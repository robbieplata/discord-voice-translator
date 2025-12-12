import logging
import re
import threading
import time
from typing import Optional

from ..core.config import (
    DEEPL_MAX_REQUEST_BYTES,
    DEEPL_MAX_REQUESTS_PER_SEC,
    DEEPL_USAGE_LOG_EVERY_SEC,
    DEEPL_USAGE_WARN_PERCENT,
    MAX_RETRIES,
    RETRY_DELAY_SEC,
)
from .base import TranslationBackend

logger = logging.getLogger("voicebot")


class DeepLBackend(TranslationBackend):
    """DeepL API translation backend."""

    def __init__(self, api_key: str):
        try:
            import deepl

            self._client = deepl.Translator(api_key)
            self._rate_lock = threading.Lock()
            self._next_allowed_at = 0.0
            self._last_usage_check_at = 0.0
        except ImportError:
            raise RuntimeError("deepl not installed: pip install deepl")

    def name(self) -> str:
        return "DeepL"

    def _maybe_log_usage(self) -> None:
        if DEEPL_USAGE_LOG_EVERY_SEC <= 0:
            return

        now = time.monotonic()
        if (now - self._last_usage_check_at) < DEEPL_USAGE_LOG_EVERY_SEC:
            return

        self._last_usage_check_at = now
        usage = self.get_usage()
        if not usage:
            return

        pct = usage.get("character_percent")
        remaining = usage.get("characters_remaining")
        limit = usage.get("character_limit")
        count = usage.get("character_count")

        if pct is not None and remaining is not None and limit:
            if float(pct) >= float(DEEPL_USAGE_WARN_PERCENT):
                logger.warning(
                    "[translate] DeepL usage high: %s%% used (%s/%s), remaining=%s",
                    pct,
                    count,
                    limit,
                    remaining,
                )
            else:
                logger.info(
                    "[translate] DeepL usage: %s%% used (%s/%s), remaining=%s",
                    pct,
                    count,
                    limit,
                    remaining,
                )

    def _throttle(self) -> None:
        """Client-side pacing to avoid bursty DeepL traffic.

        NOTE: translate() is called via asyncio.to_thread(), so this runs in a worker thread.
        """
        if DEEPL_MAX_REQUESTS_PER_SEC <= 0:
            return

        min_interval = 1.0 / float(DEEPL_MAX_REQUESTS_PER_SEC)
        with self._rate_lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                time.sleep(self._next_allowed_at - now)
                now = time.monotonic()
            self._next_allowed_at = now + min_interval

    @staticmethod
    def _chunk_text(text: str, max_bytes: int) -> list[str]:
        """Split text into chunks that (roughly) respect a UTF-8 byte ceiling."""
        if max_bytes <= 0:
            return [text]

        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return [text]

        # Prefer splitting on paragraph / sentence-ish boundaries before falling back to hard splits.
        parts: list[str] = []
        for block in re.split(r"(\n\n+)", text):
            if not block:
                continue
            parts.append(block)

        candidate_segments: list[str] = []
        for part in parts:
            if len(part.encode("utf-8")) <= max_bytes:
                candidate_segments.append(part)
                continue
            # Split on whitespace after sentence-ending punctuation (best-effort, multilingual-ish).
            sentences = re.split(r"(?<=[\.!\?。！？])\s+", part)
            if len(sentences) == 1:
                candidate_segments.append(part)
            else:
                for s in sentences:
                    if s:
                        candidate_segments.append(s + " ")

        chunks: list[str] = []
        current: list[str] = []
        current_bytes = 0

        def flush() -> None:
            nonlocal current, current_bytes
            if current:
                chunks.append("".join(current))
                current = []
                current_bytes = 0

        for seg in candidate_segments:
            seg_bytes = len(seg.encode("utf-8"))
            if seg_bytes > max_bytes:
                flush()
                # Hard split by characters (safe but may reduce translation quality)
                buf: list[str] = []
                buf_bytes = 0
                for ch in seg:
                    ch_b = len(ch.encode("utf-8"))
                    if buf and (buf_bytes + ch_b) > max_bytes:
                        chunks.append("".join(buf))
                        buf = [ch]
                        buf_bytes = ch_b
                    else:
                        buf.append(ch)
                        buf_bytes += ch_b
                if buf:
                    chunks.append("".join(buf))
                continue

            if current and (current_bytes + seg_bytes) > max_bytes:
                flush()
            current.append(seg)
            current_bytes += seg_bytes

        flush()
        return chunks if chunks else [text]

    def _translate_single(self, text: str, target_lang: str) -> str:
        import deepl

        for attempt in range(MAX_RETRIES + 1):
            try:
                self._throttle()
                self._maybe_log_usage()
                start = time.perf_counter()
                result = self._client.translate_text(text, target_lang=target_lang)
                translated = getattr(result, "text", str(result))
                elapsed = (time.perf_counter() - start) * 1000
                logger.info("⏱ [translate] DeepL: %.0fms", elapsed)
                return translated
            except deepl.DeepLException as exc:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "[translate] DeepL retry %d/%d: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        exc,
                    )
                    time.sleep(RETRY_DELAY_SEC * (attempt + 1))
                else:
                    logger.error("[translate] DeepL failed: %s", exc, exc_info=True)
                    raise
        return text

    def translate(self, text: str, target_lang: str) -> str:
        if not text:
            return text

        # DeepL text translation requests must stay under 128KiB total request size.
        # We enforce a lower cap to reduce risk of API-side rejection.
        chunks = self._chunk_text(text, DEEPL_MAX_REQUEST_BYTES)
        if len(chunks) > 1:
            logger.warning(
                "[translate] DeepL chunking input to respect request size cap: chunks=%d bytes=%d cap=%d",
                len(chunks),
                len(text.encode("utf-8")),
                DEEPL_MAX_REQUEST_BYTES,
            )

        out_parts: list[str] = []
        for chunk in chunks:
            out_parts.append(self._translate_single(chunk, target_lang=target_lang))
        return "".join(out_parts)

    def get_usage(self) -> Optional[dict]:
        """Get DeepL API usage statistics."""
        try:
            usage = self._client.get_usage()
            char_count = usage.character.count if usage.character else 0
            char_limit = usage.character.limit if usage.character else 0

            result: dict = {
                "character_count": char_count,
                "character_limit": char_limit,
            }

            if char_limit and char_count is not None:
                result["character_percent"] = round((char_count / char_limit) * 100, 1)
                result["characters_remaining"] = char_limit - char_count

            return result
        except Exception as exc:
            logger.warning("[translate] DeepL usage query failed: %s", exc)
            return None
