import asyncio
import inspect
import logging
import os
import re
import threading
import time
from typing import Optional

from ..core.config import (
    GOOGLE_MAX_INPUT_CHARS,
    GOOGLE_MAX_REQUESTS_PER_SEC,
    MAX_RETRIES,
    RETRY_DELAY_SEC,
    TRANSLATE_TIMEOUT_SEC,
)
from .base import TranslationBackend

logger = logging.getLogger("voicebot")


class GoogleTranslateBackend(TranslationBackend):
    """Google Translate backend using googletrans library (free)."""

    def __init__(self):
        try:
            import httpx
            from googletrans import Translator

            # NOTE: googletrans is an unofficial client; configure strict timeouts to
            # avoid worker threads getting stuck past the asyncio timeout wrapper.
            timeout = httpx.Timeout(float(TRANSLATE_TIMEOUT_SEC))

            # Optional: allow overriding translate host(s) for reliability.
            service_urls = os.getenv("GOOGLETRANS_SERVICE_URLS")
            urls = (
                tuple(s.strip() for s in service_urls.split(",") if s.strip())
                if service_urls
                else None
            )

            self._client = Translator(
                service_urls=urls or ("translate.googleapis.com",),
                raise_exception=True,
                timeout=timeout,
            )
            self._rate_lock = threading.Lock()
            self._next_allowed_at = 0.0
        except ImportError:
            raise RuntimeError("googletrans not installed: pip install googletrans==4.0.0-rc1")

    def name(self) -> str:
        return "Google"

    def _throttle(self) -> None:
        """Client-side pacing to avoid bursty googletrans traffic.

        NOTE: translate() is called via asyncio.to_thread(), so this runs in a worker thread.
        """
        if GOOGLE_MAX_REQUESTS_PER_SEC <= 0:
            return

        min_interval = 1.0 / float(GOOGLE_MAX_REQUESTS_PER_SEC)
        with self._rate_lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                time.sleep(self._next_allowed_at - now)
                now = time.monotonic()
            self._next_allowed_at = now + min_interval

    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        # Prefer paragraph/sentence-ish splits before falling back to hard splits.
        parts: list[str] = []
        for block in re.split(r"(\n\n+)", text):
            if block:
                parts.append(block)

        candidate_segments: list[str] = []
        for part in parts:
            if len(part) <= max_chars:
                candidate_segments.append(part)
                continue
            sentences = re.split(r"(?<=[\.!\?。！？])\s+", part)
            if len(sentences) == 1:
                candidate_segments.append(part)
            else:
                for s in sentences:
                    if s:
                        candidate_segments.append(s + " ")

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0

        for seg in candidate_segments:
            if len(seg) > max_chars:
                flush()
                for i in range(0, len(seg), max_chars):
                    chunks.append(seg[i : i + max_chars])
                continue

            if current and (current_len + len(seg)) > max_chars:
                flush()
            current.append(seg)
            current_len += len(seg)

        flush()
        return chunks if chunks else [text]

    def translate(self, text: str, target_lang: str) -> str:
        start = time.perf_counter()

        # Map DeepL codes to Google codes
        lang_map = {
            "ZH": "zh-cn",
            "EN-US": "en",
            "EN-GB": "en",
        }
        dest = lang_map.get(target_lang, target_lang.lower().split("-")[0])

        # Best-effort source language hint.
        # Our app is bidirectional EN<->ZH; providing src improves reliability and
        # reduces "no-op" translations when googletrans auto-detect is flaky.
        if dest.startswith("zh"):
            src = "en"
        elif dest == "en":
            src = "zh-cn"
        else:
            src = "auto"

        if not text:
            return text

        chunks = self._chunk_text(text, GOOGLE_MAX_INPUT_CHARS)
        if len(chunks) > 1:
            logger.warning(
                "[translate] Google chunking input: chunks=%d chars=%d cap=%d",
                len(chunks),
                len(text),
                GOOGLE_MAX_INPUT_CHARS,
            )

        out_parts: list[str] = []
        last_exc: Exception | None = None

        for chunk in chunks:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    self._throttle()
                    # googletrans is synchronous
                    result = self._client.translate(chunk, dest=dest, src=src)

                    # googletrans v4 uses an async client and returns a coroutine.
                    if inspect.iscoroutine(result):
                        try:
                            result = asyncio.run(result)
                        except RuntimeError as e:
                            # Only fall back if asyncio.run() is blocked by an already-running loop.
                            # Any other RuntimeError is likely from inside the coroutine itself.
                            if (
                                "asyncio.run() cannot be called from a running event loop"
                                not in str(e)
                            ):
                                raise

                            loop = asyncio.new_event_loop()
                            try:
                                result = loop.run_until_complete(result)
                            finally:
                                loop.close()
                    translated = getattr(result, "text", "")
                    if not (translated or "").strip():
                        logger.warning(
                            "[translate] Google returned empty translation; falling back to original chunk (len=%d)",
                            len(chunk),
                        )
                        translated = chunk
                    elif translated.strip() == chunk.strip() and chunk.strip():
                        logger.warning(
                            "[translate] Google produced no-op translation (src=%s dest=%s len=%d)",
                            src,
                            dest,
                            len(chunk),
                        )
                    out_parts.append(translated)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < MAX_RETRIES:
                        sleep_s = RETRY_DELAY_SEC * (attempt + 1)
                        logger.warning(
                            "[translate] Google retry %d/%d: %s (sleep %.1fs)",
                            attempt + 1,
                            MAX_RETRIES,
                            exc,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    logger.error("[translate] Google failed: %s", exc, exc_info=True)
                    raise

        if last_exc is not None:
            raise last_exc

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("⏱ [translate] Google: %.0fms", elapsed)
        return "".join(out_parts)

    def get_usage(self) -> Optional[dict]:
        """Google Translate (googletrans) is free with no quota tracking."""
        return {
            "note": "Free tier (googletrans library)",
            "limit": "Unofficial client (may be rate-limited/blocked); no quota tracking",
        }
