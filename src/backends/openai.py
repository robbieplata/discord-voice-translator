import logging
import time
from typing import Optional

from ..core.config import (
    MAX_RETRIES,
    OPENAI_TRANSLATION_MODEL,
    RETRY_DELAY_SEC,
    TRANSLATE_TIMEOUT_SEC,
)
from ..core.openai_guardrails import thread_limited_call, truncate_for_chat
from .base import TranslationBackend

logger = logging.getLogger("voicebot")


class OpenAIBackend(TranslationBackend):
    """OpenAI translation backend using GPT models."""

    def __init__(self, api_key: str):
        import openai

        # Per-request timeouts are also passed below for extra safety.
        self._client = openai.OpenAI(api_key=api_key, timeout=TRANSLATE_TIMEOUT_SEC)
        self._model = OPENAI_TRANSLATION_MODEL

    def name(self) -> str:
        return "OpenAI"

    def translate(self, text: str, target_lang: str) -> str:
        start = time.perf_counter()

        trunc = truncate_for_chat(text.strip())
        if trunc.truncated:
            logger.warning(
                "[translate] OpenAI input too long (%d chars), truncating to %d chars",
                trunc.original_len,
                len(trunc.text),
            )
        text = trunc.text

        # Map target lang codes to natural language
        lang_names = {
            "ZH": "Simplified Chinese",
            "EN-US": "English",
            "EN-GB": "English",
        }
        target_name = lang_names.get(target_lang, target_lang)

        from openai import APIConnectionError, RateLimitError

        for attempt in range(MAX_RETRIES + 1):
            try:

                def _do_call():
                    return self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    f"You are a translator. Translate the user's text to {target_name}. "
                                    "Output ONLY the translation, nothing else. No quotes, no explanations."
                                ),
                            },
                            {"role": "user", "content": text},
                        ],
                        temperature=0.3,
                        max_tokens=500,
                        timeout=TRANSLATE_TIMEOUT_SEC,
                    )

                response = thread_limited_call(_do_call)
                translated = response.choices[0].message.content or text
                translated = translated.strip().strip("\"'")
                elapsed = (time.perf_counter() - start) * 1000
                logger.info("⏱ [translate] OpenAI: %.0fms", elapsed)
                return translated
            except RateLimitError:
                if attempt < MAX_RETRIES:
                    sleep_s = RETRY_DELAY_SEC * (attempt + 1)
                    logger.warning(
                        "[translate] OpenAI rate-limited (attempt %d/%d); sleeping %.1fs",
                        attempt + 1,
                        MAX_RETRIES,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
            except APIConnectionError:
                if attempt < MAX_RETRIES:
                    sleep_s = RETRY_DELAY_SEC * (attempt + 1)
                    logger.warning(
                        "[translate] OpenAI connection error (attempt %d/%d); sleeping %.1fs",
                        attempt + 1,
                        MAX_RETRIES,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
            except Exception as exc:
                logger.error("[translate] OpenAI failed: %s", exc, exc_info=True)
                raise

        raise RuntimeError("OpenAI translation failed after retries")

    def get_usage(self) -> Optional[dict]:
        """OpenAI usage must be checked via dashboard."""
        return {
            "note": "Check usage at platform.openai.com",
            "model": self._model,
        }
