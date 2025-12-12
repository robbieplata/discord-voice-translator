import logging
import threading
from typing import Optional, Tuple

from ..backends import TranslationBackend

logger = logging.getLogger("voicebot")


class Translator:
    """Translator with pluggable backend and bidirectional EN<->ZH support."""

    def __init__(self, backend: TranslationBackend):
        self._backend = backend
        self._lock = threading.Lock()
        logger.info("[translate] Initialized with %s backend", backend.name())

    @property
    def backend_name(self) -> str:
        """Get the backend name."""
        with self._lock:
            return self._backend.name().lower()

    def set_backend(self, backend: TranslationBackend) -> None:
        """Swap the underlying backend in-place.

        This is used to support live backend switching while a voice session is active.
        """
        with self._lock:
            self._backend = backend
        logger.info("[translate] Switched backend to %s", backend.name())

    def translate(self, text: str, detected_lang: str) -> Tuple[str, str]:
        """
        Bidirectional translation: EN -> ZH, ZH/YUE/CMN -> EN.
        Returns (translated_text, target_lang).
        """
        clean_text = text.strip()
        if not clean_text:
            return text, detected_lang

        lang = detected_lang.lower()
        logger.debug("[translate] input detected_lang='%s' text='%s'", lang, clean_text[:80])

        # English -> Chinese
        if lang.startswith("en") or lang == "english":
            target_lang = "ZH"
            logger.debug("[translate] direction: EN->ZH")
            try:
                with self._lock:
                    backend = self._backend
                translated = backend.translate(clean_text, target_lang)
                logger.debug(
                    "[translate] EN->ZH success: '%s'", translated[:80] if translated else ""
                )
                return translated, target_lang
            except Exception as exc:
                logger.error("[translate] EN->ZH failed: %s", exc)
                raise  # Propagate error instead of silent fallback

        # Chinese/Cantonese -> English
        if lang.startswith(("zh", "yue", "cmn")) or lang in ("chinese", "mandarin", "cantonese"):
            target_lang = "EN-US"
            logger.debug("[translate] direction: ZH->EN-US")
            try:
                with self._lock:
                    backend = self._backend
                translated = backend.translate(clean_text, target_lang)
                logger.debug(
                    "[translate] ZH->EN success: '%s'", translated[:80] if translated else ""
                )
                return translated, target_lang
            except Exception as exc:
                logger.error("[translate] ZH->EN failed: %s", exc)
                raise  # Propagate error instead of silent fallback

        # No translation - unrecognized language
        logger.warning("[translate] SKIPPED - lang '%s' not configured", detected_lang)
        return text, detected_lang

    def get_usage(self) -> Optional[dict]:
        """Get usage info from the backend."""
        return self._backend.get_usage()
