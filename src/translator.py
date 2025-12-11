"""Translation module with pluggable backends."""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dotenv import load_dotenv

from .config import (
    DEEPL_MAX_RETRIES,
    DEEPL_RETRY_DELAY_SEC,
    OPENAI_TRANSLATION_MODEL,
    OPENAI_TRANSLATION_TEMPERATURE,
    OPENAI_TRANSLATION_MAX_TOKENS,
)

logger = logging.getLogger("voicebot")
load_dotenv()


class TranslationBackend(ABC):
    """Abstract base for translation backends."""
    
    @abstractmethod
    def translate(self, text: str, target_lang: str) -> str:
        """Translate text to target language. Returns translated text."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass
    
    def get_usage(self) -> Optional[dict]:
        """Get usage/quota info. Returns dict with usage details or None if unavailable."""
        return None


class DeepLBackend(TranslationBackend):
    """DeepL API translation backend."""
    
    def __init__(self, api_key: str):
        import deepl
        self._client = deepl.Translator(api_key)
        self._max_retries = DEEPL_MAX_RETRIES
        self._retry_delay = DEEPL_RETRY_DELAY_SEC
    
    def name(self) -> str:
        return "DeepL"
    
    def translate(self, text: str, target_lang: str) -> str:
        import deepl
        
        for attempt in range(self._max_retries + 1):
            try:
                start = time.perf_counter()
                result = self._client.translate_text(text, target_lang=target_lang)
                translated = getattr(result, "text", str(result))
                elapsed = (time.perf_counter() - start) * 1000
                logger.info("⏱ [translate] DeepL: %.0fms", elapsed)
                return translated
                
            except deepl.DeepLException as exc:
                if attempt < self._max_retries:
                    logger.warning("[translate] DeepL retry %d/%d: %s", 
                                   attempt + 1, self._max_retries, exc)
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    logger.error("[translate] DeepL failed: %s", exc)
                    raise
        
        return text
    
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


class GoogleTranslateBackend(TranslationBackend):
    """Google Translate backend using googletrans library (free)."""
    
    def __init__(self):
        try:
            from googletrans import Translator
            self._client = Translator()
        except ImportError:
            raise RuntimeError("googletrans not installed: pip install googletrans==4.0.0-rc1")
    
    def name(self) -> str:
        return "Google"
    
    def translate(self, text: str, target_lang: str) -> str:
        start = time.perf_counter()
        
        # Map DeepL codes to Google codes
        lang_map = {
            "ZH": "zh-cn",
            "EN-US": "en",
            "EN-GB": "en",
        }
        dest = lang_map.get(target_lang, target_lang.lower().split("-")[0])
        
        try:
            import asyncio
            result = asyncio.run(self._client.translate(text, dest=dest))
            translated = result.text
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("⏱ [translate] Google: %.0fms", elapsed)
            return translated
        except Exception as exc:
            logger.error("[translate] Google failed: %s", exc)
            raise
    
    def get_usage(self) -> Optional[dict]:
        """Google Translate (googletrans) is free with no quota tracking."""
        return {
            "note": "Free tier (googletrans library)",
            "limit": "Unlimited (rate-limited)",
        }


class OpenAIBackend(TranslationBackend):
    """OpenAI translation backend."""
    
    def __init__(self, api_key: str):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = OPENAI_TRANSLATION_MODEL
    
    def name(self) -> str:
        return "OpenAI"
    
    def translate(self, text: str, target_lang: str) -> str:
        start = time.perf_counter()
        
        # Map target lang codes to natural language
        lang_names = {
            "ZH": "Simplified Chinese",
            "EN-US": "English",
            "EN-GB": "English",
        }
        target_name = lang_names.get(target_lang, target_lang)
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a translator. Translate the user's text to {target_name}. "
                            "Output ONLY the translation, nothing else. No quotes, no explanations."
                        )
                    },
                    {"role": "user", "content": text}
                ],
                temperature=OPENAI_TRANSLATION_TEMPERATURE,
                max_tokens=OPENAI_TRANSLATION_MAX_TOKENS,
            )
            translated = response.choices[0].message.content or text
            translated = translated.strip().strip('"\'')
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("⏱ [translate] OpenAI: %.0fms", elapsed)
            return translated
        except Exception as exc:
            logger.error("[translate] OpenAI failed: %s", exc)
            raise
    
    def get_usage(self) -> Optional[dict]:
        """OpenAI usage must be checked via dashboard."""
        return {
            "note": "Check usage at platform.openai.com",
            "model": self._model,
        }


AVAILABLE_BACKENDS = ["deepl", "google", "openai"]


def list_backends() -> list[str]:
    """List all available backend names."""
    return AVAILABLE_BACKENDS.copy()


def create_backend(name: str) -> TranslationBackend:
    """Create a translation backend by name."""
    name = name.lower()
    
    if name == "deepl":
        api_key = os.getenv("DEEPL_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPL_API_KEY not set")
        return DeepLBackend(api_key)
    
    if name == "google":
        return GoogleTranslateBackend()
    
    if name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIBackend(api_key)
    
    raise ValueError(f"Unknown backend: {name}. Available: {', '.join(AVAILABLE_BACKENDS)}")


def create_default_backend() -> Tuple[TranslationBackend, str]:
    """
    Create the default translation backend based on available API keys.
    Returns (backend, backend_name).
    """
    # Try DeepL first (higher quality)
    deepl_key = os.getenv("DEEPL_API_KEY")
    if deepl_key:
        try:
            return DeepLBackend(deepl_key), "deepl"
        except Exception as exc:
            logger.warning("[translate] DeepL init failed: %s", exc)
    
    # Fall back to Google Translate
    try:
        return GoogleTranslateBackend(), "google"
    except Exception as exc:
        logger.warning("[translate] Google init failed: %s", exc)
    
    raise RuntimeError("No translation backend available. Set DEEPL_API_KEY or install googletrans.")


class Translator:
    """Translator with pluggable backend and bidirectional EN<->ZH support."""
    
    def __init__(self, backend: TranslationBackend):
        self._backend = backend
        logger.info("[translate] Initialized with %s backend", backend.name())
    
    @property
    def backend_name(self) -> str:
        """Get the backend name."""
        return self._backend.name().lower()
    
    def translate(self, text: str, detected_lang: str) -> Tuple[str, str]:
        """
        Bidirectional translation: EN -> ZH, ZH/YUE/CMN -> EN.
        Returns (translated_text, target_lang).
        """
        clean_text = text.strip()
        if not clean_text:
            return text, detected_lang
        
        lang = detected_lang.lower()
        
        # English -> Chinese
        if lang.startswith("en"):
            target_lang = "ZH"
            try:
                translated = self._backend.translate(clean_text, target_lang)
                return translated, target_lang
            except Exception as exc:
                logger.error("[translate] EN->ZH failed: %s", exc)
                return text, detected_lang
        
        # Chinese/Cantonese -> English
        if lang.startswith("zh") or lang.startswith("yue") or lang.startswith("cmn"):
            target_lang = "EN-US"
            try:
                translated = self._backend.translate(clean_text, target_lang)
                return translated, target_lang
            except Exception as exc:
                logger.error("[translate] ZH->EN failed: %s", exc)
                return text, detected_lang
        
        # No translation needed
        logger.debug("[translate] Lang '%s' not configured, skipping", detected_lang)
        return text, detected_lang
    
    def get_usage(self) -> Optional[dict]:
        """Get usage info from the backend."""
        return self._backend.get_usage()
