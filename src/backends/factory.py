import logging
import os
from typing import Tuple

from .base import TranslationBackend
from .constants import AVAILABLE_BACKENDS
from .deepl import DeepLBackend
from .google import GoogleTranslateBackend
from .openai import OpenAIBackend

logger = logging.getLogger("voicebot")


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
            logger.warning("[translate] DeepL init failed: %s", exc, exc_info=True)

    # Fall back to Google Translate
    try:
        return GoogleTranslateBackend(), "google"
    except Exception as exc:
        logger.warning("[translate] Google init failed: %s", exc, exc_info=True)

    raise RuntimeError(
        "No translation backend available. Set DEEPL_API_KEY or install googletrans."
    )
