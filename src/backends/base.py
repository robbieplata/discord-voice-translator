from abc import ABC, abstractmethod
from typing import Optional


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
