import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from .translator import Translator, create_backend, create_default_backend, list_backends
from .config import OPENAI_TTS_VOICES, OPENAI_TTS_DEFAULT_VOICE

logger = logging.getLogger("voicebot")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SETTINGS_FILE = DATA_DIR / "server_settings.json"


def _load_all_settings() -> Dict[str, dict]:
    """Load all server settings from file."""
    if not SETTINGS_FILE.exists():
        return {}
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("[persist] Failed to load settings: %s", exc)
        return {}


def _save_all_settings(settings: Dict[str, dict]) -> None:
    """Save all server settings to file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as exc:
        logger.warning("[persist] Failed to save settings: %s", exc)


class ServerState:
    """Holds per-server configuration and state."""
    
    __slots__ = ("guild_id", "translator", "backend_name", "voice")
    
    def __init__(self, guild_id: int, backend_name: Optional[str] = None, voice: Optional[str] = None):
        self.guild_id = guild_id
        self.voice = voice or OPENAI_TTS_DEFAULT_VOICE
        
        if backend_name:
            self.backend_name = backend_name
            backend = create_backend(backend_name)
        else:
            backend, self.backend_name = create_default_backend()
        
        self.translator = Translator(backend)
        logger.info("[server:%d] Initialized (backend=%s, voice=%s)", 
                    guild_id, self.backend_name, self.voice)
    
    def switch_backend(self, name: str) -> str:
        """Switch to a different translation backend. Returns new backend name."""
        name = name.lower()
        if name not in list_backends():
            raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(list_backends())}")
        
        new_backend = create_backend(name)
        self.translator = Translator(new_backend)
        self.backend_name = name
        
        logger.info("[server:%d] Switched to %s backend", self.guild_id, new_backend.name())
        return new_backend.name()
    
    def switch_voice(self, voice: str) -> str:
        """Switch TTS voice. Returns new voice name."""
        voice = voice.lower()
        if voice not in OPENAI_TTS_VOICES:
            raise ValueError(f"Unknown voice '{voice}'. Available: {', '.join(OPENAI_TTS_VOICES)}")
        
        self.voice = voice
        logger.info("[server:%d] Switched to %s voice", self.guild_id, voice)
        return voice
    
    def get_backend_name(self) -> str:
        """Get current backend name."""
        return self.translator.backend_name
    
    def get_voice(self) -> str:
        """Get current TTS voice."""
        return self.voice
    
    def get_usage(self) -> Optional[dict]:
        """Get usage info from current backend."""
        return self.translator.get_usage()


class ServerStateManager:
    """Manages ServerState instances for all guilds with persistence."""
    
    def __init__(self):
        self._states: Dict[int, ServerState] = {}
        self._settings: Dict[str, dict] = _load_all_settings()
    
    def get(self, guild_id: int) -> ServerState:
        """Get or create ServerState for a guild."""
        if guild_id not in self._states:
            saved = self._settings.get(str(guild_id), {})
            self._states[guild_id] = ServerState(
                guild_id,
                backend_name=saved.get("backend"),
                voice=saved.get("voice"),
            )
        return self._states[guild_id]
    
    def save(self, guild_id: int) -> None:
        """Save a guild's settings to persistent storage."""
        if guild_id not in self._states:
            return
        
        state = self._states[guild_id]
        self._settings[str(guild_id)] = {
            "backend": state.backend_name,
            "voice": state.voice,
        }
        _save_all_settings(self._settings)
        logger.info("[server:%d] Settings saved", guild_id)
    
    def remove(self, guild_id: int) -> None:
        """Remove ServerState for a guild."""
        if guild_id in self._states:
            del self._states[guild_id]
            logger.info("[server:%d] State removed", guild_id)
    
    def has(self, guild_id: int) -> bool:
        """Check if guild has state."""
        return guild_id in self._states


# Global manager instance
server_manager = ServerStateManager()
