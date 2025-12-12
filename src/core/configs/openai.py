from __future__ import annotations

# STT (Whisper)
WHISPER_PROMPT = ""  # Empty prompt for better language detection

# TTS
OPENAI_TTS_MODEL = "tts-1"

# Translation (when using OpenAI backend)
OPENAI_TRANSLATION_MODEL = "gpt-4o-mini"

# Sentence completion detection
SENTENCE_COMPLETION_MODEL = "gpt-4o-mini"
USE_AI_SENTENCE_DETECTION = False  # Disabled - using debounce instead
MAX_PENDING_UTTERANCES = 10

# OpenAI request timeouts (seconds)
OPENAI_STT_TIMEOUT_SEC = 20.0
OPENAI_TTS_TIMEOUT_SEC = 20.0
OPENAI_SENTENCE_TIMEOUT_SEC = 10.0

# Conservative client-side pacing / concurrency to reduce burst 429s.
OPENAI_MAX_REQUESTS_PER_SEC = 5.0
OPENAI_MAX_CONCURRENT_REQUESTS = 2

# Payload limits (defensive; OpenAI limits can vary by endpoint/model).
OPENAI_WHISPER_MAX_AUDIO_BYTES = 24 * 1024 * 1024
OPENAI_TTS_MAX_INPUT_CHARS = 3500
OPENAI_CHAT_MAX_INPUT_CHARS = 8000
