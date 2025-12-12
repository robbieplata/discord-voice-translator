# Discord Voice Translator Bot

A real-time voice translation bot for Discord that listens to voice chat, transcribes speech, translates between languages, and speaks the translation back.

## Features

- **Real-time Voice Translation**: Automatic language detection and translation
- **Multiple Translation Backends**: DeepL (recommended), Google Translate, OpenAI
- **Per-Server Settings**: Each server has its own backend and voice preferences
- **Persistent Settings**: Settings are saved and restored on restart
- **VAD**: Voice Activity Detection to filter silence and noise
- **Filtering**: Filters out Whisper hallucinations using confidence metrics

## Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/robbieplata/discord-voice-translator.git
   cd discord-voice-translator
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run with Docker Compose**
   ```bash
   docker compose up -d
   ```

## Manual Setup

### Prerequisites

- Python 3.10+
- FFmpeg installed on your system
- Discord Bot Token
- OpenAI API Key
- DeepL API Key (optional, but recommended)

### Installation

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # Activate it:
   # Windows:
   .venv\Scripts\activate
   
   # macOS / Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the bot**
   ```bash
   python -m src.main
   ```

## Code Style

This repo uses Ruff for formatting + import sorting, and pre-commit to enforce it.

```bash
pip install -r requirements.txt
pre-commit install

# one-time run across the repo
ruff format .
ruff check --fix .
```

## Configuration

Create a `.env` file with the following variables:

```env
# Required
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key

# Optional but recommended (enables DeepL backend)
DEEPL_API_KEY=your_deepl_api_key
```

## Commands

All commands are slash commands under `/translator`.

### Voice
| Command | Description |
|---------|-------------|
| `/translator join` | Join voice channel and start translating |
| `/translator leave` | Leave voice and stop translating |
| `/translator status` | Show current voice/translation status |

### Backend
| Command | Description |
|---------|-------------|
| `/translator backend show` | Show current translation backend |
| `/translator backend list` | List available translation backends |
| `/translator backend set <name>` | Switch backend (deepl, google, openai) |
| `/translator backend usage` | Show usage/quota info |

### TTS Voice
| Command | Description |
|---------|-------------|
| `/translator voice show` | Show current TTS voice |
| `/translator voice list` | List available voices |
| `/translator voice set <name>` | Change TTS voice |

### TTS Mode
| Command | Description |
|---------|-------------|
| `/translator tts show` | Show current TTS playback mode |
| `/translator tts set <mode>` | Change TTS playback mode (`normal` or `rude`) |

### Available Voices
- **Male**: `echo`, `onyx`, `fable`
- **Female**: `alloy`, `nova` (default), `shimmer`

### Utility Commands
| Command | Description |
|---------|-------------|
| `/translator ping` | Check bot latency |
| `/translator health` | Show service status |
| `/translator help` | Show all commands |

## Discord Bot Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to Bot settings and create a bot
4. Enable these Privileged Gateway Intents:
   - Message Content Intent
   - Server Members Intent (optional)
5. Copy the bot token to your `.env` file
6. Use this invite URL format:
   ```
   https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=36768768&integration_type=0&scope=bot+applications.commands
   ```

Required permissions:
- Connect (to voice channels)
- Speak (for TTS playback)
- Send Messages (for translation logs)
- Use Voice Activity

## Architecture

```
User speaks → VAD → Whisper STT → (silence debounce) → Translation (Backend) → OpenAI TTS → Playback
```

- **VAD**: Adaptive noise floor with RMS-based voice activity detection
- **STT**: OpenAI Whisper with confidence-based hallucination filtering
- **Sentence Detection**: Silence-based debounce (finalize after a short silence)
- **Translation**: Pluggable backends (DeepL, Google, OpenAI)
- **TTS**: OpenAI TTS with 6 voice options

## Configuration Notes

### Silence Debounce (Utterance Finalization)

The bot finalizes an utterance after a short period of silence (debounce), then runs STT → translate → TTS.

Settings are split across `src/core/configs/vad.py` (VAD/debounce) and `src/core/configs/openai.py` (AI sentence detection):
```python
USE_AI_SENTENCE_DETECTION = False
SILENCE_DURATION_SEC = 1.2
MAX_PENDING_UTTERANCES = 10
```

### Whisper Prompt

The `WHISPER_PROMPT` in `src/core/configs/openai.py` can be used to bias transcription toward your expected languages (optional):

```python
WHISPER_PROMPT = "This audio contains English or Mandarin Chinese speech."
```

Modify this prompt to match your expected languages for better transcription accuracy.

### DeepL Guardrails

DeepL text-translate requests have a maximum request size; this project enforces client-side guardrails to stay under the limit and reduce burst traffic.
If a translation input is too large, it will be chunked and translated in parts, then stitched back together.

### Playback Interruption ("Barge-in")

If someone starts speaking while the bot is playing TTS, the bot pauses TTS playback and resumes once the channel is quiet again (it keeps listening/receiving audio the whole time). This can be disabled by enabling "rude" TTS mode via `/translator tts set rude` to talk over conversation.

## License

MIT
