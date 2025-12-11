# Discord Voice Translator Bot

A real-time voice translation bot for Discord that listens to voice chat, transcribes speech, translates between languages, and speaks the translation back.

## Features

- **Real-time Voice Translation**: Automatic language detection and translation
- **Multiple Translation Backends**: DeepL (recommended), Google Translate, OpenAI
- **Per-Server Settings**: Each server has its own backend and voice preferences
- **Persistent Settings**: Settings are saved and restored on restart
- **Smart VAD**: Voice Activity Detection to filter silence and noise
- **Hallucination Filtering**: Filters out Whisper hallucinations using confidence metrics

## Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/discord-voice-translator.git
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

## Configuration

Create a `.env` file with the following variables:

```env
# Required
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key

# Optional (enables DeepL backend)
DEEPL_API_KEY=your_deepl_api_key
```

## Commands

All commands are slash commands only.

### Voice Commands
| Command | Description |
|---------|-------------|
| `/tr` or `/join` | Join voice channel and start translating |
| `/leave` | Leave voice and stop translating |

### Backend Commands
| Command | Description |
|---------|-------------|
| `/backend` | Show current translation backend |
| `/backend_list` | List available backends |
| `/backend_set <name>` | Switch backend (deepl, google, openai) |
| `/backend_usage` | Show usage/quota info |

### Voice Settings
| Command | Description |
|---------|-------------|
| `/voice` | Show current TTS voice |
| `/voice_list` | List available voices |
| `/voice_set <name>` | Change TTS voice |

### Available Voices
- **Male**: `echo`, `onyx`, `fable`
- **Female**: `alloy`, `nova` (default), `shimmer`

### Utility Commands
| Command | Description |
|---------|-------------|
| `/ping` | Check bot latency |
| `/health` | Show service status |
| `/help` | Show all commands |

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
User speaks → VAD → Whisper STT → Translation → OpenAI TTS → Playback
```

- **VAD**: Silence detection with RMS-based voice activity
- **STT**: OpenAI Whisper with confidence-based hallucination filtering
- **Translation**: Pluggable backends (DeepL, Google, OpenAI)
- **TTS**: OpenAI TTS with 6 voice options

## Configuration Notes

### Whisper Prompt

The `WHISPER_PROMPT` in `src/config.py` is important for accurate language detection. It hints to Whisper which languages to expect:

```python
WHISPER_PROMPT = "This audio contains English or Mandarin Chinese speech."
```

Modify this prompt to match your expected languages for better transcription accuracy.

## License

MIT