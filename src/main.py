import asyncio
import logging
import os

from .bot import bot, run_bot, setup_commands

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VOICEBOT_LOG_LEVEL = os.getenv("VOICEBOT_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logging.getLogger("voicebot").setLevel(getattr(logging, VOICEBOT_LOG_LEVEL, logging.INFO))

for name in [
    "discord",
    "discord.ext.voice_recv",
    "httpx",
    "httpcore",
    "httpcore.http2",
    "hpack",
    "h2",
    "openai",
    "urllib3",
    "deepl",
    "googletrans",
    "asyncio",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


def main():
    setup_commands(bot)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
