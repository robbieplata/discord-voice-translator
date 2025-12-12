import asyncio
import logging
import os
import signal
from typing import cast

import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient
from dotenv import load_dotenv

from .audio_sink import openai_client as _openai_client

logger = logging.getLogger("voicebot")

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def validate_config() -> None:
    """Validate required environment variables."""
    missing = [
        name
        for name, val in {
            "DISCORD_BOT_TOKEN": TOKEN,
            "OPENAI_API_KEY": OPENAI_KEY,
        }.items()
        if not val
    ]

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    if not hasattr(VoiceRecvClient, "listen"):
        raise RuntimeError("discord-ext-voice-recv not installed")


validate_config()

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


_shutdown_started: bool = False


@bot.event
async def on_ready():
    if bot.user:
        logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
        logger.info(f"Connected to {len(bot.guilds)} guild(s)")
        try:
            synced = await bot.tree.sync()
            logger.info(f"Synced {len(synced)} slash command(s)")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    """Global handler for prefix command errors."""
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing argument: `{error.param.name}`")
        return
    if isinstance(error, commands.BadArgument):
        await ctx.send(f"Bad argument: {error}")
        return
    logger.error(f"Command error: {error}", exc_info=error)
    await ctx.send("An unexpected error occurred.")


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: discord.app_commands.AppCommandError
):
    """Global handler for slash command errors."""
    logger.error(f"Slash command error: {error}", exc_info=error)
    try:
        msg = "An unexpected error occurred."
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
    except discord.HTTPException:
        pass


async def shutdown():
    """Gracefully disconnect from all voice channels."""
    global _shutdown_started
    if _shutdown_started:
        return
    _shutdown_started = True

    logger.info("Shutting down...")

    # Ensure sinks are marked closed first so on_voice_state_update won't auto-reconnect.
    try:
        from .commands import active_sinks

        for guild_id, sink in list(active_sinks.items()):
            try:
                sink.shutdown()
            except Exception as exc:
                logger.debug("Sink shutdown error (guild=%s): %s", guild_id, exc)
            active_sinks.pop(guild_id, None)
    except Exception as exc:
        logger.debug("Active sink shutdown import/error: %s", exc)

    for vc in bot.voice_clients:
        try:
            if hasattr(vc, "stop_listening"):
                cast(VoiceRecvClient, vc).stop_listening()
            await vc.disconnect(force=True)
        except Exception as e:
            logger.warning(f"Shutdown disconnect error: {e}")

    # Close shared OpenAI client to avoid aiohttp connector leaks
    try:
        close_fn = getattr(_openai_client, "close", None)
        if close_fn:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await asyncio.shield(result)
    except Exception as exc:
        logger.debug("OpenAI client close error: %s", exc)

    # Explicitly close discord.py's HTTP client session/connector.
    # (This complements bot.close() and helps avoid "Unclosed connector" warnings
    # when the loop is stopping aggressively.)
    try:
        http = getattr(bot, "http", None)
        http_close = getattr(http, "close", None) if http else None
        if callable(http_close):
            result = http_close()
            if asyncio.iscoroutine(result):
                await asyncio.shield(result)
    except Exception as exc:
        logger.debug("Discord HTTP close error: %s", exc)

    # Shield close from cancellation so aiohttp sessions/connectors get closed.
    await asyncio.shield(bot.close())


async def run_bot():
    """Run the bot."""
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            pass

    try:
        await bot.start(TOKEN)  # type: ignore
    except discord.LoginFailure:
        logger.error("Invalid Discord token!")
    except Exception as e:
        # If we're shutting down, discord.py may raise because its HTTP session is closed.
        if _shutdown_started or str(e) == "Session is closed":
            logger.debug("Bot stopped during shutdown: %s", e)
        else:
            logger.error(f"Bot error: {e}")
    finally:
        # Ensure we always run full shutdown cleanup (voice disconnect + client closes)
        # even when the event loop is being cancelled (e.g., Ctrl+C).
        if not bot.is_closed():
            try:
                await asyncio.shield(shutdown())
            except Exception as exc:
                logger.debug("Shutdown error: %s", exc)
