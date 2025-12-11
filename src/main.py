import os
import signal
import asyncio
import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient
import logging
from typing import Dict, Optional, cast
from dotenv import load_dotenv

from .audio_handler import AudioSink
from .server_state import server_manager
from .translator import list_backends
from .config import VOICE_CONNECT_TIMEOUT_SEC, OPENAI_TTS_VOICES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voicebot")

# Silence noisy loggers
for name in ["discord", "discord.ext.voice_recv", "httpcore", "httpx", 
             "openai", "urllib3", "deepl", "googletrans"]:
    logging.getLogger(name).setLevel(logging.WARNING)

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
DEEPL_KEY = os.getenv("DEEPL_API_KEY")


def ensure_service_config() -> None:
    """Validate required environment variables."""
    missing = [name for name, val in {
        "DISCORD_BOT_TOKEN": TOKEN,
        "OPENAI_API_KEY": OPENAI_KEY,
    }.items() if not val]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    if not hasattr(VoiceRecvClient, "listen"):
        raise RuntimeError("discord-ext-voice-recv not installed")


ensure_service_config()
assert TOKEN is not None

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# Track active audio sinks per guild (for invoker disconnect detection)
active_sinks: Dict[int, AudioSink] = {}


async def get_voice_client(interaction: discord.Interaction) -> Optional[VoiceRecvClient]:
    """Get voice client for the guild."""
    guild = interaction.guild
    if not guild:
        return None
    vc = discord.utils.get(bot.voice_clients, guild=guild)
    return cast(VoiceRecvClient, vc) if vc else None


def render_service_health() -> tuple[str, bool]:
    """Render service health status."""
    checks = [
        ("Discord token", bool(TOKEN), "DISCORD_BOT_TOKEN"),
        ("OpenAI", bool(OPENAI_KEY), "OPENAI_API_KEY"),
        ("DeepL", bool(DEEPL_KEY), "DEEPL_API_KEY (optional)"),
        ("Voice recv", hasattr(VoiceRecvClient, "listen"), "discord-ext-voice-recv"),
    ]

    lines = []
    for name, ok, detail in checks:
        mark = "✅" if ok else "❌"
        lines.append(f"{mark} {name} ({detail})")
    
    # Only require Discord + OpenAI
    all_ok = bool(TOKEN) and bool(OPENAI_KEY) and hasattr(VoiceRecvClient, "listen")
    return "\n".join(lines), all_ok


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
        await ctx.send(f"❌ Missing argument: `{error.param.name}`")
        return
    if isinstance(error, commands.BadArgument):
        await ctx.send(f"❌ Bad argument: {error}")
        return
    logger.error(f"Command error in {ctx.command}: {error}", exc_info=error)
    await ctx.send("❌ An unexpected error occurred.")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
    """Global handler for slash command errors."""
    logger.error(f"Slash command error: {error}", exc_info=error)
    msg = "❌ An unexpected error occurred."
    try:
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
    except discord.HTTPException:
        pass


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    """Handle voice state changes - disconnect if invoker leaves."""
    guild_id = member.guild.id
    
    # Check if the invoker left the voice channel
    if guild_id in active_sinks:
        sink = active_sinks[guild_id]
        if member.id == sink.invoker_id:
            # Invoker left or moved to a different channel
            bot_vc = discord.utils.get(bot.voice_clients, guild=member.guild)
            if bot_vc and bot_vc.channel:
                bot_channel_id = getattr(bot_vc.channel, "id", None)
                before_channel_id = getattr(before.channel, "id", None) if before.channel else None
                after_channel_id = getattr(after.channel, "id", None) if after.channel else None
                
                # Check if invoker left the bot's channel
                if before_channel_id and before_channel_id == bot_channel_id:
                    if not after.channel or after_channel_id != bot_channel_id:
                        logger.info(f"Invoker {member.display_name} left voice channel, disconnecting bot")
                        try:
                            cast(VoiceRecvClient, bot_vc).stop_listening()
                        except Exception as e:
                            logger.warning(f"Error stopping listener: {e}")
                        try:
                            await bot_vc.disconnect(force=True)
                        except Exception as e:
                            logger.warning(f"Error disconnecting: {e}")
                        active_sinks.pop(guild_id, None)
                        return
    
    # Log bot's own voice state changes
    if bot.user and member.id == bot.user.id:
        if before.channel and not after.channel:
            logger.info(f"Bot disconnected from voice in {member.guild.name}")
            active_sinks.pop(guild_id, None)
        elif before.channel and after.channel and before.channel.id != after.channel.id:
            logger.info(f"Bot moved from {before.channel.name} to {after.channel.name}")


@bot.tree.command(name="help", description="Show available commands")
async def slash_help(interaction: discord.Interaction):
    help_text = """
**Voice Commands:**
• `/tr` or `/join` — Join voice channel and start translating
• `/leave` — Leave voice and stop translating

**Backend Commands:**
• `/backend` — Show current translation backend
• `/backend_list` — List available backends
• `/backend_set <name>` — Switch backend (deepl, google, openai)
• `/backend_usage` — Show usage/quota info

**Voice Settings:**
• `/voice` — Show current TTS voice
• `/voice_list` — List available voices
• `/voice_set <name>` — Change TTS voice

**Utility:**
• `/ping` — Check bot latency
• `/health` — Show service status
• `/help` — Show this message

*Each server has its own backend and voice settings.*
"""
    await interaction.response.send_message(help_text, ephemeral=True)


@bot.tree.command(name="ping", description="Check latency")
async def slash_ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"Pong! {round(bot.latency * 1000)}ms", ephemeral=True)


@bot.tree.command(name="health", description="Show service readiness")
async def slash_health(interaction: discord.Interaction):
    report, ok = render_service_health()
    prefix = "All services look OK." if ok else "Some services are not ready."
    await interaction.response.send_message(f"{prefix}\n{report}", ephemeral=True)


@bot.tree.command(name="tr", description="Join voice channel and start translating")
async def slash_tr(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    if not interaction.guild:
        return await interaction.followup.send("❌ This command can only be used in a server.")
    
    member = interaction.guild.get_member(interaction.user.id)
    if not member or not member.voice or not member.voice.channel:
        return await interaction.followup.send("❌ Join a voice channel first!")
    
    target = member.voice.channel
    
    perms = target.permissions_for(interaction.guild.me)
    if not (perms.connect and perms.speak):
        return await interaction.followup.send("❌ I need Connect and Speak permissions!")

    vc = await get_voice_client(interaction)

    try:
        state = server_manager.get(interaction.guild.id)
        
        if vc and vc.is_connected():
            if vc.channel and vc.channel.id == target.id:
                await interaction.followup.send("✅ Already here and listening!")
                return
            await vc.move_to(target)
            await interaction.followup.send(f"📍 Moved to `{target}` — translating!")
        else:
            vc = cast(VoiceRecvClient, await target.connect(cls=VoiceRecvClient, timeout=VOICE_CONNECT_TIMEOUT_SEC))
            await interaction.followup.send(f"🎤 Joined `{target}` — translating with **{state.get_backend_name()}**!")
        
        sink = AudioSink(vc, interaction, state)
        vc.listen(sink)
        active_sinks[interaction.guild.id] = sink  # Track for invoker disconnect
        logger.info(f"Started translation in {target} (backend={state.get_backend_name()})")
    except asyncio.TimeoutError:
        await interaction.followup.send("❌ Timed out connecting to voice channel.")
    except discord.ClientException as e:
        logger.error(f"Voice connect error: {e}")
        await interaction.followup.send(f"❌ Connection error: {e}")


@bot.tree.command(name="join", description="Join voice channel and start translating (alias for /tr)")
async def slash_join(interaction: discord.Interaction):
    await slash_tr(interaction)


@bot.tree.command(name="leave", description="Leave voice and stop translating")
async def slash_leave(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    vc = await get_voice_client(interaction)
    if not vc or not vc.is_connected():
        return await interaction.followup.send("❌ Not in a voice channel.")
    
    try:
        vc.stop_listening()
    except Exception as e:
        logger.warning(f"Error stopping listener: {e}")
    
    try:
        await vc.disconnect()
    except Exception as e:
        logger.warning(f"Error disconnecting: {e}")
    
    # Clean up sink tracking
    if interaction.guild:
        active_sinks.pop(interaction.guild.id, None)
    
    await interaction.followup.send("👋 Left voice and stopped translation.")


@bot.tree.command(name="backend", description="Show current translation backend")
async def slash_backend(interaction: discord.Interaction):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    current = state.get_backend_name()
    await interaction.response.send_message(f"🌐 Current backend: **{current}**", ephemeral=True)


@bot.tree.command(name="backend_list", description="List available translation backends")
async def slash_backend_list(interaction: discord.Interaction):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    backends = list_backends()
    current = state.get_backend_name()
    lines = [f"• `{b}`{' ✅' if b == current else ''}" for b in backends]
    await interaction.response.send_message(f"📋 Available backends:\n" + "\n".join(lines), ephemeral=True)


@bot.tree.command(name="backend_set", description="Switch translation backend")
@discord.app_commands.describe(name="Backend name: deepl, google, or openai")
async def slash_backend_set(interaction: discord.Interaction, name: str):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    
    try:
        new_name = state.switch_backend(name)
        server_manager.save(interaction.guild.id)
        await interaction.response.send_message(f"✅ Switched to **{new_name}** backend!", ephemeral=True)
    except ValueError as e:
        await interaction.response.send_message(f"❌ {e}", ephemeral=True)
    except RuntimeError as e:
        await interaction.response.send_message(f"❌ Cannot switch: {e}", ephemeral=True)


@bot.tree.command(name="backend_usage", description="Show backend usage/quota info")
async def slash_backend_usage(interaction: discord.Interaction):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    current = state.get_backend_name()
    usage = state.get_usage()
    
    if not usage:
        return await interaction.response.send_message(
            f"📊 **{current}** — Usage info unavailable", ephemeral=True
        )
    
    lines = [f"📊 **{current}** Usage:"]
    if "character_count" in usage:
        lines.append(f"• Characters used: **{usage['character_count']:,}** / {usage['character_limit']:,}")
    if "character_percent" in usage:
        lines.append(f"• Usage: **{usage['character_percent']}%**")
    if "characters_remaining" in usage:
        lines.append(f"• Remaining: **{usage['characters_remaining']:,}** characters")
    if "note" in usage:
        lines.append(f"• {usage['note']}")
    if "model" in usage:
        lines.append(f"• Model: `{usage['model']}`")
    if "limit" in usage:
        lines.append(f"• Limit: {usage['limit']}")
    
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


@bot.tree.command(name="voice", description="Show current TTS voice")
async def slash_voice(interaction: discord.Interaction):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    current = state.get_voice()
    await interaction.response.send_message(f"🗣️ Current voice: **{current}**", ephemeral=True)


@bot.tree.command(name="voice_list", description="List available TTS voices")
async def slash_voice_list(interaction: discord.Interaction):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    current = state.get_voice()
    lines = [f"• `{v}`{' ✅' if v == current else ''}" for v in OPENAI_TTS_VOICES]
    await interaction.response.send_message(
        f"📋 Available voices:\n" + "\n".join(lines) + "\n\n*Male: echo, onyx, fable | Female: alloy, nova, shimmer*",
        ephemeral=True
    )


@bot.tree.command(name="voice_set", description="Change TTS voice")
@discord.app_commands.describe(name="Voice name: alloy, echo, fable, nova, onyx, shimmer")
async def slash_voice_set(interaction: discord.Interaction, name: str):
    if not interaction.guild:
        return await interaction.response.send_message("❌ Server only.", ephemeral=True)
    
    state = server_manager.get(interaction.guild.id)
    
    try:
        new_voice = state.switch_voice(name)
        server_manager.save(interaction.guild.id)
        await interaction.response.send_message(f"✅ Switched to **{new_voice}** voice!", ephemeral=True)
    except ValueError as e:
        await interaction.response.send_message(f"❌ {e}", ephemeral=True)


async def shutdown():
    """Gracefully disconnect from all voice channels."""
    logger.info("Shutting down...")
    for vc in bot.voice_clients:
        try:
            if hasattr(vc, 'stop_listening'):
                cast(VoiceRecvClient, vc).stop_listening()
            await vc.disconnect(force=True)
        except Exception as e:
            logger.warning(f"Error during shutdown disconnect: {e}")
    await bot.close()


async def main():
    """Main entry point."""
    loop = asyncio.get_running_loop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            pass
    
    try:
        if TOKEN is None:
            raise RuntimeError("DISCORD_BOT_TOKEN not set")
        await bot.start(TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid Discord token!")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main())
