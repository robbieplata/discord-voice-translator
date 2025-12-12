import asyncio
import logging
import os
from typing import Dict, Optional, cast

import discord
from discord.ext import commands
from discord.ext.voice_recv import VoiceRecvClient

from ..backends import list_backends
from ..core.config import VOICE_CONNECT_TIMEOUT_SEC
from ..core.constants import OPENAI_TTS_VOICES
from .audio_sink import AudioSink
from .state import server_manager

logger = logging.getLogger("voicebot")

# Track active audio sinks per guild
active_sinks: Dict[int, AudioSink] = {}


def render_health() -> tuple[str, bool]:
    """Render service health status."""
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    DEEPL_KEY = os.getenv("DEEPL_API_KEY")

    checks = [
        ("Discord", bool(TOKEN)),
        ("OpenAI", bool(OPENAI_KEY)),
        ("DeepL", bool(DEEPL_KEY)),
        ("Voice recv", hasattr(VoiceRecvClient, "listen")),
    ]

    lines = [f"{'✓' if ok else 'X'} {name}" for name, ok in checks]
    all_ok = bool(TOKEN) and bool(OPENAI_KEY) and hasattr(VoiceRecvClient, "listen")
    return "\n".join(lines), all_ok


def setup_commands(bot: commands.Bot) -> None:
    """Register all slash commands on the bot."""

    async def _respond_ephemeral(interaction: discord.Interaction, message: str) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)

    async def _defer_ephemeral(interaction: discord.Interaction) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

    async def leave_voice(interaction: discord.Interaction) -> None:
        await _defer_ephemeral(interaction)

        if not interaction.guild:
            return await interaction.followup.send("Server only.", ephemeral=True)

        guild_id = interaction.guild.id

        # Shutdown sink FIRST to prevent auto-reconnect
        sink = active_sinks.get(guild_id)
        if sink:
            sink.shutdown()
            active_sinks.pop(guild_id, None)

        vc = await get_voice_client(bot, interaction)
        if not vc or not vc.is_connected():
            return await interaction.followup.send("Not in a voice channel.", ephemeral=True)

        try:
            vc.stop_listening()
        except Exception as exc:
            logger.warning("Error stopping listener: %s", exc)

        try:
            await vc.disconnect(force=True)
            await interaction.followup.send("Left voice and stopped translation.", ephemeral=True)
        except Exception as exc:
            logger.error("Error disconnecting: %s", exc, exc_info=True)
            await interaction.followup.send("Disconnected with errors.", ephemeral=True)

    async def backend_name_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> list[discord.app_commands.Choice[str]]:
        try:
            items = list_backends()
        except Exception:
            items = []
        cur = (current or "").lower()
        return [discord.app_commands.Choice(name=b, value=b) for b in items if cur in b.lower()][
            :25
        ]

    async def voice_name_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> list[discord.app_commands.Choice[str]]:
        cur = (current or "").lower()
        return [
            discord.app_commands.Choice(name=v, value=v)
            for v in OPENAI_TTS_VOICES
            if cur in v.lower()
        ][:25]

    # ─── Modern grouped commands: /translator ... ───────────────────────────

    translator_group = discord.app_commands.Group(
        name="translator",
        description="Voice translation controls",
    )

    @translator_group.command(name="join", description="Join voice and start translating")
    async def translator_join(interaction: discord.Interaction) -> None:
        await join_voice(bot, interaction)

    @translator_group.command(name="leave", description="Leave voice and stop translating")
    async def translator_leave(interaction: discord.Interaction) -> None:
        await leave_voice(interaction)

    @translator_group.command(name="status", description="Show current voice/translation status")
    async def translator_status(interaction: discord.Interaction) -> None:
        await _defer_ephemeral(interaction)

        if not interaction.guild:
            return await interaction.followup.send("Server only.", ephemeral=True)

        guild_id = interaction.guild.id
        state = server_manager.get(guild_id)
        vc = await get_voice_client(bot, interaction)
        sink = active_sinks.get(guild_id)

        channel_name = None
        if vc and vc.is_connected() and getattr(vc, "channel", None):
            channel_name = getattr(vc.channel, "name", None)

        invoker_id = getattr(sink, "invoker_id", None) if sink else None
        invoker_display = None
        if invoker_id and interaction.guild:
            invoker_member = interaction.guild.get_member(int(invoker_id))
            invoker_display = invoker_member.display_name if invoker_member else f"<@{invoker_id}>"

        lines: list[str] = []
        lines.append(f"Backend: **{state.get_backend_name()}**")
        lines.append(f"Voice: **{state.get_voice()}**")
        lines.append(f"TTS mode: **{state.get_tts_mode()}**")
        if channel_name:
            lines.append(f"Connected: yes (`{channel_name}`)")
        else:
            lines.append("Connected: no")
        lines.append(
            f"Listening: {'yes' if sink and not getattr(sink, '_closed', False) else 'no'}"
        )
        if invoker_display:
            lines.append(f"Invoker: {invoker_display}")

        await interaction.followup.send("\n".join(lines), ephemeral=True)

    @translator_group.command(name="help", description="Show translator commands")
    async def translator_help(interaction: discord.Interaction) -> None:
        text = (
            "**Translator Commands:**\n"
            "• `/translator join` — Join voice and start translating\n"
            "• `/translator leave` — Leave voice\n"
            "• `/translator status` — Current status\n"
            "• `/translator ping` — Check latency\n"
            "• `/translator health` — Service status\n"
            "• `/translator backend show|list|set|usage` — Backend controls\n"
            "• `/translator voice show|list|set` — TTS voice controls\n"
            "• `/translator tts show|set` — TTS playback mode\n"
        )
        await _respond_ephemeral(interaction, text)

    @translator_group.command(name="ping", description="Check latency")
    async def translator_ping(interaction: discord.Interaction) -> None:
        await _respond_ephemeral(interaction, f"Pong! {round(bot.latency * 1000)}ms")

    @translator_group.command(name="health", description="Show service status")
    async def translator_health(interaction: discord.Interaction) -> None:
        report, ok = render_health()
        prefix = "All services ready." if ok else "Some services unavailable."
        await _respond_ephemeral(interaction, f"{prefix}\n{report}")

    backend_group = discord.app_commands.Group(
        name="backend",
        description="Translation backend controls",
        parent=translator_group,
    )

    @backend_group.command(name="show", description="Show current translation backend")
    async def translator_backend_show(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        await _respond_ephemeral(interaction, f"Current backend: **{state.get_backend_name()}**")

    @backend_group.command(name="list", description="List available translation backends")
    async def translator_backend_list(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        current = state.get_backend_name()
        lines = [f"• `{b}`{' (current)' if b == current else ''}" for b in list_backends()]
        await _respond_ephemeral(interaction, "Available backends:\n" + "\n".join(lines))

    @backend_group.command(name="set", description="Switch translation backend")
    @discord.app_commands.describe(name="Translation backend to use")
    @discord.app_commands.autocomplete(name=backend_name_autocomplete)
    async def translator_backend_set(interaction: discord.Interaction, name: str) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        try:
            new_name = state.switch_backend(name)
            server_manager.save(interaction.guild.id)
            await _respond_ephemeral(interaction, f"Switched to **{new_name}** backend.")
        except ValueError as exc:
            await _respond_ephemeral(interaction, f"Error: {exc}")
        except RuntimeError as exc:
            await _respond_ephemeral(interaction, f"Error: cannot switch: {exc}")

    @backend_group.command(name="usage", description="Show backend usage/quota info")
    async def translator_backend_usage(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        current = state.get_backend_name()
        usage = state.get_usage()

        if not usage:
            return await _respond_ephemeral(interaction, f"{current}: usage info unavailable")

        lines = [f"{current} Usage:"]
        if "character_count" in usage:
            lines.append(
                f"• Characters: **{usage['character_count']:,}** / {usage.get('character_limit', 0):,}"
            )
        if "character_percent" in usage:
            lines.append(f"• Usage: **{usage['character_percent']}%**")
        if "characters_remaining" in usage:
            lines.append(f"• Remaining: **{usage['characters_remaining']:,}**")
        for key in ("note", "model", "limit"):
            if key in usage:
                lines.append(f"• {key.title()}: {usage[key]}")
        await _respond_ephemeral(interaction, "\n".join(lines))

    voice_group = discord.app_commands.Group(
        name="voice",
        description="TTS voice controls",
        parent=translator_group,
    )

    tts_group = discord.app_commands.Group(
        name="tts",
        description="TTS playback controls",
        parent=translator_group,
    )

    @tts_group.command(name="show", description="Show current TTS playback mode")
    async def translator_tts_show(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        await _respond_ephemeral(interaction, f"Current TTS mode: **{state.get_tts_mode()}**")

    @tts_group.command(name="set", description="Change TTS playback mode")
    @discord.app_commands.describe(mode="normal pauses on barge-in; rude ignores barge-in")
    @discord.app_commands.choices(
        mode=[
            discord.app_commands.Choice(name="normal", value="normal"),
            discord.app_commands.Choice(name="rude", value="rude"),
        ]
    )
    async def translator_tts_set(
        interaction: discord.Interaction,
        mode: discord.app_commands.Choice[str],
    ) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        try:
            new_mode = state.switch_tts_mode(mode.value)
            server_manager.save(interaction.guild.id)
            await _respond_ephemeral(interaction, f"Switched to TTS mode: **{new_mode}**")
        except ValueError as exc:
            await _respond_ephemeral(interaction, f"Error: {exc}")

    @voice_group.command(name="show", description="Show current TTS voice")
    async def translator_voice_show(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        await _respond_ephemeral(interaction, f"Current voice: **{state.get_voice()}**")

    @voice_group.command(name="list", description="List available TTS voices")
    async def translator_voice_list(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        current = state.get_voice()
        lines = [f"• `{v}`{' (current)' if v == current else ''}" for v in OPENAI_TTS_VOICES]
        await _respond_ephemeral(interaction, "Available voices:\n" + "\n".join(lines))

    @voice_group.command(name="set", description="Change TTS voice")
    @discord.app_commands.describe(name="TTS voice to use")
    @discord.app_commands.autocomplete(name=voice_name_autocomplete)
    async def translator_voice_set(interaction: discord.Interaction, name: str) -> None:
        if not interaction.guild:
            return await _respond_ephemeral(interaction, "Server only.")

        state = server_manager.get(interaction.guild.id)
        try:
            new_voice = state.switch_voice(name)
            server_manager.save(interaction.guild.id)
            await _respond_ephemeral(interaction, f"Switched to **{new_voice}** voice.")
        except ValueError as exc:
            await _respond_ephemeral(interaction, f"Error: {exc}")

    bot.tree.add_command(translator_group)

    # ─── Voice State Handler ──────────────────────────────────────────────────

    @bot.event
    async def on_voice_state_update(
        member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ):
        """Auto-disconnect when invoker leaves voice channel."""
        guild_id = member.guild.id

        if guild_id in active_sinks:
            sink = active_sinks[guild_id]
            if member.id == sink.invoker_id:
                bot_vc = discord.utils.get(bot.voice_clients, guild=member.guild)
                if bot_vc and bot_vc.channel:
                    bot_channel = getattr(bot_vc.channel, "id", None)
                    before_channel = before.channel.id if before.channel else None
                    after_channel = after.channel.id if after.channel else None

                    if before_channel == bot_channel and after_channel != bot_channel:
                        logger.info(f"Invoker {member.display_name} left, disconnecting")
                        # Shutdown FIRST to prevent auto-reconnect
                        sink.shutdown()
                        active_sinks.pop(guild_id, None)
                        try:
                            cast(VoiceRecvClient, bot_vc).stop_listening()
                            await bot_vc.disconnect(force=True)
                        except Exception as e:
                            logger.warning(f"Disconnect error: {e}")
                        return

        if bot.user and member.id == bot.user.id:
            if before.channel and not after.channel:
                logger.info(f"Bot disconnected from voice in {member.guild.name}")
                # Check if we had an active sink (unexpected disconnect)
                old_sink = active_sinks.pop(guild_id, None)
                logger.debug(
                    f"[auto-reconnect] old_sink exists: {old_sink is not None}, closed: {old_sink._closed if old_sink else 'N/A'}"
                )

                if old_sink and not old_sink._closed:
                    # Unexpected disconnect - try to reconnect
                    channel = before.channel
                    # Verify channel still exists and we have permissions
                    try:
                        can_connect = channel and channel.permissions_for(member.guild.me).connect
                    except Exception as exc:
                        logger.warning(f"[auto-reconnect] Permission check failed: {exc}")
                        can_connect = False

                    logger.debug(
                        f"[auto-reconnect] channel exists: {channel is not None}, can connect: {can_connect}"
                    )
                    if can_connect and channel:
                        logger.info(f"Attempting auto-reconnect to {channel.name}")
                        await asyncio.sleep(1)  # Brief delay before reconnect
                        try:
                            vc = cast(
                                VoiceRecvClient,
                                await channel.connect(
                                    cls=VoiceRecvClient, timeout=VOICE_CONNECT_TIMEOUT_SEC
                                ),
                            )
                            state = server_manager.get(guild_id)
                            sink = AudioSink(vc, old_sink.ctx_or_int, state)
                            vc.listen(sink)
                            active_sinks[guild_id] = sink
                            logger.info(f"Auto-reconnected to {channel.name}")
                        except asyncio.TimeoutError:
                            logger.error("Auto-reconnect timed out")
                        except Exception as e:
                            logger.error(f"Auto-reconnect failed: {e}", exc_info=True)
                    else:
                        logger.info("Channel no longer accessible, not reconnecting")
                else:
                    logger.debug(
                        "[auto-reconnect] Skipping reconnect - sink was marked closed or didn't exist"
                    )


# ─── Helper Functions ─────────────────────────────────────────────────────────


async def get_voice_client(
    bot: commands.Bot, interaction: discord.Interaction
) -> Optional[VoiceRecvClient]:
    """Get voice client for the guild."""
    if not interaction.guild:
        return None
    vc = discord.utils.get(bot.voice_clients, guild=interaction.guild)
    return cast(VoiceRecvClient, vc) if vc else None


async def join_voice(bot: commands.Bot, interaction: discord.Interaction):
    """Join voice channel and start the translation pipeline."""
    await interaction.response.defer(ephemeral=True)

    if not interaction.guild:
        return await interaction.followup.send("Server only.")

    member = interaction.guild.get_member(interaction.user.id)
    if not member or not member.voice or not member.voice.channel:
        return await interaction.followup.send("Join a voice channel first.")

    target = member.voice.channel
    perms = target.permissions_for(interaction.guild.me)
    if not (perms.connect and perms.speak):
        return await interaction.followup.send("I need Connect and Speak permissions.")

    vc = await get_voice_client(bot, interaction)
    guild_id = interaction.guild.id

    try:
        state = server_manager.get(guild_id)

        # Check if already listening in this channel
        if guild_id in active_sinks and vc and vc.is_connected():
            if vc.channel and vc.channel.id == target.id:
                return await interaction.followup.send("Already here and listening.")
            # Moving to different channel - stop old listener first
            try:
                old_sink = active_sinks.pop(guild_id, None)
                if old_sink:
                    old_sink.shutdown()
                vc.stop_listening()
            except Exception as exc:
                logger.warning("Error stopping old listener: %s", exc)

            try:
                await vc.move_to(target)
            except Exception as exc:
                logger.error("Failed to move channel: %s", exc)
                return await interaction.followup.send(f"Failed to move: {exc}")

        elif vc and vc.is_connected():
            # Connected but no active sink - stop any existing listener
            try:
                vc.stop_listening()
            except Exception:
                pass
            try:
                await vc.move_to(target)
            except Exception as exc:
                logger.error("Failed to move channel: %s", exc)
                return await interaction.followup.send(f"Failed to move: {exc}")
        else:
            # Not connected - join the channel
            try:
                vc = cast(
                    VoiceRecvClient,
                    await target.connect(cls=VoiceRecvClient, timeout=VOICE_CONNECT_TIMEOUT_SEC),
                )
            except asyncio.TimeoutError:
                logger.error("Voice connection timed out")
                return await interaction.followup.send("Timed out connecting to voice channel.")
            except discord.ClientException as e:
                logger.error(f"Voice connect error: {e}", exc_info=True)
                return await interaction.followup.send(f"Connection error: {e}")
            except Exception as e:
                logger.error(f"Unexpected voice error: {e}", exc_info=True)
                return await interaction.followup.send(f"Failed to connect: {e}")

        # Create new sink and start listening
        try:
            sink = AudioSink(vc, interaction, state)
            vc.listen(sink)
            active_sinks[guild_id] = sink
        except Exception as exc:
            logger.error(f"Failed to create sink: {exc}", exc_info=True)
            try:
                await vc.disconnect(force=True)
            except Exception:
                pass
            return await interaction.followup.send(f"Failed to start listening: {exc}")

        await interaction.followup.send(
            f"Listening in `{target}` — translating with **{state.get_backend_name()}**."
        )
        logger.info(f"Started translation in {target} (backend={state.get_backend_name()})")

    except Exception as e:
        logger.error(f"FATAL: join_voice error: {e}", exc_info=True)
        await interaction.followup.send(f"Unexpected error: {e}")
