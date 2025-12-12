import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import discord
import numpy as np
from discord.ext import commands
from discord.ext.voice_recv import AudioSink as BaseAudioSink
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ..core.config import (
    MAX_PENDING_UTTERANCES,
    MAX_UTTERANCE_SEC,
    MIN_SPEECH_RMS,
    MIN_UTTERANCE_SEC,
    NOISE_FLOOR_ALPHA,
    PACKET_TIMEOUT_SEC,
    PLAYBACK_SILENCE_MAX_WAIT_SEC,
    RMS_SILENCE_THRESHOLD,
    SILENCE_DURATION_SEC,
    SPEECH_CONFIRMATION_SEC,
    SPEECH_THRESHOLD_MULTIPLIER,
    TTS_VOLUME,
    USE_AI_SENTENCE_DETECTION,
)
from ..core.constants import BYTES_PER_SECOND
from ..core.pipeline import TranslationPipeline, Utterance

if TYPE_CHECKING:
    from .state import ServerState

logger = logging.getLogger("voicebot")

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Shared async OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_KEY)


class _UserBuffer:
    """Per-user audio buffer with VAD state."""

    __slots__ = (
        "pcm",
        "silence_sec",
        "duration_sec",
        "speaking",
        "last_activity",
        "consecutive_speech_sec",
        "noise_floor",
        "speech_frames",
        "silence_frames",
        "pending_text",
        "pending_audio",
        "last_progress_log_at",
    )

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.silence_sec = 0.0
        self.duration_sec = 0.0
        self.speaking = False
        self.last_activity = time.monotonic()
        self.consecutive_speech_sec = 0.0
        self.noise_floor: float = float(RMS_SILENCE_THRESHOLD)
        self.speech_frames = 0
        self.silence_frames = 0
        self.pending_text: list[str] = []
        self.pending_audio: list[bytes] = []
        self.last_progress_log_at: float = 0.0


class AudioSink(BaseAudioSink):
    """
    Discord audio sink with VAD and translation pipeline.

    Flow:
        Discord Audio → VAD → Utterance Queue → Pipeline → Playback Queue → Discord Audio
    """

    def __init__(
        self,
        vc: discord.VoiceClient,
        ctx_or_int: Union[commands.Context, discord.Interaction],
        server_state: "ServerState",
    ):
        self.vc = vc
        self.ctx_or_int = ctx_or_int
        self.server_state = server_state
        self.buffers: Dict[int, _UserBuffer] = {}
        self.loop = vc.client.loop
        self.packet_counts: Dict[int, int] = {}

        self._utterance_queue: deque[tuple[Any, bytes]] = deque()
        self._playback_queue: deque[bytes] = deque()
        self._playing = False
        self._processing = False
        self._closed = False
        self._user_refs: Dict[int, Any] = {}  # Store user references for timeout finalization
        self._timeout_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._total_packets = 0
        self._last_packet_log = time.time()
        self._last_any_packet_at = time.monotonic()
        self._last_finalize_at: float = 0.0
        self._last_pipeline_done_at: float = 0.0
        self._last_playback_start_at: float = 0.0
        self._last_playback_end_at: float = 0.0
        self._utterance_seq: int = 0

        if isinstance(ctx_or_int, commands.Context):
            self.invoker_id = ctx_or_int.author.id
        else:
            self.invoker_id = ctx_or_int.user.id

        self._pipeline = TranslationPipeline(
            openai_client=openai_client,
            translator=server_state.translator,
            get_voice_fn=server_state.get_voice,
        )

        logger.info(
            "[audio] sink attached channel=%s backend=%s invoker=%s",
            getattr(vc.channel, "name", "?"),
            server_state.get_backend_name(),
            self.invoker_id,
        )

        # Start the timeout checker task
        self._timeout_task = self.loop.create_task(self._check_packet_timeouts())
        self._health_task = self.loop.create_task(self._health_heartbeat())

    def wants_opus(self) -> bool:
        return False

    def write(self, user: Any, data: Any) -> None:
        """Handle incoming audio packets with VAD."""
        try:
            if self._closed:
                logger.warning("[vad] write() called but sink is closed!")
                return

            if user is None:
                return

            user_id = getattr(user, "id", None)
            if user_id is None:
                return

            user_name = getattr(user, "name", "unknown")
            state = self.buffers.setdefault(user_id, _UserBuffer())
            self.packet_counts[user_id] = self.packet_counts.get(user_id, 0) + 1
            self._user_refs[user_id] = user  # Store user reference for timeout finalization

            # Track total packets and log periodically
            self._total_packets += 1
            now = time.time()
            if now - self._last_packet_log >= 5.0:
                logger.info(
                    "[vad] ✓ Received %d packets in last 5s from %d users (write() IS being called)",
                    self._total_packets,
                    len(self.buffers),
                )
                self._total_packets = 0
                self._last_packet_log = now

            pcm_bytes = self._extract_pcm(data)
            if not pcm_bytes:
                return

            self._last_any_packet_at = time.monotonic()

            state.pcm.extend(pcm_bytes)
            state.last_activity = time.monotonic()

            chunk_sec = len(pcm_bytes) / BYTES_PER_SECOND
            rms = self._rms(pcm_bytes)

            if not state.speaking:
                state.noise_floor = (
                    NOISE_FLOOR_ALPHA * rms + (1 - NOISE_FLOOR_ALPHA) * state.noise_floor
                )
                state.noise_floor = max(20, min(state.noise_floor, 200))

            speech_threshold = max(
                RMS_SILENCE_THRESHOLD, state.noise_floor * SPEECH_THRESHOLD_MULTIPLIER
            )
            is_speech = rms > speech_threshold

            if is_speech:
                state.speech_frames += 1
                state.silence_frames = 0
                state.consecutive_speech_sec += chunk_sec

                if not state.speaking and state.consecutive_speech_sec >= SPEECH_CONFIRMATION_SEC:
                    state.speaking = True
                    state.silence_sec = 0.0
                    logger.debug(
                        "[vad] %s started speaking (rms=%.0f, thresh=%.0f)",
                        user_name,
                        rms,
                        speech_threshold,
                    )
                elif state.speaking:
                    state.silence_sec = 0.0
            else:
                state.silence_frames += 1
                state.speech_frames = 0

                if state.speaking:
                    state.silence_sec += chunk_sec
                    if state.silence_sec >= SILENCE_DURATION_SEC:
                        logger.debug(
                            "[vad] %s silence=%.2fs, will finalize", user_name, state.silence_sec
                        )
                else:
                    state.consecutive_speech_sec = max(
                        0, state.consecutive_speech_sec - chunk_sec * 2
                    )

            state.duration_sec += chunk_sec

            if self._should_finalize(state):
                logger.info(
                    "[vad] ✓ %s finalizing: dur=%.2fs, silence=%.2fs, rms=%.0f",
                    user_name,
                    state.duration_sec,
                    state.silence_sec,
                    rms,
                )
                self._finalize_utterance(user, state)
            elif state.speaking and state.duration_sec > 0.5:
                # Log ongoing speech at most every ~2 seconds (avoid spamming)
                now_mono = time.monotonic()
                if now_mono - state.last_progress_log_at >= 2.0:
                    state.last_progress_log_at = now_mono
                    logger.debug(
                        "[vad] %s still speaking: dur=%.2fs, silence=%.2fs",
                        user_name,
                        state.duration_sec,
                        state.silence_sec,
                    )

        except Exception as exc:
            logger.error("[vad] CRITICAL: write() error: %s", exc, exc_info=True)
            # Don't crash the sink - continue processing other packets

    def _extract_pcm(self, data: Any) -> bytes:
        if hasattr(data, "pcm"):
            return getattr(data, "pcm") or b""
        elif isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        return b""

    def _should_finalize(self, state: _UserBuffer) -> bool:
        return (
            state.speaking
            and state.duration_sec >= MIN_UTTERANCE_SEC
            and (
                state.silence_sec >= SILENCE_DURATION_SEC or state.duration_sec >= MAX_UTTERANCE_SEC
            )
        )

    async def _check_packet_timeouts(self) -> None:
        """Background task to detect when users stop sending audio packets."""
        check_count = 0
        while not self._closed:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                if self._closed:
                    break

                check_count += 1
                now = time.monotonic()
                users_to_finalize = []

                # Log active speakers every 50 checks (5 seconds)
                if check_count % 50 == 0:
                    active_speakers = []
                    for user_id, state in self.buffers.items():
                        if state.speaking or (now - state.last_activity < 1.0):
                            user = self._user_refs.get(user_id)
                            user_name = getattr(user, "name", "?") if user else "?"
                            active_speakers.append(
                                f"{user_name}(speaking={state.speaking}, dur={state.duration_sec:.1f}s)"
                            )
                    if active_speakers:
                        logger.debug("[vad] Active speakers: %s", ", ".join(active_speakers))

                for user_id, state in list(self.buffers.items()):
                    if state.speaking and state.duration_sec >= MIN_UTTERANCE_SEC:
                        time_since_packet = now - state.last_activity
                        if time_since_packet >= PACKET_TIMEOUT_SEC:
                            user = self._user_refs.get(user_id)
                            if user:
                                users_to_finalize.append((user, state, user_id))

                for user, state, user_id in users_to_finalize:
                    user_name = getattr(user, "name", "unknown")
                    logger.info(
                        "[vad] %s packet timeout (no audio for %.1fs) - finalizing",
                        user_name,
                        now - state.last_activity,
                    )
                    self._finalize_utterance(user, state)

                # Also clear stale "speaking" flags for too-short utterances.
                # This prevents `_anyone_speaking()` from staying true forever (and blocking playback)
                # when someone makes a tiny noise that confirms speaking but never reaches MIN_UTTERANCE_SEC.
                for user_id, state in list(self.buffers.items()):
                    if not state.speaking:
                        continue
                    time_since_packet = now - state.last_activity
                    if time_since_packet < PACKET_TIMEOUT_SEC:
                        continue
                    if state.duration_sec < MIN_UTTERANCE_SEC:
                        user = self._user_refs.get(user_id)
                        user_name = getattr(user, "name", "unknown") if user else "unknown"
                        logger.debug(
                            "[vad] %s packet timeout with short utterance (dur=%.2fs) - resetting speaking state",
                            user_name,
                            state.duration_sec,
                        )
                        self.buffers[user_id] = _UserBuffer()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[vad] timeout checker error: %s", exc)

    def _finalize_utterance(self, user: Any, state: _UserBuffer) -> None:
        try:
            self._last_finalize_at = time.monotonic()
            trim_bytes = int(max(0, state.silence_sec - 0.1) * BYTES_PER_SECOND)
            if trim_bytes > 0 and trim_bytes < len(state.pcm):
                payload = bytes(state.pcm[:-trim_bytes])
            else:
                payload = bytes(state.pcm)

            user_id = getattr(user, "id", 0) or 0
            user_name = getattr(user, "name", "unknown")
            duration = len(payload) / BYTES_PER_SECOND
            rms = self._rms(payload)

            logger.info(
                "[vad] ▶ %s utterance finalized: %.2fs, %d bytes, rms=%.0f @ %.3f",
                user_name,
                duration,
                len(payload),
                rms,
                time.time(),
            )

            if user_id:
                self.buffers[user_id] = _UserBuffer()

            if self._closed:
                logger.debug("[vad] Skipping queue - sink closed")
                return

            self._utterance_queue.append((user, payload))
            logger.debug(
                "[vad] queue size: %d, processing=%s, closed=%s",
                len(self._utterance_queue),
                self._processing,
                self._closed,
            )

            # Safely schedule processing
            try:
                self.loop.call_soon_threadsafe(
                    lambda: self.loop.create_task(self._process_utterance_queue())
                )
            except RuntimeError as e:
                logger.warning("[vad] Could not schedule processing: %s", e)

        except Exception as exc:
            logger.error("[vad] CRITICAL: finalize error: %s", exc, exc_info=True)

    async def _process_utterance_queue(self) -> None:
        if self._processing:
            logger.debug("[queue] already processing, skipping")
            return

        self._processing = True
        logger.debug("[queue] started processing, queue size=%d", len(self._utterance_queue))
        try:
            while self._utterance_queue and not self._closed:
                user, pcm_bytes = self._utterance_queue.popleft()
                if self._closed:
                    break
                logger.debug("[queue] handling utterance for %s", getattr(user, "name", "?"))

                # Add timeout protection to prevent queue stalling
                try:
                    async with asyncio.timeout(30):  # 30 second timeout per utterance
                        await self._handle_utterance(user, pcm_bytes)
                except asyncio.TimeoutError:
                    user_name = getattr(user, "name", "unknown")
                    logger.error(
                        "[queue] utterance processing timeout for %s (30s limit)", user_name
                    )
                    await self._send_log(f"Processing timeout for {user_name}")
                except Exception as exc:
                    user_name = getattr(user, "name", "unknown")
                    logger.error(
                        "[queue] error processing utterance for %s: %s",
                        user_name,
                        exc,
                        exc_info=True,
                    )
                    # Continue processing queue despite error

                logger.debug("[queue] finished utterance, remaining=%d", len(self._utterance_queue))
        except asyncio.CancelledError:
            logger.debug("[audio] utterance processing cancelled")
        except Exception as exc:
            logger.error("[queue] CRITICAL queue processing error: %s", exc, exc_info=True)
        finally:
            self._processing = False
            logger.debug("[queue] processing complete")

    async def _handle_utterance(self, user: discord.Member, pcm_bytes: bytes) -> None:
        user_id = getattr(user, "id", 0) or 0
        user_name = getattr(user, "name", "unknown")

        try:
            if self._closed:
                logger.debug("[handle] skipping - sink closed")
                return

            if len(pcm_bytes) < 3000:
                logger.debug("[handle] skipping - too short (%d bytes)", len(pcm_bytes))
                return

            rms = self._rms(pcm_bytes)
            if rms < MIN_SPEECH_RMS:
                logger.warning(
                    "[handle] FILTERED low-energy audio (rms=%.1f < %d) - speak louder?",
                    rms,
                    MIN_SPEECH_RMS,
                )
                await self._send_log(
                    f"Audio too quiet from {user_name} (speak louder or check mic)"
                )
                return

            duration = len(pcm_bytes) / BYTES_PER_SECOND

            self._utterance_seq += 1
            utterance_id = f"{int(time.time() * 1000) % 1_000_000:06d}-{self._utterance_seq:04d}"

            guild_id = getattr(getattr(self.vc, "guild", None), "id", 0) or 0
            channel_name = getattr(getattr(self.vc, "channel", None), "name", None) or "?"

            logger.info(
                "[u:%s] handle start guild=%s channel=%s user=%s dur=%.1fs bytes=%d rms=%.0f",
                utterance_id,
                guild_id or "?",
                channel_name,
                user_name,
                duration,
                len(pcm_bytes),
                rms,
            )

            async with self._typing_context():
                logger.debug("[handle] checking sentence completion...")
                text_to_process = await self._check_sentence_completion(
                    user_id, pcm_bytes, duration, rms
                )
                if text_to_process is None:
                    logger.debug("[handle] sentence not complete, buffering")
                    return

                logger.debug("[handle] creating utterance object")
                utterance = Utterance(
                    pcm_bytes=pcm_bytes,
                    user_id=user_id or 0,
                    user_name=user_name,
                    duration_sec=duration,
                    rms=rms,
                    utterance_id=utterance_id,
                    guild_id=guild_id,
                    channel_name=channel_name,
                )

                # Callback to send message as soon as translation is ready (before TTS)
                translation_sent = False

                async def on_translation(translation):
                    nonlocal translation_sent
                    translation_sent = True
                    msg = f"[{user_name}] [{translation.source_lang}→{translation.target_lang}] {translation.original} → {translation.translated}"
                    await self._send_log(self._sanitize(msg))

                logger.debug("[handle] calling pipeline.process...")
                result = await self._pipeline.process(utterance, on_translation=on_translation)
                if not result:
                    logger.warning(
                        "[u:%s] pipeline returned no result user=%s", utterance_id, user_name
                    )
                    if translation_sent:
                        await self._send_log(
                            f"Translation succeeded but TTS/playback failed for {user_name} (see logs)"
                        )
                    else:
                        await self._send_log(
                            f"Failed to process speech from {user_name} (filtered or STT failed)"
                        )
                    return

                logger.debug("[handle] queueing playback (%d bytes)", len(result.audio_bytes))
                await self._queue_playback(result.audio_bytes)

                self._last_pipeline_done_at = time.monotonic()

                total_time = sum(result.timings.values())
                logger.info("[u:%s] handle done total=%.0fms", utterance_id, total_time)

        except asyncio.TimeoutError:
            logger.error("[audio] utterance processing timeout for %s", user_name, exc_info=True)
            await self._send_log(f"Translation timed out for {user_name}")
        except Exception as exc:
            logger.error(
                "[audio] utterance handler error for %s: %s", user_name, exc, exc_info=True
            )
            await self._send_log(f"Translation error for {user_name}: {str(exc)[:100]}")

    async def _check_sentence_completion(
        self, user_id: int, pcm_bytes: bytes, duration: float, rms: float
    ) -> Optional[str]:
        if not USE_AI_SENTENCE_DETECTION:
            logger.debug("[sentence] AI detection disabled, proceeding immediately")
            return ""

        # Use the SAME state object from VAD buffers to prevent state divergence
        state = self.buffers.setdefault(user_id, _UserBuffer()) if user_id else _UserBuffer()

        wav_bytes = await asyncio.to_thread(self._pipeline.pcm_to_wav, pcm_bytes)
        if not wav_bytes:
            return None

        try:
            transcription = await self._pipeline.transcribe(wav_bytes)
        except Exception:
            return None

        if not transcription.text:
            return None

        if state.pending_text:
            combined = " ".join(state.pending_text + [transcription.text])
            logger.info("  📝 Combined: '%s'", combined)
        else:
            combined = transcription.text

        at_limit = len(state.pending_text) >= MAX_PENDING_UTTERANCES
        is_complete = await self._pipeline.is_sentence_complete(combined)

        if is_complete or at_limit:
            if at_limit and not is_complete:
                logger.info("  📦 Safety limit reached")
            state.pending_text = []
            state.pending_audio = []
            return combined
        else:
            state.pending_text.append(transcription.text)
            state.pending_audio.append(pcm_bytes)
            logger.info("  ⏸ Incomplete - buffering (pending=%d)", len(state.pending_text))
            return None

    def _anyone_speaking(self) -> bool:
        now = time.monotonic()
        for state in self.buffers.values():
            if state.speaking or (now - state.last_activity < 0.5):
                return True
        return False

    async def _queue_playback(self, audio_bytes: bytes) -> None:
        try:
            if not self.vc or not self.vc.is_connected():
                logger.debug("[playback] not connected, skipping")
                return
            if self._closed:
                logger.debug("[playback] sink closed, skipping")
                return
            logger.debug(
                "[playback] queued audio (%d bytes), queue size=%d, playing=%s",
                len(audio_bytes),
                len(self._playback_queue),
                self._playing,
            )
            self._playback_queue.append(audio_bytes)
            # Process playback in a background task so utterance processing can't stall behind playback.
            if not self._playback_task or self._playback_task.done():
                self._playback_task = self.loop.create_task(self._process_playback_queue())
        except Exception as exc:
            logger.error("[playback] queue error: %s", exc, exc_info=True)

    async def _process_playback_queue(self) -> None:
        self._playing = True
        try:
            while self._playback_queue and not self._closed:
                if not self.vc or not self.vc.is_connected():
                    logger.debug("[playback] voice disconnected, aborting")
                    break

                tts_mode = "normal"
                try:
                    tts_mode = str(self.server_state.get_tts_mode()).lower()
                except Exception:
                    tts_mode = str(getattr(self.server_state, "tts_mode", "normal")).lower()

                if tts_mode != "rude":
                    logger.debug("[playback] checking if anyone speaking...")
                    wait_start = time.monotonic()
                    while self._anyone_speaking() and not self._closed:
                        if (time.monotonic() - wait_start) >= PLAYBACK_SILENCE_MAX_WAIT_SEC:
                            logger.warning(
                                "[playback] waited %.1fs for silence (max=%.1fs) - proceeding to avoid stall",
                                time.monotonic() - wait_start,
                                PLAYBACK_SILENCE_MAX_WAIT_SEC,
                            )
                            break
                        await asyncio.sleep(0.1)
                    wait_time = time.monotonic() - wait_start
                    if wait_time > 0.2:
                        logger.debug("[playback] waited %.1fs for silence", wait_time)

                if not self._playback_queue or self._closed:
                    break

                audio_bytes = self._playback_queue.popleft()
                if self._closed or not self.vc or not self.vc.is_connected():
                    break

                logger.debug("[playback] starting playback (%d bytes)", len(audio_bytes))

                try:
                    self._last_playback_start_at = time.monotonic()
                    await self._play_audio(audio_bytes)

                    # Monitor playback and pause/resume on barge-in.
                    # IMPORTANT: VoiceRecvClient.stop() stops *listening* too.
                    # We only pause/resume playback so we keep receiving audio.
                    paused_by_barge_in = False
                    pause_started_at: float = 0.0
                    while self.vc and self.vc.is_connected() and not self._closed:
                        is_playing = bool(getattr(self.vc, "is_playing", lambda: False)())
                        is_paused = bool(getattr(self.vc, "is_paused", lambda: False)())

                        tts_mode = "normal"
                        try:
                            tts_mode = str(self.server_state.get_tts_mode()).lower()
                        except Exception:
                            tts_mode = str(getattr(self.server_state, "tts_mode", "normal")).lower()

                        # Finished (not playing and not paused)
                        if not is_playing and not is_paused:
                            break

                        if tts_mode == "rude":
                            await asyncio.sleep(0.05)
                            continue

                        if self._anyone_speaking():
                            if is_playing and not is_paused:
                                logger.info("[audio] ⏸ Someone speaking - pausing TTS playback")
                                try:
                                    pause_fn = getattr(self.vc, "pause", None)
                                    if callable(pause_fn):
                                        pause_fn()
                                        paused_by_barge_in = True
                                        pause_started_at = time.monotonic()
                                    else:
                                        # Fallback: if pause isn't available, stop playback only.
                                        stop_playing = getattr(self.vc, "stop_playing", None)
                                        if callable(stop_playing):
                                            stop_playing()
                                        else:
                                            self.vc.stop()
                                except Exception as exc:
                                    logger.warning("[playback] error pausing playback: %s", exc)
                        else:
                            if paused_by_barge_in and is_paused:
                                # Resume once the channel is quiet again.
                                wait_time = time.monotonic() - pause_started_at
                                if wait_time >= PLAYBACK_SILENCE_MAX_WAIT_SEC:
                                    logger.warning(
                                        "[audio] waited %.1fs to resume (max=%.1fs) - skipping TTS to avoid stall",
                                        wait_time,
                                        PLAYBACK_SILENCE_MAX_WAIT_SEC,
                                    )
                                    try:
                                        stop_playing = getattr(self.vc, "stop_playing", None)
                                        if callable(stop_playing):
                                            stop_playing()
                                        else:
                                            self.vc.stop()
                                    except Exception as exc:
                                        logger.warning(
                                            "[playback] error stopping paused playback: %s", exc
                                        )
                                    break

                                logger.info("[audio] ▶ Resuming TTS playback")
                                try:
                                    resume_fn = getattr(self.vc, "resume", None)
                                    if callable(resume_fn):
                                        resume_fn()
                                        paused_by_barge_in = False
                                    else:
                                        # No resume method; nothing else we can do.
                                        break
                                except Exception as exc:
                                    logger.warning("[playback] error resuming playback: %s", exc)
                                    break

                        await asyncio.sleep(0.05)

                except Exception as exc:
                    logger.error("[playback] play/monitor error: %s", exc)
                    if not self.vc or not self.vc.is_connected():
                        logger.warning("[playback] voice client disconnected")
                        break

                self._last_playback_end_at = time.monotonic()

        except asyncio.CancelledError:
            logger.debug("[audio] playback processing cancelled")
        except Exception as exc:
            logger.error("[playback] CRITICAL: process error: %s", exc, exc_info=True)
        finally:
            self._playing = False
            logger.debug("[playback] processing complete")

    async def _play_audio(self, audio_bytes: bytes) -> None:
        if self._closed or not self.vc or not self.vc.is_connected():
            return

        # Stop any existing playback
        try:
            is_playing = bool(getattr(self.vc, "is_playing", lambda: False)())
            is_paused = bool(getattr(self.vc, "is_paused", lambda: False)())
            if is_playing or is_paused:
                stop_playing = getattr(self.vc, "stop_playing", None)
                if callable(stop_playing):
                    stop_playing()
                else:
                    self.vc.stop()
                await asyncio.sleep(0.05)
        except Exception as exc:
            logger.warning("[audio] stop error: %s", exc)

        try:
            ffmpeg_source = discord.FFmpegPCMAudio(
                BytesIO(audio_bytes), pipe=True, before_options="-f mp3"
            )
            source = discord.PCMVolumeTransformer(ffmpeg_source, volume=TTS_VOLUME)

            # Replace cleanup with a complete no-op to prevent ANY cascade to sink.cleanup()
            # The FFmpeg process will still be cleaned up by the OS when it terminates
            ffmpeg_source.cleanup = lambda: None

            self.vc.play(source)
            logger.debug("[audio] playback started (%d bytes)", len(audio_bytes))
        except Exception as exc:
            logger.error("[audio] CRITICAL: playback failed: %s", exc, exc_info=True)
            raise

    async def _health_heartbeat(self) -> None:
        """Periodic health log to help diagnose stalls in production."""
        while not self._closed:
            try:
                await asyncio.sleep(30)
                if self._closed:
                    break

                now = time.monotonic()
                last_pkt_age = now - self._last_any_packet_at
                last_finalize_age = (now - self._last_finalize_at) if self._last_finalize_at else -1
                last_pipeline_age = (
                    (now - self._last_pipeline_done_at) if self._last_pipeline_done_at else -1
                )
                last_play_start_age = (
                    (now - self._last_playback_start_at) if self._last_playback_start_at else -1
                )
                last_play_end_age = (
                    (now - self._last_playback_end_at) if self._last_playback_end_at else -1
                )

                logger.info(
                    "[health] packets_age=%.1fs finalize_age=%s pipeline_age=%s "
                    "play_start_age=%s play_end_age=%s q_utt=%d q_play=%d processing=%s playing=%s closed=%s",
                    last_pkt_age,
                    f"{last_finalize_age:.1f}s" if last_finalize_age >= 0 else "n/a",
                    f"{last_pipeline_age:.1f}s" if last_pipeline_age >= 0 else "n/a",
                    f"{last_play_start_age:.1f}s" if last_play_start_age >= 0 else "n/a",
                    f"{last_play_end_age:.1f}s" if last_play_end_age >= 0 else "n/a",
                    len(self._utterance_queue),
                    len(self._playback_queue),
                    self._processing,
                    self._playing,
                    self._closed,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("[health] heartbeat error: %s", exc)

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        if len(pcm_bytes) < 2:
            return 0.0
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _sanitize(text: str) -> str:
        return text.replace("@", "@\u200b").replace("<", "\\<")[:1900]

    @asynccontextmanager
    async def _typing_context(self):
        channel = self.vc.channel if self.vc else None
        if channel and hasattr(channel, "typing"):
            try:
                async with channel.typing():
                    yield
            except Exception:
                yield
        else:
            yield

    async def _send_log(self, msg: str) -> None:
        try:
            if self.vc and self.vc.channel and hasattr(self.vc.channel, "send"):
                await self.vc.channel.send(msg)
                return

            if isinstance(self.ctx_or_int, commands.Context):
                await self.ctx_or_int.send(msg)
            else:
                await self.ctx_or_int.followup.send(msg)
        except Exception as exc:
            logger.warning("[audio] failed to send log: %s", exc)

    def cleanup(self) -> None:
        """Cleanup called by discord-ext-voice-recv. Made into a no-op to prevent interference.

        CRITICAL: This method is called by discord-ext-voice-recv when the voice connection
        closes OR when audio sources finish playing. We MUST NOT do anything destructive here
        because:
        1. It's called when playback is interrupted (e.g., VoiceRecvClient.stop_playing())
        2. Doing anything here can break the listening pipeline
        3. The sink must stay alive to continue receiving audio packets

        Actual cleanup is handled by shutdown() which is only called on explicit disconnect.
        """
        logger.debug("[audio] cleanup() called and ignored (no-op by design)")
        # Absolutely nothing happens here. The sink remains fully functional.

    def shutdown(self) -> None:
        """Explicitly shutdown the sink (prevents auto-reconnect)."""
        if self._closed:
            return

        self._closed = True
        logger.info("[audio] Explicit shutdown - marking sink as closed")

        # Cancel the timeout checker task on explicit shutdown
        try:
            if self._timeout_task and not self._timeout_task.done():
                logger.info("[audio] Cancelling timeout task on shutdown")
                self._timeout_task.cancel()
        except Exception as exc:
            logger.warning("[audio] timeout task cancel error: %s", exc)

        # Cancel playback task on shutdown
        try:
            if self._playback_task and not self._playback_task.done():
                logger.info("[audio] Cancelling playback task on shutdown")
                self._playback_task.cancel()
        except Exception as exc:
            logger.warning("[audio] playback task cancel error: %s", exc)

        # Cancel health task on shutdown
        try:
            if self._health_task and not self._health_task.done():
                logger.info("[audio] Cancelling health task on shutdown")
                self._health_task.cancel()
        except Exception as exc:
            logger.warning("[audio] health task cancel error: %s", exc)

        # Clear all buffers and queues
        try:
            self.buffers.clear()
            self._utterance_queue.clear()
            self._playback_queue.clear()
            self._user_refs.clear()
        except Exception as exc:
            logger.error("[audio] buffer clear error: %s", exc)

        # Call cleanup to stop tasks and playback
        self.cleanup()
