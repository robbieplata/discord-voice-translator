"""Audio handling with VAD, STT, translation, TTS pipeline."""

import asyncio
import wave
import time
import discord
from discord.ext import commands
from discord.ext.voice_recv import AudioSink as BaseAudioSink
import numpy as np
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from io import BytesIO
from collections import deque
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging
import unicodedata
from typing import Any, Dict, Union, TYPE_CHECKING
import re

from .config import (
    SILENCE_DURATION_SEC,
    MIN_UTTERANCE_SEC,
    RMS_SILENCE_THRESHOLD,
    BYTES_PER_SECOND,
    MAX_RETRIES,
    RETRY_DELAY_SEC,
    NO_SPEECH_PROB_THRESHOLD,
    AVG_LOGPROB_THRESHOLD,
    TTS_VOLUME,
    OPENAI_TTS_MODEL,
    PINYIN_TONE_MARKS,
    PINYIN_WORDS,
    ASCII_RATIO_THRESHOLD,
    WHISPER_PROMPT,
)

if TYPE_CHECKING:
    from .server_state import ServerState

logger = logging.getLogger("voicebot")

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

openai_client = OpenAI(api_key=OPENAI_KEY)

CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


class _UserBuffer:
    """Per-user audio buffer with VAD state."""
    __slots__ = ("pcm", "silence_sec", "duration_sec", "speaking", "last_activity", "consecutive_speech_sec")

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.silence_sec = 0.0
        self.duration_sec = 0.0
        self.speaking = False
        self.last_activity = time.monotonic()
        self.consecutive_speech_sec = 0.0  # Track continuous speech to confirm not silence


class AudioSink(BaseAudioSink):
    """Custom audio sink with VAD, STT, translation, and TTS pipeline."""

    def __init__(
        self, 
        vc: discord.VoiceClient, 
        ctx_or_int: Union[commands.Context, discord.Interaction],
        server_state: "ServerState"
    ):
        self.vc = vc
        self.ctx_or_int = ctx_or_int
        self.server_state = server_state  # Reference to server state for live backend switching
        self.buffers: Dict[int, _UserBuffer] = {}
        self.loop = vc.client.loop
        self.packet_counts: Dict[int, int] = {}
        self._playback_queue: deque[bytes] = deque()
        self._utterance_queue: deque[tuple[Any, bytes]] = deque()
        self._playing = False
        self._processing = False
        self._closed = False
        
        # Track who invoked the bot (for auto-disconnect)
        if isinstance(ctx_or_int, commands.Context):
            self.invoker_id = ctx_or_int.author.id
        else:
            self.invoker_id = ctx_or_int.user.id
        
        logger.info("[audio] sink attached to channel=%s (backend=%s, invoker=%s)", 
                    getattr(vc.channel, "name", "unknown"), server_state.get_backend_name(), self.invoker_id)

    def wants_opus(self) -> bool:
        return False

    def write(self, user: Any, data: Any) -> None:
        """Handle incoming audio packets with VAD."""
        if self._closed or user is None:
            return

        user_id = getattr(user, "id", None)
        if user_id is None:
            return

        state = self.buffers.setdefault(user_id, _UserBuffer())
        self.packet_counts[user_id] = self.packet_counts.get(user_id, 0) + 1

        # Extract PCM bytes
        pcm_bytes: bytes
        if hasattr(data, "pcm"):
            pcm_bytes = getattr(data, "pcm") or b""
        elif isinstance(data, (bytes, bytearray, memoryview)):
            pcm_bytes = bytes(data)
        else:
            return

        if not pcm_bytes:
            return

        state.pcm.extend(pcm_bytes)
        state.last_activity = time.monotonic()

        chunk_sec = len(pcm_bytes) / BYTES_PER_SECOND
        rms = self._rms(pcm_bytes)

        # Periodic debug logging (every ~100 packets)
        if self.packet_counts[user_id] % 100 == 0:
            logger.debug(
                "[audio] recv user=%s packets=%d rms=%.1f dur=%.2fs",
                user_id, self.packet_counts[user_id], rms, state.duration_sec
            )

        # VAD: detect speech vs silence with hysteresis
        if rms > RMS_SILENCE_THRESHOLD:
            state.speaking = True
            state.consecutive_speech_sec += chunk_sec
            # Only reset silence counter if we have confirmed speech (not just a blip)
            if state.consecutive_speech_sec >= 0.05:  # 50ms of speech resets silence
                state.silence_sec = 0.0
        else:
            state.consecutive_speech_sec = 0.0
            if state.speaking:  # Only count silence after speech started
                state.silence_sec += chunk_sec

        state.duration_sec += chunk_sec

        # Determine if utterance should be finalized
        # Require sustained silence after meaningful speech
        should_finalize = (
            state.speaking
            and state.duration_sec >= MIN_UTTERANCE_SEC
            and state.silence_sec >= SILENCE_DURATION_SEC
        )

        if should_finalize:
            # Trim trailing silence from audio (keep ~100ms for natural ending)
            trim_bytes = int(max(0, state.silence_sec - 0.1) * BYTES_PER_SECOND)
            payload = bytes(state.pcm[:-trim_bytes]) if trim_bytes > 0 and trim_bytes < len(state.pcm) else bytes(state.pcm)
            dur = state.duration_sec
            silence = state.silence_sec
            self.buffers[user_id] = _UserBuffer()

            logger.debug(
                "[audio] finalize user=%s len=%d dur=%.2fs silence=%.2fs",
                user_id, len(payload), dur, silence
            )

            # Queue utterance for sequential processing (maintains order)
            self._utterance_queue.append((user, payload))
            self.loop.call_soon_threadsafe(
                lambda: self.loop.create_task(self._process_utterance_queue())
            )

    async def _process_utterance_queue(self) -> None:
        """Process utterances sequentially to maintain order."""
        if self._processing:
            return
        
        self._processing = True
        try:
            while self._utterance_queue and not self._closed:
                user, pcm_bytes = self._utterance_queue.popleft()
                await self._safe_handle_utterance(user, pcm_bytes)
        finally:
            self._processing = False

    async def _safe_handle_utterance(self, user: discord.Member, pcm_bytes: bytes) -> None:
        """Wrapper to catch all exceptions in utterance handling."""
        try:
            await self._handle_utterance(user, pcm_bytes)
        except Exception as exc:
            logger.error("[audio] utterance handler crashed for %s: %s",
                        getattr(user, "name", "unknown"), exc, exc_info=True)

    async def _handle_utterance(self, user: discord.Member, pcm_bytes: bytes) -> None:
        """Process utterance: STT → translate → TTS → playback."""
        if len(pcm_bytes) < 3000:
            logger.debug("[audio] skipping tiny utterance (%d bytes)", len(pcm_bytes))
            return

        # Pre-check: skip if audio energy is too low (background noise)
        avg_rms = self._rms(pcm_bytes)
        if avg_rms < RMS_SILENCE_THRESHOLD * 1.5:
            logger.debug("[audio] skipping low-energy audio (rms=%.1f)", avg_rms)
            return

        user_name = getattr(user, "name", "unknown")
        total_start = time.perf_counter()
        audio_duration = len(pcm_bytes) / BYTES_PER_SECOND
        
        logger.info("━━━ [%s] Processing %.1fs audio (rms=%.0f) ━━━", user_name, audio_duration, avg_rms)

        # Show typing indicator while processing
        async with self._typing_context():
            # Step 1: Convert PCM to WAV
            step_start = time.perf_counter()
            wav_bytes = await asyncio.to_thread(self._pcm_to_wav_bytes, pcm_bytes)
            if not wav_bytes:
                logger.warning("  ✗ WAV conversion failed")
                return
            logger.debug("  ⏱ WAV: %.0fms", (time.perf_counter() - step_start) * 1000)

            # Step 2: Transcribe (STT)
            step_start = time.perf_counter()
            transcription, detected_lang, no_speech_prob, avg_logprob = await self._transcribe_with_retry(wav_bytes)
            stt_time = (time.perf_counter() - step_start) * 1000
            
            if not transcription:
                logger.info("  ✗ No transcription (silence)")
                return
            
            # Filter using Whisper's confidence metrics
            if no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
                logger.info("  ✗ Filtered (no_speech_prob=%.2f): '%s'", no_speech_prob, transcription)
                return
            
            if avg_logprob < AVG_LOGPROB_THRESHOLD:
                logger.info("  ✗ Filtered (low confidence lp=%.2f): '%s'", avg_logprob, transcription)
                return
            
            logger.info("  ⏱ STT: %.0fms → '%s' [%s] (nsp=%.2f, lp=%.2f)", 
                        stt_time, transcription, detected_lang, no_speech_prob, avg_logprob)

            # Step 3: Adjust language detection
            detected_lang = self._adjust_lang(transcription, detected_lang)

            # Step 4: Translate (using server's current translator)
            step_start = time.perf_counter()
            try:
                translated, target_lang = await asyncio.to_thread(
                    self.server_state.translator.translate, transcription, detected_lang
                )
            except Exception as exc:
                logger.error("  ✗ Translation failed: %s", exc)
                translated, target_lang = transcription, detected_lang
            
            translate_time = (time.perf_counter() - step_start) * 1000
            logger.info("  ⏱ Translate: %.0fms → '%s' [%s]", translate_time, translated, target_lang)

            # Step 5: Synthesize TTS
            step_start = time.perf_counter()
            tts_audio = await self._synthesize_with_retry(translated)
            tts_time = (time.perf_counter() - step_start) * 1000

            if not tts_audio:
                logger.warning("  ✗ TTS returned empty audio")
                return
            
            logger.info("  ⏱ TTS: %.0fms (%d bytes)", tts_time, len(tts_audio))

            # Step 6: Queue playback
            await self._queue_playback(tts_audio)

            # Summary
            total_time = (time.perf_counter() - total_start) * 1000
            logger.info("━━━ Total: %.0fms (STT:%.0f + Trans:%.0f + TTS:%.0f) ━━━", 
                        total_time, stt_time, translate_time, tts_time)

            # Send text log to channel - sanitize for Discord
            clean_transcription = self._sanitize_for_discord(transcription)
            clean_translated = self._sanitize_for_discord(translated)
            msg = f"🎙 [{user_name}] [{detected_lang}→{target_lang}] {clean_transcription} → {clean_translated}"
            await self._send_log(msg)

    @asynccontextmanager
    async def _typing_context(self):
        """Context manager to show typing indicator in voice channel."""
        channel = self.vc.channel if self.vc else None
        if channel and hasattr(channel, 'typing'):
            try:
                async with channel.typing():
                    yield
            except Exception:
                yield  # Fallback if typing fails
        else:
            yield

    async def _transcribe_with_retry(self, wav_bytes: bytes) -> tuple[str, str, float, float]:
        """Transcribe audio with retry on transient errors."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await asyncio.to_thread(self._transcribe, wav_bytes)
            except (APIConnectionError, RateLimitError) as exc:
                if attempt < MAX_RETRIES:
                    logger.warning("  ↻ STT retry %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
                    await asyncio.sleep(RETRY_DELAY_SEC * (attempt + 1))
                else:
                    logger.error("  ✗ STT failed: %s", exc)
                    return "", "en", 1.0, -1.0
            except APIError as exc:
                logger.error("  ✗ STT API error: %s", exc)
                return "", "en", 1.0, -1.0
            except Exception as exc:
                logger.error("  ✗ STT error: %s", exc)
                return "", "en", 1.0, -1.0
        return "", "en", 1.0, -1.0

    async def _synthesize_with_retry(self, text: str) -> bytes:
        """Synthesize TTS with retry on transient errors."""
        voice = self.server_state.get_voice()
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await asyncio.to_thread(self._synthesize_tts, text, voice)
                if result:
                    return result
            except (APIConnectionError, RateLimitError) as exc:
                if attempt < MAX_RETRIES:
                    logger.warning("  ↻ TTS retry %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
                    await asyncio.sleep(RETRY_DELAY_SEC * (attempt + 1))
                else:
                    logger.error("  ✗ TTS failed: %s", exc)
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    logger.warning("  ↻ TTS retry %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
                    await asyncio.sleep(RETRY_DELAY_SEC * (attempt + 1))
                else:
                    logger.error("  ✗ TTS failed: %s", exc)
        return b""

    async def _queue_playback(self, audio_bytes: bytes) -> None:
        """Queue audio for sequential playback."""
        if not self.vc.is_connected():
            return

        self._playback_queue.append(audio_bytes)

        if not self._playing:
            await self._process_playback_queue()

    async def _process_playback_queue(self) -> None:
        """Process queued audio sequentially."""
        self._playing = True
        try:
            while self._playback_queue and self.vc.is_connected():
                audio_bytes = self._playback_queue.popleft()
                await self._play_audio(audio_bytes)

                while self.vc.is_playing():
                    await asyncio.sleep(0.1)
        finally:
            self._playing = False

    async def _play_audio(self, audio_bytes: bytes) -> None:
        """Play audio through voice client."""
        if not self.vc.is_connected():
            return

        if self.vc.is_playing():
            self.vc.stop()
            await asyncio.sleep(0.05)

        ffmpeg_source = discord.FFmpegPCMAudio(BytesIO(audio_bytes), pipe=True)
        source = discord.PCMVolumeTransformer(ffmpeg_source, volume=TTS_VOLUME)

        original_cleanup = ffmpeg_source.cleanup
        def safe_cleanup():
            try:
                original_cleanup()
            except Exception as exc:
                logger.debug("[audio] ignored ffmpeg cleanup error: %s", exc)

        ffmpeg_source.cleanup = safe_cleanup

        try:
            self.vc.play(source)
            logger.debug("[audio] playback started (%d bytes)", len(audio_bytes))
        except discord.ClientException as exc:
            logger.error("[audio] playback failed: %s", exc)
        except Exception as exc:
            logger.error("[audio] unexpected playback error: %s", exc)

    async def _send_log(self, msg: str) -> None:
        """Send translation log to voice channel's text chat."""
        try:
            # Try to send to voice channel's text chat first
            if self.vc and self.vc.channel:
                voice_channel = self.vc.channel
                # Voice channels can be sent messages directly (voice channel text chat)
                if hasattr(voice_channel, 'send'):
                    await voice_channel.send(msg)
                    return
            
            # Fallback to command channel if voice channel send fails
            if isinstance(self.ctx_or_int, commands.Context):
                await self.ctx_or_int.send(msg)
            else:
                await self.ctx_or_int.followup.send(msg)
        except discord.HTTPException as exc:
            logger.warning("[audio] failed to send log message: %s", exc)
        except Exception as exc:
            logger.error("[audio] unexpected error sending log: %s", exc)

    @staticmethod
    def _sanitize_for_discord(text: str) -> str:
        """
        Sanitize text for Discord display.
        Removes control characters and normalizes unicode.
        """
        if not text:
            return text
        
        # Normalize unicode (NFC form for consistent display)
        text = unicodedata.normalize("NFC", text)
        
        # Remove control characters except newlines and tabs
        # Keep: letters, numbers, punctuation, spaces, CJK characters
        cleaned = []
        for char in text:
            category = unicodedata.category(char)
            # Keep: Letters (L*), Numbers (N*), Punctuation (P*), Symbols (S*), 
            # Separators (Z*: spaces), and specific format chars
            if category[0] in ('L', 'N', 'P', 'S', 'Z') or char in '\n\t':
                cleaned.append(char)
            elif category == 'Cf':  # Format characters - skip most
                continue
            # Skip control characters (Cc), surrogates (Cs), etc.
        
        result = ''.join(cleaned)
        
        # Collapse multiple spaces
        result = re.sub(r' +', ' ', result)
        
        return result.strip()

    @staticmethod
    def _rms(data: bytes) -> float:
        """Calculate RMS of PCM audio."""
        if not data:
            return 0.0
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        except Exception:
            return 0.0

    @staticmethod
    def _pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
        """Convert raw PCM to WAV format."""
        if not pcm_bytes:
            return b""

        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            if samples.size == 0:
                return b""

            try:
                stereo = samples.reshape(-1, 2)
                mono = stereo.mean(axis=1).astype(np.int16)
            except ValueError:
                mono = samples

            buf = BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(mono.tobytes())
            buf.seek(0)
            return buf.read()
        except Exception as exc:
            logger.error("[audio] PCM to WAV conversion error: %s", exc)
            return b""

    @staticmethod
    def _transcribe(wav_bytes: bytes) -> tuple[str, str, float, float]:
        """Transcribe audio using OpenAI Whisper with confidence metrics."""
        with BytesIO(wav_bytes) as f:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", f),
                response_format="verbose_json",
                prompt=WHISPER_PROMPT,
            )

        transcription = (response.text or "").strip()
        detected_lang = getattr(response, "language", None) or "en"
        
        segments = getattr(response, "segments", []) or []
        if segments:
            no_speech_probs = [s.get("no_speech_prob", 0) for s in segments if isinstance(s, dict)]
            avg_logprobs = [s.get("avg_logprob", 0) for s in segments if isinstance(s, dict)]
            no_speech_prob = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
            avg_logprob = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else 0.0
        else:
            no_speech_prob = 0.0
            avg_logprob = 0.0
        
        return transcription, detected_lang, no_speech_prob, avg_logprob

    @staticmethod
    def _adjust_lang(transcription: str, detected_lang: str) -> str:
        """Post-process language detection."""
        text = transcription.strip()
        if not text:
            return detected_lang

        has_cjk = bool(CJK_PATTERN.search(text))
        if has_cjk:
            return "zh"

        lower_text = text.lower()
        
        if any(c in text for c in PINYIN_TONE_MARKS):
            return "zh"
        
        clean_text = lower_text.replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ")
        words = set(clean_text.split())
        if words & PINYIN_WORDS:
            return "zh"
        for phrase in PINYIN_WORDS:
            if phrase in lower_text:
                return "zh"

        ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
        if ascii_ratio > ASCII_RATIO_THRESHOLD and detected_lang.lower().startswith(("zh", "yue", "cmn")):
            return "en"

        return detected_lang

    @staticmethod
    def _synthesize_tts(text: str, voice: str) -> bytes:
        """Synthesize speech using OpenAI TTS."""
        clean = text.strip()
        if not clean:
            return b""

        try:
            response = openai_client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=voice,
                input=clean,
                response_format="mp3",
            )
            return response.content
        except Exception as exc:
            logger.error("[audio] OpenAI TTS synthesis failed: %s", exc)
            return b""

    def cleanup(self) -> None:
        """Clean up resources."""
        self._closed = True
        self.buffers.clear()
        self._utterance_queue.clear()
        self._playback_queue.clear()
        logger.info("[audio] sink cleanup complete")
