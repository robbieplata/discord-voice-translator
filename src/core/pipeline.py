import asyncio
import logging
import time
import wave
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, Protocol, runtime_checkable

from openai import AsyncOpenAI

from .config import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH,
    AVG_LOGPROB_THRESHOLD,
    MAX_RETRIES,
    NO_SPEECH_PROB_THRESHOLD,
    OPENAI_SENTENCE_TIMEOUT_SEC,
    OPENAI_STT_TIMEOUT_SEC,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_TIMEOUT_SEC,
    OPENAI_WHISPER_MAX_AUDIO_BYTES,
    RETRY_DELAY_SEC,
    SENTENCE_COMPLETION_MODEL,
    TRANSLATE_TIMEOUT_SEC,
    USE_AI_SENTENCE_DETECTION,
    WHISPER_PROMPT,
)
from .constants import HALLUCINATION_PATTERNS, PINYIN_TONE_MARKS
from .openai_guardrails import limited_call, truncate_for_chat, truncate_for_tts

logger = logging.getLogger("voicebot")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Utterance:
    """Audio utterance with metadata."""

    pcm_bytes: bytes
    user_id: int
    user_name: str
    duration_sec: float = 0.0
    rms: float = 0.0
    utterance_id: str = ""
    guild_id: int = 0
    channel_name: str = ""


@dataclass
class Transcription:
    """STT result with confidence metrics."""

    text: str
    language: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def is_confident(self) -> bool:
        """Check if transcription passes confidence thresholds."""
        text = self.text.strip()
        if not text:
            return False
        if self.no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
            return False
        if self.avg_logprob < AVG_LOGPROB_THRESHOLD:
            return False
        # Substring match against hallucination patterns (watermarks, etc)
        if any(p in text.lower() for p in HALLUCINATION_PATTERNS):
            return False
        return True


@dataclass
class Translation:
    """Translation result."""

    original: str
    translated: str
    source_lang: str
    target_lang: str


@dataclass
class PipelineResult:
    """Final pipeline output."""

    utterance: Utterance
    transcription: Transcription
    translation: Translation
    audio_bytes: bytes
    timings: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Definitions (Interfaces)
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class AudioInput(Protocol):
    """Protocol for receiving audio input."""

    def on_audio_received(self, user_id: int, pcm_bytes: bytes) -> Optional[Utterance]:
        """Process audio packet, return Utterance when complete."""
        ...


@runtime_checkable
class AudioOutput(Protocol):
    """Protocol for playing audio output."""

    async def play(self, audio_bytes: bytes) -> None:
        """Play audio, waiting for silence first."""
        ...

    def is_anyone_speaking(self) -> bool:
        """Check if anyone is currently speaking."""
        ...


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for speech-to-text."""

    async def transcribe(self, wav_bytes: bytes) -> Transcription:
        """Convert audio to text."""
        ...


@runtime_checkable
class TranslationProvider(Protocol):
    """Protocol for translation."""

    async def translate(self, text: str, source_lang: str) -> Translation:
        """Translate text."""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for text-to-speech."""

    async def synthesize(self, text: str, voice: str) -> bytes:
        """Convert text to audio."""
        ...


@runtime_checkable
class SentenceDetector(Protocol):
    """Protocol for sentence completion detection."""

    async def is_complete(self, text: str) -> bool:
        """Check if text is a complete sentence."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Mixins - Reusable Pipeline Components
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAISTTMixin:
    """Mixin for OpenAI Whisper STT."""

    _openai_client: AsyncOpenAI

    async def transcribe(self, wav_bytes: bytes) -> Transcription:
        """Transcribe using OpenAI Whisper."""
        if len(wav_bytes) > OPENAI_WHISPER_MAX_AUDIO_BYTES:
            logger.warning(
                "[stt] WAV payload too large: %d bytes (max=%d) - skipping",
                len(wav_bytes),
                OPENAI_WHISPER_MAX_AUDIO_BYTES,
            )
            return Transcription(text="", language="en")

        with BytesIO(wav_bytes) as f:
            async with asyncio.timeout(OPENAI_STT_TIMEOUT_SEC):
                response = await limited_call(
                    lambda: self._openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("audio.wav", f),
                        response_format="verbose_json",
                        prompt=WHISPER_PROMPT,
                    )
                )

        text = (response.text or "").strip()
        detected_lang = getattr(response, "language", None) or "en"

        # Extract confidence metrics
        segments = getattr(response, "segments", []) or []
        if segments:
            no_speech_probs = [s.get("no_speech_prob", 0) for s in segments if isinstance(s, dict)]
            avg_logprobs = [s.get("avg_logprob", 0) for s in segments if isinstance(s, dict)]
            no_speech_prob = sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
            avg_logprob = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else 0.0
        else:
            no_speech_prob, avg_logprob = 0.0, 0.0

        # Post-process language detection
        detected_lang = self._adjust_language(text, detected_lang)

        return Transcription(
            text=text,
            language=detected_lang,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
        )

    @staticmethod
    def _adjust_language(text: str, detected_lang: str) -> str:
        """Post-process language detection."""
        import re

        if not text.strip():
            return detected_lang

        # CJK characters = definitive Chinese
        if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", text):
            return "zh"

        # Pinyin tone marks = definitive Chinese romanization
        if any(c in text for c in PINYIN_TONE_MARKS):
            return "zh"

        # ASCII text analysis
        ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
        lang = detected_lang.lower()

        # If mostly ASCII (>90%), apply better heuristics
        if ascii_ratio > 0.9:
            # Normalize English language codes first
            if lang.startswith("en") or lang == "english":
                return "en"

            # Normalize Chinese language codes
            if lang.startswith(("zh", "yue", "cmn")) or lang in (
                "chinese",
                "mandarin",
                "cantonese",
            ):
                # For ASCII Chinese, check if it's actually English misdetected
                # Pure ASCII with common English words = likely English
                text_lower = text.lower()
                common_english = [
                    "hello",
                    "test",
                    "yes",
                    "no",
                    "thank",
                    "please",
                    "sorry",
                    "okay",
                    "good",
                    "bad",
                ]
                if any(word in text_lower for word in common_english):
                    logger.info(
                        "[stt] English word detected in text marked as Chinese, correcting to English"
                    )
                    return "en"
                # Otherwise it's likely pinyin
                return "zh"

        return detected_lang


class OpenAITTSMixin:
    """Mixin for OpenAI TTS."""

    _openai_client: AsyncOpenAI

    async def synthesize(self, text: str, voice: str) -> bytes:
        """Synthesize speech using OpenAI TTS."""
        clean = text.strip()
        if not clean:
            return b""

        trunc = truncate_for_tts(clean)
        if trunc.truncated:
            logger.warning(
                "[tts] input too long (%d chars), truncating to %d chars",
                trunc.original_len,
                len(trunc.text),
            )
        clean = trunc.text

        try:
            async with asyncio.timeout(OPENAI_TTS_TIMEOUT_SEC):
                response = await limited_call(
                    lambda: self._openai_client.audio.speech.create(
                        model=OPENAI_TTS_MODEL,
                        voice=voice,
                        input=clean,
                        response_format="mp3",
                    )
                )
            return response.content
        except Exception as exc:
            # Re-raise so RetryMixin can back off on transient errors (e.g., 429s).
            logger.error("[tts] OpenAI TTS failed: %s", exc)
            raise


class AISentenceDetectorMixin:
    """Mixin for AI-based sentence completion detection."""

    _openai_client: AsyncOpenAI

    async def is_sentence_complete(self, text: str) -> bool:
        """Use AI to determine if text is a complete thought/sentence."""
        if not USE_AI_SENTENCE_DETECTION:
            return True

        text = text.strip()
        if not text:
            return True

        # Heuristics first - avoid API call when possible
        if text[-1] in ".!?。！？":
            return True

        word_count = len(text.split())
        if word_count <= 3:
            return True

        # Check for incomplete endings
        last_word = text.split()[-1].lower().rstrip(",.;:")
        incomplete_endings = {
            "and",
            "or",
            "but",
            "the",
            "a",
            "an",
            "to",
            "of",
            "in",
            "for",
            "with",
            "that",
            "which",
            "who",
            "if",
            "when",
            "because",
            "so",
            "then",
        }
        if word_count >= 5 and last_word not in incomplete_endings:
            return True

        # AI for ambiguous cases
        try:
            trunc = truncate_for_chat(text)
            if trunc.truncated:
                logger.debug(
                    "[sentence] input too long (%d chars), truncating to %d chars",
                    trunc.original_len,
                    len(trunc.text),
                )
            text = trunc.text
            async with asyncio.timeout(OPENAI_SENTENCE_TIMEOUT_SEC):
                response = await limited_call(
                    lambda: self._openai_client.chat.completions.create(
                        model=SENTENCE_COMPLETION_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "Determine if speech is complete or cut off. Output ONLY: COMPLETE or INCOMPLETE.",
                            },
                            {"role": "user", "content": text},
                        ],
                        temperature=0,
                        max_tokens=5,
                    )
                )
            result = (response.choices[0].message.content or "").strip().upper()
            return result == "COMPLETE"
        except Exception as exc:
            logger.warning("[sentence] AI check failed: %s", exc)
            return True


class AudioConverterMixin:
    """Mixin for audio format conversion."""

    @staticmethod
    def pcm_to_wav(pcm_bytes: bytes) -> bytes:
        """Convert PCM to WAV format."""
        try:
            buf = BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(AUDIO_SAMPLE_WIDTH)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(pcm_bytes)
            return buf.getvalue()
        except Exception as exc:
            logger.error("[audio] PCM→WAV failed: %s", exc)
            return b""

    @staticmethod
    def calculate_rms(pcm_bytes: bytes) -> float:
        """Calculate RMS (loudness) of audio."""
        import numpy as np

        if len(pcm_bytes) < 2:
            return 0.0
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(samples) == 0:
                return 0.0
            return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        except Exception:
            return 0.0


class RetryMixin:
    """Mixin for retry logic on API calls."""

    @staticmethod
    async def with_retry(
        coro_func, *args, retries: int = MAX_RETRIES, delay: float = RETRY_DELAY_SEC
    ):  # type: ignore
        """Execute async function with retry on transient errors."""
        from openai import APIConnectionError, APIError, RateLimitError

        last_error = None
        for attempt in range(retries + 1):
            try:
                return await coro_func(*args)
            except (APIConnectionError, RateLimitError) as exc:
                last_error = exc
                if attempt < retries:
                    logger.warning("  ↻ Retry %d/%d: %s", attempt + 1, retries, exc)
                    await asyncio.sleep(delay * (attempt + 1))
            except asyncio.TimeoutError as exc:
                last_error = exc
                if attempt < retries:
                    logger.warning("  ↻ Retry %d/%d: timeout", attempt + 1, retries)
                    await asyncio.sleep(delay * (attempt + 1))
            except APIError as exc:
                logger.error("  ✗ API error: %s", exc)
                raise

        if last_error:
            raise last_error


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Executor
# ═══════════════════════════════════════════════════════════════════════════════


class TranslationPipeline(
    OpenAISTTMixin,
    OpenAITTSMixin,
    AISentenceDetectorMixin,
    AudioConverterMixin,
    RetryMixin,
):
    """
    Main translation pipeline combining all mixins.

    Pipeline stages:
        1. Audio → WAV conversion
        2. WAV → Transcription (STT)
        3. Transcription → Sentence completion check
        4. Complete sentence → Translation
        5. Translation → Audio (TTS)
    """

    def __init__(self, openai_client: AsyncOpenAI, translator, get_voice_fn):
        self._openai_client = openai_client
        self._translator = translator
        self._get_voice = get_voice_fn

    async def process(self, utterance: Utterance, on_translation=None) -> Optional[PipelineResult]:
        """Process an utterance through the full pipeline.

        Args:
            utterance: The audio utterance to process
            on_translation: Optional async callback called when translation is ready (before TTS)

        Returns:
            PipelineResult on success, None on failure
        """
        pipeline_start = time.perf_counter()
        timings = {}
        u = utterance.utterance_id or "-"
        user = utterance.user_name
        guild_id = utterance.guild_id
        channel = utterance.channel_name or "?"

        def _clip(s: str, limit: int = 200) -> str:
            s = (s or "").replace("\n", " ").strip()
            if len(s) <= limit:
                return s
            return s[: max(0, limit - 1)] + "…"

        try:
            logger.info(
                "[u:%s] pipeline start guild=%s channel=%s user=%s dur=%.1fs rms=%.0f",
                u,
                guild_id or "?",
                channel,
                user,
                utterance.duration_sec,
                utterance.rms,
            )

            # Stage 1: PCM → WAV
            logger.debug("[u:%s] stage wav start", u)
            start = time.perf_counter()
            wav_bytes = await asyncio.to_thread(self.pcm_to_wav, utterance.pcm_bytes)
            timings["wav"] = (time.perf_counter() - start) * 1000
            logger.debug("[u:%s] stage wav done %.0fms", u, timings["wav"])

            if not wav_bytes:
                logger.warning("[u:%s] wav conversion failed", u)
                return None

            # Stage 2: STT (Transcribe)
            logger.debug("[u:%s] stage stt start", u)
            start = time.perf_counter()
            transcription: Optional[Transcription] = None
            try:
                result = await self.with_retry(self.transcribe, wav_bytes)
                if isinstance(result, Transcription):
                    transcription = result
            except Exception as exc:
                logger.error("[u:%s] stt failed: %s", u, exc)
                return None
            timings["stt"] = (time.perf_counter() - start) * 1000
            logger.debug("[u:%s] stage stt done %.0fms", u, timings["stt"])

            if not transcription or not transcription.is_confident:
                if transcription:
                    logger.info(
                        "[u:%s] filtered stt nsp=%.2f lp=%.2f text=%r",
                        u,
                        transcription.no_speech_prob,
                        transcription.avg_logprob,
                        transcription.text,
                    )
                return None

            logger.info(
                "[u:%s] stt %.0fms lang=%s text=%r",
                u,
                timings["stt"],
                transcription.language,
                _clip(transcription.text),
            )

            # Stage 3: Translate
            logger.debug("[u:%s] stage translate start", u)
            start = time.perf_counter()
            try:
                async with asyncio.timeout(TRANSLATE_TIMEOUT_SEC):
                    translated_text, target_lang = await asyncio.to_thread(
                        self._translator.translate,
                        transcription.text,
                        transcription.language,
                    )
            except asyncio.TimeoutError:
                logger.error(
                    "[u:%s] translate timeout after %.1fs backend=%s",
                    u,
                    TRANSLATE_TIMEOUT_SEC,
                    getattr(self._translator, "backend_name", "unknown"),
                )
                return None
            except Exception as exc:
                logger.error("[u:%s] translate failed: %s", u, exc, exc_info=True)
                # Re-raise to let caller handle the error properly
                raise
            timings["translate"] = (time.perf_counter() - start) * 1000
            logger.debug("[u:%s] stage translate done %.0fms", u, timings["translate"])

            translation = Translation(
                original=transcription.text,
                translated=translated_text,
                source_lang=transcription.language,
                target_lang=target_lang,
            )
            logger.info(
                "[u:%s] translate %.0fms %s→%s original=%r translated=%r",
                u,
                timings["translate"],
                translation.source_lang,
                translation.target_lang,
                _clip(translation.original),
                _clip(translation.translated),
            )

            # Call translation callback before TTS (so message appears immediately)
            if on_translation:
                try:
                    logger.debug("[u:%s] calling on_translation callback", u)
                    await on_translation(translation)
                except Exception as exc:
                    logger.warning("[u:%s] translation callback error: %s", u, exc)

            # Stage 4: TTS
            logger.debug("[u:%s] stage tts start", u)
            start = time.perf_counter()
            voice = self._get_voice()
            try:
                audio_bytes = await self.with_retry(self.synthesize, translation.translated, voice)
            except Exception as exc:
                logger.error("[u:%s] tts failed: %s", u, exc)
                return None
            timings["tts"] = (time.perf_counter() - start) * 1000
            logger.debug("[u:%s] stage tts done %.0fms", u, timings["tts"])

            if not audio_bytes:
                logger.warning("[u:%s] tts returned empty audio", u)
                return None

            logger.info(
                "[u:%s] tts %.0fms voice=%s bytes=%d",
                u,
                timings["tts"],
                voice,
                len(audio_bytes),
            )

            total_pipeline = (time.perf_counter() - pipeline_start) * 1000
            logger.info(
                "[u:%s] pipeline done total=%.0fms (wav=%.0f stt=%.0f translate=%.0f tts=%.0f)",
                u,
                total_pipeline,
                timings.get("wav", 0.0),
                timings.get("stt", 0.0),
                timings.get("translate", 0.0),
                timings.get("tts", 0.0),
            )

            return PipelineResult(
                utterance=utterance,
                transcription=transcription,
                translation=translation,
                audio_bytes=audio_bytes,
                timings=timings,
            )

        except Exception as exc:
            logger.error("[u:%s] pipeline fatal: %s", u, exc, exc_info=True)
            return None
