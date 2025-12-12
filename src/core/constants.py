from __future__ import annotations

# ─── OpenAI TTS voices ───────────────────────────────────────────────────────

OPENAI_TTS_VOICES: list[str] = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_TTS_DEFAULT_VOICE: str = "nova"

# ─── Audio Format (Discord) ──────────────────────────────────────────────────

AUDIO_SAMPLE_RATE: int = 48000
AUDIO_CHANNELS: int = 2
AUDIO_SAMPLE_WIDTH: int = 2
BYTES_PER_SECOND: int = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH

# ─── Filtering / heuristics ─────────────────────────────────────────────────-

# Common Whisper hallucinations (substring match)
HALLUCINATION_PATTERNS: frozenset[str] = frozenset(
    {
        # Subtitle watermarks
        "amara.org",
        "amara org",
        "subtitles by amara",
        "subscriptionstart",
        # Chinese YouTube/video spam
        "请不吝点赞",
        "字幕提供",
        "視聴ありがとう",
        "谢谢观看",
        # Music/sound descriptions
        "♪",
        "[music]",
        "[applause]",
        "[laughter]",
    }
)

PINYIN_TONE_MARKS: str = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ"
