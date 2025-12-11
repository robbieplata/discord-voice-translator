# OpenAI TTS voices (alloy, echo, fable, onyx, nova, shimmer)
# Male-sounding: echo, onyx, fable
# Female-sounding: alloy, nova, shimmer
OPENAI_TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_TTS_DEFAULT_VOICE = "nova"
OPENAI_TTS_MODEL = "tts-1"

# VAD (Voice Activity Detection) thresholds
SILENCE_DURATION_SEC = 0.75  # Wait longer before ending utterance (was 0.45)
MIN_UTTERANCE_SEC = 0.4  # Minimum speech duration to process (was 0.3)
RMS_SILENCE_THRESHOLD = 50  # Lower = more sensitive to quiet speech (was 60)
SPEECH_CONFIRMATION_SEC = 0.1  # Require this much speech to confirm not silence

# Audio format
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_WIDTH = 2  # 16-bit
BYTES_PER_SECOND = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH

# Retry settings
MAX_RETRIES = 2
RETRY_DELAY_SEC = 0.5

# STT hallucination filtering thresholds
NO_SPEECH_PROB_THRESHOLD = 0.4
AVG_LOGPROB_THRESHOLD = -1.0

# TTS playback
TTS_VOLUME = 0.8

# Voice connection
VOICE_CONNECT_TIMEOUT_SEC = 10.0

# OpenAI translation settings
OPENAI_TRANSLATION_MODEL = "gpt-4o-mini"
OPENAI_TRANSLATION_TEMPERATURE = 0.3
OPENAI_TRANSLATION_MAX_TOKENS = 500

# DeepL retry settings
DEEPL_MAX_RETRIES = 2
DEEPL_RETRY_DELAY_SEC = 0.3

# Language detection - Pinyin/Chinese markers
PINYIN_TONE_MARKS = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ"
PINYIN_WORDS = frozenset({
    "nihao", "ni hao", "nǐ hǎo", "xiexie", "xie xie", "xièxiè",
    "zaijian", "zai jian", "zàijiàn", "duibuqi", "dui bu qi", "duìbuqǐ",
    "hao", "hen hao", "bu hao", "ni ne", "wo shi", "ta shi",
    "mingbai", "zhidao", "keyi", "meiyou", "shenme", "weishenme",
    "qing", "xing", "ting", "dui", "cuo", "shi de", "bu shi",
})
ASCII_RATIO_THRESHOLD = 0.95  # Above this, likely English even if detected as Chinese

# Whisper STT prompt (helps with language detection)
WHISPER_PROMPT = "This audio contains English or Mandarin Chinese speech."