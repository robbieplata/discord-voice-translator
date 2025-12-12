from __future__ import annotations

NO_SPEECH_PROB_THRESHOLD = 0.3  # Lower = stricter (reject if Whisper thinks it's not speech)
AVG_LOGPROB_THRESHOLD = -0.8  # Higher = stricter (reject low-confidence transcriptions)
