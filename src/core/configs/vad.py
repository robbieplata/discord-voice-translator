from __future__ import annotations

SILENCE_DURATION_SEC = 1.2  # Wait for this long of silence before finalizing (debounce)
MIN_UTTERANCE_SEC = 0.1  # Minimum speech duration to process
MAX_UTTERANCE_SEC = 30.0
RMS_SILENCE_THRESHOLD = 50  # Minimum RMS to consider as speech (filters quiet noise)
SPEECH_THRESHOLD_MULTIPLIER = 2.0
NOISE_FLOOR_ALPHA = 0.05
SPEECH_CONFIRMATION_SEC = 0.1  # Require 100ms of speech to confirm speaking
MIN_SPEECH_RMS = 50  # Minimum RMS for utterance to be processed

# When we haven't received packets for this long, assume user stopped talking.
PACKET_TIMEOUT_SEC = 0.3
