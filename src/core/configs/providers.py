from __future__ import annotations

# ─── Google Translate (googletrans) Guardrails ───────────────────────────────
GOOGLE_MAX_REQUESTS_PER_SEC = 5.0
GOOGLE_MAX_INPUT_CHARS = 4500

# ─── DeepL API Guardrails ───────────────────────────────────────────────────
DEEPL_MAX_REQUEST_BYTES = 120 * 1024
DEEPL_MAX_REQUESTS_PER_SEC = 5.0
DEEPL_USAGE_LOG_EVERY_SEC = 300.0
DEEPL_USAGE_WARN_PERCENT = 90.0
