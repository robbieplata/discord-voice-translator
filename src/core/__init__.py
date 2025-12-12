# Re-export configuration for backward compatibility.
from .pipeline import PipelineResult, Transcription, Translation, TranslationPipeline, Utterance
from .translator import Translator

__all__ = [
    "TranslationPipeline",
    "Utterance",
    "Transcription",
    "Translation",
    "PipelineResult",
    "Translator",
]
