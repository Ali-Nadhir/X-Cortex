from __future__ import annotations

from .base import ReasoningBackend
from .medgemma import MedGemmaBackend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend


def build_backend(settings) -> ReasoningBackend:
    if settings.app.mode == "mock":
        return MockBackend()
    if settings.models.reasoner == "openai":
        return OpenAICompatBackend(settings)
    return MedGemmaBackend(settings)
