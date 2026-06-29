"""llama.cpp provider transport."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp's OpenAI-compatible chat transport."""
