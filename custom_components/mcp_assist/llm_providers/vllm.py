"""vLLM provider transport."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM's OpenAI-compatible chat transport."""
