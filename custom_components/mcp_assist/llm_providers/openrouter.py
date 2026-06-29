"""OpenRouter provider transport."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter's OpenAI-compatible chat transport."""
