"""LM Studio provider transport."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio's OpenAI-compatible chat transport."""
