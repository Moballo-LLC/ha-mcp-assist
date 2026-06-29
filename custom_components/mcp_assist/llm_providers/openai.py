"""OpenAI provider transport."""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI chat-completions transport."""
