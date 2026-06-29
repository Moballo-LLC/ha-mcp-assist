"""Shared transport for OpenAI-compatible chat-completions providers."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Provider transport for OpenAI-compatible chat-completions APIs."""

    transport_name = "openai_chat"

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible chat-completions payload."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }

        if not self._uses_completion_token_limit():
            payload["temperature"] = self.temperature

        if self.max_tokens > 0:
            if self._uses_completion_token_limit():
                payload["max_completion_tokens"] = self.max_tokens
            else:
                payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        return payload

    def _uses_completion_token_limit(self) -> bool:
        """Return whether the model expects max_completion_tokens."""
        return self.model_name.startswith(("gpt-5", "o1"))
