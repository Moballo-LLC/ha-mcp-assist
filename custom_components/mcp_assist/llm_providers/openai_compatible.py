"""Shared transport for OpenAI-compatible chat-completions providers."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    """Provider transport for OpenAI-compatible chat-completions APIs."""

    transport_name = "openai_chat"
    openai_compatible_api_version = "v1"

    @classmethod
    def provider_endpoint(cls, base_url: str, endpoint: str) -> str:
        """Build an endpoint using this provider's OpenAI-compatible base path."""
        normalized_base_url = str(base_url or "").strip().rstrip("/")
        normalized_endpoint = str(endpoint or "").strip().lstrip("/")
        api_version = cls.openai_compatible_api_version.strip("/")

        if not api_version:
            return f"{normalized_base_url}/{normalized_endpoint}"
        if normalized_base_url.endswith(f"/{api_version}"):
            return f"{normalized_base_url}/{normalized_endpoint}"
        return f"{normalized_base_url}/{api_version}/{normalized_endpoint}"

    @classmethod
    def model_list_url(cls, values: dict[str, Any]) -> str:
        """Return the OpenAI-compatible models endpoint for this provider."""
        base_url = cls.model_base_url(values)
        return cls.provider_endpoint(base_url, "models") if base_url else ""

    def chat_url(self) -> str:
        """Return the OpenAI-compatible chat-completions endpoint."""
        return self.provider_endpoint(self.base_url, "chat/completions")

    def image_generation_url(self) -> str:
        """Return the OpenAI-compatible image-generation endpoint."""
        return self.provider_endpoint(self.base_url, "images/generations")

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
