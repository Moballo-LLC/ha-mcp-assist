"""OpenRouter provider transport."""

from __future__ import annotations

from typing import Any

from ..const import CONF_API_KEY, OPENROUTER_BASE_URL, SERVER_TYPE_OPENROUTER
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter's OpenAI-compatible chat transport."""

    supports_thinking_mode = True
    provider_type = SERVER_TYPE_OPENROUTER
    provider_display_name = "OpenRouter"
    default_base_url = OPENROUTER_BASE_URL
    connection_fields = (ProviderConfigField(CONF_API_KEY, kind="password"),)
    model_fetch_error = "invalid_api_key"

    def build_payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        *, stream: bool = True,
    ) -> dict[str, Any]:
        """Use OpenRouter's unified reasoning control when requested."""
        payload = super().build_payload(messages, tools, stream=stream)
        if self.thinking_mode != "default":
            payload["reasoning"] = {
                "effort": "none" if self.thinking_mode == "off" else self.thinking_mode,
            }
        return payload
