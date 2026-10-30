"""vLLM provider transport."""

from __future__ import annotations

from typing import Any

from ..const import CONF_API_KEY, CONF_LMSTUDIO_URL, DEFAULT_VLLM_URL, SERVER_TYPE_VLLM
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM's OpenAI-compatible chat transport."""

    provider_type = SERVER_TYPE_VLLM
    provider_display_name = "vLLM"
    default_base_url = DEFAULT_VLLM_URL
    supports_thinking_mode = True
    connection_fields = (
        ProviderConfigField(CONF_API_KEY, default="", required=False, kind="password"),
        ProviderConfigField(CONF_LMSTUDIO_URL, default=DEFAULT_VLLM_URL),
    )
    model_fetch_error = "cannot_connect"
    model_fetch_timeout = 5
    model_fetch_delay = 0.5

    def build_payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        *, stream: bool = True,
    ) -> dict[str, Any]:
        """Pass explicit thinking controls to the model's chat template."""
        payload = super().build_payload(messages, tools, stream=stream)
        if self.thinking_mode != "default":
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.thinking_mode != "off",
                "reasoning_effort": "low" if self.thinking_mode == "off" else self.thinking_mode,
            }
        return payload
