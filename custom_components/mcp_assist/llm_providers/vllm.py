"""vLLM provider transport."""

from __future__ import annotations

from ..const import CONF_LMSTUDIO_URL, DEFAULT_VLLM_URL, SERVER_TYPE_VLLM
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM's OpenAI-compatible chat transport."""

    provider_type = SERVER_TYPE_VLLM
    provider_display_name = "vLLM"
    default_base_url = DEFAULT_VLLM_URL
    connection_fields = (
        ProviderConfigField(CONF_LMSTUDIO_URL, default=DEFAULT_VLLM_URL),
    )
    model_fetch_error = "cannot_connect"
    model_fetch_timeout = 5
    model_fetch_delay = 0.5
