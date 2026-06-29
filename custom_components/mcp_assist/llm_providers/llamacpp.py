"""llama.cpp provider transport."""

from __future__ import annotations

from ..const import CONF_LMSTUDIO_URL, DEFAULT_LLAMACPP_URL, SERVER_TYPE_LLAMACPP
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class LlamaCppProvider(OpenAICompatibleProvider):
    """llama.cpp's OpenAI-compatible chat transport."""

    provider_type = SERVER_TYPE_LLAMACPP
    provider_display_name = "llama.cpp"
    default_base_url = DEFAULT_LLAMACPP_URL
    connection_fields = (
        ProviderConfigField(CONF_LMSTUDIO_URL, default=DEFAULT_LLAMACPP_URL),
    )
    model_fetch_error = "cannot_connect"
    model_fetch_timeout = 5
    model_fetch_delay = 0.5
