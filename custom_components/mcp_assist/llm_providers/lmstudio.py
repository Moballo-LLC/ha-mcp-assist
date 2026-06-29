"""LM Studio provider transport."""

from __future__ import annotations

from ..const import CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL, SERVER_TYPE_LMSTUDIO
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio's OpenAI-compatible chat transport."""

    provider_type = SERVER_TYPE_LMSTUDIO
    provider_display_name = "LM Studio"
    default_base_url = DEFAULT_LMSTUDIO_URL
    connection_fields = (
        ProviderConfigField(CONF_LMSTUDIO_URL, default=DEFAULT_LMSTUDIO_URL),
    )
    model_fetch_error = "cannot_connect"
    model_fetch_timeout = 5
    model_fetch_delay = 0.5
