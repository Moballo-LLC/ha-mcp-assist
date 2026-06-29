"""OpenRouter provider transport."""

from __future__ import annotations

from ..const import CONF_API_KEY, OPENROUTER_BASE_URL, SERVER_TYPE_OPENROUTER
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter's OpenAI-compatible chat transport."""

    provider_type = SERVER_TYPE_OPENROUTER
    provider_display_name = "OpenRouter"
    default_base_url = OPENROUTER_BASE_URL
    connection_fields = (ProviderConfigField(CONF_API_KEY, kind="password"),)
    model_fetch_error = "invalid_api_key"
