"""OpenAI provider transport."""

from __future__ import annotations

from ..const import (
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    OPENAI_BASE_URL,
    SERVER_TYPE_OPENAI,
)
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI chat-completions transport."""

    provider_type = SERVER_TYPE_OPENAI
    provider_display_name = "OpenAI"
    default_base_url = OPENAI_BASE_URL
    connection_fields = (
        ProviderConfigField(CONF_LMSTUDIO_URL, default=OPENAI_BASE_URL),
        ProviderConfigField(CONF_API_KEY, kind="password"),
    )
    model_fetch_error = "invalid_api_key"

    @classmethod
    def filter_model_ids(
        cls,
        model_ids: list[str],
        *,
        base_url: str,
    ) -> list[str]:
        """Filter official OpenAI results to chat models; keep custom endpoints broad."""
        official_base_urls = {
            OPENAI_BASE_URL.rstrip("/"),
            f"{OPENAI_BASE_URL.rstrip('/')}/v1",
        }
        if base_url.rstrip("/") in official_base_urls:
            model_ids = [model_id for model_id in model_ids if model_id.startswith("gpt-")]
        return sorted((model_id for model_id in model_ids if model_id), reverse=True)
