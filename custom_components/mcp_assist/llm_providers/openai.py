"""OpenAI provider transport."""

from __future__ import annotations

from ..const import (
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    OPENAI_BASE_URL,
    SERVER_TYPE_OPENAI,
)
from .base import PromptCacheUsage, ProviderConfigField
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

    @property
    def uses_official_openai_api(self) -> bool:
        """Return whether this profile targets OpenAI's official API host."""
        official_base_urls = {
            OPENAI_BASE_URL.rstrip("/"),
            f"{OPENAI_BASE_URL.rstrip('/')}/v1",
        }
        return self.base_url.rstrip("/") in official_base_urls

    def apply_prompt_cache_hints(self, payload: dict[str, object]) -> dict[str, object]:
        """Apply OpenAI prompt-cache routing hints and streaming usage collection."""
        if not self.uses_official_openai_api:
            return payload

        prepared = dict(payload)
        if self.settings.prompt_cache_key:
            prepared["prompt_cache_key"] = self.settings.prompt_cache_key

        if prepared.get("stream") is True:
            stream_options = dict(prepared.get("stream_options") or {})
            stream_options["include_usage"] = True
            prepared["stream_options"] = stream_options

        return prepared

    def extract_prompt_cache_usage(
        self,
        data: dict[str, object],
    ) -> PromptCacheUsage | None:
        """Extract OpenAI cached prompt-token counts from response usage."""
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None

        prompt_details = usage.get("prompt_tokens_details")
        cached_tokens = None
        if isinstance(prompt_details, dict):
            cached_value = prompt_details.get("cached_tokens")
            if isinstance(cached_value, int):
                cached_tokens = cached_value

        prompt_tokens = usage.get("prompt_tokens")
        return PromptCacheUsage(
            input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            cached_tokens=cached_tokens,
            cache_read_tokens=cached_tokens,
        )

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
