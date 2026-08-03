"""Hermes Agent provider metadata for config flows."""

from __future__ import annotations

from typing import Any

from ..const import (
    CONF_API_KEY,
    CONF_HERMES_SESSION_KEY,
    CONF_HERMES_URL,
    DEFAULT_HERMES_MODEL,
    DEFAULT_HERMES_SESSION_KEY,
    DEFAULT_HERMES_URL,
    SERVER_TYPE_HERMES,
)
from .base import ProviderConfigField
from .openai_compatible import OpenAICompatibleProvider


class HermesProvider(OpenAICompatibleProvider):
    """Hermes Agent's server-managed agent transport metadata.

    Hermes exposes OpenAI-compatible model discovery, but conversations bypass
    MCP Assist's client-side tool loop because Hermes runs its own prompts,
    memory, and tools on the API-server host.
    """

    provider_type = SERVER_TYPE_HERMES
    provider_display_name = "Hermes Agent (experimental)"
    default_base_url = DEFAULT_HERMES_URL
    supports_streaming = False
    supports_stateful_session_id_option = False
    uses_config_prompt_fields = False
    manages_agent_loop = True
    default_config_model_name = DEFAULT_HERMES_MODEL
    default_config_system_prompt = ""
    default_config_technical_prompt = ""
    connection_fields = (
        ProviderConfigField(CONF_HERMES_URL, default=DEFAULT_HERMES_URL),
        ProviderConfigField(CONF_API_KEY, kind="password", required=False),
    )
    provider_options_fields = (
        ProviderConfigField(
            CONF_HERMES_SESSION_KEY,
            default=DEFAULT_HERMES_SESSION_KEY,
            required=False,
        ),
    )
    model_fetch_error = "cannot_connect"

    @classmethod
    def model_base_url(cls, values: dict[str, Any]) -> str:
        """Return the configured Hermes API-server URL."""
        base_url = cls.config_value(
            values,
            CONF_HERMES_URL,
            DEFAULT_HERMES_URL,
            blank_as_default=True,
        )
        return str(base_url or "").strip().rstrip("/")

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Hermes conversations use the dedicated server-agent client."""
        del messages, tools, stream
        raise RuntimeError("Hermes conversations bypass the MCP Assist tool loop")
