"""OpenClaw provider metadata for config flows."""

from __future__ import annotations

from typing import Any

from ..const import (
    CONF_OPENCLAW_HOST,
    CONF_OPENCLAW_PORT,
    CONF_OPENCLAW_SESSION_KEY,
    CONF_OPENCLAW_TOKEN,
    CONF_OPENCLAW_USE_SSL,
    DEFAULT_OPENCLAW_HOST,
    DEFAULT_OPENCLAW_PORT,
    DEFAULT_OPENCLAW_SESSION_KEY,
    DEFAULT_OPENCLAW_USE_SSL,
    SERVER_TYPE_OPENCLAW,
)
from .base import LLMProvider, ProviderConfigField


class OpenClawProvider(LLMProvider):
    """OpenClaw Gateway config metadata.

    Runtime OpenClaw conversations bypass the HTTP LLM provider transport; this class
    exists so provider-specific config stays provider-owned.
    """

    provider_type = SERVER_TYPE_OPENCLAW
    provider_display_name = "OpenClaw"
    supports_streaming = False
    uses_config_model_step = False
    default_config_model_name = "main"
    default_config_system_prompt = ""
    default_config_technical_prompt = ""
    connection_fields = (
        ProviderConfigField(CONF_OPENCLAW_HOST, default=DEFAULT_OPENCLAW_HOST),
        ProviderConfigField(
            CONF_OPENCLAW_PORT,
            default=DEFAULT_OPENCLAW_PORT,
            kind="integer",
        ),
        ProviderConfigField(CONF_OPENCLAW_TOKEN, kind="password"),
        ProviderConfigField(
            CONF_OPENCLAW_USE_SSL,
            default=DEFAULT_OPENCLAW_USE_SSL,
            kind="boolean",
        ),
    )
    provider_options_fields = (
        ProviderConfigField(
            CONF_OPENCLAW_SESSION_KEY,
            default=DEFAULT_OPENCLAW_SESSION_KEY,
            required=False,
        ),
    )

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """OpenClaw does not use the provider HTTP payload path."""
        del messages, tools, stream
        raise RuntimeError("OpenClaw conversations bypass the LLM provider transport")
