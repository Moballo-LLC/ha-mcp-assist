"""OpenClaw provider metadata for config flows."""

from __future__ import annotations

from typing import Any

from ..const import (
    CONF_OPENCLAW_AGENT_ID,
    CONF_OPENCLAW_SESSION_MODE,
    CONF_SYSTEM_PROMPT,
    CONF_OPENCLAW_HOST,
    CONF_OPENCLAW_PORT,
    CONF_OPENCLAW_SESSION_KEY,
    CONF_OPENCLAW_TOKEN,
    CONF_OPENCLAW_USE_SSL,
    DEFAULT_OPENCLAW_HOST,
    DEFAULT_OPENCLAW_AGENT_ID,
    DEFAULT_OPENCLAW_PORT,
    DEFAULT_OPENCLAW_SESSION_KEY,
    DEFAULT_OPENCLAW_USE_SSL,
    SERVER_TYPE_OPENCLAW,
    OPENCLAW_SESSION_MODE_FIXED,
    OPENCLAW_SESSION_MODE_PER_CONVERSATION,
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
    supports_thinking_mode = True
    supports_stateful_session_id_option = False
    uses_config_model_step = False
    uses_config_prompt_fields = False
    manages_agent_loop = True
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
            CONF_OPENCLAW_AGENT_ID, default=DEFAULT_OPENCLAW_AGENT_ID, required=False,
        ),
        ProviderConfigField(
            CONF_OPENCLAW_SESSION_MODE,
            default=OPENCLAW_SESSION_MODE_FIXED,
            required=False,
            kind="select",
            options=(OPENCLAW_SESSION_MODE_FIXED, OPENCLAW_SESSION_MODE_PER_CONVERSATION),
            translation_key=CONF_OPENCLAW_SESSION_MODE,
        ),
        ProviderConfigField(
            CONF_OPENCLAW_SESSION_KEY,
            default=DEFAULT_OPENCLAW_SESSION_KEY,
            required=False,
        ),
        ProviderConfigField(CONF_SYSTEM_PROMPT, default="", required=False),
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
