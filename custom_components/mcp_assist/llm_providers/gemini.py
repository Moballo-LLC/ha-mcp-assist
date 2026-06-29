"""Gemini OpenAI-compatible provider transport."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from ..const import CONF_API_KEY, SERVER_TYPE_GEMINI
from .base import ProviderConfigField, _redacted_log_snippet
from .openai_compatible import OpenAICompatibleProvider

_LOGGER = logging.getLogger(__name__)


class GeminiProvider(OpenAICompatibleProvider):
    """Gemini's OpenAI-compatible chat transport and metadata handling."""

    provider_type = SERVER_TYPE_GEMINI
    provider_display_name = "Gemini"
    connection_fields = (ProviderConfigField(CONF_API_KEY, kind="password"),)
    model_fetch_error = "invalid_api_key"
    default_temperature = 1.0
    openai_compatible_api_version = ""

    @classmethod
    async def fetch_models(cls, hass: Any, values: dict[str, Any]) -> list[str]:
        """Fetch models from Gemini's native model-list endpoint."""
        del hass
        api_key = str(cls.config_value(values, CONF_API_KEY, "") or "")
        if not api_key:
            return []

        try:
            timeout = aiohttp.ClientTimeout(total=cls.model_fetch_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key},
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.warning(
                            "Gemini model fetch returned HTTP %d: %s",
                            resp.status,
                            _redacted_log_snippet(error_text),
                        )
                        return []

                    data = await resp.json()
                    model_ids = []
                    for model in data.get("models", []):
                        if not isinstance(model, dict):
                            continue
                        model_name = str(model.get("name") or "")
                        if model_name.startswith("models/"):
                            model_ids.append(model_name.removeprefix("models/"))
                    return sorted(
                        (
                            model_id
                            for model_id in model_ids
                            if "gemini" in model_id.lower()
                        ),
                        reverse=True,
                    )
        except Exception as err:
            _LOGGER.error("Gemini model fetch failed: %s", _redacted_log_snippet(err))
            return []

    def update_stream_metadata(self, current: Any, delta: dict[str, Any]) -> Any:
        """Capture Gemini thought signatures from streamed tool-call deltas."""
        if current is not None or "tool_calls" not in delta:
            return current

        for tool_call_delta in delta["tool_calls"]:
            google_data = tool_call_delta.get("extra_content", {}).get("google", {})
            if "thought_signature" in google_data:
                return google_data["thought_signature"]
        return current

    def prepare_stream_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        metadata: Any,
    ) -> list[dict[str, Any]]:
        """Add Gemini thought signatures to completed streamed tool calls."""
        if metadata is None:
            return tool_calls

        prepared: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            prepared_call = dict(tool_call)
            prepared_call["extra_content"] = {
                "google": {"thought_signature": metadata}
            }
            prepared.append(prepared_call)
        return prepared

    def missing_stream_metadata_warning(self, metadata: Any) -> str | None:
        """Warn when Gemini 3 streams tool calls without thought signatures."""
        if metadata is not None:
            return None
        return (
            "⚠️ No thought_signature captured for Gemini 3 "
            "(this will cause 400 error on next turn)"
        )
