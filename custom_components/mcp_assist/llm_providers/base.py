"""Provider transport contract for model-backed conversations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import re
from typing import Any

import aiohttp

from ..const import CONF_API_KEY, CONF_LMSTUDIO_URL
from ..provider_runtime import (
    build_provider_auth_headers,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSettings:
    """Runtime settings needed by provider transports."""

    server_type: str
    model_name: str
    api_key: str
    base_url: str
    timeout: int
    max_tokens: int
    temperature: float | None
    provider_options: dict[str, Any]
    display_name: str
    is_remote_service: bool
    prompt_cache_key: str | None = None


@dataclass(frozen=True)
class ProviderConfigField:
    """Provider-owned config metadata consumed by config and options flows."""

    key: str
    default: Any = None
    required: bool = True
    kind: str = "text"
    minimum: int | float | None = None
    maximum: int | float | None = None


@dataclass(frozen=True)
class StreamParseResult:
    """Provider-normalized streaming parse result."""

    delta: dict[str, Any]
    done: bool = False
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromptCacheUsage:
    """Provider-normalized prompt cache usage telemetry."""

    input_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None


def stringify_tool_arguments(arguments: Any) -> str:
    """Normalize tool arguments to a JSON string."""
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Parse tool arguments whether they arrive as a dict or JSON string."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_tool_call_arguments(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool_call function.arguments for internal and provider use."""
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        normalized_call = dict(tool_call)
        function = dict(normalized_call.get("function", {}))
        if function:
            function["arguments"] = stringify_tool_arguments(function.get("arguments"))
            normalized_call["function"] = function
        normalized.append(normalized_call)
    return normalized


def clean_json_payload(value: Any, *, strip_assistant_tool_content: bool = True) -> Any:
    """Remove None values and provider-invalid assistant tool content recursively."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            if key == "messages" and isinstance(item, list):
                cleaned_messages: list[dict[str, Any]] = []
                for message in item:
                    cleaned_message = clean_json_payload(
                        message,
                        strip_assistant_tool_content=strip_assistant_tool_content,
                    )
                    if (
                        strip_assistant_tool_content
                        and isinstance(cleaned_message, dict)
                        and cleaned_message.get("role") == "assistant"
                        and "tool_calls" in cleaned_message
                    ):
                        cleaned_message.pop("content", None)
                    cleaned_messages.append(cleaned_message)
                cleaned[key] = cleaned_messages
                continue
            cleaned[key] = clean_json_payload(
                item,
                strip_assistant_tool_content=strip_assistant_tool_content,
            )
        return cleaned

    if isinstance(value, list):
        return [
            clean_json_payload(
                item,
                strip_assistant_tool_content=strip_assistant_tool_content,
            )
            for item in value
        ]

    return value


class LLMProvider:
    """Base provider transport for a model server."""

    transport_name = "openai_chat"
    supports_streaming = True
    provider_type = ""
    provider_display_name = ""
    default_base_url: str | None = None
    connection_fields: tuple[ProviderConfigField, ...] = ()
    provider_options_fields: tuple[ProviderConfigField, ...] = ()
    model_fetch_error: str | None = None
    model_fetch_timeout = 10
    model_fetch_delay = 0.0
    uses_config_model_step = True
    default_temperature: float | None = None
    default_config_model_name = ""
    default_config_system_prompt = ""
    default_config_technical_prompt = ""

    def __init__(self, settings: ProviderSettings) -> None:
        """Initialize the provider transport."""
        self.settings = settings

    @classmethod
    def options_from_entry(cls, entry: Any) -> dict[str, Any]:
        """Return provider-specific options from a config entry."""
        del entry
        return {}

    @classmethod
    def config_display_name(cls) -> str:
        """Return the provider display name for UI surfaces."""
        return cls.provider_display_name or cls.__name__.removesuffix("Provider")

    @classmethod
    def config_value(
        cls,
        values: dict[str, Any],
        key: str,
        default: Any = None,
        *,
        blank_as_default: bool = False,
    ) -> Any:
        """Read a provider config value with optional blank-string fallback."""
        value = values.get(key, default)
        if value is None:
            return default
        if blank_as_default and isinstance(value, str) and not value.strip():
            return default
        return value

    @classmethod
    def model_base_url(cls, values: dict[str, Any]) -> str:
        """Return the base URL to use for OpenAI-compatible model listing."""
        base_url = cls.config_value(
            values,
            CONF_LMSTUDIO_URL,
            cls.default_base_url or "",
            blank_as_default=True,
        )
        return str(base_url or "").strip().rstrip("/")

    @classmethod
    def model_request_headers(cls, values: dict[str, Any]) -> dict[str, str]:
        """Return headers for model-list requests."""
        if not cls.provider_type:
            return {}
        api_key = str(cls.config_value(values, CONF_API_KEY, "") or "")
        return build_provider_auth_headers(cls.provider_type, api_key)

    @classmethod
    def filter_model_ids(
        cls,
        model_ids: list[str],
        *,
        base_url: str,
    ) -> list[str]:
        """Filter and sort provider model IDs for UI display."""
        del base_url
        models = [model_id for model_id in model_ids if model_id]
        return sorted(models)

    @classmethod
    def model_list_url(cls, values: dict[str, Any]) -> str:
        """Return the URL this provider uses to list supported models."""
        del values
        return ""

    @classmethod
    async def fetch_models(cls, hass: Any, values: dict[str, Any]) -> list[str]:
        """Fetch models from the provider-owned model-list endpoint."""
        del hass
        if cls.model_fetch_error is None:
            return []

        base_url = cls.model_base_url(values)
        if not base_url:
            return []

        models_url = cls.model_list_url(values)
        if not models_url:
            return []

        _LOGGER.info(
            "Starting %s model fetch from %s",
            cls.config_display_name(),
            _redacted_log_snippet(base_url),
        )
        try:
            if cls.model_fetch_delay > 0:
                await asyncio.sleep(cls.model_fetch_delay)

            timeout = aiohttp.ClientTimeout(total=cls.model_fetch_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    models_url,
                    headers=cls.model_request_headers(values),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        _LOGGER.warning(
                            "%s model fetch returned HTTP %d: %s",
                            cls.config_display_name(),
                            resp.status,
                            _redacted_log_snippet(error_text),
                        )
                        return []

                    data = await resp.json()
                    model_ids = [
                        str(model.get("id") or "")
                        for model in data.get("data", [])
                        if isinstance(model, dict)
                    ]
                    return cls.filter_model_ids(model_ids, base_url=base_url)
        except Exception as err:
            _LOGGER.error(
                "%s model fetch failed: %s",
                cls.config_display_name(),
                _redacted_log_snippet(err),
            )
            return []

    @property
    def server_type(self) -> str:
        """Return the configured server type."""
        return self.settings.server_type

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self.settings.model_name

    @property
    def base_url(self) -> str:
        """Return the provider base URL without trailing slash."""
        return self.settings.base_url.rstrip("/")

    @property
    def max_tokens(self) -> int:
        """Return the maximum response-token setting."""
        return self.settings.max_tokens

    @property
    def temperature(self) -> float | None:
        """Return the configured sampling temperature."""
        return self.settings.temperature

    @property
    def display_name(self) -> str:
        """Return the user-facing provider display name."""
        return self.settings.display_name

    @property
    def is_remote_service(self) -> bool:
        """Return whether the provider is a hosted remote service."""
        return self.settings.is_remote_service

    def headers(self) -> dict[str, str]:
        """Return provider request headers."""
        return build_provider_auth_headers(self.server_type, self.settings.api_key)

    def chat_url(self) -> str:
        """Return the provider chat endpoint."""
        raise NotImplementedError

    def image_generation_url(self) -> str:
        """Return the provider image-generation endpoint, if supported."""
        raise NotImplementedError

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build the provider request payload."""
        raise NotImplementedError

    def clean_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Clean a request payload before JSON serialization."""
        return clean_json_payload(payload)

    def apply_prompt_cache_hints(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply provider-specific prompt-cache hints before sending a payload."""
        return payload

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Prepare a request payload for transport."""
        return self.clean_payload(self.apply_prompt_cache_hints(payload))

    def extract_prompt_cache_usage(
        self,
        data: dict[str, Any],
    ) -> PromptCacheUsage | None:
        """Extract provider-specific prompt-cache usage from a response."""
        del data
        return None

    def prepare_messages_for_stream(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prepare messages for streaming transport."""
        prepared: list[dict[str, Any]] = []
        for message in messages:
            cleaned = dict(message)
            if cleaned.get("content") is None:
                cleaned["content"] = ""
            if cleaned.get("role") == "assistant" and cleaned.get("tool_calls"):
                cleaned.pop("content", None)
            prepared.append(cleaned)
        return prepared

    def parse_http_message(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a normalized assistant message from a non-streaming response."""
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"No response from {self.server_type}")
        return choices[0].get("message", {})

    def parse_stream_line(self, line: str) -> StreamParseResult | None:
        """Return a normalized delta from a provider stream line."""
        if not line.startswith("data: "):
            return None
        if line == "data: [DONE]":
            return StreamParseResult(delta={}, done=True)

        data = json.loads(line[6:])
        choices = data.get("choices") or []
        if not choices:
            return StreamParseResult(delta={}, usage=data.get("usage"))
        return StreamParseResult(delta=choices[0].get("delta", {}), usage=data.get("usage"))

    def normalize_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize provider tool calls to the internal function-call shape."""
        return self.format_tool_calls(tool_calls)

    def format_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Format one tool call for this provider's conversation history."""
        return normalize_tool_call_arguments([tool_call])[0]

    def format_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format tool calls for this provider's conversation history."""
        return [self.format_tool_call(tool_call) for tool_call in tool_calls]

    def build_tool_call_assistant_message(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        response_text: str = "",
    ) -> dict[str, Any]:
        """Build the assistant message to append before tool results."""
        return {"role": "assistant", "tool_calls": self.format_tool_calls(tool_calls)}

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> dict[str, Any]:
        """Build a tool-result message for this provider."""
        del tool_name
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    def update_stream_metadata(self, current: Any, delta: dict[str, Any]) -> Any:
        """Update provider-specific stream metadata from a delta."""
        return current

    def prepare_stream_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        metadata: Any,
    ) -> list[dict[str, Any]]:
        """Apply provider-specific stream metadata to completed tool calls."""
        return tool_calls

    def missing_stream_metadata_warning(self, metadata: Any) -> str | None:
        """Return a warning when provider-required stream metadata is absent."""
        return None

    def is_invalid_tool_arguments_error(
        self,
        *,
        status: int,
        error_text: str,
    ) -> bool:
        """Return whether a provider error reports malformed model tool arguments."""
        del status, error_text
        return False

    def context_window_error_message(
        self,
        *,
        token_count: str | None = None,
        light_context_mode: bool = False,
    ) -> str:
        """Return provider-specific guidance for context-window errors."""
        del light_context_mode
        if token_count:
            return (
                f"The conversation has exceeded the model's {token_count} token limit. "
                "Start a new conversation or reduce the history limit in Advanced Settings."
            )
        return (
            "The conversation has exceeded the model's token limit. Start a new "
            "conversation or reduce the history limit in Advanced Settings."
        )

    def model_unavailable_message(self, error_text: str) -> str | None:
        """Return provider-specific guidance for missing or unavailable models."""
        del error_text
        return None


def _redacted_log_snippet(value: Any, *, max_chars: int = 200) -> str:
    """Return a short provider log snippet with common secrets redacted."""
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+[^\s,;}]+", "Bearer [redacted]", text)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*)[^\s,;}]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"://[^/\s:@]+:[^@\s/]+@", "://[redacted]@", text)
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|api\s+key|password|secret|token|key)[\"']?"
        r"(?:\s+\w+){0,3}\s*[:=]\s*[\"']?)[^\"'\s,;}]+([\"']?)",
        r"\1[redacted]\2",
        text,
    )
    text = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[redacted]", text)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated {len(text) - max_chars} chars]"
