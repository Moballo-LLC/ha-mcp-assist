"""Provider transport contract for model-backed conversations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..provider_runtime import (
    build_openai_compatible_endpoint,
    build_provider_auth_headers,
)


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


@dataclass(frozen=True)
class StreamParseResult:
    """Provider-normalized streaming parse result."""

    delta: dict[str, Any]
    done: bool = False


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

    def __init__(self, settings: ProviderSettings) -> None:
        """Initialize the provider transport."""
        self.settings = settings

    @classmethod
    def options_from_entry(cls, entry: Any) -> dict[str, Any]:
        """Return provider-specific options from a config entry."""
        del entry
        return {}

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
        """Return the provider chat-completions endpoint."""
        return build_openai_compatible_endpoint(self.base_url, "chat/completions")

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
        choice = data["choices"][0]
        return StreamParseResult(delta=choice.get("delta", {}))

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
