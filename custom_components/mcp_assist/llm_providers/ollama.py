"""Ollama provider transport."""

from __future__ import annotations

import json
from typing import Any

from ..const import (
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_NUM_CTX,
)
from .base import LLMProvider, StreamParseResult, parse_tool_arguments


class OllamaProvider(LLMProvider):
    """Ollama native chat transport."""

    transport_name = "ollama_chat"

    @classmethod
    def options_from_entry(cls, entry: Any) -> dict[str, Any]:
        """Return Ollama-specific options from a config entry."""
        keep_alive = cls._entry_value(
            entry,
            CONF_OLLAMA_KEEP_ALIVE,
            DEFAULT_OLLAMA_KEEP_ALIVE,
        )
        context_window = cls._entry_value(
            entry,
            CONF_OLLAMA_NUM_CTX,
            DEFAULT_OLLAMA_NUM_CTX,
        )
        try:
            context_window_int = int(context_window)
        except (TypeError, ValueError):
            context_window_int = int(DEFAULT_OLLAMA_NUM_CTX)
        return {
            "keep_alive": str(keep_alive),
            "context_window": context_window_int,
        }

    def chat_url(self) -> str:
        """Return Ollama's native chat endpoint."""
        return f"{self.base_url}/api/chat"

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build an Ollama native chat payload."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._build_messages(messages),
            "stream": stream,
            "keep_alive": self._keep_alive_value(),
            "options": {},
        }

        if self.temperature is not None:
            payload["options"]["temperature"] = self.temperature

        if self.max_tokens > 0:
            payload["options"]["num_predict"] = self.max_tokens

        context_window = self._context_window_value()
        if context_window > 0:
            payload["options"]["num_ctx"] = context_window

        if tools:
            payload["tools"] = tools

        return payload

    def parse_http_message(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return Ollama's direct assistant message."""
        return data.get("message", {})

    def parse_stream_line(self, line: str) -> StreamParseResult | None:
        """Return a normalized delta from an Ollama JSONL stream line."""
        if not line:
            return None

        data = json.loads(line)
        if data.get("done"):
            return StreamParseResult(delta={}, done=True)

        message = data.get("message", {})
        delta: dict[str, Any] = {}
        if message.get("content"):
            delta["content"] = message["content"]
        if "tool_calls" in message:
            delta["tool_calls"] = message["tool_calls"]
        return StreamParseResult(delta=delta)

    def normalize_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize tool calls to this provider's native argument shape."""
        return self.format_tool_calls(tool_calls)

    def format_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Convert one tool call to this provider's native argument shape."""
        formatted_call = dict(tool_call)
        function = dict(formatted_call.get("function", {}))
        if function:
            function["arguments"] = parse_tool_arguments(function.get("arguments"))
            formatted_call["function"] = function
        return formatted_call

    def build_tool_call_assistant_message(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        response_text: str = "",
    ) -> dict[str, Any]:
        """Build an Ollama assistant tool-call message."""
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": self.format_tool_calls(tool_calls),
        }
        if response_text:
            assistant_msg["content"] = response_text
        return assistant_msg

    def build_tool_result_message(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> dict[str, Any]:
        """Build Ollama's native tool-result message."""
        del tool_call_id
        return {
            "role": "tool",
            "tool_name": tool_name,
            "content": content,
        }

    def context_window_error_message(
        self,
        *,
        token_count: str | None = None,
        light_context_mode: bool = False,
    ) -> str:
        """Return Ollama-specific guidance for context-window errors."""
        token_text = f" The request was about {token_count} tokens." if token_count else ""
        if light_context_mode:
            return (
                "Ollama rejected the request because it exceeded the "
                f"model's context window.{token_text} Try raising the "
                "Ollama Context Window if the model supports it, "
                "reducing Max History Messages, or disabling optional "
                "tool families."
            )
        return (
            "Ollama rejected the request because it exceeded the "
            f"model's context window.{token_text} Enable Context Mode: "
            "Light for this profile, reduce Max History Messages, "
            "disable optional tool families, or raise the Ollama "
            "Context Window if the model supports it."
        )

    def model_unavailable_message(self, error_text: str) -> str | None:
        """Return Ollama-specific guidance for unloaded models."""
        if "model not loaded" in error_text or "pull the model" in error_text:
            return (
                f"The model '{self.model_name}' isn't loaded in Ollama. "
                f"Run 'ollama pull {self.model_name}' to download it first."
            )
        return None

    def _build_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert provider-neutral messages to Ollama's message shape."""
        ollama_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "tool":
                tool_message = {"role": "tool", "content": message.get("content", "")}
                if message.get("tool_name"):
                    tool_message["tool_name"] = message["tool_name"]
                ollama_messages.append(tool_message)
                continue

            if message.get("role") == "assistant" and message.get("tool_calls"):
                assistant_message = dict(message)
                assistant_message["tool_calls"] = self.format_tool_calls(
                    message["tool_calls"]
                )
                ollama_messages.append(assistant_message)
                continue

            ollama_messages.append(message)

        return ollama_messages

    def _keep_alive_value(self) -> int | str:
        """Return Ollama keep_alive as seconds or duration string."""
        keep_alive = self.settings.provider_options.get(
            "keep_alive",
            DEFAULT_OLLAMA_KEEP_ALIVE,
        )
        try:
            return int(keep_alive)
        except (TypeError, ValueError):
            return str(keep_alive)

    def _context_window_value(self) -> int:
        """Return the configured context-window override."""
        value = self.settings.provider_options.get(
            "context_window",
            DEFAULT_OLLAMA_NUM_CTX,
        )
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(DEFAULT_OLLAMA_NUM_CTX)

    @staticmethod
    def _entry_value(entry: Any, key: str, default: Any) -> Any:
        """Read a config-entry value from options first, then data."""
        value = entry.options.get(key, entry.data.get(key))
        return default if value is None else value
