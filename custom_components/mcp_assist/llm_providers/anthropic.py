"""Anthropic provider transport."""

from __future__ import annotations

from typing import Any
import uuid

from ..const import DEFAULT_MAX_TOKENS
from .base import (
    LLMProvider,
    normalize_tool_call_arguments,
    parse_tool_arguments,
    stringify_tool_arguments,
)

ANTHROPIC_UNSUPPORTED_TOOL_NAMES = {"analyze_image", "generate_image"}


class AnthropicProvider(LLMProvider):
    """Anthropic native Messages API transport."""

    transport_name = "anthropic_messages"
    supports_streaming = False

    def chat_url(self) -> str:
        """Return Anthropic's native Messages endpoint."""
        return f"{self.base_url}/v1/messages"

    def headers(self) -> dict[str, str]:
        """Build headers for Anthropic's native Messages API."""
        return {
            "x-api-key": self.settings.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build Anthropic's native Messages API payload."""
        del stream
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            if role == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    system_parts.append(content)
                continue

            if role == "tool":
                self.append_message(
                    anthropic_messages,
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": str(message.get("content") or ""),
                        }
                    ],
                )
                continue

            if role == "assistant" and message.get("tool_calls"):
                content_blocks = self.text_content_blocks(message.get("content"))
                content_blocks.extend(self.tool_use_blocks(message.get("tool_calls", [])))
                self.append_message(anthropic_messages, "assistant", content_blocks)
                continue

            if role in {"user", "assistant"}:
                self.append_message(
                    anthropic_messages,
                    role,
                    self.text_content_blocks(message.get("content")),
                )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max(1, int(self.max_tokens or DEFAULT_MAX_TOKENS)),
            "messages": anthropic_messages,
        }

        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        anthropic_tools = self.convert_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        return payload

    def convert_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Convert provider-neutral function tools to Anthropic tool schemas."""
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools or []:
            function = tool.get("function", {})
            name = function.get("name")
            if not name or name in ANTHROPIC_UNSUPPORTED_TOOL_NAMES:
                continue

            input_schema = function.get("parameters") or {
                "type": "object",
                "properties": {},
            }
            anthropic_tools.append(
                {
                    "name": name,
                    "description": function.get("description", ""),
                    "input_schema": input_schema,
                }
            )
        return anthropic_tools

    def response_to_tool_calls(
        self, content_blocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tool_use blocks to internal function-call shape."""
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            tool_calls.append(
                {
                    "id": block.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": stringify_tool_arguments(block.get("input") or {}),
                    },
                }
            )
        return tool_calls

    def parse_http_message(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return Anthropic content blocks as a provider-neutral assistant message."""
        content_blocks = [
            block for block in data.get("content", []) if isinstance(block, dict)
        ]
        text_blocks = [
            str(block.get("text") or "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_blocks).strip(),
        }
        tool_calls = self.response_to_tool_calls(content_blocks)
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def normalize_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize Anthropic-derived tool calls to internal format."""
        return normalize_tool_call_arguments(tool_calls)

    def build_tool_call_assistant_message(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        response_text: str = "",
    ) -> dict[str, Any]:
        """Build an assistant message that preserves Anthropic pre-tool text."""
        message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if response_text:
            message["content"] = response_text
        return message

    def append_message(
        self,
        messages: list[dict[str, Any]],
        role: str,
        content_blocks: list[dict[str, Any]],
    ) -> None:
        """Append an Anthropic message, merging adjacent messages with the same role."""
        if not content_blocks:
            return
        if messages and messages[-1].get("role") == role:
            messages[-1].setdefault("content", []).extend(content_blocks)
            return
        messages.append({"role": role, "content": content_blocks})

    def text_content_blocks(self, content: Any) -> list[dict[str, str]]:
        """Convert provider-neutral message content to Anthropic text blocks."""
        if isinstance(content, list):
            blocks: list[dict[str, str]] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                else:
                    text = str(item or "").strip()
                if text:
                    blocks.append({"type": "text", "text": text})
            return blocks

        text = str(content or "").strip()
        return [{"type": "text", "text": text}] if text else []

    def tool_use_blocks(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert internal tool calls to Anthropic tool_use blocks."""
        blocks: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name")
            if not name:
                continue
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                    "name": name,
                    "input": parse_tool_arguments(function.get("arguments")),
                }
            )
        return blocks
