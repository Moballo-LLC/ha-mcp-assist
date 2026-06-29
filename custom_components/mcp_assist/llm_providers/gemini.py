"""Gemini OpenAI-compatible provider transport."""

from __future__ import annotations

from typing import Any

from .openai_compatible import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    """Gemini's OpenAI-compatible chat transport and metadata handling."""

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
