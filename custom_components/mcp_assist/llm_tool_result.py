"""Compatibility helpers for Home Assistant LLM tool results."""

from __future__ import annotations

from typing import Any

from homeassistant.helpers import llm


def normalize_llm_tool_result(result: Any) -> tuple[dict[str, Any], bool]:
    """Return JSON data and the error flag across old and new HA result APIs."""
    tool_result_type = getattr(llm, "ToolResult", None)
    if tool_result_type is not None and isinstance(result, tool_result_type):
        data = result.data
        if isinstance(data, dict):
            return data, bool(result.error)
        return {"result": data}, bool(result.error)

    if isinstance(result, dict):
        return result, False
    return {"result": result}, False
