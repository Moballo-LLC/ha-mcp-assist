"""Provider-neutral MCP tool schema helpers."""

from __future__ import annotations

import json
from typing import Any

ADAPTIVE_TOOL_CATALOG_NAME = "list_available_tools"
ADAPTIVE_TOOL_SCHEMA_NAME = "load_tool_schemas"
ADAPTIVE_META_TOOL_NAMES = frozenset(
    {
        ADAPTIVE_TOOL_CATALOG_NAME,
        ADAPTIVE_TOOL_SCHEMA_NAME,
    }
)


def compact_text(text: str, *, max_len: int = 160) -> str:
    """Compact instructional text for lower token usage."""
    normalized = " ".join(str(text).split()).strip()
    if not normalized:
        return ""

    for separator in (". ", "\n", "; "):
        if separator in normalized:
            normalized = normalized.split(separator, 1)[0].strip()
            break

    if len(normalized) <= max_len:
        return normalized

    truncated = normalized[: max_len - 1].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 40:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:.") + "."


def compact_schema_for_llm(schema: Any, *, keep_description: bool = False) -> Any:
    """Strip nonessential JSON-schema verbosity before sending tools to the LLM."""
    if isinstance(schema, list):
        compacted_list = [
            compact_schema_for_llm(item, keep_description=keep_description)
            for item in schema
        ]
        return [item for item in compacted_list if item not in (None, {}, [])]

    if not isinstance(schema, dict):
        return schema

    compacted: dict[str, Any] = {}

    for key, value in schema.items():
        if key in {"$schema", "title", "default", "examples", "example"}:
            continue

        if key == "description":
            if keep_description:
                compact_description = compact_text(str(value), max_len=120)
                if compact_description:
                    compacted[key] = compact_description
            continue

        if key == "properties":
            properties: dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                compact_prop = compact_schema_for_llm(prop_schema)
                if compact_prop:
                    properties[prop_name] = compact_prop
            if properties:
                compacted[key] = properties
            continue

        if key == "required":
            if value:
                compacted[key] = value
            continue

        if key == "additionalProperties":
            continue

        compact_value = compact_schema_for_llm(
            value, keep_description=keep_description
        )
        if compact_value not in (None, {}, []):
            compacted[key] = compact_value

    return compacted


def build_tool_routing_summary(routing_hints: Any) -> str:
    """Build a compact description suffix from optional routing hints."""
    if not isinstance(routing_hints, dict):
        return ""

    preferred_when = str(routing_hints.get("preferred_when") or "").strip()
    if preferred_when:
        compact_preferred = compact_text(preferred_when, max_len=90)
        if compact_preferred:
            return f"Use for: {compact_preferred}"

    example_queries = routing_hints.get("example_queries")
    if isinstance(example_queries, list):
        cleaned_examples = [
            str(item).strip() for item in example_queries if str(item).strip()
        ][:1]
        if cleaned_examples:
            compact_example = compact_text(cleaned_examples[0], max_len=80)
            if compact_example:
                return f"Example: {compact_example}"

    keywords = routing_hints.get("keywords")
    if isinstance(keywords, list):
        cleaned_keywords = [
            str(item).strip() for item in keywords if str(item).strip()
        ][:3]
        if cleaned_keywords:
            return f"Keywords: {', '.join(cleaned_keywords)}"

    return ""


def convert_mcp_tools_to_llm_tools(
    tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert MCP tools to a compact provider-neutral function schema."""
    llm_tools = []

    for tool in tools:
        parameters = compact_schema_for_llm(
            tool.get("inputSchema", {}), keep_description=False
        )
        if not parameters:
            parameters = {"type": "object", "properties": {}}
        elif parameters.get("type") == "object" and "properties" not in parameters:
            parameters["properties"] = {}

        llm_description = compact_text(
            str(tool.get("llmDescription") or tool.get("llm_description") or ""),
            max_len=120,
        )
        base_description = compact_text(
            llm_description or tool.get("description", ""),
            max_len=140,
        )
        description_parts = [base_description.rstrip(" .")]
        routing_summary = build_tool_routing_summary(tool.get("routingHints"))
        if routing_summary:
            description_parts.append(routing_summary)

        llm_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": compact_text(
                        " | ".join(part for part in description_parts if part),
                        max_len=220,
                    ),
                    "parameters": parameters,
                },
            }
        )

    return llm_tools


def build_adaptive_meta_tools() -> list[dict[str, Any]]:
    """Return tiny meta tools for on-demand tool discovery in adaptive mode."""
    return [
        {
            "type": "function",
            "function": {
                "name": ADAPTIVE_TOOL_CATALOG_NAME,
                "description": (
                    "Search optional, built-in, and custom MCP tools before loading "
                    "their full schemas."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": ADAPTIVE_TOOL_SCHEMA_NAME,
                "description": (
                    "Load full schemas for specific optional/custom tools so they can "
                    "be called on the next turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_names": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                },
            },
        },
    ]


def build_adaptive_llm_tools(
    tools: list[dict[str, Any]],
    *,
    base_tool_names: frozenset[str],
    loaded_tool_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Return the LLM-facing tool surface for adaptive context mode."""
    selected_tool_names = set(base_tool_names) | set(loaded_tool_names)
    selected_tools = [
        tool
        for tool in tools
        if str(tool.get("name") or "") in selected_tool_names
    ]
    return [
        *convert_mcp_tools_to_llm_tools(selected_tools),
        *build_adaptive_meta_tools(),
    ]


def json_size_bytes(value: Any) -> int:
    """Return UTF-8 JSON size without exposing the JSON itself."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return len(text.encode("utf-8"))


def estimate_tokens_from_bytes(byte_count: int) -> int:
    """Return a rough conservative token estimate for compact JSON payloads."""
    return (byte_count + 3) // 4
