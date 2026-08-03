"""Tool-effect metadata used to enforce read-only conversation profiles."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ToolEffect(str, Enum):
    """Describe whether an MCP tool can change external or persisted state."""

    READ_ONLY = "read_only"
    WRITE = "write"
    HIGH_RISK = "high_risk"


_SHIPPED_TOOL_EFFECTS: dict[str, ToolEffect] = {
    # Core discovery and context tools.
    "discover_entities": ToolEffect.READ_ONLY,
    "discover_devices": ToolEffect.READ_ONLY,
    "get_entity_details": ToolEffect.READ_ONLY,
    "get_device_details": ToolEffect.READ_ONLY,
    "list_areas": ToolEffect.READ_ONLY,
    "list_domains": ToolEffect.READ_ONLY,
    "get_index": ToolEffect.READ_ONLY,
    "list_assist_tools": ToolEffect.READ_ONLY,
    "get_assist_prompt": ToolEffect.READ_ONLY,
    "get_assist_context_snapshot": ToolEffect.READ_ONLY,
    "analyze_image": ToolEffect.READ_ONLY,
    "get_image": ToolEffect.READ_ONLY,
    "set_conversation_state": ToolEffect.READ_ONLY,
    # Core tools that can invoke arbitrary Home Assistant behavior.
    "call_assist_tool": ToolEffect.HIGH_RISK,
    "perform_action": ToolEffect.HIGH_RISK,
    "run_script": ToolEffect.HIGH_RISK,
    "run_automation": ToolEffect.HIGH_RISK,
    "generate_image": ToolEffect.WRITE,
    # Calculator and unit conversion.
    "add": ToolEffect.READ_ONLY,
    "subtract": ToolEffect.READ_ONLY,
    "multiply": ToolEffect.READ_ONLY,
    "divide": ToolEffect.READ_ONLY,
    "sqrt": ToolEffect.READ_ONLY,
    "power": ToolEffect.READ_ONLY,
    "round_number": ToolEffect.READ_ONLY,
    "average": ToolEffect.READ_ONLY,
    "min_value": ToolEffect.READ_ONLY,
    "max_value": ToolEffect.READ_ONLY,
    "convert_unit": ToolEffect.READ_ONLY,
    "evaluate_expression": ToolEffect.READ_ONLY,
    # Search, maps, URL, weather, recorder, and response reads.
    "search": ToolEffect.READ_ONLY,
    "read_url": ToolEffect.READ_ONLY,
    "search_wikipedia": ToolEffect.READ_ONLY,
    "search_google_places": ToolEffect.READ_ONLY,
    "get_google_place_details": ToolEffect.READ_ONLY,
    "get_google_route": ToolEffect.READ_ONLY,
    "get_weather_forecast": ToolEffect.READ_ONLY,
    "get_entity_history": ToolEffect.READ_ONLY,
    "analyze_entity_history": ToolEffect.READ_ONLY,
    "get_entity_state_at_time": ToolEffect.READ_ONLY,
    "get_calendar_events": ToolEffect.READ_ONLY,
    "list_response_services": ToolEffect.READ_ONLY,
    "call_service_with_response": ToolEffect.HIGH_RISK,
    # Home Assistant LLM API bridge.
    "list_llm_apis": ToolEffect.READ_ONLY,
    "list_llm_api_tools": ToolEffect.READ_ONLY,
    "get_llm_api_prompt": ToolEffect.READ_ONLY,
    "call_llm_api_tool": ToolEffect.HIGH_RISK,
    # Persisted MCP Assist memory.
    "list_memory_categories": ToolEffect.READ_ONLY,
    "recall_memories": ToolEffect.READ_ONLY,
    "remember_memory": ToolEffect.WRITE,
    "forget_memory": ToolEffect.WRITE,
    # Music Assistant queries and controls.
    "list_music_assistant_players": ToolEffect.READ_ONLY,
    "list_music_assistant_instances": ToolEffect.READ_ONLY,
    "search_music_assistant": ToolEffect.READ_ONLY,
    "get_music_assistant_library": ToolEffect.READ_ONLY,
    "get_music_assistant_queue": ToolEffect.READ_ONLY,
    "play_music_assistant": ToolEffect.WRITE,
    "control_music_assistant_player": ToolEffect.WRITE,
    "transfer_music_assistant_queue": ToolEffect.WRITE,
}


def get_tool_effect(
    tool_name: str,
    tool_definition: dict[str, Any] | None = None,
) -> ToolEffect:
    """Return a tool's declared effect, failing closed for unknown tools."""
    if effect := _SHIPPED_TOOL_EFFECTS.get(tool_name):
        return effect

    annotations = (
        tool_definition.get("annotations")
        if isinstance(tool_definition, dict)
        else None
    )
    if not isinstance(annotations, dict):
        return ToolEffect.HIGH_RISK
    if annotations.get("destructiveHint") is True:
        return ToolEffect.HIGH_RISK
    if annotations.get("readOnlyHint") is True:
        return ToolEffect.READ_ONLY
    if annotations.get("destructiveHint") is False:
        return ToolEffect.WRITE
    return ToolEffect.HIGH_RISK


def annotate_tool_effect(tool_definition: dict[str, Any]) -> dict[str, Any]:
    """Return a tool definition with explicit standard MCP effect hints."""
    tool_name = str(tool_definition.get("name") or "")
    effect = get_tool_effect(tool_name, tool_definition)
    current_annotations = tool_definition.get("annotations")
    annotations = (
        dict(current_annotations) if isinstance(current_annotations, dict) else {}
    )
    annotations["readOnlyHint"] = effect is ToolEffect.READ_ONLY
    annotations["destructiveHint"] = effect is ToolEffect.HIGH_RISK
    return {**tool_definition, "annotations": annotations}


def tool_requires_control(
    tool_name: str,
    tool_definition: dict[str, Any] | None = None,
) -> bool:
    """Return whether a profile must allow control before using a tool."""
    return get_tool_effect(tool_name, tool_definition) is not ToolEffect.READ_ONLY
