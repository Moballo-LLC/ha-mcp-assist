"""Tests for MCP tool-effect classifications and annotations."""

from custom_components.mcp_assist.tool_effects import (
    ToolEffect,
    annotate_tool_effect,
    get_tool_effect,
)


def test_destructive_shipped_tools_are_annotated_high_risk() -> None:
    """Deletion and mixed queue tools should request destructive approval."""
    for tool_name in (
        "forget_memory",
        "play_music_assistant",
        "control_music_assistant_player",
    ):
        assert get_tool_effect(tool_name) is ToolEffect.HIGH_RISK
        annotated = annotate_tool_effect(
            {
                "name": tool_name,
                "annotations": {"title": "Preserved annotation"},
            }
        )
        assert annotated["annotations"] == {
            "title": "Preserved annotation",
            "readOnlyHint": False,
            "destructiveHint": True,
        }
