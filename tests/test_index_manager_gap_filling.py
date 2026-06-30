"""Tests for IndexManager LLM gap-filling helpers."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.const import DOMAIN
from custom_components.mcp_assist.index_manager import IndexManager


def test_gap_filling_parser_recovers_complete_categories_from_truncated_tail(hass) -> None:
    """A truncated LLM response should not discard complete inferred categories."""
    manager = IndexManager(hass)

    inferred = manager._parse_inferred_types(
        (
            '{"presence": {"pattern": "binary_sensor.*_presence", "count": 2, '
            '"description": "Presence sensors"}, "lights": {"pattern": "light.'
        )
    )

    assert inferred == {
        "presence": {
            "pattern": "binary_sensor.*_presence",
            "count": 2,
            "description": "Presence sensors",
        }
    }


@pytest.mark.asyncio
async def test_gap_filling_uses_profile_agent_when_system_entry_is_first(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """LLM inference should skip the shared system entry and use a profile agent."""
    system_entry_factory()
    profile_entry = profile_entry_factory()
    seen_calls = []

    class FakeAgent:
        async def async_process(self, conversation_input):
            raise AssertionError("gap filling should not use full conversation processing")

        async def async_call_llm_without_tools(self, messages, *, transport):
            seen_calls.append((messages, transport))
            return (
                '{"presence": {"pattern": "binary_sensor.*_presence", '
                '"count": 2, "description": "Presence sensors"}}'
            )

    hass.data.setdefault(DOMAIN, {})[profile_entry.entry_id] = {"agent": FakeAgent()}
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 2
    assert seen_calls == [([{"role": "user", "content": "infer entities"}], "index_gap_filling")]
