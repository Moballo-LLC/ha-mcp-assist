"""Tests for IndexManager LLM gap-filling helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.mcp_assist.const import DOMAIN
from custom_components.mcp_assist.index_manager import IndexManager


@pytest.mark.asyncio
async def test_gap_filling_uses_profile_agent_when_system_entry_is_first(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """LLM inference should skip the shared system entry and use a profile agent."""
    system_entry_factory()
    profile_entry = profile_entry_factory()
    seen_inputs = []

    class FakeAgent:
        async def async_process(self, conversation_input):
            seen_inputs.append(conversation_input)
            return SimpleNamespace(
                response=SimpleNamespace(
                    speech={
                        "plain": {
                            "speech": (
                                '{"presence": {"pattern": "binary_sensor.*_presence", '
                                '"count": 2, "description": "Presence sensors"}}'
                            )
                        }
                    }
                )
            )

    hass.data.setdefault(DOMAIN, {})[profile_entry.entry_id] = {"agent": FakeAgent()}
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 2
    assert seen_inputs[0].agent_id == profile_entry.entry_id
