"""Tests for IndexManager LLM gap-filling helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.mcp_assist.const import (
    CONF_SERVER_TYPE,
    DOMAIN,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENCLAW,
)
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
    assert seen_calls == [
        ([{"role": "user", "content": "infer entities"}], "index_gap_filling")
    ]


@pytest.mark.asyncio
async def test_gap_filling_skips_openclaw_when_http_profile_is_available(
    hass, profile_entry_factory
) -> None:
    """OpenClaw profiles should not block direct no-tools inference."""
    openclaw_entry = profile_entry_factory(
        title="OpenClaw - Test Profile",
        unique_id="openclaw-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENCLAW},
    )
    openai_entry = profile_entry_factory(
        title="OpenAI - Test Profile",
        unique_id="openai-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENAI},
    )
    seen_calls = []

    class OpenClawAgent:
        async def async_process(self, conversation_input):
            raise AssertionError("direct-capable profile should be preferred")

    class OpenAIAgent:
        async def async_call_llm_without_tools(self, messages, *, transport):
            seen_calls.append((messages, transport))
            return (
                '{"presence": {"pattern": "binary_sensor.*_presence", '
                '"count": 3, "description": "Presence sensors"}}'
            )

    hass.data.setdefault(DOMAIN, {})[openclaw_entry.entry_id] = {
        "agent": OpenClawAgent()
    }
    hass.data.setdefault(DOMAIN, {})[openai_entry.entry_id] = {"agent": OpenAIAgent()}
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 3
    assert seen_calls == [
        ([{"role": "user", "content": "infer entities"}], "index_gap_filling")
    ]


@pytest.mark.asyncio
async def test_gap_filling_prefers_remote_profile_over_local_entry_order(
    hass, profile_entry_factory
) -> None:
    """Auto-ranking should choose a hosted direct provider before local providers."""
    ollama_entry = profile_entry_factory(
        title="Ollama - Test Profile",
        unique_id="ollama-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA},
    )
    openai_entry = profile_entry_factory(
        title="OpenAI - Test Profile",
        unique_id="openai-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENAI},
    )
    seen_calls = []

    class OllamaAgent:
        async def async_call_llm_without_tools(self, messages, *, transport):
            seen_calls.append(("ollama", messages, transport))
            return (
                '{"presence": {"pattern": "binary_sensor.*_presence", '
                '"count": 4, "description": "Presence sensors"}}'
            )

    class OpenAIAgent:
        async def async_call_llm_without_tools(self, messages, *, transport):
            seen_calls.append(("openai", messages, transport))
            return (
                '{"presence": {"pattern": "binary_sensor.*_presence", '
                '"count": 5, "description": "Presence sensors"}}'
            )

    hass.data.setdefault(DOMAIN, {})[ollama_entry.entry_id] = {"agent": OllamaAgent()}
    hass.data.setdefault(DOMAIN, {})[openai_entry.entry_id] = {"agent": OpenAIAgent()}
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 5
    assert [call[0] for call in seen_calls] == ["openai"]


@pytest.mark.asyncio
async def test_gap_filling_tries_next_profile_when_first_direct_profile_fails(
    hass, profile_entry_factory
) -> None:
    """A failing direct-capable profile should not block later profiles."""
    failing_entry = profile_entry_factory(
        title="OpenAI - Test Profile",
        unique_id="openai-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENAI},
    )
    working_entry = profile_entry_factory(
        title="Ollama - Test Profile",
        unique_id="ollama-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA},
    )
    seen_calls = []

    class FailingAgent:
        async def async_call_llm_without_tools(self, messages, *, transport):
            seen_calls.append(("failing", messages, transport))
            return "not json"

    class WorkingAgent:
        async def async_call_llm_without_tools(self, messages, *, transport):
            seen_calls.append(("working", messages, transport))
            return (
                '{"presence": {"pattern": "binary_sensor.*_presence", '
                '"count": 5, "description": "Presence sensors"}}'
            )

    hass.data.setdefault(DOMAIN, {})[failing_entry.entry_id] = {"agent": FailingAgent()}
    hass.data.setdefault(DOMAIN, {})[working_entry.entry_id] = {"agent": WorkingAgent()}
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 5
    assert [call[0] for call in seen_calls] == ["failing", "working"]


@pytest.mark.asyncio
async def test_gap_filling_uses_openclaw_conversation_when_it_is_only_profile(
    hass, profile_entry_factory
) -> None:
    """OpenClaw-only installs should keep the prior conversation fallback behavior."""
    openclaw_entry = profile_entry_factory(
        title="OpenClaw - Test Profile",
        unique_id="openclaw-profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENCLAW},
    )
    seen_inputs = []

    class OpenClawAgent:
        async def async_process(self, conversation_input):
            seen_inputs.append(conversation_input)
            return SimpleNamespace(
                response=SimpleNamespace(
                    speech={
                        "plain": {
                            "speech": (
                                '{"presence": {"pattern": "binary_sensor.*_presence", '
                                '"count": 4, "description": "Presence sensors"}}'
                            )
                        }
                    }
                )
            )

    hass.data.setdefault(DOMAIN, {})[openclaw_entry.entry_id] = {
        "agent": OpenClawAgent()
    }
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 4
    assert seen_inputs[0].agent_id == openclaw_entry.entry_id
