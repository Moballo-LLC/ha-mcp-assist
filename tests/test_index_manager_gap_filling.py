"""Tests for IndexManager LLM gap-filling helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.mcp_assist.const import (
    CONF_SERVER_TYPE,
    DOMAIN,
    SERVER_TYPE_HERMES,
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
@pytest.mark.parametrize(
    ("server_type", "profile_name"),
    [
        (SERVER_TYPE_OPENCLAW, "OpenClaw"),
        (SERVER_TYPE_HERMES, "Hermes Agent"),
    ],
)
async def test_gap_filling_uses_server_managed_conversation_when_it_is_only_profile(
    hass, profile_entry_factory, server_type: str, profile_name: str
) -> None:
    """Server-managed-only installs should use the conversation transport."""
    profile_entry = profile_entry_factory(
        title=f"{profile_name} - Test Profile",
        unique_id=f"{server_type}-profile",
        data={CONF_SERVER_TYPE: server_type},
    )
    seen_inputs = []

    class ServerManagedAgent:
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

    hass.data.setdefault(DOMAIN, {})[profile_entry.entry_id] = {
        "agent": ServerManagedAgent()
    }
    manager = IndexManager(hass)

    inferred = await manager._call_llm_for_inference("infer entities")

    assert inferred["presence"]["count"] == 4
    assert seen_inputs[0].agent_id == profile_entry.entry_id


_PATTERN_ENTITIES = [
    "binary_sensor.front_person_detected",
    "binary_sensor.back_person_detected",
]


@pytest.mark.asyncio
async def test_gap_filling_reuses_result_when_patterns_unchanged(hass, monkeypatch) -> None:
    """Registry churn with unchanged entity patterns must not re-call the LLM."""
    manager = IndexManager(hass)
    calls = []

    async def fake_call(prompt):
        calls.append(prompt)
        return {"person_detection": {"pattern": "p", "count": 2, "description": "d"}}

    monkeypatch.setattr(manager, "_call_llm_for_inference", fake_call)

    first = await manager._infer_entity_types(list(_PATTERN_ENTITIES))
    second = await manager._infer_entity_types(list(reversed(_PATTERN_ENTITIES)))

    assert first == second
    assert len(calls) == 1

    await manager._infer_entity_types(
        [*_PATTERN_ENTITIES, "sensor.kitchen_ble_area", "sensor.office_ble_area"]
    )
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_gap_filling_failure_waits_for_cooldown_before_retry(hass, monkeypatch) -> None:
    """A failed or empty LLM response should not be retried on every refresh."""
    from datetime import timedelta

    from custom_components.mcp_assist import index_manager as index_manager_module

    manager = IndexManager(hass)
    calls = []

    async def failing_call(prompt):
        calls.append(prompt)
        raise ValueError("No-tools provider response was empty")

    monkeypatch.setattr(manager, "_call_llm_for_inference", failing_call)

    assert await manager._infer_entity_types(list(_PATTERN_ENTITIES)) == {}
    assert await manager._infer_entity_types(list(_PATTERN_ENTITIES)) == {}
    assert len(calls) == 1

    manager._last_inference_at -= timedelta(
        seconds=index_manager_module.GAP_FILL_FAILURE_RETRY_SECONDS + 1
    )
    await manager._infer_entity_types(list(_PATTERN_ENTITIES))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_gap_filling_empty_result_is_retried_after_cooldown(
    hass, monkeypatch
) -> None:
    """A valid but empty `{}` response must not be cached as a success forever."""
    from datetime import timedelta

    from custom_components.mcp_assist import index_manager as index_manager_module

    manager = IndexManager(hass)
    calls = []

    async def empty_call(prompt):
        calls.append(prompt)
        return {}

    monkeypatch.setattr(manager, "_call_llm_for_inference", empty_call)

    assert await manager._infer_entity_types(list(_PATTERN_ENTITIES)) == {}
    assert await manager._infer_entity_types(list(_PATTERN_ENTITIES)) == {}
    assert len(calls) == 1
    assert manager._last_inference_succeeded is False

    manager._last_inference_at -= timedelta(
        seconds=index_manager_module.GAP_FILL_FAILURE_RETRY_SECONDS + 1
    )
    await manager._infer_entity_types(list(_PATTERN_ENTITIES))
    assert len(calls) == 2
