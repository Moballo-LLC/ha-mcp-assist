"""Ordinary lifecycle, privacy and diagnostic read boundaries."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.mcp_assist import runtime_observation as observation
from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.diagnostics import build_assist_diagnostics
from custom_components.mcp_assist.mcp_server import MCPServer
from custom_components.mcp_assist.runtime_observation import (
    note_request_failure, observe, observe_request, project_request_outcomes,
)


class Worker:
    @observe("tool")
    async def tool(self, result):
        return result

    @observe("provider")
    async def inner(self, result):
        return result

    @observe("provider")
    async def outer(self, result):
        return await self.inner(result)

    @observe("tool")
    async def fail(self, error):
        raise error

    @observe("tool")
    async def caught(self, error):
        note_request_failure(self, "tool", error)
        return {"error": "private diagnostic details"}

    @observe("tool")
    async def waiting(self, started, release):
        started.set()
        await release.wait()
        return {"content": []}


def stage(owner, name):
    return project_request_outcomes(owner)["stages"][name]


def test_read_only_unknown_projection_does_not_initialize():
    owner = Worker()
    assert project_request_outcomes(owner) == {"status": "not_observed"}
    assert vars(owner) == {}
    assert project_request_outcomes(None) == {"status": "not_observed"}


async def test_private_results_and_errors_are_returned_unchanged_and_not_retained():
    owner = Worker()
    secret = "private-canary-never-retain"
    result = {"content": [{"text": secret}], "structuredContent": {"token": secret}}
    assert await owner.tool(result) is result
    error = ValueError(secret)
    with pytest.raises(ValueError) as caught:
        await owner.fail(error)
    assert caught.value is error
    snapshot = project_request_outcomes(owner)
    assert secret not in json.dumps(snapshot)
    assert stage(owner, "tool")["last_failure_type"] == "validation"
    assert stage(owner, "tool")["active"] == 0


async def test_nested_provider_dispatch_counts_once_and_profiles_stay_independent():
    first, second = Worker(), Worker()
    assert await first.outer("reply") == "reply"
    assert stage(first, "provider")["started"] == 1
    assert stage(first, "provider")["success"] == 1
    assert project_request_outcomes(second) == {"status": "not_observed"}
    await second.inner("another reply")
    assert stage(second, "provider")["started"] == 1


async def test_cancellation_retains_uncertainty_and_normal_recovery():
    owner = Worker()
    started, release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(owner.waiting(started, release))
    await started.wait()
    assert stage(owner, "tool")["active"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    value = stage(owner, "tool")
    assert value["cancelled"] == value["uncertain"] == 1
    assert value["last_terminal_uncertain"] is True
    assert value["active"] == 0
    original_cancel = value["last_cancelled_at"]
    await owner.tool({"content": []})
    value = stage(owner, "tool")
    assert value["success"] == 1
    assert value["last_terminal_uncertain"] is False
    assert value["uncertain"] == 1  # Later completion grants no replay authority.
    assert value["last_cancelled_at"] == original_cancel


async def test_independent_concurrent_requests_are_not_deduplicated():
    owner = Worker()
    started, release = asyncio.Event(), asyncio.Event()
    tasks = [asyncio.create_task(owner.waiting(started, release)) for _ in range(3)]
    await started.wait()
    await asyncio.sleep(0)
    assert stage(owner, "tool")["active"] == 3
    release.set()
    await asyncio.gather(*tasks)
    assert stage(owner, "tool")["started"] == 3
    assert stage(owner, "tool")["success"] == 3
    assert stage(owner, "tool")["active"] == 0


async def test_timeout_caught_by_dispatch_is_typed_and_uncertain():
    owner = Worker()
    await owner.caught(TimeoutError("private timeout details"))
    value = stage(owner, "tool")
    assert value["failure"] == value["uncertain"] == 1
    assert value["last_failure_type"] == "timeout"
    assert "private timeout" not in json.dumps(project_request_outcomes(owner))


async def test_unknown_result_remains_uncertain_and_diagnostic_reads_do_not_freshen():
    owner = Worker()
    await owner.tool(object())
    first = project_request_outcomes(owner)
    assert first["stages"]["tool"]["uncertain"] == 1
    assert first["stages"]["tool"]["success"] == 0
    first["stages"]["tool"]["last_terminal_at"] = "changed copy"
    assert project_request_outcomes(owner)["stages"]["tool"]["last_terminal_at"] != "changed copy"
    assert project_request_outcomes(owner) == project_request_outcomes(owner)


async def test_counters_and_stage_inventory_are_bounded_without_limiting_requests(monkeypatch):
    monkeypatch.setattr(observation, "_MAX_COUNT", 3)
    owner = Worker()
    for _ in range(10):
        assert await owner.tool({"content": []}) == {"content": []}
    value = stage(owner, "tool")
    assert value["started"] == value["success"] == 3
    assert value["saturated"] is True
    assert value["active"] == 0
    with pytest.raises(ValueError):
        observe_request(owner, "unbounded-stage")
    assert set(project_request_outcomes(owner)["stages"]) == {"tool"}


@pytest.mark.parametrize("provider", ["hermes", "openclaw"])
async def test_actual_external_provider_dispatch_is_observed_once(
    hass, profile_entry_factory, monkeypatch, provider
):
    entry = profile_entry_factory(data={"server_type": provider})
    agent = MCPAssistConversationEntity(hass, entry)
    client = SimpleNamespace(send_message=AsyncMock(return_value="normal reply"))
    if provider == "hermes":
        monkeypatch.setattr(agent, "_get_hermes_client", Mock(return_value=client))
    else:
        hass.data.setdefault("mcp_assist", {})[entry.entry_id] = {"openclaw_client": client}
    result = object()
    monkeypatch.setattr(agent, "_build_response_result", AsyncMock(return_value=result))
    method = getattr(agent, f"_handle_{provider}_message")
    assert await method(SimpleNamespace(text="private input"), "private conversation") is result
    assert client.send_message.await_count == 1
    assert stage(agent, "provider")["started"] == stage(agent, "provider")["success"] == 1
    assert "private" not in json.dumps(project_request_outcomes(agent))


async def test_actual_provider_pipeline_and_server_tool_boundaries(hass, profile_entry_factory, monkeypatch):
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(agent, "_get_llm_provider", Mock(return_value=SimpleNamespace(supports_streaming=False)))
    monkeypatch.setattr(agent, "_call_llm_http", AsyncMock(return_value="normal reply"))
    assert await agent._call_llm([]) == "normal reply"
    assert stage(agent, "provider")["success"] == 1
    server = MCPServer(hass, 12345, entry)
    result = {"content": [], "isError": True}
    monkeypatch.setattr(server, "_handle_tool_call", AsyncMock(return_value=result))
    assert await server.handle_tool_call({}) == result
    assert stage(server, "tool")["failure"] == 1
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": agent}, "shared_mcp_server": server}
    diagnostics = build_assist_diagnostics(hass, entry)
    assert diagnostics["ordinary_outcomes"] == project_request_outcomes(agent)


async def test_conversation_errors_and_success_are_distinct():
    class Conversation:
        @observe("conversation")
        async def request(self, response):
            return SimpleNamespace(response=response)
    owner = Conversation()
    await owner.request(SimpleNamespace(error_code="unknown"))
    assert stage(owner, "conversation")["failure"] == 1
    await owner.request(SimpleNamespace(error_code=None))
    assert stage(owner, "conversation")["success"] == 1
    assert stage(owner, "conversation")["last_outcome"] == "success"


@pytest.mark.parametrize("reply, timeout, outcome", [({}, False, "uncertain"), (None, True, "failure")])
async def test_actual_client_unknown_reply_and_caught_timeout(
    hass, profile_entry_factory, monkeypatch, reply, timeout, outcome
):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.tool_effects import ToolEffect

    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    monkeypatch.setattr(agent, "_is_tool_configured_for_profile", Mock(return_value=True))
    monkeypatch.setattr(agent, "_get_tool_effect", Mock(return_value=ToolEffect.READ_ONLY))
    response = SimpleNamespace(status=200, json=AsyncMock(return_value=reply))
    response_context = Mock(__aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=False))
    post = Mock(side_effect=TimeoutError("private timeout") if timeout else None,
                return_value=response_context)
    session_context = Mock(__aenter__=AsyncMock(return_value=SimpleNamespace(post=post)),
                           __aexit__=AsyncMock(return_value=False))
    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", Mock(return_value=session_context))
    await agent._call_mcp_tool("example_tool", {"private": "argument"})
    value = stage(agent, "tool")
    assert value[outcome] == 1
    assert value["success"] == 0
    assert value["active"] == 0
    assert value["last_terminal_uncertain"] is True
    if timeout:
        assert value["last_failure_type"] == "timeout"
    assert "private" not in json.dumps(project_request_outcomes(agent))
    assert post.call_count == 1
