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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "reply, outcome, failure, uncertain, attempts",
    [
        ("success", "success", None, False, 1),
        ("non_200", "failure", "provider", False, 1),
        ("empty", "uncertain", None, True, 1),
        ("timeout", "failure", "timeout", True, 2),
        ("connection", "failure", "connection", False, 1),
        ("malformed", "failure", "validation", False, 1),
    ],
)
async def test_actual_no_tools_provider_outcomes_and_recovery(
    hass, profile_entry_factory, monkeypatch, nested,
    reply, outcome, failure, uncertain, attempts,
):
    """Fallback text cannot turn a failed normal provider dispatch into success."""
    from custom_components.mcp_assist import agent as agent_module

    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data={
        "server_type": "ollama", "lmstudio_url": "http://provider.example.invalid",
    }))
    provider = agent._get_llm_provider()
    response = SimpleNamespace(
        status=503 if reply == "non_200" else 200,
        text=AsyncMock(return_value="private response details"),
        json=AsyncMock(return_value={"message": {
            "role": "assistant", "content": "normal reply" if reply == "success" else "",
        }}),
    )
    if reply == "malformed":
        response.json.side_effect = ValueError("private response details")
    response_context = Mock(
        __aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=False),
    )
    errors = {"timeout": TimeoutError, "connection": ConnectionError}
    post = Mock(
        side_effect=errors[reply]("private transport details") if reply in errors else None,
        return_value=response_context,
    )
    session_context = Mock(
        __aenter__=AsyncMock(return_value=SimpleNamespace(post=post)),
        __aexit__=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", Mock(return_value=session_context))
    monkeypatch.setattr(agent, "_log_initial_llm_payload_metrics", Mock())

    async def final_response(messages):
        return await agent._call_llm_final_response_without_tools(
            messages, provider, transport="test_final",
        )

    if nested:
        monkeypatch.setattr(agent, "_call_llm_streaming", final_response)
        request = agent._call_llm
    else:
        request = final_response
    messages = [{"role": "user", "content": "private input"}]
    expected = "normal reply" if reply == "success" else agent_module.TOOL_BUDGET_FALLBACK_RESPONSE
    assert await request(messages) == expected
    value = stage(agent, "provider")
    assert value["started"] == value[outcome] == 1
    assert value["success"] == (1 if outcome == "success" else 0)
    assert value["failure"] == (1 if outcome == "failure" else 0)
    assert value["uncertain"] == (1 if uncertain else 0)
    assert value["active"] == 0
    assert value["last_outcome"] == outcome
    assert value["last_failure_type"] == failure
    assert value["last_terminal_uncertain"] is uncertain
    assert post.call_count == attempts
    assert "private" not in json.dumps(project_request_outcomes(agent))

    post.side_effect = None
    response.status = 200
    response.json.side_effect = None
    response.json.return_value = {"message": {"role": "assistant", "content": "recovered reply"}}
    assert await request(messages) == "recovered reply"
    recovered = stage(agent, "provider")
    assert recovered["started"] == 2
    assert recovered["active"] == 0
    assert recovered["success"] == value["success"] + 1
    assert recovered["failure"] == value["failure"]
    assert recovered["uncertain"] == value["uncertain"]
    assert recovered["last_outcome"] == "success"
    assert recovered["last_failure_type"] is None
    assert recovered["last_terminal_uncertain"] is False
    assert post.call_count == attempts + 1


class _StreamLines:
    def __init__(self, lines):
        self.lines = iter(line.encode() for line in lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.lines)
        except StopIteration:
            raise StopAsyncIteration from None


def _mock_responses(monkeypatch, aiohttp_module, specs):
    posts = []
    contexts = []
    for status, payload, lines in specs:
        response = SimpleNamespace(
            status=status,
            headers={},
            content=_StreamLines(lines),
            json=AsyncMock(return_value=payload),
            text=AsyncMock(return_value=json.dumps(payload)),
        )
        contexts.append(Mock(
            __aenter__=AsyncMock(return_value=response),
            __aexit__=AsyncMock(return_value=False),
        ))

    session = Mock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    def post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        return contexts.pop(0)

    session.post = Mock(side_effect=post)
    monkeypatch.setattr(aiohttp_module, "ClientSession", Mock(return_value=session))
    return posts


def _tool_response(transport, arguments, *, status=200, payload_error=None):
    payload = (
        {"error": payload_error}
        if payload_error is not None
        else {"message": {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-1",
                "function": {"name": "list_available_tools", "arguments": arguments},
            }],
        }}
    )
    lines = (
        [json.dumps(payload), json.dumps({"done": True})]
        if transport == "streaming" and status == 200 else []
    )
    return status, payload, lines


def _text_response(transport, text):
    payload = {"message": {"content": text}}
    lines = [json.dumps(payload), json.dumps({"done": True})] if transport == "streaming" else []
    return 200, payload, lines


@pytest.mark.parametrize("transport", ["streaming", "http"])
@pytest.mark.parametrize("trigger", ["http_error", "malformed_tool_call"])
async def test_invalid_tool_fallback_is_uncertain_and_recovers(
    hass, profile_entry_factory, monkeypatch, transport, trigger
):
    from custom_components.mcp_assist import agent as agent_module

    agent = MCPAssistConversationEntity(hass, profile_entry_factory(
        data={"server_type": "ollama"},
        options={"context_mode": "light"},
    ))
    provider = agent._get_llm_provider()
    monkeypatch.setattr(agent, "_get_llm_provider", Mock(return_value=provider))
    if transport == "http":
        provider.supports_streaming = False
    monkeypatch.setattr(agent, "_get_mcp_tools", AsyncMock(return_value=[{
        "type": "function",
        "function": {"name": "list_available_tools", "parameters": {"type": "object"}},
    }]))
    execute = AsyncMock(return_value=[])
    monkeypatch.setattr(agent, "_execute_tool_calls", execute)

    if trigger == "http_error":
        repeated = [
            _tool_response(
                transport, "", status=500,
                payload_error="invalid tool call arguments: invalid JSON",
            )
            for _ in range(2)
        ]
    else:
        repeated = [
            _tool_response(transport, 1),
            _tool_response(transport, 2),
        ]
    posts = _mock_responses(
        monkeypatch, agent_module.aiohttp,
        repeated + [_text_response(transport, "Recovered.")],
    )
    messages = [{"role": "user", "content": "private request"}]

    assert await agent._call_llm(messages) == agent_module.INVALID_TOOL_ARGUMENT_FALLBACK_RESPONSE
    assert len(posts) == 2
    assert "invalid JSON arguments" in posts[1]["json"]["messages"][-1]["content"]
    value = stage(agent, "provider")
    assert value["started"] == value["uncertain"] == 1
    assert value["success"] == value["failure"] == value["active"] == 0
    assert value["last_outcome"] == "uncertain"
    assert value["last_terminal_uncertain"] is True
    execute.assert_not_awaited()

    assert await agent._call_llm(messages) == "Recovered."
    value = stage(agent, "provider")
    assert len(posts) == 3
    assert value["started"] == 2
    assert value["success"] == 1
    assert value["failure"] == 0
    assert value["uncertain"] == 1
    assert value["last_outcome"] == "success"
    assert value["last_terminal_uncertain"] is False
    assert value["active"] == 0


@pytest.mark.parametrize("transport", ["streaming", "http"])
async def test_iteration_guard_fallback_is_uncertain_and_recovers(
    hass, profile_entry_factory, monkeypatch, transport
):
    from custom_components.mcp_assist import agent as agent_module

    agent = MCPAssistConversationEntity(hass, profile_entry_factory(
        data={"server_type": "ollama", "max_iterations": 1},
        options={"context_mode": "light", "max_iterations": 1},
    ))
    provider = agent._get_llm_provider()
    monkeypatch.setattr(agent, "_get_llm_provider", Mock(return_value=provider))
    if transport == "http":
        provider.supports_streaming = False
    monkeypatch.setattr(agent, "_get_mcp_tools", AsyncMock(return_value=[{
        "type": "function",
        "function": {"name": "list_available_tools", "parameters": {"type": "object"}},
    }]))
    execute = AsyncMock(return_value=[{"content": "completed without a tool-role message"}])
    monkeypatch.setattr(agent, "_execute_tool_calls", execute)
    posts = _mock_responses(
        monkeypatch, agent_module.aiohttp,
        [_tool_response(transport, "{}") for _ in range(2)]
        + [_text_response(transport, "Recovered.")],
    )
    messages = [{"role": "user", "content": "private request"}]

    assert await agent._call_llm(messages) == agent_module.TOOL_BUDGET_FALLBACK_RESPONSE
    assert len(posts) == 2
    assert execute.await_count == 2
    value = stage(agent, "provider")
    assert value["started"] == value["uncertain"] == 1
    assert value["success"] == value["failure"] == value["active"] == 0
    assert value["last_outcome"] == "uncertain"
    assert value["last_terminal_uncertain"] is True

    assert await agent._call_llm(messages) == "Recovered."
    value = stage(agent, "provider")
    assert len(posts) == 3
    assert value["started"] == 2
    assert value["success"] == 1
    assert value["failure"] == 0
    assert value["uncertain"] == 1
    assert value["last_outcome"] == "success"
    assert value["last_terminal_uncertain"] is False
    assert value["active"] == 0


@pytest.mark.parametrize(
    ("operation", "media_provider"),
    [
        ("no_tools", "ollama"),
        ("analyze", "ollama"),
        ("analyze", "openai"),
        ("generate", "ollama"),
        ("unsupported_image", "ollama"),
    ],
)
async def test_provider_http_and_local_image_failures_keep_distinct_categories(
    hass, profile_entry_factory, monkeypatch, operation, media_provider
):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist import mcp_server as server_module

    if operation == "no_tools":
        owner = MCPAssistConversationEntity(
            hass, profile_entry_factory(data={"server_type": "ollama"})
        )
        async def call():
            return await owner.async_call_llm_without_tools([
                {"role": "user", "content": "private request"}
            ])
        posts = _mock_responses(
            monkeypatch, agent_module.aiohttp,
            [(503, {"error": "provider unavailable"}, [])],
        )
    else:
        owner = MCPServer(hass, 12345, profile_entry_factory())

        def image_url():
            if operation == "unsupported_image":
                raise NotImplementedError
            return "https://provider.example.invalid/api/images"

        provider = SimpleNamespace(
            server_type=media_provider,
            model_name="example-model",
            image_model="example-image-model",
            display_name="Example provider",
            settings=SimpleNamespace(timeout=10, provider_options={}),
            headers=lambda: {},
            chat_url=lambda: "https://provider.example.invalid/api/chat",
            image_generation_url=image_url,
        )
        monkeypatch.setattr(owner, "_get_model_provider", Mock(return_value=provider))
        monkeypatch.setattr(owner, "_get_image_model_provider", AsyncMock(return_value=provider))
        if operation == "analyze":
            async def call():
                return await owner._analyze_image_with_provider(
                    question="private question", image_bytes=b"image",
                    mime_type="image/png", detail="auto", context=None,
                )
        else:
            async def call():
                return await owner._generate_image_with_provider(
                    prompt="private prompt", size=None, quality=None, style=None,
                    background=None, context=None,
                )
        specs = [] if operation == "unsupported_image" else [
            (503, {"error": "provider unavailable"}, [])
        ]
        posts = _mock_responses(monkeypatch, server_module.aiohttp, specs)

    with pytest.raises(ValueError):
        await call()

    value = stage(owner, "provider")
    assert value["failure"] == 1
    assert value["success"] == value["uncertain"] == value["active"] == 0
    assert value["last_failure_type"] == (
        "validation" if operation == "unsupported_image" else "provider"
    )
    assert len(posts) == (0 if operation == "unsupported_image" else 1)
