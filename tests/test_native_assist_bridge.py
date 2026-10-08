"""Native Assist compatibility and recoverable MCP tool errors."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
import pytest
import voluptuous as vol

from custom_components.mcp_assist.const import CONF_ENABLE_ASSIST_BRIDGE
from custom_components.mcp_assist.mcp_server import MCPServer


@pytest.fixture
def bridge(hass, system_entry_factory, profile_entry_factory):
    system_entry_factory(data={CONF_ENABLE_ASSIST_BRIDGE: True})
    server = MCPServer(hass, 8099, profile_entry_factory())
    api = SimpleNamespace(
        api=SimpleNamespace(id="assist", name="Assist"),
        tools=[SimpleNamespace(name="homeassistant__GetLiveContext")],
        async_call_tool=AsyncMock(return_value={"result": "Example context"}),
    )
    server._get_assist_api_instance = AsyncMock(return_value=api)
    return server, api


async def call(server, name, arguments):
    return await server.process_mcp_message({
        "jsonrpc": "2.0", "id": 42, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })


@pytest.mark.parametrize("name", ["GetLiveContext", "homeassistant__GetLiveContext"])
async def test_native_names_resolve_without_replaying(bridge, name):
    server, api = bridge
    response = await call(server, "call_assist_tool", {"tool_name": name, "arguments": {}})
    assert "error" not in response
    assert not response["result"].get("isError", False)
    api.async_call_tool.assert_awaited_once()
    assert api.async_call_tool.call_args.args[0].tool_name == "homeassistant__GetLiveContext"


async def test_namespaced_context_snapshot_is_detected_and_called(bridge):
    server, api = bridge
    assert server._assist_api_has_live_context_tool(api)
    response = await call(server, "get_assist_context_snapshot", {})
    assert "Example context" in response["result"]["content"][0]["text"]
    assert api.async_call_tool.call_args.args[0].tool_name == "homeassistant__GetLiveContext"


async def test_exact_legacy_name_wins_and_ambiguous_alias_is_not_called(bridge):
    server, api = bridge
    api.tools.append(SimpleNamespace(name="other__GetLiveContext"))
    assert not server._assist_api_has_live_context_tool(api)
    response = await call(server, "call_assist_tool", {"tool_name": "GetLiveContext"})
    assert response["result"]["isError"]
    api.async_call_tool.assert_not_awaited()
    api.tools.append(SimpleNamespace(name="GetLiveContext"))
    await call(server, "call_assist_tool", {"tool_name": "GetLiveContext"})
    assert api.async_call_tool.call_args.args[0].tool_name == "GetLiveContext"


@pytest.mark.parametrize("arguments", [
    {}, {"tool_name": "UnknownExampleTool"},
    {"tool_name": "GetLiveContext", "arguments": []},
    {"tool_name": "GetLiveContext", "arguments": None},
    {"tool_name": "wrong__GetLiveContext"},
])
async def test_invalid_native_calls_are_tool_errors(bridge, arguments):
    server, api = bridge
    response = await call(server, "call_assist_tool", arguments)
    assert "error" not in response
    assert response["result"]["isError"]
    api.async_call_tool.assert_not_awaited()


@pytest.mark.parametrize("error_type", [HomeAssistantError, vol.Invalid])
@pytest.mark.parametrize("tool_name", ["call_assist_tool", "get_assist_context_snapshot"])
async def test_expected_native_failures_are_sanitized_tool_errors(bridge, error_type, tool_name, caplog):
    server, api = bridge
    api.async_call_tool.side_effect = error_type("Bearer example-secret-canary")
    response = await call(server, tool_name, {"tool_name": "GetLiveContext"})
    assert "error" not in response
    assert response["result"]["isError"]
    assert "example-secret-canary" not in json.dumps(response) + caplog.text
    api.async_call_tool.assert_awaited_once()


@pytest.mark.parametrize("tool_name", ["call_assist_tool", "get_assist_context_snapshot"])
async def test_api_unavailability_is_recoverable(bridge, tool_name):
    server, api = bridge
    server._get_assist_api_instance.side_effect = HomeAssistantError("example-secret-canary")
    response = await call(server, tool_name, {"tool_name": "GetLiveContext"})
    assert response["result"]["isError"]
    assert "example-secret-canary" not in json.dumps(response)
    api.async_call_tool.assert_not_awaited()


async def test_failed_context_result_is_sanitized(bridge):
    server, api = bridge
    api.async_call_tool.return_value = {"success": False, "error": "example-secret-canary"}
    response = await call(server, "get_assist_context_snapshot", {})
    assert response["result"]["isError"]
    assert "example-secret-canary" not in json.dumps(response)


async def test_native_cancellation_propagates_without_retry(bridge):
    server, api = bridge
    api.async_call_tool.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await call(server, "call_assist_tool", {"tool_name": "GetLiveContext"})
    api.async_call_tool.assert_awaited_once()


async def test_native_failed_context_result_is_sanitized(bridge, caplog):
    if not hasattr(llm, "ToolResult"):
        pytest.skip("Native ToolResult was introduced in HA 2026.10")
    server, api = bridge
    api.async_call_tool.return_value = llm.ToolResult(
        data={"result": "example-secret-canary"}, error=True
    )
    response = await call(server, "get_assist_context_snapshot", {})
    assert response["result"]["isError"] is True
    assert "example-secret-canary" not in json.dumps(response) + caplog.text
    api.async_call_tool.assert_awaited_once()
