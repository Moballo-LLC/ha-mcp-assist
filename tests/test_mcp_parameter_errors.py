"""Malformed MCP envelopes are client errors without tool execution."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist.mcp_server import MCPServer


@pytest.mark.parametrize("method", ["initialize", "tools/list", "tools/call"])
@pytest.mark.parametrize("params", [[], None, "Bearer example-canary", 3, False])
async def test_non_object_params_are_client_errors(
    hass, system_entry_factory, profile_entry_factory, method, params, caplog
):
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.handle_tool_call = AsyncMock(return_value={"content": []})
    server.handle_initialize = AsyncMock()
    server.handle_tools_list = AsyncMock()
    request = SimpleNamespace(
        remote="127.0.0.1", headers={}, query={},
        json=AsyncMock(return_value={
            "jsonrpc": "2.0", "id": 42, "method": method, "params": params,
        }),
    )
    reply = await server.handle_mcp_request(request)
    response = json.loads(reply.text)
    assert reply.status == 200
    assert response["error"]["code"] == -32602
    assert response["id"] == 42
    assert "example-canary" not in reply.text + caplog.text
    assert "Error in MCP method" not in caplog.text
    server.handle_tool_call.assert_not_awaited()
    server.handle_initialize.assert_not_awaited()
    server.handle_tools_list.assert_not_awaited()
    recovered = await server.process_mcp_message({
        "jsonrpc": "2.0", "id": 43, "method": "tools/call",
        "params": {"name": "example_tool", "arguments": {}},
    })
    assert "error" not in recovered
    server.handle_tool_call.assert_awaited_once()


@pytest.mark.parametrize("field", ["arguments", "context"])
@pytest.mark.parametrize("value", [[], None, "example-canary", False])
async def test_non_object_tool_fields_do_not_dispatch(
    hass, system_entry_factory, profile_entry_factory, field, value
):
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.handle_tool_call = AsyncMock()
    response = await server.process_mcp_message({
        "jsonrpc": "2.0", "id": "parameter-test", "method": "tools/call",
        "params": {"name": "example_tool", field: value},
    })
    assert response["error"]["code"] == -32602
    assert "example-canary" not in json.dumps(response)
    server.handle_tool_call.assert_not_awaited()
