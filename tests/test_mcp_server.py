"""Tests for MCP server configuration-sensitive behavior."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

from aiohttp import ClientSession, WSCloseCode
import yarl
from homeassistant.components.weather import WeatherEntityFeature
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util import dt as dt_util
import pytest
from pytest_socket import disable_socket, enable_socket
import voluptuous as vol

import custom_components.mcp_assist.mcp_server as mcp_server_module
from custom_components.mcp_assist.tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.tools.packages.llm_api_bridge.llm_api_bridge import (
    LLM_API_BRIDGE_TOOL_DEFINITIONS,
    LLMApiBridgeTool,
)
from custom_components.mcp_assist.tools.packages.memory.memory import MemoryTool
from custom_components.mcp_assist.tools.packages.recorder.recorder import RECORDER_TOOL_DEFINITIONS
from custom_components.mcp_assist.tools.packages.response_service.response_services import (
    RESPONSE_SERVICE_TOOL_DEFINITIONS,
)
from custom_components.mcp_assist.tools.packages.weather_forecast.weather import WEATHER_TOOL_DEFINITIONS
from custom_components.mcp_assist.const import (
    CONF_API_KEY,
    CONF_ALLOWED_IPS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_LMSTUDIO_URL,
    CONF_LLM_API_ALLOWLIST,
    CONF_MODEL_NAME,
    CONF_SERVER_TYPE,
    DOMAIN,
    SERVER_TYPE_OPENAI,
)
from custom_components.mcp_assist.mcp_server import MCPServer

BUILTIN_SPECS = load_builtin_tool_toggle_specs()


class _EchoLLMTool(llm.Tool):
    """Small fake Home Assistant LLM tool for bridge tests."""

    name = "EchoIntent"
    description = "Echo a value through a fake third-party LLM API."
    parameters = vol.Schema({vol.Required("value"): str})

    async def async_call(
        self,
        hass,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        """Return the provided value and basic context metadata."""
        return {
            "echo": tool_input.tool_args["value"],
            "external": tool_input.external,
            "platform": llm_context.platform,
        }


class _FakeLLMAPI(llm.API):
    """Small fake third-party Home Assistant LLM API for bridge tests."""

    async def async_get_api_instance(
        self,
        llm_context: llm.LLMContext,
    ) -> llm.APIInstance:
        """Return a fake API instance with one tool."""
        return llm.APIInstance(
            api=self,
            api_prompt="Use EchoIntent for fake LLM intent testing.",
            llm_context=llm_context,
            tools=[_EchoLLMTool()],
        )


def test_create_assist_llm_context_supports_legacy_user_prompt(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Older HA LLMContext constructors should receive a blank user prompt."""

    class _LegacyLLMContext:
        def __init__(
            self,
            *,
            platform: str,
            context: Any,
            language: str,
            assistant: str,
            device_id: str | None,
            user_prompt: str,
        ) -> None:
            self.platform = platform
            self.context = context
            self.language = language
            self.assistant = assistant
            self.device_id = device_id
            self.user_prompt = user_prompt

    monkeypatch.setattr(mcp_server_module.llm, "LLMContext", _LegacyLLMContext)
    server = MCPServer(hass, 8099, profile_entry_factory())

    context = server._create_assist_llm_context()

    assert context.platform == DOMAIN
    assert context.language == "*"
    assert context.assistant == "conversation"
    assert context.device_id is None
    assert context.user_prompt == ""


def test_create_llm_tool_input_supports_legacy_external_keyword(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Older HA ToolInput constructors should not reject bridge calls."""

    class _LegacyToolInput:
        def __init__(self, *, tool_name, tool_args) -> None:
            self.tool_name = tool_name
            self.tool_args = tool_args

    monkeypatch.setattr(mcp_server_module.llm, "ToolInput", _LegacyToolInput)
    server = MCPServer(hass, 8099, profile_entry_factory())

    tool_input = server._create_llm_tool_input(
        "EchoIntent",
        {"value": "hello"},
        external=True,
    )

    assert tool_input.tool_name == "EchoIntent"
    assert tool_input.tool_args == {"value": "hello"}
    assert not hasattr(tool_input, "external")


def _json_payload_from_text_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the JSON payload appended after a text header."""
    return json.loads(result["content"][0]["text"].split("\n\n", 1)[1])


def test_sanitize_log_value_escapes_line_breaks() -> None:
    """User-controlled log values should not be able to add log lines."""
    assert mcp_server_module._sanitize_log_value("one\ntwo\rthree") == (
        "one\\ntwo\\rthree"
    )


def test_sanitize_log_value_redacts_common_secret_markers() -> None:
    """User-controlled log values should not expose common secret markers."""
    sanitized = mcp_server_module._sanitize_log_value(
        "Authorization: Bearer abc123 api_key=secret-token password=hunter2 "
        '{"api_key":"quoted-secret","token":"quoted-token"} '
        "{'secret': 'dict-secret'}"
    )

    assert "Authorization" not in sanitized
    assert "Bearer" not in sanitized
    assert "api_key" not in sanitized
    assert "password" not in sanitized
    assert "abc123" not in sanitized
    assert "secret-token" not in sanitized
    assert "hunter2" not in sanitized
    assert "quoted-secret" not in sanitized
    assert "quoted-token" not in sanitized
    assert "dict-secret" not in sanitized
    assert "[redacted]" in sanitized


def _builtin_spec(tool_name: str):
    """Return the built-in packaged-tool spec for a tool name."""
    for spec in BUILTIN_SPECS:
        if tool_name in spec.tool_names:
            return spec
    raise AssertionError(f"Missing built-in spec for {tool_name}")


def _domain_package_tool_stub() -> SimpleNamespace:
    """Return a custom-tool stub exposing the built-in domain package definitions."""
    tool_definitions = [
        *RECORDER_TOOL_DEFINITIONS,
        *RESPONSE_SERVICE_TOOL_DEFINITIONS,
        *WEATHER_TOOL_DEFINITIONS,
    ]
    tool_names = {tool["name"] for tool in tool_definitions}
    return SimpleNamespace(
        get_tool_definitions=lambda: list(tool_definitions),
        is_custom_tool=lambda tool_name: tool_name in tool_names,
        get_cache_signature=lambda: tuple(sorted(tool_names)),
    )


def _llm_api_bridge_tool_stub() -> SimpleNamespace:
    """Return a custom-tool stub exposing the built-in LLM API Bridge package."""
    tool_definitions = list(LLM_API_BRIDGE_TOOL_DEFINITIONS)
    tool_names = {tool["name"] for tool in tool_definitions}
    return SimpleNamespace(
        get_tool_definitions=lambda: list(tool_definitions),
        is_custom_tool=lambda tool_name: tool_name in tool_names,
        get_builtin_toggle_spec=lambda name: _builtin_spec(name),
        get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
        get_cache_signature=lambda: tuple(sorted(tool_names)),
    )


def _is_exact_http_origin(url: object, *, scheme: str, host: str, port: int) -> bool:
    """Return whether a URL matches an exact origin."""
    parsed_url = yarl.URL(str(url))
    parsed_port = parsed_url.explicit_port
    if parsed_port is None and parsed_url.scheme == "http":
        parsed_port = 80
    if parsed_port is None and parsed_url.scheme == "https":
        parsed_port = 443
    return (
        parsed_url.scheme == scheme
        and parsed_url.host == host
        and parsed_port == port
        and parsed_url.user is None
        and parsed_url.password is None
    )


def _is_allowlisted_external_image_url(url: object) -> bool:
    """Return whether a URL is under the exact allowlisted external image base."""
    parsed_url = yarl.URL(str(url))
    if not _is_exact_http_origin(
        parsed_url,
        scheme="https",
        host="images.example.com",
        port=443,
    ):
        return False
    return parsed_url.path == "/weather" or parsed_url.path.startswith("/weather/")


def test_server_collects_allowed_ips_from_url_and_shared_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The MCP server should whitelist localhost, LM Studio host, and configured networks."""
    system_entry_factory(data={CONF_ALLOWED_IPS: "10.0.0.0/24,192.168.1.25"})
    entry = profile_entry_factory(
        data={CONF_LMSTUDIO_URL: "http://192.168.50.12:11434"}
    )

    server = MCPServer(hass, 8099, entry)

    assert "127.0.0.1" in server.allowed_ips
    assert "::1" in server.allowed_ips
    assert "192.168.50.12" in server.allowed_ips
    assert "10.0.0.0/24" in server.allowed_ips
    assert "192.168.1.25" in server.allowed_ips


@pytest.mark.asyncio
async def test_server_applies_shared_allowed_ips_and_reloads_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Running servers should refresh shared allowlists and package tools."""
    system_entry = system_entry_factory(
        data={CONF_ALLOWED_IPS: "10.0.0.0/24"}
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = SimpleNamespace(
        reload_tool_packages=AsyncMock(
            return_value={
                "external_custom_tools_enabled": False,
                "external_packages": [],
                "built_in_packages": [],
            }
        )
    )
    server.broadcast_notification = AsyncMock()
    removed_ws = Mock()
    removed_ws.close = AsyncMock()
    kept_ws = Mock()
    kept_ws.close = AsyncMock()
    server._websocket_clients = {
        removed_ws: "10.0.0.10",
        kept_ws: "192.168.1.25",
    }
    removed_sse = Mock()
    removed_sse.write_eof = AsyncMock()
    kept_sse = Mock()
    kept_sse.write_eof = AsyncMock()
    server.sse_clients = [removed_sse, kept_sse]
    server._sse_client_ips = {
        removed_sse: "10.0.0.10",
        kept_sse: "192.168.1.25",
    }
    removed_progress = asyncio.Queue()
    kept_progress = asyncio.Queue()
    server.progress_queues = {removed_progress, kept_progress}
    server._progress_queue_ips = {
        removed_progress: "10.0.0.10",
        kept_progress: "192.168.1.25",
    }

    assert "10.0.0.0/24" in server.allowed_ips

    hass.config_entries.async_update_entry(
        system_entry,
        data={**system_entry.data, CONF_ALLOWED_IPS: "192.168.1.25"},
    )

    result = await server.async_apply_shared_settings()

    assert "192.168.1.25" in server.allowed_ips
    assert "10.0.0.0/24" not in server.allowed_ips
    assert result["allowed_ips"] == server.allowed_ips
    removed_ws.close.assert_awaited_once_with(
        code=WSCloseCode.POLICY_VIOLATION,
        message=b"IP no longer authorized",
    )
    kept_ws.close.assert_not_awaited()
    removed_sse.write_eof.assert_awaited_once()
    kept_sse.write_eof.assert_not_awaited()
    assert removed_sse not in server.sse_clients
    assert kept_sse in server.sse_clients
    assert removed_progress not in server.progress_queues
    assert removed_progress.get_nowait() is None
    assert kept_progress in server.progress_queues
    assert kept_progress.empty()
    server.tools.reload_tool_packages.assert_awaited_once()
    server.broadcast_notification.assert_awaited_once_with(
        "notifications/tools/list_changed"
    )


@pytest.mark.asyncio
async def test_detail_tools_skip_non_json_serializable_values(
    hass,
    profile_entry_factory,
    system_entry_factory,
) -> None:
    """Detail tools should return usable JSON even with custom attribute objects."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    unserializable = object()
    invalid_key = object()
    server.discovery = SimpleNamespace(
        get_entity_details=AsyncMock(
            return_value={
                "sensor.custom": {
                    "state": "on",
                    "raw_object": unserializable,
                    "nested": {"keep": 1, "drop": unserializable},
                    "bad_float": float("nan"),
                    "items": ["ok", unserializable, float("inf")],
                    math.inf: "drop non-finite float key",
                    invalid_key: "drop invalid key",
                }
            }
        ),
        get_device_details=AsyncMock(
            return_value={
                "device-id": {
                    "name": "Device",
                    "identifiers": {"ok", unserializable},
                }
            }
        ),
    )

    entity_result = await server.tool_get_entity_details(
        {"entity_ids": ["sensor.custom"]}
    )
    device_result = await server.tool_get_device_details(
        {"device_ids": ["device-id"]}
    )

    entity_payload = json.loads(entity_result["content"][0]["text"])
    device_payload = json.loads(device_result["content"][0]["text"])
    entity_details = entity_payload["sensor.custom"]

    assert entity_details["state"] == "on"
    assert entity_details["nested"] == {"keep": 1}
    assert entity_details["items"] == ["ok"]
    assert "bad_float" not in entity_details
    assert "Infinity" not in entity_result["content"][0]["text"]
    assert "NaN" not in entity_result["content"][0]["text"]
    assert "raw_object" not in entity_details
    assert device_payload["device-id"]["identifiers"] == ["ok"]


@pytest.mark.asyncio
async def test_server_start_serves_health_endpoint(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Starting the MCP server should expose a working health endpoint."""
    system_entry_factory()
    server = MCPServer(hass, 0, profile_entry_factory())

    enable_socket()
    await server.start()
    try:
        assert server.site is not None
        sockets = server.site._server.sockets  # type: ignore[attr-defined]
        assert sockets
        bound_port = sockets[0].getsockname()[1]

        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{bound_port}/health") as response:
                assert response.status == 200
                payload = await response.json()

        assert payload["status"] == "healthy"
        assert payload["server"] == "ha-entity-discovery"
        assert payload["tools_available"] > 0
    finally:
        await server.stop()
        disable_socket()


@pytest.mark.asyncio
async def test_health_endpoint_rejects_unauthorized_ip(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Health diagnostics should use the same IP whitelist as other handlers."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    response = await server.handle_health(SimpleNamespace(remote="192.168.1.50"))

    assert response.status == 403
    assert response.text == "Forbidden: IP not authorized"


@pytest.mark.asyncio
async def test_prompt_overhead_endpoint_rejects_unauthorized_ip(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Prompt overhead diagnostics should use the same IP whitelist."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    response = await server.handle_prompt_overhead_diagnostics(
        SimpleNamespace(remote="192.168.1.50", query={})
    )

    assert response.status == 403
    assert response.text == "Forbidden: IP not authorized"


@pytest.mark.asyncio
async def test_prompt_overhead_diagnostics_reports_metadata_only(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Prompt overhead diagnostics should summarize size without raw schemas."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    tools = [
        {
            "name": "sample_status",
            "description": "private-schema-marker external status tool",
            "llmDescription": "Get sample status.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "private-schema-marker should not leak",
                    }
                },
            },
        },
        {
            "name": "run_script",
            "description": "Run a script.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    async def fake_handle_tools_list() -> dict[str, Any]:
        return {"tools": tools}

    def get_tool_source_info(tool_name: str) -> dict[str, str] | None:
        if tool_name != "sample_status":
            return None
        return {
            "source": "external_custom",
            "package_id": "sample_package",
            "package_name": "Sample Package",
        }

    server.handle_tools_list = fake_handle_tools_list
    server.tools = SimpleNamespace(get_tool_source_info=get_tool_source_info)

    response = await server.handle_prompt_overhead_diagnostics(
        SimpleNamespace(remote="127.0.0.1", query={"top": "5"})
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["status"] == "ok"
    assert payload["standard_context"]["tool_count"] == 2
    assert payload["standard_context"]["compact_llm_tool_schema_bytes"] > 0
    assert payload["standard_context"]["approx_llm_tool_schema_tokens"] > 0
    assert payload["adaptive_context"]["tool_count"] == 3
    assert payload["adaptive_context"]["compact_llm_tool_schema_bytes"] > 0
    assert any(
        group["source"] == "adaptive_meta"
        for group in payload["adaptive_context"]["top_tool_groups_by_schema_bytes"]
    )
    assert payload["light_context"]["tool_count"] == 1
    assert payload["light_context"]["top_tools_by_schema_bytes"][0]["name"] == "run_script"
    assert payload["standard_context"]["top_tool_groups_by_schema_bytes"][0][
        "package_id"
    ] in {"sample_package", "core"}
    assert "private-schema-marker" not in response.text
    assert "inputSchema" not in response.text


def test_tool_enablement_follows_shared_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Optional tool families should respect their shared enable/disable toggles."""
    system_entry_factory(
        data={
            CONF_ENABLE_DEVICE_TOOLS: False,
            CONF_ENABLE_ASSIST_BRIDGE: False,
            CONF_ENABLE_LLM_API_BRIDGE: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_MEMORY_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: False,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("discover_entities") is True
    assert server._is_tool_enabled("discover_devices") is False
    assert server._is_tool_enabled("list_assist_tools") is False
    assert server._is_tool_enabled("list_llm_apis") is False
    assert server._is_tool_enabled("get_calendar_events") is False
    assert server._is_tool_enabled("call_service_with_response") is False
    assert server._is_tool_enabled("get_weather_forecast") is False
    assert server._is_tool_enabled("analyze_entity_history") is False
    assert server._is_tool_enabled("list_memory_categories") is False
    assert server._is_tool_enabled("remember_memory") is False
    assert server._is_tool_enabled("add") is False
    assert server._is_tool_enabled("convert_unit") is False
    assert server._is_tool_enabled("play_music_assistant") is False
    assert server._is_tool_enabled("control_music_assistant_player") is False


def test_unit_conversion_can_stay_enabled_when_calculator_is_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Unit conversion should be independently gateable from calculator math tools."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("add") is False
    assert server._is_tool_enabled("convert_unit") is True


def test_weather_forecast_tool_and_weather_services_can_be_disabled_independently(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather forecast capability should be hideable without disabling all response services."""
    system_entry_factory(
        data={
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: True,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())

    assert server._is_tool_enabled("call_service_with_response") is True
    assert server._is_tool_enabled("get_weather_forecast") is False
    assert "Weather forecast support is disabled" in (
        server._get_domain_capability_error("weather") or ""
    )


@pytest.mark.asyncio
async def test_handle_tools_list_filters_disabled_tool_families(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Disabled optional tools should not be advertised in the MCP tool list."""
    system_entry_factory(
        data={
            CONF_ENABLE_DEVICE_TOOLS: False,
            CONF_ENABLE_ASSIST_BRIDGE: False,
            CONF_ENABLE_LLM_API_BRIDGE: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_MEMORY_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: False,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {"name": "add", "description": "calc", "inputSchema": {}},
            {"name": "convert_unit", "description": "convert", "inputSchema": {}},
        ],
        is_custom_tool=lambda tool_name: tool_name in {"add", "convert_unit"},
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "discover_entities" in tool_names
    assert "discover_devices" not in tool_names
    assert "list_assist_tools" not in tool_names
    assert "list_llm_apis" not in tool_names
    assert "get_calendar_events" not in tool_names
    assert "call_service_with_response" not in tool_names
    assert "get_weather_forecast" not in tool_names
    assert "get_entity_history" not in tool_names
    assert "list_memory_categories" not in tool_names
    assert "remember_memory" not in tool_names
    assert "play_music_assistant" not in tool_names
    assert "add" not in tool_names
    assert "convert_unit" not in tool_names


@pytest.mark.asyncio
async def test_handle_tools_list_includes_music_assistant_package_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Built-in packaged Music Assistant tools should surface through the custom tool loader."""
    system_entry_factory(data={CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: True})
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {
                "name": "play_music_assistant",
                "description": "play",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_music_assistant_queue",
                "description": "queue",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "control_music_assistant_player",
                "description": "control",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
        is_custom_tool=lambda tool_name: tool_name
        in {
            "play_music_assistant",
            "get_music_assistant_queue",
            "control_music_assistant_player",
        },
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "play_music_assistant" in tool_names
    assert "get_music_assistant_queue" in tool_names
    assert "control_music_assistant_player" in tool_names


@pytest.mark.asyncio
async def test_handle_tool_call_routes_music_assistant_to_custom_tool_loader(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Music Assistant calls should be dispatched through the packaged custom tool loader."""
    system_entry_factory(data={CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: True})
    server = MCPServer(hass, 8099, profile_entry_factory())
    calls = []

    async def handle_tool_call(tool_name, arguments, *, context=None):
        calls.append((tool_name, arguments, context))
        return {"content": [{"type": "text", "text": "called package"}]}

    server.tools = SimpleNamespace(
        is_custom_tool=lambda tool_name: tool_name == "play_music_assistant",
        handle_tool_call=handle_tool_call,
    )

    result = await server.handle_tool_call(
        {
            "name": "play_music_assistant",
            "arguments": {"media_type": "track", "media_id": "song"},
            "context": {"profile_entry_id": "profile-1"},
        }
    )

    assert result["content"][0]["text"] == "called package"
    assert calls == [
        (
            "play_music_assistant",
            {"media_type": "track", "media_id": "song"},
            {"profile_entry_id": "profile-1"},
        )
    ]


@pytest.mark.asyncio
async def test_handle_tool_call_routes_domain_packages_to_custom_tool_loader(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Domain package tool calls should be dispatched through the custom tool loader."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    calls = []

    async def handle_tool_call(tool_name, arguments, *, context=None):
        calls.append((tool_name, arguments, context))
        return {"content": [{"type": "text", "text": f"called {tool_name}"}]}

    server.tools = SimpleNamespace(
        is_custom_tool=lambda tool_name: tool_name
        in {"get_calendar_events", "get_entity_history"},
        handle_tool_call=handle_tool_call,
    )

    calendar_result = await server.handle_tool_call(
        {
            "name": "get_calendar_events",
            "arguments": {"query": "Mariners"},
            "context": {"profile_entry_id": "profile-1"},
        }
    )
    history_result = await server.handle_tool_call(
        {
            "name": "get_entity_history",
            "arguments": {"entity_id": "binary_sensor.front_door"},
        }
    )

    assert calendar_result["content"][0]["text"] == "called get_calendar_events"
    assert history_result["content"][0]["text"] == "called get_entity_history"
    assert calls == [
        (
            "get_calendar_events",
            {"query": "Mariners"},
            {"profile_entry_id": "profile-1"},
        ),
        (
            "get_entity_history",
            {"entity_id": "binary_sensor.front_door"},
            {},
        ),
    ]


@pytest.mark.asyncio
async def test_handle_tool_call_logs_metadata_without_arguments_or_results(
    hass, profile_entry_factory, system_entry_factory, caplog
) -> None:
    """Tool dispatch logs should describe argument shape without raw values."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    async def handle_tool_call(tool_name, arguments, *, context=None):
        return {"content": [{"type": "text", "text": "tool-result-secret"}]}

    server.tools = SimpleNamespace(
        is_custom_tool=lambda tool_name: tool_name == "private_tool_status",
        handle_tool_call=handle_tool_call,
    )

    with caplog.at_level(logging.DEBUG, logger=mcp_server_module._LOGGER.name):
        result = await server.handle_tool_call(
            {
                "name": "private_tool_status",
                "arguments": {"entity_id": "light.private", "password": "super-secret"},
                "context": {"profile_entry_id": "profile-private"},
            }
        )

    assert result["content"][0]["text"] == "tool-result-secret"
    assert "private_tool_status" in caplog.text
    assert "argument_count=2" in caplog.text
    assert "context_count=1" in caplog.text
    assert "argument_bytes=" in caplog.text
    assert "entity_id" not in caplog.text
    assert "password" not in caplog.text
    assert "profile_entry_id" not in caplog.text
    assert "light.private" not in caplog.text
    assert "super-secret" not in caplog.text
    assert "profile-private" not in caplog.text
    assert "tool-result-secret" not in caplog.text


@pytest.mark.asyncio
async def test_handle_tools_list_can_keep_unit_conversion_when_calculator_math_is_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Package-based built-ins should still honor the separate global math/unit toggles."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {"name": "add", "description": "calc", "inputSchema": {}},
            {"name": "convert_unit", "description": "convert", "inputSchema": {}},
        ],
        is_custom_tool=lambda tool_name: tool_name in {"add", "convert_unit"},
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "add" not in tool_names
    assert "convert_unit" in tool_names


@pytest.mark.asyncio
async def test_handle_tools_list_hides_web_search_tools_when_shared_toggle_is_off(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The shared web-search toggle should still hide search and read_url together."""
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = SimpleNamespace(
        get_tool_definitions=lambda: [
            {"name": "search", "description": "search", "inputSchema": {}},
            {"name": "read_url", "description": "read", "inputSchema": {}},
        ],
        is_custom_tool=lambda tool_name: tool_name in {"search", "read_url"},
    )

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "search" not in tool_names
    assert "read_url" not in tool_names


def test_package_based_search_and_read_url_can_be_toggled_independently(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared built-in package toggles should not force search and read_url together."""
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": False,
            CONF_ENABLE_WEB_SEARCH: False,
            "search_provider": "brave",
        }
    )
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = SimpleNamespace(
        get_builtin_toggle_spec=lambda name: _builtin_spec(name),
        get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
    )

    assert server._is_tool_enabled("search") is True
    assert server._is_tool_enabled("read_url") is False


@pytest.mark.asyncio
async def test_default_tool_list_stays_streamlined(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Compatibility aliases and optional bridge tools should stay out of the default list."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = _domain_package_tool_stub()

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}

    assert "get_entity_history" in tool_names
    assert "get_calendar_events" in tool_names
    assert "list_memory_categories" not in tool_names
    assert "remember_memory" not in tool_names
    assert "get_last_entity_event" not in tool_names
    assert "list_assist_tools" not in tool_names
    assert "list_llm_apis" not in tool_names

    get_index_tool = next(
        tool for tool in result["tools"] if tool["name"] == "get_index"
    )
    assert "start of a conversation" not in get_index_tool["description"]
    assert "do not call by default" in get_index_tool["description"]


@pytest.mark.asyncio
async def test_llm_api_bridge_lists_allowlisted_registered_apis(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The optional LLM API bridge should list registered non-Assist APIs."""
    unregister = llm.async_register_api(
        hass,
        _FakeLLMAPI(hass=hass, id="llm_intents", name="LLM Intents"),
    )
    try:
        system_entry_factory(
            data={
                CONF_ENABLE_LLM_API_BRIDGE: True,
                CONF_LLM_API_ALLOWLIST: "llm_intents\nmissing_api",
            }
        )
        server = MCPServer(hass, 8099, profile_entry_factory())
        server.tools = _llm_api_bridge_tool_stub()
        bridge_tool = LLMApiBridgeTool(hass)

        tools_result = await server.handle_tools_list()
        tool_names = {tool["name"] for tool in tools_result["tools"]}
        list_result = await bridge_tool.handle_call("list_llm_apis", {})
        payload = _json_payload_from_text_result(list_result)

        assert {
            "list_llm_apis",
            "list_llm_api_tools",
            "call_llm_api_tool",
            "get_llm_api_prompt",
        } <= tool_names
        assert payload["enabled"] is True
        assert payload["allowed_api_ids"] == ["llm_intents", "missing_api"]
        assert payload["missing_allowed_api_ids"] == ["missing_api"]
        assert payload["apis"] == [
            {"id": "llm_intents", "name": "LLM Intents", "allowed": True}
        ]
    finally:
        unregister()


@pytest.mark.asyncio
async def test_llm_api_bridge_can_inspect_call_and_read_prompt(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Allowlisted third-party LLM API tools should be inspectable and callable."""
    unregister = llm.async_register_api(
        hass,
        _FakeLLMAPI(hass=hass, id="llm_intents", name="LLM Intents"),
    )
    try:
        system_entry_factory(
            data={
                CONF_ENABLE_LLM_API_BRIDGE: True,
                CONF_LLM_API_ALLOWLIST: "llm_intents",
            }
        )
        bridge_tool = LLMApiBridgeTool(hass)

        tools_result = await bridge_tool.handle_call(
            "list_llm_api_tools", {"api_id": "llm_intents"}
        )
        tools_payload = _json_payload_from_text_result(tools_result)
        call_result = await bridge_tool.handle_call(
            "call_llm_api_tool",
            {
                "api_id": "llm_intents",
                "tool_name": "EchoIntent",
                "arguments": {"value": "hello"},
            },
        )
        prompt_result = await bridge_tool.handle_call(
            "get_llm_api_prompt", {"api_id": "llm_intents"}
        )

        assert tools_payload["api_id"] == "llm_intents"
        assert tools_payload["tools"][0]["name"] == "EchoIntent"
        assert tools_payload["tools"][0]["input_schema"]["required"] == ["value"]
        assert '"echo": "hello"' in call_result["content"][0]["text"]
        assert '"external": true' in call_result["content"][0]["text"]
        assert '"platform": "mcp_assist"' in call_result["content"][0]["text"]
        assert "Use EchoIntent for fake LLM intent testing." in (
            prompt_result["content"][0]["text"]
        )
    finally:
        unregister()


@pytest.mark.asyncio
async def test_llm_api_bridge_rejects_unallowlisted_and_assist_api(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The bridge should fail closed for untrusted API ids and the built-in Assist API."""
    system_entry_factory(
        data={
            CONF_ENABLE_LLM_API_BRIDGE: True,
            CONF_LLM_API_ALLOWLIST: "llm_intents",
        }
    )
    bridge_tool = LLMApiBridgeTool(hass)

    with pytest.raises(HomeAssistantError, match="not allowlisted"):
        await bridge_tool.handle_call("list_llm_api_tools", {"api_id": "other_api"})

    with pytest.raises(HomeAssistantError, match="Assist API"):
        await bridge_tool.handle_call(
            "list_llm_api_tools", {"api_id": llm.LLM_API_ASSIST}
        )


@pytest.mark.asyncio
async def test_handle_tools_list_uses_cache_for_stable_signature(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Repeated tools/list requests should reuse the cached tool surface when settings are unchanged."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    custom_tools = SimpleNamespace(
        tools={},
        get_cache_signature=lambda: ("stable",),
        get_tool_definitions=lambda: [
            {"name": "search", "description": "search", "inputSchema": {"type": "object", "properties": {}}},
        ],
    )
    server.tools = custom_tools

    result_one = await server.handle_tools_list()
    custom_tools.get_tool_definitions = lambda: (_ for _ in ()).throw(
        AssertionError("tools/list should have been served from cache")
    )
    result_two = await server.handle_tools_list()

    assert result_one == result_two


@pytest.mark.asyncio
async def test_handle_tools_list_invalidates_cache_when_custom_tool_surface_changes(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A changed external custom-tool surface should invalidate the cached tools/list response."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    state = {
        "signature": ("v1",),
        "definitions": [
            {
                "name": "sample_tool_status",
                "description": "sample status",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
    }
    server.tools = SimpleNamespace(
        tools={},
        get_cache_signature=lambda: state["signature"],
        get_tool_definitions=lambda: state["definitions"],
    )

    first = await server.handle_tools_list()

    state["signature"] = ("v2",)
    state["definitions"] = [
        {
            "name": "sample_tool_history",
            "description": "sample history",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]

    second = await server.handle_tools_list()

    first_names = {tool["name"] for tool in first["tools"]}
    second_names = {tool["name"] for tool in second["tools"]}

    assert "sample_tool_status" in first_names
    assert "sample_tool_history" not in first_names
    assert "sample_tool_history" in second_names
    assert "sample_tool_status" not in second_names


@pytest.mark.asyncio
async def test_handle_tool_call_rejects_disabled_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Disabled tools should fail closed even if called directly."""
    system_entry_factory(data={CONF_ENABLE_DEVICE_TOOLS: False})
    server = MCPServer(hass, 8099, profile_entry_factory())

    with pytest.raises(ValueError, match="disabled"):
        await server.handle_tool_call({"name": "discover_devices", "arguments": {}})


@pytest.mark.asyncio
async def test_memory_tools_store_recall_and_forget(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Memory tools should store, search, and delete persisted memories."""
    system_entry_factory(data={CONF_ENABLE_MEMORY_TOOLS: True})
    memory_tool = MemoryTool(hass)
    await memory_tool.initialize()
    try:
        remembered = await memory_tool.handle_call(
            "remember_memory",
            {"memory": "Front door code is changing next week", "category": "household"},
        )
        recalled = await memory_tool.handle_call(
            "recall_memories", {"query": "front door code"}
        )
        memory_id = recalled["memories"][0]["id"]
        forgotten = await memory_tool.handle_call(
            "forget_memory", {"memory_id": memory_id}
        )
        recalled_again = await memory_tool.handle_call(
            "recall_memories", {"query": "front door code"}
        )
    finally:
        await memory_tool.async_shutdown()

    assert "Stored memory" in remembered["content"][0]["text"]
    assert recalled["result_count"] == 1
    assert forgotten["deleted_count"] == 1
    assert recalled_again["result_count"] == 0


@pytest.mark.asyncio
async def test_list_memory_categories_reports_suggestions_and_counts(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Memory category discovery should report suggestions and active counts."""
    system_entry_factory(data={CONF_ENABLE_MEMORY_TOOLS: True})
    memory_tool = MemoryTool(hass)
    await memory_tool.initialize()
    try:
        await memory_tool.handle_call(
            "remember_memory",
            {"memory": "The den lamp is called the reading light", "category": "alias"},
        )
        await memory_tool.handle_call(
            "remember_memory",
            {
                "memory": "The office is usually 69 degrees overnight",
                "category": "normal",
            },
        )

        result = await memory_tool.handle_call("list_memory_categories", {})
    finally:
        await memory_tool.async_shutdown()
    category_counts = {item["category"]: item["count"] for item in result["categories"]}

    assert "Suggested memory categories" in result["content"][0]["text"]
    assert category_counts["device_alias"] == 1
    assert category_counts["baseline"] == 1
    assert result["total_count"] == 2


@pytest.mark.asyncio
async def test_tool_discover_entities_reports_paging_metadata(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Entity discovery responses should tell callers when more results are available."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.discover_entities_page = AsyncMock(
        return_value={
            "items": [
                {
                    "entity_id": "light.kitchen",
                    "name": "Kitchen Light",
                    "state": "on",
                    "area": "Kitchen",
                },
                {
                    "entity_id": "light.pantry",
                    "name": "Pantry Light",
                    "state": "on",
                    "area": "Kitchen",
                },
            ],
            "total_found": 5,
            "returned_count": 2,
            "remaining_count": 3,
            "offset": 0,
            "limit": 2,
            "has_more": True,
            "next_offset": 2,
        }
    )

    result = await server.tool_discover_entities({"domain": "light", "limit": 2})

    assert "Showing 1-2 of 5 entities; 3 more available (next_offset=2):" in (
        result["content"][0]["text"]
    )
    assert result["pagination"]["next_offset"] == 2
    assert len(result["entities"]) == 2


@pytest.mark.asyncio
async def test_tool_discover_devices_reports_paging_metadata(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Device discovery responses should include paging metadata too."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.discover_devices_page = AsyncMock(
        return_value={
            "items": [
                {
                    "device_id": "abc123",
                    "name": "Kitchen Speaker",
                    "entity_count": 3,
                    "domains": ["media_player"],
                    "entities_preview": [{"entity_id": "media_player.kitchen"}],
                }
            ],
            "total_found": 4,
            "returned_count": 1,
            "remaining_count": 3,
            "offset": 1,
            "limit": 1,
            "has_more": True,
            "next_offset": 2,
        }
    )

    result = await server.tool_discover_devices({"domain": "media_player", "limit": 1, "offset": 1})

    assert "Showing 2-2 of 4 devices; 2 more available (next_offset=2)" in (
        result["content"][0]["text"]
    )
    assert result["pagination"]["offset"] == 1
    assert result["pagination"]["next_offset"] == 2


def test_validate_service_blocks_music_assistant_when_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Generic service actions should also respect the Music Assistant toggle."""
    system_entry_factory(data={CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False})
    server = MCPServer(hass, 8099, profile_entry_factory())

    with pytest.raises(ValueError, match="Music Assistant support is disabled"):
        server.validate_service("music_assistant", "play_media")


@pytest.mark.asyncio
async def test_list_domains_hides_disabled_music_assistant_domain(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Domain summaries should not advertise disabled optional domains."""
    system_entry_factory(data={CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: False})
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.list_domains = AsyncMock(
        return_value=[
            {"domain": "light", "count": 1},
            {"domain": "music_assistant", "count": 2},
        ]
    )

    result = await server.tool_list_domains()
    text = result["content"][0]["text"]

    assert "light: 1 entities" in text
    assert "music_assistant" not in text


def test_general_discovery_results_group_by_area_and_sort_names(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """General discovery output should be grouped by area and sorted predictably."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    text = server._format_general_discovery_results(
        [
            {
                "entity_id": "light.primary_bedroom_bedside_right",
                "name": "Primary Bedroom Bedside Lamp: Right",
                "state": "on",
                "area": "Primary Bedroom",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.office_desk_strip",
                "name": "Office Desk Light Strip",
                "state": "on",
                "area": "Office",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.primary_bedroom_bedside_left",
                "name": "Primary Bedroom Bedside Lamp: Left",
                "state": "on",
                "area": "Primary Bedroom",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.office_cans",
                "name": "Office Cans",
                "state": "on",
                "area": "Office",
                "floor": "Downstairs",
            },
            {
                "entity_id": "light.primary_bathroom_vanity",
                "name": "Primary Bathroom Vanity",
                "state": "on",
                "area": "Primary Bathroom",
                "floor": "Downstairs",
            },
        ]
    )

    assert "Found 5 entities across 3 groups:" in text
    assert text.index("Office (2):") < text.index("Primary Bathroom (1):")
    assert text.index("Primary Bathroom (1):") < text.index("Primary Bedroom (2):")
    assert text.index("Office Cans") < text.index("Office Desk Light Strip")
    assert text.index("Primary Bedroom Bedside Lamp: Left") < text.index(
        "Primary Bedroom Bedside Lamp: Right"
    )


def test_general_discovery_results_keep_no_area_bucket_last(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Ungrouped entities should appear after named rooms."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    text = server._format_general_discovery_results(
        [
            {
                "entity_id": "light.loft_lamp",
                "name": "Loft Lamp",
                "state": "on",
                "floor": "Upstairs",
            },
            {
                "entity_id": "light.kitchen_pendants",
                "name": "Kitchen Pendants",
                "state": "on",
                "area": "Kitchen",
                "floor": "Downstairs",
            },
        ]
    )

    assert text.index("Kitchen (Downstairs) (1):") < text.index(
        "No area (Upstairs) (1):"
    )


def test_prepare_response_service_data_uses_supported_weather_forecast_type(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather response calls should default to a supported forecast type."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "supported_features": int(
                WeatherEntityFeature.FORECAST_HOURLY
                | WeatherEntityFeature.FORECAST_TWICE_DAILY
            )
        },
    )

    prepared = server._prepare_response_service_data(
        "weather",
        "get_forecasts",
        {},
        resolved_target={"entity_id": ["weather.home"]},
    )

    assert prepared["type"] == "twice_daily"


def test_prepare_response_service_data_adjusts_unsupported_weather_forecast_type(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather response calls should correct unsupported forecast types."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "supported_features": int(
                WeatherEntityFeature.FORECAST_HOURLY
                | WeatherEntityFeature.FORECAST_TWICE_DAILY
            )
        },
    )

    prepared = server._prepare_response_service_data(
        "weather",
        "get_forecasts",
        {"type": "daily"},
        resolved_target={"entity_id": ["weather.home"]},
    )

    assert prepared["type"] == "twice_daily"


@pytest.mark.asyncio
async def test_call_service_with_response_uses_supported_weather_forecast_type(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Weather response-service calls should use the entity's supported type."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "supported_features": int(
                WeatherEntityFeature.FORECAST_HOURLY
                | WeatherEntityFeature.FORECAST_TWICE_DAILY
            )
        },
    )

    server._get_response_service_info = AsyncMock(
        return_value=(
            {
                "fields": {"type": {"required": True}},
                "target": {"entity": {"domain": "weather"}},
            },
            None,
        )
    )
    server.resolve_target = AsyncMock(return_value={"entity_id": ["weather.home"]})
    async_call_mock = AsyncMock(
        return_value={
            "weather.home": {
                "forecast": [
                    {
                        "datetime": "2026-04-13T08:00:00-07:00",
                        "condition": "sunny",
                        "temperature": 72,
                    }
                ]
            }
        }
    )
    monkeypatch.setattr(type(hass.services), "async_call", async_call_mock)

    result = await server.tool_call_service_with_response(
        {
            "domain": "weather",
            "service": "get_forecasts",
            "target": {"entity_id": "weather.home"},
            "data": {},
        }
    )

    async_call_mock.assert_awaited_once()
    service_data = async_call_mock.await_args.kwargs["service_data"]
    assert service_data["type"] == "twice_daily"
    assert "Forecast type used: twice_daily." in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_weather_forecast_discovers_entity_and_summarizes_tomorrow(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather forecast helper should use an HA weather entity instead of acting source-less."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "weather.home",
        "sunny",
        {
            "friendly_name": "Weather",
            "supported_features": int(WeatherEntityFeature.FORECAST_TWICE_DAILY),
        },
    )
    server.discovery.discover_entities = AsyncMock(
        return_value=[
            {
                "entity_id": "weather.home",
                "name": "Weather",
                "forecast_service_supported": True,
            }
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "weather.home": {
                    "forecast": [
                        {
                            "datetime": (
                                dt_util.now().replace(hour=9, minute=0, second=0, microsecond=0)
                                + timedelta(days=1)
                            ).isoformat(),
                            "condition": "sunny",
                            "temperature": 72,
                            "is_daytime": True,
                        },
                        {
                            "datetime": (
                                dt_util.now().replace(hour=21, minute=0, second=0, microsecond=0)
                                + timedelta(days=1)
                            ).isoformat(),
                            "condition": "cloudy",
                            "temperature": 61,
                            "is_daytime": False,
                        },
                    ]
                }
            },
        }
    )

    result = await server.tool_get_weather_forecast({"when": "tomorrow"})

    server.tool_call_service_with_response.assert_awaited_once()
    request_args = server.tool_call_service_with_response.await_args.args[0]
    assert request_args["domain"] == "weather"
    assert request_args["service"] == "get_forecasts"
    assert request_args["target"] == {"entity_id": ["weather.home"]}
    assert request_args["data"]["type"] == "twice_daily"
    assert "Tomorrow for Weather:" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_calendar_events_discovers_calendar_and_summarizes_next_event(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Calendar helper should discover a relevant calendar and summarize the next event."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "calendar.mariners_baseball",
        "off",
        {"friendly_name": "Mariners Baseball"},
    )
    server.discovery.discover_entities = AsyncMock(
        return_value=[
            {
                "entity_id": "calendar.mariners_baseball",
                "name": "Mariners Baseball",
            }
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "calendar.mariners_baseball": {
                    "events": [
                        {
                            "summary": "Rangers at Mariners",
                            "start": "2026-04-14T18:40:00-07:00",
                            "end": "2026-04-14T21:40:00-07:00",
                            "location": "T-Mobile Park",
                        }
                    ]
                }
            },
        }
    )

    result = await server.tool_get_calendar_events({"query": "Mariners", "limit": 1})

    server.tool_call_service_with_response.assert_awaited_once()
    request_args = server.tool_call_service_with_response.await_args.args[0]
    assert request_args["domain"] == "calendar"
    assert request_args["service"] == "get_events"
    assert request_args["target"] == {"entity_id": ["calendar.mariners_baseball"]}
    assert "start_date_time" in request_args["data"]
    assert "end_date_time" in request_args["data"]
    assert "Next matching calendar event:" in result["content"][0]["text"]
    assert "Rangers at Mariners" in result["content"][0]["text"]
    assert result["structuredContent"]["selected_calendars"] == [
        {
            "entity_id": "calendar.mariners_baseball",
            "name": "Mariners Baseball",
        }
    ]


@pytest.mark.asyncio
async def test_get_calendar_events_falls_back_to_event_text_search(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """General query text should fall back to event-text matching across calendars."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "calendar.ash_jason",
        "off",
        {"friendly_name": "Ash & Jason"},
    )
    server.discovery.discover_entities = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "entity_id": "calendar.ash_jason",
                    "name": "Ash & Jason",
                }
            ],
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "calendar.ash_jason": {
                    "events": [
                        {
                            "summary": "Dentist cleaning",
                            "description": "Downtown appointment",
                            "start": "2026-04-18T09:00:00-07:00",
                            "end": "2026-04-18T10:00:00-07:00",
                        },
                        {
                            "summary": "Birthday party",
                            "start": "2026-04-19T14:00:00-07:00",
                            "end": "2026-04-19T16:00:00-07:00",
                        },
                    ]
                }
            },
        }
    )

    result = await server.tool_get_calendar_events({"query": "dentist", "limit": 1})

    assert server.discovery.discover_entities.await_count == 2
    first_call = server.discovery.discover_entities.await_args_list[0]
    second_call = server.discovery.discover_entities.await_args_list[1]
    assert first_call.kwargs["name_contains"] == "dentist"
    assert "name_contains" not in second_call.kwargs or second_call.kwargs["name_contains"] is None
    assert "Dentist cleaning" in result["content"][0]["text"]
    assert result["structuredContent"]["event_text"] == "dentist"


@pytest.mark.asyncio
async def test_get_calendar_events_prefers_named_sports_calendar_over_personal_event_match(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Generic words like 'game' should not force a fallback to personal calendar events."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "calendar.mariners_baseball",
        "off",
        {"friendly_name": "Mariners Baseball"},
    )
    hass.states.async_set(
        "calendar.ash_jason",
        "off",
        {"friendly_name": "Ash & Jason"},
    )
    server.discovery.discover_entities = AsyncMock(
        side_effect=[
            [],
            [
                {
                    "entity_id": "calendar.mariners_baseball",
                    "name": "Mariners Baseball",
                }
            ],
        ]
    )
    server.tool_call_service_with_response = AsyncMock(
        return_value={
            "content": [{"type": "text", "text": "ok"}],
            "response": {
                "calendar.mariners_baseball": {
                    "events": [
                        {
                            "summary": "Astros @ Mariners",
                            "start": "2026-04-12T13:10:00-07:00",
                            "end": "2026-04-12T16:10:00-07:00",
                            "location": "T-Mobile Park",
                        }
                    ]
                },
                "calendar.ash_jason": {
                    "events": [
                        {
                            "summary": "Mariners Game <3",
                            "start": "2026-04-18T15:00:00-07:00",
                            "end": "2026-04-18T20:00:00-07:00",
                        }
                    ]
                },
            },
        }
    )

    result = await server.tool_get_calendar_events({"query": "Mariners game", "limit": 1})

    assert server.discovery.discover_entities.await_count == 2
    first_call = server.discovery.discover_entities.await_args_list[0]
    second_call = server.discovery.discover_entities.await_args_list[1]
    assert first_call.kwargs["name_contains"] == "Mariners game"
    assert second_call.kwargs["name_contains"] == "mariners"

    request_args = server.tool_call_service_with_response.await_args.args[0]
    assert request_args["target"] == {"entity_id": ["calendar.mariners_baseball"]}
    assert "Astros @ Mariners" in result["content"][0]["text"]
    assert "Mariners Game <3" not in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_resolve_weather_forecast_target_uses_generic_ranking(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Weather target resolution should not rely on install-specific entity names."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.discovery.discover_entities = AsyncMock(
        return_value=[
            {
                "entity_id": "weather.weather",
                "name": "Weather",
                "forecast_service_supported": False,
                "forecast_available": False,
            },
            {
                "entity_id": "weather.acme_sky",
                "name": "Acme Sky",
                "forecast_service_supported": True,
                "forecast_available": True,
            },
            {
                "entity_id": "weather.zeta_forecast",
                "name": "Zeta Forecast",
                "forecast_service_supported": True,
                "forecast_available": True,
            },
        ]
    )

    resolved_target, entity_info = await server._resolve_weather_forecast_target()

    assert resolved_target == {"entity_id": ["weather.acme_sky"]}
    assert entity_info == {
        "entity_id": "weather.acme_sky",
        "name": "Acme Sky",
    }


@pytest.mark.asyncio
async def test_observe_action_outcome_confirms_expected_lock_state(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Mechanical actions should confirm completion once the expected state is reached."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "lock.front_door_deadbolt",
        "locked",
        {"friendly_name": "Front Door Deadbolt"},
    )

    result = await server._observe_action_outcome(
        domain="lock",
        service="lock",
        entity_ids=["lock.front_door_deadbolt"],
    )

    assert result["status"] == "confirmed"
    assert result["progress_phrase"] == "locking"
    assert result["state_lines"] == ["  • Front Door Deadbolt: locked"]


@pytest.mark.asyncio
async def test_tool_perform_action_reports_pending_lock_transition(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Slow lock transitions should be reported as pending, not as failures."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.resolve_target = AsyncMock(
        return_value={"entity_id": ["lock.front_door_deadbolt"]}
    )
    server._observe_action_outcome = AsyncMock(
        return_value={
            "status": "pending",
            "progress_phrase": "locking",
            "state_lines": ["  • Front Door Deadbolt: unlocked"],
        }
    )
    async_call_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(type(hass.services), "async_call", async_call_mock)

    result = await server.tool_perform_action(
        {
            "domain": "lock",
            "action": "lock",
            "target": {"entity_id": "lock.front_door_deadbolt"},
            "data": {},
        }
    )

    text = result["content"][0]["text"]
    async_call_mock.assert_awaited_once()
    assert text.startswith("✅ Sent lock.lock")
    assert "may still be locking" in text
    assert "Current states right now:" in text
    assert "Front Door Deadbolt: unlocked" in text
    assert "Successfully executed lock.lock" not in text


def test_history_resolution_prefers_related_contact_sensor_for_open_requests(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Open-history requests should prefer a strongly matching door sensor over a lock."""
    system_entry_factory()
    entry = profile_entry_factory()
    server = MCPServer(hass, 8099, entry)

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    lock_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_lock")},
        name="Front Door Deadbolt",
    )
    contact_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_contact")},
        name="Front Door",
    )

    entity_registry.async_get_or_create(
        "lock",
        "test",
        "front_door_deadbolt",
        suggested_object_id="front_door_deadbolt",
        device_id=lock_device.id,
    )
    entity_registry.async_get_or_create(
        "binary_sensor",
        "test",
        "front_door_contact",
        suggested_object_id="front_door",
        device_id=contact_device.id,
    )

    hass.states.async_set(
        "lock.front_door_deadbolt",
        "locked",
        {"friendly_name": "Front Door Deadbolt"},
    )
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"friendly_name": "Front Door", "device_class": "door"},
    )

    with patch(
        "custom_components.mcp_assist.tools.packages.recorder.history.async_should_expose",
        return_value=True,
    ):
        history_entity_id, resolution_note = server._resolve_history_entity_for_request(
            "lock.front_door_deadbolt",
            None,
            "opened",
        )

    assert history_entity_id == "binary_sensor.front_door"
    assert resolution_note is not None
    assert "binary_sensor.front_door" in resolution_note


@pytest.mark.asyncio
async def test_analyze_history_falls_back_to_related_contact_sensor_when_primary_has_no_matches(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Count analysis should try strong related entities before returning zero matches."""
    system_entry_factory()
    entry = profile_entry_factory()
    server = MCPServer(hass, 8099, entry)

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    lock_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_lock_count")},
        name="Front Door Deadbolt",
    )
    contact_device = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "front_door_contact_count")},
        name="Front Door",
    )

    entity_registry.async_get_or_create(
        "lock",
        "test",
        "front_door_deadbolt_count",
        suggested_object_id="front_door_deadbolt",
        device_id=lock_device.id,
    )
    entity_registry.async_get_or_create(
        "binary_sensor",
        "test",
        "front_door_contact_count",
        suggested_object_id="front_door",
        device_id=contact_device.id,
    )

    hass.states.async_set(
        "lock.front_door_deadbolt",
        "locked",
        {"friendly_name": "Front Door Deadbolt"},
    )
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"friendly_name": "Front Door", "device_class": "door"},
    )

    now = dt_util.utcnow()

    async def fake_fetch(entity_id, *args, **kwargs):
        if entity_id == "lock.front_door_deadbolt":
            return [
                SimpleNamespace(
                    state="locked",
                    last_changed=now - timedelta(hours=12),
                    last_updated=now - timedelta(hours=12),
                )
            ]
        if entity_id == "binary_sensor.front_door":
            return [
                SimpleNamespace(
                    state="off",
                    last_changed=now - timedelta(hours=24),
                    last_updated=now - timedelta(hours=24),
                ),
                SimpleNamespace(
                    state="on",
                    last_changed=now - timedelta(hours=18),
                    last_updated=now - timedelta(hours=18),
                ),
                SimpleNamespace(
                    state="off",
                    last_changed=now - timedelta(hours=17, minutes=55),
                    last_updated=now - timedelta(hours=17, minutes=55),
                ),
            ]
        raise AssertionError(f"Unexpected entity history fetch for {entity_id}")

    server._fetch_entity_history_states = AsyncMock(side_effect=fake_fetch)

    with patch(
        "custom_components.mcp_assist.tools.packages.recorder.history.async_should_expose",
        return_value=True,
    ):
        result = await server.tool_analyze_entity_history(
            {
                "entity_id": "lock.front_door_deadbolt",
                "event": "opened",
                "analysis": "count",
                "hours": 24,
            }
        )

    text = result["content"][0]["text"]

    assert "Using related entity binary_sensor.front_door" in text
    assert "Front Door (binary_sensor.front_door)" in text
    assert "Recorded opened event during the last 24 hours: 1" in text


@pytest.mark.asyncio
async def test_analyze_history_counts_calendar_yesterday_transitions(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Calendar-day count questions should query the exact local day window."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"friendly_name": "Front Door", "device_class": "door"},
    )

    fixed_now = datetime(2026, 4, 20, 21, 0, tzinfo=timezone.utc)
    expected_start = datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc)
    expected_end = datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "custom_components.mcp_assist.tools.packages.recorder.history.dt_util.now",
        lambda: fixed_now,
    )
    monkeypatch.setattr(
        "custom_components.mcp_assist.tools.packages.recorder.history.dt_util.utcnow",
        lambda: fixed_now,
    )

    history_rows = []
    for index in range(35):
        history_rows.append(
            SimpleNamespace(
                state="on",
                last_changed=expected_start + timedelta(minutes=index * 10 + 1),
                last_updated=expected_start + timedelta(minutes=index * 10 + 1),
            )
        )
        history_rows.append(
            SimpleNamespace(
                state="off",
                last_changed=expected_start + timedelta(minutes=index * 10 + 2),
                last_updated=expected_start + timedelta(minutes=index * 10 + 2),
            )
        )

    async def fake_fetch(entity_id, *args, **kwargs):
        assert entity_id == "binary_sensor.front_door"
        assert kwargs["start_time"] == expected_start
        assert kwargs["end_time"] == expected_end
        assert kwargs["include_start_time_state"] is False
        return history_rows

    server._fetch_entity_history_states = AsyncMock(side_effect=fake_fetch)

    result = await server.tool_analyze_entity_history(
        {
            "entity_id": "binary_sensor.front_door",
            "event": "opened",
            "analysis": "count",
            "period": "yesterday",
        }
    )

    text = result["content"][0]["text"]
    assert "Recorded opened events during yesterday: 35" in text
    assert "Counted using recorder state: on" in text


@pytest.mark.asyncio
async def test_get_entity_history_timeline_keeps_newest_states_first(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Timeline history should keep the recorder's newest-first ordering."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set(
        "binary_sensor.front_door",
        "off",
        {"friendly_name": "Front Door"},
    )
    await hass.async_block_till_done()

    newest = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    middle = newest - timedelta(minutes=15)
    oldest = newest - timedelta(minutes=30)
    history_rows = [
        SimpleNamespace(state="open", last_changed=newest, last_updated=newest),
        SimpleNamespace(state="closed", last_changed=middle, last_updated=middle),
        SimpleNamespace(state="jammed", last_changed=oldest, last_updated=oldest),
    ]

    server._fetch_entity_history_states = AsyncMock(return_value=history_rows)
    marker_by_time = {
        newest: "newest",
        middle: "middle",
        oldest: "oldest",
    }
    server._format_relative_absolute_time = lambda when: marker_by_time[when]

    result = await server.tool_get_entity_history(
        {
            "entity_id": "binary_sensor.front_door",
            "hours": 24,
            "limit": 3,
        }
    )

    text = result["content"][0]["text"]
    timeline_lines = [line for line in text.splitlines() if line.startswith("• ")]
    assert timeline_lines == [
        "• newest → open",
        "• middle → closed",
        "• oldest → jammed",
    ]
    server._fetch_entity_history_states.assert_awaited_once()
    fetch_kwargs = server._fetch_entity_history_states.await_args.kwargs
    assert fetch_kwargs["descending"] is True
    assert fetch_kwargs["limit"] == 3


def test_format_relative_absolute_time_uses_local_timezone(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Relative/absolute time formatting should render recorder timestamps in local time."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    original_tz = dt_util.DEFAULT_TIME_ZONE
    frozen_now = datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc)
    historical_when = datetime(2026, 5, 1, 1, 15, tzinfo=timezone.utc)

    try:
        dt_util.set_default_time_zone(ZoneInfo("America/Los_Angeles"))
        with patch.object(mcp_server_module.dt_util, "utcnow", return_value=frozen_now):
            formatted = server._format_relative_absolute_time(historical_when)
    finally:
        dt_util.set_default_time_zone(original_tz)

    assert formatted == "45 minutes ago at 6:15 PM PDT today"


@pytest.mark.asyncio
async def test_get_entity_history_count_mode_delegates_to_analyzer(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The generic history tool should not return a limited timeline for count mode."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tool_analyze_entity_history = AsyncMock(
        return_value={"content": [{"type": "text", "text": "counted"}]}
    )

    result = await server.tool_get_entity_history(
        {
            "entity_id": "binary_sensor.front_door",
            "event": "opened",
            "mode": "count",
            "period": "yesterday",
        }
    )

    assert result["content"][0]["text"] == "counted"
    server.tool_analyze_entity_history.assert_awaited_once_with(
        {
            "entity_id": "binary_sensor.front_door",
            "event": "opened",
            "mode": "count",
            "period": "yesterday",
            "analysis": "count",
        }
    )


@pytest.mark.asyncio
async def test_handle_tools_list_includes_media_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The core MCP tool list should include the generic image/media helpers."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server.tools = _domain_package_tool_stub()

    result = await server.handle_tools_list()
    tool_names = {tool["name"] for tool in result["tools"]}
    tool_map = {tool["name"]: tool for tool in result["tools"]}

    assert "analyze_image" in tool_names
    assert "get_image" in tool_names
    assert "generate_image" in tool_names
    assert "get_calendar_events" in tool_names
    assert tool_map["analyze_image"]["llmDescription"] == (
        "Analyze an image or camera snapshot with the active multimodal model."
    )
    assert tool_map["get_calendar_events"]["llmDescription"] == (
        "Get upcoming Home Assistant calendar events, schedules, or subscribed team games."
    )


@pytest.mark.asyncio
async def test_reload_external_custom_tools_clears_cache_and_notifies_clients(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Reloading external tools should invalidate the cached surface and notify clients."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    server._cached_tools_list = [{"name": "stale"}]
    server._cached_tools_signature = ("stale",)
    server.tools = SimpleNamespace(
        reload_tool_packages=AsyncMock(
            return_value={
                "external_custom_tools_enabled": True,
                "external_packages": [],
                "built_in_packages": [],
            }
        )
    )
    server.broadcast_notification = AsyncMock()

    diagnostics = await server.reload_external_custom_tools()

    assert diagnostics["external_custom_tools_enabled"] is True
    assert server._cached_tools_list is None
    assert server._cached_tools_signature is None
    server.broadcast_notification.assert_awaited_once_with(
        "notifications/tools/list_changed"
    )


@pytest.mark.asyncio
async def test_tool_get_image_returns_image_block_from_local_file(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """get_image should return an MCP image block for local image files."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    image_path = tmp_path / "guest_wifi.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsample")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    result = await server.tool_get_image({"image_path": "guest_wifi.png"})

    assert result["isError"] is False
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["mimeType"] == "image/png"
    assert base64.b64decode(result["content"][1]["data"]) == b"\x89PNG\r\n\x1a\nsample"


@pytest.mark.asyncio
async def test_tool_get_image_accepts_absolute_path_inside_config_root(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Absolute image paths should still work when they stay inside the config root."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    image_path = tmp_path / "absolute_guest_wifi.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nabsolute")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    result = await server.tool_get_image({"image_path": str(image_path)})

    assert result["isError"] is False
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["mimeType"] == "image/png"


@pytest.mark.asyncio
async def test_tool_analyze_image_returns_answer_and_optional_image_block(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """analyze_image should return the model answer and optional image content."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    image_path = tmp_path / "driveway.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\ndriveway")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    monkeypatch.setattr(
        server,
        "_analyze_image_with_provider",
        AsyncMock(return_value="A white SUV is in the driveway."),
    )

    result = await server.tool_analyze_image(
        {"image_path": "driveway.png", "include_image": True}
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "A white SUV is in the driveway."
    assert result["content"][1]["type"] == "image"
    assert result["structuredContent"]["source"]["type"] == "image_path"


@pytest.mark.asyncio
async def test_analyze_image_uses_provider_owned_chat_url(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Image analysis should ask the provider transport for its chat URL."""
    system_entry_factory()
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: "https://proxy.example.invalid/v1",
            CONF_API_KEY: "sk-test",
            CONF_MODEL_NAME: "gpt-4o-mini",
        }
    )
    server = MCPServer(hass, 8099, entry)
    calls: list[dict[str, Any]] = []

    class _ProviderResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def json(self):
            return {"choices": [{"message": {"content": "Looks clear."}}]}

    class _ProviderSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": headers, "payload": json})
            return _ProviderResponse()

    monkeypatch.setattr(mcp_server_module.aiohttp, "ClientSession", _ProviderSession)

    result = await server._analyze_image_with_provider(
        question="What is shown?",
        image_bytes=b"fake-image",
        mime_type="image/png",
        detail="auto",
        context=None,
    )

    assert result == "Looks clear."
    assert calls[0]["url"] == "https://proxy.example.invalid/v1/chat/completions"
    assert calls[0]["headers"] == {"Authorization": "Bearer sk-test"}
    assert calls[0]["payload"]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_generate_image_uses_provider_owned_generation_url(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Image generation should ask the provider transport for its generation URL."""
    system_entry_factory()
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: "https://proxy.example.invalid",
            CONF_API_KEY: "sk-test",
            CONF_MODEL_NAME: "gpt-image-1",
        }
    )
    server = MCPServer(hass, 8099, entry)
    calls: list[dict[str, Any]] = []

    class _ProviderResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def json(self):
            return {
                "data": [
                    {
                        "b64_json": base64.b64encode(b"fake-png").decode("ascii"),
                        "revised_prompt": "A concise prompt",
                    }
                ]
            }

    class _ProviderSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": headers, "payload": json})
            return _ProviderResponse()

    monkeypatch.setattr(mcp_server_module.aiohttp, "ClientSession", _ProviderSession)

    image_bytes, mime_type, metadata = await server._generate_image_with_provider(
        prompt="Draw a clean diagram.",
        size=None,
        quality=None,
        style=None,
        background=None,
        context=None,
    )

    assert image_bytes == b"fake-png"
    assert mime_type == "image/png"
    assert calls[0]["url"] == "https://proxy.example.invalid/v1/images/generations"
    assert calls[0]["headers"] == {"Authorization": "Bearer sk-test"}
    assert calls[0]["payload"]["model"] == "gpt-image-1"
    assert metadata["model"] == "gpt-image-1"
    assert metadata["revised_prompt"] == "A concise prompt"


def test_resolve_local_image_path_rejects_path_traversal(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Local image resolution should reject traversal outside the config root."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    with pytest.raises(ValueError, match="stay inside"):
        server._resolve_local_image_path("../secrets.txt")


def test_resolve_fetchable_http_image_url_keeps_supported_sources(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Image URL validation should keep HA-local and allowlisted remote URLs working."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    monkeypatch.setattr(
        mcp_server_module.network_helper,
        "get_url",
        lambda *args, **kwargs: "http://ha.local:8123",
    )
    monkeypatch.setattr(
        mcp_server_module.network_helper,
        "is_hass_url",
        lambda hass_obj, url: _is_exact_http_origin(
            url,
            scheme="http",
            host="ha.local",
            port=8123,
        ),
    )
    monkeypatch.setattr(
        hass.config,
        "is_allowed_external_url",
        _is_allowlisted_external_image_url,
    )
    monkeypatch.setattr(
        hass.config,
        "allowlist_external_urls",
        ["https://images.example.com/weather/"],
    )

    assert (
        str(server._resolve_fetchable_http_image_url("/api/image_proxy/front_door"))
        == "http://ha.local:8123/api/image_proxy/front_door"
    )
    assert (
        str(
            server._resolve_fetchable_http_image_url(
                "http://ha.local:8123/api/image/serve/abc123/512x512"
            )
        )
        == "http://ha.local:8123/api/image/serve/abc123/512x512"
    )
    assert (
        str(
            server._resolve_fetchable_http_image_url(
                "https://images.example.com/weather/radar.png"
            )
        )
        == "https://images.example.com/weather/radar.png"
    )

    with pytest.raises(ValueError, match="allowlisted"):
        server._resolve_fetchable_http_image_url(
            "https://169.254.169.254/latest/meta-data"
        )

    with pytest.raises(ValueError, match="allowlisted"):
        server._resolve_fetchable_http_image_url(
            "https://images.example.com.evil.test/weather/radar.png"
        )

    with pytest.raises(ValueError, match="allowlisted"):
        server._resolve_fetchable_http_image_url(
            "https://images.example.com/weather-evil/radar.png"
        )

    with pytest.raises(ValueError, match="image-serving path"):
        server._resolve_fetchable_http_image_url("http://ha.local:8123/api/states")

    with pytest.raises(ValueError, match="parent path"):
        server._resolve_fetchable_http_image_url(
            "http://ha.local:8123/api/image_proxy/../config"
        )


def test_resolve_fetchable_http_image_url_sanitizes_double_slash_paths(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Double-slash paths should stay on the trusted authority, not become a network-path URL."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())

    monkeypatch.setattr(
        mcp_server_module.network_helper,
        "get_url",
        lambda *args, **kwargs: "http://ha.local:8123",
    )
    monkeypatch.setattr(
        mcp_server_module.network_helper,
        "is_hass_url",
        lambda hass_obj, url: _is_exact_http_origin(
            url,
            scheme="http",
            host="ha.local",
            port=8123,
        ),
    )
    monkeypatch.setattr(
        hass.config,
        "is_allowed_external_url",
        _is_allowlisted_external_image_url,
    )
    monkeypatch.setattr(
        hass.config,
        "allowlist_external_urls",
        ["https://images.example.com/weather/"],
    )

    assert (
        str(
            server._resolve_fetchable_http_image_url(
                "http://ha.local:8123//api/image_proxy/front_door"
            )
        )
        == "http://ha.local:8123/api/image_proxy/front_door"
    )
    assert (
        server._build_safe_http_request_path(
            yarl.URL("https://images.example.com//weather/radar.png")
        )
        == "/weather/radar.png"
    )

    with pytest.raises(ValueError, match="allowlisted"):
        server._resolve_fetchable_http_image_url(
            "https://images.example.com//evil.test/weather/radar.png"
        )


@pytest.mark.asyncio
async def test_fetch_http_image_url_uses_validated_absolute_request_url(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Image fetches should request the validated absolute URL, not a raw user path."""
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        mcp_server_module.network_helper,
        "get_url",
        lambda *args, **kwargs: "http://ha.local:8123",
    )

    class _FakeImageResponse:
        status = 200
        headers = {"Content-Type": "image/png"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def read(self) -> bytes:
            return b"image-bytes"

    class _FakeImageSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def get(self, url: str, **kwargs):
            calls.append((url, kwargs))
            return _FakeImageResponse()

    monkeypatch.setattr(mcp_server_module.aiohttp, "ClientSession", _FakeImageSession)

    image_bytes, mime_type = await server._fetch_http_image_url(
        "/api/image_proxy/front_door?token=preview"
    )

    assert image_bytes == b"image-bytes"
    assert mime_type == "image/png"
    assert calls == [
        (
            "http://ha.local:8123/api/image_proxy/front_door?token=preview",
            {"allow_redirects": False},
        )
    ]
