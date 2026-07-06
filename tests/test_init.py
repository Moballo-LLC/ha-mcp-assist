"""Tests for integration setup helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mcp_assist import (
    _migrate_brave_search_tool_name,
    _async_apply_shared_mcp_settings,
    async_setup_entry,
    async_unload_entry,
    ensure_system_entry,
)
from custom_components.mcp_assist.const import (
    CONF_ALLOWED_IPS,
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_LLM_API_ALLOWLIST,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MCP_PORT,
    CONF_PROFILE_NAME,
    CONF_SEARCH_PROVIDER,
    CONF_SEARXNG_URL,
    CONF_TECHNICAL_PROMPT,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_ENABLE_LLM_API_BRIDGE,
    DEFAULT_LLM_API_ALLOWLIST,
    DEFAULT_MCP_PORT,
    DOMAIN,
    SERVICE_CLEAR_CHAT_LOGS,
    SERVICE_GET_CHAT_LOGS,
    SERVICE_RELOAD_EXTERNAL_CUSTOM_TOOLS,
    SERVICE_VALIDATE_EXTERNAL_CUSTOM_TOOLS,
    SYSTEM_ENTRY_UNIQUE_ID,
)


def _mock_system_entry_init(hass, data: dict) -> MockConfigEntry:
    """Simulate creation of the shared system entry without full HA dependency setup."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Shared MCP Server Settings",
        unique_id=SYSTEM_ENTRY_UNIQUE_ID,
        source="system",
        data=data,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_migrate_brave_search_tool_name_updates_legacy_prompt(
    hass, profile_entry_factory
) -> None:
    """Legacy brave_search references should be migrated in-place."""
    entry = profile_entry_factory(
        options={CONF_TECHNICAL_PROMPT: "Use brave_search for current events."}
    )

    await _migrate_brave_search_tool_name(hass, entry)

    assert entry.options[CONF_TECHNICAL_PROMPT] == "Use search for current events."


@pytest.mark.asyncio
async def test_ensure_system_entry_copies_shared_settings_from_first_profile(
    hass, profile_entry_factory
) -> None:
    """System entry creation should copy the shared MCP settings from the first profile."""
    profile_entry_factory(
        data={
            CONF_MCP_PORT: 1883,
            CONF_SEARCH_PROVIDER: "none",
            CONF_BRAVE_API_KEY: "",
            CONF_ALLOWED_IPS: "",
            CONF_ENABLE_GAP_FILLING: True,
        },
        options={
            CONF_MCP_PORT: 8124,
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "duckduckgo",
            CONF_BRAVE_API_KEY: "abc123",
            CONF_SEARXNG_URL: "http://search.local",
            CONF_ALLOWED_IPS: "10.0.0.0/24",
            CONF_INCLUDE_CURRENT_USER: False,
            CONF_INCLUDE_HOME_LOCATION: False,
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: True,
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: True,
            CONF_ENABLE_GAP_FILLING: False,
            CONF_ENABLE_ASSIST_BRIDGE: False,
            CONF_ENABLE_LLM_API_BRIDGE: True,
            CONF_LLM_API_ALLOWLIST: "llm_intents",
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_MEMORY_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
            CONF_ENABLE_DEVICE_TOOLS: False,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: True,
            CONF_MEMORY_DEFAULT_TTL_DAYS: 14,
            CONF_MEMORY_MAX_TTL_DAYS: 90,
            CONF_MEMORY_MAX_ITEMS: 250,
        },
    )

    async_init = AsyncMock(
        side_effect=lambda domain, context, data: _mock_system_entry_init(hass, data)
    )
    with patch.object(hass.config_entries.flow, "async_init", async_init):
        system_entry = await ensure_system_entry(hass)

    assert system_entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID
    assert system_entry.data[CONF_MCP_PORT] == 8124
    assert system_entry.data[CONF_ENABLE_WEB_SEARCH] is True
    assert system_entry.data[CONF_SEARCH_PROVIDER] == "duckduckgo"
    assert system_entry.data[CONF_BRAVE_API_KEY] == "abc123"
    assert system_entry.data[CONF_SEARXNG_URL] == "http://search.local"
    assert system_entry.data[CONF_ALLOWED_IPS] == "10.0.0.0/24"
    assert system_entry.data[CONF_INCLUDE_CURRENT_USER] is False
    assert system_entry.data[CONF_INCLUDE_HOME_LOCATION] is False
    assert system_entry.data[CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS] is True
    assert system_entry.data[CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS] is True
    assert system_entry.data[CONF_ENABLE_GAP_FILLING] is False
    assert system_entry.data[CONF_ENABLE_LLM_API_BRIDGE] is True
    assert system_entry.data[CONF_LLM_API_ALLOWLIST] == "llm_intents"
    assert system_entry.data[CONF_ENABLE_DEVICE_TOOLS] is False
    assert system_entry.data[CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT] is True
    assert system_entry.data[CONF_ENABLE_WEATHER_FORECAST_TOOL] is False
    assert system_entry.data[CONF_ENABLE_MEMORY_TOOLS] is True
    assert system_entry.data[CONF_ENABLE_UNIT_CONVERSION_TOOLS] is True
    assert system_entry.data[CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS] is True
    assert system_entry.data[CONF_MEMORY_DEFAULT_TTL_DAYS] == 14
    assert system_entry.data[CONF_MEMORY_MAX_TTL_DAYS] == 90
    assert system_entry.data[CONF_MEMORY_MAX_ITEMS] == 250


@pytest.mark.asyncio
async def test_ensure_system_entry_uses_defaults_without_profiles(hass) -> None:
    """System entry creation should fall back to defaults when no profiles exist yet."""
    async_init = AsyncMock(
        side_effect=lambda domain, context, data: _mock_system_entry_init(hass, data)
    )
    with patch.object(hass.config_entries.flow, "async_init", async_init):
        system_entry = await ensure_system_entry(hass)

    assert system_entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID
    assert system_entry.data[CONF_MCP_PORT] == DEFAULT_MCP_PORT
    assert CONF_ENABLE_WEB_SEARCH in system_entry.data
    assert CONF_SEARXNG_URL in system_entry.data
    assert CONF_INCLUDE_CURRENT_USER in system_entry.data
    assert CONF_INCLUDE_HOME_LOCATION in system_entry.data
    assert CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS in system_entry.data
    assert CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS in system_entry.data
    assert system_entry.data[CONF_ENABLE_LLM_API_BRIDGE] == DEFAULT_ENABLE_LLM_API_BRIDGE
    assert system_entry.data[CONF_LLM_API_ALLOWLIST] == DEFAULT_LLM_API_ALLOWLIST
    assert system_entry.data[CONF_ENABLE_DEVICE_TOOLS] == DEFAULT_ENABLE_DEVICE_TOOLS
    assert CONF_ENABLE_WEATHER_FORECAST_TOOL in system_entry.data
    assert CONF_ENABLE_MEMORY_TOOLS in system_entry.data
    assert CONF_ENABLE_UNIT_CONVERSION_TOOLS in system_entry.data
    assert CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS in system_entry.data
    assert CONF_MEMORY_DEFAULT_TTL_DAYS in system_entry.data
    assert CONF_MEMORY_MAX_TTL_DAYS in system_entry.data
    assert CONF_MEMORY_MAX_ITEMS in system_entry.data


@pytest.mark.asyncio
async def test_async_setup_and_unload_reuse_shared_runtime_objects(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Profiles should share the MCP server and index manager until the last unload."""
    system_entry_factory()
    entry_one = profile_entry_factory(title="Ollama - One", unique_id=f"{DOMAIN}_one", data={CONF_PROFILE_NAME: "One"})
    entry_two = profile_entry_factory(title="Ollama - Two", unique_id=f"{DOMAIN}_two", data={CONF_PROFILE_NAME: "Two"})

    index_manager = SimpleNamespace(start=AsyncMock(), async_stop=AsyncMock())
    mcp_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())

    with (
        patch("custom_components.mcp_assist.IndexManager", return_value=index_manager) as index_cls,
        patch("custom_components.mcp_assist.MCPServer", return_value=mcp_server) as server_cls,
        patch.object(hass.config_entries, "async_forward_entry_setups", AsyncMock(return_value=True)),
        patch.object(hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)),
    ):
        assert await async_setup_entry(hass, entry_one) is True
        assert await async_setup_entry(hass, entry_two) is True
        assert hass.data[DOMAIN]["mcp_refcount"] == 2
        assert index_cls.call_count == 1
        assert server_cls.call_count == 1

        assert await async_unload_entry(hass, entry_one) is True
        assert hass.data[DOMAIN]["mcp_refcount"] == 1
        mcp_server.stop.assert_not_called()
        index_manager.async_stop.assert_not_called()

        assert await async_unload_entry(hass, entry_two) is True
        mcp_server.stop.assert_awaited_once()
        index_manager.async_stop.assert_awaited_once()
        assert "shared_mcp_server" not in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_apply_shared_mcp_settings_restarts_server_for_port_change(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared port changes should restart only the shared MCP server."""
    system_entry_factory(data={CONF_MCP_PORT: 8124})
    profile_entry = profile_entry_factory()
    old_server = SimpleNamespace(port=8090, stop=AsyncMock())
    new_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = old_server
    hass.data[DOMAIN]["mcp_port"] = 8090
    order = Mock()
    order.attach_mock(new_server.start, "new_start")
    order.attach_mock(old_server.stop, "old_stop")

    with patch("custom_components.mcp_assist.MCPServer", return_value=new_server) as server_cls:
        await _async_apply_shared_mcp_settings(hass)

    old_server.stop.assert_awaited_once()
    new_server.start.assert_awaited_once()
    new_server.stop.assert_not_awaited()
    assert order.mock_calls == [call.new_start(), call.old_stop()]
    server_cls.assert_called_once_with(hass, 8124, profile_entry)
    assert hass.data[DOMAIN]["shared_mcp_server"] is new_server
    assert hass.data[DOMAIN]["mcp_port"] == 8124


@pytest.mark.asyncio
async def test_apply_shared_mcp_settings_restarts_with_active_profile_entry(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared port restarts should preserve the profile used by the running server."""
    system_entry_factory(data={CONF_MCP_PORT: 8124})
    profile_entry_factory(
        title="Ollama - First",
        unique_id=f"{DOMAIN}_first",
        data={CONF_PROFILE_NAME: "First"},
    )
    active_profile = profile_entry_factory(
        title="Ollama - Active",
        unique_id=f"{DOMAIN}_active",
        data={CONF_PROFILE_NAME: "Active"},
    )
    old_server = SimpleNamespace(port=8090, entry=active_profile, stop=AsyncMock())
    new_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = old_server
    hass.data[DOMAIN]["mcp_port"] = 8090

    with patch("custom_components.mcp_assist.MCPServer", return_value=new_server) as server_cls:
        await _async_apply_shared_mcp_settings(hass)

    server_cls.assert_called_once_with(hass, 8124, active_profile)
    assert hass.data[DOMAIN]["shared_mcp_server"] is new_server


@pytest.mark.asyncio
async def test_apply_shared_mcp_settings_keeps_old_server_when_new_port_fails(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A failed port restart should leave the working shared server in place."""
    system_entry_factory(data={CONF_MCP_PORT: 8124})
    profile_entry = profile_entry_factory()
    old_server = SimpleNamespace(port=8090, stop=AsyncMock())
    new_server = SimpleNamespace(start=AsyncMock(side_effect=OSError("address in use")), stop=AsyncMock())
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = old_server
    hass.data[DOMAIN]["mcp_port"] = 8090

    with (
        patch("custom_components.mcp_assist.MCPServer", return_value=new_server) as server_cls,
        pytest.raises(OSError, match="address in use"),
    ):
        await _async_apply_shared_mcp_settings(hass)

    new_server.start.assert_awaited_once()
    new_server.stop.assert_not_awaited()
    old_server.stop.assert_not_awaited()
    server_cls.assert_called_once_with(hass, 8124, profile_entry)
    assert hass.data[DOMAIN]["shared_mcp_server"] is old_server
    assert hass.data[DOMAIN]["mcp_port"] == 8090


@pytest.mark.asyncio
async def test_async_setup_registers_reload_service_and_last_unload_removes_it(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The shared external-tool reload service should track the shared server lifecycle."""
    system_entry_factory()
    entry = profile_entry_factory()
    index_manager = SimpleNamespace(start=AsyncMock())
    mcp_server = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(),
        reload_external_custom_tools=AsyncMock(
            return_value={"enabled": True, "loaded_tools": [], "load_errors": []}
        ),
    )

    with (
        patch("custom_components.mcp_assist.IndexManager", return_value=index_manager),
        patch("custom_components.mcp_assist.MCPServer", return_value=mcp_server),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(return_value=True),
        ),
        patch.object(
            hass.config_entries,
            "async_unload_platforms",
            AsyncMock(return_value=True),
        ),
    ):
        assert await async_setup_entry(hass, entry) is True
        assert hass.services.has_service(DOMAIN, SERVICE_RELOAD_EXTERNAL_CUSTOM_TOOLS)
        assert hass.services.has_service(DOMAIN, SERVICE_VALIDATE_EXTERNAL_CUSTOM_TOOLS)
        assert hass.services.has_service(DOMAIN, SERVICE_GET_CHAT_LOGS)
        assert hass.services.has_service(DOMAIN, SERVICE_CLEAR_CHAT_LOGS)

        response = await hass.services.async_call(
            DOMAIN,
            SERVICE_RELOAD_EXTERNAL_CUSTOM_TOOLS,
            blocking=True,
            return_response=True,
        )

        assert response == {"enabled": True, "loaded_tools": [], "load_errors": []}
        mcp_server.reload_external_custom_tools.assert_awaited_once()

        mcp_server.validate_external_custom_tools = AsyncMock(
            return_value={
                "enabled": True,
                "valid": True,
                "loaded_tools": [],
                "load_errors": [],
            }
        )
        validate_response = await hass.services.async_call(
            DOMAIN,
            SERVICE_VALIDATE_EXTERNAL_CUSTOM_TOOLS,
            blocking=True,
            return_response=True,
        )

        assert validate_response == {
            "enabled": True,
            "valid": True,
            "loaded_tools": [],
            "load_errors": [],
        }
        mcp_server.validate_external_custom_tools.assert_awaited_once()

        chat_log_manager = hass.data[DOMAIN]["chat_log_manager"]
        await chat_log_manager.async_record(
            {
                "created_at": "2026-06-01T00:00:00+00:00",
                "profile_entry_id": entry.entry_id,
                "conversation_id": "conv-1",
                "user_text": "status",
                "assistant_text": "All good.",
                "tools": [],
            }
        )
        logs_response = await hass.services.async_call(
            DOMAIN,
            SERVICE_GET_CHAT_LOGS,
            {"profile_entry_id": entry.entry_id},
            blocking=True,
            return_response=True,
        )

        assert logs_response["count"] == 1
        assert logs_response["logs"][0]["conversation_id"] == "conv-1"

        clear_response = await hass.services.async_call(
            DOMAIN,
            SERVICE_CLEAR_CHAT_LOGS,
            {"conversation_id": "conv-1"},
            blocking=True,
            return_response=True,
        )

        assert clear_response == {"deleted_count": 1, "remaining_count": 0}

        assert await async_unload_entry(hass, entry) is True
        assert not hass.services.has_service(
            DOMAIN,
            SERVICE_RELOAD_EXTERNAL_CUSTOM_TOOLS,
        )
        assert not hass.services.has_service(
            DOMAIN,
            SERVICE_VALIDATE_EXTERNAL_CUSTOM_TOOLS,
        )
        assert not hass.services.has_service(DOMAIN, SERVICE_GET_CHAT_LOGS)
        assert not hass.services.has_service(DOMAIN, SERVICE_CLEAR_CHAT_LOGS)


@pytest.mark.asyncio
async def test_failed_setup_releases_shared_server(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A setup that fails after refcount increment must release the shared server."""
    system_entry_factory()
    entry = profile_entry_factory()
    index_manager = SimpleNamespace(start=AsyncMock(), async_stop=AsyncMock())
    mcp_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())

    with (
        patch("custom_components.mcp_assist.IndexManager", return_value=index_manager),
        patch("custom_components.mcp_assist.MCPServer", return_value=mcp_server),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(side_effect=RuntimeError("platform boom")),
        ),
    ):
        with pytest.raises(Exception):
            await async_setup_entry(hass, entry)

    # The single reference taken during setup must have been released, so the
    # shared server is stopped and the port is freed for the next attempt.
    mcp_server.stop.assert_awaited_once()
    index_manager.async_stop.assert_awaited_once()
    assert "shared_mcp_server" not in hass.data[DOMAIN]
    assert "mcp_refcount" not in hass.data[DOMAIN]
    assert entry.entry_id not in hass.data[DOMAIN]


@pytest.mark.asyncio
async def test_failed_second_setup_keeps_server_for_first_profile(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """One profile's failed setup must not tear down the server used by another."""
    system_entry_factory()
    entry_one = profile_entry_factory(
        title="Ollama - One", unique_id=f"{DOMAIN}_one", data={CONF_PROFILE_NAME: "One"}
    )
    entry_two = profile_entry_factory(
        title="Ollama - Two", unique_id=f"{DOMAIN}_two", data={CONF_PROFILE_NAME: "Two"}
    )
    index_manager = SimpleNamespace(start=AsyncMock(), async_stop=AsyncMock())
    mcp_server = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())

    forward = AsyncMock(side_effect=[True, RuntimeError("platform boom")])
    with (
        patch("custom_components.mcp_assist.IndexManager", return_value=index_manager),
        patch("custom_components.mcp_assist.MCPServer", return_value=mcp_server),
        patch.object(hass.config_entries, "async_forward_entry_setups", forward),
    ):
        assert await async_setup_entry(hass, entry_one) is True
        with pytest.raises(Exception):
            await async_setup_entry(hass, entry_two)

    # Refcount is back to 1 (only entry_one) and the server stays up.
    assert hass.data[DOMAIN]["mcp_refcount"] == 1
    mcp_server.stop.assert_not_called()
    assert hass.data[DOMAIN]["shared_mcp_server"] is mcp_server
    assert entry_two.entry_id not in hass.data[DOMAIN]
