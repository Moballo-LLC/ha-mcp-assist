"""Conditional forms use native Core validation and preserve older form behavior."""
from __future__ import annotations

import pytest
from homeassistant import data_entry_flow

from custom_components.mcp_assist import config_flow as flow
from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY, CONF_ENABLE_INDEX_AUTO_REFRESH, CONF_ENABLE_LLM_API_BRIDGE,
    CONF_INDEX_REFRESH_DELAY_SECONDS, CONF_LLM_API_ALLOWLIST, CONF_SEARCH_PROVIDER,
    CONF_SEARXNG_URL,
)
from custom_components.mcp_assist.tools.builtin_catalog import load_builtin_tool_toggle_specs


def fields(schema):
    return {key.schema: key for key in schema.schema}


def test_search_and_bridge_fields_follow_their_enabling_choices():
    specs = load_builtin_tool_toggle_specs()
    search = next(spec for spec in specs if spec.requires_search_provider)
    schema = flow._build_shared_tools_section({}, specs).schema
    markers = fields(schema)
    if not hasattr(data_entry_flow, "Optional"):
        assert all(getattr(marker, "visible", None) is None for marker in markers.values())
        return
    assert markers[CONF_SEARCH_PROVIDER].visible == {"field": search.shared_setting_key, "value": True}
    assert markers[CONF_BRAVE_API_KEY].visible == [
        {"field": search.shared_setting_key, "value": True},
        {"field": CONF_SEARCH_PROVIDER, "value": "brave"},
    ]
    assert markers[CONF_SEARXNG_URL].visible[-1] == {"field": CONF_SEARCH_PROVIDER, "value": "searxng"}
    assert markers[CONF_LLM_API_ALLOWLIST].visible == {"field": CONF_ENABLE_LLM_API_BRIDGE, "value": True}
    stripped, values = data_entry_flow._strip_hidden_fields(schema, {
        search.shared_setting_key: False, CONF_ENABLE_LLM_API_BRIDGE: False,
        CONF_SEARCH_PROVIDER: "searxng", CONF_SEARXNG_URL: "stale",
        CONF_BRAVE_API_KEY: "example-key", CONF_LLM_API_ALLOWLIST: "homeassistant",
    })
    validated = stripped(values)
    assert CONF_SEARCH_PROVIDER not in validated
    assert CONF_SEARXNG_URL not in validated
    assert CONF_BRAVE_API_KEY not in validated
    assert CONF_LLM_API_ALLOWLIST not in validated


def test_older_core_keeps_existing_fields_and_defaults(monkeypatch):
    monkeypatch.setattr(flow, "ConditionalOptional", None, raising=False)
    monkeypatch.setattr(flow, "ConditionalRequired", None, raising=False)
    schema = flow._build_shared_tools_section({CONF_BRAVE_API_KEY: "saved-example-key"},
                                             load_builtin_tool_toggle_specs()).schema
    markers = fields(schema)
    assert getattr(markers[CONF_BRAVE_API_KEY], "visible", None) is None
    assert markers[CONF_BRAVE_API_KEY].default() == "saved-example-key"
    assert CONF_SEARXNG_URL in markers
    assert CONF_LLM_API_ALLOWLIST in markers


def test_hiding_tools_preserves_stored_provider_choice_keys_and_allowlist():
    specs = load_builtin_tool_toggle_specs()
    search = next(spec for spec in specs if spec.requires_search_provider)
    previous = {CONF_SEARCH_PROVIDER: "brave", CONF_BRAVE_API_KEY: "saved-example-key",
                CONF_LLM_API_ALLOWLIST: "homeassistant"}
    values = flow._normalize_shared_tool_inputs(
        {search.shared_setting_key: False, CONF_ENABLE_LLM_API_BRIDGE: False}, specs,
        previous_values=previous,
    )
    assert values[CONF_SEARCH_PROVIDER] == "brave"
    assert values[CONF_BRAVE_API_KEY] == "saved-example-key"
    assert values[CONF_LLM_API_ALLOWLIST] == "homeassistant"


def test_auto_refresh_timing_is_conditional_and_falls_back_on_older_core():
    from custom_components.mcp_assist.const import CONF_ENABLE_GAP_FILLING, CONF_MAX_ENTITIES_PER_DISCOVERY
    schema = flow._build_shared_discovery_section({
        CONF_ENABLE_GAP_FILLING: True, CONF_ENABLE_INDEX_AUTO_REFRESH: True,
        CONF_INDEX_REFRESH_DELAY_SECONDS: 5, CONF_MAX_ENTITIES_PER_DISCOVERY: 100,
    }).schema
    marker = fields(schema)[CONF_INDEX_REFRESH_DELAY_SECONDS]
    expected = {"field": CONF_ENABLE_INDEX_AUTO_REFRESH, "value": True}
    assert getattr(marker, "visible", None) == (expected if hasattr(data_entry_flow, "Optional") else None)


@pytest.mark.parametrize("mode", ["fixed", "per_conversation"])
def test_openclaw_session_key_visibility_and_saved_value(mode):
    from custom_components.mcp_assist.const import CONF_OPENCLAW_SESSION_KEY, CONF_OPENCLAW_SESSION_MODE
    from custom_components.mcp_assist.llm_providers.openclaw import OpenClawProvider
    provider_fields = OpenClawProvider.config_provider_options_fields()
    items = flow._build_provider_field_schema_items(provider_fields)
    schema = flow._build_provider_section(items).schema
    marker = fields(schema)[CONF_OPENCLAW_SESSION_KEY]
    expected = [{"field": CONF_OPENCLAW_SESSION_MODE, "value": "fixed"}]
    assert getattr(marker, "visible", None) == (expected if hasattr(data_entry_flow, "Optional") else None)
    values = flow._preserve_inactive_profile_values(
        {CONF_OPENCLAW_SESSION_MODE: mode}, {CONF_OPENCLAW_SESSION_KEY: "saved-session"}, provider_fields,
    )
    assert values.get(CONF_OPENCLAW_SESSION_KEY) == ("saved-session" if mode == "per_conversation" else None)


async def test_options_save_preserves_hidden_shared_settings(
    hass, profile_entry_factory, system_entry_factory, monkeypatch,
):
    from unittest.mock import AsyncMock
    from custom_components.mcp_assist.const import CONF_MCP_PORT
    specs = load_builtin_tool_toggle_specs()
    search = next(spec for spec in specs if spec.requires_search_provider)
    system = system_entry_factory(data={
        CONF_SEARCH_PROVIDER: "brave", CONF_BRAVE_API_KEY: "saved-example-key",
        CONF_LLM_API_ALLOWLIST: "homeassistant", CONF_ENABLE_LLM_API_BRIDGE: True,
    })
    profile = profile_entry_factory()
    options = flow.MCPAssistOptionsFlow()
    options.hass = hass
    options.handler = profile.entry_id
    options.profile_options = {}
    monkeypatch.setattr("custom_components.mcp_assist._async_apply_shared_mcp_settings", AsyncMock())
    result = await options.async_step_mcp_server({
        CONF_MCP_PORT: 8090, search.shared_setting_key: False, CONF_ENABLE_LLM_API_BRIDGE: False,
    })
    assert result["type"] == "create_entry"
    assert system.data[CONF_BRAVE_API_KEY] == "saved-example-key"
    assert system.data[CONF_LLM_API_ALLOWLIST] == "homeassistant"
    assert system.data[CONF_SEARCH_PROVIDER] == "brave"



def test_native_serialized_form_contains_visibility_conditions():
    if not hasattr(data_entry_flow, "add_visible_conditions_to_serialized_schema"):
        return
    try:
        from probatio import to_field_list
    except ImportError:
        from voluptuous_serialize import convert as to_field_list
    schema = flow._build_shared_tools_section({}, load_builtin_tool_toggle_specs()).schema
    from homeassistant.helpers import config_validation as cv
    serialized = to_field_list(schema, custom_serializer=cv.custom_serializer)
    data_entry_flow.add_visible_conditions_to_serialized_schema(schema, serialized)
    rendered = {field["name"]: field for field in serialized}
    assert rendered[CONF_LLM_API_ALLOWLIST]["visible"] == {
        "field": CONF_ENABLE_LLM_API_BRIDGE, "value": True,
    }
