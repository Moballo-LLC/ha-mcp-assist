"""Native maintenance signals, exposure, paging, and optional package boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from homeassistant.helpers import entity_registry as er

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import (
    CONF_CONTEXT_MODE, CONF_CONTROL_HA, CONTEXT_MODE_ADAPTIVE, CONTEXT_MODE_LIGHT,
    CONTEXT_MODE_STANDARD, DOMAIN,
)
from custom_components.mcp_assist.custom_tool_api import MCPAssistCustomToolManifest
from custom_components.mcp_assist.tool_effects import ToolEffect, get_tool_effect
from custom_components.mcp_assist.tools import CustomToolsLoader
from custom_components.mcp_assist.tools.builtin_catalog import load_builtin_tool_toggle_specs
from custom_components.mcp_assist.tools.external_loader import ExternalCustomToolLoader
from custom_components.mcp_assist.tools.packages.maintenance import maintenance


PACKAGE_ROOT = Path("custom_components/mcp_assist/tools/packages")


@pytest.fixture
def tool(hass, monkeypatch):
    """Use real HA states with an explicit mocked conversation exposure boundary."""
    monkeypatch.setattr(maintenance, "async_should_expose", lambda _hass, assistant, entity_id: (
        assistant == "conversation" and entity_id != "sensor.unexposed_battery"
    ))
    return maintenance.MaintenanceTool(hass, MCPAssistCustomToolManifest(
        schema_version=1, tool_id="maintenance", name="Maintenance Status",
        description="Native signals", version="1.0.0", entrypoint="tool:MaintenancePackageTool",
    ), PACKAGE_ROOT / "maintenance")


async def test_native_signals_exposure_and_allowlisted_output(hass, tool):
    """No unexposed entities or arbitrary provider attributes enter either output."""
    hass.states.async_set("sensor.unexposed_battery", "1", {
        "device_class": "battery", "unit_of_measurement": "%",
    })
    hass.states.async_set("sensor.low_battery", "30", {
        "device_class": "battery", "unit_of_measurement": "%", "friendly_name": "Example battery",
        "private_url": "https://example.invalid/private", "api_key": "example-sensitive-value",
    })
    hass.states.async_set("binary_sensor.low_battery", "on", {"device_class": "battery"})
    hass.states.async_set("binary_sensor.problem", "on", {"device_class": "problem"})
    hass.states.async_set("update.software", "on", {"release_url": "https://example.invalid/release"})
    hass.states.async_set("sensor.offline", "unavailable")
    hass.states.async_set("sensor.uncertain", "unknown")
    hass.states.async_set("sensor.normal", "75", {"device_class": "battery", "unit_of_measurement": "%"})
    hass.states.async_set("binary_sensor.normal", "off", {"device_class": "problem"})
    hass.states.async_set("update.current", "off")
    hass.states.async_set("switch.battery_name", "on", {"device_class": "battery"})
    result = await tool.handle_call("get_maintenance_status", {})
    data = result["structuredContent"]
    assert data["scope"] == "exposed_entities_only"
    assert data["exposed_count"] == 10
    assert data["matched_count"] == data["page_count"] == 6
    assert data["next_offset"] is None
    assert data["truncated"] is False
    assert [(item["category"], item["entity_id"]) for item in data["items"]] == [
        ("problems", "binary_sensor.problem"), ("unavailable", "sensor.offline"),
        ("unavailable", "sensor.uncertain"), ("batteries", "binary_sensor.low_battery"),
        ("batteries", "sensor.low_battery"), ("updates", "update.software"),
    ]
    assert data["items"][4] == {
        "category": "batteries", "entity_id": "sensor.low_battery",
        "friendly_name": "Example battery", "state": "30", "battery_percentage": 30.0,
    }
    assert "battery_percentage" not in data["items"][3]
    for item in data["items"]:
        assert set(item) <= {"category", "entity_id", "friendly_name", "state", "battery_percentage"}
    assert "example.invalid" not in str(result)
    assert "example-sensitive-value" not in str(result)
    assert "sensor.unexposed_battery" not in str(result)
    assert "does not establish whole-home health" in result["content"][0]["text"]


@pytest.mark.parametrize("state,attributes", [
    ("NaN", {"device_class": "battery", "unit_of_measurement": "%"}),
    ("inf", {"device_class": "battery", "unit_of_measurement": "%"}),
    ("-inf", {"device_class": "battery", "unit_of_measurement": "%"}),
    ("-1", {"device_class": "battery", "unit_of_measurement": "%"}),
    ("101", {"device_class": "battery", "unit_of_measurement": "%"}),
    ("not numeric", {"device_class": "battery", "unit_of_measurement": "%"}),
    ("5", {"device_class": "battery", "unit_of_measurement": "V"}),
    ("5", {"unit_of_measurement": "%"}),
])
async def test_malformed_or_nonpercentage_batteries_do_not_match(hass, tool, state, attributes):
    hass.states.async_set("sensor.battery", state, attributes)
    data = (await tool.handle_call("get_maintenance_status", {
        "category": "batteries", "battery_threshold": 100,
    }))["structuredContent"]
    assert data["items"] == []
    assert data["exposed_count"] == 1


async def test_threshold_and_category_filtering(hass, tool):
    hass.states.async_set("sensor.zero", "0", {"device_class": "battery", "unit_of_measurement": "%"})
    hass.states.async_set("sensor.low", "30", {"device_class": "battery", "unit_of_measurement": "%"})
    hass.states.async_set("sensor.full", "100", {"device_class": "battery", "unit_of_measurement": "%"})
    hass.states.async_set("binary_sensor.battery", "unavailable", {"device_class": "battery"})
    hass.states.async_set("update.software", "unknown")
    for threshold, expected in ((1, ["sensor.zero"]), (30, ["sensor.low", "sensor.zero"]),
                                (100, ["sensor.full", "sensor.low", "sensor.zero"])):
        data = (await tool.handle_call("get_maintenance_status", {
            "category": "batteries", "battery_threshold": threshold,
        }))["structuredContent"]
        assert [item["entity_id"] for item in data["items"]] == expected
    data = (await tool.handle_call("get_maintenance_status", {"category": "unavailable"}))["structuredContent"]
    assert [item["state"] for item in data["items"]] == ["unavailable", "unknown"]
    assert (await tool.handle_call("get_maintenance_status", {"category": "updates"}))["structuredContent"]["matched_count"] == 0


async def test_hidden_ui_entity_remains_visible_when_exposed(hass, tool):
    entry = er.async_get(hass).async_get_or_create(
        "binary_sensor", "test", "example_problem", suggested_object_id="example_problem",
        hidden_by=er.RegistryEntryHider.USER,
    )
    hass.states.async_set(entry.entity_id, "on", {"device_class": "problem"})
    data = (await tool.handle_call("get_maintenance_status", {}))["structuredContent"]
    assert [item["entity_id"] for item in data["items"]] == [entry.entity_id]


async def test_paging_is_deterministic_and_retains_full_match_count(hass, tool):
    for entity_id in ("update.z", "update.a", "update.m"):
        hass.states.async_set(entity_id, "on")
    for offset, expected, next_offset in (
        (0, ["update.a", "update.m"], 2), (2, ["update.z"], None), (10, [], None),
    ):
        data = (await tool.handle_call("get_maintenance_status", {
            "limit": 2, "offset": offset,
        }))["structuredContent"]
        assert [item["entity_id"] for item in data["items"]] == expected
        assert data["page_count"] == len(expected)
        assert data["matched_count"] == 3
        assert data["next_offset"] == next_offset
        assert data["truncated"] is True


@pytest.mark.parametrize("arguments", [
    {"category": "missing"}, {"category": []}, {"battery_threshold": True},
    {"battery_threshold": 0}, {"battery_threshold": 101}, {"limit": 0}, {"limit": 101},
    {"limit": "25"}, {"offset": -1}, {"offset": 1.5}, {"unexpected": True},
])
async def test_invalid_arguments_do_not_scan(hass, tool, monkeypatch, arguments):
    scan = Mock(side_effect=AssertionError("unexpected state scan"))
    monkeypatch.setattr(type(hass.states), "async_all", scan)
    assert (await tool.handle_call("get_maintenance_status", arguments))["isError"] is True
    scan.assert_not_called()


async def test_package_startup_does_not_scan_entities(hass, monkeypatch):
    scan = Mock(side_effect=AssertionError("unexpected startup scan"))
    monkeypatch.setattr(type(hass.states), "async_all", scan)
    loader = ExternalCustomToolLoader(
        hass, tools_root=PACKAGE_ROOT, module_namespace="mcp_assist_maintenance_startup_tests",
        require_tool_name_prefix=False,
    )
    packages = await loader.load(allowed_tool_ids={"maintenance"})
    assert loader.last_load_errors == []
    assert [package.manifest.tool_id for package in packages] == ["maintenance"]
    scan.assert_not_called()
    await packages[0].instance.async_shutdown()


@pytest.mark.parametrize("shared,profile,expected", [
    (None, None, False), (False, True, False), (True, None, True),
    (True, True, True), (True, False, False),
])
def test_optional_package_gating_and_read_only_profile(
    hass, tool, system_entry_factory, profile_entry_factory, shared, profile, expected,
):
    system_entry_factory(options={} if shared is None else {"enable_maintenance_tools": shared})
    options = {CONF_CONTROL_HA: False}
    if profile is not None:
        options["profile_enable_maintenance_tools"] = profile
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(options=options))
    spec = next(spec for spec in load_builtin_tool_toggle_specs() if spec.package_id == "maintenance")
    definition = tool.get_tool_definitions()[0]
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(tools=SimpleNamespace(
        get_builtin_toggle_spec=lambda name: spec if name == "get_maintenance_status" else None,
        get_tool_definition=lambda name: definition,
    ))
    assert get_tool_effect("get_maintenance_status") is ToolEffect.READ_ONLY
    assert agent._is_tool_enabled_for_profile("get_maintenance_status") is expected
    assert bool(agent._filter_mcp_tools_for_profile([definition])) is expected


async def test_disabled_default_package_not_loaded_or_advertised(hass, system_entry_factory):
    loader = CustomToolsLoader(hass, system_entry_factory())
    await loader.initialize()
    try:
        assert "maintenance" not in {package.manifest.tool_id for package in loader.builtin_packages}
        assert "get_maintenance_status" not in {tool["name"] for tool in loader.get_tool_definitions()}
        assert "get_maintenance_status" not in loader.get_builtin_prompt_instructions()
    finally:
        await loader.shutdown()


@pytest.mark.parametrize("context_mode", [
    CONTEXT_MODE_STANDARD, CONTEXT_MODE_ADAPTIVE, CONTEXT_MODE_LIGHT,
])
async def test_profile_prompt_omits_disabled_package_guidance(
    hass, system_entry_factory, profile_entry_factory, context_mode,
):
    shared = system_entry_factory(options={
        "enable_maintenance_tools": True, "enable_calculator_tools": True,
    })
    loader = CustomToolsLoader(hass, shared)
    await loader.initialize()
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(tools=loader)
    try:
        for enabled in (False, True):
            agent = MCPAssistConversationEntity(hass, profile_entry_factory(options={
                CONF_CONTEXT_MODE: context_mode, "profile_enable_maintenance_tools": enabled,
            }))
            instructions = agent._get_builtin_tool_instructions()
            assert ("get_maintenance_status" in instructions) is enabled
            assert "Calculator" in instructions
            optional = agent._build_optional_technical_instructions("")
            assert ("get_maintenance_status" in optional) is enabled
            definitions = agent._filter_mcp_tools_for_profile(loader.get_tool_definitions())
            advertised = enabled and context_mode != CONTEXT_MODE_LIGHT
            assert ("get_maintenance_status" in {tool["name"] for tool in definitions}) is advertised
        assert "get_maintenance_status" in loader.get_builtin_prompt_instructions()
    finally:
        await loader.shutdown()
