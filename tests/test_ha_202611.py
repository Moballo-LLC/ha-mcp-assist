"""Coverage for capabilities introduced by Home Assistant 2026.11."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from homeassistant.helpers import area_registry as ar, entity_registry as er

from custom_components.mcp_assist.discovery import SmartDiscovery
from custom_components.mcp_assist.mcp_server import MCPServer


async def test_target_excludes_an_area_without_expanding_exposure(
    hass, profile_entry_factory, system_entry_factory, monkeypatch,
):
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    area = ar.async_get(hass).async_create("Kitchen")
    registry = er.async_get(hass)
    for name in ("kitchen", "hall", "private"):
        entity = registry.async_get_or_create("light", "test", name, suggested_object_id=name)
        if name == "kitchen":
            registry.async_update_entity(entity.entity_id, area_id=area.id)
        hass.states.async_set(entity.entity_id, "on")
    monkeypatch.setattr("custom_components.mcp_assist.mcp_server.async_should_expose",
                        lambda hass, assistant, entity_id: entity_id != "light.private")
    monkeypatch.setattr("custom_components.mcp_assist.targeting.async_should_expose",
                        lambda hass, assistant, entity_id: entity_id != "light.private")
    target = {"entity_id": ["light.kitchen", "light.hall"], "exclude": {"area_id": area.id}}
    from custom_components.mcp_assist.targeting import async_extract_excluded_entity_ids
    if async_extract_excluded_entity_ids is None:
        with pytest.raises(ValueError, match="does not support native target exclusions"):
            await server.resolve_target(target)
        return
    assert await server.resolve_target(target) == {"entity_id": ["light.hall"]}


@pytest.mark.parametrize("exclude", [{"entity_id": "light.desk"}, {"exclude": {}}, "desk"])
async def test_empty_or_malformed_exclusions_never_call_a_service(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, exclude,
):
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set("light.desk", "on")
    monkeypatch.setattr("custom_components.mcp_assist.mcp_server.async_should_expose", lambda *a: True)
    with pytest.raises(ValueError):
        await server.resolve_target({"entity_id": "light.desk", "exclude": exclude})


async def test_action_data_cannot_override_an_excluded_target(
    hass, profile_entry_factory, system_entry_factory, monkeypatch,
):
    system_entry_factory()
    server = MCPServer(hass, 8099, profile_entry_factory())
    hass.states.async_set("light.hall", "on")
    hass.states.async_set("light.kitchen", "on")
    monkeypatch.setattr("custom_components.mcp_assist.mcp_server.async_should_expose", lambda *a: True)
    monkeypatch.setattr(server, "validate_service", lambda *a: "turn_off")
    call = AsyncMock()
    monkeypatch.setattr(type(hass.services), "async_call", call)
    result = await server.tool_perform_action({
        "domain": "light", "action": "turn_off",
        "target": {"entity_id": ["light.hall", "light.kitchen"],
                   "exclude": {"entity_id": "light.kitchen"}},
        "data": {"entity_id": "light.kitchen"},
    })
    assert result.get("isError") is True
    call.assert_not_awaited()


async def test_discovery_uses_computed_registry_names(hass, monkeypatch):
    registry = er.async_get(hass)
    entry = registry.async_get_or_create("sensor", "test", "temperature")
    hass.states.async_set(entry.entity_id, "20", {"friendly_name": "Legacy name"})
    monkeypatch.setattr("custom_components.mcp_assist.discovery.async_should_expose", lambda *a: True)
    monkeypatch.setattr(er, "async_get_full_entity_name", lambda *a: "Desk Temperature", raising=False)
    discovery = SmartDiscovery(hass)
    result = await discovery.discover_entities_page(name_contains="Desk Temperature")
    assert result["items"][0]["name"] == "Desk Temperature"
    monkeypatch.delattr(er, "async_get_full_entity_name")
    assert discovery._create_entity_info(hass.states.get(entry.entity_id))["name"] == "Legacy name"


async def test_internal_bridge_identity_is_task_local_and_external_calls_have_no_user(
    hass, profile_entry_factory, system_entry_factory, monkeypatch,
):
    import asyncio
    from types import SimpleNamespace
    from homeassistant.core import Context
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity
    from custom_components.mcp_assist.const import CONF_ENABLE_LLM_API_BRIDGE
    from custom_components.mcp_assist.request_context import REQUEST_HA_CONTEXT
    from custom_components.mcp_assist.tools.packages.llm_api_bridge.llm_api_bridge import LLMApiBridgeTool

    system_entry_factory(data={CONF_ENABLE_LLM_API_BRIDGE: True})
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    tool = LLMApiBridgeTool(hass)
    captured = []
    async def handle(params):
        await asyncio.sleep(0)
        captured.append(tool._create_assist_llm_context().context.user_id)
        return {"content": [{"type": "text", "text": "ok"}]}
    hass.data.setdefault("mcp_assist", {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=handle)
    async def call_as(user_id):
        await agent._call_mcp_tool("list_llm_api_tools", {"api_id": "homeassistant"},
                                   SimpleNamespace(context=Context(user_id=user_id)))
    await asyncio.gather(call_as("user-a"), call_as("user-b"))
    assert sorted(captured) == ["user-a", "user-b"]
    assert REQUEST_HA_CONTEXT.get() is None
    assert tool._create_assist_llm_context().context.user_id is None


async def test_admin_llm_api_keeps_cores_real_authorization(hass, monkeypatch):
    from homeassistant.auth.const import GROUP_ID_ADMIN
    from homeassistant.core import Context
    from homeassistant.exceptions import HomeAssistantError
    from homeassistant.helpers import llm
    from custom_components.mcp_assist.request_context import REQUEST_HA_CONTEXT
    from custom_components.mcp_assist.tools.packages.llm_api_bridge.llm_api_bridge import LLMApiBridgeTool

    if not hasattr(llm.API, "requires_admin"):
        pytest.skip("This Core release predates administrator-only LLM APIs")
    class AdminAPI(llm.API):
        async def async_get_api_instance(self, llm_context):
            return llm.APIInstance(api=self, api_prompt="", llm_context=llm_context, tools=[])
    api = AdminAPI(hass=hass, id="homeassistant", name="Home Assistant", requires_admin=True)
    llm.async_register_api(hass, api)
    admin = await hass.auth.async_create_user("Test admin", group_ids=[GROUP_ID_ADMIN])
    member = await hass.auth.async_create_user("Test member")
    tool = LLMApiBridgeTool(hass)
    monkeypatch.setattr(tool, "_allowed_llm_api_ids", lambda: ("homeassistant",))
    for user_id in (None, member.id, admin.id):
        token = REQUEST_HA_CONTEXT.set(Context(user_id=user_id))
        try:
            if user_id == admin.id:
                instance = await tool._get_llm_api_bridge_api_instance("homeassistant")
                assert instance.llm_context.context.user_id == admin.id
            else:
                with pytest.raises(HomeAssistantError):
                    await tool._get_llm_api_bridge_api_instance("homeassistant")
        finally:
            REQUEST_HA_CONTEXT.reset(token)


async def test_child_devices_inherit_parent_area_in_discovery_and_targets(
    hass, profile_entry_factory, system_entry_factory, monkeypatch,
):
    from homeassistant.helpers import device_registry as dr
    if not hasattr(dr.DeviceRegistry, "async_get_or_create_child"):
        pytest.skip("This Core release predates child devices")
    system_entry_factory()
    profile = profile_entry_factory()
    areas = ar.async_get(hass)
    area = areas.async_create("Office")
    devices = dr.async_get(hass)
    parent = devices.async_get_or_create(config_entry_id=profile.entry_id,
        identifiers={("test", "power-strip")}, name="Power strip")
    devices.async_update_device(parent.id, area_id=area.id)
    child = devices.async_get_or_create_child(config_entry_id=profile.entry_id,
        identifiers={("test", "outlet")}, parent_device_id=parent.id, name="Outlet")
    entity = er.async_get(hass).async_get_or_create("switch", "test", "outlet",
        device_id=child.id, original_name="Outlet")
    hass.states.async_set(entity.entity_id, "on")
    monkeypatch.setattr("custom_components.mcp_assist.discovery.async_should_expose", lambda *a: True)
    monkeypatch.setattr("custom_components.mcp_assist.mcp_server.async_should_expose", lambda *a: True)
    discovery = SmartDiscovery(hass)
    context = discovery._get_device_context(child, areas)
    assert context["area_id"] == area.id
    assert (await discovery.get_entities_by_area(area.id))[0]["entity_id"] == entity.entity_id
    server = MCPServer(hass, 8099, profile)
    assert await server.resolve_target({"area_id": area.id}) == {"entity_id": [entity.entity_id]}
