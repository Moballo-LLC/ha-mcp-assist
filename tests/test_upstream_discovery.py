"""Regression coverage for upstream v0.19 discovery improvements."""

import pytest
from homeassistant.helpers import area_registry as ar, entity_registry as er

from custom_components.mcp_assist.discovery import SmartDiscovery


@pytest.mark.parametrize("area_filter", [None, "Kitchen"])
async def test_name_filter_remains_active(hass, monkeypatch, area_filter):
    """A name containing 'all' is still a name, including within an area."""
    monkeypatch.setattr(
        "custom_components.mcp_assist.discovery.async_should_expose", lambda *args: True
    )
    area = ar.async_get(hass).async_create("Kitchen")
    registry = er.async_get(hass)
    for name in ("hallway", "desk"):
        entity = registry.async_get_or_create("light", "test", name, suggested_object_id=name)
        registry.async_update_entity(entity.entity_id, area_id=area.id)
        hass.states.async_set(entity.entity_id, "on", {"friendly_name": name})
    result = await SmartDiscovery(hass).discover_entities_page(
        domain="light", area=area_filter, name_contains="hallway"
    )
    assert [item["entity_id"] for item in result["items"] if item["entity_id"] != "_summary"] == ["light.hallway"]


@pytest.mark.parametrize("domain", ["all", "*", " ALL "])
async def test_wildcard_domain_means_no_filter(hass, monkeypatch, domain):
    monkeypatch.setattr(
        "custom_components.mcp_assist.discovery.async_should_expose", lambda *args: True
    )
    hass.states.async_set("light.desk", "on")
    result = await SmartDiscovery(hass).discover_entities_page(domain=domain)
    assert [item["entity_id"] for item in result["items"] if item["entity_id"] != "_summary"] == ["light.desk"]
