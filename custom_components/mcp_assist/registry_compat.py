"""Compatibility helpers for Home Assistant registry collections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def iter_registry_items(items: Any) -> Iterable[Any]:
    """Iterate registry entries across older mapping and newer collection APIs."""
    if isinstance(items, Mapping):
        return items.values()
    return items


def entity_display_name(hass: Any, state: Any) -> str:
    """Use Core's computed entity name, with legacy state names as fallback."""
    from homeassistant.helpers import entity_registry as er

    get_name = getattr(er, "async_get_full_entity_name", None)
    if get_name is not None and (entry := er.async_get(hass).async_get(state.entity_id)):
        try:
            if name := get_name(hass, entry):
                return name
        except (AttributeError, TypeError):
            pass
    return state.name


def device_effective_area_id(hass: Any, device: Any) -> str | None:
    """Include parent-area inheritance where Core supports child devices."""
    from homeassistant.helpers import device_registry as dr

    get_area = getattr(dr, "async_get_effective_area_id", None)
    return get_area(hass, device) if get_area is not None else device.area_id
