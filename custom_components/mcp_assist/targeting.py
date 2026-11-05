"""Native target exclusions without widening conversation exposure."""
from __future__ import annotations

from typing import Any

from homeassistant.components.homeassistant import async_should_expose

try:
    from homeassistant.helpers.target import (
        TargetSelection,
        async_extract_excluded_entity_ids,
        async_extract_referenced_entity_ids,
    )
except ImportError:  # Older supported Core releases have no native exclusions.
    TargetSelection = None
    async_extract_excluded_entity_ids = None
    async_extract_referenced_entity_ids = None

TARGET_KEYS = ("entity_id", "device_id", "area_id", "floor_id", "label_id")


def target_schema() -> dict[str, Any]:
    """Advertise native exclusions only when the installed Core supports them."""
    properties = {
        key: {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}
        for key in TARGET_KEYS
    }
    schema = {"type": "object", "properties": properties, "minProperties": 1,
              "additionalProperties": False,
              "description": "Entity, device, area, floor, or label IDs; resolved to exposed entities."}
    if async_extract_excluded_entity_ids is not None:
        properties["exclude"] = {
            "type": "object", "properties": dict(properties), "additionalProperties": False,
            "description": "Leave out matching entities, devices, areas, floors, or labels.",
        }
    return schema


def apply_target_exclusions(hass: Any, entity_ids: set[str], target: dict) -> set[str]:
    """Apply Core's group-aware exclusion semantics to an already authorized target."""
    if "exclude" not in target:
        return entity_ids
    exclude = target["exclude"]
    if not isinstance(exclude, dict) or set(exclude).difference(TARGET_KEYS):
        raise ValueError("exclude must contain entity_id, device_id, area_id, floor_id, or label_id")
    for value in exclude.values():
        if not isinstance(value, str) and not (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        ):
            raise ValueError("Exclusion IDs must be strings or lists of strings")
    if async_extract_excluded_entity_ids is None:
        raise ValueError("This Home Assistant release does not support native target exclusions")
    selection = TargetSelection({"entity_id": sorted(entity_ids), "exclude": exclude})
    selected = async_extract_referenced_entity_ids(hass, selection)
    return {
        entity_id for entity_id in selected.referenced | selected.indirectly_referenced
        if hass.states.get(entity_id) is not None
        and async_should_expose(hass, "conversation", entity_id)
    }
