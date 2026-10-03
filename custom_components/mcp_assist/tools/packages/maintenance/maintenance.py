"""Read current maintenance signals from exposed Home Assistant entities."""

from __future__ import annotations

import math
from typing import Any

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.core import State

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


CATEGORY_PRIORITY = {"problems": 0, "unavailable": 1, "batteries": 2, "updates": 3}


class MaintenanceTool(MCPAssistExternalTool):
    """Summarize native entity signals without attributes or corrective actions."""

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the optional read-only tool schema."""
        return [{
            "name": "get_maintenance_status",
            "description": "Read exposed entities' low batteries, updates, unavailable states, "
            "and native problem signals. Results do not establish whole-home health.",
            "annotations": {"readOnlyHint": True, "destructiveHint": False},
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "batteries", "updates", "unavailable", "problems"],
                        "default": "all",
                    },
                    "battery_threshold": {
                        "type": "integer", "minimum": 1, "maximum": 100, "default": 30,
                    },
                    "limit": {
                        "type": "integer", "minimum": 1, "maximum": 100, "default": 25,
                    },
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "additionalProperties": False,
            },
        }]

    @staticmethod
    def _signal(state: State, threshold: int) -> tuple[str, float | None] | None:
        """Match only native domains, device classes, units, and current states."""
        if state.state in {"unavailable", "unknown"}:
            return "unavailable", None
        domain = state.domain
        device_class = state.attributes.get("device_class")
        if domain == "binary_sensor" and state.state == "on":
            if device_class == "problem":
                return "problems", None
            if device_class == "battery":
                return "batteries", None
        if domain == "update" and state.state == "on":
            return "updates", None
        if (
            domain == "sensor"
            and device_class == "battery"
            and state.attributes.get("unit_of_measurement") == "%"
        ):
            try:
                percentage = float(state.state)
            except (TypeError, ValueError, OverflowError):
                return None
            if math.isfinite(percentage) and 0 <= percentage <= threshold:
                return "batteries", percentage
        return None

    async def handle_call(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Scan on demand and return a deterministic page of exposed signals."""
        if tool_name != "get_maintenance_status":
            return self.error("Unknown maintenance tool.")
        if set(arguments) - {"category", "battery_threshold", "limit", "offset"}:
            return self.error("Unsupported maintenance argument.")
        category = arguments.get("category", "all")
        if not isinstance(category, str) or category not in {"all", *CATEGORY_PRIORITY}:
            return self.error("Invalid maintenance category.")
        values = {}
        for name, default, minimum, maximum in (
            ("battery_threshold", 30, 1, 100), ("limit", 25, 1, 100),
            ("offset", 0, 0, None),
        ):
            value = arguments.get(name, default)
            if (
                type(value) is not int or value < minimum
                or (maximum is not None and value > maximum)
            ):
                return self.error(f"Invalid {name}.")
            values[name] = value

        matched = []
        exposed_count = 0
        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, "conversation", state.entity_id):
                continue
            exposed_count += 1
            signal = self._signal(state, values["battery_threshold"])
            if signal is None or (category != "all" and signal[0] != category):
                continue
            name = state.attributes.get("friendly_name")
            item = {
                "category": signal[0], "entity_id": state.entity_id,
                "friendly_name": name if isinstance(name, str) else state.entity_id,
                "state": state.state,
            }
            if signal[1] is not None:
                item["battery_percentage"] = signal[1]
            matched.append(item)
        matched.sort(key=lambda item: (CATEGORY_PRIORITY[item["category"]], item["entity_id"]))
        offset, limit = values["offset"], values["limit"]
        page = matched[offset:offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(matched) else None
        data = {
            "schema_version": 1, "scope": "exposed_entities_only", "category": category,
            "battery_threshold": values["battery_threshold"], "exposed_count": exposed_count,
            "matched_count": len(matched), "offset": offset, "page_count": len(page),
            "next_offset": next_offset, "truncated": len(page) < len(matched), "items": page,
        }
        lines = [
            f"Maintenance signals among exposed entities only: checked {exposed_count}; "
            f"matched {len(matched)}; returned {len(page)}.",
            "This does not establish whole-home health.",
        ]
        for item in page:
            percentage = (
                f" ({item['battery_percentage']:g}%)" if "battery_percentage" in item else ""
            )
            lines.append(
                f"{item['category']}: {item['friendly_name']} ({item['entity_id']}): "
                f"{item['state']}{percentage}"
            )
        return self.ok("\n".join(lines), structured_content=data)
