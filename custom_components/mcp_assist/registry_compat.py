"""Compatibility helpers for Home Assistant registry collections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def iter_registry_items(items: Any) -> Iterable[Any]:
    """Iterate registry entries across older mapping and newer collection APIs."""
    if isinstance(items, Mapping):
        return items.values()
    return items
