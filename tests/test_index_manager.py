"""Tests for Smart Index Manager lifecycle behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from custom_components.mcp_assist.index_manager import IndexManager


@pytest.mark.asyncio
async def test_index_manager_stop_removes_listeners_and_cancels_refresh(
) -> None:
    """Stopping the manager should release listeners and pending refresh work."""
    unsubscribe_callbacks: list[Mock] = []

    def _async_listen(event_type, listener):
        del event_type, listener
        unsubscribe = Mock()
        unsubscribe_callbacks.append(unsubscribe)
        return unsubscribe

    hass = SimpleNamespace(bus=SimpleNamespace(async_listen=_async_listen))
    manager = IndexManager(hass)

    await manager.start()
    await manager.start()

    assert len(unsubscribe_callbacks) >= 3
    registered_count = len(unsubscribe_callbacks)

    manager._refresh_debounce_seconds = 3600
    manager._schedule_refresh()
    refresh_task = manager._refresh_task

    assert refresh_task is not None
    assert not refresh_task.done()

    await manager.async_stop()

    assert len(unsubscribe_callbacks) == registered_count
    for unsubscribe in unsubscribe_callbacks:
        unsubscribe.assert_called_once()
    assert manager._refresh_task is None
    assert refresh_task.done()
