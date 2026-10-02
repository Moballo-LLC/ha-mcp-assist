"""Tests for Smart Index Manager lifecycle behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.mcp_assist.index_manager import IndexManager


def _fake_hass(listeners: list | None = None) -> SimpleNamespace:
    def _async_listen(event_type, listener):
        if listeners is not None:
            listeners.append(listener)
        return Mock()

    return SimpleNamespace(
        bus=SimpleNamespace(async_listen=_async_listen),
        config_entries=SimpleNamespace(async_entries=lambda domain: []),
    )


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


@pytest.mark.asyncio
async def test_get_index_coalesces_concurrent_generation() -> None:
    """Concurrent get_index callers must trigger only one index build."""
    manager = IndexManager(_fake_hass())
    calls = 0

    async def _fake_generate():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        manager._first_index_generated = True
        return {"areas": []}

    manager.generate_index = _fake_generate
    manager._is_gap_filling_enabled = AsyncMock(return_value=False)

    results = await asyncio.gather(
        manager.get_index(), manager.get_index(), manager.get_index()
    )

    assert calls == 1
    assert all(result == {"areas": []} for result in results)


@pytest.mark.asyncio
async def test_schedule_refresh_coalesces_into_single_loop_task() -> None:
    """Rapid registry events must not cancel/replace an in-flight refresh loop."""
    manager = IndexManager(_fake_hass())
    manager._refresh_debounce_seconds = 3600

    manager._schedule_refresh()
    first_task = manager._refresh_task
    manager._schedule_refresh()
    manager._schedule_refresh()

    # Same loop task is reused (old code cancelled + recreated on every event).
    assert manager._refresh_task is first_task
    assert not first_task.done()
    assert manager._refresh_pending is True

    await manager.async_stop()
    assert first_task.done()


@pytest.mark.asyncio
async def test_first_index_schedules_deferred_gap_fill_when_enabled() -> None:
    """After the first index, one follow-up refresh runs gap-filling on stable systems."""
    manager = IndexManager(_fake_hass())
    manager._refresh_debounce_seconds = 3600  # keep the follow-up task pending

    async def _fake_generate():
        manager._first_index_generated = True
        return {"areas": []}

    manager.generate_index = _fake_generate
    manager._is_gap_filling_enabled = AsyncMock(return_value=True)

    await manager.get_index()

    assert manager._initial_gap_fill_scheduled is True
    assert manager._refresh_task is not None and not manager._refresh_task.done()

    await manager.async_stop()


@pytest.mark.asyncio
async def test_first_index_skips_deferred_gap_fill_when_disabled() -> None:
    """No follow-up refresh should be scheduled when gap-filling is disabled."""
    manager = IndexManager(_fake_hass())

    async def _fake_generate():
        manager._first_index_generated = True
        return {"areas": []}

    manager.generate_index = _fake_generate
    manager._is_gap_filling_enabled = AsyncMock(return_value=False)

    await manager.get_index()

    assert manager._initial_gap_fill_scheduled is False
    assert manager._refresh_task is None


@pytest.mark.asyncio
async def test_debounce_resets_quiet_window_on_events_during_wait(monkeypatch) -> None:
    """A burst of registry events must coalesce into a single refresh."""
    manager = IndexManager(_fake_hass())
    refresh_calls = 0

    async def _fake_refresh() -> None:
        nonlocal refresh_calls
        refresh_calls += 1

    manager.refresh_index = _fake_refresh

    sleep_count = 0

    async def _fake_sleep(_delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            # A registry event lands mid-wait; the window must restart.
            manager._schedule_refresh()

    monkeypatch.setattr(
        "custom_components.mcp_assist.index_manager.asyncio.sleep", _fake_sleep
    )

    manager._schedule_refresh()
    await manager._refresh_task

    assert refresh_calls == 1  # one rebuild, not one per event
    assert sleep_count == 2  # the quiet window was reset exactly once


@pytest.mark.asyncio
async def test_registry_change_skips_refresh_when_auto_refresh_disabled(monkeypatch) -> None:
    """Turning off automatic refresh should ignore registry change events."""
    listeners: list = []
    manager = IndexManager(_fake_hass(listeners))
    monkeypatch.setattr(manager, "_is_auto_refresh_enabled", lambda: False)
    await manager.start()

    for listener in listeners:
        listener(SimpleNamespace(event_type="entity_registry_updated"))

    assert manager._refresh_task is None
    assert manager._refresh_pending is False


@pytest.mark.asyncio
async def test_registry_change_schedules_refresh_by_default() -> None:
    """Automatic refresh stays on when no shared setting is stored."""
    listeners: list = []
    manager = IndexManager(_fake_hass(listeners))
    manager._refresh_debounce_seconds = 3600
    await manager.start()

    listeners[0](SimpleNamespace(event_type="entity_registry_updated"))

    assert manager._refresh_task is not None
    await manager.async_stop()


def test_refresh_settings_default_without_system_entry() -> None:
    """Defaults apply when no shared settings entry exists."""
    manager = IndexManager(_fake_hass())

    assert manager._is_auto_refresh_enabled() is True
    assert manager._get_refresh_delay_seconds() == 60


@pytest.mark.asyncio
async def test_refresh_settings_read_from_system_entry(hass, system_entry_factory) -> None:
    """Shared settings control automatic refresh and its quiet window."""
    from custom_components.mcp_assist.const import (
        CONF_ENABLE_INDEX_AUTO_REFRESH,
        CONF_INDEX_REFRESH_DELAY_SECONDS,
    )

    system_entry_factory(
        options={
            CONF_ENABLE_INDEX_AUTO_REFRESH: False,
            CONF_INDEX_REFRESH_DELAY_SECONDS: 3600,
        }
    )
    manager = IndexManager(hass)

    assert manager._is_auto_refresh_enabled() is False
    assert manager._get_refresh_delay_seconds() == 3600


@pytest.mark.asyncio
async def test_refresh_delay_is_clamped_to_supported_range(hass, system_entry_factory) -> None:
    """Out-of-range stored delays are clamped instead of breaking refreshes."""
    from custom_components.mcp_assist.const import CONF_INDEX_REFRESH_DELAY_SECONDS

    entry = system_entry_factory(options={CONF_INDEX_REFRESH_DELAY_SECONDS: 1})
    manager = IndexManager(hass)
    assert manager._get_refresh_delay_seconds() == 10

    hass.config_entries.async_update_entry(
        entry, options={CONF_INDEX_REFRESH_DELAY_SECONDS: "not-a-number"}
    )
    assert manager._get_refresh_delay_seconds() == 60
