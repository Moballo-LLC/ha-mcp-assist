"""Tests for built-in Music Assistant tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from custom_components.mcp_assist.custom_tools.music_assistant import MusicAssistantTool


def _stub_target_resolution(
    tool: MusicAssistantTool,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entity_id: str = "media_player.kitchen",
    name: str = "Kitchen Speaker",
) -> None:
    """Stub Music Assistant target resolution for focused service-call tests."""

    async def resolve_targets(**_kwargs: Any) -> tuple[list[str], str]:
        return [entity_id], f"Resolved Music Assistant players: {name}"

    monkeypatch.setattr(tool, "_resolve_music_assistant_player_targets", resolve_targets)
    monkeypatch.setattr(tool, "_friendly_names_for_entities", lambda _ids: [name])


@pytest.mark.asyncio
async def test_control_music_assistant_player_routes_pause_to_media_player_service(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause should use the Music Assistant resolver and media_player pause service."""
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = []

    async def pause_service(call):
        calls.append(dict(call.data))

    hass.services.async_register("media_player", "media_pause", pause_service)

    result = await tool.handle_call(
        "control_music_assistant_player",
        {"action": "pause", "media_player": "Kitchen"},
    )

    assert result["isError"] is False
    assert calls == [{"entity_id": ["media_player.kitchen"]}]
    assert "Completed Music Assistant pause" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_control_music_assistant_player_sets_volume_as_media_player_level(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """set_volume should accept 0-100 and call Home Assistant's 0-1 volume API."""
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = []

    async def volume_service(call):
        calls.append(dict(call.data))

    hass.services.async_register("media_player", "volume_set", volume_service)

    result = await tool.handle_call(
        "control_music_assistant_player",
        {
            "action": "set_volume",
            "media_player": "Kitchen",
            "volume_level": 42,
        },
    )

    assert result["isError"] is False
    assert calls == [
        {
            "entity_id": ["media_player.kitchen"],
            "volume_level": 0.42,
        }
    ]
    assert "Volume: 42%." in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_control_music_assistant_player_skips_from_current_position(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """skip should seek relative to the player's current playback position."""
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = []
    updated_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    hass.states.async_set(
        "media_player.kitchen",
        "playing",
        {
            "friendly_name": "Kitchen Speaker",
            "media_position": 60,
            "media_position_updated_at": updated_at,
        },
    )

    async def seek_service(call):
        calls.append(dict(call.data))

    hass.services.async_register("media_player", "media_seek", seek_service)

    result = await tool.handle_call(
        "control_music_assistant_player",
        {
            "action": "skip",
            "media_player": "Kitchen",
            "seconds": -15,
        },
    )

    assert result["isError"] is False
    assert len(calls) == 1
    assert calls[0]["entity_id"] == "media_player.kitchen"
    assert calls[0]["seek_position"] >= 70


@pytest.mark.asyncio
async def test_transfer_music_assistant_queue_requires_distinct_single_targets(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue transfer should resolve source and target players before service call."""
    tool = MusicAssistantTool(hass)
    calls = []

    async def resolve_targets(**kwargs: Any) -> tuple[list[str], str]:
        if kwargs.get("media_player") == "Kitchen":
            return ["media_player.kitchen"], "Resolved Music Assistant players: Kitchen"
        if kwargs.get("media_player") == "Office":
            return ["media_player.office"], "Resolved Music Assistant players: Office"
        return [], ""

    monkeypatch.setattr(tool, "_resolve_music_assistant_player_targets", resolve_targets)
    monkeypatch.setattr(
        tool,
        "_friendly_names_for_entities",
        lambda ids: [
            "Kitchen Speaker" if entity_id == "media_player.kitchen" else "Office Speaker"
            for entity_id in ids
        ],
    )

    async def transfer_service(call):
        calls.append(dict(call.data))

    hass.services.async_register(
        "music_assistant",
        "transfer_queue",
        transfer_service,
    )

    result = await tool.handle_call(
        "transfer_music_assistant_queue",
        {
            "source_media_player": "Kitchen",
            "media_player": "Office",
            "auto_play": True,
        },
    )

    assert result["isError"] is False
    assert calls == [
        {
            "entity_id": "media_player.office",
            "source_player": "media_player.kitchen",
            "auto_play": True,
        }
    ]
    assert "Transferred Music Assistant queue" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_transfer_music_assistant_queue_allows_inferred_source(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue transfer may omit source so Music Assistant can infer the active queue."""
    tool = MusicAssistantTool(hass)
    service_calls = []
    resolver_calls = []

    async def resolve_targets(**kwargs: Any) -> tuple[list[str], str]:
        resolver_calls.append(kwargs)
        if kwargs.get("media_player") == "Office":
            return ["media_player.office"], "Resolved Music Assistant players: Office"
        return [], ""

    monkeypatch.setattr(tool, "_resolve_music_assistant_player_targets", resolve_targets)
    monkeypatch.setattr(
        tool,
        "_friendly_names_for_entities",
        lambda ids: ["Office Speaker" for _entity_id in ids],
    )

    async def transfer_service(call):
        service_calls.append(dict(call.data))

    hass.services.async_register(
        "music_assistant",
        "transfer_queue",
        transfer_service,
    )

    result = await tool.handle_call(
        "transfer_music_assistant_queue",
        {
            "media_player": "Office",
        },
    )

    assert result["isError"] is False
    assert service_calls == [{"entity_id": "media_player.office"}]
    assert resolver_calls == [
        {
            "area": None,
            "floor": None,
            "label": None,
            "media_player": "Office",
        }
    ]
    assert "Transferred active Music Assistant queue to Office Speaker" in result["content"][0]["text"]
