"""Tests for built-in Music Assistant tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import SupportsResponse
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.mcp_assist.tools.packages.music_assistant.music_assistant import MusicAssistantTool


def _add_music_assistant_entry(
    hass,
    *,
    entry_id: str = "ma_entry",
    title: str = "Music Assistant",
) -> MockConfigEntry:
    """Add a Music Assistant config entry for response-service tests."""
    entry = MockConfigEntry(
        domain="music_assistant",
        entry_id=entry_id,
        title=title,
    )
    entry.add_to_hass(hass)
    return entry


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


def _stub_playback_services(hass, response):
    """Record local search/play/shuffle calls without contacting a provider."""
    calls = []

    async def search_service(call):
        calls.append(("search", dict(call.data)))
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(call)
        return response

    async def play_service(call):
        calls.append(("play", dict(call.data)))

    async def shuffle_service(call):
        calls.append(("shuffle", dict(call.data)))

    hass.services.async_register(
        "music_assistant", "search", search_service,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register("music_assistant", "play_media", play_service)
    hass.services.async_register("media_player", "shuffle_set", shuffle_service)
    return calls


@pytest.mark.parametrize("name", ["Moonbeam FM", "Moonbeem FM", "Moonbeam FM Radio"])
@pytest.mark.parametrize("nested", [False, True])
async def test_play_music_assistant_resolves_name_before_playback(hass, monkeypatch, name, nested):
    """A shortened or slightly misspelled fictional station resolves to its URI."""
    entry = _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    items = [
        {"name": "Other Sounds Radio", "uri": "library://radio/9", "media_type": "radio"},
        {"name": "Moonbeam FM Radio", "uri": "radiobrowser://radio/22",
         "media_type": "radio", "favorite": False},
        {"name": "Moonbeam FM Radio", "uri": "library://radio/2", "media_type": "radio"},
    ]
    calls = _stub_playback_services(hass, {"radio": {"items": items} if nested else items})

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": name, "shuffle": True, "enqueue": "replace",
    })

    assert result["isError"] is False
    assert calls == [
        ("search", {"config_entry_id": entry.entry_id, "name": name,
                    "media_type": "radio", "limit": 20}),
        ("play", {"media_id": "library://radio/2", "media_type": "radio",
                  "enqueue": "replace", "entity_id": ["media_player.kitchen"]}),
        ("shuffle", {"shuffle": True, "entity_id": ["media_player.kitchen"]}),
    ]


@pytest.mark.parametrize("response", [
    {"radio": []},
    {"radio": [{"name": "Distant Echo Radio", "uri": "library://radio/9"}]},
    {"radio": [{"name": "Moonbeam FMX", "uri": "library://radio/1"},
               {"name": "Moonbeam FM", "uri": "library://radio/2"}]},
    {"radio": [{"name": "Moonbeem FM", "uri": "library://radio/1"},
               {"name": "Moonbeem FMS", "uri": "library://radio/2"}]},
    {"radio": [{"name": "Moonbeam FM", "uri": "radiobrowser://radio/1"},
               {"name": "Moonbeam FM", "uri": "radiobrowser://radio/2"}]},
    {"radio": [{"name": "Moonbeam FM Radio"}]},
    {"radio": [{"name": "Moonbeam FM Radio", "uri": "Moonbeam FM"}]},
    {"radio": [{"name": "Moonbeam FM Radio", "uri": "library://radio/1",
                "media_type": "playlist"}]},
    {"radio": ["Moonbeam FM Radio"]},
    {"radio": {"items": "invalid"}},
    {"error": "search unavailable"},
    {"radio": [] , "provider": {"error": "partial results"}},
    [],
    None,
    RuntimeError("Search unavailable"),
])
async def test_play_music_assistant_fails_closed_on_uncertain_search(
    hass, monkeypatch, response
):
    """Empty, weak, ambiguous, malformed, truncated, and failed searches never play."""
    _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = _stub_playback_services(hass, response)

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": "Moonbeam FM", "shuffle": True,
    })

    assert result["isError"] is True
    assert "canonical URI" in result["content"][0]["text"]
    assert [service for service, _data in calls] == ["search"]


@pytest.mark.parametrize("media_id, expected", [
    ("library://radio/2", "library://radio/2"),
    ("spotify://track/example", "spotify://track/example"),
    ("https://example.invalid/stream", "https://example.invalid/stream"),
    ("42", "42"),
    (["library://radio/1", "library://radio/2"], ["library://radio/1", "library://radio/2"]),
    ("library://radio/1;library://radio/2",
     ["library://radio/1", "library://radio/2"]),
])
async def test_play_music_assistant_preserves_direct_identifiers(
    hass, monkeypatch, media_id, expected
):
    """Explicit URIs, numeric IDs, and multi-item requests retain direct playback."""
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = _stub_playback_services(hass, None)

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": media_id,
    })

    assert result["isError"] is False
    assert calls == [("play", {"media_id": expected, "media_type": "radio",
                                "entity_id": ["media_player.kitchen"]})]


async def test_play_music_assistant_resolves_each_human_readable_list_item(
    hass, monkeypatch
):
    """Every name in a multi-item request is searched before the list is played."""
    _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    responses = {
        "Moonbeam FM": {"radio": [
            {"name": "Moonbeam FM", "uri": "library://radio/2"},
        ]},
        "Sunrise FM": {"radio": [
            {"name": "Sunrise FM", "uri": "library://radio/3"},
        ]},
    }
    calls = _stub_playback_services(hass, lambda call: responses[call.data["name"]])

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": ["Moonbeam FM", "Sunrise FM"],
    })

    assert result["isError"] is False
    assert [service for service, _data in calls] == ["search", "search", "play"]
    assert calls[2][1]["media_id"] == ["library://radio/2", "library://radio/3"]


async def test_play_music_assistant_prefers_favorite_duplicate_station(hass, monkeypatch):
    """Equivalent provider copies collapse to one station and prefer the favorite URI."""
    _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = _stub_playback_services(hass, {"radio": [
        {"name": "Moonbeam FM Radio", "uri": "radiobrowser://radio/22",
         "favorite": False},
        {"name": "Moonbeam FM", "uri": "library://radio/2", "favorite": True},
    ]})

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": "Moonbeam FM",
    })

    assert result["isError"] is False
    assert [service for service, _data in calls] == ["search", "play"]
    assert calls[1][1]["media_id"] == "library://radio/2"


async def test_play_music_assistant_never_prefers_a_constraint_mismatch(
    hass, monkeypatch
):
    """Favorite status cannot revive a same-name result rejected by constraints."""
    _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = _stub_playback_services(hass, {"radio": [
        {"name": "Moonbeam FM", "uri": "library://radio/2", "favorite": False,
         "artists": [{"name": "Example Broadcaster"}]},
        {"name": "Moonbeam FM", "uri": "radiobrowser://radio/22", "favorite": True,
         "artists": [{"name": "Different Broadcaster"}]},
    ]})

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": "Moonbeam FM",
        "artist": "Example Broadcaster",
    })

    assert result["isError"] is False
    assert [service for service, _data in calls] == ["search", "play"]
    assert calls[1][1]["media_id"] == "library://radio/2"


@pytest.mark.parametrize("missing_artist", [False, True])
async def test_play_music_assistant_verifies_artist_and_album_constraints(
    hass, monkeypatch, missing_artist
):
    """Search constraints must also be evidenced by the selected media item's metadata."""
    _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    item = {"name": "Paper Stars", "uri": "library://track/2", "media_type": "track",
            "album": {"name": "Folded Sky"}}
    if not missing_artist:
        item["artists"] = [{"name": "The Example Band"}]
    calls = _stub_playback_services(hass, {"tracks": [item]})

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "track", "media_id": "Paper Stars", "artist": "The Example Band",
        "album": "Folded Sky",
    })

    assert result["isError"] is missing_artist
    assert calls[0][1]["artist"] == "The Example Band"
    assert calls[0][1]["album"] == "Folded Sky"
    assert [service for service, _data in calls] == (
        ["search"] if missing_artist else ["search", "play"]
    )


async def test_play_music_assistant_search_uses_target_instance(hass, monkeypatch):
    """Multiple configured servers are disambiguated using the selected player's entry."""
    entry = _add_music_assistant_entry(hass)
    _add_music_assistant_entry(hass, entry_id="ma_other", title="Other")
    registry = er.async_get(hass)
    entity = registry.async_get_or_create(
        "media_player", "music_assistant", "example", config_entry=entry,
    )
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch, entity_id=entity.entity_id)
    item = {"name": "Moonbeam FM", "uri": "library://radio/2"}
    calls = _stub_playback_services(hass, {"radio": [item, item]})

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": "Moonbeam FM",
    })

    assert result["isError"] is False
    assert calls[0][1]["config_entry_id"] == entry.entry_id


async def test_play_music_assistant_without_search_does_not_fall_back(hass, monkeypatch):
    """An unavailable search service cannot send a guessed name to playback."""
    _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    _stub_target_resolution(tool, monkeypatch)
    calls = _stub_playback_services(hass, None)
    hass.services.async_remove("music_assistant", "search")

    result = await tool.handle_call("play_music_assistant", {
        "media_type": "radio", "media_id": "Moonbeam FM",
    })

    assert result["isError"] is True
    assert calls == []


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


def test_music_assistant_search_and_library_schemas_include_library_media_types(
    hass,
) -> None:
    """Search and library discovery should expose Music Assistant library types."""
    tool = MusicAssistantTool(hass)
    definitions = {definition["name"]: definition for definition in tool.get_tool_definitions()}

    search_schema = definitions["search_music_assistant"]["inputSchema"]["properties"][
        "media_type"
    ]
    library_schema = definitions["get_music_assistant_library"]["inputSchema"][
        "properties"
    ]["media_type"]
    playback_schema = definitions["play_music_assistant"]["inputSchema"]["properties"][
        "media_type"
    ]

    search_types = set(search_schema["oneOf"][0]["enum"])
    search_array_types = set(search_schema["oneOf"][1]["items"]["enum"])
    library_types = set(library_schema["enum"])
    playback_types = set(playback_schema["enum"])

    assert {"audiobook", "podcast", "playlist"}.issubset(search_types)
    assert {"audiobooks", "podcasts", "playlists"}.issubset(search_types)
    assert search_types == search_array_types
    assert {"audiobook", "podcast", "playlist"}.issubset(library_types)
    assert {"audiobooks", "podcasts", "playlists"}.issubset(library_types)
    assert "podcast" not in playback_types
    assert "podcasts" not in playback_types


@pytest.mark.asyncio
async def test_get_music_assistant_library_supports_podcast_media_type(hass) -> None:
    """Library browsing should pass podcast media types to Music Assistant."""
    entry = _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    calls = []

    async def get_library_service(call):
        calls.append(dict(call.data))
        return {"items": [{"name": "The Helpful Home"}]}

    hass.services.async_register(
        "music_assistant",
        "get_library",
        get_library_service,
        supports_response=SupportsResponse.ONLY,
    )

    result = await tool.handle_call(
        "get_music_assistant_library",
        {"media_type": "podcasts", "limit": 5, "offset": 2},
    )

    assert result["isError"] is False
    assert calls == [
        {
            "config_entry_id": entry.entry_id,
            "media_type": "podcast",
            "limit": 5,
            "offset": 2,
        }
    ]
    assert result["response"] == {"items": [{"name": "The Helpful Home"}]}


@pytest.mark.asyncio
async def test_search_music_assistant_supports_audiobook_and_podcast_filters(
    hass,
) -> None:
    """Search should normalize audiobook and podcast media type filters."""
    entry = _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    calls = []

    async def search_service(call):
        calls.append(dict(call.data))
        return {
            "podcast": {"items": [{"name": "The Helpful Home"}]},
            "audiobook": {"items": [{"name": "A Calm Manual"}]},
        }

    hass.services.async_register(
        "music_assistant",
        "search",
        search_service,
        supports_response=SupportsResponse.ONLY,
    )

    result = await tool.handle_call(
        "search_music_assistant",
        {
            "name": "home",
            "media_type": ["podcasts", "audiobooks", "podcast"],
            "library_only": True,
            "limit": 3,
        },
    )

    assert result["isError"] is False
    assert calls == [
        {
            "config_entry_id": entry.entry_id,
            "name": "home",
            "limit": 3,
            "media_type": ["podcast", "audiobook"],
            "library_only": True,
        }
    ]
    assert "The Helpful Home" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_search_music_assistant_can_search_within_native_media_result(
    hass, monkeypatch
) -> None:
    """A copied native media result scopes search to its Music Assistant player."""
    entry = _add_music_assistant_entry(hass)
    tool = MusicAssistantTool(hass)
    resolver = AsyncMock(return_value=(["media_player.office"], "Resolved Office"))
    monkeypatch.setattr(tool, "_resolve_music_assistant_player_targets", resolver)
    responses = [{"media_player.office": {"result": [{"title": "Example track"}]}}]
    calls = []

    async def search_media_service(call):
        calls.append(dict(call.data))
        return responses[0]

    hass.services.async_register(
        "media_player",
        "search_media",
        search_media_service,
        supports_response=SupportsResponse.ONLY,
    )

    result = await tool.handle_call(
        "search_music_assistant",
        {
            "name": "track",
            "config_entry_id": entry.entry_id,
            "within_media_content_id": "artist/example",
            "within_media_content_type": "artist",
            "media_player": "Office",
        },
    )

    assert result["isError"] is False
    assert result["response"] == {
        "media_player.office": {"result": [{"title": "Example track"}]}
    }
    resolver.assert_awaited_once_with(
        area=None,
        floor=None,
        label=None,
        media_player="Office",
        config_entry_id=entry.entry_id,
    )
    expected_service_data = {
            "entity_id": ["media_player.office"],
            "search_query": "track",
            "media_content_id": "artist/example",
            "media_content_type": "artist",
        }
    assert calls == [expected_service_data]

    service_error = {"media_player.office": {"error": "search unavailable"}}
    responses[0] = service_error
    error_result = await tool.handle_call(
        "search_music_assistant",
        {
            "name": "track",
            "config_entry_id": entry.entry_id,
            "within_media_content_id": "artist/example",
            "within_media_content_type": "artist",
            "media_player": "Office",
        },
    )
    assert error_result["isError"] is True
    assert error_result["response"] == service_error
    assert "❌ Home Assistant returned an error" in error_result["content"][0]["text"]
    assert calls == [expected_service_data, expected_service_data]


@pytest.mark.parametrize("selector", [None, "media_player.office"])
async def test_contextual_search_uses_real_catalog_and_selected_instance(
    hass, monkeypatch, selector
):
    """Registry-backed searches keep only exposed players in the chosen instance."""
    entry = _add_music_assistant_entry(hass)
    other = _add_music_assistant_entry(hass, entry_id="ma_other", title="Other")
    registry = er.async_get(hass)
    for name, config_entry in [("office", entry), ("hidden", entry), ("other", other)]:
        entity = registry.async_get_or_create(
            "media_player", "music_assistant", name,
            suggested_object_id=name, config_entry=config_entry,
        )
        hass.states.async_set(entity.entity_id, "idle", {"friendly_name": name.title()})
    monkeypatch.setattr(
        "custom_components.mcp_assist.tools.packages.music_assistant.music_assistant.async_should_expose",
        lambda _hass, _assistant, entity_id: entity_id != "media_player.hidden",
    )
    tool = MusicAssistantTool(hass)
    calls = []
    response = {"media_player.office": {"result": [{
        "title": "Example Track", "media_content_id": "track/example",
        "media_content_type": "track",
    }]}}

    async def search_service(call):
        calls.append(dict(call.data))
        return response

    hass.services.async_register(
        "media_player", "search_media", search_service,
        supports_response=SupportsResponse.ONLY,
    )
    arguments = {
        "name": "track", "config_entry_id": entry.entry_id,
        "within_media_content_id": "artist/example",
        "within_media_content_type": "artist",
    }
    if selector:
        arguments["media_player"] = selector
    result = await tool.handle_call("search_music_assistant", arguments)
    assert result["isError"] is False
    assert result["response"] == response
    assert calls == [{
        "entity_id": ["media_player.office"], "search_query": "track",
        "media_content_id": "artist/example", "media_content_type": "artist",
    }]

    arguments["media_player"] = "media_player.other"
    rejected = await tool.handle_call("search_music_assistant", arguments)
    assert rejected["isError"] is True
    assert len(calls) == 1
