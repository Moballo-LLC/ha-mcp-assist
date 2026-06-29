"""Built-in Music Assistant packaged tools for MCP Assist."""

from __future__ import annotations

import copy
from datetime import date, datetime, time
import json
import logging
from typing import Any, Dict, List, Tuple

from homeassistant.components.homeassistant import async_should_expose
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

try:
    from homeassistant.helpers import floor_registry as fr
except ImportError:  # pragma: no cover - older Home Assistant versions
    fr = None

try:
    from homeassistant.helpers import label_registry as lr
except ImportError:  # pragma: no cover - older Home Assistant versions
    lr = None

from ....const import DOMAIN, MAX_ENTITIES_PER_DISCOVERY
from ....discovery import EntityDiscovery

_LOGGER = logging.getLogger(__name__)

MUSIC_ASSISTANT_PLAYBACK_MEDIA_TYPES = ("track", "album", "artist", "playlist", "radio")
MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPES = (
    "track",
    "album",
    "artist",
    "playlist",
    "radio",
    "audiobook",
    "podcast",
)
MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_SET = set(MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPES)
MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_TEXT = ", ".join(MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPES)
MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_ALIASES = {
    "tracks": "track",
    "albums": "album",
    "artists": "artist",
    "playlists": "playlist",
    "radios": "radio",
    "radio station": "radio",
    "radio stations": "radio",
    "stations": "radio",
    "audiobooks": "audiobook",
    "podcasts": "podcast",
}
MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_INPUTS = (
    *MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPES,
    *MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_ALIASES,
)

MUSIC_ASSISTANT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_music_assistant_players",
        "description": "List Music Assistant media_player entities only. Use this to inspect or disambiguate valid Music Assistant playback targets by name, area, floor, or label without mixing in unrelated media_player entities.",
        "llmDescription": "List Music Assistant players by area, floor, label, or name.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "Optional area name or alias to filter Music Assistant players. If the value matches a floor name or alias instead, it will search that floor.",
                },
                "floor": {
                    "type": "string",
                    "description": "Optional floor name or alias to filter Music Assistant players.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label name to filter Music Assistant players.",
                },
                "name_contains": {
                    "type": "string",
                    "description": "Optional text to match against Music Assistant player names, aliases, related device names, area aliases, and floor aliases.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of Music Assistant players to return (default: 20, max: {MAX_ENTITIES_PER_DISCOVERY})",
                    "default": 20,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "play_music_assistant",
        "description": "Play music using the Home Assistant Music Assistant integration. This resolves only Music Assistant players, supports area/floor/label targeting, and is safer than generic media_player playback when Music Assistant is in use.",
        "llmDescription": "Play music on Music Assistant players.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "media_type": {
                    "type": "string",
                    "enum": list(MUSIC_ASSISTANT_PLAYBACK_MEDIA_TYPES),
                    "description": "The type of content to play.",
                },
                "media_id": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "The Music Assistant media identifier, URI, name, or list of items to play.",
                },
                "artist": {
                    "type": "string",
                    "description": "Optional artist name to narrow track/album playback.",
                },
                "album": {
                    "type": "string",
                    "description": "Optional album name to narrow track playback.",
                },
                "media_description": {
                    "type": "string",
                    "description": "Optional natural-language description of the requested media for logging and result summaries.",
                },
                "area": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional area name or alias, resolved only to Music Assistant players in that area.",
                },
                "floor": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional floor name or alias, resolved only to Music Assistant players on that floor.",
                },
                "label": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional label name, resolved only to Music Assistant players carrying that label context.",
                },
                "media_player": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional Music Assistant player entity_id, friendly name, or alias. Only Music Assistant players are matched.",
                },
                "shuffle": {
                    "type": "boolean",
                    "description": "Optional shuffle state to apply after starting playback.",
                },
                "radio_mode": {
                    "type": "boolean",
                    "description": "Optional Music Assistant radio mode flag.",
                },
                "enqueue": {
                    "type": "string",
                    "enum": ["play", "replace", "next", "replace_next", "add"],
                    "description": "Optional queue behavior for Music Assistant playback.",
                },
            },
            "required": ["media_type", "media_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_music_assistant_instances",
        "description": "List configured Music Assistant integration instances. Use this when multiple Music Assistant servers are configured and you need a specific instance for library discovery.",
        "llmDescription": "List configured Music Assistant instances.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_music_assistant",
        "description": "Search the Music Assistant library and providers using a resolved Music Assistant instance. Prefer this over generic service calls when you want LLM-friendly music discovery results and automatic instance resolution.",
        "llmDescription": "Search Music Assistant for tracks, artists, albums, playlists, radio, audiobooks, or podcasts.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The track, artist, album, playlist, radio station, audiobook, or podcast name to search for.",
                },
                "media_type": {
                    "oneOf": [
                        {
                            "type": "string",
                            "enum": list(MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_INPUTS),
                        },
                        {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_INPUTS),
                            },
                        },
                    ],
                    "description": "Optional Music Assistant media type or list of media types to narrow the search.",
                },
                "artist": {
                    "type": "string",
                    "description": "Optional artist constraint for track or album searches.",
                },
                "album": {
                    "type": "string",
                    "description": "Optional album constraint for track searches.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional result limit per media type (default: 10, max: 50).",
                    "default": 10,
                },
                "library_only": {
                    "type": "boolean",
                    "description": "When true, limit results to items already in the Music Assistant library.",
                },
                "config_entry_id": {
                    "type": "string",
                    "description": "Optional Music Assistant config entry ID. Use this when multiple Music Assistant instances exist.",
                },
                "instance": {
                    "type": "string",
                    "description": "Optional Music Assistant instance title/name. Use list_music_assistant_instances first if needed.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_music_assistant_library",
        "description": "Browse or filter the Music Assistant library using a resolved Music Assistant instance. Use this for curated discovery like favorite artists, random tracks, or filtered library views.",
        "llmDescription": "Browse or filter the Music Assistant library.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "media_type": {
                    "type": "string",
                    "enum": list(MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_INPUTS),
                    "description": "The library media type to list.",
                },
                "search": {
                    "type": "string",
                    "description": "Optional filter text to narrow the library results.",
                },
                "favorite": {
                    "type": "boolean",
                    "description": "Optional favorite filter.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of results to return (default: 25, max: 100).",
                    "default": 25,
                },
                "offset": {
                    "type": "integer",
                    "description": "Optional pagination offset.",
                    "default": 0,
                },
                "order_by": {
                    "type": "string",
                    "description": "Optional Music Assistant sort field.",
                },
                "album_artists_only": {
                    "type": "boolean",
                    "description": "Optional Music Assistant album_artists_only flag for artist library views.",
                },
                "album_type": {
                    "type": "string",
                    "description": "Optional Music Assistant album_type filter for album library views.",
                },
                "config_entry_id": {
                    "type": "string",
                    "description": "Optional Music Assistant config entry ID. Use this when multiple Music Assistant instances exist.",
                },
                "instance": {
                    "type": "string",
                    "description": "Optional Music Assistant instance title/name. Use list_music_assistant_instances first if needed.",
                },
            },
            "required": ["media_type"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_music_assistant_queue",
        "description": "Read the current Music Assistant queue for specific Music Assistant players. This resolves only Music Assistant players and returns queue details for one or more target players.",
        "llmDescription": "Read the current Music Assistant queue.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "area": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional area name or alias to resolve Music Assistant players.",
                },
                "floor": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional floor name or alias to resolve Music Assistant players.",
                },
                "label": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional label name to resolve Music Assistant players.",
                },
                "media_player": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional Music Assistant player entity_id, friendly name, or alias.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "control_music_assistant_player",
        "description": "Control Music Assistant playback, queue settings, and volume for resolved Music Assistant players. Use this instead of generic media_player services when the target is managed by Music Assistant.",
        "llmDescription": "Pause, resume, skip, stop, seek, adjust volume, mute, shuffle, repeat, or clear Music Assistant queues.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "pause",
                        "resume",
                        "toggle",
                        "stop",
                        "next",
                        "previous",
                        "seek",
                        "skip",
                        "set_volume",
                        "volume_up",
                        "volume_down",
                        "mute",
                        "shuffle",
                        "repeat",
                        "clear_queue",
                    ],
                    "description": "The Music Assistant control action to perform.",
                },
                "area": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional area name or alias to resolve Music Assistant players.",
                },
                "floor": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional floor name or alias to resolve Music Assistant players.",
                },
                "label": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional label name to resolve Music Assistant players.",
                },
                "media_player": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional Music Assistant player entity_id, friendly name, or alias.",
                },
                "position": {
                    "type": "integer",
                    "description": "Required for seek. Absolute playback position in seconds from the start of the current item.",
                },
                "seconds": {
                    "type": "integer",
                    "description": "Optional for skip. Relative seconds to move from the current playback position. Defaults to 10 and can be negative.",
                    "default": 10,
                },
                "volume_level": {
                    "type": "integer",
                    "description": "Required for set_volume. Target volume from 0 to 100.",
                },
                "muted": {
                    "type": "boolean",
                    "description": "Required for mute. True to mute, false to unmute.",
                },
                "shuffle": {
                    "type": "boolean",
                    "description": "Required for shuffle. True to enable shuffle, false to disable it.",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["off", "one", "all"],
                    "description": "Required for repeat. Repeat mode to apply to the queue.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "transfer_music_assistant_queue",
        "description": "Transfer a Music Assistant queue from one Music Assistant player to another resolved Music Assistant player.",
        "llmDescription": "Move the active Music Assistant queue from one player to another.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "source_media_player": {
                    "type": "string",
                    "description": "Source Music Assistant player entity_id, friendly name, or alias whose queue should move.",
                },
                "area": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional target area name or alias to resolve one Music Assistant player.",
                },
                "floor": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional target floor name or alias to resolve one Music Assistant player.",
                },
                "label": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Optional target label name to resolve one Music Assistant player.",
                },
                "media_player": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Target Music Assistant player entity_id, friendly name, or alias.",
                },
                "auto_play": {
                    "type": "boolean",
                    "description": "Optional Music Assistant auto_play flag for the transferred queue.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
]

MUSIC_ASSISTANT_TOOL_NAMES = {
    str(tool_definition["name"]) for tool_definition in MUSIC_ASSISTANT_TOOL_DEFINITIONS
}


def describe_music_assistant_item(item: Any) -> str | None:
    """Build a compact description for a Music Assistant media item."""
    if not isinstance(item, dict):
        return str(item) if item is not None else None

    name = item.get("name") or item.get("title") or item.get("uri")
    artist = item.get("artist")
    if isinstance(artist, dict):
        artist = artist.get("name")
    elif isinstance(artist, list) and artist:
        first_artist = artist[0]
        artist = first_artist.get("name") if isinstance(first_artist, dict) else str(first_artist)

    if name and artist and str(artist).strip().casefold() not in str(name).casefold():
        return f"{artist} - {name}"
    if name:
        return str(name)
    return None


def summarize_music_assistant_collection_response(response: Any) -> List[str]:
    """Summarize Music Assistant search/library payloads."""
    if not isinstance(response, dict):
        return []

    collections: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(response.get("items"), list):
        collections.append(("items", response))
    else:
        for key, value in response.items():
            if isinstance(value, dict) and isinstance(value.get("items"), list):
                collections.append((str(key), value))

    if not collections:
        return []

    lines = ["Summary:"]
    for key, payload in collections:
        items = payload.get("items") or []
        preview = describe_music_assistant_item(items[0]) if items else None
        line = f"- {key}: {len(items)} item{'s' if len(items) != 1 else ''}"
        if preview:
            line += f"; first: {preview}"
        lines.append(line)
    return lines


def summarize_music_assistant_queue_response(response: Any) -> List[str]:
    """Summarize Music Assistant queue payloads."""
    if not isinstance(response, dict):
        return []

    queue_payloads: List[Tuple[str, Dict[str, Any]]] = []
    if "items" in response or "current_item" in response:
        queue_payloads.append(("queue", response))
    else:
        for key, value in response.items():
            if isinstance(value, dict) and (
                "items" in value or "current_item" in value
            ):
                queue_payloads.append((str(key), value))

    if not queue_payloads:
        return []

    lines = ["Summary:"]
    for key, payload in queue_payloads:
        items = payload.get("items") or []
        current_item = payload.get("current_item")
        detail_parts = [f"{len(items)} queued item{'s' if len(items) != 1 else ''}"]
        current_preview = describe_music_assistant_item(current_item)
        if current_preview:
            detail_parts.append(f"current: {current_preview}")
        lines.append(f"- {key}: {'; '.join(detail_parts)}")
    return lines


def summarize_music_assistant_response(service: str, response: Any) -> List[str]:
    """Build concise summaries for Music Assistant response payloads."""
    if service in {"search", "get_library"}:
        return summarize_music_assistant_collection_response(response)
    if service == "get_queue":
        return summarize_music_assistant_queue_response(response)
    return []


class MusicAssistantTool:
    """Built-in packaged Music Assistant tools."""

    def __init__(self, hass) -> None:
        """Initialize the Music Assistant tool bundle."""
        self.hass = hass
        self.discovery = EntityDiscovery(hass)

    async def initialize(self) -> None:
        """Initialize the tool bundle."""
        return None

    async def async_shutdown(self) -> None:
        """Clean up tool resources."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles a tool."""
        return tool_name in MUSIC_ASSISTANT_TOOL_NAMES

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Music Assistant MCP tool definitions."""
        return copy.deepcopy(MUSIC_ASSISTANT_TOOL_DEFINITIONS)

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a Music Assistant tool call."""
        if tool_name == "list_music_assistant_players":
            return await self.tool_list_music_assistant_players(arguments)
        if tool_name == "play_music_assistant":
            return await self.tool_play_music_assistant(arguments)
        if tool_name == "list_music_assistant_instances":
            return await self.tool_list_music_assistant_instances(arguments)
        if tool_name == "search_music_assistant":
            return await self.tool_search_music_assistant(arguments)
        if tool_name == "get_music_assistant_library":
            return await self.tool_get_music_assistant_library(arguments)
        if tool_name == "get_music_assistant_queue":
            return await self.tool_get_music_assistant_queue(arguments)
        if tool_name == "control_music_assistant_player":
            return await self.tool_control_music_assistant_player(arguments)
        if tool_name == "transfer_music_assistant_queue":
            return await self.tool_transfer_music_assistant_queue(arguments)
        return self._text_result(
            f"Unknown Music Assistant tool: {tool_name}",
            is_error=True,
        )

    async def tool_list_music_assistant_players(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """List Music Assistant players only."""
        self._publish_progress(
            "tool_start",
            "Listing Music Assistant players",
            tool="list_music_assistant_players",
            args=args,
        )

        try:
            players = await self._discover_music_assistant_players(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                name_contains=args.get("name_contains"),
                limit=self._coerce_int_arg(
                    args.get("limit"),
                    default=20,
                    minimum=1,
                    maximum=MAX_ENTITIES_PER_DISCOVERY,
                ),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        self._publish_progress(
            "tool_complete",
            f"Music Assistant player listing complete: found {len(players)} players",
            tool="list_music_assistant_players",
            count=len(players),
        )

        if not players:
            return self._text_result("No exposed Music Assistant players matched that query.")

        payload = {"count": len(players), "players": players}
        header = f"Found {len(players)} Music Assistant player{'s' if len(players) != 1 else ''}."
        return self._text_result(
            header + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)
        )

    async def tool_play_music_assistant(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Play media using the Music Assistant integration."""
        if not self.hass.services.has_service("music_assistant", "play_media"):
            return self._text_result(
                "The Home Assistant Music Assistant integration is not available or does not expose music_assistant.play_media."
            )

        media_type = str(args.get("media_type") or "").strip().lower()
        if media_type not in MUSIC_ASSISTANT_PLAYBACK_MEDIA_TYPES:
            return self._text_result(
                "❌ Error: media_type must be one of track, album, artist, playlist, or radio."
            )

        normalized_media_id = self._normalize_music_assistant_media_id(args.get("media_id"))
        if normalized_media_id in (None, "", []):
            return self._text_result("❌ Error: media_id is required.")

        try:
            resolved_player_ids, resolution_text = await self._resolve_music_assistant_player_targets(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                media_player=args.get("media_player"),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        service_data: Dict[str, Any] = {
            "media_id": normalized_media_id,
            "media_type": media_type,
        }
        for key in ("artist", "album", "enqueue"):
            value = args.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                service_data[key] = value

        radio_mode = args.get("radio_mode")
        if isinstance(radio_mode, bool):
            service_data["radio_mode"] = radio_mode

        shuffle = args.get("shuffle")
        shuffle_requested = isinstance(shuffle, bool)
        media_description = str(args.get("media_description") or "").strip()

        self._publish_progress(
            "tool_start",
            "Starting Music Assistant playback",
            tool="play_music_assistant",
            media_type=media_type,
            target_count=len(resolved_player_ids),
        )

        try:
            await self.hass.services.async_call(
                domain="music_assistant",
                service="play_media",
                service_data={**service_data, "entity_id": resolved_player_ids},
                blocking=True,
                return_response=False,
            )

            if shuffle_requested and self.hass.services.has_service("media_player", "shuffle_set"):
                await self.hass.services.async_call(
                    domain="media_player",
                    service="shuffle_set",
                    service_data={
                        "shuffle": shuffle,
                        "entity_id": resolved_player_ids,
                    },
                    blocking=True,
                    return_response=False,
                )
        except Exception as err:
            error_msg = f"Music Assistant playback failed: {err}"
            _LOGGER.exception(error_msg)
            return self._text_result(f"❌ Error: {error_msg}")

        self._publish_progress(
            "tool_complete",
            "Music Assistant playback started",
            tool="play_music_assistant",
            success=True,
            target_count=len(resolved_player_ids),
        )

        player_names = self._friendly_names_for_entities(resolved_player_ids)
        target_text = ", ".join(player_names)
        description_text = media_description or (
            ", ".join(normalized_media_id)
            if isinstance(normalized_media_id, list)
            else str(normalized_media_id)
        )

        text_parts = [
            f"✅ Started Music Assistant playback for {description_text} on {target_text}.",
            f"Media type: {media_type}",
        ]
        if resolution_text:
            text_parts.append(resolution_text)
        if shuffle_requested:
            text_parts.append(f"Shuffle set to {'on' if shuffle else 'off'}.")

        return self._text_result("\n".join(text_parts))

    async def tool_list_music_assistant_instances(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """List configured Music Assistant instances."""
        del args

        instances = self._get_music_assistant_instances()
        if not instances:
            return self._text_result(
                "No Music Assistant config entries are currently configured in Home Assistant."
            )

        players = [
            dict(record["entity_info"])
            for record in self._get_music_assistant_player_catalog()
        ]
        player_counts: Dict[str, int] = {}
        for player in players:
            config_entry_id = player.get("config_entry_id")
            if config_entry_id:
                player_counts[config_entry_id] = player_counts.get(config_entry_id, 0) + 1

        payload = {
            "count": len(instances),
            "instances": [
                {
                    "config_entry_id": entry.entry_id,
                    "title": entry.title,
                    "player_count": player_counts.get(entry.entry_id, 0),
                }
                for entry in instances
            ],
        }
        return self._text_result(json.dumps(payload, indent=2, ensure_ascii=False))

    async def tool_search_music_assistant(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Search Music Assistant content."""
        name = str(args.get("name") or "").strip()
        if not name:
            return self._text_result("❌ Error: name is required for search_music_assistant.")

        try:
            config_entry = self._resolve_music_assistant_instance(
                config_entry_id=args.get("config_entry_id"),
                instance=args.get("instance"),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        service_data: Dict[str, Any] = {
            "config_entry_id": config_entry.entry_id,
            "name": name,
            "limit": self._coerce_int_arg(
                args.get("limit"), default=10, minimum=1, maximum=50
            ),
        }
        for key in ("artist", "album"):
            value = args.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                service_data[key] = value
        media_type = args.get("media_type")
        normalized_media_type = self._normalize_music_assistant_media_type_filter(media_type)
        if media_type is not None and normalized_media_type is None:
            return self._text_result(
                f"❌ Error: media_type must be {MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_TEXT}, "
                "or a list of those values."
            )
        if normalized_media_type not in (None, [], ""):
            service_data["media_type"] = normalized_media_type
        if isinstance(args.get("library_only"), bool):
            service_data["library_only"] = args["library_only"]

        return await self._call_music_assistant_response_service(
            service="search",
            service_data=service_data,
            summary_label=f"Music Assistant search for {name}",
        )

    async def tool_get_music_assistant_library(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Browse the Music Assistant library."""
        media_type = self._normalize_music_assistant_media_type_value(args.get("media_type"))
        if media_type is None:
            return self._text_result(
                f"❌ Error: media_type must be one of {MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_TEXT}."
            )

        try:
            config_entry = self._resolve_music_assistant_instance(
                config_entry_id=args.get("config_entry_id"),
                instance=args.get("instance"),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        service_data: Dict[str, Any] = {
            "config_entry_id": config_entry.entry_id,
            "media_type": media_type,
            "limit": self._coerce_int_arg(
                args.get("limit"), default=25, minimum=1, maximum=100
            ),
            "offset": self._coerce_int_arg(
                args.get("offset"), default=0, minimum=0, maximum=10000
            ),
        }
        for key in ("search", "order_by", "album_type"):
            value = args.get(key)
            if isinstance(value, str):
                value = value.strip()
            if value not in (None, ""):
                service_data[key] = value
        for key in ("favorite", "album_artists_only"):
            if isinstance(args.get(key), bool):
                service_data[key] = args[key]

        return await self._call_music_assistant_response_service(
            service="get_library",
            service_data=service_data,
            summary_label=f"Music Assistant {media_type} library",
        )

    async def tool_get_music_assistant_queue(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read Music Assistant queue details for target players."""
        if not self.hass.services.has_service("music_assistant", "get_queue"):
            return self._text_result(
                "The Home Assistant Music Assistant integration is not available or does not expose music_assistant.get_queue."
            )

        try:
            resolved_player_ids, resolution_text = await self._resolve_music_assistant_player_targets(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                media_player=args.get("media_player"),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        result = await self._call_music_assistant_response_service(
            service="get_queue",
            service_data={"entity_id": resolved_player_ids},
            summary_label="Music Assistant queue",
        )

        if resolution_text and result.get("content"):
            result["content"][0]["text"] += f"\n\n{resolution_text}"

        return result

    async def tool_control_music_assistant_player(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Control playback, queue settings, or volume for Music Assistant players."""
        action = str(args.get("action") or "").strip().lower()
        if action not in {
            "pause",
            "resume",
            "toggle",
            "stop",
            "next",
            "previous",
            "seek",
            "skip",
            "set_volume",
            "volume_up",
            "volume_down",
            "mute",
            "shuffle",
            "repeat",
            "clear_queue",
        }:
            return self._text_result(
                "❌ Error: action must be one of pause, resume, toggle, stop, next, previous, seek, skip, set_volume, volume_up, volume_down, mute, shuffle, repeat, or clear_queue."
            )

        try:
            resolved_player_ids, resolution_text = await self._resolve_music_assistant_player_targets(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                media_player=args.get("media_player"),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        service = {
            "pause": "media_pause",
            "resume": "media_play",
            "toggle": "media_play_pause",
            "stop": "media_stop",
            "next": "media_next_track",
            "previous": "media_previous_track",
            "volume_up": "volume_up",
            "volume_down": "volume_down",
            "clear_queue": "clear_playlist",
        }.get(action)

        service_data: Dict[str, Any] = {}
        detail_text: str | None = None

        if action == "seek":
            if args.get("position") is None:
                return self._text_result("❌ Error: position is required for seek.")
            service = "media_seek"
            service_data["seek_position"] = self._coerce_int_arg(
                args.get("position"),
                default=0,
                minimum=0,
                maximum=24 * 60 * 60,
            )
            detail_text = f"Position: {service_data['seek_position']} seconds."
        elif action == "skip":
            service = "media_seek"
            seconds = self._coerce_int_arg(
                args.get("seconds"),
                default=10,
                minimum=-(24 * 60 * 60),
                maximum=24 * 60 * 60,
            )
            return await self._skip_music_assistant_players(
                resolved_player_ids=resolved_player_ids,
                seconds=seconds,
                resolution_text=resolution_text,
            )
        elif action == "set_volume":
            if args.get("volume_level") is None:
                return self._text_result(
                    "❌ Error: volume_level is required for set_volume."
                )
            service = "volume_set"
            volume_percent = self._coerce_int_arg(
                args.get("volume_level"),
                default=0,
                minimum=0,
                maximum=100,
            )
            service_data["volume_level"] = volume_percent / 100
            detail_text = f"Volume: {volume_percent}%."
        elif action == "mute":
            if not isinstance(args.get("muted"), bool):
                return self._text_result("❌ Error: muted is required for mute.")
            service = "volume_mute"
            service_data["is_volume_muted"] = args["muted"]
            detail_text = f"Muted: {'yes' if args['muted'] else 'no'}."
        elif action == "shuffle":
            if not isinstance(args.get("shuffle"), bool):
                return self._text_result("❌ Error: shuffle is required for shuffle.")
            service = "shuffle_set"
            service_data["shuffle"] = args["shuffle"]
            detail_text = f"Shuffle: {'on' if args['shuffle'] else 'off'}."
        elif action == "repeat":
            repeat = str(args.get("repeat") or "").strip().lower()
            if repeat not in {"off", "one", "all"}:
                return self._text_result(
                    "❌ Error: repeat is required for repeat and must be off, one, or all."
                )
            service = "repeat_set"
            service_data["repeat"] = repeat
            detail_text = f"Repeat: {repeat}."

        if service is None:
            return self._text_result(
                f"❌ Error: Unsupported Music Assistant control action: {action}."
            )

        return await self._call_music_assistant_media_player_service(
            service=service,
            service_data=service_data,
            action_label=action,
            resolved_player_ids=resolved_player_ids,
            resolution_text=resolution_text,
            detail_text=detail_text,
        )

    async def tool_transfer_music_assistant_queue(
        self,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer a Music Assistant queue between two players."""
        if not self.hass.services.has_service("music_assistant", "transfer_queue"):
            return self._text_result(
                "The Home Assistant Music Assistant integration is not available or does not expose music_assistant.transfer_queue."
            )

        source_values = self._normalize_target_values(args.get("source_media_player"))
        if not any(
            self._normalize_target_values(args.get(key))
            for key in ("area", "floor", "label", "media_player")
        ):
            return self._text_result(
                "❌ Error: target area, floor, label, or media_player is required for transfer_music_assistant_queue."
            )

        try:
            if source_values:
                source_player_ids, _source_resolution_text = await self._resolve_music_assistant_player_targets(
                    media_player=args.get("source_media_player"),
                )
            else:
                source_player_ids = []
            target_player_ids, target_resolution_text = await self._resolve_music_assistant_player_targets(
                area=args.get("area"),
                floor=args.get("floor"),
                label=args.get("label"),
                media_player=args.get("media_player"),
            )
        except ValueError as err:
            return self._text_result(f"❌ Error: {err}")

        if source_player_ids and len(source_player_ids) != 1:
            return self._text_result(
                "❌ Error: source_media_player must resolve to exactly one Music Assistant player."
            )
        if len(target_player_ids) != 1:
            return self._text_result(
                "❌ Error: target selectors must resolve to exactly one Music Assistant player."
            )
        if source_player_ids and source_player_ids[0] == target_player_ids[0]:
            return self._text_result(
                "❌ Error: source and target Music Assistant players must be different."
            )

        service_data: Dict[str, Any] = {
            "entity_id": target_player_ids[0],
        }
        if source_player_ids:
            service_data["source_player"] = source_player_ids[0]
        if isinstance(args.get("auto_play"), bool):
            service_data["auto_play"] = args["auto_play"]

        self._publish_progress(
            "tool_start",
            "Transferring Music Assistant queue",
            tool="transfer_music_assistant_queue",
        )

        try:
            await self.hass.services.async_call(
                domain="music_assistant",
                service="transfer_queue",
                service_data=service_data,
                blocking=True,
                return_response=False,
            )
        except Exception as err:
            error_msg = f"Music Assistant queue transfer failed: {err}"
            _LOGGER.exception(error_msg)
            return self._text_result(f"❌ Error: {error_msg}")

        self._publish_progress(
            "tool_complete",
            "Music Assistant queue transferred",
            tool="transfer_music_assistant_queue",
            success=True,
        )

        target_name = self._friendly_names_for_entities(target_player_ids)[0]
        if source_player_ids:
            source_name = self._friendly_names_for_entities(source_player_ids)[0]
            transfer_text = (
                f"✅ Transferred Music Assistant queue from {source_name} to {target_name}."
            )
        else:
            transfer_text = f"✅ Transferred active Music Assistant queue to {target_name}."
        text_parts = [
            transfer_text,
            target_resolution_text,
        ]
        if isinstance(args.get("auto_play"), bool):
            text_parts.append(f"Auto play: {'yes' if args['auto_play'] else 'no'}.")
        return self._text_result("\n".join(text_parts))

    def _publish_progress(self, stage: str, message: str, **payload: Any) -> None:
        """Forward progress updates to the shared MCP server when available."""
        server = self.hass.data.get(DOMAIN, {}).get("shared_mcp_server")
        publish_progress = getattr(server, "publish_progress", None) if server else None
        if callable(publish_progress):
            publish_progress(stage, message, **payload)

    @staticmethod
    def _text_result(
        text: str,
        *,
        is_error: bool = False,
        response: Any | None = None,
    ) -> Dict[str, Any]:
        """Build a standard text MCP result."""
        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }
        if response is not None:
            result["response"] = response
        return result

    def _get_music_assistant_instances(self) -> List[Any]:
        """Return configured Music Assistant config entries."""
        return list(self.hass.config_entries.async_entries("music_assistant"))

    def _resolve_music_assistant_instance(
        self,
        *,
        config_entry_id: Any = None,
        instance: Any = None,
    ) -> Any:
        """Resolve a single Music Assistant config entry."""
        instances = self._get_music_assistant_instances()
        if not instances:
            raise ValueError("No Music Assistant instances are configured in Home Assistant.")

        config_entry_id_text = str(config_entry_id or "").strip()
        instance_text = str(instance or "").strip().casefold()

        matched = instances
        if config_entry_id_text:
            matched = [
                entry for entry in matched if entry.entry_id == config_entry_id_text
            ]
            if not matched:
                raise ValueError(
                    f"No Music Assistant instance matched config_entry_id '{config_entry_id_text}'."
                )

        if instance_text:
            exact_matches = [
                entry for entry in matched if entry.title.casefold() == instance_text
            ]
            if exact_matches:
                matched = exact_matches
            else:
                partial_matches = [
                    entry for entry in matched if instance_text in entry.title.casefold()
                ]
                if partial_matches:
                    matched = partial_matches
                else:
                    raise ValueError(
                        f"No Music Assistant instance matched '{instance}'. Use list_music_assistant_instances first."
                    )

        if len(matched) == 1:
            return matched[0]

        raise ValueError(
            "Multiple Music Assistant instances are configured. Use list_music_assistant_instances and pass config_entry_id or instance."
        )

    def _is_music_assistant_entity(self, entity_entry: Any) -> bool:
        """Return whether an entity registry entry belongs to Music Assistant."""
        if entity_entry is None:
            return False

        platform = str(getattr(entity_entry, "platform", "") or "").strip()
        if platform == "music_assistant":
            return True

        config_entry_id = getattr(entity_entry, "config_entry_id", None)
        if config_entry_id:
            config_entry = self.hass.config_entries.async_get_entry(config_entry_id)
            if config_entry and config_entry.domain == "music_assistant":
                return True

        return False

    def _get_music_assistant_player_catalog(self) -> List[Dict[str, Any]]:
        """Build a catalog of exposed Music Assistant players."""
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        catalog: List[Dict[str, Any]] = []
        for state_obj in self.hass.states.async_all():
            if state_obj.domain != "media_player":
                continue
            if not async_should_expose(self.hass, "conversation", state_obj.entity_id):
                continue

            entity_entry = entity_registry.async_get(state_obj.entity_id)
            if not self._is_music_assistant_entity(entity_entry):
                continue

            entity_context = self.discovery._get_entity_context(
                entity_entry,
                device_registry,
                area_registry,
                floor_registry,
                label_registry,
            )
            entity_info = self.discovery._create_entity_info(
                state_obj,
                entity_entry=entity_entry,
                entity_context=entity_context,
            )

            config_entry_id = getattr(entity_entry, "config_entry_id", None)
            config_entry = (
                self.hass.config_entries.async_get_entry(config_entry_id)
                if config_entry_id
                else None
            )
            if config_entry_id:
                entity_info["config_entry_id"] = config_entry_id
            if config_entry:
                entity_info["instance_title"] = config_entry.title
            entity_info["integration"] = "music_assistant"
            for attr in (
                "media_title",
                "media_artist",
                "media_album_name",
                "source",
                "volume_level",
                "is_volume_muted",
            ):
                if attr in state_obj.attributes:
                    entity_info[attr] = self._serialize_service_response_value(
                        state_obj.attributes.get(attr)
                    )

            catalog.append(
                {
                    "entity_id": state_obj.entity_id,
                    "state_obj": state_obj,
                    "entity_entry": entity_entry,
                    "entity_context": entity_context,
                    "entity_info": entity_info,
                }
            )

        catalog.sort(
            key=lambda record: (
                record["entity_info"].get("name", "").casefold(),
                record["entity_id"],
            )
        )
        return catalog

    def _resolve_music_assistant_area_values(
        self,
        area_values: List[str],
        area_registry: Any,
        floor_registry: Any,
    ) -> Tuple[set[str], set[str]]:
        """Resolve Music Assistant area selectors, with floor fallback."""
        area_ids: set[str] = set()
        floor_ids: set[str] = set()

        for value in area_values:
            area_entry = self.discovery._resolve_area_entry(value, area_registry)
            if area_entry is not None:
                area_ids.add(area_entry.id)
                continue

            floor_entry = self.discovery._resolve_floor_entry(value, floor_registry)
            if floor_entry is not None:
                floor_ids.add(floor_entry.floor_id)
                continue

            raise ValueError(
                f"No Home Assistant area or floor matched '{value}' for Music Assistant targeting."
            )

        return area_ids, floor_ids

    def _match_music_assistant_player_term(
        self,
        catalog: List[Dict[str, Any]],
        search_term: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Resolve a player selector term to the strongest matching players."""
        normalized_term = str(search_term or "").strip().casefold()
        if not normalized_term:
            return [], []

        scored_matches: List[Tuple[int, Dict[str, Any], List[str]]] = []
        for record in catalog:
            score, reasons = self.discovery._get_entity_match_details(
                normalized_term,
                record["state_obj"],
                record["entity_entry"],
                record["entity_context"],
            )
            if score > 0:
                scored_matches.append((score, record, reasons))

        if not scored_matches:
            raise ValueError(
                f"No Music Assistant player matched '{search_term}'. Use list_music_assistant_players first."
            )

        best_score = max(score for score, _, _ in scored_matches)
        best_matches = [
            {
                **record,
                "match_score": score,
                "match_reasons": reasons,
            }
            for score, record, reasons in scored_matches
            if score == best_score
        ]

        if len(best_matches) > 5:
            raise ValueError(
                f"Music Assistant player selector '{search_term}' is too broad. Use list_music_assistant_players to narrow it down."
            )

        return best_matches, [match["entity_info"]["name"] for match in best_matches]

    async def _discover_music_assistant_players(
        self,
        *,
        area: Any = None,
        floor: Any = None,
        label: Any = None,
        name_contains: Any = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Discover Music Assistant players with alias-aware filters."""
        catalog = self._get_music_assistant_player_catalog()
        if not catalog:
            return []

        area_values = self._normalize_target_values(area)
        floor_values = self._normalize_target_values(floor)
        label_values = self._normalize_target_values(label)

        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        area_ids: set[str] = set()
        floor_ids: set[str] = set()
        if area_values:
            resolved_area_ids, resolved_floor_ids = self._resolve_music_assistant_area_values(
                area_values,
                area_registry,
                floor_registry,
            )
            area_ids.update(resolved_area_ids)
            floor_ids.update(resolved_floor_ids)

        for value in floor_values:
            floor_entry = self.discovery._resolve_floor_entry(value, floor_registry)
            if floor_entry is None:
                raise ValueError(
                    f"No Home Assistant floor matched '{value}' for Music Assistant filtering."
                )
            floor_ids.add(floor_entry.floor_id)

        label_ids: set[str] = set()
        for value in label_values:
            label_entry = self.discovery._resolve_label_entry(value, label_registry)
            if label_entry is None:
                raise ValueError(
                    f"No Home Assistant label matched '{value}' for Music Assistant filtering."
                )
            label_ids.add(label_entry.label_id)

        filtered_records = []
        for record in catalog:
            info = record["entity_info"]
            if area_ids and info.get("area_id") not in area_ids:
                continue
            if floor_ids and info.get("floor_id") not in floor_ids:
                continue
            if label_ids and not label_ids.intersection(set(info.get("label_ids", []))):
                continue
            filtered_records.append(record)

        search_term = str(name_contains or "").strip().casefold()
        if search_term:
            scored_records = []
            for record in filtered_records:
                score, reasons = self.discovery._get_entity_match_details(
                    search_term,
                    record["state_obj"],
                    record["entity_entry"],
                    record["entity_context"],
                )
                if score <= 0:
                    continue
                entity_info = dict(record["entity_info"])
                entity_info["match_score"] = score
                entity_info["match_reasons"] = reasons
                scored_records.append((score, entity_info))

            scored_records.sort(
                key=lambda item: (-item[0], item[1].get("name", "").casefold(), item[1]["entity_id"])
            )
            return [entity_info for _, entity_info in scored_records[:limit]]

        return [dict(record["entity_info"]) for record in filtered_records[:limit]]

    async def _resolve_music_assistant_player_targets(
        self,
        *,
        area: Any = None,
        floor: Any = None,
        label: Any = None,
        media_player: Any = None,
    ) -> Tuple[List[str], str]:
        """Resolve selectors to concrete Music Assistant player entity IDs."""
        catalog = self._get_music_assistant_player_catalog()
        if not catalog:
            raise ValueError("No exposed Music Assistant players are available.")

        area_values = self._normalize_target_values(area)
        floor_values = self._normalize_target_values(floor)
        label_values = self._normalize_target_values(label)
        media_player_values = self._normalize_target_values(media_player)

        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass) if fr else None
        label_registry = lr.async_get(self.hass) if lr else None

        selector_sets: List[set[str]] = []

        if area_values:
            area_ids, floor_ids = self._resolve_music_assistant_area_values(
                area_values,
                area_registry,
                floor_registry,
            )
            matched = {
                record["entity_id"]
                for record in catalog
                if (
                    (area_ids and record["entity_info"].get("area_id") in area_ids)
                    or (floor_ids and record["entity_info"].get("floor_id") in floor_ids)
                )
            }
            if not matched:
                raise ValueError(
                    f"No Music Assistant players matched area selector(s): {', '.join(area_values)}"
                )
            selector_sets.append(matched)

        if floor_values:
            floor_ids = set()
            for value in floor_values:
                floor_entry = self.discovery._resolve_floor_entry(value, floor_registry)
                if floor_entry is None:
                    raise ValueError(
                        f"No Home Assistant floor matched '{value}' for Music Assistant targeting."
                    )
                floor_ids.add(floor_entry.floor_id)

            matched = {
                record["entity_id"]
                for record in catalog
                if record["entity_info"].get("floor_id") in floor_ids
            }
            if not matched:
                raise ValueError(
                    f"No Music Assistant players matched floor selector(s): {', '.join(floor_values)}"
                )
            selector_sets.append(matched)

        if label_values:
            label_ids = set()
            for value in label_values:
                label_entry = self.discovery._resolve_label_entry(value, label_registry)
                if label_entry is None:
                    raise ValueError(
                        f"No Home Assistant label matched '{value}' for Music Assistant targeting."
                    )
                label_ids.add(label_entry.label_id)

            matched = {
                record["entity_id"]
                for record in catalog
                if label_ids.intersection(set(record["entity_info"].get("label_ids", [])))
            }
            if not matched:
                raise ValueError(
                    f"No Music Assistant players matched label selector(s): {', '.join(label_values)}"
                )
            selector_sets.append(matched)

        if media_player_values:
            matched_entity_ids: set[str] = set()
            for value in media_player_values:
                matches, _match_names = self._match_music_assistant_player_term(catalog, value)
                matched_entity_ids.update(match["entity_id"] for match in matches)
            selector_sets.append(matched_entity_ids)

        if selector_sets:
            resolved_entity_ids = sorted(set.intersection(*selector_sets))
            if not resolved_entity_ids:
                raise ValueError(
                    "No Music Assistant players matched the combined selectors."
                )
        else:
            if len(catalog) == 1:
                resolved_entity_ids = [catalog[0]["entity_id"]]
            else:
                raise ValueError(
                    "Music Assistant playback needs a target player, area, floor, or label when multiple Music Assistant players are exposed."
                )

        return (
            resolved_entity_ids,
            "Resolved Music Assistant players: "
            + ", ".join(self._friendly_names_for_entities(resolved_entity_ids)),
        )

    @staticmethod
    def _normalize_music_assistant_media_id(value: Any) -> Any:
        """Normalize Music Assistant media_id input, including semicolon lists."""
        if value is None:
            return None

        if isinstance(value, (list, tuple, set)):
            normalized_values = []
            for item in value:
                item_text = str(item).strip()
                if item_text:
                    normalized_values.append(item_text)
            if not normalized_values:
                return None
            return normalized_values if len(normalized_values) > 1 else normalized_values[0]

        value_text = str(value).strip()
        if not value_text:
            return None

        if ";" in value_text:
            parts = [part.strip() for part in value_text.split(";") if part.strip()]
            if parts:
                return parts if len(parts) > 1 else parts[0]

        return value_text

    @staticmethod
    def _normalize_music_assistant_media_type_filter(value: Any) -> Any:
        """Normalize Music Assistant media_type filters."""
        if value is None:
            return None

        if isinstance(value, (list, tuple, set)):
            normalized = []
            for item in value:
                item_text = MusicAssistantTool._normalize_music_assistant_media_type_value(item)
                if item_text is not None and item_text not in normalized:
                    normalized.append(item_text)
            return normalized or None

        value_text = str(value).strip().lower()
        if not value_text:
            return None
        if "," in value_text:
            parts = [
                part.strip()
                for part in value_text.replace(";", ",").split(",")
                if part.strip()
            ]
            normalized = []
            for part in parts:
                item_text = MusicAssistantTool._normalize_music_assistant_media_type_value(part)
                if item_text is not None and item_text not in normalized:
                    normalized.append(item_text)
            return normalized or None
        return MusicAssistantTool._normalize_music_assistant_media_type_value(value_text)

    @staticmethod
    def _normalize_music_assistant_media_type_value(value: Any) -> str | None:
        """Normalize one Music Assistant library/search media type value."""
        value_text = str(value).strip().lower()
        if not value_text:
            return None
        value_text = MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_ALIASES.get(value_text, value_text)
        return value_text if value_text in MUSIC_ASSISTANT_LIBRARY_MEDIA_TYPE_SET else None

    async def _call_music_assistant_response_service(
        self,
        *,
        service: str,
        service_data: Dict[str, Any],
        summary_label: str,
    ) -> Dict[str, Any]:
        """Call a Music Assistant response service and format the result."""
        if not self.hass.services.has_service("music_assistant", service):
            return self._text_result(
                f"The Home Assistant Music Assistant integration does not expose music_assistant.{service}."
            )

        self._publish_progress(
            "tool_start",
            f"Calling Music Assistant {service}",
            tool=f"music_assistant.{service}",
        )

        try:
            response = await self.hass.services.async_call(
                domain="music_assistant",
                service=service,
                service_data=service_data,
                blocking=True,
                return_response=True,
            )
        except Exception as err:
            error_msg = f"Music Assistant {service} failed: {err}"
            _LOGGER.exception(error_msg)
            return self._text_result(f"❌ Error: {error_msg}")

        self._publish_progress(
            "tool_complete",
            f"Music Assistant {service} completed",
            tool=f"music_assistant.{service}",
            success=True,
        )

        serialized_response = self._serialize_service_response_value(response)
        text_parts = [f"✅ Retrieved {summary_label}."]
        summary_lines = summarize_music_assistant_response(service, serialized_response)
        if summary_lines:
            text_parts.append("")
            text_parts.extend(summary_lines)
        text_parts.append("")
        text_parts.append("Response:")
        text_parts.append(json.dumps(serialized_response, indent=2, ensure_ascii=False))

        return self._text_result(
            "\n".join(text_parts),
            response=serialized_response,
        )

    async def _call_music_assistant_media_player_service(
        self,
        *,
        service: str,
        service_data: Dict[str, Any],
        action_label: str,
        resolved_player_ids: List[str],
        resolution_text: str,
        detail_text: str | None = None,
    ) -> Dict[str, Any]:
        """Call a media_player service for resolved Music Assistant players."""
        if not self.hass.services.has_service("media_player", service):
            return self._text_result(
                f"Home Assistant does not expose media_player.{service}."
            )

        self._publish_progress(
            "tool_start",
            f"Running Music Assistant {action_label}",
            tool="control_music_assistant_player",
            action=action_label,
            target_count=len(resolved_player_ids),
        )

        try:
            await self.hass.services.async_call(
                domain="media_player",
                service=service,
                service_data={**service_data, "entity_id": resolved_player_ids},
                blocking=True,
                return_response=False,
            )
        except Exception as err:
            error_msg = f"Music Assistant {action_label} failed: {err}"
            _LOGGER.exception(error_msg)
            return self._text_result(f"❌ Error: {error_msg}")

        self._publish_progress(
            "tool_complete",
            f"Music Assistant {action_label} completed",
            tool="control_music_assistant_player",
            action=action_label,
            success=True,
            target_count=len(resolved_player_ids),
        )

        target_text = ", ".join(self._friendly_names_for_entities(resolved_player_ids))
        text_parts = [
            f"✅ Completed Music Assistant {action_label.replace('_', ' ')} for {target_text}."
        ]
        if detail_text:
            text_parts.append(detail_text)
        if resolution_text:
            text_parts.append(resolution_text)
        return self._text_result("\n".join(text_parts))

    async def _skip_music_assistant_players(
        self,
        *,
        resolved_player_ids: List[str],
        seconds: int,
        resolution_text: str,
    ) -> Dict[str, Any]:
        """Seek each Music Assistant player relative to its current position."""
        if not self.hass.services.has_service("media_player", "media_seek"):
            return self._text_result("Home Assistant does not expose media_player.media_seek.")

        seek_positions: Dict[str, int] = {}
        for entity_id in resolved_player_ids:
            state_obj = self.hass.states.get(entity_id)
            current_position = self._current_music_assistant_position_seconds(state_obj)
            if current_position is None:
                return self._text_result(
                    f"❌ Error: {entity_id} does not expose a current media_position for skip."
                )
            seek_positions[entity_id] = max(0, int(current_position) + seconds)

        self._publish_progress(
            "tool_start",
            "Skipping Music Assistant playback",
            tool="control_music_assistant_player",
            action="skip",
            target_count=len(resolved_player_ids),
        )

        try:
            for entity_id, seek_position in seek_positions.items():
                await self.hass.services.async_call(
                    domain="media_player",
                    service="media_seek",
                    service_data={
                        "entity_id": entity_id,
                        "seek_position": seek_position,
                    },
                    blocking=True,
                    return_response=False,
                )
        except Exception as err:
            error_msg = f"Music Assistant skip failed: {err}"
            _LOGGER.exception(error_msg)
            return self._text_result(f"❌ Error: {error_msg}")

        self._publish_progress(
            "tool_complete",
            "Music Assistant skip completed",
            tool="control_music_assistant_player",
            action="skip",
            success=True,
            target_count=len(resolved_player_ids),
        )

        target_text = ", ".join(self._friendly_names_for_entities(resolved_player_ids))
        text_parts = [
            f"✅ Skipped Music Assistant playback by {seconds} second{'s' if seconds != 1 else ''} for {target_text}."
        ]
        if resolution_text:
            text_parts.append(resolution_text)
        return self._text_result("\n".join(text_parts))

    @staticmethod
    def _current_music_assistant_position_seconds(state_obj: Any) -> int | None:
        """Return the current media position, accounting for elapsed play time."""
        if state_obj is None:
            return None

        raw_position = state_obj.attributes.get("media_position")
        if raw_position is None:
            return None
        try:
            position = float(raw_position)
        except (TypeError, ValueError):
            return None

        updated_at = state_obj.attributes.get("media_position_updated_at")
        if (
            state_obj.state == "playing"
            and isinstance(updated_at, datetime)
        ):
            now = datetime.now(updated_at.tzinfo) if updated_at.tzinfo else datetime.now()
            position += max(0, (now - updated_at).total_seconds())

        return max(0, int(position))

    def _friendly_names_for_entities(self, entity_ids: List[str]) -> List[str]:
        """Resolve entity IDs to friendly names."""
        names = []
        for entity_id in entity_ids:
            state = self.hass.states.get(entity_id)
            if state and state.name:
                names.append(state.name)
            else:
                names.append(entity_id)
        return names

    def _serialize_service_response_value(self, value: Any) -> Any:
        """Serialize HA service response data to JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): self._serialize_service_response_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_service_response_value(item) for item in value]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return str(value)

    @staticmethod
    def _normalize_target_values(value: Any) -> List[str]:
        """Normalize scalar or list selector values to unique strings."""
        if value is None:
            return []

        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        normalized = []
        seen = set()
        for item in raw_values:
            if item is None:
                continue
            item_text = str(item).strip()
            if not item_text or item_text in seen:
                continue
            seen.add(item_text)
            normalized.append(item_text)

        return normalized

    @staticmethod
    def _coerce_int_arg(
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        """Coerce an integer-like tool argument safely."""
        if value is None:
            parsed = default
        elif isinstance(value, bool):
            parsed = default
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(value)
        else:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                parsed = default

        return max(minimum, min(parsed, maximum))
