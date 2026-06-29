"""Tests for the built-in Google Places and Routes tools."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.const import (
    CONF_GOOGLE_MAPS_API_KEY,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
)
from custom_components.mcp_assist.custom_tools import google_maps as google_maps_module


class _FakeGoogleResponse:
    """Minimal async response for Google Maps tests."""

    def __init__(
        self,
        *,
        status: int = 200,
        json_data: dict | None = None,
        text: str = "",
    ) -> None:
        self.status = status
        self._json_data = json_data or {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self, *args, **kwargs):
        return self._json_data

    async def text(self) -> str:
        return self._text


class _FakeGoogleSession:
    """Minimal async session that records Google Maps requests."""

    def __init__(self, response: _FakeGoogleResponse, captured: dict) -> None:
        self._response = response
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url, **kwargs):
        self._captured.setdefault("calls", []).append(("POST", url, kwargs))
        return self._response

    def get(self, url, **kwargs):
        self._captured.setdefault("calls", []).append(("GET", url, kwargs))
        return self._response


def _google_client_session(response: _FakeGoogleResponse, captured: dict):
    """Build a fake aiohttp ClientSession factory."""

    def _client_session(**kwargs):
        captured.setdefault("sessions", []).append(kwargs)
        return _FakeGoogleSession(response, captured)

    return _client_session


def _set_home_coordinates(hass) -> None:
    """Set deterministic Home Assistant home coordinates."""
    hass.config.latitude = 47.6205
    hass.config.longitude = -122.3493


def test_google_maps_tool_definitions_include_places_and_routes(hass) -> None:
    """Google Maps tools should advertise Places and Routes surfaces."""
    definitions = {
        definition["name"]: definition
        for definition in google_maps_module.GoogleMapsTool(hass).get_tool_definitions()
    }

    assert set(definitions) == {
        "search_google_places",
        "get_google_place_details",
        "get_google_route",
    }
    assert "open" in definitions["search_google_places"]["preferred_when"]
    assert "traffic-aware" in definitions["get_google_route"]["description"]


@pytest.mark.asyncio
async def test_search_google_places_returns_place_details(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Places Text Search should send field masks and normalize place results."""
    _set_home_coordinates(hass)
    system_entry_factory(
        data={
            CONF_GOOGLE_MAPS_API_KEY: "maps-key",
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: True,
        }
    )
    captured = {}
    monkeypatch.setattr(
        google_maps_module.aiohttp,
        "ClientSession",
        _google_client_session(
            _FakeGoogleResponse(
                json_data={
                    "places": [
                        {
                            "id": "places/abc123",
                            "displayName": {"text": "Coffee House"},
                            "formattedAddress": "1 Pike St, Seattle, WA",
                            "nationalPhoneNumber": "(206) 555-0100",
                            "rating": 4.5,
                            "userRatingCount": 321,
                            "businessStatus": "OPERATIONAL",
                            "currentOpeningHours": {
                                "openNow": True,
                                "weekdayDescriptions": ["Monday: 7 AM-5 PM"],
                            },
                            "googleMapsUri": "https://maps.google.com/?cid=1",
                            "websiteUri": "https://example.com",
                            "location": {"latitude": 47.6, "longitude": -122.3},
                        }
                    ]
                }
            ),
            captured,
        ),
    )

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "search_google_places",
        {"query": "coffee near me", "limit": 50},
    )

    assert result["isError"] is False
    session_headers = captured["sessions"][0]["headers"]
    assert session_headers["X-Goog-Api-Key"] == "maps-key"
    assert "places.currentOpeningHours" in session_headers["X-Goog-FieldMask"]
    method, url, kwargs = captured["calls"][0]
    assert method == "POST"
    assert url == google_maps_module.GOOGLE_PLACES_TEXT_SEARCH_URL
    assert kwargs["json"]["pageSize"] == 10
    assert kwargs["json"]["locationBias"]["circle"]["center"] == {
        "latitude": 47.6205,
        "longitude": -122.3493,
    }
    place = result["structuredContent"]["places"][0]
    assert place["name"] == "Coffee House"
    assert place["open_now"] is True
    assert place["phone"] == "(206) 555-0100"
    assert "Coffee House" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_search_google_places_omits_home_bias_when_location_context_disabled(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Places should not send home coordinates unless tool-call context allows it."""
    _set_home_coordinates(hass)
    system_entry_factory(data={CONF_GOOGLE_MAPS_API_KEY: "maps-key"})
    captured = {}
    monkeypatch.setattr(
        google_maps_module.aiohttp,
        "ClientSession",
        _google_client_session(_FakeGoogleResponse(json_data={"places": []}), captured),
    )

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "search_google_places",
        {"query": "coffee"},
    )

    assert result["isError"] is False
    assert "locationBias" not in captured["calls"][0][2]["json"]


@pytest.mark.asyncio
async def test_get_google_place_details_accepts_prefixed_place_id(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Place Details should accept resource-prefixed place IDs."""
    system_entry_factory(data={CONF_GOOGLE_MAPS_API_KEY: "maps-key"})
    captured = {}
    monkeypatch.setattr(
        google_maps_module.aiohttp,
        "ClientSession",
        _google_client_session(
            _FakeGoogleResponse(
                json_data={
                    "id": "places/abc123",
                    "displayName": {"text": "Coffee House"},
                    "formattedAddress": "1 Pike St, Seattle, WA",
                }
            ),
            captured,
        ),
    )

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "get_google_place_details",
        {"place_id": "places/abc123"},
    )

    assert result["isError"] is False
    assert captured["calls"][0][0] == "GET"
    assert captured["calls"][0][1] == "https://places.googleapis.com/v1/places/abc123"
    assert result["structuredContent"]["place"]["place_id"] == "places/abc123"


@pytest.mark.asyncio
async def test_get_google_route_defaults_origin_to_home_and_uses_traffic(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routes should default origin to HA home and request traffic-aware driving."""
    _set_home_coordinates(hass)
    system_entry_factory(
        data={
            CONF_GOOGLE_MAPS_API_KEY: "maps-key",
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: True,
        }
    )
    captured = {}
    monkeypatch.setattr(
        google_maps_module.aiohttp,
        "ClientSession",
        _google_client_session(
            _FakeGoogleResponse(
                json_data={
                    "routes": [
                        {
                            "duration": "1800s",
                            "staticDuration": "1500s",
                            "distanceMeters": 21000,
                            "localizedValues": {
                                "duration": {"text": "30 min"},
                                "staticDuration": {"text": "25 min"},
                                "distance": {"text": "21 km"},
                            },
                            "description": "I-5 S",
                            "warnings": ["Tolls on route"],
                        }
                    ]
                }
            ),
            captured,
        ),
    )

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "get_google_route",
        {
            "destination": "Seattle-Tacoma International Airport",
            "departure_time": "now",
            "avoid_tolls": True,
        },
    )

    assert result["isError"] is False
    method, url, kwargs = captured["calls"][0]
    assert method == "POST"
    assert url == google_maps_module.GOOGLE_ROUTES_URL
    body = kwargs["json"]
    assert body["origin"]["location"]["latLng"] == {
        "latitude": 47.6205,
        "longitude": -122.3493,
    }
    assert body["destination"] == {"address": "Seattle-Tacoma International Airport"}
    assert body["travelMode"] == "DRIVE"
    assert body["routingPreference"] == "TRAFFIC_AWARE"
    assert body["routeModifiers"]["avoidTolls"] is True
    assert "departureTime" not in body
    route = result["structuredContent"]["route"]
    assert route["duration"] == "30 min"
    assert route["traffic_delay_seconds"] == 300
    assert route["origin"] == "Home Assistant home"
    assert "Traffic delay" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_google_route_requires_origin_when_location_context_disabled(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routes should not use HA home coordinates unless tool-call context allows it."""
    _set_home_coordinates(hass)
    system_entry_factory(data={CONF_GOOGLE_MAPS_API_KEY: "maps-key"})

    def _client_session(**_kwargs):
        raise AssertionError("HTTP should not be called without an origin")

    monkeypatch.setattr(google_maps_module.aiohttp, "ClientSession", _client_session)

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "get_google_route",
        {"destination": "Seattle-Tacoma International Airport"},
    )

    assert result["isError"] is True
    assert "origin is required" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_google_route_rejects_drive_arrival_time(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-transit arrival-time requests would otherwise misrepresent leave-by."""
    _set_home_coordinates(hass)
    system_entry_factory(
        data={
            CONF_GOOGLE_MAPS_API_KEY: "maps-key",
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: True,
        }
    )

    def _client_session(**_kwargs):
        raise AssertionError("HTTP should not be called for unsupported arrival_time")

    monkeypatch.setattr(google_maps_module.aiohttp, "ClientSession", _client_session)

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "get_google_route",
        {
            "destination": "Dinner",
            "arrival_time": "2026-06-29T18:30:00-07:00",
        },
    )

    assert result["isError"] is True
    assert "arrival_time is only supported for transit" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_get_google_route_rejects_avoid_flags_for_non_drive_modes(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid modifiers should not be sent for route modes that do not support them."""
    system_entry_factory(data={CONF_GOOGLE_MAPS_API_KEY: "maps-key"})

    def _client_session(**_kwargs):
        raise AssertionError("HTTP should not be called for unsupported avoid flags")

    monkeypatch.setattr(google_maps_module.aiohttp, "ClientSession", _client_session)

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "get_google_route",
        {
            "origin": "Downtown Seattle",
            "destination": "Capitol Hill",
            "travel_mode": "transit",
            "avoid_tolls": True,
        },
    )

    assert result["isError"] is True
    assert "only supported for driving routes" in result["content"][0]["text"]


def test_google_route_waypoint_accepts_bare_place_ids(hass) -> None:
    """Search result place IDs should be directly routeable."""
    waypoint = google_maps_module.GoogleMapsTool(hass)._build_route_waypoint(
        "ChIJ1234567890abcdef"
    )

    assert waypoint == {"placeId": "ChIJ1234567890abcdef"}


@pytest.mark.asyncio
async def test_google_maps_tools_require_api_key(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google Maps tools should fail before HTTP when no API key is configured."""

    def _client_session(**_kwargs):
        raise AssertionError("HTTP should not be called without an API key")

    monkeypatch.setattr(google_maps_module.aiohttp, "ClientSession", _client_session)

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "search_google_places",
        {"query": "coffee"},
    )

    assert result["isError"] is True
    assert "API key is required" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_search_google_places_rejects_invalid_location_bias(
    hass,
    system_entry_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid location bias should fail before HTTP."""
    system_entry_factory(data={CONF_GOOGLE_MAPS_API_KEY: "maps-key"})

    def _client_session(**_kwargs):
        raise AssertionError("HTTP should not be called for invalid input")

    monkeypatch.setattr(google_maps_module.aiohttp, "ClientSession", _client_session)

    result = await google_maps_module.GoogleMapsTool(hass).handle_call(
        "search_google_places",
        {"query": "coffee", "location_bias": "Seattle"},
    )

    assert result["isError"] is True
    assert "location_bias must be" in result["content"][0]["text"]
