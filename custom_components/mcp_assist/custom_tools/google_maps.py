"""Google Places and Routes custom tools for MCP Assist."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any

import aiohttp
from homeassistant.util import dt as dt_util

from custom_components.mcp_assist import get_system_entry
from custom_components.mcp_assist.const import (
    CONF_GOOGLE_MAPS_API_KEY,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    DEFAULT_GOOGLE_MAPS_API_KEY,
    DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
)

_LOGGER = logging.getLogger(__name__)

GOOGLE_PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
GOOGLE_MAPS_USER_AGENT = (
    "ha-mcp-assist/1.0 (https://github.com/Moballo-LLC/ha-mcp-assist)"
)
GOOGLE_PLACES_FIELD_MASK = ",".join(
    (
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.nationalPhoneNumber",
        "places.internationalPhoneNumber",
        "places.rating",
        "places.userRatingCount",
        "places.businessStatus",
        "places.currentOpeningHours",
        "places.googleMapsUri",
        "places.websiteUri",
        "places.location",
    )
)
GOOGLE_PLACE_DETAILS_FIELD_MASK = ",".join(
    field.removeprefix("places.") for field in GOOGLE_PLACES_FIELD_MASK.split(",")
)
GOOGLE_ROUTES_FIELD_MASK = ",".join(
    (
        "routes.duration",
        "routes.staticDuration",
        "routes.distanceMeters",
        "routes.description",
        "routes.warnings",
        "routes.localizedValues",
        "routes.legs.duration",
        "routes.legs.staticDuration",
        "routes.legs.distanceMeters",
        "routes.legs.localizedValues",
    )
)

MAX_GOOGLE_PLACE_RESULTS = 10
_LAT_LNG_RE = re.compile(
    r"^\s*(?P<lat>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<lng>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


class GoogleMapsTool:
    """Search Google Places and calculate Google Routes."""

    def __init__(self, hass) -> None:
        """Initialize the Google Maps tool bundle."""
        self.hass = hass

    async def initialize(self) -> None:
        """Initialize the tool bundle."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles the requested tool."""
        return tool_name in {
            "search_google_places",
            "get_google_place_details",
            "get_google_route",
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions for Google Places and Routes."""
        return [
            {
                "name": "search_google_places",
                "description": (
                    "Search Google Places for businesses and points of interest. "
                    "Returns address, phone, rating, current open status, and links "
                    "when Google provides them."
                ),
                "llmDescription": "Search Google Places for place details.",
                "keywords": ["places", "business", "open now", "phone", "address", "rating"],
                "preferred_when": (
                    "Use when the user asks whether a place is open, where it is, "
                    "how to call it, or how it is rated."
                ),
                "returns": (
                    "Google Places results with place IDs, names, addresses, phone "
                    "numbers, ratings, open status, and map URLs."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Place, business, or point-of-interest search text.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of places to return (default 5, max 10).",
                            "minimum": 1,
                            "maximum": MAX_GOOGLE_PLACE_RESULTS,
                            "default": 5,
                        },
                        "location_bias": {
                            "type": "string",
                            "description": (
                                "Optional location bias as 'lat,lng'. If omitted, "
                                "Home Assistant home coordinates are used when available."
                            ),
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_google_place_details",
                "description": (
                    "Fetch Google Places details for a place ID returned by "
                    "search_google_places, including open status, phone, address, "
                    "rating, website, and Google Maps URL when available."
                ),
                "llmDescription": "Fetch details for a Google Place ID.",
                "keywords": ["place details", "open now", "phone", "address", "rating"],
                "preferred_when": (
                    "Use after search_google_places when the user wants details for a "
                    "specific result."
                ),
                "returns": "A Google Place details object with formatted text.",
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "place_id": {
                            "type": "string",
                            "description": "Google Places place ID, such as places/ChIJ... or ChIJ...",
                        }
                    },
                    "required": ["place_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_google_route",
                "description": (
                    "Calculate a Google Routes ETA and distance between an origin and "
                    "destination. Defaults origin to Home Assistant home coordinates "
                    "and uses traffic-aware driving routes."
                ),
                "llmDescription": "Calculate traffic-aware Google Routes travel time.",
                "keywords": ["route", "ETA", "traffic", "travel time", "directions"],
                "preferred_when": (
                    "Use for travel time, leave-by, commute, traffic-aware ETA, and "
                    "distance questions."
                ),
                "returns": (
                    "Route duration, static duration, traffic delay, distance, warnings, "
                    "and structured route metadata."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": (
                                "Destination address, place ID, or 'lat,lng'. Prefix place "
                                "IDs with 'place_id:' when ambiguous."
                            ),
                        },
                        "origin": {
                            "type": "string",
                            "description": (
                                "Optional origin address, place ID, or 'lat,lng'. May be "
                                "omitted for Home Assistant home only when home-location "
                                "tool sharing is enabled."
                            ),
                        },
                        "travel_mode": {
                            "type": "string",
                            "enum": ["drive", "walk", "bicycle", "transit"],
                            "description": "Travel mode. Defaults to drive.",
                            "default": "drive",
                        },
                        "departure_time": {
                            "type": "string",
                            "description": (
                                "Optional ISO timestamp for leaving. Use 'now' or omit for "
                                "current traffic-aware driving routes."
                            ),
                        },
                        "arrival_time": {
                            "type": "string",
                            "description": (
                                "Optional ISO timestamp for transit arrival-by routing. Do "
                                "not send with departure_time or non-transit travel modes."
                            ),
                        },
                        "routing_preference": {
                            "type": "string",
                            "enum": [
                                "traffic_aware",
                                "traffic_aware_optimal",
                                "traffic_unaware",
                            ],
                            "description": "Driving route preference. Defaults to traffic_aware.",
                            "default": "traffic_aware",
                        },
                        "avoid_tolls": {
                            "type": "boolean",
                            "description": "Avoid toll roads when possible.",
                        },
                        "avoid_highways": {
                            "type": "boolean",
                            "description": "Avoid highways when possible.",
                        },
                        "avoid_ferries": {
                            "type": "boolean",
                            "description": "Avoid ferries when possible.",
                        },
                    },
                    "required": ["destination"],
                    "additionalProperties": False,
                },
            },
        ]

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle a Google Maps tool call."""
        if tool_name == "search_google_places":
            return await self._search_google_places(arguments)
        if tool_name == "get_google_place_details":
            return await self._get_google_place_details(arguments)
        if tool_name == "get_google_route":
            return await self._get_google_route(arguments)
        return self._text_result(
            f"Unknown Google Maps tool: {tool_name}",
            is_error=True,
        )

    async def _search_google_places(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Search Google Places using text search."""
        api_key = self._get_api_key()
        if not api_key:
            return self._missing_api_key_result()

        query = str(arguments.get("query") or "").strip()
        if not query:
            return self._text_result("Error: query is required.", is_error=True)

        limit = self._coerce_int_arg(
            arguments.get("limit"),
            default=5,
            minimum=1,
            maximum=MAX_GOOGLE_PLACE_RESULTS,
        )
        body: dict[str, Any] = {"textQuery": query, "pageSize": limit}
        try:
            location_bias = self._build_location_bias(arguments.get("location_bias"))
        except ValueError as err:
            return self._text_result(f"Error: {err}", is_error=True)
        if location_bias is not None:
            body["locationBias"] = location_bias

        try:
            payload = await self._request_json(
                "POST",
                GOOGLE_PLACES_TEXT_SEARCH_URL,
                api_key=api_key,
                field_mask=GOOGLE_PLACES_FIELD_MASK,
                json_body=body,
            )
        except asyncio.TimeoutError:
            return self._text_result("Error: Google Places search timed out.", is_error=True)
        except aiohttp.ClientError as err:
            return self._text_result(
                f"Error: Google Places request failed: {err}",
                is_error=True,
            )
        except ValueError as err:
            return self._text_result(f"Error: {err}", is_error=True)

        places = [
            self._normalize_place(place)
            for place in payload.get("places", [])
            if isinstance(place, dict)
        ]
        structured = {"query": query, "count": len(places), "places": places}
        return self._text_result(
            self._format_places_text(f"Google Places results for '{query}'", places),
            structured_content=structured,
        )

    async def _get_google_place_details(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Fetch details for a Google Place ID."""
        api_key = self._get_api_key()
        if not api_key:
            return self._missing_api_key_result()

        place_id = self._normalize_place_id(arguments.get("place_id"))
        if place_id is None:
            return self._text_result("Error: place_id is required.", is_error=True)

        try:
            place = await self._request_json(
                "GET",
                GOOGLE_PLACE_DETAILS_URL.format(place_id=place_id),
                api_key=api_key,
                field_mask=GOOGLE_PLACE_DETAILS_FIELD_MASK,
            )
        except asyncio.TimeoutError:
            return self._text_result("Error: Google Place Details timed out.", is_error=True)
        except aiohttp.ClientError as err:
            return self._text_result(
                f"Error: Google Place Details request failed: {err}",
                is_error=True,
            )
        except ValueError as err:
            return self._text_result(f"Error: {err}", is_error=True)

        normalized = self._normalize_place(place)
        return self._text_result(
            self._format_places_text("Google Place details", [normalized]),
            structured_content={"place": normalized},
        )

    async def _get_google_route(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Compute a Google Routes ETA."""
        api_key = self._get_api_key()
        if not api_key:
            return self._missing_api_key_result()

        destination = str(arguments.get("destination") or "").strip()
        if not destination:
            return self._text_result("Error: destination is required.", is_error=True)

        origin = str(arguments.get("origin") or "").strip()
        if not origin:
            home_waypoint = self._home_location_waypoint()
            if home_waypoint is None:
                return self._text_result(
                    "Error: origin is required because Home Assistant home coordinates are unavailable.",
                    is_error=True,
                )
            origin_waypoint = home_waypoint
            origin_label = "Home Assistant home"
        else:
            origin_waypoint = self._build_route_waypoint(origin)
            origin_label = origin

        travel_mode = self._normalize_travel_mode(arguments.get("travel_mode"))
        if travel_mode is None:
            return self._text_result(
                "Error: travel_mode must be drive, walk, bicycle, or transit.",
                is_error=True,
            )

        departure_time = str(arguments.get("departure_time") or "").strip()
        arrival_time = str(arguments.get("arrival_time") or "").strip()
        if departure_time and arrival_time:
            return self._text_result(
                "Error: send either departure_time or arrival_time, not both.",
                is_error=True,
            )
        if arrival_time and travel_mode != "TRANSIT":
            return self._text_result(
                "Error: arrival_time is only supported for transit routes. "
                "Use departure_time for drive, walk, or bicycle routes.",
                is_error=True,
            )
        avoid_tolls = bool(arguments.get("avoid_tolls", False))
        avoid_highways = bool(arguments.get("avoid_highways", False))
        avoid_ferries = bool(arguments.get("avoid_ferries", False))
        if travel_mode != "DRIVE" and any(
            (avoid_tolls, avoid_highways, avoid_ferries)
        ):
            return self._text_result(
                "Error: avoid_tolls, avoid_highways, and avoid_ferries are only "
                "supported for driving routes.",
                is_error=True,
            )

        desired_arrival_time: datetime | None = None
        if arrival_time:
            try:
                desired_arrival_time = self._parse_route_time(arrival_time)
            except ValueError as err:
                return self._text_result(f"Error: {err}", is_error=True)

        body: dict[str, Any] = {
            "origin": origin_waypoint,
            "destination": self._build_route_waypoint(destination),
            "travelMode": travel_mode,
            "computeAlternativeRoutes": False,
        }
        routing_preference = self._normalize_routing_preference(
            arguments.get("routing_preference")
        )
        if routing_preference is None:
            return self._text_result(
                "Error: routing_preference must be traffic_aware, "
                "traffic_aware_optimal, or traffic_unaware.",
                is_error=True,
            )
        if travel_mode == "DRIVE":
            body["routingPreference"] = routing_preference
            body["routeModifiers"] = {
                "avoidTolls": avoid_tolls,
                "avoidHighways": avoid_highways,
                "avoidFerries": avoid_ferries,
            }

        try:
            if desired_arrival_time is not None and travel_mode == "TRANSIT":
                body["arrivalTime"] = self._format_route_time(desired_arrival_time)
            if departure_time and departure_time.casefold() != "now":
                body["departureTime"] = self._normalize_route_time(departure_time)
        except ValueError as err:
            return self._text_result(f"Error: {err}", is_error=True)

        try:
            payload = await self._request_json(
                "POST",
                GOOGLE_ROUTES_URL,
                api_key=api_key,
                field_mask=GOOGLE_ROUTES_FIELD_MASK,
                json_body=body,
            )
        except asyncio.TimeoutError:
            return self._text_result("Error: Google Routes request timed out.", is_error=True)
        except aiohttp.ClientError as err:
            return self._text_result(
                f"Error: Google Routes request failed: {err}",
                is_error=True,
            )
        except ValueError as err:
            return self._text_result(f"Error: {err}", is_error=True)

        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            return self._text_result("No Google Routes results returned.", is_error=True)

        route = self._normalize_route(routes[0])
        route["origin"] = origin_label
        route["destination"] = destination
        route["travel_mode"] = travel_mode.lower()
        if desired_arrival_time is not None:
            route["desired_arrival_time"] = self._format_route_time(desired_arrival_time)
            if route.get("duration_seconds") is not None:
                leave_by = desired_arrival_time - timedelta(
                    seconds=route["duration_seconds"]
                )
                route["leave_by"] = self._format_route_time(leave_by)
        return self._text_result(
            self._format_route_text(route),
            structured_content={"route": route},
        )

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        api_key: str,
        field_mask: str,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a Google Maps Platform request and return JSON."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "Content-Type": "application/json",
            "User-Agent": GOOGLE_MAPS_USER_AGENT,
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": field_mask,
        }
        timeout = aiohttp.ClientTimeout(total=15)

        async with aiohttp.ClientSession(headers=headers) as session:
            if method == "GET":
                request = session.get(url, timeout=timeout)
            else:
                request = session.post(url, json=json_body or {}, timeout=timeout)
            async with request as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise ValueError(
                        f"Google Maps returned HTTP {response.status}: {error_text[:200]}"
                    )
                payload = await response.json(content_type=None)

        if not isinstance(payload, dict):
            raise ValueError("Google Maps returned an unexpected response shape.")
        return payload

    def _get_api_key(self) -> str:
        """Read the shared Google Maps API key."""
        system_entry = get_system_entry(self.hass)
        if system_entry is None:
            return DEFAULT_GOOGLE_MAPS_API_KEY
        value = system_entry.options.get(
            CONF_GOOGLE_MAPS_API_KEY,
            system_entry.data.get(CONF_GOOGLE_MAPS_API_KEY, DEFAULT_GOOGLE_MAPS_API_KEY),
        )
        return str(value or "").strip()

    @staticmethod
    def _missing_api_key_result() -> dict[str, Any]:
        """Return a consistent missing-key result."""
        return GoogleMapsTool._text_result(
            "Error: Google Maps API key is required. Add it in shared MCP server settings.",
            is_error=True,
        )

    def _build_location_bias(self, value: Any) -> dict[str, Any] | None:
        """Build an optional Places locationBias object."""
        value_text = str(value or "").strip()
        if value_text:
            lat_lng = self._parse_lat_lng(value_text)
            if lat_lng is None:
                raise ValueError("location_bias must be a latitude,longitude value.")
            return {
                "circle": {
                    "center": {
                        "latitude": lat_lng[0],
                        "longitude": lat_lng[1],
                    },
                    "radius": 50000.0,
                }
            }

        home = self._home_lat_lng()
        if home is None:
            return None
        return {
            "circle": {
                "center": {
                    "latitude": home[0],
                    "longitude": home[1],
                },
                "radius": 50000.0,
            }
        }

    def _home_location_waypoint(self) -> dict[str, Any] | None:
        """Return a route waypoint for the Home Assistant home location."""
        home = self._home_lat_lng()
        if home is None:
            return None
        return self._location_waypoint(home[0], home[1])

    def _home_lat_lng(self) -> tuple[float, float] | None:
        """Return configured Home Assistant latitude and longitude when available."""
        if not self._home_location_sharing_enabled():
            return None
        latitude = getattr(self.hass.config, "latitude", None)
        longitude = getattr(self.hass.config, "longitude", None)
        try:
            lat = float(latitude)
            lng = float(longitude)
        except (TypeError, ValueError):
            return None
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return None
        return lat, lng

    def _home_location_sharing_enabled(self) -> bool:
        """Return whether shared settings allow tools to receive home coordinates."""
        system_entry = get_system_entry(self.hass)
        if system_entry is None:
            return DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS
        value = system_entry.options.get(
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
            system_entry.data.get(
                CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
            ),
        )
        return bool(value)

    def _build_route_waypoint(self, value: str) -> dict[str, Any]:
        """Build a Routes waypoint from address, place ID, or lat/lng input."""
        lat_lng = self._parse_lat_lng(value)
        if lat_lng is not None:
            return self._location_waypoint(lat_lng[0], lat_lng[1])

        if value.casefold().startswith("place_id:"):
            place_id = value.split(":", 1)[1].strip()
            return {"placeId": self._strip_place_resource_prefix(place_id)}
        if value.casefold().startswith("places/"):
            return {"placeId": self._strip_place_resource_prefix(value)}
        if self._looks_like_place_id(value):
            return {"placeId": value.strip()}
        return {"address": value}

    @staticmethod
    def _location_waypoint(latitude: float, longitude: float) -> dict[str, Any]:
        """Build a Routes latitude/longitude waypoint."""
        return {
            "location": {
                "latLng": {
                    "latitude": latitude,
                    "longitude": longitude,
                }
            }
        }

    @staticmethod
    def _parse_lat_lng(value: str) -> tuple[float, float] | None:
        """Parse a 'lat,lng' string when valid."""
        match = _LAT_LNG_RE.match(value)
        if not match:
            return None
        latitude = float(match.group("lat"))
        longitude = float(match.group("lng"))
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return None
        return latitude, longitude

    @staticmethod
    def _normalize_place_id(value: Any) -> str | None:
        """Normalize a Google Places resource ID."""
        value_text = str(value or "").strip()
        if not value_text:
            return None
        place_id = GoogleMapsTool._strip_place_resource_prefix(value_text)
        if not re.fullmatch(r"[A-Za-z0-9_\-:.]+", place_id):
            return None
        return place_id

    @staticmethod
    def _looks_like_place_id(value: str) -> bool:
        """Return whether a bare string looks like a Google Place ID."""
        value_text = str(value or "").strip()
        return bool(
            len(value_text) >= 12
            and re.fullmatch(r"[A-Za-z0-9_-]+", value_text)
            and not re.fullmatch(r"[A-Za-z]+", value_text)
        )

    @staticmethod
    def _strip_place_resource_prefix(value: str) -> str:
        """Strip the Places resource prefix when present."""
        value_text = str(value or "").strip()
        if value_text.casefold().startswith("places/"):
            return value_text.split("/", 1)[1]
        return value_text

    @staticmethod
    def _normalize_travel_mode(value: Any) -> str | None:
        """Normalize user-facing travel mode values to Routes API values."""
        normalized = str(value or "drive").strip().casefold()
        return {
            "drive": "DRIVE",
            "driving": "DRIVE",
            "walk": "WALK",
            "walking": "WALK",
            "bicycle": "BICYCLE",
            "bike": "BICYCLE",
            "bicycling": "BICYCLE",
            "transit": "TRANSIT",
        }.get(normalized)

    @staticmethod
    def _normalize_routing_preference(value: Any) -> str | None:
        """Normalize user-facing routing preferences to Routes API values."""
        normalized = str(value or "traffic_aware").strip().casefold()
        return {
            "traffic_aware": "TRAFFIC_AWARE",
            "traffic": "TRAFFIC_AWARE",
            "traffic_aware_optimal": "TRAFFIC_AWARE_OPTIMAL",
            "optimal": "TRAFFIC_AWARE_OPTIMAL",
            "traffic_unaware": "TRAFFIC_UNAWARE",
            "no_traffic": "TRAFFIC_UNAWARE",
        }.get(normalized)

    @staticmethod
    def _normalize_route_time(value: str) -> str:
        """Normalize an ISO-like timestamp to RFC3339 UTC."""
        return GoogleMapsTool._format_route_time(GoogleMapsTool._parse_route_time(value))

    @staticmethod
    def _parse_route_time(value: str) -> datetime:
        """Parse an ISO-like timestamp."""
        parsed = dt_util.parse_datetime(value)
        if parsed is None:
            raise ValueError(f"Invalid route timestamp: {value}")
        if parsed.tzinfo is None:
            parsed = dt_util.as_local(parsed)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _format_route_time(value: datetime) -> str:
        """Format a route timestamp as RFC3339 UTC."""
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _normalize_place(place: dict[str, Any]) -> dict[str, Any]:
        """Normalize Google Places payloads for MCP output."""
        display_name = place.get("displayName")
        if isinstance(display_name, dict):
            name = str(display_name.get("text") or "").strip()
        else:
            name = str(display_name or "").strip()
        opening_hours = place.get("currentOpeningHours")
        open_now = None
        weekday_descriptions: list[str] = []
        if isinstance(opening_hours, dict):
            if isinstance(opening_hours.get("openNow"), bool):
                open_now = opening_hours["openNow"]
            weekday_descriptions = [
                str(item)
                for item in opening_hours.get("weekdayDescriptions", [])
                if str(item).strip()
            ]
        location = place.get("location") if isinstance(place.get("location"), dict) else {}
        return {
            "place_id": place.get("id", ""),
            "name": name,
            "address": place.get("formattedAddress", ""),
            "phone": place.get("nationalPhoneNumber")
            or place.get("internationalPhoneNumber")
            or "",
            "international_phone": place.get("internationalPhoneNumber", ""),
            "rating": place.get("rating"),
            "user_rating_count": place.get("userRatingCount"),
            "business_status": place.get("businessStatus", ""),
            "open_now": open_now,
            "weekday_descriptions": weekday_descriptions,
            "google_maps_url": place.get("googleMapsUri", ""),
            "website": place.get("websiteUri", ""),
            "location": {
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
            },
        }

    @staticmethod
    def _normalize_route(route: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Google Routes response."""
        duration_seconds = GoogleMapsTool._duration_to_seconds(route.get("duration"))
        static_duration_seconds = GoogleMapsTool._duration_to_seconds(
            route.get("staticDuration")
        )
        traffic_delay_seconds = None
        if duration_seconds is not None and static_duration_seconds is not None:
            traffic_delay_seconds = max(0, duration_seconds - static_duration_seconds)
        localized = route.get("localizedValues")
        if not isinstance(localized, dict):
            localized = {}
        return {
            "duration": GoogleMapsTool._localized_value(localized.get("duration")),
            "duration_seconds": duration_seconds,
            "static_duration": GoogleMapsTool._localized_value(
                localized.get("staticDuration")
            ),
            "static_duration_seconds": static_duration_seconds,
            "traffic_delay_seconds": traffic_delay_seconds,
            "distance": GoogleMapsTool._localized_value(localized.get("distance")),
            "distance_meters": route.get("distanceMeters"),
            "description": route.get("description", ""),
            "warnings": route.get("warnings", []),
        }

    @staticmethod
    def _duration_to_seconds(value: Any) -> int | None:
        """Parse a Google duration string like '123s'."""
        if not isinstance(value, str) or not value.endswith("s"):
            return None
        try:
            return int(float(value[:-1]))
        except ValueError:
            return None

    @staticmethod
    def _localized_value(value: Any) -> str:
        """Extract a localized text value from Routes payloads."""
        if isinstance(value, dict):
            return str(value.get("text") or "")
        return ""

    @staticmethod
    def _format_places_text(label: str, places: list[dict[str, Any]]) -> str:
        """Format place results for an MCP text response."""
        if not places:
            return f"{label}: no places found."

        lines = [f"{label}:"]
        for index, place in enumerate(places, 1):
            lines.append("")
            lines.append(f"{index}. {place['name'] or place['place_id']}")
            if place.get("address"):
                lines.append(f"   Address: {place['address']}")
            if place.get("open_now") is not None:
                lines.append(f"   Open now: {'yes' if place['open_now'] else 'no'}")
            if place.get("phone"):
                lines.append(f"   Phone: {place['phone']}")
            if place.get("rating") is not None:
                rating_text = f"{place['rating']}"
                if place.get("user_rating_count") is not None:
                    rating_text += f" ({place['user_rating_count']} reviews)"
                lines.append(f"   Rating: {rating_text}")
            if place.get("google_maps_url"):
                lines.append(f"   Google Maps: {place['google_maps_url']}")
            if place.get("website"):
                lines.append(f"   Website: {place['website']}")
            lines.append(f"   Place ID: {place['place_id']}")
        return "\n".join(lines)

    @staticmethod
    def _format_route_text(route: dict[str, Any]) -> str:
        """Format a route result for an MCP text response."""
        lines = [
            f"Google route from {route['origin']} to {route['destination']}:",
            f"- Travel mode: {route['travel_mode']}",
        ]
        if route.get("duration"):
            lines.append(f"- ETA: {route['duration']}")
        if route.get("distance"):
            lines.append(f"- Distance: {route['distance']}")
        if route.get("traffic_delay_seconds") is not None:
            minutes = round(route["traffic_delay_seconds"] / 60)
            lines.append(f"- Traffic delay: about {minutes} minute{'s' if minutes != 1 else ''}")
        if route.get("leave_by"):
            lines.append(f"- Leave by: {route['leave_by']}")
        if route.get("warnings"):
            lines.append("- Warnings: " + "; ".join(str(item) for item in route["warnings"]))
        return "\n".join(lines)

    @staticmethod
    def _coerce_int_arg(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        """Coerce an integer-like tool argument to a bounded value."""
        if isinstance(value, bool) or value is None:
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

    @staticmethod
    def _text_result(
        text: str,
        *,
        is_error: bool = False,
        structured_content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a standard MCP text response."""
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }
        if structured_content is not None:
            result["structuredContent"] = structured_content
        return result
