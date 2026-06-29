"""SearXNG Search custom tool for MCP Assist."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

NEWS_QUERY_HINTS = (
    "news",
    "headline",
    "headlines",
    "latest",
    "today",
    "current",
    "right now",
    "breaking",
    "what's happening",
    "what is happening",
)


class SearXNGSearchTool:
    """SearXNG search tool for self-hosted search instances."""

    def __init__(self, hass, base_url: str | None = None) -> None:
        """Initialize SearXNG search tool."""
        self.hass = hass
        self.base_url = self._normalize_base_url(base_url)

    async def initialize(self) -> None:
        """Initialize the tool."""
        if not self.base_url:
            raise ValueError(
                "SearXNG search provider selected but no SearXNG URL is configured."
            )

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name == "search"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get MCP tool definition for SearXNG Search."""
        return [
            {
                "name": "search",
                "description": (
                    "Search the web for up-to-date information, including current "
                    "events and live news, using a self-hosted SearXNG instance."
                ),
                "llmDescription": "Search the web or news for current information.",
                "keywords": ["news", "latest", "current", "today", "right now", "web"],
                "example_queries": [
                    "What's happening right now in Iran?",
                    "Latest Home Assistant release news today",
                ],
                "preferred_when": (
                    "Use for web, internet, current-events, breaking-news, and "
                    "latest-information questions."
                ),
                "returns": (
                    "Search results with titles, URLs, snippets, and structured "
                    "result metadata."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        },
                        "count": {
                            "type": "number",
                            "description": "Number of results to return (default 5, max 20)",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                        },
                        "mode": {
                            "type": "string",
                            "description": (
                                "Search mode: auto treats current-events style "
                                "queries as news-oriented."
                            ),
                            "enum": ["auto", "web", "news"],
                            "default": "auto",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a SearXNG search."""
        del tool_name

        query = arguments.get("query")
        count = self._normalize_count(arguments.get("count", 5))
        mode = self._normalize_mode(arguments.get("mode"), query)
        query_to_send = self._query_for_mode(query, mode)

        _LOGGER.debug(
            "SearXNG Search: '%s' (count: %s, mode: %s)",
            query_to_send,
            count,
            mode,
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": "HA-MCP-Assist/1.0",
        }
        params = {
            "q": query_to_send,
            "format": "json",
            "language": "en",
            "safesearch": "1",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        _LOGGER.error(
                            "SearXNG Search error %s: %s",
                            response.status,
                            error,
                        )
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"❌ Search failed (HTTP {response.status}): "
                                        f"{error[:200]}"
                                    ),
                                }
                            ]
                        }

                    data = await response.json()
                    results = [
                        self._normalize_result(item)
                        for item in data.get("results", [])[:count]
                    ]

                    heading = "News results" if mode == "news" else "Search results"
                    text_results = f"🔍 {heading} for '{query}':\n\n"
                    for index, result in enumerate(results, 1):
                        text_results += f"{index}. **{result['title']}**\n"
                        if result.get("source") or result.get("date"):
                            details = " | ".join(
                                part
                                for part in [
                                    result.get("source", ""),
                                    result.get("date", ""),
                                ]
                                if part
                            )
                            if details:
                                text_results += f"   {details}\n"
                        text_results += f"   {result['url']}\n"
                        text_results += f"   {result['snippet']}\n\n"

                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": text_results,
                            }
                        ],
                        "structuredContent": {
                            "query": query,
                            "provider_query": query_to_send,
                            "mode": mode,
                            "count": len(results),
                            "results": results,
                        },
                    }

        except asyncio.TimeoutError:
            _LOGGER.error("SearXNG Search timeout")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "❌ Search timeout - please try again",
                    }
                ]
            }
        except Exception as err:
            _LOGGER.error("SearXNG Search exception: %s", err)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"❌ Search error: {err}",
                    }
                ]
            }

    @staticmethod
    def _normalize_base_url(raw_base_url: str | None) -> str | None:
        """Normalize a configured SearXNG base URL to the search endpoint."""
        base_url = str(raw_base_url or "").strip().rstrip("/")
        if not base_url:
            return None
        if base_url.endswith("/search"):
            return base_url
        return f"{base_url}/search"

    @staticmethod
    def _normalize_count(raw_count: Any) -> int:
        """Normalize requested result count."""
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 5
        return max(1, min(count, 20))

    def _normalize_mode(self, raw_mode: Any, query: Any) -> str:
        """Normalize requested search mode."""
        normalized = str(raw_mode or "auto").strip().lower()
        if normalized not in {"auto", "web", "news"}:
            normalized = "auto"
        if normalized == "auto":
            lowered_query = str(query or "").strip().lower()
            if any(hint in lowered_query for hint in NEWS_QUERY_HINTS):
                return "news"
            return "web"
        return normalized

    def _query_for_mode(self, query: Any, mode: str) -> str:
        """Slightly bias news queries toward fresher web results."""
        normalized_query = str(query or "").strip()
        if mode != "news":
            return normalized_query
        lowered_query = normalized_query.lower()
        if any(hint in lowered_query for hint in NEWS_QUERY_HINTS):
            return normalized_query
        return f"{normalized_query} latest news"

    @staticmethod
    def _normalize_result(raw_result: dict[str, Any]) -> dict[str, str]:
        """Normalize SearXNG results into one structure."""
        engines = raw_result.get("engines")
        engine_source = raw_result.get("engine")
        if not engine_source and isinstance(engines, list):
            engine_source = ", ".join(str(engine) for engine in engines if engine)

        return {
            "title": str(raw_result.get("title", "") or ""),
            "url": str(raw_result.get("url", "") or ""),
            "snippet": str(
                raw_result.get("content")
                or raw_result.get("snippet")
                or raw_result.get("description")
                or ""
            ),
            "source": str(engine_source or ""),
            "date": str(
                raw_result.get("publishedDate")
                or raw_result.get("published_date")
                or raw_result.get("date")
                or ""
            ),
        }
