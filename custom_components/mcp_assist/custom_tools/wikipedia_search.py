"""Wikipedia search custom tool for MCP Assist."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

DEFAULT_WIKIPEDIA_LANGUAGE = "en"
MAX_WIKIPEDIA_RESULTS = 10
WIKIPEDIA_USER_AGENT = (
    "ha-mcp-assist/1.0 (https://github.com/Moballo-LLC/ha-mcp-assist)"
)
_LANGUAGE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class WikipediaSearchError(Exception):
    """Raised when a Wikipedia search request fails."""


class WikipediaSearchTool:
    """Search Wikipedia through the MediaWiki OpenSearch API."""

    def __init__(self, hass) -> None:
        """Initialize the Wikipedia search tool."""
        self.hass = hass

    async def initialize(self) -> None:
        """Initialize the tool."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this tool handles the requested name."""
        return tool_name == "search_wikipedia"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions for Wikipedia search."""
        return [
            {
                "name": "search_wikipedia",
                "description": (
                    "Search Wikipedia articles for lightweight background and reference "
                    "information. Use this when an encyclopedia result is enough and a "
                    "full web search is unnecessary."
                ),
                "llmDescription": "Search Wikipedia for reference articles.",
                "keywords": ["wikipedia", "encyclopedia", "reference", "background"],
                "example_queries": [
                    "Search Wikipedia for heat pumps",
                    "Find a Wikipedia article about Ada Lovelace",
                ],
                "preferred_when": (
                    "Use for stable background, reference, and encyclopedia-style "
                    "questions. Use web search instead for current events or latest "
                    "information."
                ),
                "returns": (
                    "Wikipedia article search results with titles, URLs, descriptions, "
                    "and structured result metadata."
                ),
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The Wikipedia search query.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                "Maximum number of results to return (default 5, max 10)."
                            ),
                            "minimum": 1,
                            "default": 5,
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "Optional Wikipedia language subdomain, such as en, es, "
                                "de, or simple. Defaults to en."
                            ),
                            "default": DEFAULT_WIKIPEDIA_LANGUAGE,
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
        """Run a Wikipedia search."""
        if not self.handles_tool(tool_name):
            return self._text_result(
                f"Unknown Wikipedia search tool: {tool_name}",
                is_error=True,
            )

        query = str(arguments.get("query") or "").strip()
        if not query:
            return self._text_result("Error: query is required.", is_error=True)

        language = self._normalize_language(arguments.get("language"))
        if language is None:
            return self._text_result(
                "Error: language must be a Wikipedia subdomain such as en, es, de, or simple.",
                is_error=True,
            )

        limit = self._coerce_int_arg(
            arguments.get("limit"),
            default=5,
            minimum=1,
            maximum=MAX_WIKIPEDIA_RESULTS,
        )

        _LOGGER.debug(
            "Wikipedia search: %r (language: %s, limit: %s)",
            query,
            language,
            limit,
        )

        try:
            results = await self._search_wikipedia(query, language, limit)
        except asyncio.TimeoutError:
            return self._text_result("Error: Wikipedia search timed out.", is_error=True)
        except WikipediaSearchError as err:
            return self._text_result(f"Error: {err}", is_error=True)
        except aiohttp.ClientError as err:
            _LOGGER.warning("Wikipedia search client error: %s", err)
            return self._text_result(
                f"Error: Wikipedia search request failed: {err}",
                is_error=True,
            )
        except Exception as err:
            _LOGGER.exception("Unexpected Wikipedia search error")
            return self._text_result(
                f"Error: unexpected Wikipedia search failure: {err}",
                is_error=True,
            )

        structured_content = {
            "query": query,
            "language": language,
            "count": len(results),
            "results": results,
        }
        return self._text_result(
            self._format_results_text(query, language, results),
            structured_content=structured_content,
        )

    async def _search_wikipedia(
        self,
        query: str,
        language: str,
        limit: int,
    ) -> list[dict[str, str]]:
        """Call Wikipedia OpenSearch and normalize the response."""
        endpoint = f"https://{language}.wikipedia.org/w/api.php"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": WIKIPEDIA_USER_AGENT,
        }
        params = {
            "action": "opensearch",
            "search": query,
            "limit": str(limit),
            "namespace": "0",
            "format": "json",
            "redirects": "resolve",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise WikipediaSearchError(
                        f"Wikipedia returned HTTP {response.status}: {error_text[:200]}"
                    )
                payload = await response.json(content_type=None)

        return self._normalize_opensearch_payload(payload)

    @staticmethod
    def _normalize_opensearch_payload(payload: Any) -> list[dict[str, str]]:
        """Normalize the OpenSearch array payload into result dictionaries."""
        if not isinstance(payload, list) or len(payload) < 4:
            raise WikipediaSearchError("Wikipedia returned an unexpected response shape.")

        titles = payload[1]
        descriptions = payload[2]
        urls = payload[3]
        if not isinstance(titles, list) or not isinstance(urls, list):
            raise WikipediaSearchError("Wikipedia returned malformed search results.")
        if not isinstance(descriptions, list):
            descriptions = []

        results = []
        for index, title in enumerate(titles):
            title_text = str(title or "").strip()
            url = str(urls[index] if index < len(urls) else "").strip()
            if not title_text or not url:
                continue
            description = str(
                descriptions[index] if index < len(descriptions) else ""
            ).strip()
            results.append(
                {
                    "title": title_text,
                    "url": url,
                    "description": description,
                }
            )
        return results

    @staticmethod
    def _format_results_text(
        query: str,
        language: str,
        results: list[dict[str, str]],
    ) -> str:
        """Format Wikipedia search results for an MCP text response."""
        if not results:
            return f"No Wikipedia results found for '{query}' on {language}.wikipedia.org."

        lines = [f"Wikipedia results for '{query}' on {language}.wikipedia.org:"]
        for index, result in enumerate(results, 1):
            lines.append("")
            lines.append(f"{index}. {result['title']}")
            lines.append(f"   {result['url']}")
            if result.get("description"):
                lines.append(f"   {result['description']}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_language(value: Any) -> str | None:
        """Normalize and validate a Wikipedia language subdomain."""
        language = str(value or DEFAULT_WIKIPEDIA_LANGUAGE).strip().lower()
        if not language or len(language) > 24:
            return None
        if not _LANGUAGE_RE.fullmatch(language):
            return None
        return language

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
