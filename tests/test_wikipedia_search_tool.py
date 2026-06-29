"""Tests for the built-in Wikipedia search tool."""

from __future__ import annotations

import aiohttp
import pytest

from custom_components.mcp_assist.custom_tools import wikipedia_search as wikipedia_module


class _FakeWikipediaResponse:
    """Minimal async response for Wikipedia search tests."""

    def __init__(
        self,
        *,
        status: int = 200,
        json_data=None,
        text: str = "",
    ) -> None:
        self.status = status
        self._json_data = json_data
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self, *args, **kwargs):
        return self._json_data

    async def text(self) -> str:
        return self._text


class _FakeWikipediaSession:
    """Minimal async session that records Wikipedia requests."""

    def __init__(self, response: _FakeWikipediaResponse, captured: dict) -> None:
        self._response = response
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url, **kwargs):
        self._captured["url"] = url
        self._captured["request_kwargs"] = kwargs
        return self._response


def _wikipedia_client_session(
    response: _FakeWikipediaResponse,
    captured: dict,
):
    """Build a fake aiohttp ClientSession factory."""

    def _client_session(**kwargs):
        captured["session_kwargs"] = kwargs
        return _FakeWikipediaSession(response, captured)

    return _client_session


def test_wikipedia_search_tool_definition_has_reference_routing_metadata(hass) -> None:
    """Wikipedia search should advertise reference-tool routing hints."""
    definition = wikipedia_module.WikipediaSearchTool(hass).get_tool_definitions()[0]

    assert definition["name"] == "search_wikipedia"
    assert "wikipedia" in definition["keywords"]
    assert "current events" in definition["preferred_when"]
    assert definition["returns"]
    assert definition["inputSchema"]["properties"]["limit"]["maximum"] == 10


@pytest.mark.asyncio
async def test_search_wikipedia_returns_structured_results(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wikipedia OpenSearch responses should become MCP text and structured content."""
    captured = {}
    payload = [
        "Ada",
        ["Ada Lovelace", "Ada"],
        ["English mathematician and writer", ""],
        [
            "https://en.wikipedia.org/wiki/Ada_Lovelace",
            "https://en.wikipedia.org/wiki/Ada",
        ],
    ]
    monkeypatch.setattr(
        wikipedia_module.aiohttp,
        "ClientSession",
        _wikipedia_client_session(
            _FakeWikipediaResponse(json_data=payload),
            captured,
        ),
    )

    result = await wikipedia_module.WikipediaSearchTool(hass).handle_call(
        "search_wikipedia",
        {"query": "Ada", "limit": 50, "language": "EN"},
    )

    assert result["isError"] is False
    assert captured["url"] == "https://en.wikipedia.org/w/api.php"
    assert captured["session_kwargs"]["headers"]["User-Agent"].startswith(
        "ha-mcp-assist/"
    )
    assert captured["request_kwargs"]["params"] == {
        "action": "opensearch",
        "search": "Ada",
        "limit": "10",
        "namespace": "0",
        "format": "json",
        "redirects": "resolve",
    }
    assert result["structuredContent"] == {
        "query": "Ada",
        "language": "en",
        "count": 2,
        "results": [
            {
                "title": "Ada Lovelace",
                "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                "description": "English mathematician and writer",
            },
            {
                "title": "Ada",
                "url": "https://en.wikipedia.org/wiki/Ada",
                "description": "",
            },
        ],
    }
    assert "Ada Lovelace" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_search_wikipedia_rejects_invalid_language_without_fetch(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Language input should not be able to alter the fixed Wikipedia host shape."""

    def _client_session(**_kwargs):
        raise AssertionError("Wikipedia request should not be created")

    monkeypatch.setattr(wikipedia_module.aiohttp, "ClientSession", _client_session)

    result = await wikipedia_module.WikipediaSearchTool(hass).handle_call(
        "search_wikipedia",
        {"query": "Ada", "language": "en.wikipedia.org"},
    )

    assert result["isError"] is True
    assert "language must be a Wikipedia subdomain" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_search_wikipedia_reports_http_errors(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP failures should become MCP error results."""
    captured = {}
    monkeypatch.setattr(
        wikipedia_module.aiohttp,
        "ClientSession",
        _wikipedia_client_session(
            _FakeWikipediaResponse(status=503, text="try again later"),
            captured,
        ),
    )

    result = await wikipedia_module.WikipediaSearchTool(hass).handle_call(
        "search_wikipedia",
        {"query": "Ada"},
    )

    assert result["isError"] is True
    assert "HTTP 503" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_search_wikipedia_reports_client_errors(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aiohttp client failures should become MCP error results."""

    class _FailingSession:
        async def __aenter__(self):
            raise aiohttp.ClientError("network down")

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(
        wikipedia_module.aiohttp,
        "ClientSession",
        lambda **_kwargs: _FailingSession(),
    )

    result = await wikipedia_module.WikipediaSearchTool(hass).handle_call(
        "search_wikipedia",
        {"query": "Ada"},
    )

    assert result["isError"] is True
    assert "request failed" in result["content"][0]["text"]
