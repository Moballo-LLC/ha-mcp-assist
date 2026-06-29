"""Tests for search and URL-reading tool helpers."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

from custom_components.mcp_assist.custom_tools import brave_search as brave_module
from custom_components.mcp_assist.custom_tools import read_url as read_url_module
from custom_components.mcp_assist.custom_tools import searxng_search as searxng_module

sys.modules.setdefault("ddgs", types.SimpleNamespace(DDGS=object))
sys.modules.setdefault("duckduckgo_search", types.SimpleNamespace(DDGS=object))
ddg_module = importlib.import_module(
    "custom_components.mcp_assist.custom_tools.duckduckgo_search"
)


def test_search_tool_definitions_include_current_events_routing_metadata(hass) -> None:
    """Built-in search helpers should advertise live-news routing hints."""
    brave_definition = brave_module.BraveSearchTool(hass, api_key="secret").get_tool_definitions()[0]
    ddg_definition = ddg_module.DuckDuckGoSearchTool(hass).get_tool_definitions()[0]
    searxng_definition = searxng_module.SearXNGSearchTool(
        hass,
        base_url="http://search.local",
    ).get_tool_definitions()[0]

    for definition in (brave_definition, ddg_definition, searxng_definition):
        assert "news" in definition["keywords"]
        assert definition["preferred_when"]
        assert definition["returns"]


class _FakeResponse:
    """Minimal async HTTP response stub."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
        json_data: dict | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._text = text
        self._json_data = json_data or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return self._text

    async def json(self) -> dict:
        return self._json_data


class _FakeSession:
    """Minimal async HTTP session stub."""

    def __init__(self, *, response: _FakeResponse | None = None, error=None) -> None:
        self._response = response
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


@pytest.mark.asyncio
async def test_brave_search_formats_successful_results(hass, monkeypatch) -> None:
    """Brave search should clamp counts and format returned results."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            json_data={
                "web": {
                    "results": [
                        {
                            "title": "Weather Result",
                            "url": "https://example.com/weather",
                            "description": "Forecast details",
                        }
                    ]
                }
            }
        )
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(brave_module.aiohttp, "ClientSession", _client_session)
    tool = brave_module.BraveSearchTool(hass, api_key="secret")

    result = await tool.handle_call("search", {"query": "weather", "count": 50})

    assert "Weather Result" in result["content"][0]["text"]
    assert "https://example.com/weather" in result["content"][0]["text"]
    assert result["structuredContent"]["mode"] == "web"
    assert result["structuredContent"]["results"][0]["url"] == "https://example.com/weather"
    assert fake_session.calls[0][1]["params"]["count"] == "20"
    assert fake_session.calls[0][1]["headers"]["X-Subscription-Token"] == "secret"


@pytest.mark.asyncio
async def test_brave_search_returns_timeout_error(hass, monkeypatch) -> None:
    """Brave search should return a friendly timeout error payload."""
    fake_session = _FakeSession(error=asyncio.TimeoutError())

    def _client_session():
        return fake_session

    monkeypatch.setattr(brave_module.aiohttp, "ClientSession", _client_session)
    tool = brave_module.BraveSearchTool(hass, api_key="secret")

    result = await tool.handle_call("search", {"query": "weather"})

    assert result["content"][0]["text"] == "❌ Search timeout - please try again"


@pytest.mark.asyncio
async def test_searxng_search_requires_url(hass) -> None:
    """SearXNG should fail clearly when selected without a configured URL."""
    tool = searxng_module.SearXNGSearchTool(hass)

    with pytest.raises(ValueError, match="no SearXNG URL"):
        await tool.initialize()


@pytest.mark.asyncio
async def test_searxng_search_formats_successful_results(hass, monkeypatch) -> None:
    """SearXNG search should normalize URL, query, and returned results."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            json_data={
                "results": [
                    {
                        "title": "Weather Result",
                        "url": "https://example.com/weather",
                        "content": "Forecast details",
                        "engines": ["duckduckgo", "brave"],
                        "publishedDate": "2026-06-01",
                    },
                    {
                        "title": "Second Result",
                        "url": "https://example.com/second",
                        "content": "More details",
                    },
                ]
            }
        )
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(searxng_module.aiohttp, "ClientSession", _client_session)
    tool = searxng_module.SearXNGSearchTool(hass, base_url="http://search.local/")

    result = await tool.handle_call("search", {"query": "weather", "count": 1})

    assert "Weather Result" in result["content"][0]["text"]
    assert "duckduckgo, brave" in result["content"][0]["text"]
    assert result["structuredContent"]["mode"] == "web"
    assert result["structuredContent"]["count"] == 1
    assert result["structuredContent"]["results"][0]["snippet"] == "Forecast details"
    assert fake_session.calls[0][0] == "http://search.local/search"
    assert fake_session.calls[0][1]["params"]["q"] == "weather"
    assert fake_session.calls[0][1]["params"]["format"] == "json"


@pytest.mark.asyncio
async def test_searxng_search_biases_news_mode_queries(hass, monkeypatch) -> None:
    """News mode should bias SearXNG queries toward fresh results."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            json_data={
                "results": [
                    {
                        "title": "Mariners update",
                        "url": "https://example.com/mariners",
                        "content": "Trade deadline news",
                        "engine": "brave",
                    }
                ]
            }
        )
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(searxng_module.aiohttp, "ClientSession", _client_session)
    tool = searxng_module.SearXNGSearchTool(
        hass,
        base_url="http://search.local/search",
    )

    result = await tool.handle_call(
        "search",
        {"query": "mariners", "mode": "news"},
    )

    assert result["structuredContent"]["mode"] == "news"
    assert result["structuredContent"]["provider_query"] == "mariners latest news"
    assert fake_session.calls[0][1]["params"]["q"] == "mariners latest news"


@pytest.mark.asyncio
async def test_searxng_search_returns_timeout_error(hass, monkeypatch) -> None:
    """SearXNG search should return a friendly timeout error payload."""
    fake_session = _FakeSession(error=asyncio.TimeoutError())

    def _client_session():
        return fake_session

    monkeypatch.setattr(searxng_module.aiohttp, "ClientSession", _client_session)
    tool = searxng_module.SearXNGSearchTool(
        hass,
        base_url="http://search.local",
    )

    result = await tool.handle_call("search", {"query": "weather"})

    assert result["content"][0]["text"] == "❌ Search timeout - please try again"


@pytest.mark.asyncio
async def test_duckduckgo_search_formats_executor_results(hass, monkeypatch) -> None:
    """DuckDuckGo search should format normalized executor results."""
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    def _search_sync(query: str, count: int, mode: str):
        assert query == "mariners"
        assert count == 2
        assert mode == "web"
        return [
            {
                "title": "Schedule",
                "url": "https://example.com/schedule",
                "snippet": "Upcoming games",
                "source": "",
                "date": "",
            }
        ]

    monkeypatch.setattr(tool, "_search_sync", _search_sync)

    result = await tool.handle_call("search", {"query": "mariners", "count": 2})

    assert "Schedule" in result["content"][0]["text"]
    assert "https://example.com/schedule" in result["content"][0]["text"]
    assert result["structuredContent"]["mode"] == "web"
    assert result["structuredContent"]["results"][0]["url"] == "https://example.com/schedule"


def test_duckduckgo_search_sync_normalizes_ddgs_results(monkeypatch, hass) -> None:
    """The synchronous DDG wrapper should normalize provider field names."""

    class _FakeDDGS:
        def text(self, query, **kwargs):
            assert query == "bus"
            assert kwargs["max_results"] == 3
            assert kwargs["backend"] == "duckduckgo"
            return [
                {"title": "Route 372", "href": "https://example.com/372", "body": "ETA"}
            ]

    monkeypatch.setattr(ddg_module, "DDGS", _FakeDDGS)
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    results = tool._search_sync("bus", 3, "web")

    assert results == [
        {
            "title": "Route 372",
            "url": "https://example.com/372",
            "snippet": "ETA",
            "source": "",
            "date": "",
        }
    ]


def test_duckduckgo_search_sync_uses_news_mode_for_news_queries(monkeypatch, hass) -> None:
    """News-mode DDGS searches should use the provider news endpoint."""

    class _FakeDDGS:
        def news(self, query, **kwargs):
            assert query == "Iran latest"
            assert kwargs["max_results"] == 2
            assert kwargs["backend"] == "duckduckgo"
            return [
                {
                    "title": "Iran update",
                    "url": "https://example.com/iran",
                    "body": "Top development",
                    "source": "Reuters",
                    "date": "2026-04-12",
                }
            ]

        def text(self, query, **kwargs):
            del query, kwargs
            raise AssertionError("news mode should not call text()")

    monkeypatch.setattr(ddg_module, "DDGS", _FakeDDGS)
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    results = tool._search_sync("Iran latest", 2, "news")

    assert results == [
        {
            "title": "Iran update",
            "url": "https://example.com/iran",
            "snippet": "Top development",
            "source": "Reuters",
            "date": "2026-04-12",
        }
    ]


def test_duckduckgo_search_sync_preserves_duckduckgo_backend_on_news_fallback(
    monkeypatch, hass
) -> None:
    """News fallback should still use DuckDuckGo-only web search."""

    calls = []

    class _FakeDDGS:
        def news(self, query, **kwargs):
            assert query == "Iran latest"
            assert kwargs["backend"] == "duckduckgo"
            raise RuntimeError("news unavailable")

        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            return [
                {"title": "Iran fallback", "href": "https://example.com/fallback", "body": "Fallback"}
            ]

    monkeypatch.setattr(ddg_module, "DDGS", _FakeDDGS)
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    results = tool._search_sync("Iran latest", 2, "news")

    assert calls == [
        (
            "Iran latest",
            {
                "max_results": 2,
                "region": "us-en",
                "safesearch": "moderate",
                "backend": "duckduckgo",
            },
        )
    ]
    assert results[0]["title"] == "Iran fallback"


def test_duckduckgo_search_sync_preserves_legacy_text_backend(monkeypatch, hass) -> None:
    """Legacy duckduckgo_search fallback should keep its supported text backend."""

    class _FakeDDGS:
        def text(self, query, **kwargs):
            assert query == "bus"
            assert kwargs["backend"] == "auto"
            return [
                {"title": "Route 372", "href": "https://example.com/372", "body": "ETA"}
            ]

    monkeypatch.setattr(ddg_module, "DDGS", _FakeDDGS)
    monkeypatch.setattr(ddg_module, "_USES_RENAMED_DDGS", False)
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    results = tool._search_sync("bus", 3, "web")

    assert results[0]["title"] == "Route 372"


def test_duckduckgo_search_sync_omits_legacy_news_backend(monkeypatch, hass) -> None:
    """Legacy duckduckgo_search fallback news calls should omit unsupported backend."""

    class _FakeDDGS:
        def news(self, query, **kwargs):
            assert query == "Iran latest"
            assert "backend" not in kwargs
            return [
                {
                    "title": "Iran update",
                    "url": "https://example.com/iran",
                    "body": "Top development",
                    "source": "Reuters",
                    "date": "2026-04-12",
                }
            ]

        def text(self, query, **kwargs):
            del query, kwargs
            raise AssertionError("news mode should not call text()")

    monkeypatch.setattr(ddg_module, "DDGS", _FakeDDGS)
    monkeypatch.setattr(ddg_module, "_USES_RENAMED_DDGS", False)
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    results = tool._search_sync("Iran latest", 2, "news")

    assert results[0]["title"] == "Iran update"


@pytest.mark.asyncio
async def test_duckduckgo_search_returns_error_payload_on_failure(
    hass, monkeypatch
) -> None:
    """DuckDuckGo search failures should become MCP error text."""
    tool = ddg_module.DuckDuckGoSearchTool(hass)

    def _search_sync(query: str, count: int, mode: str):
        del query, count, mode
        raise RuntimeError("search backend failed")

    monkeypatch.setattr(tool, "_search_sync", _search_sync)

    result = await tool.handle_call("search", {"query": "mariners"})

    assert result["content"][0]["text"] == "❌ Search error: search backend failed"


@pytest.mark.asyncio
async def test_read_url_extracts_html_text_and_decodes_entities(hass) -> None:
    """HTML extraction should strip tags and decode common entities."""
    tool = read_url_module.ReadUrlTool(hass)

    text = await tool._extract_text(
        """
        <html>
          <head><style>body { color: red; }</style></head>
          <body>
            <script>console.log("ignore")</script >
            <h1>Hello &amp; welcome</h1>
            <p>Line&nbsp;two</p>
          </body>
        </html>
        """,
        "text/html",
    )

    assert text == "Hello & welcome Line two"


@pytest.mark.asyncio
async def test_read_url_handles_valid_html_pages(hass, monkeypatch) -> None:
    """Read URL should fetch, extract, and format supported HTML pages."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="""
            <html>
              <head><title>Example Page</title></head>
              <body><p>Hello world</p></body>
            </html>
            """,
        )
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(read_url_module.aiohttp, "ClientSession", _client_session)
    tool = read_url_module.ReadUrlTool(hass)

    result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/page", "summary": False},
    )

    assert "📖 **Example Page**" in result["content"][0]["text"]
    assert "Hello world" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_read_url_rejects_invalid_urls_and_timeouts(hass, monkeypatch) -> None:
    """Read URL should fail cleanly for invalid URLs and request timeouts."""
    tool = read_url_module.ReadUrlTool(hass)

    invalid = await tool.handle_call("read_url", {"url": "notaurl"})
    assert invalid["content"][0]["text"] == "❌ Invalid URL format"

    unsupported = await tool.handle_call("read_url", {"url": "ftp://example.com"})
    assert unsupported["content"][0]["text"] == "❌ Unsupported URL scheme: ftp"

    fake_session = _FakeSession(error=asyncio.TimeoutError())

    def _client_session():
        return fake_session

    monkeypatch.setattr(read_url_module.aiohttp, "ClientSession", _client_session)
    timed_out = await tool.handle_call("read_url", {"url": "https://example.com"})

    assert timed_out["content"][0]["text"] == "❌ Timeout reading URL"
