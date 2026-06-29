"""Tests for search and URL-reading tool helpers."""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
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


def _allow_public_example_resolution(tool) -> None:
    """Stub URL reader DNS resolution to a public address."""

    async def _resolve_host_addresses(host: str, port: int):
        assert host in {"example.com", "news.example.com"}
        assert port in {80, 443}
        return {ipaddress.ip_address("93.184.216.34")}

    tool._resolve_host_addresses = _resolve_host_addresses


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


class _FakeContent:
    """Minimal stream reader stub."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def read(self, n: int = -1) -> bytes:
        if n < 0:
            return self._body
        return self._body[:n]


class _FakeResponse:
    """Minimal async HTTP response stub."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
        content_bytes: bytes | None = None,
        charset: str | None = None,
        json_data: dict | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._text = text
        self._json_data = json_data or {}
        self.charset = charset
        if content_bytes is not None:
            self.content = _FakeContent(content_bytes)

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

    def __init__(
        self,
        *,
        response: _FakeResponse | None = None,
        responses: list[_FakeResponse] | None = None,
        error=None,
    ) -> None:
        self._response = response
        self._responses = list(responses or [])
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
        if self._responses:
            return self._responses.pop(0)
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
    _allow_public_example_resolution(tool)

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
    _allow_public_example_resolution(tool)
    timed_out = await tool.handle_call("read_url", {"url": "https://example.com"})

    assert timed_out["content"][0]["text"] == "❌ Timeout reading URL"


@pytest.mark.asyncio
async def test_read_url_blocks_local_and_private_urls_before_fetch(
    hass,
    monkeypatch,
) -> None:
    """Read URL should not fetch local/private network targets by default."""
    tool = read_url_module.ReadUrlTool(hass)

    def _client_session():
        raise AssertionError("unsafe URLs should be rejected before opening a session")

    monkeypatch.setattr(read_url_module.aiohttp, "ClientSession", _client_session)

    localhost = await tool.handle_call("read_url", {"url": "http://localhost:8123/api"})
    assert localhost["content"][0]["text"] == "❌ Local URLs must be allowlisted in Home Assistant"

    loopback = await tool.handle_call("read_url", {"url": "http://127.0.0.1:8123/api"})
    assert loopback["content"][0]["text"] == (
        "❌ Private or local URLs must be allowlisted in Home Assistant"
    )

    shared_address = await tool.handle_call("read_url", {"url": "http://100.64.0.1/status"})
    assert shared_address["content"][0]["text"] == (
        "❌ Private or local URLs must be allowlisted in Home Assistant"
    )

    async def _resolve_private_host(host: str, port: int):
        assert host == "internal.example"
        assert port == 443
        return {ipaddress.ip_address("10.0.0.5")}

    tool._resolve_host_addresses = _resolve_private_host
    private_dns = await tool.handle_call("read_url", {"url": "https://internal.example"})
    assert private_dns["content"][0]["text"] == (
        "❌ Private or local URLs must be allowlisted in Home Assistant"
    )


@pytest.mark.asyncio
async def test_read_url_allows_explicitly_allowlisted_private_url(
    hass,
    monkeypatch,
) -> None:
    """Home Assistant's external URL allowlist should opt private URLs back in."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain"},
            text="health ok",
        )
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(read_url_module.aiohttp, "ClientSession", _client_session)
    monkeypatch.setattr(
        hass.config,
        "is_allowed_external_url",
        lambda url: url == "http://127.0.0.1:8123/health",
    )
    tool = read_url_module.ReadUrlTool(hass)

    result = await tool.handle_call(
        "read_url",
        {"url": "http://127.0.0.1:8123/health"},
    )

    assert "health ok" in result["content"][0]["text"]
    assert fake_session.calls[0][0] == "http://127.0.0.1:8123/health"


@pytest.mark.asyncio
async def test_read_url_revalidates_redirect_targets(
    hass,
    monkeypatch,
) -> None:
    """Redirects should be blocked when they target local/private network space."""
    fake_session = _FakeSession(
        responses=[
            _FakeResponse(
                status=302,
                headers={"Location": "http://169.254.169.254/latest/meta-data"},
            ),
        ]
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(read_url_module.aiohttp, "ClientSession", _client_session)
    tool = read_url_module.ReadUrlTool(hass)
    _allow_public_example_resolution(tool)

    result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/start"},
    )

    assert result["content"][0]["text"] == (
        "❌ Unsafe redirect blocked: Private or local URLs must be allowlisted in Home Assistant"
    )
    assert len(fake_session.calls) == 1


@pytest.mark.asyncio
async def test_read_url_rejects_oversized_responses(
    hass,
    monkeypatch,
) -> None:
    """Read URL should reject oversized responses before decoding full page text."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain", "Content-Length": "513000"},
            text="too large",
        )
    )

    def _client_session():
        return fake_session

    monkeypatch.setattr(read_url_module.aiohttp, "ClientSession", _client_session)
    tool = read_url_module.ReadUrlTool(hass)
    _allow_public_example_resolution(tool)

    length_result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/huge"},
    )
    assert length_result["content"][0]["text"] == "❌ URL response is too large to read safely"

    streamed_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain"},
            content_bytes=b"x" * (tool.max_response_bytes + 2),
        )
    )
    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        lambda: streamed_session,
    )

    streamed_result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/chunked"},
    )
    assert streamed_result["content"][0]["text"] == "❌ URL response is too large to read safely"
