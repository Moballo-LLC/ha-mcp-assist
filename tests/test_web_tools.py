"""Tests for search and URL-reading tool helpers."""

from __future__ import annotations

import asyncio
import importlib
import ipaddress
import logging
import socket
import sys
import types
from urllib.parse import urlparse

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

    def __init__(
        self,
        body: bytes | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self._body = body or b""
        self._chunks = list(chunks or [])

    async def read(self, n: int = -1) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if n < 0:
            chunk = self._body
            self._body = b""
            return chunk
        chunk = self._body[:n]
        self._body = self._body[n:]
        return chunk


class _FakeResponse:
    """Minimal async HTTP response stub."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
        content_bytes: bytes | None = None,
        content_chunks: list[bytes] | None = None,
        charset: str | None = None,
        json_data: dict | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._text = text
        self._json_data = json_data or {}
        self.charset = charset
        if content_bytes is not None or content_chunks is not None:
            self.content = _FakeContent(content_bytes, content_chunks)

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
        self.client_session_kwargs: list[dict] = []

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


def _read_url_client_session(fake_session: _FakeSession):
    """Build a ClientSession factory that records constructor kwargs."""

    def _client_session(**kwargs):
        fake_session.client_session_kwargs.append(kwargs)
        return fake_session

    return _client_session


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
async def test_duckduckgo_search_formats_executor_results(
    hass, monkeypatch, caplog
) -> None:
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

    with caplog.at_level(logging.DEBUG, logger=ddg_module._LOGGER.name):
        result = await tool.handle_call("search", {"query": "mariners", "count": 2})

    assert "Schedule" in result["content"][0]["text"]
    assert "https://example.com/schedule" in result["content"][0]["text"]
    assert result["structuredContent"]["mode"] == "web"
    assert result["structuredContent"]["results"][0]["url"] == "https://example.com/schedule"
    assert "DuckDuckGo Search request" in caplog.text
    assert "query_chars=8" in caplog.text
    assert "mariners" not in caplog.text


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

    assert result["content"][0]["text"] == "❌ Search error: RuntimeError"


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
async def test_read_url_prefers_main_content_over_page_chrome(hass) -> None:
    """HTML extraction should prefer article/main content over navigation chrome."""
    tool = read_url_module.ReadUrlTool(hass)

    text = await tool._extract_text(
        """
        <html>
          <body>
            <header>Site header</header>
            <nav>Menu link</nav>
            <main id="main-content">
              <h1>Useful article</h1>
              <p>Important body text.</p>
            </main>
            <aside>Related links</aside>
            <footer>Footer text</footer>
          </body>
        </html>
        """,
        "text/html",
    )

    assert text == "Useful article Important body text."


@pytest.mark.asyncio
async def test_read_url_excludes_content_named_boilerplate(hass) -> None:
    """Navigation ids/classes should not make boilerplate preferred content."""
    tool = read_url_module.ReadUrlTool(hass)

    text = await tool._extract_text(
        """
        <html>
          <body>
            <nav class="content-navigation">Menu link</nav>
            <section>
              <h1>Useful article</h1>
              <p>Important body text.</p>
            </section>
          </body>
        </html>
        """,
        "text/html",
    )

    assert text == "Useful article Important body text."


@pytest.mark.asyncio
async def test_read_url_unwinds_malformed_preferred_markup(hass) -> None:
    """Malformed optional tags should not leak page chrome into main content."""
    tool = read_url_module.ReadUrlTool(hass)

    text = await tool._extract_text(
        """
        <html>
          <body>
            <main>
              <ul>
                <li>Useful item one
                <li>Useful item two
              </main>
            <footer>Footer text</footer>
            <section>Related link</section>
          </body>
        </html>
        """,
        "text/html",
    )

    assert text == "Useful item one Useful item two"


@pytest.mark.asyncio
async def test_read_url_summary_keeps_longer_excerpt(hass, monkeypatch) -> None:
    """Summary mode should keep a useful excerpt instead of stopping at 1000 chars."""
    long_text = "x" * 1200
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain"},
            text=long_text,
        )
    )

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
    tool = read_url_module.ReadUrlTool(hass)
    _allow_public_example_resolution(tool)

    result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/long", "summary": True},
    )

    assert "Length: 1200 chars" in result["content"][0]["text"]
    assert long_text in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_read_url_cleans_wikipedia_page_chrome(hass) -> None:
    """Wikipedia HTML fallback should remove common article chrome."""
    tool = read_url_module.ReadUrlTool(hass)
    response = _FakeResponse(
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="""
        <html>
          <head><title>Example - Wikipedia</title></head>
          <body>
            <main id="mw-content-text">
              <p>From Wikipedia, the free encyclopedia</p>
              <p>Contents hide</p>
              <p>Useful encyclopedia article text. [edit]</p>
            </main>
          </body>
        </html>
        """,
    )

    result = await tool._format_response(
        response,
        urlparse("https://en.wikipedia.org/wiki/Example"),
        "https://en.wikipedia.org/wiki/Example",
        summary_only=False,
    )

    result_text = result["content"][0]["text"]
    assert "Useful encyclopedia article text." in result_text
    assert "From Wikipedia" not in result_text
    assert "Contents hide" not in result_text
    assert "[edit]" not in result_text


def test_read_url_wikipedia_detection_requires_real_wikipedia_host(hass) -> None:
    """Wikipedia cleanup should not trigger for lookalike hostnames."""
    tool = read_url_module.ReadUrlTool(hass)

    assert tool._is_wikipedia_url(urlparse("https://en.wikipedia.org/wiki/Example"))
    assert not tool._is_wikipedia_url(urlparse("https://evilwikipedia.org/wiki/Example"))
    assert not tool._is_wikipedia_url(urlparse("https://wikipedia.org.example/wiki/Example"))


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

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
    tool = read_url_module.ReadUrlTool(hass)
    _allow_public_example_resolution(tool)

    result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/page", "summary": False},
    )

    assert "📖 **Example Page**" in result["content"][0]["text"]
    assert "Hello world" in result["content"][0]["text"]
    assert fake_session.calls[0][0] == "https://example.com/page"
    assert "Host" not in fake_session.calls[0][1]["headers"]
    assert "server_hostname" not in fake_session.calls[0][1]
    assert isinstance(
        fake_session.client_session_kwargs[0]["connector"],
        read_url_module.aiohttp.TCPConnector,
    )


@pytest.mark.asyncio
async def test_read_url_debug_logs_shape_without_sensitive_query(
    hass, monkeypatch, caplog
) -> None:
    """Read URL debug logs should not include raw paths or query values."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain"},
            text="safe response",
        )
    )
    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )

    tool = read_url_module.ReadUrlTool(hass)
    _allow_public_example_resolution(tool)

    with caplog.at_level(logging.DEBUG, logger=read_url_module._LOGGER.name):
        result = await tool.handle_call(
            "read_url",
            {"url": "https://example.com/private/path?api_key=secret-token"},
        )

    assert "safe response" in result["content"][0]["text"]
    assert "scheme=https" in caplog.text
    assert "host=example.com" in caplog.text
    assert "query_present=True" in caplog.text
    assert "private/path" not in caplog.text
    assert "api_key" not in caplog.text
    assert "secret-token" not in caplog.text


@pytest.mark.asyncio
async def test_read_url_pins_all_validated_dns_addresses(hass) -> None:
    """Validated hostnames should keep every safe address for connector fallback."""
    tool = read_url_module.ReadUrlTool(hass)

    async def _resolve_host_addresses(host: str, port: int):
        assert host == "example.com"
        assert port == 443
        return {
            ipaddress.ip_address("93.184.216.34"),
            ipaddress.ip_address("2606:2800:220:1:248:1893:25c8:1946"),
        }

    tool._resolve_host_addresses = _resolve_host_addresses

    target = await tool._validate_fetchable_url("https://example.com/page")
    resolver = read_url_module._PinnedHostResolver(
        "example.com",
        target.resolved_addresses,
    )

    resolved = await resolver.resolve("example.com", 443, socket.AF_UNSPEC)

    assert target.request_url == "https://example.com/page"
    assert {item["host"] for item in resolved} == {
        "93.184.216.34",
        "2606:2800:220:1:248:1893:25c8:1946",
    }


@pytest.mark.asyncio
async def test_read_url_rejects_invalid_urls_and_timeouts(hass, monkeypatch) -> None:
    """Read URL should fail cleanly for invalid URLs and request timeouts."""
    tool = read_url_module.ReadUrlTool(hass)

    invalid = await tool.handle_call("read_url", {"url": "notaurl"})
    assert invalid["content"][0]["text"] == "❌ Invalid URL format"

    unsupported = await tool.handle_call("read_url", {"url": "ftp://example.com"})
    assert unsupported["content"][0]["text"] == "❌ Unsupported URL scheme: ftp"

    fake_session = _FakeSession(error=asyncio.TimeoutError())

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
    _allow_public_example_resolution(tool)
    timed_out = await tool.handle_call("read_url", {"url": "https://example.com"})

    assert timed_out["content"][0]["text"] == "❌ Timeout reading URL"


@pytest.mark.asyncio
async def test_read_url_reports_dns_resolution_failures(hass) -> None:
    """DNS resolver errors should become tool error text."""
    tool = read_url_module.ReadUrlTool(hass)

    async def _raise_dns_error(host: str, port: int):
        assert host == "missing.example"
        assert port == 443
        raise socket.gaierror("name not known")

    tool._resolve_host_addresses = _raise_dns_error

    result = await tool.handle_call("read_url", {"url": "https://missing.example"})

    assert result["content"][0]["text"] == "❌ Unable to resolve URL host: missing.example"


@pytest.mark.asyncio
async def test_read_url_reports_dns_resolution_timeouts(hass) -> None:
    """DNS resolver timeouts should become tool error text."""
    tool = read_url_module.ReadUrlTool(hass)

    async def _raise_timeout(host: str, port: int):
        assert host == "slow.example"
        assert port == 443
        raise asyncio.TimeoutError

    tool._resolve_host_addresses = _raise_timeout

    result = await tool.handle_call("read_url", {"url": "https://slow.example"})

    assert result["content"][0]["text"] == "❌ Timed out resolving URL host: slow.example"


@pytest.mark.asyncio
async def test_read_url_blocks_local_and_private_urls_before_fetch(
    hass,
    monkeypatch,
) -> None:
    """Read URL should not fetch local/private network targets by default."""
    tool = read_url_module.ReadUrlTool(hass)

    def _client_session(**kwargs):
        del kwargs
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

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
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

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
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

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
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
        _read_url_client_session(streamed_session),
    )

    streamed_result = await tool.handle_call(
        "read_url",
        {"url": "https://example.com/chunked"},
    )
    assert streamed_result["content"][0]["text"] == "❌ URL response is too large to read safely"


@pytest.mark.asyncio
async def test_read_url_reads_chunked_body_until_eof(hass, monkeypatch) -> None:
    """Chunked responses should be read until EOF within the size limit."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain"},
            content_chunks=[b"hello ", b"world", b""],
        )
    )

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
    tool = read_url_module.ReadUrlTool(hass)
    _allow_public_example_resolution(tool)

    result = await tool.handle_call("read_url", {"url": "https://example.com/chunked"})

    assert "hello world" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_read_url_rejects_oversized_chunked_body_after_multiple_reads(
    hass,
    monkeypatch,
) -> None:
    """Chunked responses should not bypass the size limit with a small first chunk."""
    fake_session = _FakeSession(
        response=_FakeResponse(
            headers={"Content-Type": "text/plain"},
            content_chunks=[b"abc", b"def", b""],
        )
    )

    monkeypatch.setattr(
        read_url_module.aiohttp,
        "ClientSession",
        _read_url_client_session(fake_session),
    )
    tool = read_url_module.ReadUrlTool(hass)
    tool.max_response_bytes = 5
    _allow_public_example_resolution(tool)

    result = await tool.handle_call("read_url", {"url": "https://example.com/chunked"})

    assert result["content"][0]["text"] == "❌ URL response is too large to read safely"
