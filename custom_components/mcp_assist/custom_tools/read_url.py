"""Read URL custom tool for MCP Assist."""
import asyncio
import aiohttp
from dataclasses import dataclass
import html as html_lib
from html.parser import HTMLParser
import ipaddress
import logging
import re
import socket
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

_LOGGER = logging.getLogger(__name__)

_HTTP_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}
_MARKDOWN_ESCAPE_REPLACEMENTS = {
    r"\_": "_",
    r"\*": "*",
    r"\[": "[",
    r"\]": "]",
    r"\(": "(",
    r"\)": ")",
}


def _clean_markdown_escapes(text: str) -> str:
    """Clean common escaped markdown characters from prose text."""
    for source, replacement in _MARKDOWN_ESCAPE_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text


@dataclass(frozen=True)
class _FetchTarget:
    """Validated URL plus request details pinned to a checked address."""

    display_url: str
    request_url: str
    resolved_addresses: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...] = ()


class _PinnedHostResolver(aiohttp.abc.AbstractResolver):
    """aiohttp resolver that only returns already-validated addresses."""

    def __init__(
        self,
        hostname: str,
        addresses: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...],
    ) -> None:
        """Initialize the resolver."""
        self._hostname = hostname.rstrip(".").casefold()
        self._addresses = addresses

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: int = socket.AF_UNSPEC,
    ) -> list[dict[str, Any]]:
        """Resolve only the expected host to its validated addresses."""
        if host.rstrip(".").casefold() != self._hostname:
            raise OSError(f"Unexpected host for pinned resolver: {host}")

        results: list[dict[str, Any]] = []
        for address in self._addresses:
            address_family = (
                socket.AF_INET6
                if isinstance(address, ipaddress.IPv6Address)
                else socket.AF_INET
            )
            if family not in (socket.AF_UNSPEC, 0, address_family):
                continue
            results.append(
                {
                    "hostname": host,
                    "host": str(address),
                    "port": port,
                    "family": address_family,
                    "proto": 0,
                    "flags": socket.AI_NUMERICHOST,
                }
            )

        if not results:
            raise OSError(f"No validated addresses available for {host}")
        return results

    async def close(self) -> None:
        """Close resolver resources."""
        return None


class ReadUrlTool:
    """Tool to read and extract content from URLs."""

    SUMMARY_MAX_CHARS = 6000
    MAX_CONTENT_CHARS = 50000
    WIKIPEDIA_LEAD_MARKERS = (
        "from wikipedia, the free encyclopedia",
        "aus wikipedia, der freien enzyklopädie",
    )

    def __init__(self, hass):
        """Initialize Read URL tool."""
        self.hass = hass
        self.max_content_length = self.MAX_CONTENT_CHARS
        self.summary_max_length = self.SUMMARY_MAX_CHARS
        self.max_response_bytes = 512000  # Max response bytes to read before decoding
        self.max_redirects = 5

    async def initialize(self):
        """Initialize the tool."""
        pass  # No logging needed

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name == "read_url"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definition for Read URL."""
        return [{
            "name": "read_url",
            "description": "Read and extract text content from a webpage URL",
            "llmDescription": "Read webpage text from a URL.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to read"
                    },
                    "summary": {
                        "type": "boolean",
                        "description": "Return a summary instead of full content (default false)",
                        "default": False
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }]

    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read and extract content from URL."""
        url = arguments.get("url")
        summary_only = arguments.get("summary", False)

        try:
            current_target = await self._validate_fetchable_url(url)
        except ValueError as err:
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ {err}"
                }]
            }

        current_url = current_target.display_url
        parsed = urlparse(current_url)
        _LOGGER.debug(
            "Reading URL: scheme=%s host=%s path_chars=%d query_present=%s",
            parsed.scheme,
            parsed.hostname or "",
            len(parsed.path or ""),
            bool(parsed.query),
        )

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; mcp-assist/1.0)"
            }

            for _redirect_count in range(self.max_redirects + 1):
                request_kwargs = {
                    "headers": headers,
                    "timeout": aiohttp.ClientTimeout(total=15),
                    "allow_redirects": False,
                }
                session_kwargs: dict[str, Any] = {}
                connector = self._build_pinned_connector(current_target)
                if connector is not None:
                    session_kwargs["connector"] = connector

                async with aiohttp.ClientSession(**session_kwargs) as session:
                    async with session.get(
                        current_target.request_url,
                        **request_kwargs,
                    ) as response:
                        if response.status in _HTTP_REDIRECT_STATUSES:
                            location = str(response.headers.get("Location") or "").strip()
                            if not location:
                                return {
                                    "content": [{
                                        "type": "text",
                                        "text": "❌ Redirect response did not include a Location header"
                                    }]
                                }
                            try:
                                current_target = await self._validate_fetchable_url(
                                    urljoin(current_url, location)
                                )
                            except ValueError as err:
                                return {
                                    "content": [{
                                        "type": "text",
                                        "text": f"❌ Unsafe redirect blocked: {err}"
                                    }]
                                }
                            current_url = current_target.display_url
                            parsed = urlparse(current_url)
                            continue

                        return await self._format_response(
                            response,
                            parsed,
                            current_url,
                            summary_only,
                        )

            return {
                "content": [{
                    "type": "text",
                    "text": "❌ Too many redirects while reading URL"
                }]
            }

        except asyncio.TimeoutError:
            return {
                "content": [{
                    "type": "text",
                    "text": "❌ Timeout reading URL"
                }]
            }
        except Exception as e:
            _LOGGER.error(f"Read URL exception: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Error reading URL: {e!s}"
                }]
            }

    async def _format_response(
        self,
        response: Any,
        parsed,
        url: str,
        summary_only: bool,
    ) -> Dict[str, Any]:
        """Validate, read, and format a successful URL response."""
        if response.status != 200:
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ HTTP {response.status}: Failed to fetch URL"
                }]
            }

        content_type = response.headers.get('Content-Type', '')
        lower_content_type = content_type.lower()
        if 'text/html' not in lower_content_type and 'text/plain' not in lower_content_type:
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Unsupported content type: {content_type}"
                }]
            }

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_response_bytes:
                    return {
                        "content": [{
                            "type": "text",
                            "text": "❌ URL response is too large to read safely"
                        }]
                    }
            except ValueError:
                pass

        try:
            html_text = await self._read_limited_text(response)
        except ValueError as err:
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ {err}"
                }]
            }

        text = await self._extract_text(html_text, lower_content_type)
        if 'text/html' in lower_content_type:
            if self._is_wikipedia_url(parsed):
                text = self._trim_wikipedia_html_fallback(text)

        title = parsed.netloc
        lower_html = html_text.lower()
        if '<title>' in lower_html and '</title>' in lower_html:
            title_start = lower_html.index('<title>') + 7
            title_end = lower_html.index('</title>', title_start)
            title = html_text[title_start:title_end]
        title = self._clean_title(title)

        truncated = False
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "..."
            truncated = True

        if summary_only and len(text) > self.summary_max_length:
            text = text[:self.summary_max_length].rstrip() + "..."

        result_text = f"📖 **{title}**\n"
        result_text += f"URL: {url}\n"
        result_text += f"Length: {len(text)} chars"
        if truncated:
            result_text += " (truncated)"
        result_text += f"\n\n{text}"

        return {
            "content": [{
                "type": "text",
                "text": result_text
            }]
        }

    async def _read_limited_text(self, response: Any) -> str:
        """Read response text without allowing unbounded in-memory bodies."""
        content = getattr(response, "content", None)
        if content is None:
            return await response.text()

        chunks: list[bytes] = []
        total_size = 0
        while True:
            chunk = await content.read(self.max_response_bytes + 1 - total_size)
            if not chunk:
                break
            chunks.append(chunk)
            total_size += len(chunk)
            if total_size > self.max_response_bytes:
                raise ValueError("URL response is too large to read safely")

        raw_body = b"".join(chunks)
        if len(raw_body) > self.max_response_bytes:
            raise ValueError("URL response is too large to read safely")

        charset = getattr(response, "charset", None) or "utf-8"
        return raw_body.decode(charset, errors="replace")

    async def _validate_fetchable_url(self, url: Any) -> _FetchTarget:
        """Validate that a URL is safe for the URL reader to fetch."""
        raw_url = str(url or "").strip()
        parsed = urlparse(raw_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        if parsed.username or parsed.password:
            raise ValueError("URLs must not include embedded credentials")

        sanitized_url = parsed._replace(fragment="").geturl()
        if self._is_allowlisted_url(sanitized_url):
            return _FetchTarget(display_url=sanitized_url, request_url=sanitized_url)

        host = parsed.hostname
        if not host:
            raise ValueError("Invalid URL host")

        normalized_host = host.rstrip(".").casefold()
        if normalized_host in _LOCAL_HOSTNAMES or normalized_host.endswith(".localhost"):
            raise ValueError("Local URLs must be allowlisted in Home Assistant")

        try:
            ip_address = ipaddress.ip_address(normalized_host)
        except ValueError:
            addresses = await self._validated_resolved_host_addresses(
                normalized_host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
            )
            return _FetchTarget(
                display_url=sanitized_url,
                request_url=sanitized_url,
                resolved_addresses=addresses,
            )
        else:
            if self._is_private_or_local_address(ip_address):
                raise ValueError("Private or local URLs must be allowlisted in Home Assistant")

        return _FetchTarget(display_url=sanitized_url, request_url=sanitized_url)

    def _build_pinned_connector(self, target: _FetchTarget) -> aiohttp.TCPConnector | None:
        """Build a connector that pins DNS names to validated addresses."""
        if not target.resolved_addresses:
            return None

        host = urlparse(target.display_url).hostname
        if not host:
            return None

        return aiohttp.TCPConnector(
            resolver=_PinnedHostResolver(host, target.resolved_addresses),
            use_dns_cache=False,
        )

    def _is_allowlisted_url(self, url: str) -> bool:
        """Return whether Home Assistant explicitly allows this external URL."""
        is_allowed_external_url = getattr(self.hass.config, "is_allowed_external_url", None)
        if not callable(is_allowed_external_url):
            return False
        try:
            return bool(is_allowed_external_url(url))
        except Exception as err:
            parsed_url = urlparse(url)
            _LOGGER.debug(
                "Unable to evaluate external URL allowlist for scheme=%s host=%s path_chars=%d: %s",
                parsed_url.scheme,
                parsed_url.hostname or "",
                len(parsed_url.path or ""),
                err,
            )
            return False

    async def _validated_resolved_host_addresses(
        self,
        host: str,
        port: int,
    ) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        """Resolve a hostname and return checked public addresses."""
        try:
            addresses = await asyncio.wait_for(
                self._resolve_host_addresses(host, port),
                timeout=5,
            )
        except asyncio.TimeoutError as err:
            raise ValueError(f"Timed out resolving URL host: {host}") from err
        except OSError as err:
            raise ValueError(f"Unable to resolve URL host: {host}") from err
        if not addresses:
            raise ValueError(f"Unable to resolve URL host: {host}")
        if any(self._is_private_or_local_address(address) for address in addresses):
            raise ValueError("Private or local URLs must be allowlisted in Home Assistant")
        return tuple(sorted(addresses, key=str))

    async def _resolve_host_addresses(
        self,
        host: str,
        port: int,
    ) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        """Resolve a hostname to IP addresses without blocking the event loop."""
        loop = asyncio.get_running_loop()
        addrinfos = await loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for _family, _type, _proto, _canonname, sockaddr in addrinfos:
            try:
                addresses.add(ipaddress.ip_address(sockaddr[0]))
            except ValueError:
                continue
        return addresses

    @staticmethod
    def _is_private_or_local_address(
        address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        """Return whether an address points at local, private, or reserved space."""
        return any(
            (
                address.is_loopback,
                address.is_link_local,
                address.is_private,
                address.is_multicast,
                address.is_reserved,
                address.is_unspecified,
                not address.is_global,
            )
        )

    @staticmethod
    def _is_wikipedia_url(parsed: Any) -> bool:
        """Return whether a parsed URL points to a Wikipedia article page."""
        host = (parsed.hostname or "").lower()
        is_wikipedia_host = host == "wikipedia.org" or host.endswith(".wikipedia.org")
        return is_wikipedia_host and parsed.path.startswith("/wiki/")

    @staticmethod
    def _clean_title(title: str) -> str:
        """Strip lightweight markup from a page title."""
        if not title:
            return title
        return html_lib.unescape(re.sub(r"<[^>]+>", "", title)).strip()

    def _trim_wikipedia_html_fallback(self, text: str) -> str:
        """Remove common Wikipedia chrome when reading article HTML."""
        if not text:
            return text

        lower_text = text.lower()
        for marker in self.WIKIPEDIA_LEAD_MARKERS:
            marker_index = lower_text.find(marker)
            if marker_index != -1:
                text = text[marker_index + len(marker):].lstrip()
                break

        text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\[\s*Bearbeiten\s*\|\s*Quelltext bearbeiten\s*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\bContents\s+hide\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bInhaltsverzeichnis\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    async def _extract_text(self, html: str, content_type: str) -> str:
        """Extract text from HTML without BeautifulSoup."""
        if 'text/plain' in content_type:
            return html

        try:
            parser = _MainContentParser()
            parser.feed(html)
            parser.close()
            text = parser.get_text()
            if text:
                return text
        except Exception as err:
            _LOGGER.debug("HTML main-content parsing failed: %s", err)

        return self._strip_tags_fallback(html)

    @staticmethod
    def _strip_tags_fallback(html: str) -> str:
        """Fallback HTML tag stripping when parser extraction fails."""
        html = re.sub(
            r"<script\b[^>]*>.*?</script\b[^>]*>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(
            r"<style\b[^>]*>.*?</style\b[^>]*>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        html = re.sub(r"<[^>]+>", " ", html)
        html = html_lib.unescape(html)
        return re.sub(r"\s+", " ", html).strip()


class _MainContentParser(HTMLParser):
    """Lightweight HTML parser that favors article and main content."""

    _SKIP_TAGS = {"head", "noscript", "script", "style", "title"}
    _BOILERPLATE_TAGS = {"nav", "header", "footer", "aside"}
    _LITERAL_TAGS = {"code", "kbd", "pre", "samp"}
    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    _BLOCK_TAGS = {
        "article",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
    }
    _PREFERRED_TAGS = {"article", "main"}
    _PREFERRED_KEYWORDS = {
        "article",
        "bodycontent",
        "content",
        "entry-content",
        "main-content",
        "mw-content-text",
        "post",
    }

    def __init__(self) -> None:
        """Initialize parser state."""
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._boilerplate_depth = 0
        self._preferred_depth = 0
        self._literal_depth = 0
        self._tag_stack: List[Tuple[str, bool, bool, bool, bool]] = []
        self._preferred_parts: List[Tuple[str, bool]] = []
        self._body_parts: List[Tuple[str, bool]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        """Track whether a tag starts skippable, boilerplate, or preferred content."""
        tag = tag.lower()
        if tag in self._VOID_TAGS:
            if tag == "br":
                self._append_separator()
            return

        is_skip = tag in self._SKIP_TAGS
        is_boilerplate = tag in self._BOILERPLATE_TAGS
        is_literal = tag in self._LITERAL_TAGS
        is_preferred = (
            not is_skip
            and not is_boilerplate
            and self._is_preferred_container(tag, attrs)
        )

        self._tag_stack.append((tag, is_skip, is_boilerplate, is_preferred, is_literal))

        if is_skip:
            self._skip_depth += 1
        if is_boilerplate:
            self._boilerplate_depth += 1
        if is_preferred:
            self._preferred_depth += 1
        if is_literal:
            self._literal_depth += 1

    def handle_startendtag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        """Handle self-closing tags."""
        tag = tag.lower()
        if tag in self._VOID_TAGS:
            if tag == "br":
                self._append_separator()
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        """Leave tracked tag scopes."""
        tag = tag.lower()
        matching_index = -1
        for index in range(len(self._tag_stack) - 1, -1, -1):
            if self._tag_stack[index][0] == tag:
                matching_index = index
                break

        if matching_index >= 0:
            while len(self._tag_stack) > matching_index:
                self._pop_tag_scope()

        if tag in self._BLOCK_TAGS:
            self._append_separator()

    def handle_data(self, data: str) -> None:
        """Collect visible page text."""
        if self._skip_depth > 0 or self._boilerplate_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        is_literal = self._literal_depth > 0
        if self._preferred_depth > 0:
            self._preferred_parts.append((text, is_literal))
        elif self._boilerplate_depth == 0:
            self._body_parts.append((text, is_literal))

    def get_text(self) -> str:
        """Return preferred main content, falling back to visible body text."""
        preferred = self._normalize_text(self._preferred_parts)
        if preferred:
            return preferred
        return self._normalize_text(self._body_parts)

    def _append_separator(self) -> None:
        """Add a loose text separator."""
        if self._skip_depth > 0 or self._boilerplate_depth > 0:
            return
        if self._preferred_depth > 0:
            self._preferred_parts.append((" ", False))
        elif self._boilerplate_depth == 0:
            self._body_parts.append((" ", False))

    def _pop_tag_scope(self) -> None:
        """Pop one tracked tag scope and update active depths."""
        _, is_skip, is_boilerplate, is_preferred, is_literal = self._tag_stack.pop()
        if is_skip and self._skip_depth > 0:
            self._skip_depth -= 1
        if is_boilerplate and self._boilerplate_depth > 0:
            self._boilerplate_depth -= 1
        if is_preferred and self._preferred_depth > 0:
            self._preferred_depth -= 1
        if is_literal and self._literal_depth > 0:
            self._literal_depth -= 1

    def _is_preferred_container(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> bool:
        """Return whether a container likely holds main article content."""
        if tag in self._PREFERRED_TAGS:
            return True

        attr_map = {key.lower(): (value or "") for key, value in attrs}
        if attr_map.get("role", "").lower() == "main":
            return True

        attribute_values = {
            value.casefold()
            for key in ("id", "class")
            for value in attr_map.get(key, "").split()
            if value
        }
        return bool(attribute_values & self._PREFERRED_KEYWORDS)

    @staticmethod
    def _normalize_text(parts: List[Tuple[str, bool]]) -> str:
        """Collapse collected text fragments into a single readable string."""
        if not parts:
            return ""
        processed_parts = [
            text if is_literal else _clean_markdown_escapes(text)
            for text, is_literal in parts
        ]
        return re.sub(r"\s+", " ", " ".join(processed_parts)).strip()
