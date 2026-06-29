"""Read URL custom tool for MCP Assist."""
import asyncio
import aiohttp
import html as html_lib
import ipaddress
import logging
import socket
from typing import Dict, Any, List
from urllib.parse import urljoin, urlparse

_LOGGER = logging.getLogger(__name__)

_HTTP_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}


class ReadUrlTool:
    """Tool to read and extract content from URLs."""

    def __init__(self, hass):
        """Initialize Read URL tool."""
        self.hass = hass
        self.max_content_length = 50000  # Max characters to return
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

        _LOGGER.debug(f"Reading URL: {url}")

        try:
            current_url = await self._validate_fetchable_url(url)
        except ValueError as err:
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ {err}"
                }]
            }

        parsed = urlparse(current_url)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; mcp-assist/1.0)"
            }

            async with aiohttp.ClientSession() as session:
                for _redirect_count in range(self.max_redirects + 1):
                    async with session.get(
                        current_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15),
                        allow_redirects=False
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
                                current_url = await self._validate_fetchable_url(
                                    urljoin(current_url, location)
                                )
                            except ValueError as err:
                                return {
                                    "content": [{
                                        "type": "text",
                                        "text": f"❌ Unsafe redirect blocked: {err}"
                                    }]
                                }
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
        if 'text/html' not in content_type and 'text/plain' not in content_type:
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

        text = await self._extract_text(html_text, content_type)

        title = parsed.netloc
        lower_html = html_text.lower()
        if '<title>' in lower_html and '</title>' in lower_html:
            title_start = lower_html.index('<title>') + 7
            title_end = lower_html.index('</title>', title_start)
            title = html_lib.unescape(html_text[title_start:title_end]).strip()

        truncated = False
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "..."
            truncated = True

        if summary_only and len(text) > 1000:
            text = text[:1000] + "..."

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

        raw_body = await content.read(self.max_response_bytes + 1)
        if len(raw_body) > self.max_response_bytes:
            raise ValueError("URL response is too large to read safely")

        charset = getattr(response, "charset", None) or "utf-8"
        return raw_body.decode(charset, errors="replace")

    async def _validate_fetchable_url(self, url: Any) -> str:
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
            return sanitized_url

        host = parsed.hostname
        if not host:
            raise ValueError("Invalid URL host")

        normalized_host = host.rstrip(".").casefold()
        if normalized_host in _LOCAL_HOSTNAMES or normalized_host.endswith(".localhost"):
            raise ValueError("Local URLs must be allowlisted in Home Assistant")

        try:
            ip_address = ipaddress.ip_address(normalized_host)
        except ValueError:
            await self._validate_resolved_host_addresses(
                normalized_host,
                parsed.port or (443 if parsed.scheme == "https" else 80),
            )
        else:
            if self._is_private_or_local_address(ip_address):
                raise ValueError("Private or local URLs must be allowlisted in Home Assistant")

        return sanitized_url

    def _is_allowlisted_url(self, url: str) -> bool:
        """Return whether Home Assistant explicitly allows this external URL."""
        is_allowed_external_url = getattr(self.hass.config, "is_allowed_external_url", None)
        if not callable(is_allowed_external_url):
            return False
        try:
            return bool(is_allowed_external_url(url))
        except Exception as err:
            _LOGGER.debug("Unable to evaluate external URL allowlist for %s: %s", url, err)
            return False

    async def _validate_resolved_host_addresses(self, host: str, port: int) -> None:
        """Resolve a hostname and reject private or local network addresses."""
        addresses = await asyncio.wait_for(
            self._resolve_host_addresses(host, port),
            timeout=5,
        )
        if not addresses:
            raise ValueError(f"Unable to resolve URL host: {host}")
        if any(self._is_private_or_local_address(address) for address in addresses):
            raise ValueError("Private or local URLs must be allowlisted in Home Assistant")

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

    async def _extract_text(self, html: str, content_type: str) -> str:
        """Extract text from HTML without BeautifulSoup."""
        if 'text/plain' in content_type:
            return html

        import re

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

        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

        html = re.sub(r'<[^>]+>', ' ', html)

        html = html_lib.unescape(html)

        lines = html.split('\n')
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        text = '\n'.join(lines)

        text = re.sub(r'\s+', ' ', text)

        return text.strip()
