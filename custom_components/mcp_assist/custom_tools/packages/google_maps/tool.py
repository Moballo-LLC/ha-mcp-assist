"""Built-in Google Places and Routes package wrapper."""

from __future__ import annotations

from typing import Any

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool
from custom_components.mcp_assist.custom_tools.google_maps import GoogleMapsTool


class GoogleMapsPackageTool(MCPAssistExternalTool):
    """Expose Google Places and Routes through the package API."""

    def __init__(self, hass, manifest, tool_dir) -> None:
        """Initialize the wrapper and delegated Google Maps tool bundle."""
        super().__init__(hass, manifest, tool_dir)
        self._delegate = GoogleMapsTool(hass)

    async def initialize(self) -> None:
        """Initialize the delegated Google Maps tool bundle."""
        await self._delegate.initialize()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the delegated Google Maps tool definitions."""
        return self._delegate.get_tool_definitions()

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Delegate Google Maps tool calls."""
        return await self._delegate.handle_call(tool_name, arguments)
