"""Built-in Wikipedia search package wrapper."""

from __future__ import annotations

from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool
from custom_components.mcp_assist.custom_tools.wikipedia_search import WikipediaSearchTool


class WikipediaSearchPackageTool(MCPAssistExternalTool):
    """Expose Wikipedia search through the package API."""

    def __init__(self, hass, manifest, tool_dir) -> None:
        """Initialize the wrapper and delegated Wikipedia search tool."""
        super().__init__(hass, manifest, tool_dir)
        self._delegate = WikipediaSearchTool(hass)

    async def initialize(self) -> None:
        """Initialize the delegated Wikipedia search tool."""
        await self._delegate.initialize()

    def get_tool_definitions(self) -> list[dict]:
        """Return the delegated Wikipedia search tool definition."""
        return self._delegate.get_tool_definitions()

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """Delegate Wikipedia search tool calls."""
        return await self._delegate.handle_call(tool_name, arguments)
