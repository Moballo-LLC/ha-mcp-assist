"""Trusted, task-local Home Assistant identity for in-process tool dispatch."""
from contextvars import ContextVar

from homeassistant.core import Context

REQUEST_HA_CONTEXT: ContextVar[Context | None] = ContextVar("mcp_assist_ha_context", default=None)


def current_ha_context() -> Context:
    """External MCP payloads never supply a Home Assistant user identity."""
    return REQUEST_HA_CONTEXT.get() or Context()
