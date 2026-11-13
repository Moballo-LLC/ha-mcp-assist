"""Lifetime-bound Home Assistant identity for direct in-process bridge calls."""
from contextvars import ContextVar
from dataclasses import dataclass

from homeassistant.core import Context


@dataclass
class HAContextScope:
    """Shared revocation state inherited by child tasks."""
    context: Context
    tool_name: str
    active: bool = True


REQUEST_HA_CONTEXT: ContextVar[HAContextScope | None] = ContextVar(
    "mcp_assist_ha_context", default=None
)


def current_ha_context() -> Context:
    """Return native identity only while the direct bridge dispatch is active."""
    scope = REQUEST_HA_CONTEXT.get()
    return scope.context if scope is not None and scope.active else Context()
