"""Trusted native dispatch identity and current Home Assistant access checks."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

from homeassistant.auth.permissions.const import POLICY_CONTROL, POLICY_READ
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context

from .const import (
    CONF_CONTROL_HA, CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_CONTROL_HA, DEFAULT_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS, DOMAIN,
)

TRUSTED_CALL_CONTEXT_API_VERSION = 1


class ToolAccessDenied(PermissionError):
    """The requested identity or access is not authorized."""
    code = "access_denied"

    def __init__(self):
        super().__init__(self.code)


class ToolAccessUnavailable(Exception):
    """A dependency is temporarily unavailable."""
    code = "dependency_unavailable"

    def __init__(self):
        super().__init__(self.code)


@dataclass(frozen=True)
class TrustedCallContext:
    user_id: str
    profile_entry_id: str
    dispatch_id: str
    ha_context: Context


@dataclass
class _DispatchScope:
    entry: object
    tool_name: str
    context: Context | None
    dispatch_id: str
    active: bool = True
    acquired: bool = False


ACTIVE_LOADER_TOOL: ContextVar[str | None] = ContextVar("mcp_assist_active_loader_tool", default=None)

REQUEST_TRUSTED_SCOPE: ContextVar[_DispatchScope | None] = ContextVar(
    "mcp_assist_trusted_scope", default=None
)


def new_dispatch_scope(entry, tool_name, context):
    return _DispatchScope(entry, tool_name, context, str(uuid4()))


def get_trusted_call_context(hass) -> TrustedCallContext:
    """Return native identity only during its opted-in dispatch's lifetime."""
    scope = REQUEST_TRUSTED_SCOPE.get()
    if scope is None or not scope.active:
        raise ToolAccessDenied()
    context = scope.context
    if (not isinstance(context, Context) or not context.user_id
            or hass.config_entries.async_get_entry(scope.entry.entry_id) is not scope.entry):
        raise ToolAccessDenied()
    result = TrustedCallContext(context.user_id, scope.entry.entry_id, scope.dispatch_id, context)
    if not scope.active:
        raise ToolAccessDenied()
    scope.acquired = True
    return result


async def async_check_profile_entity_access(
    hass, *, user_id: str, profile_entry_id: str, entity_ids=(),
    permission: str = "control", package_id: str | None = None,
) -> Context:
    """Recheck live profile, user, exposure, and HA permissions."""
    if (not isinstance(user_id, str) or not 0 < len(user_id) <= 128
            or not isinstance(profile_entry_id, str) or not 0 < len(profile_entry_id) <= 128
            or permission not in ("control", "read") or not isinstance(entity_ids, (list, tuple))
            or (package_id is not None and (not isinstance(package_id, str) or not package_id))):
        raise ToolAccessDenied()
    entry = hass.config_entries.async_get_entry(profile_entry_id)
    if entry is None or entry.domain != DOMAIN or entry.disabled_by is not None:
        raise ToolAccessDenied()
    if entry.state is not ConfigEntryState.LOADED:
        raise ToolAccessUnavailable()
    if permission == "control" and not entry.options.get(
        CONF_CONTROL_HA, entry.data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA)
    ):
        raise ToolAccessDenied()
    if not entry.options.get(CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                             entry.data.get(CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                                            DEFAULT_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS)):
        raise ToolAccessDenied()
    server = hass.data.get(DOMAIN, {}).get("shared_mcp_server")
    loader = getattr(server, "tools", None)
    enabled = getattr(loader, "external_tools_enabled", None)
    package_enabled = getattr(loader, "is_external_package_enabled", None)
    ready = getattr(loader, "external_tools_ready", None)
    if (not callable(enabled) or not callable(ready)
            or (package_id is not None and not callable(package_enabled))):
        raise ToolAccessUnavailable()
    if enabled() is not True:
        raise ToolAccessDenied()
    if ready() is not True:
        raise ToolAccessUnavailable()
    if package_id is not None:
        package_status = package_enabled(package_id)
        if package_status is None:
            raise ToolAccessUnavailable()
        if package_status is not True:
            raise ToolAccessDenied()
    user = await hass.auth.async_get_user(user_id)
    if user is None or not user.is_active:
        raise ToolAccessDenied()
    policy = POLICY_CONTROL if permission == "control" else POLICY_READ
    for entity_id in entity_ids:
        if (not isinstance(entity_id, str) or not 0 < len(entity_id) <= 255
                or not async_should_expose(hass, "conversation", entity_id)
                or not user.permissions.check_entity(entity_id, policy)):
            raise ToolAccessDenied()
        if hass.states.get(entity_id) is None:
            raise ToolAccessUnavailable()
    return Context(user_id=user_id)
