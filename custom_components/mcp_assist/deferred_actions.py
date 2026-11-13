"""Supported authorization boundary for deterministic deferred action consumers.

Only the in-process conversation dispatcher can supply the scheduling principal.
Stored IDs are reauthorized against live HA policy on every later dispatch.
This module never calls a service or grants generic administrator authority.
"""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
from typing import Any

from homeassistant.auth.permissions.const import POLICY_CONTROL, POLICY_READ
from homeassistant.components.homeassistant import async_should_expose
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context

from .const import (
    CONF_CONTROL_HA, CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_CONTROL_HA, DEFAULT_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS, DOMAIN,
)

DEFERRED_ACTION_API_VERSION = 1
REQUEST_DEFERRED_ACTION: ContextVar[tuple[Any, str, str, Context] | None] = ContextVar(
    "mcp_assist_deferred_action_request", default=None,
)
REQUEST_TOOL_CALL_ID: ContextVar[str | None] = ContextVar("mcp_assist_logical_tool_call", default=None)


class DeferredActionAuthorizationError(PermissionError):
    """No raw principal, profile, or target data in authorization errors."""

    def __init__(self) -> None:
        super().__init__("deferred_action_not_authorized")


class DeferredActionUnavailable(DeferredActionAuthorizationError):
    """A loaded dependency is temporarily unavailable; never grants permission."""

    code = "dependency_unavailable"


async def async_authorize_deferred_action(
    hass, owner: dict[str, str], entity_ids: list[str], *, tool_name: str,
    read_entity_ids: list[str] | None = None,
) -> Context:
    """Recheck a stored owner/profile, exposure and HA control permissions.

    A successful result is an HA service context for this check only. Consumers
    must invoke it again immediately before each service call, including after
    database waits; it is not a persistent capability or an effect receipt.
    """
    read_entity_ids = [] if read_entity_ids is None else read_entity_ids
    if (not isinstance(owner, dict) or not isinstance(entity_ids, list)
            or not isinstance(read_entity_ids, list) or len(entity_ids) + len(read_entity_ids) > 32
            or not isinstance(tool_name, str)):
        raise DeferredActionAuthorizationError()
    user_id, profile_id = owner.get("user_id"), owner.get("profile_id")
    if not all(isinstance(value, str) and 0 < len(value) <= 128 for value in (user_id, profile_id)):
        raise DeferredActionAuthorizationError()
    entry = hass.config_entries.async_get_entry(profile_id)
    if entry is None or entry.domain != DOMAIN or entry.disabled_by is not None:
        raise DeferredActionAuthorizationError()
    if entry.state is not ConfigEntryState.LOADED:
        raise DeferredActionUnavailable()
    if not entry.options.get(CONF_CONTROL_HA, entry.data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA)):
        raise DeferredActionAuthorizationError()
    if not entry.options.get(CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                             entry.data.get(CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                                            DEFAULT_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS)):
        raise DeferredActionAuthorizationError()
    server = hass.data.get(DOMAIN, {}).get("shared_mcp_server")
    loader = getattr(server, "tools", None)
    checker = getattr(loader, "requires_deferred_action_context", None)
    enabled = getattr(loader, "_external_custom_tools_enabled", None)
    if callable(enabled) and enabled() is not True:
        raise DeferredActionAuthorizationError()
    if not callable(checker) or checker(tool_name) is not True or not callable(enabled):
        raise DeferredActionUnavailable()
    user = await hass.auth.async_get_user(user_id)
    if user is None or not user.is_active:
        raise DeferredActionAuthorizationError()
    for entity_id, permission in [(item, POLICY_CONTROL) for item in entity_ids] + [(item, POLICY_READ) for item in read_entity_ids]:
        if (not isinstance(entity_id, str) or len(entity_id) > 255
                or not async_should_expose(hass, "conversation", entity_id)
                or not user.permissions.check_entity(entity_id, permission)):
            raise DeferredActionAuthorizationError()
        if hass.states.get(entity_id) is None:
            raise DeferredActionUnavailable()
    return Context(user_id=user_id)


async def async_get_deferred_action_principal(hass) -> dict[str, str]:
    """Get an authenticated creation identity without trusting MCP JSON fields."""
    scope = REQUEST_DEFERRED_ACTION.get()
    if scope is None:
        raise DeferredActionAuthorizationError()
    entry, tool_name, call_id, context = scope
    if (not isinstance(context, Context) or not context.user_id or not context.id
            or not isinstance(call_id, str) or not 0 < len(call_id) <= 200
            or hass.config_entries.async_get_entry(entry.entry_id) is not entry):
        raise DeferredActionAuthorizationError()
    owner = {"user_id": context.user_id, "profile_id": entry.entry_id}
    await async_authorize_deferred_action(hass, owner, [], tool_name=tool_name)
    creation_key = hashlib.sha256((context.id + ":" + call_id).encode()).hexdigest()
    return {**owner, "creation_key": creation_key}
