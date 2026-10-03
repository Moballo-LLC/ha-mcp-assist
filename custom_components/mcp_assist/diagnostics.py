"""On-demand diagnostics built from allowlisted configuration and cached metadata."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant

from . import const
from .localization import get_language_instruction

_PROVIDERS = frozenset(
    value for name, value in vars(const).items() if name.startswith("SERVER_TYPE_")
)
_CONTEXT_MODES = frozenset({"standard", "adaptive", "light"})
_MAX_CACHED_TOOLS = 1000
_MAX_CACHED_SCHEMA_BYTES = 256 * 1024
_MAX_CACHED_SCHEMA_NODES = 8192
_MAX_CACHED_SCHEMA_DEPTH = 32


class _CacheSizeLimit(ValueError):
    """Cached schemas exceed the diagnostic measurement budget."""


def _check_schema_budget(value: Any) -> None:
    """Bound traversal and a conservative encoded size before serialization."""
    remaining_bytes = _MAX_CACHED_SCHEMA_BYTES
    remaining_nodes = _MAX_CACHED_SCHEMA_NODES

    def visit(item: Any, depth: int) -> None:
        nonlocal remaining_bytes, remaining_nodes
        remaining_nodes -= 1
        if remaining_nodes < 0 or depth > _MAX_CACHED_SCHEMA_DEPTH:
            raise _CacheSizeLimit
        if type(item) is str:
            cost = 2 + 6 * len(item)
        elif type(item) is int:
            cost = 2 + item.bit_length() // 3
        elif type(item) in (dict, list):
            children = len(item) * (2 if type(item) is dict else 1)
            if children > remaining_nodes:
                raise _CacheSizeLimit
            cost = 2 + max(0, len(item) - 1) + (len(item) if type(item) is dict else 0)
        elif type(item) is float:
            cost = 32
        elif item is None or type(item) is bool:
            cost = 5
        else:
            raise TypeError
        remaining_bytes -= cost
        if remaining_bytes < 0:
            raise _CacheSizeLimit
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise TypeError
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) is list:
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)


def _configured(entry: ConfigEntry, key: str, default: Any = None) -> Any:
    return entry.options.get(key, entry.data.get(key, default))


def _choice(value: Any, choices: frozenset[str]) -> str:
    return value if isinstance(value, str) and value in choices else "unknown"


def _selection(entry: ConfigEntry, profile_key: str, model_key: str) -> str:
    if _configured(entry, profile_key):
        return "profile"
    return "explicit" if _configured(entry, model_key) else "unset"


def _prompt_mode(entry: ConfigEntry, mode_key: str, prompt_key: str, default_prompt: str) -> str:
    """Infer legacy prompt modes using the same rules as the conversation agent."""
    explicit_mode = _configured(entry, mode_key)
    if explicit_mode in (const.PROMPT_MODE_DEFAULT, const.PROMPT_MODE_CUSTOM):
        return explicit_mode
    stored_prompt = _configured(entry, prompt_key)
    return (
        const.PROMPT_MODE_DEFAULT if stored_prompt in (None, "", default_prompt)
        else const.PROMPT_MODE_CUSTOM
    )


def _cached_tools(agent: Any) -> dict[str, Any]:
    tools = getattr(agent, "_cached_profile_mcp_tools", None)
    if not isinstance(tools, list):
        return {"status": "not_built"}
    if len(tools) > _MAX_CACHED_TOOLS:
        return {"status": "too_large", "count": len(tools)}
    try:
        _check_schema_budget(tools)
        size = len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode())
    except _CacheSizeLimit:
        return {"status": "too_large", "count": len(tools)}
    except (TypeError, ValueError, OverflowError, RecursionError):
        return {"status": "unavailable"}
    return {"status": "cached", "count": len(tools), "schema_bytes": size}


def _resolution(agent: Any, selection: str, attribute: str) -> str:
    if selection != "profile":
        return "not_applicable"
    if agent is None:
        return "runtime_unavailable"
    return "resolved" if isinstance(getattr(agent, attribute, None), Mapping) else "unresolved"


def build_assist_diagnostics(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    """Project metadata without discovery, model calls, or configuration contents.

    External diagnostic tools can reuse this helper. An unresolved profile can be
    normal before its first request; resolution is not proof of provider success.
    """
    domain_data = hass.data.get(const.DOMAIN, {})
    if not isinstance(domain_data, Mapping):
        domain_data = {}
    runtime = domain_data.get(entry.entry_id, {})
    agent = runtime.get("agent") if isinstance(runtime, Mapping) else None
    if getattr(agent, "entry", None) is not entry:
        agent = None
    shared = entry.unique_id == const.SYSTEM_ENTRY_UNIQUE_ID
    result: dict[str, Any] = {
        "schema_version": 1,
        "entry_kind": "system" if shared else "profile",
        "loaded": entry.state is ConfigEntryState.LOADED,
        "disabled": entry.disabled_by is not None,
        "boundary": "configuration_and_cached_metadata_not_provider_or_device_health",
    }
    if shared:
        result["server_runtime_present"] = domain_data.get("shared_mcp_server") is not None
        result["external_tools_enabled"] = _configured(
            entry, const.CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
            const.DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
        ) is True
        result["bearer_auth_configured"] = bool(_configured(entry, const.CONF_MCP_BEARER_TOKEN))
        return result
    provider = _choice(
        _configured(entry, const.CONF_SERVER_TYPE, const.DEFAULT_SERVER_TYPE), _PROVIDERS
    )
    model_selection = (
        _selection(entry, const.CONF_MODEL_PROFILE, const.CONF_MODEL_NAME)
        if provider == const.SERVER_TYPE_OPENAI
        else ("explicit" if _configured(entry, const.CONF_MODEL_NAME) else "unset")
    )
    result.update(
        provider=provider,
        agent_runtime_present=agent is not None,
        context_mode=_choice(
            _configured(entry, const.CONF_CONTEXT_MODE, const.DEFAULT_CONTEXT_MODE), _CONTEXT_MODES
        ),
        model_selection=model_selection,
        model_profile_resolution=_resolution(agent, model_selection, "resolved_model_profile"),
        credential_configured=bool(_configured(
            entry,
            const.CONF_OPENCLAW_TOKEN
            if provider == const.SERVER_TYPE_OPENCLAW
            else const.CONF_API_KEY,
        )),
        cached_tools=_cached_tools(agent),
    )
    if provider == const.SERVER_TYPE_OPENAI:
        image_selection = _selection(
            entry, const.CONF_OPENAI_IMAGE_MODEL_PROFILE, const.CONF_OPENAI_IMAGE_MODEL
        )
        result["image_model_selection"] = image_selection
        result["image_profile_resolution"] = _resolution(
            agent, image_selection, "resolved_image_model_profile"
        )
    result["prompts"] = {
        label: {
            "mode": _prompt_mode(entry, mode_key, prompt_key, default_prompt),
            "configured_characters": len(value) if isinstance(value, str) else None,
        }
        for label, mode_key, prompt_key, default_prompt in (
            ("system", const.CONF_SYSTEM_PROMPT_MODE, const.CONF_SYSTEM_PROMPT,
             get_language_instruction(hass.config.language) or const.DEFAULT_SYSTEM_PROMPT),
            ("technical", const.CONF_TECHNICAL_PROMPT_MODE, const.CONF_TECHNICAL_PROMPT,
             const.DEFAULT_TECHNICAL_PROMPT),
        )
        for value in (_configured(entry, prompt_key),)
    }
    result["limits"] = {
        label: value if type(value) is int and 0 <= value <= 1_000_000 else None
        for label, key, default in (
            ("history", const.CONF_MAX_HISTORY, const.DEFAULT_MAX_HISTORY),
            ("iterations", const.CONF_MAX_ITERATIONS, const.DEFAULT_MAX_ITERATIONS),
            ("output_tokens", const.CONF_MAX_TOKENS, const.DEFAULT_MAX_TOKENS),
        )
        for value in (_configured(entry, key, default),)
    }
    return result


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return support metadata using Home Assistant's diagnostics download."""
    return build_assist_diagnostics(hass, entry)
