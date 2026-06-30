"""Config flow for MCP Assist integration."""

from __future__ import annotations

import ipaddress
import logging
import re
import secrets
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TemplateSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .tools.builtin_catalog import (
    BuiltInToolToggleSpec,
    get_builtin_profile_setting_value,
    get_builtin_shared_setting_value,
    load_builtin_tool_toggle_specs,
)
from .localization import get_language_instruction, get_follow_up_phrases, get_end_words
from .llm_providers import (
    ProviderConfigField,
    get_llm_provider_class,
    provider_selector_options,
)

from .const import (
    DOMAIN,
    SYSTEM_ENTRY_UNIQUE_ID,
    CONF_PROFILE_NAME,
    CONF_SERVER_TYPE,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_AUTO_START,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_CONTROL_HA,
    CONF_FOLLOW_UP_MODE,
    CONF_RESPONSE_MODE,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
    CONF_MAX_HISTORY,
    CONF_CONTEXT_MODE,
    CONF_MAX_ITERATIONS,
    CONF_DEBUG_MODE,
    CONF_CHAT_LOG_MODE,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_BRAVE_API_KEY,
    CONF_GOOGLE_MAPS_API_KEY,
    CONF_SEARXNG_URL,
    CONF_ALLOWED_IPS,
    CONF_MCP_BEARER_TOKEN,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_LLM_API_ALLOWLIST,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MAX_ENTITIES_PER_DISCOVERY,
    DEFAULT_MAX_ENTITIES_PER_DISCOVERY,
    CONF_FOLLOW_UP_PHRASES,
    CONF_END_WORDS,
    CONF_CLEAN_RESPONSES,
    CONF_TIMEOUT,
    SERVER_TYPE_OPENCLAW,
    DEFAULT_SERVER_TYPE,
    CONF_OPENCLAW_HOST,
    CONF_OPENCLAW_PORT,
    CONF_OPENCLAW_TOKEN,
    CONF_OPENCLAW_USE_SSL,
    DEFAULT_OPENCLAW_HOST,
    DEFAULT_OPENCLAW_PORT,
    DEFAULT_OPENCLAW_USE_SSL,
    DEFAULT_MCP_PORT,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    PROMPT_MODE_DEFAULT,
    PROMPT_MODE_CUSTOM,
    CONTEXT_MODE_LIGHT,
    CONTEXT_MODE_STANDARD,
    DEFAULT_CONTROL_HA,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_HISTORY,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_DEBUG_MODE,
    DEFAULT_CHAT_LOG_MODE,
    DEFAULT_BRAVE_API_KEY,
    DEFAULT_GOOGLE_MAPS_API_KEY,
    DEFAULT_SEARXNG_URL,
    DEFAULT_ALLOWED_IPS,
    DEFAULT_MCP_BEARER_TOKEN,
    DEFAULT_INCLUDE_CURRENT_USER,
    DEFAULT_INCLUDE_HOME_LOCATION,
    DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    DEFAULT_SEARCH_PROVIDER,
    DEFAULT_ENABLE_WEB_SEARCH,
    DEFAULT_ENABLE_GAP_FILLING,
    DEFAULT_ENABLE_ASSIST_BRIDGE,
    DEFAULT_ENABLE_LLM_API_BRIDGE,
    DEFAULT_LLM_API_ALLOWLIST,
    DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
    DEFAULT_ENABLE_RECORDER_TOOLS,
    DEFAULT_ENABLE_MEMORY_TOOLS,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
    DEFAULT_ENABLE_DEVICE_TOOLS,
    DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    DEFAULT_MEMORY_MAX_TTL_DAYS,
    DEFAULT_MEMORY_MAX_ITEMS,
    DEFAULT_FOLLOW_UP_PHRASES,
    DEFAULT_END_WORDS,
    DEFAULT_CLEAN_RESPONSES,
    DEFAULT_TIMEOUT,
    TOOL_FAMILY_ASSIST_BRIDGE,
    TOOL_FAMILY_DEVICE,
    TOOL_FAMILY_EXTERNAL_CUSTOM,
    TOOL_FAMILY_LLM_API_BRIDGE,
    TOOL_FAMILY_MEMORY,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
    parse_llm_api_allowlist,
)

_LOGGER = logging.getLogger(__name__)
_SENSITIVE_LOG_FIELD_PATTERN = (
    r"api[_-]?key|api\s+key|authorization|bearer|password|secret|token|\bkey\b"
)
_SENSITIVE_LOG_FIELD_RE = re.compile(
    rf"(?i)({_SENSITIVE_LOG_FIELD_PATTERN})"
)
_SENSITIVE_LOG_FIELD_VALUE_RE = re.compile(
    rf"(?i)([\"']?(?:{_SENSITIVE_LOG_FIELD_PATTERN})[\"']?"
    r"(?:\s+\w+){0,3}\s*[:=]\s*[\"']?)[^\"'\s,;}]+([\"']?)"
)


def _redacted_log_snippet(value: Any, *, max_chars: int = 200) -> str:
    """Return a short config-flow log snippet with common secrets redacted."""
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+[^\s,;}]+", "Bearer [redacted]", text)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*)[^\s,;}]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"://[^/\s:@]+:[^@\s/]+@", "://[redacted]@", text)
    text = _SENSITIVE_LOG_FIELD_VALUE_RE.sub(r"\1[redacted]\2", text)
    text = re.sub(
        r"(?i)([?&]key=)[^&\s]+",
        r"\1[redacted]",
        text,
    )
    text = _SENSITIVE_LOG_FIELD_RE.sub("[redacted]", text)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated {len(text) - max_chars} chars]"


def _prompt_mode_selector() -> SelectSelector:
    """Build a prompt source selector."""
    return SelectSelector(
        SelectSelectorConfig(
            options=[PROMPT_MODE_DEFAULT, PROMPT_MODE_CUSTOM],
            mode=SelectSelectorMode.DROPDOWN,
            translation_key="prompt_source_mode",
        )
    )


def _context_mode_selector() -> SelectSelector:
    """Build a model context-size selector."""
    return SelectSelector(
        SelectSelectorConfig(
            options=[
                {"value": CONTEXT_MODE_STANDARD, "label": "Standard"},
                {"value": CONTEXT_MODE_LIGHT, "label": "Light"},
            ],
            mode=SelectSelectorMode.DROPDOWN,
        )
    )


def _provider_display_name(server_type: str) -> str:
    """Return the user-facing provider display name."""
    return get_llm_provider_class(server_type).config_display_name()


def _provider_default_url(server_type: str) -> str:
    """Return the default URL for providers that use a configurable endpoint."""
    return str(get_llm_provider_class(server_type).default_base_url or "")


def _provider_field_default(
    field: ProviderConfigField,
    current_values: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    """Resolve a provider field default from in-progress, option, data, then spec."""
    default = field.default
    if options is not None or data is not None:
        default = (options or {}).get(field.key, (data or {}).get(field.key, default))
    return _get_form_value(current_values, field.key, default)


def _provider_field_marker(
    field: ProviderConfigField,
    current_values: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> vol.Required | vol.Optional:
    """Build a voluptuous marker for a provider-owned field."""
    default = _provider_field_default(field, current_values, options, data)
    marker = vol.Required if field.required else vol.Optional
    if default is None:
        return marker(field.key)
    return marker(field.key, default=default)


def _provider_field_validator(field: ProviderConfigField) -> Any:
    """Build a selector or validator for a provider-owned field."""
    if field.kind == "password":
        return TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
    if field.kind == "boolean":
        return BooleanSelector()
    if field.kind == "integer":
        validator: Any = vol.Coerce(int)
        if field.minimum is not None or field.maximum is not None:
            validator = vol.All(
                validator,
                vol.Range(min=field.minimum, max=field.maximum),
            )
        return validator
    return str


def _build_provider_field_schema_items(
    fields: tuple[ProviderConfigField, ...],
    current_values: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[Any, Any]:
    """Build schema items for provider-owned fields."""
    return {
        _provider_field_marker(field, current_values, options, data): (
            _provider_field_validator(field)
        )
        for field in fields
    }


def _merge_provider_values(*sources: dict[str, Any] | None) -> dict[str, Any]:
    """Merge profile/config values for provider hooks."""
    merged: dict[str, Any] = {}
    for source in sources:
        if source:
            merged.update(source)
    return merged


def _get_default_system_prompt(hass: HomeAssistant) -> str:
    """Get the localized default system prompt."""
    return get_language_instruction(hass.config.language) or DEFAULT_SYSTEM_PROMPT


def _infer_prompt_mode(
    explicit_mode: Any, stored_prompt: Any, default_prompt: str
) -> str:
    """Infer prompt mode with backward compatibility for older entries."""
    if explicit_mode in {PROMPT_MODE_DEFAULT, PROMPT_MODE_CUSTOM}:
        return explicit_mode
    if stored_prompt in (None, "", default_prompt):
        return PROMPT_MODE_DEFAULT
    return PROMPT_MODE_CUSTOM


def _format_installed_llm_api_options(hass: HomeAssistant) -> str:
    """Return a short display string for registered third-party LLM APIs."""
    get_apis = getattr(llm, "async_get_apis", None)
    if not callable(get_apis):
        return "not available on this Home Assistant version"

    try:
        registered_apis = sorted(
            (
                api
                for api in get_apis(hass)
                if getattr(api, "id", None) != llm.LLM_API_ASSIST
            ),
            key=lambda api: (
                str(getattr(api, "name", "")).casefold(),
                str(getattr(api, "id", "")),
            ),
        )
    except Exception as err:
        _LOGGER.debug("Unable to list registered third-party LLM APIs: %s", err)
        return "unable to read installed APIs right now"

    if not registered_apis:
        return "none currently registered"

    formatted_options = []
    for api in registered_apis:
        api_id = str(getattr(api, "id", "")).strip()
        if not api_id:
            continue
        api_name = str(getattr(api, "name", "") or api_id).strip()
        formatted_options.append(f"{api_name} ({api_id})")
    return ", ".join(formatted_options) or "none currently registered"


def _get_current_prompt_mode(
    current_values: dict[str, Any] | None,
    *,
    mode_key: str,
    stored_mode: Any,
    stored_prompt: Any,
    default_prompt: str,
) -> str:
    """Get the prompt mode from current form values or stored settings."""
    if current_values and mode_key in current_values:
        current_mode = current_values.get(mode_key)
        if current_mode in {PROMPT_MODE_DEFAULT, PROMPT_MODE_CUSTOM}:
            return current_mode

    return _infer_prompt_mode(stored_mode, stored_prompt, default_prompt)


def _get_prompt_text_default(
    current_values: dict[str, Any] | None,
    *,
    prompt_key: str,
    stored_mode: Any = None,
    stored_prompt: Any,
    default_prompt: str = "",
) -> str:
    """Get the effective prompt text to prefill in the form."""
    if current_values and prompt_key in current_values:
        value = current_values.get(prompt_key)
        return "" if value is None else str(value)

    inferred_mode = _infer_prompt_mode(stored_mode, stored_prompt, default_prompt)
    if inferred_mode == PROMPT_MODE_DEFAULT or stored_prompt in (None, ""):
        return str(default_prompt)

    return str(stored_prompt)


def _normalize_prompt_inputs(
    user_input: dict[str, Any], server_type: str, default_system_prompt: str
) -> dict[str, Any]:
    """Normalize prompt override inputs before storing."""
    normalized = dict(user_input)

    def _normalize_prompt(prompt_key: str, mode_key: str, default_prompt: str) -> None:
        raw_value = normalized.get(prompt_key, "")
        text = "" if raw_value is None else str(raw_value)
        if not text.strip() or text.strip() == default_prompt.strip():
            normalized.pop(prompt_key, None)
            normalized[mode_key] = PROMPT_MODE_DEFAULT
        else:
            normalized[prompt_key] = text
            normalized[mode_key] = PROMPT_MODE_CUSTOM

    if server_type == SERVER_TYPE_OPENCLAW:
        normalized[CONF_SYSTEM_PROMPT_MODE] = PROMPT_MODE_DEFAULT
        normalized.pop(CONF_SYSTEM_PROMPT, None)
    else:
        _normalize_prompt(
            CONF_SYSTEM_PROMPT,
            CONF_SYSTEM_PROMPT_MODE,
            default_system_prompt,
        )

    _normalize_prompt(
        CONF_TECHNICAL_PROMPT,
        CONF_TECHNICAL_PROMPT_MODE,
        DEFAULT_TECHNICAL_PROMPT,
    )

    return normalized


def _get_form_value(
    current_values: dict[str, Any] | None, key: str, fallback: Any
) -> Any:
    """Prefer in-progress form values over stored defaults."""
    if current_values and key in current_values:
        return current_values[key]
    return fallback


def generate_mcp_bearer_token() -> str:
    """Generate a high-entropy bearer token for external MCP clients."""
    return secrets.token_urlsafe(32)


def _normalize_mcp_bearer_token(value: Any) -> str:
    """Normalize the optional shared MCP bearer token."""
    return str(value or "").strip()


def _validate_mcp_bearer_token(value: Any) -> bool:
    """Return whether an optional MCP bearer token is strong enough."""
    token = _normalize_mcp_bearer_token(value)
    return not token or len(token) >= 16


def _optional_with_suggested_value(key: str, suggested_value: str | None) -> vol.Optional:
    """Build an optional marker that pre-fills a value without forcing it."""
    if suggested_value not in (None, ""):
        return vol.Optional(key, description={"suggested_value": suggested_value})
    return vol.Optional(key)


TOOLS_SECTION_KEY = "tools"
DISCOVERY_SECTION_KEY = "discovery"
CONTEXT_SECTION_KEY = "context"
MEMORY_SECTION_KEY = "memory"
PROFILE_SECTION_KEY = "profile"
CONNECTION_SECTION_KEY = "connection"
MODEL_SECTION_KEY = "model_fields"
PROMPTS_SECTION_KEY = "prompts"
CONVERSATION_SECTION_KEY = "conversation"
PERFORMANCE_SECTION_KEY = "performance"
PROVIDER_SECTION_KEY = "provider"
ADVANCED_SECTION_KEY = "advanced_settings"
DISABLE_ASSIST_BRIDGE_FIELD = "disable_assist_bridge"
DISABLE_CUSTOM_TOOLS_FIELD = "disable_custom_tools"
DISABLE_DEVICE_FIELD = "disable_device"
DISABLE_LLM_API_BRIDGE_FIELD = "disable_llm_api_bridge"
DISABLE_MEMORY_FIELD = "disable_memory"
DISABLE_MUSIC_ASSISTANT_FIELD = "disable_music_assistant"
DISABLE_RECORDER_FIELD = "disable_recorder"
DISABLE_RESPONSE_SERVICE_FIELD = "disable_response_service"
DISABLE_WEATHER_FORECAST_FIELD = "disable_weather_forecast"

STATIC_TOOL_FAMILY_ALPHABETICAL = [
    TOOL_FAMILY_ASSIST_BRIDGE,
    TOOL_FAMILY_EXTERNAL_CUSTOM,
    TOOL_FAMILY_DEVICE,
    TOOL_FAMILY_LLM_API_BRIDGE,
    TOOL_FAMILY_MEMORY,
]

PROFILE_DISABLE_FIELD_BY_FAMILY = {
    TOOL_FAMILY_ASSIST_BRIDGE: DISABLE_ASSIST_BRIDGE_FIELD,
    TOOL_FAMILY_EXTERNAL_CUSTOM: DISABLE_CUSTOM_TOOLS_FIELD,
    TOOL_FAMILY_DEVICE: DISABLE_DEVICE_FIELD,
    TOOL_FAMILY_LLM_API_BRIDGE: DISABLE_LLM_API_BRIDGE_FIELD,
    TOOL_FAMILY_MEMORY: DISABLE_MEMORY_FIELD,
}

STATIC_TOOL_FAMILY_SHARED_LABELS = {
    TOOL_FAMILY_ASSIST_BRIDGE: "Assist Bridge",
    TOOL_FAMILY_EXTERNAL_CUSTOM: "Custom Tools",
    TOOL_FAMILY_DEVICE: "Device Tools",
    TOOL_FAMILY_LLM_API_BRIDGE: "LLM API Bridge",
    TOOL_FAMILY_MEMORY: "Memory",
}

STATIC_TOOL_FAMILY_PROFILE_DISABLE_LABELS = {
    TOOL_FAMILY_ASSIST_BRIDGE: "Disable Assist Bridge",
    TOOL_FAMILY_EXTERNAL_CUSTOM: "Disable Custom Tools",
    TOOL_FAMILY_DEVICE: "Disable Device Tools",
    TOOL_FAMILY_LLM_API_BRIDGE: "Disable LLM API Bridge",
    TOOL_FAMILY_MEMORY: "Disable Memory",
}


def _flatten_section_values(
    user_input: dict[str, Any], *section_keys: str
) -> dict[str, Any]:
    """Flatten nested section payloads into a plain config dict."""
    normalized = dict(user_input)

    for section_key in section_keys:
        section_values = normalized.pop(section_key, None)
        if isinstance(section_values, dict):
            normalized.update(section_values)

    return normalized


async def _async_load_builtin_tool_toggle_specs(
    hass: HomeAssistant,
) -> tuple[BuiltInToolToggleSpec, ...]:
    """Load built-in packaged-tool metadata asynchronously."""
    return await hass.async_add_executor_job(load_builtin_tool_toggle_specs)


def _builtin_shared_field_key(spec: BuiltInToolToggleSpec) -> str:
    """Return the stable shared-form checkbox key for a built-in package."""
    return spec.shared_setting_key


def _builtin_profile_disable_field_key(spec: BuiltInToolToggleSpec) -> str:
    """Return the stable profile disable checkbox key for a built-in package."""
    return f"disable_{spec.package_id}"


def _profile_tool_disabled_default(
    current_values: dict[str, Any] | None,
    family: str,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> bool:
    """Return whether a profile tool family should default to disabled in the form."""
    options = options or {}
    data = data or {}
    disable_field = PROFILE_DISABLE_FIELD_BY_FAMILY[family]
    if current_values and disable_field in current_values:
        return bool(current_values[disable_field])

    setting_key, _default = TOOL_FAMILY_PROFILE_SETTINGS[family]
    stored_value = options.get(setting_key, data.get(setting_key))
    return stored_value is False


def _builtin_profile_tool_disabled_default(
    current_values: dict[str, Any] | None,
    spec: BuiltInToolToggleSpec,
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> bool:
    """Return whether a built-in packaged tool should default to disabled."""
    options = options or {}
    data = data or {}
    disable_field = _builtin_profile_disable_field_key(spec)
    if current_values and disable_field in current_values:
        return bool(current_values[disable_field])

    stored_value = get_builtin_profile_setting_value(
        spec,
        lambda key, default=None: options.get(key, data.get(key, default)),
    )
    return stored_value is False


def _apply_profile_tool_disables(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...] = (),
) -> dict[str, Any]:
    """Map profile disable checkboxes to stored profile enable flags."""
    normalized = dict(user_input)
    packaged_profile_setting_keys = {
        spec.profile_setting_key for spec in built_in_specs
    }
    for family in STATIC_TOOL_FAMILY_ALPHABETICAL:
        disable_field = PROFILE_DISABLE_FIELD_BY_FAMILY[family]
        setting_key, _default = TOOL_FAMILY_PROFILE_SETTINGS[family]
        if setting_key in packaged_profile_setting_keys:
            continue

        disabled = bool(normalized.pop(disable_field, False))
        if disabled:
            normalized[setting_key] = False
        else:
            normalized.pop(setting_key, None)

    for spec in built_in_specs:
        disable_field = _builtin_profile_disable_field_key(spec)
        disabled = bool(normalized.pop(disable_field, False))
        if not disabled and spec.profile_disable_label in normalized:
            disabled = bool(normalized.pop(spec.profile_disable_label, False))
        if disabled:
            normalized[spec.profile_setting_key] = False
        else:
            normalized.pop(spec.profile_setting_key, None)

    return normalized


def _normalize_search_provider(value: Any) -> str:
    """Normalize a stored search provider value."""
    normalized = str(value or "").strip().casefold()
    return normalized or DEFAULT_SEARCH_PROVIDER


def _infer_web_search_enabled(
    explicit_enabled: Any,
    search_provider: Any,
    legacy_enable_custom_tools: Any = False,
) -> bool:
    """Infer whether web search should be enabled."""
    if explicit_enabled is not None:
        return bool(explicit_enabled)

    provider = _normalize_search_provider(search_provider)
    if provider != DEFAULT_SEARCH_PROVIDER:
        return True

    return bool(legacy_enable_custom_tools)


def _shared_search_provider_requires_url(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
) -> bool:
    """Return whether the selected shared search provider needs a URL."""
    if _normalize_search_provider(user_input.get(CONF_SEARCH_PROVIDER)) != "searxng":
        return False

    built_in_search_enabled = any(
        bool(user_input.get(spec.shared_setting_key, spec.shared_default))
        for spec in built_in_specs
        if spec.requires_search_provider
    )
    legacy_web_search_enabled = bool(
        user_input.get(CONF_ENABLE_WEB_SEARCH, DEFAULT_ENABLE_WEB_SEARCH)
    )

    return built_in_search_enabled or legacy_web_search_enabled


def _validate_shared_search_settings(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
    errors: dict[str, str],
) -> None:
    """Validate provider-specific shared search settings."""
    if _shared_search_provider_requires_url(user_input, built_in_specs) and not str(
        user_input.get(CONF_SEARXNG_URL, "")
    ).strip():
        errors[CONF_SEARXNG_URL] = "searxng_url_required"


def _shared_google_maps_enabled(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
) -> bool:
    """Return whether Google Maps built-in tools are enabled in shared settings."""
    return any(
        spec.package_id == "google_maps"
        and bool(user_input.get(spec.shared_setting_key, spec.shared_default))
        for spec in built_in_specs
    )


def _validate_shared_google_maps_settings(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
    errors: dict[str, str],
) -> None:
    """Validate Google Maps settings for the optional tool package."""
    if _shared_google_maps_enabled(user_input, built_in_specs) and not str(
        user_input.get(CONF_GOOGLE_MAPS_API_KEY, "")
    ).strip():
        errors[CONF_GOOGLE_MAPS_API_KEY] = "google_maps_api_key_required"


def _normalize_shared_tool_inputs(
    user_input: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...] = (),
) -> dict[str, Any]:
    """Normalize shared tool-family settings before storing."""
    normalized = dict(user_input)

    for spec in built_in_specs:
        field_key = _builtin_shared_field_key(spec)
        if field_key in normalized:
            normalized[spec.shared_setting_key] = bool(normalized.pop(field_key))
            continue

        legacy_label_key = spec.shared_label
        if legacy_label_key in normalized:
            normalized[spec.shared_setting_key] = bool(normalized.pop(legacy_label_key))

    search_provider = _normalize_search_provider(
        normalized.get(CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER)
    )

    built_in_search_enabled = any(
        bool(normalized.get(spec.shared_setting_key, spec.shared_default))
        for spec in built_in_specs
        if spec.requires_search_provider
    )
    legacy_web_search_enabled = bool(
        normalized.get(CONF_ENABLE_WEB_SEARCH, DEFAULT_ENABLE_WEB_SEARCH)
    )

    if (
        built_in_search_enabled or legacy_web_search_enabled
    ) and search_provider == DEFAULT_SEARCH_PROVIDER:
        normalized[CONF_SEARCH_PROVIDER] = "duckduckgo"
    else:
        normalized[CONF_SEARCH_PROVIDER] = search_provider

    normalized[CONF_LLM_API_ALLOWLIST] = ", ".join(
        parse_llm_api_allowlist(
            normalized.get(CONF_LLM_API_ALLOWLIST, DEFAULT_LLM_API_ALLOWLIST)
        )
    )

    memory_max_ttl = normalized.get(
        CONF_MEMORY_MAX_TTL_DAYS,
        DEFAULT_MEMORY_MAX_TTL_DAYS,
    )
    try:
        memory_max_ttl = max(1, int(memory_max_ttl))
    except (TypeError, ValueError):
        memory_max_ttl = DEFAULT_MEMORY_MAX_TTL_DAYS
    normalized[CONF_MEMORY_MAX_TTL_DAYS] = memory_max_ttl

    memory_default_ttl = normalized.get(
        CONF_MEMORY_DEFAULT_TTL_DAYS,
        DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    )
    try:
        memory_default_ttl = int(memory_default_ttl)
    except (TypeError, ValueError):
        memory_default_ttl = DEFAULT_MEMORY_DEFAULT_TTL_DAYS
    normalized[CONF_MEMORY_DEFAULT_TTL_DAYS] = max(1, min(memory_default_ttl, memory_max_ttl))

    return normalized


def _build_profile_tools_section(
    current_values: dict[str, Any] | None,
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
    options: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> section:
    """Build the per-profile tool preferences section."""
    options = options or {}
    data = data or {}
    profile_tool_entries: list[tuple[str, vol.Optional, type[bool]]] = []
    packaged_profile_setting_keys = {
        spec.profile_setting_key for spec in built_in_specs
    }
    for family in STATIC_TOOL_FAMILY_ALPHABETICAL:
        setting_key, _default = TOOL_FAMILY_PROFILE_SETTINGS[family]
        if setting_key in packaged_profile_setting_keys:
            continue

        disable_field = PROFILE_DISABLE_FIELD_BY_FAMILY[family]
        profile_tool_entries.append(
            (
                STATIC_TOOL_FAMILY_PROFILE_DISABLE_LABELS[family].casefold(),
                vol.Optional(
                    disable_field,
                    default=_profile_tool_disabled_default(
                        current_values, family, options, data
                    ),
                ),
                bool,
            )
        )

    for spec in built_in_specs:
        profile_tool_entries.append(
            (
                spec.profile_disable_label.casefold(),
                vol.Optional(
                    _builtin_profile_disable_field_key(spec),
                    default=_builtin_profile_tool_disabled_default(
                        current_values,
                        spec,
                        options,
                        data,
                    ),
                ),
                bool,
            )
        )

    profile_tool_fields = {
        marker: value_type
        for _label, marker, value_type in sorted(
            profile_tool_entries,
            key=lambda item: item[0],
        )
    }

    return section(
        vol.Schema(profile_tool_fields),
        {"collapsed": False},
    )


def _build_shared_tools_section(
    defaults: dict[str, Any],
    built_in_specs: tuple[BuiltInToolToggleSpec, ...],
) -> section:
    """Build the shared MCP server optional tools section."""
    shared_tool_entries: list[
        tuple[str, str, vol.Optional, type[bool], BuiltInToolToggleSpec | None]
    ] = []
    packaged_shared_setting_keys = {
        spec.shared_setting_key for spec in built_in_specs
    }
    for family in STATIC_TOOL_FAMILY_ALPHABETICAL:
        setting_key, default = TOOL_FAMILY_SHARED_SETTINGS[family]
        if setting_key in packaged_shared_setting_keys:
            continue

        shared_tool_entries.append(
            (
                STATIC_TOOL_FAMILY_SHARED_LABELS[family].casefold(),
                setting_key,
                vol.Optional(
                    setting_key,
                    default=_get_form_value(defaults, setting_key, default),
                ),
                bool,
                None,
            )
        )

    for spec in built_in_specs:
        shared_tool_entries.append(
            (
                spec.shared_label.casefold(),
                spec.shared_setting_key,
                vol.Optional(
                    _builtin_shared_field_key(spec),
                    default=_get_form_value(
                        defaults,
                        spec.shared_setting_key,
                        spec.shared_default,
                    ),
                ),
                bool,
                spec,
            )
        )

    shared_tool_fields: dict[Any, Any] = {}
    for _label, setting_key, marker, value_type, spec in sorted(
        shared_tool_entries,
        key=lambda item: item[0],
    ):
        shared_tool_fields[marker] = value_type

        if spec and spec.package_id == "google_maps":
            shared_tool_fields[
                vol.Optional(
                    CONF_GOOGLE_MAPS_API_KEY,
                    default=_get_form_value(
                        defaults,
                        CONF_GOOGLE_MAPS_API_KEY,
                        DEFAULT_GOOGLE_MAPS_API_KEY,
                    ),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))

        if setting_key == CONF_ENABLE_LLM_API_BRIDGE:
            shared_tool_fields[
                vol.Optional(
                    CONF_LLM_API_ALLOWLIST,
                    default=_get_form_value(
                        defaults,
                        CONF_LLM_API_ALLOWLIST,
                        DEFAULT_LLM_API_ALLOWLIST,
                    ),
                )
            ] = TextSelector(TextSelectorConfig(multiline=True))

        if setting_key == CONF_ENABLE_MEMORY_TOOLS:
            shared_tool_fields[
                vol.Optional(
                    CONF_MEMORY_DEFAULT_TTL_DAYS,
                    default=_get_form_value(
                        defaults,
                        CONF_MEMORY_DEFAULT_TTL_DAYS,
                        DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1, max=3650))
            shared_tool_fields[
                vol.Optional(
                    CONF_MEMORY_MAX_TTL_DAYS,
                    default=_get_form_value(
                        defaults,
                        CONF_MEMORY_MAX_TTL_DAYS,
                        DEFAULT_MEMORY_MAX_TTL_DAYS,
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=1, max=3650))
            shared_tool_fields[
                vol.Optional(
                    CONF_MEMORY_MAX_ITEMS,
                    default=_get_form_value(
                        defaults,
                        CONF_MEMORY_MAX_ITEMS,
                        DEFAULT_MEMORY_MAX_ITEMS,
                    ),
                )
            ] = vol.All(vol.Coerce(int), vol.Range(min=10, max=5000))

        if spec and spec.requires_search_provider:
            shared_tool_fields[
                vol.Required(
                    CONF_SEARCH_PROVIDER,
                    default=_get_form_value(
                        defaults, CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER
                    ),
                )
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "duckduckgo", "label": "DuckDuckGo"},
                        {
                            "value": "brave",
                            "label": "Brave Search (requires API key)",
                        },
                        {
                            "value": "searxng",
                            "label": "SearXNG (self-hosted)",
                        },
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
            shared_tool_fields[
                vol.Optional(
                    CONF_BRAVE_API_KEY,
                    default=_get_form_value(
                        defaults, CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY
                    ),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
            shared_tool_fields[
                vol.Optional(
                    CONF_SEARXNG_URL,
                    default=_get_form_value(
                        defaults, CONF_SEARXNG_URL, DEFAULT_SEARXNG_URL
                    ),
                )
            ] = TextSelector(TextSelectorConfig(type=TextSelectorType.URL))

    return section(
        vol.Schema(shared_tool_fields),
        {"collapsed": False},
    )


def _build_shared_discovery_section(defaults: dict[str, Any]) -> section:
    """Build the shared MCP server discovery settings section."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_ENABLE_GAP_FILLING,
                    default=defaults[CONF_ENABLE_GAP_FILLING],
                ): bool,
                vol.Optional(
                    CONF_MAX_ENTITIES_PER_DISCOVERY,
                    default=defaults[CONF_MAX_ENTITIES_PER_DISCOVERY],
                ): vol.All(vol.Coerce(int), vol.Range(min=20, max=500)),
            }
        ),
        {"collapsed": False},
    )


def _build_shared_context_section(defaults: dict[str, Any]) -> section:
    """Build the shared AI-context settings section."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_INCLUDE_CURRENT_USER,
                    default=defaults[CONF_INCLUDE_CURRENT_USER],
                ): bool,
                vol.Optional(
                    CONF_INCLUDE_HOME_LOCATION,
                    default=defaults[CONF_INCLUDE_HOME_LOCATION],
                ): bool,
                vol.Optional(
                    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                    default=defaults[CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS],
                ): bool,
                vol.Optional(
                    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                    default=defaults[CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS],
                ): bool,
            }
        ),
        {"collapsed": False},
    )


def _build_shared_memory_section(defaults: dict[str, Any]) -> section:
    """Build shared persisted-memory settings."""
    return section(
        vol.Schema(
            {
                vol.Optional(
                    CONF_MEMORY_DEFAULT_TTL_DAYS,
                    default=defaults[CONF_MEMORY_DEFAULT_TTL_DAYS],
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=3650)),
                vol.Optional(
                    CONF_MEMORY_MAX_TTL_DAYS,
                    default=defaults[CONF_MEMORY_MAX_TTL_DAYS],
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=3650)),
                vol.Optional(
                    CONF_MEMORY_MAX_ITEMS,
                    default=defaults[CONF_MEMORY_MAX_ITEMS],
                ): vol.All(vol.Coerce(int), vol.Range(min=10, max=5000)),
            }
        ),
        {"collapsed": False},
    )


def _build_profile_identity_section(profile_name: str) -> section:
    """Build the profile identity section."""
    return section(
        vol.Schema(
            {
                vol.Required(CONF_PROFILE_NAME, default=profile_name): str,
            }
        ),
        {"collapsed": False},
    )


def _build_connection_section(schema_items: dict[Any, Any]) -> section:
    """Wrap connection-related fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_model_section(current_model: str, model_field: Any) -> section:
    """Build the model-selection section."""
    return section(
        vol.Schema(
            {
                vol.Required(CONF_MODEL_NAME, default=current_model): model_field,
            }
        ),
        {"collapsed": False},
    )


def _build_prompt_section(
    *,
    system_prompt_value: str | None = None,
    technical_prompt_value: str | None,
    include_system_prompt: bool = True,
) -> section:
    """Build the prompt editing section."""
    schema_items: dict[Any, Any] = {}
    if include_system_prompt:
        schema_items[
            _optional_with_suggested_value(CONF_SYSTEM_PROMPT, system_prompt_value)
        ] = TemplateSelector(TemplateSelectorConfig())
    schema_items[
        _optional_with_suggested_value(
            CONF_TECHNICAL_PROMPT, technical_prompt_value
        )
    ] = TemplateSelector(TemplateSelectorConfig())
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_conversation_section(schema_items: dict[Any, Any]) -> section:
    """Wrap conversation-behavior fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_performance_section(schema_items: dict[Any, Any]) -> section:
    """Wrap performance-related fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_provider_section(schema_items: dict[Any, Any]) -> section:
    """Wrap provider-specific fields in a section."""
    return section(vol.Schema(schema_items), {"collapsed": False})


def _build_advanced_section(schema_items: dict[Any, Any]) -> section:
    """Wrap advanced settings in a collapsed section."""
    return section(vol.Schema(schema_items), {"collapsed": True})


def _needs_prompt_followup(
    user_input: dict[str, Any], server_type: str
) -> bool:
    """Config-flow forms are static per step; prompt followup is no longer used."""
    del user_input, server_type
    return False


def validate_allowed_ips(allowed_ips_str: str) -> tuple[bool, str]:
    """Validate comma-separated list of IP addresses and CIDR ranges.

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not allowed_ips_str or not allowed_ips_str.strip():
        # Empty is valid (no additional IPs)
        return True, ""

    # Parse comma-separated values
    ip_list = [ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()]

    for ip_entry in ip_list:
        try:
            # Try parsing as IP network (handles both individual IPs and CIDR)
            ipaddress.ip_network(ip_entry, strict=False)
        except ValueError:
            # Invalid IP or CIDR format
            return False, f"Invalid IP address or CIDR range: {ip_entry}"

    return True, ""


STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_PROFILE_NAME): str,
        vol.Required(CONF_SERVER_TYPE, default=DEFAULT_SERVER_TYPE): SelectSelector(
            SelectSelectorConfig(
                options=provider_selector_options(),
                mode=SelectSelectorMode.LIST,
            )
        ),
    }
)

STEP_MCP_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MCP_PORT, default=DEFAULT_MCP_PORT): vol.Coerce(int),
        vol.Required(CONF_AUTO_START, default=True): bool,
    }
)


class MCPAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for MCP Assist."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize."""
        self.step1_data: dict[str, Any] = {}
        self.step2_data: dict[str, Any] = {}
        self.step3_data: dict[str, Any] = {}
        self.step4_data: dict[str, Any] = {}
        self._generated_mcp_bearer_token = generate_mcp_bearer_token()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 1 - profile name and server type."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate profile name is not empty
            profile_name = user_input.get(CONF_PROFILE_NAME, "").strip()
            if not profile_name:
                errors[CONF_PROFILE_NAME] = "profile_name_required"
            else:
                # Store data and move to step 2
                self.step1_data = user_input
                return await self.async_step_server()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_server(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 2 - server configuration (URL for local, API key for cloud)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Store data and move to next step
            self.step2_data = user_input
            server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
            if server_type == SERVER_TYPE_OPENCLAW:
                return await self.async_step_openclaw_pairing()
            return await self.async_step_model()

        # Get server type from step 1 to build dynamic schema
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        provider_class = get_llm_provider_class(server_type)
        server_schema = vol.Schema(
            _build_provider_field_schema_items(provider_class.connection_fields)
        )

        return self.async_show_form(
            step_id="server",
            data_schema=server_schema,
            errors=errors,
        )

    async def async_step_openclaw_pairing(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle OpenClaw device pairing step."""
        errors: dict[str, str] = {}

        from .openclaw_client import (
            OpenClawClient, OpenClawDeviceAuth, DevicePairingRequiredError,
            OpenClawConnectionError, OpenClawAuthError,
        )

        # Get or create device auth
        if "openclaw_device_auth" not in self.hass.data.get(DOMAIN, {}):
            self.hass.data.setdefault(DOMAIN, {})
            device_auth = OpenClawDeviceAuth(self.hass)
            await device_auth.async_load()
            self.hass.data[DOMAIN]["openclaw_device_auth"] = device_auth

        device_auth = self.hass.data[DOMAIN]["openclaw_device_auth"]
        device_id = device_auth.device_id

        if user_input is not None or not hasattr(self, "_pairing_attempted"):
            # Attempt connection to test pairing
            self._pairing_attempted = True
            client = OpenClawClient(
                host=self.step2_data.get(CONF_OPENCLAW_HOST, DEFAULT_OPENCLAW_HOST),
                port=self.step2_data.get(CONF_OPENCLAW_PORT, DEFAULT_OPENCLAW_PORT),
                token=self.step2_data.get(CONF_OPENCLAW_TOKEN, ""),
                use_ssl=self.step2_data.get(CONF_OPENCLAW_USE_SSL, DEFAULT_OPENCLAW_USE_SSL),
                device_auth=device_auth,
                timeout=30,
            )

            try:
                await client.connect()
                await client.disconnect()
                # Device is approved — proceed to model step
                return await self.async_step_model()
            except DevicePairingRequiredError:
                errors["base"] = "openclaw_not_paired"
            except OpenClawAuthError as err:
                _LOGGER.error("OpenClaw auth error: %s", err)
                errors["base"] = "openclaw_connection_failed"
            except OpenClawConnectionError as err:
                _LOGGER.error("OpenClaw connection error: %s", err)
                errors["base"] = "openclaw_connection_failed"
            except Exception as err:
                _LOGGER.error("OpenClaw unexpected error: %s: %s", type(err).__name__, err)
                errors["base"] = "openclaw_connection_failed"

        return self.async_show_form(
            step_id="openclaw_pairing",
            data_schema=vol.Schema({}),
            errors=errors,
            description_placeholders={
                "device_id": device_id,
            },
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 3 - model selection and prompts."""
        errors: dict[str, str] = {}

        # Get server type to determine model source
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        provider_class = get_llm_provider_class(server_type)
        default_system_prompt = _get_default_system_prompt(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input, MODEL_SECTION_KEY, PROMPTS_SECTION_KEY
            )
            user_input = _normalize_prompt_inputs(
                user_input, server_type, default_system_prompt
            )
            # Store data and move to step 4 (advanced)
            self.step3_data = user_input
            return await self.async_step_advanced()

        models = []
        current_values = getattr(self, "step3_data", {})

        if not provider_class.uses_config_model_step:
            self.step3_data = {
                CONF_MODEL_NAME: provider_class.default_config_model_name,
                CONF_SYSTEM_PROMPT: provider_class.default_config_system_prompt,
                CONF_TECHNICAL_PROMPT: provider_class.default_config_technical_prompt,
                CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_DEFAULT,
                CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_DEFAULT,
            }
            return await self.async_step_advanced()
        else:
            provider_values = _merge_provider_values(
                self.step1_data,
                self.step2_data,
                current_values,
            )
            models = await provider_class.fetch_models(self.hass, provider_values)
            _LOGGER.debug(
                "Fetched %d %s models",
                len(models),
                provider_class.config_display_name(),
            )
            if provider_class.model_fetch_error and not models:
                errors["base"] = provider_class.model_fetch_error

        # Build dynamic schema based on whether models were fetched
        current_model = current_values.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME)
        if models:
            # Show dropdown with available models (custom_value allows free text input)
            _LOGGER.info("Showing model dropdown with %d models", len(models))
            model_field = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            )
        else:
            # Show text input as fallback
            _LOGGER.info("No models fetched, showing text input")
            model_field = str

        system_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_SYSTEM_PROMPT,
            stored_prompt=None,
            default_prompt=default_system_prompt,
        )
        technical_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_TECHNICAL_PROMPT,
            stored_prompt=None,
            default_prompt=DEFAULT_TECHNICAL_PROMPT,
        )

        schema_dict: dict[Any, Any] = {
            MODEL_SECTION_KEY: _build_model_section(current_model, model_field),
            PROMPTS_SECTION_KEY: _build_prompt_section(
                system_prompt_value=system_prompt_suggestion,
                technical_prompt_value=technical_prompt_suggestion,
            ),
        }

        model_schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="model",
            data_schema=model_schema,
            errors=errors,
            description_placeholders={
                "server_info": "Select a model. The prompt fields are prefilled with the current effective prompts so you can review, copy, or edit them directly. If you leave a prompt unchanged, the integration keeps using the built-in version from code. Models are automatically loaded from your server."
            },
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle step 4 - advanced settings."""
        errors: dict[str, str] = {}
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        # Get server type to determine which fields to show
        server_type = self.step1_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        provider_class = get_llm_provider_class(server_type)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                CONVERSATION_SECTION_KEY,
                PERFORMANCE_SECTION_KEY,
                PROVIDER_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _apply_profile_tool_disables(user_input, built_in_specs)

            # For OpenClaw, set defaults for LLM-specific fields (not shown in UI)
            if not provider_class.uses_config_model_step:
                user_input[CONF_TEMPERATURE] = DEFAULT_TEMPERATURE
                user_input[CONF_MAX_TOKENS] = DEFAULT_MAX_TOKENS
                user_input[CONF_MAX_HISTORY] = DEFAULT_MAX_HISTORY
                user_input[CONF_MAX_ITERATIONS] = DEFAULT_MAX_ITERATIONS
                if CONF_TIMEOUT not in user_input:
                    user_input[CONF_TIMEOUT] = 60

            # Validate MCP port
            mcp_port = user_input.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            if not 1024 <= mcp_port <= 65535:
                errors[CONF_MCP_PORT] = "invalid_port"

            # Validate allowed IPs
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs: %s", error_msg)

            if not errors:
                # Check if this is the first profile (MCP server doesn't exist yet)
                is_first_profile = "shared_mcp_server" not in self.hass.data.get(
                    DOMAIN, {}
                )

                if is_first_profile:
                    # First profile - store step 4 data and proceed to MCP server config
                    self.step4_data = user_input
                    return await self.async_step_mcp_server()
                else:
                    # Subsequent profile - use existing shared MCP server settings
                    # Get MCP settings from shared server
                    mcp_port = self.hass.data[DOMAIN].get("mcp_port", DEFAULT_MCP_PORT)

                    # Get search provider from any existing entry (they all share it)
                    # Find first entry to copy shared settings from
                    existing_entry = None
                    for entry in self.hass.config_entries.async_entries(DOMAIN):
                        existing_entry = entry
                        break

                    # Copy shared settings from existing entry
                    shared_settings = {}
                    if existing_entry:
                        shared_settings = {
                            CONF_MCP_PORT: existing_entry.data.get(
                                CONF_MCP_PORT, mcp_port
                            ),
                            CONF_SEARCH_PROVIDER: existing_entry.data.get(
                                CONF_SEARCH_PROVIDER, DEFAULT_SEARCH_PROVIDER
                            ),
                            CONF_BRAVE_API_KEY: existing_entry.data.get(
                                CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY
                            ),
                            CONF_SEARXNG_URL: existing_entry.data.get(
                                CONF_SEARXNG_URL, DEFAULT_SEARXNG_URL
                            ),
                            CONF_ALLOWED_IPS: existing_entry.data.get(
                                CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS
                            ),
                            CONF_MCP_BEARER_TOKEN: existing_entry.data.get(
                                CONF_MCP_BEARER_TOKEN,
                                DEFAULT_MCP_BEARER_TOKEN,
                            ),
                            CONF_INCLUDE_CURRENT_USER: existing_entry.data.get(
                                CONF_INCLUDE_CURRENT_USER,
                                DEFAULT_INCLUDE_CURRENT_USER,
                            ),
                            CONF_INCLUDE_HOME_LOCATION: existing_entry.data.get(
                                CONF_INCLUDE_HOME_LOCATION,
                                DEFAULT_INCLUDE_HOME_LOCATION,
                            ),
                            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: existing_entry.data.get(
                                CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                                DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                            ),
                            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: existing_entry.data.get(
                                CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                                DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                            ),
                            CONF_ENABLE_GAP_FILLING: existing_entry.data.get(
                                CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING
                            ),
                            CONF_ENABLE_ASSIST_BRIDGE: existing_entry.data.get(
                                CONF_ENABLE_ASSIST_BRIDGE,
                                DEFAULT_ENABLE_ASSIST_BRIDGE,
                            ),
                            CONF_ENABLE_LLM_API_BRIDGE: existing_entry.data.get(
                                CONF_ENABLE_LLM_API_BRIDGE,
                                DEFAULT_ENABLE_LLM_API_BRIDGE,
                            ),
                            CONF_LLM_API_ALLOWLIST: existing_entry.data.get(
                                CONF_LLM_API_ALLOWLIST,
                                DEFAULT_LLM_API_ALLOWLIST,
                            ),
                            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
                            ),
                            CONF_ENABLE_WEATHER_FORECAST_TOOL: existing_entry.data.get(
                                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                                DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
                            ),
                            CONF_ENABLE_RECORDER_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_RECORDER_TOOLS,
                                DEFAULT_ENABLE_RECORDER_TOOLS,
                            ),
                            CONF_ENABLE_CALCULATOR_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_CALCULATOR_TOOLS,
                                DEFAULT_ENABLE_CALCULATOR_TOOLS,
                            ),
                            CONF_ENABLE_UNIT_CONVERSION_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_UNIT_CONVERSION_TOOLS,
                                existing_entry.data.get(
                                    CONF_ENABLE_CALCULATOR_TOOLS,
                                    DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
                                ),
                            ),
                            CONF_ENABLE_DEVICE_TOOLS: existing_entry.data.get(
                                CONF_ENABLE_DEVICE_TOOLS,
                                DEFAULT_ENABLE_DEVICE_TOOLS,
                            ),
                            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: existing_entry.data.get(
                                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                                DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                            ),
                        }
                        for spec in built_in_specs:
                            shared_settings[spec.shared_setting_key] = (
                                existing_entry.data.get(
                                    spec.shared_setting_key,
                                    get_builtin_shared_setting_value(
                                        spec,
                                        lambda key, default=None: existing_entry.data.get(
                                            key, default
                                        ),
                                    ),
                                )
                            )

                    # Combine data from steps 1-4 + shared settings
                    combined_data = {
                        **self.step1_data,
                        **self.step2_data,
                        **self.step3_data,
                        **user_input,  # Step 4 data
                        **shared_settings,  # Copy from existing entry
                    }

                    # Create config entry (same as before)
                    profile_name = combined_data[CONF_PROFILE_NAME]
                    server_type = combined_data.get(
                        CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE
                    )

                    server_display = _provider_display_name(server_type)

                    unique_id = f"{DOMAIN}_{server_type}_{profile_name.lower().replace(' ', '_')}"
                    await self.async_set_unique_id(unique_id)
                    self._abort_if_unique_id_configured()

                    return self.async_create_entry(
                        title=f"{server_display} - {profile_name}",
                        data=combined_data,
                    )

        default_temp = (
            provider_class.default_temperature
            if provider_class.default_temperature is not None
            else DEFAULT_TEMPERATURE
        )

        # Build schema based on server type
        if not provider_class.uses_config_model_step:
            advanced_schema_dict = {
                CONVERSATION_SECTION_KEY: _build_conversation_section(
                    {
                        vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
                        vol.Required(
                            CONF_RESPONSE_MODE, default=DEFAULT_RESPONSE_MODE
                        ): SelectSelector(
                            SelectSelectorConfig(
                                options=[
                                    {"value": "none", "label": "None"},
                                    {"value": "default", "label": "Smart"},
                                    {"value": "always", "label": "Always"},
                                ],
                                mode=SelectSelectorMode.DROPDOWN,
                            )
                        ),
                        vol.Optional(
                            CONF_FOLLOW_UP_PHRASES,
                            default=get_follow_up_phrases(self.hass.config.language),
                        ): TextSelector(TextSelectorConfig(multiline=True)),
                        vol.Optional(
                            CONF_END_WORDS,
                            default=get_end_words(self.hass.config.language),
                        ): TextSelector(TextSelectorConfig(multiline=True)),
                        vol.Optional(
                            CONF_CLEAN_RESPONSES, default=DEFAULT_CLEAN_RESPONSES
                        ): bool,
                    }
                ),
                PERFORMANCE_SECTION_KEY: _build_performance_section(
                    {
                        vol.Required(CONF_TIMEOUT, default=60): vol.All(
                            vol.Coerce(int), vol.Range(min=5, max=300)
                        ),
                        vol.Required(
                            CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE
                        ): bool,
                        vol.Required(
                            CONF_CHAT_LOG_MODE, default=DEFAULT_CHAT_LOG_MODE
                        ): bool,
                    }
                ),
                PROVIDER_SECTION_KEY: _build_provider_section(
                    _build_provider_field_schema_items(
                        provider_class.provider_options_fields
                    )
                ),
            }
        else:
            performance_schema_items: dict[Any, Any] = {
                vol.Required(CONF_TEMPERATURE, default=default_temp): vol.All(
                    vol.Coerce(float), vol.Range(min=0.0, max=1.0)
                ),
                vol.Required(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): vol.Coerce(
                    int
                ),
                vol.Required(CONF_MAX_HISTORY, default=DEFAULT_MAX_HISTORY): vol.Coerce(
                    int
                ),
                vol.Required(
                    CONF_CONTEXT_MODE, default=DEFAULT_CONTEXT_MODE
                ): _context_mode_selector(),
                vol.Required(
                    CONF_MAX_ITERATIONS, default=DEFAULT_MAX_ITERATIONS
                ): vol.Coerce(int),
                vol.Required(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(
                    vol.Coerce(int), vol.Range(min=5, max=300)
                ),
                vol.Required(CONF_DEBUG_MODE, default=DEFAULT_DEBUG_MODE): bool,
                vol.Required(CONF_CHAT_LOG_MODE, default=DEFAULT_CHAT_LOG_MODE): bool,
            }
            conversation_schema_items: dict[Any, Any] = {
                vol.Required(CONF_CONTROL_HA, default=DEFAULT_CONTROL_HA): bool,
                vol.Required(
                    CONF_RESPONSE_MODE, default=DEFAULT_RESPONSE_MODE
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            {"value": "none", "label": "None"},
                            {"value": "default", "label": "Smart"},
                            {"value": "always", "label": "Always"},
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_FOLLOW_UP_PHRASES,
                    default=get_follow_up_phrases(self.hass.config.language),
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_END_WORDS, default=get_end_words(self.hass.config.language)
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(CONF_CLEAN_RESPONSES, default=DEFAULT_CLEAN_RESPONSES): bool,
            }
            advanced_schema_dict = {
                CONVERSATION_SECTION_KEY: _build_conversation_section(
                    conversation_schema_items
                ),
                PERFORMANCE_SECTION_KEY: _build_performance_section(
                    performance_schema_items
                ),
            }

            provider_schema_items = _build_provider_field_schema_items(
                provider_class.provider_options_fields
            )
            if provider_schema_items:
                advanced_schema_dict[PROVIDER_SECTION_KEY] = _build_provider_section(
                    provider_schema_items
                )

        advanced_schema_dict[TOOLS_SECTION_KEY] = _build_profile_tools_section(
            getattr(self, "step4_data", {}),
            built_in_specs,
        )

        advanced_schema = vol.Schema(advanced_schema_dict)

        # Set description based on server type
        if server_type == SERVER_TYPE_OPENCLAW:
            description_placeholders = {
                "advanced_info": (
                    "OpenClaw manages model selection, token limits, history, and "
                    "tool execution on the gateway. Only conversation, timeout, "
                    "session, and profile-level tool settings are shown here. The "
                    "Tools section still lets this profile disable specific shared "
                    "MCP tool families."
                )
            }
        else:
            description_placeholders = {
                "advanced_info": (
                    "These settings are organized by conversation behavior, performance, "
                    "provider-specific options, and tools. The Tools section only affects "
                    "this profile and can disable specific shared MCP tool families for "
                    "smaller models."
                )
            }

        return self.async_show_form(
            step_id="advanced",
            data_schema=advanced_schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_mcp_server(self, user_input=None) -> FlowResult:
        """Handle step 5 - shared MCP server settings (first profile only)."""
        errors: dict[str, str] = {}
        current_values = user_input or {}
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                CONTEXT_SECTION_KEY,
                DISCOVERY_SECTION_KEY,
                MEMORY_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _normalize_shared_tool_inputs(user_input, built_in_specs)
            current_values = user_input

            # Validate MCP port
            mcp_port = user_input.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            if not 1024 <= mcp_port <= 65535:
                errors[CONF_MCP_PORT] = "invalid_port"

            # Validate allowed IPs
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs: %s", error_msg)

            user_input[CONF_MCP_BEARER_TOKEN] = _normalize_mcp_bearer_token(
                user_input.get(
                    CONF_MCP_BEARER_TOKEN,
                    self._generated_mcp_bearer_token,
                )
            )
            if not _validate_mcp_bearer_token(user_input[CONF_MCP_BEARER_TOKEN]):
                errors[CONF_MCP_BEARER_TOKEN] = "mcp_bearer_token_too_short"

            _validate_shared_search_settings(user_input, built_in_specs, errors)
            _validate_shared_google_maps_settings(user_input, built_in_specs, errors)

            if not errors:
                # Create/update system entry with shared settings
                from . import get_system_entry

                system_entry = get_system_entry(self.hass)

                if not system_entry:
                    # Create system entry with shared settings
                    await self.hass.config_entries.flow.async_init(
                        DOMAIN, context={"source": "system"}, data=user_input
                    )
                    _LOGGER.info(
                        "Created system entry with shared MCP settings from initial setup"
                    )
                else:
                    # Update existing system entry
                    self.hass.config_entries.async_update_entry(
                        system_entry, data={**system_entry.data, **user_input}
                    )
                    _LOGGER.info(
                        "Updated existing system entry with shared MCP settings"
                    )

                # Combine data from steps 1-4 (profile settings only, no shared settings)
                combined_data = {
                    **self.step1_data,
                    **self.step2_data,
                    **self.step3_data,
                    **self.step4_data,
                }

                # Create profile config entry
                profile_name = combined_data[CONF_PROFILE_NAME]
                server_type = combined_data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)

                server_display = _provider_display_name(server_type)

                unique_id = (
                    f"{DOMAIN}_{server_type}_{profile_name.lower().replace(' ', '_')}"
                )
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"{server_display} - {profile_name}",
                    data=combined_data,
                )

        shared_defaults = {
            CONF_SEARCH_PROVIDER: _get_form_value(
                current_values,
                CONF_SEARCH_PROVIDER,
                DEFAULT_SEARCH_PROVIDER,
            ),
            CONF_ENABLE_WEB_SEARCH: _get_form_value(
                current_values,
                CONF_ENABLE_WEB_SEARCH,
                _infer_web_search_enabled(
                    current_values.get(CONF_ENABLE_WEB_SEARCH),
                    current_values.get(CONF_SEARCH_PROVIDER),
                ),
            ),
            CONF_BRAVE_API_KEY: _get_form_value(
                current_values,
                CONF_BRAVE_API_KEY,
                DEFAULT_BRAVE_API_KEY,
            ),
            CONF_GOOGLE_MAPS_API_KEY: _get_form_value(
                current_values,
                CONF_GOOGLE_MAPS_API_KEY,
                DEFAULT_GOOGLE_MAPS_API_KEY,
            ),
            CONF_SEARXNG_URL: _get_form_value(
                current_values,
                CONF_SEARXNG_URL,
                DEFAULT_SEARXNG_URL,
            ),
            CONF_INCLUDE_CURRENT_USER: _get_form_value(
                current_values,
                CONF_INCLUDE_CURRENT_USER,
                DEFAULT_INCLUDE_CURRENT_USER,
            ),
            CONF_INCLUDE_HOME_LOCATION: _get_form_value(
                current_values,
                CONF_INCLUDE_HOME_LOCATION,
                DEFAULT_INCLUDE_HOME_LOCATION,
            ),
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: _get_form_value(
                current_values,
                CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
            ),
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: _get_form_value(
                current_values,
                CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
            ),
            CONF_ENABLE_GAP_FILLING: _get_form_value(
                current_values,
                CONF_ENABLE_GAP_FILLING,
                DEFAULT_ENABLE_GAP_FILLING,
            ),
            CONF_MAX_ENTITIES_PER_DISCOVERY: _get_form_value(
                current_values,
                CONF_MAX_ENTITIES_PER_DISCOVERY,
                DEFAULT_MAX_ENTITIES_PER_DISCOVERY,
            ),
            CONF_ENABLE_ASSIST_BRIDGE: _get_form_value(
                current_values,
                CONF_ENABLE_ASSIST_BRIDGE,
                DEFAULT_ENABLE_ASSIST_BRIDGE,
            ),
            CONF_ENABLE_LLM_API_BRIDGE: _get_form_value(
                current_values,
                CONF_ENABLE_LLM_API_BRIDGE,
                DEFAULT_ENABLE_LLM_API_BRIDGE,
            ),
            CONF_LLM_API_ALLOWLIST: _get_form_value(
                current_values,
                CONF_LLM_API_ALLOWLIST,
                DEFAULT_LLM_API_ALLOWLIST,
            ),
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
            ),
            CONF_ENABLE_WEATHER_FORECAST_TOOL: _get_form_value(
                current_values,
                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
            ),
            CONF_ENABLE_RECORDER_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RECORDER_TOOLS,
                DEFAULT_ENABLE_RECORDER_TOOLS,
            ),
            CONF_ENABLE_MEMORY_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_MEMORY_TOOLS,
                DEFAULT_ENABLE_MEMORY_TOOLS,
            ),
            CONF_ENABLE_DEVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_DEVICE_TOOLS,
                DEFAULT_ENABLE_DEVICE_TOOLS,
            ),
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: _get_form_value(
                current_values,
                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
            ),
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
            ),
            CONF_MEMORY_DEFAULT_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_DEFAULT_TTL_DAYS,
                DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
            ),
            CONF_MEMORY_MAX_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_TTL_DAYS,
                DEFAULT_MEMORY_MAX_TTL_DAYS,
            ),
            CONF_MEMORY_MAX_ITEMS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_ITEMS,
                DEFAULT_MEMORY_MAX_ITEMS,
            ),
        }
        for spec in built_in_specs:
            shared_defaults[spec.shared_setting_key] = _get_form_value(
                current_values,
                spec.shared_setting_key,
                get_builtin_shared_setting_value(
                    spec,
                    lambda key, default=None: current_values.get(key, default),
                ),
            )

        # Build schema for MCP server settings
        mcp_schema = vol.Schema(
            {
                vol.Required(
                    CONF_MCP_PORT,
                    default=_get_form_value(
                        current_values, CONF_MCP_PORT, DEFAULT_MCP_PORT
                    ),
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_ALLOWED_IPS,
                    default=_get_form_value(
                        current_values, CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS
                    ),
                ): str,
                vol.Optional(
                    CONF_MCP_BEARER_TOKEN,
                    default=_get_form_value(
                        current_values,
                        CONF_MCP_BEARER_TOKEN,
                        self._generated_mcp_bearer_token,
                    ),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                CONTEXT_SECTION_KEY: _build_shared_context_section(shared_defaults),
                DISCOVERY_SECTION_KEY: _build_shared_discovery_section(shared_defaults),
                TOOLS_SECTION_KEY: _build_shared_tools_section(
                    shared_defaults,
                    built_in_specs,
                ),
            }
        )

        return self.async_show_form(
            step_id="mcp_server",
            data_schema=mcp_schema,
            errors=errors,
            description_placeholders={
                "info": (
                    "⚠️ These settings define the shared MCP server capabilities "
                    "available to all profiles and external MCP clients. Individual "
                    "profiles can still disable specific tool families later."
                ),
                "installed_llm_apis": _format_installed_llm_api_options(self.hass),
            },
        )

    async def async_step_system(self, data: dict[str, Any]) -> FlowResult:
        """Handle programmatic creation of system entry (no UI)."""
        # Set unique ID for system entry
        await self.async_set_unique_id(SYSTEM_ENTRY_UNIQUE_ID)
        self._abort_if_unique_id_configured()

        # Create system entry with provided data
        return self.async_create_entry(
            title="Shared MCP Server Settings",
            data=data,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get options flow for this handler."""
        return MCPAssistOptionsFlow()


class MCPAssistOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for MCP Assist integration."""

    def __init__(self) -> None:
        """Initialize options flow."""
        super().__init__()
        self.profile_options: dict[str, Any] = {}

    def _get_search_provider_default(self, options: dict, data: dict) -> str:
        """Get default search provider with backward compatibility."""
        # Check if search_provider is already set
        provider = options.get(CONF_SEARCH_PROVIDER, data.get(CONF_SEARCH_PROVIDER))
        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if options.get(
            CONF_ENABLE_CUSTOM_TOOLS, data.get(CONF_ENABLE_CUSTOM_TOOLS, False)
        ):
            return "brave"

        return DEFAULT_SEARCH_PROVIDER

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        # Check if this is the system entry - skip directly to MCP server settings
        if self.config_entry.unique_id == SYSTEM_ENTRY_UNIQUE_ID:
            return await self.async_step_mcp_server()

        errors: dict[str, str] = {}
        server_type = self.config_entry.data.get(CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE)
        provider_class = get_llm_provider_class(server_type)
        default_system_prompt = _get_default_system_prompt(self.hass)
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                PROFILE_SECTION_KEY,
                CONNECTION_SECTION_KEY,
                MODEL_SECTION_KEY,
                PROMPTS_SECTION_KEY,
                CONVERSATION_SECTION_KEY,
                PROVIDER_SECTION_KEY,
                ADVANCED_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _apply_profile_tool_disables(user_input, built_in_specs)
            user_input = _normalize_prompt_inputs(
                user_input, server_type, default_system_prompt
            )

            if not errors:
                # Support both old and new config keys
                if (
                    CONF_FOLLOW_UP_MODE in user_input
                    and CONF_RESPONSE_MODE not in user_input
                ):
                    user_input[CONF_RESPONSE_MODE] = user_input[CONF_FOLLOW_UP_MODE]
                    del user_input[CONF_FOLLOW_UP_MODE]

                # For OpenClaw, ensure model name and empty system prompt are set
                server_type = self.config_entry.data.get(
                    CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE
                )
                if server_type == SERVER_TYPE_OPENCLAW:
                    if CONF_MODEL_NAME not in user_input:
                        user_input[CONF_MODEL_NAME] = "main"
                    user_input[CONF_SYSTEM_PROMPT] = ""
                    user_input[CONF_TECHNICAL_PROMPT] = ""
                    user_input[CONF_SYSTEM_PROMPT_MODE] = PROMPT_MODE_DEFAULT
                    user_input[CONF_TECHNICAL_PROMPT_MODE] = PROMPT_MODE_DEFAULT

                # Store profile settings and proceed to MCP server settings
                self.profile_options = user_input
                return await self.async_step_mcp_server()

        # Get current values from options, then data, then defaults
        options = self.config_entry.options
        data = self.config_entry.data
        current_values = self.profile_options or {}

        # Handle backward compatibility
        response_mode_value = options.get(
            CONF_RESPONSE_MODE, options.get(CONF_FOLLOW_UP_MODE, DEFAULT_RESPONSE_MODE)
        )

        # Fetch models based on server type
        models = []
        current_model = current_values.get(
            CONF_MODEL_NAME,
            options.get(CONF_MODEL_NAME, data.get(CONF_MODEL_NAME, DEFAULT_MODEL_NAME)),
        )

        if provider_class.uses_config_model_step:
            provider_values = _merge_provider_values(data, options, current_values)
            models = await provider_class.fetch_models(self.hass, provider_values)

        # Build model selector based on whether models were fetched
        if models:
            # Show dropdown with available models (custom_value allows free text input)
            model_selector = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            )
        else:
            # Show text input as fallback
            model_selector = str

        schema_dict: dict[Any, Any] = {
            PROFILE_SECTION_KEY: _build_profile_identity_section(
                _get_form_value(
                    current_values,
                    CONF_PROFILE_NAME,
                    options.get(
                        CONF_PROFILE_NAME,
                        data.get(CONF_PROFILE_NAME, "Default"),
                    ),
                )
            ),
        }

        system_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_SYSTEM_PROMPT,
            stored_mode=options.get(
                CONF_SYSTEM_PROMPT_MODE, data.get(CONF_SYSTEM_PROMPT_MODE)
            ),
            stored_prompt=options.get(CONF_SYSTEM_PROMPT, data.get(CONF_SYSTEM_PROMPT)),
            default_prompt=default_system_prompt,
        )
        technical_prompt_suggestion = _get_prompt_text_default(
            current_values,
            prompt_key=CONF_TECHNICAL_PROMPT,
            stored_mode=options.get(
                CONF_TECHNICAL_PROMPT_MODE,
                data.get(CONF_TECHNICAL_PROMPT_MODE),
            ),
            stored_prompt=options.get(
                CONF_TECHNICAL_PROMPT, data.get(CONF_TECHNICAL_PROMPT)
            ),
            default_prompt=DEFAULT_TECHNICAL_PROMPT,
        )

        connection_schema_items = _build_provider_field_schema_items(
            provider_class.connection_fields,
            current_values,
            options,
            data,
        )

        schema_dict[CONNECTION_SECTION_KEY] = _build_connection_section(
            connection_schema_items
        )

        if provider_class.uses_config_model_step:
            schema_dict[MODEL_SECTION_KEY] = _build_model_section(
                current_model, model_selector
            )

        if provider_class.uses_config_model_step:
            schema_dict[PROMPTS_SECTION_KEY] = _build_prompt_section(
                include_system_prompt=True,
                system_prompt_value=system_prompt_suggestion,
                technical_prompt_value=technical_prompt_suggestion,
            )

        provider_schema_items: dict[Any, Any] = {}

        if server_type == SERVER_TYPE_OPENCLAW:
            schema_dict[CONVERSATION_SECTION_KEY] = _build_conversation_section(
                {
                    vol.Required(
                        CONF_CONTROL_HA,
                        default=_get_form_value(
                            current_values,
                            CONF_CONTROL_HA,
                            options.get(
                                CONF_CONTROL_HA,
                                data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA),
                            ),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_RESPONSE_MODE, default=response_mode_value
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": "none", "label": "None"},
                                {"value": "default", "label": "Smart"},
                                {"value": "always", "label": "Always"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FOLLOW_UP_PHRASES,
                        default=options.get(
                            CONF_FOLLOW_UP_PHRASES,
                            data.get(CONF_FOLLOW_UP_PHRASES, DEFAULT_FOLLOW_UP_PHRASES),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_END_WORDS,
                        default=options.get(
                            CONF_END_WORDS, data.get(CONF_END_WORDS, DEFAULT_END_WORDS)
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_CLEAN_RESPONSES,
                        default=_get_form_value(
                            current_values,
                            CONF_CLEAN_RESPONSES,
                            options.get(
                                CONF_CLEAN_RESPONSES,
                                data.get(
                                    CONF_CLEAN_RESPONSES,
                                    DEFAULT_CLEAN_RESPONSES,
                                ),
                            ),
                        ),
                    ): bool,
                }
            )
            provider_schema_items = _build_provider_field_schema_items(
                provider_class.provider_options_fields,
                current_values,
                options,
                data,
            )
            advanced_schema_items: dict[Any, Any] = {
                vol.Required(
                    CONF_TIMEOUT,
                    default=_get_form_value(
                        current_values,
                        CONF_TIMEOUT,
                        options.get(
                            CONF_TIMEOUT,
                            data.get(CONF_TIMEOUT, 60),
                        ),
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                vol.Required(
                    CONF_DEBUG_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_DEBUG_MODE,
                        options.get(
                            CONF_DEBUG_MODE,
                            data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE),
                        ),
                    ),
                ): bool,
                vol.Required(
                    CONF_CHAT_LOG_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_CHAT_LOG_MODE,
                        options.get(
                            CONF_CHAT_LOG_MODE,
                            data.get(CONF_CHAT_LOG_MODE, DEFAULT_CHAT_LOG_MODE),
                        ),
                    ),
                ): bool,
            }
        else:
            schema_dict[CONVERSATION_SECTION_KEY] = _build_conversation_section(
                {
                    vol.Required(
                        CONF_CONTROL_HA,
                        default=_get_form_value(
                            current_values,
                            CONF_CONTROL_HA,
                            options.get(
                                CONF_CONTROL_HA,
                                data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA),
                            ),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_RESPONSE_MODE,
                        default=_get_form_value(
                            current_values,
                            CONF_RESPONSE_MODE,
                            response_mode_value,
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": "none", "label": "None"},
                                {"value": "default", "label": "Smart"},
                                {"value": "always", "label": "Always"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FOLLOW_UP_PHRASES,
                        default=_get_form_value(
                            current_values,
                            CONF_FOLLOW_UP_PHRASES,
                            options.get(
                                CONF_FOLLOW_UP_PHRASES,
                                data.get(
                                    CONF_FOLLOW_UP_PHRASES,
                                    DEFAULT_FOLLOW_UP_PHRASES,
                                ),
                            ),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Optional(
                        CONF_END_WORDS,
                        default=_get_form_value(
                            current_values,
                            CONF_END_WORDS,
                            options.get(
                                CONF_END_WORDS,
                                data.get(CONF_END_WORDS, DEFAULT_END_WORDS),
                            ),
                        ),
                    ): TextSelector(TextSelectorConfig(multiline=True)),
                    vol.Required(
                        CONF_CLEAN_RESPONSES,
                        default=_get_form_value(
                            current_values,
                            CONF_CLEAN_RESPONSES,
                            options.get(
                                CONF_CLEAN_RESPONSES,
                                data.get(
                                    CONF_CLEAN_RESPONSES,
                                    DEFAULT_CLEAN_RESPONSES,
                                ),
                            ),
                        ),
                    ): bool,
                }
            )
            advanced_schema_items = {
                vol.Required(
                    CONF_TEMPERATURE,
                    default=_get_form_value(
                        current_values,
                        CONF_TEMPERATURE,
                        options.get(
                            CONF_TEMPERATURE,
                            data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                        ),
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Required(
                    CONF_MAX_TOKENS,
                    default=_get_form_value(
                        current_values,
                        CONF_MAX_TOKENS,
                        options.get(
                            CONF_MAX_TOKENS,
                            data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Required(
                    CONF_MAX_HISTORY,
                    default=_get_form_value(
                        current_values,
                        CONF_MAX_HISTORY,
                        options.get(
                            CONF_MAX_HISTORY,
                            data.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Required(
                    CONF_CONTEXT_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_CONTEXT_MODE,
                        options.get(
                            CONF_CONTEXT_MODE,
                            data.get(CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE),
                        ),
                    ),
                ): _context_mode_selector(),
                vol.Required(
                    CONF_MAX_ITERATIONS,
                    default=_get_form_value(
                        current_values,
                        CONF_MAX_ITERATIONS,
                        options.get(
                            CONF_MAX_ITERATIONS,
                            data.get(
                                CONF_MAX_ITERATIONS,
                                DEFAULT_MAX_ITERATIONS,
                            ),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Required(
                    CONF_TIMEOUT,
                    default=_get_form_value(
                        current_values,
                        CONF_TIMEOUT,
                        options.get(
                            CONF_TIMEOUT,
                            data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                        ),
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                vol.Required(
                    CONF_DEBUG_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_DEBUG_MODE,
                        options.get(
                            CONF_DEBUG_MODE,
                            data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE),
                        ),
                    ),
                ): bool,
                vol.Required(
                    CONF_CHAT_LOG_MODE,
                    default=_get_form_value(
                        current_values,
                        CONF_CHAT_LOG_MODE,
                        options.get(
                            CONF_CHAT_LOG_MODE,
                            data.get(CONF_CHAT_LOG_MODE, DEFAULT_CHAT_LOG_MODE),
                        ),
                    ),
                ): bool,
            }
            provider_schema_items = _build_provider_field_schema_items(
                provider_class.provider_options_fields,
                current_values,
                options,
                data,
            )

        if provider_schema_items:
            schema_dict[PROVIDER_SECTION_KEY] = _build_provider_section(
                provider_schema_items
            )
        schema_dict[TOOLS_SECTION_KEY] = _build_profile_tools_section(
            current_values,
            built_in_specs,
            options,
            data,
        )
        schema_dict[ADVANCED_SECTION_KEY] = _build_advanced_section(
            advanced_schema_items
        )

        options_schema = vol.Schema(schema_dict)

        # Set description based on server type
        if server_type == SERVER_TYPE_OPENCLAW:
            description_placeholders = {
                "server_info": (
                    "OpenClaw manages the model and system prompt on the gateway. "
                    "Use the connection, conversation, provider, and advanced "
                    "sections here to control how this Home Assistant profile "
                    "connects and follows up. The Tools section can still disable "
                    "specific shared MCP tool families for this profile."
                )
            }
        else:
            description_placeholders = {
                "server_info": (
                    "Configure this conversation profile. These settings only affect "
                    "this profile. The prompt fields are prefilled with the current "
                    "effective prompts so you can review, copy, or edit them directly. "
                    "If you leave a prompt unchanged, the integration keeps using the "
                    "built-in version from code. "
                    "The Tools section can disable specific shared MCP tool families "
                    "for smaller models."
                )
            }

        return self.async_show_form(
            step_id="init",
            data_schema=options_schema,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_mcp_server(self, user_input=None):
        """Configure shared MCP server settings (affects all profiles)."""
        errors: dict[str, str] = {}
        current_values = user_input or {}
        built_in_specs = await _async_load_builtin_tool_toggle_specs(self.hass)

        if user_input is not None:
            user_input = _flatten_section_values(
                user_input,
                CONTEXT_SECTION_KEY,
                DISCOVERY_SECTION_KEY,
                MEMORY_SECTION_KEY,
                TOOLS_SECTION_KEY,
            )
            user_input = _normalize_shared_tool_inputs(user_input, built_in_specs)
            current_values = user_input

            # Validate MCP port
            mcp_port = user_input.get(CONF_MCP_PORT, DEFAULT_MCP_PORT)
            if not 1024 <= mcp_port <= 65535:
                errors[CONF_MCP_PORT] = "invalid_port"

            # Validate allowed IPs
            allowed_ips_str = user_input.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS)
            is_valid, error_msg = validate_allowed_ips(allowed_ips_str)
            if not is_valid:
                errors[CONF_ALLOWED_IPS] = "invalid_ip"
                _LOGGER.warning("Invalid allowed IPs in options: %s", error_msg)

            if CONF_MCP_BEARER_TOKEN in user_input:
                token_value = user_input[CONF_MCP_BEARER_TOKEN]
            else:
                from . import get_system_entry

                system_entry = get_system_entry(self.hass)
                if system_entry:
                    token_value = system_entry.options.get(
                        CONF_MCP_BEARER_TOKEN,
                        system_entry.data.get(
                            CONF_MCP_BEARER_TOKEN,
                            DEFAULT_MCP_BEARER_TOKEN,
                        ),
                    )
                else:
                    token_value = self.config_entry.options.get(
                        CONF_MCP_BEARER_TOKEN,
                        self.config_entry.data.get(
                            CONF_MCP_BEARER_TOKEN,
                            DEFAULT_MCP_BEARER_TOKEN,
                        ),
                    )
            user_input[CONF_MCP_BEARER_TOKEN] = _normalize_mcp_bearer_token(token_value)
            if not _validate_mcp_bearer_token(user_input[CONF_MCP_BEARER_TOKEN]):
                errors[CONF_MCP_BEARER_TOKEN] = "mcp_bearer_token_too_short"

            _validate_shared_search_settings(user_input, built_in_specs, errors)
            _validate_shared_google_maps_settings(user_input, built_in_specs, errors)

            if not errors:
                # Import get_system_entry
                from . import get_system_entry, _async_apply_shared_mcp_settings

                shared_settings_applied = True
                # Update system entry with shared MCP settings
                system_entry = get_system_entry(self.hass)
                if system_entry:
                    previous_system_data = dict(system_entry.data)
                    new_system_data = {**previous_system_data, **user_input}
                    self.hass.config_entries.async_update_entry(
                        system_entry, data=new_system_data
                    )
                    try:
                        await _async_apply_shared_mcp_settings(self.hass)
                    except Exception as err:
                        self.hass.config_entries.async_update_entry(
                            system_entry, data=previous_system_data
                        )
                        shared_settings_applied = False
                        errors["base"] = "mcp_apply_failed"
                        _LOGGER.warning(
                            "Failed to apply shared MCP settings live: %s",
                            type(err).__name__,
                        )
                    if shared_settings_applied:
                        _LOGGER.info("Updated system entry with shared MCP settings")
                else:
                    _LOGGER.error("System entry not found when saving shared settings")
                    shared_settings_applied = False
                    errors["base"] = "mcp_apply_failed"

                if shared_settings_applied:
                    # Update profile entry with per-profile settings only
                    # Update entry title if profile name changed
                    new_profile_name = self.profile_options.get(CONF_PROFILE_NAME)
                    old_profile_name = self.config_entry.options.get(
                        CONF_PROFILE_NAME, self.config_entry.data.get(CONF_PROFILE_NAME)
                    )
                    if new_profile_name and new_profile_name != old_profile_name:
                        server_type = self.config_entry.data.get(
                            CONF_SERVER_TYPE, DEFAULT_SERVER_TYPE
                        )
                        server_display = _provider_display_name(server_type)
                        self.hass.config_entries.async_update_entry(
                            self.config_entry,
                            title=f"{server_display} - {new_profile_name}",
                        )

                    # Save profile settings only (not shared settings)
                    return self.async_create_entry(title="", data=self.profile_options)

        # Get current values from system entry
        from . import get_system_entry

        system_entry = get_system_entry(self.hass)

        # Get shared settings from system entry (with fallback to profile for backward compat)
        if system_entry:
            sys_options = system_entry.options
            sys_data = system_entry.data
        else:
            # Fallback to profile entry for backward compatibility
            sys_options = self.config_entry.options
            sys_data = self.config_entry.data

        shared_defaults = {
            CONF_SEARCH_PROVIDER: _get_form_value(
                current_values,
                CONF_SEARCH_PROVIDER,
                self._get_search_provider_default(sys_options, sys_data),
            ),
            CONF_ENABLE_WEB_SEARCH: _get_form_value(
                current_values,
                CONF_ENABLE_WEB_SEARCH,
                _infer_web_search_enabled(
                    sys_options.get(
                        CONF_ENABLE_WEB_SEARCH,
                        sys_data.get(CONF_ENABLE_WEB_SEARCH),
                    ),
                    self._get_search_provider_default(sys_options, sys_data),
                    sys_options.get(
                        CONF_ENABLE_CUSTOM_TOOLS,
                        sys_data.get(CONF_ENABLE_CUSTOM_TOOLS, False),
                    ),
                ),
            ),
            CONF_BRAVE_API_KEY: _get_form_value(
                current_values,
                CONF_BRAVE_API_KEY,
                sys_options.get(
                    CONF_BRAVE_API_KEY,
                    sys_data.get(CONF_BRAVE_API_KEY, DEFAULT_BRAVE_API_KEY),
                ),
            ),
            CONF_GOOGLE_MAPS_API_KEY: _get_form_value(
                current_values,
                CONF_GOOGLE_MAPS_API_KEY,
                sys_options.get(
                    CONF_GOOGLE_MAPS_API_KEY,
                    sys_data.get(CONF_GOOGLE_MAPS_API_KEY, DEFAULT_GOOGLE_MAPS_API_KEY),
                ),
            ),
            CONF_SEARXNG_URL: _get_form_value(
                current_values,
                CONF_SEARXNG_URL,
                sys_options.get(
                    CONF_SEARXNG_URL,
                    sys_data.get(CONF_SEARXNG_URL, DEFAULT_SEARXNG_URL),
                ),
            ),
            CONF_INCLUDE_CURRENT_USER: _get_form_value(
                current_values,
                CONF_INCLUDE_CURRENT_USER,
                sys_options.get(
                    CONF_INCLUDE_CURRENT_USER,
                    sys_data.get(
                        CONF_INCLUDE_CURRENT_USER,
                        DEFAULT_INCLUDE_CURRENT_USER,
                    ),
                ),
            ),
            CONF_INCLUDE_HOME_LOCATION: _get_form_value(
                current_values,
                CONF_INCLUDE_HOME_LOCATION,
                sys_options.get(
                    CONF_INCLUDE_HOME_LOCATION,
                    sys_data.get(
                        CONF_INCLUDE_HOME_LOCATION,
                        DEFAULT_INCLUDE_HOME_LOCATION,
                    ),
                ),
            ),
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: _get_form_value(
                current_values,
                CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                sys_options.get(
                    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                    sys_data.get(
                        CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                        DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
                    ),
                ),
            ),
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: _get_form_value(
                current_values,
                CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                sys_options.get(
                    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                    sys_data.get(
                        CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                        DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
                    ),
                ),
            ),
            CONF_ENABLE_GAP_FILLING: _get_form_value(
                current_values,
                CONF_ENABLE_GAP_FILLING,
                sys_options.get(
                    CONF_ENABLE_GAP_FILLING,
                    sys_data.get(CONF_ENABLE_GAP_FILLING, DEFAULT_ENABLE_GAP_FILLING),
                ),
            ),
            CONF_ENABLE_ASSIST_BRIDGE: _get_form_value(
                current_values,
                CONF_ENABLE_ASSIST_BRIDGE,
                sys_options.get(
                    CONF_ENABLE_ASSIST_BRIDGE,
                    sys_data.get(
                        CONF_ENABLE_ASSIST_BRIDGE, DEFAULT_ENABLE_ASSIST_BRIDGE
                    ),
                ),
            ),
            CONF_ENABLE_LLM_API_BRIDGE: _get_form_value(
                current_values,
                CONF_ENABLE_LLM_API_BRIDGE,
                sys_options.get(
                    CONF_ENABLE_LLM_API_BRIDGE,
                    sys_data.get(
                        CONF_ENABLE_LLM_API_BRIDGE, DEFAULT_ENABLE_LLM_API_BRIDGE
                    ),
                ),
            ),
            CONF_LLM_API_ALLOWLIST: _get_form_value(
                current_values,
                CONF_LLM_API_ALLOWLIST,
                sys_options.get(
                    CONF_LLM_API_ALLOWLIST,
                    sys_data.get(CONF_LLM_API_ALLOWLIST, DEFAULT_LLM_API_ALLOWLIST),
                ),
            ),
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                sys_options.get(
                    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
                        DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
                    ),
                ),
            ),
            CONF_ENABLE_WEATHER_FORECAST_TOOL: _get_form_value(
                current_values,
                CONF_ENABLE_WEATHER_FORECAST_TOOL,
                sys_options.get(
                    CONF_ENABLE_WEATHER_FORECAST_TOOL,
                    sys_data.get(
                        CONF_ENABLE_WEATHER_FORECAST_TOOL,
                        DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
                    ),
                ),
            ),
            CONF_ENABLE_RECORDER_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_RECORDER_TOOLS,
                sys_options.get(
                    CONF_ENABLE_RECORDER_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_RECORDER_TOOLS,
                        DEFAULT_ENABLE_RECORDER_TOOLS,
                    ),
                ),
            ),
            CONF_ENABLE_MEMORY_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_MEMORY_TOOLS,
                sys_options.get(
                    CONF_ENABLE_MEMORY_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_MEMORY_TOOLS,
                        DEFAULT_ENABLE_MEMORY_TOOLS,
                    ),
                ),
            ),
            CONF_ENABLE_DEVICE_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_DEVICE_TOOLS,
                sys_options.get(
                    CONF_ENABLE_DEVICE_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_DEVICE_TOOLS, DEFAULT_ENABLE_DEVICE_TOOLS
                    ),
                ),
            ),
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: _get_form_value(
                current_values,
                CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                sys_options.get(
                    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                    sys_data.get(
                        CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                        DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
                    ),
                ),
            ),
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: _get_form_value(
                current_values,
                CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                sys_options.get(
                    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                    sys_data.get(
                        CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                        DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
                    ),
                ),
            ),
            CONF_MEMORY_DEFAULT_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_DEFAULT_TTL_DAYS,
                sys_options.get(
                    CONF_MEMORY_DEFAULT_TTL_DAYS,
                    sys_data.get(
                        CONF_MEMORY_DEFAULT_TTL_DAYS,
                        DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
                    ),
                ),
            ),
            CONF_MEMORY_MAX_TTL_DAYS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_TTL_DAYS,
                sys_options.get(
                    CONF_MEMORY_MAX_TTL_DAYS,
                    sys_data.get(
                        CONF_MEMORY_MAX_TTL_DAYS,
                        DEFAULT_MEMORY_MAX_TTL_DAYS,
                    ),
                ),
            ),
            CONF_MEMORY_MAX_ITEMS: _get_form_value(
                current_values,
                CONF_MEMORY_MAX_ITEMS,
                sys_options.get(
                    CONF_MEMORY_MAX_ITEMS,
                    sys_data.get(
                        CONF_MEMORY_MAX_ITEMS,
                        DEFAULT_MEMORY_MAX_ITEMS,
                    ),
                ),
            ),
            CONF_MAX_ENTITIES_PER_DISCOVERY: _get_form_value(
                current_values,
                CONF_MAX_ENTITIES_PER_DISCOVERY,
                sys_options.get(
                    CONF_MAX_ENTITIES_PER_DISCOVERY,
                    sys_data.get(
                        CONF_MAX_ENTITIES_PER_DISCOVERY,
                        DEFAULT_MAX_ENTITIES_PER_DISCOVERY,
                    ),
                ),
            ),
        }
        for spec in built_in_specs:
            shared_defaults[spec.shared_setting_key] = _get_form_value(
                current_values,
                spec.shared_setting_key,
                get_builtin_shared_setting_value(
                    spec,
                    lambda key, default=None: sys_options.get(
                        key, sys_data.get(key, default)
                    ),
                ),
            )

        # Build schema for MCP server settings
        mcp_schema = vol.Schema(
            {
                vol.Required(
                    CONF_MCP_PORT,
                    default=_get_form_value(
                        current_values,
                        CONF_MCP_PORT,
                        sys_options.get(
                            CONF_MCP_PORT,
                            sys_data.get(CONF_MCP_PORT, DEFAULT_MCP_PORT),
                        ),
                    ),
                ): vol.Coerce(int),
                vol.Optional(
                    CONF_ALLOWED_IPS,
                    default=_get_form_value(
                        current_values,
                        CONF_ALLOWED_IPS,
                        sys_options.get(
                            CONF_ALLOWED_IPS,
                            sys_data.get(CONF_ALLOWED_IPS, DEFAULT_ALLOWED_IPS),
                        ),
                    ),
                ): str,
                vol.Optional(
                    CONF_MCP_BEARER_TOKEN,
                    default=_get_form_value(
                        current_values,
                        CONF_MCP_BEARER_TOKEN,
                        sys_options.get(
                            CONF_MCP_BEARER_TOKEN,
                            sys_data.get(
                                CONF_MCP_BEARER_TOKEN,
                                DEFAULT_MCP_BEARER_TOKEN,
                            ),
                        ),
                    ),
                ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
                CONTEXT_SECTION_KEY: _build_shared_context_section(shared_defaults),
                DISCOVERY_SECTION_KEY: _build_shared_discovery_section(shared_defaults),
                TOOLS_SECTION_KEY: _build_shared_tools_section(
                    shared_defaults,
                    built_in_specs,
                ),
            }
        )

        return self.async_show_form(
            step_id="mcp_server",
            data_schema=mcp_schema,
            errors=errors,
            description_placeholders={
                "warning": (
                    "⚠️ These settings are shared across ALL MCP Assist profiles and "
                    "external MCP clients. Individual profiles can still opt into a "
                    "smaller subset in their own settings."
                ),
                "installed_llm_apis": _format_installed_llm_api_options(self.hass),
            },
        )
