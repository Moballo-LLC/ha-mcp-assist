"""Tests for MCP Assist config flow helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import voluptuous as vol
import voluptuous_serialize
from homeassistant.data_entry_flow import FlowResultType, section
from homeassistant.helpers.selector import TemplateSelector

from custom_components.mcp_assist import config_flow as config_flow_module
from custom_components.mcp_assist.config_flow import (
    ADVANCED_SECTION_KEY,
    CONNECTION_SECTION_KEY,
    CONVERSATION_SECTION_KEY,
    CONTEXT_SECTION_KEY,
    DISCOVERY_SECTION_KEY,
    DISABLE_ASSIST_BRIDGE_FIELD,
    DISABLE_CUSTOM_TOOLS_FIELD,
    DISABLE_DEVICE_FIELD,
    DISABLE_LLM_API_BRIDGE_FIELD,
    DISABLE_MEMORY_FIELD,
    DISABLE_MUSIC_ASSISTANT_FIELD,
    DISABLE_RECORDER_FIELD,
    DISABLE_RESPONSE_SERVICE_FIELD,
    DISABLE_WEATHER_FORECAST_FIELD,
    MCPAssistConfigFlow,
    MCPAssistOptionsFlow,
    MEMORY_SECTION_KEY,
    MODEL_SECTION_KEY,
    PERFORMANCE_SECTION_KEY,
    PROFILE_SECTION_KEY,
    PROMPTS_SECTION_KEY,
    PROVIDER_SECTION_KEY,
    PROFILE_DISABLE_FIELD_BY_FAMILY,
    SERVER_SECTION_KEY,
    STATIC_TOOL_FAMILY_ALPHABETICAL,
    TOOLS_SECTION_KEY,
    _build_profile_tools_section,
    _build_shared_tools_section,
    _format_installed_llm_api_options,
    _apply_profile_tool_disables,
    _infer_prompt_mode,
    _needs_prompt_followup,
    _normalize_shared_tool_inputs,
    _normalize_prompt_inputs,
    _redacted_log_snippet,
    validate_allowed_ips,
)
from custom_components.mcp_assist.tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.llm_providers.ollama import OllamaProvider
from custom_components.mcp_assist.const import (
    CONF_API_KEY,
    CONF_ALLOWED_IPS,
    CONF_BRAVE_API_KEY,
    CONF_CHAT_LOG_MODE,
    CONF_CLEAN_RESPONSES,
    CONF_CONTROL_HA,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_MODE,
    CONF_ENABLE_GAP_FILLING,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_GOOGLE_MAPS_API_KEY,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_END_WORDS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_FOLLOW_UP_PHRASES,
    CONF_LLM_API_ALLOWLIST,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_MAX_ENTITIES_PER_DISCOVERY,
    CONF_MAX_HISTORY,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MCP_PORT,
    CONF_MCP_BEARER_TOKEN,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    CONF_OPENCLAW_SESSION_KEY,
    CONF_PROFILE_NAME,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_PROFILE_ENABLE_LLM_API_BRIDGE,
    CONF_RESPONSE_MODE,
    CONF_SEARCH_PROVIDER,
    CONF_SEARXNG_URL,
    CONF_SERVER_TYPE,
    CONF_STATEFUL_SESSION_ID,
    CONF_SYSTEM_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TEMPERATURE,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_TIMEOUT,
    DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    DEFAULT_MEMORY_MAX_TTL_DAYS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TECHNICAL_PROMPT,
    OPENAI_BASE_URL,
    PROMPT_MODE_CUSTOM,
    PROMPT_MODE_DEFAULT,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENCLAW,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
)

BUILTIN_SPECS = load_builtin_tool_toggle_specs()
BUILTIN_SPEC_BY_PACKAGE = {spec.package_id: spec for spec in BUILTIN_SPECS}
PROFILE_BUILTIN_ORDER = [
    spec.profile_disable_label
    for spec in sorted(BUILTIN_SPECS, key=lambda item: item.profile_disable_label.casefold())
]
SHARED_BUILTIN_ORDER = [
    spec.shared_label
    for spec in sorted(BUILTIN_SPECS, key=lambda item: item.shared_label.casefold())
]


def _builtin_shared_key(package_id: str) -> str:
    """Return the shared form key for a built-in packaged tool."""
    return BUILTIN_SPEC_BY_PACKAGE[package_id].shared_setting_key


def _builtin_profile_key(package_id: str) -> str:
    """Return the profile-disable form key for a built-in packaged tool."""
    return f"disable_{package_id}"


PROFILE_TOOL_ORDER = [
    DISABLE_ASSIST_BRIDGE_FIELD,
    _builtin_profile_key("calculator"),
    DISABLE_CUSTOM_TOOLS_FIELD,
    DISABLE_DEVICE_FIELD,
    _builtin_profile_key("google_maps"),
    DISABLE_LLM_API_BRIDGE_FIELD,
    DISABLE_MEMORY_FIELD,
    DISABLE_MUSIC_ASSISTANT_FIELD,
    _builtin_profile_key("read_url"),
    DISABLE_RECORDER_FIELD,
    DISABLE_RESPONSE_SERVICE_FIELD,
    _builtin_profile_key("search"),
    _builtin_profile_key("unit_conversion"),
    DISABLE_WEATHER_FORECAST_FIELD,
    _builtin_profile_key("wikipedia_search"),
]

SHARED_TOOL_SECTION_ORDER = [
    CONF_ENABLE_ASSIST_BRIDGE,
    _builtin_shared_key("calculator"),
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_DEVICE_TOOLS,
    _builtin_shared_key("google_maps"),
    CONF_GOOGLE_MAPS_API_KEY,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_LLM_API_ALLOWLIST,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    _builtin_shared_key("read_url"),
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    _builtin_shared_key("search"),
    CONF_SEARCH_PROVIDER,
    CONF_BRAVE_API_KEY,
    CONF_SEARXNG_URL,
    _builtin_shared_key("unit_conversion"),
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    _builtin_shared_key("wikipedia_search"),
]

SHARED_MEMORY_SECTION_ORDER = [
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
]
CONFIG_STRINGS_PATH = Path("custom_components/mcp_assist/strings.json")
TRANSLATION_DIR = Path("custom_components/mcp_assist/translations")


def _config_string_paths(data: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Return leaf translation paths from a config string tree."""
    if isinstance(data, dict):
        paths: set[tuple[str, ...]] = set()
        for key, value in data.items():
            paths.update(_config_string_paths(value, (*prefix, key)))
        return paths

    return {prefix}


def _load_config_strings() -> list[tuple[Path, dict[str, Any]]]:
    """Load base and localized config string files."""
    paths = [CONFIG_STRINGS_PATH, *sorted(TRANSLATION_DIR.glob("*.json"))]
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    ]


def _section_field_names(form_section: section) -> set[str]:
    """Return normalized field names from a flow section."""
    return {
        getattr(marker, "schema", marker)
        for marker in form_section.schema.schema.keys()
    }


def _schema_marker_by_field(data_schema: vol.Schema) -> dict[str, Any]:
    """Return schema markers by normalized field name."""
    return {
        getattr(marker, "schema", marker): marker
        for marker in data_schema.schema.keys()
    }


def _schema_section(data_schema: vol.Schema, key: str) -> section:
    """Return a section from a schema keyed by voluptuous markers."""
    for marker, value in data_schema.schema.items():
        if getattr(marker, "schema", marker) == key:
            return value
    raise KeyError(key)


def test_builtin_tool_toggle_specs_include_ui_descriptions() -> None:
    """Built-in packaged-tool metadata should carry UI subtitles for both forms."""
    for spec in BUILTIN_SPECS:
        assert spec.shared_description
        assert spec.profile_disable_description


def test_infer_prompt_mode_defaults_when_prompt_matches_builtin() -> None:
    """Legacy prompt storage should infer default mode when the prompt matches the built-in text."""
    assert _infer_prompt_mode(None, "builtin prompt", "builtin prompt") == PROMPT_MODE_DEFAULT
    assert _infer_prompt_mode(None, "custom prompt", "builtin prompt") == PROMPT_MODE_CUSTOM


def test_normalize_prompt_inputs_drops_default_prompt_text() -> None:
    """Blank prompt overrides should fall back to built-in defaults."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT: "",
            CONF_TECHNICAL_PROMPT: "   ",
        },
        server_type="ollama",
        default_system_prompt="builtin system",
    )

    assert CONF_SYSTEM_PROMPT not in normalized
    assert CONF_TECHNICAL_PROMPT not in normalized
    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert normalized[CONF_TECHNICAL_PROMPT_MODE] == PROMPT_MODE_DEFAULT


def test_normalize_prompt_inputs_marks_nonblank_prompts_as_custom() -> None:
    """Nonblank prompt overrides should be stored as custom values."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT: "Be formal",
            CONF_TECHNICAL_PROMPT: "Always inspect attributes",
        },
        server_type="ollama",
        default_system_prompt="builtin system",
    )

    assert normalized[CONF_SYSTEM_PROMPT] == "Be formal"
    assert normalized[CONF_TECHNICAL_PROMPT] == "Always inspect attributes"
    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_CUSTOM
    assert normalized[CONF_TECHNICAL_PROMPT_MODE] == PROMPT_MODE_CUSTOM


def test_normalize_prompt_inputs_for_openclaw_forces_default_system_prompt() -> None:
    """OpenClaw should always ignore custom system prompts."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_SYSTEM_PROMPT: "custom",
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: "keep this one",
        },
        server_type=SERVER_TYPE_OPENCLAW,
        default_system_prompt="builtin system",
    )

    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert CONF_SYSTEM_PROMPT not in normalized
    assert normalized[CONF_TECHNICAL_PROMPT] == "keep this one"


def test_normalize_prompt_inputs_treats_builtin_prompt_text_as_default() -> None:
    """Prompt text matching the built-in default should not be stored as custom."""
    normalized = _normalize_prompt_inputs(
        {
            CONF_SYSTEM_PROMPT: "builtin system",
            CONF_TECHNICAL_PROMPT: DEFAULT_TECHNICAL_PROMPT,
        },
        server_type="ollama",
        default_system_prompt="builtin system",
    )

    assert CONF_SYSTEM_PROMPT not in normalized
    assert CONF_TECHNICAL_PROMPT not in normalized
    assert normalized[CONF_SYSTEM_PROMPT_MODE] == PROMPT_MODE_DEFAULT
    assert normalized[CONF_TECHNICAL_PROMPT_MODE] == PROMPT_MODE_DEFAULT


def test_needs_prompt_followup_when_switching_prompt_visibility() -> None:
    """Prompt followup is disabled because the fields are always visible."""
    assert _needs_prompt_followup({}, server_type="ollama") is False


def test_validate_allowed_ips_accepts_ips_and_cidr_ranges() -> None:
    """Allowed IP parsing should accept both addresses and CIDR entries."""
    assert validate_allowed_ips("192.168.1.10,10.0.0.0/24") == (True, "")


def test_validate_allowed_ips_rejects_invalid_values() -> None:
    """Allowed IP parsing should reject malformed values."""
    is_valid, message = validate_allowed_ips("192.168.1.10,not-an-ip")

    assert is_valid is False
    assert "not-an-ip" in message


def test_redacted_log_snippet_removes_provider_secret_markers() -> None:
    """Provider error snippets should not echo common credential fields."""
    snippet = _redacted_log_snippet(
        "request failed: https://api.example/models?key=gemini-secret "
        "https://api.example/models?key=sk-live-value "
        "https://user:password@example.local/v1/models "
        "Authorization: Bearer openai-secret api_key=custom-secret "
        '{"api_key":"quoted-secret","error":"API key provided: sk-prose-secret"}'
    )

    assert "gemini-secret" not in snippet
    assert "sk-live-value" not in snippet
    assert "password" not in snippet
    assert "openai-secret" not in snippet
    assert "custom-secret" not in snippet
    assert "quoted-secret" not in snippet
    assert "sk-prose-secret" not in snippet
    assert "Authorization" not in snippet
    assert "Bearer" not in snippet
    assert "api_key" not in snippet
    assert "[redacted]" in snippet


async def test_system_flow_creates_shared_config_entry(hass) -> None:
    """The system-source config flow should create the shared config entry without UI steps."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "system"}

    result = await flow.async_step_system({CONF_MCP_PORT: 7788})

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Shared MCP Server Settings"
    assert result["data"][CONF_MCP_PORT] == 7788


async def test_model_step_always_shows_prompt_fields_without_mode_dropdowns(hass) -> None:
    """The model step should always expose prompt textareas directly."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}
    flow.step2_data = {CONF_LMSTUDIO_URL: "http://localhost:11434"}

    with patch(
        "custom_components.mcp_assist.llm_providers.ollama.OllamaProvider.fetch_models",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_model()

    top_level_keys = set(_schema_marker_by_field(result["data_schema"]))
    assert MODEL_SECTION_KEY in top_level_keys
    assert PROMPTS_SECTION_KEY in top_level_keys

    prompts_section = _schema_section(result["data_schema"], PROMPTS_SECTION_KEY)
    assert isinstance(prompts_section, section)

    prompt_keys = {
        getattr(key, "schema", key) for key in prompts_section.schema.schema.keys()
    }
    assert CONF_SYSTEM_PROMPT in prompt_keys
    assert CONF_TECHNICAL_PROMPT in prompt_keys
    assert all(
        isinstance(selector, TemplateSelector)
        for selector in prompts_section.schema.schema.values()
    )


async def test_server_step_uses_provider_specific_connection_fields(hass) -> None:
    """Provider setup should show the connection fields required by that provider."""
    ollama_flow = MCPAssistConfigFlow()
    ollama_flow.hass = hass
    ollama_flow.context = {"source": "user"}
    ollama_flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}

    ollama_result = await ollama_flow.async_step_server()
    ollama_markers = _schema_marker_by_field(ollama_result["data_schema"])

    assert set(ollama_markers) == {CONF_LMSTUDIO_URL}
    assert ollama_markers[CONF_LMSTUDIO_URL].default() == DEFAULT_OLLAMA_URL

    openai_flow = MCPAssistConfigFlow()
    openai_flow.hass = hass
    openai_flow.context = {"source": "user"}
    openai_flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OPENAI}

    openai_result = await openai_flow.async_step_server()
    openai_markers = _schema_marker_by_field(openai_result["data_schema"])

    assert set(openai_markers) == {CONF_LMSTUDIO_URL, CONF_API_KEY}
    assert openai_markers[CONF_LMSTUDIO_URL].default() == OPENAI_BASE_URL


async def test_model_step_prompt_overrides_are_optional(hass) -> None:
    """Prompt fields should be optional and prefilled with the effective prompts."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}
    flow.step2_data = {CONF_LMSTUDIO_URL: "http://localhost:11434"}

    with patch(
        "custom_components.mcp_assist.llm_providers.ollama.OllamaProvider.fetch_models",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_model()

    prompts_section = _schema_section(result["data_schema"], PROMPTS_SECTION_KEY)
    marker_by_key = {
        getattr(marker, "schema", marker): marker
        for marker in prompts_section.schema.schema.keys()
    }

    assert isinstance(marker_by_key[CONF_SYSTEM_PROMPT], vol.Optional)
    assert isinstance(marker_by_key[CONF_TECHNICAL_PROMPT], vol.Optional)
    assert marker_by_key[CONF_SYSTEM_PROMPT].description["suggested_value"]
    assert marker_by_key[CONF_TECHNICAL_PROMPT].description["suggested_value"]


def test_apply_profile_tool_disables_marks_checked_tools_disabled() -> None:
    """Checked profile tool toggles should store disabled flags."""
    normalized = _apply_profile_tool_disables(
        {
            DISABLE_DEVICE_FIELD: True,
            DISABLE_ASSIST_BRIDGE_FIELD: True,
            DISABLE_CUSTOM_TOOLS_FIELD: True,
            DISABLE_LLM_API_BRIDGE_FIELD: True,
            _builtin_profile_key("calculator"): True,
            _builtin_profile_key("search"): True,
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_PROFILE_ENABLE_DEVICE_TOOLS] is False
    assert normalized[CONF_PROFILE_ENABLE_ASSIST_BRIDGE] is False
    assert normalized[CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS] is False
    assert normalized[CONF_PROFILE_ENABLE_LLM_API_BRIDGE] is False
    assert normalized["profile_enable_calculator_tools"] is False
    assert normalized["profile_enable_search_tool"] is False


def test_apply_profile_tool_disables_leaves_unchecked_tools_inherited() -> None:
    """Unchecked profile tool toggles should fall back to shared settings."""
    normalized = _apply_profile_tool_disables(
        {
            DISABLE_DEVICE_FIELD: False,
            DISABLE_ASSIST_BRIDGE_FIELD: False,
            DISABLE_CUSTOM_TOOLS_FIELD: False,
            DISABLE_LLM_API_BRIDGE_FIELD: False,
            _builtin_profile_key("calculator"): False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: False,
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_PROFILE_ENABLE_LLM_API_BRIDGE: False,
            "profile_enable_calculator_tools": False,
        },
        BUILTIN_SPECS,
    )

    assert CONF_PROFILE_ENABLE_DEVICE_TOOLS not in normalized
    assert CONF_PROFILE_ENABLE_ASSIST_BRIDGE not in normalized
    assert CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS not in normalized
    assert CONF_PROFILE_ENABLE_LLM_API_BRIDGE not in normalized
    assert "profile_enable_calculator_tools" not in normalized


def test_normalize_shared_tool_inputs_maps_built_in_fields_to_setting_keys() -> None:
    """Built-in toggle fields should store real shared setting keys."""
    normalized = _normalize_shared_tool_inputs(
        {
            _builtin_shared_key("search"): True,
            _builtin_shared_key("read_url"): True,
            CONF_SEARCH_PROVIDER: "",
        },
        BUILTIN_SPECS,
    )

    assert normalized["enable_search_tool"] is True
    assert normalized["enable_read_url_tool"] is True
    assert normalized[CONF_SEARCH_PROVIDER] == "duckduckgo"


def test_normalize_shared_tool_inputs_accepts_legacy_built_in_labels() -> None:
    """Older in-flight form data using built-in labels should still normalize."""
    normalized = _normalize_shared_tool_inputs(
        {
            BUILTIN_SPEC_BY_PACKAGE["search"].shared_label: True,
            BUILTIN_SPEC_BY_PACKAGE["read_url"].shared_label: True,
            CONF_SEARCH_PROVIDER: "",
        },
        BUILTIN_SPECS,
    )

    assert normalized["enable_search_tool"] is True
    assert normalized["enable_read_url_tool"] is True
    assert BUILTIN_SPEC_BY_PACKAGE["search"].shared_label not in normalized
    assert BUILTIN_SPEC_BY_PACKAGE["read_url"].shared_label not in normalized


def test_normalize_shared_tool_inputs_infers_provider_from_legacy_web_search() -> None:
    """Legacy web-search enablement should still infer a real provider."""
    normalized = _normalize_shared_tool_inputs(
        {
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "",
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_SEARCH_PROVIDER] == "duckduckgo"


def test_normalize_shared_tool_inputs_normalizes_llm_api_allowlist() -> None:
    """LLM API allowlists should accept commas, newlines, and repeated ids."""
    normalized = _normalize_shared_tool_inputs(
        {
            CONF_ENABLE_LLM_API_BRIDGE: True,
            CONF_LLM_API_ALLOWLIST: "llm_intents\nmusic_api, llm_intents",
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_ENABLE_LLM_API_BRIDGE] is True
    assert normalized[CONF_LLM_API_ALLOWLIST] == "llm_intents, music_api"


def test_format_installed_llm_api_options_lists_registered_third_party_apis(
    hass, monkeypatch
) -> None:
    """The shared settings helper should show copyable installed third-party API ids."""
    monkeypatch.setattr(
        config_flow_module.llm,
        "async_get_apis",
        lambda hass_arg: [
            SimpleNamespace(
                id=config_flow_module.llm.LLM_API_ASSIST,
                name="Assist",
            ),
            SimpleNamespace(id="llm_intents", name="LLM Intents"),
            SimpleNamespace(id="calendar_tools", name="Calendar Tools"),
        ],
    )

    assert _format_installed_llm_api_options(hass) == (
        "Calendar Tools (calendar_tools), LLM Intents (llm_intents)"
    )


def test_normalize_shared_tool_inputs_clamps_memory_ttls() -> None:
    """Shared memory TTL settings should be coerced into a safe valid range."""
    normalized = _normalize_shared_tool_inputs(
        {
            CONF_MEMORY_MAX_TTL_DAYS: 3,
            CONF_MEMORY_DEFAULT_TTL_DAYS: 99,
        },
        BUILTIN_SPECS,
    )

    assert normalized[CONF_MEMORY_MAX_TTL_DAYS] == 3
    assert normalized[CONF_MEMORY_DEFAULT_TTL_DAYS] == 3

    fallback = _normalize_shared_tool_inputs(
        {
            CONF_MEMORY_MAX_TTL_DAYS: "oops",
            CONF_MEMORY_DEFAULT_TTL_DAYS: "nope",
        },
        BUILTIN_SPECS,
    )

    assert fallback[CONF_MEMORY_MAX_TTL_DAYS] == DEFAULT_MEMORY_MAX_TTL_DAYS
    assert fallback[CONF_MEMORY_DEFAULT_TTL_DAYS] == DEFAULT_MEMORY_DEFAULT_TTL_DAYS


async def test_advanced_step_groups_profile_tools_into_checkbox_section(hass) -> None:
    """Advanced settings should expose per-profile tool disable checkboxes."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}

    result = await flow.async_step_advanced()

    top_level_keys = set(_schema_marker_by_field(result["data_schema"]))
    assert CONVERSATION_SECTION_KEY in top_level_keys
    assert PERFORMANCE_SECTION_KEY in top_level_keys
    performance_section = _schema_section(result["data_schema"], PERFORMANCE_SECTION_KEY)
    tools_section = _schema_section(result["data_schema"], TOOLS_SECTION_KEY)
    assert CONF_CONTEXT_MODE in _section_field_names(performance_section)
    assert CONF_CHAT_LOG_MODE in _section_field_names(performance_section)
    assert isinstance(tools_section, section)

    section_keys = [
        getattr(key, "schema", key) for key in tools_section.schema.schema.keys()
    ]
    assert section_keys == PROFILE_TOOL_ORDER
    assert all(value is bool for value in tools_section.schema.schema.values())
    marker_by_key = {
        getattr(marker, "schema", marker): marker
        for marker in tools_section.schema.schema.keys()
    }
    assert marker_by_key[_builtin_profile_key("calculator")].description is None
    assert marker_by_key[_builtin_profile_key("search")].description is None


async def test_advanced_step_preserves_provider_fields_from_sections(hass) -> None:
    """Initial setup should flatten provider-owned fields before shared MCP setup."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}
    flow.step1_data = {
        CONF_PROFILE_NAME: "Test Profile",
        CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
    }
    flow.step2_data = {CONF_LMSTUDIO_URL: "http://localhost:11434"}
    flow.step3_data = {CONF_MODEL_NAME: "qwen3"}

    result = await flow.async_step_advanced(
        {
            CONVERSATION_SECTION_KEY: {
                CONF_CONTROL_HA: True,
                CONF_RESPONSE_MODE: "default",
                CONF_FOLLOW_UP_PHRASES: "Anything else?",
                CONF_END_WORDS: "stop",
                CONF_CLEAN_RESPONSES: True,
            },
            PERFORMANCE_SECTION_KEY: {
                CONF_TEMPERATURE: 0.4,
                CONF_MAX_TOKENS: 2048,
                CONF_MAX_HISTORY: 8,
                CONF_CONTEXT_MODE: "standard",
                CONF_MAX_ITERATIONS: 6,
                CONF_TIMEOUT: 45,
                CONF_DEBUG_MODE: False,
                CONF_CHAT_LOG_MODE: True,
            },
            PROVIDER_SECTION_KEY: {
                CONF_OLLAMA_NUM_CTX: 12288,
                CONF_OLLAMA_KEEP_ALIVE: "30m",
            },
            TOOLS_SECTION_KEY: {
                DISABLE_DEVICE_FIELD: True,
            },
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "mcp_server"
    assert flow.step4_data[CONF_OLLAMA_NUM_CTX] == 12288
    assert flow.step4_data[CONF_OLLAMA_KEEP_ALIVE] == "30m"
    assert flow.step4_data[CONF_TEMPERATURE] == 0.4
    assert flow.step4_data[CONF_CONTEXT_MODE] == "standard"
    assert flow.step4_data[CONF_PROFILE_ENABLE_DEVICE_TOOLS] is False


async def test_shared_mcp_step_groups_context_discovery_and_tools(
    hass, monkeypatch
) -> None:
    """Shared MCP settings should group tool-specific settings next to their tools."""
    monkeypatch.setattr(
        config_flow_module.llm,
        "async_get_apis",
        lambda hass_arg: [
            SimpleNamespace(
                id=config_flow_module.llm.LLM_API_ASSIST,
                name="Assist",
            ),
            SimpleNamespace(id="llm_intents", name="LLM Intents"),
        ],
    )
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}

    result = await flow.async_step_mcp_server()

    server_section = _schema_section(result["data_schema"], SERVER_SECTION_KEY)
    context_section = _schema_section(result["data_schema"], CONTEXT_SECTION_KEY)
    discovery_section = _schema_section(result["data_schema"], DISCOVERY_SECTION_KEY)
    memory_section = _schema_section(result["data_schema"], MEMORY_SECTION_KEY)
    tools_section = _schema_section(result["data_schema"], TOOLS_SECTION_KEY)

    assert isinstance(server_section, section)
    assert isinstance(context_section, section)
    assert isinstance(discovery_section, section)
    assert isinstance(memory_section, section)
    assert isinstance(tools_section, section)
    top_level_keys = {
        getattr(key, "schema", key) for key in result["data_schema"].schema.keys()
    }
    assert top_level_keys == {
        SERVER_SECTION_KEY,
        CONTEXT_SECTION_KEY,
        DISCOVERY_SECTION_KEY,
        MEMORY_SECTION_KEY,
        TOOLS_SECTION_KEY,
    }
    token_marker = next(
        key
        for key in server_section.schema.schema
        if getattr(key, "schema", key) == CONF_MCP_BEARER_TOKEN
    )
    token_default = token_marker.default
    token_value = token_default() if callable(token_default) else token_default
    assert isinstance(token_value, str)
    assert len(token_value) >= 32

    server_keys = [
        getattr(key, "schema", key) for key in server_section.schema.schema.keys()
    ]
    context_keys = {
        getattr(key, "schema", key) for key in context_section.schema.schema.keys()
    }
    discovery_keys = {
        getattr(key, "schema", key) for key in discovery_section.schema.schema.keys()
    }
    memory_keys = [
        getattr(key, "schema", key) for key in memory_section.schema.schema.keys()
    ]
    tool_keys = [
        getattr(key, "schema", key) for key in tools_section.schema.schema.keys()
    ]

    assert server_keys == [
        CONF_MCP_PORT,
        CONF_ALLOWED_IPS,
        CONF_MCP_BEARER_TOKEN,
    ]
    assert context_keys == {
        CONF_INCLUDE_CURRENT_USER,
        CONF_INCLUDE_HOME_LOCATION,
        CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
        CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    }
    assert discovery_keys == {
        CONF_ENABLE_GAP_FILLING,
        CONF_MAX_ENTITIES_PER_DISCOVERY,
    }
    assert memory_keys == SHARED_MEMORY_SECTION_ORDER
    assert tool_keys == SHARED_TOOL_SECTION_ORDER
    tool_markers = {
        getattr(marker, "schema", marker): marker
        for marker in tools_section.schema.schema.keys()
    }
    search_selector = tools_section.schema.schema[tool_markers[CONF_SEARCH_PROVIDER]]
    assert {
        option["value"] for option in search_selector.config["options"]
    } == {"duckduckgo", "brave", "searxng"}
    external_default = tool_markers[CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS].default
    assert external_default() is False if callable(external_default) else external_default is False
    recorder_default = tool_markers[_builtin_shared_key("recorder")].default
    assert recorder_default() is True if callable(recorder_default) else recorder_default is True
    response_default = tool_markers[_builtin_shared_key("response_service")].default
    assert response_default() is True if callable(response_default) else response_default is True
    assert tool_markers[_builtin_shared_key("calculator")].description is None
    assert tool_markers[_builtin_shared_key("read_url")].description is None
    assert result["description_placeholders"]["installed_llm_apis"] == (
        "LLM Intents (llm_intents)"
    )


async def test_shared_mcp_step_requires_searxng_url_when_selected(hass) -> None:
    """SearXNG search should not save without a base URL."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}

    result = await flow.async_step_mcp_server(
        {
            CONF_MCP_PORT: 8090,
            _builtin_shared_key("search"): True,
            CONF_SEARCH_PROVIDER: "searxng",
            CONF_SEARXNG_URL: " ",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"][CONF_SEARXNG_URL] == "searxng_url_required"


async def test_shared_mcp_step_requires_google_maps_api_key_when_enabled(hass) -> None:
    """Google Maps tools should not save without an API key."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}

    result = await flow.async_step_mcp_server(
        {
            CONF_MCP_PORT: 8090,
            _builtin_shared_key("google_maps"): True,
            CONF_GOOGLE_MAPS_API_KEY: " ",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"][CONF_GOOGLE_MAPS_API_KEY] == "google_maps_api_key_required"


async def test_shared_mcp_step_rejects_short_bearer_token(hass) -> None:
    """Shared MCP bearer tokens should be strong when enabled."""
    flow = MCPAssistConfigFlow()
    flow.hass = hass
    flow.context = {"source": "user"}

    result = await flow.async_step_mcp_server(
        {
            CONF_MCP_PORT: 8090,
            CONF_MCP_BEARER_TOKEN: "short",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"][CONF_MCP_BEARER_TOKEN] == "mcp_bearer_token_too_short"


async def test_options_mcp_step_requires_searxng_url_when_selected(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared options should reject SearXNG without a base URL."""
    system_entry_factory()
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}

    result = await flow.async_step_mcp_server(
        {
            CONF_MCP_PORT: 8090,
            _builtin_shared_key("search"): True,
            CONF_SEARCH_PROVIDER: "searxng",
            CONF_SEARXNG_URL: "",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"][CONF_SEARXNG_URL] == "searxng_url_required"


async def test_options_mcp_step_requires_google_maps_api_key_when_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared options should reject Google Maps tools without an API key."""
    system_entry_factory()

    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}

    result = await flow.async_step_mcp_server(
        {
            CONF_MCP_PORT: 8090,
            _builtin_shared_key("google_maps"): True,
            CONF_GOOGLE_MAPS_API_KEY: "",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"][CONF_GOOGLE_MAPS_API_KEY] == "google_maps_api_key_required"


async def test_options_mcp_step_rejects_invalid_shared_port(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared options should not persist an invalid MCP server port."""
    system_entry = system_entry_factory(data={CONF_MCP_PORT: 8090})
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}

    result = await flow.async_step_mcp_server(
        {
            CONF_MCP_PORT: 80,
            CONF_ALLOWED_IPS: "127.0.0.1",
            _builtin_shared_key("search"): False,
            CONF_SEARCH_PROVIDER: "none",
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"][CONF_MCP_PORT] == "invalid_port"
    assert system_entry.data[CONF_MCP_PORT] == 8090


async def test_options_mcp_step_preserves_bearer_token_default(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared options should keep the saved MCP bearer token in form defaults."""
    system_entry_factory(data={CONF_MCP_BEARER_TOKEN: "saved-token-123456"})
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}

    result = await flow.async_step_mcp_server()

    server_section = _schema_section(result["data_schema"], SERVER_SECTION_KEY)
    token_marker = next(
        key
        for key in server_section.schema.schema
        if getattr(key, "schema", key) == CONF_MCP_BEARER_TOKEN
    )
    token_default = token_marker.default
    token_value = token_default() if callable(token_default) else token_default
    assert token_value == "saved-token-123456"


async def test_options_mcp_step_regenerates_bearer_token_from_sentinel(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Entering FFFF should rotate the shared MCP bearer token."""
    system_entry = system_entry_factory(
        data={
            CONF_MCP_PORT: 8090,
            CONF_ALLOWED_IPS: "127.0.0.1",
            CONF_MCP_BEARER_TOKEN: "old-token-123456",
        }
    )
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}
    apply_mock = AsyncMock()

    with (
        patch(
            "custom_components.mcp_assist.config_flow.generate_mcp_bearer_token",
            return_value="regenerated-token-123456",
        ),
        patch("custom_components.mcp_assist._async_apply_shared_mcp_settings", apply_mock),
    ):
        result = await flow.async_step_mcp_server(
            {
                SERVER_SECTION_KEY: {
                    CONF_MCP_PORT: 8090,
                    CONF_ALLOWED_IPS: "127.0.0.1",
                    CONF_MCP_BEARER_TOKEN: "ffff",
                },
                TOOLS_SECTION_KEY: {
                    _builtin_shared_key("search"): False,
                    CONF_SEARCH_PROVIDER: "none",
                },
            }
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert system_entry.data[CONF_MCP_BEARER_TOKEN] == "regenerated-token-123456"
    apply_mock.assert_awaited_once_with(hass)


async def test_options_mcp_step_preserves_google_maps_api_key_default(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Shared options should keep the saved Google Maps key in form defaults."""
    system_entry_factory(data={CONF_GOOGLE_MAPS_API_KEY: "saved-maps-key"})
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}

    result = await flow.async_step_mcp_server()

    tools_section = _schema_section(result["data_schema"], TOOLS_SECTION_KEY)
    tool_markers = {
        getattr(marker, "schema", marker): marker
        for marker in tools_section.schema.schema.keys()
    }
    maps_default = tool_markers[CONF_GOOGLE_MAPS_API_KEY].default
    resolved_default = maps_default() if callable(maps_default) else maps_default
    assert resolved_default == "saved-maps-key"


async def test_options_mcp_step_applies_shared_settings_to_running_server(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Saving shared MCP settings should apply them without a full HA restart."""
    system_entry = system_entry_factory(data={CONF_MCP_BEARER_TOKEN: "saved-token-123456"})
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}
    apply_mock = AsyncMock()

    with patch("custom_components.mcp_assist._async_apply_shared_mcp_settings", apply_mock):
        result = await flow.async_step_mcp_server(
            {
                CONF_MCP_PORT: 8124,
                CONF_ALLOWED_IPS: "192.168.1.25",
                _builtin_shared_key("search"): False,
                CONF_SEARCH_PROVIDER: "none",
            }
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert system_entry.data[CONF_MCP_PORT] == 8124
    assert system_entry.data[CONF_ALLOWED_IPS] == "192.168.1.25"
    assert system_entry.data[CONF_MCP_BEARER_TOKEN] == "saved-token-123456"
    apply_mock.assert_awaited_once_with(hass)


async def test_options_mcp_step_rolls_back_shared_settings_when_apply_fails(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A failed live apply should not persist an unusable shared MCP port."""
    system_entry = system_entry_factory(
        data={CONF_MCP_PORT: 8090, CONF_ALLOWED_IPS: "10.0.0.0/24"}
    )
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id
    flow.profile_options = {}
    apply_mock = AsyncMock(side_effect=OSError("address in use"))

    with patch("custom_components.mcp_assist._async_apply_shared_mcp_settings", apply_mock):
        result = await flow.async_step_mcp_server(
            {
                CONF_MCP_PORT: 8124,
                CONF_ALLOWED_IPS: "192.168.1.25",
                _builtin_shared_key("search"): False,
                CONF_SEARCH_PROVIDER: "none",
            }
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "mcp_apply_failed"
    assert system_entry.data[CONF_MCP_PORT] == 8090
    assert system_entry.data[CONF_ALLOWED_IPS] == "10.0.0.0/24"
    apply_mock.assert_awaited_once_with(hass)


def test_built_in_tool_checkboxes_rely_on_translation_subtitles() -> None:
    """Built-in packaged tool checkboxes should not override translated subtitles inline."""
    shared_section = _build_shared_tools_section({}, BUILTIN_SPECS)
    profile_section = _build_profile_tools_section({}, BUILTIN_SPECS, {}, {})

    shared_names = {
        _builtin_shared_key("calculator"),
        _builtin_shared_key("read_url"),
        _builtin_shared_key("search"),
        _builtin_shared_key("unit_conversion"),
        _builtin_shared_key("wikipedia_search"),
    }
    profile_names = {
        _builtin_profile_key("calculator"),
        _builtin_profile_key("read_url"),
        _builtin_profile_key("search"),
        _builtin_profile_key("unit_conversion"),
        _builtin_profile_key("wikipedia_search"),
    }

    shared_checkbox_fields = {
        marker: value
        for marker, value in shared_section.schema.schema.items()
        if getattr(marker, "schema", marker) in shared_names
    }
    profile_checkbox_fields = {
        marker: value
        for marker, value in profile_section.schema.schema.items()
        if getattr(marker, "schema", marker) in profile_names
    }

    shared_serialized = voluptuous_serialize.convert(vol.Schema(shared_checkbox_fields))
    profile_serialized = voluptuous_serialize.convert(vol.Schema(profile_checkbox_fields))

    shared_by_name = {item["name"]: item for item in shared_serialized}
    profile_by_name = {item["name"]: item for item in profile_serialized}

    for name in shared_names:
        assert "description" not in shared_by_name[name]

    for name in profile_names:
        assert "description" not in profile_by_name[name]


def test_optional_tool_family_checkbox_sets_stay_in_sync() -> None:
    """Every optional tool family should exist in both shared and profile checkbox builders."""
    assert set(STATIC_TOOL_FAMILY_ALPHABETICAL) <= set(TOOL_FAMILY_PROFILE_SETTINGS)
    assert set(STATIC_TOOL_FAMILY_ALPHABETICAL) <= set(TOOL_FAMILY_SHARED_SETTINGS)
    assert set(STATIC_TOOL_FAMILY_ALPHABETICAL) == set(PROFILE_DISABLE_FIELD_BY_FAMILY)


def test_tool_translations_cover_all_declared_tool_fields() -> None:
    """Shared/profile tool fields should still have labels and descriptions."""
    expected_profile_fields = {
        PROFILE_DISABLE_FIELD_BY_FAMILY[family]
        for family in STATIC_TOOL_FAMILY_ALPHABETICAL
    }
    expected_server_fields = {
        CONF_ALLOWED_IPS,
        CONF_MCP_BEARER_TOKEN,
        CONF_MCP_PORT,
    }
    expected_memory_fields = set(SHARED_MEMORY_SECTION_ORDER)
    expected_shared_fields = {
        TOOL_FAMILY_SHARED_SETTINGS[family][0]
        for family in STATIC_TOOL_FAMILY_ALPHABETICAL
    }
    expected_profile_fields.update(
        _builtin_profile_key(spec.package_id) for spec in BUILTIN_SPECS
    )
    expected_shared_fields.update(
        _builtin_shared_key(spec.package_id) for spec in BUILTIN_SPECS
    )
    expected_shared_fields.update(
        {
            CONF_BRAVE_API_KEY,
            CONF_GOOGLE_MAPS_API_KEY,
            CONF_LLM_API_ALLOWLIST,
            CONF_SEARCH_PROVIDER,
            CONF_SEARXNG_URL,
        }
    )

    for path, strings in _load_config_strings():
        for root, advanced_step in (("config", "advanced"), ("options", "init")):
            advanced_tools = strings[root]["step"][advanced_step]["sections"][
                TOOLS_SECTION_KEY
            ]
            shared_sections = strings[root]["step"]["mcp_server"]["sections"]
            shared_server = shared_sections[SERVER_SECTION_KEY]
            shared_memory = shared_sections[MEMORY_SECTION_KEY]
            shared_tools = shared_sections[TOOLS_SECTION_KEY]

            assert expected_profile_fields <= set(advanced_tools["data"]), path
            assert expected_profile_fields <= set(advanced_tools["data_description"]), path
            assert expected_server_fields <= set(shared_server["data"]), path
            assert expected_server_fields <= set(shared_server["data_description"]), path
            assert expected_memory_fields <= set(shared_memory["data"]), path
            assert expected_memory_fields <= set(shared_memory["data_description"]), path
            assert expected_shared_fields <= set(shared_tools["data"]), path
            assert expected_shared_fields <= set(shared_tools["data_description"]), path
            assert expected_memory_fields.isdisjoint(shared_tools["data"]), path
            assert expected_memory_fields.isdisjoint(shared_tools["data_description"]), path


def test_translation_files_match_base_config_string_keys() -> None:
    """Localized string files should keep all config-flow translation keys."""
    base_strings = json.loads(CONFIG_STRINGS_PATH.read_text(encoding="utf-8"))
    base_paths = _config_string_paths(base_strings)

    for path in sorted(TRANSLATION_DIR.glob("*.json")):
        localized_paths = _config_string_paths(
            json.loads(path.read_text(encoding="utf-8"))
        )

        assert localized_paths == base_paths, path


def test_provider_section_translations_cover_provider_specific_fields() -> None:
    """Provider-only settings should have section translations in both config and options flows."""
    strings = json.loads(
        Path("custom_components/mcp_assist/strings.json").read_text(encoding="utf-8")
    )

    expected_provider_fields = {
        CONF_OLLAMA_KEEP_ALIVE,
        CONF_OLLAMA_NUM_CTX,
        CONF_OPENCLAW_SESSION_KEY,
        CONF_STATEFUL_SESSION_ID,
    }

    for root, step in (("config", "advanced"), ("options", "init")):
        provider_section = strings[root]["step"][step]["sections"][PROVIDER_SECTION_KEY]
        assert expected_provider_fields <= set(provider_section["data"])
        assert expected_provider_fields <= set(provider_section["data_description"])


def test_performance_translations_cover_context_mode() -> None:
    """Context mode should have labels in setup and options performance sections."""
    strings = json.loads(
        Path("custom_components/mcp_assist/strings.json").read_text(encoding="utf-8")
    )

    for root, step, section_key in (
        ("config", "advanced", PERFORMANCE_SECTION_KEY),
        ("options", "init", ADVANCED_SECTION_KEY),
    ):
        performance_section = strings[root]["step"][step]["sections"][section_key]
        assert CONF_CONTEXT_MODE in performance_section["data"]
        assert CONF_CONTEXT_MODE in performance_section["data_description"]


async def test_options_step_groups_profile_settings_into_sections(
    hass, profile_entry_factory
) -> None:
    """Options flow should organize profile settings into clear sections."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory()
    flow.handler = entry.entry_id

    with patch(
        "custom_components.mcp_assist.llm_providers.lmstudio.LMStudioProvider.fetch_models",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_init()

    top_level_keys = set(_schema_marker_by_field(result["data_schema"]))
    assert PROFILE_SECTION_KEY in top_level_keys
    assert MODEL_SECTION_KEY in top_level_keys
    assert PROMPTS_SECTION_KEY in top_level_keys
    assert CONVERSATION_SECTION_KEY in top_level_keys
    assert ADVANCED_SECTION_KEY in top_level_keys
    assert TOOLS_SECTION_KEY in top_level_keys
    advanced_section = _schema_section(result["data_schema"], ADVANCED_SECTION_KEY)
    assert CONF_CONTEXT_MODE in _section_field_names(advanced_section)
    assert CONF_CHAT_LOG_MODE in _section_field_names(advanced_section)


async def test_options_step_for_ollama_keeps_provider_fields_in_provider_section(
    hass, profile_entry_factory
) -> None:
    """Ollama provider-only settings should live in the provider section, not advanced."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
            CONF_LMSTUDIO_URL: "http://localhost:11434",
        }
    )
    flow.handler = entry.entry_id

    with patch(
        "custom_components.mcp_assist.llm_providers.ollama.OllamaProvider.fetch_models",
        AsyncMock(return_value=["qwen3"]),
    ):
        result = await flow.async_step_init()

    provider_section = _schema_section(result["data_schema"], PROVIDER_SECTION_KEY)
    advanced_section = _schema_section(result["data_schema"], ADVANCED_SECTION_KEY)

    assert isinstance(provider_section, section)
    assert _section_field_names(provider_section) == {
        CONF_OLLAMA_NUM_CTX,
        CONF_OLLAMA_KEEP_ALIVE,
        CONF_STATEFUL_SESSION_ID,
    }
    assert CONF_OLLAMA_NUM_CTX not in _section_field_names(advanced_section)
    assert CONF_OLLAMA_KEEP_ALIVE not in _section_field_names(advanced_section)


async def test_options_submit_preserves_provider_fields_from_sections(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Options flow should save provider-owned fields from the standard sectioned form."""
    system_entry_factory()
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
            CONF_LMSTUDIO_URL: "http://localhost:11434",
        }
    )
    flow.handler = entry.entry_id

    result = await flow.async_step_init(
        {
            PROFILE_SECTION_KEY: {CONF_PROFILE_NAME: "Updated Profile"},
            CONNECTION_SECTION_KEY: {
                CONF_LMSTUDIO_URL: "http://ollama.example.invalid:11434",
            },
            MODEL_SECTION_KEY: {CONF_MODEL_NAME: "qwen3"},
            PROMPTS_SECTION_KEY: {
                CONF_SYSTEM_PROMPT: "Use short answers",
                CONF_TECHNICAL_PROMPT: "Prefer entity ids",
            },
            CONVERSATION_SECTION_KEY: {
                CONF_CONTROL_HA: True,
                CONF_RESPONSE_MODE: "always",
                CONF_FOLLOW_UP_PHRASES: "Anything else?",
                CONF_END_WORDS: "stop",
                CONF_CLEAN_RESPONSES: False,
            },
            PROVIDER_SECTION_KEY: {
                CONF_OLLAMA_NUM_CTX: 16384,
                CONF_OLLAMA_KEEP_ALIVE: "1h",
            },
            ADVANCED_SECTION_KEY: {
                CONF_TEMPERATURE: 0.2,
                CONF_MAX_TOKENS: 4096,
                CONF_MAX_HISTORY: 12,
                CONF_CONTEXT_MODE: "light",
                CONF_MAX_ITERATIONS: 7,
                CONF_TIMEOUT: 90,
                CONF_DEBUG_MODE: True,
                CONF_CHAT_LOG_MODE: False,
            },
            TOOLS_SECTION_KEY: {
                DISABLE_DEVICE_FIELD: True,
            },
        }
    )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "mcp_server"
    assert flow.profile_options[CONF_LMSTUDIO_URL] == (
        "http://ollama.example.invalid:11434"
    )
    assert flow.profile_options[CONF_MODEL_NAME] == "qwen3"
    assert flow.profile_options[CONF_OLLAMA_NUM_CTX] == 16384
    assert flow.profile_options[CONF_OLLAMA_KEEP_ALIVE] == "1h"
    assert flow.profile_options[CONF_RESPONSE_MODE] == "always"
    assert flow.profile_options[CONF_CONTEXT_MODE] == "light"
    assert flow.profile_options[CONF_PROFILE_ENABLE_DEVICE_TOOLS] is False


async def test_options_step_for_ollama_uses_provider_default_url_when_missing(
    hass, profile_entry_factory
) -> None:
    """Older Ollama entries without a stored URL should use Ollama's default."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        data={CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA, CONF_LMSTUDIO_URL: ""}
    )
    hass.config_entries.async_update_entry(
        entry,
        data={
            key: value
            for key, value in entry.data.items()
            if key != CONF_LMSTUDIO_URL
        },
    )
    flow.handler = entry.entry_id

    fetch_models = AsyncMock(return_value=["qwen3"])
    with patch(
        "custom_components.mcp_assist.llm_providers.ollama.OllamaProvider.fetch_models",
        fetch_models,
    ):
        result = await flow.async_step_init()

    fetch_models.assert_awaited_once()
    provider_values = fetch_models.await_args.args[1]
    assert OllamaProvider.model_base_url(provider_values) == DEFAULT_OLLAMA_URL
    assert OllamaProvider.model_base_url({CONF_LMSTUDIO_URL: ""}) == DEFAULT_OLLAMA_URL
    assert CONNECTION_SECTION_KEY in _schema_marker_by_field(result["data_schema"])
    connection_section = _schema_section(result["data_schema"], CONNECTION_SECTION_KEY)
    assert CONF_LMSTUDIO_URL in _section_field_names(connection_section)


async def test_options_step_for_openai_fetches_models_from_custom_base_url(
    hass, profile_entry_factory
) -> None:
    """OpenAI-compatible options should discover models from the saved base URL."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        title="OpenAI - Test Profile",
        unique_id="mcp_assist_openai_test_profile",
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_API_KEY: "sk-test",
            CONF_LMSTUDIO_URL: "https://proxy.example.com/v1",
        },
    )
    flow.handler = entry.entry_id

    fetch_models = AsyncMock(return_value=["proxy-model"])
    with patch(
        "custom_components.mcp_assist.llm_providers.openai.OpenAIProvider.fetch_models",
        fetch_models,
    ):
        result = await flow.async_step_init()

    fetch_models.assert_awaited_once()
    assert fetch_models.await_args.args[1][CONF_API_KEY] == "sk-test"
    assert (
        fetch_models.await_args.args[1][CONF_LMSTUDIO_URL]
        == "https://proxy.example.com/v1"
    )
    assert CONNECTION_SECTION_KEY in _schema_marker_by_field(result["data_schema"])
    connection_section = _schema_section(result["data_schema"], CONNECTION_SECTION_KEY)
    assert {CONF_LMSTUDIO_URL, CONF_API_KEY} <= _section_field_names(
        connection_section
    )


async def test_options_step_for_openclaw_hides_model_prompts_and_uses_provider_section(
    hass, profile_entry_factory
) -> None:
    """OpenClaw options should hide model/prompts and keep the session key in provider settings."""
    flow = MCPAssistOptionsFlow()
    flow.hass = hass
    entry = profile_entry_factory(
        title="OpenClaw - Test Profile",
        unique_id="mcp_assist_openclaw_test_profile",
        data={CONF_SERVER_TYPE: SERVER_TYPE_OPENCLAW},
    )
    flow.handler = entry.entry_id

    result = await flow.async_step_init()

    top_level_keys = set(_schema_marker_by_field(result["data_schema"]))
    provider_section = _schema_section(result["data_schema"], PROVIDER_SECTION_KEY)
    advanced_section = _schema_section(result["data_schema"], ADVANCED_SECTION_KEY)

    assert MODEL_SECTION_KEY not in top_level_keys
    assert PROMPTS_SECTION_KEY not in top_level_keys
    assert PROVIDER_SECTION_KEY in top_level_keys
    assert isinstance(provider_section, section)
    assert _section_field_names(provider_section) == {CONF_OPENCLAW_SESSION_KEY}
    assert CONF_OPENCLAW_SESSION_KEY not in _section_field_names(advanced_section)
