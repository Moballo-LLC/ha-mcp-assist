"""MCP Assist conversation agent."""

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation import chat_log
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    intent,
    area_registry as ar,
    device_registry as dr,
    llm,
)
from homeassistant.helpers.template import Template
from homeassistant.util import dt as dt_util

from .tools.builtin_catalog import (
    BuiltInToolToggleSpec,
    is_builtin_package_enabled_for_profile,
)
from .provider_runtime import resolve_provider_runtime_config
from .tool_schema import (
    ADAPTIVE_META_TOOL_NAMES,
    ADAPTIVE_TOOL_CATALOG_NAME,
    ADAPTIVE_TOOL_SCHEMA_NAME,
    build_adaptive_llm_tools,
    build_tool_routing_summary,
    compact_schema_for_llm,
    compact_text,
    convert_mcp_tools_to_llm_tools,
    json_size_bytes,
    match_adaptive_tool_definitions,
    normalize_adaptive_query_terms,
    score_adaptive_tool_match,
    tool_definition_name,
)
from .llm_providers import (
    LLMProvider,
    ProviderSettings,
    build_provider_settings,
    create_llm_provider,
    normalize_tool_call_arguments,
    parse_tool_arguments,
    stringify_tool_arguments,
)
from .localization import get_language_instruction
from .const import (
    DOMAIN,
    CONF_PROFILE_NAME,
    CONF_MODEL_NAME,
    CONF_MCP_PORT,
    CONF_SYSTEM_PROMPT,
    CONF_TECHNICAL_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_DEBUG_MODE,
    CONF_CHAT_LOG_MODE,
    CONF_MAX_ITERATIONS,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_FOLLOW_UP_MODE,
    CONF_RESPONSE_MODE,
    CONF_MAX_HISTORY,
    CONF_CONTEXT_MODE,
    CONF_SERVER_TYPE,
    CONF_CONTROL_HA,
    CONF_SEARCH_PROVIDER,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
    CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_MCP_BEARER_TOKEN,
    CONF_FOLLOW_UP_PHRASES,
    CONF_END_WORDS,
    CONF_CLEAN_RESPONSES,
    CONF_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    PROMPT_MODE_DEFAULT,
    PROMPT_MODE_CUSTOM,
    CONTEXT_MODE_ADAPTIVE,
    CONTEXT_MODE_LIGHT,
    CONTEXT_MODE_STANDARD,
    DEFAULT_DEBUG_MODE,
    DEFAULT_CHAT_LOG_MODE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_RESPONSE_MODE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_MCP_PORT,
    DEFAULT_CONTROL_HA,
    DEFAULT_FOLLOW_UP_PHRASES,
    DEFAULT_END_WORDS,
    DEFAULT_CLEAN_RESPONSES,
    DEFAULT_TIMEOUT,
    DEFAULT_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_INCLUDE_CURRENT_USER,
    DEFAULT_INCLUDE_HOME_LOCATION,
    DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS,
    DEFAULT_MCP_BEARER_TOKEN,
    LIGHT_CONTEXT_MAX_HISTORY,
    LIGHT_CONTEXT_TOOL_NAMES,
    RESPONSE_MODE_INSTRUCTIONS,
    DEVICE_TECHNICAL_INSTRUCTIONS,
    MEMORY_TECHNICAL_INSTRUCTIONS,
    ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS,
    LLM_API_BRIDGE_TECHNICAL_INSTRUCTIONS,
    MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS,
    SERVER_TYPE_OPENCLAW,
    TOOL_FAMILY_EXTERNAL_CUSTOM,
    TOOL_FAMILY_LLM_API_BRIDGE,
    TOOL_FAMILY_PROFILE_SETTINGS,
    TOOL_FAMILY_SHARED_SETTINGS,
    get_optional_tool_family,
    CONF_OPENCLAW_SESSION_KEY,
    DEFAULT_OPENCLAW_SESSION_KEY,
)
from .conversation_history import ConversationHistory

_LOGGER = logging.getLogger(__name__)


class RecoverableStreamingFallbackError(Exception):
    """Raised when streaming can safely fall back to provider HTTP transport."""


class EmptyStreamingResponseError(RecoverableStreamingFallbackError):
    """Raised when streaming completes without content or tool calls."""


class StatefulStreamingRequestError(Exception):
    """Raised when a stateful streaming request may already be provider-visible."""

# Tool schemas are invalidated by settings and custom-tool signatures; this TTL is
# just a safety refresh, not the primary change detector.
MCP_TOOL_CACHE_TTL_SECONDS = 300.0
ADAPTIVE_RETAINED_SCHEMA_LIMIT = 2
ADAPTIVE_FOLLOW_UP_MAX_WORDS = 8
ADAPTIVE_FOLLOW_UP_PREFIXES = (
    "and ",
    "also ",
    "how about ",
    "what about ",
    "what if ",
    "then ",
)
ADAPTIVE_FOLLOW_UP_REFERENCE_TERMS = frozenset(
    {
        "again",
        "it",
        "same",
        "that",
        "them",
        "there",
        "these",
        "those",
    }
)
MAX_TOOL_RESULT_CHARS = 8000
MAX_TOOL_RESULT_LINES = 120
TOOL_BUDGET_FINAL_TOOL_RESULT_CHARS = 2000
TOOL_BUDGET_FINAL_TOOL_RESULT_LINES = 40
MAX_PROVIDER_LOG_CHARS = 500
PROVIDER_HTTP_TIMEOUT_ATTEMPTS = 2
TOOL_HISTORY_ARGUMENT_KEYS = (
    "entity_id",
    "entity_ids",
    "area",
    "floor",
    "domain",
    "device_class",
    "state",
    "action",
    "service",
    "script",
    "automation",
    "name",
)
TOOL_BUDGET_EXHAUSTED_RESULT = (
    "Tool call skipped because this request reached the configured tool-call "
    "budget. Answer from the tool results already available without calling more "
    "tools."
)
TOOL_BUDGET_FINAL_INSTRUCTION = (
    "The configured MCP tool-call budget has been reached. Answer the user's "
    "request now using only the tool results already available. Do not request "
    "more tools. If the available results are partial, say that briefly."
)
TOOL_BUDGET_FALLBACK_RESPONSE = (
    "I used the available tool budget before I could finish. Try a narrower "
    "request, or raise Max Tool Iterations if this needs a broader check."
)
TOOLLESS_RETRY_INSTRUCTION = (
    "You just said you would check, but no MCP tool call was made. This is still "
    "the same user request. Do not ask the user to wait or confirm. Call the "
    "appropriate MCP tool now using the most specific available filters, or give "
    "the final answer only if no tool is needed. After using tools, answer in "
    "the user's language."
)
INVALID_TOOL_ARGUMENT_RETRY_INSTRUCTION = (
    "The previous MCP tool call had invalid JSON arguments and was not executed. "
    "This is still the same user request. Do not ask the user to confirm. Call "
    "the needed MCP tool again with a complete JSON object in function.arguments, "
    "or answer only if no tool is needed. After using tools, answer in the "
    "user's language."
)
INVALID_TOOL_ARGUMENT_FALLBACK_RESPONSE = (
    "I couldn't complete that because the model produced invalid tool-call "
    "arguments more than once. Try again, or use a model with stronger "
    "tool-calling support."
)
TOOLLESS_RESPONSE_PATTERNS = (
    re.compile(
        r"\b(?:i(?:'|’)?ll|i will|i(?:'|’)?m going to|i am going to|let me|i can)\s+"
        r"(?:check|look|look up|find|see|verify|inspect|search)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*checking\b", re.IGNORECASE),
)
TOOLLESS_RESPONSE_PREFIXES = tuple(
    phrase.casefold()
    for phrase in (
        # German
        "ich prüfe",
        "ich überprüfe",
        "ich schaue",
        "ich werde prüfen",
        "ich werde nachsehen",
        "lass mich",
        # French
        "je vais vérifier",
        "je vérifie",
        "je vais regarder",
        "je vais chercher",
        "laissez-moi vérifier",
        # Spanish
        "voy a comprobar",
        "voy a revisar",
        "voy a verificar",
        "déjame comprobar",
        "permíteme comprobar",
        "revisaré",
        "comprobaré",
        # Italian
        "controllo",
        "verifico",
        "controllerò",
        "lasciami controllare",
        # Dutch
        "ik controleer",
        "ik ga controleren",
        "ik kijk",
        "laat me controleren",
        # Polish
        "sprawdzę",
        "sprawdzam",
        "już sprawdzam",
        "pozwól mi sprawdzić",
        # Portuguese
        "vou verificar",
        "vou conferir",
        "deixe-me verificar",
        "verifico",
        # Russian
        "я проверю",
        "сейчас проверю",
        "проверю",
        "я посмотрю",
        "посмотрю",
        # Chinese
        "我来查",
        "我会检查",
        "让我看看",
        "正在检查",
        "我查一下",
        # Japanese
        "確認します",
        "調べます",
        "見てみます",
        "確認してみます",
        # Korean
        "확인하겠습니다",
        "확인해볼게요",
        "확인해 보겠습니다",
        # Nordic languages
        "jag kontrollerar",
        "jag ska kontrollera",
        "jag kollar",
        "låt mig kontrollera",
        "jeg sjekker",
        "jeg skal sjekke",
        "la meg sjekke",
        "jeg tjekker",
        "jeg vil tjekke",
        "lad mig tjekke",
        # Finnish, Czech, Greek, Turkish, Filipino
        "tarkistan",
        "katson",
        "zkontroluji",
        "podívám se",
        "ověřím",
        "θα ελέγξω",
        "ελέγχω",
        "ας ελέγξω",
        "θα κοιτάξω",
        "kontrol edeceğim",
        "kontrol ediyorum",
        "bakacağım",
        "kontrol edeyim",
        "susuriin ko",
        "titingnan ko",
        # Arabic, Hindi
        "سأتحقق",
        "سوف أتحقق",
        "دعني أتحقق",
        "سأفحص",
        "سأبحث",
        "मैं जांच",
        "मैं जाँच",
        "मैं देख",
        "जांचता हूँ",
        "जाँचता हूँ",
        "देखता हूँ",
    )
)
_REQUEST_USER_INPUT: ContextVar[ConversationInput | None] = ContextVar(
    "mcp_assist_request_user_input", default=None
)
_REQUEST_CONVERSATION_ID: ContextVar[str | None] = ContextVar(
    "mcp_assist_request_conversation_id", default=None
)
_ADAPTIVE_LOADED_TOOL_NAMES: ContextVar[set[str] | frozenset[str] | None] = ContextVar(
    "mcp_assist_adaptive_loaded_tool_names", default=None
)
_REQUEST_TOOL_HISTORY_SUMMARIES: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "mcp_assist_request_tool_history_summaries", default=None
)
_PERSISTENT_CHAT_LOG_RECORD: ContextVar[dict[str, Any] | None] = ContextVar(
    "mcp_assist_persistent_chat_log_record", default=None
)
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


@dataclass(frozen=True)
class ToolCallBudgetPlan:
    """Provider tool-call plan with original-order indexes preserved."""

    executable_calls: list[dict[str, Any]]
    executable_indexes: list[int]
    skipped_results_by_index: dict[int, dict[str, Any]]
    original_count: int
    exhausted: bool


@dataclass(frozen=True)
class ProviderHttpResponse:
    """Provider HTTP response data after network transport succeeds."""

    status: int
    data: dict[str, Any] | None = None
    error_text: str = ""


class ProviderResponseTimeoutError(Exception):
    """Expected provider timeout surfaced without a noisy traceback."""

    def __init__(
        self,
        *,
        provider_name: str,
        timeout_seconds: int,
        transport: str,
        attempts: int,
        iteration: int,
    ) -> None:
        self.provider_name = provider_name
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.attempts = attempts
        self.iteration = iteration
        super().__init__(
            f"{provider_name} {transport} request timed out after "
            f"{timeout_seconds} seconds"
        )


def _json_size_bytes(value: Any) -> int:
    """Return UTF-8 JSON size for logs without logging the JSON itself."""
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(value)
    return len(serialized.encode("utf-8"))


def _adaptive_loaded_tool_names() -> set[str]:
    """Return mutable request-scoped adaptive schema names."""
    loaded_names = _ADAPTIVE_LOADED_TOOL_NAMES.get()
    if isinstance(loaded_names, set):
        return loaded_names
    mutable_names = set(loaded_names or ())
    _ADAPTIVE_LOADED_TOOL_NAMES.set(mutable_names)
    return mutable_names


def _mapping_key_summary(value: Any, *, max_keys: int = 10) -> str:
    """Return a compact key-only summary for dictionaries."""
    if not isinstance(value, dict) or not value:
        return "none"
    keys = sorted(str(key) for key in value)
    visible = keys[:max_keys]
    suffix = f", +{len(keys) - max_keys} more" if len(keys) > max_keys else ""
    return ", ".join(visible) + suffix


def _redacted_log_snippet(value: Any, *, max_chars: int = 500) -> str:
    """Return a short log snippet with common secret fields redacted."""
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+[^\s,;}]+", "Bearer [redacted]", text)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*)[^\s,;}]+",
        r"\1[redacted]",
        text,
    )
    text = _SENSITIVE_LOG_FIELD_VALUE_RE.sub(r"\1[redacted]\2", text)
    text = _SENSITIVE_LOG_FIELD_RE.sub("[redacted]", text)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated {len(text) - max_chars} chars]"


def _provider_log_snippet(value: Any, max_chars: int = MAX_PROVIDER_LOG_CHARS) -> str:
    """Return a single-line, redacted provider detail safe for logs."""
    text = str(value or "")
    text = re.sub(
        r"(?i)(authorization)([\"']?\s*[:=]\s*[\"']?)[^,\"'}\r\n]+",
        r"\1\2[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)(api(?:[_\s-]?key)|access(?:[_\s-]?token)|x-api-key)"
        r"([\"']?\s*[:=]\s*[\"']?)[^,\"'}\s]+",
        r"\1\2[redacted]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{8,}\b", "[redacted]", text)
    text = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "[redacted]", text)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated {len(text) - max_chars} chars]"


def _exception_log_snippet(err: BaseException) -> str:
    """Return a useful log detail even for exceptions with an empty string value."""
    snippet = _provider_log_snippet(err)
    return snippet or err.__class__.__name__


class MCPAssistConversationEntity(ConversationEntity):
    """MCP Assist conversation entity with multi-provider support."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the MCP Assist conversation entity."""
        super().__init__()

        self.hass = hass
        self.entry = entry
        self.history = ConversationHistory()
        self._current_chat_log = None  # ChatLog for debug view tracking
        self._cached_profile_mcp_tools: list[dict[str, Any]] | None = None
        self._cached_profile_mcp_tools_key: tuple[Any, ...] | None = None
        self._cached_profile_mcp_tools_fetched_at = 0.0

        # Entity attributes
        profile_name = entry.data.get("profile_name", "MCP Assist")

        # Static configuration (doesn't change)
        runtime_config = resolve_provider_runtime_config(entry)
        self.server_type = runtime_config.server_type
        server_display_name = runtime_config.display_name

        # Set entity attributes
        self._attr_unique_id = entry.entry_id
        self._attr_name = f"{server_display_name} - {profile_name}"
        self._attr_suggested_object_id = (
            f"{self.server_type}_{profile_name.lower().replace(' ', '_')}"
        )

        # Device info
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"{server_display_name} - {profile_name}",
            manufacturer="MCP Assist",
            model=server_display_name,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        # All other config values are now dynamic properties (see @property methods below)

        # Log the actual configuration being used
        if self.debug_mode:
            _LOGGER.debug(f"🔍 Server Type: {self.server_type}")
            _LOGGER.debug(f"🔍 Base URL: {self.base_url_dynamic}")
            _LOGGER.debug("🔍 Debug mode: ON")
            _LOGGER.debug(f"🔍 Max iterations: {self.max_iterations}")

        _LOGGER.info(
            "MCP Assist Agent initialized - Server: %s, Model: %s, MCP Port: %d, URL: %s",
            self.server_type,
            self.model_name,
            self.mcp_port,
            self.base_url_dynamic,
        )

    def _get_shared_setting(self, key: str, default: Any) -> Any:
        """Get a shared setting from system entry with fallback to profile entry."""
        # Import here to avoid circular dependency
        from . import get_system_entry

        # Try to get from system entry first
        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        # Fallback to profile entry for backward compatibility
        value = self.entry.options.get(key, self.entry.data.get(key))
        if value is not None:
            return value

        # Return default
        return default

    def _get_profile_setting(self, key: str, default: Any) -> Any:
        """Get a profile-specific setting from this conversation profile."""
        value = self.entry.options.get(key, self.entry.data.get(key))
        if value is not None:
            return value
        return default

    def _mcp_request_headers(self) -> dict[str, str] | None:
        """Build auth headers for internal MCP server JSON-RPC calls."""
        token = str(
            self._get_shared_setting(CONF_MCP_BEARER_TOKEN, DEFAULT_MCP_BEARER_TOKEN)
            or ""
        ).strip()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}"}

    def _mcp_post_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build aiohttp POST kwargs for internal MCP JSON-RPC calls."""
        kwargs: dict[str, Any] = {"json": payload}
        if headers := self._mcp_request_headers():
            kwargs["headers"] = headers
        return kwargs

    def _provider_request_headers(self, provider: LLMProvider) -> dict[str, str]:
        """Build model-provider headers with request-scoped Assist identity."""
        headers = dict(provider.headers())
        conversation_id = str(_REQUEST_CONVERSATION_ID.get() or "").strip()
        if (
            conversation_id
            and "\r" not in conversation_id
            and "\n" not in conversation_id
        ):
            headers["X-Session-Id"] = conversation_id
        return headers

    def _is_optional_tool_family_enabled(self, family: str) -> bool:
        """Return whether an optional tool family is enabled for this profile."""
        if family == "unit_conversion":
            shared_enabled = self._get_shared_setting(
                CONF_ENABLE_UNIT_CONVERSION_TOOLS,
                None,
            )
            if shared_enabled is None:
                shared_enabled = self._get_shared_setting(
                    CONF_ENABLE_CALCULATOR_TOOLS,
                    DEFAULT_ENABLE_CALCULATOR_TOOLS,
                )

            profile_enabled = self._get_profile_setting(
                CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
                None,
            )
            if profile_enabled is None:
                profile_enabled = self._get_profile_setting(
                    CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
                    DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS,
                )

            return bool(shared_enabled and profile_enabled)

        shared_key, shared_default = TOOL_FAMILY_SHARED_SETTINGS[family]
        profile_key, profile_default = TOOL_FAMILY_PROFILE_SETTINGS[family]
        return bool(
            self._get_shared_setting(shared_key, shared_default)
            and self._get_profile_setting(profile_key, profile_default)
        )

    def _get_builtin_toggle_specs(self) -> tuple[BuiltInToolToggleSpec, ...]:
        """Return built-in packaged-tool metadata from the shared custom tool loader."""
        tools = self._get_shared_tools_loader()
        if tools is None:
            return ()

        getter = getattr(tools, "get_builtin_toggle_specs", None)
        if not callable(getter):
            return ()

        try:
            return tuple(getter() or ())
        except Exception as err:
            _LOGGER.debug("Unable to read built-in packaged tool specs: %s", err)
            return ()

    def _get_builtin_toggle_spec(
        self,
        tool_name: str,
    ) -> BuiltInToolToggleSpec | None:
        """Return built-in packaged-tool metadata for a tool name, if any."""
        tools = self._get_shared_tools_loader()
        if tools is not None:
            getter = getattr(tools, "get_builtin_toggle_spec", None)
            if callable(getter):
                try:
                    return getter(tool_name)
                except Exception as err:
                    _LOGGER.debug(
                        "Unable to read built-in packaged tool metadata for %s: %s",
                        tool_name,
                        err,
                    )

        return None

    def _is_builtin_package_enabled(
        self,
        spec: BuiltInToolToggleSpec,
    ) -> bool:
        """Return whether a built-in packaged tool is enabled for this profile."""
        return is_builtin_package_enabled_for_profile(
            spec,
            self._get_shared_setting,
            self._get_profile_setting,
            search_provider=self.search_provider,
        )

    def _is_tool_enabled_for_profile(self, tool_name: str) -> bool:
        """Return whether a tool should be visible to this profile."""
        built_in_spec = self._get_builtin_toggle_spec(tool_name)
        if built_in_spec is not None:
            return self._is_builtin_package_enabled(built_in_spec)

        family = get_optional_tool_family(tool_name)
        if family is not None:
            return self._is_optional_tool_family_enabled(family)
        if self._is_external_custom_tool(tool_name):
            return self.external_custom_tools_enabled
        return True

    # Dynamic configuration properties - read from entry.options/data each time
    @property
    def base_url_dynamic(self) -> str:
        """Get base URL (dynamic for local servers)."""
        return resolve_provider_runtime_config(self.entry).base_url

    @property
    def model_name(self) -> str:
        """Get model name (dynamic)."""
        return self.entry.options.get(
            CONF_MODEL_NAME, self.entry.data.get(CONF_MODEL_NAME, "")
        )

    @property
    def mcp_port(self) -> int:
        """Get MCP port (shared setting)."""
        return self._get_shared_setting(CONF_MCP_PORT, DEFAULT_MCP_PORT)

    @property
    def debug_mode(self) -> bool:
        """Get debug mode (dynamic)."""
        return self.entry.options.get(
            CONF_DEBUG_MODE, self.entry.data.get(CONF_DEBUG_MODE, DEFAULT_DEBUG_MODE)
        )

    @property
    def chat_log_mode(self) -> bool:
        """Get persistent chat log mode (dynamic)."""
        return self.entry.options.get(
            CONF_CHAT_LOG_MODE,
            self.entry.data.get(CONF_CHAT_LOG_MODE, DEFAULT_CHAT_LOG_MODE),
        )

    @property
    def clean_responses(self) -> bool:
        """Get clean responses setting (dynamic)."""
        return self.entry.options.get(
            CONF_CLEAN_RESPONSES,
            self.entry.data.get(CONF_CLEAN_RESPONSES, DEFAULT_CLEAN_RESPONSES),
        )

    @property
    def max_iterations(self) -> int:
        """Get max iterations (dynamic)."""
        return self.entry.options.get(
            CONF_MAX_ITERATIONS,
            self.entry.data.get(CONF_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS),
        )

    @property
    def max_history(self) -> int:
        """Get max history messages/turns (dynamic)."""
        return self.entry.options.get(
            CONF_MAX_HISTORY, self.entry.data.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)
        )

    @property
    def context_mode(self) -> str:
        """Get model context mode (dynamic)."""
        value = self.entry.options.get(
            CONF_CONTEXT_MODE,
            self.entry.data.get(CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE),
        )
        if value in {CONTEXT_MODE_STANDARD, CONTEXT_MODE_ADAPTIVE, CONTEXT_MODE_LIGHT}:
            return value
        return DEFAULT_CONTEXT_MODE

    @property
    def adaptive_context_mode(self) -> bool:
        """Return whether this profile should load optional tool schemas on demand."""
        return self.context_mode == CONTEXT_MODE_ADAPTIVE

    @property
    def light_context_mode(self) -> bool:
        """Return whether this profile should send reduced model context."""
        return self.context_mode == CONTEXT_MODE_LIGHT

    @property
    def max_tokens(self) -> int:
        """Get max tokens (dynamic)."""
        return self.entry.options.get(
            CONF_MAX_TOKENS, self.entry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        )

    @property
    def temperature(self) -> float:
        """Get temperature (dynamic)."""
        return self.entry.options.get(
            CONF_TEMPERATURE, self.entry.data.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        )

    @property
    def follow_up_mode(self) -> str:
        """Get response mode (dynamic, with backward compatibility)."""
        return self.entry.options.get(
            CONF_RESPONSE_MODE,
            self.entry.data.get(
                CONF_RESPONSE_MODE,
                self.entry.options.get(
                    CONF_FOLLOW_UP_MODE,
                    self.entry.data.get(CONF_FOLLOW_UP_MODE, DEFAULT_RESPONSE_MODE),
                ),
            ),
        )

    def _build_provider_settings(self) -> ProviderSettings:
        """Build current provider settings from dynamic profile options."""
        return build_provider_settings(
            self.entry,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            prompt_cache_key=self._build_prompt_cache_key(),
        )

    def _get_llm_provider(self) -> LLMProvider:
        """Return the active provider transport."""
        return create_llm_provider(self._build_provider_settings())

    @property
    def search_provider(self) -> str:
        """Get search provider (shared setting) with backward compatibility."""
        provider = self._get_shared_setting(CONF_SEARCH_PROVIDER, None)

        if provider:
            return provider

        # Backward compat: if old enable_custom_tools was True, default to "brave"
        if self._get_shared_setting(CONF_ENABLE_CUSTOM_TOOLS, False):
            return "brave"

        return "none"

    @property
    def music_assistant_support_enabled(self) -> bool:
        """Get effective Music Assistant support setting for this profile."""
        return self._is_optional_tool_family_enabled("music_assistant")

    @property
    def assist_bridge_enabled(self) -> bool:
        """Get effective Assist bridge setting for this profile."""
        return self._is_optional_tool_family_enabled("assist_bridge")

    @property
    def llm_api_bridge_enabled(self) -> bool:
        """Get effective third-party LLM API bridge setting for this profile."""
        return self._is_optional_tool_family_enabled(TOOL_FAMILY_LLM_API_BRIDGE)

    @property
    def external_custom_tools_enabled(self) -> bool:
        """Get effective external custom tool setting for this profile."""
        return self._is_optional_tool_family_enabled(TOOL_FAMILY_EXTERNAL_CUSTOM)

    @property
    def memory_tools_enabled(self) -> bool:
        """Get effective persisted memory tool setting for this profile."""
        return self._is_optional_tool_family_enabled("memory")

    @property
    def device_tools_enabled(self) -> bool:
        """Get effective device tool setting for this profile."""
        return self._is_optional_tool_family_enabled("device")

    @property
    def web_search_tools_enabled(self) -> bool:
        """Get effective web-search tool setting for this profile."""
        specs = self._get_builtin_toggle_specs()
        if specs:
            return any(
                self._is_builtin_package_enabled(spec)
                for spec in specs
                if spec.package_id in {"search", "read_url"}
            )
        return self._is_optional_tool_family_enabled("web_search")

    def _get_shared_tools_loader(self) -> Any | None:
        """Return the shared custom tool loader, if available."""
        server = self.hass.data.get(DOMAIN, {}).get("shared_mcp_server")
        return getattr(server, "tools", None) if server else None

    def _is_external_custom_tool(self, tool_name: str) -> bool:
        """Return whether a tool name comes from an external custom tool package."""
        tools = self._get_shared_tools_loader()
        if tools is None:
            return False

        checker = getattr(tools, "is_external_custom_tool", None)
        if not callable(checker):
            return False

        try:
            return bool(checker(tool_name))
        except Exception as err:
            _LOGGER.debug(
                "Unable to classify external custom tool %s: %s", tool_name, err
            )
            return False

    def _get_builtin_tool_instructions(self) -> str:
        """Return prompt additions from loaded built-in packaged tools."""
        tools = self._get_shared_tools_loader()
        if tools is None:
            return ""

        getter = getattr(tools, "get_builtin_prompt_instructions", None)
        if not callable(getter):
            return ""

        try:
            return str(getter() or "").strip()
        except Exception as err:
            _LOGGER.debug(
                "Unable to read built-in packaged tool prompt instructions: %s",
                err,
            )
            return ""

    def _build_disabled_tool_family_instructions(self) -> str:
        """Build prompt instructions for disabled optional tool families."""
        lines: list[str] = []

        if not self.device_tools_enabled:
            lines.append(
                "- Device tools are disabled. Do not call discover_devices or get_device_details. Use discover_entities and get_entity_details instead."
            )

        if not self.assist_bridge_enabled:
            lines.append(
                "- Native Assist bridge tools are disabled. Do not call list_assist_tools, call_assist_tool, get_assist_prompt, or get_assist_context_snapshot."
            )

        if not self.llm_api_bridge_enabled:
            lines.append(
                "- Third-party LLM API bridge tools are disabled. Do not call list_llm_apis, list_llm_api_tools, call_llm_api_tool, or get_llm_api_prompt."
            )

        if not self.memory_tools_enabled:
            lines.append(
                "- Memory tools are disabled. Do not call list_memory_categories, remember_memory, recall_memories, or forget_memory."
            )

        if not self.external_custom_tools_enabled:
            lines.append(
                "- External custom tools are disabled. Do not call tools provided by user-defined packages."
            )

        for spec in self._get_builtin_toggle_specs():
            if self._is_builtin_package_enabled(spec):
                continue
            lines.append(
                f"- {spec.package_name} is disabled for this profile. Do not call {', '.join(spec.tool_names)}."
            )

        if not lines:
            return ""

        return "## Disabled Optional Tool Families\n" + "\n".join(lines)

    def _build_optional_technical_instructions(self, current_area: str) -> str:
        """Build optional prompt sections for enabled capability families."""
        sections: list[str] = []

        if self.device_tools_enabled:
            sections.append(DEVICE_TECHNICAL_INSTRUCTIONS.strip())

        if self.memory_tools_enabled:
            sections.append(MEMORY_TECHNICAL_INSTRUCTIONS.strip())

        if self.assist_bridge_enabled:
            sections.append(ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS.strip())

        if self.llm_api_bridge_enabled:
            sections.append(LLM_API_BRIDGE_TECHNICAL_INSTRUCTIONS.strip())

        if self.music_assistant_support_enabled:
            sections.append(
                self._render_prompt_template(
                    MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS,
                    {"current_area": current_area},
                ).strip()
            )

        built_in_tool_instructions = self._get_builtin_tool_instructions()
        if built_in_tool_instructions:
            sections.append(built_in_tool_instructions)

        if self.external_custom_tools_enabled:
            external_custom_tool_instructions = (
                self._get_external_custom_tool_instructions()
            )
            if external_custom_tool_instructions:
                sections.append(external_custom_tool_instructions)

        return "\n\n".join(section for section in sections if section)

    def _build_adaptive_technical_instructions(self) -> str:
        """Build compact routing guidance for adaptive context mode."""
        return (
            "## Adaptive Tool Loading\n"
            "- Start with the advertised Home Assistant tools for entity discovery and control.\n"
            "- When a request needs optional, built-in package, or custom tools, "
            f"call {ADAPTIVE_TOOL_CATALOG_NAME} with a short query, then call "
            f"{ADAPTIVE_TOOL_SCHEMA_NAME} for the exact tool names you need.\n"
            "- Do not ask the user to approve tool discovery; use these routing "
            "tools in the same turn when needed."
        )

    def _get_external_custom_tool_instructions(self) -> str:
        """Return prompt additions from loaded external custom tool packages."""
        if not self.external_custom_tools_enabled:
            return ""

        tools = self._get_shared_tools_loader()
        if tools is None:
            return ""

        try:
            return str(tools.get_external_prompt_instructions() or "").strip()
        except Exception as err:
            _LOGGER.debug(
                "Unable to read external custom tool prompt instructions: %s", err
            )
            return ""

    @staticmethod
    def _compact_text(text: str, *, max_len: int = 160) -> str:
        """Compact instructional text for lower token usage."""
        return compact_text(text, max_len=max_len)

    def _compact_schema_for_llm(self, schema: Any, *, keep_description: bool = False) -> Any:
        """Strip nonessential JSON-schema verbosity before sending tools to the LLM."""
        return compact_schema_for_llm(schema, keep_description=keep_description)

    def _convert_mcp_tools_to_llm_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tools to a compact provider-neutral function schema."""
        return convert_mcp_tools_to_llm_tools(tools)

    def _build_tool_routing_summary(self, routing_hints: Any) -> str:
        """Build a compact description suffix from optional routing hints."""
        return build_tool_routing_summary(routing_hints)

    @staticmethod
    def _json_size_bytes(value: Any) -> int:
        """Return UTF-8 JSON size for debug metrics without exposing contents."""
        return json_size_bytes(value)

    @classmethod
    def _content_char_count(cls, value: Any) -> int:
        """Return an approximate text size for message content."""
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        if isinstance(value, dict):
            if "text" in value:
                return cls._content_char_count(value.get("text"))
            if "content" in value:
                return cls._content_char_count(value.get("content"))
            return sum(cls._content_char_count(item) for item in value.values())
        if isinstance(value, list):
            return sum(cls._content_char_count(item) for item in value)
        return len(str(value))

    @classmethod
    def _message_content_char_count(cls, messages: list[dict[str, Any]]) -> int:
        """Return total content characters across provider messages."""
        return sum(
            cls._content_char_count(message.get("content")) for message in messages
        )

    def _log_initial_llm_payload_metrics(
        self,
        *,
        transport: str,
        iteration: int,
        payload: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """Log compact first-payload metrics for prompt latency debugging."""
        if iteration != 0:
            return
        if not (self.debug_mode or _LOGGER.isEnabledFor(logging.DEBUG)):
            return

        tool_count = len(tools or [])
        log = _LOGGER.info if self.debug_mode else _LOGGER.debug
        log(
            (
                "Initial LLM payload metrics: provider=%s transport=%s "
                "payload_bytes=%d messages=%d message_chars=%d tools=%d "
                "tool_schema_bytes=%d"
            ),
            self.server_type,
            transport,
            self._json_size_bytes(payload),
            len(messages),
            self._message_content_char_count(messages),
            tool_count,
            self._json_size_bytes(tools or []),
        )

    def _build_prompt_cache_key(self) -> str:
        """Build a stable, non-identifying prompt-cache key for this profile."""
        source = {
            "entry_id": self.entry.entry_id,
            "unique_id": self.entry.unique_id,
            "server_type": self.server_type,
            "model_name": self.model_name,
            "context_mode": self.context_mode,
        }
        digest = hashlib.sha256(
            json.dumps(source, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        return f"ha-mcp-assist-{digest}"

    def _prepare_provider_payload(
        self,
        provider: LLMProvider,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare a provider payload without exposing provider-specific details."""
        return provider.prepare_payload(payload)

    def _log_prompt_cache_usage(
        self,
        provider: LLMProvider,
        data: dict[str, Any],
        *,
        transport: str,
        iteration: int,
    ) -> None:
        """Log provider prompt-cache usage without logging prompt contents."""
        usage = provider.extract_prompt_cache_usage(data)
        if usage is None:
            return

        cached_tokens = usage.cached_tokens
        cache_read_tokens = usage.cache_read_tokens
        cache_creation_tokens = usage.cache_creation_tokens
        has_cache_activity = any(
            value
            for value in (
                cached_tokens,
                cache_read_tokens,
                cache_creation_tokens,
            )
        )
        if not (
            has_cache_activity
            or self.debug_mode
            or _LOGGER.isEnabledFor(logging.DEBUG)
        ):
            return

        input_tokens = usage.input_tokens
        cache_hit_pct = None
        if input_tokens and cached_tokens is not None:
            cache_hit_pct = round((cached_tokens / input_tokens) * 100, 1)

        log = _LOGGER.info if self.debug_mode or has_cache_activity else _LOGGER.debug
        log(
            (
                "Prompt cache usage: provider=%s transport=%s iteration=%d "
                "input_tokens=%s cached_tokens=%s cache_read_tokens=%s "
                "cache_creation_tokens=%s cache_hit_pct=%s"
            ),
            provider.server_type,
            transport,
            iteration,
            input_tokens,
            cached_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            cache_hit_pct,
        )

    def _build_mcp_tool_cache_key(self) -> tuple[Any, ...]:
        """Build a cache key for the current profile-visible MCP tool surface."""
        return (
            self.mcp_port,
            self.search_provider,
            self.assist_bridge_enabled,
            self.llm_api_bridge_enabled,
            self.memory_tools_enabled,
            self.external_custom_tools_enabled,
            self.device_tools_enabled,
            self.music_assistant_support_enabled,
            self.web_search_tools_enabled,
            self.context_mode,
            tuple(
                (
                    spec.package_id,
                    self._is_builtin_package_enabled(spec),
                )
                for spec in self._get_builtin_toggle_specs()
            ),
            self._get_external_custom_tool_cache_signature(),
        )

    def _get_external_custom_tool_cache_signature(self) -> tuple[Any, ...]:
        """Return a cache signature for loaded built-in/external packaged tools."""
        server = self.hass.data.get(DOMAIN, {}).get("shared_mcp_server")
        tools = getattr(server, "tools", None) if server else None
        if tools is None:
            return ()

        get_cache_signature = getattr(tools, "get_cache_signature", None)
        if callable(get_cache_signature):
            try:
                raw_signature = get_cache_signature()
                if isinstance(raw_signature, tuple):
                    return raw_signature
                return (raw_signature,)
            except Exception as err:
                _LOGGER.debug(
                    "Unable to read external custom tool cache signature: %s", err
                )

        get_builtin_prompt_instructions = getattr(
            tools, "get_builtin_prompt_instructions", None
        )
        if callable(get_builtin_prompt_instructions):
            try:
                return (str(get_builtin_prompt_instructions() or "").strip(),)
            except Exception as err:
                _LOGGER.debug(
                    "Unable to read built-in packaged tool prompt instructions: %s",
                    err,
                )

        get_external_prompt_instructions = getattr(
            tools, "get_external_prompt_instructions", None
        )
        if callable(get_external_prompt_instructions):
            try:
                return (str(get_external_prompt_instructions() or "").strip(),)
            except Exception as err:
                _LOGGER.debug(
                    "Unable to read external custom tool prompt instructions: %s", err
                )

        return ()

    def _compact_tool_result_for_llm(
        self,
        tool_name: str,
        content: Any,
        *,
        max_chars: int = MAX_TOOL_RESULT_CHARS,
        max_lines: int = MAX_TOOL_RESULT_LINES,
    ) -> str:
        """Keep tool results useful while avoiding oversized follow-up payloads."""
        text = "" if content is None else str(content)
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return ""

        original_length = len(text)
        original_lines = text.count("\n") + 1

        if original_length <= max_chars and original_lines <= max_lines:
            return text

        lines = text.splitlines()
        truncated_lines = lines[:max_lines]
        compacted = "\n".join(truncated_lines).strip()

        if len(compacted) > max_chars:
            compacted = compacted[:max_chars].rstrip()
            last_break = max(compacted.rfind("\n"), compacted.rfind(" "))
            if last_break > int(max_chars * 0.7):
                compacted = compacted[:last_break].rstrip()

        omitted_lines = max(0, original_lines - len(truncated_lines))
        omitted_chars = max(0, original_length - len(compacted))
        hint = (
            "Use narrower filters, paging, or a more specific follow-up tool call if you need the omitted detail."
        )
        if tool_name in {"discover_entities", "discover_devices"}:
            hint = (
                "Use limit/offset paging or narrower filters if you need more of the result set."
            )
        elif tool_name in {"get_entity_details", "get_device_details", "get_index"}:
            hint = (
                "Call again with a narrower target if you need more of this structured detail."
            )

        summary_parts: list[str] = []
        if omitted_lines:
            summary_parts.append(f"{omitted_lines} more lines")
        if omitted_chars:
            summary_parts.append(f"{omitted_chars} more chars")
        summary = ", ".join(summary_parts) or "additional content omitted"

        return (
            f"{compacted}\n\n"
            f"[Tool result truncated for model context: {summary}. {hint}]"
        )

    @property
    def attribution(self) -> str:
        """Return attribution."""
        return f"Powered by {self._get_server_display_name()} with MCP entity discovery"

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"  # Support all languages

    @property
    def supported_features(self) -> int:
        """Return supported features."""
        features = ConversationEntityFeature(0)

        # Check if home control is enabled in config
        control_enabled = self.entry.options.get(
            CONF_CONTROL_HA, self.entry.data.get(CONF_CONTROL_HA, DEFAULT_CONTROL_HA)
        )

        if control_enabled:
            features |= ConversationEntityFeature.CONTROL

        return features

    @property
    def follow_up_phrases(self) -> str:
        """Return follow-up phrases for pattern detection."""
        return self.entry.options.get(
            CONF_FOLLOW_UP_PHRASES,
            self.entry.data.get(CONF_FOLLOW_UP_PHRASES, DEFAULT_FOLLOW_UP_PHRASES),
        )

    @property
    def end_words(self) -> str:
        """Return end conversation words for user ending detection."""
        return self.entry.options.get(
            CONF_END_WORDS, self.entry.data.get(CONF_END_WORDS, DEFAULT_END_WORDS)
        )

    @property
    def profile_name(self) -> str:
        """Return profile name."""
        return self.entry.data.get(CONF_PROFILE_NAME, "MCP Assist")

    @property
    def timeout(self) -> int:
        """Get request timeout in seconds (dynamic)."""
        return self.entry.options.get(
            CONF_TIMEOUT, self.entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        )

    @property
    def session_key(self) -> str:
        """Get OpenClaw session key (dynamic)."""
        return self.entry.options.get(
            CONF_OPENCLAW_SESSION_KEY,
            self.entry.data.get(CONF_OPENCLAW_SESSION_KEY, DEFAULT_OPENCLAW_SESSION_KEY),
        )

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

        # Store entity reference for index manager to access
        if self.entry.entry_id in self.hass.data[DOMAIN]:
            self.hass.data[DOMAIN][self.entry.entry_id]["agent"] = self

        _LOGGER.info("Conversation entity registered: %s", self._attr_name)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)

        # Remove entity reference
        if self.entry.entry_id in self.hass.data.get(DOMAIN, {}):
            self.hass.data[DOMAIN][self.entry.entry_id].pop("agent", None)

        await super().async_will_remove_from_hass()
        _LOGGER.info("Conversation entity unregistered: %s", self._attr_name)

    def _get_server_display_name(self) -> str:
        """Get friendly display name for the server type."""
        return self._get_llm_provider().display_name

    def _get_friendly_error_message(self, error: Exception) -> str:
        """Convert technical errors to user-friendly TTS messages."""
        error_str = str(error).lower()
        error_full = str(error)  # Keep original case for extracting details
        provider = self._get_llm_provider()

        # Category A: Connection/Network Errors
        if any(
            x in error_str
            for x in [
                "connection",
                "refused",
                "cannot connect",
                "no route",
                "unreachable",
            ]
        ):
            if provider.is_remote_service:
                return f"I couldn't reach {self._get_server_display_name()}'s API servers. Please check your internet connection and try again."
            return f"I couldn't connect to {self._get_server_display_name()} at {self.base_url_dynamic}. Please check that the server is running and the address is correct in your integration settings."

        if "timeout" in error_str or "timed out" in error_str:
            return f"The {self._get_server_display_name()} server took too long to respond. This might be because the model is slow or busy. Try again or consider using a faster model."

        # Category B: Authentication
        if any(
            x in error_str
            for x in [
                "401",
                "403",
                "unauthorized",
                "invalid_api_key",
                "invalid api key",
            ]
        ):
            return f"Your {self._get_server_display_name()} API key is invalid or missing. Please check your API key in the integration settings."

        if "insufficient_quota" in error_str or "permission denied" in error_str:
            return f"Your {self._get_server_display_name()} account doesn't have permission for this operation. Check your account status and billing."

        # Category C: Resource Limits
        if (
            "rate limit" in error_str
            or "429" in error_str
            or "too many requests" in error_str
        ):
            return f"You've hit {self._get_server_display_name()}'s rate limit. Wait a minute and try again, or upgrade your plan for higher limits."

        if "quota exceeded" in error_str or "insufficient credits" in error_str:
            return f"Your {self._get_server_display_name()} account has run out of credits or quota. Check your billing and add credits to continue."

        if (
            "maximum context length" in error_str
            or "context_length_exceeded" in error_str
            or "too many tokens" in error_str
            or ("tokens" in error_str and "exceed" in error_str)
            or "context window" in error_str
        ):
            # Try to extract token limit if present
            token_match = re.search(r"(\d+)\s*tokens?", error_str)
            return provider.context_window_error_message(
                token_count=token_match.group(1) if token_match else None,
                light_context_mode=self.light_context_mode,
            )

        # Category D: Model Errors
        provider_model_message = provider.model_unavailable_message(error_str)
        if provider_model_message is not None:
            return provider_model_message

        if "404" in error_str or ("model" in error_str and "not found" in error_str):
            return f"The model '{self.model_name}' wasn't found on {self._get_server_display_name()}. Check that the model name is correct in your integration settings."

        # Category E: OpenClaw Gateway Errors
        if "not_paired" in error_str or "device pairing" in error_str or "not paired" in error_str:
            return "This device needs to be approved on your OpenClaw server. Run 'openclaw devices approve' or use the OpenClaw Control UI."

        if "openclaw" in error_str and "timeout" in error_str:
            return "The OpenClaw gateway took too long to respond. The agent may be busy processing a complex request. Try again or increase the timeout in settings."

        if "websocket" in error_str or "connection closed" in error_str:
            return "Lost connection to the OpenClaw gateway. Check that the gateway is running and accessible."

        # Category F: MCP Errors
        if (
            f"localhost:{self.mcp_port}" in error_str
            or f"127.0.0.1:{self.mcp_port}" in error_str
        ):
            return f"I couldn't connect to the MCP server on port {self.mcp_port}. The integration may not have initialized correctly. Try restarting Home Assistant."

        # Category F: Response Errors
        if "empty response" in error_str or "no response" in error_str:
            return f"The {self._get_server_display_name()} server returned an empty response. This sometimes happens with certain models. Try rephrasing your request."

        if "json" in error_str and (
            "parse" in error_str or "decode" in error_str or "malformed" in error_str
        ):
            return f"I received a malformed response from {self._get_server_display_name()}. This might be a temporary server issue. Please try again."

        # Category G: Generic fallback
        # Extract first meaningful part of error (up to 100 chars, stop at newline)
        error_snippet = error_full.split("\n")[0][:100]
        return f"An unexpected error occurred while talking to {self._get_server_display_name()}. The error was: {error_snippet}. Check the Home Assistant logs for more details."

    def _record_tool_calls_to_chatlog(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Record tool calls to ChatLog for debug view."""
        if not self._current_chat_log:
            return

        try:
            # Convert tool calls to llm.ToolInput format
            llm_tool_calls = []
            for tc in tool_calls:
                tool_input = llm.ToolInput(
                    id=tc.get("id", str(uuid.uuid4())),
                    tool_name=tc.get("function", {}).get("name", "unknown"),
                    tool_args=self._parse_tool_arguments(
                        tc.get("function", {}).get("arguments")
                    ),
                    external=True,  # MCP tools are executed externally, not by ChatLog
                )
                llm_tool_calls.append(tool_input)

            # Add assistant content with tool calls
            assistant_content = chat_log.AssistantContent(
                agent_id=self.entity_id, tool_calls=llm_tool_calls
            )
            self._current_chat_log.async_add_assistant_content_without_tools(
                assistant_content
            )

            if self.debug_mode:
                _LOGGER.debug(f"📊 Recorded {len(tool_calls)} tool calls to ChatLog")
        except Exception as e:
            _LOGGER.error(f"Error recording tool calls to ChatLog: {e}")

    def _stringify_tool_arguments(self, arguments: Any) -> str:
        """Normalize tool arguments to a JSON string."""
        return stringify_tool_arguments(arguments)

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        """Parse tool arguments whether they arrive as a dict or JSON string."""
        return parse_tool_arguments(arguments)

    def _tool_call_log_summary(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Summarize tool calls for logs without exposing argument values."""
        summaries: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                function = {}

            raw_arguments = function.get("arguments")
            parsed_arguments = self._parse_tool_arguments(raw_arguments)
            summaries.append(
                {
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type"),
                    "name": function.get("name"),
                    "argument_keys": _mapping_key_summary(parsed_arguments),
                    "argument_bytes": _json_size_bytes(raw_arguments),
                }
            )
        return summaries

    def _normalize_tool_call_arguments(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize tool_call function.arguments for internal and provider use."""
        return normalize_tool_call_arguments(tool_calls)

    @staticmethod
    def _normalize_stream_tool_call_index(
        raw_index: Any,
        stream_index_offset: int | None,
    ) -> tuple[int, int]:
        """Normalize streamed tool-call indexes that start above zero."""
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = 0

        if stream_index_offset is None:
            stream_index_offset = index

        return max(0, index - stream_index_offset), stream_index_offset

    @classmethod
    def _ensure_stream_tool_call_slot(
        cls,
        tc: Dict[str, Any],
        current_tool_calls: List[Dict[str, Any]],
        stream_index_offset: int | None,
    ) -> tuple[int, int | None]:
        """Resolve a streamed tool call's slot, growing the buffer to fit.

        Providers that supply an ``index`` (OpenAI-style deltas) keep the
        offset-normalized index, and any gap left by a sparse index is
        backfilled so indexing never raises.

        Some providers omit ``index`` entirely, in two different shapes:
        Ollama sends each *complete* tool call as its own fragment (carrying a
        name), while some OpenAI-compatible servers stream a *single* call as
        argument fragments where only the first carries the id/name. So an
        index-less fragment starts a new slot only when it carries an ``id`` or
        a function ``name``; a name-less fragment continues the current call,
        instead of either collapsing distinct calls onto one slot or splitting
        one call's arguments across several slots.
        """
        raw_index = tc.get("index")
        if raw_index is not None:
            idx, stream_index_offset = cls._normalize_stream_tool_call_index(
                raw_index, stream_index_offset
            )
        else:
            function = tc.get("function") or {}
            starts_new_call = bool(tc.get("id") or function.get("name"))
            if starts_new_call or not current_tool_calls:
                idx = len(current_tool_calls)
            else:
                idx = len(current_tool_calls) - 1

        while idx >= len(current_tool_calls):
            current_tool_calls.append({})

        return idx, stream_index_offset

    @staticmethod
    def _compact_streamed_tool_calls(
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Drop empty streamed tool-call placeholders before execution."""
        return [tool_call for tool_call in tool_calls if tool_call]

    @staticmethod
    def _tool_call_arguments_are_valid(arguments: Any) -> bool:
        """Return whether tool arguments are a complete JSON object."""
        if arguments is None:
            return True
        if isinstance(arguments, dict):
            return True
        if isinstance(arguments, str):
            if not arguments.strip():
                return True
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return False
            return isinstance(parsed, dict)
        return False

    @classmethod
    def _tool_call_is_valid_for_execution(cls, tool_call: Dict[str, Any]) -> bool:
        """Return whether a provider tool call can safely be executed."""
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return False
        name = str(function.get("name") or "").strip()
        if not name:
            return False
        return cls._tool_call_arguments_are_valid(function.get("arguments"))

    @classmethod
    def _partition_valid_tool_calls(
        cls,
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split provider tool calls into executable and malformed calls."""
        valid_tool_calls: List[Dict[str, Any]] = []
        invalid_tool_calls: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            if cls._tool_call_is_valid_for_execution(tool_call):
                valid_tool_calls.append(tool_call)
            else:
                invalid_tool_calls.append(tool_call)

        return valid_tool_calls, invalid_tool_calls

    @classmethod
    def _invalid_tool_call_names(cls, tool_calls: List[Dict[str, Any]]) -> str:
        """Return a compact list of malformed tool-call names for logs/prompts."""
        names = []
        for tool_call in tool_calls:
            name = cls._tool_call_name(tool_call)
            if name not in names:
                names.append(name)
        return ", ".join(names)

    @classmethod
    def _append_invalid_tool_call_retry_messages(
        cls,
        conversation_messages: List[Dict[str, Any]],
        response_text: str,
        invalid_tool_calls: List[Dict[str, Any]],
    ) -> None:
        """Append corrective context after malformed tool-call arguments."""
        if response_text.strip():
            conversation_messages.append(
                {"role": "assistant", "content": response_text.strip()}
            )

        tool_names = cls._invalid_tool_call_names(invalid_tool_calls)
        instruction = INVALID_TOOL_ARGUMENT_RETRY_INSTRUCTION
        if tool_names:
            instruction = f"{instruction} Affected tool(s): {tool_names}."
        conversation_messages.append({"role": "system", "content": instruction})

    def _record_tool_result_to_chatlog(
        self, tool_call_id: str, tool_name: str, tool_result: Dict[str, Any]
    ) -> None:
        """Record a single tool result to ChatLog for debug view."""
        if not self._current_chat_log:
            return

        try:
            result_content = chat_log.ToolResultContent(
                agent_id=self.entity_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_result=tool_result,
            )
            # Use callback method to add tool result
            self._current_chat_log.async_add_assistant_content_without_tools(
                result_content
            )

            if self.debug_mode:
                _LOGGER.debug(f"📊 Recorded tool result for {tool_name} to ChatLog")
        except Exception as e:
            _LOGGER.error(f"Error recording tool result to ChatLog: {e}")

    def _build_persistent_chat_log_record(
        self, user_input: ConversationInput, conversation_id: str
    ) -> dict[str, Any]:
        """Build the initial persisted chat log record for an opt-in conversation."""
        now = dt_util.utcnow()
        return {
            "id": uuid.uuid4().hex[:12],
            "created_at": now.isoformat(),
            "_started_monotonic": time.monotonic(),
            "profile_entry_id": self.entry.entry_id,
            "profile_name": self.entry.data.get(CONF_PROFILE_NAME, "Default"),
            "conversation_id": conversation_id,
            "server_type": self.server_type,
            "model": self.model_name,
            "language": user_input.language,
            "user_text": user_input.text,
            "tools": [],
        }

    async def _finish_persistent_chat_log_record(
        self,
        *,
        assistant_text: str | None = None,
        error: str | None = None,
        continue_conversation: bool | None = None,
    ) -> None:
        """Persist the current chat log record, if one is active."""
        record = _PERSISTENT_CHAT_LOG_RECORD.get()
        if not record or record.get("_saved"):
            return

        record["_saved"] = True
        record["completed_at"] = dt_util.utcnow().isoformat()
        started_monotonic = record.get("_started_monotonic")
        if isinstance(started_monotonic, (int, float)):
            record["duration_ms"] = int(
                max(0.0, time.monotonic() - started_monotonic) * 1000
            )
        if assistant_text is not None:
            record["assistant_text"] = assistant_text
        if error is not None:
            record["error"] = error
        if continue_conversation is not None:
            record["continue_conversation"] = continue_conversation

        manager = self.hass.data.get(DOMAIN, {}).get("chat_log_manager")
        if manager is None:
            _LOGGER.debug("Chat Log Mode enabled, but chat log manager is unavailable")
            return

        try:
            await manager.async_record(record)
        except Exception as err:
            _LOGGER.error("Error persisting MCP Assist chat log: %s", err)

    def _start_persistent_tool_log(
        self,
        *,
        tool_call_id: str,
        tool_name: str | None,
        arguments: dict[str, Any],
        raw_arguments: Any = None,
    ) -> dict[str, Any] | None:
        """Add a tool-call entry to the current persisted chat log."""
        record = _PERSISTENT_CHAT_LOG_RECORD.get()
        if not record:
            return None

        tool_entry: dict[str, Any] = {
            "id": tool_call_id,
            "name": tool_name or "unknown",
            "started_at": dt_util.utcnow().isoformat(),
            "arguments": arguments,
        }
        if (
            isinstance(raw_arguments, str)
            and raw_arguments.strip()
            and not arguments
            and raw_arguments.strip() != "{}"
        ):
            tool_entry["raw_arguments"] = raw_arguments

        record.setdefault("tools", []).append(tool_entry)
        return tool_entry

    def _finish_persistent_tool_log(
        self,
        tool_entry: dict[str, Any] | None,
        *,
        result: dict[str, Any] | None = None,
        llm_content: str | None = None,
        error: str | None = None,
    ) -> None:
        """Complete a tool-call entry in the current persisted chat log."""
        if not tool_entry:
            return

        tool_entry["completed_at"] = dt_util.utcnow().isoformat()
        if result is not None:
            tool_entry["result"] = result
        if llm_content is not None:
            tool_entry["llm_content"] = llm_content
        if error is not None:
            tool_entry["error"] = error

    @staticmethod
    def _compact_history_argument_value(value: Any) -> str:
        """Return a tiny argument value for follow-up context."""
        if isinstance(value, list):
            items = [
                str(item)
                for item in value[:4]
                if isinstance(item, (str, int, float, bool))
            ]
            suffix = ", ..." if len(value) > len(items) else ""
            return compact_text(", ".join(items) + suffix, max_len=90)
        if isinstance(value, (str, int, float, bool)):
            return compact_text(str(value), max_len=90)
        return ""

    def _record_tool_history_summary(
        self,
        tool_name: str | None,
        arguments: dict[str, Any],
        result: dict[str, Any] | None = None,
        *,
        error: str | None = None,
    ) -> None:
        """Record compact real-tool context for future follow-up turns."""
        if not tool_name or tool_name in ADAPTIVE_META_TOOL_NAMES:
            return

        summaries = _REQUEST_TOOL_HISTORY_SUMMARIES.get()
        if summaries is None:
            return

        compact_arguments: dict[str, str] = {}
        for key in TOOL_HISTORY_ARGUMENT_KEYS:
            if key not in arguments:
                continue
            value = self._compact_history_argument_value(arguments[key])
            if value:
                compact_arguments[key] = value

        summary: dict[str, Any] = {
            "type": "mcp_tool",
            "tool": tool_name,
            "arguments": compact_arguments,
        }
        if error:
            summary["status"] = "error"
        elif isinstance(result, dict) and (result.get("isError") or "error" in result):
            summary["status"] = "error"
        else:
            summary["status"] = "ok"

        summaries.append(summary)

    @staticmethod
    def _format_history_tool_context(actions: Any) -> str:
        """Return compact prior tool context for model history messages."""
        if not isinstance(actions, list):
            return ""

        parts: list[str] = []
        for action in actions:
            if not isinstance(action, dict) or action.get("type") != "mcp_tool":
                continue
            tool_name = str(action.get("tool") or "")
            if not tool_name:
                continue
            arguments = action.get("arguments")
            argument_parts = []
            if isinstance(arguments, dict):
                argument_parts = [
                    f"{key}={value}"
                    for key, value in arguments.items()
                    if value
                ]
            status = str(action.get("status") or "ok")
            call_text = tool_name
            if argument_parts:
                call_text += f"({', '.join(argument_parts[:4])})"
            if status != "ok":
                call_text += f" status={status}"
            parts.append(call_text)

        if not parts:
            return ""
        return "Tool context: " + "; ".join(parts[-6:])

    async def _async_handle_message(
        self, user_input: ConversationInput, chat_log_instance: chat_log.ChatLog
    ) -> ConversationResult:
        """Process user input and return response.

        Called by the base ConversationEntity.async_process which manages
        ChatSession and ChatLog lifecycle automatically.
        """
        _LOGGER.info(
            "🎤 Voice request started: text_chars=%d",
            len(str(user_input.text or "")),
        )

        # Store ChatLog for tool execution methods to access
        self._current_chat_log = chat_log_instance
        user_input_token: Token[ConversationInput | None] = _REQUEST_USER_INPUT.set(
            user_input
        )
        conversation_id = chat_log_instance.conversation_id
        conversation_id_token: Token[str | None] = _REQUEST_CONVERSATION_ID.set(
            conversation_id
        )
        adaptive_loaded_tools_token: Token[set[str] | frozenset[str] | None] = (
            _ADAPTIVE_LOADED_TOOL_NAMES.set(set())
        )
        tool_history_token: Token[list[dict[str, Any]] | None] = (
            _REQUEST_TOOL_HISTORY_SUMMARIES.set([])
        )
        persistent_chat_log = (
            self._build_persistent_chat_log_record(user_input, conversation_id)
            if self.chat_log_mode
            else None
        )
        chat_log_token: Token[dict[str, Any] | None] = (
            _PERSISTENT_CHAT_LOG_RECORD.set(persistent_chat_log)
        )

        try:
            return await self._async_handle_message_inner(
                user_input, conversation_id
            )
        finally:
            # Clean up
            _PERSISTENT_CHAT_LOG_RECORD.reset(chat_log_token)
            _ADAPTIVE_LOADED_TOOL_NAMES.reset(adaptive_loaded_tools_token)
            _REQUEST_TOOL_HISTORY_SUMMARIES.reset(tool_history_token)
            _REQUEST_CONVERSATION_ID.reset(conversation_id_token)
            _REQUEST_USER_INPUT.reset(user_input_token)
            self._current_chat_log = None

    async def _async_handle_message_inner(
        self, user_input: ConversationInput, conversation_id: str
    ) -> ConversationResult:
        """Process user input with ChatLog tracking."""
        try:
            _LOGGER.debug("Conversation ID: %s", conversation_id)

            # OpenClaw: bypass entire LLM/MCP pipeline — server handles everything
            if self.server_type == SERVER_TYPE_OPENCLAW:
                return await self._handle_openclaw_message(user_input, conversation_id)

            # Get conversation history
            metrics_enabled = _LOGGER.isEnabledFor(logging.DEBUG)
            setup_started_at = time.monotonic() if metrics_enabled else 0.0
            history_started_at = time.monotonic() if metrics_enabled else 0.0
            history = self.history.get_history(conversation_id)
            history_ms = (
                (time.monotonic() - history_started_at) * 1000
                if metrics_enabled
                else 0.0
            )
            _LOGGER.debug("History retrieved: %d turns", len(history))

            # Build system prompt with context
            prompt_started_at = time.monotonic() if metrics_enabled else 0.0
            system_prompt = await self._build_system_prompt_with_context(user_input)
            prompt_ms = (
                (time.monotonic() - prompt_started_at) * 1000
                if metrics_enabled
                else 0.0
            )
            if self.debug_mode:
                _LOGGER.info(
                    f"📝 System prompt built, length: {len(system_prompt)} chars"
                )

            adaptive_started_at = time.monotonic() if metrics_enabled else 0.0
            await self._prepare_adaptive_tools_for_request(
                user_input.text,
                history=history,
            )
            adaptive_ms = (
                (time.monotonic() - adaptive_started_at) * 1000
                if metrics_enabled
                else 0.0
            )

            # Build conversation messages
            messages = self._build_messages(system_prompt, user_input.text, history)
            if metrics_enabled:
                _LOGGER.debug(
                    (
                        "Initial prompt metrics: prompt_chars=%d messages=%d "
                        "message_chars=%d history_turns=%d history_ms=%.1f "
                        "prompt_build_ms=%.1f adaptive_prepare_ms=%.1f setup_ms=%.1f"
                    ),
                    len(system_prompt),
                    len(messages),
                    self._message_content_char_count(messages),
                    len(history),
                    history_ms,
                    prompt_ms,
                    adaptive_ms,
                    (time.monotonic() - setup_started_at) * 1000,
                )

            if self.debug_mode:
                _LOGGER.info(f"📨 Messages built: {len(messages)} messages")
                for i, msg in enumerate(messages):
                    role = msg.get("role")
                    content_len = (
                        len(msg.get("content", "")) if msg.get("content") else 0
                    )
                    _LOGGER.info(
                        f"  Message {i}: role={role}, content_length={content_len}"
                    )

            # Call LLM API
            _LOGGER.info(f"📡 Calling {self.server_type} API...")
            response_text = await self._call_llm(messages)
            _LOGGER.info(
                f"✅ {self.server_type} response received, length: %d",
                len(response_text),
            )

            return await self._build_response_result(
                response_text, user_input, conversation_id
            )

        except ProviderResponseTimeoutError as err:
            _LOGGER.warning(
                (
                    "Provider request timed out: provider=%s transport=%s "
                    "timeout=%ss attempts=%d iteration=%d"
                ),
                err.provider_name,
                err.transport,
                err.timeout_seconds,
                err.attempts,
                err.iteration,
            )
            await self._finish_persistent_chat_log_record(error=str(err))

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                self._get_friendly_error_message(err),
            )

            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )

        except Exception as err:
            _LOGGER.exception("Error processing conversation")
            await self._finish_persistent_chat_log_record(error=str(err))

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                self._get_friendly_error_message(err),
            )

            return ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )

    async def _handle_openclaw_message(
        self, user_input: ConversationInput, conversation_id: str
    ) -> ConversationResult:
        """Handle a message via the OpenClaw Gateway WebSocket."""
        client = self.hass.data.get(DOMAIN, {}).get(
            self.entry.entry_id, {}
        ).get("openclaw_client")

        if not client:
            raise RuntimeError("OpenClaw client not available — check integration setup")

        _LOGGER.info("📡 Sending to OpenClaw gateway")
        response_text = await client.send_message(user_input.text, self.session_key)
        _LOGGER.info(
            "✅ OpenClaw response received, length: %d", len(response_text)
        )

        return await self._build_response_result(
            response_text, user_input, conversation_id
        )

    async def _build_response_result(
        self,
        response_text: str,
        user_input: ConversationInput,
        conversation_id: str,
    ) -> ConversationResult:
        """Shared post-response pipeline: clean, log, detect follow-up, build result."""
        # Strip thinking tags from reasoning models
        response_text, thinking_content = self._strip_thinking_tags(response_text)
        if thinking_content and self.debug_mode:
            _LOGGER.info(
                "🧠 Thinking content stripped: %d chars",
                len(thinking_content),
            )

        _LOGGER.info("💬 Response text ready, length: %d", len(response_text))

        # Parse response and execute any Home Assistant actions
        actions_taken = await self._execute_actions(response_text, user_input)
        tool_history_summaries = _REQUEST_TOOL_HISTORY_SUMMARIES.get()
        if tool_history_summaries:
            actions_taken.extend(tool_history_summaries)

        # Add final assistant response to ChatLog
        if self._current_chat_log:
            final_content = chat_log.AssistantContent(
                agent_id=self.entity_id, content=response_text
            )
            self._current_chat_log.async_add_assistant_content_without_tools(
                final_content
            )

        # Store in conversation history
        self.history.add_turn(
            conversation_id, user_input.text, response_text, actions=actions_taken
        )

        # Create intent response
        intent_response = intent.IntentResponse(language=user_input.language)
        cleaned_text = self._clean_text_for_tts(response_text)
        intent_response.async_set_speech(cleaned_text)

        # Check if user wants to end (stopwords+1 algorithm)
        user_wants_to_end = False
        if self.follow_up_mode in ["default", "always"]:
            user_wants_to_end = self._detect_user_ending_intent(user_input.text)
            if user_wants_to_end and self.debug_mode:
                _LOGGER.info("🎯 User ending intent detected (stopwords+1)")

        # Determine follow-up mode
        if user_wants_to_end:
            continue_conversation = False
        elif self.follow_up_mode == "always":
            continue_conversation = True
        elif self.follow_up_mode == "none":
            continue_conversation = False
        else:  # "default" - smart mode
            if hasattr(self, "_expecting_response"):
                continue_conversation = self._expecting_response
                delattr(self, "_expecting_response")
                if self.debug_mode:
                    _LOGGER.info("🎯 Using LLM's set_conversation_state indication")
            else:
                continue_conversation = self._detect_follow_up_patterns(
                    response_text
                )
                if self.debug_mode:
                    if continue_conversation:
                        _LOGGER.info("🎯 Pattern detection triggered continuation")
                    else:
                        _LOGGER.info("🎯 No patterns detected, closing conversation")

        if self.debug_mode:
            _LOGGER.info(
                f"🎯 Follow-up mode: {self.follow_up_mode}, Continue: {continue_conversation}"
            )

        await self._finish_persistent_chat_log_record(
            assistant_text=response_text,
            continue_conversation=continue_conversation,
        )

        return ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
            continue_conversation=continue_conversation,
        )

    def _detect_user_ending_intent(self, text: str) -> bool:
        """Detect if user wants to end conversation using stopwords+1 algorithm.

        Handles both single words and multi-word phrases.

        Returns True if:
        - User message contains at least one stop word/phrase, AND
        - User message has ≤1 non-stop word (excluding agent name and matched phrases)

        Examples:
        - "stop" → True (0 non-stop words)
        - "no thanks" → True (both are stop words)
        - "no thank you" → True ("thank you" is a stop phrase)
        - "bye Jarvis" → True (Jarvis removed, 0 non-stop)
        - "ok please" → True (1 non-stop word)
        - "no turn on light" → False (3 non-stop words)
        """
        if not text:
            return False

        # Parse end words from config
        end_words_raw = [
            word.strip().lower() for word in self.end_words.split(",") if word.strip()
        ]
        if not end_words_raw:
            return False

        # Separate multi-word phrases from single words
        multi_word_phrases = [phrase for phrase in end_words_raw if " " in phrase]
        single_words = [word for word in end_words_raw if " " not in word]

        # Normalize text
        text_lower = text.lower().strip()

        # Check if any multi-word phrases are present and remove them
        has_stop_word = False
        remaining_text = text_lower

        for phrase in multi_word_phrases:
            if phrase in remaining_text:
                has_stop_word = True
                # Replace matched phrase with spaces to preserve word boundaries
                remaining_text = remaining_text.replace(phrase, " ")

        # Split remaining text into words
        words = remaining_text.split()

        # Remove agent name
        profile_name_lower = self.profile_name.lower()
        words = [word for word in words if word != profile_name_lower]

        # Check if any single-word stop words are present
        for word in words:
            if word in single_words:
                has_stop_word = True

        if not has_stop_word:
            return False

        # Count non-stop words (words not in single_words list)
        non_stop_words = [
            word for word in words if word not in single_words and word.strip()
        ]

        # End if ≤1 non-stop word
        return len(non_stop_words) <= 1

    def _detect_follow_up_patterns(self, text: str) -> bool:
        """Detect if the response expects a follow-up based on patterns."""
        if not text:
            return False

        # Debug logging to see what we're checking
        if self.debug_mode:
            _LOGGER.info(
                f"🔍 Pattern detection - Full response length: {len(text)} chars"
            )
            _LOGGER.info("🔍 Pattern detection - Checking trailing response window")

        # Check last 200 characters for efficiency
        check_text = text[-200:].lower()

        # Pattern 1: Ends with a question mark
        if check_text.rstrip().endswith("?"):
            if self.debug_mode:
                _LOGGER.info("📊 Question detected: phrase ends with question mark")
            return True

        # Pattern 2: Question phrases (user-configurable)
        question_phrases = [
            phrase.strip().lower()
            for phrase in self.follow_up_phrases.split(",")
            if phrase.strip()
        ]

        for phrase in question_phrases:
            if phrase in check_text:
                if self.debug_mode:
                    _LOGGER.info(f"📊 Follow-up phrase detected: '{phrase}'")
                return True

        return False

    async def _get_current_area(self, user_input: ConversationInput) -> str:
        """Get the area of the satellite/device making the request."""
        try:
            # Try to get device_id from context
            device_id = (
                user_input.device_id if hasattr(user_input, "device_id") else None
            )

            if not device_id:
                _LOGGER.debug("No device_id in conversation input")
                return "Unknown"

            # Get device registry and look up device
            device_reg = dr.async_get(self.hass)
            device_entry = device_reg.async_get(device_id)

            if not device_entry:
                _LOGGER.debug("No device found for device_id: %s", device_id)
                return "Unknown"

            # Get area from device
            area_id = device_entry.area_id
            if not area_id:
                _LOGGER.debug("Device %s has no assigned area", device_id)
                return "Unknown"

            # Get area registry and look up area name
            area_reg = ar.async_get(self.hass)
            area_entry = area_reg.async_get_area(area_id)

            if not area_entry:
                _LOGGER.debug("Area ID %s not found in registry", area_id)
                return "Unknown"

            area_name = area_entry.name
            _LOGGER.info(
                "📍 Current area detected: %s (from device %s)", area_name, device_id
            )
            return area_name

        except Exception as e:
            _LOGGER.warning("Error getting current area: %s", e)
            return "Unknown"

    def _get_home_location(self) -> str:
        """Return a compact home-location summary for prompt context."""
        if not self._get_shared_setting(
            CONF_INCLUDE_HOME_LOCATION, DEFAULT_INCLUDE_HOME_LOCATION
        ):
            return ""

        location_name = str(
            getattr(self.hass.config, "location_name", "") or ""
        ).strip()
        latitude = getattr(self.hass.config, "latitude", None)
        longitude = getattr(self.hass.config, "longitude", None)

        coordinates = ""
        try:
            if latitude is not None and longitude is not None:
                coordinates = f"{float(latitude):.4f}, {float(longitude):.4f}"
        except (TypeError, ValueError):
            coordinates = ""

        if location_name and coordinates:
            return f"{location_name} ({coordinates})"
        if location_name:
            return location_name
        return coordinates

    async def _get_current_user_name(self, user_input: ConversationInput) -> str:
        """Return the current Home Assistant user name for prompt context."""
        if not self._get_shared_setting(
            CONF_INCLUDE_CURRENT_USER, DEFAULT_INCLUDE_CURRENT_USER
        ):
            return ""

        try:
            user_id = getattr(getattr(user_input, "context", None), "user_id", None)
            if not user_id:
                return ""

            user = await self.hass.auth.async_get_user(user_id)
            if not user:
                return ""

            return str(getattr(user, "name", "") or "").strip()
        except Exception as err:
            _LOGGER.debug("Unable to resolve current HA user: %s", err)
            return ""

    async def _get_current_user_context(
        self, user_input: ConversationInput | None
    ) -> dict[str, str]:
        """Return current HA user metadata for MCP tool call context."""
        if not self._get_shared_setting(
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
            DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
        ):
            return {}

        user_id = str(
            getattr(getattr(user_input, "context", None), "user_id", None) or ""
        ).strip()
        if not user_id:
            return {}

        context = {"user_id": user_id}
        try:
            user = await self.hass.auth.async_get_user(user_id)
            user_name = str(getattr(user, "name", "") or "").strip() if user else ""
            if user_name:
                context["user_name"] = user_name
        except Exception as err:
            _LOGGER.debug("Unable to resolve current HA user for tool context: %s", err)
        return context

    def _get_home_location_context(self) -> dict[str, Any]:
        """Return home-location metadata for MCP tool call context."""
        if not self._get_shared_setting(
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
            DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
        ):
            return {}

        context: dict[str, Any] = {}

        location_name = str(
            getattr(self.hass.config, "location_name", "") or ""
        ).strip()
        if location_name:
            context["home_location"] = location_name
            context["home_location_name"] = location_name

        latitude = getattr(self.hass.config, "latitude", None)
        longitude = getattr(self.hass.config, "longitude", None)
        try:
            if latitude is not None and longitude is not None:
                context["home_latitude"] = float(latitude)
                context["home_longitude"] = float(longitude)
        except (TypeError, ValueError):
            pass
        return context

    async def _build_mcp_tool_call_context(
        self, user_input: ConversationInput | None = None
    ) -> dict[str, Any]:
        """Build privacy-gated MCP tool call context for the active profile."""
        context: dict[str, Any] = {
            "profile_entry_id": self.entry.entry_id,
            "profile_name": self.profile_name,
        }
        context.update(await self._get_current_user_context(user_input))
        context.update(self._get_home_location_context())
        return context

    def _get_default_system_prompt(self) -> str:
        """Get the built-in localized default system prompt."""
        return (
            get_language_instruction(self.hass.config.language)
            or DEFAULT_SYSTEM_PROMPT
        )

    def _resolve_prompt_setting(
        self, *, prompt_key: str, mode_key: str, default_prompt: str
    ) -> str:
        """Resolve a prompt using Default/Custom mode with backward compatibility."""
        options = self.entry.options
        data = self.entry.data

        explicit_mode = options.get(mode_key, data.get(mode_key))
        stored_prompt = options.get(prompt_key, data.get(prompt_key))

        if explicit_mode == PROMPT_MODE_CUSTOM:
            return "" if stored_prompt is None else str(stored_prompt)

        if explicit_mode == PROMPT_MODE_DEFAULT:
            return default_prompt

        if stored_prompt in (None, "", default_prompt):
            return default_prompt

        return str(stored_prompt)

    @staticmethod
    def _prompt_references_variable(template_text: str, variable_name: str) -> bool:
        """Return whether a prompt references a supported template variable."""
        legacy_placeholder = f"{{{variable_name}}}"
        if legacy_placeholder in template_text:
            return True
        for block in re.findall(r"{[{%#]-?.*?-?[}%]}", template_text, re.DOTALL):
            if re.search(r"\b" + re.escape(variable_name) + r"\b", block):
                return True
        return False

    def _replace_legacy_prompt_placeholders(
        self,
        prompt: str,
        variables: dict[str, Any],
    ) -> str:
        """Replace legacy {name} placeholders after Jinja rendering."""
        rendered = prompt
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    def _render_prompt_template(
        self,
        template_text: str,
        variables: dict[str, Any],
    ) -> str:
        """Render a Jinja prompt template with legacy placeholder fallback."""
        if not template_text:
            return ""

        try:
            rendered = Template(template_text, self.hass).async_render(
                variables=variables,
                parse_result=False,
            )
        except Exception as err:
            _LOGGER.warning(
                "Failed to render Jinja prompt template, using raw text: %s",
                err,
            )
            rendered = template_text

        return self._replace_legacy_prompt_placeholders(str(rendered), variables)

    async def _build_prompt_template_variables(
        self,
        user_input: ConversationInput | None,
        *templates: str,
    ) -> tuple[dict[str, Any], str]:
        """Collect prompt template variables, fetching costly values lazily."""
        now = dt_util.now()
        combined_template = "\n".join(template for template in templates if template)

        variables: dict[str, Any] = {
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
        }

        current_area = "Unknown"
        if (
            self._prompt_references_variable(combined_template, "current_area")
            or self.music_assistant_support_enabled
        ) and user_input is not None:
            current_area = await self._get_current_area(user_input)
        variables["current_area"] = current_area

        if (
            self._prompt_references_variable(combined_template, "current_user")
            or self._prompt_references_variable(
                combined_template,
                "current_user_context",
            )
        ) and user_input is not None:
            current_user = await self._get_current_user_name(user_input)
        else:
            current_user = ""
        variables["current_user"] = current_user
        variables["current_user_context"] = (
            f"Current user: {current_user}" if current_user else ""
        )

        if self._prompt_references_variable(
            combined_template,
            "home_location",
        ) or self._prompt_references_variable(
            combined_template,
            "home_location_context",
        ):
            home_location = self._get_home_location()
        else:
            home_location = ""
        variables["home_location"] = home_location
        variables["home_location_context"] = (
            f"Home location: {home_location}" if home_location else ""
        )

        if self._prompt_references_variable(combined_template, "response_mode"):
            variables["response_mode"] = RESPONSE_MODE_INSTRUCTIONS.get(
                self.follow_up_mode,
                RESPONSE_MODE_INSTRUCTIONS["default"],
            )
        else:
            variables["response_mode"] = ""

        if self._prompt_references_variable(combined_template, "index"):
            index_manager = self.hass.data.get(DOMAIN, {}).get("index_manager")
            if index_manager:
                index = await index_manager.get_index()
                variables["index"] = json.dumps(index, separators=(",", ":"))
            else:
                variables["index"] = "{}"
                _LOGGER.warning("IndexManager not available, using empty index")
        else:
            variables["index"] = "{}"

        return variables, current_area

    async def _build_system_prompt_with_context(
        self, user_input: ConversationInput
    ) -> str:
        """Build the compact system prompt used for model calls."""
        try:
            if self.entry.data.get(CONF_SERVER_TYPE) == SERVER_TYPE_OPENCLAW:
                system_prompt_template = ""
            else:
                system_prompt_template = self._resolve_prompt_setting(
                    prompt_key=CONF_SYSTEM_PROMPT,
                    mode_key=CONF_SYSTEM_PROMPT_MODE,
                    default_prompt=self._get_default_system_prompt(),
                )
            technical_prompt_template = self._resolve_prompt_setting(
                prompt_key=CONF_TECHNICAL_PROMPT,
                    mode_key=CONF_TECHNICAL_PROMPT_MODE,
                    default_prompt=DEFAULT_TECHNICAL_PROMPT,
            )

            variables, current_area = await self._build_prompt_template_variables(
                user_input,
                system_prompt_template,
                technical_prompt_template,
            )
            system_prompt = self._render_prompt_template(
                system_prompt_template,
                variables,
            )
            technical_prompt = self._render_prompt_template(
                technical_prompt_template,
                variables,
            )

            technical_prompt = re.sub(r"\n{3,}", "\n\n", technical_prompt).strip()

            if self.light_context_mode:
                optional_instructions = ""
            elif self.adaptive_context_mode:
                optional_instructions = self._build_adaptive_technical_instructions()
            else:
                optional_instructions = self._build_optional_technical_instructions(
                    current_area
                )
            if optional_instructions:
                technical_prompt = (
                    f"{technical_prompt.rstrip()}\n\n{optional_instructions}"
                )

            if system_prompt:
                return f"{system_prompt}\n\n{technical_prompt}"
            return technical_prompt

        except Exception as e:
            _LOGGER.error("Error building system prompt: %s", e)
            return "You are a Home Assistant voice assistant. Use MCP tools to control devices."

    async def _get_home_context(self) -> str:
        """Get lightweight home context (areas and domains) to help LLM with discovery."""
        try:
            # Fetch areas
            areas_result = await self._call_mcp_tool("list_areas", {})
            areas_text = ""
            if "result" in areas_result:
                areas_text = areas_result["result"]

            # Fetch domains
            domains_result = await self._call_mcp_tool("list_domains", {})
            domains_text = ""
            if "result" in domains_result:
                domains_text = domains_result["result"]

            # Format context section
            context = "# Your Home Configuration\n\n"
            if areas_text:
                context += f"{areas_text}\n\n"
            if domains_text:
                context += f"{domains_text}\n"

            _LOGGER.debug("Home context added: %d characters", len(context))
            return context

        except Exception as e:
            _LOGGER.warning("Could not fetch home context: %s", e)
            return ""

    def _build_system_prompt(self) -> str:
        """Build system prompt (legacy sync version - note: cannot include index without async)."""
        try:
            now = dt_util.now()
            if self.entry.data.get(CONF_SERVER_TYPE) == SERVER_TYPE_OPENCLAW:
                system_prompt_template = ""
            else:
                system_prompt_template = self._resolve_prompt_setting(
                    prompt_key=CONF_SYSTEM_PROMPT,
                    mode_key=CONF_SYSTEM_PROMPT_MODE,
                    default_prompt=self._get_default_system_prompt(),
                )
            technical_prompt_template = self._resolve_prompt_setting(
                prompt_key=CONF_TECHNICAL_PROMPT,
                mode_key=CONF_TECHNICAL_PROMPT_MODE,
                default_prompt=DEFAULT_TECHNICAL_PROMPT,
            )

            home_location = self._get_home_location()
            variables: dict[str, Any] = {
                "time": now.strftime("%H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "current_area": "Unknown",
                "current_user": "",
                "current_user_context": "",
                "home_location": home_location,
                "home_location_context": (
                    f"Home location: {home_location}" if home_location else ""
                ),
                "index": "{}",
                "response_mode": RESPONSE_MODE_INSTRUCTIONS.get(
                    self.follow_up_mode, RESPONSE_MODE_INSTRUCTIONS["default"]
                ),
            }

            system_prompt = self._render_prompt_template(
                system_prompt_template,
                variables,
            )
            technical_prompt = self._render_prompt_template(
                technical_prompt_template,
                variables,
            )

            technical_prompt = re.sub(r"\n{3,}", "\n\n", technical_prompt).strip()

            if self.light_context_mode:
                optional_instructions = ""
            elif self.adaptive_context_mode:
                optional_instructions = self._build_adaptive_technical_instructions()
            else:
                optional_instructions = self._build_optional_technical_instructions(
                    "Unknown"
                )
            if optional_instructions:
                technical_prompt = (
                    f"{technical_prompt.rstrip()}\n\n{optional_instructions}"
                )

            if system_prompt:
                return f"{system_prompt}\n\n{technical_prompt}"
            return technical_prompt

        except Exception as e:
            _LOGGER.error("Error building system prompt: %s", e)
            # Return a basic prompt as fallback
            return "You are a Home Assistant voice assistant. Use MCP tools to control devices."

    def _build_messages(
        self, system_prompt: str, user_text: str, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build provider-neutral conversation messages."""
        messages = [{"role": "system", "content": system_prompt}]

        # OpenClaw manages its own session history on the gateway.
        if self.server_type != SERVER_TYPE_OPENCLAW:
            history_limit = max(0, self.max_history)
            if self.light_context_mode:
                history_limit = min(history_limit, LIGHT_CONTEXT_MAX_HISTORY)
            if history_limit > 0:
                for turn in history[-history_limit:]:
                    messages.append({"role": "user", "content": turn["user"]})
                    assistant_content = str(turn["assistant"])
                    tool_context = self._format_history_tool_context(
                        turn.get("actions")
                    )
                    if tool_context:
                        assistant_content = (
                            f"{assistant_content.rstrip()}\n{tool_context}"
                        )
                    messages.append(
                        {"role": "assistant", "content": assistant_content}
                    )

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    async def _fetch_mcp_tools_from_server(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch profile-visible MCP tool definitions from the MCP server."""
        try:
            mcp_url = f"http://localhost:{self.mcp_port}"

            # Get tools list from MCP server
            timeout = aiohttp.ClientTimeout(total=5)
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1,
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{mcp_url}/",
                    **self._mcp_post_kwargs(payload),
                ) as response:
                    if response.status != 200:
                        _LOGGER.warning("Failed to get MCP tools: %d", response.status)
                        return None

                    data = await response.json()
                    if "result" in data and "tools" in data["result"]:
                        tools = self._filter_mcp_tools_for_profile(
                            data["result"]["tools"]
                        )
                        _LOGGER.info(
                            "Retrieved %d MCP tools after profile filtering",
                            len(tools),
                        )

                        tool_names = []
                        for tool in tools:
                            tool_names.append(tool["name"])

                        _LOGGER.info("MCP tools available: %s", ", ".join(tool_names))
                        if "perform_action" in tool_names:
                            _LOGGER.info("✅ perform_action tool is available")
                        else:
                            _LOGGER.warning("⚠️ perform_action tool NOT found!")

                        return tools
                    return None

        except Exception as err:
            _LOGGER.error("Failed to get MCP tools: %s", err)
            return None

    async def _get_profile_mcp_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Return profile-visible MCP tool definitions, using a short-lived cache."""
        cache_key = self._build_mcp_tool_cache_key()
        now = time.monotonic()

        if (
            self._cached_profile_mcp_tools is not None
            and self._cached_profile_mcp_tools_key == cache_key
            and (now - self._cached_profile_mcp_tools_fetched_at)
            < MCP_TOOL_CACHE_TTL_SECONDS
        ):
            _LOGGER.debug(
                "Using cached MCP tool definitions for profile: tools=%d age_ms=%.1f",
                len(self._cached_profile_mcp_tools),
                (now - self._cached_profile_mcp_tools_fetched_at) * 1000,
            )
            return list(self._cached_profile_mcp_tools)

        fetch_started_at = time.monotonic()
        tools = await self._fetch_mcp_tools_from_server()
        fetch_ms = (time.monotonic() - fetch_started_at) * 1000
        if tools is not None:
            self._cached_profile_mcp_tools = list(tools)
            self._cached_profile_mcp_tools_key = cache_key
            self._cached_profile_mcp_tools_fetched_at = now
            _LOGGER.debug(
                "Fetched MCP tool definitions for profile: tools=%d latency_ms=%.1f",
                len(tools),
                fetch_ms,
            )
            return list(tools)

        if (
            self._cached_profile_mcp_tools is not None
            and self._cached_profile_mcp_tools_key == cache_key
        ):
            _LOGGER.warning(
                "Using stale cached MCP tools after refresh failure: "
                "tools=%d fetch_latency_ms=%.1f",
                len(self._cached_profile_mcp_tools),
                fetch_ms,
            )
            return list(self._cached_profile_mcp_tools)

        _LOGGER.debug(
            "MCP tool definition fetch returned no tools: latency_ms=%.1f", fetch_ms
        )
        return None

    async def _get_mcp_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Return available LLM-facing MCP tools for the current context mode."""
        tools = await self._get_profile_mcp_tools()
        if tools is None:
            return None
        return self._build_llm_tools_for_context(tools)

    def _build_llm_tools_for_context(
        self,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build the advertised LLM tool surface for the current context mode."""
        if self.adaptive_context_mode:
            return build_adaptive_llm_tools(
                tools,
                base_tool_names=LIGHT_CONTEXT_TOOL_NAMES,
                loaded_tool_names=_adaptive_loaded_tool_names(),
            )

        return self._convert_mcp_tools_to_llm_tools(tools)

    @staticmethod
    def _tool_definition_name(tool: Dict[str, Any]) -> str:
        """Return the MCP tool name for a raw tool definition."""
        return tool_definition_name(tool)

    def _adaptive_tool_catalog_entry(self, tool: Dict[str, Any]) -> dict[str, Any]:
        """Return one compact catalog entry for model-facing adaptive discovery."""
        tool_name = self._tool_definition_name(tool)
        routing_summary = self._build_tool_routing_summary(tool.get("routingHints"))
        summary = self._compact_text(
            str(tool.get("llmDescription") or tool.get("llm_description") or "")
            or str(tool.get("description") or ""),
            max_len=110,
        )
        if routing_summary:
            summary = self._compact_text(
                f"{summary.rstrip(' .')} | {routing_summary}",
                max_len=180,
            )
        family = get_optional_tool_family(tool_name)
        entry: dict[str, Any] = {
            "name": tool_name,
            "summary": summary,
            "schema_loaded": tool_name in _adaptive_loaded_tool_names(),
        }
        if family:
            entry["family"] = family
        return entry

    def _match_adaptive_tool_definitions(
        self,
        tools: List[Dict[str, Any]],
        *,
        query: str = "",
        tool_names: list[str] | None = None,
        limit: int = 20,
    ) -> list[Dict[str, Any]]:
        """Return adaptive catalog matches from profile-visible raw tools."""
        return match_adaptive_tool_definitions(
            tools,
            query=query,
            tool_names=tool_names,
            limit=limit,
            base_tool_names=LIGHT_CONTEXT_TOOL_NAMES,
        )

    def _select_initial_adaptive_tool_names(
        self,
        tools: List[Dict[str, Any]],
        user_text: str,
        *,
        limit: int = 2,
        minimum_score: int = 18,
    ) -> frozenset[str]:
        """Return highly likely optional tool schemas to preload for this request."""
        scored: list[tuple[int, str]] = []
        loaded_names = _adaptive_loaded_tool_names()
        for tool in tools:
            tool_name = self._tool_definition_name(tool)
            if (
                not tool_name
                or tool_name in LIGHT_CONTEXT_TOOL_NAMES
                or tool_name in loaded_names
                or tool_name in ADAPTIVE_META_TOOL_NAMES
            ):
                continue
            score = score_adaptive_tool_match(
                tool,
                user_text,
                base_tool_names=LIGHT_CONTEXT_TOOL_NAMES,
            )
            if score >= minimum_score:
                scored.append((score, tool_name))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return frozenset(name for _score, name in scored[:limit])

    @staticmethod
    def _is_bounded_adaptive_follow_up(user_text: str) -> bool:
        """Return whether a short utterance explicitly refers to prior context."""
        normalized = " ".join(str(user_text or "").split()).casefold()
        words = re.findall(r"\w+", normalized, flags=re.UNICODE)
        if not words or len(words) > ADAPTIVE_FOLLOW_UP_MAX_WORDS:
            return False
        if normalized.startswith(ADAPTIVE_FOLLOW_UP_PREFIXES):
            return True
        return bool(set(words) & ADAPTIVE_FOLLOW_UP_REFERENCE_TERMS)

    def _select_retained_adaptive_tool_names(
        self,
        tools: List[Dict[str, Any]],
        user_text: str,
        history: List[Dict[str, Any]],
    ) -> frozenset[str]:
        """Retain recently used optional schemas for a same-topic follow-up."""
        if not history:
            return frozenset()

        previous_turn = history[-1]
        actions = previous_turn.get("actions")
        if not isinstance(actions, list):
            return frozenset()

        available_tools = {
            self._tool_definition_name(tool): tool
            for tool in tools
            if self._tool_definition_name(tool)
        }
        candidate_names: list[str] = []
        for action in reversed(actions):
            if not isinstance(action, dict) or action.get("type") != "mcp_tool":
                continue
            tool_name = str(action.get("tool") or "")
            if (
                not tool_name
                or tool_name in LIGHT_CONTEXT_TOOL_NAMES
                or tool_name in ADAPTIVE_META_TOOL_NAMES
                or tool_name not in available_tools
                or tool_name in candidate_names
            ):
                continue
            candidate_names.append(tool_name)
            if len(candidate_names) >= ADAPTIVE_RETAINED_SCHEMA_LIMIT:
                break

        if not candidate_names:
            return frozenset()

        previous_user_text = str(previous_turn.get("user") or "")
        current_terms = set(normalize_adaptive_query_terms(user_text))
        previous_terms = set(normalize_adaptive_query_terms(previous_user_text))
        same_topic = bool(current_terms & previous_terms)
        if not same_topic:
            same_topic = any(
                score_adaptive_tool_match(
                    available_tools[name],
                    user_text,
                    base_tool_names=LIGHT_CONTEXT_TOOL_NAMES,
                )
                > 0
                for name in candidate_names
            )
        if not same_topic:
            same_topic = self._is_bounded_adaptive_follow_up(user_text)

        return frozenset(candidate_names if same_topic else ())

    async def _prepare_adaptive_tools_for_request(
        self,
        user_text: str,
        *,
        history: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Preload obvious adaptive schemas before the first model turn."""
        if not self.adaptive_context_mode:
            return

        tools = await self._get_profile_mcp_tools() or []
        retained_names = self._select_retained_adaptive_tool_names(
            tools,
            user_text,
            history or [],
        )
        preload_names = self._select_initial_adaptive_tool_names(tools, user_text)
        selected_names = retained_names | preload_names
        if not selected_names:
            return

        _adaptive_loaded_tool_names().update(selected_names)
        _LOGGER.debug(
            "Adaptive context prepared tool schemas: retained=%s matched=%s",
            ", ".join(sorted(retained_names)) or "none",
            ", ".join(sorted(preload_names)) or "none",
        )

    @staticmethod
    def _normalize_requested_tool_names(value: Any) -> list[str]:
        """Return normalized tool names from a string or list argument."""
        if isinstance(value, str):
            raw_names = value.split(",")
        elif isinstance(value, list):
            raw_names = value
        else:
            raw_names = []
        names: list[str] = []
        for raw_name in raw_names:
            name = str(raw_name or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        """Return an integer clamped to a small safe range."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return min(max(parsed, minimum), maximum)

    async def _handle_adaptive_meta_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle adaptive tool catalog/schema meta tools without MCP round-trips."""
        profile_tools = await self._get_profile_mcp_tools() or []
        query = str(arguments.get("query") or "").strip()

        if tool_name == ADAPTIVE_TOOL_CATALOG_NAME:
            limit = self._bounded_int(
                arguments.get("limit"),
                default=20,
                minimum=1,
                maximum=50,
            )
            matches = self._match_adaptive_tool_definitions(
                profile_tools,
                query=query,
                limit=limit,
            )
            payload = {
                "tools": [
                    self._adaptive_tool_catalog_entry(tool)
                    for tool in matches
                ],
                "loaded_tool_schemas": sorted(_adaptive_loaded_tool_names()),
                "next_step": (
                    f"Call {ADAPTIVE_TOOL_SCHEMA_NAME} with the exact tool names "
                    "you need before using optional or custom tools."
                ),
            }
        elif tool_name == ADAPTIVE_TOOL_SCHEMA_NAME:
            requested_names = self._normalize_requested_tool_names(
                arguments.get("tool_names")
            )
            limit = self._bounded_int(
                arguments.get("limit"),
                default=8,
                minimum=1,
                maximum=8,
            )
            matches = self._match_adaptive_tool_definitions(
                profile_tools,
                query=query,
                tool_names=requested_names,
                limit=limit,
            )
            matched_names = [self._tool_definition_name(tool) for tool in matches]
            if matched_names:
                _adaptive_loaded_tool_names().update(matched_names)
            missing_names = [
                name for name in requested_names if name not in set(matched_names)
            ]
            payload = {
                "loaded_tools": [
                    self._adaptive_tool_catalog_entry(tool)
                    for tool in matches
                ],
                "not_found": missing_names,
                "next_step": (
                    "The loaded tool schemas will be available in the next model "
                    "call. Use the loaded tool directly if it is needed to answer."
                ),
            }
        else:
            payload = {"error": f"Unknown adaptive meta tool: {tool_name}"}

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ]
        }

    def _filter_mcp_tools_for_profile(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter shared MCP tools down to the subset enabled for this profile."""
        profile_tools = [
            tool
            for tool in tools
            if self._is_tool_enabled_for_profile(tool.get("name", ""))
        ]

        if self.light_context_mode:
            filtered_tools = [
                tool
                for tool in profile_tools
                if tool.get("name", "") in LIGHT_CONTEXT_TOOL_NAMES
            ]
            _LOGGER.info(
                "Light context mode exposing %d of %d profile-visible MCP tools",
                len(filtered_tools),
                len(profile_tools),
            )
        else:
            filtered_tools = profile_tools

        filtered_names = [tool.get("name", "") for tool in filtered_tools]
        _LOGGER.info("Profile-visible MCP tools: %s", ", ".join(filtered_names))
        return filtered_tools

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_input: ConversationInput | None = None,
    ) -> Dict[str, Any]:
        """Execute a single MCP tool and return the result."""
        _LOGGER.info(
            "🔧 Executing MCP tool: %s (argument_keys=%s, argument_bytes=%d)",
            tool_name,
            _mapping_key_summary(arguments),
            _json_size_bytes(arguments),
        )

        if not self._is_tool_enabled_for_profile(tool_name):
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Tool '{tool_name}' is disabled for this profile. "
                            "Use this profile's enabled tools instead."
                        ),
                    }
                ],
            }

        try:
            mcp_url = f"http://localhost:{self.mcp_port}"

            # Create JSON-RPC request for tool execution
            request_id = f"tool_{uuid.uuid4().hex[:8]}"
            tool_user_input = (
                user_input if user_input is not None else _REQUEST_USER_INPUT.get()
            )
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                    "context": await self._build_mcp_tool_call_context(
                        tool_user_input
                    ),
                },
                "id": request_id,
            }

            _LOGGER.debug(
                "MCP request prepared: id=%s tool=%s argument_keys=%s context_keys=%s payload_bytes=%d",
                request_id,
                tool_name,
                _mapping_key_summary(arguments),
                _mapping_key_summary(payload["params"].get("context")),
                _json_size_bytes(payload),
            )

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{mcp_url}/",
                    **self._mcp_post_kwargs(payload),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            "MCP tool call failed: status=%s error=%s",
                            response.status,
                            _redacted_log_snippet(error_text),
                        )
                        return {
                            "error": f"Tool execution failed with HTTP {response.status}"
                        }

                    data = await response.json()
                    _LOGGER.debug(
                        "MCP response received: id=%s has_result=%s has_error=%s response_bytes=%d",
                        request_id,
                        "result" in data,
                        "error" in data,
                        _json_size_bytes(data),
                    )

                    if "result" in data:
                        return self._normalize_mcp_tool_response(data["result"])
                    if "error" in data:
                        return {"error": data["error"]}
                    return {"result": str(data.get("result", ""))}

        except Exception as e:
            _LOGGER.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    def _normalize_mcp_tool_response(self, result: Any) -> Dict[str, Any]:
        """Normalize an MCP JSON-RPC tool result into one predictable shape."""
        if isinstance(result, dict):
            if any(
                key in result
                for key in ("content", "structuredContent", "isError", "_meta")
            ):
                return result
            return {"result": result}
        return {"result": result}

    def _format_tool_result_for_llm(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Serialize a tool result for follow-up LLM turns without losing structure."""
        if "error" in result:
            error_data = result["error"]
            if isinstance(error_data, dict):
                error_message = error_data.get("message", str(error_data))
            else:
                error_message = str(error_data)
            return self._compact_tool_result_for_llm(
                tool_name,
                f"ERROR: {error_message}",
            )

        simple_text = self._extract_simple_text_result(result)
        if simple_text is not None:
            return self._compact_tool_result_for_llm(tool_name, simple_text)

        if "result" in result and not any(
            key in result for key in ("content", "structuredContent", "isError", "_meta")
        ):
            raw_result = result["result"]
            if isinstance(raw_result, str):
                return self._compact_tool_result_for_llm(tool_name, raw_result)
            return self._compact_tool_result_for_llm(
                tool_name,
                json.dumps(
                    self._sanitize_tool_result_for_llm(raw_result),
                    ensure_ascii=False,
                    default=str,
                ),
            )

        return self._compact_tool_result_for_llm(
            tool_name,
            json.dumps(
                self._sanitize_tool_result_for_llm(result),
                ensure_ascii=False,
                default=str,
            ),
        )

    def _extract_simple_text_result(self, result: Dict[str, Any]) -> str | None:
        """Return plain text when the MCP result is a single text block."""
        content = result.get("content")
        if (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
            and content[0].get("type") == "text"
            and "structuredContent" not in result
            and "_meta" not in result
        ):
            return str(content[0].get("text") or "")
        return None

    def _sanitize_tool_result_for_llm(self, value: Any) -> Any:
        """Sanitize structured tool results so binary/image payloads stay compact."""
        if isinstance(value, dict):
            if value.get("type") == "image":
                data_field = value.get("data")
                omitted_bytes = 0
                if isinstance(data_field, str):
                    omitted_bytes = int(len(data_field) * 0.75)
                return {
                    "type": "image",
                    "mimeType": value.get("mimeType"),
                    "data": f"[binary image omitted: ~{omitted_bytes} bytes]",
                }

            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                if key in {"data", "blob", "b64_json"} and isinstance(item, str):
                    sanitized[key] = f"[omitted {len(item)} chars]"
                    continue
                sanitized[key] = self._sanitize_tool_result_for_llm(item)
            return sanitized

        if isinstance(value, list):
            return [self._sanitize_tool_result_for_llm(item) for item in value]

        if isinstance(value, str) and len(value) > 2000:
            return self._compact_tool_result_for_llm("structured_result", value)

        return value

    def _strip_thinking_tags(self, text: str) -> tuple[str, str]:
        """Strip thinking/reasoning tags from model output.

        Reasoning models output chain-of-thought in various tag formats:
        - <think>...</think> (Qwen3, DeepSeek R1, GPT-OSS)
        - <|thought|>...<|/thought|> (some fine-tuned models)

        This content should not be shown to users or spoken via TTS.

        Returns:
            Tuple of (cleaned_text, thinking_content)
            - cleaned_text: Response with thinking tags removed
            - thinking_content: The extracted thinking content (for debug logs)
        """
        # Match all known thinking tag formats (case insensitive, multiline)
        patterns = [
            r"<think>(.*?)</think>",
            r"<\|thought\|>(.*?)<\|/thought\|>",
        ]

        all_matches = []
        cleaned_text = text

        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            all_matches.extend(matches)
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)

        if not all_matches:
            return text, ""

        thinking_content = "\n".join(all_matches)

        # Clean up extra whitespace that might be left
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)  # Multiple newlines
        cleaned_text = cleaned_text.strip()

        return cleaned_text, thinking_content

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS to handle special characters properly."""
        # ALWAYS run character normalization (existing fixes)
        # Replace ALL apostrophe variants with standard apostrophe
        text = text.replace(
            """, "'")  # U+2019 RIGHT SINGLE QUOTATION MARK
        text = text.replace(""",
            "'",
        )  # U+2018 LEFT SINGLE QUOTATION MARK
        text = text.replace("´", "'")  # U+00B4 ACUTE ACCENT
        text = text.replace("`", "'")  # U+0060 GRAVE ACCENT
        text = text.replace("′", "'")  # U+2032 PRIME
        text = text.replace("‛", "'")  # U+201B SINGLE HIGH-REVERSED-9 QUOTATION MARK
        text = text.replace("ʻ", "'")  # U+02BB MODIFIER LETTER TURNED COMMA
        text = text.replace("ʼ", "'")  # U+02BC MODIFIER LETTER APOSTROPHE
        text = text.replace("ˈ", "'")  # U+02C8 MODIFIER LETTER VERTICAL LINE
        text = text.replace("ˊ", "'")  # U+02CA MODIFIER LETTER ACUTE ACCENT
        text = text.replace("ˋ", "'")  # U+02CB MODIFIER LETTER GRAVE ACCENT

        # Replace smart quotes
        text = text.replace('"', '"')  # U+201C LEFT DOUBLE QUOTATION MARK
        text = text.replace('"', '"')  # U+201D RIGHT DOUBLE QUOTATION MARK
        text = text.replace("„", '"')  # U+201E DOUBLE LOW-9 QUOTATION MARK
        text = text.replace("‟", '"')  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK

        # Replace dashes with commas for pauses
        text = text.replace("—", ", ")  # U+2014 EM DASH
        text = text.replace("–", ", ")  # U+2013 EN DASH
        text = text.replace("‒", ", ")  # U+2012 FIGURE DASH
        text = text.replace("―", ", ")  # U+2015 HORIZONTAL BAR

        # Other fixes
        text = text.replace("…", "...")  # U+2026 HORIZONTAL ELLIPSIS
        text = text.replace("•", "-")  # U+2022 BULLET

        # Normalize stray spaces before punctuation so spoken/displayed text
        # does not come out as "it , the weather entity is available."
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)

        # ONLY apply aggressive cleaning if clean_responses enabled
        if not self.clean_responses:
            return text

        # 1. Strip emojis
        text = re.sub(
            r"[\U00010000-\U0010ffff]", "", text
        )  # Supplementary planes (most emojis)
        text = re.sub(
            r"[\u2600-\u26FF\u2700-\u27BF]", "", text
        )  # Misc symbols & dingbats
        text = re.sub(r"[\uE000-\uF8FF]", "", text)  # Private use area

        # 2. Remove markdown (order matters - bold before italic)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold** → bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # *italic* → italic
        text = re.sub(r"__(.+?)__", r"\1", text)  # __bold__ → bold
        text = re.sub(r"_(.+?)_", r"\1", text)  # _italic_ → italic
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # [text](url) → text
        text = re.sub(r"`([^`]+)`", r"\1", text)  # `code` → code
        text = re.sub(r"```[\s\S]+?```", "", text)  # ```code block``` → removed
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # # Header → Header

        # 3. Convert symbols to words
        SYMBOL_MAP = {
            "°C": " degrees celsius",
            "°F": " degrees fahrenheit",
            "°": " degrees",
            "%": " percent",
            "€": " euros",
            "£": " pounds",
            "$": " dollars",
            "&": " and",
            "+": " plus",
            "=": " equals",
            "<": " less than",
            ">": " greater than",
            "@": " at",
            "#": " number",
            "×": " times",
            "÷": " divided by",
        }
        for symbol, word in SYMBOL_MAP.items():
            text = text.replace(symbol, word)

        # 4. Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    async def _trigger_tts(self, text: str):
        """Streaming interim TTS is intentionally disabled.

        Home Assistant already handles speaking the final response, and the old
        hardcoded media_player target was both environment-specific and added
        avoidable work during each request.
        """
        del text
        return

    async def _execute_single_tool_call(
        self, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single tool call and return a provider-ready tool message."""
        tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        arguments_str = function.get("arguments")

        _LOGGER.info(f"📞 Processing tool call {tool_call_id}: {tool_name}")

        tool_entry: dict[str, Any] | None = None
        try:
            arguments = self._parse_tool_arguments(arguments_str)
            tool_entry = self._start_persistent_tool_log(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                raw_arguments=arguments_str,
            )

            # Execute the tool
            if tool_name in ADAPTIVE_META_TOOL_NAMES:
                result = await self._handle_adaptive_meta_tool(tool_name, arguments)
            else:
                result = await self._call_mcp_tool(tool_name, arguments)

            # Format result for LLM consumption
            content = self._format_tool_result_for_llm(tool_name, result)
            self._finish_persistent_tool_log(
                tool_entry,
                result=result,
                llm_content=content,
            )
            self._record_tool_history_summary(tool_name, arguments, result)

            if tool_name == "set_conversation_state" and content:
                if "conversation_state:true" in content.lower():
                    self._expecting_response = True
                    _LOGGER.debug(
                        "🔄 Conversation will continue - expecting response"
                    )
                elif "conversation_state:false" in content.lower():
                    self._expecting_response = False
                    _LOGGER.debug(
                        "🔄 Conversation will close - not expecting response"
                    )

            return self._get_llm_provider().build_tool_result_message(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=content if content is not None else "",
            )

        except Exception as e:
            _LOGGER.error(f"Error executing tool {tool_name}: {e}")
            self._finish_persistent_tool_log(tool_entry, error=str(e))
            self._record_tool_history_summary(
                tool_name,
                self._parse_tool_arguments(arguments_str),
                error=str(e),
            )
            error_content = json.dumps({"error": str(e)})
            return self._get_llm_provider().build_tool_result_message(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=error_content,
            )

    @property
    def tool_call_budget(self) -> int:
        """Return the per-request MCP tool-call budget."""
        try:
            return max(1, int(self.max_iterations))
        except (TypeError, ValueError):
            return max(1, int(DEFAULT_MAX_ITERATIONS))

    @property
    def model_turn_limit(self) -> int:
        """Return the per-request model-call limit around the tool budget."""
        adaptive_turn_allowance = 4 if self.adaptive_context_mode else 1
        return self.tool_call_budget + adaptive_turn_allowance

    @staticmethod
    def _tool_call_id(tool_call: Dict[str, Any]) -> str:
        """Return a stable ID for a provider-normalized tool call."""
        return str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}")

    @staticmethod
    def _tool_call_name(tool_call: Dict[str, Any]) -> str:
        """Return the function name for a provider-normalized tool call."""
        function = tool_call.get("function") or {}
        return str(function.get("name") or "unknown")

    @classmethod
    def _is_budgeted_tool_call(cls, tool_call: Dict[str, Any]) -> bool:
        """Return whether a tool call should consume the user's MCP tool budget."""
        return cls._tool_call_name(tool_call) not in ADAPTIVE_META_TOOL_NAMES

    @classmethod
    def _count_budgeted_tool_calls(cls, tool_calls: List[Dict[str, Any]]) -> int:
        """Return how many calls in a list consume the MCP tool budget."""
        return sum(1 for tool_call in tool_calls if cls._is_budgeted_tool_call(tool_call))

    def _build_budget_skipped_tool_result(
        self,
        provider: LLMProvider,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a provider-ready tool result for a skipped over-budget call."""
        return provider.build_tool_result_message(
            tool_call_id=self._tool_call_id(tool_call),
            tool_name=self._tool_call_name(tool_call),
            content=json.dumps(
                {
                    "error": TOOL_BUDGET_EXHAUSTED_RESULT,
                    "budget_exhausted": True,
                }
            ),
        )

    def _prepare_tool_calls_for_budget(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        tool_calls_used: int,
        provider: LLMProvider,
    ) -> ToolCallBudgetPlan:
        """Split requested tool calls into executable and skipped budget results."""
        remaining = max(0, self.tool_call_budget - tool_calls_used)
        executable_tool_calls: list[dict[str, Any]] = []
        executable_indexes: list[int] = []
        skipped_tool_calls: list[tuple[int, dict[str, Any]]] = []
        budgeted_calls_added = 0

        for index, tool_call in enumerate(tool_calls):
            if not self._is_budgeted_tool_call(tool_call):
                executable_tool_calls.append(tool_call)
                executable_indexes.append(index)
                continue
            if budgeted_calls_added < remaining:
                executable_tool_calls.append(tool_call)
                executable_indexes.append(index)
                budgeted_calls_added += 1
                continue
            skipped_tool_calls.append((index, tool_call))

        if skipped_tool_calls:
            _LOGGER.warning(
                "Skipping %d over-budget tool calls after %d/%d calls were used",
                len(skipped_tool_calls),
                tool_calls_used,
                self.tool_call_budget,
            )

        skipped_results_by_index = {
            index: self._build_budget_skipped_tool_result(provider, tool_call)
            for index, tool_call in skipped_tool_calls
        }
        return ToolCallBudgetPlan(
            executable_calls=executable_tool_calls,
            executable_indexes=executable_indexes,
            skipped_results_by_index=skipped_results_by_index,
            original_count=len(tool_calls),
            exhausted=bool(skipped_tool_calls),
        )

    @staticmethod
    def _merge_tool_results_in_call_order(
        plan: ToolCallBudgetPlan,
        executable_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge executed and skipped tool results back into tool-call order."""
        results_by_index = dict(plan.skipped_results_by_index)
        for index, result in zip(
            plan.executable_indexes,
            executable_results,
            strict=True,
        ):
            results_by_index[index] = result
        return [
            results_by_index[index]
            for index in range(plan.original_count)
            if index in results_by_index
        ]

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute a list of tool calls and return provider-ready results."""
        if not tool_calls:
            return []

        results: list[dict[str, Any] | None] = [None] * len(tool_calls)
        concurrent_calls: list[tuple[int, dict[str, Any]]] = []

        for index, tool_call in enumerate(tool_calls):
            if self._tool_call_name(tool_call) in ADAPTIVE_META_TOOL_NAMES:
                # Meta tools update request-scoped schema state, so run them in
                # this task rather than inside the gathered child tasks below.
                results[index] = await self._execute_single_tool_call(tool_call)
            else:
                concurrent_calls.append((index, tool_call))

        if concurrent_calls:
            concurrent_results = await asyncio.gather(
                *(
                    self._execute_single_tool_call(tool_call)
                    for _index, tool_call in concurrent_calls
                )
            )
            for (index, _tool_call), result in zip(
                concurrent_calls,
                concurrent_results,
                strict=True,
            ):
                results[index] = result

        return [result for result in results if result is not None]

    @staticmethod
    def _has_tool_results(messages: List[Dict[str, Any]]) -> bool:
        """Return whether the conversation already has tool results to summarize."""
        return any(message.get("role") == "tool" for message in messages)

    @classmethod
    def _tool_names_by_call_id(
        cls,
        messages: List[Dict[str, Any]],
    ) -> dict[str, str]:
        """Return tool names keyed by provider tool-call id."""
        tool_names: dict[str, str] = {}
        for message in messages:
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            for tool_call in message.get("tool_calls", []):
                tool_call_id = str(tool_call.get("id") or "")
                if tool_call_id:
                    tool_names[tool_call_id] = cls._tool_call_name(tool_call)
        return tool_names

    def _compact_tool_results_for_final_response(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return messages with tighter tool-result content for no-tools finalization."""
        tool_names_by_id = self._tool_names_by_call_id(messages)
        compacted_messages: list[dict[str, Any]] = []

        for message in messages:
            if message.get("role") != "tool":
                compacted_messages.append(message)
                continue

            compacted_message = dict(message)
            tool_call_id = str(compacted_message.get("tool_call_id") or "")
            tool_name = str(
                compacted_message.get("tool_name")
                or tool_names_by_id.get(tool_call_id)
                or "tool_result"
            )
            compacted_message["content"] = self._compact_tool_result_for_llm(
                tool_name,
                compacted_message.get("content"),
                max_chars=TOOL_BUDGET_FINAL_TOOL_RESULT_CHARS,
                max_lines=TOOL_BUDGET_FINAL_TOOL_RESULT_LINES,
            )
            compacted_messages.append(compacted_message)

        return compacted_messages

    @staticmethod
    def _is_toolless_check_preamble(response_text: str) -> bool:
        """Return whether a response only promises to check without using tools."""
        text = re.sub(r"\s+", " ", str(response_text or "")).strip()
        if not text or len(text) > 240 or "?" in text:
            return False
        if any(pattern.search(text) for pattern in TOOLLESS_RESPONSE_PATTERNS):
            return True
        normalized = text.casefold().strip(" \t\r\n.,!;:…。！？¡¿")
        return any(
            normalized.startswith(prefix)
            for prefix in TOOLLESS_RESPONSE_PREFIXES
        )

    def _should_retry_toolless_response(
        self,
        response_text: str,
        *,
        tools: List[Dict[str, Any]] | None,
        retry_used: bool,
        tool_calls_used: int = 0,
    ) -> bool:
        """Return whether to give the model one more chance to call a tool."""
        if retry_used or not tools or tool_calls_used > 0:
            return False
        return self._is_toolless_check_preamble(response_text)

    @staticmethod
    def _append_toolless_retry_messages(
        conversation_messages: List[Dict[str, Any]],
        response_text: str,
    ) -> None:
        """Append corrective context after a tool-less preamble response."""
        if response_text.strip():
            conversation_messages.append(
                {"role": "assistant", "content": response_text.strip()}
            )
        conversation_messages.append(
            {"role": "system", "content": TOOLLESS_RETRY_INSTRUCTION}
        )

    async def _call_llm_final_response_without_tools(
        self,
        conversation_messages: List[Dict[str, Any]],
        provider: LLMProvider,
        *,
        transport: str,
    ) -> str:
        """Ask the provider for a final answer with tools disabled."""
        return await self._call_llm_without_tools(
            conversation_messages,
            provider,
            transport=transport,
            final_instruction=TOOL_BUDGET_FINAL_INSTRUCTION,
            fallback_response=TOOL_BUDGET_FALLBACK_RESPONSE,
            compact_tool_results=True,
        )

    async def async_call_llm_without_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        transport: str = "direct_no_tools",
    ) -> str:
        """Call the active provider directly with tools disabled."""
        provider = self._get_llm_provider()
        return await self._call_llm_without_tools(
            messages,
            provider,
            transport=transport,
        )

    async def _call_llm_without_tools(
        self,
        conversation_messages: List[Dict[str, Any]],
        provider: LLMProvider,
        *,
        transport: str,
        final_instruction: str | None = None,
        fallback_response: str | None = None,
        compact_tool_results: bool = False,
    ) -> str:
        """Ask a provider for a no-tools response."""
        base_messages = (
            self._compact_tool_results_for_final_response(conversation_messages)
            if compact_tool_results
            else list(conversation_messages)
        )
        final_messages = list(base_messages)
        if final_instruction:
            final_messages.append({"role": "system", "content": final_instruction})

        payload = provider.build_payload(final_messages, [], stream=False)
        clean_payload = self._prepare_provider_payload(provider, payload)
        self._log_initial_llm_payload_metrics(
            transport=transport,
            iteration=-1,
            payload=clean_payload,
            messages=final_messages,
            tools=clean_payload.get("tools"),
        )

        try:
            response = await self._request_provider_http_response(
                provider,
                clean_payload,
                iteration=-1,
            )
            if response.status != 200:
                if fallback_response is None:
                    raise ValueError(
                        f"No-tools provider response failed with status {response.status}: "
                        f"{_provider_log_snippet(response.error_text)}"
                    )
                _LOGGER.warning(
                    "%s no-tools response failed: transport=%s status=%s body=%s",
                    self.server_type,
                    transport,
                    response.status,
                    _provider_log_snippet(response.error_text),
                )
                return fallback_response

            data = response.data or {}
            self._log_prompt_cache_usage(
                provider,
                data,
                transport=transport,
                iteration=-1,
            )
            message = provider.parse_http_message(data)
            final_content = str(message.get("content") or "").strip()
            if final_content or fallback_response is not None:
                return final_content or str(fallback_response)
            raise ValueError("No-tools provider response was empty")
        except Exception as err:
            if fallback_response is None:
                raise
            _LOGGER.warning(
                "%s no-tools response failed: transport=%s error=%s",
                self.server_type,
                transport,
                _exception_log_snippet(err),
            )
            return fallback_response

    async def _test_streaming_basic(self) -> bool:
        """Test basic streaming without tools to isolate connection issues."""
        provider = self._get_llm_provider()
        payload = provider.build_payload(
            [{"role": "user", "content": "Say hello"}],
            stream=True,
        )
        clean_payload = self._prepare_provider_payload(provider, payload)

        _LOGGER.info("🧪 Testing basic streaming to: %s", provider.chat_url())
        _LOGGER.info(f"🧪 Model: {self.model_name}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # This synthetic probe must not join the active Assist session.
                async with session.post(
                    provider.chat_url(),
                    headers=provider.headers(),
                    json=clean_payload,
                ) as response:
                    _LOGGER.info(
                        f"✅ Basic streaming connected! Status: {response.status}"
                    )
                    _LOGGER.info(
                        "📋 Header keys: %s",
                        _mapping_key_summary(dict(response.headers)),
                    )

                    # Try to read first few lines
                    line_count = 0
                    async for line in response.content:
                        line_length = len(line.decode("utf-8", errors="replace"))
                        _LOGGER.info(
                            "📨 Streaming probe line %d: %d chars",
                            line_count,
                            line_length,
                        )
                        line_count += 1
                        if line_count >= 3:
                            break

                    _LOGGER.info(
                        f"✅ Basic streaming works! Received {line_count} lines"
                    )
                    return True

        except aiohttp.ClientConnectionError as e:
            _LOGGER.debug("Streaming probe connection error: %s", e)
            return False
        except Exception as e:
            _LOGGER.debug(
                "Basic streaming probe failed: %s: %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return False

    async def _call_llm_streaming(self, messages: List[Dict[str, Any]]) -> str:
        """Stream LLM responses with immediate TTS feedback."""
        _LOGGER.info(f"🚀 Starting streaming {self.server_type} conversation")

        # Test streaming once and cache result
        if not hasattr(self, "_streaming_available"):
            self._streaming_available = await self._test_streaming_basic()

        if not self._streaming_available:
            _LOGGER.debug("Streaming not available; using provider HTTP transport")
            raise RecoverableStreamingFallbackError("Streaming not available")

        tools: list[dict[str, Any]] | None = None
        provider = self._get_llm_provider()
        session_scoped = "X-Session-Id" in self._provider_request_headers(provider)
        conversation_messages = list(messages)
        tool_calls_used = 0
        toolless_retry_used = False
        invalid_tool_retry_used = False
        empty_stream_responses = 0

        # Buffers for streaming
        tool_arg_buffers = {}  # index -> partial JSON string
        tool_names = {}  # index -> tool name
        tool_ids = {}  # index -> tool_call_id
        response_text = ""
        sentence_buffer = ""
        completed_tools = set()

        for iteration in range(self.model_turn_limit):
            _LOGGER.info(f"🔄 Stream iteration {iteration + 1}")
            tools = await self._get_mcp_tools()
            if not tools and iteration == 0:
                _LOGGER.warning("No MCP tools available - proceeding without tools")
            if self.debug_mode and iteration == 0:
                _LOGGER.info(f"🎯 Using model: {self.model_name}")

            # Debug logging for iteration 2+ if enabled
            if self.debug_mode and iteration >= 1:
                _LOGGER.info(
                    f"🔄 Iteration {iteration + 1}: {len(conversation_messages)} messages to send"
                )
                for i, msg in enumerate(conversation_messages):
                    role = msg.get("role")
                    has_tool_calls = "tool_calls" in msg
                    tool_call_id = msg.get("tool_call_id", "")
                    content = msg.get("content", "")
                    content_len = len(str(content)) if content else 0
                    _LOGGER.info(
                        "  Msg %d: %s, tool_calls=%s, tool_call_id=%s, content_chars=%d",
                        i,
                        role,
                        has_tool_calls,
                        tool_call_id,
                        content_len,
                    )

            cleaned_messages = provider.prepare_messages_for_stream(
                conversation_messages
            )
            payload = provider.build_payload(cleaned_messages, tools, stream=True)

            # Debug: Log actual cleaned payload being sent in iteration 2+
            if self.debug_mode and iteration >= 1:
                _LOGGER.info(
                    f"📤 Sending {len(cleaned_messages)} messages to LLM (iteration {iteration + 1}):"
                )
                _LOGGER.info(f"📤 Model: {self.model_name}")
                _LOGGER.info(f"📤 Temperature: {payload.get('temperature', 'default')}")
                _LOGGER.info(
                    f"📤 Max tokens: {payload.get('max_tokens', payload.get('max_completion_tokens', 'default'))}"
                )
                for i, msg in enumerate(cleaned_messages):
                    role = msg.get("role")
                    content = msg.get("content", "")
                    content_len = len(str(content)) if content else 0
                    _LOGGER.info("  [%d] %s: %d chars", i, role, content_len)

            clean_payload = self._prepare_provider_payload(provider, payload)

            self._log_initial_llm_payload_metrics(
                transport="streaming",
                iteration=iteration,
                payload=clean_payload,
                messages=cleaned_messages,
                tools=clean_payload.get("tools"),
            )

            has_tool_calls = False
            current_tool_calls = []
            stream_tool_index_offset = None
            stream_metadata = None
            request_dispatched = False
            request_rejected = False

            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    url = provider.chat_url()
                    headers = self._provider_request_headers(provider)

                    _LOGGER.info(f"📡 Streaming to: {url}")
                    if self.debug_mode:
                        _LOGGER.debug(
                            f"📦 Payload size: {len(json.dumps(clean_payload))} bytes"
                        )
                        _LOGGER.debug(f"🔧 Using model: {self.model_name}")

                    # Once a stateful request is dispatched, a transport failure may
                    # happen after the provider has recorded the turn. An explicit
                    # non-success response is a safe rejection and can still fall back.
                    request_context = session.post(
                        url, headers=headers, json=clean_payload
                    )
                    request_dispatched = True
                    async with request_context as response:
                        _LOGGER.info(
                            f"🔌 Connection established, status: {response.status}"
                        )
                        if self.debug_mode:
                            _LOGGER.debug(
                                "📋 Response header keys: %s",
                                _mapping_key_summary(dict(response.headers)),
                            )

                        if response.status != 200:
                            request_rejected = True
                            try:
                                error_data = await response.json()
                                error_text = json.dumps(error_data, indent=2)
                            except Exception:
                                error_text = await response.text()
                            if provider.is_invalid_tool_arguments_error(
                                status=response.status,
                                error_text=error_text,
                            ):
                                _LOGGER.info(
                                    "%s streaming reported malformed tool-call arguments on iteration %d",
                                    self.server_type,
                                    iteration + 1,
                                )
                                if self._has_tool_results(conversation_messages):
                                    return await self._call_llm_final_response_without_tools(
                                        conversation_messages,
                                        provider,
                                        transport="streaming_invalid_tool_final",
                                    )
                                if invalid_tool_retry_used:
                                    return INVALID_TOOL_ARGUMENT_FALLBACK_RESPONSE
                                self._append_invalid_tool_call_retry_messages(
                                    conversation_messages,
                                    response_text,
                                    [],
                                )
                                invalid_tool_retry_used = True
                                response_text = ""
                                sentence_buffer = ""
                                tool_arg_buffers.clear()
                                tool_names.clear()
                                tool_ids.clear()
                                completed_tools.clear()
                                continue

                            _LOGGER.warning(
                                "Streaming failed with status %s", response.status
                            )
                            _LOGGER.debug(
                                "Provider streaming error response: %s",
                                _provider_log_snippet(error_text),
                            )
                            raise Exception(
                                f"Streaming failed: {_provider_log_snippet(error_text)}"
                            )  # Raise to trigger fallback

                        if self.debug_mode:
                            _LOGGER.debug("📖 Starting to read stream...")

                        async for line in response.content:
                            if not line:
                                continue

                            line_str = line.decode("utf-8").strip()

                            try:
                                parsed_stream = provider.parse_stream_line(line_str)
                                if parsed_stream is None:
                                    continue
                                if parsed_stream.usage:
                                    self._log_prompt_cache_usage(
                                        provider,
                                        {"usage": parsed_stream.usage},
                                        transport="streaming",
                                        iteration=iteration,
                                    )
                                if parsed_stream.done:
                                    break

                                delta = parsed_stream.delta
                                previous_stream_metadata = stream_metadata
                                stream_metadata = provider.update_stream_metadata(
                                    stream_metadata,
                                    delta,
                                )
                                if (
                                    stream_metadata is not None
                                    and stream_metadata != previous_stream_metadata
                                ):
                                    _LOGGER.debug(
                                        "Captured provider stream metadata (%d chars)",
                                        len(str(stream_metadata)),
                                    )

                                # Handle streamed content
                                if "content" in delta and delta["content"]:
                                    empty_stream_responses = 0
                                    chunk = delta["content"]
                                    response_text += chunk
                                    sentence_buffer += chunk

                                    # Trigger TTS on complete sentence
                                    if any(
                                        sentence_buffer.endswith(p)
                                        for p in [". ", "! ", "? ", ".\n", "!\n", "?\n"]
                                    ):
                                        await self._trigger_tts(sentence_buffer.strip())
                                        sentence_buffer = ""

                                # Handle streamed tool calls
                                if "tool_calls" in delta:
                                    has_tool_calls = True
                                    for tc in delta["tool_calls"]:
                                        idx, stream_tool_index_offset = (
                                            self._ensure_stream_tool_call_slot(
                                                tc,
                                                current_tool_calls,
                                                stream_tool_index_offset,
                                            )
                                        )

                                        if "id" in tc:
                                            tool_ids[idx] = tc["id"]
                                            current_tool_calls[idx]["id"] = tc["id"]
                                            # Add the required type field
                                            current_tool_calls[idx]["type"] = "function"

                                        if "function" in tc:
                                            func = tc["function"]
                                            if "name" in func:
                                                tool_names[idx] = func["name"]
                                                if (
                                                    "function"
                                                    not in current_tool_calls[idx]
                                                ):
                                                    current_tool_calls[idx][
                                                        "function"
                                                    ] = {}
                                                current_tool_calls[idx]["function"][
                                                    "name"
                                                ] = func["name"]
                                                _LOGGER.info(
                                                    f"🔧 Tool streaming: {func['name']}"
                                                )

                                            if "arguments" in func:
                                                if (
                                                    "function"
                                                    not in current_tool_calls[idx]
                                                ):
                                                    current_tool_calls[idx][
                                                        "function"
                                                    ] = {}

                                                raw_arguments = func["arguments"]
                                                if isinstance(raw_arguments, dict):
                                                    current_tool_calls[idx]["function"][
                                                        "arguments"
                                                    ] = self._stringify_tool_arguments(
                                                        raw_arguments
                                                    )

                                                    tool_name = tool_names.get(idx)
                                                    if (
                                                        tool_name
                                                        and idx not in completed_tools
                                                    ):
                                                        completed_tools.add(idx)
                                                        if (
                                                            tool_name
                                                            == "discover_entities"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Looking for devices..."
                                                            )
                                                        elif (
                                                            tool_name
                                                            == "perform_action"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Controlling the device..."
                                                            )
                                                    continue

                                                if idx not in tool_arg_buffers:
                                                    tool_arg_buffers[idx] = ""
                                                tool_arg_buffers[idx] += (
                                                    self._stringify_tool_arguments(
                                                        raw_arguments
                                                    )
                                                )

                                                # Try to parse arguments
                                                try:
                                                    json.loads(
                                                        tool_arg_buffers[idx]
                                                    )
                                                    # Valid JSON - save it
                                                    if (
                                                        "function"
                                                        not in current_tool_calls[idx]
                                                    ):
                                                        current_tool_calls[idx][
                                                            "function"
                                                        ] = {}
                                                    current_tool_calls[idx]["function"][
                                                        "arguments"
                                                    ] = tool_arg_buffers[idx]

                                                    # Quick feedback for tool execution
                                                    tool_name = tool_names.get(idx)
                                                    if (
                                                        tool_name
                                                        and idx not in completed_tools
                                                    ):
                                                        completed_tools.add(idx)
                                                        if (
                                                            tool_name
                                                            == "discover_entities"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Looking for devices..."
                                                            )
                                                        elif (
                                                            tool_name
                                                            == "perform_action"
                                                        ):
                                                            await self._trigger_tts(
                                                                "Controlling the device..."
                                                            )

                                                except json.JSONDecodeError:
                                                    # Still accumulating arguments
                                                    pass

                            except Exception as e:
                                _LOGGER.debug(f"Stream parsing: {e}")

            except Exception as stream_error:
                _LOGGER.warning(
                    "Streaming iteration %d failed: %s",
                    iteration + 1,
                    stream_error,
                )
                if iteration == 0:
                    if request_dispatched and not request_rejected and session_scoped:
                        raise StatefulStreamingRequestError(
                            "Stateful streaming failed after the request was dispatched"
                        ) from stream_error
                    # An unscoped or explicitly rejected request can safely fall back.
                    raise stream_error
                else:
                    # Later iteration failed, return what we have
                    break

            # Handle any remaining sentence
            if sentence_buffer.strip():
                await self._trigger_tts(sentence_buffer.strip())
                sentence_buffer = ""

            current_tool_calls = self._compact_streamed_tool_calls(current_tool_calls)

            # If we got tool calls, execute them
            if has_tool_calls and current_tool_calls:
                empty_stream_responses = 0
                _LOGGER.info(
                    f"⚡ Executing {len(current_tool_calls)} streamed tool calls"
                )
                if self.debug_mode:
                    _LOGGER.debug(
                        f"📝 Discarding intermediate narration: {len(response_text)} chars"
                    )
                    _LOGGER.debug(
                        "📊 Tool calls structure: %s",
                        json.dumps(
                            self._tool_call_log_summary(current_tool_calls),
                            indent=2,
                        ),
                    )

                valid_tool_calls, invalid_tool_calls = self._partition_valid_tool_calls(
                    current_tool_calls
                )
                if invalid_tool_calls:
                    _LOGGER.info(
                        "Skipping %d streamed tool call(s) with invalid JSON "
                        "arguments: %s",
                        len(invalid_tool_calls),
                        self._invalid_tool_call_names(invalid_tool_calls),
                    )

                if not valid_tool_calls:
                    if invalid_tool_retry_used:
                        return INVALID_TOOL_ARGUMENT_FALLBACK_RESPONSE
                    self._append_invalid_tool_call_retry_messages(
                        conversation_messages,
                        response_text,
                        invalid_tool_calls,
                    )
                    invalid_tool_retry_used = True
                    response_text = ""
                    sentence_buffer = ""
                    tool_arg_buffers.clear()
                    tool_names.clear()
                    tool_ids.clear()
                    completed_tools.clear()
                    continue

                prepared_tool_calls = provider.prepare_stream_tool_calls(
                    valid_tool_calls,
                    stream_metadata,
                )
                warning = provider.missing_stream_metadata_warning(stream_metadata)
                if warning:
                    _LOGGER.warning(warning)

                assistant_msg = provider.build_tool_call_assistant_message(
                    prepared_tool_calls,
                    response_text=response_text,
                )
                conversation_messages.append(assistant_msg)

                # Record tool calls to ChatLog for debug view
                self._record_tool_calls_to_chatlog(valid_tool_calls)

                budget_plan = self._prepare_tool_calls_for_budget(
                    valid_tool_calls,
                    tool_calls_used=tool_calls_used,
                    provider=provider,
                )

                # Execute tools
                executable_tool_results = await self._execute_tool_calls(
                    budget_plan.executable_calls
                )
                tool_calls_used += self._count_budgeted_tool_calls(
                    budget_plan.executable_calls
                )
                tool_results = self._merge_tool_results_in_call_order(
                    budget_plan,
                    executable_tool_results,
                )

                # Record tool results to ChatLog for debug view
                for idx, result in enumerate(tool_results):
                    if idx < len(valid_tool_calls):
                        tc = valid_tool_calls[idx]
                        tool_call_id = result.get(
                            "tool_call_id", tc.get("id", "unknown")
                        )
                        tool_name = tc.get("function", {}).get("name", "unknown")
                        # Parse content as JSON if possible, otherwise use as-is
                        try:
                            tool_result_data = json.loads(result.get("content", "{}"))
                        except Exception:
                            tool_result_data = {"result": result.get("content", "")}
                        self._record_tool_result_to_chatlog(
                            tool_call_id, tool_name, tool_result_data
                        )

                conversation_messages.extend(tool_results)
                if invalid_tool_calls and not invalid_tool_retry_used:
                    self._append_invalid_tool_call_retry_messages(
                        conversation_messages,
                        "",
                        invalid_tool_calls,
                    )
                    invalid_tool_retry_used = True

                # Reset for next iteration - we don't want intermediate narration in final response
                response_text = (
                    ""  # Clear accumulated text since it was just pre-tool narration
                )
                tool_arg_buffers.clear()
                tool_names.clear()
                tool_ids.clear()
                completed_tools.clear()

                if budget_plan.exhausted or tool_calls_used >= self.tool_call_budget:
                    _LOGGER.info(
                        "Tool-call budget exhausted after %d/%d calls; requesting final answer without tools",
                        tool_calls_used,
                        self.tool_call_budget,
                    )
                    return await self._call_llm_final_response_without_tools(
                        conversation_messages,
                        provider,
                        transport="streaming_budget_final",
                    )

                # Continue to get next response after tools
                continue
            else:
                # No tool calls, return the response
                if response_text:
                    if self._should_retry_toolless_response(
                        response_text,
                        tools=tools,
                        retry_used=toolless_retry_used,
                        tool_calls_used=tool_calls_used,
                    ):
                        _LOGGER.info(
                            "Model returned a tool-less check preamble; retrying with corrective tool-use instruction"
                        )
                        self._append_toolless_retry_messages(
                            conversation_messages,
                            response_text,
                        )
                        toolless_retry_used = True
                        response_text = ""
                        sentence_buffer = ""
                        continue
                    return response_text
                else:
                    # No content and no tools, might need another iteration
                    empty_stream_responses += 1
                    if empty_stream_responses == 1:
                        if session_scoped:
                            raise StatefulStreamingRequestError(
                                "Stateful streaming returned an empty response"
                            )
                        _LOGGER.debug("Empty response from streaming, retrying once")
                        continue
                    if self._has_tool_results(conversation_messages):
                        return await self._call_llm_final_response_without_tools(
                            conversation_messages,
                            provider,
                            transport="streaming_empty_final",
                        )
                    raise EmptyStreamingResponseError(
                        "Streaming returned an empty response"
                    )

        # Hit the model-call guard before a final response arrived.
        if response_text:
            return response_text
        if self._has_tool_results(conversation_messages):
            return await self._call_llm_final_response_without_tools(
                conversation_messages,
                provider,
                transport="streaming_iteration_final",
            )
        else:
            return TOOL_BUDGET_FALLBACK_RESPONSE

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Call LLM API with MCP tools and handle tool execution loop."""
        provider = self._get_llm_provider()
        if not provider.supports_streaming:
            return await self._call_llm_http(messages, provider=provider)

        # Try streaming first, fallback to HTTP if needed
        try:
            return await self._call_llm_streaming(messages)
        except StatefulStreamingRequestError:
            raise
        except RecoverableStreamingFallbackError as e:
            _LOGGER.debug("%s; using provider HTTP transport", e)
            return await self._call_llm_http(messages, provider=provider)
        except Exception as e:
            _LOGGER.warning(
                "Streaming failed (%s), using provider HTTP transport",
                _exception_log_snippet(e),
            )
            return await self._call_llm_http(messages, provider=provider)

    async def _request_provider_http_response(
        self,
        provider: LLMProvider,
        clean_payload: dict[str, Any],
        *,
        iteration: int,
    ) -> ProviderHttpResponse:
        """POST a provider HTTP request with bounded retry for transport timeouts."""
        headers = self._provider_request_headers(provider)
        max_attempts = (
            1
            if "X-Session-Id" in headers
            else PROVIDER_HTTP_TIMEOUT_ATTEMPTS
        )
        for attempt in range(1, max_attempts + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        provider.chat_url(),
                        headers=headers,
                        json=clean_payload,
                    ) as response:
                        if response.status != 200:
                            return ProviderHttpResponse(
                                status=response.status,
                                error_text=await response.text(),
                            )

                        return ProviderHttpResponse(
                            status=response.status,
                            data=await response.json(),
                        )
            except asyncio.TimeoutError as err:
                if attempt < max_attempts:
                    _LOGGER.warning(
                        (
                            "%s HTTP transport timed out after %ss on "
                            "iteration %d; retrying once"
                        ),
                        self.server_type,
                        self.timeout,
                        iteration + 1,
                    )
                    continue

                raise ProviderResponseTimeoutError(
                    provider_name=self._get_server_display_name(),
                    timeout_seconds=self.timeout,
                    transport="HTTP",
                    attempts=attempt,
                    iteration=iteration + 1,
                ) from err

        raise ProviderResponseTimeoutError(
            provider_name=self._get_server_display_name(),
            timeout_seconds=self.timeout,
            transport="HTTP",
            attempts=max_attempts,
            iteration=iteration + 1,
        )

    async def _call_llm_http(
        self,
        messages: List[Dict[str, Any]],
        *,
        provider: LLMProvider | None = None,
    ) -> str:
        """Call the active provider with its non-streaming HTTP transport."""
        _LOGGER.info(f"🚀 Using provider HTTP transport for {self.server_type}")

        tools: list[dict[str, Any]] | None = None

        if provider is None:
            provider = self._get_llm_provider()

        # Keep a mutable copy of messages for the conversation
        conversation_messages = list(messages)
        tool_calls_used = 0
        toolless_retry_used = False
        invalid_tool_retry_used = False

        # Tool execution loop
        for iteration in range(self.model_turn_limit):
            _LOGGER.info(
                f"🔄 HTTP Iteration {iteration + 1}: Calling {self.server_type} with {len(conversation_messages)} messages"
            )
            tools = await self._get_mcp_tools()
            if not tools and iteration == 0:
                _LOGGER.warning("No MCP tools available - proceeding without tools")

            payload = provider.build_payload(conversation_messages, tools, stream=False)
            clean_payload = self._prepare_provider_payload(provider, payload)
            self._log_initial_llm_payload_metrics(
                transport="http",
                iteration=iteration,
                payload=clean_payload,
                messages=conversation_messages,
                tools=clean_payload.get("tools"),
            )

            http_response = await self._request_provider_http_response(
                provider,
                clean_payload,
                iteration=iteration,
            )
            if http_response.status != 200:
                error_text = http_response.error_text
                error_snippet = _provider_log_snippet(error_text)
                if provider.is_invalid_tool_arguments_error(
                    status=http_response.status,
                    error_text=error_text,
                ):
                    _LOGGER.info(
                        "%s HTTP transport reported malformed tool-call arguments on iteration %d",
                        self.server_type,
                        iteration + 1,
                    )
                    if self._has_tool_results(conversation_messages):
                        return await self._call_llm_final_response_without_tools(
                            conversation_messages,
                            provider,
                            transport="http_invalid_tool_final",
                        )
                    if invalid_tool_retry_used:
                        return INVALID_TOOL_ARGUMENT_FALLBACK_RESPONSE
                    self._append_invalid_tool_call_retry_messages(
                        conversation_messages,
                        "",
                        [],
                    )
                    invalid_tool_retry_used = True
                    continue

                _LOGGER.warning(
                    "%s API error: status=%s body=%s",
                    self.server_type,
                    http_response.status,
                    error_snippet,
                )
                raise Exception(
                    f"{self.server_type} API error {http_response.status}: {error_snippet}"
                )

            data = http_response.data or {}
            self._log_prompt_cache_usage(
                provider,
                data,
                transport="http",
                iteration=iteration,
            )
            message = provider.parse_http_message(data)

            # Check if there are tool calls to execute
            if "tool_calls" in message and message["tool_calls"]:
                valid_raw_tool_calls, invalid_tool_calls = (
                    self._partition_valid_tool_calls(message["tool_calls"])
                )
                if invalid_tool_calls:
                    _LOGGER.info(
                        "Skipping %d tool call(s) with invalid JSON arguments: %s",
                        len(invalid_tool_calls),
                        self._invalid_tool_call_names(invalid_tool_calls),
                    )

                if not valid_raw_tool_calls:
                    if invalid_tool_retry_used:
                        return INVALID_TOOL_ARGUMENT_FALLBACK_RESPONSE
                    self._append_invalid_tool_call_retry_messages(
                        conversation_messages,
                        str(message.get("content") or ""),
                        invalid_tool_calls,
                    )
                    invalid_tool_retry_used = True
                    continue

                tool_calls = provider.normalize_tool_calls(valid_raw_tool_calls)
                _LOGGER.info(
                    f"🛠️ {self.server_type} requested {len(tool_calls)} tool calls"
                )

                # Ensure each tool_call has the required type field
                for tc in tool_calls:
                    if "type" not in tc:
                        tc["type"] = "function"
                    if "function" in tc:
                        _LOGGER.debug(
                            "  - %s: %s",
                            tc["function"].get("name"),
                            _provider_log_snippet(tc["function"].get("arguments")),
                        )

                assistant_msg = provider.build_tool_call_assistant_message(
                    tool_calls,
                    response_text=str(message.get("content") or ""),
                )
                conversation_messages.append(assistant_msg)

                # Record tool calls to ChatLog for debug view
                self._record_tool_calls_to_chatlog(tool_calls)

                budget_plan = self._prepare_tool_calls_for_budget(
                    tool_calls,
                    tool_calls_used=tool_calls_used,
                    provider=provider,
                )

                # Execute the tool calls
                _LOGGER.info("⚡ Executing tool calls against MCP server...")
                executable_tool_results = await self._execute_tool_calls(
                    budget_plan.executable_calls
                )
                tool_calls_used += self._count_budgeted_tool_calls(
                    budget_plan.executable_calls
                )
                tool_results = self._merge_tool_results_in_call_order(
                    budget_plan,
                    executable_tool_results,
                )

                # Record tool results to ChatLog for debug view
                for idx, result in enumerate(tool_results):
                    if idx < len(tool_calls):
                        tc = tool_calls[idx]
                        tool_call_id = result.get(
                            "tool_call_id", tc.get("id", "unknown")
                        )
                        tool_name = tc.get("function", {}).get("name", "unknown")
                        try:
                            tool_result_data = json.loads(result.get("content", "{}"))
                        except Exception:
                            tool_result_data = {"result": result.get("content", "")}
                        self._record_tool_result_to_chatlog(
                            tool_call_id, tool_name, tool_result_data
                        )

                # Add tool results to conversation
                conversation_messages.extend(tool_results)
                if invalid_tool_calls and not invalid_tool_retry_used:
                    self._append_invalid_tool_call_retry_messages(
                        conversation_messages,
                        "",
                        invalid_tool_calls,
                    )
                    invalid_tool_retry_used = True

                _LOGGER.info(f"📊 Added {len(tool_results)} tool results to conversation")

                if budget_plan.exhausted or tool_calls_used >= self.tool_call_budget:
                    _LOGGER.info(
                        "Tool-call budget exhausted after %d/%d calls; requesting final answer without tools",
                        tool_calls_used,
                        self.tool_call_budget,
                    )
                    return await self._call_llm_final_response_without_tools(
                        conversation_messages,
                        provider,
                        transport="http_budget_final",
                    )

                # Continue the loop to get next response
                continue

            # No more tool calls, we have the final response
            final_content = str(message.get("content") or "").strip()
            _LOGGER.info(f"💬 Final response received (length: {len(final_content)})")
            if self._should_retry_toolless_response(
                final_content,
                tools=tools,
                retry_used=toolless_retry_used,
                tool_calls_used=tool_calls_used,
            ):
                _LOGGER.info(
                    "Model returned a tool-less check preamble; retrying with corrective tool-use instruction"
                )
                self._append_toolless_retry_messages(
                    conversation_messages,
                    final_content,
                )
                toolless_retry_used = True
                continue
            return final_content

        # If we hit the model-call guard, return what we have.
        _LOGGER.warning(
            "⚠️ Hit maximum model turns (%d) around tool-call budget (%d)",
            self.model_turn_limit,
            self.tool_call_budget,
        )
        if self._has_tool_results(conversation_messages):
            return await self._call_llm_final_response_without_tools(
                conversation_messages,
                provider,
                transport="http_iteration_final",
            )
        return TOOL_BUDGET_FALLBACK_RESPONSE

    async def _execute_actions(
        self, response_text: str, user_input: ConversationInput
    ) -> List[Dict[str, Any]]:
        """Parse response for any action information.

        NOTE: With MCP tools, the model requests actions directly via the MCP server.
        We don't need to parse intents or execute them; this only reports what happened.
        """
        actions_taken = []

        # MCP tools execute actions directly, so this only logs what was mentioned.
        # The actual actions have already been performed via MCP's perform_action tool

        _LOGGER.info(
            "MCP-enabled response completed. Actions were executed via MCP tools if needed."
        )

        # We could parse the response to extract what was done for logging purposes
        # but the actual execution happens through MCP, not here

        if (
            "turned on" in response_text.lower()
            or "turning on" in response_text.lower()
        ):
            actions_taken.append(
                {"type": "mcp_action", "description": "Turned on devices via MCP"}
            )
        elif (
            "turned off" in response_text.lower()
            or "turning off" in response_text.lower()
        ):
            actions_taken.append(
                {"type": "mcp_action", "description": "Turned off devices via MCP"}
            )
        elif "toggled" in response_text.lower():
            actions_taken.append(
                {"type": "mcp_action", "description": "Toggled devices via MCP"}
            )

        return actions_taken
