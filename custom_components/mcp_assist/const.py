"""Constants for the MCP Assist integration."""

DOMAIN = "mcp_assist"
SYSTEM_ENTRY_UNIQUE_ID = "mcp_assist_system_settings"

# Server type options
SERVER_TYPE_LMSTUDIO = "lmstudio"
SERVER_TYPE_LLAMACPP = "llamacpp"
SERVER_TYPE_OLLAMA = "ollama"
SERVER_TYPE_OPENAI = "openai"
SERVER_TYPE_GEMINI = "gemini"
SERVER_TYPE_ANTHROPIC = "anthropic"
SERVER_TYPE_OPENROUTER = "openrouter"
SERVER_TYPE_OPENCLAW = "openclaw"
SERVER_TYPE_VLLM = "vllm"

# Configuration keys
CONF_PROFILE_NAME = "profile_name"
CONF_SERVER_TYPE = "server_type"
CONF_API_KEY = "api_key"
CONF_LMSTUDIO_URL = "lmstudio_url"
CONF_MODEL_NAME = "model_name"
CONF_MCP_PORT = "mcp_port"
CONF_AUTO_START = "auto_start"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TECHNICAL_PROMPT = "technical_prompt"
CONF_SYSTEM_PROMPT_MODE = "system_prompt_mode"
CONF_TECHNICAL_PROMPT_MODE = "technical_prompt_mode"
CONF_CONTROL_HA = "control_home_assistant"
CONF_RESPONSE_MODE = "response_mode"
CONF_FOLLOW_UP_MODE = "follow_up_mode"  # Keep for backward compatibility
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_MAX_HISTORY = "max_history"
CONF_CONTEXT_MODE = "context_mode"
CONF_MAX_ITERATIONS = "max_iterations"
CONF_DEBUG_MODE = "debug_mode"
CONF_CHAT_LOG_MODE = "chat_log_mode"
CONF_ENABLE_CUSTOM_TOOLS = "enable_custom_tools"
CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS = "enable_external_custom_tools"
CONF_BRAVE_API_KEY = "brave_api_key"
CONF_GOOGLE_MAPS_API_KEY = "google_maps_api_key"
CONF_SEARXNG_URL = "searxng_url"
CONF_ALLOWED_IPS = "allowed_ips"
CONF_INCLUDE_CURRENT_USER = "include_current_user"
CONF_INCLUDE_HOME_LOCATION = "include_home_location"
CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS = "include_current_user_in_tool_calls"
CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS = "include_home_location_in_tool_calls"
CONF_SEARCH_PROVIDER = "search_provider"
CONF_ENABLE_WEB_SEARCH = "enable_web_search"
CONF_ENABLE_GAP_FILLING = "enable_gap_filling"
CONF_ENABLE_ASSIST_BRIDGE = "enable_assist_bridge"
CONF_ENABLE_LLM_API_BRIDGE = "enable_llm_api_bridge"
CONF_LLM_API_ALLOWLIST = "llm_api_allowlist"
CONF_ENABLE_RESPONSE_SERVICE_TOOLS = "enable_response_service_tools"
CONF_ENABLE_WEATHER_FORECAST_TOOL = "enable_weather_forecast_tool"
CONF_ENABLE_RECORDER_TOOLS = "enable_recorder_tools"
CONF_ENABLE_MEMORY_TOOLS = "enable_memory_tools"
CONF_ENABLE_CALCULATOR_TOOLS = "enable_calculator_tools"
CONF_ENABLE_UNIT_CONVERSION_TOOLS = "enable_unit_conversion_tools"
CONF_ENABLE_DEVICE_TOOLS = "enable_device_tools"
CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT = "enable_music_assistant_support"
CONF_MEMORY_DEFAULT_TTL_DAYS = "memory_default_ttl_days"
CONF_MEMORY_MAX_TTL_DAYS = "memory_max_ttl_days"
CONF_MEMORY_MAX_ITEMS = "memory_max_items"
CONF_PROFILE_ENABLE_WEB_SEARCH = "profile_enable_web_search"
CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS = "profile_enable_external_custom_tools"
CONF_PROFILE_ENABLE_ASSIST_BRIDGE = "profile_enable_assist_bridge"
CONF_PROFILE_ENABLE_LLM_API_BRIDGE = "profile_enable_llm_api_bridge"
CONF_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS = "profile_enable_response_service_tools"
CONF_PROFILE_ENABLE_WEATHER_FORECAST_TOOL = "profile_enable_weather_forecast_tool"
CONF_PROFILE_ENABLE_RECORDER_TOOLS = "profile_enable_recorder_tools"
CONF_PROFILE_ENABLE_MEMORY_TOOLS = "profile_enable_memory_tools"
CONF_PROFILE_ENABLE_CALCULATOR_TOOLS = "profile_enable_calculator_tools"
CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS = "profile_enable_unit_conversion_tools"
CONF_PROFILE_ENABLE_DEVICE_TOOLS = "profile_enable_device_tools"
CONF_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT = "profile_enable_music_assistant_support"
CONF_OLLAMA_KEEP_ALIVE = "ollama_keep_alive"
CONF_OLLAMA_NUM_CTX = "ollama_num_ctx"
CONF_FOLLOW_UP_PHRASES = "follow_up_phrases"
CONF_END_WORDS = "end_words"
CONF_CLEAN_RESPONSES = "clean_responses"
CONF_TIMEOUT = "timeout"

# Default values
DEFAULT_SERVER_TYPE = "lmstudio"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"
DEFAULT_LLAMACPP_URL = "http://localhost:8080"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# OpenClaw Gateway defaults
CONF_OPENCLAW_HOST = "openclaw_host"
CONF_OPENCLAW_PORT = "openclaw_port"
CONF_OPENCLAW_TOKEN = "openclaw_token"
CONF_OPENCLAW_USE_SSL = "openclaw_use_ssl"
CONF_OPENCLAW_SESSION_KEY = "openclaw_session_key"
DEFAULT_OPENCLAW_HOST = "localhost"
DEFAULT_OPENCLAW_PORT = 18789
DEFAULT_OPENCLAW_USE_SSL = True
DEFAULT_OPENCLAW_SESSION_KEY = "main"
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_MCP_PORT = 8090
DEFAULT_API_KEY = ""

# Cloud provider base URLs
OPENAI_BASE_URL = "https://api.openai.com"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

# No hardcoded model lists - models are fetched dynamically from provider APIs
DEFAULT_MODEL_NAME = "model"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful Home Assistant voice assistant. For Home Assistant state "
    "or control requests, use MCP tools before replying. Respond naturally and "
    "conversationally to user requests."
)
PROMPT_MODE_DEFAULT = "default"
PROMPT_MODE_CUSTOM = "custom"
CONTEXT_MODE_STANDARD = "standard"
CONTEXT_MODE_ADAPTIVE = "adaptive"
CONTEXT_MODE_LIGHT = "light"
DEFAULT_SYSTEM_PROMPT_MODE = PROMPT_MODE_DEFAULT
DEFAULT_TECHNICAL_PROMPT_MODE = PROMPT_MODE_DEFAULT
DEFAULT_CONTEXT_MODE = CONTEXT_MODE_ADAPTIVE
DEFAULT_CONTROL_HA = True
DEFAULT_RESPONSE_MODE = "default"
DEFAULT_FOLLOW_UP_MODE = "default"  # Keep for backward compatibility
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 500
DEFAULT_MAX_HISTORY = 10
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_DEBUG_MODE = False
DEFAULT_CHAT_LOG_MODE = False
DEFAULT_CHAT_LOG_MAX_ENTRIES = 50
DEFAULT_ENABLE_CUSTOM_TOOLS = False
DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS = False
DEFAULT_BRAVE_API_KEY = ""
DEFAULT_GOOGLE_MAPS_API_KEY = ""
DEFAULT_SEARXNG_URL = ""
DEFAULT_ALLOWED_IPS = ""
DEFAULT_INCLUDE_CURRENT_USER = True
DEFAULT_INCLUDE_HOME_LOCATION = True
DEFAULT_INCLUDE_CURRENT_USER_IN_TOOL_CALLS = False
DEFAULT_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS = False
DEFAULT_SEARCH_PROVIDER = "none"
DEFAULT_ENABLE_WEB_SEARCH = False
DEFAULT_ENABLE_GAP_FILLING = True
DEFAULT_ENABLE_ASSIST_BRIDGE = False
DEFAULT_ENABLE_LLM_API_BRIDGE = False
DEFAULT_LLM_API_ALLOWLIST = ""
DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS = True
DEFAULT_ENABLE_WEATHER_FORECAST_TOOL = True
DEFAULT_ENABLE_RECORDER_TOOLS = True
DEFAULT_ENABLE_MEMORY_TOOLS = False
DEFAULT_ENABLE_CALCULATOR_TOOLS = False
DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS = False
DEFAULT_ENABLE_DEVICE_TOOLS = True
DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT = False
DEFAULT_MEMORY_DEFAULT_TTL_DAYS = 30
DEFAULT_MEMORY_MAX_TTL_DAYS = 365
DEFAULT_MEMORY_MAX_ITEMS = 500
DEFAULT_PROFILE_ENABLE_WEB_SEARCH = True
DEFAULT_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS = True
DEFAULT_PROFILE_ENABLE_ASSIST_BRIDGE = True
DEFAULT_PROFILE_ENABLE_LLM_API_BRIDGE = True
DEFAULT_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS = True
DEFAULT_PROFILE_ENABLE_WEATHER_FORECAST_TOOL = True
DEFAULT_PROFILE_ENABLE_RECORDER_TOOLS = True
DEFAULT_PROFILE_ENABLE_MEMORY_TOOLS = True
DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS = True
DEFAULT_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS = True
DEFAULT_PROFILE_ENABLE_DEVICE_TOOLS = True
DEFAULT_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT = True
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"  # 5 minutes
DEFAULT_OLLAMA_NUM_CTX = 0  # 0 = use model default
LIGHT_CONTEXT_MAX_HISTORY = 2
DEFAULT_FOLLOW_UP_PHRASES = (
    "anything else, what else, would you, do you, should i, can i, which, "
    "how can, what about, is there"
)
DEFAULT_END_WORDS = (
    "stop, cancel, no, nope, thanks, thank you, bye, goodbye, done, never mind, "
    "nevermind, forget it, that's all, that's it"
)
DEFAULT_CLEAN_RESPONSES = False
DEFAULT_TIMEOUT = 30

CUSTOM_TOOLS_DIRECTORY = "mcp-assist-tools"
CUSTOM_TOOL_SHARED_DIRECTORY = "__shared__"
CUSTOM_TOOL_SETTINGS_DIRECTORY = "mcp-assist-tool-settings"
CUSTOM_TOOL_SCHEMA_VERSION = 1
CUSTOM_TOOL_MANIFEST_FILENAME = "mcp_tool.json"
SERVICE_RELOAD_EXTERNAL_CUSTOM_TOOLS = "reload_external_custom_tools"
SERVICE_GET_CHAT_LOGS = "get_chat_logs"
SERVICE_CLEAR_CHAT_LOGS = "clear_chat_logs"

TOOL_FAMILY_DEVICE = "device"
TOOL_FAMILY_EXTERNAL_CUSTOM = "external_custom"
TOOL_FAMILY_ASSIST_BRIDGE = "assist_bridge"
TOOL_FAMILY_LLM_API_BRIDGE = "llm_api_bridge"
TOOL_FAMILY_RESPONSE_SERVICE = "response_service"
TOOL_FAMILY_WEATHER_FORECAST = "weather_forecast"
TOOL_FAMILY_RECORDER = "recorder"
TOOL_FAMILY_MEMORY = "memory"
TOOL_FAMILY_CALCULATOR = "calculator"
TOOL_FAMILY_UNIT_CONVERSION = "unit_conversion"
TOOL_FAMILY_MUSIC_ASSISTANT = "music_assistant"
TOOL_FAMILY_WEB_SEARCH = "web_search"

OPTIONAL_TOOL_FAMILY_TOOL_NAMES = {
    TOOL_FAMILY_DEVICE: frozenset({"discover_devices", "get_device_details"}),
    TOOL_FAMILY_ASSIST_BRIDGE: frozenset(
        {
            "list_assist_tools",
            "call_assist_tool",
            "get_assist_prompt",
            "get_assist_context_snapshot",
        }
    ),
    TOOL_FAMILY_LLM_API_BRIDGE: frozenset(
        {
            "list_llm_apis",
            "list_llm_api_tools",
            "call_llm_api_tool",
            "get_llm_api_prompt",
        }
    ),
    TOOL_FAMILY_RESPONSE_SERVICE: frozenset(
        {
            "get_calendar_events",
            "list_response_services",
            "call_service_with_response",
        }
    ),
    TOOL_FAMILY_WEATHER_FORECAST: frozenset({"get_weather_forecast"}),
    TOOL_FAMILY_RECORDER: frozenset(
        {
            "get_entity_history",
            "get_last_entity_event",
            "analyze_entity_history",
            "get_entity_state_at_time",
        }
    ),
    TOOL_FAMILY_MEMORY: frozenset(
        {
            "list_memory_categories",
            "remember_memory",
            "recall_memories",
            "forget_memory",
        }
    ),
    TOOL_FAMILY_CALCULATOR: frozenset(
        {
            "add",
            "subtract",
            "multiply",
            "divide",
            "sqrt",
            "power",
            "round_number",
            "average",
            "min_value",
            "max_value",
            "evaluate_expression",
        }
    ),
    TOOL_FAMILY_UNIT_CONVERSION: frozenset({"convert_unit"}),
    TOOL_FAMILY_MUSIC_ASSISTANT: frozenset(
        {
            "list_music_assistant_players",
            "play_music_assistant",
            "list_music_assistant_instances",
            "search_music_assistant",
            "get_music_assistant_library",
            "get_music_assistant_queue",
            "control_music_assistant_player",
            "transfer_music_assistant_queue",
        }
    ),
    TOOL_FAMILY_WEB_SEARCH: frozenset({"search", "read_url"}),
}

OPTIONAL_TOOL_NAME_TO_FAMILY = {
    tool_name: family
    for family, tool_names in OPTIONAL_TOOL_FAMILY_TOOL_NAMES.items()
    for tool_name in tool_names
}

LIGHT_CONTEXT_TOOL_NAMES = frozenset(
    {
        "discover_entities",
        "get_entity_details",
        "list_areas",
        "list_domains",
        "get_index",
        "perform_action",
        "set_conversation_state",
        "run_script",
        "run_automation",
        "discover_devices",
        "get_device_details",
    }
)

TOOL_FAMILY_SHARED_SETTINGS = {
    TOOL_FAMILY_DEVICE: (
        CONF_ENABLE_DEVICE_TOOLS,
        DEFAULT_ENABLE_DEVICE_TOOLS,
    ),
    TOOL_FAMILY_EXTERNAL_CUSTOM: (
        CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
        DEFAULT_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    ),
    TOOL_FAMILY_ASSIST_BRIDGE: (
        CONF_ENABLE_ASSIST_BRIDGE,
        DEFAULT_ENABLE_ASSIST_BRIDGE,
    ),
    TOOL_FAMILY_LLM_API_BRIDGE: (
        CONF_ENABLE_LLM_API_BRIDGE,
        DEFAULT_ENABLE_LLM_API_BRIDGE,
    ),
    TOOL_FAMILY_RESPONSE_SERVICE: (
        CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
        DEFAULT_ENABLE_RESPONSE_SERVICE_TOOLS,
    ),
    TOOL_FAMILY_RECORDER: (
        CONF_ENABLE_RECORDER_TOOLS,
        DEFAULT_ENABLE_RECORDER_TOOLS,
    ),
    TOOL_FAMILY_MEMORY: (
        CONF_ENABLE_MEMORY_TOOLS,
        DEFAULT_ENABLE_MEMORY_TOOLS,
    ),
    TOOL_FAMILY_WEATHER_FORECAST: (
        CONF_ENABLE_WEATHER_FORECAST_TOOL,
        DEFAULT_ENABLE_WEATHER_FORECAST_TOOL,
    ),
    TOOL_FAMILY_CALCULATOR: (
        CONF_ENABLE_CALCULATOR_TOOLS,
        DEFAULT_ENABLE_CALCULATOR_TOOLS,
    ),
    TOOL_FAMILY_UNIT_CONVERSION: (
        CONF_ENABLE_UNIT_CONVERSION_TOOLS,
        DEFAULT_ENABLE_UNIT_CONVERSION_TOOLS,
    ),
    TOOL_FAMILY_MUSIC_ASSISTANT: (
        CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
        DEFAULT_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    ),
    TOOL_FAMILY_WEB_SEARCH: (
        CONF_ENABLE_WEB_SEARCH,
        DEFAULT_ENABLE_WEB_SEARCH,
    ),
}

TOOL_FAMILY_PROFILE_SETTINGS = {
    TOOL_FAMILY_DEVICE: (
        CONF_PROFILE_ENABLE_DEVICE_TOOLS,
        DEFAULT_PROFILE_ENABLE_DEVICE_TOOLS,
    ),
    TOOL_FAMILY_EXTERNAL_CUSTOM: (
        CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
        DEFAULT_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    ),
    TOOL_FAMILY_ASSIST_BRIDGE: (
        CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
        DEFAULT_PROFILE_ENABLE_ASSIST_BRIDGE,
    ),
    TOOL_FAMILY_LLM_API_BRIDGE: (
        CONF_PROFILE_ENABLE_LLM_API_BRIDGE,
        DEFAULT_PROFILE_ENABLE_LLM_API_BRIDGE,
    ),
    TOOL_FAMILY_RESPONSE_SERVICE: (
        CONF_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS,
        DEFAULT_PROFILE_ENABLE_RESPONSE_SERVICE_TOOLS,
    ),
    TOOL_FAMILY_WEATHER_FORECAST: (
        CONF_PROFILE_ENABLE_WEATHER_FORECAST_TOOL,
        DEFAULT_PROFILE_ENABLE_WEATHER_FORECAST_TOOL,
    ),
    TOOL_FAMILY_RECORDER: (
        CONF_PROFILE_ENABLE_RECORDER_TOOLS,
        DEFAULT_PROFILE_ENABLE_RECORDER_TOOLS,
    ),
    TOOL_FAMILY_MEMORY: (
        CONF_PROFILE_ENABLE_MEMORY_TOOLS,
        DEFAULT_PROFILE_ENABLE_MEMORY_TOOLS,
    ),
    TOOL_FAMILY_CALCULATOR: (
        CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
        DEFAULT_PROFILE_ENABLE_CALCULATOR_TOOLS,
    ),
    TOOL_FAMILY_UNIT_CONVERSION: (
        CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
        DEFAULT_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    ),
    TOOL_FAMILY_MUSIC_ASSISTANT: (
        CONF_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT,
        DEFAULT_PROFILE_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    ),
    TOOL_FAMILY_WEB_SEARCH: (
        CONF_PROFILE_ENABLE_WEB_SEARCH,
        DEFAULT_PROFILE_ENABLE_WEB_SEARCH,
    ),
}


def get_optional_tool_family(tool_name: str) -> str | None:
    """Return the optional tool family for a tool name, if any."""
    return OPTIONAL_TOOL_NAME_TO_FAMILY.get(tool_name)


def parse_llm_api_allowlist(value: object) -> tuple[str, ...]:
    """Normalize a comma/newline-separated LLM API allowlist."""
    seen: set[str] = set()
    allowed_api_ids: list[str] = []
    for item in str(value or "").replace("\n", ",").split(","):
        api_id = item.strip()
        if not api_id or api_id in seen:
            continue
        seen.add(api_id)
        allowed_api_ids.append(api_id)
    return tuple(allowed_api_ids)

# MCP Server settings
MCP_SERVER_NAME = "ha-entity-discovery"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Entity discovery limits
MAX_ENTITIES_PER_DISCOVERY = 50  # Default, can be overridden in system settings
MAX_DISCOVERY_RESULTS = 100
CONF_MAX_ENTITIES_PER_DISCOVERY = "max_entities_per_discovery"
DEFAULT_MAX_ENTITIES_PER_DISCOVERY = 50

RESPONSE_MODE_INSTRUCTIONS = {
    "none": (
        "Follow-up behavior: do not ask follow-up questions. Complete the task and end."
    ),
    "default": (
        "Follow-up behavior: ask a short, specific follow-up only when it is "
        "genuinely helpful. If you ask one, call set_conversation_state. "
        "Otherwise end after completing the task."
    ),
    "always": (
        "Follow-up behavior: usually ask a short, specific follow-up after the "
        "task. If you ask one, call set_conversation_state. End naturally when "
        "the user indicates they are done."
    ),
}

DEFAULT_TECHNICAL_PROMPT = """You control a Home Assistant smart home through MCP tools.

Rules:
- Never invent entity IDs or claim an action happened unless a tool confirmed it.
- For Home Assistant tasks, discover the target first.
- For direct read-only checks or simple control requests, call the needed tools in the same turn. Do not ask the user to confirm that you should look something up.
- Do not reply only with a plan or promise to check; make the tool call before answering.
- Ask a follow-up only when the target or requested action is genuinely ambiguous, risky, or unsafe.
- Prefer entity-first control. Use device tools only when physical-device context matters.
- Floors, labels, and aliases are valid discovery inputs.
- Treat the tool-call budget as limited. Prefer one specific discovery call with filters such as domain, device_class, area, floor, label, or state before broader searches.
- Call get_index() only when you need a broad system overview.

Core workflow:
1. discover_entities(...) for most requests.
2. perform_action(...) for changes.
3. get_entity_details(...) when exact state or attributes matter.
4. run_script(...) for scripts with return data.
5. run_automation(...) for manual automation triggering.
6. Discover calendar or todo entities first, then use perform_action(...) for supported writes.

Responses:
- Keep replies short and plain text.
- Use friendly names, not raw entity IDs.
- For time answers, prefer relative time plus local absolute time when available.
- When listing multiple items, group by area when possible, otherwise use a stable order.

{{ response_mode }}

{{ current_user_context }}
Current area: {{ current_area }}
{{ home_location_context }}
Current time: {{ time }}
Current date: {{ date }}"""

DEVICE_TECHNICAL_INSTRUCTIONS = """
Device tools are enabled.
- Use discover_devices / get_device_details when the user means a physical device or you need related entities on the same device.
- Prefer discover_entities for most direct control.
"""

MEMORY_TECHNICAL_INSTRUCTIONS = """
Memory tools are enabled.
- Use remember_memory only when the user explicitly asks you to remember something.
- Use list_memory_categories when choosing or explaining memory categories.
- Prefer stable categories such as preference, routine, device_alias, automation_note, baseline, correction, maintenance, and household.
- Use recall_memories for stored facts or preferences.
- Use forget_memory when the user asks to remove or update stored memory.
"""

ASSIST_BRIDGE_TECHNICAL_INSTRUCTIONS = """
Assist bridge tools are enabled.
- Use list_assist_tools / call_assist_tool only as fallback or debugging.
- Prefer MCP Assist discovery and control tools first.
"""

LLM_API_BRIDGE_TECHNICAL_INSTRUCTIONS = """
Third-party Home Assistant LLM API bridge tools are enabled.
- Use list_llm_apis to inspect allowlisted third-party LLM APIs.
- Use list_llm_api_tools before call_llm_api_tool so arguments match that API's schema.
- Prefer MCP Assist native tools first for common Home Assistant discovery and control.
"""

CALCULATOR_TECHNICAL_INSTRUCTIONS = """
Calculator tools are enabled.
- Use calculator tools for exact arithmetic and compound expressions instead of mental math.
"""

UNIT_CONVERSION_TECHNICAL_INSTRUCTIONS = """
Unit conversion tools are enabled.
- Use convert_unit for temperatures, measurements, data sizes, rates, and other exact unit conversions.
"""

MUSIC_ASSISTANT_TECHNICAL_INSTRUCTIONS = """
Music Assistant support is enabled.
- Prefer Music Assistant tools for playback, playback control, target discovery, search, library, and queue questions.
- Only target Music Assistant players, not arbitrary media_player entities.
- If no target is given and the current area is known, use area="{{ current_area }}".
"""
