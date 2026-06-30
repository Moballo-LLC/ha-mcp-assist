"""Tests for profile-specific tool gating in the conversation agent."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist import agent as agent_module
from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.const import (
    CONF_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_CLEAN_RESPONSES,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_MAX_HISTORY,
    CONF_CONTEXT_MODE,
    CONF_MAX_ITERATIONS,
    CONF_MCP_BEARER_TOKEN,
    CONF_MAX_TOKENS,
    CONF_MODEL_NAME,
    CONF_PROFILE_NAME,
    CONF_SERVER_TYPE,
    CONF_SYSTEM_PROMPT,
    CONF_SYSTEM_PROMPT_MODE,
    CONF_TECHNICAL_PROMPT,
    CONF_TECHNICAL_PROMPT_MODE,
    CONF_PROFILE_ENABLE_ASSIST_BRIDGE,
    CONF_PROFILE_ENABLE_CALCULATOR_TOOLS,
    CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_PROFILE_ENABLE_LLM_API_BRIDGE,
    CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_PROFILE_ENABLE_WEB_SEARCH,
    CONF_CHAT_LOG_MODE,
    CONF_DEBUG_MODE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TECHNICAL_PROMPT,
    DOMAIN,
    CONTEXT_MODE_ADAPTIVE,
    CONTEXT_MODE_LIGHT,
    CONTEXT_MODE_STANDARD,
    DEFAULT_CONTEXT_MODE,
    PROMPT_MODE_CUSTOM,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_ANTHROPIC,
)
from custom_components.mcp_assist.tool_schema import (
    ADAPTIVE_TOOL_CATALOG_NAME,
    ADAPTIVE_TOOL_SCHEMA_NAME,
    match_adaptive_tool_definitions,
    normalize_adaptive_query_terms,
    score_adaptive_tool_match,
)

BUILTIN_SPECS = load_builtin_tool_toggle_specs()


def _builtin_spec(tool_name: str):
    """Return the built-in spec that declares a tool name."""
    for spec in BUILTIN_SPECS:
        if tool_name in spec.tool_names:
            return spec
    raise AssertionError(f"Missing built-in spec for {tool_name}")


def _tool(name: str) -> dict[str, object]:
    """Build a minimal MCP tool definition."""
    return {
        "name": name,
        "description": name,
        "inputSchema": {"type": "object", "properties": {}},
    }


def test_provider_log_snippet_redacts_and_truncates_details() -> None:
    """Provider details written to logs should be compact and secret-safe."""
    snippet = agent_module._provider_log_snippet(
        'first line\n{"api_key":"secret-value","Authorization":"Bearer sk-leaked-value",'
        '"error":"Incorrect API key: sk-live-secret1234567890",'
        '"details":"API key provided: sk-prose-secret",'
        '"google":"AIzaSyExampleKeyValue1234567890",'
        '"local":"API key: my-local-secret",'
        '"message":"'
        + ("x" * 80)
        + '"}',
        max_chars=120,
    )

    assert "\n" not in snippet
    assert "secret-value" not in snippet
    assert "sk-leaked-value" not in snippet
    assert "sk-live-secret1234567890" not in snippet
    assert "sk-prose-secret" not in snippet
    assert "AIzaSyExampleKeyValue1234567890" not in snippet
    assert "my-local-secret" not in snippet
    assert 'api_key":"[redacted]' in snippet
    assert 'Authorization":"[redacted]' in snippet
    assert "truncated" in snippet


class _FakeAnthropicResponse:
    """Minimal async response for Anthropic API tests."""

    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self) -> dict[str, object]:
        return self._payload

    async def text(self) -> str:
        return str(self._payload)


class _FakeAnthropicSession:
    """Minimal aiohttp session that records Anthropic requests."""

    def __init__(self, responses: list[_FakeAnthropicResponse], posts: list[dict]) -> None:
        self._responses = responses
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, **kwargs):
        self._posts.append({"url": url, **kwargs})
        return self._responses.pop(0)


class _FakeStreamContent:
    """Minimal async byte iterator for streaming response tests."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeStreamingResponse(_FakeAnthropicResponse):
    """Minimal async response with aiohttp-style streaming content."""

    def __init__(
        self,
        lines: list[str] | None = None,
        *,
        payload: dict[str, object] | None = None,
        status: int = 200,
        text: str | None = None,
    ) -> None:
        super().__init__(payload or {}, status=status)
        self.content = _FakeStreamContent(lines or [])
        self._text = text

    async def text(self) -> str:
        if self._text is not None:
            return self._text
        return await super().text()


def test_stream_tool_call_index_normalization_handles_nonzero_offsets() -> None:
    """Streamed tool-call indexes should normalize providers that start above zero."""
    offset = None

    first_index, offset = MCPAssistConversationEntity._normalize_stream_tool_call_index(
        1,
        offset,
    )
    second_index, offset = MCPAssistConversationEntity._normalize_stream_tool_call_index(
        2,
        offset,
    )
    repeated_first_index, offset = (
        MCPAssistConversationEntity._normalize_stream_tool_call_index(1, offset)
    )

    assert first_index == 0
    assert second_index == 1
    assert repeated_first_index == 0
    assert offset == 1


def test_toolless_check_preamble_detects_supported_non_english_phrases(
    hass, profile_entry_factory
) -> None:
    """Tool-less retry should not only work for English model preambles."""
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())

    assert agent._is_toolless_check_preamble("Voy a comprobar las luces de arriba.")
    assert agent._is_toolless_check_preamble("Je vais vérifier les lumières.")
    assert agent._is_toolless_check_preamble("確認します。")
    assert agent._is_toolless_check_preamble("سأتحقق من الأضواء.")


def test_compact_streamed_tool_calls_drops_empty_placeholders() -> None:
    """Empty streamed tool-call slots should not be executed."""
    compacted = MCPAssistConversationEntity._compact_streamed_tool_calls(
        [
            {},
            {"id": "call-1", "function": {"name": "get_index"}},
            {},
        ]
    )

    assert compacted == [{"id": "call-1", "function": {"name": "get_index"}}]


def test_partition_valid_tool_calls_rejects_partial_json_arguments() -> None:
    """Malformed streamed tool arguments should be retried, not executed as {}."""
    valid, invalid = MCPAssistConversationEntity._partition_valid_tool_calls(
        [
            {
                "id": "call-1",
                "function": {
                    "name": "discover_entities",
                    "arguments": '{"domain": "light"}',
                },
            },
            {
                "id": "call-2",
                "function": {
                    "name": "discover_entities",
                    "arguments": '{"domain": "light"',
                },
            },
        ]
    )

    assert [call["id"] for call in valid] == ["call-1"]
    assert [call["id"] for call in invalid] == ["call-2"]


def test_profile_tool_enablement_respects_shared_and_profile_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A profile can only use optional tools when both shared and profile settings allow it."""
    system_entry_factory(
        data={
            CONF_ENABLE_ASSIST_BRIDGE: True,
            CONF_ENABLE_LLM_API_BRIDGE: True,
            CONF_ENABLE_DEVICE_TOOLS: False,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_LLM_API_BRIDGE: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: True,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)

    assert agent.assist_bridge_enabled is False
    assert agent.llm_api_bridge_enabled is False
    assert agent.device_tools_enabled is False


def test_unit_conversion_tool_enablement_has_backward_compatible_fallbacks(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Unit conversion should inherit older calculator settings when its own flags are absent."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: None,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS: None,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)

    assert agent._is_tool_enabled_for_profile("convert_unit") is True


def test_profile_tool_filtering_hides_disabled_optional_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Only profile-enabled optional tools should be exposed to the LLM."""
    system_entry_factory(
        data={
            CONF_ENABLE_ASSIST_BRIDGE: True,
            CONF_ENABLE_LLM_API_BRIDGE: True,
            CONF_ENABLE_DEVICE_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_LLM_API_BRIDGE: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("discover_entities"),
            _tool("discover_devices"),
            _tool("list_assist_tools"),
            _tool("list_llm_apis"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "discover_devices" not in tool_names
    assert "list_assist_tools" not in tool_names
    assert "list_llm_apis" not in tool_names


def test_light_context_mode_advertises_core_profile_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Light context mode should keep the LLM-facing tool schema small."""
    system_entry_factory(
        data={
            CONF_ENABLE_DEVICE_TOOLS: True,
            CONF_ENABLE_WEB_SEARCH: True,
        }
    )
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_LIGHT})

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("discover_entities"),
            _tool("get_entity_details"),
            _tool("perform_action"),
            _tool("discover_devices"),
            _tool("get_device_details"),
            _tool("get_weather_forecast"),
            _tool("get_entity_history"),
            _tool("search"),
            _tool("sample_custom_tool"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert tool_names == {
        "discover_entities",
        "get_entity_details",
        "perform_action",
        "discover_devices",
        "get_device_details",
    }


def test_adaptive_context_mode_advertises_core_and_meta_tools(
    hass, profile_entry_factory
) -> None:
    """Adaptive context should start small and add selected schemas on demand."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    tools = [
        _tool("discover_entities"),
        _tool("get_entity_details"),
        _tool("perform_action"),
        _tool("sample_custom_tool"),
    ]

    advertised = agent._build_llm_tools_for_context(tools)
    advertised_names = {tool["function"]["name"] for tool in advertised}

    assert "discover_entities" in advertised_names
    assert "perform_action" in advertised_names
    assert ADAPTIVE_TOOL_CATALOG_NAME in advertised_names
    assert ADAPTIVE_TOOL_SCHEMA_NAME in advertised_names
    assert "sample_custom_tool" not in advertised_names

    token = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.set(
        frozenset({"sample_custom_tool"})
    )
    try:
        advertised = agent._build_llm_tools_for_context(tools)
    finally:
        agent_module._ADAPTIVE_LOADED_TOOL_NAMES.reset(token)

    advertised_names = {tool["function"]["name"] for tool in advertised}
    assert "sample_custom_tool" in advertised_names


def test_profile_tool_filtering_can_hide_convert_unit_without_hiding_add(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Profiles should still be able to disable unit conversion independently of math tools."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_CALCULATOR_TOOLS: True,
            CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("add"),
            _tool("convert_unit"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "add" in tool_names
    assert "convert_unit" not in tool_names


def test_profile_tool_filtering_hides_web_search_tools_for_disabled_profile(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Profiles should still be able to hide search and read_url together."""
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_WEB_SEARCH: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("search"),
            _tool("read_url"),
            _tool("discover_entities"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "search" not in tool_names
    assert "read_url" not in tool_names


def test_optional_technical_instructions_include_external_custom_tool_guidance(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Loaded external custom tools should be able to extend prompt guidance."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status",
            get_external_prompt_instructions=lambda: "## External Custom Tools\nUse sample_tool_status when asked for custom status."
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "External Custom Tools" in instructions
    assert "sample_tool_status" in instructions


def test_optional_technical_instructions_include_built_in_package_guidance(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Built-in packaged tools should contribute prompt guidance through the loader."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
        }
    )
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(
            get_builtin_prompt_instructions=lambda: (
                "## Optional Built-In Tool Packages\n"
                "Use calculator tools for arithmetic questions."
            ),
            get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "Optional Built-In Tool Packages" in instructions
    assert "calculator tools" in instructions


def test_profile_tool_filtering_hides_disabled_external_custom_tools(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """External custom tools should also respect profile-level tool disables."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory(
        options={CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS: False}
    )
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status"
        )
    )

    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("discover_entities"),
            _tool("sample_tool_status"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "sample_tool_status" not in tool_names


def test_optional_technical_instructions_omit_external_custom_tool_guidance_when_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """External custom tool prompt guidance should disappear when the profile disables it."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory(
        options={CONF_PROFILE_ENABLE_EXTERNAL_CUSTOM_TOOLS: False}
    )
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status",
            get_external_prompt_instructions=lambda: "## External Custom Tools\nUse sample_tool_status when asked for custom status.",
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "External Custom Tools" not in instructions
    assert "sample_tool_status" not in instructions


@pytest.mark.asyncio
async def test_adaptive_prompt_uses_compact_tool_loading_guidance(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Adaptive mode should not inline full optional custom-tool instructions."""
    system_entry_factory(data={CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True})
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(
            get_external_prompt_instructions=lambda: (
                "## External Custom Tools\nUse sample_tool_status for custom status."
            ),
        )
    )

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None)
    )

    assert "Adaptive Tool Loading" in prompt
    assert ADAPTIVE_TOOL_CATALOG_NAME in prompt
    assert ADAPTIVE_TOOL_SCHEMA_NAME in prompt
    assert "sample_tool_status" not in prompt


def test_profile_tool_filtering_can_disable_search_without_hiding_read_url(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Built-in packaged tools should be independently gateable per profile."""
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": True,
        }
    )
    entry = profile_entry_factory(
        options={
            "profile_enable_search_tool": False,
            "profile_enable_read_url_tool": True,
        }
    )
    agent = MCPAssistConversationEntity(hass, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(
            get_builtin_toggle_spec=lambda name: _builtin_spec(name),
            get_builtin_toggle_specs=lambda: BUILTIN_SPECS,
        )
    )

    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("search"),
            _tool("read_url"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "search" not in tool_names
    assert "read_url" in tool_names


@pytest.mark.asyncio
async def test_profile_disabled_tool_is_rejected_before_mcp_call(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Direct tool calls should also fail closed when the profile disabled that family."""
    system_entry_factory(data={CONF_ENABLE_ASSIST_BRIDGE: True})
    entry = profile_entry_factory(options={CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False})

    agent = MCPAssistConversationEntity(hass, entry)
    result = await agent._call_mcp_tool("list_assist_tools", {})

    assert result["isError"] is True
    assert "disabled for this profile" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_profile_disabled_unit_conversion_is_rejected_before_mcp_call(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Direct MCP calls should still respect the profile unit-conversion toggle."""
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_CALCULATOR_TOOLS: True,
            CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    result = await agent._call_mcp_tool("convert_unit", {"value": 1, "from_unit": "m", "to_unit": "ft"})

    assert result["isError"] is True
    assert "disabled for this profile" in result["content"][0]["text"]


def test_build_messages_respects_configured_max_history(
    hass, profile_entry_factory
) -> None:
    """Conversation message building should honor the configured history limit."""
    entry = profile_entry_factory(options={CONF_MAX_HISTORY: 2})
    agent = MCPAssistConversationEntity(hass, entry)

    history = [
        {"user": "u1", "assistant": "a1"},
        {"user": "u2", "assistant": "a2"},
        {"user": "u3", "assistant": "a3"},
        {"user": "u4", "assistant": "a4"},
    ]

    messages = agent._build_messages("system", "current", history)

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
        {"role": "user", "content": "current"},
    ]


def test_build_messages_includes_compact_tool_history_context(
    hass, profile_entry_factory
) -> None:
    """Follow-up turns should retain compact MCP target/action context."""
    entry = profile_entry_factory(options={CONF_MAX_HISTORY: 2})
    agent = MCPAssistConversationEntity(hass, entry)
    history = [
        {
            "user": "Are any kitchen lights on?",
            "assistant": "The kitchen pendant is on.",
            "actions": [
                {
                    "type": "mcp_tool",
                    "tool": "discover_entities",
                    "status": "ok",
                    "arguments": {
                        "area": "Kitchen",
                        "domain": "light",
                        "state": "on",
                    },
                }
            ],
        }
    ]

    messages = agent._build_messages("system", "turn it off", history)

    assert messages[2]["role"] == "assistant"
    assert "The kitchen pendant is on." in messages[2]["content"]
    assert "Tool context: discover_entities(area=Kitchen, domain=light, state=on)" in (
        messages[2]["content"]
    )


def test_tool_history_summary_marks_error_result_status(
    hass, profile_entry_factory
) -> None:
    """MCP failure result dictionaries should not be summarized as successful."""
    entry = profile_entry_factory(options={CONF_MAX_HISTORY: 2})
    agent = MCPAssistConversationEntity(hass, entry)
    summaries: list[dict[str, object]] = []
    token = agent_module._REQUEST_TOOL_HISTORY_SUMMARIES.set(summaries)

    try:
        agent._record_tool_history_summary(
            "perform_action",
            {"entity_id": "light.kitchen"},
            {"error": "Service call failed"},
        )
    finally:
        agent_module._REQUEST_TOOL_HISTORY_SUMMARIES.reset(token)

    assert summaries[0]["status"] == "error"
    assert "status=error" in agent._format_history_tool_context(summaries)


def test_build_messages_supports_zero_history(
    hass, profile_entry_factory
) -> None:
    """A zero history limit should omit prior turns entirely."""
    entry = profile_entry_factory(options={CONF_MAX_HISTORY: 0})
    agent = MCPAssistConversationEntity(hass, entry)

    history = [{"user": "u1", "assistant": "a1"}]

    messages = agent._build_messages("system", "current", history)

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current"},
    ]


def test_message_content_char_count_handles_provider_blocks(
    hass, profile_entry_factory
) -> None:
    """Prompt metrics should count provider text blocks without schema noise."""
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())

    assert agent._message_content_char_count(
        [
            {"role": "system", "content": "abc"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
            {
                "role": "tool",
                "content": {"type": "tool_result", "content": "result"},
            },
        ]
    ) == 14


def test_initial_payload_metrics_log_size_without_content(
    hass, profile_entry_factory, caplog
) -> None:
    """Debug metrics should help size first prompts without leaking prompt text."""
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    messages = [{"role": "user", "content": "please keep this private"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "discover_entities", "parameters": {}},
        }
    ]
    payload = {"model": "test-model", "messages": messages, "tools": tools}

    with caplog.at_level(logging.DEBUG, logger=agent_module._LOGGER.name):
        agent._log_initial_llm_payload_metrics(
            transport="http",
            iteration=0,
            payload=payload,
            messages=messages,
            tools=tools,
        )
        agent._log_initial_llm_payload_metrics(
            transport="http",
            iteration=1,
            payload=payload,
            messages=messages,
            tools=tools,
        )

    assert caplog.text.count("Initial LLM payload metrics") == 1
    assert "payload_bytes=" in caplog.text
    assert "message_chars=24" in caplog.text
    assert "please keep this private" not in caplog.text


def test_initial_payload_metrics_log_at_info_when_debug_mode_enabled(
    hass, profile_entry_factory, caplog
) -> None:
    """Profile Debug Mode should surface first-payload metrics without DEBUG logs."""
    agent = MCPAssistConversationEntity(
        hass, profile_entry_factory(data={CONF_DEBUG_MODE: True})
    )
    messages = [{"role": "user", "content": "private debug message"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "discover_entities", "parameters": {}},
        }
    ]
    payload = {"model": "test-model", "messages": messages, "tools": tools}

    with caplog.at_level(logging.INFO, logger=agent_module._LOGGER.name):
        agent._log_initial_llm_payload_metrics(
            transport="http",
            iteration=0,
            payload=payload,
            messages=messages,
            tools=tools,
        )

    assert "Initial LLM payload metrics" in caplog.text
    assert "payload_bytes=" in caplog.text
    assert "private debug message" not in caplog.text


def test_build_messages_light_context_caps_history_to_two_turns(
    hass, profile_entry_factory
) -> None:
    """Light context mode should override large history settings with a small cap."""
    entry = profile_entry_factory(
        options={CONF_MAX_HISTORY: 10, CONF_CONTEXT_MODE: CONTEXT_MODE_LIGHT}
    )
    agent = MCPAssistConversationEntity(hass, entry)

    history = [
        {"user": "u1", "assistant": "a1"},
        {"user": "u2", "assistant": "a2"},
        {"user": "u3", "assistant": "a3"},
        {"user": "u4", "assistant": "a4"},
    ]

    messages = agent._build_messages("system", "current", history)

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
        {"role": "user", "content": "current"},
    ]


def test_chat_log_mode_defaults_off_and_can_be_enabled(
    hass, profile_entry_factory
) -> None:
    """Persistent chat logging should be opt-in per profile."""
    default_agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    enabled_agent = MCPAssistConversationEntity(
        hass,
        profile_entry_factory(
            unique_id=f"{DOMAIN}_chat_log_profile",
            options={CONF_CHAT_LOG_MODE: True},
        ),
    )

    assert default_agent.chat_log_mode is False
    assert enabled_agent.chat_log_mode is True


def test_context_mode_defaults_to_adaptive_for_unknown_values(
    hass, profile_entry_factory
) -> None:
    """Unknown stored context mode values should fall back to adaptive behavior."""
    default_agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    invalid_agent = MCPAssistConversationEntity(
        hass,
        profile_entry_factory(
            unique_id=f"{DOMAIN}_context_mode_profile",
            options={CONF_CONTEXT_MODE: "unexpected"},
        ),
    )

    assert default_agent.context_mode == DEFAULT_CONTEXT_MODE
    assert default_agent.context_mode == CONTEXT_MODE_ADAPTIVE
    assert invalid_agent.context_mode == DEFAULT_CONTEXT_MODE


def test_ollama_context_error_suggests_light_context_mode(
    hass, profile_entry_factory
) -> None:
    """Ollama context-window errors should point users to the small-context setting."""
    entry = profile_entry_factory(data={CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA})
    agent = MCPAssistConversationEntity(hass, entry)

    message = agent._get_friendly_error_message(
        Exception(
            'ollama API error 400: {"error":"request (12772 tokens) exceed"}'
        )
    )

    assert "Context Mode: Light" in message
    assert "12772 tokens" in message


def test_ollama_token_rate_limit_is_not_reported_as_context_error(
    hass, profile_entry_factory
) -> None:
    """Token-based rate limits should not trigger light-context guidance."""
    entry = profile_entry_factory(data={CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA})
    agent = MCPAssistConversationEntity(hass, entry)

    message = agent._get_friendly_error_message(
        Exception(
            "ollama API error 429: rate limit exceeded: 1000 tokens per minute"
        )
    )

    assert "rate limit" in message
    assert "Context Mode: Light" not in message


@pytest.mark.asyncio
async def test_default_prompt_does_not_fetch_index_unless_requested(
    hass, profile_entry_factory
) -> None:
    """The built-in prompt path should not inline the smart index on every turn."""

    class FailIndexManager:
        async def get_index(self) -> dict[str, object]:
            raise AssertionError("get_index should not be called for default prompts")

    hass.data.setdefault(DOMAIN, {})["index_manager"] = FailIndexManager()
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(SimpleNamespace(device_id=None))

    assert "Current area:" in prompt
    assert "get_index()" in prompt
    assert "## Index" not in prompt


@pytest.mark.asyncio
async def test_default_prompt_includes_current_user_and_home_location_context(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Default prompts should include current HA user and home location when enabled."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: True,
            CONF_INCLUDE_HOME_LOCATION: True,
        }
    )
    hass.config.location_name = "Test Home"
    hass.config.latitude = 12.3456
    hass.config.longitude = -65.4321
    monkeypatch.setattr(
        hass.auth,
        "async_get_user",
        AsyncMock(return_value=SimpleNamespace(name="Jason")),
    )
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None, context=SimpleNamespace(user_id="user-123"))
    )

    assert "Current user: Jason" in prompt
    assert "Home location: Test Home (12.3456, -65.4321)" in prompt


@pytest.mark.asyncio
async def test_default_prompt_omits_optional_identity_context_when_disabled(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Shared privacy settings should allow identity/location prompt context to be omitted."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: False,
            CONF_INCLUDE_HOME_LOCATION: False,
        }
    )
    monkeypatch.setattr(
        hass.auth,
        "async_get_user",
        AsyncMock(return_value=SimpleNamespace(name="Jason")),
    )
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None, context=SimpleNamespace(user_id="user-123"))
    )

    assert "Current user:" not in prompt
    assert "Home location:" not in prompt


@pytest.mark.asyncio
async def test_mcp_tool_call_context_includes_enabled_user_and_home_location(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Tool packages should receive identity/location context when explicitly enabled."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: False,
            CONF_INCLUDE_HOME_LOCATION: False,
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: True,
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: True,
        }
    )
    hass.config.location_name = "Example Home"
    hass.config.latitude = 12.3456
    hass.config.longitude = -65.4321
    monkeypatch.setattr(
        hass.auth,
        "async_get_user",
        AsyncMock(return_value=SimpleNamespace(name="Jason")),
    )
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)

    context = await agent._build_mcp_tool_call_context(
        SimpleNamespace(context=SimpleNamespace(user_id="user-123"))
    )

    assert context["profile_entry_id"] == entry.entry_id
    assert context["profile_name"] == "Kitchen Profile"
    assert context["user_id"] == "user-123"
    assert context["user_name"] == "Jason"
    assert context["home_location"] == "Example Home"
    assert context["home_location_name"] == "Example Home"
    assert context["home_latitude"] == 12.3456
    assert context["home_longitude"] == -65.4321


@pytest.mark.asyncio
async def test_mcp_tool_call_context_omits_disabled_user_and_home_location(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Prompt context settings should not automatically share metadata with tools."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: True,
            CONF_INCLUDE_HOME_LOCATION: True,
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: False,
            CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS: False,
        }
    )
    hass.config.location_name = "Test Home"
    monkeypatch.setattr(
        hass.auth,
        "async_get_user",
        AsyncMock(return_value=SimpleNamespace(name="Jason")),
    )
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)

    context = await agent._build_mcp_tool_call_context(
        SimpleNamespace(context=SimpleNamespace(user_id="user-123"))
    )

    assert context == {
        "profile_entry_id": entry.entry_id,
        "profile_name": "Kitchen Profile",
    }


@pytest.mark.asyncio
async def test_custom_prompt_with_index_placeholder_fetches_index(
    hass, profile_entry_factory
) -> None:
    """Custom prompts that explicitly request {index} should still work."""

    class StubIndexManager:
        async def get_index(self) -> dict[str, object]:
            return {"areas": ["Kitchen"], "domains": {"light": 3}}

    hass.data.setdefault(DOMAIN, {})["index_manager"] = StubIndexManager()
    entry = profile_entry_factory(
        options={
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: "Index:{index}",
        }
    )
    agent = MCPAssistConversationEntity(hass, entry)

    prompt = await agent._build_system_prompt_with_context(SimpleNamespace(device_id=None))

    assert 'Index:{"areas":["Kitchen"],"domains":{"light":3}}' in prompt


@pytest.mark.asyncio
async def test_jinja_prompt_templates_render_with_context(
    hass,
    profile_entry_factory,
    monkeypatch,
) -> None:
    """Custom prompts should support Jinja while preserving context variables."""

    class StubIndexManager:
        async def get_index(self) -> dict[str, object]:
            return {"areas": ["Kitchen"]}

    hass.data.setdefault(DOMAIN, {})["index_manager"] = StubIndexManager()
    entry = profile_entry_factory(
        options={
            CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_SYSTEM_PROMPT: "User={{ current_user }}",
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: (
                "Area={{ current_area }} Index={{ index }} Legacy={date}"
            ),
        }
    )
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(
        agent,
        "_get_current_user_name",
        AsyncMock(return_value="Jason"),
    )
    monkeypatch.setattr(
        agent,
        "_get_current_area",
        AsyncMock(return_value="Kitchen"),
    )

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None)
    )

    assert "User=Jason" in prompt
    assert "Area=Kitchen" in prompt
    assert 'Index={"areas":["Kitchen"]}' in prompt
    assert "Legacy={date}" not in prompt


@pytest.mark.asyncio
async def test_jinja_prompt_templates_detect_variables_in_statements(
    hass,
    profile_entry_factory,
    monkeypatch,
) -> None:
    """Variables referenced only in Jinja statements should still be populated."""
    entry = profile_entry_factory(
        options={
            CONF_SYSTEM_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_SYSTEM_PROMPT: (
                "{% if current_user == 'Jason' %}Known user{% else %}Unknown user{% endif %}"
            ),
            CONF_TECHNICAL_PROMPT_MODE: PROMPT_MODE_CUSTOM,
            CONF_TECHNICAL_PROMPT: (
                "{% set selected_area = current_area %}Area={{ selected_area }}"
            ),
        }
    )
    agent = MCPAssistConversationEntity(hass, entry)
    get_user = AsyncMock(return_value="Jason")
    get_area = AsyncMock(return_value="Kitchen")
    monkeypatch.setattr(agent, "_get_current_user_name", get_user)
    monkeypatch.setattr(agent, "_get_current_area", get_area)

    prompt = await agent._build_system_prompt_with_context(
        SimpleNamespace(device_id=None)
    )

    assert "Known user" in prompt
    assert "Unknown user" not in prompt
    assert "Area=Kitchen" in prompt
    get_user.assert_awaited_once()
    get_area.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_mcp_tools_uses_short_lived_cache(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Repeated tool fetches with the same profile surface should reuse the cache."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_STANDARD})
    agent = MCPAssistConversationEntity(hass, entry)
    fetch_mock = AsyncMock(return_value=[_tool("discover_entities")])
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", fetch_mock)

    first = await agent._get_mcp_tools()
    second = await agent._get_mcp_tools()

    assert first == second
    fetch_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_mcp_tools_refetches_when_external_custom_tool_signature_changes(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """External custom tool changes should invalidate the profile MCP-tool cache immediately."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_STANDARD})
    agent = MCPAssistConversationEntity(hass, entry)
    state = {"signature": ("v1",)}
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        tools=SimpleNamespace(get_cache_signature=lambda: state["signature"])
    )
    fetch_mock = AsyncMock(
        side_effect=[
            [_tool("sample_tool_status")],
            [_tool("sample_tool_history")],
        ]
    )
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", fetch_mock)

    first = await agent._get_mcp_tools()
    state["signature"] = ("v2",)
    second = await agent._get_mcp_tools()

    assert first != second
    assert first[0]["function"]["name"] == "sample_tool_status"
    assert second[0]["function"]["name"] == "sample_tool_history"
    assert fetch_mock.await_count == 2


@pytest.mark.asyncio
async def test_get_mcp_tools_uses_stale_cache_on_refresh_failure(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """A transient tools/list failure should fall back to the last cached tool surface."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_STANDARD})
    agent = MCPAssistConversationEntity(hass, entry)
    cached_tools = [_tool("discover_entities")]
    agent._cached_profile_mcp_tools = list(cached_tools)
    agent._cached_profile_mcp_tools_key = agent._build_mcp_tool_cache_key()
    agent._cached_profile_mcp_tools_fetched_at = 0
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", AsyncMock(return_value=None))
    monkeypatch.setattr("custom_components.mcp_assist.agent.time.monotonic", lambda: 9999.0)

    result = await agent._get_mcp_tools()

    assert result == agent._convert_mcp_tools_to_llm_tools(cached_tools)


@pytest.mark.asyncio
async def test_adaptive_meta_tools_catalog_and_load_schemas(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Adaptive meta tools should expose a compact catalog before loading schemas."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    custom_tool = {
        "name": "sample_tool_status",
        "description": "private-schema-marker user-facing description",
        "llmDescription": "Get sample custom status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_details": {
                    "type": "boolean",
                    "description": "private-schema-marker should stay hidden",
                }
            },
        },
        "routingHints": {
            "keywords": ["status", "sample"],
            "preferred_when": "Use when the user asks for custom package health.",
        },
    }
    monkeypatch.setattr(
        agent,
        "_get_profile_mcp_tools",
        AsyncMock(return_value=[_tool("discover_entities"), custom_tool]),
    )
    token = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.set(frozenset())

    try:
        catalog_result = await agent._handle_adaptive_meta_tool(
            ADAPTIVE_TOOL_CATALOG_NAME,
            {"query": "sample"},
        )
        catalog_payload = json.loads(catalog_result["content"][0]["text"])

        assert catalog_payload["tools"][0]["name"] == "sample_tool_status"
        assert catalog_payload["tools"][0]["schema_loaded"] is False
        assert "inputSchema" not in catalog_result["content"][0]["text"]
        assert "private-schema-marker" not in catalog_result["content"][0]["text"]

        load_result = await agent._handle_adaptive_meta_tool(
            ADAPTIVE_TOOL_SCHEMA_NAME,
            {"tool_names": ["sample_tool_status"]},
        )
        load_payload = json.loads(load_result["content"][0]["text"])

        assert load_payload["loaded_tools"][0]["name"] == "sample_tool_status"
        assert load_payload["not_found"] == []
        assert "sample_tool_status" in agent_module._ADAPTIVE_LOADED_TOOL_NAMES.get()
    finally:
        agent_module._ADAPTIVE_LOADED_TOOL_NAMES.reset(token)


@pytest.mark.asyncio
async def test_adaptive_schema_load_survives_execute_tool_calls(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Schema loads should update the next advertised tool surface."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    tools = [
        _tool("discover_entities"),
        _tool("sample_tool_status"),
    ]
    monkeypatch.setattr(
        agent,
        "_get_profile_mcp_tools",
        AsyncMock(return_value=tools),
    )
    token = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.set(frozenset())

    try:
        await agent._execute_tool_calls(
            [
                {
                    "id": "load-1",
                    "function": {
                        "name": ADAPTIVE_TOOL_SCHEMA_NAME,
                        "arguments": json.dumps(
                            {"tool_names": ["sample_tool_status"]}
                        ),
                    },
                }
            ]
        )
        advertised = agent._build_llm_tools_for_context(tools)
    finally:
        agent_module._ADAPTIVE_LOADED_TOOL_NAMES.reset(token)

    advertised_names = {tool["function"]["name"] for tool in advertised}
    assert "sample_tool_status" in advertised_names


@pytest.mark.asyncio
async def test_adaptive_schema_load_survives_mixed_execute_tool_calls(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Schema loads should remain visible when real tools run concurrently."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    tools = [_tool("discover_entities"), _tool("sample_tool_status")]
    monkeypatch.setattr(
        agent,
        "_get_profile_mcp_tools",
        AsyncMock(return_value=tools),
    )
    monkeypatch.setattr(
        agent,
        "_call_mcp_tool",
        AsyncMock(return_value={"content": [{"type": "text", "text": "ok"}]}),
    )
    token = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.set(frozenset())

    try:
        await agent._execute_tool_calls(
            [
                {
                    "id": "call-1",
                    "function": {
                        "name": "discover_entities",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "load-1",
                    "function": {
                        "name": ADAPTIVE_TOOL_SCHEMA_NAME,
                        "arguments": json.dumps(
                            {"tool_names": ["sample_tool_status"]}
                        ),
                    },
                },
            ]
        )
        advertised = agent._build_llm_tools_for_context(tools)
    finally:
        agent_module._ADAPTIVE_LOADED_TOOL_NAMES.reset(token)

    advertised_names = {tool["function"]["name"] for tool in advertised}
    assert "sample_tool_status" in advertised_names


@pytest.mark.asyncio
async def test_adaptive_preloads_obvious_optional_tool_from_user_query(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Adaptive mode should avoid extra turns for high-confidence optional tools."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    weather_tool = {
        **_tool("get_weather_forecast"),
        "llmDescription": "Get weather forecast data.",
        "routingHints": {
            "keywords": ["weather", "forecast"],
            "preferred_when": "Use when the user asks about weather.",
        },
    }
    monkeypatch.setattr(
        agent,
        "_get_profile_mcp_tools",
        AsyncMock(return_value=[_tool("discover_entities"), weather_tool]),
    )
    token = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.set(frozenset())

    try:
        await agent._prepare_adaptive_tools_for_request(
            "What is the weather tomorrow?"
        )
        loaded_names = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.get()
    finally:
        agent_module._ADAPTIVE_LOADED_TOOL_NAMES.reset(token)

    assert "get_weather_forecast" in loaded_names


@pytest.mark.asyncio
async def test_adaptive_preloads_optional_tool_from_localized_query(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Adaptive query aliases should bridge non-English user text to tool metadata."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_ADAPTIVE})
    agent = MCPAssistConversationEntity(hass, entry)
    weather_tool = {
        **_tool("get_weather_forecast"),
        "llmDescription": "Get weather forecast data.",
        "routingHints": {
            "keywords": ["weather", "forecast"],
            "preferred_when": "Use when the user asks about weather.",
        },
    }
    monkeypatch.setattr(
        agent,
        "_get_profile_mcp_tools",
        AsyncMock(return_value=[_tool("discover_entities"), weather_tool]),
    )
    token = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.set(frozenset())

    try:
        await agent._prepare_adaptive_tools_for_request(
            "¿Qué tiempo hará mañana?"
        )
        loaded_names = agent_module._ADAPTIVE_LOADED_TOOL_NAMES.get()
    finally:
        agent_module._ADAPTIVE_LOADED_TOOL_NAMES.reset(token)

    assert "get_weather_forecast" in loaded_names


def test_adaptive_query_terms_expand_unicode_aliases() -> None:
    """Adaptive query matching should tokenize and expand non-English terms."""
    assert "weather" in normalize_adaptive_query_terms("¿Qué tiempo hará mañana?")
    assert "weather" in normalize_adaptive_query_terms("明天天气怎么样")
    assert "search" in normalize_adaptive_query_terms("buscar en la web")
    assert "access" in normalize_adaptive_query_terms(
        "How many times was the front door opened today?"
    )
    assert "url" in normalize_adaptive_query_terms(
        "Summarize https://example.com"
    )


def test_adaptive_tool_scoring_avoids_substring_false_positives() -> None:
    """Adaptive matching should avoid unrelated tiny substring matches."""
    query = "How many times was the front door opened today?"
    image_tool = {
        "name": "analyze_image",
        "description": "Analyze an image.",
        "inputSchema": {"type": "object", "properties": {}},
    }
    waste_tool = {
        "name": "waste_status",
        "description": "Return upcoming waste collection status.",
        "inputSchema": {"type": "object", "properties": {}},
    }
    access_tool = {
        "name": "home_access_history",
        "llmDescription": "Get lock and door access history or event counts.",
        "description": "Return lock and garage-door access history.",
        "routingHints": {
            "preferred_when": "Door/garage open/close counts and lock access history.",
        },
        "inputSchema": {"type": "object", "properties": {}},
    }

    assert score_adaptive_tool_match(image_tool, query) == 0
    assert score_adaptive_tool_match(waste_tool, query) == 0
    assert score_adaptive_tool_match(access_tool, query) >= 18

    matches = match_adaptive_tool_definitions(
        [image_tool, waste_tool, access_tool],
        query=query,
        limit=3,
    )

    assert [tool["name"] for tool in matches] == ["home_access_history"]


def test_adaptive_tool_scoring_prefers_read_url_for_urls() -> None:
    """URL queries should load URL-reading tools, not incidental substring matches."""
    read_tool = {
        "name": "read_url",
        "description": "Read and summarize a web page URL.",
        "inputSchema": {"type": "object", "properties": {}},
    }
    convert_tool = {
        "name": "convert_unit",
        "description": "Convert units of measure.",
        "inputSchema": {"type": "object", "properties": {}},
    }
    security_tool = {
        "name": "home_security_posture",
        "description": "Summarize home security posture.",
        "inputSchema": {"type": "object", "properties": {}},
    }

    matches = match_adaptive_tool_definitions(
        [convert_tool, security_tool, read_tool],
        query="Summarize https://example.com",
        limit=3,
    )

    assert [tool["name"] for tool in matches] == ["read_url"]


@pytest.mark.asyncio
async def test_fetch_mcp_tools_sends_shared_bearer_token(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Internal tools/list requests should authenticate when MCP bearer auth is enabled."""
    system_entry_factory(data={CONF_MCP_BEARER_TOKEN: "internal-token-123456"})
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    captured: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {"tools": [_tool("discover_entities")]},
            }

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, json, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    tools = await agent._fetch_mcp_tools_from_server()

    assert tools is not None
    assert tools[0]["name"] == "discover_entities"
    assert captured["payload"]["method"] == "tools/list"
    assert captured["headers"] == {
        "Authorization": "Bearer internal-token-123456"
    }


def test_compact_tool_result_for_llm_truncates_large_payloads(
    hass, profile_entry_factory
) -> None:
    """Oversized tool results should be trimmed before re-entering the model loop."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    large_result = "\n".join(f"line {index}" for index in range(300))
    compacted = agent._compact_tool_result_for_llm(
        "discover_entities", large_result
    )

    assert "Tool result truncated for model context" in compacted
    assert "Use limit/offset paging" in compacted
    assert len(compacted) < len(large_result)


@pytest.mark.asyncio
async def test_trigger_tts_is_a_noop(
    hass, profile_entry_factory
) -> None:
    """Interim streaming TTS should simply return without doing extra work."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    assert await agent._trigger_tts("Hello there.") is None


def test_convert_mcp_tools_to_llm_tools_compacts_schema(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """The LLM-facing tool schema should drop nonessential verbosity."""
    system_entry_factory(data={CONF_ENABLE_CALCULATOR_TOOLS: True})
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "discover_entities",
            "description": (
                "Find and list Home Assistant entities by criteria like area, floor, "
                "label, type, domain, device_class, current state, or aliases. "
                "Prefer this for most direct control."
            ),
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Area name or alias to search in. Check get_index() to see available areas.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entities to return.",
                        "default": 20,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    function = compact_tools[0]["function"]
    parameters = function["parameters"]

    assert function["description"] == "Find and list Home Assistant entities by criteria like area, floor, label, type, domain, device_class, current state, or aliases"
    assert "$schema" not in parameters
    assert "additionalProperties" not in parameters
    assert "required" not in parameters
    assert "description" not in parameters["properties"]["area"]
    assert "default" not in parameters["properties"]["limit"]


def test_convert_mcp_tools_to_llm_tools_keeps_empty_object_properties(
    hass, profile_entry_factory
) -> None:
    """OpenAI requires object schemas to include properties, even when empty."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "list_music_assistant_instances",
            "description": "List configured Music Assistant instances.",
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    parameters = compact_tools[0]["function"]["parameters"]

    assert parameters == {"type": "object", "properties": {}}


def test_convert_mcp_tools_to_llm_tools_appends_routing_hints(
    hass, profile_entry_factory
) -> None:
    """Routing hints should improve tool selection without needing long prompt text."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "sample_tool_status",
            "description": "Return custom status.",
            "inputSchema": {"type": "object", "properties": {}},
            "routingHints": {
                "keywords": ["status", "custom"],
                "preferred_when": "Use when the user asks for custom package health.",
                "returns": "A short status summary.",
                "example_queries": ["What's the custom status?"],
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    description = compact_tools[0]["function"]["description"]

    assert "Return custom status" in description
    assert "Use for: Use when the user asks for custom package health" in description
    assert "Keywords:" not in description
    assert "Example:" not in description


def test_convert_mcp_tools_to_llm_tools_keeps_compact_routing_with_llm_description(
    hass, profile_entry_factory
) -> None:
    """Explicit LLM descriptions should stay compact while preserving one routing hint."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    tools = [
        {
            "name": "sample_tool_status",
            "description": "Very long UI-facing description that should not be passed through to the model.",
            "llmDescription": "Get sample status.",
            "inputSchema": {"type": "object", "properties": {}},
            "routingHints": {
                "preferred_when": "Use when the user asks for custom package health.",
            },
        }
    ]

    compact_tools = agent._convert_mcp_tools_to_llm_tools(tools)
    description = compact_tools[0]["function"]["description"]

    assert description.startswith("Get sample status")
    assert "Use for: Use when the user asks for custom package health" in description
    assert "Very long UI-facing description" not in description


def test_agent_does_not_expose_provider_specific_transport_helpers() -> None:
    """Provider quirks should live under llm_providers, not on the agent."""
    provider_specific_helpers = {
        "_append_anthropic_message",
        "_build_anthropic_payload",
        "_build_ollama_payload",
        "_build_openai_payload",
        "_call_anthropic_messages",
        "_format_tool_calls_for_ollama",
        "_get_anthropic_provider",
    }

    for helper_name in provider_specific_helpers:
        assert not hasattr(MCPAssistConversationEntity, helper_name)


def test_default_prompt_uses_tools_without_extra_confirmation() -> None:
    """Routine voice-friendly home checks should happen in one assistant turn."""
    assert "use MCP tools before replying" in DEFAULT_SYSTEM_PROMPT
    assert "call the needed tools in the same turn" in DEFAULT_TECHNICAL_PROMPT
    assert "Do not ask the user to confirm" in DEFAULT_TECHNICAL_PROMPT
    assert "Do not reply only with a plan" in DEFAULT_TECHNICAL_PROMPT
    assert "Treat the tool-call budget as limited" in DEFAULT_TECHNICAL_PROMPT


@pytest.mark.asyncio
async def test_anthropic_messages_tool_loop_uses_native_endpoint(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Anthropic calls should use /v1/messages and round-trip tool_result blocks."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_ANTHROPIC,
            CONF_API_KEY: "anthropic-key",
            CONF_MODEL_NAME: "claude-sonnet-4-5",
        },
        options={CONF_MAX_ITERATIONS: 3, CONF_MAX_TOKENS: 100},
    )
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(
        agent,
        "_get_mcp_tools",
        AsyncMock(
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "discover_entities",
                        "description": "Find entities.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_image",
                        "description": "Analyze an image.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "generate_image",
                        "description": "Generate an image.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]
        ),
    )
    execute_mock = AsyncMock(
        return_value=[
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "Kitchen light is on",
            }
        ]
    )
    monkeypatch.setattr(agent, "_execute_tool_calls", execute_mock)

    posts: list[dict] = []
    responses = [
        _FakeAnthropicResponse(
            {
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "discover_entities",
                        "input": {"area": "Kitchen"},
                    },
                ],
                "stop_reason": "tool_use",
            }
        ),
        _FakeAnthropicResponse(
            {
                "content": [
                    {"type": "text", "text": "The kitchen light is on."}
                ],
                "stop_reason": "end_turn",
            }
        ),
    ]

    def _client_session(**kwargs):
        del kwargs
        return _FakeAnthropicSession(responses, posts)

    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", _client_session)
    metrics_calls: list[dict] = []
    monkeypatch.setattr(
        agent,
        "_log_initial_llm_payload_metrics",
        lambda **kwargs: metrics_calls.append(kwargs),
    )

    result = await agent._call_llm([{"role": "user", "content": "Check kitchen"}])

    assert result == "The kitchen light is on."
    assert [post["url"] for post in posts] == [
        "https://api.anthropic.com/v1/messages",
        "https://api.anthropic.com/v1/messages",
    ]
    assert posts[0]["headers"]["x-api-key"] == "anthropic-key"
    assert posts[0]["headers"]["anthropic-version"] == "2023-06-01"
    assert "tool_choice" not in posts[0]["json"]
    assert posts[0]["json"]["tools"][0]["input_schema"]["type"] == "object"
    assert [tool["name"] for tool in metrics_calls[0]["tools"]] == [
        "discover_entities"
    ]
    execute_mock.assert_awaited_once()
    assert execute_mock.await_args.args[0][0]["function"]["arguments"] == (
        '{"area": "Kitchen"}'
    )
    assert posts[1]["json"]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "Kitchen light is on",
            }
        ],
    }


@pytest.mark.asyncio
async def test_tool_budget_skips_extra_calls_and_finishes_without_tools(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Over-budget tool calls should get results and force a no-tools final answer."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_ANTHROPIC,
            CONF_API_KEY: "anthropic-key",
            CONF_MODEL_NAME: "claude-sonnet-4-5",
        },
        options={CONF_MAX_ITERATIONS: 1, CONF_MAX_TOKENS: 100},
    )
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(
        agent,
        "_get_mcp_tools",
        AsyncMock(
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "discover_entities",
                        "description": "Find entities.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_entity_details",
                        "description": "Get entity details.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]
        ),
    )
    execute_mock = AsyncMock(
        return_value=[
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "content": "Kitchen light is on",
            }
        ]
    )
    monkeypatch.setattr(agent, "_execute_tool_calls", execute_mock)

    posts: list[dict] = []
    responses = [
        _FakeAnthropicResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "discover_entities",
                        "input": {"floor": "downstairs", "domain": "light"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "get_entity_details",
                        "input": {"entity_id": "light.downstairs_lamp"},
                    },
                ],
                "stop_reason": "tool_use",
            }
        ),
        _FakeAnthropicResponse(
            {
                "content": [
                    {"type": "text", "text": "One downstairs light is on."}
                ],
                "stop_reason": "end_turn",
            }
        ),
    ]

    def _client_session(**kwargs):
        del kwargs
        return _FakeAnthropicSession(responses, posts)

    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", _client_session)
    monkeypatch.setattr(agent, "_log_initial_llm_payload_metrics", lambda **kwargs: None)

    result = await agent._call_llm(
        [{"role": "user", "content": "Are there any lights on downstairs?"}]
    )

    assert result == "One downstairs light is on."
    execute_mock.assert_awaited_once()
    executed_calls = execute_mock.await_args.args[0]
    assert [call["function"]["name"] for call in executed_calls] == [
        "discover_entities"
    ]
    assert "tools" in posts[0]["json"]
    assert "tools" not in posts[1]["json"]
    assert "tool-call budget has been reached" in posts[1]["json"]["system"]

    final_tool_result_blocks = posts[1]["json"]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in final_tool_result_blocks] == [
        "toolu_1",
        "toolu_2",
    ]
    skipped_content = json.loads(final_tool_result_blocks[1]["content"])
    assert skipped_content["budget_exhausted"] is True
    assert "configured tool-call budget" in skipped_content["error"]


def test_adaptive_meta_tool_calls_do_not_consume_tool_budget(
    hass, profile_entry_factory
) -> None:
    """Schema discovery should not spend the user's real MCP tool-call budget."""
    entry = profile_entry_factory(options={CONF_MAX_ITERATIONS: 1})
    agent = MCPAssistConversationEntity(hass, entry)
    provider = agent._get_llm_provider()
    tool_calls = [
        {"id": "meta-1", "function": {"name": ADAPTIVE_TOOL_CATALOG_NAME}},
        {"id": "call-1", "function": {"name": "discover_entities"}},
        {"id": "call-2", "function": {"name": "perform_action"}},
    ]

    plan = agent._prepare_tool_calls_for_budget(
        tool_calls,
        tool_calls_used=0,
        provider=provider,
    )

    assert [call["function"]["name"] for call in plan.executable_calls] == [
        ADAPTIVE_TOOL_CATALOG_NAME,
        "discover_entities",
    ]
    assert list(plan.skipped_results_by_index) == [2]
    assert "budget_exhausted" in json.dumps(plan.skipped_results_by_index[2])
    assert agent._count_budgeted_tool_calls(plan.executable_calls) == 1
    assert plan.exhausted is True


def test_tool_budget_results_preserve_original_call_order(
    hass, profile_entry_factory
) -> None:
    """Skipped real calls should stay in order when later meta calls can run."""
    entry = profile_entry_factory(options={CONF_MAX_ITERATIONS: 1})
    agent = MCPAssistConversationEntity(hass, entry)
    provider = agent._get_llm_provider()
    tool_calls = [
        {"id": "call-1", "function": {"name": "discover_entities"}},
        {"id": "call-2", "function": {"name": "perform_action"}},
        {"id": "load-1", "function": {"name": ADAPTIVE_TOOL_SCHEMA_NAME}},
    ]

    plan = agent._prepare_tool_calls_for_budget(
        tool_calls,
        tool_calls_used=0,
        provider=provider,
    )
    executable_results = [
        provider.build_tool_result_message(
            tool_call_id="call-1",
            tool_name="discover_entities",
            content="{}",
        ),
        provider.build_tool_result_message(
            tool_call_id="load-1",
            tool_name=ADAPTIVE_TOOL_SCHEMA_NAME,
            content="{}",
        ),
    ]

    results = agent._merge_tool_results_in_call_order(plan, executable_results)

    assert [result["tool_call_id"] for result in results] == [
        "call-1",
        "call-2",
        "load-1",
    ]
    skipped_content = json.loads(results[1]["content"])
    assert skipped_content["budget_exhausted"] is True


@pytest.mark.asyncio
async def test_ollama_toolless_check_preamble_retries_with_tool_call(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Ollama-style pre-tool narration should self-correct in the same turn."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
            CONF_MODEL_NAME: "qwen-tool-model",
        },
        options={
            CONF_CONTEXT_MODE: CONTEXT_MODE_LIGHT,
            CONF_MAX_ITERATIONS: 1,
            CONF_MAX_TOKENS: 100,
        },
    )
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(
        agent,
        "_get_mcp_tools",
        AsyncMock(
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "discover_entities",
                        "description": "Find entities.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        ),
    )
    execute_mock = AsyncMock(
        return_value=[
            {
                "role": "tool",
                "tool_name": "discover_entities",
                "content": "No upstairs lights are on.",
            }
        ]
    )
    monkeypatch.setattr(agent, "_execute_tool_calls", execute_mock)

    posts: list[dict] = []
    responses = [
        _FakeAnthropicResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll check if there are any lights on upstairs for you.",
                }
            }
        ),
        _FakeAnthropicResponse(
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "discover_entities",
                                "arguments": {
                                    "domain": "light",
                                    "floor": "upstairs",
                                    "state": "on",
                                },
                            }
                        }
                    ],
                }
            }
        ),
        _FakeAnthropicResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": "No upstairs lights are on.",
                }
            }
        ),
    ]

    def _client_session(**kwargs):
        del kwargs
        return _FakeAnthropicSession(responses, posts)

    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", _client_session)
    monkeypatch.setattr(agent, "_log_initial_llm_payload_metrics", lambda **kwargs: None)

    result = await agent._call_llm_http(
        [{"role": "user", "content": "Are there any lights on upstairs?"}]
    )

    assert result == "No upstairs lights are on."
    assert len(posts) == 3
    assert "tools" in posts[0]["json"]
    assert "tools" in posts[1]["json"]
    assert "tools" not in posts[2]["json"]

    retry_messages = posts[1]["json"]["messages"]
    assert retry_messages[-2] == {
        "role": "assistant",
        "content": "I'll check if there are any lights on upstairs for you.",
    }
    assert retry_messages[-1]["role"] == "system"
    assert "no MCP tool call was made" in retry_messages[-1]["content"]

    execute_mock.assert_awaited_once()
    executed_calls = execute_mock.await_args.args[0]
    assert [call["function"]["name"] for call in executed_calls] == [
        "discover_entities"
    ]
    assert executed_calls[0]["function"]["arguments"] == {
        "domain": "light",
        "floor": "upstairs",
        "state": "on",
    }


@pytest.mark.asyncio
async def test_ollama_invalid_tool_arguments_retry_without_malformed_history(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Malformed tool-call JSON should prompt a clean retry without bad history."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
            CONF_MODEL_NAME: "qwen-tool-model",
        },
        options={
            CONF_CONTEXT_MODE: CONTEXT_MODE_LIGHT,
            CONF_MAX_ITERATIONS: 1,
            CONF_MAX_TOKENS: 100,
        },
    )
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(
        agent,
        "_get_mcp_tools",
        AsyncMock(
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "discover_entities",
                        "description": "Find entities.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        ),
    )
    execute_mock = AsyncMock(
        return_value=[
            {
                "role": "tool",
                "tool_name": "discover_entities",
                "content": "No upstairs lights are on.",
            }
        ]
    )
    monkeypatch.setattr(agent, "_execute_tool_calls", execute_mock)

    posts: list[dict] = []
    responses = [
        _FakeAnthropicResponse(
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "discover_entities",
                                "arguments": '{"domain": "light"',
                            }
                        }
                    ],
                }
            }
        ),
        _FakeAnthropicResponse(
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "discover_entities",
                                "arguments": {
                                    "domain": "light",
                                    "floor": "upstairs",
                                    "state": "on",
                                },
                            }
                        }
                    ],
                }
            }
        ),
        _FakeAnthropicResponse(
            {
                "message": {
                    "role": "assistant",
                    "content": "No upstairs lights are on.",
                }
            }
        ),
    ]

    def _client_session(**kwargs):
        del kwargs
        return _FakeAnthropicSession(responses, posts)

    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", _client_session)
    monkeypatch.setattr(agent, "_log_initial_llm_payload_metrics", lambda **kwargs: None)

    result = await agent._call_llm_http(
        [{"role": "user", "content": "Are there any lights on upstairs?"}]
    )

    assert result == "No upstairs lights are on."
    assert len(posts) == 3
    retry_messages = posts[1]["json"]["messages"]
    assert not any(message.get("tool_calls") for message in retry_messages)
    assert retry_messages[-1]["role"] == "system"
    assert "invalid JSON arguments" in retry_messages[-1]["content"]
    assert "function.arguments" in retry_messages[-1]["content"]

    execute_mock.assert_awaited_once()
    executed_calls = execute_mock.await_args.args[0]
    assert executed_calls[0]["function"]["arguments"] == {
        "domain": "light",
        "floor": "upstairs",
        "state": "on",
    }


@pytest.mark.asyncio
async def test_ollama_streaming_invalid_tool_error_finishes_without_tools(
    hass, profile_entry_factory, monkeypatch, caplog
) -> None:
    """llama-server malformed tool-call 500s after tool results should not hard fail."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA,
            CONF_MODEL_NAME: "qwen-tool-model",
        },
        options={
            CONF_CONTEXT_MODE: CONTEXT_MODE_LIGHT,
            CONF_MAX_ITERATIONS: 3,
            CONF_MAX_TOKENS: 100,
        },
    )
    agent = MCPAssistConversationEntity(hass, entry)
    agent._streaming_available = True
    monkeypatch.setattr(
        agent,
        "_get_mcp_tools",
        AsyncMock(
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "discover_entities",
                        "description": "Find entities.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        ),
    )
    monkeypatch.setattr(
        agent,
        "_execute_tool_calls",
        AsyncMock(
            return_value=[
                {
                    "role": "tool",
                    "tool_name": "discover_entities",
                    "content": "No upstairs lights are on.",
                }
            ]
        ),
    )
    monkeypatch.setattr(agent, "_log_initial_llm_payload_metrics", lambda **kwargs: None)

    invalid_tool_error = (
        '{"error":"llama-server returned invalid tool call arguments for '
        '\\"discover_entities\\": unexpected end of JSON input"}'
    )
    posts: list[dict] = []
    responses = [
        _FakeStreamingResponse(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "discover_entities",
                                        "arguments": {
                                            "domain": "light",
                                            "floor": "upstairs",
                                            "state": "on",
                                        },
                                    }
                                }
                            ],
                        }
                    }
                ),
                json.dumps({"done": True}),
            ]
        ),
        _FakeStreamingResponse(
            status=500,
            payload={
                "error": (
                    "llama-server returned invalid tool call arguments for "
                    '"discover_entities": unexpected end of JSON input'
                )
            },
            text=invalid_tool_error,
        ),
        _FakeStreamingResponse(
            payload={
                "message": {
                    "role": "assistant",
                    "content": "No upstairs lights are on.",
                }
            }
        ),
    ]

    def _client_session(**kwargs):
        del kwargs
        return _FakeAnthropicSession(responses, posts)

    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", _client_session)

    with caplog.at_level(logging.WARNING, logger=agent_module._LOGGER.name):
        result = await agent._call_llm_streaming(
            [{"role": "user", "content": "Are there any lights on upstairs?"}]
        )

    assert result == "No upstairs lights are on."
    assert len(posts) == 3
    assert "tools" in posts[1]["json"]
    assert "tools" not in posts[2]["json"]
    assert (
        posts[2]["json"]["messages"][-1]["content"]
        == agent_module.TOOL_BUDGET_FINAL_INSTRUCTION
    )
    assert "Streaming failed with status 500" not in caplog.text
    assert "Streaming iteration 2 failed" not in caplog.text


def test_toolless_check_retry_does_not_fire_after_tool_results(
    hass, profile_entry_factory
) -> None:
    """A post-tool summary that starts like a check should not trigger extra turns."""
    entry = profile_entry_factory(options={CONF_CONTEXT_MODE: CONTEXT_MODE_LIGHT})
    agent = MCPAssistConversationEntity(hass, entry)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "discover_entities",
                "description": "Find entities.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    assert agent._should_retry_toolless_response(
        "I'll check if any kitchen lights are on.",
        tools=tools,
        retry_used=False,
        tool_calls_used=0,
    )
    assert not agent._should_retry_toolless_response(
        "I can see the kitchen light is on.",
        tools=tools,
        retry_used=False,
        tool_calls_used=1,
    )


def test_format_tool_result_for_llm_preserves_structured_results_without_binary(
    hass, profile_entry_factory
) -> None:
    """Structured MCP results should survive, but binary image payloads should be compacted."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    formatted = agent._format_tool_result_for_llm(
        "analyze_image",
        {
            "content": [
                {"type": "text", "text": "White SUV in the driveway."},
                {"type": "image", "mimeType": "image/jpeg", "data": "a" * 4096},
            ],
            "structuredContent": {"source": {"camera_entity_id": "camera.driveway"}},
            "isError": False,
        },
    )

    assert "White SUV in the driveway." in formatted
    assert "[binary image omitted:" in formatted
    assert "camera.driveway" in formatted


def test_redacted_log_snippet_removes_common_secret_markers() -> None:
    """Log snippets should redact common secret-bearing field names."""
    snippet = agent_module._redacted_log_snippet(
        'Authorization: Bearer abc123 api_key=secret-token password=hunter2 '
        '{"api_key":"quoted-secret","token":"quoted-token"} '
        '{"error":"API key provided: sk-prose-secret"}',
    )

    assert "Authorization" not in snippet
    assert "Bearer" not in snippet
    assert "api_key" not in snippet
    assert "password" not in snippet
    assert "abc123" not in snippet
    assert "secret-token" not in snippet
    assert "hunter2" not in snippet
    assert "quoted-secret" not in snippet
    assert "quoted-token" not in snippet
    assert "sk-prose-secret" not in snippet
    assert "[redacted]" in snippet


def test_friendly_error_message_uses_sanitized_provider_token_details(
    hass, profile_entry_factory
) -> None:
    """Provider errors should keep token-limit context after body redaction."""
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)

    message = agent._get_friendly_error_message(
        Exception(
            'ollama API error 400: {"error":{"message":"request (12772 tokens) '
            'exceed maximum context length","api_key":"[redacted]"}}'
        )
    )

    assert "12772 token limit" in message
    assert "api_key" not in message


def test_follow_up_pattern_debug_logs_do_not_include_response_tail(
    hass, profile_entry_factory, caplog
) -> None:
    """Follow-up detection should log metadata without response content."""
    entry = profile_entry_factory(data={CONF_DEBUG_MODE: True})
    agent = MCPAssistConversationEntity(hass, entry)

    with caplog.at_level(logging.INFO, logger=agent_module._LOGGER.name):
        assert agent._detect_follow_up_patterns(
            "Please do not log private-token-123?"
        )

    assert "Pattern detection - Full response length" in caplog.text
    assert "Checking trailing response window" in caplog.text
    assert "private-token-123" not in caplog.text


def test_tool_call_log_summary_omits_argument_values(hass, profile_entry_factory) -> None:
    """Streaming tool-call debug logs should keep metadata without raw arguments."""
    entry = profile_entry_factory(data={CONF_DEBUG_MODE: True})
    agent = MCPAssistConversationEntity(hass, entry)
    summary = agent._tool_call_log_summary(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read_url",
                    "arguments": (
                        '{"url":"https://private.example/path?api_key=sk-secret",'
                        '"entity_id":"light.private","api_key":"sk-secret"}'
                    ),
                },
            }
        ]
    )

    log_text = str(summary)

    assert summary[0]["name"] == "read_url"
    assert summary[0]["argument_keys"] == "api_key, entity_id, url"
    assert summary[0]["argument_bytes"] > 0
    assert "https://private.example" not in log_text
    assert "light.private" not in log_text
    assert "sk-secret" not in log_text


@pytest.mark.asyncio
async def test_call_mcp_tool_includes_profile_context(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """MCP tool calls should identify the active profile for settings-aware tools."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER: False,
            CONF_INCLUDE_HOME_LOCATION: False,
        }
    )
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)
    captured: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
            }

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    result = await agent._call_mcp_tool("sample_tool_status", {})

    assert result["content"][0]["text"] == "ok"
    assert captured["payload"]["params"]["context"] == {
        "profile_entry_id": entry.entry_id,
        "profile_name": "Kitchen Profile",
    }


@pytest.mark.asyncio
async def test_call_mcp_tool_sends_shared_bearer_token(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Internal tools/call requests should authenticate when MCP bearer auth is enabled."""
    system_entry_factory(data={CONF_MCP_BEARER_TOKEN: "internal-token-123456"})
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)
    captured: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
            }

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, json, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    result = await agent._call_mcp_tool("sample_tool_status", {})

    assert result["content"][0]["text"] == "ok"
    assert captured["payload"]["method"] == "tools/call"
    assert captured["headers"] == {
        "Authorization": "Bearer internal-token-123456"
    }


@pytest.mark.asyncio
async def test_call_mcp_tool_logs_metadata_without_arguments_or_results(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, caplog
) -> None:
    """MCP exchange logs should not include raw arguments or result content."""
    system_entry_factory()
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "super-secret-result"}],
                    "isError": False,
                },
            }

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    with caplog.at_level(logging.DEBUG, logger=agent_module._LOGGER.name):
        result = await agent._call_mcp_tool(
            "sample_tool_status",
            {"entity_id": "light.private", "password": "super-secret"},
        )

    assert result["content"][0]["text"] == "super-secret-result"
    assert "sample_tool_status" in caplog.text
    assert "argument_keys=entity_id, password" in caplog.text
    assert "payload_bytes=" in caplog.text
    assert "light.private" not in caplog.text
    assert "super-secret" not in caplog.text
    assert "super-secret-result" not in caplog.text


@pytest.mark.asyncio
async def test_call_mcp_tool_non_200_returns_status_without_raw_body(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, caplog
) -> None:
    """MCP HTTP failures should not send raw response bodies back to the model."""
    system_entry_factory()
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)

    class _FakeResponse:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def text(self):
            return "provider token secret-body"

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    with caplog.at_level(logging.ERROR, logger=agent_module._LOGGER.name):
        result = await agent._call_mcp_tool("sample_tool_status", {"query": "private"})

    assert result == {"error": "Tool execution failed with HTTP 500"}
    assert "secret-body" not in result["error"]
    assert "secret-body" not in caplog.text
    assert "token" not in caplog.text
    assert "[redacted]" in caplog.text


@pytest.mark.asyncio
async def test_execute_single_tool_call_records_persistent_tool_log(
    hass, profile_entry_factory, monkeypatch
) -> None:
    """Tool names, arguments, and results should be captured for Chat Log Mode."""
    entry = profile_entry_factory(options={CONF_CHAT_LOG_MODE: True})
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(
        agent,
        "_call_mcp_tool",
        AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "Kitchen light turned on"}],
                "isError": False,
            }
        ),
    )
    record = {
        "id": "record-1",
        "created_at": "2026-06-01T00:00:00+00:00",
        "tools": [],
    }
    token = agent_module._PERSISTENT_CHAT_LOG_RECORD.set(record)

    try:
        result = await agent._execute_single_tool_call(
            {
                "id": "call-1",
                "function": {
                    "name": "perform_action",
                    "arguments": '{"entity_id":"light.kitchen","action":"turn_on"}',
                },
            }
        )
    finally:
        agent_module._PERSISTENT_CHAT_LOG_RECORD.reset(token)

    assert result["tool_call_id"] == "call-1"
    assert record["tools"] == [
        {
            "id": "call-1",
            "name": "perform_action",
            "started_at": record["tools"][0]["started_at"],
            "arguments": {"entity_id": "light.kitchen", "action": "turn_on"},
            "completed_at": record["tools"][0]["completed_at"],
            "result": {
                "content": [{"type": "text", "text": "Kitchen light turned on"}],
                "isError": False,
            },
            "llm_content": "Kitchen light turned on",
        }
    ]


@pytest.mark.asyncio
async def test_finish_persistent_chat_log_saves_with_manager(
    hass, profile_entry_factory
) -> None:
    """Completed chat log records should be persisted through the shared manager."""
    entry = profile_entry_factory(options={CONF_CHAT_LOG_MODE: True})
    agent = MCPAssistConversationEntity(hass, entry)
    manager = SimpleNamespace(async_record=AsyncMock())
    hass.data.setdefault(DOMAIN, {})["chat_log_manager"] = manager
    record = {
        "id": "record-1",
        "created_at": "2026-06-01T00:00:00+00:00",
        "_started_monotonic": 1.0,
        "tools": [],
    }
    token = agent_module._PERSISTENT_CHAT_LOG_RECORD.set(record)

    try:
        await agent._finish_persistent_chat_log_record(
            assistant_text="Done.",
            continue_conversation=False,
        )
    finally:
        agent_module._PERSISTENT_CHAT_LOG_RECORD.reset(token)

    manager.async_record.assert_awaited_once()
    saved = manager.async_record.await_args.args[0]
    assert saved["assistant_text"] == "Done."
    assert saved["continue_conversation"] is False
    assert saved["completed_at"]
    assert saved["_saved"] is True


@pytest.mark.asyncio
async def test_call_mcp_tool_uses_task_local_user_context(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Concurrent requests must not borrow user context from shared entity state."""
    system_entry_factory(
        data={
            CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS: True,
            CONF_INCLUDE_HOME_LOCATION: False,
        }
    )

    async def async_get_user(user_id):
        await asyncio.sleep(0)
        return SimpleNamespace(name=f"Name {user_id}")

    monkeypatch.setattr(hass.auth, "async_get_user", async_get_user)
    entry = profile_entry_factory(data={CONF_PROFILE_NAME: "Kitchen Profile"})
    agent = MCPAssistConversationEntity(hass, entry)
    captured: list[dict[str, object]] = []

    class _FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
            }

    class _FakeSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json):
            del url
            captured.append(json)
            return _FakeResponse()

    monkeypatch.setattr(
        "custom_components.mcp_assist.agent.aiohttp.ClientSession",
        _FakeSession,
    )

    async def call_as(user_id: str) -> None:
        token = agent_module._REQUEST_USER_INPUT.set(
            SimpleNamespace(context=SimpleNamespace(user_id=user_id))
        )
        try:
            await agent._call_mcp_tool("sample_tool_status", {})
        finally:
            agent_module._REQUEST_USER_INPUT.reset(token)

    await asyncio.gather(call_as("user-a"), call_as("user-b"))

    contexts = {
        payload["params"]["context"]["user_id"]: payload["params"]["context"]
        for payload in captured
    }
    assert set(contexts) == {"user-a", "user-b"}
    assert contexts["user-a"]["user_name"] == "Name user-a"
    assert contexts["user-b"]["user_name"] == "Name user-b"


def test_clean_text_for_tts_removes_spaces_before_punctuation(
    hass, profile_entry_factory
) -> None:
    """Final speech text should not keep stray spaces before punctuation."""
    entry = profile_entry_factory(data={CONF_CLEAN_RESPONSES: False})
    agent = MCPAssistConversationEntity(hass, entry)

    cleaned = agent._clean_text_for_tts(
        "I can use it , the weather entity is available ."
    )

    assert cleaned == "I can use it, the weather entity is available."
