"""Tests for profile-specific tool gating in the conversation agent."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist import agent as agent_module
from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.custom_tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.const import (
    CONF_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_ASSIST_BRIDGE,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_INCLUDE_CURRENT_USER,
    CONF_INCLUDE_HOME_LOCATION,
    CONF_INCLUDE_CURRENT_USER_IN_TOOL_CALLS,
    CONF_INCLUDE_HOME_LOCATION_IN_TOOL_CALLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_CLEAN_RESPONSES,
    CONF_ENABLE_DEVICE_TOOLS,
    CONF_MAX_HISTORY,
    CONF_MAX_ITERATIONS,
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
    CONF_PROFILE_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_PROFILE_ENABLE_DEVICE_TOOLS,
    CONF_PROFILE_ENABLE_WEB_SEARCH,
    CONF_CHAT_LOG_MODE,
    CONF_DEBUG_MODE,
    DOMAIN,
    PROMPT_MODE_CUSTOM,
    SERVER_TYPE_ANTHROPIC,
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


def test_profile_tool_enablement_respects_shared_and_profile_settings(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """A profile can only use optional tools when both shared and profile settings allow it."""
    system_entry_factory(
        data={
            CONF_ENABLE_ASSIST_BRIDGE: True,
            CONF_ENABLE_DEVICE_TOOLS: False,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: True,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)

    assert agent.assist_bridge_enabled is False
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
            CONF_ENABLE_DEVICE_TOOLS: True,
        }
    )
    entry = profile_entry_factory(
        options={
            CONF_PROFILE_ENABLE_ASSIST_BRIDGE: False,
            CONF_PROFILE_ENABLE_DEVICE_TOOLS: False,
        }
    )

    agent = MCPAssistConversationEntity(hass, entry)
    filtered = agent._filter_mcp_tools_for_profile(
        [
            _tool("discover_entities"),
            _tool("discover_devices"),
            _tool("list_assist_tools"),
        ]
    )

    tool_names = {tool["name"] for tool in filtered}
    assert "discover_entities" in tool_names
    assert "discover_devices" not in tool_names
    assert "list_assist_tools" not in tool_names


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
        custom_tools=SimpleNamespace(
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
        custom_tools=SimpleNamespace(
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
        custom_tools=SimpleNamespace(
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
        custom_tools=SimpleNamespace(
            is_external_custom_tool=lambda name: name == "sample_tool_status",
            get_external_prompt_instructions=lambda: "## External Custom Tools\nUse sample_tool_status when asked for custom status.",
        )
    )

    instructions = agent._build_optional_technical_instructions("Kitchen")

    assert "External Custom Tools" not in instructions
    assert "sample_tool_status" not in instructions


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
        custom_tools=SimpleNamespace(
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
    ]

    messages = agent._build_messages("system", "current", history)

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "current"},
    ]


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
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    fetch_mock = AsyncMock(return_value=[{"type": "function", "function": {"name": "discover_entities"}}])
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
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    state = {"signature": ("v1",)}
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        custom_tools=SimpleNamespace(get_cache_signature=lambda: state["signature"])
    )
    fetch_mock = AsyncMock(
        side_effect=[
            [{"type": "function", "function": {"name": "sample_tool_status"}}],
            [{"type": "function", "function": {"name": "sample_tool_history"}}],
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
    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    cached_tools = [{"type": "function", "function": {"name": "discover_entities"}}]
    agent._cached_llm_tools = list(cached_tools)
    agent._cached_llm_tools_key = agent._build_mcp_tool_cache_key()
    agent._cached_llm_tools_fetched_at = 0
    monkeypatch.setattr(agent, "_fetch_mcp_tools_from_server", AsyncMock(return_value=None))
    monkeypatch.setattr("custom_components.mcp_assist.agent.time.monotonic", lambda: 9999.0)

    result = await agent._get_mcp_tools()

    assert result == cached_tools


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


def test_build_anthropic_payload_uses_native_messages_shape(
    hass, profile_entry_factory
) -> None:
    """Claude should receive native Messages payloads, not OpenAI chat payloads."""
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_ANTHROPIC,
            CONF_MODEL_NAME: "claude-sonnet-4-5",
        },
        options={CONF_MAX_TOKENS: 321},
    )
    agent = MCPAssistConversationEntity(hass, entry)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "discover_entities",
                "description": "Find entities.",
                "parameters": {
                    "type": "object",
                    "properties": {"area": {"type": "string"}},
                },
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
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Find kitchen lights."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "discover_entities",
                        "arguments": '{"area":"Kitchen"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Kitchen light"},
    ]

    payload = agent._build_anthropic_payload(messages, tools)

    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["max_tokens"] == 321
    assert payload["system"] == "You are helpful."
    assert "stream" not in payload
    assert payload["tools"] == [
        {
            "name": "discover_entities",
            "description": "Find entities.",
            "input_schema": {
                "type": "object",
                "properties": {"area": {"type": "string"}},
            },
        }
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Find kitchen lights."}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "discover_entities",
                    "input": {"area": "Kitchen"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "Kitchen light",
                }
            ],
        },
    ]


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
                }
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
