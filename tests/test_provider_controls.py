"""Opt-in provider controls adapted from upstream v0.19."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import DOMAIN
from custom_components.mcp_assist.llm_providers import build_provider_settings, create_llm_provider
from custom_components.mcp_assist.provider_runtime import build_provider_auth_headers


@pytest.mark.parametrize(("provider", "model", "mode", "expected"), [
    ("ollama", "qwen3", "off", {"think": False}),
    ("ollama", "qwen3", "high", {"think": True}),
    ("ollama", "gpt-oss:20b", "off", {"think": "low"}),
    ("ollama", "gpt-oss:20b", "high", {"think": "high"}),
    ("vllm", "qwen3", "off", {"chat_template_kwargs": {
        "enable_thinking": False, "reasoning_effort": "low",
    }}),
    ("llamacpp", "qwen3", "medium", {"chat_template_kwargs": {
        "enable_thinking": True, "reasoning_effort": "medium",
    }}),
    ("openrouter", "example/model", "off", {"reasoning": {"effort": "none"}}),
    ("gemini", "gemini-2.5-flash", "off", {"reasoning_effort": "none"}),
    ("gemini", "gemini-3-pro-preview", "off", {"reasoning_effort": "low"}),
    ("gemini", "gemini-3-pro-preview", "medium", {"reasoning_effort": "high"}),
    ("gemini", "gemini-3.1-pro-preview", "off", {"reasoning_effort": "low"}),
    ("gemini", "gemini-3.1-pro-preview", "medium", {"reasoning_effort": "medium"}),
    ("gemini", "gemini-3-flash-preview", "off", {"reasoning_effort": "minimal"}),
    ("gemini", "gemini-3.1-flash-lite-preview", "off", {"reasoning_effort": "minimal"}),
    ("gemini", "gemini-3.5-flash-lite", "off", {"reasoning_effort": "minimal"}),
    ("gemini", "gemini-3.6-flash", "off", {"reasoning_effort": "minimal"}),
    ("gemini", "gemini-3.8-flash", "off", {"reasoning_effort": "low"}),
    ("gemini", "gemini-4-argon", "off", {"reasoning_effort": "low"}),
    ("gemini", "models/gemini-3-pro-preview", "medium", {"reasoning_effort": "high"}),
    ("openai", "gpt-5", "off", {"reasoning": {"effort": "minimal"}}),
    ("openai", "gpt-6.1-sol", "off", {"reasoning": {"effort": "low"}}),
    ("openai", "gpt-6-astra", "off", {"reasoning": {"effort": "low"}}),
    ("openai", "gpt-6-sol", "off", {"reasoning": {"effort": "none"}}),
    ("openai", "o3", "off", {"reasoning": {"effort": "low"}}),
    ("openai", "gpt-5-pro", "low", {"reasoning": {"effort": "high"}}),
    ("openai", "gpt-4.1", "high", {}),
])
@pytest.mark.parametrize("stream", [True, False])
def test_thinking_controls_reach_both_transports(
    profile_entry_factory, provider, model, mode, expected, stream
):
    entry = profile_entry_factory(data={
        "server_type": provider, "model_name": model,
        "lmstudio_url": "https://api.openai.com" if provider == "openai" else "http://localhost:8000",
    }, options={"thinking_mode": mode, "openai_api_transport": "responses"})
    runtime = create_llm_provider(build_provider_settings(entry, max_tokens=500, temperature=0.5))
    payload = runtime.build_payload([{"role": "user", "content": "Hello"}], stream=stream)
    controls = {k: payload[k] for k in ("think", "chat_template_kwargs", "reasoning", "reasoning_effort") if k in payload}
    assert controls == expected


@pytest.mark.parametrize("provider", ["ollama", "vllm", "llamacpp", "openrouter", "gemini", "openai"])
@pytest.mark.parametrize("mode", [None, "default", "invalid"])
def test_default_thinking_keeps_wire_contract(profile_entry_factory, provider, mode):
    entry = profile_entry_factory(data={"server_type": provider}, options={"thinking_mode": mode, "openai_api_transport": "responses"})
    runtime = create_llm_provider(build_provider_settings(entry, max_tokens=500, temperature=0.5))
    payload = runtime.build_payload([{"role": "user", "content": "Hello"}])
    assert not {"think", "chat_template_kwargs", "reasoning", "reasoning_effort"} & payload.keys()


@pytest.mark.parametrize("provider", ["lmstudio", "llamacpp", "vllm", "ollama"])
def test_local_auth_is_opt_in_and_consistent(profile_entry_factory, provider):
    assert build_provider_auth_headers(provider, "") == {}
    entry = profile_entry_factory(data={"server_type": provider, "api_key": "test-local-key"})
    runtime = create_llm_provider(build_provider_settings(entry, max_tokens=500, temperature=0.5))
    expected = {"Authorization": "Bearer test-local-key"}
    assert runtime.headers() == expected
    assert runtime.model_request_headers(dict(entry.data)) == expected


@pytest.mark.parametrize(("options", "expected"), [
    ({}, ("main", None)),
    ({"openclaw_agent_id": "voice"}, ("agent:voice:main", "voice")),
    ({"openclaw_agent_id": "VOICE", "openclaw_session_key": "satellite"},
     ("agent:voice:satellite", "voice")),
    ({"openclaw_session_key": "agent:other:fixed"}, ("agent:other:fixed", None)),
    ({"openclaw_session_mode": "per_conversation", "openclaw_agent_id": "voice"},
     ("agent:voice:ha-chat-123", "voice")),
    ({"openclaw_session_mode": "per_conversation"}, ("agent:main:ha-chat-123", "main")),
])
def test_openclaw_session_routing(hass, profile_entry_factory, options, expected):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(options=options))
    assert agent._openclaw_session_target("CHAT-123") == expected


@pytest.mark.parametrize("area", ["Kitchen", "Unknown"])
async def test_openclaw_sends_room_and_persona_without_local_discovery(
    hass, profile_entry_factory, monkeypatch, area
):
    entry = profile_entry_factory(data={"server_type": "openclaw"}, options={
        "system_prompt": "Answer briefly", "thinking_mode": "low",
    })
    agent = MCPAssistConversationEntity(hass, entry)
    client = SimpleNamespace(send_message=AsyncMock(return_value="Ready"))
    hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})["openclaw_client"] = client
    monkeypatch.setattr(agent, "_get_current_area", AsyncMock(return_value=area))
    monkeypatch.setattr(agent, "_build_response_result", AsyncMock(return_value="done"))
    await agent._handle_openclaw_message(SimpleNamespace(text="Hello"), "chat-123")
    client.send_message.assert_awaited_once_with(
        "[Room: Kitchen] Hello" if area == "Kitchen" else "Hello", "main", agent_id=None,
        extra_system_prompt="Answer briefly\n\nSpeaking from: Kitchen" if area == "Kitchen" else "Answer briefly",
        thinking="low",
    )


async def test_default_prompt_is_stable_and_custom_context_still_renders(
    hass, profile_entry_factory, monkeypatch
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    get_area = AsyncMock(side_effect=["Kitchen", "Office"])
    monkeypatch.setattr(agent, "_get_current_area", get_area)
    user = SimpleNamespace(device_id="satellite")
    first = await agent._build_system_prompt_with_context(user)
    context = await agent._build_turn_context(user)
    second = await agent._build_system_prompt_with_context(user)
    next_context = await agent._build_turn_context(user)
    assert first == second
    assert "Current area: Kitchen" in context
    assert "Current area: Office" in next_context
    assert "Current time:" in context and "Current time:" not in first

    hass.config_entries.async_update_entry(agent.entry, options={
        "technical_prompt_mode": "custom", "technical_prompt": "Time: {{ time }}",
    })
    assert await agent._build_turn_context(user) == ""
    assert "Time: " in await agent._build_system_prompt_with_context(user)


@pytest.mark.parametrize("music_enabled", [True, False])
async def test_custom_prompt_keeps_optional_music_room_context(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, music_enabled
):
    system_entry_factory(data={"enable_music_assistant_support": music_enabled})
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(options={
        "context_mode": "standard", "technical_prompt_mode": "custom",
        "technical_prompt": "Use the available tools.",
    }))
    area = AsyncMock(return_value="Kitchen")
    monkeypatch.setattr(agent, "_get_current_area", area)
    user = SimpleNamespace(device_id="satellite")
    prompt = await agent._build_system_prompt_with_context(user)
    context = await agent._build_turn_context(user)
    if music_enabled:
        assert context == "Current area: Kitchen"
        area.assert_awaited_once()
    else:
        assert context == ""
        area.assert_not_awaited()
    assert "Current time:" not in prompt + context
