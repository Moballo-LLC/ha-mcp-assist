"""Speech regressions adapted from upstream v0.19."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from custom_components.mcp_assist.agent import MCPAssistConversationEntity


@pytest.mark.parametrize(("text", "expected"), [
    ("Answer. <think>unfinished reasoning", "Answer."),
    ("internal reasoning</think>Answer.", "Answer."),
    ("<|thought|>unfinished", ""),
    ("internal<|/thought|>Answer.", "Answer."),
    ("Answer. <THINK>", "Answer."),
    ("<think>internal</think>Answer.", "Answer."),
])
def test_incomplete_thoughts_are_not_spoken(hass, profile_entry_factory, text, expected):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    assert agent._strip_thinking_tags(text)[0] == expected


@pytest.mark.parametrize(("language", "expected"), [
    ("de", "21 Grad Celsius, 40 Prozent"),
    ("fr-CA", "21 degrés Celsius, 40 pour cent"),
    ("en", "21 degrees celsius, 40 percent"),
    ("xx", "21°C, 40%"),
])
def test_voice_symbols_follow_language(hass, profile_entry_factory, language, expected):
    hass.config.language = language
    agent = MCPAssistConversationEntity(
        hass, profile_entry_factory(options={"clean_responses": True})
    )
    assert agent._clean_text_for_tts("21°C, 40%") == expected


def test_voice_symbols_use_pipeline_language(hass, profile_entry_factory):
    hass.config.language = "de"
    agent = MCPAssistConversationEntity(
        hass, profile_entry_factory(options={"clean_responses": True})
    )
    assert agent._clean_text_for_tts("21°C", language="en") == "21 degrees celsius"
    assert agent._clean_text_for_tts("21°C", language="xx") == "21°C"


async def test_response_uses_voice_pipeline_language(hass, profile_entry_factory, monkeypatch):
    hass.config.language = "de"
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(options={
        "clean_responses": True, "follow_up_mode": "none",
    }))
    monkeypatch.setattr(agent, "_execute_actions", AsyncMock(return_value=[]))
    result = await agent._build_response_result(
        "21°C", SimpleNamespace(text="Temperature?", language="en"), "chat-1"
    )
    assert result.response.speech["plain"]["speech"] == "21 degrees celsius"
