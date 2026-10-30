"""Speech regressions adapted from upstream v0.19."""

import pytest

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
