"""Home Assistant ChatLog ownership across overlapping conversation requests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from tests.test_model_profiles import user_input


@pytest.mark.parametrize("second_outcome", ["success", "error", "cancel"])
async def test_overlapping_conversations_keep_chat_logs(
    hass, profile_entry_factory, monkeypatch, second_outcome,
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    monkeypatch.setattr(agent, "_execute_actions", AsyncMock(return_value=[]))
    second_started = asyncio.Event()
    first_finished = asyncio.Event()
    logs = {name: SimpleNamespace(
        conversation_id=name, async_add_assistant_content_without_tools=Mock(),
    ) for name in ("first", "second")}

    async def handle(user, conversation_id):
        if conversation_id == "first":
            await second_started.wait()
            if second_outcome != "success":
                await first_finished.wait()
        else:
            second_started.set()
            if second_outcome == "error":
                raise RuntimeError("Example request failure")
            if second_outcome == "cancel":
                raise asyncio.CancelledError()
            await first_finished.wait()
        agent._record_tool_calls_to_chatlog([{
            "id": conversation_id, "function": {
                "name": "discover_entities", "arguments": {"query": conversation_id},
            },
        }])
        agent._record_tool_result_to_chatlog(
            conversation_id, "discover_entities", {"text": conversation_id},
        )
        return await agent._build_response_result(conversation_id, user, conversation_id)

    monkeypatch.setattr(agent, "_async_handle_message_inner", handle)
    first = asyncio.create_task(agent._async_handle_message(user_input(), logs["first"]))
    second = asyncio.create_task(agent._async_handle_message(user_input(), logs["second"]))
    try:
        if second_outcome == "success":
            await asyncio.wait_for(first, timeout=5)
            first_finished.set()
            await asyncio.wait_for(second, timeout=5)
        else:
            with pytest.raises(RuntimeError if second_outcome == "error" else asyncio.CancelledError):
                await asyncio.wait_for(second, timeout=5)
            first_finished.set()
            await asyncio.wait_for(first, timeout=5)
    finally:
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
    first_items = [call.args[0] for call in logs["first"].async_add_assistant_content_without_tools.call_args_list]
    assert len(first_items) == 3
    assert first_items[0].tool_calls[0].id == "first"
    assert first_items[1].tool_call_id == "first"
    assert first_items[2].content == "first"
    second_items = [call.args[0] for call in logs["second"].async_add_assistant_content_without_tools.call_args_list]
    if second_outcome == "success":
        assert len(second_items) == 3
        assert second_items[0].tool_calls[0].id == "second"
        assert second_items[1].tool_call_id == "second"
        assert second_items[2].content == "second"
    else:
        assert second_items == []
    assert agent._current_chat_log is None


async def test_nested_conversations_restore_outer_chat_log(
    hass, profile_entry_factory, monkeypatch,
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    other = MCPAssistConversationEntity(hass, profile_entry_factory())
    monkeypatch.setattr(agent, "_execute_actions", AsyncMock(return_value=[]))
    logs = {name: SimpleNamespace(
        conversation_id=name, async_add_assistant_content_without_tools=Mock(),
    ) for name in ("outer", "inner")}

    async def handle(user, conversation_id):
        assert agent._current_chat_log is logs[conversation_id]
        assert other._current_chat_log is None
        other._record_tool_result_to_chatlog("unrelated", "discover_entities", {})
        if conversation_id == "outer":
            await agent._async_handle_message(user, logs["inner"])
            assert agent._current_chat_log is logs["outer"]
        return await agent._build_response_result(conversation_id, user, conversation_id)

    monkeypatch.setattr(agent, "_async_handle_message_inner", handle)
    await agent._async_handle_message(user_input(), logs["outer"])
    for name, log in logs.items():
        log.async_add_assistant_content_without_tools.assert_called_once()
        assert log.async_add_assistant_content_without_tools.call_args.args[0].content == name
    assert agent._current_chat_log is None
