"""Tests for persisted MCP Assist chat logs."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.chat_log_manager import ChatLogManager


@pytest.mark.asyncio
async def test_chat_log_manager_records_lists_and_filters(hass) -> None:
    """Chat logs should be returned newest first and support common filters."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()

    first = await manager.async_record(
        {
            "created_at": "2026-06-01T00:00:00+00:00",
            "profile_entry_id": "profile-a",
            "conversation_id": "conv-a",
            "user_text": "turn on the light",
            "assistant_text": "Done.",
            "tools": [
                {
                    "name": "perform_action",
                    "arguments": {"entity_id": "light.kitchen"},
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                }
            ],
        }
    )
    second = await manager.async_record(
        {
            "created_at": "2026-06-01T00:01:00+00:00",
            "profile_entry_id": "profile-b",
            "conversation_id": "conv-b",
            "user_text": "what is open",
            "assistant_text": "The garage is open.",
            "tools": [],
        }
    )

    all_logs = await manager.async_list()
    filtered = await manager.async_list(profile_entry_id="profile-a")

    assert [item["id"] for item in all_logs] == [second["id"], first["id"]]
    assert [item["conversation_id"] for item in filtered] == ["conv-a"]


@pytest.mark.asyncio
async def test_chat_log_manager_prunes_and_clears(hass) -> None:
    """Chat log storage should stay bounded and be clearable."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()

    await manager.async_record({"created_at": "1", "conversation_id": "one"}, max_entries=2)
    await manager.async_record({"created_at": "2", "conversation_id": "two"}, max_entries=2)
    await manager.async_record({"created_at": "3", "conversation_id": "three"}, max_entries=2)

    remaining = await manager.async_list()
    deleted = await manager.async_clear(conversation_id="two")

    assert [item["conversation_id"] for item in remaining] == ["three", "two"]
    assert deleted == {"deleted_count": 1, "remaining_count": 1}
    assert [item["conversation_id"] for item in await manager.async_list()] == ["three"]


@pytest.mark.asyncio
async def test_chat_log_manager_bounds_large_values(hass) -> None:
    """Large strings should be truncated before storage."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()

    stored = await manager.async_record(
        {
            "created_at": "2026-06-01T00:00:00+00:00",
            "assistant_text": "x" * 13000,
        }
    )

    assert stored["assistant_text"].endswith("[truncated 1000 chars]")
