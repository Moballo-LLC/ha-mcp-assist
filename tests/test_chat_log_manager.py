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
async def test_chat_log_manager_projects_duplicate_tool_content(hass) -> None:
    """Response projections should omit payloads without mutating storage."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()
    await manager.async_record(
        {
            "created_at": "2026-06-01T00:00:00+00:00",
            "conversation_id": "conv-a",
            "user_text": "check the light",
            "assistant_text": "The light is on.",
            "tools": [
                {
                    "id": "call-1",
                    "name": "get_entity_details",
                    "started_at": "2026-06-01T00:00:01+00:00",
                    "completed_at": "2026-06-01T00:00:02+00:00",
                    "arguments": {"entity_id": "light.kitchen"},
                    "raw_arguments": '{"entity_id":"light.kitchen"}',
                    "result": {
                        "content": [{"type": "text", "text": "raw result"}]
                    },
                    "llm_content": "model-facing result",
                }
            ],
        }
    )

    raw_tool = (await manager.async_list(projection="raw"))[0]["tools"][0]
    model_tool = (await manager.async_list(projection="model"))[0]["tools"][0]
    compact_tool = (await manager.async_list(projection="compact"))[0]["tools"][0]
    full_tool = (await manager.async_list())[0]["tools"][0]

    assert "result" in raw_tool
    assert "llm_content" not in raw_tool
    assert "result" not in model_tool
    assert model_tool["llm_content"] == "model-facing result"
    assert compact_tool == {
        "id": "call-1",
        "name": "get_entity_details",
        "started_at": "2026-06-01T00:00:01+00:00",
        "completed_at": "2026-06-01T00:00:02+00:00",
        "argument_keys": ["entity_id"],
        "status": "ok",
    }
    assert "result" in full_tool
    assert "llm_content" in full_tool


def test_chat_log_manager_rejects_unknown_projection() -> None:
    """Unknown projections should not fall back to an unexpectedly large view."""
    with pytest.raises(ValueError, match="projection must be one of"):
        ChatLogManager.normalize_projection("everything")


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


@pytest.mark.asyncio
async def test_chat_log_manager_uses_delayed_save(hass, monkeypatch) -> None:
    """Recording should debounce writes via async_delay_save, not async_save."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()

    delay_calls: list[float] = []
    immediate_calls: list[object] = []
    monkeypatch.setattr(
        manager._store,
        "async_delay_save",
        lambda data_func, delay=0: delay_calls.append(delay),
    )

    async def _fail_immediate(_data):
        immediate_calls.append(_data)

    monkeypatch.setattr(manager._store, "async_save", _fail_immediate)

    await manager.async_record({"created_at": "1", "conversation_id": "one"})

    assert delay_calls and delay_calls[0] > 0
    assert immediate_calls == []


@pytest.mark.asyncio
async def test_chat_log_manager_handles_circular_references(hass) -> None:
    """A self-referential record must not blow the stack when normalized."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()

    payload: dict = {"created_at": "1", "conversation_id": "loop"}
    payload["self"] = payload  # cycle

    # Must not raise RecursionError; the cycle is replaced with a marker.
    stored = await manager.async_record(payload)

    assert "[circular reference]" in str(stored)


@pytest.mark.asyncio
async def test_chat_log_manager_bounds_deep_nesting(hass) -> None:
    """Deeply nested structures must be truncated, not recursed without limit."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()

    deep: dict = {}
    cursor = deep
    for _ in range(50):
        cursor["child"] = {}
        cursor = cursor["child"]

    stored = await manager.async_record({"created_at": "1", "data": deep})

    # Walk down and confirm a max-depth marker appears rather than 50 levels.
    text = str(stored)
    assert "max depth" in text


@pytest.mark.asyncio
async def test_chat_log_manager_clear_saves_immediately(hass, monkeypatch) -> None:
    """Clearing logs must flush synchronously, not via the debounced delay."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()
    await manager.async_record({"created_at": "1", "conversation_id": "one"})

    immediate_saves: list[dict] = []
    delayed_saves: list[float] = []

    async def _immediate(data):
        immediate_saves.append(data)

    monkeypatch.setattr(manager._store, "async_save", _immediate)
    monkeypatch.setattr(
        manager._store,
        "async_delay_save",
        lambda data_func, delay=0: delayed_saves.append(delay),
    )

    result = await manager.async_clear()

    assert result["deleted_count"] == 1
    assert immediate_saves and immediate_saves[0] == {"entries": []}
    assert delayed_saves == []


@pytest.mark.asyncio
async def test_chat_log_manager_shutdown_flushes_pending_save(hass, monkeypatch) -> None:
    """Shutdown must flush the current state synchronously before being dropped."""
    manager = ChatLogManager(hass)
    await manager.async_initialize()
    await manager.async_record({"created_at": "1", "conversation_id": "keep"})

    immediate_saves: list[dict] = []

    async def _immediate(data):
        immediate_saves.append(data)

    monkeypatch.setattr(manager._store, "async_save", _immediate)

    await manager.async_shutdown()

    assert immediate_saves
    assert len(immediate_saves[-1]["entries"]) == 1
