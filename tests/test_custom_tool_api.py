"""Tests for the public custom tool helper API."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from custom_components.mcp_assist import custom_tool_api as custom_tool_api_module
from custom_components.mcp_assist.const import DOMAIN
from custom_components.mcp_assist.custom_tool_api import (
    analyze_image,
    async_recorder_query,
    entity_snapshot,
    format_datetime,
    format_entity_state,
    format_relative_time,
    mcp_error_result,
    mcp_image_result,
    mcp_json_result,
    mcp_text_result,
)


def test_result_builders_return_standard_mcp_payloads() -> None:
    """Helper result builders should keep custom tools on the MCP result shape."""
    assert mcp_text_result("ok") == {
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
    }
    assert mcp_error_result("nope") == {
        "content": [{"type": "text", "text": "nope"}],
        "isError": True,
    }

    json_result = mcp_json_result({"status": "ok"})
    assert json_result["isError"] is False
    assert json_result["structuredContent"] == {"status": "ok"}
    assert '"status": "ok"' in json_result["content"][0]["text"]

    image_result = mcp_image_result(b"image-bytes", "image/png", text="image")
    assert image_result["content"][0] == {"type": "text", "text": "image"}
    assert image_result["content"][1]["type"] == "image"
    assert image_result["content"][1]["mimeType"] == "image/png"


def test_entity_and_time_helpers_format_tool_friendly_values(hass) -> None:
    """Entity/time helpers should provide compact stable text and snapshots."""
    hass.states.async_set(
        "sensor.example_temperature",
        "72",
        {"friendly_name": "Example Temperature", "unit_of_measurement": "F"},
    )

    assert format_entity_state(hass, "sensor.example_temperature") == (
        "Example Temperature: 72 F"
    )
    assert entity_snapshot(hass, "sensor.example_temperature")["state"] == "72"
    assert entity_snapshot(hass, "sensor.missing") is None
    assert format_datetime(datetime(2026, 6, 1, 12, 30, tzinfo=timezone.utc)).startswith(
        "2026-06-01T"
    )
    assert (
        format_relative_time(
            datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
            now=datetime(2026, 6, 1, 12, 5, tzinfo=timezone.utc),
        )
        == "5 minutes ago"
    )


@pytest.mark.asyncio
async def test_analyze_image_helper_preserves_call_context(hass) -> None:
    """Image helpers should compose through core MCP tools with supplied context."""
    captured: dict[str, object] = {}

    class _StubServer:
        async def handle_tool_call(self, params):
            captured.update(params)
            return {
                "content": [{"type": "text", "text": "image answer"}],
                "isError": False,
            }

    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = _StubServer()

    result = await analyze_image(
        hass,
        question="What is here?",
        camera_entity_id="camera.example",
        include_image=True,
        context={"profile_entry_id": "profile-1"},
    )

    assert result["content"][0]["text"] == "image answer"
    assert captured["name"] == "analyze_image"
    assert captured["arguments"] == {
        "camera_entity_id": "camera.example",
        "question": "What is here?",
        "detail": "auto",
        "include_image": True,
    }
    assert captured["context"] == {"profile_entry_id": "profile-1"}


@pytest.mark.asyncio
async def test_recorder_query_helper_uses_recorder_executor(monkeypatch, hass) -> None:
    """Recorder helpers should run read-only SQL inside the recorder executor."""
    executor_calls = 0

    class _Rows:
        def mappings(self):
            return self

        def fetchmany(self, count):
            assert count == 1
            return [{"state": "on"}]

        def all(self):
            pytest.fail("limit should be applied before fetching all rows")

    class _Session:
        def execute(self, statement, parameters):
            assert str(statement).startswith("SELECT")
            assert parameters == {"entity_id": "light.example"}
            return _Rows()

    class _RecorderInstance:
        async def async_add_executor_job(self, job):
            nonlocal executor_calls
            executor_calls += 1
            return job()

    @contextmanager
    def _session_scope(**kwargs):
        assert kwargs["hass"] is hass
        assert kwargs["read_only"] is True
        yield _Session()

    monkeypatch.setattr(
        "homeassistant.helpers.recorder.get_instance",
        lambda target_hass: _RecorderInstance(),
    )
    monkeypatch.setattr(
        "homeassistant.helpers.recorder.session_scope",
        _session_scope,
    )

    rows = await async_recorder_query(
        hass,
        "SELECT state FROM states WHERE entity_id = :entity_id",
        {"entity_id": "light.example"},
        limit=1,
    )

    assert rows == [{"state": "on"}]
    assert executor_calls == 1


@pytest.mark.asyncio
async def test_recorder_job_helper_supports_component_recorder_imports(
    monkeypatch,
    hass,
) -> None:
    """Recorder helpers should also work on HA releases before helpers.recorder."""
    executor_calls = 0

    class _RecorderInstance:
        async def async_add_executor_job(self, job):
            nonlocal executor_calls
            executor_calls += 1
            return job()

    class _Session:
        pass

    @contextmanager
    def _session_scope(**kwargs):
        assert kwargs["hass"] is hass
        assert kwargs["read_only"] is True
        yield _Session()

    def _import_module(name):
        if name == "homeassistant.helpers.recorder":
            raise ImportError("helpers recorder is not available")
        if name == "homeassistant.components.recorder":
            return SimpleNamespace(get_instance=lambda target_hass: _RecorderInstance())
        if name == "homeassistant.components.recorder.util":
            return SimpleNamespace(session_scope=_session_scope)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(custom_tool_api_module.importlib, "import_module", _import_module)

    result = await custom_tool_api_module.async_run_recorder_job(
        hass,
        lambda session: session.__class__.__name__,
    )

    assert result == "_Session"
    assert executor_calls == 1


@pytest.mark.asyncio
async def test_recorder_query_helper_rejects_writes(hass) -> None:
    """Recorder query helper should only expose the read-only path."""
    with pytest.raises(ValueError, match="read-only"):
        await async_recorder_query(hass, "DELETE FROM states")


@pytest.mark.asyncio
async def test_recorder_query_helper_rejects_writes_hidden_behind_ctes(hass) -> None:
    """Writable statements after or inside CTEs should not pass as read-only."""
    blocked_sql = [
        "WITH stale AS (SELECT 1) DELETE FROM states WHERE entity_id = :entity_id",
        "WITH deleted AS (DELETE FROM states RETURNING *) SELECT * FROM deleted",
        "WITH stale AS (SELECT 1) UPDATE states SET state = 'off'",
    ]

    for sql in blocked_sql:
        with pytest.raises(ValueError, match="read-only"):
            await async_recorder_query(hass, sql, {"entity_id": "light.example"})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT state INTO OUTFILE '/tmp/mcp_assist_states.txt' FROM states",
        "SELECT state INTO DUMPFILE '/tmp/mcp_assist_states.bin' FROM states",
        "SELECT state FROM states INTO OUTFILE '/tmp/mcp_assist_states.txt'",
        "SELECT state /*!50000 INTO OUTFILE '/tmp/mcp_assist_states.txt' */ FROM states",
        "SELECT state INTO /*!50000 OUTFILE */ '/tmp/mcp_assist_states.txt' FROM states",
        "SELECT state /*!50000 INTO */ OUTFILE '/tmp/mcp_assist_states.txt' FROM states",
        """
        WITH exported AS (
            SELECT state INTO OUTFILE '/tmp/mcp_assist_states.txt' FROM states
        )
        SELECT * FROM exported
        """,
    ],
)
def test_read_only_sql_validator_rejects_select_file_outputs(sql: str) -> None:
    """MySQL/MariaDB SELECT file-output clauses are write-capable."""
    assert not custom_tool_api_module._is_read_only_sql(sql)


def test_read_only_sql_validator_allows_select_ctes_and_ignores_literals() -> None:
    """Safe SELECT CTEs should still be valid, even with write words in strings."""
    assert custom_tool_api_module._is_read_only_sql(
        """
        WITH recent AS (
            SELECT state FROM states WHERE entity_id = :entity_id
        )
        SELECT * FROM recent
        """
    )
    assert custom_tool_api_module._is_read_only_sql(
        "SELECT 'delete from states' AS example_text FROM states"
    )
    assert custom_tool_api_module._is_read_only_sql(
        "SELECT 'INTO OUTFILE /tmp/not-written' AS example_text FROM states"
    )
    assert custom_tool_api_module._is_read_only_sql(
        "SELECT state /* INTO OUTFILE '/tmp/not-written' */ FROM states"
    )
    assert custom_tool_api_module._is_read_only_sql(
        "SELECT /*!50000 'DELETE FROM states' AS example_text, */ state FROM states"
    )
