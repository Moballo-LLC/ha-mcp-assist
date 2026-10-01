"""Generic contracts for recent statistics, comparisons, and source evidence."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.mcp_assist.mcp_server import MCPServer
from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.tools.packages.recorder import history
from custom_components.mcp_assist.tools.packages.recorder.coverage import source_coverage

ENTITY = "sensor.example_energy"
START = datetime(2026, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def statistics_server(hass, profile_entry_factory, monkeypatch):
    """Use Recorder's executor boundary without opening a database."""
    server = MCPServer(hass, 8099, profile_entry_factory())
    jobs = []

    class Recorder:
        async def async_add_executor_job(self, job):
            jobs.append(job)
            return job()

    monkeypatch.setattr(history, "get_recorder_instance", lambda _: Recorder())
    monkeypatch.setattr(history, "async_should_expose", lambda *_: True)
    monkeypatch.setattr(history.recorder_statistics, "get_metadata", Mock(return_value={
        ENTITY: (1, {"has_sum": True, "has_mean": True, "unit_of_measurement": "kWh"})
    }))
    monkeypatch.setattr(history.recorder_statistics, "get_display_unit", Mock(return_value="kWh"))
    query = Mock(return_value={ENTITY: []})
    monkeypatch.setattr(history.recorder_statistics, "statistics_during_period", query)
    return server, query, jobs


def arguments(start=START, end=START + timedelta(hours=1), **extra):
    return {"entity_id": ENTITY, "period": "custom", "start_datetime": start.isoformat(),
            "end_datetime": end.isoformat(), "bucket": "5minute", "metric": "change", **extra}


def rows(start=START, count=12, value=1, seconds=300):
    return [{"start": start + timedelta(seconds=seconds * index), "change": value}
            for index in range(count)]


@pytest.mark.asyncio
async def test_recent_native_query_and_summary_ignore_display_limit(statistics_server):
    server, query, jobs = statistics_server
    query.return_value = {ENTITY: rows()}
    result = await server.tool_get_entity_statistics(arguments(bucket="auto", limit=1))
    data = result["structuredContent"]
    assert query.call_args.args[1:5] == (START, START + timedelta(hours=1), {ENTITY}, "5minute")
    assert len(jobs) == 3
    assert data["schema_version"] == 1 and data["available"]
    assert data["summary"]["change"] == 12
    assert data["coverage"]["complete"]
    assert data["bucket_count"] == 12 and len(data["buckets"]) == 1 and data["truncated"]


@pytest.mark.asyncio
async def test_alignment_reports_partial_window(statistics_server):
    server, query, _ = statistics_server
    query.return_value = {ENTITY: rows(START + timedelta(minutes=5), count=10)}
    data = (await server.tool_get_entity_statistics(arguments(
        start=START + timedelta(minutes=2), end=START + timedelta(minutes=58)
    )))["structuredContent"]
    assert data["effective_window"] == {"start": (START + timedelta(minutes=5)).isoformat(),
                                        "end": (START + timedelta(minutes=55)).isoformat()}
    assert data["coverage"]["missing_buckets"] == 0
    assert not data["coverage"]["complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hours, resolution, grouping", [
    (12, "5minute", "5minute"), (13, "hour", "hour"),
    (72, "hour", "hour"), (73, "hour", "day"),
])
async def test_auto_resolution_thresholds(statistics_server, hours, resolution, grouping):
    server, query, _ = statistics_server
    data = (await server.tool_get_entity_statistics(arguments(
        end=START + timedelta(hours=hours), bucket="auto"
    )))["structuredContent"]
    assert query.call_args.args[4] == resolution
    assert data["bucket"] == grouping


@pytest.mark.asyncio
async def test_five_minute_cap_applies_per_interval(statistics_server):
    server, query, jobs = statistics_server
    result = await server.tool_get_entity_statistics(arguments(
        end=START + timedelta(hours=73), compare_previous=True
    ))
    assert result["isError"] and not jobs
    query.assert_not_called()
    await server.tool_get_entity_statistics(arguments(
        end=START + timedelta(hours=72), compare_previous=True
    ))
    assert query.call_args.args[2] - query.call_args.args[1] == timedelta(hours=144)


@pytest.mark.asyncio
@pytest.mark.parametrize("previous, percent", [(0, None), (-1, None), (1, 100)])
async def test_comparison_reuses_one_query_and_units(statistics_server, previous, percent):
    server, query, jobs = statistics_server
    previous_start = START - timedelta(hours=1)
    query.return_value = {ENTITY: rows(previous_start, value=previous) + rows(value=2)}
    data = (await server.tool_get_entity_statistics(arguments(compare_previous=True)))["structuredContent"]
    query.assert_called_once()
    assert len(jobs) == 3
    assert query.call_args.args[1] == previous_start
    comparison = data["comparison"]
    assert comparison["comparable"]
    assert comparison["previous"]["requested_window"]["end"] == START.isoformat()
    assert comparison["differences"]["change"] == {"absolute": 24 - previous * 12, "percent": percent}


@pytest.mark.asyncio
async def test_metric_gaps_and_untrusted_rows_suppress_comparison(statistics_server):
    server, query, _ = statistics_server
    current = rows()
    current[0]["change"] = float("nan")
    current[1]["change"] = True
    query.return_value = {ENTITY: rows(START - timedelta(hours=1)) + current + [
        current[2], {"start": START + timedelta(hours=1), "change": 1000},
        {"start": START + timedelta(seconds=1), "change": 1000},
    ]}
    data = (await server.tool_get_entity_statistics(arguments(compare_previous=True)))["structuredContent"]
    assert data["summary"]["change"] == 9
    assert not data["coverage"]["per_metric"]["change"]["complete"]
    assert data["coverage"]["per_metric"]["change"]["missing_buckets"] == 3
    assert not data["comparison"]["comparable"]
    assert data["comparison"]["differences"] == {}


@pytest.mark.asyncio
async def test_short_term_absence_is_unavailable_not_zero(statistics_server):
    server, _, _ = statistics_server
    result = await server.tool_get_entity_statistics(arguments())
    assert result["structuredContent"]["available"] is False
    assert "summary" not in result["structuredContent"]
    assert "five-minute" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [50, 100])
@pytest.mark.parametrize("uncertain", [False, True])
async def test_comparison_and_coverage_survive_model_compaction(
    statistics_server, profile_entry_factory, limit, uncertain,
):
    server, query, _ = statistics_server
    if uncertain:
        server.hass.states.async_set(ENTITY, "1", {"recorder_coverage": {
            "estimated_periods": {"months": ["2026-01"]},
        }})
    samples = rows(START - timedelta(hours=72), count=144, seconds=3600, value=123456.123456)
    for row in samples:
        row.update(mean=123.456789, min=100.123456, max=150.234567)
    query.return_value = {ENTITY: samples}
    result = await server.tool_get_entity_statistics(arguments(
        end=START + timedelta(hours=72), bucket="auto", metric="all",
        compare_previous=True, limit=limit,
    ))
    agent = MCPAssistConversationEntity(server.hass, profile_entry_factory())
    formatted = agent._format_tool_result_for_llm("get_entity_statistics", result)
    assert "Tool result truncated for model context" in formatted
    assert "Previous equal-length interval" in formatted
    if uncertain:
        assert "Source-reported coverage is uncertain" in formatted
        assert "Comparison unavailable: incomplete coverage or source-reported uncertainty." in formatted
    else:
        assert "Comparison differences:" in formatted


@pytest.mark.asyncio
async def test_scope_denial_precedes_state_and_database(statistics_server, monkeypatch):
    server, query, jobs = statistics_server
    monkeypatch.setattr(history, "async_should_expose", lambda *_: False)
    with monkeypatch.context() as guard:
        guard.setattr(type(server.hass.states), "get", Mock(side_effect=AssertionError("state accessed")))
        result = await server.tool_get_entity_statistics(arguments())
    assert result["isError"]
    assert not jobs
    query.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("annotation", [
    {"started_at": "2026-01-02T00:10:00+00:00"},
    {"estimated_periods": {"days": ["2026-01-01", "2026-01-02"]}},
    {"counter_resets": True},
    {"started_at": "2026-01-01T00:00:00"},
    {"estimated_periods": {"months": ["2026-13"]}},
])
async def test_source_uncertainty_blocks_otherwise_complete_comparison(statistics_server, annotation):
    server, query, _ = statistics_server
    server.hass.states.async_set(ENTITY, "1", {"recorder_coverage": annotation})
    query.return_value = {ENTITY: rows(START - timedelta(hours=1)) + rows()}
    data = (await server.tool_get_entity_statistics(arguments(compare_previous=True)))["structuredContent"]
    assert data["coverage"]["complete"]
    assert data["source_coverage"]["uncertain"]
    assert not data["comparison"]["comparable"]


def test_source_annotations_project_local_dates_with_exclusive_end():
    old_zone = dt_util.DEFAULT_TIME_ZONE
    try:
        dt_util.set_default_time_zone(ZoneInfo("America/New_York"))
        start = datetime(2026, 1, 2, 5, tzinfo=timezone.utc)
        data = source_coverage({"recorder_coverage": {
            "counter_resets": 2, "note": "Imported records", "private_key": "omit",
            "estimated_periods": {"days": ["2026-01-01", "2026-01-02", "2026-01-03"],
                                  "months": ["2025-12", "2026-01", "2026-02"]},
        }}, start, start + timedelta(days=1))
        assert data["estimated_periods"] == {"days": ["2026-01-02"], "months": ["2026-01"]}
        assert "private_key" not in data
        assert data["counter_resets"] == 2 and data["uncertain"]
    finally:
        dt_util.set_default_time_zone(old_zone)


@pytest.mark.parametrize("annotation", [
    {"estimated_periods": {"days": ["2026-01-01"] * 401}},
    {"estimated_periods": {"months": ["2026-01"] * 25}},
    {"note": "x" * 1201},
    {"estimated_periods": {"unknown_before": "20260101"}},
])
def test_unbounded_or_noncanonical_annotations_do_not_leak(annotation):
    result = source_coverage({"recorder_coverage": annotation}, START, START + timedelta(hours=1))
    assert result == {
        "available": False, "uncertain": True,
        "reason": "Source coverage annotations are malformed or exceed their bounds.",
    }


def test_rolling_duration_and_complete_day_average_across_dst(statistics_server, monkeypatch):
    server, _, _ = statistics_server
    old_zone = dt_util.DEFAULT_TIME_ZONE
    zone = ZoneInfo("America/New_York")
    try:
        dt_util.set_default_time_zone(zone)
        monkeypatch.setattr(history.dt_util, "now", lambda: datetime(2026, 3, 9, 0, tzinfo=zone))
        start, end, _ = server._statistics_window({}, "last_24_hours")
        assert end - start == timedelta(hours=24)
        day_start = datetime(2026, 3, 8, 5, tzinfo=timezone.utc)
        day_end = datetime(2026, 3, 9, 4, tzinfo=timezone.utc)
        changes = {row["start"]: row["change"] for row in rows(day_start, count=23, seconds=3600)}
        assert server._complete_statistic_period_totals(changes, day_start, day_end, "day") == [23]
        changes.pop(day_start)
        assert server._complete_statistic_period_totals(changes, day_start, day_end, "day") == []
    finally:
        dt_util.set_default_time_zone(old_zone)
