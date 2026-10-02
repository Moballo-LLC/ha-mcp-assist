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


def with_comparison_baselines(samples, current_start, previous_start=None, seconds=300):
    """Add finite sum evidence and exact predecessors, preserving malformed rows."""
    prepared = [dict(row, sum=row["start"].timestamp() / seconds) for row in samples]
    observed = {row["start"] for row in prepared}
    for start in (previous_start, current_start):
        if start is None:
            continue
        predecessor = start - timedelta(seconds=seconds)
        if predecessor not in observed:
            prepared.append({"start": predecessor, "sum": predecessor.timestamp() / seconds})
            observed.add(predecessor)
    return prepared


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_boundary", [False, True])
@pytest.mark.parametrize("compare_previous", [False, True])
async def test_native_change_baselines_do_not_bridge_excluded_boundary(
    statistics_server, monkeypatch, missing_boundary, compare_previous,
):
    """Exercise HA's actual change augmentation across an off-grid split."""
    server, query, _ = statistics_server
    now = datetime(2026, 1, 3, 12, 2, 17, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(history.dt_util, "now", lambda: now)
    baseline_start = datetime(2026, 1, 3, 10, tzinfo=timezone.utc)
    current_start = baseline_start + timedelta(hours=1, minutes=5)
    statistics = history.recorder_statistics
    monkeypatch.setattr(statistics, "get_instance", lambda _: Mock())
    monkeypatch.setattr(statistics, "_statistics_at_time", Mock(return_value=[]))
    native_result = {ENTITY: [
        {"start": baseline_start + timedelta(minutes=index * 5), "sum": 100 + index}
        for index in range(24)
        if not missing_boundary or index != 12
    ]}
    statistics._augment_result_with_change(
        server.hass, Mock(), baseline_start, None, {"change", "sum"},
        statistics.StatisticsShortTerm, {ENTITY: (1, {"statistic_id": ENTITY})}, native_result,
    )
    first_current = next(row for row in native_result[ENTITY] if row["start"] == current_start)
    assert first_current["change"] == (2 if missing_boundary else 1)
    query.side_effect = lambda *args: {ENTITY: [
        row for row in native_result[ENTITY] if args[1] <= row["start"] < args[2]
    ]}
    call = {
        "entity_id": ENTITY, "period": "last_hour", "bucket": "auto",
        "metric": "change",
    }
    if compare_previous:
        call["compare_previous"] = True
    data = (await server.tool_get_entity_statistics(call))["structuredContent"]
    assert query.call_args.args[1] == (baseline_start if compare_previous else current_start - timedelta(minutes=5))
    assert query.call_args.args[-1] == {"change", "sum"}
    if missing_boundary:
        assert data["summary"]["change"] == 10
        assert data["coverage"]["per_metric"]["change"]["observed_buckets"] == 10
        assert data["coverage"]["effective_complete"] is False
    else:
        assert data["summary"]["change"] == 11
        assert data["coverage"]["effective_complete"] is True
    analyses = [data]
    if compare_previous:
        comparison = data["comparison"]
        assert comparison["change_baselines_available"] == {"current": not missing_boundary, "previous": True}
        assert comparison["previous"]["summary"]["change"] == 11
        if missing_boundary:
            assert comparison["comparable"] is False
            assert comparison["differences"] == {}
            assert "immediate predecessor" in comparison["reason"].casefold()
        else:
            assert comparison["comparable"] is True
            assert comparison["differences"]["change"] == {"absolute": 0, "percent": 0}
        analyses.append(comparison["previous"])
    else:
        assert "comparison" not in data
    for analysis in analyses:
        assert "sum" not in analysis["summary"]
        assert all("sum" not in bucket for bucket in analysis["buckets"])


@pytest.mark.asyncio
@pytest.mark.parametrize("compare_previous, period, slot, invalid", [
    (True, "previous", "predecessor", "missing"),
    (True, "previous", "predecessor", None),
    (True, "previous", "start", float("nan")),
    (True, "current", "predecessor", float("inf")),
    (True, "current", "start", None),
    (False, "current", "predecessor", "missing"),
    (False, "current", "predecessor", None),
    (False, "current", "start", float("nan")),
    (False, "current", "predecessor", float("inf")),
    (False, "current", "start", None),
])
async def test_finite_mocked_change_does_not_override_invalid_sum_baseline(
    statistics_server, compare_previous, period, slot, invalid,
):
    server, query, _ = statistics_server
    previous_start = START - timedelta(hours=1)
    samples = with_comparison_baselines(rows(previous_start) + rows(value=2), START, previous_start)
    target = START if period == "current" else previous_start
    if slot == "predecessor":
        target -= timedelta(minutes=5)
    if invalid == "missing":
        samples = [row for row in samples if row["start"] != target]
    else:
        next(row for row in samples if row["start"] == target)["sum"] = invalid
    query.return_value = {ENTITY: samples}
    call = arguments(**({"compare_previous": True} if compare_previous else {}))
    data = (await server.tool_get_entity_statistics(call))["structuredContent"]
    affected = data
    if compare_previous:
        comparison = data["comparison"]
        assert comparison["change_baselines_available"][period] is False
        assert comparison["comparable"] is False
        assert comparison["differences"] == {}
        assert "immediate predecessor" in comparison["reason"].casefold()
        affected = data if period == "current" else comparison["previous"]
    else:
        assert "comparison" not in data
    missing = 2 if slot == "start" else 1
    assert affected["coverage"]["per_metric"]["change"]["missing_buckets"] == missing
    assert affected["summary"]["change"] == (12 - missing) * (2 if period == "current" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("compare_previous", [False, True])
async def test_internal_sum_alone_does_not_establish_available_change(statistics_server, compare_previous):
    server, query, _ = statistics_server
    previous_start = START - timedelta(hours=1)
    samples = with_comparison_baselines(rows(previous_start) + rows(), START, previous_start)
    for row in samples:
        if row["start"] >= START:
            row.pop("change", None)
    query.return_value = {ENTITY: samples}
    result = await server.tool_get_entity_statistics(arguments(
        **({"compare_previous": True} if compare_previous else {})
    ))
    assert result["structuredContent"]["available"] is False
    assert "summary" not in result["structuredContent"]
    assert "no finite statistic values" in result["content"][0]["text"].casefold()


@pytest.mark.asyncio
@pytest.mark.parametrize("compare_previous", [False, True])
@pytest.mark.parametrize("fault, missing_metric, missing_rows, bridged_delta", [
    ("interior_missing", 2, 1, 2),
    ("delayed_first", 2, 1, 3),
    ("consecutive_none", 3, 0, 3),
])
async def test_native_interior_gaps_never_inflate_summary_or_calendar_average(
    statistics_server, monkeypatch, compare_previous, fault, missing_metric, missing_rows, bridged_delta,
):
    """HA bridges absent sums; only immediate source-slot deltas are reportable."""
    server, query, _ = statistics_server
    old_zone = dt_util.DEFAULT_TIME_ZONE
    try:
        dt_util.set_default_time_zone(timezone.utc)
        statistics = history.recorder_statistics
        monkeypatch.setattr(statistics, "get_instance", lambda _: Mock())
        monkeypatch.setattr(statistics, "_statistics_at_time", Mock(return_value=[]))
        baseline_start = START - timedelta(hours=25)
        samples = [{"start": baseline_start + timedelta(hours=index), "sum": 100 + index}
                   for index in range(49)]
        if fault == "interior_missing":
            samples = [row for row in samples if row["start"] != START + timedelta(hours=6)]
            bridge_start = START + timedelta(hours=7)
        elif fault == "delayed_first":
            samples = [row for row in samples if row["start"] not in {START - timedelta(hours=1), START}]
            bridge_start = START + timedelta(hours=1)
        else:
            for row in samples:
                if row["start"] in {START + timedelta(hours=6), START + timedelta(hours=7)}:
                    row["sum"] = None
            bridge_start = START + timedelta(hours=8)
        native_result = {ENTITY: samples}
        statistics._augment_result_with_change(
            server.hass, Mock(), baseline_start, None, {"change", "sum"},
            statistics.Statistics, {ENTITY: (1, {"statistic_id": ENTITY})}, native_result,
        )
        assert next(row for row in samples if row["start"] == bridge_start)["change"] == bridged_delta
        query.side_effect = lambda *args: {ENTITY: [
            row for row in samples if args[1] <= row["start"] < args[2]
        ]}
        data = (await server.tool_get_entity_statistics(arguments(
            end=START + timedelta(days=1), bucket="day",
            **({"compare_previous": True} if compare_previous else {}),
        )))["structuredContent"]
        assert data["summary"]["change"] == 24 - missing_metric
        assert data["summary"]["complete_day_count"] == 0
        assert data["summary"]["average_per_complete_day"] is None
        assert data["coverage"]["missing_buckets"] == missing_rows
        assert data["coverage"]["per_metric"]["change"]["missing_buckets"] == missing_metric
        assert data["coverage"]["effective_complete"] is False
        assert data["buckets"] == [{
            "start": "2026-01-02", "sample_count": 24 - missing_rows, "change": 24 - missing_metric,
        }]
        if compare_previous:
            assert data["comparison"]["comparable"] is False
            assert data["comparison"]["differences"] == {}
        else:
            assert "comparison" not in data
    finally:
        dt_util.set_default_time_zone(old_zone)


@pytest.mark.asyncio
async def test_recent_native_query_and_summary_ignore_display_limit(statistics_server):
    server, query, jobs = statistics_server
    query.return_value = {ENTITY: with_comparison_baselines(rows(), START)}
    result = await server.tool_get_entity_statistics(arguments(bucket="auto", limit=1))
    data = result["structuredContent"]
    assert query.call_args.args[1:5] == (START - timedelta(minutes=5), START + timedelta(hours=1), {ENTITY}, "5minute")
    assert query.call_args.args[-1] == {"change", "sum"}
    assert len(jobs) == 3
    assert data["schema_version"] == 1 and data["available"]
    assert data["summary"]["change"] == 12
    assert data["coverage"]["complete"]
    assert data["bucket_count"] == 12 and len(data["buckets"]) == 1 and data["truncated"]


@pytest.mark.asyncio
async def test_alignment_reports_partial_window(statistics_server):
    server, query, _ = statistics_server
    query.return_value = {ENTITY: with_comparison_baselines(
        rows(START + timedelta(minutes=5), count=10), START + timedelta(minutes=5),
    )}
    data = (await server.tool_get_entity_statistics(arguments(
        start=START + timedelta(minutes=2), end=START + timedelta(minutes=58)
    )))["structuredContent"]
    assert data["effective_window"] == {"start": (START + timedelta(minutes=5)).isoformat(),
                                        "end": (START + timedelta(minutes=55)).isoformat()}
    assert data["coverage"]["missing_buckets"] == 0
    assert not data["coverage"]["complete"]
    assert data["coverage"]["effective_complete"]
    assert data["coverage"]["per_metric"]["change"]["effective_complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("period, bucket, hours, seconds, count", [
    ("last_hour", "auto", 1, 300, 11),
    ("last_24_hours", "auto", 24, 3600, 23),
    ("last_24_hours", "5minute", 24, 300, 287),
])
async def test_off_grid_rolling_comparison_uses_only_complete_effective_windows(
    statistics_server, profile_entry_factory, monkeypatch, period, bucket, hours, seconds, count,
):
    """Real clock seconds must not disable an honest comparison of retained buckets."""
    server, query, jobs = statistics_server
    now = datetime(2026, 1, 3, 12, 2, 17, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(history.dt_util, "now", lambda: now)
    requested_start = now - timedelta(hours=hours)
    effective_end = now.replace(minute=0, second=0, microsecond=0)
    effective_start = effective_end - timedelta(seconds=count * seconds)
    previous_start = effective_start - timedelta(hours=hours)
    previous_end = effective_end - timedelta(hours=hours)
    query.return_value = {ENTITY: with_comparison_baselines(
        rows(previous_start, count=count, seconds=seconds) +
        rows(effective_start, count=count, seconds=seconds, value=2),
        effective_start, previous_start, seconds,
    )}
    result = await server.tool_get_entity_statistics({
        "entity_id": ENTITY, "period": period, "bucket": bucket,
        "metric": "change", "compare_previous": True, "limit": 1,
    })
    data = result["structuredContent"]
    comparison = data["comparison"]
    assert data["requested_window"] == {"start": requested_start.isoformat(), "end": now.isoformat()}
    assert data["effective_window"] == {"start": effective_start.isoformat(), "end": effective_end.isoformat()}
    assert comparison["previous"]["effective_window"] == {
        "start": previous_start.isoformat(), "end": previous_end.isoformat(),
    }
    assert comparison["comparable"]
    assert comparison["scope"] == "effective_windows"
    assert comparison["boundaries_aligned"] is False
    assert comparison["effective_duration_seconds"] == count * seconds
    assert comparison["qualification"] == (
        "Comparison covers effective windows only; requested boundary fragments are excluded."
    )
    for coverage in (data["coverage"], comparison["previous"]["coverage"]):
        assert coverage["complete"] is False
        assert coverage["effective_complete"] is True
        assert coverage["per_metric"]["change"]["complete"] is False
        assert coverage["per_metric"]["change"]["effective_complete"] is True
    assert comparison["differences"]["change"] == {"absolute": count, "percent": 100}
    assert data["summary"]["change"] == 2 * count
    assert data["truncated"] and len(data["buckets"]) == 1
    query.assert_called_once()
    assert len(jobs) == 3
    assert query.call_args.args[1:3] == (previous_start - timedelta(seconds=seconds), effective_end)
    assert query.call_args.args[-1] == {"change", "sum"}
    agent = MCPAssistConversationEntity(server.hass, profile_entry_factory())
    text = agent._format_tool_result_for_llm("get_entity_statistics", result).casefold()
    assert "effective windows" in text
    assert "excluded" in text or "omitted" in text
    assert "comparison differences:" in text
    assert text.index(comparison["qualification"].casefold()) < text.index("displayed ")


@pytest.mark.asyncio
@pytest.mark.parametrize("affected_period, missing_kind", [
    ("current", "bucket"), ("previous", "bucket"),
    ("current", "finite_metric"), ("previous", "finite_metric"),
])
async def test_effective_comparison_requires_each_period_and_selected_metric(
    statistics_server, monkeypatch, affected_period, missing_kind,
):
    server, query, _ = statistics_server
    now = datetime(2026, 1, 3, 12, 2, 17, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(history.dt_util, "now", lambda: now)
    current = rows(datetime(2026, 1, 3, 11, 5, tzinfo=timezone.utc), count=11, value=2)
    previous = rows(datetime(2026, 1, 3, 10, 5, tzinfo=timezone.utc), count=11)
    for row in current + previous:
        row.update(mean=4, min=3, max=5)
    affected = current if affected_period == "current" else previous
    if missing_kind == "bucket":
        affected.pop(3)
    else:
        affected[3]["mean"] = float("nan")
    query.return_value = {ENTITY: with_comparison_baselines(
        previous + current, current[0]["start"], previous[0]["start"],
    )}
    data = (await server.tool_get_entity_statistics({
        "entity_id": ENTITY, "period": "last_hour", "bucket": "auto",
        "metric": "all", "compare_previous": True,
    }))["structuredContent"]
    comparison = data["comparison"]
    affected_coverage = data["coverage"] if affected_period == "current" else comparison["previous"]["coverage"]
    assert affected_coverage["effective_complete"] is False
    assert affected_coverage["per_metric"]["mean"]["effective_complete"] is False
    if missing_kind == "finite_metric":
        assert affected_coverage["missing_buckets"] == 0
        assert affected_coverage["per_metric"]["change"]["effective_complete"] is True
    assert comparison["comparable"] is False
    assert comparison["differences"] == {}


@pytest.mark.asyncio
async def test_unequal_fully_covered_effective_intervals_are_not_comparable(statistics_server):
    server, query, _ = statistics_server
    current_start = START + timedelta(minutes=5)
    previous_start = START - timedelta(minutes=30)
    query.return_value = {ENTITY: with_comparison_baselines(
        rows(previous_start, count=6) + rows(current_start, count=5, value=2),
        current_start, previous_start,
    )}
    data = (await server.tool_get_entity_statistics(arguments(
        start=START + timedelta(minutes=2), end=START + timedelta(minutes=34), compare_previous=True,
    )))["structuredContent"]
    comparison = data["comparison"]
    assert data["coverage"]["effective_complete"]
    assert comparison["previous"]["coverage"]["effective_complete"]
    current_window = data["effective_window"]
    previous_window = comparison["previous"]["effective_window"]
    assert datetime.fromisoformat(current_window["end"]) - datetime.fromisoformat(current_window["start"]) == timedelta(minutes=25)
    assert datetime.fromisoformat(previous_window["end"]) - datetime.fromisoformat(previous_window["start"]) == timedelta(minutes=30)
    assert comparison["comparable"] is False
    assert comparison["differences"] == {}
    assert comparison["effective_duration_seconds"] is None
    assert "equal" in comparison["reason"].casefold()


@pytest.mark.asyncio
async def test_rolling_dst_comparison_preserves_equal_elapsed_utc_intervals(statistics_server, monkeypatch):
    server, query, _ = statistics_server
    old_zone = dt_util.DEFAULT_TIME_ZONE
    zone = ZoneInfo("America/New_York")
    try:
        dt_util.set_default_time_zone(zone)
        now = datetime(2026, 3, 9, 0, 2, 17, 123456, tzinfo=zone)
        monkeypatch.setattr(history.dt_util, "now", lambda: now)
        effective_end = datetime(2026, 3, 9, 4, tzinfo=timezone.utc)
        effective_start = effective_end - timedelta(hours=23)
        previous_start = effective_start - timedelta(hours=24)
        query.return_value = {ENTITY: with_comparison_baselines(
            rows(previous_start, count=23, seconds=3600) +
            rows(effective_start, count=23, seconds=3600, value=2),
            effective_start, previous_start, 3600,
        )}
        data = (await server.tool_get_entity_statistics({
            "entity_id": ENTITY, "period": "last_24_hours", "bucket": "auto",
            "metric": "change", "compare_previous": True,
        }))["structuredContent"]
        comparison = data["comparison"]
        for window in (data["requested_window"], comparison["previous"]["requested_window"]):
            assert datetime.fromisoformat(window["end"]) - datetime.fromisoformat(window["start"]) == timedelta(hours=24)
        assert comparison["effective_duration_seconds"] == 23 * 3600
        assert comparison["comparable"]
        assert comparison["differences"]["change"]["percent"] == 100
        assert comparison["boundaries_aligned"] is False
    finally:
        dt_util.set_default_time_zone(old_zone)


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
    assert query.call_args.args[2] - query.call_args.args[1] == timedelta(hours=144, minutes=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("previous, percent", [(0, None), (-1, None), (1, 100)])
async def test_comparison_reuses_one_query_and_units(statistics_server, previous, percent):
    server, query, jobs = statistics_server
    previous_start = START - timedelta(hours=1)
    query.return_value = {ENTITY: with_comparison_baselines(
        rows(previous_start, value=previous) + rows(value=2), START, previous_start,
    )}
    data = (await server.tool_get_entity_statistics(arguments(compare_previous=True)))["structuredContent"]
    query.assert_called_once()
    assert len(jobs) == 3
    assert query.call_args.args[1] == previous_start - timedelta(minutes=5)
    assert query.call_args.args[-1] == {"change", "sum"}
    comparison = data["comparison"]
    assert comparison["comparable"]
    assert comparison["boundaries_aligned"] is True
    assert comparison["qualification"] is None
    assert comparison["previous"]["requested_window"]["end"] == START.isoformat()
    assert comparison["differences"]["change"] == {"absolute": 24 - previous * 12, "percent": percent}


@pytest.mark.asyncio
async def test_metric_gaps_and_untrusted_rows_suppress_comparison(statistics_server):
    server, query, _ = statistics_server
    current = rows()
    current[0]["change"] = float("nan")
    current[1]["change"] = True
    query.return_value = {ENTITY: with_comparison_baselines(rows(START - timedelta(hours=1)) + current + [
        current[2], {"start": START + timedelta(hours=1), "change": 1000},
        {"start": START + timedelta(seconds=1), "change": 1000},
    ], START, START - timedelta(hours=1))}
    data = (await server.tool_get_entity_statistics(arguments(compare_previous=True)))["structuredContent"]
    assert data["summary"]["change"] == 8
    assert not data["coverage"]["per_metric"]["change"]["complete"]
    assert data["coverage"]["per_metric"]["change"]["missing_buckets"] == 4
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
    query.return_value = {ENTITY: with_comparison_baselines(samples, START, START - timedelta(hours=72), 3600)}
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
    query.return_value = {ENTITY: with_comparison_baselines(
        rows(START - timedelta(hours=1)) + rows(), START, START - timedelta(hours=1),
    )}
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


@pytest.fixture
def statistics_dst_timezone(hass):
    old_zone = dt_util.DEFAULT_TIME_ZONE
    zone = ZoneInfo("Europe/Berlin")
    dt_util.set_default_time_zone(zone)
    try:
        yield zone
    finally:
        dt_util.set_default_time_zone(old_zone)


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_custom_interval_orders_absolute_instants_across_dst_fold(
    statistics_server, statistics_dst_timezone, reverse,
):
    server, query, jobs = statistics_server
    start = datetime(2026, 10, 25, 0, 50, tzinfo=timezone.utc)
    end = datetime(2026, 10, 25, 1, 10, tzinfo=timezone.utc)
    query.return_value = {ENTITY: with_comparison_baselines(rows(start, count=4), start)}
    result = await server.tool_get_entity_statistics(arguments(
        start=end if reverse else start, end=start if reverse else end,
    ))
    if reverse:
        assert result["isError"] is True
        assert "later than" in result["content"][0]["text"]
        query.assert_not_called()
        assert jobs == []
    else:
        assert not result.get("isError")
        data = result["structuredContent"]
        assert data["requested_window"] == {"start": start.isoformat(), "end": end.isoformat()}
        assert data["effective_window"] == data["requested_window"]
        assert data["coverage"]["complete"] is True
        assert data["summary"]["change"] == 4
        assert query.call_args.args[1:3] == (start - timedelta(minutes=5), end)


@pytest.mark.parametrize("now", [
    datetime(2026, 3, 29, 1, 30, tzinfo=timezone.utc),
    datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc),
])
@pytest.mark.parametrize("period,hours", [("last_hour", 1), ("last_24_hours", 24)])
def test_rolling_intervals_keep_elapsed_duration_across_dst(
    statistics_server, statistics_dst_timezone, monkeypatch, now, period, hours,
):
    server, _, _ = statistics_server
    zone = statistics_dst_timezone
    monkeypatch.setattr(history.dt_util, "now", lambda: now.astimezone(zone))
    start, end, _ = server._statistics_window({}, period)
    assert end == now
    assert end - start == timedelta(hours=hours)


@pytest.mark.parametrize("now,hours", [
    (datetime(2026, 3, 30, tzinfo=ZoneInfo("Europe/Berlin")), 23),
    (datetime(2026, 10, 26, tzinfo=ZoneInfo("Europe/Berlin")), 25),
])
def test_calendar_day_keeps_local_boundaries_across_dst(
    statistics_server, statistics_dst_timezone, monkeypatch, now, hours,
):
    server, _, _ = statistics_server
    monkeypatch.setattr(history.dt_util, "now", lambda: now)
    start, end, _ = server._statistics_window({}, "yesterday")
    assert end - start == timedelta(hours=hours)
    assert dt_util.as_local(start).date() == now.date() - timedelta(days=1)
    assert dt_util.as_local(start).hour == dt_util.as_local(end).hour == 0
    changes = {row["start"]: row["change"] for row in rows(start, count=hours, seconds=3600)}
    assert server._complete_statistic_period_totals(changes, start, end, "day") == [hours]
