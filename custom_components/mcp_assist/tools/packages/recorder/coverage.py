"""Bounded, optional coverage annotations supplied by a statistic's entity."""

from collections.abc import Mapping
from datetime import date, timedelta

from homeassistant.util import dt as dt_util


def source_coverage(attributes, start, end):
    """Project documented recorder_coverage fields as source-reported evidence."""
    if "recorder_coverage" not in attributes:
        return {}
    raw = attributes["recorder_coverage"]
    invalid = {
        "available": False,
        "uncertain": True,
        "reason": "Source coverage annotations are malformed or exceed their bounds.",
    }
    if not isinstance(raw, Mapping):
        return invalid
    result = {"available": True, "uncertain": False}
    started = raw.get("started_at")
    if started is not None:
        if not isinstance(started, str) or len(started) > 64:
            return invalid
        parsed = dt_util.parse_datetime(started)
        if parsed is None or parsed.tzinfo is None:
            return invalid
        result["started_at"] = dt_util.as_utc(parsed).isoformat()
        result["includes_pre_collection_time"] = start < dt_util.as_utc(parsed)
        result["uncertain"] |= result["includes_pre_collection_time"]
    note = raw.get("note")
    if note is not None:
        if not isinstance(note, str) or len(note) > 1200:
            return invalid
        result["note"] = note
    resets = raw.get("counter_resets")
    if resets is not None:
        if isinstance(resets, bool) or not isinstance(resets, int) or resets < 0:
            return invalid
        result["counter_resets"] = resets

    periods = raw.get("estimated_periods", {})
    if not isinstance(periods, Mapping):
        return invalid
    local_start = dt_util.as_local(start).date()
    local_end = dt_util.as_local(end - timedelta(microseconds=1)).date()
    matched = {"days": [], "months": []}
    for key, bound in (("days", 400), ("months", 24)):
        values = periods.get(key, [])
        if not isinstance(values, list) or len(values) > bound:
            return invalid
        for value in values:
            if not isinstance(value, str):
                return invalid
            try:
                parsed_date = date.fromisoformat(value if key == "days" else value + "-01")
            except ValueError:
                return invalid
            if value != parsed_date.isoformat()[:10 if key == "days" else 7]:
                return invalid
            first = local_start.isoformat()[:10 if key == "days" else 7]
            last = local_end.isoformat()[:10 if key == "days" else 7]
            if first <= value <= last:
                matched[key].append(value)
    older = periods.get("unknown_before")
    if older is not None:
        try:
            older_date = date.fromisoformat(older)
        except (TypeError, ValueError):
            return invalid
        if older != older_date.isoformat():
            return invalid
        matched["unknown_before"] = older
        result["includes_older_unverified_time"] = local_start < older_date
        result["uncertain"] |= result["includes_older_unverified_time"]
    result["estimated_periods"] = matched
    result["uncertain"] |= bool(matched["days"] or matched["months"])
    return result
