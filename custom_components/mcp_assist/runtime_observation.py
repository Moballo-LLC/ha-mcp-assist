"""Small in-memory counters for troubleshooting ordinary request outcomes."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps

from aiohttp import ClientConnectionError

_STAGES = frozenset({"conversation", "provider", "tool"})
_MAX_COUNT = 2**31 - 1
_ATTRIBUTE = "_ordinary_request_outcomes"
_ACTIVE: ContextVar[tuple] = ContextVar("mcp_assist_observed_requests", default=())


class ProviderResponseError(ValueError):
    """A provider HTTP failure that retains ValueError compatibility."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_type(error: BaseException) -> str:
    """Return a fixed category, never the exception's message or type name."""
    name = type(error).__name__.casefold()
    if isinstance(error, TimeoutError) or "timeout" in name:
        return "timeout"
    if "auth" in name:
        return "authorization"
    if isinstance(error, (ConnectionError, ClientConnectionError)) or "connection" in name:
        return "connection"
    if isinstance(error, ProviderResponseError):
        return "provider"
    if isinstance(error, (ValueError, TypeError)):
        return "validation"
    if "provider" in name or "stream" in name:
        return "provider"
    return "unknown"


def _result_outcome(stage: str, result: object) -> tuple[str, str | None]:
    if stage == "tool" and isinstance(result, dict):
        if result.get("isError") is True or "error" in result:
            return "failure", "tool"
        if any(key in result for key in ("content", "structuredContent", "result")):
            return "success", None
    if stage == "conversation":
        response = getattr(result, "response", None)
        if response is not None and hasattr(response, "error_code"):
            return ("failure", "conversation") if response.error_code is not None else ("success", None)
    if stage == "provider":
        if isinstance(result, str) and result.strip():
            return "success", None
        if (isinstance(result, tuple) and len(result) == 3
                and isinstance(result[0], bytes) and result[0]
                and isinstance(result[1], str) and isinstance(result[2], dict)):
            return "success", None
    return "uncertain", None


class RequestOutcomes:
    """Three fixed stages, saturating counters, no arguments or identifiers."""

    def __init__(self) -> None:
        self.started_at = _now()
        self.stages: dict[str, dict] = {}

    @staticmethod
    def _increment(stage: dict, key: str) -> None:
        if stage[key] == _MAX_COUNT:
            stage["saturated"] = True
        else:
            stage[key] += 1

    def begin(self, stage: str) -> None:
        if stage not in _STAGES:
            raise ValueError("Unknown request stage")
        value = self.stages.setdefault(stage, {
            "started": 0, "active": 0, "success": 0, "failure": 0,
            "cancelled": 0, "uncertain": 0, "saturated": False,
        })
        self._increment(value, "started")
        self._increment(value, "active")
        value["last_started_at"] = _now()

    def finish(self, stage: str, outcome: str, failure: str | None = None) -> None:
        if outcome not in {"success", "failure", "cancelled", "uncertain"}:
            raise ValueError("Unknown request outcome")
        value = self.stages[stage]
        value["active"] = max(0, value["active"] - 1)
        self._increment(value, outcome)
        at = _now()
        value["last_terminal_at"] = at
        value["last_outcome"] = outcome
        value["last_" + outcome + "_at"] = at
        value["last_failure_type"] = failure
        # Cancellation/timeout cannot prove whether a remote operation completed.
        uncertain = outcome in ("cancelled", "uncertain") or failure == "timeout"
        value["last_terminal_uncertain"] = uncertain
        if uncertain and outcome != "uncertain":
            self._increment(value, "uncertain")
            value["last_uncertain_at"] = at


def project_request_outcomes(owner: object | None) -> dict:
    """Read existing counters only; this does not initialize or probe anything."""
    counters = vars(owner).get(_ATTRIBUTE) if owner is not None and hasattr(owner, "__dict__") else None
    if not isinstance(counters, RequestOutcomes):
        return {"status": "not_observed"}
    return {"status": "observed", "window_started_at": counters.started_at,
            "stages": {key: dict(value) for key, value in counters.stages.items()},
            "boundary": "protocol_completion_not_device_or_remote_terminal_proof"}


class observe_request:
    """Observe one normal dispatch, excluding nested dispatch helpers."""

    def __init__(self, owner: object, stage: str) -> None:
        if stage not in _STAGES:
            raise ValueError("Unknown request stage")
        self.owner = owner
        self.stage = stage
        self.key = (id(owner), stage)
        self.counters: RequestOutcomes | None = None
        self.token = None
        self.outcome = "uncertain"
        self.failure: str | None = None
        self.uncertain_result = False

    def __enter__(self):
        active = _ACTIVE.get()
        if not any(scope.key == self.key for scope in active):
            self.counters = vars(self.owner).get(_ATTRIBUTE)
            if not isinstance(self.counters, RequestOutcomes):
                self.counters = RequestOutcomes()
                setattr(self.owner, _ATTRIBUTE, self.counters)
            self.counters.begin(self.stage)
            self.token = _ACTIVE.set((*active, self))
        return self

    def complete(self, result: object) -> None:
        if self.failure is not None:
            self.outcome = "failure"
        else:
            self.outcome, self.failure = (
                ("uncertain", None) if self.uncertain_result
                else _result_outcome(self.stage, result)
            )

    def __exit__(self, error_type, error, traceback):
        if self.counters is not None:
            try:
                if isinstance(error, asyncio.CancelledError):
                    self.outcome, self.failure = "cancelled", None
                elif error is not None:
                    self.outcome, self.failure = "failure", _failure_type(error)
                self.counters.finish(self.stage, self.outcome, self.failure)
            finally:
                _ACTIVE.reset(self.token)
        return False


def note_request_failure(owner: object, stage: str, error: BaseException) -> None:
    """Retain a typed failure when a normal dispatcher returns an error result."""
    for scope in reversed(_ACTIVE.get()):
        if scope.key == (id(owner), stage):
            scope.outcome, scope.failure = "failure", _failure_type(error)
            return


def note_request_uncertain(owner: object, stage: str) -> None:
    """Keep a malformed reply uncertain when the dispatcher returns a fallback."""
    for scope in reversed(_ACTIVE.get()):
        if scope.key == (id(owner), stage):
            scope.uncertain_result = True
            return


def observe(stage: str):
    """Wrap a normal operation without changing its result or retry behavior."""
    if stage not in _STAGES:
        raise ValueError("Unknown request stage")

    def decorate(function):
        @wraps(function)
        async def wrapped(self, *args, **kwargs):
            with observe_request(self, stage) as outcome:
                result = await function(self, *args, **kwargs)
                outcome.complete(result)
                return result
        return wrapped
    return decorate
