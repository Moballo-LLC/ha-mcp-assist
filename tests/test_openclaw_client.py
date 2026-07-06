"""Tests for OpenClaw client behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from custom_components.mcp_assist.openclaw_client import (
    OpenClawClient,
    OpenClawConnectionError,
    _AgentRun,
)


def test_openclaw_client_uses_configured_locale() -> None:
    """OpenClaw handshake locale should follow the Home Assistant language."""
    client = OpenClawClient(
        host="internal.example",
        port=18789,
        token="test-token",
        use_ssl=True,
        device_auth=SimpleNamespace(),
        locale="fr_CA",
    )

    assert client._locale == "fr-CA"


def test_openclaw_client_locale_falls_back_to_english() -> None:
    """Empty OpenClaw locale should retain the previous English fallback."""
    client = OpenClawClient(
        host="internal.example",
        port=18789,
        token="test-token",
        use_ssl=True,
        device_auth=SimpleNamespace(),
        locale="",
    )

    assert client._locale == "en-US"


def _make_client() -> OpenClawClient:
    return OpenClawClient(
        host="internal.example",
        port=18789,
        token="test-token",
        use_ssl=False,
        device_auth=SimpleNamespace(),
    )


class _ImmediateAckWs:
    """Fake websocket that delivers the ack and completion during send()."""

    def __init__(self, client: OpenClawClient) -> None:
        self._client = client

    async def send(self, raw: str) -> None:
        msg = json.loads(raw)
        if msg.get("method") != "agent":
            return
        # Simulate the receive loop processing the acknowledgment and the
        # completion event back to back, before send_message resumes.
        await self._client._handle_message(
            {"type": "res", "id": msg["id"], "ok": True, "payload": {"runId": "run-1"}}
        )
        self._client._handle_agent_event(
            {"runId": "run-1", "output": "All done", "status": "ok", "summary": "All done"}
        )


@pytest.mark.asyncio
async def test_send_message_survives_ack_and_completion_during_send() -> None:
    """An ack/completion processed during the send await must not be dropped."""
    client = _make_client()
    client._connected = True
    client._ws = _ImmediateAckWs(client)

    result = await asyncio.wait_for(client.send_message("hi", "main"), timeout=5)

    assert result == "All done"
    assert client._agent_runs == {}
    assert client._early_agent_events == {}


class _DyingWs:
    """Fake websocket whose receive iteration ends immediately."""

    def __init__(self, error: Exception | None) -> None:
        self._error = error

    def __aiter__(self) -> "_DyingWs":
        return self

    async def __anext__(self) -> str:
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("boom"), None])
async def test_receive_loop_end_fails_inflight_waiters(error: Exception | None) -> None:
    """Waiters must fail fast when the connection dies, not hit full timeouts."""
    client = _make_client()
    client._connected = True
    client._ws = _DyingWs(error)

    pending = asyncio.get_running_loop().create_future()
    client._pending_requests["req-1"] = pending
    run = _AgentRun("run-1")
    client._agent_runs["run-1"] = run

    await client._receive_loop()

    assert client._connected is False
    assert isinstance(pending.exception(), OpenClawConnectionError)
    assert run.complete_event.is_set()
    assert run.status == "error"
    assert client._pending_requests == {}
    assert client._agent_runs == {}


@pytest.mark.asyncio
async def test_concurrent_connect_only_connects_once() -> None:
    """Concurrent reconnect attempts must not open two sockets."""
    client = _make_client()
    calls = 0

    async def fake_connect_locked() -> None:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        client._connected = True
        client._ws = object()

    client._connect_locked = fake_connect_locked

    await asyncio.gather(client.connect(), client.connect())

    assert calls == 1


def test_early_agent_event_buffer_is_bounded() -> None:
    """Unclaimed early events must not grow without bound."""
    client = _make_client()

    for i in range(client._MAX_EARLY_EVENT_RUNS * 2):
        client._handle_agent_event({"runId": f"run-{i}", "output": "x"})

    assert len(client._early_agent_events) == client._MAX_EARLY_EVENT_RUNS

    for _ in range(client._MAX_EARLY_EVENTS_PER_RUN * 2):
        client._handle_agent_event({"runId": "run-hot", "output": "x"})

    assert len(client._early_agent_events["run-hot"]) == client._MAX_EARLY_EVENTS_PER_RUN
