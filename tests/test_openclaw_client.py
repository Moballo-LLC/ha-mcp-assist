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

    await client._receive_loop(client._ws)

    assert client._connected is False
    assert isinstance(pending.exception(), OpenClawConnectionError)
    assert run.complete_event.is_set()
    assert run.status == "error"
    assert client._pending_requests == {}
    assert client._agent_runs == {}


@pytest.mark.asyncio
async def test_stale_receive_loop_does_not_touch_new_connection() -> None:
    """A stale loop from a replaced socket must not disconnect the new one."""
    client = _make_client()
    new_ws = object()
    client._connected = True
    client._ws = new_ws

    new_pending = asyncio.get_running_loop().create_future()
    client._pending_requests["new-req"] = new_pending
    new_run = _AgentRun("new-run")
    client._agent_runs["new-run"] = new_run

    # The old loop (bound to a since-replaced socket) finally exits.
    await client._receive_loop(_DyingWs(RuntimeError("old socket died")))

    # The live connection and its in-flight work are untouched.
    assert client._connected is True
    assert client._ws is new_ws
    assert not new_pending.done()
    assert not new_run.complete_event.is_set()
    assert "new-req" in client._pending_requests
    assert "new-run" in client._agent_runs


@pytest.mark.asyncio
async def test_reconnect_tears_down_stale_socket_and_tasks() -> None:
    """Reconnecting must cancel the previous receive loop and close its socket."""
    client = _make_client()

    closed = False

    class _OldWs:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(3600)  # block like a live socket until cancelled

        async def close(self):
            nonlocal closed
            closed = True

    old_ws = _OldWs()
    client._ws = old_ws
    client._connected = False  # keepalive failure left the socket behind
    old_receive = asyncio.create_task(client._receive_loop(old_ws))
    client._receive_task = old_receive
    await asyncio.sleep(0)  # let the loop start blocking on the old socket

    connected_ws = object()

    async def fake_connect_locked() -> None:
        # Mirror the real method: tear down the previous connection first.
        await client._teardown_connection("Reconnecting to OpenClaw Gateway")
        client._ws = connected_ws
        client._connected = True

    client._connect_locked = fake_connect_locked
    await client.connect()

    assert closed is True
    assert old_receive.cancelled() or old_receive.done()
    assert client._ws is connected_ws
    assert client._connected is True


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


@pytest.mark.asyncio
async def test_buffered_completion_survives_connection_loss_and_replays() -> None:
    """A completion buffered before run registration must survive a socket close.

    Sequence: send_message registers the ack future and sends; the receive loop
    delivers the ack (resolving the future) and the completion event (buffered,
    since the run isn't registered yet), then the socket closes. The buffered
    completion must be preserved so send_message can replay it instead of
    waiting out the full agent timeout.
    """
    client = _make_client()
    client._connected = True
    ws = _DyingWs(None)  # yields nothing, then a graceful close
    client._ws = ws

    ack_future = asyncio.get_running_loop().create_future()
    client._pending_requests["req-1"] = ack_future

    await client._handle_message(
        {"type": "res", "id": "req-1", "ok": True, "payload": {"runId": "run-1"}}
    )
    client._handle_agent_event({"runId": "run-1", "status": "ok", "summary": "done"})
    assert "run-1" in client._early_agent_events

    # Socket closes; connection-loss cleanup runs on the owning loop.
    await client._receive_loop(ws)

    # The ack was delivered (not failed) and the completion is still buffered.
    assert ack_future.done() and ack_future.exception() is None
    assert "run-1" in client._early_agent_events

    # send_message now registers the run and replays the buffered events.
    run = _AgentRun("run-1")
    client._agent_runs["run-1"] = run
    for payload in client._early_agent_events.pop("run-1", []):
        client._apply_agent_event(run, payload)

    assert run.complete_event.is_set()
    assert run.status == "ok"
    assert run.summary == "done"


@pytest.mark.asyncio
async def test_connection_loss_does_not_overwrite_completed_run() -> None:
    """A run that already completed ok must not be flipped to error on close."""
    client = _make_client()
    client._connected = True
    ws = _DyingWs(RuntimeError("socket died"))
    client._ws = ws

    run = _AgentRun("run-1")
    run.set_complete("ok", "done")
    client._agent_runs["run-1"] = run

    await client._receive_loop(ws)

    assert run.status == "ok"
    assert run.summary == "done"
