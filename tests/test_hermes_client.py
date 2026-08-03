"""Tests for the Hermes Agent server-managed client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist import hermes_client as hermes_module
from custom_components.mcp_assist.hermes_client import (
    HermesClient,
    HermesConnectionError,
    HermesError,
    HermesSessionError,
    HermesStreamingUnsupportedError,
)


def _make_client(*, api_key: str = "test-api-key") -> HermesClient:
    return HermesClient(
        base_url="http://hermes.example.invalid:8642",
        api_key=api_key,
        session_key="homeassistant",
        model="hermes-agent",
        timeout=30,
    )


def test_authenticated_request_uses_server_session_and_only_current_message() -> None:
    """Authenticated Hermes requests should rely on server transcript state."""
    client = _make_client()
    client._store_session_id(
        "conversation-1",
        {"X-Hermes-Session-Id": "transcript-1"},
    )

    url, headers, payload = client.build_request(
        "What changed?",
        "conversation-1",
        [{"user": "Earlier question", "assistant": "Earlier answer"}],
        stream=True,
        idempotency_key="request-1",
    )

    assert url == "http://hermes.example.invalid:8642/v1/chat/completions"
    assert headers == {
        "Content-Type": "application/json",
        "Idempotency-Key": "request-1",
        "Authorization": "Bearer test-api-key",
        "X-Hermes-Session-Key": "homeassistant",
        "X-Hermes-Session-Id": "transcript-1",
    }
    assert payload == {
        "model": "hermes-agent",
        "messages": [{"role": "user", "content": "What changed?"}],
        "stream": True,
    }


def test_unauthenticated_request_replays_local_history_without_session_headers() -> None:
    """Hermes without API auth should not claim server-side session privileges."""
    client = _make_client(api_key="")

    _, headers, payload = client.build_request(
        "What changed?",
        "conversation-1",
        [{"user": "Earlier question", "assistant": "Earlier answer"}],
        stream=False,
        idempotency_key="request-1",
    )

    assert headers == {
        "Content-Type": "application/json",
        "Idempotency-Key": "request-1",
    }
    assert payload["messages"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "What changed?"},
    ]


def test_chat_url_does_not_duplicate_version_prefix() -> None:
    """A configured /v1 URL should still produce one version segment."""
    client = HermesClient(
        base_url="http://hermes.example.invalid:8642/v1/",
        api_key="",
        session_key="",
        model="hermes-agent",
        timeout=30,
    )

    assert client.chat_url == "http://hermes.example.invalid:8642/v1/chat/completions"


def test_session_key_rejects_header_injection() -> None:
    """User-configured session keys must be safe to place in HTTP headers."""
    with pytest.raises(HermesError, match="control characters"):
        HermesClient(
            base_url="http://hermes.example.invalid:8642",
            api_key="test-api-key",
            session_key="safe\r\nInjected: value",
            model="hermes-agent",
            timeout=30,
        )


class _SseContent:
    """Async line iterator for a fake SSE response."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    def __aiter__(self) -> "_SseContent":
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration as err:
            raise StopAsyncIteration from err


class _SseResponse:
    """Successful fake Hermes streaming response."""

    status = 200
    headers = {
        "Content-Type": "text/event-stream",
        "X-Hermes-Session-Id": "transcript-2",
    }

    def __init__(self) -> None:
        payloads = [
            {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
            {"tool": "web_search", "status": "running"},
            {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
        ]
        self.content = _SseContent(
            [
                f"data: {json.dumps(payloads[0])}\n".encode(),
                b"event: hermes.tool.progress\n",
                f"data: {json.dumps(payloads[1])}\n".encode(),
                b"data: not-json\n",
                f"data: {json.dumps(payloads[2])}\n".encode(),
                b"data: [DONE]\n",
            ]
        )

    async def __aenter__(self) -> "_SseResponse":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _SseSession:
    """Fake aiohttp session that records the outbound request."""

    def __init__(self, *, timeout: object, calls: list[dict[str, object]]) -> None:
        self._calls = calls

    async def __aenter__(self) -> "_SseSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def post(self, url: str, **kwargs: object) -> _SseResponse:
        self._calls.append({"url": url, **kwargs})
        return _SseResponse()


@pytest.mark.asyncio
async def test_streaming_response_collects_text_and_rotated_session_id(
    monkeypatch,
) -> None:
    """SSE tool-progress events should not obscure the final assistant text."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        hermes_module.aiohttp,
        "ClientSession",
        lambda *, timeout: _SseSession(timeout=timeout, calls=calls),
    )
    client = _make_client()

    response = await client.send_message("Hello", "conversation-1", [])

    assert response == "Hello world"
    assert calls[0]["json"]["stream"] is True
    _, next_headers, _ = client.build_request(
        "Next",
        "conversation-1",
        [],
        stream=True,
        idempotency_key="request-2",
    )
    assert next_headers["X-Hermes-Session-Id"] == "transcript-2"


@pytest.mark.asyncio
async def test_stale_session_is_cleared_and_retried_once() -> None:
    """Only an explicitly rejected transcript session should be retried."""
    client = _make_client()
    client._session_ids["conversation-1"] = "stale-session"
    client._call_streaming = AsyncMock(
        side_effect=[HermesSessionError("expired session"), "Recovered"]
    )

    response = await client.send_message("Hello", "conversation-1", [])

    assert response == "Recovered"
    assert client._call_streaming.await_count == 2
    assert "conversation-1" not in client._session_ids


@pytest.mark.asyncio
async def test_http_fallback_requires_explicit_streaming_rejection() -> None:
    """A transport failure must not duplicate a server-side agent run."""
    client = _make_client()
    client._call_streaming = AsyncMock(
        side_effect=HermesConnectionError("connection dropped")
    )
    client._call_http = AsyncMock(return_value="duplicate")

    with pytest.raises(HermesConnectionError, match="connection dropped"):
        await client.send_message("Run an action", "conversation-1", [])

    client._call_http.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_fallback_runs_after_explicit_streaming_rejection() -> None:
    """Hermes may explicitly request a non-streaming retry before agent work starts."""
    client = _make_client()
    client._call_streaming = AsyncMock(
        side_effect=HermesStreamingUnsupportedError("stream is unsupported")
    )
    client._call_http = AsyncMock(return_value="Fallback response")

    response = await client.send_message("Hello", "conversation-1", [])

    assert response == "Fallback response"
    client._call_http.assert_awaited_once()
