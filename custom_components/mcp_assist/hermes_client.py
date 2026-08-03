"""Client for Hermes Agent's server-managed OpenAI-compatible API."""

from __future__ import annotations

from collections import OrderedDict
import json
import logging
import uuid
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)

_MAX_TRACKED_SESSIONS = 128
_MAX_HEADER_LENGTH = 512


class HermesError(Exception):
    """Base exception for Hermes Agent errors."""


class HermesAuthError(HermesError):
    """Hermes rejected the configured API key."""


class HermesBusyError(HermesError):
    """Hermes reached its concurrent-agent limit."""


class HermesConnectionError(HermesError):
    """Hermes could not be reached or its response could not be completed."""


class HermesSessionError(HermesError):
    """Hermes rejected a stored transcript session identifier."""


class HermesStreamingUnsupportedError(HermesError):
    """Hermes explicitly rejected streaming before starting an agent run."""


def _safe_error_text(value: Any, limit: int = 300) -> str:
    """Return a bounded, single-line provider error description."""
    text = "".join(
        character
        for character in str(value or "")
        if character >= " " or character == "\t"
    ).strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class HermesClient:
    """Call a Hermes Agent API server while preserving transcript continuity."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session_key: str,
        model: str,
        timeout: int,
        debug: bool = False,
    ) -> None:
        """Initialize the Hermes client without performing network work."""
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._api_key = self._validate_header_value(
            api_key,
            name="Hermes API key",
            allow_empty=True,
        )
        self._session_key = self._validate_header_value(
            session_key,
            name="Hermes memory session key",
            allow_empty=True,
        )
        self._model = str(model or "hermes-agent").strip() or "hermes-agent"
        self._timeout = int(timeout)
        self._debug = debug
        self._session_ids: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def _validate_header_value(
        value: Any,
        *,
        name: str,
        allow_empty: bool = False,
    ) -> str:
        """Validate a value before placing it in an HTTP header."""
        normalized = str(value or "").strip()
        if not normalized and allow_empty:
            return ""
        if not normalized:
            raise HermesError(f"{name} cannot be empty")
        if "\r" in normalized or "\n" in normalized or "\x00" in normalized:
            raise HermesError(f"{name} contains invalid control characters")
        if len(normalized) > _MAX_HEADER_LENGTH:
            raise HermesError(f"{name} is too long")
        return normalized

    @property
    def chat_url(self) -> str:
        """Return the Hermes chat-completions endpoint."""
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/chat/completions"
        return f"{self._base_url}/v1/chat/completions"

    def _request_headers(
        self,
        conversation_id: str,
        *,
        idempotency_key: str,
    ) -> dict[str, str]:
        """Build request headers for a Hermes turn."""
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        }
        if not self._api_key:
            return headers

        headers["Authorization"] = f"Bearer {self._api_key}"
        if self._session_key:
            headers["X-Hermes-Session-Key"] = self._session_key

        session_id = self._session_ids.get(conversation_id)
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id
        return headers

    def _request_messages(
        self,
        text: str,
        conversation_id: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build the smallest safe transcript for the configured auth mode."""
        if self._api_key and self._session_ids.get(conversation_id):
            return [{"role": "user", "content": text}]

        messages: list[dict[str, str]] = []
        for turn in history:
            user_text = str(turn.get("user") or "")
            assistant_text = str(turn.get("assistant") or "")
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": text})
        return messages

    def build_request(
        self,
        text: str,
        conversation_id: str,
        history: list[dict[str, Any]],
        *,
        stream: bool,
        idempotency_key: str,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build a Hermes request without exposing MCP Assist's client tools."""
        headers = self._request_headers(
            conversation_id,
            idempotency_key=idempotency_key,
        )
        payload = {
            "model": self._model,
            "messages": self._request_messages(text, conversation_id, history),
            "stream": stream,
        }
        return self.chat_url, headers, payload

    async def send_message(
        self,
        text: str,
        conversation_id: str,
        history: list[dict[str, Any]],
        *,
        _retried_session: bool = False,
    ) -> str:
        """Send one turn, retrying only a server-rejected stale session."""
        idempotency_key = str(uuid.uuid4())
        try:
            return await self._call_streaming(
                text,
                conversation_id,
                history,
                idempotency_key=idempotency_key,
            )
        except HermesSessionError:
            if _retried_session:
                raise
            self._session_ids.pop(conversation_id, None)
            return await self.send_message(
                text,
                conversation_id,
                history,
                _retried_session=True,
            )
        except HermesStreamingUnsupportedError:
            return await self._call_http(
                text,
                conversation_id,
                history,
                idempotency_key=idempotency_key,
            )

    async def _call_streaming(
        self,
        text: str,
        conversation_id: str,
        history: list[dict[str, Any]],
        *,
        idempotency_key: str,
    ) -> str:
        """Call Hermes using its SSE chat-completions transport."""
        url, headers, payload = self.build_request(
            text,
            conversation_id,
            history,
            stream=True,
            idempotency_key=idempotency_key,
        )
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        await self._raise_response_error(response)

                    self._store_session_id(conversation_id, response.headers)
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "application/json" in content_type:
                        return self._parse_http_payload(await response.json())

                    response_text = ""
                    finish_reason = None
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data_text = line[5:].lstrip()
                        if data_text == "[DONE]":
                            break
                        try:
                            data = json.loads(data_text)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices")
                        if not choices:
                            if self._debug and data.get("tool"):
                                _LOGGER.debug(
                                    "Hermes tool progress: tool=%s status=%s",
                                    _safe_error_text(data.get("tool"), 100),
                                    _safe_error_text(data.get("status"), 40),
                                )
                            continue

                        choice = choices[0] if isinstance(choices[0], dict) else {}
                        delta = choice.get("delta")
                        if isinstance(delta, dict) and isinstance(
                            delta.get("content"), str
                        ):
                            response_text += delta["content"]
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])

                    return self._finish_response(
                        response_text,
                        finish_reason=finish_reason,
                    )
        except HermesError:
            raise
        except aiohttp.ClientError as err:
            raise HermesConnectionError(
                "Could not complete the Hermes Agent request"
            ) from err
        except TimeoutError as err:
            raise HermesConnectionError("Hermes Agent response timed out") from err

    async def _call_http(
        self,
        text: str,
        conversation_id: str,
        history: list[dict[str, Any]],
        *,
        idempotency_key: str,
    ) -> str:
        """Use non-streaming HTTP after an explicit streaming rejection."""
        url, headers, payload = self.build_request(
            text,
            conversation_id,
            history,
            stream=False,
            idempotency_key=idempotency_key,
        )
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        await self._raise_response_error(response)
                    self._store_session_id(conversation_id, response.headers)
                    return self._parse_http_payload(await response.json())
        except HermesError:
            raise
        except aiohttp.ClientError as err:
            raise HermesConnectionError(
                "Could not complete the Hermes Agent request"
            ) from err
        except TimeoutError as err:
            raise HermesConnectionError("Hermes Agent response timed out") from err

    def _store_session_id(
        self,
        conversation_id: str,
        response_headers: Any,
    ) -> None:
        """Capture an echoed or rotated session ID for the next authenticated turn."""
        if not self._api_key:
            return
        raw_session_id = response_headers.get("X-Hermes-Session-Id")
        if not raw_session_id:
            return
        try:
            session_id = self._validate_header_value(
                raw_session_id,
                name="Hermes transcript session ID",
            )
        except HermesError:
            _LOGGER.warning("Ignored an invalid Hermes transcript session ID")
            return

        self._session_ids[conversation_id] = session_id
        self._session_ids.move_to_end(conversation_id)
        while len(self._session_ids) > _MAX_TRACKED_SESSIONS:
            self._session_ids.popitem(last=False)

    @staticmethod
    def _payload_error_message(payload: Any) -> str:
        """Extract a bounded error message from a Hermes response payload."""
        if not isinstance(payload, dict):
            return ""
        error = payload.get("error")
        if isinstance(error, dict):
            return _safe_error_text(error.get("message"))
        if error:
            return _safe_error_text(error)
        hermes = payload.get("hermes")
        if isinstance(hermes, dict):
            return _safe_error_text(hermes.get("error"))
        return ""

    def _parse_http_payload(self, payload: Any) -> str:
        """Extract assistant text from a non-streaming chat completion."""
        if not isinstance(payload, dict):
            raise HermesConnectionError("Hermes returned an invalid JSON response")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise HermesConnectionError("Hermes returned no completion choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        response_text = (
            str(message.get("content") or "") if isinstance(message, dict) else ""
        )
        return self._finish_response(
            response_text,
            finish_reason=choice.get("finish_reason"),
        )

    @staticmethod
    def _finish_response(
        response_text: str,
        *,
        finish_reason: Any,
    ) -> str:
        """Validate Hermes completion state without retrying visible requests."""
        normalized_reason = str(finish_reason or "").casefold()
        if normalized_reason == "error":
            raise HermesConnectionError("Hermes Agent reported an error")
        if not response_text:
            raise HermesConnectionError("Hermes Agent returned an empty response")
        if normalized_reason == "length":
            _LOGGER.warning("Hermes Agent response was truncated")
        return response_text

    async def _raise_response_error(self, response: Any) -> None:
        """Map a non-success Hermes response to a stable exception type."""
        try:
            payload = await response.json(content_type=None)
            message = self._payload_error_message(payload)
        except Exception:
            message = _safe_error_text(await response.text())

        if response.status == 429:
            raise HermesBusyError(
                "Hermes Agent is busy handling other requests; try again shortly"
            )
        if response.status in {401, 403}:
            raise HermesAuthError("Hermes Agent rejected the configured API key")
        if response.status in {400, 404} and "session" in message.casefold():
            raise HermesSessionError("Hermes transcript session was rejected")
        if (
            response.status in {400, 404, 405, 422}
            and "stream" in message.casefold()
            and any(
                term in message.casefold()
                for term in ("unsupported", "not support", "not allowed", "invalid")
            )
        ):
            raise HermesStreamingUnsupportedError(message)

        raise HermesConnectionError(
            f"Hermes Agent request failed with HTTP {response.status}"
        )
