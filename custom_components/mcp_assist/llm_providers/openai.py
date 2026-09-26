"""OpenAI provider transport."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from ..const import (
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_OPENAI_API_TRANSPORT,
    CONF_OPENAI_IMAGE_MODEL,
    CONF_OPENAI_IMAGE_API,
    DEFAULT_OPENAI_API_TRANSPORT,
    DEFAULT_OPENAI_IMAGE_MODEL,
    OPENAI_API_TRANSPORT_AUTO,
    OPENAI_API_TRANSPORT_CHAT_COMPLETIONS,
    OPENAI_API_TRANSPORT_RESPONSES,
    OPENAI_IMAGE_API_IMAGES,
    OPENAI_IMAGE_MODEL_AUTO,
    OPENAI_BASE_URL,
    SERVER_TYPE_OPENAI,
)
from .base import (
    PromptCacheUsage,
    ProviderConfigField,
    ProviderSettings,
    ProviderStreamError,
    StreamParseResult,
    _sse_data_payload,
)
from .openai_compatible import OpenAICompatibleProvider

_RESPONSES_OUTPUT_KEY = "_responses_output"
_OPENAI_API_HOSTS = {
    "api.openai.com",
    *(f"{region}.api.openai.com" for region in ("us", "eu", "au", "ca", "jp", "in", "sg", "kr", "gb", "ae")),
}
_OPENAI_API_TRANSPORTS = {
    OPENAI_API_TRANSPORT_AUTO,
    OPENAI_API_TRANSPORT_RESPONSES,
    OPENAI_API_TRANSPORT_CHAT_COMPLETIONS,
}
_CHAT_COMPLETIONS_ONLY_MODEL_PREFIXES = (
    "gpt-audio",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview",
)


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI Responses and chat-completions transport."""

    provider_type = SERVER_TYPE_OPENAI
    provider_display_name = "OpenAI"
    default_base_url = OPENAI_BASE_URL
    connection_fields = (
        ProviderConfigField(CONF_LMSTUDIO_URL, default=OPENAI_BASE_URL),
        ProviderConfigField(CONF_API_KEY, kind="password"),
    )
    provider_options_fields = (
        ProviderConfigField(
            CONF_OPENAI_API_TRANSPORT,
            default=DEFAULT_OPENAI_API_TRANSPORT,
            kind="select",
            options=(
                OPENAI_API_TRANSPORT_AUTO,
                OPENAI_API_TRANSPORT_RESPONSES,
                OPENAI_API_TRANSPORT_CHAT_COMPLETIONS,
            ),
            translation_key="openai_api_transport",
        ),
        ProviderConfigField(
            CONF_OPENAI_IMAGE_MODEL,
            default=OPENAI_IMAGE_MODEL_AUTO,
            kind="select",
            options=(OPENAI_IMAGE_MODEL_AUTO, DEFAULT_OPENAI_IMAGE_MODEL, "gpt-image-2.5-sunburst"),
            custom_value=True,
        ),
        ProviderConfigField(
            CONF_OPENAI_IMAGE_API,
            default=OPENAI_API_TRANSPORT_AUTO,
            kind="select",
            options=(
                OPENAI_API_TRANSPORT_AUTO,
                OPENAI_IMAGE_API_IMAGES,
                OPENAI_API_TRANSPORT_RESPONSES,
            ),
            translation_key="openai_image_api",
        ),
    )
    model_fetch_error = "invalid_api_key"

    def __init__(self, settings: ProviderSettings) -> None:
        """Initialize the OpenAI transport and response-item replay state."""
        super().__init__(settings)
        self._pending_response_output: list[dict[str, Any]] | None = None
        self._streamed_text_seen = False

    @classmethod
    def options_from_entry(cls, entry: Any) -> dict[str, Any]:
        """Return the selected OpenAI transport and image model."""
        data = getattr(entry, "data", {}) or {}
        options = getattr(entry, "options", {}) or {}
        configured = options.get(
            CONF_OPENAI_API_TRANSPORT,
            data.get(CONF_OPENAI_API_TRANSPORT),
        )
        transport = (
            OPENAI_API_TRANSPORT_CHAT_COMPLETIONS
            if configured in (None, "")
            else str(configured)
        )
        if transport not in _OPENAI_API_TRANSPORTS:
            transport = DEFAULT_OPENAI_API_TRANSPORT
        configured_image_model = options.get(
            CONF_OPENAI_IMAGE_MODEL, data.get(CONF_OPENAI_IMAGE_MODEL)
        )
        image_model = str(configured_image_model or "").strip() or OPENAI_IMAGE_MODEL_AUTO
        image_api = options.get(
            CONF_OPENAI_IMAGE_API, data.get(CONF_OPENAI_IMAGE_API, OPENAI_API_TRANSPORT_AUTO)
        )
        if image_api not in (
            OPENAI_API_TRANSPORT_AUTO, OPENAI_IMAGE_API_IMAGES, OPENAI_API_TRANSPORT_RESPONSES
        ):
            image_api = OPENAI_API_TRANSPORT_AUTO
        return {
            CONF_OPENAI_API_TRANSPORT: transport,
            CONF_OPENAI_IMAGE_MODEL: image_model,
            CONF_OPENAI_IMAGE_API: image_api,
        }

    @staticmethod
    def is_image_model_id(model: str) -> bool:
        """Identify known OpenAI image-only IDs without guessing custom capabilities."""
        return model.startswith("gpt-image-") or model in {"dall-e-2", "dall-e-3"}

    @classmethod
    def default_image_model(cls, model: str, base_url: str) -> str:
        """Keep legacy image IDs and every custom endpoint's existing model."""
        if cls.is_image_model_id(model) or not cls._is_official_openai_base_url(base_url):
            return model
        return DEFAULT_OPENAI_IMAGE_MODEL

    @property
    def image_model(self) -> str:
        """Return this provider instance's selected image model."""
        configured = str(self.settings.provider_options.get(CONF_OPENAI_IMAGE_MODEL) or "").strip()
        if configured and configured != OPENAI_IMAGE_MODEL_AUTO:
            return configured
        return self.default_image_model(self.model_name, self.base_url)

    @property
    def uses_responses_image_api(self) -> bool:
        """Resolve image routing independently of custom endpoints' chat support."""
        configured = self.settings.provider_options.get(CONF_OPENAI_IMAGE_API)
        if configured == OPENAI_API_TRANSPORT_RESPONSES:
            return True
        if configured == OPENAI_IMAGE_API_IMAGES:
            return False
        return (
            self.uses_official_openai_api
            and self.uses_responses_api
            and not self.is_image_model_id(self.model_name)
            and self.image_model not in {"dall-e-2", "dall-e-3"}
        )

    @classmethod
    def _is_official_openai_base_url(cls, base_url: str) -> bool:
        """Return whether a base URL targets OpenAI's official API host."""
        try:
            parsed = urlsplit(str(base_url or "").strip())
            port = parsed.port
        except ValueError:
            return False
        hostname = (parsed.hostname or "").rstrip(".")
        path = parsed.path.rstrip("/")
        return (
            parsed.scheme.lower() == "https"
            and hostname in _OPENAI_API_HOSTS
            and port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and path in {"", "/v1"}
        )

    @classmethod
    def _configured_transport_from_values(cls, values: dict[str, Any] | None) -> str:
        """Return a validated transport selection from config-flow values."""
        configured = (values or {}).get(
            CONF_OPENAI_API_TRANSPORT,
            DEFAULT_OPENAI_API_TRANSPORT,
        )
        transport = str(configured or DEFAULT_OPENAI_API_TRANSPORT)
        return (
            transport
            if transport in _OPENAI_API_TRANSPORTS
            else DEFAULT_OPENAI_API_TRANSPORT
        )

    @classmethod
    def _resolve_api_transport(cls, configured: str, base_url: str) -> str:
        """Resolve Automatic to the safest default for the configured endpoint."""
        if configured != OPENAI_API_TRANSPORT_AUTO:
            return configured
        if cls._is_official_openai_base_url(base_url):
            return OPENAI_API_TRANSPORT_RESPONSES
        return OPENAI_API_TRANSPORT_CHAT_COMPLETIONS

    @property
    def uses_official_openai_api(self) -> bool:
        """Return whether this profile targets OpenAI's official API host."""
        return self._is_official_openai_base_url(self.base_url)

    @property
    def configured_api_transport(self) -> str:
        """Return the profile's validated OpenAI API transport selection."""
        return self._configured_transport_from_values(self.settings.provider_options)

    @property
    def api_transport(self) -> str:
        """Return the effective API transport for this endpoint."""
        return self._resolve_api_transport(
            self.configured_api_transport,
            self.base_url,
        )

    @property
    def uses_responses_api(self) -> bool:
        """Return whether this profile uses the Responses API."""
        return self.api_transport == OPENAI_API_TRANSPORT_RESPONSES

    @property
    def requires_stream_terminal_event(self) -> bool:
        """Require Responses streams to prove they reached a terminal event."""
        return self.uses_responses_api

    def chat_url(self) -> str:
        """Return the selected OpenAI generation endpoint."""
        if self.uses_responses_api:
            return self.provider_endpoint(self.base_url, "responses")
        return super().chat_url()

    def image_generation_url(self) -> str:
        """Return the image endpoint independently of the conversation endpoint."""
        if self.uses_responses_image_api:
            return self.provider_endpoint(self.base_url, "responses")
        return super().image_generation_url()

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build a request for the selected OpenAI API transport."""
        if not self.uses_responses_api:
            if (
                tools
                and self.uses_official_openai_api
                and self.requires_responses_for_tools(self.model_name)
            ):
                raise ValueError(
                    "This model requires the Responses API for tool calling. "
                    "Choose Responses API or Automatic in the profile's Generation API setting."
                )
            payload = super().build_payload(messages, tools, stream=stream)
            if (
                tools
                and self.uses_official_openai_api
                and any(
                    self._matches_model_family(self.model_name, family)
                    for family in ("gpt-6-sol", "gpt-6-luna")
                )
            ):
                payload["reasoning_effort"] = "none"
            return payload

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": self._build_responses_input(messages),
            "stream": stream,
            "store": False,
        }

        if self.is_reasoning_model(self.model_name):
            payload["include"] = ["reasoning.encrypted_content"]
        else:
            payload["temperature"] = self.temperature

        if self.max_tokens > 0:
            payload["max_output_tokens"] = self.max_tokens

        response_tools = self._build_responses_tools(tools or [])
        if response_tools:
            payload["tools"] = response_tools
            payload["tool_choice"] = "auto"

        return payload

    @classmethod
    def _build_responses_tools(
        cls,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Chat Completions function schemas to Responses tools."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict) or not function.get("name"):
                continue
            converted_tool = {
                "type": "function",
                "name": function["name"],
                "description": function.get("description"),
                "parameters": function.get("parameters", {}),
                "strict": function.get("strict", False),
            }
            converted.append(converted_tool)
        return converted

    @classmethod
    def _build_responses_input(
        cls,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert internal chat-shaped history to Responses input items."""
        input_items: list[dict[str, Any]] = []
        for message in messages:
            response_output = message.get(_RESPONSES_OUTPUT_KEY)
            if isinstance(response_output, list):
                input_items.extend(item for item in response_output if isinstance(item, dict))
                continue

            role = str(message.get("role") or "user")
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if call_id:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": cls._stringify_response_tool_output(
                                message.get("content")
                            ),
                        }
                    )
                continue

            content = message.get("content")
            if content not in (None, "", []):
                input_message = {
                    "role": role,
                    "content": cls._convert_responses_content(content),
                }
                if role == "assistant" and message.get("phase") in {"commentary", "final_answer"}:
                    input_message["phase"] = message["phase"]
                input_items.append(input_message)

            tool_calls = message.get("tool_calls")
            if role == "assistant" and isinstance(tool_calls, list):
                input_items.extend(cls._tool_calls_to_response_items(tool_calls))

        return input_items

    @staticmethod
    def _convert_responses_content(content: Any) -> Any:
        """Convert Chat Completions multimodal content to Responses content."""
        if not isinstance(content, list):
            return content

        converted: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                converted.append(block)
                continue
            block_type = block.get("type")
            if block_type == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
                continue
            if block_type == "image_url":
                image_url = block.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    detail = image_url.get("detail")
                else:
                    url = image_url
                    detail = None
                converted_image = {"type": "input_image", "image_url": url}
                if detail is not None:
                    converted_image["detail"] = detail
                converted.append(converted_image)
                continue
            converted.append(dict(block))
        return converted

    @staticmethod
    def _stringify_response_tool_output(content: Any) -> str:
        """Return a Responses-compatible function output string."""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    @classmethod
    def _tool_calls_to_response_items(
        cls,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert normalized internal function calls to Responses items."""
        items: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = tool_call.get("function")
            call_id = str(tool_call.get("id") or "")
            if not isinstance(function, dict) or not call_id:
                continue
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": function.get("name"),
                    "arguments": cls._stringify_response_tool_output(
                        function.get("arguments", "{}")
                    ),
                }
            )
        return items

    def apply_prompt_cache_hints(self, payload: dict[str, object]) -> dict[str, object]:
        """Apply official OpenAI prompt-cache and stream-usage hints."""
        if not self.uses_official_openai_api:
            return payload

        prepared = dict(payload)
        if self.settings.prompt_cache_key:
            prepared["prompt_cache_key"] = self.settings.prompt_cache_key

        if prepared.get("stream") is True and not self.uses_responses_api:
            stream_options = dict(prepared.get("stream_options") or {})
            stream_options["include_usage"] = True
            prepared["stream_options"] = stream_options

        return prepared

    def extract_prompt_cache_usage(
        self,
        data: dict[str, object],
    ) -> PromptCacheUsage | None:
        """Extract OpenAI cached input-token counts from response usage."""
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None

        prompt_details = usage.get("input_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = usage.get("prompt_tokens_details")
        cached_tokens = None
        cache_write_tokens = None
        if isinstance(prompt_details, dict):
            cached_value = prompt_details.get("cached_tokens")
            if isinstance(cached_value, int):
                cached_tokens = cached_value
            cache_write_value = prompt_details.get("cache_write_tokens")
            if isinstance(cache_write_value, int):
                cache_write_tokens = cache_write_value

        input_tokens = usage.get("input_tokens")
        if not isinstance(input_tokens, int):
            input_tokens = usage.get("prompt_tokens")
        return PromptCacheUsage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            cached_tokens=cached_tokens,
            cache_read_tokens=cached_tokens,
            cache_creation_tokens=cache_write_tokens,
        )

    def raise_for_non_retryable_error(self, *, status: int, error_text: str) -> None:
        """Do not replay official OpenAI safety, rate-limit, or overload rejections."""
        if not self.uses_official_openai_api:
            return
        try:
            data = json.loads(error_text)
        except (ValueError, TypeError):
            data = None
        error = data.get("error") if isinstance(data, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if code == "misalignment_policy_violation":
            raise ProviderStreamError(
                "OpenAI stopped this conversation for review (misalignment_policy_violation). "
                "Review the conversation and any actions already taken; it will not be retried."
            )
        if status in {429, 503}:
            raise ProviderStreamError(
                f"OpenAI rejected the request (HTTP {status}). "
                "No automatic retry was sent; check rate limits, quota, or availability "
                "before trying again."
            )

    def parse_http_message(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize a response from the selected OpenAI API."""
        if not self.uses_responses_api:
            return super().parse_http_message(data)
        if isinstance(data.get("error"), dict):
            self.raise_for_non_retryable_error(
                status=200, error_text=json.dumps({"error": data["error"]})
            )
        status = data.get("status")
        if "status" in data and status != "completed":
            raise ValueError(f"OpenAI Responses API ended with status {status}")
        item_status = self._noncompleted_output_item_status(data)
        if item_status is not None:
            raise ValueError(
                f"OpenAI Responses API contained an output item with status {item_status}"
            )

        message = self._normalize_responses_message(data)
        self._pending_response_output = self._response_output_items(data)
        return message

    def parse_stream_line(self, line: str) -> StreamParseResult | None:
        """Normalize Chat Completions chunks or typed Responses events."""
        if not self.uses_responses_api:
            return super().parse_stream_line(line)
        payload = _sse_data_payload(line)
        if payload is None:
            return None

        data = json.loads(payload)
        event_type = data.get("type")
        if event_type == "response.created":
            self._streamed_text_seen = False
            return None
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            text_delta = str(data.get("delta") or "")
            self._streamed_text_seen = self._streamed_text_seen or bool(text_delta)
            return StreamParseResult(delta={"content": text_delta})
        if event_type == "response.incomplete":
            raise ProviderStreamError("OpenAI Responses stream ended incomplete")
        if event_type == "response.completed":
            response = data.get("response")
            if not isinstance(response, dict):
                raise ProviderStreamError(
                    "OpenAI Responses stream completed without a response"
                )
            status = response.get("status")
            if "status" in response and status != "completed":
                raise ProviderStreamError(
                    f"OpenAI Responses stream ended with status {status}"
                )
            item_status = self._noncompleted_output_item_status(response)
            if item_status is not None:
                raise ProviderStreamError(
                    "OpenAI Responses stream contained an output item with "
                    f"status {item_status}"
                )
            message = self._normalize_responses_message(response)
            delta: dict[str, Any] = {
                _RESPONSES_OUTPUT_KEY: self._response_output_items(response)
            }
            if message.get("content") and not self._streamed_text_seen:
                delta["content"] = message["content"]
            if message.get("tool_calls"):
                delta["tool_calls"] = [
                    {**tool_call, "index": index}
                    for index, tool_call in enumerate(message["tool_calls"])
                ]
            usage = response.get("usage")
            self._streamed_text_seen = False
            return StreamParseResult(
                delta=delta,
                done=True,
                usage=usage if isinstance(usage, dict) else None,
            )
        if event_type in {"error", "response.failed", "response.cancelled"}:
            error_response = data.get("response", data)
            if isinstance(error_response, dict):
                error = error_response.get("error", error_response)
                self.raise_for_non_retryable_error(
                    status=200, error_text=json.dumps({"error": error})
                )
            raise ProviderStreamError("OpenAI Responses stream failed")
        return None

    @staticmethod
    def _noncompleted_output_item_status(data: dict[str, Any]) -> str | None:
        """Return the first explicit non-completed output-item status."""
        output = data.get("output")
        if not isinstance(output, list):
            return None
        for item in output:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if "status" in item and status != "completed":
                return str(status)
        return None

    @classmethod
    def _normalize_responses_message(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert Responses output items to the internal assistant shape."""
        output = data.get("output")
        if not isinstance(output, list):
            raise ValueError("No response from OpenAI Responses API")

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if not isinstance(content, dict):
                        continue
                    if content.get("type") == "output_text":
                        text = str(content.get("text") or "")
                    elif content.get("type") == "refusal":
                        text = str(content.get("refusal") or "")
                    else:
                        continue
                    if text:
                        text_parts.append(text)
                continue
            if item.get("type") != "function_call":
                continue
            call_id = str(item.get("call_id") or item.get("id") or "")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": cls._stringify_response_tool_output(
                            item.get("arguments", "{}")
                        ),
                    },
                }
            )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    @staticmethod
    def _response_output_items(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return raw output items needed for a stateless follow-up request."""
        output = data.get("output")
        if not isinstance(output, list):
            return []
        return [dict(item) for item in output if isinstance(item, dict)]

    def update_stream_metadata(self, current: Any, delta: dict[str, Any]) -> Any:
        """Capture completed Responses output items for the next tool-loop turn."""
        output = delta.get(_RESPONSES_OUTPUT_KEY)
        return output if isinstance(output, list) else current

    def prepare_stream_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        metadata: Any,
    ) -> list[dict[str, Any]]:
        """Save streamed Responses output items before building assistant history."""
        if self.uses_responses_api and isinstance(metadata, list):
            self._pending_response_output = metadata
        return tool_calls

    def build_assistant_message(
        self, response_text: str, *, metadata: Any = None
    ) -> dict[str, Any]:
        """Keep reasoning and assistant phases when retrying a text-only preamble."""
        message = super().build_assistant_message(response_text, metadata=metadata)
        if self.uses_responses_api:
            output = metadata if isinstance(metadata, list) else self._pending_response_output
            self._pending_response_output = None
            if output:
                message[_RESPONSES_OUTPUT_KEY] = output
        return message

    def build_tool_call_assistant_message(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        response_text: str = "",
    ) -> dict[str, Any]:
        """Build internal tool history and retain Responses reasoning items."""
        message = super().build_tool_call_assistant_message(
            tool_calls,
            response_text=response_text,
        )
        if not self.uses_responses_api:
            return message

        valid_call_ids = {str(call.get("id") or "") for call in tool_calls}
        response_output = self._pending_response_output or []
        self._pending_response_output = None
        filtered_output = [
            item
            for item in response_output
            if item.get("type") != "function_call"
            or str(item.get("call_id") or item.get("id") or "") in valid_call_ids
        ]
        if not any(item.get("type") == "function_call" for item in filtered_output):
            if response_text.strip():
                filtered_output.append(
                    {"role": "assistant", "content": response_text.strip()}
                )
            filtered_output.extend(self._tool_calls_to_response_items(tool_calls))
        message[_RESPONSES_OUTPUT_KEY] = filtered_output
        return message

    @classmethod
    def filter_model_ids(
        cls,
        model_ids: list[str],
        *,
        base_url: str,
        values: dict[str, Any] | None = None,
    ) -> list[str]:
        """Filter official OpenAI results for the selected generation API."""
        if cls._is_official_openai_base_url(base_url):
            configured = cls._configured_transport_from_values(values)
            transport = cls._resolve_api_transport(configured, base_url)
            model_ids = [
                model_id
                for model_id in model_ids
                if (model_id.startswith("gpt-") or cls.is_reasoning_model(model_id))
                and not cls.is_deep_research_model(model_id)
                and (
                    (
                        transport == OPENAI_API_TRANSPORT_RESPONSES
                        and not cls.is_chat_completions_only_model(model_id)
                    )
                    or (
                        transport == OPENAI_API_TRANSPORT_CHAT_COMPLETIONS
                        and not cls.requires_responses_for_tools(model_id)
                    )
                )
            ]
        return sorted((model_id for model_id in model_ids if model_id), reverse=True)

    @staticmethod
    def is_chat_completions_only_model(model_name: str) -> bool:
        """Return whether OpenAI documents a model family as Chat-only."""
        name = str(model_name or "").strip().lower().rsplit("/", 1)[-1]
        return name.startswith(_CHAT_COMPLETIONS_ONLY_MODEL_PREFIXES)

    @classmethod
    def requires_responses_for_tools(cls, model_name: str) -> bool:
        """Include Astra, whose Chat Completions support excludes tool calling."""
        return cls.is_responses_only_model(model_name) or cls._matches_model_family(
            model_name, "gpt-6-astra"
        )

    @staticmethod
    def is_deep_research_model(model_name: str) -> bool:
        """Return whether a model requires OpenAI built-in data-source tools."""
        name = str(model_name or "").strip().lower().rsplit("/", 1)[-1]
        return "deep-research" in name

    @classmethod
    def model_configuration_error(
        cls,
        model_name: str,
        *,
        base_url: str,
        values: dict[str, Any] | None = None,
    ) -> str | None:
        """Reject known official-OpenAI model and transport mismatches."""
        if not cls._is_official_openai_base_url(base_url):
            return None
        if cls.is_deep_research_model(model_name):
            return "deep_research_model_not_supported"
        configured = cls._configured_transport_from_values(values)
        transport = cls._resolve_api_transport(configured, base_url)
        if (
            transport == OPENAI_API_TRANSPORT_CHAT_COMPLETIONS
            and cls.requires_responses_for_tools(model_name)
        ):
            return "model_requires_responses_api"
        if (
            transport == OPENAI_API_TRANSPORT_RESPONSES
            and cls.is_chat_completions_only_model(model_name)
        ):
            return "model_requires_chat_completions_api"
        return None
