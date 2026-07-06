"""Tests for LLM provider transport helpers."""

from __future__ import annotations

import logging

import pytest

from custom_components.mcp_assist.const import (
    CONF_API_KEY,
    CONF_LMSTUDIO_URL,
    CONF_OLLAMA_KEEP_ALIVE,
    CONF_OLLAMA_NUM_CTX,
    CONF_OPENCLAW_HOST,
    CONF_OPENCLAW_PORT,
    CONF_OPENCLAW_SESSION_KEY,
    CONF_OPENCLAW_TOKEN,
    CONF_OPENCLAW_USE_SSL,
    DEFAULT_LLAMACPP_URL,
    DEFAULT_LMSTUDIO_URL,
    DEFAULT_OLLAMA_KEEP_ALIVE,
    DEFAULT_OLLAMA_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OPENCLAW_HOST,
    DEFAULT_OPENCLAW_PORT,
    DEFAULT_OPENCLAW_SESSION_KEY,
    DEFAULT_OPENCLAW_USE_SSL,
    DEFAULT_VLLM_URL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENCLAW,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_VLLM,
)
from custom_components.mcp_assist.llm_providers import base as base_module
from custom_components.mcp_assist.llm_providers import gemini as gemini_module
from custom_components.mcp_assist.llm_providers import ollama as ollama_module
from custom_components.mcp_assist.llm_providers import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    LlamaCppProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenClawProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderSettings,
    VLLMProvider,
    get_llm_provider_class,
    provider_selector_options,
    create_llm_provider,
)


PROVIDER_CLASSES: tuple[tuple[str, type[LLMProvider]], ...] = (
    (SERVER_TYPE_LMSTUDIO, LMStudioProvider),
    (SERVER_TYPE_LLAMACPP, LlamaCppProvider),
    (SERVER_TYPE_OLLAMA, OllamaProvider),
    (SERVER_TYPE_OPENAI, OpenAIProvider),
    (SERVER_TYPE_GEMINI, GeminiProvider),
    (SERVER_TYPE_ANTHROPIC, AnthropicProvider),
    (SERVER_TYPE_OPENROUTER, OpenRouterProvider),
    (SERVER_TYPE_OPENCLAW, OpenClawProvider),
    (SERVER_TYPE_VLLM, VLLMProvider),
)


def _settings(
    server_type: str = SERVER_TYPE_LMSTUDIO,
    *,
    model_name: str = "test-model",
    max_tokens: int = 100,
    temperature: float | None = 0.25,
    provider_options: dict[str, object] | None = None,
    base_url: str = "https://provider.example.invalid",
    display_name: str = "Test Provider",
    is_remote_service: bool = False,
    prompt_cache_key: str | None = None,
) -> ProviderSettings:
    """Build provider settings for unit tests."""
    return ProviderSettings(
        server_type=server_type,
        model_name=model_name,
        api_key="test-key",
        base_url=base_url,
        timeout=30,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_options=provider_options or {},
        display_name=display_name,
        is_remote_service=is_remote_service,
        prompt_cache_key=prompt_cache_key,
    )


@pytest.mark.parametrize(
    ("server_type", "provider_class"),
    PROVIDER_CLASSES,
)
def test_create_llm_provider_returns_provider_class(
    server_type: str,
    provider_class: type[LLMProvider],
) -> None:
    """Each configured server type should get its own provider class."""
    assert isinstance(create_llm_provider(_settings(server_type)), provider_class)


def test_provider_selector_options_are_owned_by_provider_classes() -> None:
    """Provider labels shown in config flow should come from provider metadata."""
    assert provider_selector_options() == [
        {"value": server_type, "label": provider_class.config_display_name()}
        for server_type, provider_class in PROVIDER_CLASSES
    ]


def test_ollama_detects_llama_server_invalid_tool_argument_errors() -> None:
    """Ollama should own llama-server malformed tool-call error detection."""
    provider = OllamaProvider(_settings(SERVER_TYPE_OLLAMA))

    assert provider.is_invalid_tool_arguments_error(
        status=500,
        error_text=(
            '{"error":"llama-server returned invalid tool call arguments for '
            '\\"discover_entities\\": unexpected end of JSON input"}'
        ),
    )
    assert not provider.is_invalid_tool_arguments_error(
        status=400,
        error_text="invalid tool call arguments: unexpected end of JSON input",
    )
    assert not provider.is_invalid_tool_arguments_error(
        status=500,
        error_text="model not loaded",
    )


@pytest.mark.parametrize(
    (
        "server_type",
        "provider_class",
        "display_name",
        "connection_fields",
        "provider_options_fields",
        "default_base_url",
        "model_fetch_error",
        "uses_config_model_step",
        "supports_streaming",
    ),
    [
        (
            SERVER_TYPE_LMSTUDIO,
            LMStudioProvider,
            "LM Studio",
            ((CONF_LMSTUDIO_URL, DEFAULT_LMSTUDIO_URL, "text", True),),
            (),
            DEFAULT_LMSTUDIO_URL,
            "cannot_connect",
            True,
            True,
        ),
        (
            SERVER_TYPE_LLAMACPP,
            LlamaCppProvider,
            "llama.cpp",
            ((CONF_LMSTUDIO_URL, DEFAULT_LLAMACPP_URL, "text", True),),
            (),
            DEFAULT_LLAMACPP_URL,
            "cannot_connect",
            True,
            True,
        ),
        (
            SERVER_TYPE_OLLAMA,
            OllamaProvider,
            "Ollama",
            ((CONF_LMSTUDIO_URL, DEFAULT_OLLAMA_URL, "text", True),),
            (
                (CONF_OLLAMA_NUM_CTX, DEFAULT_OLLAMA_NUM_CTX, "integer", False),
                (CONF_OLLAMA_KEEP_ALIVE, DEFAULT_OLLAMA_KEEP_ALIVE, "text", False),
            ),
            DEFAULT_OLLAMA_URL,
            "cannot_connect",
            True,
            True,
        ),
        (
            SERVER_TYPE_OPENAI,
            OpenAIProvider,
            "OpenAI",
            (
                (CONF_LMSTUDIO_URL, OPENAI_BASE_URL, "text", True),
                (CONF_API_KEY, None, "password", True),
            ),
            (),
            OPENAI_BASE_URL,
            "invalid_api_key",
            True,
            True,
        ),
        (
            SERVER_TYPE_GEMINI,
            GeminiProvider,
            "Gemini",
            ((CONF_API_KEY, None, "password", True),),
            (),
            None,
            "invalid_api_key",
            True,
            True,
        ),
        (
            SERVER_TYPE_ANTHROPIC,
            AnthropicProvider,
            "Claude",
            ((CONF_API_KEY, None, "password", True),),
            (),
            None,
            None,
            True,
            False,
        ),
        (
            SERVER_TYPE_OPENROUTER,
            OpenRouterProvider,
            "OpenRouter",
            ((CONF_API_KEY, None, "password", True),),
            (),
            OPENROUTER_BASE_URL,
            "invalid_api_key",
            True,
            True,
        ),
        (
            SERVER_TYPE_OPENCLAW,
            OpenClawProvider,
            "OpenClaw",
            (
                (CONF_OPENCLAW_HOST, DEFAULT_OPENCLAW_HOST, "text", True),
                (CONF_OPENCLAW_PORT, DEFAULT_OPENCLAW_PORT, "integer", True),
                (CONF_OPENCLAW_TOKEN, None, "password", True),
                (CONF_OPENCLAW_USE_SSL, DEFAULT_OPENCLAW_USE_SSL, "boolean", True),
            ),
            (
                (CONF_OPENCLAW_SESSION_KEY, DEFAULT_OPENCLAW_SESSION_KEY, "text", False),
            ),
            None,
            None,
            False,
            False,
        ),
        (
            SERVER_TYPE_VLLM,
            VLLMProvider,
            "vLLM",
            ((CONF_LMSTUDIO_URL, DEFAULT_VLLM_URL, "text", True),),
            (),
            DEFAULT_VLLM_URL,
            "cannot_connect",
            True,
            True,
        ),
    ],
)
def test_provider_classes_expose_config_metadata(
    server_type: str,
    provider_class: type[LLMProvider],
    display_name: str,
    connection_fields: tuple[tuple[str, object, str, bool], ...],
    provider_options_fields: tuple[tuple[str, object, str, bool], ...],
    default_base_url: str | None,
    model_fetch_error: str | None,
    uses_config_model_step: bool,
    supports_streaming: bool,
) -> None:
    """Each provider should own the metadata consumed by config and options flows."""
    assert get_llm_provider_class(server_type) is provider_class
    assert provider_class.provider_type == server_type
    assert provider_class.config_display_name() == display_name
    assert provider_class.default_base_url == default_base_url
    assert provider_class.model_fetch_error == model_fetch_error
    assert provider_class.uses_config_model_step is uses_config_model_step
    assert provider_class.supports_streaming is supports_streaming
    assert tuple(
        (field.key, field.default, field.kind, field.required)
        for field in provider_class.connection_fields
    ) == connection_fields
    assert tuple(
        (field.key, field.default, field.kind, field.required)
        for field in provider_class.provider_options_fields
    ) == provider_options_fields


def test_openai_compatible_provider_owns_versioned_endpoints() -> None:
    """OpenAI-compatible providers should own root-vs-/v1 endpoint handling."""
    root_provider = OpenAIProvider(
        _settings(SERVER_TYPE_OPENAI, base_url="https://api.example.invalid")
    )
    v1_provider = OpenAIProvider(
        _settings(SERVER_TYPE_OPENAI, base_url="https://api.example.invalid/v1")
    )

    assert root_provider.chat_url() == (
        "https://api.example.invalid/v1/chat/completions"
    )
    assert v1_provider.chat_url() == (
        "https://api.example.invalid/v1/chat/completions"
    )
    assert OpenAIProvider.model_list_url(
        {CONF_LMSTUDIO_URL: "https://api.example.invalid/v1/"}
    ) == "https://api.example.invalid/v1/models"


def test_gemini_provider_owns_openai_compatible_endpoint_shape() -> None:
    """Gemini's OpenAI-compatible path should not get an extra /v1 segment."""
    provider = GeminiProvider(
        _settings(
            SERVER_TYPE_GEMINI,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
    )

    assert provider.chat_url() == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


async def test_ollama_provider_fetches_models_from_native_tags_endpoint(
    monkeypatch,
) -> None:
    """Ollama should own native supported-model discovery."""
    calls: list[str] = []

    class TagsResponse:
        status = 200

        async def __aenter__(self) -> "TagsResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def json(self) -> dict[str, list[dict[str, str]]]:
            return {
                "models": [
                    {"name": "llama3.2"},
                    {"model": "qwen3"},
                ]
            }

    class TagsSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "TagsSession":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def get(self, url: str) -> TagsResponse:
            calls.append(url)
            return TagsResponse()

    monkeypatch.setattr(OllamaProvider, "model_fetch_delay", 0)
    monkeypatch.setattr(ollama_module.aiohttp, "ClientSession", TagsSession)

    models = await OllamaProvider.fetch_models(
        None,
        {CONF_LMSTUDIO_URL: "http://ollama.example.invalid"},
    )

    assert calls == ["http://ollama.example.invalid/api/tags"]
    assert models == ["llama3.2", "qwen3"]


async def test_openai_model_fetch_redacts_base_url_userinfo(
    monkeypatch,
    caplog,
) -> None:
    """Provider model-fetch logs should not expose credentials embedded in URLs."""

    class ModelsResponse:
        status = 200

        async def __aenter__(self) -> "ModelsResponse":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def json(self) -> dict[str, list[dict[str, str]]]:
            return {"data": [{"id": "gpt-5-mini"}]}

    class ModelsSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "ModelsSession":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def get(self, *args: object, **kwargs: object) -> ModelsResponse:
            return ModelsResponse()

    monkeypatch.setattr(base_module.aiohttp, "ClientSession", ModelsSession)

    with caplog.at_level(
        logging.INFO,
        logger="custom_components.mcp_assist.llm_providers.base",
    ):
        models = await OpenAIProvider.fetch_models(
            None,
            {
                CONF_LMSTUDIO_URL: "https://user:pass@proxy.example.invalid/v1",
                CONF_API_KEY: "sk-test",
            },
        )

    assert models == ["gpt-5-mini"]
    assert "user:pass" not in caplog.text
    assert "https://[redacted]@proxy.example.invalid/v1" in caplog.text


@pytest.mark.parametrize(
    ("server_type", "provider_class"),
    [
        (SERVER_TYPE_LMSTUDIO, LMStudioProvider),
        (SERVER_TYPE_LLAMACPP, LlamaCppProvider),
        (SERVER_TYPE_OPENAI, OpenAIProvider),
        (SERVER_TYPE_GEMINI, GeminiProvider),
        (SERVER_TYPE_OPENROUTER, OpenRouterProvider),
        (SERVER_TYPE_VLLM, VLLMProvider),
    ],
)
def test_openai_compatible_providers_build_chat_payloads(
    server_type: str,
    provider_class: type[LLMProvider],
) -> None:
    """OpenAI-compatible providers should share the standard chat payload shape."""
    provider = provider_class(_settings(server_type, max_tokens=50, temperature=0.4))

    payload = provider.build_payload(
        [{"role": "user", "content": "Hello"}],
        [{"type": "function", "function": {"name": "ping", "parameters": {}}}],
        stream=True,
    )

    assert payload == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "temperature": 0.4,
        "max_tokens": 50,
        "tools": [{"type": "function", "function": {"name": "ping", "parameters": {}}}],
        "tool_choice": "auto",
    }


def test_openclaw_provider_rejects_http_payload_path() -> None:
    """OpenClaw should stay registered without pretending to be an HTTP transport."""
    provider = OpenClawProvider(_settings(SERVER_TYPE_OPENCLAW))

    with pytest.raises(RuntimeError, match="bypass"):
        provider.build_payload([{"role": "user", "content": "Hello"}])


def test_openai_provider_uses_completion_tokens_for_gpt5() -> None:
    """GPT-5 and o1-family models should use OpenAI's newer token limit field."""
    provider = OpenAIProvider(
        _settings(SERVER_TYPE_OPENAI, model_name="gpt-5-mini", max_tokens=321)
    )

    payload = provider.build_payload(
        [{"role": "user", "content": "Hello"}],
        stream=False,
    )

    assert payload["max_completion_tokens"] == 321
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.parametrize("model_name", ["o3", "o3-mini", "o4-mini", "o1-preview"])
def test_openai_provider_uses_completion_tokens_for_o_series(model_name: str) -> None:
    """o3/o4-series reasoning models must use max_completion_tokens, no temperature."""
    provider = OpenAIProvider(
        _settings(SERVER_TYPE_OPENAI, model_name=model_name, max_tokens=200)
    )

    payload = provider.build_payload(
        [{"role": "user", "content": "Hello"}],
        stream=False,
    )

    assert payload["max_completion_tokens"] == 200
    assert "max_tokens" not in payload
    assert "temperature" not in payload


@pytest.mark.parametrize("model_name", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"])
def test_openai_provider_keeps_standard_params_for_chat_models(model_name: str) -> None:
    """Non-reasoning chat models keep max_tokens and temperature."""
    provider = OpenAIProvider(
        _settings(SERVER_TYPE_OPENAI, model_name=model_name, max_tokens=200, temperature=0.4)
    )

    payload = provider.build_payload(
        [{"role": "user", "content": "Hello"}],
        stream=False,
    )

    assert payload["max_tokens"] == 200
    assert payload["temperature"] == 0.4
    assert "max_completion_tokens" not in payload


def test_is_reasoning_model_classification() -> None:
    """The reasoning-model heuristic should match the o-series and GPT-5 only."""
    is_reasoning = OpenAIProvider.is_reasoning_model
    assert is_reasoning("o1")
    assert is_reasoning("o3-mini")
    assert is_reasoning("o4-mini")
    assert is_reasoning("gpt-5")
    assert is_reasoning("openai/o3-mini")  # OpenRouter-style prefix
    assert not is_reasoning("gpt-4o")
    assert not is_reasoning("gpt-4o-mini")
    assert not is_reasoning("omni-model")  # "o" not followed by a digit
    assert not is_reasoning("llama-3.1-8b")
    assert not is_reasoning("")


def test_openai_filter_model_ids_keeps_reasoning_models() -> None:
    """The official OpenAI model list should include o-series reasoning models."""
    filtered = OpenAIProvider.filter_model_ids(
        ["gpt-4o", "o3-mini", "o1", "whisper-1", "text-embedding-3-large", "o4-mini"],
        base_url=OPENAI_BASE_URL,
    )

    assert "o3-mini" in filtered
    assert "o1" in filtered
    assert "o4-mini" in filtered
    assert "gpt-4o" in filtered
    assert "whisper-1" not in filtered
    assert "text-embedding-3-large" not in filtered


def test_openai_filter_model_ids_excludes_responses_only_models() -> None:
    """Responses-API-only o-series models must not appear in the chat dropdown."""
    filtered = OpenAIProvider.filter_model_ids(
        ["gpt-4o", "o3", "o3-mini", "o1-pro", "o3-pro", "o3-pro-2025-06-10", "o3-deep-research", "o4-mini-deep-research"],
        base_url=OPENAI_BASE_URL,
    )

    # Chat-completions reasoning models stay.
    assert "o3" in filtered
    assert "o3-mini" in filtered
    assert "gpt-4o" in filtered
    # Responses-only / deep-research variants (incl. dated snapshots) are excluded.
    assert "o1-pro" not in filtered
    assert "o3-pro" not in filtered
    assert "o3-pro-2025-06-10" not in filtered
    assert "o3-deep-research" not in filtered
    assert "o4-mini-deep-research" not in filtered


def test_is_responses_only_model_classification() -> None:
    """Only deep-research and o-series *-pro models are Responses-only."""
    is_responses_only = OpenAIProvider.is_responses_only_model
    assert is_responses_only("o3-pro")
    assert is_responses_only("o1-pro")
    assert is_responses_only("o3-deep-research")
    assert is_responses_only("o4-mini-deep-research")
    assert is_responses_only("openai/o3-pro")
    # Dated snapshot IDs must also be caught.
    assert is_responses_only("o3-pro-2025-06-10")
    assert is_responses_only("o1-pro-2025-03-19")
    assert is_responses_only("o3-deep-research-2025-06-26")
    assert not is_responses_only("o3")
    assert not is_responses_only("o3-mini")
    assert not is_responses_only("gpt-4o")
    # "pro" must be a segment, not any substring.
    assert not is_responses_only("gpt-4-proxy")


def test_openai_provider_applies_prompt_cache_key_and_stream_usage() -> None:
    """Official OpenAI requests should opt into cache routing and stream usage."""
    provider = OpenAIProvider(
        _settings(
            SERVER_TYPE_OPENAI,
            base_url=OPENAI_BASE_URL,
            prompt_cache_key="ha-mcp-assist-cache-key",
        )
    )

    payload = provider.prepare_payload(
        provider.build_payload(
            [{"role": "user", "content": "Hello"}],
            stream=True,
        )
    )

    assert payload["prompt_cache_key"] == "ha-mcp-assist-cache-key"
    assert payload["stream_options"] == {"include_usage": True}


def test_openai_provider_skips_prompt_cache_fields_for_custom_base_url() -> None:
    """Custom OpenAI-compatible endpoints should not receive OpenAI-only fields."""
    provider = OpenAIProvider(
        _settings(
            SERVER_TYPE_OPENAI,
            base_url="https://openai-compatible.example.invalid",
            prompt_cache_key="ha-mcp-assist-cache-key",
        )
    )

    payload = provider.prepare_payload(
        provider.build_payload(
            [{"role": "user", "content": "Hello"}],
            stream=True,
        )
    )

    assert "prompt_cache_key" not in payload
    assert "stream_options" not in payload


def test_openai_compatible_providers_do_not_emit_openai_cache_fields() -> None:
    """OpenAI-compatible local endpoints should not receive OpenAI-only fields."""
    provider = LMStudioProvider(
        _settings(
            SERVER_TYPE_LMSTUDIO,
            prompt_cache_key="ha-mcp-assist-cache-key",
        )
    )

    payload = provider.prepare_payload(
        provider.build_payload(
            [{"role": "user", "content": "Hello"}],
            stream=True,
        )
    )

    assert "prompt_cache_key" not in payload
    assert "stream_options" not in payload


def test_openai_provider_extracts_prompt_cache_usage() -> None:
    """OpenAI usage metadata should expose cached prompt tokens."""
    provider = OpenAIProvider(_settings(SERVER_TYPE_OPENAI))

    usage = provider.extract_prompt_cache_usage(
        {
            "usage": {
                "prompt_tokens": 4096,
                "prompt_tokens_details": {"cached_tokens": 3072},
            }
        }
    )

    assert usage is not None
    assert usage.input_tokens == 4096
    assert usage.cached_tokens == 3072
    assert usage.cache_read_tokens == 3072


def test_stream_parser_allows_usage_only_chunks() -> None:
    """Streaming usage chunks should not be treated as malformed deltas."""
    provider = OpenAIProvider(_settings(SERVER_TYPE_OPENAI))

    parsed = provider.parse_stream_line(
        'data: {"choices":[],"usage":{"prompt_tokens":4096,'
        '"prompt_tokens_details":{"cached_tokens":2048}}}'
    )

    assert parsed is not None
    assert parsed.delta == {}
    assert parsed.usage == {
        "prompt_tokens": 4096,
        "prompt_tokens_details": {"cached_tokens": 2048},
    }


def test_ollama_provider_uses_native_tool_shapes() -> None:
    """Ollama should receive native tool arguments and tool-result identifiers."""
    provider = OllamaProvider(
        _settings(
            SERVER_TYPE_OLLAMA,
            provider_options={"keep_alive": "-1", "context_window": 8192},
        )
    )

    payload = provider.build_payload(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "discover_entities",
                            "arguments": '{"area": "Kitchen"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "tool_name": "discover_entities",
                "content": "Kitchen light",
            },
        ],
        stream=False,
    )
    tool_result = provider.build_tool_result_message(
        tool_call_id="call_1",
        tool_name="discover_entities",
        content="Kitchen light",
    )
    formatted_tool_call = provider.format_tool_call(
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "discover_entities",
                "arguments": '{"area": "Office"}',
            },
        }
    )

    assert payload["keep_alive"] == -1
    assert payload["options"]["num_ctx"] == 8192
    assert payload["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
        "area": "Kitchen"
    }
    assert payload["messages"][1] == {
        "role": "tool",
        "tool_name": "discover_entities",
        "content": "Kitchen light",
    }
    assert tool_result == {
        "role": "tool",
        "tool_name": "discover_entities",
        "content": "Kitchen light",
    }
    assert formatted_tool_call["function"]["arguments"] == {"area": "Office"}


def test_gemini_provider_captures_streamed_thought_signature() -> None:
    """Gemini's streamed thought signature should stay in the Gemini provider."""
    provider = GeminiProvider(_settings(SERVER_TYPE_GEMINI))
    delta = {
        "tool_calls": [
            {
                "extra_content": {
                    "google": {"thought_signature": "thought-signature"}
                }
            }
        ]
    }
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "discover_entities", "arguments": "{}"},
        }
    ]

    metadata = provider.update_stream_metadata(None, delta)
    prepared = provider.prepare_stream_tool_calls(tool_calls, metadata)

    assert metadata == "thought-signature"
    assert prepared[0]["extra_content"] == {
        "google": {"thought_signature": "thought-signature"}
    }
    assert "extra_content" not in tool_calls[0]
    assert provider.missing_stream_metadata_warning(metadata) is None


async def test_gemini_model_fetch_redacts_exception_urls(monkeypatch, caplog) -> None:
    """Gemini model-fetch failures should not log API keys embedded in URLs."""

    class RaisingSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "RaisingSession":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def get(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError(
                "Cannot decode https://generativelanguage.googleapis.com/v1beta/"
                "models?key=gemini-secret"
            )

    monkeypatch.setattr(gemini_module.aiohttp, "ClientSession", RaisingSession)

    with caplog.at_level(
        logging.ERROR,
        logger="custom_components.mcp_assist.llm_providers.gemini",
    ):
        models = await GeminiProvider.fetch_models(
            None,
            {CONF_API_KEY: "gemini-secret"},
        )

    assert models == []
    assert "gemini-secret" not in caplog.text
    assert "key=[redacted]" in caplog.text


def test_anthropic_provider_uses_native_messages_shape() -> None:
    """Anthropic should receive native messages and filtered native tool schemas."""
    provider = AnthropicProvider(
        _settings(SERVER_TYPE_ANTHROPIC, model_name="claude-sonnet-4-5")
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "discover_entities",
                "description": "Find entities.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_image",
                "description": "Analyze an image.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    payload = provider.build_payload(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Find kitchen lights."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "discover_entities",
                            "arguments": '{"area":"Kitchen"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "Kitchen light"},
        ],
        tools,
    )

    assert provider.supports_streaming is False
    assert payload["system"] == "You are helpful."
    assert [tool["name"] for tool in payload["tools"]] == ["discover_entities"]
    assert payload["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "Kitchen light",
            }
        ],
    }


def test_anthropic_provider_normalizes_http_message() -> None:
    """Anthropic response parsing should stay in the Anthropic provider."""
    provider = AnthropicProvider(_settings(SERVER_TYPE_ANTHROPIC))

    message = provider.parse_http_message(
        {
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "discover_entities",
                    "input": {"area": "Kitchen"},
                },
            ],
            "stop_reason": "tool_use",
        }
    )

    assert message == {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {
                    "name": "discover_entities",
                    "arguments": '{"area": "Kitchen"}',
                },
            }
        ],
    }
