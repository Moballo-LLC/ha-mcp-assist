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
from custom_components.mcp_assist.llm_providers import gemini as gemini_module
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
    display_name: str = "Test Provider",
    is_remote_service: bool = False,
) -> ProviderSettings:
    """Build provider settings for unit tests."""
    return ProviderSettings(
        server_type=server_type,
        model_name=model_name,
        api_key="test-key",
        base_url="https://provider.example.invalid",
        timeout=30,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_options=provider_options or {},
        display_name=display_name,
        is_remote_service=is_remote_service,
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
