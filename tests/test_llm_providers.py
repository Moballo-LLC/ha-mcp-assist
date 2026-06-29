"""Tests for LLM provider transport helpers."""

from __future__ import annotations

import pytest

from custom_components.mcp_assist.const import (
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_VLLM,
)
from custom_components.mcp_assist.llm_providers import (
    AnthropicProvider,
    GeminiProvider,
    LlamaCppProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderSettings,
    VLLMProvider,
    create_llm_provider,
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
    [
        (SERVER_TYPE_LMSTUDIO, LMStudioProvider),
        (SERVER_TYPE_LLAMACPP, LlamaCppProvider),
        (SERVER_TYPE_OLLAMA, OllamaProvider),
        (SERVER_TYPE_OPENAI, OpenAIProvider),
        (SERVER_TYPE_GEMINI, GeminiProvider),
        (SERVER_TYPE_ANTHROPIC, AnthropicProvider),
        (SERVER_TYPE_OPENROUTER, OpenRouterProvider),
        (SERVER_TYPE_VLLM, VLLMProvider),
    ],
)
def test_create_llm_provider_returns_provider_class(
    server_type: str,
    provider_class: type,
) -> None:
    """Each configured server type should get its own provider class."""
    assert isinstance(create_llm_provider(_settings(server_type)), provider_class)


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
