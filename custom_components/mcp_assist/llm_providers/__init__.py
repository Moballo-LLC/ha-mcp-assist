"""LLM provider transports for MCP Assist."""

from __future__ import annotations

from typing import Any

from .anthropic import AnthropicProvider
from .base import (
    LLMProvider,
    ProviderSettings,
    StreamParseResult,
    normalize_tool_call_arguments,
    parse_tool_arguments,
    stringify_tool_arguments,
)
from .gemini import GeminiProvider
from .llamacpp import LlamaCppProvider
from .lmstudio import LMStudioProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider
from .vllm import VLLMProvider
from ..provider_runtime import resolve_provider_runtime_config
from ..const import (
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_OLLAMA,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_VLLM,
)


def _provider_classes() -> dict[str, type[LLMProvider]]:
    """Return the configured provider class registry."""
    return {
        SERVER_TYPE_LMSTUDIO: LMStudioProvider,
        SERVER_TYPE_LLAMACPP: LlamaCppProvider,
        SERVER_TYPE_OLLAMA: OllamaProvider,
        SERVER_TYPE_OPENAI: OpenAIProvider,
        SERVER_TYPE_GEMINI: GeminiProvider,
        SERVER_TYPE_ANTHROPIC: AnthropicProvider,
        SERVER_TYPE_OPENROUTER: OpenRouterProvider,
        SERVER_TYPE_VLLM: VLLMProvider,
    }


def build_provider_settings(
    entry: Any,
    *,
    max_tokens: int,
    temperature: float | None,
) -> ProviderSettings:
    """Build current provider settings from a conversation profile entry."""
    runtime_config = resolve_provider_runtime_config(entry)
    provider_class = _provider_classes().get(
        runtime_config.server_type,
        LMStudioProvider,
    )
    return ProviderSettings(
        server_type=runtime_config.server_type,
        model_name=runtime_config.model_name,
        api_key=runtime_config.api_key,
        base_url=runtime_config.base_url,
        timeout=runtime_config.timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        provider_options=provider_class.options_from_entry(entry),
        display_name=runtime_config.display_name,
        is_remote_service=runtime_config.is_remote_service,
    )


def create_llm_provider(settings: ProviderSettings) -> LLMProvider:
    """Create the provider transport for the configured server type."""
    provider_class = _provider_classes().get(settings.server_type, LMStudioProvider)
    return provider_class(settings)


__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "LLMProvider",
    "LlamaCppProvider",
    "LMStudioProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderSettings",
    "StreamParseResult",
    "VLLMProvider",
    "build_provider_settings",
    "create_llm_provider",
    "normalize_tool_call_arguments",
    "parse_tool_arguments",
    "stringify_tool_arguments",
]
