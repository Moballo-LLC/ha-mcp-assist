"""Resolve selected models through the standard OpenAI-compatible API."""

from __future__ import annotations

from contextvars import ContextVar
import json
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp

from .const import SERVER_TYPE_OPENAI
from .llm_providers.openai import OpenAIProvider
from .provider_runtime import ProviderRuntimeConfig, build_provider_auth_headers

_MAX_BYTES = 1024 * 1024
MCP_MODEL_REQUEST_HEADER = "X-MCP-Assist-Model-Request"
ModelSelectionSnapshot = tuple[Any, str | None, str | None]
REQUEST_MODEL_SELECTION: ContextVar[ModelSelectionSnapshot | None] = ContextVar(
    "mcp_assist_model_selection", default=None
)


class ModelResolutionError(ValueError):
    """Safe failure before generation when model retrieval fails."""

    def __init__(self) -> None:
        super().__init__("The selected model could not be retrieved from the endpoint.")


async def async_resolve_model(runtime: ProviderRuntimeConfig, model: str) -> str:
    """Freeze a model ID once for a logical request to a custom endpoint.

    Retrieval is opt-in for endpoints that resolve aliases. Ordinary compatible
    endpoints receive the selected ID directly without a metadata request.
    """
    if (not runtime.resolve_model_before_request
            or runtime.server_type != SERVER_TYPE_OPENAI or not model):
        return model
    if OpenAIProvider._is_official_openai_base_url(runtime.base_url):
        return model
    if not isinstance(model, str) or len(model) > 512:
        raise ModelResolutionError()
    try:
        parts = urlsplit(runtime.base_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment):
            raise ModelResolutionError()
        endpoint = OpenAIProvider.provider_endpoint(runtime.base_url, "models")
        url = f"{endpoint}/{quote(model, safe='/')}"
        headers = build_provider_auth_headers(runtime.server_type, runtime.api_key)
        timeout = aiohttp.ClientTimeout(total=min(max(runtime.timeout, 1), 15))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, allow_redirects=False) as response:
                if response.status != 200:
                    raise ModelResolutionError()
                if response.content_length is not None and response.content_length > _MAX_BYTES:
                    raise ModelResolutionError()
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > _MAX_BYTES:
                        raise ModelResolutionError()
                payload = json.loads(body)
                result = payload.get("id") if isinstance(payload, dict) else None
                if not isinstance(result, str) or not result.strip() or len(result) > 512:
                    raise ModelResolutionError()
                return result
    except Exception:
        raise ModelResolutionError() from None
