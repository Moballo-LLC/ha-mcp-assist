"""Regression tests for provider-owned image transport dispatch."""

from __future__ import annotations

import base64
from typing import Any

import pytest

import custom_components.mcp_assist.mcp_server as mcp_server_module
from custom_components.mcp_assist.const import (
    CONF_API_KEY,
    CONF_HERMES_URL,
    CONF_LMSTUDIO_URL,
    CONF_MODEL_NAME,
    CONF_OPENAI_API_TRANSPORT,
    CONF_OPENAI_IMAGE_API,
    CONF_OPENAI_IMAGE_MODEL,
    CONF_SERVER_TYPE,
    OPENAI_API_TRANSPORT_AUTO,
    OPENAI_API_TRANSPORT_CHAT_COMPLETIONS,
    OPENAI_API_TRANSPORT_RESPONSES,
    OPENAI_BASE_URL,
    SERVER_TYPE_GEMINI,
    SERVER_TYPE_HERMES,
    SERVER_TYPE_LLAMACPP,
    SERVER_TYPE_LMSTUDIO,
    SERVER_TYPE_ANTHROPIC,
    SERVER_TYPE_OPENAI,
    SERVER_TYPE_OPENROUTER,
    SERVER_TYPE_VLLM,
)
from custom_components.mcp_assist.mcp_server import MCPServer
from custom_components.mcp_assist.agent import MCPAssistConversationEntity


def _image_response() -> dict[str, Any]:
    return {
        "data": [{"b64_json": base64.b64encode(b"fake-png").decode("ascii")}]
    }


def _install_image_response(monkeypatch, response_body: dict[str, Any]):
    calls: list[dict[str, Any]] = []

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def json(self):
            return response_body

    class _Session:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]):
            calls.append({"url": url, "headers": headers, "payload": json})
            return _Response()

    monkeypatch.setattr(mcp_server_module.aiohttp, "ClientSession", _Session)
    return calls


async def _dispatch_image(server: MCPServer, **arguments):
    return await server.handle_tool_call(
        {"name": "generate_image", "arguments": {"prompt": "Draw a house", **arguments}}
    )


@pytest.mark.parametrize(
    ("server_type", "expected_url", "expected_headers"),
    [
        (
            SERVER_TYPE_LMSTUDIO,
            "https://provider.example.invalid/v1/images/generations",
            {},
        ),
        (
            SERVER_TYPE_LLAMACPP,
            "https://provider.example.invalid/v1/images/generations",
            {},
        ),
        (
            SERVER_TYPE_VLLM,
            "https://provider.example.invalid/v1/images/generations",
            {},
        ),
        (
            SERVER_TYPE_GEMINI,
            "https://generativelanguage.googleapis.com/v1beta/openai/images/generations",
            {"Authorization": "Bearer test-provider-key"},
        ),
        (
            SERVER_TYPE_OPENROUTER,
            "https://openrouter.ai/api/v1/images/generations",
            {
                "Authorization": "Bearer test-provider-key",
                "HTTP-Referer": "https://github.com/Moballo-LLC/ha-mcp-assist",
                "X-Title": "MCP Assist for Home Assistant",
            },
        ),
        (
            SERVER_TYPE_HERMES,
            "https://provider.example.invalid/v1/images/generations",
            {"Authorization": "Bearer test-provider-key"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_compatible_provider_image_dispatch_keeps_owned_images_route(
    hass,
    profile_entry_factory,
    system_entry_factory,
    monkeypatch,
    server_type,
    expected_url,
    expected_headers,
):
    system_entry_factory()
    config = {
        CONF_SERVER_TYPE: server_type,
        CONF_LMSTUDIO_URL: "https://provider.example.invalid/v1",
        CONF_HERMES_URL: "https://provider.example.invalid/v1",
        CONF_API_KEY: "test-provider-key",
        CONF_MODEL_NAME: "custom-chat-model",
    }
    entry = profile_entry_factory(data=config)
    server = MCPServer(hass, 8099, entry)
    calls = _install_image_response(monkeypatch, _image_response())

    result = await _dispatch_image(server, style="vivid")

    assert result["isError"] is False
    assert len(calls) == 1
    assert calls[0]["url"] == expected_url
    assert not calls[0]["url"].endswith("/responses")
    assert calls[0]["headers"] == expected_headers
    assert calls[0]["payload"]["model"] == "custom-chat-model"
    assert calls[0]["payload"]["response_format"] == "b64_json"
    assert calls[0]["payload"]["style"] == "vivid"
    assert result["structuredContent"]["model"] == "custom-chat-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation_transport",
    [
        OPENAI_API_TRANSPORT_AUTO,
        OPENAI_API_TRANSPORT_CHAT_COMPLETIONS,
        OPENAI_API_TRANSPORT_RESPONSES,
    ],
)
@pytest.mark.parametrize("profile_model", ["custom-chat-model", "gpt-image-custom"])
@pytest.mark.asyncio
async def test_custom_openai_defaults_to_images_api_independent_of_chat_transport(
    hass,
    profile_entry_factory,
    system_entry_factory,
    monkeypatch,
    conversation_transport,
    profile_model,
):
    system_entry_factory()
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: "https://proxy.example.invalid/v1",
            CONF_API_KEY: "sk-test-key",
            CONF_MODEL_NAME: profile_model,
        },
        options={CONF_OPENAI_API_TRANSPORT: conversation_transport},
    )
    server = MCPServer(hass, 8099, entry)
    calls = _install_image_response(monkeypatch, _image_response())

    result = await _dispatch_image(server, style="vivid")

    assert result["isError"] is False
    assert len(calls) == 1
    assert calls[0]["url"] == "https://proxy.example.invalid/v1/images/generations"
    assert calls[0]["headers"] == {"Authorization": "Bearer sk-test-key"}
    assert calls[0]["payload"]["model"] == profile_model
    assert calls[0]["payload"]["response_format"] == "b64_json"
    assert calls[0]["payload"]["style"] == "vivid"


@pytest.mark.asyncio
async def test_custom_openai_can_explicitly_select_responses_image_tool(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
):
    system_entry_factory()
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: "https://proxy.example.invalid/v1",
            CONF_API_KEY: "sk-test-key",
            CONF_MODEL_NAME: "custom-chat-model",
        },
        options={
            CONF_OPENAI_API_TRANSPORT: OPENAI_API_TRANSPORT_CHAT_COMPLETIONS,
            CONF_OPENAI_IMAGE_MODEL: "custom-image-v2",
            CONF_OPENAI_IMAGE_API: OPENAI_API_TRANSPORT_RESPONSES,
        },
    )
    server = MCPServer(hass, 8099, entry)
    calls = _install_image_response(
        monkeypatch,
        {
            "status": "completed",
            "output": [
                {
                    "type": "image_generation_call",
                    "result": base64.b64encode(b"fake-png").decode("ascii"),
                }
            ],
        },
    )

    result = await _dispatch_image(server, size="1024x1024")

    assert result["isError"] is False
    assert len(calls) == 1
    assert calls[0]["url"] == "https://proxy.example.invalid/v1/responses"
    assert calls[0]["headers"] == {"Authorization": "Bearer sk-test-key"}
    assert calls[0]["payload"] == {
        "model": "custom-chat-model",
        "input": "Draw a house",
        "tools": [
            {"type": "image_generation", "model": "custom-image-v2", "size": "1024x1024"}
        ],
        "tool_choice": {"type": "image_generation"},
        "store": False,
    }


def test_loaded_agent_metadata_reports_image_model_for_compatible_provider_only(
    hass, profile_entry_factory
):
    compatible = MCPAssistConversationEntity(
        hass,
        profile_entry_factory(
            data={
                CONF_SERVER_TYPE: SERVER_TYPE_LMSTUDIO,
                CONF_MODEL_NAME: "custom-chat-model",
            }
        ),
    )
    unsupported = MCPAssistConversationEntity(
        hass,
        profile_entry_factory(
            data={CONF_SERVER_TYPE: SERVER_TYPE_ANTHROPIC, CONF_MODEL_NAME: "claude-example"}
        ),
    )

    assert compatible.extra_state_attributes == {
        "image_model": "custom-chat-model",
        "image_model_available": True,
    }
    assert unsupported.extra_state_attributes == {"image_model_available": False}


@pytest.mark.parametrize(
    ("image_model", "expected_response_format"),
    [("dall-e-2", "b64_json"), ("dall-e-3", "b64_json"), ("gpt-image-1", None)],
)
@pytest.mark.asyncio
async def test_official_openai_legacy_image_models_keep_images_api_route(
    hass,
    profile_entry_factory,
    system_entry_factory,
    monkeypatch,
    image_model,
    expected_response_format,
):
    system_entry_factory()
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: OPENAI_BASE_URL,
            CONF_API_KEY: "sk-test-key",
            CONF_MODEL_NAME: image_model,
        },
        options={CONF_OPENAI_API_TRANSPORT: OPENAI_API_TRANSPORT_AUTO},
    )
    server = MCPServer(hass, 8099, entry)
    calls = _install_image_response(monkeypatch, _image_response())

    result = await _dispatch_image(server)

    assert result["isError"] is False
    assert len(calls) == 1
    assert calls[0]["url"] == "https://api.openai.com/v1/images/generations"
    assert calls[0]["headers"] == {"Authorization": "Bearer sk-test-key"}
    assert calls[0]["payload"]["model"] == image_model
    if expected_response_format is None:
        assert "response_format" not in calls[0]["payload"]
    else:
        assert calls[0]["payload"]["response_format"] == expected_response_format


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        (
            {"status": "completed", "output": [{"type": "message", "content": []}]},
            "no usable image data",
        ),
        ({"status": "failed", "output": []}, "did not complete"),
        ({"status": "incomplete", "output": []}, "did not complete"),
    ],
)
@pytest.mark.asyncio
async def test_responses_image_errors_do_not_fall_back_to_images_api(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, body, expected_error
):
    system_entry_factory()
    entry = profile_entry_factory(
        data={
            CONF_SERVER_TYPE: SERVER_TYPE_OPENAI,
            CONF_LMSTUDIO_URL: "https://proxy.example.invalid/v1",
            CONF_API_KEY: "sk-test-key",
            CONF_MODEL_NAME: "custom-chat-model",
        },
        options={
            CONF_OPENAI_IMAGE_MODEL: "custom-image-v2",
            CONF_OPENAI_IMAGE_API: OPENAI_API_TRANSPORT_RESPONSES,
        },
    )
    server = MCPServer(hass, 8099, entry)
    calls = _install_image_response(monkeypatch, body)

    result = await _dispatch_image(server)

    assert result["isError"] is True
    assert expected_error in result["content"][0]["text"]
    assert len(calls) == 1
    assert calls[0]["url"] == "https://proxy.example.invalid/v1/responses"
