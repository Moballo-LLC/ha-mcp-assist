"""Image policy integrity and one-snapshot provider dispatch."""

import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist import agent as agent_module
from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import (
    CONF_LMSTUDIO_URL, CONF_MCP_BEARER_TOKEN, CONF_MCP_PORT, DOMAIN,
    CONF_OPENAI_IMAGE_API, CONF_OPENAI_IMAGE_MODEL, CONF_OPENAI_IMAGE_MODEL_PROFILE,
    CONF_SERVER_TYPE, OPENAI_API_TRANSPORT_RESPONSES, SERVER_TYPE_OLLAMA,
)
from custom_components.mcp_assist.mcp_server import MCPServer
from custom_components.mcp_assist.model_profiles import (
    MCP_PROFILE_REQUEST_HEADER,
    ModelProfileResolutionError, REQUEST_RESOLVED_PROFILES,
    async_resolve_model_profile, async_resolve_model_profiles,
    validate_model_profile, validate_model_profiles,
)
from custom_components.mcp_assist.provider_runtime import resolve_provider_runtime_config
from tests.test_image_transport_compatibility import _install_image_response
from tests.test_model_profiles import Response, entry_data, mock_session, policy, prepare, user_input


def image_policy(model="example-image", text_model="example-text", effort="high"):
    value = policy(text_model, effort)
    value.update(imageModels={"image": model}, imageProfiles={
        "illustration": {"label": "Illustration", "modelAlias": "image"}},
        imageBindings={"visual": "illustration"})
    refresh_revision(value)
    value["resolvedImageProfiles"] = {"illustration": {"model": model}}
    return value


def refresh_revision(value):
    keys = ("schemaVersion", "models", "profiles", "bindings",
            "imageModels", "imageProfiles", "imageBindings")
    value["revision"] = hashlib.sha256(json.dumps(
        {key: value[key] for key in keys if key in value}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def image_entry_data(**options):
    return {**entry_data(), CONF_OPENAI_IMAGE_MODEL_PROFILE: "visual", **options}


def test_full_policy_digest_resolves_text_and_image():
    value = image_policy()
    text, image = validate_model_profiles(value, "assistant", "visual")
    assert text == validate_model_profile(value, "assistant")
    assert image.model == "example-image"
    assert image.revision == text.revision
    assert validate_model_profiles(value, image_reference="illustration")[1] == image
    value["imageModels"]["image"] = "changed-image"
    value["resolvedImageProfiles"]["illustration"]["model"] = "changed-image"
    with pytest.raises(ModelProfileResolutionError):
        validate_model_profile(value, "assistant")


@pytest.mark.parametrize("mutate", [
    lambda v: v.pop("imageModels"), lambda v: v.pop("imageProfiles"),
    lambda v: v.pop("imageBindings"), lambda v: v.pop("resolvedImageProfiles"),
    lambda v: v["imageProfiles"]["illustration"].update(reasoningEffort="high"),
    lambda v: v["imageProfiles"]["illustration"].update(modelAlias="missing"),
    lambda v: v["imageBindings"].update(visual="missing"),
    lambda v: v["resolvedImageProfiles"]["illustration"].update(model="wrong"),
    lambda v: v["imageModels"].update(image="bad model"),
])
def test_invalid_image_maps_rejected_for_text_and_image(mutate):
    value = image_policy()
    mutate(value)
    refresh_revision(value)
    with pytest.raises(ModelProfileResolutionError):
        validate_model_profiles(value, "assistant", "visual")
    with pytest.raises(ModelProfileResolutionError):
        validate_model_profile(value, "assistant")


async def test_one_authenticated_fetch_resolves_both(profile_entry_factory, monkeypatch):
    calls = mock_session(monkeypatch, Response(json.dumps(image_policy()).encode()))
    runtime = resolve_provider_runtime_config(profile_entry_factory(data=image_entry_data()))
    text, image = await async_resolve_model_profiles(runtime, "assistant", "visual")
    assert (text.model, image.model) == ("example-text", "example-image")
    assert calls == [("https://example.invalid/v1/model-profiles", {
        "headers": {"Authorization": "Bearer example-key"},
        "params": {"include_images": "true"}, "allow_redirects": False})]


@pytest.mark.parametrize("image_reference", ["visual", "missing"])
async def test_text_only_policy_cannot_resolve_image(
    profile_entry_factory, monkeypatch, image_reference
):
    calls = mock_session(monkeypatch, Response(json.dumps(policy()).encode()))
    runtime = resolve_provider_runtime_config(profile_entry_factory(data=image_entry_data()))
    with pytest.raises(ModelProfileResolutionError):
        await async_resolve_model_profiles(runtime, "assistant", image_reference)
    assert len(calls) == 1


async def test_unsupported_image_provider_stops_before_generation(
    hass, profile_entry_factory, monkeypatch
):
    entry = profile_entry_factory(data=image_entry_data(**{CONF_SERVER_TYPE: SERVER_TYPE_OLLAMA}))
    agent = MCPAssistConversationEntity(hass, entry)
    calls = mock_session(monkeypatch, Response(b"{}"))
    generation = AsyncMock(return_value="Answer")
    monkeypatch.setattr(agent, "_call_llm", generation)
    result = await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert result.response.error_code is not None
    assert not calls and not generation.called
    assert agent.image_model is None and agent.resolved_image_model_profile is None


@pytest.mark.parametrize("responses", [False, True])
async def test_image_payload_uses_resolved_model_in_logical_request(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, responses
):
    system_entry_factory()
    entry = profile_entry_factory(data=image_entry_data(**{
        CONF_OPENAI_IMAGE_MODEL: "explicit-fallback",
        CONF_OPENAI_IMAGE_API: OPENAI_API_TRANSPORT_RESPONSES if responses else "images",
    }))
    agent = MCPAssistConversationEntity(hass, entry)
    server = MCPServer(hass, 8099, entry)
    prepare(agent, monkeypatch)
    current = image_policy()
    lookup = AsyncMock(side_effect=lambda *args: validate_model_profiles(
        deepcopy(current), "assistant", "visual"))
    monkeypatch.setattr(agent_module, "async_resolve_model_profiles", lookup)
    response = ({"status": "completed", "output": [{"type": "image_generation_call",
                 "result": base64.b64encode(b"fake-png").decode()}]} if responses
                else {"data": [{"b64_json": base64.b64encode(b"fake-png").decode()}]})
    calls = _install_image_response(monkeypatch, response)
    monkeypatch.setattr(server, "_normalize_image_payload", lambda data, mime, _: (data, mime))

    async def generate(messages):
        for index in range(2):
            assert agent.image_model == "example-image"
            if index == 1:
                current.update(image_policy("next-image", "next-text", "low"))
                hass.config_entries.async_update_entry(entry, options={
                    CONF_OPENAI_IMAGE_MODEL_PROFILE: "changed-reference",
                    CONF_OPENAI_IMAGE_MODEL: "changed-fallback",
                    CONF_OPENAI_IMAGE_API: OPENAI_API_TRANSPORT_RESPONSES if responses else "images",
                })
            await server._generate_image_with_provider(prompt="Illustration", size=None,
                quality=None, style=None, background=None, context={"profile_entry_id": entry.entry_id})
        # Provider rebuilt after the tool still consumes the frozen snapshot.
        assert agent._get_llm_provider().image_model == "example-image"
        return "Answer"

    monkeypatch.setattr(agent, "_call_llm", generate)
    assert agent.image_model is None
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert lookup.await_count == 1
    assert len(calls) == 2
    for call in calls:
        payload = call["payload"]
        if responses:
            assert payload["model"] == "example-text"
            assert payload["tools"][0]["model"] == "example-image"
            assert payload["reasoning"] == {"effort": "high"}
        else:
            assert payload["model"] == "example-image"
        assert "visual" not in json.dumps(payload)
    assert agent.resolved_image_model_profile is None  # Preference changed during the request.
    assert agent.image_model is None
    assert REQUEST_RESOLVED_PROFILES.get() is None


async def test_image_metadata_clears_after_failed_new_lookup(
    hass, profile_entry_factory, monkeypatch
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=image_entry_data()))
    prepare(agent, monkeypatch)
    resolved = validate_model_profiles(image_policy(), "assistant", "visual")
    lookup = AsyncMock(side_effect=[resolved, ModelProfileResolutionError()])
    monkeypatch.setattr(agent_module, "async_resolve_model_profiles", lookup)
    monkeypatch.setattr(agent, "_call_llm", AsyncMock(return_value="Answer"))
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    metadata = agent.resolved_image_model_profile
    assert metadata == {"reference": "visual", "model": "example-image",
                        "profile_id": "illustration", "revision": resolved[1].revision}
    metadata["model"] = "tampered"
    assert agent.image_model == "example-image"
    assert agent.extra_state_attributes == {
        "image_model_available": True, "image_model": "example-image",
        "image_model_profile": "visual", "resolved_image_model_profile": "illustration",
        "image_model_policy_revision": resolved[1].revision,
    }
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert agent.image_model is None and agent.resolved_image_model_profile is None
    assert agent.extra_state_attributes == {
        "image_model_available": False, "image_model_profile": "visual",
    }


async def test_cancelled_image_request_clears_scopes_and_metadata(
    hass, profile_entry_factory, monkeypatch
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=image_entry_data()))
    prepare(agent, monkeypatch)
    monkeypatch.setattr(agent_module, "async_resolve_model_profiles", AsyncMock(
        return_value=validate_model_profiles(image_policy(), "assistant", "visual")))
    monkeypatch.setattr(agent, "_call_llm", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert agent.resolved_image_model_profile is None
    assert REQUEST_RESOLVED_PROFILES.get() is None
    assert agent_module._REQUEST_IMAGE_MODEL_PROFILE.get() is None


@pytest.mark.parametrize("responses", [False, True])
async def test_standalone_image_tool_resolves_one_policy(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, responses
):
    from custom_components.mcp_assist import model_profiles as profiles_module

    system_entry_factory()
    entry = profile_entry_factory(data=image_entry_data(**{
        CONF_OPENAI_IMAGE_API: OPENAI_API_TRANSPORT_RESPONSES if responses else "images",
    }))
    server = MCPServer(hass, 8099, entry)
    lookup = AsyncMock(return_value=validate_model_profiles(image_policy(), "assistant", "visual"))
    monkeypatch.setattr(profiles_module, "async_resolve_model_profiles", lookup)
    response = ({"status": "completed", "output": [{"type": "image_generation_call",
                 "result": base64.b64encode(b"fake-png").decode()}]} if responses
                else {"data": [{"b64_json": base64.b64encode(b"fake-png").decode()}]})
    calls = _install_image_response(monkeypatch, response)
    monkeypatch.setattr(server, "_normalize_image_payload", lambda data, mime, _: (data, mime))
    await server._generate_image_with_provider(prompt="Illustration", size=None,
        quality=None, style=None, background=None, context=None)
    assert lookup.await_count == 1
    payload = calls[0]["payload"]
    if responses:
        assert payload["model"] == "example-text"
        assert payload["tools"][0]["model"] == "example-image"
        assert payload["reasoning"] == {"effort": "high"}
    else:
        assert payload["model"] == "example-image"
        assert "reasoning" not in payload
    assert REQUEST_RESOLVED_PROFILES.get() is None


async def test_real_tool_loop_freezes_both_and_next_request_adopts_policy(
    hass, profile_entry_factory, monkeypatch
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=image_entry_data()))
    prepare(agent, monkeypatch)
    current = image_policy()
    lookup = AsyncMock(side_effect=lambda *args: validate_model_profiles(
        deepcopy(current), "assistant", "visual"))
    monkeypatch.setattr(agent_module, "async_resolve_model_profiles", lookup)
    observed = []
    monkeypatch.setattr(agent, "_get_mcp_tools", AsyncMock(return_value=[{
        "type": "function", "function": {"name": "generate_image", "description": "Illustrate",
        "parameters": {"type": "object", "properties": {}}},
    }]))

    async def execute(calls):
        current.update(image_policy("next-image", "next-text"))
        return [{"role": "tool", "tool_call_id": "call-1", "content": "Generated."}]

    async def response(provider, payload, *, iteration):
        observed.append((payload["model"], provider.image_model))
        message = ({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "generate_image", "arguments": "{}"},
        }]} if iteration == 0 else {"role": "assistant", "content": "Answer."})
        return agent_module.ProviderHttpResponse(status=200, data={"choices": [{"message": message}]})

    monkeypatch.setattr(agent, "_execute_tool_calls", execute)
    monkeypatch.setattr(agent, "_request_provider_http_response", response)
    monkeypatch.setattr(agent, "_call_llm", agent._call_llm_http)
    for _ in range(2):
        await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert observed == [("example-text", "example-image"), ("example-text", "example-image"),
                        ("next-text", "next-image"), ("next-text", "next-image")]
    assert lookup.await_count == 2
    assert agent.image_model == "next-image"
    assert agent.resolved_model_profile["revision"] == agent.resolved_image_model_profile["revision"]


async def test_image_only_reference_keeps_saved_conversation_model(
    hass, profile_entry_factory, monkeypatch
):
    from custom_components.mcp_assist.const import CONF_MODEL_PROFILE

    agent = MCPAssistConversationEntity(hass, profile_entry_factory(
        data=image_entry_data(**{CONF_MODEL_PROFILE: ""})))
    prepare(agent, monkeypatch)
    calls = mock_session(monkeypatch, Response(json.dumps(image_policy()).encode()))

    async def generate(messages):
        provider = agent._get_llm_provider()
        assert provider.model_name == "saved-model"
        assert provider.image_model == "example-image"
        return "Answer"

    monkeypatch.setattr(agent, "_call_llm", generate)
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert len(calls) == 1
    assert agent.resolved_model_profile is None


async def test_disabling_reference_restores_loaded_concrete_image_fallback(
    hass, profile_entry_factory, monkeypatch
):
    entry = profile_entry_factory(data=image_entry_data(**{
        CONF_OPENAI_IMAGE_MODEL: "explicit-image",
    }))
    agent = MCPAssistConversationEntity(hass, entry)
    assert agent.image_model is None
    hass.config_entries.async_update_entry(entry, options={CONF_OPENAI_IMAGE_MODEL_PROFILE: ""})
    assert agent.image_model == "explicit-image"
    assert agent.resolved_image_model_profile is None
    assert agent.extra_state_attributes == {
        "image_model": "explicit-image", "image_model_available": True,
    }


async def test_concurrent_image_snapshots_remain_entry_and_request_scoped(
    hass, profile_entry_factory, monkeypatch
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=image_entry_data()))
    other = MCPAssistConversationEntity(hass, profile_entry_factory(data=image_entry_data()))
    prepare(agent, monkeypatch)
    count = 0
    arrivals = 0
    ready = asyncio.Event()
    observed = {}

    async def lookup(*args):
        nonlocal count
        count += 1
        return validate_model_profiles(image_policy(f"image-{count}", f"text-{count}"),
                                       "assistant", "visual")

    async def generate(messages):
        nonlocal arrivals
        model = agent.image_model
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await ready.wait()
        observed[model] = agent._get_llm_provider().image_model
        assert other.image_model is None
        assert other.resolved_image_model_profile is None
        with pytest.raises(ValueError, match="not been resolved"):
            _ = other._get_llm_provider().image_model
        return "Answer"

    monkeypatch.setattr(agent_module, "async_resolve_model_profiles", lookup)
    monkeypatch.setattr(agent, "_call_llm", generate)
    await asyncio.gather(*(agent._async_handle_message(
        user_input(str(index)), SimpleNamespace(conversation_id=str(index))) for index in range(2)))
    assert observed == {"image-1": "image-1", "image-2": "image-2"}
    assert REQUEST_RESOLVED_PROFILES.get() is None


async def test_scalar_image_attributes_drop_stale_resolution_on_reference_change(
    hass, profile_entry_factory, monkeypatch
):
    entry = profile_entry_factory(data=image_entry_data())
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    monkeypatch.setattr(agent_module, "async_resolve_model_profiles", AsyncMock(
        return_value=validate_model_profiles(image_policy(), "assistant", "visual")))
    monkeypatch.setattr(agent, "_call_llm", AsyncMock(return_value="Answer"))
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert agent.extra_state_attributes["resolved_image_model_profile"] == "illustration"
    hass.config_entries.async_update_entry(entry, options={
        CONF_OPENAI_IMAGE_MODEL_PROFILE: "next-visual",
    })
    assert agent.resolved_image_model_profile is None
    assert agent.extra_state_attributes == {
        "image_model_available": False, "image_model_profile": "next-visual",
    }


async def test_standalone_actual_provider_reaches_real_policy_resolution(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
):
    from custom_components.mcp_assist.llm_providers.openai import OpenAIProvider

    system_entry_factory()
    entry = profile_entry_factory(data=image_entry_data(**{
        CONF_OPENAI_IMAGE_MODEL: "explicit-fallback",
    }))
    server = MCPServer(hass, 8099, entry)
    unresolved = server._get_model_provider(None)
    assert isinstance(unresolved, OpenAIProvider)
    with pytest.raises(ValueError, match="not been resolved"):
        _ = unresolved.image_model
    calls = mock_session(monkeypatch, Response(json.dumps(image_policy()).encode()))
    # No provider factory or policy resolver is mocked: only the HTTP boundary is replaced.
    provider = await server._get_image_model_provider(None)
    assert isinstance(provider, OpenAIProvider)
    assert provider.image_model == "example-image"
    assert provider.model_name == "example-text"
    assert calls == [("https://example.invalid/v1/model-profiles", {
        "headers": {"Authorization": "Bearer example-key"},
        "params": {"include_images": "true"}, "allow_redirects": False})]


@pytest.mark.parametrize("mode,expanded", [
    ("text-wrapper", False), ("text-wrapper", True),
    ("text-combined", False), ("text-combined", True), ("both", True),
])
async def test_actual_http_policy_lookup_negotiates_image_capability(
    aiohttp_server, socket_enabled, profile_entry_factory, mode, expanded
):
    from aiohttp import web

    received = []

    async def get_policy(request):
        received.append((dict(request.query), request.headers.get("Authorization"), request.path))
        # Legacy endpoints may ignore the query and continue serving a text-only policy.
        return web.json_response(image_policy() if expanded else policy())

    app = web.Application()
    app.router.add_get("/v1/model-profiles", get_policy)
    server = await aiohttp_server(app)
    runtime = replace(
        resolve_provider_runtime_config(profile_entry_factory(data=image_entry_data())),
        base_url=str(server.make_url("/v1")),
    )
    if mode == "text-wrapper":
        text = await async_resolve_model_profile(runtime, "assistant")
        image = None
    else:
        text, image = await async_resolve_model_profiles(
            runtime, "assistant", "visual" if mode == "both" else "",
        )
    assert text.model == ("example-text" if expanded else "example-model")
    assert (image.model if image else None) == ("example-image" if mode == "both" else None)
    assert received == [({"include_images": "true"}, "Bearer example-key", "/v1/model-profiles")]


@pytest.mark.parametrize("responses", [False, True])
async def test_conversation_snapshot_crosses_actual_mcp_http_request(
    hass, aiohttp_server, socket_enabled, profile_entry_factory, system_entry_factory,
    monkeypatch, responses,
):
    from aiohttp import web
    import pytest_socket

    pytest_socket.socket_allow_hosts(["127.0.0.1", "::1"])
    system_entry = system_entry_factory(data={CONF_MCP_BEARER_TOKEN: "example-mcp-key"})
    current = image_policy()
    policy_reads = []
    generated = []
    handler_contexts = []
    entry = profile_entry_factory(data=image_entry_data(**{
        CONF_OPENAI_IMAGE_API: OPENAI_API_TRANSPORT_RESPONSES if responses else "images",
    }))
    mcp = MCPServer(hass, 8099, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = mcp

    async def get_policy(request):
        policy_reads.append(deepcopy(current))
        return web.json_response(current)

    async def generate_image(request):
        generated.append(await request.json())
        encoded = base64.b64encode(b"example-png").decode()
        return web.json_response(
            {"status": "completed", "output": [
                {"type": "image_generation_call", "result": encoded}
            ]} if responses else {"data": [{"b64_json": encoded}]}
        )

    async def mcp_request(request):
        # The server was started outside the caller's conversation context.
        handler_contexts.append(REQUEST_RESOLVED_PROFILES.get())
        assert request.headers["Authorization"] == "Bearer example-mcp-key"
        assert request.headers[MCP_PROFILE_REQUEST_HEADER]
        return await mcp.handle_mcp_request(request)

    app = web.Application()
    app.router.add_get("/v1/model-profiles", get_policy)
    app.router.add_post("/v1/responses" if responses else "/v1/images/generations", generate_image)
    app.router.add_post("/", mcp_request)
    http = await aiohttp_server(app)
    hass.config_entries.async_update_entry(system_entry, options={CONF_MCP_PORT: http.port})
    hass.config_entries.async_update_entry(entry, options={
        CONF_LMSTUDIO_URL: str(http.make_url("/v1")), CONF_MCP_PORT: http.port,
    })
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    monkeypatch.setattr(mcp, "_normalize_image_payload", lambda data, mime, _: (data, mime))

    async def generate(messages):
        current.update(image_policy("next-image", "next-text", "low"))
        result = await agent._call_mcp_tool("generate_image", {"prompt": "Illustration"})
        assert not result.get("isError") and "error" not in result
        assert agent._get_llm_provider().model_name == "example-text"
        assert agent._get_llm_provider().image_model == "example-image"
        return "Answer"

    monkeypatch.setattr(agent, "_call_llm", generate)
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert handler_contexts == [None]
    assert len(policy_reads) == 1 and len(generated) == 1
    if responses:
        assert generated[0]["model"] == "example-text"
        assert generated[0]["reasoning"] == {"effort": "high"}
        assert generated[0]["tools"][0]["model"] == "example-image"
    else:
        assert generated[0]["model"] == "example-image"
    assert mcp._profile_requests == {}
    assert REQUEST_RESOLVED_PROFILES.get() is None


@pytest.mark.parametrize("invalid", ["entry", "id", "tool", "remote", "unknown", "replay"])
async def test_http_snapshot_rejects_wrong_binding_and_replay(
    hass, profile_entry_factory, invalid,
):
    entry = profile_entry_factory(data=image_entry_data())
    other = profile_entry_factory(data=image_entry_data())
    server = MCPServer(hass, 8099, entry)
    snapshot = (entry, *validate_model_profiles(image_policy(), "assistant", "visual"))
    token = server.register_profile_request(snapshot, "request", "generate_image")
    payload = {"jsonrpc": "2.0", "id": "request", "method": "tools/call", "params": {
        "name": "generate_image", "context": {"profile_entry_id": entry.entry_id},
    }}
    request = SimpleNamespace(headers={MCP_PROFILE_REQUEST_HEADER: token}, remote="127.0.0.1")
    if invalid == "entry":
        payload["params"]["context"]["profile_entry_id"] = other.entry_id
    elif invalid == "id":
        payload["id"] = "different-request"
    elif invalid == "tool":
        payload["params"]["name"] = "analyze_image"
    elif invalid == "remote":
        request.remote = "192.0.2.1"
    elif invalid == "unknown":
        request.headers[MCP_PROFILE_REQUEST_HEADER] = "unknown"
    else:
        assert server._take_profile_request(request, payload) == snapshot
    with pytest.raises(ModelProfileResolutionError):
        server._take_profile_request(request, payload)
    server.discard_profile_request(token)
    assert server._profile_requests == {}


async def test_outstanding_http_snapshots_and_registry_limit(hass, profile_entry_factory):
    entry = profile_entry_factory(data=image_entry_data())
    server = MCPServer(hass, 8099, entry)
    snapshots = [(entry, *validate_model_profiles(image_policy(f"image-{i}", f"text-{i}"),
                 "assistant", "visual")) for i in range(128)]
    tokens = [server.register_profile_request(snapshot, str(i), "generate_image")
              for i, snapshot in enumerate(snapshots)]
    with pytest.raises(ModelProfileResolutionError):
        server.register_profile_request(snapshots[0], "overflow", "generate_image")
    for i in reversed(range(128)):
        request = SimpleNamespace(headers={MCP_PROFILE_REQUEST_HEADER: tokens[i]}, remote="::1")
        payload = {"id": str(i), "method": "tools/call", "params": {
            "name": "generate_image", "context": {"profile_entry_id": entry.entry_id},
        }}
        assert server._take_profile_request(request, payload) == snapshots[i]
    assert server._profile_requests == {}
    server.register_profile_request(snapshots[0], "shutdown", "generate_image")
    await server.stop()
    assert server._profile_requests == {}


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_http_dispatch_releases_snapshot(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, cancel,
):
    system_entry_factory()
    entry = profile_entry_factory(data=image_entry_data())
    server = MCPServer(hass, 8099, entry)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = server
    agent = MCPAssistConversationEntity(hass, entry)
    snapshot = (entry, *validate_model_profiles(image_policy(), "assistant", "visual"))
    token = REQUEST_RESOLVED_PROFILES.set(snapshot)

    def fail_session(**kwargs):
        assert len(server._profile_requests) == 1
        raise asyncio.CancelledError() if cancel else OSError("Example connection failure")

    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", fail_session)
    try:
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await agent._call_mcp_tool("generate_image", {"prompt": "Illustration"})
        else:
            result = await agent._call_mcp_tool("generate_image", {"prompt": "Illustration"})
            assert "error" in result
    finally:
        REQUEST_RESOLVED_PROFILES.reset(token)
    assert server._profile_requests == {}
