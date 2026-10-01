"""Generic compatible-endpoint profile contracts and request isolation."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

from custom_components.mcp_assist import agent as agent_module
from custom_components.mcp_assist import model_profiles as profiles_module
from custom_components.mcp_assist.agent import MCPAssistConversationEntity
from custom_components.mcp_assist.const import (
    CONF_API_KEY, CONF_CHAT_LOG_MODE, CONF_LMSTUDIO_URL, CONF_MODEL_NAME,
    CONF_MODEL_PROFILE, CONF_OPENAI_API_TRANSPORT, CONF_SERVER_TYPE,
    SERVER_TYPE_OPENAI, SERVER_TYPE_OLLAMA, DOMAIN,
)
from custom_components.mcp_assist.llm_providers.openai import OpenAIProvider
from custom_components.mcp_assist.model_profiles import (
    ModelProfileResolutionError, async_resolve_model_profile, validate_model_profile,
)
from custom_components.mcp_assist.provider_runtime import resolve_provider_runtime_config


def policy(model="example-model", effort="high", *, bindings=None):
    value = {"schemaVersion": 1, "models": {"primary": model}, "profiles": {
        "focused": {"label": "Focused", "modelAlias": "primary", "reasoningEffort": effort}
    }, "bindings": {"assistant": "focused"} if bindings is None else bindings}
    value["revision"] = hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    value["resolvedProfiles"] = {"focused": {"model": model, "effort": effort}}
    return value


def entry_data(reference="assistant"):
    return {CONF_SERVER_TYPE: SERVER_TYPE_OPENAI, CONF_MODEL_NAME: "saved-model",
            CONF_MODEL_PROFILE: reference, CONF_API_KEY: "example-key",
            CONF_LMSTUDIO_URL: "https://example.invalid/v1", CONF_CHAT_LOG_MODE: False}


def user_input(text="Test"):
    return ConversationInput(text=text, context=Context(), conversation_id="test",
                             device_id=None, satellite_id=None, language="en", agent_id="test")


def prepare(agent, monkeypatch):
    monkeypatch.setattr(agent, "_build_system_prompt_with_context", AsyncMock(return_value="system"))
    monkeypatch.setattr(agent, "_prepare_adaptive_tools_for_request", AsyncMock())
    monkeypatch.setattr(agent, "_build_response_result", AsyncMock(return_value="done"))


def test_binding_and_profile_resolution():
    value = policy()
    assert validate_model_profile(value, "assistant") == validate_model_profile(value, "focused")
    value["nativeExport"] = {"ok": False, "error": "example"}
    assert validate_model_profile(value, "assistant").model == "example-model"


def test_direct_profile_resolution_without_bindings():
    value = policy(bindings={})
    assert validate_model_profile(value, "focused").model == "example-model"
    with pytest.raises(ModelProfileResolutionError):
        validate_model_profile(value, "assistant")


@pytest.mark.parametrize("mapping", ["models", "profiles", "resolvedProfiles"])
def test_required_policy_maps_remain_nonempty(mapping):
    value = policy(bindings={})
    value[mapping] = {}
    with pytest.raises(ModelProfileResolutionError):
        validate_model_profile(value, "focused")


@pytest.mark.parametrize("mutate", [
    lambda v: v.update(schemaVersion=True), lambda v: v.update(schemaVersion=2),
    lambda v: v.update(revision="0" * 64), lambda v: v.update(models=[]),
    lambda v: v["bindings"].update(assistant="missing"),
    lambda v: v["profiles"]["focused"].update(reasoningEffort="invalid"),
    lambda v: v["resolvedProfiles"]["focused"].update(model="mismatch"),
    lambda v: v["models"].update(primary="bad model"),
])
def test_invalid_policy_rejected(mutate):
    value = policy()
    mutate(value)
    with pytest.raises(ModelProfileResolutionError):
        validate_model_profile(value, "assistant")


class Response:
    def __init__(self, body, status=200, content_length=None):
        self.body, self.status, self.content_length = body, status, content_length
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def iter_chunked(self, size):
        for offset in range(0, len(self.body), size):
            yield self.body[offset:offset+size]


def mock_session(monkeypatch, response):
    calls = []

    class Session:
        def __init__(self, **kwargs):
            assert 0 < kwargs["timeout"].total <= 15

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return response

    monkeypatch.setattr(profiles_module.aiohttp, "ClientSession", Session)
    return calls


async def test_authenticated_lookup_preserves_base_path(profile_entry_factory, monkeypatch):
    calls = mock_session(monkeypatch, Response(json.dumps(policy()).encode()))
    runtime = resolve_provider_runtime_config(profile_entry_factory(data=entry_data()))
    assert (await async_resolve_model_profile(runtime, "assistant")).model == "example-model"
    assert calls == [("https://example.invalid/v1/model-profiles", {
        "headers": {"Authorization": "Bearer example-key"},
        "allow_redirects": False})]


@pytest.mark.parametrize("response", [
    Response(b"{}", status=302), Response(b"{}", status=401),
    Response(b"{}", content_length=1024*1024+1), Response(b"x"*(1024*1024+1)),
    Response(b"not json"), Response(b"{}"),
])
async def test_lookup_rejects_redirect_oversize_and_bad_schema(
    profile_entry_factory, monkeypatch, response
):
    mock_session(monkeypatch, response)
    runtime = resolve_provider_runtime_config(profile_entry_factory(data=entry_data()))
    with pytest.raises(ModelProfileResolutionError):
        await async_resolve_model_profile(runtime, "assistant")


@pytest.mark.parametrize("base_url,key", [
    ("https://api.openai.com/v1", "example-key"),
    ("https://eu.api.openai.com/v1", "example-key"),
    ("https://example.invalid/v1", ""),
    ("https://example.invalid/v1?token=example", "example-key"),
])
async def test_lookup_rejects_unsupported_endpoint_before_network(
    profile_entry_factory, monkeypatch, base_url, key
):
    calls = mock_session(monkeypatch, Response(b"{}"))
    runtime = resolve_provider_runtime_config(profile_entry_factory(data=entry_data()))
    with pytest.raises(ModelProfileResolutionError) as error:
        await async_resolve_model_profile(replace(runtime, base_url=base_url, api_key=key), "assistant")
    assert calls == []
    assert base_url not in str(error.value)
    assert key not in str(error.value) if key else True


@pytest.mark.parametrize("server_type,reference", [
    (SERVER_TYPE_OPENAI, ""), (SERVER_TYPE_OPENAI, "  "), (SERVER_TYPE_OLLAMA, "assistant"),
])
async def test_blank_or_other_provider_never_looks_up(
    hass, profile_entry_factory, monkeypatch, server_type, reference
):
    entry = profile_entry_factory(data={**entry_data(reference), CONF_SERVER_TYPE: server_type})
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    lookup = AsyncMock(side_effect=AssertionError("unexpected lookup"))
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    call = AsyncMock(return_value="answer")
    monkeypatch.setattr(agent, "_call_llm", call)
    assert await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test")) == "done"
    lookup.assert_not_called()
    call.assert_awaited_once()
    assert agent.resolved_model_profile is None


async def test_lookup_failure_has_no_saved_model_fallback(hass, profile_entry_factory, monkeypatch):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data()))
    prepare(agent, monkeypatch)
    lookup = AsyncMock(side_effect=ModelProfileResolutionError())
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    call = AsyncMock()
    monkeypatch.setattr(agent, "_call_llm", call)
    result = await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert result.response.error_code is not None
    call.assert_not_called()
    agent._build_system_prompt_with_context.assert_not_called()
    assert agent.resolved_model_profile is None
    assert agent_module._REQUEST_MODEL_PROFILE.get() is None


async def test_tool_followups_freeze_pair_and_next_request_adopts_revision(
    hass, profile_entry_factory, monkeypatch
):
    entry = profile_entry_factory(data=entry_data())
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    current = policy()
    lookup = AsyncMock(side_effect=lambda *args: validate_model_profile(deepcopy(current), "assistant"))
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    observed = []

    tool_calls = []
    monkeypatch.setattr(agent, "_get_mcp_tools", AsyncMock(return_value=[{
        "type": "function", "function": {"name": "discover_entities",
        "description": "Find entities", "parameters": {"type": "object", "properties": {}}},
    }]))

    async def execute(calls):
        tool_calls.extend(calls)
        current.update(policy("later-model", "low"))
        return [{"role": "tool", "tool_call_id": "call-1", "content": "Found."}]

    async def response(provider, payload, *, iteration):
        observed.append(payload)
        assert agent.model_name == payload["model"]
        message = ({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "discover_entities", "arguments": "{}"},
        }]} if iteration == 0 else {"role": "assistant", "content": "Answer."})
        return agent_module.ProviderHttpResponse(status=200, data={"choices": [{"message": message}]})

    monkeypatch.setattr(agent, "_execute_tool_calls", execute)
    monkeypatch.setattr(agent, "_request_provider_http_response", response)
    monkeypatch.setattr(agent, "_call_llm", agent._call_llm_http)
    for _ in range(2):
        await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert [item["model"] for item in observed] == [
        "example-model", "example-model", "later-model", "later-model"]
    assert [item["reasoning_effort"] for item in observed] == ["high", "high", "low", "low"]
    assert lookup.await_count == 2
    assert len(tool_calls) == 2
    assert agent.model_name == "saved-model"
    assert agent.resolved_model_profile["revision"] == current["revision"]
    metadata = agent.resolved_model_profile
    metadata["model"] = "tampered"
    assert agent.resolved_model_profile["model"] == "later-model"
    assert agent.image_model == "saved-model"


async def test_concurrent_requests_isolate_pair(hass, profile_entry_factory, monkeypatch):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data()))
    prepare(agent, monkeypatch)
    count = 0
    ready = asyncio.Event()
    arrivals = 0
    observed = {}

    async def lookup(*args):
        nonlocal count
        count += 1
        return validate_model_profile(policy(f"model-{count}"), "assistant")

    async def call(messages):
        nonlocal arrivals
        initial = agent.model_name
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await ready.wait()
        observed[initial] = agent._build_provider_settings().model_name
        return "answer"

    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    monkeypatch.setattr(agent, "_call_llm", call)
    await asyncio.gather(*(agent._async_handle_message(
        user_input(str(i)), SimpleNamespace(conversation_id=str(i))) for i in range(2)))
    assert observed == {"model-1": "model-1", "model-2": "model-2"}
    assert agent_module._REQUEST_MODEL_PROFILE.get() is None


@pytest.mark.parametrize("transport,key", [("responses", "reasoning"), ("chat_completions", "reasoning_effort")])
def test_profile_effort_projected_only_for_resolved_selection(
    hass, profile_entry_factory, transport, key
):
    entry = profile_entry_factory(data=entry_data(), options={CONF_OPENAI_API_TRANSPORT: transport})
    agent = MCPAssistConversationEntity(hass, entry)
    settings = agent._build_provider_settings()
    assert key not in OpenAIProvider(settings).build_payload([])
    resolved = validate_model_profile(policy(), "assistant")
    settings = replace(settings, model_name=resolved.model, provider_options={
        **settings.provider_options, "_resolved_model_profile": resolved})
    payload = OpenAIProvider(settings).build_payload([])
    assert payload[key] == ({"effort": "high"} if transport == "responses" else "high")


async def test_nested_other_entry_does_not_inherit_profile(hass, profile_entry_factory, monkeypatch):
    selected = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data()))
    other = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data("")))
    prepare(selected, monkeypatch)
    prepare(other, monkeypatch)
    resolved = validate_model_profile(policy(), "assistant")
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", AsyncMock(return_value=resolved))

    async def nested(messages):
        assert other.model_name == "saved-model"
        assert other._build_provider_settings().model_name == "saved-model"
        return "answer"

    async def outer(messages):
        assert selected.model_name == resolved.model
        await other._async_handle_message(user_input(), SimpleNamespace(conversation_id="other"))
        assert selected.model_name == resolved.model
        return "answer"

    monkeypatch.setattr(other, "_call_llm", nested)
    monkeypatch.setattr(selected, "_call_llm", outer)
    await selected._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert other.resolved_model_profile is None
    assert selected.model_name == "saved-model"


def test_provider_reads_opt_in_profile_options_only(profile_entry_factory):
    entry = profile_entry_factory(data=entry_data(), options={CONF_MODEL_PROFILE: " focused "})
    assert OpenAIProvider.options_from_entry(entry)[CONF_MODEL_PROFILE] == "focused"


async def test_cancelled_request_clears_metadata_and_scope(hass, profile_entry_factory, monkeypatch):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data()))
    prepare(agent, monkeypatch)
    resolved = validate_model_profile(policy(), "assistant")
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", AsyncMock(return_value=resolved))
    monkeypatch.setattr(agent, "_call_llm", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert agent.resolved_model_profile is None
    assert agent_module._REQUEST_MODEL_PROFILE.get() is None


async def test_new_blank_request_clears_last_resolution(hass, profile_entry_factory, monkeypatch):
    entry = profile_entry_factory(data=entry_data())
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    resolved = validate_model_profile(policy(), "assistant")
    lookup = AsyncMock(return_value=resolved)
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    monkeypatch.setattr(agent, "_call_llm", AsyncMock(return_value="Answer."))
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert agent.resolved_model_profile is not None
    hass.config_entries.async_update_entry(entry, options={CONF_MODEL_PROFILE: ""})
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert agent.resolved_model_profile is None
    assert lookup.await_count == 1


async def test_preference_change_during_lookup_does_not_relabel_metadata(
    hass, profile_entry_factory, monkeypatch
):
    entry = profile_entry_factory(data=entry_data())
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()
    call_started = asyncio.Event()
    release_call = asyncio.Event()
    references = []
    models = []

    async def lookup(runtime, reference):
        references.append(reference)
        if reference == "assistant":
            lookup_started.set()
            await release_lookup.wait()
            return validate_model_profile(policy("prior-model", "high"), reference)
        return validate_model_profile(policy("next-model", "low"), reference)

    async def call(messages):
        models.append(agent.model_name)
        if agent.model_name == "prior-model":
            assert agent._build_provider_settings().model_name == "prior-model"
            assert agent.resolved_model_profile is None
            call_started.set()
            await release_call.wait()
            assert agent.model_name == "prior-model"
        return "Answer."

    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    monkeypatch.setattr(agent, "_call_llm", call)
    task = asyncio.create_task(agent._async_handle_message(
        user_input(), SimpleNamespace(conversation_id="test")))
    await lookup_started.wait()
    hass.config_entries.async_update_entry(entry, options={CONF_MODEL_PROFILE: "focused"})
    release_lookup.set()
    await call_started.wait()
    assert agent.model_profile == "focused"
    assert agent.resolved_model_profile is None
    assert agent.model_name == "saved-model"
    release_call.set()
    await task
    assert agent.resolved_model_profile is None
    await agent._async_handle_message(user_input(), SimpleNamespace(conversation_id="test"))
    assert references == ["assistant", "focused"]
    assert models == ["prior-model", "next-model"]
    assert agent.resolved_model_profile["model"] == "next-model"
    assert agent._last_resolved_model_profile_reference == "focused"


@pytest.mark.parametrize("failure", [None, "timeout", "error"])
async def test_persisted_chat_log_records_resolved_model(
    hass, profile_entry_factory, monkeypatch, failure
):
    entry = profile_entry_factory(data={**entry_data(), CONF_CHAT_LOG_MODE: True})
    agent = MCPAssistConversationEntity(hass, entry)
    prepare(agent, monkeypatch)
    monkeypatch.setattr(agent, "_build_response_result",
                        MCPAssistConversationEntity._build_response_result.__get__(agent))
    manager = SimpleNamespace(async_record=AsyncMock())
    hass.data.setdefault(DOMAIN, {})["chat_log_manager"] = manager
    resolved = validate_model_profile(policy(), "assistant")
    monkeypatch.setattr(agent_module, "async_resolve_model_profile", AsyncMock(return_value=resolved))
    error = (agent_module.ProviderResponseTimeoutError(
        provider_name="OpenAI", timeout_seconds=1, transport="HTTP", attempts=1, iteration=0
    ) if failure == "timeout" else RuntimeError("Request failed") if failure else None)
    monkeypatch.setattr(agent, "_call_llm", AsyncMock(return_value="Answer.", side_effect=error))
    chat_log = SimpleNamespace(conversation_id="test",
                              async_add_assistant_content_without_tools=lambda content: None)

    result = await agent._async_handle_message(user_input(), chat_log)

    assert (result.response.error_code is not None) == bool(failure)
    manager.async_record.assert_awaited_once()
    saved = manager.async_record.await_args.args[0]
    assert saved["model"] == "example-model"
    assert saved["model"] != agent.model_name
    assert agent_module._PERSISTENT_CHAT_LOG_RECORD.get() is None


@pytest.mark.parametrize("failure", ["timeout", "error", "cancelled"])
@pytest.mark.parametrize("failure_stage", ["lookup", "generation"])
async def test_older_failed_request_preserves_newer_resolution(
    hass, profile_entry_factory, monkeypatch, failure, failure_stage
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data()))
    prepare(agent, monkeypatch)
    older_started = asyncio.Event()
    release_older = asyncio.Event()
    count = 0
    older = validate_model_profile(policy("older-model"), "assistant")
    newer = validate_model_profile(policy("newer-model"), "assistant")

    async def fail_older():
        older_started.set()
        await release_older.wait()
        if failure == "cancelled":
            raise asyncio.CancelledError
        if failure == "timeout":
            raise agent_module.ProviderResponseTimeoutError(
                provider_name="OpenAI", timeout_seconds=1, transport="HTTP",
                attempts=1, iteration=0,
            )
        raise ModelProfileResolutionError()

    async def lookup(*args):
        nonlocal count
        count += 1
        if count == 1:
            if failure_stage == "lookup":
                await fail_older()
            return older
        return newer

    async def call(messages):
        if agent.model_name == "older-model":
            await fail_older()
        return "Answer."

    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    monkeypatch.setattr(agent, "_call_llm", call)
    task = asyncio.create_task(agent._async_handle_message(
        user_input("Older"), SimpleNamespace(conversation_id="older")))
    try:
        await asyncio.wait_for(older_started.wait(), timeout=1)
        assert await agent._async_handle_message(
            user_input("Newer"), SimpleNamespace(conversation_id="newer")) == "done"
        metadata = agent.resolved_model_profile
        assert metadata["model"] == "newer-model"
        release_older.set()
        if failure == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.response.error_code is not None
        assert agent.resolved_model_profile == metadata
        assert agent_module._REQUEST_MODEL_PROFILE.get() is None
    finally:
        release_older.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("older_active", [True, False])
@pytest.mark.parametrize("failure", ["timeout", "error", "cancelled"])
async def test_newer_lookup_failure_preserves_prior_resolution(
    hass, profile_entry_factory, monkeypatch, older_active, failure
):
    agent = MCPAssistConversationEntity(hass, profile_entry_factory(data=entry_data()))
    prepare(agent, monkeypatch)
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()
    prior = validate_model_profile(policy("prior-model"), "assistant")
    count = 0

    async def lookup(*args):
        nonlocal count
        count += 1
        if count == 1:
            return prior
        lookup_started.set()
        await release_lookup.wait()
        if failure == "cancelled":
            raise asyncio.CancelledError
        if failure == "timeout":
            raise agent_module.ProviderResponseTimeoutError(
                provider_name="OpenAI", timeout_seconds=1, transport="HTTP",
                attempts=1, iteration=0,
            )
        raise ModelProfileResolutionError()

    async def call(messages):
        assert agent.model_name == "prior-model"
        generation_started.set()
        if older_active:
            await release_generation.wait()
        return "Answer."

    monkeypatch.setattr(agent_module, "async_resolve_model_profile", lookup)
    monkeypatch.setattr(agent, "_call_llm", call)
    older_task = asyncio.create_task(agent._async_handle_message(
        user_input("Older"), SimpleNamespace(conversation_id="older")))
    tasks = [older_task]
    try:
        await asyncio.wait_for(generation_started.wait(), timeout=1)
        if not older_active:
            assert await older_task == "done"
        metadata = agent.resolved_model_profile
        assert metadata["model"] == "prior-model"
        newer_task = asyncio.create_task(agent._async_handle_message(
            user_input("Newer"), SimpleNamespace(conversation_id="newer")))
        tasks.append(newer_task)
        await asyncio.wait_for(lookup_started.wait(), timeout=1)
        assert agent.resolved_model_profile == metadata
        release_lookup.set()
        if failure == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await newer_task
        else:
            result = await newer_task
            assert result.response.error_code is not None
        assert agent.resolved_model_profile == metadata
        release_generation.set()
        assert await older_task == "done"
        assert agent.resolved_model_profile == metadata
        assert agent_module._REQUEST_MODEL_PROFILE.get() is None
    finally:
        release_generation.set()
        release_lookup.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
