"""Standard compatible endpoint model retrieval."""

from dataclasses import replace
import json

import pytest

from custom_components.mcp_assist import model_selection
from custom_components.mcp_assist.const import (
    CONF_API_KEY, CONF_LMSTUDIO_URL, CONF_MODEL_NAME, CONF_SERVER_TYPE, SERVER_TYPE_OPENAI,
)
from custom_components.mcp_assist.provider_runtime import resolve_provider_runtime_config


class Response:
    def __init__(self, body=b'{}', status=200, content_length=None):
        self.body, self.status, self.content_length = body, status, content_length
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def iter_chunked(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start:start + size]


def session(monkeypatch, response):
    calls = []

    class Session:
        def __init__(self, **kwargs):
            assert 0 < kwargs['timeout'].total <= 15

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return response

    monkeypatch.setattr(model_selection.aiohttp, 'ClientSession', Session)
    return calls


@pytest.fixture
def runtime(profile_entry_factory):
    return resolve_provider_runtime_config(profile_entry_factory(data={
        CONF_SERVER_TYPE: SERVER_TYPE_OPENAI, CONF_MODEL_NAME: 'selection',
        CONF_API_KEY: 'example-key', CONF_LMSTUDIO_URL: 'https://example.invalid/v1',
    }))


async def test_retrieves_opaque_id_and_authenticates(runtime, monkeypatch):
    selected = 'profile/example'
    frozen = 'profile/example@' + 'a' * 64
    calls = session(monkeypatch, Response(json.dumps({'id': frozen}).encode()))
    assert await model_selection.async_resolve_model(runtime, selected) == frozen
    assert calls == [('https://example.invalid/v1/models/profile/example', {
        'headers': {'Authorization': 'Bearer example-key'}, 'allow_redirects': False,
    })]


@pytest.mark.parametrize('status', [404, 405])
async def test_unsupported_retrieve_keeps_selection(runtime, monkeypatch, status):
    session(monkeypatch, Response(status=status))
    assert await model_selection.async_resolve_model(runtime, 'vendor/model') == 'vendor/model'


@pytest.mark.parametrize('response', [
    Response(status=302), Response(status=401), Response(status=500),
    Response(content_length=1024 * 1024 + 1), Response(b'x' * (1024 * 1024 + 1)),
    Response(b'invalid'), Response(b'{}'), Response(b'{"id":""}'),
    Response(json.dumps({'id': 'x' * 513}).encode()),
])
async def test_rejects_failures(runtime, monkeypatch, response):
    session(monkeypatch, response)
    with pytest.raises(model_selection.ModelResolutionError):
        await model_selection.async_resolve_model(runtime, 'selection')


async def test_optional_auth_and_official_endpoint(runtime, monkeypatch):
    calls = session(monkeypatch, Response(b'{"id":"concrete"}'))
    assert await model_selection.async_resolve_model(replace(runtime, api_key=''), 'a') == 'concrete'
    assert calls[0][1]['headers'] == {}
    assert await model_selection.async_resolve_model(
        replace(runtime, base_url='https://api.openai.com/v1'), 'a'
    ) == 'a'
    assert len(calls) == 1


async def test_image_resolution_is_lazy_and_shared(hass, profile_entry_factory, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock
    from custom_components.mcp_assist import agent as agent_module

    entry = profile_entry_factory(data={
        CONF_SERVER_TYPE: SERVER_TYPE_OPENAI, CONF_MODEL_NAME: 'text',
        CONF_LMSTUDIO_URL: 'https://example.invalid/v1',
    })
    agent = agent_module.MCPAssistConversationEntity(hass, entry)
    resolve = AsyncMock(return_value='opaque-frozen-image')
    monkeypatch.setattr(agent_module, 'async_resolve_model', resolve)
    scoped = agent_module.RequestModelSelection(entry, 'opaque-image')
    token = agent_module._REQUEST_IMAGE_MODEL.set(scoped)
    try:
        assert resolve.await_count == 0
        await asyncio.gather(
            agent._resolve_request_image_model(), agent._resolve_request_image_model()
        )
        assert scoped.resolved_image == 'opaque-frozen-image'
        assert resolve.await_count == 1
        await agent._resolve_request_image_model()
        assert resolve.await_count == 1
    finally:
        agent_module._REQUEST_IMAGE_MODEL.reset(token)


async def test_text_selection_is_frozen_per_concurrent_request(hass, profile_entry_factory, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from homeassistant.components.conversation import ConversationInput
    from homeassistant.core import Context
    from custom_components.mcp_assist import agent as agent_module

    entry = profile_entry_factory(data={
        CONF_SERVER_TYPE: SERVER_TYPE_OPENAI, CONF_MODEL_NAME: 'alias',
        CONF_LMSTUDIO_URL: 'https://example.invalid/v1',
    })
    agent = agent_module.MCPAssistConversationEntity(hass, entry)
    started = asyncio.Event()
    release = asyncio.Event()
    resolved = []

    async def lookup(runtime, model):
        value = f'frozen-{len(resolved)}'
        resolved.append(value)
        return value

    async def call_llm(messages):
        first = agent._build_provider_settings().model_name
        if first == 'frozen-0':
            started.set()
            await release.wait()
        else:
            await started.wait()
            release.set()
        assert agent._build_provider_settings().model_name == first
        return first

    monkeypatch.setattr(agent_module, 'async_resolve_model', lookup)
    monkeypatch.setattr(agent, '_build_system_prompt_with_context', AsyncMock(return_value='system'))
    monkeypatch.setattr(agent, '_prepare_adaptive_tools_for_request', AsyncMock())
    monkeypatch.setattr(agent, '_build_turn_context', AsyncMock(return_value=''))
    monkeypatch.setattr(agent, '_call_llm', call_llm)
    monkeypatch.setattr(agent, '_build_response_result', AsyncMock(side_effect=lambda text, *_: text))
    def request(name):
        return ConversationInput(text='test', context=Context(), conversation_id=name,
                                 device_id=None, satellite_id=None, language='en', agent_id='test')
    results = await asyncio.gather(*(
        agent._async_handle_message(request(name), SimpleNamespace(conversation_id=name))
        for name in ('one', 'two')
    ))
    assert set(results) == {'frozen-0', 'frozen-1'}
    assert len(resolved) == 2
    assert agent_module.REQUEST_MODEL_SELECTION.get() is None


def test_model_snapshot_bridge_is_bound_and_single_use(hass, profile_entry_factory):
    from types import SimpleNamespace
    from custom_components.mcp_assist.mcp_server import MCPServer
    from custom_components.mcp_assist.model_selection import (
        MCP_MODEL_REQUEST_HEADER, ModelResolutionError,
    )

    entry = profile_entry_factory()
    server = MCPServer(hass, 8099, entry)
    snapshot = (entry, 'opaque-text', 'opaque-image')
    token = server.register_model_request(snapshot, 'request-1', 'generate_image')
    request = SimpleNamespace(remote='127.0.0.1', headers={MCP_MODEL_REQUEST_HEADER: token})
    data = {'method': 'tools/call', 'id': 'request-1', 'params': {
        'name': 'generate_image', 'context': {'profile_entry_id': entry.entry_id},
    }}
    assert server._take_model_request(request, data) == snapshot
    with pytest.raises(ModelResolutionError):
        server._take_model_request(request, data)
    token = server.register_model_request(snapshot, 'request-1', 'generate_image')
    request.headers[MCP_MODEL_REQUEST_HEADER] = token
    request.remote = '192.0.2.1'
    with pytest.raises(ModelResolutionError):
        server._take_model_request(request, data)


async def test_analyze_image_forwards_frozen_text_without_image_lookup(
    hass, profile_entry_factory, monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.const import DOMAIN

    entry = profile_entry_factory(data={
        CONF_SERVER_TYPE: SERVER_TYPE_OPENAI, CONF_MODEL_NAME: 'alias',
        CONF_LMSTUDIO_URL: 'https://example.invalid/v1',
    })
    agent = agent_module.MCPAssistConversationEntity(hass, entry)
    server = SimpleNamespace(register_model_request=Mock(return_value='token'),
                             discard_model_request=Mock())
    hass.data.setdefault(DOMAIN, {})['shared_mcp_server'] = server
    lookup = AsyncMock()
    monkeypatch.setattr(agent_module, 'async_resolve_model', lookup)
    monkeypatch.setattr(agent, '_is_tool_configured_for_profile', lambda name: True)
    monkeypatch.setattr(agent, '_get_tool_effect', lambda name: agent_module.ToolEffect.READ_ONLY)
    monkeypatch.setattr(agent, '_build_mcp_tool_call_context', AsyncMock(return_value={}))
    captured = {}

    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        async def json(self): return {'result': {'content': [], 'isError': False}}

    class Session:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        def post(self, url, **kwargs):
            captured.update(kwargs)
            return Response()

    monkeypatch.setattr(agent_module.aiohttp, 'ClientSession', Session)
    token = agent_module.REQUEST_MODEL_SELECTION.set((entry, 'frozen-text', None))
    try:
        await agent._call_mcp_tool('analyze_image', {})
    finally:
        agent_module.REQUEST_MODEL_SELECTION.reset(token)
    assert server.register_model_request.call_args.args[0] == (entry, 'frozen-text', None)
    assert server.register_model_request.call_args.args[2] == 'analyze_image'
    assert captured['headers'][agent_module.MCP_MODEL_REQUEST_HEADER] == 'token'
    lookup.assert_not_awaited()
