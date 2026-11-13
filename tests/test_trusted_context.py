"""Native trusted context and live authorization boundaries."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context
from homeassistant.auth.permissions.const import POLICY_READ

from custom_components.mcp_assist import trusted_context as api
from custom_components.mcp_assist.const import DOMAIN
from custom_components.mcp_assist.custom_tool_api import call_mcp_tool
from custom_components.mcp_assist.tools.external_loader import ExternalCustomToolLoader


@pytest.fixture
def authority(hass, monkeypatch):
    entry = SimpleNamespace(entry_id="profile", domain=DOMAIN, state=ConfigEntryState.LOADED,
                            disabled_by=None, options={"control_home_assistant": True}, data={})
    user = SimpleNamespace(is_active=True, permissions=SimpleNamespace(check_entity=Mock(return_value=True)))
    monkeypatch.setattr(hass.config_entries, "async_get_entry", lambda key: entry if key == "profile" else None)
    monkeypatch.setattr(hass.auth, "async_get_user", AsyncMock(return_value=user))
    monkeypatch.setattr(api, "async_should_expose", lambda *args: True)
    loader = SimpleNamespace(external_tools_enabled=lambda: True, external_tools_ready=lambda: True,
                             is_external_package_enabled=lambda package: package == "example")
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(tools=loader)
    hass.states.async_set("light.example", "off")
    return entry, user, loader


@pytest.mark.asyncio
async def test_read_control_and_current_policy(hass, authority):
    entry, user, loader = authority
    kwargs = dict(user_id="user", profile_entry_id="profile", entity_ids=["light.example"],
                  package_id="example")
    entry.options["control_home_assistant"] = False
    assert (await api.async_check_profile_entity_access(hass, permission="read", **kwargs)).user_id == "user"
    user.permissions.check_entity.assert_called_with("light.example", POLICY_READ)
    with pytest.raises(api.ToolAccessDenied):
        await api.async_check_profile_entity_access(hass, **kwargs)
    entry.options["control_home_assistant"] = True
    user.is_active = False
    with pytest.raises(api.ToolAccessDenied):
        await api.async_check_profile_entity_access(hass, **kwargs)
    user.is_active = True
    loader.is_external_package_enabled = lambda _: None
    with pytest.raises(api.ToolAccessUnavailable):
        await api.async_check_profile_entity_access(hass, **kwargs)
    assert not issubclass(api.ToolAccessUnavailable, api.ToolAccessDenied)


@pytest.mark.asyncio
async def test_scope_revocation_and_nested_call(hass, authority):
    scope = api.new_dispatch_scope(authority[0], "example", Context(user_id="user"))
    token = api.REQUEST_TRUSTED_SCOPE.set(scope)
    gate = asyncio.Event()
    async def child():
        await gate.wait()
        with pytest.raises(api.ToolAccessDenied):
            api.get_trusted_call_context(hass)
    try:
        first = api.get_trusted_call_context(hass)
        assert first == api.get_trusted_call_context(hass)
        assert first.dispatch_id != api.new_dispatch_scope(authority[0], "example", scope.context).dispatch_id
        async def nested(_):
            with pytest.raises(api.ToolAccessDenied):
                api.get_trusted_call_context(hass)
            return {}
        hass.data[DOMAIN]["shared_mcp_server"].handle_tool_call = nested
        await call_mcp_tool(hass, "nested", context={"user_id": "spoofed"})
        task = asyncio.create_task(child())
        scope.active = False
        gate.set()
        await task
    finally:
        api.REQUEST_TRUSTED_SCOPE.reset(token)
    with pytest.raises(api.ToolAccessDenied):
        api.get_trusted_call_context(hass)


@pytest.mark.parametrize("api_value", [{"other": 1}, {"trusted_call_context": True},
                                       {"trusted_call_context": 2}, [], None])
def test_manifest_rejects_invalid_api(hass, tmp_path, api_value):
    directory = tmp_path / "example"
    directory.mkdir()
    manifest = {"schema_version": 1, "id": "example", "name": "Example", "description": "Example",
                "version": "1", "entrypoint": "tool:Tool", "api": api_value}
    (directory / "mcp_tool.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ExternalCustomToolLoader(hass)._load_manifest_from_disk(directory)


def test_manifest_accepts_opt_in(hass, tmp_path):
    directory = tmp_path / "example"
    directory.mkdir()
    manifest = {"schema_version": 1, "id": "example", "name": "Example", "description": "Example",
                "version": "1", "entrypoint": "tool:Tool", "api": {"trusted_call_context": 1}}
    (directory / "mcp_tool.json").write_text(json.dumps(manifest))
    assert ExternalCustomToolLoader(hass)._load_manifest_from_disk(directory).api == manifest["api"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["return", "cancel", "timeout"])
async def test_agent_revokes_inherited_scope(hass, profile_entry_factory, monkeypatch, outcome):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity

    entry = profile_entry_factory(data={"control_home_assistant": True})
    agent = MCPAssistConversationEntity(hass, entry)
    loader = SimpleNamespace(requires_trusted_call_context=lambda name: True)
    monkeypatch.setattr(agent, "_get_shared_tools_loader", lambda: loader)
    monkeypatch.setattr(agent, "_is_tool_configured_for_profile", lambda name: True)
    monkeypatch.setattr(agent, "_build_mcp_tool_call_context", AsyncMock(return_value={"user_id": "spoof"}))
    gate = asyncio.Event()
    entered = asyncio.Event()
    children = []
    dispatch_ids = []

    async def child():
        await gate.wait()
        with pytest.raises(api.ToolAccessDenied):
            api.get_trusted_call_context(hass)

    async def handle(_):
        identity = api.get_trusted_call_context(hass)
        assert identity.user_id == "native-user"
        dispatch_ids.append(identity.dispatch_id)
        children.append(asyncio.create_task(child()))
        entered.set()
        if outcome != "return":
            await asyncio.Event().wait()
        return {"content": [{"type": "text", "text": "ok"}]}

    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=handle)
    if outcome == "timeout":
        real_timeout = asyncio.timeout
        monkeypatch.setattr(agent_module.asyncio, "timeout", lambda _: real_timeout(0.01))
    task = asyncio.create_task(agent._call_mcp_tool(
        "example", {}, SimpleNamespace(context=Context(user_id="native-user"))))
    await entered.wait()
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        if outcome == "timeout":
            body = json.loads(result["content"][0]["text"])
            assert body == {"error": "tool_result_uncertain", "dispatch_id": dispatch_ids[0]}
    gate.set()
    await asyncio.gather(*children)


@pytest.mark.asyncio
async def test_real_external_loader_registry_and_nested_scope(hass, tmp_path, authority, monkeypatch):
    from custom_components.mcp_assist.tools import CustomToolsLoader
    from tests.test_custom_tools_loader import _write_external_tool_package

    package_dir = _write_external_tool_package(tmp_path, tool_id="example")
    manifest_path = package_dir / "mcp_tool.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["api"] = {"trusted_call_context": 1}
    manifest_path.write_text(json.dumps(manifest))
    external = ExternalCustomToolLoader(hass, tools_root=package_dir.parent,
                                        settings_root=tmp_path / "settings")
    packages = await external.load()
    assert len(packages) == 1
    loader = CustomToolsLoader(hass)
    loader._external_loader = external
    loader.external_tools = packages
    monkeypatch.setattr(loader, "_external_custom_tools_enabled", lambda: True)
    loader._refresh_tool_registry()
    assert loader.requires_trusted_call_context("example_status")
    assert not loader.requires_trusted_call_context("unknown")
    registry_entry = loader._tool_registry["example_status"]
    registry_entry.is_external = False
    assert not loader.requires_trusted_call_context("example_status")
    registry_entry.is_external = True
    registry_entry.loaded_tool_package.manifest.api.clear()
    assert not loader.requires_trusted_call_context("example_status")
    registry_entry.loaded_tool_package.manifest.api["trusted_call_context"] = 1
    assert loader.is_external_package_enabled("example")
    assert not loader.is_external_package_enabled("unknown")
    calls = []

    async def handle(name, arguments):
        calls.append(name)
        if len(calls) == 1:
            assert api.get_trusted_call_context(hass).user_id == "user"
            await loader.handle_tool_call(name, arguments)
            assert api.get_trusted_call_context(hass).user_id == "user"
        else:
            with pytest.raises(api.ToolAccessDenied):
                api.get_trusted_call_context(hass)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(packages[0].instance, "handle_call", handle)
    scope = api.new_dispatch_scope(authority[0], "example_status", Context(user_id="user"))
    token = api.REQUEST_TRUSTED_SCOPE.set(scope)
    try:
        await loader.handle_tool_call("example_status", {})
        assert len(calls) == 2
        with pytest.raises(ValueError):
            await loader.handle_tool_call("unknown", {})
    finally:
        api.REQUEST_TRUSTED_SCOPE.reset(token)
        await packages[0].instance.async_shutdown()


@pytest.mark.asyncio
async def test_concurrent_provider_calls_have_isolated_native_identity_and_unique_ids(
    hass, profile_entry_factory, monkeypatch,
):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity

    entries = [profile_entry_factory(unique_id=f"profile-{i}") for i in range(2)]
    agents = [MCPAssistConversationEntity(hass, entry) for entry in entries]
    for agent in agents:
        monkeypatch.setattr(agent, "_get_shared_tools_loader", lambda: SimpleNamespace(
            requires_trusted_call_context=lambda _: True))
        monkeypatch.setattr(agent, "_is_tool_configured_for_profile", lambda _: True)
        monkeypatch.setattr(agent, "_build_mcp_tool_call_context", AsyncMock(return_value={"user_id": "spoof"}))
    identities = []
    barrier = asyncio.Event()

    async def handle(_):
        identity = api.get_trusted_call_context(hass)
        identities.append(identity)
        if len(identities) == 4:
            barrier.set()
        await barrier.wait()
        assert api.get_trusted_call_context(hass) == identity
        return {"content": [{"type": "text", "text": "ok"}]}

    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=handle)

    async def call(index, provider_id):
        token = agent_module._REQUEST_USER_INPUT.set(SimpleNamespace(context=Context(user_id=f"user-{index}")))
        try:
            call_data = {"function": {"name": "example", "arguments": "{}"}}
            if provider_id is not None:
                call_data["id"] = provider_id
            await agents[index]._execute_single_tool_call(call_data)
        finally:
            agent_module._REQUEST_USER_INPUT.reset(token)

    await asyncio.gather(call(0, None), call(0, "same"), call(1, "same"), call(1, None))
    assert len({item.dispatch_id for item in identities}) == 4
    assert {(item.user_id, item.profile_entry_id) for item in identities} == {
        ("user-0", entries[0].entry_id), ("user-1", entries[1].entry_id)}


@pytest.mark.asyncio
async def test_timeout_without_native_identity_has_no_dispatch_id(hass, profile_entry_factory, monkeypatch):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity

    entry = profile_entry_factory()
    agent = MCPAssistConversationEntity(hass, entry)
    monkeypatch.setattr(agent, "_get_shared_tools_loader", lambda: SimpleNamespace(
        requires_trusted_call_context=lambda _: True))
    monkeypatch.setattr(agent, "_is_tool_configured_for_profile", lambda _: True)
    monkeypatch.setattr(agent, "_build_mcp_tool_call_context", AsyncMock(return_value={"user_id": "spoof"}))
    async def handle(_):
        with pytest.raises(api.ToolAccessDenied):
            api.get_trusted_call_context(hass)
        await asyncio.Event().wait()
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=handle)
    real_timeout = asyncio.timeout
    monkeypatch.setattr(agent_module.asyncio, "timeout", lambda _: real_timeout(0.01))
    result = await agent._call_mcp_tool("example", {}, SimpleNamespace(context=Context()))
    assert "dispatch_id" not in json.dumps(result)
    assert "tool_result_uncertain" not in json.dumps(result)


@pytest.mark.asyncio
async def test_ordinary_external_tool_uses_http_without_native_scope(hass, profile_entry_factory, monkeypatch):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity

    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    monkeypatch.setattr(agent, "_get_shared_tools_loader", lambda: SimpleNamespace(
        requires_trusted_call_context=lambda _: False))
    monkeypatch.setattr(agent, "_is_tool_configured_for_profile", lambda _: True)
    monkeypatch.setattr(agent, "_build_mcp_tool_call_context", AsyncMock(return_value={"user_id": "spoof"}))
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(
        handle_tool_call=AsyncMock(side_effect=AssertionError("Unexpected in-process dispatch")))
    captured = []
    class Response:
        status = 200
        async def __aenter__(self):
            with pytest.raises(api.ToolAccessDenied):
                api.get_trusted_call_context(hass)
            return self
        async def __aexit__(self, *args):
            return False
        async def json(self):
            return {"result": {"content": [{"type": "text", "text": "ok"}]}}
    class Session:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def post(self, url, **kwargs):
            captured.append(kwargs["json"])
            return Response()
    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", Session)
    result = await agent._call_mcp_tool("example", {}, SimpleNamespace(context=Context(user_id="native")))
    assert result["content"][0]["text"] == "ok"
    assert captured[0]["params"]["context"]["user_id"] == "spoof"
