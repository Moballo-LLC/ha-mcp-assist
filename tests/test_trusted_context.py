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

_TEST_INSTANCE = object()


@pytest.fixture(autouse=True)
def fake_instance_binding():
    token = api.ACTIVE_TRUSTED_INSTANCE.set(_TEST_INSTANCE)
    yield
    api.ACTIVE_TRUSTED_INSTANCE.reset(token)


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
            api._get_trusted_call_context(hass, _TEST_INSTANCE)
    try:
        first = api._get_trusted_call_context(hass, _TEST_INSTANCE)
        assert first == api._get_trusted_call_context(hass, _TEST_INSTANCE)
        assert first.dispatch_id != api.new_dispatch_scope(authority[0], "example", scope.context).dispatch_id
        async def nested(_):
            with pytest.raises(api.ToolAccessDenied):
                api._get_trusted_call_context(hass, _TEST_INSTANCE)
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
        api._get_trusted_call_context(hass, _TEST_INSTANCE)


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
    assert ExternalCustomToolLoader(hass)._load_manifest_from_disk(directory).api == (("trusted_call_context", 1),)


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
            api._get_trusted_call_context(hass, _TEST_INSTANCE)

    async def handle(_):
        identity = api._get_trusted_call_context(hass, _TEST_INSTANCE)
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
    assert loader.is_external_package_enabled("example")
    assert not loader.is_external_package_enabled("unknown")
    calls = []
    other_instance = type(packages[0].instance)(
        hass, packages[0].manifest, packages[0].instance.tool_dir
    )
    service_checks = []
    async def other_service(_call):
        with pytest.raises(api.ToolAccessDenied):
            other_instance.get_trusted_call_context()
        service_checks.append(True)
    hass.services.async_register("test", "other_package", other_service)

    async def handle(name, arguments):
        calls.append(name)
        if len(calls) == 1:
            assert packages[0].instance.get_trusted_call_context().user_id == "user"
            await hass.services.async_call("test", "other_package", blocking=True)
            assert service_checks == [True]
            await loader.handle_tool_call(name, arguments)
            assert packages[0].instance.get_trusted_call_context().user_id == "user"
        else:
            with pytest.raises(api.ToolAccessDenied):
                packages[0].instance.get_trusted_call_context()
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
        identity = api._get_trusted_call_context(hass, _TEST_INSTANCE)
        identities.append(identity)
        if len(identities) == 4:
            barrier.set()
        await barrier.wait()
        assert api._get_trusted_call_context(hass, _TEST_INSTANCE) == identity
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
            api._get_trusted_call_context(hass, _TEST_INSTANCE)
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
                api._get_trusted_call_context(hass, _TEST_INSTANCE)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["return", "cancel", "timeout"])
@pytest.mark.parametrize("direct_bridge", [False, True])
async def test_real_bridge_identity_is_not_delegated_or_inherited_after_dispatch(
    hass, profile_entry_factory, monkeypatch, outcome, direct_bridge,
):
    from homeassistant.helpers import llm
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity
    from custom_components.mcp_assist.request_context import current_ha_context
    from custom_components.mcp_assist.tools import CustomToolsLoader
    from custom_components.mcp_assist.tools.packages.llm_api_bridge.llm_api_bridge import LLMApiBridgeTool

    agent = MCPAssistConversationEntity(hass, profile_entry_factory())
    monkeypatch.setattr(agent, "_get_shared_tools_loader", lambda: SimpleNamespace(
        requires_trusted_call_context=lambda name: name == "example_status"))
    monkeypatch.setattr(agent, "_is_tool_configured_for_profile", lambda _: True)
    monkeypatch.setattr(agent, "_build_mcp_tool_call_context", AsyncMock(return_value={"user_id": "spoof"}))
    bridge = LLMApiBridgeTool(hass)
    monkeypatch.setattr(bridge, "_allowed_llm_api_ids", lambda: ("example_api",))
    observed = []
    async def get_api(hass_arg, api_id, llm_context):
        observed.append(llm_context.context.user_id)
        async def invoke(tool_input):
            return {"ok": True}
        return SimpleNamespace(api=SimpleNamespace(id=api_id), async_call_tool=invoke)
    monkeypatch.setattr(llm, "async_get_api", get_api)
    args = {"api_id": "example_api", "tool_name": "Example", "arguments": {}}
    loader = CustomToolsLoader(hass)
    async def loader_call(name, arguments, *, context=None):
        return await bridge.handle_call(name, arguments)
    monkeypatch.setattr(loader, "_handle_tool_call", loader_call)
    gate = asyncio.Event()
    entered = asyncio.Event()
    children = []
    async def invoke_bridge():
        return await bridge.handle_call("call_llm_api_tool", args)
    async def child():
        await gate.wait()
        if not direct_bridge:
            with pytest.raises(api.ToolAccessDenied):
                api._get_trusted_call_context(hass, _TEST_INSTANCE)
        await invoke_bridge()
    async def server_call(params):
        if params["name"] == "call_llm_api_tool" and not direct_bridge:
            return await loader.handle_tool_call(params["name"], params["arguments"])
        if direct_bridge:
            await invoke_bridge()
        else:
            assert api._get_trusted_call_context(hass, _TEST_INSTANCE).user_id == "native"
            assert current_ha_context().user_id is None
            await call_mcp_tool(hass, "call_llm_api_tool", args, context={"user_id": "spoof"})
            # Also exercise direct loader dispatch from the external tool.
            await loader.handle_tool_call("call_llm_api_tool", args)
        children.append(asyncio.create_task(child()))
        entered.set()
        if outcome != "return":
            await asyncio.Event().wait()
        return {"content": [{"type": "text", "text": "ok"}]}
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=server_call)
    if outcome == "timeout":
        real_timeout = asyncio.timeout
        monkeypatch.setattr(agent_module.asyncio, "timeout", lambda _: real_timeout(0.01))
    name = "call_llm_api_tool" if direct_bridge else "example_status"
    task = asyncio.create_task(agent._call_mcp_tool(name, args, SimpleNamespace(context=Context(user_id="native"))))
    await entered.wait()
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    gate.set()
    await asyncio.gather(*children)
    assert observed == (["native", None] if direct_bridge else [None, None, None])


@pytest.mark.asyncio
async def test_nested_bridge_calls_clear_active_legacy_identity(hass, monkeypatch):
    from homeassistant.helpers import llm
    from custom_components.mcp_assist.request_context import HAContextScope, REQUEST_HA_CONTEXT
    from custom_components.mcp_assist.tools import CustomToolsLoader
    from custom_components.mcp_assist.tools.packages.llm_api_bridge.llm_api_bridge import LLMApiBridgeTool

    bridge = LLMApiBridgeTool(hass)
    monkeypatch.setattr(bridge, "_allowed_llm_api_ids", lambda: ("example_api",))
    observed = []
    async def get_api(hass_arg, api_id, llm_context):
        observed.append(llm_context.context.user_id)
        async def invoke(tool_input):
            return {"ok": True}
        return SimpleNamespace(api=SimpleNamespace(id=api_id), async_call_tool=invoke)
    monkeypatch.setattr(llm, "async_get_api", get_api)
    loader = CustomToolsLoader(hass)
    args = {"api_id": "example_api", "tool_name": "Example", "arguments": {}}
    depth = 0
    async def loader_call(name, arguments, *, context=None):
        nonlocal depth
        depth += 1
        try:
            if depth == 1:
                await bridge.handle_call(name, arguments)
                await call_mcp_tool(hass, name, arguments)
                await loader.handle_tool_call(name, arguments)
                await bridge.handle_call(name, arguments)
                return {}
            return await bridge.handle_call(name, arguments)
        finally:
            depth -= 1
    monkeypatch.setattr(loader, "_handle_tool_call", loader_call)
    async def server_call(params):
        return await loader.handle_tool_call(params["name"], params["arguments"])
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=server_call)
    scope = HAContextScope(Context(user_id="native"), "call_llm_api_tool")
    token = REQUEST_HA_CONTEXT.set(scope)
    try:
        await loader.handle_tool_call("call_llm_api_tool", args)
    finally:
        scope.active = False
        REQUEST_HA_CONTEXT.reset(token)
    assert observed == ["native", None, None, "native"]


def test_manifest_preserves_positional_construction_and_hashability():
    from custom_components.mcp_assist.custom_tool_api import MCPAssistCustomToolManifest
    old = MCPAssistCustomToolManifest(1, "example", "Example", "Description", "1", "tool:Tool",
                                     ("description",), "prompt.md")
    assert old.prompt_append_file == "prompt.md"
    assert old.api == ()
    opted = MCPAssistCustomToolManifest(1, "example", "Example", "Description", "1", "tool:Tool",
                                       ("description",), "prompt.md",
                                       api=(("trusted_call_context", 1),))
    assert isinstance(hash(old), int)
    assert isinstance(hash(opted), int)


@pytest.mark.asyncio
async def test_entity_validation_precedes_exposure_and_distinguishes_unavailable(hass, authority, monkeypatch):
    from homeassistant.helpers import entity_registry as er
    entry, _, _ = authority
    exposure = Mock(return_value=True)
    monkeypatch.setattr(api, "async_should_expose", exposure)
    kwargs = dict(user_id="user", profile_entry_id=entry.entry_id)
    for entity_id in ("invalid", "light.invalid name", "light.absent"):
        with pytest.raises(api.ToolAccessDenied):
            await api.async_check_profile_entity_access(hass, entity_ids=[entity_id], **kwargs)
        exposure.assert_not_called()
    registry = er.async_get(hass)
    registered = registry.async_get_or_create("light", "test", "example-id", suggested_object_id="registered")
    with pytest.raises(api.ToolAccessUnavailable):
        await api.async_check_profile_entity_access(hass, entity_ids=[registered.entity_id], **kwargs)
    exposure.assert_not_called()
    registry.async_update_entity(registered.entity_id, disabled_by=er.RegistryEntryDisabler.USER)
    with pytest.raises(api.ToolAccessDenied):
        await api.async_check_profile_entity_access(hass, entity_ids=[registered.entity_id], **kwargs)
    exposure.assert_not_called()
