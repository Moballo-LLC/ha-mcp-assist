"""Trusted principal creation and current-policy deferred authorization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context
from homeassistant.auth.permissions.const import POLICY_CONTROL, POLICY_READ

from custom_components.mcp_assist import deferred_actions as api
from custom_components.mcp_assist.const import DOMAIN


@pytest.fixture
def authority(hass, monkeypatch):
    entry = SimpleNamespace(entry_id="profile", domain=DOMAIN, state=ConfigEntryState.LOADED,
                            disabled_by=None, options={"control_home_assistant": True}, data={})
    user = SimpleNamespace(is_active=True, permissions=SimpleNamespace(check_entity=Mock(return_value=True)))
    monkeypatch.setattr(hass.config_entries, "async_get_entry", lambda key: entry if key == "profile" else None)
    monkeypatch.setattr(hass.auth, "async_get_user", AsyncMock(return_value=user))
    monkeypatch.setattr(api, "async_should_expose", lambda *args: True)
    loader = SimpleNamespace(requires_deferred_action_context=lambda name: name == "schedule_create",
                             _external_custom_tools_enabled=lambda: True)
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(tools=loader)
    hass.states.async_set("light.example", "off")
    return entry, user, loader


@pytest.mark.asyncio
async def test_trigger_uses_read_permission_and_action_still_needs_control(hass, authority):
    user = authority[1]
    hass.states.async_set("person.example", "home")
    user.permissions.check_entity.side_effect = lambda entity_id, policy: policy == (POLICY_READ if entity_id.startswith("person.") else POLICY_CONTROL)
    context = await api.async_authorize_deferred_action(hass,
        {"user_id": "user", "profile_id": "profile"}, ["light.example"],
        read_entity_ids=["person.example"], tool_name="schedule_create")
    assert context.user_id == "user"
    with pytest.raises(api.DeferredActionAuthorizationError):
        await api.async_authorize_deferred_action(hass,
            {"user_id": "user", "profile_id": "profile"}, ["person.example"], tool_name="schedule_create")


@pytest.mark.asyncio
async def test_payload_context_is_not_an_authenticated_principal(hass, authority):
    with pytest.raises(api.DeferredActionAuthorizationError):
        await api.async_get_deferred_action_principal(hass)


@pytest.mark.asyncio
async def test_creation_key_is_scoped_to_the_actual_request_and_call(hass, authority):
    entry, _, _ = authority
    context = Context(user_id="user")
    token = api.REQUEST_DEFERRED_ACTION.set((entry, "schedule_create", "provider-call", context))
    try:
        one = await api.async_get_deferred_action_principal(hass)
        assert one == await api.async_get_deferred_action_principal(hass)
        assert one["user_id"] == "user" and one["profile_id"] == "profile"
        assert len(one["creation_key"]) == 64
        api.REQUEST_DEFERRED_ACTION.set((entry, "schedule_create", "provider-call", Context(user_id="user")))
        assert (await api.async_get_deferred_action_principal(hass))["creation_key"] != one["creation_key"]
    finally:
        api.REQUEST_DEFERRED_ACTION.reset(token)


@pytest.mark.parametrize("call_id,context", [(None, Context(user_id="user")), ("", Context(user_id="user")),
                                            ("call", Context()), ("call", {"user_id": "user"})])
@pytest.mark.asyncio
async def test_missing_native_identity_or_stable_call_id_fails_closed(hass, authority, call_id, context):
    token = api.REQUEST_DEFERRED_ACTION.set((authority[0], "schedule_create", call_id, context))
    try:
        with pytest.raises(api.DeferredActionAuthorizationError):
            await api.async_get_deferred_action_principal(hass)
    finally:
        api.REQUEST_DEFERRED_ACTION.reset(token)


@pytest.mark.asyncio
async def test_current_permissions_profile_and_exposure_rechecked(hass, authority, monkeypatch):
    entry, user, loader = authority
    owner = {"user_id": "user", "profile_id": "profile"}
    async def check():
        return await api.async_authorize_deferred_action(hass, owner, ["light.example"], tool_name="schedule_create")
    assert (await check()).user_id == "user"
    for attribute, value, original in (("is_active", False, True),):
        setattr(user, attribute, value)
        with pytest.raises(api.DeferredActionAuthorizationError):
            await check()
        setattr(user, attribute, original)
    user.permissions.check_entity.return_value = False
    with pytest.raises(api.DeferredActionAuthorizationError):
        await check()
    user.permissions.check_entity.return_value = True
    entry.options["control_home_assistant"] = False
    with pytest.raises(api.DeferredActionAuthorizationError):
        await check()
    entry.options["control_home_assistant"] = True
    entry.options["profile_enable_external_custom_tools"] = False
    with pytest.raises(api.DeferredActionAuthorizationError):
        await check()
    entry.options.clear()
    loader._external_custom_tools_enabled = lambda: False
    with pytest.raises(api.DeferredActionAuthorizationError):
        await check()
    loader._external_custom_tools_enabled = lambda: True
    entry.state = ConfigEntryState.NOT_LOADED
    with pytest.raises(api.DeferredActionAuthorizationError):
        await check()
    entry.state = ConfigEntryState.LOADED
    monkeypatch.setattr(api, "async_should_expose", lambda *args: False)
    with pytest.raises(api.DeferredActionAuthorizationError):
        await check()


@pytest.mark.asyncio
async def test_replaced_profile_and_unknown_tool_rejected(hass, authority, monkeypatch):
    entry, _, _ = authority
    token = api.REQUEST_DEFERRED_ACTION.set((entry, "schedule_create", "call", Context(user_id="user")))
    try:
        monkeypatch.setattr(hass.config_entries, "async_get_entry", lambda key: SimpleNamespace(**vars(entry)))
        with pytest.raises(api.DeferredActionAuthorizationError):
            await api.async_get_deferred_action_principal(hass)
    finally:
        api.REQUEST_DEFERRED_ACTION.reset(token)
    with pytest.raises(api.DeferredActionAuthorizationError):
        await api.async_authorize_deferred_action(hass, {"user_id": "user", "profile_id": "profile"}, [], tool_name="other")


@pytest.mark.parametrize("lost_response", [False, True])
@pytest.mark.asyncio
async def test_agent_dispatch_keeps_native_scope_in_process_and_resets_it(
    hass, profile_entry_factory, monkeypatch, lost_response,
):
    from custom_components.mcp_assist import agent as agent_module
    from custom_components.mcp_assist.agent import MCPAssistConversationEntity

    entry = profile_entry_factory(data={"control_home_assistant": True})
    agent = MCPAssistConversationEntity(hass, entry)
    loader = SimpleNamespace(requires_deferred_action_context=lambda name: name == "schedule_create")
    monkeypatch.setattr(agent, "_get_shared_tools_loader", lambda: loader)
    monkeypatch.setattr(agent, "_is_tool_configured_for_profile", lambda name: True)
    monkeypatch.setattr(agent, "_build_mcp_tool_call_context", AsyncMock(return_value={"user_id": "spoofed"}))
    context = Context(user_id="native-user")
    async def handle(params):
        scope = api.REQUEST_DEFERRED_ACTION.get()
        assert scope[0] is entry and scope[1:3] == ("schedule_create", "provider-call")
        assert scope[3] is context
        if lost_response:
            raise TimeoutError("fixture lost response after acceptance")
        return {"content": [{"type": "text", "text": "accepted"}]}
    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = SimpleNamespace(handle_tool_call=handle)
    http = Mock(side_effect=AssertionError("Deferred identity must not cross HTTP"))
    monkeypatch.setattr(agent_module.aiohttp, "ClientSession", http)
    token = api.REQUEST_TOOL_CALL_ID.set("provider-call")
    try:
        result = await agent._call_mcp_tool("schedule_create", {}, SimpleNamespace(context=context))
        if lost_response:
            import hashlib
            import json
            body = json.loads(result["content"][0]["text"])
            assert body["recovery_id"] == hashlib.sha256((context.id + ":provider-call").encode()).hexdigest()
            assert body["error"] == "deferred_result_uncertain"
        else:
            assert result["content"][0]["text"] == "accepted"
        assert api.REQUEST_DEFERRED_ACTION.get() is None
        http.assert_not_called()
    finally:
        api.REQUEST_TOOL_CALL_ID.reset(token)
