"""Privacy and request-cost boundaries for downloadable diagnostics."""

import json
from types import SimpleNamespace

import pytest

from custom_components.mcp_assist.diagnostics import (
    async_get_config_entry_diagnostics,
    build_assist_diagnostics,
)


@pytest.mark.asyncio
async def test_diagnostics_allowlist_omits_private_configuration(hass, profile_entry_factory):
    secret = "private-value-must-never-appear"
    entry = profile_entry_factory(data={
        "server_type": "openai", "model_profile": secret, "model_name": secret,
        "api_key": secret, "lmstudio_url": f"https://{secret}.example.invalid/{secret}",
        "system_prompt": secret, "system_prompt_mode": "custom",
        "technical_prompt": secret, "profile_name": secret,
        "openai_image_model_profile": secret,
    })
    agent = SimpleNamespace(
        entry=entry,
        resolved_model_profile={"model": secret, "revision": secret},
        resolved_image_model_profile=None,
        _cached_profile_mcp_tools=[{"name": secret, "inputSchema": {"secret": secret}}],
    )
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": agent}}
    result = await async_get_config_entry_diagnostics(hass, entry)
    assert secret not in json.dumps(result)
    assert entry.entry_id not in json.dumps(result)
    assert result["model_profile_resolution"] == "resolved"
    assert result["image_profile_resolution"] == "unresolved"
    assert result["prompts"]["system"] == {
        "mode": "custom", "configured_characters": len(secret),
    }
    assert result["cached_tools"]["count"] == 1
    assert result["cached_tools"]["schema_bytes"] > len(secret)


def test_unloaded_and_wrong_agent_identity_have_no_live_evidence(hass, profile_entry_factory):
    entry = profile_entry_factory(data={"server_type": "openai", "model_profile": "example-profile"})
    result = build_assist_diagnostics(hass, entry)
    assert result["agent_runtime_present"] is False
    assert result["model_profile_resolution"] == "runtime_unavailable"
    assert result["cached_tools"] == {"status": "not_built"}
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": SimpleNamespace(
        entry=object(), resolved_model_profile={"model": "example"},
        _cached_profile_mcp_tools=[],
    )}}
    assert build_assist_diagnostics(hass, entry) == result


def test_system_diagnostics_do_not_discover_or_build_tools(hass, system_entry_factory):
    entry = system_entry_factory(data={"mcp_bearer_token": "private-secret"})
    class Server:
        def __getattr__(self, name):
            raise AssertionError(f"Diagnostics must not call server {name}")
    hass.data["mcp_assist"] = {"shared_mcp_server": Server()}
    result = build_assist_diagnostics(hass, entry)
    assert result["entry_kind"] == "system"
    assert result["server_runtime_present"] is True
    assert result["bearer_auth_configured"] is True
    assert "private-secret" not in json.dumps(result)
    assert "cached_tools" not in result


@pytest.mark.parametrize("tools, expected", [
    (None, {"status": "not_built"}),
    ([], {"status": "cached", "count": 0, "schema_bytes": 2}),
    ([{"bad": object()}], {"status": "unavailable"}),
    ([{}] * 1001, {"status": "too_large", "count": 1001}),
])
def test_cached_tools_are_on_demand_and_bounded(hass, profile_entry_factory, tools, expected):
    entry = profile_entry_factory(options={"max_history": "secret", "max_iterations": True})
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": SimpleNamespace(
        entry=entry, _cached_profile_mcp_tools=tools,
    )}}
    result = build_assist_diagnostics(hass, entry)
    assert result["cached_tools"] == expected
    assert result["limits"]["history"] is None
    assert result["limits"]["iterations"] is None


def test_unknown_enum_values_are_not_exported(hass, profile_entry_factory):
    entry = profile_entry_factory(options={
        "server_type": "private-secret", "context_mode": "private-secret",
        "system_prompt_mode": "private-secret",
    })
    result = build_assist_diagnostics(hass, entry)
    assert "private-secret" not in json.dumps(result)
    assert result["provider"] == "unknown"


def test_non_openai_provider_ignores_inactive_profile_reference(hass, profile_entry_factory):
    entry = profile_entry_factory(data={
        "server_type": "ollama", "model_name": "example-model", "model_profile": "inactive",
    })
    result = build_assist_diagnostics(hass, entry)
    assert result["model_selection"] == "explicit"
    assert result["model_profile_resolution"] == "not_applicable"
