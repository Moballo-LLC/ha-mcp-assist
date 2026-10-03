"""Privacy and request-cost boundaries for downloadable diagnostics."""

import json
from types import SimpleNamespace

import pytest

from custom_components.mcp_assist import const
from custom_components.mcp_assist import diagnostics as diagnostics_module
from custom_components.mcp_assist.localization import get_language_instruction
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


def test_openclaw_credential_is_reported_as_boolean_without_exposing_token(
    hass, profile_entry_factory,
):
    secret = "synthetic-openclaw-token"
    entry = profile_entry_factory(data={
        "server_type": "openclaw", "openclaw_token": secret, "api_key": "inactive-key",
    })

    result = build_assist_diagnostics(hass, entry)

    assert result["credential_configured"] is True
    assert secret not in json.dumps(result)
    entry_without_token = profile_entry_factory(data={
        "server_type": "openclaw", "api_key": "inactive-key",
    })
    assert build_assist_diagnostics(hass, entry_without_token)["credential_configured"] is False


def test_cached_tool_schema_with_lone_surrogate_reports_size_unavailable(
    hass, profile_entry_factory,
):
    entry = profile_entry_factory()
    tools = [{"name": "bad-\ud800-schema"}]
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": SimpleNamespace(
        entry=entry, _cached_profile_mcp_tools=tools,
    )}}

    result = build_assist_diagnostics(hass, entry)

    assert result["cached_tools"] == {"status": "unavailable"}
    assert "bad-" not in json.dumps(result)

    tools[:] = [{"name": "valid schema"}]
    result = build_assist_diagnostics(hass, entry)
    assert result["cached_tools"]["status"] == "cached"
    assert result["cached_tools"]["schema_bytes"] == len(
        json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode()
    )


@pytest.mark.parametrize("shape", ["large_string", "many_strings", "wide", "deep", "large_number"])
def test_oversized_cached_schemas_are_rejected_before_serialization(
    hass, profile_entry_factory, monkeypatch, shape,
):
    if shape == "large_string":
        schema = {"description": "private-description" * 100_000}
    elif shape == "many_strings":
        schema = {"enum": ["private-value" * 1000] * 100}
    elif shape == "wide":
        schema = {"enum": [None] * 100_000}
    elif shape == "deep":
        schema = {}
        for _ in range(100):
            schema = {"nested": schema}
    else:
        schema = {"default": 1 << 1_000_000}
    entry = profile_entry_factory()
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": SimpleNamespace(
        entry=entry, _cached_profile_mcp_tools=[{"inputSchema": schema}],
    )}}
    with monkeypatch.context() as context:
        context.setattr(diagnostics_module.json, "dumps", lambda *args, **kwargs: (
            pytest.fail("Oversized cache must be rejected before serialization")
        ))
        result = build_assist_diagnostics(hass, entry)
    assert result["cached_tools"] == {"status": "too_large", "count": 1}
    assert "private-" not in json.dumps(result)


def test_small_cached_schema_size_matches_compact_utf8_json(hass, profile_entry_factory):
    tools = [{"name": "example", "inputSchema": {"enum": [
        "snowman \u2603", "\\quote\"\n", None, True, False, 3.14, -12345,
    ]}}]
    entry = profile_entry_factory()
    hass.data["mcp_assist"] = {entry.entry_id: {"agent": SimpleNamespace(
        entry=entry, _cached_profile_mcp_tools=tools,
    )}}
    result = build_assist_diagnostics(hass, entry)
    assert result["cached_tools"] == {
        "status": "cached", "count": 1,
        "schema_bytes": len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode()),
    }


@pytest.mark.parametrize("label", ["system", "technical"])
@pytest.mark.parametrize("storage", ["data", "options"])
@pytest.mark.parametrize("mode,prompt,expected", [
    (None, "synthetic-custom-prompt", "custom"),
    ("invalid-mode", "synthetic-custom-prompt", "custom"),
    ("default", "synthetic-custom-prompt", "default"),
    ("custom", None, "custom"),
    (None, None, "default"),
    (None, "", "default"),
    (None, "builtin", "default"),
])
def test_prompt_modes_match_legacy_runtime_inference(
    hass, profile_entry_factory, label, storage, mode, prompt, expected,
):
    default_prompt = (
        get_language_instruction(hass.config.language) or const.DEFAULT_SYSTEM_PROMPT
        if label == "system" else const.DEFAULT_TECHNICAL_PROMPT
    )
    values = {f"{label}_prompt": default_prompt if prompt == "builtin" else prompt}
    if mode is not None:
        values[f"{label}_prompt_mode"] = mode
    entry = profile_entry_factory(**{storage: values})
    result = build_assist_diagnostics(hass, entry)
    assert result["prompts"][label]["mode"] == expected
    assert "synthetic-custom-prompt" not in json.dumps(result)
    assert "invalid-mode" not in json.dumps(result)


def test_prompt_mode_options_override_data_and_infer_localized_default(
    hass, profile_entry_factory,
):
    hass.config.language = "de"
    localized = get_language_instruction("de")
    entry = profile_entry_factory(
        data={"system_prompt": "synthetic-custom-prompt", "system_prompt_mode": "custom"},
        options={"system_prompt": localized, "system_prompt_mode": None},
    )
    result = build_assist_diagnostics(hass, entry)
    assert result["prompts"]["system"]["mode"] == "default"
    assert localized not in json.dumps(result)
