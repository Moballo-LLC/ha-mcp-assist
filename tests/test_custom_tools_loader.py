"""Tests for custom tool loading."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from custom_components.mcp_assist.const import (
    CONF_BRAVE_API_KEY,
    CONF_ENABLE_CALCULATOR_TOOLS,
    CONF_ENABLE_CUSTOM_TOOLS,
    CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS,
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_ENABLE_MEMORY_TOOLS,
    CONF_GOOGLE_MAPS_API_KEY,
    CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT,
    CONF_ENABLE_RECORDER_TOOLS,
    CONF_ENABLE_RESPONSE_SERVICE_TOOLS,
    CONF_ENABLE_UNIT_CONVERSION_TOOLS,
    CONF_ENABLE_WEB_SEARCH,
    CONF_ENABLE_WEATHER_FORECAST_TOOL,
    CONF_LLM_API_ALLOWLIST,
    CONF_SEARCH_PROVIDER,
    CONF_SEARXNG_URL,
    CUSTOM_TOOL_MANIFEST_FILENAME,
    CUSTOM_TOOL_SETTINGS_DIRECTORY,
    CUSTOM_TOOLS_DIRECTORY,
    DOMAIN,
)
from custom_components.mcp_assist.tools import CustomToolsLoader
from custom_components.mcp_assist.tools.builtin_catalog import (
    load_builtin_tool_toggle_specs,
)
from custom_components.mcp_assist.tools.external_loader import ExternalCustomToolLoader

TOOLS_ROOT = Path("custom_components/mcp_assist/tools")
LEGACY_CUSTOM_TOOLS_ROOT = Path("custom_components/mcp_assist/custom_tools")
LEGACY_SERVER_TOOLS_ROOT = Path("custom_components/mcp_assist/server_tools")
CUSTOM_TOOL_FRAMEWORK_FILES = {
    "__init__.py",
    "builtin_catalog.py",
    "external_loader.py",
    "schema_utils.py",
    "tool_runtime.py",
}
LEGACY_ROOT_TOOL_MODULES = {
    "brave_search",
    "calculator",
    "duckduckgo_search",
    "google_maps",
    "llm_api_bridge",
    "memory",
    "music_assistant",
    "read_url",
    "recorder",
    "response_services",
    "searxng_search",
    "weather",
    "wikipedia_search",
}


class _StubTool:
    """Simple stub custom tool implementation."""

    def __init__(self, hass, *args) -> None:
        self.hass = hass
        self.args = args

    async def initialize(self) -> None:
        return None

    def get_tool_definitions(self):
        return [
            {
                "name": self.__class__.__name__,
                "description": "Stub tool definition.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

    def handles_tool(self, tool_name: str) -> bool:
        return tool_name == self.__class__.__name__

    async def handle_call(self, tool_name, arguments):
        return {"content": [{"type": "text", "text": tool_name}]}


def test_builtin_tool_implementations_live_in_manifest_package_dirs() -> None:
    """Built-in tools should mirror the external custom-tool package layout."""
    assert not LEGACY_CUSTOM_TOOLS_ROOT.exists()
    assert not LEGACY_SERVER_TOOLS_ROOT.exists()
    assert {
        path.name for path in TOOLS_ROOT.glob("*.py")
    } == CUSTOM_TOOL_FRAMEWORK_FILES

    package_dirs = [
        path
        for path in sorted((TOOLS_ROOT / "packages").iterdir())
        if path.is_dir() and not path.name.startswith((".", "__"))
    ]
    assert package_dirs
    for package_dir in package_dirs:
        assert (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).is_file()
        assert (package_dir / "tool.py").is_file()

    legacy_imports = {
        f"custom_components.mcp_assist.tools.{module}"
        for module in LEGACY_ROOT_TOOL_MODULES
    }
    for path in TOOLS_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(legacy_import in text for legacy_import in legacy_imports), path


@pytest.mark.asyncio
async def test_all_builtin_tool_packages_load_and_match_manifest_tool_names(
    hass,
    system_entry_factory,
) -> None:
    """Every built-in package should load through the public custom-tool format."""
    system_entry_factory(
        data={
            CONF_BRAVE_API_KEY: "test-key",
            CONF_SEARCH_PROVIDER: "brave",
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
        }
    )
    specs_by_package_id = {
        spec.package_id: spec for spec in load_builtin_tool_toggle_specs()
    }
    loader = ExternalCustomToolLoader(
        hass,
        tools_root=TOOLS_ROOT / "packages",
        module_namespace="mcp_assist_builtin_tool_contract_tests",
        require_tool_name_prefix=False,
        package_log_label="built-in test tool package",
        prompt_package_label="Built-in test tool package",
    )
    loaded_packages = []

    try:
        loaded_packages = await loader.load(
            allowed_tool_ids=set(specs_by_package_id)
        )

        assert loader.last_load_errors == []
        assert {package.manifest.tool_id for package in loaded_packages} == set(
            specs_by_package_id
        )
        for package in loaded_packages:
            expected_tool_names = specs_by_package_id[package.manifest.tool_id].tool_names
            assert package.tool_names == expected_tool_names
            assert [tool["name"] for tool in package.tool_definitions] == list(
                expected_tool_names
            )
            for tool_name in expected_tool_names:
                assert package.instance.handles_tool(tool_name)
    finally:
        for package in loaded_packages:
            await package.instance.async_shutdown()


def _write_external_tool_package(
    config_root: Path,
    *,
    tool_id: str = "sample_tool",
    tool_name: str | None = None,
    include_prompt_file: bool = True,
    directory_name: str | None = None,
    manifest_filename: str = CUSTOM_TOOL_MANIFEST_FILENAME,
) -> Path:
    """Write a valid external custom tool package to the temp config root."""
    package_dir = config_root / CUSTOM_TOOLS_DIRECTORY / (directory_name or tool_id)
    package_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "id": tool_id,
        "name": "Sample Tool",
        "description": "Example external tool package for testing.",
        "version": "1.0.0",
        "entrypoint": "tool:SampleTool",
        "capabilities": ["Provides a custom sample status lookup."],
    }
    if include_prompt_file:
        manifest["prompt_append_file"] = "prompt.md"

    (package_dir / manifest_filename).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    if include_prompt_file:
        (package_dir / "prompt.md").write_text(
            "Use sample_tool_status when the user asks for the sample system status.",
            encoding="utf-8",
        )

    resolved_tool_name = tool_name or f"{tool_id}_status"
    (package_dir / "tool.py").write_text(
        f'''from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class SampleTool(MCPAssistExternalTool):
    async def initialize(self) -> None:
        return None

    def get_settings_schema(self):
        return {{
            "type": "object",
            "properties": {{
                "status_text": {{
                    "type": "string",
                    "default": "sample ok",
                }}
            }},
        }}

    def get_tool_definitions(self):
        return [{{
            "name": "{resolved_tool_name}",
            "description": "Return a sample status.",
            "llmDescription": "Return sample status.",
            "inputSchema": {{"type": "object", "properties": {{}}}},
            "keywords": ["sample", "status"],
            "example_queries": ["What's the sample status?"],
            "preferred_when": "Use for sample package status questions.",
            "returns": "A short sample status string.",
        }}]

    async def handle_call(self, tool_name, arguments):
        settings = self.get_settings()
        return {{
            "content": [{{"type": "text", "text": settings.get("status_text", "sample ok")}}],
            "isError": False,
        }}

    def get_prompt_instructions(self) -> str:
        return "Prefer {resolved_tool_name} for sample-status questions."
''',
        encoding="utf-8",
    )
    return package_dir


@pytest.mark.asyncio
async def test_initialize_skips_calculator_when_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Calculator tools should not be loaded when the shared toggle is disabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(data={CONF_ENABLE_CALCULATOR_TOOLS: False})
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "calculator" not in loader.tools


@pytest.mark.asyncio
async def test_initialize_loads_calculator_bundle_when_only_unit_conversion_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Unit conversion should load its own package without loading math tools."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "calculator" not in loader.tools
    assert "unit_conversion" in loader.tools


@pytest.mark.asyncio
async def test_initialize_loads_music_assistant_bundle_when_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Music Assistant should load as a built-in packaged tool when enabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_MUSIC_ASSISTANT_SUPPORT: True,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "music_assistant" in loader.tools
    tool_names = {tool["name"] for tool in loader.get_tool_definitions()}
    assert "play_music_assistant" in tool_names
    assert "get_music_assistant_queue" in tool_names
    assert "control_music_assistant_player" in tool_names
    assert "transfer_music_assistant_queue" in tool_names
    music_assistant_spec = loader.get_builtin_toggle_spec("play_music_assistant")
    assert music_assistant_spec is not None
    assert music_assistant_spec.package_id == "music_assistant"
    assert (
        loader.get_builtin_toggle_spec("control_music_assistant_player")
        == music_assistant_spec
    )


@pytest.mark.asyncio
async def test_initialize_loads_llm_api_bridge_bundle_when_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """LLM API Bridge should load as a built-in packaged tool when enabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_LLM_API_BRIDGE: True,
            CONF_LLM_API_ALLOWLIST: "llm_intents",
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "llm_api_bridge" in loader.tools
    tool_names = {tool["name"] for tool in loader.get_tool_definitions()}
    assert {
        "list_llm_apis",
        "list_llm_api_tools",
        "call_llm_api_tool",
        "get_llm_api_prompt",
    } <= tool_names
    llm_api_bridge_spec = loader.get_builtin_toggle_spec("list_llm_apis")
    assert llm_api_bridge_spec is not None
    assert llm_api_bridge_spec.package_id == "llm_api_bridge"
    assert "list_llm_apis" in loader.get_builtin_prompt_instructions()


@pytest.mark.asyncio
async def test_initialize_loads_memory_bundle_when_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Memory tools should load as a built-in packaged tool when enabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_MEMORY_TOOLS: True,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert "memory" in loader.tools
    tool_names = {tool["name"] for tool in loader.get_tool_definitions()}
    assert {
        "list_memory_categories",
        "remember_memory",
        "recall_memories",
        "forget_memory",
    } <= tool_names
    memory_spec = loader.get_builtin_toggle_spec("remember_memory")
    assert memory_spec is not None
    assert memory_spec.package_id == "memory"


@pytest.mark.asyncio
async def test_initialize_loads_domain_tool_bundles_when_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Recorder, response-service, and weather tools should load as built-in packages."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_RECORDER_TOOLS: True,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: True,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: False,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)

    await loader.initialize()

    assert {"recorder", "response_service", "weather_forecast"}.issubset(
        loader.tools
    )
    tool_names = {tool["name"] for tool in loader.get_tool_definitions()}
    assert "get_entity_history" in tool_names
    assert "get_calendar_events" in tool_names
    assert "get_weather_forecast" in tool_names
    assert loader.get_builtin_toggle_spec("get_entity_history").package_id == "recorder"
    assert (
        loader.get_builtin_toggle_spec("get_calendar_events").package_id
        == "response_service"
    )
    assert (
        loader.get_builtin_toggle_spec("get_weather_forecast").package_id
        == "weather_forecast"
    )
    builtin_prompt = loader.get_builtin_prompt_instructions()
    assert "point-in-time questions" in builtin_prompt
    assert "native Home Assistant service responses over web search" in builtin_prompt
    assert "chooses a supported forecast type" in builtin_prompt


@pytest.mark.asyncio
async def test_builtin_tool_packages_use_executor_for_manifest_reads(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Built-in package manifests should load through the shared package loader."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    executor_calls: list[str] = []
    original_async_add_executor_job = hass.async_add_executor_job

    async def _track_executor_job(func, *args):
        executor_calls.append(getattr(func, "__name__", repr(func)))
        return await original_async_add_executor_job(func, *args)

    monkeypatch.setattr(hass, "async_add_executor_job", _track_executor_job)

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "calculator" in loader.tools
    assert "_discover_package_dirs" in executor_calls
    assert "_load_manifest_from_disk" in executor_calls
    # Importing the package entrypoint runs its top-level code + disk reads,
    # so it must also be dispatched off the event loop.
    assert "_instantiate_tool" in executor_calls


@pytest.mark.asyncio
async def test_package_diagnostics_include_loaded_builtin_packages(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Manifest-package diagnostics should include built-in package metadata."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    diagnostics = loader.get_package_diagnostics()

    assert diagnostics["built_in_packages"][0]["id"] == "calculator"
    assert diagnostics["built_in_prompt_instructions"].startswith(
        "## Optional Built-In Tool Packages"
    )
    assert diagnostics["external_custom_tools_enabled"] is False
    assert loader.get_loaded_builtin_tool_info()[0]["tool_names"]
    builtin_prompt = loader.get_builtin_prompt_instructions()
    assert "Calculator" in builtin_prompt
    assert "Capabilities:" not in builtin_prompt


@pytest.mark.asyncio
async def test_builtin_tool_registry_exposes_package_classification_and_toggle_metadata(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Built-in packaged tools should register as built-in tools with toggle metadata."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_UNIT_CONVERSION_TOOLS: True,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    calculator_spec = loader.get_builtin_toggle_spec("add")
    unit_conversion_spec = loader.get_builtin_toggle_spec("convert_unit")

    assert loader.is_custom_tool("add") is True
    assert loader.is_builtin_custom_tool("add") is True
    assert loader.is_external_custom_tool("add") is False
    assert calculator_spec is not None
    assert calculator_spec.package_id == "calculator"
    assert unit_conversion_spec is not None
    assert unit_conversion_spec.package_id == "unit_conversion"
    assert loader.get_builtin_toggle_specs()
    assert loader.get_builtin_toggle_spec("missing_tool") is None


@pytest.mark.asyncio
async def test_initialize_loads_google_maps_when_enabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Google Maps should load as its own optional built-in package."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            "enable_google_maps_tools": True,
            CONF_GOOGLE_MAPS_API_KEY: "test-key",
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "google_maps" in loader.tools
    tool_names = {definition["name"] for definition in loader.get_tool_definitions()}
    assert {
        "search_google_places",
        "get_google_place_details",
        "get_google_route",
    } <= tool_names
    google_maps_spec = loader.get_builtin_toggle_spec("get_google_route")
    assert google_maps_spec is not None
    assert google_maps_spec.package_id == "google_maps"
    google_maps_package = next(
        package
        for package in loader.builtin_packages
        if package.manifest.tool_id == "google_maps"
    )
    assert len(google_maps_package.prompt_instructions) <= 420
    assert "home-location tool sharing is enabled" in google_maps_package.prompt_instructions
    assert "routes starting from Home Assistant home" in google_maps_package.prompt_instructions
    assert "ask/pass origin" in google_maps_package.prompt_instructions
    assert "arrival_time is transit-only" in google_maps_package.prompt_instructions


@pytest.mark.asyncio
async def test_initialize_loads_wikipedia_search_when_enabled(
    hass,
    profile_entry_factory,
    system_entry_factory,
) -> None:
    """Wikipedia search should load as its own optional built-in package."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            "enable_wikipedia_search_tool": True,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "wikipedia_search" in loader.tools
    assert any(
        definition["name"] == "search_wikipedia"
        for definition in loader.get_tool_definitions()
    )
    wikipedia_spec = loader.get_builtin_toggle_spec("search_wikipedia")
    assert wikipedia_spec is not None
    assert wikipedia_spec.package_id == "wikipedia_search"


@pytest.mark.asyncio
async def test_initialize_loads_search_and_read_url_for_brave(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Search-enabled setups should load the provider and read_url tools."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "brave",
            CONF_BRAVE_API_KEY: "secret",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
        }
    )

    brave_module = types.SimpleNamespace(BraveSearchTool=type("BraveSearchTool", (_StubTool,), {}))
    read_url_module = types.SimpleNamespace(ReadUrlTool=type("ReadUrlTool", (_StubTool,), {}))
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.search.brave_search",
        brave_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.read_url.read_url",
        read_url_module,
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert {"search", "read_url"}.issubset(loader.tools)
    assert "calculator" not in loader.tools


@pytest.mark.asyncio
async def test_initialize_loads_search_and_read_url_for_duckduckgo(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """DuckDuckGo should load through the same built-in package pathway."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": True,
            CONF_SEARCH_PROVIDER: "duckduckgo",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    duckduckgo_module = types.SimpleNamespace(
        DuckDuckGoSearchTool=type("DuckDuckGoSearchTool", (_StubTool,), {})
    )
    read_url_module = types.SimpleNamespace(ReadUrlTool=type("ReadUrlTool", (_StubTool,), {}))
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.search.duckduckgo_search",
        duckduckgo_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.read_url.read_url",
        read_url_module,
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert {"search", "read_url"}.issubset(loader.tools)
    assert "calculator" not in loader.tools


@pytest.mark.asyncio
async def test_initialize_loads_search_and_read_url_for_searxng(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """SearXNG should load through the same built-in package pathway."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": True,
            CONF_SEARCH_PROVIDER: "searxng",
            CONF_SEARXNG_URL: "http://search.local",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    searxng_module = types.SimpleNamespace(
        SearXNGSearchTool=type("SearXNGSearchTool", (_StubTool,), {})
    )
    read_url_module = types.SimpleNamespace(ReadUrlTool=type("ReadUrlTool", (_StubTool,), {}))
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.search.searxng_search",
        searxng_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.read_url.read_url",
        read_url_module,
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert {"search", "read_url"}.issubset(loader.tools)
    assert loader.tools["search"]._delegate.args == ("http://search.local",)
    assert "calculator" not in loader.tools


@pytest.mark.asyncio
async def test_initialize_can_enable_search_without_read_url(
    hass, profile_entry_factory, system_entry_factory, monkeypatch
) -> None:
    """Built-in packaged tools should be independently toggleable."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            "enable_search_tool": True,
            "enable_read_url_tool": False,
            CONF_SEARCH_PROVIDER: "brave",
            CONF_BRAVE_API_KEY: "secret",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    brave_module = types.SimpleNamespace(
        BraveSearchTool=type("BraveSearchTool", (_StubTool,), {})
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.search.brave_search",
        brave_module,
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "search" in loader.tools
    assert "read_url" not in loader.tools


@pytest.mark.asyncio
async def test_initialize_skips_search_when_web_search_disabled(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Search tools should not load when the web-search tool family is disabled."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_WEB_SEARCH: False,
            CONF_SEARCH_PROVIDER: "brave",
            CONF_BRAVE_API_KEY: "secret",
            CONF_ENABLE_CALCULATOR_TOOLS: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "search" not in loader.tools
    assert "read_url" not in loader.tools


def test_get_search_provider_keeps_backward_compatibility(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Legacy enable_custom_tools should still imply Brave search when search_provider is unset."""
    profile_entry = profile_entry_factory(options={CONF_ENABLE_CUSTOM_TOOLS: True})
    system_entry_factory(data={CONF_SEARCH_PROVIDER: None}, options={})
    loader = CustomToolsLoader(hass, profile_entry)

    assert loader._get_search_provider() == "brave"


@pytest.mark.asyncio
async def test_external_custom_tools_are_disabled_by_default(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """User-defined custom tools should not load unless explicitly enabled."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.external_tools == []
    assert loader.get_external_prompt_instructions() == ""
    assert "sample_tool" not in loader.tools


@pytest.mark.asyncio
async def test_external_custom_tool_validation_does_not_mutate_live_registry(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Validation should inspect packages without exposing them as live tools."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    diagnostics = await loader.validate_external_tool_packages()

    assert diagnostics["enabled"] is False
    assert diagnostics["valid"] is True
    assert diagnostics["load_errors"] == []
    assert diagnostics["loaded_tools"][0]["id"] == "sample_tool"
    assert loader.external_tools == []
    assert "sample_tool" not in loader.tools
    assert "sample_tool_status" not in {
        tool_definition["name"] for tool_definition in loader.get_tool_definitions()
    }


@pytest.mark.asyncio
async def test_external_custom_tool_package_loads_and_handles_calls(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Enabled external custom tools should load from the HA config directory."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    tool_names = {
        tool_definition["name"] for tool_definition in loader.get_tool_definitions()
    }
    assert "sample_tool_status" in tool_names
    external_prompt = loader.get_external_prompt_instructions()
    assert "External Custom Tools" in external_prompt
    assert "sample_tool_status" in external_prompt
    assert "Capabilities:" not in external_prompt
    assert "Custom tool package" not in external_prompt
    assert loader.get_loaded_external_tool_info() == [
        {
            "id": "sample_tool",
            "name": "Sample Tool",
            "version": "1.0.0",
            "description": "Example external tool package for testing.",
            "tool_names": ["sample_tool_status"],
            "capabilities": ["Provides a custom sample status lookup."],
        }
    ]

    result = await loader.handle_tool_call("sample_tool_status", {})
    assert result["isError"] is False
    assert result["content"][0]["text"] == "sample ok"

    tool_definition = next(
        tool
        for tool in loader.get_tool_definitions()
        if tool["name"] == "sample_tool_status"
    )
    assert tool_definition["routingHints"]["keywords"] == ["sample", "status"]
    assert tool_definition["routingHints"]["preferred_when"] == (
        "Use for sample package status questions."
    )
    assert tool_definition["llmDescription"] == "Return sample status."


@pytest.mark.asyncio
async def test_cache_signature_includes_tool_surface_and_prompt_instructions(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Cache signatures should change with the loaded tool definitions and prompts."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    tool_definitions, builtin_prompt, external_prompt = loader.get_cache_signature()

    assert any('"name":"add"' in definition for definition in tool_definitions)
    assert any(
        '"name":"sample_tool_status"' in definition for definition in tool_definitions
    )
    assert "Optional Built-In Tool Packages" in builtin_prompt
    assert "Calculator" in builtin_prompt
    assert "External Custom Tools" in external_prompt
    assert "sample_tool_status" in external_prompt
    assert len(external_prompt) <= 2200


@pytest.mark.asyncio
async def test_cache_signature_is_precomputed_after_registry_refresh(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Cache-signature reads should not repeatedly serialize every tool definition."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    expected_signature = loader.get_cache_signature()

    def fail_json_dumps(*args, **kwargs):
        raise AssertionError("cache signature should be precomputed")

    monkeypatch.setattr(
        "custom_components.mcp_assist.tools.json.dumps",
        fail_json_dumps,
    )

    assert loader.get_cache_signature() == expected_signature


@pytest.mark.asyncio
async def test_external_tool_package_id_cannot_conflict_with_builtin_package(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External package ids should not collide with built-in package ids."""
    _write_external_tool_package(
        tmp_path,
        tool_id="search",
        tool_name="search_status",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: True,
            CONF_SEARCH_PROVIDER: "brave",
            CONF_BRAVE_API_KEY: "secret",
        }
    )

    brave_module = types.SimpleNamespace(
        BraveSearchTool=type("BraveSearchTool", (_StubTool,), {})
    )
    read_url_module = types.SimpleNamespace(
        ReadUrlTool=type("ReadUrlTool", (_StubTool,), {})
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.search.brave_search",
        brave_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "custom_components.mcp_assist.tools.packages.read_url.read_url",
        read_url_module,
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "search" in loader.tools
    assert loader.get_loaded_external_tool_info() == []


@pytest.mark.asyncio
async def test_external_custom_tool_package_uses_executor_for_manifest_and_prompt_reads(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External package file reads should be delegated to the executor."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    executor_calls: list[str] = []
    original_async_add_executor_job = hass.async_add_executor_job

    async def _track_executor_job(func, *args):
        executor_calls.append(getattr(func, "__name__", repr(func)))
        return await original_async_add_executor_job(func, *args)

    monkeypatch.setattr(hass, "async_add_executor_job", _track_executor_job)

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert "_discover_package_dirs" in executor_calls
    assert "_load_manifest_from_disk" in executor_calls
    assert "_read_prompt_append_file_from_disk" in executor_calls


@pytest.mark.asyncio
async def test_invalid_external_tool_package_is_skipped_without_crashing(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Invalid external packages should be skipped while valid loading continues."""
    _write_external_tool_package(tmp_path, directory_name="wrong_dir_name")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
            CONF_ENABLE_RECORDER_TOOLS: False,
            CONF_ENABLE_RESPONSE_SERVICE_TOOLS: False,
            CONF_ENABLE_WEATHER_FORECAST_TOOL: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.external_tools == []
    assert loader.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_external_tool_name_must_be_namespaced(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External tool names must be prefixed with the manifest id."""
    _write_external_tool_package(tmp_path, tool_name="status")
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.external_tools == []


@pytest.mark.asyncio
async def test_legacy_manifest_filename_is_not_supported(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Only the canonical mcp_tool.json manifest filename should be accepted."""
    _write_external_tool_package(
        tmp_path,
        manifest_filename="manifest.json",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    assert loader.get_loaded_external_tool_info() == []


@pytest.mark.asyncio
async def test_external_custom_tool_uses_shared_and_profile_settings(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External tools should receive merged shared and per-profile settings."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    settings_root = tmp_path / CUSTOM_TOOL_SETTINGS_DIRECTORY
    settings_root.mkdir(parents=True, exist_ok=True)
    (settings_root / "sample_tool.json").write_text(
        json.dumps({"status_text": "shared status"}),
        encoding="utf-8",
    )
    profile_settings_dir = settings_root / "profiles" / profile_entry.entry_id
    profile_settings_dir.mkdir(parents=True, exist_ok=True)
    (profile_settings_dir / "sample_tool.json").write_text(
        json.dumps({"status_text": "profile status"}),
        encoding="utf-8",
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    shared_result = await loader.handle_tool_call("sample_tool_status", {})
    profile_result = await loader.handle_tool_call(
        "sample_tool_status",
        {},
        context={"profile_entry_id": profile_entry.entry_id},
    )

    assert shared_result["content"][0]["text"] == "shared status"
    assert profile_result["content"][0]["text"] == "profile status"


@pytest.mark.asyncio
async def test_external_custom_tool_profile_settings_use_executor(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Per-profile settings should be loaded through the executor on tool calls."""
    _write_external_tool_package(tmp_path)
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    settings_root = tmp_path / CUSTOM_TOOL_SETTINGS_DIRECTORY
    settings_root.mkdir(parents=True, exist_ok=True)
    profile_settings_dir = settings_root / "profiles" / profile_entry.entry_id
    profile_settings_dir.mkdir(parents=True, exist_ok=True)
    (profile_settings_dir / "sample_tool.json").write_text(
        json.dumps({"status_text": "profile status"}),
        encoding="utf-8",
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    executor_calls: list[str] = []
    original_async_add_executor_job = hass.async_add_executor_job

    async def _track_executor_job(func, *args):
        executor_calls.append(getattr(func, "__name__", repr(func)))
        return await original_async_add_executor_job(func, *args)

    monkeypatch.setattr(hass, "async_add_executor_job", _track_executor_job)

    result = await loader.handle_tool_call(
        "sample_tool_status",
        {},
        context={"profile_entry_id": profile_entry.entry_id},
    )

    assert result["content"][0]["text"] == "profile status"
    assert "_load_settings_file_with_status" in executor_calls


@pytest.mark.asyncio
async def test_external_custom_tool_argument_validation_returns_mcp_error(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Invalid external-tool arguments should be rejected before tool execution."""
    package_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "typed_tool"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "typed_tool",
                "name": "Typed Tool",
                "description": "Typed external tool for schema validation tests.",
                "version": "1.0.0",
                "entrypoint": "tool:TypedTool",
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "tool.py").write_text(
        """from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class TypedTool(MCPAssistExternalTool):
    def get_tool_definitions(self):
        return [{
            "name": "typed_tool_echo",
            "description": "Echo an integer value.",
            "inputSchema": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
        }]

    async def handle_call(self, tool_name, arguments):
        return {
            "content": [{"type": "text", "text": str(arguments["count"])}],
            "isError": False,
        }
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    result = await loader.handle_tool_call("typed_tool_echo", {"count": "nope"})

    assert result["isError"] is True
    assert "typed_tool_echo arguments.count must be an integer" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_external_custom_tool_runtime_errors_return_mcp_error_payload(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Manifest-based tool failures should return MCP errors instead of bubbling."""
    package_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "broken_tool"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "broken_tool",
                "name": "Broken Tool",
                "description": "External tool that raises while handling a call.",
                "version": "1.0.0",
                "entrypoint": "tool:BrokenTool",
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "tool.py").write_text(
        """from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class BrokenTool(MCPAssistExternalTool):
    def get_tool_definitions(self):
        return [{
            "name": "broken_tool_status",
            "description": "Raise an error.",
            "inputSchema": {"type": "object", "properties": {}},
        }]

    async def handle_call(self, tool_name, arguments):
        raise RuntimeError("tool execution failed")
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    result = await loader.handle_tool_call("broken_tool_status", {})

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution failed"


@pytest.mark.asyncio
async def test_external_custom_tools_can_import_shared_helpers(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """Packages should be able to import helper modules from __shared__ safely."""
    shared_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "__shared__"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "helpers.py").write_text(
        "def message():\n    return 'shared helper ok'\n",
        encoding="utf-8",
    )

    package_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "shared_tool"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "shared_tool",
                "name": "Shared Tool",
                "description": "External tool that uses a shared helper.",
                "version": "1.0.0",
                "entrypoint": "tool:SharedTool",
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "tool.py").write_text(
        """from custom_components.mcp_assist.custom_tool_api import (
    MCPAssistExternalTool,
    load_external_shared_module,
)

helpers = load_external_shared_module(__file__, "helpers")


class SharedTool(MCPAssistExternalTool):
    def get_tool_definitions(self):
        return [{
            "name": "shared_tool_status",
            "description": "Return shared helper output.",
            "inputSchema": {"type": "object", "properties": {}},
        }]

    async def handle_call(self, tool_name, arguments):
        return {
            "content": [{"type": "text", "text": helpers.message()}],
            "isError": False,
        }
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    result = await loader.handle_tool_call("shared_tool_status", {})

    assert result["isError"] is False
    assert result["content"][0]["text"] == "shared helper ok"

    (shared_dir / "helpers.py").write_text(
        "def message():\n    return 'shared helper updated'\n",
        encoding="utf-8",
    )

    await loader.reload_tool_packages()
    reloaded_result = await loader.handle_tool_call("shared_tool_status", {})

    assert reloaded_result["isError"] is False
    assert reloaded_result["content"][0]["text"] == "shared helper updated"


@pytest.mark.asyncio
async def test_reload_tool_packages_refreshes_builtin_toggle_specs(
    hass, profile_entry_factory, system_entry_factory
) -> None:
    """Reloading package tools should refresh built-in toggle metadata first."""
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_CALCULATOR_TOOLS: True,
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )
    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    loader._load_builtin_toggle_specs = AsyncMock()
    loader._initialize_builtin_tools = AsyncMock()
    loader._initialize_external_tools = AsyncMock()

    await loader.reload_tool_packages()

    loader._load_builtin_toggle_specs.assert_awaited_once()
    loader._initialize_builtin_tools.assert_awaited_once()
    loader._initialize_external_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_external_tool_can_call_core_mcp_tool_with_profile_context(
    hass, profile_entry_factory, system_entry_factory, monkeypatch, tmp_path
) -> None:
    """External tools should be able to invoke core MCP tools through the shared server."""
    package_dir = tmp_path / CUSTOM_TOOLS_DIRECTORY / "bridge_tool"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / CUSTOM_TOOL_MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "bridge_tool",
                "name": "Bridge Tool",
                "description": "External tool that bridges to a core MCP tool.",
                "version": "1.0.0",
                "entrypoint": "tool:BridgeTool",
            }
        ),
        encoding="utf-8",
    )
    (package_dir / "tool.py").write_text(
        """from custom_components.mcp_assist.custom_tool_api import MCPAssistExternalTool


class BridgeTool(MCPAssistExternalTool):
    def get_tool_definitions(self):
        return [{
            "name": "bridge_tool_status",
            "description": "Call a core MCP tool.",
            "inputSchema": {"type": "object", "properties": {}},
        }]

    async def handle_call(self, tool_name, arguments):
        del tool_name, arguments
        return await self.call_mcp_tool("sample_core_tool", {"value": "ok"})
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hass.config,
        "path",
        lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    profile_entry = profile_entry_factory()
    system_entry_factory(
        data={
            CONF_ENABLE_EXTERNAL_CUSTOM_TOOLS: True,
            CONF_ENABLE_CALCULATOR_TOOLS: False,
            CONF_ENABLE_WEB_SEARCH: False,
        }
    )

    captured: dict[str, object] = {}

    class _StubServer:
        async def handle_tool_call(self, params):
            captured.update(params)
            return {
                "content": [{"type": "text", "text": "core ok"}],
                "isError": False,
            }

    hass.data.setdefault(DOMAIN, {})["shared_mcp_server"] = _StubServer()

    loader = CustomToolsLoader(hass, profile_entry)
    await loader.initialize()

    result = await loader.handle_tool_call(
        "bridge_tool_status",
        {},
        context={"profile_entry_id": profile_entry.entry_id, "profile_name": "Test"},
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "core ok"
    assert captured["name"] == "sample_core_tool"
    assert captured["arguments"] == {"value": "ok"}
    assert captured["context"]["profile_entry_id"] == profile_entry.entry_id
    assert captured["context"]["profile_name"] == "Test"
