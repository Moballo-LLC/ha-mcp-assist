"""Repository maintenance checks for Home Assistant manifest metadata."""

from __future__ import annotations

import json
from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_components" / "mcp_assist" / "manifest.json"
HACS = ROOT / "hacs.json"
RUNTIME_REQUIREMENTS = ROOT / "requirements_runtime.txt"


def _runtime_requirements() -> list[str]:
    return [
        line
        for line in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def _requirement_names(requirements: list[str]) -> list[str]:
    return [Requirement(requirement).name.lower() for requirement in requirements]


def test_runtime_requirements_track_manifest_packages() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert _requirement_names(_runtime_requirements()) == _requirement_names(
        manifest["requirements"]
    )


def test_duckduckgo_runtime_uses_renamed_ddgs_package() -> None:
    """The DuckDuckGo provider should install the maintained ddgs package."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    requirement_names = _requirement_names(manifest["requirements"])

    assert "ddgs" in requirement_names
    assert "duckduckgo-search" not in requirement_names


def test_hacs_display_name_keeps_drop_in_integration_domain() -> None:
    """HACS branding can differ while the installable integration stays drop-in."""

    hacs = json.loads(HACS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert hacs["name"] == "HA MCP Assist"
    assert manifest["domain"] == "mcp_assist"
    assert MANIFEST.parent.name == manifest["domain"]
