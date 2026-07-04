"""Repository maintenance checks for Home Assistant manifest metadata."""

from __future__ import annotations

import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_components" / "mcp_assist" / "manifest.json"
HACS = ROOT / "hacs.json"
RUNTIME_REQUIREMENTS = ROOT / "requirements_runtime.txt"


def _runtime_requirements() -> list[str]:
    return [
        line.strip()
        for line in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _requirement_names(requirements: list[str]) -> list[str]:
    return [Requirement(requirement).name.lower() for requirement in requirements]


def _requirements_by_name(requirements: list[str]) -> dict[str, Requirement]:
    return {
        requirement.name.lower(): requirement
        for requirement in (Requirement(requirement) for requirement in requirements)
    }


def _highest_lower_bound(requirement: Requirement) -> Version | None:
    lower_bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">", ">="}
    ]
    return max(lower_bounds, default=None)


def _non_lower_bound_specifiers(requirement: Requirement) -> set[tuple[str, str]]:
    return {
        (specifier.operator, specifier.version)
        for specifier in requirement.specifier
        if specifier.operator not in {">", ">="}
    }


def test_runtime_requirements_track_manifest_packages() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert _requirement_names(_runtime_requirements()) == _requirement_names(
        manifest["requirements"]
    )


def test_runtime_requirements_do_not_raise_manifest_bounds() -> None:
    """Dependabot's runtime mirror must not narrow HA compatibility."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_requirements = _requirements_by_name(manifest["requirements"])

    for name, runtime_requirement in _requirements_by_name(_runtime_requirements()).items():
        manifest_requirement = manifest_requirements[name]
        runtime_lower_bound = _highest_lower_bound(runtime_requirement)
        manifest_lower_bound = _highest_lower_bound(manifest_requirement)

        assert not runtime_lower_bound or (
            manifest_lower_bound and runtime_lower_bound <= manifest_lower_bound
        ), (
            f"{runtime_requirement} is stricter than manifest requirement "
            f"{manifest_requirement}"
        )

        assert _non_lower_bound_specifiers(runtime_requirement) <= _non_lower_bound_specifiers(
            manifest_requirement
        ), (
            f"{runtime_requirement} adds caps, pins, or exclusions not present in "
            f"{manifest_requirement}"
        )


def test_duckduckgo_runtime_uses_renamed_ddgs_package() -> None:
    """The DuckDuckGo provider should install the maintained ddgs package."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    requirement_names = _requirement_names(manifest["requirements"])

    assert "ddgs" in requirement_names
    assert "duckduckgo-search" not in requirement_names


def test_display_names_keep_drop_in_integration_domain() -> None:
    """Repository and HA branding can change while the integration stays drop-in."""

    hacs = json.loads(HACS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert hacs["name"] == "HA MCP Assist"
    assert manifest["name"] == "HA MCP Assist"
    assert manifest["domain"] == "mcp_assist"
    assert MANIFEST.parent.name == manifest["domain"]
