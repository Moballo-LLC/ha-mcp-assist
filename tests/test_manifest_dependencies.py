"""Repository maintenance checks for Home Assistant manifest metadata."""

from __future__ import annotations

import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
import pytest
import yaml

from scripts.check_manifest_core_dependencies import (
    check_manifest_core_dependencies,
    find_core_dependency_overlaps,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_components" / "mcp_assist" / "manifest.json"
HACS = ROOT / "hacs.json"
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
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


def test_dependabot_excludes_runtime_compatibility_mirror() -> None:
    """Routine version updates must not raise Home Assistant support floors."""

    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    pip_updates = [
        update
        for update in config["updates"]
        if update["package-ecosystem"] == "pip" and update["directory"] == "/"
    ]

    assert len(pip_updates) == 1
    excluded_paths = set(pip_updates[0].get("exclude-paths", []))
    assert RUNTIME_REQUIREMENTS.name in excluded_paths
    assert "custom_components/mcp_assist/manifest.json" in excluded_paths


def test_runtime_surfaces_do_not_duplicate_installed_homeassistant_core() -> None:
    """The manifest and Dependabot mirror cannot redeclare Core dependencies."""
    check_manifest_core_dependencies()


@pytest.mark.parametrize(
    ("integration_requirement", "core_requirement"),
    [
        ("AIOHTTP>=3.8.0", "aiohttp>=3.8"),
        ("PyYAML>=6.0", "pyyaml>=6.0"),
        ("cryptography>=41.0.0", "Cryptography>=41"),
        ("My_Package>=1", "my-package>=1"),
    ],
)
def test_core_overlap_guard_normalizes_distribution_names(
    integration_requirement: str, core_requirement: str
) -> None:
    """Case and separator variants of a direct Core package are rejected."""
    assert find_core_dependency_overlaps(
        [integration_requirement], [core_requirement]
    ) == {canonicalize_name(Requirement(integration_requirement).name)}


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
