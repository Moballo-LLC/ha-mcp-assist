#!/usr/bin/env python3
"""Reject runtime requirements that Home Assistant Core already provides."""

from __future__ import annotations

from importlib.metadata import distribution
import json
from pathlib import Path
from typing import Iterable

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "custom_components" / "mcp_assist" / "manifest.json"
RUNTIME_REQUIREMENTS = ROOT / "requirements_runtime.txt"


def normalized_requirement_names(requirements: Iterable[str]) -> set[str]:
    """Return PEP 503 normalized distribution names for requirement strings."""
    return {
        canonicalize_name(Requirement(requirement).name)
        for requirement in requirements
    }


def find_core_dependency_overlaps(
    integration_requirements: Iterable[str],
    homeassistant_requirements: Iterable[str],
) -> set[str]:
    """Find normalized requirements declared by both the integration and Core."""
    return normalized_requirement_names(integration_requirements) & (
        normalized_requirement_names(homeassistant_requirements)
    )


def _runtime_requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def check_manifest_core_dependencies() -> None:
    """Fail if either runtime declaration duplicates a Core direct requirement."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    runtime_requirements = _runtime_requirements(RUNTIME_REQUIREMENTS)
    core_requirements = distribution("homeassistant").requires
    if not core_requirements:
        raise SystemExit(
            "Home Assistant distribution metadata has no direct requirements; "
            "cannot verify integration dependency ownership."
        )

    conflicts = {
        "manifest": find_core_dependency_overlaps(
            manifest.get("requirements", []), core_requirements
        ),
        "runtime mirror": find_core_dependency_overlaps(
            runtime_requirements, core_requirements
        ),
    }
    conflicts = {surface: names for surface, names in conflicts.items() if names}
    if conflicts:
        details = "; ".join(
            f"{surface}: {', '.join(sorted(names))}"
            for surface, names in conflicts.items()
        )
        raise SystemExit(
            "Remove Home Assistant Core dependencies from integration runtime "
            f"requirements ({details})."
        )


if __name__ == "__main__":
    check_manifest_core_dependencies()
    print("Integration runtime requirements do not duplicate Home Assistant Core dependencies.")
