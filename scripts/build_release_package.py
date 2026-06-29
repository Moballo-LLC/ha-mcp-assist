#!/usr/bin/env python3
"""Build the HACS release package for HA MCP Assist."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "mcp_assist"
DEFAULT_ARCHIVE = ROOT / "dist" / "mcp_assist.zip"
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def should_include(path: Path) -> bool:
    """Return whether a component file should be packaged."""
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def build_package(archive: Path) -> None:
    """Build the release zip archive."""
    if not COMPONENT.is_dir():
        raise SystemExit(f"Component directory not found: {COMPONENT}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()

    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        for path in sorted(COMPONENT.rglob("*")):
            if should_include(path):
                zip_file.write(path, path.relative_to(COMPONENT.parent))

    try:
        display_path = archive.relative_to(ROOT)
    except ValueError:
        display_path = archive
    print(f"Built {display_path}")


def main() -> None:
    """Run the package builder."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to write the release zip.",
    )
    args = parser.parse_args()

    archive = args.archive
    if not archive.is_absolute():
        archive = ROOT / archive
    build_package(archive)


if __name__ == "__main__":
    main()
