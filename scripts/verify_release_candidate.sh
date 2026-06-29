#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

find_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"
    return
  fi

  if [[ -x /tmp/ha-mcp-assist-py314-venv/bin/python ]]; then
    echo /tmp/ha-mcp-assist-py314-venv/bin/python
    return
  fi

  for candidate in python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  echo "Python was not found. Set PYTHON=/path/to/python." >&2
  exit 1
}

PYTHON_BIN="$(find_python)"
export PYTHON="$PYTHON_BIN"

if [[ "${REQUIRE_MAIN_ANCESTOR:-0}" == "1" ]]; then
  git fetch --no-tags --depth=1 origin main
  git merge-base --is-ancestor HEAD origin/main || {
    echo "Release candidate commit must be reachable from origin/main." >&2
    exit 1
  }
fi

if [[ "${SKIP_LOCAL_VERIFY:-0}" != "1" ]]; then
  scripts/verify_local.sh --all
fi

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

tag = os.environ.get("RELEASE_TAG", "").strip()
manifest_path = Path("custom_components/mcp_assist/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
version = str(manifest.get("version", "")).strip()

if not version:
    raise SystemExit("manifest.json must declare a non-empty version")

if tag:
    if not tag.startswith("v"):
        raise SystemExit(f"Release tag {tag!r} must start with 'v'")
    if tag.removeprefix("v") != version:
        raise SystemExit(
            f"Release tag {tag!r} does not match manifest version {version!r}"
        )
PY

rm -rf dist
"$PYTHON_BIN" scripts/build_release_package.py --archive dist/mcp_assist.zip

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import compileall
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

archive = Path("dist/mcp_assist.zip")
required_files = {
    "mcp_assist/__init__.py",
    "mcp_assist/agent.py",
    "mcp_assist/config_flow.py",
    "mcp_assist/manifest.json",
    "mcp_assist/mcp_server.py",
    "mcp_assist/strings.json",
    "mcp_assist/translations/en.json",
}

if not archive.is_file():
    raise SystemExit(f"Release archive was not built: {archive}")

with ZipFile(archive) as zip_file:
    bad_member = zip_file.testzip()
    if bad_member is not None:
        raise SystemExit(f"Release archive has a corrupt member: {bad_member}")

    names = set(zip_file.namelist())
    missing = sorted(required_files - names)
    if missing:
        raise SystemExit(f"Release archive is missing required files: {missing}")

    forbidden = sorted(
        name
        for name in names
        if (
            name.startswith("/")
            or ".." in Path(name).parts
            or not name.startswith("mcp_assist/")
            or "__pycache__" in Path(name).parts
            or Path(name).suffix in {".pyc", ".pyo"}
        )
    )
    if forbidden:
        raise SystemExit(f"Release archive contains forbidden paths: {forbidden[:10]}")

    manifest = json.loads(zip_file.read("mcp_assist/manifest.json"))
    tag = os.environ.get("RELEASE_TAG", "").strip()
    if tag and manifest.get("version") != tag.removeprefix("v"):
        raise SystemExit("Packaged manifest version does not match RELEASE_TAG")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file.extractall(temp_dir)
        extracted_component = Path(temp_dir) / "mcp_assist"
        if not compileall.compile_dir(
            extracted_component,
            quiet=1,
            force=True,
        ):
            raise SystemExit("Packaged integration failed compileall")

print("Release candidate package verified.")
PY
