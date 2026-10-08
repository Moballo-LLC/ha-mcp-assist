#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 HA_VERSION [HARNESS_VERSION] [PYTHON]" >&2
  exit 2
fi

HA_VERSION="$1"
PYTHON_BIN="${3:-python}"

case "$HA_VERSION" in
  2026.10.0) HARNESS_VERSION="0.13.371" ;;
  2026.9.3) HARNESS_VERSION="0.13.366" ;;
  2026.8.3) HARNESS_VERSION="0.13.357" ;;
  *)
    echo "Unsupported HA version: $HA_VERSION" >&2
    exit 2
    ;;
esac

if [[ $# -ge 2 && "$2" != "$HARNESS_VERSION" ]]; then
  echo "Harness $2 does not match HA $HA_VERSION; expected $HARNESS_VERSION" >&2
  exit 2
fi

TEMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT
sed '/^[[:space:]]*pytest-homeassistant-custom-component==/d' requirements_test.txt > "$TEMP_REQUIREMENTS"

"$PYTHON_BIN" -m pip install -r "$TEMP_REQUIREMENTS"
"$PYTHON_BIN" -m pip install "pytest-homeassistant-custom-component==$HARNESS_VERSION"

"$PYTHON_BIN" - "$HA_VERSION" "$HARNESS_VERSION" <<'PY'
import sys

from homeassistant.const import __version__ as actual_ha
from pytest_homeassistant_custom_component import const as harness_const

expected_ha, expected_harness = sys.argv[1:]
harness_ha = (
    f"{harness_const.MAJOR_VERSION}.{harness_const.MINOR_VERSION}."
    f"{harness_const.PATCH_VERSION}"
)
if actual_ha != expected_ha or harness_ha != expected_ha:
    raise SystemExit(
        f"Expected HA {expected_ha} via harness {expected_harness}; "
        f"installed HA is {actual_ha}, harness HA is {harness_ha}"
    )
print(f"Verified Home Assistant {actual_ha} with harness {expected_harness}")
PY
