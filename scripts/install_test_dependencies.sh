#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--environment" ]]; then
  [[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: $0 --environment FILE [PYTHON]" >&2; exit 2; }
  exec "${3:-python}" "$ROOT/scripts/test_environment.py" install --environment "$2"
fi
[[ $# -le 3 ]] || { echo "Usage: $0 [TARGET [HARNESS_VERSION [PYTHON]]]" >&2; exit 2; }
PYTHON_BIN="${3:-python}"
ENVIRONMENT="$(mktemp)"
trap 'rm -f "$ENVIRONMENT"' EXIT
ARGS=(resolve "${1:-default}" --output "$ENVIRONMENT")
if [[ $# -ge 2 && -n "$2" ]]; then ARGS+=(--expected-harness "$2"); fi
"$PYTHON_BIN" "$ROOT/scripts/test_environment.py" "${ARGS[@]}"
"$PYTHON_BIN" "$ROOT/scripts/test_environment.py" install --environment "$ENVIRONMENT"
