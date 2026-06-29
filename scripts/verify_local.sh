#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:---all}"
case "$MODE" in
  --all)
    RUN_STATIC=1
    RUN_TESTS=1
    ;;
  --static)
    RUN_STATIC=1
    RUN_TESTS=0
    ;;
  --pytest)
    RUN_STATIC=0
    RUN_TESTS=1
    ;;
  *)
    echo "Usage: $0 [--all|--static|--pytest]" >&2
    exit 2
    ;;
esac

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

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip wheel
  "$PYTHON_BIN" -m pip install -r requirements_test.txt
fi

echo "Using Python: $("$PYTHON_BIN" --version)"

if [[ "$RUN_STATIC" == "1" ]]; then
  "$PYTHON_BIN" -m ruff check custom_components tests
  "$PYTHON_BIN" -m compileall -q custom_components tests
  "$PYTHON_BIN" -m json.tool custom_components/mcp_assist/strings.json >/dev/null
  "$PYTHON_BIN" -m json.tool custom_components/mcp_assist/translations/en.json >/dev/null
  git diff --check
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  mkdir -p test-results
  "$PYTHON_BIN" -m pytest -ra --junitxml=test-results/pytest.xml tests
fi
