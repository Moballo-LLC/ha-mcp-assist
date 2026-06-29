"""Repository maintenance checks for localized Home Assistant strings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STRINGS = ROOT / "custom_components" / "mcp_assist" / "strings.json"
TRANSLATIONS = ROOT / "custom_components" / "mcp_assist" / "translations"
PLACEHOLDER_PATTERN = re.compile(r"\{[A-Za-z0-9_]+\}")


def _flatten_strings(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    strings: dict[str, str] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            strings.update(_flatten_strings(value, path))
        else:
            strings[path] = value
    return strings


def test_translation_files_match_source_strings() -> None:
    source = _flatten_strings(json.loads(STRINGS.read_text(encoding="utf-8")))
    source_keys = set(source)
    failures: list[str] = []

    for path in sorted(TRANSLATIONS.glob("*.json")):
        translated = _flatten_strings(json.loads(path.read_text(encoding="utf-8")))
        translated_keys = set(translated)

        missing = sorted(source_keys - translated_keys)
        extra = sorted(translated_keys - source_keys)
        if missing:
            failures.append(f"{path.name} missing keys: {missing}")
        if extra:
            failures.append(f"{path.name} extra keys: {extra}")

        for key in sorted(source_keys & translated_keys):
            source_value = source[key]
            translated_value = translated[key]
            if PLACEHOLDER_PATTERN.findall(translated_value) != PLACEHOLDER_PATTERN.findall(
                source_value
            ):
                failures.append(f"{path.name} placeholder mismatch at {key}")
            if source_value.count("\n") != translated_value.count("\n"):
                failures.append(f"{path.name} newline mismatch at {key}")
            if "QZXP" in translated_value or "QZXS" in translated_value or "QZXITEM" in translated_value:
                failures.append(f"{path.name} guard token leaked at {key}")

    assert not failures
