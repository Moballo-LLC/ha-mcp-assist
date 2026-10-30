"""Runtime language data stays in the Home Assistant translation files."""

import json
from pathlib import Path
import threading

import pytest

from custom_components.mcp_assist import localization


@pytest.mark.parametrize(("variant", "base"), [
    ("fr-CA", "fr"), ("FR_ca", "fr"), ("zh-Hans", "zh-hans"),
    ("zh_Hans_CN", "zh-hans"), ("en-US", "en"),
])
def test_locale_variants_use_the_bundled_language(variant, base):
    assert localization.get_language_instruction(variant) == localization.get_language_instruction(base)
    assert localization.get_follow_up_phrases(variant) == localization.get_follow_up_phrases(base)
    assert localization.get_end_words(variant) == localization.get_end_words(base)
    assert localization.get_spoken_symbols(variant) == localization.get_spoken_symbols(base)


def test_unknown_language_falls_back_for_phrases_but_not_spoken_units():
    assert localization.get_language_instruction("unknown") == ""
    assert localization.get_follow_up_phrases("unknown") == localization.get_follow_up_phrases("en")
    assert localization.get_end_words("unknown") == localization.get_end_words("en")
    assert localization.get_spoken_symbols("unknown") == []


async def test_locale_files_load_off_loop_and_lookups_do_not_read_files(hass, monkeypatch):
    localization._read_localizations.cache_clear()
    monkeypatch.setattr(localization, "_COMMON", {})
    original = Path.read_text
    threads = []

    def read(path, *args, **kwargs):
        threads.append(threading.get_ident())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    await localization.async_load_localizations(hass)
    assert threads and threading.get_ident() not in threads
    threads.clear()
    await localization.async_load_localizations(hass)
    assert "Deutsch" in localization.get_language_instruction("de")
    assert localization.get_spoken_symbols("fr")[0] == ("°C", " degrés Celsius")
    assert threads == []


def test_runtime_strings_are_complete_and_hassfest_compatible():
    from homeassistant.helpers import config_validation as cv

    root = Path("custom_components/mcp_assist")
    base = json.loads((root / "strings.json").read_text())["common"]
    for path in (root / "translations").glob("*.json"):
        common = json.loads(path.read_text())["common"]
        assert common.keys() == base.keys(), path
        for key, value in common.items():
            assert value == value.strip(), (path, key)
            assert cv.string_with_no_html(value) == value
            if key.startswith("spoken_"):
                assert value.count("{value}") == 1, (path, key)
                assert value.format(value="") == value.replace("{value}", "")
