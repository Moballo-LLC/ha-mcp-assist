"""Look up runtime language defaults from Home Assistant translation files."""

from functools import cache
import json
import logging
from pathlib import Path

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)
_COMMON: dict[str, dict[str, str]] = {}

# Match longer unit symbols before their single-character prefixes.
_SPOKEN_SYMBOL_KEYS = (
    ("°C", "celsius"), ("°F", "fahrenheit"), ("°", "degrees"),
    ("%", "percent"), ("€", "euros"), ("£", "pounds"), ("$", "dollars"),
    ("&", "and"), ("+", "plus"), ("=", "equals"), ("<", "less_than"),
    (">", "greater_than"), ("@", "at"), ("#", "number"),
    ("×", "times"), ("÷", "divided_by"),
)


@cache
def _read_localizations() -> dict[str, dict[str, str]]:
    """Read bundled translation data once, in the executor."""
    root = Path(__file__).parent / "translations"
    return {
        path.stem.lower(): json.loads(path.read_text(encoding="utf-8"))["common"]
        for path in root.glob("*.json")
    }


async def async_load_localizations(hass: HomeAssistant) -> None:
    """Prepare synchronous lookups before config forms or profiles use them."""
    if not _COMMON:
        _COMMON.update(await hass.async_add_executor_job(_read_localizations))


def _language_data(language_code: str) -> dict[str, str]:
    """Resolve exact locales, then progressively less-specific language codes."""
    code = (language_code or "en").replace("_", "-").lower()
    while code:
        if code in _COMMON:
            return _COMMON[code]
        code = code.rsplit("-", 1)[0] if "-" in code else ""
    return {}


def get_language_instruction(language_code: str) -> str:
    """Return the locale's prompt instruction, or use the default English prompt."""
    return _language_data(language_code).get("language_instruction", "")


def _phrases(language_code: str, key: str) -> str:
    """Get localized detection phrases with the established English fallback."""
    data = _language_data(language_code)
    if key not in data:
        _LOGGER.warning("Language '%s' has no %s; using English defaults", language_code, key)
    return data.get(key, _COMMON["en"][key])


def get_follow_up_phrases(language_code: str) -> str:
    """Return comma-separated follow-up phrases."""
    return _phrases(language_code, "follow_up_phrases")


def get_end_words(language_code: str) -> str:
    """Return comma-separated conversation-ending phrases."""
    return _phrases(language_code, "end_words")


def get_supported_languages() -> list[str]:
    """Return languages with a localized prompt, excluding the English default."""
    return sorted(code for code, data in _COMMON.items() if data.get("language_instruction"))


def get_spoken_symbols(language_code: str) -> list[tuple[str, str]]:
    """Return ordered replacements; never substitute English for an unknown locale."""
    data = _language_data(language_code)
    replacements = []
    for symbol, key in _SPOKEN_SYMBOL_KEYS:
        template = data.get(f"spoken_{key}", "{value}" + symbol)
        replacement = template.replace("{value}", "")
        if replacement != symbol:
            replacements.append((symbol, replacement))
    return replacements
