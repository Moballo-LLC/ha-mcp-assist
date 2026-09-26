import json
import re
from pathlib import Path


SELECTOR_OPTION_KEY = re.compile(r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$")
INTEGRATION_DIR = Path(__file__).parents[1] / "custom_components" / "mcp_assist"
TRANSLATION_FILES = [
    INTEGRATION_DIR / "strings.json",
    *sorted((INTEGRATION_DIR / "translations").glob("*.json")),
]


def test_selector_option_translation_keys_follow_hassfest_format():
    invalid_keys = []

    for path in TRANSLATION_FILES:
        translation = json.loads(path.read_text(encoding="utf-8"))
        for selector, definition in translation.get("selector", {}).items():
            for key in definition.get("options", {}):
                if not SELECTOR_OPTION_KEY.fullmatch(key):
                    invalid_keys.append(f"{path.name}: selector.{selector}.options.{key}")

    assert not invalid_keys, "Invalid selector option translation keys: " + ", ".join(
        invalid_keys
    )
