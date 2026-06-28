"""Brand asset tests for MCP Assist."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


BRAND_DIR = Path("custom_components/mcp_assist/brand")
SUPPORTED_FILES = {
    "icon.png",
    "icon@2x.png",
    "dark_icon.png",
    "dark_icon@2x.png",
    "logo.png",
    "logo@2x.png",
    "dark_logo.png",
    "dark_logo@2x.png",
}
EXPECTED_SIZES = {
    "icon.png": (256, 256),
    "dark_icon.png": (256, 256),
    "icon@2x.png": (512, 512),
    "dark_icon@2x.png": (512, 512),
    "logo.png": (256, 256),
    "dark_logo.png": (256, 256),
    "logo@2x.png": (512, 512),
    "dark_logo@2x.png": (512, 512),
}


def test_brand_assets_use_supported_home_assistant_filenames() -> None:
    """The brand folder should only contain documented asset filenames."""

    files = {path.name for path in BRAND_DIR.glob("*.png")}
    assert files == SUPPORTED_FILES


def test_brand_assets_are_valid_pngs_with_expected_sizes() -> None:
    """The shipped brand assets should match the expected local brand sizes."""

    for name, expected_size in EXPECTED_SIZES.items():
        path = BRAND_DIR / name
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.size == expected_size
            assert image.mode == "RGBA"
