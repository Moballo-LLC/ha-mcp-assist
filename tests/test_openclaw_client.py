"""Tests for OpenClaw client behavior."""

from __future__ import annotations

from types import SimpleNamespace

from custom_components.mcp_assist.openclaw_client import OpenClawClient


def test_openclaw_client_uses_configured_locale() -> None:
    """OpenClaw handshake locale should follow the Home Assistant language."""
    client = OpenClawClient(
        host="internal.example",
        port=18789,
        token="test-token",
        use_ssl=True,
        device_auth=SimpleNamespace(),
        locale="fr_CA",
    )

    assert client._locale == "fr-CA"


def test_openclaw_client_locale_falls_back_to_english() -> None:
    """Empty OpenClaw locale should retain the previous English fallback."""
    client = OpenClawClient(
        host="internal.example",
        port=18789,
        token="test-token",
        use_ssl=True,
        device_auth=SimpleNamespace(),
        locale="",
    )

    assert client._locale == "en-US"
