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


def test_build_ws_url_percent_encodes_token() -> None:
    """Tokens with URL metacharacters must be percent-encoded in the query."""
    client = OpenClawClient(
        host="wss://gateway.example/",
        port=443,
        token="a&b/c #d",
        use_ssl=True,
        device_auth=SimpleNamespace(),
    )

    url = client._build_ws_url()

    assert url == "wss://gateway.example:443/?token=a%26b%2Fc%20%23d"
    # The raw, unencoded token must not appear in the URL.
    assert "a&b/c #d" not in url


def test_build_ws_url_strips_scheme_prefix_and_selects_ws() -> None:
    """Host protocol prefixes are stripped and ws is used without SSL."""
    client = OpenClawClient(
        host="http://10.0.0.5",
        port=18789,
        token="plain",
        use_ssl=False,
        device_auth=SimpleNamespace(),
    )

    assert client._build_ws_url() == "ws://10.0.0.5:18789/?token=plain"
