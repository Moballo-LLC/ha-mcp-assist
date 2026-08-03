"""Tests for outbound secret redaction."""

from __future__ import annotations

import json

from custom_components.mcp_assist.secret_redaction import (
    redact_exception,
    redact_secrets,
    redact_secret_text,
)


CANARY = "not-a-real-secret-canary-12345"


def test_redact_secrets_covers_nested_credentials_and_strings() -> None:
    """Nested credential fields and credential-bearing strings must be redacted."""
    value = {
        "safe": "keep me",
        "nested": [
            {"api_key": CANARY},
            {"apiToken": CANARY},
            {"clientSecret": CANARY},
            {"aws_secret_access_key": CANARY},
            {"provider_access_token": "short-lived"},
            {"providerAuthorization": "short-auth"},
            {"providerAuthToken": "short-token"},
            {"providerIdToken": "short-id"},
            {"vendorCookie": "short-cookie"},
            {"headers": {"Authorization": f"Bearer {CANARY}"}},
            {
                "urls": [
                    f"https://user:{CANARY}@example.com/resource",
                    f"https://example.com/resource?access_token={CANARY}&page=2",
                    f"https://example.com/resource?page=2&signature={CANARY}",
                    f"https://example.com/bearer/{CANARY}/resource",
                ]
            },
        ],
        "error": f"Request failed: Authorization: Bearer {CANARY}",
        "ordinary_mapping": {
            "key": "temperature",
            "page_token": "page-123456",
            "token": "page-123456",
            "signature": "detached-signature",
        },
        "opaque_mapping": {
            "page_token": "fedcba9876543210fedcba9876543210",
            "request_token": "abcdefghijklmnopqrstuvwxyzabcdef",
            "token": "0123456789abcdef0123456789abcdef",
            "signature": "abcdef0123456789abcdef0123456789",
        },
        "ordinary_urls": [
            "https://example.com/items?token=page-123456",
            "https://example.com/items?signature=detached-signature",
            "https://example.com/items?page_token=page-123456",
            "https://example.com/items/token/page-123456/details",
        ],
        "instructions": "Press key: Enter",
        "token_count": 42,
        "max_tokens": 500,
    }

    redacted = redact_secrets(value)
    serialized = json.dumps(redacted)

    assert CANARY not in serialized
    assert redacted["safe"] == "keep me"
    for item in redacted["nested"][3:9]:
        assert next(iter(item.values())) == "[redacted]"
    assert redacted["ordinary_mapping"] == {
        "key": "temperature",
        "page_token": "page-123456",
        "token": "page-123456",
        "signature": "detached-signature",
    }
    assert redacted["opaque_mapping"]["page_token"] == "[redacted]"
    assert redacted["opaque_mapping"]["request_token"] == "[redacted]"
    assert redacted["opaque_mapping"]["token"] == "[redacted]"
    assert redacted["opaque_mapping"]["signature"] == "[redacted]"
    assert redacted["ordinary_urls"] == [
        "https://example.com/items?token=page-123456",
        "https://example.com/items?signature=detached-signature",
        "https://example.com/items?page_token=page-123456",
        "https://example.com/items/token/page-123456/details",
    ]
    assert redacted["instructions"] == "Press key: Enter"
    assert redacted["token_count"] == 42
    assert redacted["max_tokens"] == 500
    assert serialized.count("[redacted]") >= 7


def test_redact_secrets_handles_cycles_without_revealing_values() -> None:
    """Self-referential values must remain bounded and secret-safe."""
    value: dict[str, object] = {"password": CANARY}
    value["self"] = value

    redacted = redact_secrets(value)
    serialized = json.dumps(redacted)

    assert CANARY not in serialized
    assert "circular reference" in serialized


def test_redact_secrets_covers_plural_cookie_containers() -> None:
    """Conventional cookie collections must not expose their nested values."""
    value = {
        "cookies": {"sessionid": "short-secret"},
        "set_cookies": ["sessionid=short-secret"],
        "providerCookies": {"sessionid": "short-secret"},
        "providerSetCookies": ["sessionid=short-secret"],
    }

    assert redact_secrets(value) == {
        "cookies": "[redacted]",
        "set_cookies": "[redacted]",
        "providerCookies": "[redacted]",
        "providerSetCookies": "[redacted]",
    }


def test_redact_secrets_covers_passphrase_labels() -> None:
    """Passphrase fields and provider-prefixed variants must be protected."""
    value = {
        "passphrase": CANARY,
        "providerPassphrase": "short-secret",
        "error": "passphrase=short-secret",
    }

    assert redact_secrets(value) == {
        "passphrase": "[redacted]",
        "providerPassphrase": "[redacted]",
        "error": "passphrase=[redacted]",
    }


def test_redact_secrets_covers_compound_secret_token_labels() -> None:
    """Explicit secret-token compounds must not use ambiguous token heuristics."""
    value = {
        "secret_token": "short-secret",
        "providerSecretToken": "short-provider-secret",
        "error": "secret.token=short-text-secret",
    }

    assert redact_secrets(value) == {
        "secret_token": "[redacted]",
        "providerSecretToken": "[redacted]",
        "error": "secret.token=[redacted]",
    }


def test_redact_secrets_covers_storage_account_key_labels() -> None:
    """Storage account keys and provider-prefixed variants must be protected."""
    value = {
        "AccountKey": CANARY,
        "azureAccountKey": "short-provider-secret",
        "error": "account.key=short-text-secret",
    }

    assert redact_secrets(value) == {
        "AccountKey": "[redacted]",
        "azureAccountKey": "[redacted]",
        "error": "account.key=[redacted]",
    }


def test_redact_secrets_preserves_boolean_and_empty_status_values() -> None:
    """Sensitive-looking keys must preserve values that cannot contain secrets."""
    value = {
        "auth": False,
        "authorization": True,
        "cookie": [],
        "credentials": {},
        "password": "",
        "secret": None,
    }

    assert redact_secrets(value) == value


def test_redact_secrets_preserves_known_authentication_status_strings() -> None:
    """Authentication metadata should remain usable without exposing credentials."""
    value = {
        "auth": "none",
        "authorization": "unsupported",
        "providerAuth": "oauth2",
        "auth_token": "none",
        "password": "none",
    }

    assert redact_secrets(value) == {
        "auth": "none",
        "authorization": "unsupported",
        "providerAuth": "oauth2",
        "auth_token": "[redacted]",
        "password": "[redacted]",
    }


def test_redact_secrets_preserves_numeric_authentication_statuses() -> None:
    """Numeric auth statuses should retain their values and types."""
    value = {
        "auth": 401,
        "authorization": 200,
        "providerAuth": 403.0,
        "credential": 123456,
        "auth_token": 401,
    }

    assert redact_secrets(value) == {
        "auth": 401,
        "authorization": 200,
        "providerAuth": 403.0,
        "credential": "[redacted]",
        "auth_token": "[redacted]",
    }


def test_redact_secrets_preserves_deep_acyclic_payloads() -> None:
    """Deep ordinary results should retain their complete shape."""
    value: dict[str, object] = {"api_key": CANARY, "value": "leaf"}
    for _ in range(50):
        value = {"nested": value}

    redacted = redact_secrets(value)
    cursor = redacted
    for _ in range(50):
        assert list(cursor) == ["nested"]
        cursor = cursor["nested"]

    assert cursor == {"api_key": "[redacted]", "value": "leaf"}


def test_redact_secrets_is_idempotent() -> None:
    """Repeated boundary checks must not corrupt existing redaction markers."""
    value = {
        "error": f"Authorization: Bearer {CANARY}",
        "url": f"https://example.com/resource?access_token={CANARY}",
    }

    redacted = redact_secrets(value)

    assert redact_secrets(redacted) == redacted


def test_redact_secret_text_preserves_ordinary_auth_words() -> None:
    """Auth vocabulary in ordinary prose must not be mistaken for a credential."""
    text = "Basic lighting controls use bearer authentication."

    assert redact_secret_text(text) == text


def test_redact_secret_text_handles_long_alternating_case_input() -> None:
    """Camel-prefix matching must remain linear on adversarial ordinary text."""
    value = "aA" * 2048

    assert redact_secret_text(value) == value


def test_redact_secret_text_preserves_known_authentication_status_strings() -> None:
    """Serialized authentication metadata should match structured behavior."""
    value = (
        'auth=none; payload={"authorization":"unsupported"}; '
        "providerAuth=oauth2; auth_token=none"
    )

    assert redact_secret_text(value) == (
        'auth=none; payload={"authorization":"unsupported"}; '
        "providerAuth=oauth2; auth_token=[redacted]"
    )


def test_redact_secret_text_preserves_authentication_status_queries() -> None:
    """Auth status query values should remain usable while tokens stay protected."""
    value = (
        "https://example.com/callback?auth=none&authorization=unsupported"
        "&providerAuth=401&access_token=none"
    )

    assert redact_secret_text(value) == (
        "https://example.com/callback?auth=none&authorization=unsupported"
        "&providerAuth=401&access_token=[redacted]"
    )


def test_redact_secret_text_preserves_serialized_data_after_query_secrets() -> None:
    """Query redaction must not consume the enclosing serialized payload."""
    value = (
        '{"url":"https://example.com/?access_token=short-secret","state":"ok"}; '
        r'payload={\"url\":\"https://example.com/?api_key=another-secret\",'
        r'\"state\":\"ok\"}'
    )

    assert redact_secret_text(value) == (
        '{"url":"https://example.com/?access_token=[redacted]","state":"ok"}; '
        r'payload={\"url\":\"https://example.com/?api_key=[redacted]\",'
        r'\"state\":\"ok\"}'
    )


def test_redact_secret_text_decodes_query_names_before_classification() -> None:
    """Encoded separators in credential names must not bypass redaction."""
    value = (
        "https://example.com/callback?access%5Ftoken=short-secret"
        "&api%2Dkey=another-short-secret"
        f"&page%5Ftoken={CANARY}"
        "&auth%2Estatus=enabled"
        "&ordinary%5Ffield=keep"
    )

    assert redact_secret_text(value) == (
        "https://example.com/callback?access%5Ftoken=[redacted]"
        "&api%2Dkey=[redacted]"
        "&page%5Ftoken=[redacted]"
        "&auth%2Estatus=enabled"
        "&ordinary%5Ffield=keep"
    )


def test_redact_secret_text_decodes_form_encoded_query_names() -> None:
    """Form-encoded spaces in sensitive query names must be classified."""
    value = "https://example.com/callback?access+token=short-secret"

    assert redact_secret_text(value) == (
        "https://example.com/callback?access+token=[redacted]"
    )


def test_redact_secret_text_decodes_first_form_field_names() -> None:
    """An encoded first form field must not require a URL delimiter."""
    value = (
        "access%5Ftoken=short-secret&state=ok\n"
        "  api%2Dkey=another-short-secret&ordinary=keep"
    )

    assert redact_secret_text(value) == (
        "access%5Ftoken=[redacted]&state=ok\n"
        "  api%2Dkey=[redacted]&ordinary=keep"
    )


def test_redact_secret_text_decodes_embedded_form_field_names() -> None:
    """Encoded form fields embedded in exception prose must be redacted."""
    value = (
        "response body: access%5Ftoken=short-secret&state=ok; "
        'payload="api%2Dkey=another-short-secret"'
    )

    assert redact_secret_text(value) == (
        "response body: access%5Ftoken=[redacted]&state=ok; "
        'payload="api%2Dkey=[redacted]"'
    )


def test_redact_secret_text_handles_many_query_delimiters() -> None:
    """Encoded-name scanning must remain linear across repeated delimiters."""
    value = "#" + "?!" * 4096

    assert redact_secret_text(value) == value


def test_redact_secret_text_covers_standalone_auth_credentials() -> None:
    """Standalone HTTP auth values should be redacted without a header key."""
    lowercase_token = "abcdefghijklmnopqrstuvwxyzabcdef"
    value = (
        f"Basic dXNlcjpwYXNzd29yZA==; Bearer {CANARY}; "
        f"Bearer {lowercase_token}"
    )

    redacted = redact_secret_text(value)

    assert "dXNlcjpwYXNzd29yZA==" not in redacted
    assert CANARY not in redacted
    assert lowercase_token not in redacted
    assert redacted.count("[redacted]") == 3


def test_redact_secret_text_covers_complete_sensitive_header_lines() -> None:
    """Multipart authorization and cookie headers must redact complete values."""
    value = (
        'Authorization: Digest username="alice", nonce="short-nonce", '
        'response="short-secret"\n'
        "Proxy-Authorization: Basic dTpw\n"
        "Cookie: first=a; sessionid=short-secret\n"
        "Set-Cookie: sessionid=short-secret; HttpOnly\n"
        "status=failed"
    )

    assert redact_secret_text(value) == (
        "Authorization: [redacted]\n"
        "Proxy-Authorization: [redacted]\n"
        "Cookie: [redacted]\n"
        "Set-Cookie: [redacted]\n"
        "status=failed"
    )


def test_redact_secret_text_preserves_authentication_status_header_lines() -> None:
    """Known auth header statuses should survive while cookies remain protected."""
    value = (
        "Authorization: unsupported\n"
        "Proxy-Authorization: none\n"
        "Cookie: disabled"
    )

    assert redact_secret_text(value) == (
        "Authorization: unsupported\n"
        "Proxy-Authorization: none\n"
        "Cookie: [redacted]"
    )


def test_redact_secret_text_covers_short_basic_credentials() -> None:
    """Short valid Basic credentials must reach base64 validation."""
    assert redact_secret_text("Basic dTpw") == "Basic [redacted]"


def test_redact_secret_text_covers_multiline_private_keys() -> None:
    """A labeled PEM value must not expose its body or consume later fields."""
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        f"MII{CANARY}\n"
        "-----END PRIVATE KEY-----"
    )
    value = f"private_key={private_key}\nstatus=failed"

    redacted = redact_secret_text(value)

    assert CANARY not in redacted
    assert "BEGIN PRIVATE KEY" not in redacted
    assert redacted == "private_key=[redacted]\nstatus=failed"


def test_redact_secret_text_preserves_ordinary_credential_named_routes() -> None:
    """Credential vocabulary in route names should not corrupt ordinary URLs."""
    ordinary = "https://example.com/auth/callback /password/reset"
    opaque = f"https://example.com/bearer/{CANARY}/resource"

    assert redact_secret_text(ordinary) == ordinary
    assert redact_secret_text(opaque) == (
        "https://example.com/bearer/[redacted]/resource"
    )


def test_redact_secret_text_covers_short_unambiguous_path_credentials() -> None:
    """Strong token and key path labels should protect short credentials."""
    value = (
        "https://example.com/api_key/short-secret/resource "
        "https://example.com/bearer/short-secret"
    )

    redacted = redact_secret_text(value)

    assert "short-secret" not in redacted
    assert redacted.count("[redacted]") == 2


def test_redact_secret_text_covers_generic_credential_labels() -> None:
    """Credential labels in exception-like text remain protected."""
    value = f"Request failed: token={CANARY}"

    redacted = redact_secret_text(value)

    assert CANARY not in redacted
    assert redacted.count("[redacted]") == 1


def test_redact_secret_text_covers_prefixed_credential_labels() -> None:
    """Provider-prefixed credential names receive the same protection."""
    value = (
        f"provider_api_token={CANARY}; "
        f"https://example.com/object?x-amz-signature={CANARY}"
    )

    redacted = redact_secret_text(value)

    assert CANARY not in redacted
    assert redacted.count("[redacted]") == 2


def test_redact_secret_text_covers_every_canonical_token_label() -> None:
    """Strong token keys stay strong in text even when their values are short."""
    value = (
        "provider_session_token=short-session; "
        'payload={"csrf_token":"short-csrf"}; '
        "openclaw_token=short-openclaw; "
        "providerSessionToken=short-camel; "
        r'payload={\"providerCsrfToken\": \"short-csrf-camel\"}'
    )

    redacted = redact_secret_text(value)

    assert "short-session" not in redacted
    assert "short-csrf" not in redacted
    assert "short-openclaw" not in redacted
    assert "short-camel" not in redacted
    assert "short-csrf-camel" not in redacted
    assert redacted.count("[redacted]") == 5


def test_redact_secret_text_covers_dotted_credential_labels() -> None:
    """Dotted text labels should match equivalent structured credential keys."""
    value = (
        "api.key=short-secret; private.key: short-private; "
        r'payload={"provider.api.key":"short-provider"}'
    )

    redacted = redact_secret_text(value)

    assert "short-secret" not in redacted
    assert "short-private" not in redacted
    assert "short-provider" not in redacted
    assert redacted.count("[redacted]") == 3


def test_redact_secret_text_preserves_ordinary_generic_labels() -> None:
    """Pagination tokens and detached signatures remain usable tool-result text."""
    value = (
        "next page token=page-123456; "
        "signature=AbCdEf0123456789-AbCdEf0123456789; "
        'token="this is ordinary pagination guidance for the next request"'
    )

    assert redact_secret_text(value) == value


def test_redact_secret_text_covers_complete_quoted_credentials() -> None:
    """Quoted credential values must be redacted through their closing quote."""
    value = (
        'password="correct horse battery staple"; '
        f"api_token='opaque {CANARY} value'"
    )

    redacted = redact_secret_text(value)

    assert "correct horse battery staple" not in redacted
    assert CANARY not in redacted
    assert redacted == 'password="[redacted]"; api_token=\'[redacted]\''


def test_redact_secret_text_covers_unquoted_passwords_with_spaces() -> None:
    """Strong unquoted labels redact through a safe structural delimiter."""
    value = "password=correct horse battery staple; retry=false"

    assert redact_secret_text(value) == "password=[redacted]; retry=false"


def test_redact_secret_text_covers_escaped_json_credentials() -> None:
    """Credential values in an escaped JSON snippet must not leak after a space."""
    value = r'payload={\"password\": \"correct horse battery staple\"}'

    assert redact_secret_text(value) == r'payload={\"password\": \"[redacted]\"}'


def test_redact_secret_text_covers_multiply_escaped_json_credentials() -> None:
    """Repeated JSON serialization must not hide a quoted credential label."""
    value = r'payload={\\\"password\\\": \\\"correct horse battery staple\\\"}'

    assert redact_secret_text(value) == r'payload={\\\"password\\\": \\\"[redacted]\\\"}'


def test_redact_secret_text_covers_nested_escaped_quotes() -> None:
    """Nested escaped quotes must not terminate credential redaction early."""
    value = r'payload={\"password\": \"foo \\\"bar\\\" baz\"}'

    redacted = redact_secret_text(value)

    assert "foo" not in redacted
    assert "bar" not in redacted
    assert "baz" not in redacted
    assert redacted == r'payload={\"password\": \"[redacted]\"}'


def test_redact_secret_text_preserves_fields_after_trailing_backslash() -> None:
    """A credential-ending backslash must not consume the following field."""
    value = r'password="secret\\"; next="keep"'

    assert redact_secret_text(value) == r'password="[redacted]"; next="keep"'


def test_redact_secret_text_uses_final_url_userinfo_delimiter() -> None:
    """A literal at sign inside userinfo must not leave a password suffix."""
    value = "https://user:correct@horse@example.com/resource"

    assert redact_secret_text(value) == (
        "https://user:[redacted]@example.com/resource"
    )


def test_redact_secret_text_preserves_username_only_url_userinfo() -> None:
    """Username-only URL userinfo is not secret credential material."""
    value = "https://alice@example.com/resource"

    assert redact_secret_text(value) == value


def test_redaction_handles_objects_with_broken_string_conversion() -> None:
    """A hostile value must not make the outbound firewall fail open."""

    class BrokenStringError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("broken string conversion")

    error = BrokenStringError()

    assert redact_secrets({"value": error}) == {
        "value": "[BrokenStringError omitted]"
    }
    assert redact_exception(error) == "BrokenStringError"
