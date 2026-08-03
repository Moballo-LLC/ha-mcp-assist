"""Bounded redaction for values crossing persistence and response boundaries."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
import re
from typing import Any
from urllib.parse import unquote_plus


REDACTION_MARKER = "[redacted]"
_CIRCULAR_REFERENCE_MARKER = "[circular reference]"

_SAFE_TOKEN_KEYS = frozenset(
    {
        "input_tokens",
        "max_tokens",
        "output_tokens",
        "token_count",
        "tokens_available",
        "tokens_used",
        "total_tokens",
    }
)
_SENSITIVE_KEY_NAMES = frozenset(
    {
        "account_key",
        "access_token",
        "api_key",
        "api_token",
        "auth",
        "auth_token",
        "authorization",
        "bearer",
        "bearer_token",
        "client_secret",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "csrf_token",
        "id_token",
        "mcp_bearer_token",
        "oauth_token",
        "openclaw_token",
        "passwd",
        "passphrase",
        "password",
        "private_key",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "secret_token",
        "session_token",
        "set_cookie",
        "set_cookies",
        "x_api_key",
    }
)
_SENSITIVE_KEY_SUFFIXES = tuple(f"_{key}" for key in _SENSITIVE_KEY_NAMES)
_SENSITIVE_KEY_COMPACT_NAMES = frozenset(
    key.replace("_", "") for key in _SENSITIVE_KEY_NAMES
)
_SENSITIVE_KEY_COMPACT_SUFFIXES = tuple(
    suffix.replace("_", "") for suffix in _SENSITIVE_KEY_SUFFIXES
)
_SAFE_TOKEN_COMPACT_KEYS = frozenset(key.replace("_", "") for key in _SAFE_TOKEN_KEYS)
_STATUS_CAPABLE_AUTH_KEYS = frozenset(
    {"auth", "authorization", "credential", "credentials"}
)
_STATUS_CAPABLE_AUTH_KEY_SUFFIXES = tuple(
    f"_{key}" for key in _STATUS_CAPABLE_AUTH_KEYS
)
_STATUS_CAPABLE_AUTH_COMPACT_KEYS = frozenset(
    key.replace("_", "") for key in _STATUS_CAPABLE_AUTH_KEYS
)
_STATUS_CAPABLE_AUTH_COMPACT_SUFFIXES = tuple(
    suffix.replace("_", "") for suffix in _STATUS_CAPABLE_AUTH_KEY_SUFFIXES
)
_NON_SECRET_AUTH_STATUS_VALUES = frozenset(
    {
        "allowed",
        "anonymous",
        "authenticated",
        "basic",
        "bearer",
        "denied",
        "digest",
        "disabled",
        "enabled",
        "failed",
        "false",
        "missing",
        "none",
        "not_applicable",
        "not_configured",
        "not_required",
        "null",
        "oauth",
        "oauth2",
        "optional",
        "required",
        "supported",
        "true",
        "unauthenticated",
        "unavailable",
        "unknown",
        "unsupported",
    }
)
_ROUTE_LIKE_PATH_KEY_NAMES = frozenset(
    {
        "auth",
        "authorization",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "passwd",
        "passphrase",
        "password",
        "secret",
        "set_cookie",
        "set_cookies",
    }
)
_UNAMBIGUOUS_PATH_KEY_NAMES = _SENSITIVE_KEY_NAMES - _ROUTE_LIKE_PATH_KEY_NAMES

_STRONG_FIELD_NAME_PARTS = "|".join(
    re.escape(name).replace("_", r"[\s_.-]?")
    for name in sorted(_SENSITIVE_KEY_NAMES, key=len, reverse=True)
)
_STRONG_FIELD_NAME_PATTERN = rf"(?:{_STRONG_FIELD_NAME_PARTS})"
_AMBIGUOUS_FIELD_NAME_PATTERN = r"(?:token)"
_SEPARATED_FIELD_PREFIX = r"(?:[a-z0-9]+[_.-])*"
_CAMEL_FIELD_PREFIX = r"(?-i:[A-Za-z0-9]*?(?=[A-Z]))"
_PREFIXED_STRONG_FIELD_NAME_PATTERN = (
    rf"(?:{_SEPARATED_FIELD_PREFIX}{_STRONG_FIELD_NAME_PATTERN}"
    rf"|{_CAMEL_FIELD_PREFIX}{_STRONG_FIELD_NAME_PATTERN})"
)
_PREFIXED_AMBIGUOUS_TOKEN_PATTERN = (
    rf"(?:{_SEPARATED_FIELD_PREFIX}{_AMBIGUOUS_FIELD_NAME_PATTERN}"
    rf"|{_CAMEL_FIELD_PREFIX}{_AMBIGUOUS_FIELD_NAME_PATTERN})"
)
_UNAMBIGUOUS_PATH_FIELD_NAME_PARTS = "|".join(
    re.escape(name).replace("_", r"[\s_.-]?")
    for name in sorted(_UNAMBIGUOUS_PATH_KEY_NAMES, key=len, reverse=True)
)
_PREFIXED_UNAMBIGUOUS_PATH_FIELD_NAME_PATTERN = (
    rf"(?:{_SEPARATED_FIELD_PREFIX}(?:{_UNAMBIGUOUS_PATH_FIELD_NAME_PARTS})"
    rf"|{_CAMEL_FIELD_PREFIX}(?:{_UNAMBIGUOUS_PATH_FIELD_NAME_PARTS}))"
)
_AMBIGUOUS_QUERY_NAME_PATTERN = (
    rf"(?:{_SEPARATED_FIELD_PREFIX}(?:token|signature|sig|key))"
)
_QUERY_SECRET_VALUE_PATTERN = r"""(?P<secret>[^&#;\s\\\"',}\]\[)<`]+)"""
_URL_USERINFO_RE = re.compile(
    r"(?i)(?P<scheme>\b[a-z][a-z0-9+.-]*://)"
    r"(?P<username>[^/?#\s:]*):(?P<password>[^/?#\s]+@)"
)
_SENSITIVE_HEADER_LINE_RE = re.compile(
    r"(?im)(?P<prefix>(?<![a-z0-9_-])(?P<field>"
    r"(?:authorization|proxy[\s_.-]?authorization|cookie|set[\s_.-]?cookie))"
    r"[ \t]*:[ \t]*)"
    r"(?P<secret>[^\r\n]*(?:\r?\n[ \t]+[^\r\n]*)*)"
)
_STRONG_QUERY_PARAM_RE = re.compile(
    rf"(?ix)(?P<prefix>[?&#;]"
    rf"(?P<field>{_PREFIXED_STRONG_FIELD_NAME_PATTERN})=)"
    rf"{_QUERY_SECRET_VALUE_PATTERN}"
)
_AMBIGUOUS_QUERY_PARAM_RE = re.compile(
    rf"(?ix)(?P<prefix>[?&#;]{_AMBIGUOUS_QUERY_NAME_PATTERN}=)"
    rf"{_QUERY_SECRET_VALUE_PATTERN}"
)
_PERCENT_ENCODED_QUERY_PARAM_RE = re.compile(
    r"(?i)(?P<separator>^|[^a-z0-9_.%+~-])"
    r"(?P<field>[a-z0-9_.+~-]*%[0-9a-f]{2}[a-z0-9_.%+~-]*)="
    rf"{_QUERY_SECRET_VALUE_PATTERN}"
)
_FORM_ENCODED_QUERY_PARAM_RE = re.compile(
    r"(?i)(?P<separator>^|[^a-z0-9_.%+~-])"
    r"(?P<field>[a-z0-9_.%~-]*\+[a-z0-9_.%+~-]*)="
    rf"{_QUERY_SECRET_VALUE_PATTERN}"
)
_AMBIGUOUS_QUOTED_LABEL_START_RE = re.compile(
    rf"(?ix)(?P<prefix>(?<![a-z0-9_])(?:\\*[\"'])?"
    rf"{_PREFIXED_AMBIGUOUS_TOKEN_PATTERN}(?:\\*[\"'])?"
    r"(?:\s+(?:is|provided|value))?\s*[:=]\s*)"
    r"(?P<quote>\\*[\"'])"
)
_STRONG_QUOTED_LABEL_START_RE = re.compile(
    rf"(?ix)(?P<prefix>(?<![a-z0-9_])(?:\\*[\"'])?"
    rf"(?P<field>{_PREFIXED_STRONG_FIELD_NAME_PATTERN})(?:\\*[\"'])?"
    r"(?:\s+(?:is|provided|value))?\s*[:=]\s*)"
    r"(?P<quote>\\*[\"'])"
)
_AMBIGUOUS_LABELED_SECRET_RE = re.compile(
    rf"(?ix)(?P<prefix>(?<![a-z0-9_])(?:\\*[\"'])?"
    rf"{_PREFIXED_AMBIGUOUS_TOKEN_PATTERN}(?:\\*[\"'])?"
    r"(?:\s+(?:is|provided|value))?\s*[:=]"
    r"(?!\s*(?:\\*[\"']|\[redacted\]))\s*)"
    r"(?P<secret>(?:(?:bearer|basic)\s+)?[^\s,;&}\]\[)\"']+)"
)
_STRONG_LABELED_SECRET_RE = re.compile(
    rf"(?ix)(?P<prefix>(?<![a-z0-9_])(?:\\*[\"'])?"
    rf"(?P<field>{_PREFIXED_STRONG_FIELD_NAME_PATTERN})(?:\\*[\"'])?"
    r"(?:\s+(?:is|provided|value))?\s*[:=]"
    r"(?!\s*(?:\\*[\"']|\[redacted\]))\s*)"
    r"(?P<secret>[^\n,;&}\]\[)\"']+)"
)
_AUTH_SCHEME_RE = re.compile(
    r"(?i)\b(?P<scheme>bearer|basic)\s+(?P<secret>[a-z0-9._~+/=-]{4,})"
)
_UNAMBIGUOUS_SECRET_PATH_RE = re.compile(
    rf"(?ix)(?P<prefix>/{_PREFIXED_UNAMBIGUOUS_PATH_FIELD_NAME_PATTERN}(?:/|=))"
    r"(?P<secret>[^/?#\s\"'&]+)"
)
_STRONG_SECRET_PATH_RE = re.compile(
    rf"(?ix)(?P<prefix>/{_PREFIXED_STRONG_FIELD_NAME_PATTERN}(?:/|=))"
    r"(?P<secret>[^/?#\s\"'&]+)"
)
_AMBIGUOUS_TOKEN_PATH_RE = re.compile(
    rf"(?ix)(?P<prefix>/{_PREFIXED_AMBIGUOUS_TOKEN_PATTERN}(?:/|=))"
    r"(?P<secret>[^/?#\s\"'&]+)"
)
_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9_-]{8,}|"
    r"AIza[A-Za-z0-9_-]{16,}|"
    r"ya29\.[A-Za-z0-9_-]{12,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"glpat-[A-Za-z0-9_-]{12,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r")\b"
)
_OPAQUE_HEX_RE = re.compile(r"(?i)^[0-9a-f]{32,}$")
_PRIVATE_KEY_PEM_RE = re.compile(
    r"-----BEGIN (?P<label>(?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?)-----"
    r".*?"
    r"-----END (?P=label)-----",
    re.DOTALL,
)


def _canonical_key(value: Any) -> str:
    """Return a separator-normalized key for sensitivity checks."""
    try:
        text = str(value)
    except Exception:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_sensitive_key(value: Any) -> bool:
    """Return whether a mapping key conventionally stores a credential."""
    canonical = _canonical_key(value)
    compact = canonical.replace("_", "")
    if canonical in _SAFE_TOKEN_KEYS or compact in _SAFE_TOKEN_COMPACT_KEYS:
        return False
    return (
        canonical in _SENSITIVE_KEY_NAMES
        or canonical.endswith(_SENSITIVE_KEY_SUFFIXES)
        or compact in _SENSITIVE_KEY_COMPACT_NAMES
        or compact.endswith(_SENSITIVE_KEY_COMPACT_SUFFIXES)
    )


def _is_non_secret_auth_status(key: Any, value: Any) -> bool:
    """Return whether an auth-like field contains known status metadata."""
    canonical_key = _canonical_key(key)
    compact_key = canonical_key.replace("_", "")
    is_status_key = (
        canonical_key in _STATUS_CAPABLE_AUTH_KEYS
        or canonical_key.endswith(_STATUS_CAPABLE_AUTH_KEY_SUFFIXES)
        or compact_key in _STATUS_CAPABLE_AUTH_COMPACT_KEYS
        or compact_key.endswith(_STATUS_CAPABLE_AUTH_COMPACT_SUFFIXES)
    )
    if not is_status_key:
        return False
    if isinstance(value, int) and not isinstance(value, bool):
        return 100 <= value <= 599
    if isinstance(value, float):
        return value.is_integer() and 100 <= value <= 599
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    return (
        _canonical_key(candidate) in _NON_SECRET_AUTH_STATUS_VALUES
        or bool(re.fullmatch(r"[1-5][0-9]{2}", candidate))
    )


def _looks_like_basic_credentials(value: str) -> bool:
    """Return whether a value decodes like an HTTP Basic userinfo payload."""
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return False
    return b":" in decoded


def _looks_like_bearer_token(value: str) -> bool:
    """Avoid treating ordinary prose after the word bearer as a credential."""
    if len(value) < 12:
        return False
    has_lower = any(char.islower() for char in value)
    has_upper = any(char.isupper() for char in value)
    has_digit = any(char.isdigit() for char in value)
    has_symbol = any(not char.isalnum() for char in value)
    return has_digit or has_symbol or (
        len(value) >= 20 and (has_lower or has_upper)
    )


def _redact_auth_scheme(match: re.Match[str]) -> str:
    """Redact an auth scheme only when the following value looks credential-like."""
    scheme = match.group("scheme")
    matched_secret = match.group("secret")
    secret = matched_secret.rstrip(".,;:!?")
    trailing_punctuation = matched_secret[len(secret) :]
    if scheme.casefold() == "basic":
        should_redact = _looks_like_basic_credentials(secret)
    else:
        should_redact = _looks_like_bearer_token(secret)
    if not should_redact:
        return match.group(0)
    return f"{scheme} {REDACTION_MARKER}{trailing_punctuation}"


def _looks_like_opaque_credential(value: str) -> bool:
    """Return whether an ambiguous token-like value looks credential-like."""
    candidate = value.strip()
    if candidate == REDACTION_MARKER:
        return True
    if _KNOWN_TOKEN_RE.search(candidate) or _OPAQUE_HEX_RE.fullmatch(candidate):
        return True
    if len(candidate) < 24 or any(char.isspace() for char in candidate):
        return False
    character_classes = sum(
        (
            any(char.islower() for char in candidate),
            any(char.isupper() for char in candidate),
            any(char.isdigit() for char in candidate),
            any(not char.isalnum() and not char.isspace() for char in candidate),
        )
    )
    return character_classes >= 3 or (
        len(candidate) >= 32 and character_classes >= 2
    ) or (
        len(candidate) >= 32 and len(set(candidate.casefold())) >= 6
    )


def _redact_ambiguous_labeled_secret(match: re.Match[str]) -> str:
    """Redact an ambiguous label only when its value resembles a credential."""
    if not _looks_like_opaque_credential(match.group("secret")):
        return match.group(0)
    quote = match.groupdict().get("quote")
    if quote:
        return f"{match.group('prefix')}{quote}{REDACTION_MARKER}{quote}"
    return f"{match.group('prefix')}{REDACTION_MARKER}"


def _find_quoted_value_end(text: str, start: int, delimiter: str) -> int | None:
    """Find a closing quote while respecting ordinary and escaped JSON quotes."""
    quote = delimiter[-1]
    position = start
    while (position := text.find(quote, position)) >= 0:
        slash_count = 0
        cursor = position - 1
        while cursor >= start and text[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        delimiter_slashes = len(delimiter) - 1
        escape_cycle = 2 * (delimiter_slashes + 1)
        if slash_count % escape_cycle == delimiter_slashes:
            return position - delimiter_slashes
        position += 1
    return None


def _redact_quoted_labeled_values(
    text: str,
    pattern: re.Pattern[str],
    *,
    ambiguous: bool,
) -> str:
    """Redact complete labeled quoted values without stopping at nested quotes."""
    parts: list[str] = []
    output_cursor = 0
    search_position = 0
    while match := pattern.search(text, search_position):
        delimiter = match.group("quote")
        value_end = _find_quoted_value_end(text, match.end(), delimiter)
        if value_end is None:
            candidate = text[match.end() :]
            if ambiguous and not _looks_like_opaque_credential(candidate):
                break
            if not ambiguous and (
                not candidate
                or _is_non_secret_auth_status(match.group("field"), candidate)
            ):
                break
            parts.extend((text[output_cursor : match.end()], REDACTION_MARKER))
            output_cursor = len(text)
            break

        candidate = text[match.end() : value_end]
        next_position = value_end + len(delimiter)
        if ambiguous and not _looks_like_opaque_credential(candidate):
            search_position = next_position
            continue
        if not ambiguous and (
            not candidate
            or _is_non_secret_auth_status(match.group("field"), candidate)
        ):
            search_position = next_position
            continue

        parts.extend(
            (
                text[output_cursor : match.end()],
                REDACTION_MARKER,
                delimiter,
            )
        )
        output_cursor = next_position
        search_position = next_position

    if not parts:
        return text
    parts.append(text[output_cursor:])
    return "".join(parts)


def _is_ambiguous_mapping_key(value: Any) -> bool:
    """Return whether a mapping key may hold either ordinary data or a secret."""
    canonical = _canonical_key(value)
    return canonical in {"sig", "signature", "token"} or canonical.endswith(
        ("_sig", "_signature", "_token")
    )


def _is_ambiguous_query_key(value: Any) -> bool:
    """Return whether a query key needs credential-shape confirmation."""
    canonical = _canonical_key(value)
    return canonical in {"key", "sig", "signature", "token"} or canonical.endswith(
        ("_key", "_sig", "_signature", "_token")
    )


def _looks_like_ambiguous_mapping_credential(value: Any) -> bool:
    """Apply conservative credential heuristics to ambiguous mapping fields."""
    return isinstance(value, str) and _looks_like_opaque_credential(value)


def _redact_strong_labeled_secret(match: re.Match[str]) -> str:
    """Preserve known auth metadata while redacting strong labeled values."""
    if _is_non_secret_auth_status(match.group("field"), match.group("secret")):
        return match.group(0)
    return f"{match.group('prefix')}{REDACTION_MARKER}"


def _redact_encoded_query_param(match: re.Match[str]) -> str:
    """Classify percent- or form-encoded query names before redacting values."""
    raw_field = match.group("field")
    if "+" not in raw_field and not re.search(r"(?i)%[0-9a-f]{2}", raw_field):
        return match.group(0)

    decoded_field = unquote_plus(raw_field)
    secret = match.group("secret")
    should_redact = _is_sensitive_key(decoded_field) and not _is_non_secret_auth_status(
        decoded_field,
        secret,
    )
    should_redact = should_redact or (
        _is_ambiguous_query_key(decoded_field)
        and _looks_like_opaque_credential(secret)
    )
    if not should_redact:
        return match.group(0)
    return (
        f"{match.group('separator')}{raw_field}="
        f"{REDACTION_MARKER}"
    )


def _has_potential_secret_value(value: Any) -> bool:
    """Return whether a value can contain non-empty credential material."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(
        value,
        (str, bytes, bytearray, Mapping, list, tuple, set, frozenset),
    ):
        try:
            return bool(value)
        except Exception:
            return True
    return True


def redact_secret_text(
    value: Any,
    *,
    _include_ambiguous_labels: bool = True,
) -> str:
    """Redact credential patterns from one string-like value."""
    try:
        text = str(value)
    except Exception:
        return f"[{type(value).__name__} omitted]"
    text = _PRIVATE_KEY_PEM_RE.sub(REDACTION_MARKER, text)
    text = _URL_USERINFO_RE.sub(
        lambda match: (
            f"{match.group('scheme')}{match.group('username')}:"
            f"{REDACTION_MARKER}@"
        ),
        text,
    )
    text = _SENSITIVE_HEADER_LINE_RE.sub(_redact_strong_labeled_secret, text)
    text = _PERCENT_ENCODED_QUERY_PARAM_RE.sub(
        _redact_encoded_query_param,
        text,
    )
    text = _FORM_ENCODED_QUERY_PARAM_RE.sub(
        _redact_encoded_query_param,
        text,
    )
    text = _STRONG_QUERY_PARAM_RE.sub(_redact_strong_labeled_secret, text)
    text = _redact_quoted_labeled_values(
        text,
        _STRONG_QUOTED_LABEL_START_RE,
        ambiguous=False,
    )
    text = _STRONG_LABELED_SECRET_RE.sub(_redact_strong_labeled_secret, text)
    if _include_ambiguous_labels:
        text = _AMBIGUOUS_QUERY_PARAM_RE.sub(
            _redact_ambiguous_labeled_secret,
            text,
        )
        text = _redact_quoted_labeled_values(
            text,
            _AMBIGUOUS_QUOTED_LABEL_START_RE,
            ambiguous=True,
        )
        text = _AMBIGUOUS_LABELED_SECRET_RE.sub(
            _redact_ambiguous_labeled_secret,
            text,
        )
    text = _AUTH_SCHEME_RE.sub(_redact_auth_scheme, text)
    text = _UNAMBIGUOUS_SECRET_PATH_RE.sub(
        lambda match: f"{match.group('prefix')}{REDACTION_MARKER}",
        text,
    )
    text = _STRONG_SECRET_PATH_RE.sub(_redact_ambiguous_labeled_secret, text)
    if _include_ambiguous_labels:
        text = _AMBIGUOUS_TOKEN_PATH_RE.sub(
            _redact_ambiguous_labeled_secret,
            text,
        )
    return _KNOWN_TOKEN_RE.sub(REDACTION_MARKER, text)


def redact_exception(error: BaseException) -> str:
    """Return a useful exception summary without raw credential material."""
    try:
        detail = redact_secret_text(str(error)).strip()
    except Exception:
        return type(error).__name__
    if detail:
        return f"{type(error).__name__}: {detail}"
    return type(error).__name__


def redact_secrets(
    value: Any,
    *,
    _include_ambiguous_labels: bool = True,
) -> Any:
    """Recursively redact secrets while preserving ordinary JSON-like shapes."""
    root: list[Any] = [None]
    tasks: list[tuple[Any, Any, Any, frozenset[int], bool]] = [
        (root, 0, value, frozenset(), False)
    ]

    while tasks:
        parent, slot, current, ancestors, force_redaction = tasks.pop()
        if force_redaction:
            parent[slot] = REDACTION_MARKER
            continue
        if current is None or isinstance(current, (bool, int, float)):
            parent[slot] = current
            continue
        if isinstance(current, str):
            parent[slot] = redact_secret_text(
                current,
                _include_ambiguous_labels=_include_ambiguous_labels,
            )
            continue

        if isinstance(current, Mapping):
            if id(current) in ancestors:
                parent[slot] = _CIRCULAR_REFERENCE_MARKER
                continue
            redacted: dict[str, Any] = {}
            parent[slot] = redacted
            child_ancestors = ancestors | {id(current)}
            child_tasks = []
            for key, item in current.items():
                safe_key = redact_secret_text(
                    key,
                    _include_ambiguous_labels=_include_ambiguous_labels,
                )
                has_value = _has_potential_secret_value(item)
                should_redact = (
                    _is_sensitive_key(key)
                    and has_value
                    and not _is_non_secret_auth_status(key, item)
                )
                should_redact = should_redact or (
                    _include_ambiguous_labels
                    and has_value
                    and _is_ambiguous_mapping_key(key)
                    and _looks_like_ambiguous_mapping_credential(item)
                )
                child_tasks.append(
                    (redacted, safe_key, item, child_ancestors, should_redact)
                )
            tasks.extend(reversed(child_tasks))
            continue

        if isinstance(current, (list, tuple, set, frozenset)):
            if id(current) in ancestors:
                parent[slot] = _CIRCULAR_REFERENCE_MARKER
                continue
            items = list(current)
            redacted_items: list[Any] = [None] * len(items)
            parent[slot] = redacted_items
            child_ancestors = ancestors | {id(current)}
            tasks.extend(
                (redacted_items, index, item, child_ancestors, False)
                for index, item in reversed(list(enumerate(items)))
            )
            continue

        try:
            parent[slot] = redact_secret_text(
                current,
                _include_ambiguous_labels=_include_ambiguous_labels,
            )
        except Exception:
            parent[slot] = f"[{type(current).__name__} omitted]"

    return root[0]


def redact_secrets_for_storage_migration(value: Any) -> Any:
    """Redact only unambiguous credentials when rewriting legacy storage."""
    return redact_secrets(value, _include_ambiguous_labels=False)
