"""Opt-in model-profile resolution for compatible OpenAI endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .llm_providers.openai import OpenAIProvider
from .provider_runtime import ProviderRuntimeConfig, build_provider_auth_headers

_MAX_BYTES = 1024 * 1024
_ID = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,119}\Z")
_REVISION = re.compile(r"[a-f0-9]{64}\Z")
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max", "ultra"})


class ModelProfileResolutionError(ValueError):
    """Safe failure that must stop a request before model generation."""

    def __init__(self) -> None:
        super().__init__(
            "The configured model profile could not be resolved. "
            "Check the compatible endpoint's model-profile policy and authentication."
        )


@dataclass(frozen=True)
class ResolvedModelProfile:
    """An immutable selection used for one logical conversation request."""

    model: str
    effort: str
    profile_id: str
    revision: str


def validate_model_profile(payload: Any, reference: str) -> ResolvedModelProfile:
    """Validate the policy and its selected resolved model/effort pair."""
    try:
        if not isinstance(payload, dict) or type(payload.get("schemaVersion")) is not int:
            raise ValueError
        if payload["schemaVersion"] != 1 or not _ID.fullmatch(reference):
            raise ValueError
        revision = payload["revision"]
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise ValueError
        for name in ("models", "profiles", "bindings", "resolvedProfiles"):
            mapping = payload[name]
            if not isinstance(mapping, dict) or not 1 <= len(mapping) <= 64:
                raise ValueError
            if any(not isinstance(key, str) or not _ID.fullmatch(key) for key in mapping):
                raise ValueError
        models, profiles, bindings = (payload[name] for name in ("models", "profiles", "bindings"))
        if any(not isinstance(model, str) or not _MODEL.fullmatch(model) for model in models.values()):
            raise ValueError
        for profile in profiles.values():
            if not isinstance(profile, dict) or set(profile) != {
                "label", "modelAlias", "reasoningEffort"
            }:
                raise ValueError
            label = profile["label"]
            if not isinstance(label, str) or not label.strip() or len(label) > 60:
                raise ValueError
            if re.search(r"[\x00-\x1f\x7f-\x9f]", label):
                raise ValueError
            if profile["modelAlias"] not in models or profile["reasoningEffort"] not in _EFFORTS:
                raise ValueError
        if any(not isinstance(value, str) or value not in profiles for value in bindings.values()):
            raise ValueError
        policy = {name: payload[name] for name in ("schemaVersion", "models", "profiles", "bindings")}
        canonical = json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != revision:
            raise ValueError
        profile_id = bindings.get(reference, reference)
        profile = profiles[profile_id]
        model = models[profile["modelAlias"]]
        effort = profile["reasoningEffort"]
        if payload["resolvedProfiles"].get(profile_id) != {"model": model, "effort": effort}:
            raise ValueError
        return ResolvedModelProfile(model, effort, profile_id, revision)
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise ModelProfileResolutionError() from None


async def async_resolve_model_profile(
    runtime: ProviderRuntimeConfig, reference: str
) -> ResolvedModelProfile:
    """Fetch one authenticated policy without redirects or unbounded reads."""
    try:
        parts = urlsplit(runtime.base_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or OpenAIProvider._is_official_openai_base_url(runtime.base_url)
            or not _ID.fullmatch(reference)
        ):
            raise ModelProfileResolutionError()
        headers = build_provider_auth_headers(runtime.server_type, runtime.api_key)
        if not headers.get("Authorization"):
            raise ModelProfileResolutionError()
        url = OpenAIProvider.provider_endpoint(runtime.base_url, "model-profiles")
        timeout = aiohttp.ClientTimeout(total=min(max(runtime.timeout, 1), 15))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, allow_redirects=False) as response:
                if response.status != 200:
                    raise ModelProfileResolutionError()
                if response.content_length is not None and response.content_length > _MAX_BYTES:
                    raise ModelProfileResolutionError()
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > _MAX_BYTES:
                        raise ModelProfileResolutionError()
                return validate_model_profile(json.loads(body), reference)
    except Exception:
        raise ModelProfileResolutionError() from None
