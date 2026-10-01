"""Opt-in model-profile resolution for compatible OpenAI endpoints."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .const import SERVER_TYPE_OPENAI
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


@dataclass(frozen=True)
class ResolvedImageModelProfile:
    """An immutable image selection for one logical request."""

    model: str
    profile_id: str
    revision: str


ResolvedProfilesSnapshot = tuple[
    Any, ResolvedModelProfile | None, ResolvedImageModelProfile | None
]
MCP_PROFILE_REQUEST_HEADER = "X-MCP-Assist-Profile-Request"
REQUEST_RESOLVED_PROFILES: ContextVar[ResolvedProfilesSnapshot | None] = ContextVar(
    "mcp_assist_resolved_provider_profiles", default=None
)


def _validate_policy(payload: Any) -> None:
    """Validate both policy namespaces and the canonical full-policy digest."""
    if not isinstance(payload, dict) or type(payload.get("schemaVersion")) is not int:
        raise ValueError
    if payload["schemaVersion"] != 1:
        raise ValueError
    revision = payload["revision"]
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ValueError
    image_names = ("imageModels", "imageProfiles", "imageBindings")
    image_present = [name in payload for name in image_names]
    if any(image_present) and not all(image_present):
        raise ValueError
    namespaces = [("models", "profiles", "bindings", "resolvedProfiles", True)]
    if all(image_present):
        namespaces.append((*image_names, "resolvedImageProfiles", False))
    elif "resolvedImageProfiles" in payload:
        raise ValueError
    for model_name, profile_name, binding_name, resolved_name, text in namespaces:
        for name in (model_name, profile_name, binding_name, resolved_name):
            mapping = payload[name]
            minimum_size = 0 if name == binding_name else 1
            if not isinstance(mapping, dict) or not minimum_size <= len(mapping) <= 64:
                raise ValueError
            if any(not isinstance(key, str) or not _ID.fullmatch(key) for key in mapping):
                raise ValueError
        models, profiles, bindings = (payload[name] for name in (
            model_name, profile_name, binding_name))
        if any(not isinstance(model, str) or not _MODEL.fullmatch(model)
               for model in models.values()):
            raise ValueError
        for profile_id, profile in profiles.items():
            fields = {"label", "modelAlias", "reasoningEffort"} if text else {"label", "modelAlias"}
            if not isinstance(profile, dict) or set(profile) != fields:
                raise ValueError
            label = profile["label"]
            if not isinstance(label, str) or not label.strip() or len(label) > 60:
                raise ValueError
            if re.search(r"[\x00-\x1f\x7f-\x9f]", label):
                raise ValueError
            if profile["modelAlias"] not in models:
                raise ValueError
            resolved = {"model": models[profile["modelAlias"]]}
            if text:
                if profile["reasoningEffort"] not in _EFFORTS:
                    raise ValueError
                resolved["effort"] = profile["reasoningEffort"]
            if payload[resolved_name].get(profile_id) != resolved:
                raise ValueError
        if set(payload[resolved_name]) != set(profiles):
            raise ValueError
        if any(not isinstance(value, str) or value not in profiles for value in bindings.values()):
            raise ValueError
    names = ("schemaVersion", "models", "profiles", "bindings")
    if all(image_present):
        names += image_names
    canonical = json.dumps({name: payload[name] for name in names}, sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != revision:
        raise ValueError


def validate_model_profiles(
    payload: Any, reference: str = "", image_reference: str = ""
) -> tuple[ResolvedModelProfile | None, ResolvedImageModelProfile | None]:
    """Resolve optional text and image references from the same validated snapshot."""
    try:
        _validate_policy(payload)
        text = image = None
        for ref in (reference, image_reference):
            if ref and (not isinstance(ref, str) or not _ID.fullmatch(ref)):
                raise ValueError
        if reference:
            profile_id = payload["bindings"].get(reference, reference)
            profile = payload["profiles"][profile_id]
            text = ResolvedModelProfile(payload["models"][profile["modelAlias"]],
                                        profile["reasoningEffort"], profile_id, payload["revision"])
        if image_reference:
            profile_id = payload["imageBindings"].get(image_reference, image_reference)
            profile = payload["imageProfiles"][profile_id]
            image = ResolvedImageModelProfile(payload["imageModels"][profile["modelAlias"]],
                                              profile_id, payload["revision"])
        return text, image
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise ModelProfileResolutionError() from None


def validate_model_profile(payload: Any, reference: str) -> ResolvedModelProfile:
    """Resolve a text reference, including the optional image policy in its digest."""
    if not isinstance(reference, str) or not _ID.fullmatch(reference):
        raise ModelProfileResolutionError()
    return validate_model_profiles(payload, reference)[0]


async def async_resolve_model_profiles(
    runtime: ProviderRuntimeConfig, reference: str = "", image_reference: str = ""
) -> tuple[ResolvedModelProfile | None, ResolvedImageModelProfile | None]:
    """Fetch one authenticated policy without redirects or unbounded reads."""
    try:
        parts = urlsplit(runtime.base_url)
        if (
            runtime.server_type != SERVER_TYPE_OPENAI
            or parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or OpenAIProvider._is_official_openai_base_url(runtime.base_url)
            or not (reference or image_reference)
            or any(ref and not _ID.fullmatch(ref) for ref in (reference, image_reference))
        ):
            raise ModelProfileResolutionError()
        headers = build_provider_auth_headers(runtime.server_type, runtime.api_key)
        if not headers.get("Authorization"):
            raise ModelProfileResolutionError()
        url = OpenAIProvider.provider_endpoint(runtime.base_url, "model-profiles")
        timeout = aiohttp.ClientTimeout(total=min(max(runtime.timeout, 1), 15))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url, headers=headers, params={"include_images": "true"}, allow_redirects=False
            ) as response:
                if response.status != 200:
                    raise ModelProfileResolutionError()
                if response.content_length is not None and response.content_length > _MAX_BYTES:
                    raise ModelProfileResolutionError()
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > _MAX_BYTES:
                        raise ModelProfileResolutionError()
                return validate_model_profiles(json.loads(body), reference, image_reference)
    except Exception:
        raise ModelProfileResolutionError() from None


async def async_resolve_model_profile(
    runtime: ProviderRuntimeConfig, reference: str
) -> ResolvedModelProfile:
    """Resolve the existing text-only opt-in contract."""
    return (await async_resolve_model_profiles(runtime, reference))[0]
