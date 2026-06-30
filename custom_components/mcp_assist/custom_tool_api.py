"""Public API for user-defined MCP Assist custom tools.

This module is intentionally stable and documented so local tools placed under
the Home Assistant config directory can import it directly:

    <home-assistant-config>/mcp-assist-tools/<tool_id>/tool.py

Tool-package metadata should live in:

    <home-assistant-config>/mcp-assist-tools/<tool_id>/mcp_tool.json
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, TypeVar

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from .const import CUSTOM_TOOL_SHARED_DIRECTORY, CUSTOM_TOOLS_DIRECTORY, DOMAIN

_T = TypeVar("_T")

_CURRENT_EXTERNAL_TOOL_CALL_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "mcp_assist_external_tool_call_context",
    default=None,
)
_EXTERNAL_SHARED_MODULE_CACHE: dict[str, tuple[int, str]] = {}


@dataclass(frozen=True)
class MCPAssistCustomToolManifest:
    """Validated metadata for a user-defined custom tool package."""

    schema_version: int
    tool_id: str
    name: str
    description: str
    version: str
    entrypoint: str
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    prompt_append_file: str | None = None


class MCPAssistExternalTool:
    """Base class for user-defined MCP Assist custom tool packages.

    Custom tool packages should subclass this class and implement:
    - get_tool_definitions()
    - handle_call()

    Optional hooks:
    - initialize()
    - async_shutdown()
    - get_prompt_instructions()
    - get_settings_schema()
    """

    def __init__(
        self,
        hass: HomeAssistant,
        manifest: MCPAssistCustomToolManifest,
        tool_dir: Path,
    ) -> None:
        """Initialize the external tool instance."""
        self.hass = hass
        self.manifest = manifest
        self.tool_dir = tool_dir
        self.settings: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the tool package.

        Override when the tool needs lightweight setup such as reading a local
        config file or preparing cached state.
        """

    async def async_shutdown(self) -> None:
        """Clean up any tool resources before MCP Assist shuts down."""

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions exposed by this package."""
        raise NotImplementedError

    def get_settings_schema(self) -> dict[str, Any]:
        """Return an optional shared-settings schema for this package.

        When provided, MCP Assist will load package settings from
        `<config>/mcp-assist-tool-settings/<tool_id>.json`, validate them, and
        make the merged shared/profile settings available via `get_settings()`.
        """

        return {}

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this package handles the given tool name."""
        return tool_name in {
            str(tool.get("name") or "")
            for tool in self.get_tool_definitions()
        }

    async def handle_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a tool call and return standard MCP content."""
        raise NotImplementedError

    def get_prompt_instructions(self) -> str:
        """Return optional prompt instructions for the LLM.

        Keep this short and procedural. The loader will append the returned
        text to the technical instructions only when external custom tools are
        enabled.
        """

        return ""

    def get_settings(self) -> dict[str, Any]:
        """Return the effective settings for the current tool call."""
        context = self.get_call_context()
        settings = context.get("settings") if isinstance(context, dict) else None
        if isinstance(settings, dict):
            return dict(settings)
        return dict(self.settings)

    def get_shared_settings(self) -> dict[str, Any]:
        """Return the package's shared settings."""
        return dict(self.settings)

    def get_profile_settings(self) -> dict[str, Any]:
        """Return the current profile override settings, if any."""
        context = self.get_call_context()
        profile_settings = (
            context.get("profile_settings") if isinstance(context, dict) else None
        )
        if isinstance(profile_settings, dict):
            return dict(profile_settings)
        return {}

    def get_call_context(self) -> dict[str, Any]:
        """Return MCP Assist metadata for the current tool call."""
        return get_external_tool_call_context()

    async def call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke any MCP Assist tool using the current profile call context."""
        return await call_mcp_tool(
            self.hass,
            tool_name,
            arguments,
            context=self.get_call_context(),
        )

    async def analyze_image(
        self,
        *,
        question: str,
        camera_entity_id: str | None = None,
        image_path: str | None = None,
        image_url: str | None = None,
        include_image: bool = False,
        detail: str = "auto",
    ) -> dict[str, Any]:
        """Analyze an image using the active MCP Assist profile provider."""
        return await analyze_image(
            self.hass,
            question=question,
            camera_entity_id=camera_entity_id,
            image_path=image_path,
            image_url=image_url,
            include_image=include_image,
            detail=detail,
            context=self.get_call_context(),
        )

    async def get_image(
        self,
        *,
        camera_entity_id: str | None = None,
        image_path: str | None = None,
        image_url: str | None = None,
    ) -> dict[str, Any]:
        """Fetch an image through MCP Assist's validated image-source handling."""
        return await get_image(
            self.hass,
            camera_entity_id=camera_entity_id,
            image_path=image_path,
            image_url=image_url,
            context=self.get_call_context(),
        )

    async def async_run_recorder_job(
        self,
        job: Callable[[Any], _T],
    ) -> _T:
        """Run a read-only recorder job in the recorder executor."""
        return await async_run_recorder_job(self.hass, job)

    async def async_recorder_query(
        self,
        sql: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a bounded read-only recorder SQL query."""
        return await async_recorder_query(
            self.hass,
            sql,
            parameters,
            limit=limit,
        )

    def ok(
        self,
        text: Any,
        *,
        structured_content: dict[str, Any] | None = None,
        content: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a standard successful MCP text result."""
        return mcp_text_result(
            text,
            structured_content=structured_content,
            content=content,
        )

    def error(
        self,
        message: Any,
        *,
        structured_content: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a standard MCP error result."""
        return mcp_error_result(
            message,
            structured_content=structured_content,
        )

    def _set_loaded_settings(self, settings: dict[str, Any]) -> None:
        """Internal helper used by the loader to attach validated settings."""
        self.settings = dict(settings or {})

    def _push_call_context(self, context: dict[str, Any] | None) -> Token:
        """Internal helper used by the loader to scope call metadata."""
        return _CURRENT_EXTERNAL_TOOL_CALL_CONTEXT.set(dict(context or {}))

    def _reset_call_context(self, token: Token) -> None:
        """Internal helper used by the loader to restore prior call metadata."""
        _CURRENT_EXTERNAL_TOOL_CALL_CONTEXT.reset(token)


def load_external_shared_module(
    caller_file: str | Path,
    module_name: str,
) -> Any:
    """Load a reusable helper module from `<config>/mcp-assist-tools/__shared__`.

    This lets multiple narrow external tool packages share code without custom
    `sys.path` shims.
    """

    caller_path = Path(caller_file).resolve()
    tools_root = _find_external_tools_root(caller_path)
    shared_root = (tools_root / CUSTOM_TOOL_SHARED_DIRECTORY).resolve()
    if not shared_root.is_dir():
        raise ImportError(
            f"Shared helper directory does not exist: {shared_root}"
        )

    relative_path = Path(*module_name.split("."))
    module_path = (shared_root / relative_path).with_suffix(".py")
    if not module_path.is_file():
        module_path = shared_root / relative_path / "__init__.py"
    if not module_path.is_file():
        raise ImportError(
            f"Unable to resolve shared helper module {module_name!r} in {shared_root}"
        )

    try:
        module_path.resolve().relative_to(shared_root)
    except ValueError as err:
        raise ImportError("Shared helper module must stay within __shared__") from err

    module_path = module_path.resolve()
    module_cache_key = str(module_path)
    module_mtime_ns = module_path.stat().st_mtime_ns
    cached = _EXTERNAL_SHARED_MODULE_CACHE.get(module_cache_key)
    if cached is not None:
        cached_mtime_ns, cached_module_name = cached
        if cached_mtime_ns == module_mtime_ns:
            module = sys.modules.get(cached_module_name)
            if module is not None:
                return module
        else:
            sys.modules.pop(cached_module_name, None)

    path_hash = hashlib.sha1(module_cache_key.encode("utf-8")).hexdigest()[:12]
    unique_module_name = (
        f"mcp_assist_external_tools.shared.{module_name.replace('.', '_')}.{path_hash}_{module_mtime_ns}"
    )
    module = sys.modules.get(unique_module_name)
    if module is not None:
        _EXTERNAL_SHARED_MODULE_CACHE[module_cache_key] = (
            module_mtime_ns,
            unique_module_name,
        )
        return module

    spec = importlib.util.spec_from_file_location(unique_module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import shared helper from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(unique_module_name, None)
        raise
    _EXTERNAL_SHARED_MODULE_CACHE[module_cache_key] = (
        module_mtime_ns,
        unique_module_name,
    )
    return module


def get_external_tool_call_context() -> dict[str, Any]:
    """Return the current scoped external-tool call context."""
    context = _CURRENT_EXTERNAL_TOOL_CALL_CONTEXT.get()
    if isinstance(context, dict):
        return dict(context)
    return {}


async def call_mcp_tool(
    hass: HomeAssistant,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke any MCP Assist tool through the shared MCP server."""
    server = hass.data.get(DOMAIN, {}).get("shared_mcp_server")
    if server is None:
        raise RuntimeError("Shared MCP server is not running.")

    handle_tool_call = getattr(server, "handle_tool_call", None)
    if not callable(handle_tool_call):
        raise RuntimeError(
            "This MCP Assist build does not support tool invocation from external packages."
        )

    return await handle_tool_call(
        {
            "name": tool_name,
            "arguments": dict(arguments or {}),
            "context": dict(context or {}),
        }
    )


def mcp_text_result(
    text: Any,
    *,
    is_error: bool = False,
    structured_content: dict[str, Any] | None = None,
    content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a standard MCP tool result with a text content block."""
    result_content = [{"type": "text", "text": str(text)}]
    if content:
        result_content.extend(content)

    result: dict[str, Any] = {
        "content": result_content,
        "isError": bool(is_error),
    }
    if structured_content is not None:
        result["structuredContent"] = structured_content
    return result


def mcp_error_result(
    message: Any,
    *,
    structured_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard MCP error result."""
    return mcp_text_result(
        message,
        is_error=True,
        structured_content=structured_content,
    )


def mcp_json_result(
    payload: Any,
    *,
    text: str | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    """Build a text-plus-structured MCP result for JSON-serializable data."""
    result_text = text
    if result_text is None:
        result_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return mcp_text_result(
        result_text,
        is_error=is_error,
        structured_content=payload if isinstance(payload, dict) else {"result": payload},
    )


def mcp_image_result(
    image_bytes: bytes,
    mime_type: str,
    *,
    text: str = "Fetched image.",
    structured_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard MCP image result from raw image bytes."""
    import base64

    return mcp_text_result(
        text,
        structured_content=structured_content,
        content=[
            {
                "type": "image",
                "mimeType": str(mime_type or "image/jpeg"),
                "data": base64.b64encode(image_bytes).decode("ascii"),
            }
        ],
    )


def entity_snapshot(hass: HomeAssistant, entity_id: str) -> dict[str, Any] | None:
    """Return a compact, JSON-friendly snapshot for a Home Assistant entity."""
    state = hass.states.get(entity_id)
    if state is None:
        return None

    return {
        "entity_id": state.entity_id,
        "state": state.state,
        "name": state.name,
        "unit_of_measurement": state.attributes.get("unit_of_measurement"),
        "device_class": state.attributes.get("device_class"),
        "last_changed": _format_datetime_value(state.last_changed),
        "last_updated": _format_datetime_value(state.last_updated),
    }


def format_entity_state(
    hass: HomeAssistant,
    entity_id: str,
    *,
    include_last_changed: bool = False,
    missing_text: str | None = None,
) -> str:
    """Format a Home Assistant entity's current state for tool output."""
    state = hass.states.get(entity_id)
    if state is None:
        return missing_text if missing_text is not None else f"{entity_id}: unavailable"
    return format_state(state, include_last_changed=include_last_changed)


def format_state(
    state: State,
    *,
    include_last_changed: bool = False,
) -> str:
    """Format a Home Assistant State object for compact tool output."""
    label = str(state.name or state.entity_id)
    value = str(state.state)
    unit = state.attributes.get("unit_of_measurement")
    if unit:
        value = f"{value} {unit}"

    rendered = f"{label}: {value}"
    if include_last_changed:
        rendered += f" (last changed {format_datetime(state.last_changed)})"
    return rendered


def format_datetime(
    value: datetime | date | str | None,
    *,
    fallback: str = "unknown",
) -> str:
    """Format a date/time value in Home Assistant local time for tool output."""
    formatted = _format_datetime_value(value)
    return formatted if formatted else fallback


def format_relative_time(
    value: datetime | None,
    *,
    now: datetime | None = None,
    fallback: str = "unknown",
) -> str:
    """Return a compact relative age such as '5 minutes ago'."""
    if value is None:
        return fallback

    current = now or dt_util.utcnow()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    compared = value
    if compared.tzinfo is None:
        compared = compared.replace(tzinfo=timezone.utc)

    seconds = int((current - compared).total_seconds())
    suffix = "ago"
    if seconds < 0:
        seconds = abs(seconds)
        suffix = "from now"

    for unit_name, unit_seconds in (
        ("day", 86400),
        ("hour", 3600),
        ("minute", 60),
    ):
        if seconds >= unit_seconds:
            amount = seconds // unit_seconds
            return f"{amount} {unit_name}{'' if amount == 1 else 's'} {suffix}"
    return f"{seconds} second{'' if seconds == 1 else 's'} {suffix}"


async def analyze_image(
    hass: HomeAssistant,
    *,
    question: str,
    camera_entity_id: str | None = None,
    image_path: str | None = None,
    image_url: str | None = None,
    include_image: bool = False,
    detail: str = "auto",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze an image using the active MCP Assist profile provider."""
    arguments = _image_source_arguments(
        camera_entity_id=camera_entity_id,
        image_path=image_path,
        image_url=image_url,
    )
    arguments["question"] = str(question or "").strip()
    arguments["detail"] = str(detail or "auto")
    if include_image:
        arguments["include_image"] = True
    return await call_mcp_tool(
        hass,
        "analyze_image",
        arguments,
        context=context,
    )


async def get_image(
    hass: HomeAssistant,
    *,
    camera_entity_id: str | None = None,
    image_path: str | None = None,
    image_url: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch an image through MCP Assist's validated image-source handling."""
    return await call_mcp_tool(
        hass,
        "get_image",
        _image_source_arguments(
            camera_entity_id=camera_entity_id,
            image_path=image_path,
            image_url=image_url,
        ),
        context=context,
    )


async def async_run_recorder_job(
    hass: HomeAssistant,
    job: Callable[[Any], _T],
) -> _T:
    """Run a read-only recorder job in Home Assistant's recorder executor."""
    get_instance, session_scope = _recorder_helpers()
    recorder_instance = get_instance(hass)

    def _execute_job() -> _T:
        with session_scope(hass=hass, read_only=True) as session:
            return job(session)

    return await recorder_instance.async_add_executor_job(_execute_job)


async def async_recorder_query(
    hass: HomeAssistant,
    sql: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run a bounded read-only recorder SQL query in the recorder executor."""
    normalized_sql = str(sql or "").strip()
    if not _is_read_only_sql(normalized_sql):
        raise ValueError("Recorder helper only accepts read-only SELECT/WITH queries.")
    if limit is not None and limit < 1:
        raise ValueError("Recorder query limit must be at least 1.")

    def _query(session: Any) -> list[dict[str, Any]]:
        from sqlalchemy import text

        mapped_rows = session.execute(
            text(normalized_sql),
            dict(parameters or {}),
        ).mappings()
        rows = mapped_rows.fetchmany(limit) if limit is not None else mapped_rows.all()
        normalized_rows = [dict(row) for row in rows]
        return normalized_rows

    return await async_run_recorder_job(hass, _query)


def _recorder_helpers() -> tuple[Callable[[HomeAssistant], Any], Callable[..., Any]]:
    """Return recorder helpers across supported Home Assistant releases."""
    try:
        recorder_helpers = importlib.import_module("homeassistant.helpers.recorder")
        return recorder_helpers.get_instance, recorder_helpers.session_scope
    except (AttributeError, ImportError):
        recorder_component = importlib.import_module("homeassistant.components.recorder")
        recorder_util = importlib.import_module("homeassistant.components.recorder.util")
        return recorder_component.get_instance, recorder_util.session_scope


def _find_external_tools_root(caller_path: Path) -> Path:
    for candidate in [caller_path.parent, *caller_path.parents]:
        if candidate.name == CUSTOM_TOOLS_DIRECTORY:
            return candidate
    raise ImportError(
        f"Unable to locate {CUSTOM_TOOLS_DIRECTORY!r} from {caller_path}"
    )


def _image_source_arguments(
    *,
    camera_entity_id: str | None,
    image_path: str | None,
    image_url: str | None,
) -> dict[str, Any]:
    arguments = {
        key: value
        for key, value in {
            "camera_entity_id": camera_entity_id,
            "image_path": image_path,
            "image_url": image_url,
        }.items()
        if value
    }
    if len(arguments) != 1:
        raise ValueError(
            "Provide exactly one image source: camera_entity_id, image_path, or image_url."
        )
    return arguments


def _format_datetime_value(value: datetime | date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        local_value = dt_util.as_local(value)
        return local_value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    text_value = str(value).strip()
    return text_value or None


def _is_read_only_sql(sql: str) -> bool:
    stripped = sql.lstrip(" \t\r\n(").lower()
    first_word = stripped.split(None, 1)[0] if stripped else ""
    return first_word in {"select", "with"}
