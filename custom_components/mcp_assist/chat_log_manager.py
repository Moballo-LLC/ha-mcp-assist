"""Persistent chat log storage for MCP Assist."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DEFAULT_CHAT_LOG_MAX_ENTRIES, DOMAIN

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}_chat_logs"
_MAX_STRING_CHARS = 12000
_MAX_COLLECTION_ITEMS = 100
_MAX_DEPTH = 8
CHAT_LOG_PROJECTION_FULL = "full"
CHAT_LOG_PROJECTION_COMPACT = "compact"
CHAT_LOG_PROJECTION_RAW = "raw"
CHAT_LOG_PROJECTION_MODEL = "model"
CHAT_LOG_PROJECTIONS = frozenset(
    {
        CHAT_LOG_PROJECTION_FULL,
        CHAT_LOG_PROJECTION_COMPACT,
        CHAT_LOG_PROJECTION_RAW,
        CHAT_LOG_PROJECTION_MODEL,
    }
)
# Debounce persistence: chat logs are written once per conversation turn and
# the whole (multi-MB) file is rewritten each time, so coalesce rapid writes.
_SAVE_DELAY_SECONDS = 15


class ChatLogManager:
    """Manage opt-in persisted conversation logs for troubleshooting."""

    def __init__(self, hass) -> None:
        """Initialize the chat log manager."""
        self.hass = hass
        self._store: Store[dict[str, Any]] = Store(
            hass,
            _STORAGE_VERSION,
            _STORAGE_KEY,
        )
        self._lock = asyncio.Lock()
        self._loaded = False
        self._entries: list[dict[str, Any]] = []

    async def async_initialize(self) -> None:
        """Load persisted chat logs."""
        async with self._lock:
            await self._ensure_loaded_locked()

    async def async_shutdown(self) -> None:
        """Flush any pending debounced save before this manager is dropped.

        On an integration reload/unload the manager is discarded while a
        delayed save may still be scheduled. Without flushing, a new manager
        reloads the stale file and the old delayed callback can later overwrite
        it — losing recent logs or resurrecting cleared ones. Writing now also
        cancels the pending delayed save inside the Store.
        """
        async with self._lock:
            if self._loaded:
                await self._store.async_save(self._data_to_save())

    async def async_record(
        self,
        record: dict[str, Any],
        *,
        max_entries: int = DEFAULT_CHAT_LOG_MAX_ENTRIES,
    ) -> dict[str, Any]:
        """Persist one completed conversation record."""
        normalized = self._normalize_record(record)
        effective_max = self._positive_int(
            max_entries, default=DEFAULT_CHAT_LOG_MAX_ENTRIES
        )

        async with self._lock:
            await self._ensure_loaded_locked()
            self._entries.append(normalized)
            if len(self._entries) > effective_max:
                self._entries = self._entries[-effective_max:]
            await self._save_locked()

        return dict(normalized)

    async def async_list(
        self,
        *,
        limit: int | None = None,
        profile_entry_id: str | None = None,
        conversation_id: str | None = None,
        projection: str = CHAT_LOG_PROJECTION_FULL,
    ) -> list[dict[str, Any]]:
        """Return matching logs newest first."""
        normalized_projection = self.normalize_projection(projection)
        async with self._lock:
            await self._ensure_loaded_locked()
            entries = list(reversed(self._entries))

        if profile_entry_id:
            entries = [
                entry
                for entry in entries
                if entry.get("profile_entry_id") == profile_entry_id
            ]
        if conversation_id:
            entries = [
                entry
                for entry in entries
                if entry.get("conversation_id") == conversation_id
            ]
        if limit is not None:
            entries = entries[: self._positive_int(limit, default=len(entries))]

        return [
            self._project_entry(entry, normalized_projection)
            for entry in entries
        ]

    @staticmethod
    def normalize_projection(value: Any) -> str:
        """Return a supported chat-log response projection."""
        projection = str(value or CHAT_LOG_PROJECTION_FULL).strip().casefold()
        if projection not in CHAT_LOG_PROJECTIONS:
            choices = ", ".join(sorted(CHAT_LOG_PROJECTIONS))
            raise ValueError(f"projection must be one of: {choices}")
        return projection

    @classmethod
    def _project_entry(
        cls,
        entry: dict[str, Any],
        projection: str,
    ) -> dict[str, Any]:
        """Return one response-only view without mutating stored records."""
        projected = dict(entry)
        tools = entry.get("tools")
        if not isinstance(tools, list):
            return projected

        projected_tools: list[Any] = []
        for tool in tools:
            if not isinstance(tool, dict):
                projected_tools.append(tool)
                continue

            projected_tool = dict(tool)
            if projection == CHAT_LOG_PROJECTION_RAW:
                projected_tool.pop("llm_content", None)
            elif projection == CHAT_LOG_PROJECTION_MODEL:
                projected_tool.pop("result", None)
            elif projection == CHAT_LOG_PROJECTION_COMPACT:
                projected_tool = cls._compact_tool_entry(projected_tool)
            projected_tools.append(projected_tool)

        projected["tools"] = projected_tools
        return projected

    @staticmethod
    def _compact_tool_entry(tool: dict[str, Any]) -> dict[str, Any]:
        """Return tool-call metadata without raw or model-facing payloads."""
        compact = {
            key: tool[key]
            for key in (
                "id",
                "name",
                "started_at",
                "completed_at",
                "error",
            )
            if key in tool
        }
        arguments = tool.get("arguments")
        if isinstance(arguments, dict):
            compact["argument_keys"] = sorted(str(key) for key in arguments)
        result = tool.get("result")
        result_is_error = isinstance(result, dict) and (
            bool(result.get("isError")) or "error" in result
        )
        if "error" in tool or result_is_error:
            compact["status"] = "error"
        elif "completed_at" in tool:
            compact["status"] = "ok"
        else:
            compact["status"] = "in_progress"
        return compact

    async def async_clear(
        self,
        *,
        profile_entry_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, int]:
        """Clear matching logs and return deletion counts."""
        async with self._lock:
            await self._ensure_loaded_locked()
            original_count = len(self._entries)

            if not profile_entry_id and not conversation_id:
                self._entries = []
            else:
                self._entries = [
                    entry
                    for entry in self._entries
                    if not self._matches_filters(
                        entry,
                        profile_entry_id=profile_entry_id,
                        conversation_id=conversation_id,
                    )
                ]

            deleted_count = original_count - len(self._entries)
            if deleted_count:
                # Deletion must be durable before we return: a debounced save
                # would leave cleared (possibly sensitive) logs on disk for the
                # delay window, and a reload in that gap would resurrect them.
                await self._save_locked(immediate=True)

        return {"deleted_count": deleted_count, "remaining_count": len(self._entries)}

    async def _ensure_loaded_locked(self) -> None:
        """Load persisted storage on first use."""
        if self._loaded:
            return

        stored = await self._store.async_load()
        raw_entries = stored.get("entries", []) if isinstance(stored, dict) else []
        self._entries = [
            normalized
            for item in raw_entries
            if (normalized := self._normalize_loaded_record(item)) is not None
        ]
        self._loaded = True

    async def _save_locked(self, *, immediate: bool = False) -> None:
        """Persist the current chat log list.

        Record writes are debounced (``async_delay_save``) to avoid rewriting
        the whole file every turn; Home Assistant flushes any pending delayed
        save on shutdown, so nothing is lost. Destructive clears pass
        ``immediate=True`` to make the deletion durable before returning.
        """
        if immediate:
            await self._store.async_save(self._data_to_save())
        else:
            self._store.async_delay_save(self._data_to_save, _SAVE_DELAY_SECONDS)

    def _data_to_save(self) -> dict[str, Any]:
        """Return the payload for the store (read at flush time)."""
        return {"entries": list(self._entries)}

    def _normalize_loaded_record(self, item: Any) -> dict[str, Any] | None:
        """Normalize a loaded record from storage."""
        if not isinstance(item, dict):
            return None

        record_id = self._normalize_text(item.get("id"))
        created_at = self._normalize_text(item.get("created_at"))
        if not record_id or not created_at:
            return None

        return self._json_safe(item)

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize a new record before storage."""
        public_record = {
            key: value for key, value in record.items() if not str(key).startswith("_")
        }
        normalized = self._json_safe(public_record)
        normalized.setdefault("id", uuid4().hex[:12])
        normalized.setdefault("created_at", dt_util.utcnow().isoformat())
        normalized.setdefault("tools", [])
        return normalized

    def _json_safe(
        self,
        value: Any,
        _depth: int = 0,
        _ancestors: frozenset[int] = frozenset(),
    ) -> Any:
        """Convert values to a bounded JSON-serializable shape.

        Guards against runaway or self-referential structures: nesting is
        capped at _MAX_DEPTH, and a value that appears in its own ancestry
        (a cycle) is replaced with a marker instead of recursing forever.
        """
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate_text(value)

        if isinstance(value, (dict, list, tuple, set)):
            if id(value) in _ancestors:
                return "[circular reference]"
            if _depth >= _MAX_DEPTH:
                return f"[max depth {_MAX_DEPTH} exceeded]"
            child_ancestors = _ancestors | {id(value)}
            child_depth = _depth + 1

            if isinstance(value, dict):
                safe_dict: dict[str, Any] = {}
                for index, (key, item_value) in enumerate(value.items()):
                    if index >= _MAX_COLLECTION_ITEMS:
                        safe_dict["_truncated_items"] = len(value) - _MAX_COLLECTION_ITEMS
                        break
                    safe_dict[str(key)] = self._json_safe(
                        item_value, child_depth, child_ancestors
                    )
                return safe_dict

            sequence = list(value)
            items = [
                self._json_safe(item, child_depth, child_ancestors)
                for item in sequence[:_MAX_COLLECTION_ITEMS]
            ]
            if len(sequence) > _MAX_COLLECTION_ITEMS:
                items.append(
                    {"_truncated_items": len(sequence) - _MAX_COLLECTION_ITEMS}
                )
            return items

        return self._truncate_text(str(value))

    @staticmethod
    def _truncate_text(value: str) -> str:
        """Bound stored strings to keep logs reviewable."""
        if len(value) <= _MAX_STRING_CHARS:
            return value
        remaining = len(value) - _MAX_STRING_CHARS
        return f"{value[:_MAX_STRING_CHARS]}... [truncated {remaining} chars]"

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        """Normalize optional string values."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _positive_int(value: Any, *, default: int) -> int:
        """Coerce a value to a positive integer."""
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return max(1, default)

    @staticmethod
    def _matches_filters(
        entry: dict[str, Any],
        *,
        profile_entry_id: str | None,
        conversation_id: str | None,
    ) -> bool:
        """Return whether a record matches the provided filters."""
        if profile_entry_id and entry.get("profile_entry_id") != profile_entry_id:
            return False
        if conversation_id and entry.get("conversation_id") != conversation_id:
            return False
        return True
