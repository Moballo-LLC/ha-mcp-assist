"""Built-in persisted memory tools."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import (
    CONF_MEMORY_DEFAULT_TTL_DAYS,
    CONF_MEMORY_MAX_ITEMS,
    CONF_MEMORY_MAX_TTL_DAYS,
    DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
    DEFAULT_MEMORY_MAX_ITEMS,
    DEFAULT_MEMORY_MAX_TTL_DAYS,
)
from ..memory_manager import MemoryManager
from .tool_runtime import HomeAssistantToolRuntime

_LOGGER = logging.getLogger(__name__)

MEMORY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_memory_categories",
        "description": (
            "List suggested memory categories and active counts. Use this before "
            "storing or filtering memories when the right category is unclear."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "remember_memory",
        "description": (
            "Store a short fact, preference, or instruction for later recall. "
            "Use this only when the user explicitly asks you to remember something. "
            "Memories persist across conversations and automatically expire after a TTL."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "The fact, preference, or instruction to store.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional short category. Prefer list_memory_categories "
                        "suggestions such as 'preference', 'routine', 'device_alias', "
                        "'automation_note', 'baseline', 'correction', 'maintenance', "
                        "or 'household'."
                    ),
                },
                "ttl_days": {
                    "type": "integer",
                    "description": (
                        "Optional retention time in days. If omitted, the shared "
                        "default TTL is used and capped by the shared maximum TTL."
                    ),
                    "minimum": 1,
                    "maximum": 3650,
                },
            },
            "required": ["memory"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall_memories",
        "description": (
            "Search active stored memories by query or category, or list recent "
            "memories when no query is given. Use this for requests like 'what "
            "do you remember about my coffee preference?'"
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search text to match against stored memory text.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category filter. Use list_memory_categories to "
                        "inspect suggested categories and active counts."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (default: 5).",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "forget_memory",
        "description": (
            "Delete one stored memory by id or by query/category match. Use this "
            "when the user asks you to forget or update something previously remembered."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Specific memory id to delete.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Search text to find a memory to delete when the id is not known."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category filter when deleting by query. Use "
                        "list_memory_categories if the category is unclear."
                    ),
                },
                "forget_all_matches": {
                    "type": "boolean",
                    "description": (
                        "Delete every matching memory instead of only the best match."
                    ),
                    "default": False,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
]
MEMORY_TOOL_NAMES = {str(tool["name"]) for tool in MEMORY_TOOL_DEFINITIONS}


class MemoryTool(HomeAssistantToolRuntime):
    """Expose persisted MCP Assist memory tools."""

    def __init__(self, hass) -> None:
        """Initialize the memory tool bundle."""
        super().__init__(hass)
        self.memory_manager = MemoryManager(hass)

    async def initialize(self) -> None:
        """Initialize persisted memory storage."""
        await self.memory_manager.async_initialize()

    async def async_shutdown(self) -> None:
        """Shut down persisted memory storage."""
        await self.memory_manager.async_shutdown()

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles the requested tool."""
        return tool_name in MEMORY_TOOL_NAMES

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return memory tool definitions."""
        return deepcopy(MEMORY_TOOL_DEFINITIONS)

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle memory tool calls."""
        if tool_name == "list_memory_categories":
            return await self.tool_list_memory_categories(arguments)
        if tool_name == "remember_memory":
            return await self.tool_remember_memory(arguments)
        if tool_name == "recall_memories":
            return await self.tool_recall_memories(arguments)
        if tool_name == "forget_memory":
            return await self.tool_forget_memory(arguments)
        raise ValueError(f"Unknown memory tool: {tool_name}")

    async def tool_list_memory_categories(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """List suggested memory categories and active counts."""
        del args

        self.publish_progress(
            "tool_start",
            "Listing memory categories",
            tool="list_memory_categories",
        )

        try:
            result = await self.memory_manager.list_categories()
        except Exception as err:
            _LOGGER.error("Failed to list memory categories: %s", _safe_log_value(err))
            return self._build_text_tool_result(
                f"Failed to list memory categories: {err}",
                is_error=True,
            )

        self.publish_progress(
            "tool_complete",
            "Memory categories listed",
            tool="list_memory_categories",
            count=result["total_count"],
        )

        lines = ["Suggested memory categories:"]
        for category in result["categories"]:
            lines.append(
                f"- {category['category']}: {category['description']} "
                f"({category['count']} active)"
            )

        custom_categories = result["custom_categories"]
        if custom_categories:
            lines.append("Custom categories already in use:")
            for category in custom_categories:
                lines.append(f"- {category['category']}: {category['count']} active")

        if result["uncategorized_count"]:
            lines.append(f"Uncategorized active memories: {result['uncategorized_count']}")
        lines.append(f"Total active memories: {result['total_count']}")

        return {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            **result,
        }

    async def tool_remember_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a persisted memory with TTL."""
        memory_text = " ".join(str(args.get("memory") or "").split()).strip()
        if not memory_text:
            return self._build_text_tool_result(
                "Memory text is required.",
                is_error=True,
            )

        ttl_days = args.get("ttl_days")
        category = args.get("category")
        self.publish_progress(
            "tool_start",
            "Storing memory",
            tool="remember_memory",
        )

        try:
            stored = await self.memory_manager.remember(
                memory_text,
                default_ttl_days=self._memory_default_ttl_days(),
                max_ttl_days=self._memory_max_ttl_days(),
                ttl_days=None if ttl_days is None else self._coerce_int_arg(
                    ttl_days,
                    default=self._memory_default_ttl_days(),
                    minimum=1,
                    maximum=self._memory_max_ttl_days(),
                ),
                category=category,
                max_items=self._memory_max_items(),
            )
        except Exception as err:
            _LOGGER.error("Failed to store memory: %s", _safe_log_value(err))
            return self._build_text_tool_result(
                f"Failed to store memory: {err}",
                is_error=True,
            )

        self.publish_progress(
            "tool_complete",
            "Memory stored",
            tool="remember_memory",
            memory_id=stored["id"],
        )

        expires_at = dt_util.parse_datetime(stored["expires_at"])
        expires_text = (
            self._format_relative_absolute_time(expires_at)
            if expires_at is not None
            else "later"
        )
        category_text = (
            f" Category: {stored['category']}." if stored.get("category") else ""
        )
        prune_text = (
            " "
            f"{stored['pruned_count']} old memories were pruned to stay within "
            "the configured limit."
            if stored.get("pruned_count")
            else ""
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Stored memory [{stored['id']}].{category_text} "
                        f"It expires {expires_text}.{prune_text}"
                    ),
                }
            ],
            "memory": stored,
        }

    async def tool_recall_memories(self, args: dict[str, Any]) -> dict[str, Any]:
        """Recall stored memories by query or category."""
        limit = self._coerce_int_arg(
            args.get("limit"),
            default=5,
            minimum=1,
            maximum=50,
        )
        query = args.get("query")
        category = args.get("category")

        self.publish_progress(
            "tool_start",
            "Searching stored memories",
            tool="recall_memories",
        )

        try:
            result = await self.memory_manager.recall(
                query=None if query is None else str(query),
                category=None if category is None else str(category),
                limit=limit,
            )
        except Exception as err:
            _LOGGER.error("Failed to recall memories: %s", _safe_log_value(err))
            return self._build_text_tool_result(
                f"Failed to recall memories: {err}",
                is_error=True,
            )

        items = result["items"]
        self.publish_progress(
            "tool_complete",
            "Memory recall complete",
            tool="recall_memories",
            count=result["returned_count"],
            total=result["total_found"],
        )

        if not items:
            return {
                "content": [{"type": "text", "text": "No active memories matched."}],
                "memories": [],
                "result_count": 0,
            }

        header = (
            f"Found {result['returned_count']} of {result['total_found']} "
            "active memories:"
            if result["remaining_count"] > 0
            else f"Found {result['returned_count']} active memories:"
        )
        lines = [header]
        for memory in items:
            expires_at = dt_util.parse_datetime(str(memory.get("expires_at") or ""))
            expires_text = (
                self._format_relative_absolute_time(expires_at)
                if expires_at is not None
                else "later"
            )
            category_text = f" [{memory['category']}]" if memory.get("category") else ""
            lines.append(
                f"- {memory['id']}{category_text}: {memory['text']} "
                f"(expires {expires_text})"
            )
        if result["remaining_count"] > 0:
            lines.append(
                f"{result['remaining_count']} more memories matched but were not shown."
            )

        return {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            "memories": items,
            "result_count": result["total_found"],
        }

    async def tool_forget_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Delete stored memories by id or query."""
        memory_id = args.get("memory_id")
        query = args.get("query")
        category = args.get("category")
        forget_all_matches = bool(args.get("forget_all_matches", False))

        self.publish_progress(
            "tool_start",
            "Deleting stored memory",
            tool="forget_memory",
        )

        try:
            result = await self.memory_manager.forget(
                memory_id=None if memory_id is None else str(memory_id),
                query=None if query is None else str(query),
                category=None if category is None else str(category),
                delete_all_matches=forget_all_matches,
            )
        except Exception as err:
            _LOGGER.error("Failed to forget memory: %s", _safe_log_value(err))
            return self._build_text_tool_result(
                f"Failed to forget memory: {err}",
                is_error=True,
            )

        self.publish_progress(
            "tool_complete",
            "Memory deletion complete",
            tool="forget_memory",
            deleted=result["deleted_count"],
        )

        if result["deleted_count"] == 0:
            return {
                "content": [{"type": "text", "text": "No matching memories were deleted."}],
                "deleted_count": 0,
                "deleted": [],
            }

        deleted = result["deleted"]
        lines = [f"Deleted {result['deleted_count']} memory item(s):"]
        for memory in deleted[:10]:
            category_text = f" [{memory['category']}]" if memory.get("category") else ""
            lines.append(f"- {memory['id']}{category_text}: {memory['text']}")
        if len(deleted) > 10:
            lines.append(f"{len(deleted) - 10} additional deleted memories were omitted.")

        return {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            "deleted_count": result["deleted_count"],
            "deleted": deleted,
        }

    def _get_shared_setting(self, key: str, default: Any = None) -> Any:
        """Get a shared setting from system settings, with server fallback support."""
        server = self._server
        get_setting = getattr(server, "_get_shared_setting", None) if server else None
        if callable(get_setting):
            return get_setting(key, default)

        from .. import get_system_entry

        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        return default

    def _memory_default_ttl_days(self) -> int:
        """Return the default TTL for new memories."""
        configured_max = self._memory_max_ttl_days()
        return self._coerce_int_arg(
            self._get_shared_setting(
                CONF_MEMORY_DEFAULT_TTL_DAYS,
                DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
            ),
            default=DEFAULT_MEMORY_DEFAULT_TTL_DAYS,
            minimum=1,
            maximum=configured_max,
        )

    def _memory_max_ttl_days(self) -> int:
        """Return the maximum TTL allowed for memories."""
        return self._coerce_int_arg(
            self._get_shared_setting(
                CONF_MEMORY_MAX_TTL_DAYS,
                DEFAULT_MEMORY_MAX_TTL_DAYS,
            ),
            default=DEFAULT_MEMORY_MAX_TTL_DAYS,
            minimum=1,
            maximum=3650,
        )

    def _memory_max_items(self) -> int:
        """Return the maximum number of memories to keep."""
        return self._coerce_int_arg(
            self._get_shared_setting(
                CONF_MEMORY_MAX_ITEMS,
                DEFAULT_MEMORY_MAX_ITEMS,
            ),
            default=DEFAULT_MEMORY_MAX_ITEMS,
            minimum=10,
            maximum=5000,
        )


def _safe_log_value(value: Any) -> str:
    """Return a compact log-safe representation."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r")
