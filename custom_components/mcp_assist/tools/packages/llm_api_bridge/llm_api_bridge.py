"""Built-in third-party Home Assistant LLM API bridge tools."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time
import inspect
import json
import logging
from typing import Any

import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from ....const import (
    CONF_ENABLE_LLM_API_BRIDGE,
    CONF_LLM_API_ALLOWLIST,
    DEFAULT_ENABLE_LLM_API_BRIDGE,
    DEFAULT_LLM_API_ALLOWLIST,
    DOMAIN,
    parse_llm_api_allowlist,
)
from ...tool_runtime import HomeAssistantToolRuntime

_LOGGER = logging.getLogger(__name__)

_API_ID_DESCRIPTION = (
    "Third-party Home Assistant LLM API id shown by list_llm_apis, "
    "for example 'llm_intents'."
)

LLM_API_BRIDGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_llm_apis",
        "description": (
            "List registered third-party Home Assistant LLM APIs and whether "
            "each is allowlisted for MCP Assist. The built-in Assist API is "
            "handled by Assist Bridge tools and is not included here."
        ),
        "llmDescription": "List allowlisted third-party Home Assistant LLM APIs.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_llm_api_tools",
        "description": (
            "List the tools exposed by an allowlisted third-party Home "
            "Assistant LLM API. Use this before calling call_llm_api_tool."
        ),
        "llmDescription": (
            "List tools from an allowlisted third-party Home Assistant LLM API."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "api_id": {
                    "type": "string",
                    "description": _API_ID_DESCRIPTION,
                },
            },
            "required": ["api_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "call_llm_api_tool",
        "description": (
            "Call a tool exposed by an allowlisted third-party Home Assistant "
            "LLM API. Arguments must match the schema returned by "
            "list_llm_api_tools."
        ),
        "llmDescription": (
            "Call a tool on an allowlisted third-party Home Assistant LLM API."
        ),
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "api_id": {
                    "type": "string",
                    "description": _API_ID_DESCRIPTION,
                },
                "tool_name": {
                    "type": "string",
                    "description": (
                        "Exact tool name exposed by the selected API. Use "
                        "list_llm_api_tools first if unsure."
                    ),
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the selected tool.",
                    "additionalProperties": True,
                },
            },
            "required": ["api_id", "tool_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_llm_api_prompt",
        "description": (
            "Get the prompt text exposed by an allowlisted third-party Home "
            "Assistant LLM API for compatibility checks and debugging."
        ),
        "llmDescription": "Get prompt text from an allowlisted third-party LLM API.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "api_id": {
                    "type": "string",
                    "description": _API_ID_DESCRIPTION,
                },
            },
            "required": ["api_id"],
            "additionalProperties": False,
        },
    },
]
LLM_API_BRIDGE_TOOL_NAMES = {
    str(tool["name"]) for tool in LLM_API_BRIDGE_TOOL_DEFINITIONS
}


class LLMApiBridgeTool(HomeAssistantToolRuntime):
    """Expose allowlisted third-party Home Assistant LLM API tools."""

    async def initialize(self) -> None:
        """Initialize the bridge tool package."""
        return None

    async def async_shutdown(self) -> None:
        """Shut down the bridge tool package."""
        return None

    def handles_tool(self, tool_name: str) -> bool:
        """Return whether this bundle handles the requested tool."""
        return tool_name in LLM_API_BRIDGE_TOOL_NAMES

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return bridge tool definitions."""
        return deepcopy(LLM_API_BRIDGE_TOOL_DEFINITIONS)

    async def handle_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle third-party LLM API bridge tool calls."""
        if tool_name == "list_llm_apis":
            return await self.tool_list_llm_apis(arguments)
        if tool_name == "list_llm_api_tools":
            return await self.tool_list_llm_api_tools(arguments)
        if tool_name == "call_llm_api_tool":
            return await self.tool_call_llm_api_tool(arguments)
        if tool_name == "get_llm_api_prompt":
            return await self.tool_get_llm_api_prompt(arguments)
        raise ValueError(f"Unknown LLM API bridge tool: {tool_name}")

    async def tool_list_llm_apis(self, args: dict[str, Any]) -> dict[str, Any]:
        """List registered third-party Home Assistant LLM APIs."""
        del args

        allowed_api_ids = self._allowed_llm_api_ids()
        allowed_api_id_set = set(allowed_api_ids)
        registered_apis = sorted(
            (
                api
                for api in llm.async_get_apis(self.hass)
                if api.id != llm.LLM_API_ASSIST
            ),
            key=lambda api: (api.name.casefold(), api.id),
        )
        registered_api_ids = {api.id for api in registered_apis}
        apis_payload = [
            {
                "id": api.id,
                "name": api.name,
                "allowed": api.id in allowed_api_id_set,
            }
            for api in registered_apis
        ]
        allowed_count = sum(1 for api in apis_payload if api["allowed"])
        missing_allowed_api_ids = sorted(allowed_api_id_set - registered_api_ids)
        payload = {
            "enabled": self._llm_api_bridge_enabled(),
            "allowed_api_ids": list(allowed_api_ids),
            "missing_allowed_api_ids": missing_allowed_api_ids,
            "api_count": len(apis_payload),
            "allowed_api_count": allowed_count,
            "apis": apis_payload,
        }
        header = (
            f"Found {len(apis_payload)} third-party Home Assistant LLM APIs; "
            f"{allowed_count} are allowlisted for MCP Assist."
        )
        if missing_allowed_api_ids:
            header += (
                " Some configured API ids are not currently registered: "
                + ", ".join(missing_allowed_api_ids)
                + "."
            )

        return self._json_text_result(header, payload)

    async def tool_list_llm_api_tools(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """List tools exposed by an allowlisted third-party LLM API."""
        llm_api = await self._get_llm_api_bridge_api_instance(args.get("api_id"))
        tools_payload = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": self._format_llm_tool_input_schema(
                    tool, llm_api.custom_serializer
                ),
            }
            for tool in llm_api.tools
        ]
        payload = {
            "api_id": llm_api.api.id,
            "api_name": llm_api.api.name,
            "tool_count": len(tools_payload),
            "tools": tools_payload,
        }
        header = (
            f"Found {len(tools_payload)} tools from third-party LLM API "
            f"{llm_api.api.name} ({llm_api.api.id})."
        )
        return self._json_text_result(header, payload)

    async def tool_call_llm_api_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        """Call a tool exposed by an allowlisted third-party LLM API."""
        tool_name = str(args.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("tool_name is required")

        tool_arguments = args.get("arguments") or {}
        if not isinstance(tool_arguments, dict):
            raise ValueError("arguments must be an object")

        llm_api = await self._get_llm_api_bridge_api_instance(args.get("api_id"))
        tool_response = await self._call_llm_api_tool(
            llm_api, tool_name, tool_arguments
        )
        serialized_response = self._serialize_service_response_value(tool_response)

        text_parts = [
            f"✅ Called third-party LLM API `{llm_api.api.id}` tool `{tool_name}`."
        ]
        summary_lines = self._build_assist_tool_response_summary(serialized_response)
        if summary_lines:
            text_parts.append("")
            text_parts.extend(summary_lines)
        text_parts.append("")
        text_parts.append("Response:")
        text_parts.append(json.dumps(serialized_response, indent=2, ensure_ascii=False))

        return {"content": [{"type": "text", "text": "\n".join(text_parts)}]}

    async def tool_get_llm_api_prompt(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get prompt text for an allowlisted third-party LLM API."""
        llm_api = await self._get_llm_api_bridge_api_instance(args.get("api_id"))
        prompt = llm_api.api_prompt or ""
        description = (
            f"Prompt for third-party Home Assistant LLM API "
            f"{llm_api.api.name} ({llm_api.api.id})"
        )
        text = description + (
            "\n\n" + prompt if prompt else "\n\nNo prompt text provided."
        )
        return {"content": [{"type": "text", "text": text}]}

    def _json_text_result(self, header: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a text result with an appended JSON payload."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": header
                    + "\n\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False),
                }
            ]
        }

    def _get_shared_setting(self, key: str, default: Any = None) -> Any:
        """Get a shared setting from system settings, with server fallback support."""
        server = self._server
        get_setting = getattr(server, "_get_shared_setting", None) if server else None
        if callable(get_setting):
            return get_setting(key, default)

        from .... import get_system_entry

        system_entry = get_system_entry(self.hass)
        if system_entry:
            value = system_entry.options.get(key, system_entry.data.get(key))
            if value is not None:
                return value

        return default

    def _llm_api_bridge_enabled(self) -> bool:
        """Return whether third-party LLM API bridge tools are enabled."""
        return bool(
            self._get_shared_setting(
                CONF_ENABLE_LLM_API_BRIDGE,
                DEFAULT_ENABLE_LLM_API_BRIDGE,
            )
        )

    def _allowed_llm_api_ids(self) -> tuple[str, ...]:
        """Return third-party LLM API ids allowlisted for MCP Assist."""
        return tuple(
            api_id
            for api_id in parse_llm_api_allowlist(
                self._get_shared_setting(
                    CONF_LLM_API_ALLOWLIST,
                    DEFAULT_LLM_API_ALLOWLIST,
                )
            )
            if api_id != llm.LLM_API_ASSIST
        )

    def _validate_llm_api_bridge_api_id(self, api_id: object) -> str:
        """Validate a requested third-party LLM API id against the allowlist."""
        validated_api_id = str(api_id or "").strip()
        if not validated_api_id:
            raise ValueError("api_id is required")

        if validated_api_id == llm.LLM_API_ASSIST:
            raise HomeAssistantError(
                "The built-in Assist API is available through Assist Bridge tools, "
                "not the third-party LLM API bridge."
            )

        allowed_api_ids = self._allowed_llm_api_ids()
        if validated_api_id not in allowed_api_ids:
            raise HomeAssistantError(
                f"LLM API '{validated_api_id}' is not allowlisted for MCP Assist. "
                "Add its API id to the shared LLM API allowlist before calling it."
            )

        return validated_api_id

    async def _get_llm_api_bridge_api_instance(
        self,
        api_id: object,
    ) -> llm.APIInstance:
        """Get an allowlisted third-party Home Assistant LLM API instance."""
        validated_api_id = self._validate_llm_api_bridge_api_id(api_id)
        try:
            return await llm.async_get_api(
                self.hass,
                validated_api_id,
                self._create_assist_llm_context(),
            )
        except HomeAssistantError as err:
            raise HomeAssistantError(
                f"LLM API '{validated_api_id}' is not currently available: {err}"
            ) from err

    def _create_assist_llm_context(self) -> llm.LLMContext:
        """Create an LLM context compatible with Home Assistant LLM APIs."""
        kwargs: dict[str, Any] = {
            "platform": DOMAIN,
            "context": Context(),
            "language": "*",
            "assistant": conversation.DOMAIN,
            "device_id": None,
        }
        if "user_prompt" in inspect.signature(llm.LLMContext).parameters:
            kwargs["user_prompt"] = ""
        return llm.LLMContext(**kwargs)

    def _format_llm_tool_input_schema(
        self,
        tool: llm.Tool,
        custom_serializer,
    ) -> dict[str, Any]:
        """Convert a Home Assistant LLM tool schema to JSON schema for inspection."""
        try:
            input_schema = convert(
                tool.parameters, custom_serializer=custom_serializer
            )
        except Exception as err:
            _LOGGER.debug(
                "Failed to convert Home Assistant LLM tool schema for %s: %s",
                _safe_log_value(tool.name),
                _safe_log_value(err),
            )
            return {"type": "object", "properties": {}}

        return (
            input_schema
            if isinstance(input_schema, dict)
            else {"type": "object", "properties": {}}
        )

    async def _call_llm_api_tool(
        self,
        llm_api: llm.APIInstance,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a Home Assistant LLM API tool safely."""
        tool_input = self._create_llm_tool_input(
            tool_name,
            arguments,
            external=True,
        )

        try:
            result = await llm_api.async_call_tool(tool_input)
        except (HomeAssistantError, vol.Invalid) as err:
            raise HomeAssistantError(
                "Error calling Home Assistant LLM API "
                f"'{llm_api.api.id}' tool '{tool_name}': {err}"
            ) from err

        if not isinstance(result, dict):
            return {"result": self._serialize_service_response_value(result)}
        return result

    def _create_llm_tool_input(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        external: bool,
    ) -> llm.ToolInput:
        """Create a ToolInput while supporting HA versions without external."""
        kwargs = {"tool_name": tool_name, "tool_args": arguments}
        try:
            return llm.ToolInput(**kwargs, external=external)
        except TypeError as err:
            if "external" not in str(err):
                raise
            return llm.ToolInput(**kwargs)

    def _build_assist_tool_response_summary(self, response: Any) -> list[str]:
        """Build a concise summary for a Home Assistant LLM tool response."""
        if not isinstance(response, dict):
            return []

        lines: list[str] = []

        speech = response.get("speech")
        if isinstance(speech, dict):
            plain_speech = speech.get("plain")
            if isinstance(plain_speech, dict) and plain_speech.get("speech"):
                lines.append("Summary:")
                lines.append(f"- Speech: {plain_speech['speech']}")

        data = response.get("data")
        if isinstance(data, dict) and (
            "success" in data or "failed" in data or "targets" in data
        ):
            if not lines:
                lines.append("Summary:")
            detail_parts = []
            if "success" in data:
                detail_parts.append(f"success={data['success']}")
            if "failed" in data:
                detail_parts.append(f"failed={data['failed']}")
            targets = data.get("targets")
            if isinstance(targets, list):
                detail_parts.append(f"targets={len(targets)}")
            if detail_parts:
                lines.append("- Result: " + ", ".join(detail_parts))

        response_type = response.get("response_type")
        if response_type and not lines:
            lines.append("Summary:")
            lines.append(f"- Response type: {response_type}")

        return lines

    def _serialize_service_response_value(self, value: Any) -> Any:
        """Serialize HA LLM API response data to JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): self._serialize_service_response_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_service_response_value(item) for item in value]
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.hex()
        return str(value)


def _safe_log_value(value: Any) -> str:
    """Return a compact log-safe representation for schema conversion errors."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r")
