# Tool Reference

MCP Assist exposes Home Assistant through MCP tools. The exact tool list depends
on shared server settings, per-profile overrides, provider capabilities, and
which Home Assistant integrations are installed.

## Core Tools

These are the main tools the assistant uses for Home Assistant discovery and
control.

| Tool | Purpose |
| --- | --- |
| `get_index` | Return the compact Smart Entity Index |
| `discover_entities` | Find exposed entities by area, domain, device class, state, name, or inferred type |
| `get_entity_details` | Read exact state, attributes, and metadata for one or more entities |
| `perform_action` | Execute supported Home Assistant actions against exposed entities |
| `run_script` | Run a Home Assistant script and return response data when available |
| `run_automation` | Trigger a Home Assistant automation manually |
| `list_areas` | List Home Assistant areas |
| `list_domains` | List entity domains available to the assistant |
| `set_conversation_state` | Track whether the assistant expects a follow-up |

## Device Tools

Device tools help when the user refers to a physical device instead of a single
entity.

| Tool | Purpose |
| --- | --- |
| `discover_devices` | Find Home Assistant devices and related entities |
| `get_device_details` | Read device metadata and associated exposed entities |

Use these for requests like "turn off the thermostat display" or "what entities
belong to the bedroom lamp device?"

## Assist Bridge Tools

Assist bridge tools expose additional Home Assistant Assist context when
enabled.

| Tool | Purpose |
| --- | --- |
| `list_assist_tools` | List available Assist bridge tools |
| `call_assist_tool` | Call an exposed Assist tool |
| `get_assist_prompt` | Read Assist prompt context |
| `get_assist_context_snapshot` | Inspect the current Assist context snapshot |

## Response-Service Read Tools

These tools read structured data from Home Assistant services that return
responses.

| Tool | Purpose |
| --- | --- |
| `get_calendar_events` | Read calendar events from exposed calendar entities |
| `list_response_services` | List response-capable Home Assistant services |
| `call_service_with_response` | Call a response-capable service directly |

For normal weather questions, prefer `get_weather_forecast`.

## Weather Forecast

| Tool | Purpose |
| --- | --- |
| `get_weather_forecast` | Find and summarize Home Assistant weather forecasts |

Requirements:

- A `weather.` entity exposed to the conversation assistant.
- The **Weather Forecast** tool family enabled.
- A weather integration that supports at least one forecast type.

## Recorder History

Recorder tools answer questions about past entity state.

| Tool | Purpose |
| --- | --- |
| `get_entity_history` | Read recent state history for an entity |
| `get_last_entity_event` | Find the last matching state event |
| `analyze_entity_history` | Count, summarize, or analyze state changes over a period |
| `get_entity_state_at_time` | Read an entity state at a point in time |

These tools require Home Assistant recorder data for the relevant entities and
time range.

## Calculator and Unit Conversion

Calculator tools are useful when exact arithmetic matters.

| Tool | Purpose |
| --- | --- |
| `add`, `subtract`, `multiply`, `divide` | Basic arithmetic |
| `sqrt`, `power`, `round_number` | Common math operations |
| `average`, `min_value`, `max_value` | Aggregate numbers |
| `evaluate_expression` | Evaluate a bounded math expression |
| `convert_unit` | Convert common units |

Calculator and unit conversion are separate tool families so profiles can expose
one without the other.

## Memory

Memory tools persist user-approved facts and preferences.

| Tool | Purpose |
| --- | --- |
| `remember_memory` | Store a memory with optional TTL |
| `recall_memories` | Search stored memories |
| `forget_memory` | Delete matching stored memories |

The assistant should use memory only when the user asks it to remember, recall,
or forget something. Memories are shared across MCP Assist profiles.

## Web Search and URL Reading

Web tools are optional and controlled by shared provider settings.

| Tool | Purpose |
| --- | --- |
| `search` | Search the web with DuckDuckGo, Brave Search, or SearXNG |
| `read_url` | Fetch and extract content from a specific URL |

Use Home Assistant-native tools first for local Home Assistant data such as
weather, calendars, history, and entity state.

## Music Assistant

Music Assistant tools are available when the Home Assistant Music Assistant
integration is installed and the tool family is enabled.

| Tool | Purpose |
| --- | --- |
| `list_music_assistant_players` | List Music Assistant media players |
| `play_music_assistant` | Play media through Music Assistant |
| `list_music_assistant_instances` | List configured Music Assistant instances |
| `search_music_assistant` | Search Music Assistant library content |
| `get_music_assistant_library` | Browse library content |
| `get_music_assistant_queue` | Inspect a player queue |

Use player names, areas, floors, labels, or entity IDs to narrow ambiguous
player requests.

## Image Tools

Image tools depend on provider and source support.

| Tool | Purpose |
| --- | --- |
| `analyze_image` | Ask the active multimodal model about a camera snapshot, image entity, URL, or local image |
| `get_image` | Return an image as an MCP image content block |
| `generate_image` | Generate an image when the active provider exposes compatible image generation |

Only use these when the provider and client can support the requested image
workflow.

## External Custom Tools

External custom tools are user-provided Python packages under:

```text
<home-assistant-config>/mcp-assist-tools
```

They are disabled by default and should only be enabled for packages you trust.
See [External Custom Tools](custom-tools.md).

## Tool Selection Tips

- Use `discover_entities` before `perform_action` unless the entity ID is known
  and unambiguous.
- Use `get_entity_details` when exact state or attributes matter.
- Use device tools when the user refers to physical hardware rather than one
  entity.
- Use Home Assistant-native reads before web search for local data.
- Keep optional tool families disabled unless a profile needs them.
