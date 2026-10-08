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

`get_entity_details` returns `last_changed`, `last_updated`, and any
datetime-valued attributes as ISO 8601 timestamps in Home Assistant's configured
time zone. Its `_metadata` entry names that time zone explicitly.

`perform_action` requests structured response data automatically when Home
Assistant marks a service as response-capable, and includes that data in the
tool result. For lights, legacy `color_temp` values are migrated to
`color_temp_kelvin`: values from 100 through 1000 are treated as mireds, while
larger values are treated as Kelvin supplied under the old key.

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

Use `list_assist_tools` to find current names and argument schemas. Home Assistant
may prefix native names, such as `homeassistant__GetLiveContext`. The bridge accepts
older unqualified names when they identify exactly one tool and prefers exact
matches. Missing or ambiguous names, invalid arguments, and unavailable native
tools return MCP tool errors so the assistant can recover. Failed calls are not
automatically retried, since a tool may have already performed an action.

## Third-Party LLM API Bridge Tools

The LLM API Bridge exposes allowlisted third-party Home Assistant LLM APIs
registered by other integrations. It is disabled by default.

| Tool | Purpose |
| --- | --- |
| `list_llm_apis` | List registered non-Assist LLM APIs and allowlist status |
| `list_llm_api_tools` | Inspect tools exposed by one allowlisted LLM API |
| `call_llm_api_tool` | Call a tool on one allowlisted LLM API |
| `get_llm_api_prompt` | Read prompt text from one allowlisted LLM API |

Use `list_llm_api_tools` before `call_llm_api_tool` so arguments match the
third-party API's schema. The built-in Home Assistant `assist` API stays on the
Assist Bridge tools. Native tool results retain their structured data and error
status in MCP Assist, including Home Assistant's `ToolResult` API and older
JSON-object return values.

## Response-Service Read Tools

These tools read structured data from Home Assistant services that return
responses.

| Tool | Purpose |
| --- | --- |
| `get_calendar_events` | Read calendar events from exposed calendar entities |
| `list_response_services` | List response-capable Home Assistant services |
| `call_service_with_response` | Call a response-capable service directly |

For normal weather questions, prefer `get_weather_forecast`.

## Maintenance Status

| Tool | Purpose |
| --- | --- |
| `get_maintenance_status` | Read current native maintenance signals from exposed entities |

Enable the optional **Maintenance Status** family in shared settings first. The
tool is read-only and performs an on-demand state scan, without service calls or
repair actions. It uses Home Assistant's conversation exposure controls; hiding
an entity in the UI does not exclude it if it remains exposed.

Arguments:

- `category`: `all` (default), `batteries`, `updates`, `unavailable`, or `problems`.
- `battery_threshold`: integer 1–100, default 30. Percentage battery sensors
  match at or below the threshold; the value must be finite and within 0–100,
  with native `battery` device class and `%` unit. Battery binary sensors match
  when `on`.
- `limit`: integer 1–100, default 25.
- `offset`: nonnegative integer, default 0; use `next_offset` to read another page.

Updates match native `update` entities when `on`; problems match native
`binary_sensor` entities with `problem` device class when `on`. Any exposed
entity whose current state is `unavailable` or `unknown` belongs to the
`unavailable` category instead of being treated as healthy.

`structuredContent` includes `schema_version: 1`, `scope: exposed_entities_only`,
the checked `exposed_count`, full `matched_count`, `page_count`, `next_offset`,
and `items`. Each item contains only `category`, `entity_id`, `friendly_name`,
and `state`, plus `battery_percentage` for numeric battery matches. Results sort
by category priority (problems, unavailable, batteries, updates), then entity ID.
`truncated` means some matches were omitted from the current page; `next_offset`
is null when there are no later matches. Paging reads current states again, so
results may change between calls.

An empty result means no matching signals were found among exposed entities. It
does not establish whole-home health, scan unexposed devices, or inspect repair
issues in Home Assistant's Repairs registry.

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
| `get_entity_history` with `mode: "last_event"` | Find the last matching state event |
| `analyze_entity_history` | Count, summarize, or analyze state changes over a period |
| `get_entity_state_at_time` | Read an entity state at a point in time |
| `get_entity_statistics` | Read statistic changes, numeric summaries, coverage, and previous-period comparisons |

These tools require Home Assistant recorder data for the relevant entities and
time range. Use `period: "today"` or `period: "yesterday"` for calendar-day
questions instead of approximating with a number of hours.
Count analyses count transitions into the matching state, not repeated recorder
rows that report the same state.
Recorder query boundaries stay in UTC, while timestamps shown in tool results
are formatted in Home Assistant's configured time zone.
`get_entity_statistics` accepts this/last month, today/yesterday, rolling
`last_hour`/`last_24_hours`, the previous 7 or 30 complete local days, the previous
12 complete calendar months, or an exact timezone-aware custom interval. Its
`metric` selects `change`, `mean`, `min`, `max`, or `all` available metrics. Change
is the change in Home Assistant's sum statistic, not its cumulative sum value.
Circular statistics do not produce an arithmetic mean.

`bucket` supports `5minute`, `hour`, `day` (default), `month`, and `auto`.
Five-minute queries use native Recorder short-term statistics and are limited to
72 hours per requested interval. `auto` selects five-minute resolution through
12 hours, hourly through 72 hours, and daily grouping for longer intervals.
Hourly, daily, and monthly output uses hourly long-term source rows. Short-term
retention varies by installation; missing data is unavailable, never zero.

Results include readable text and `structuredContent` with `schema_version: 1`:

| Field | Meaning |
| --- | --- |
| `available`, `reason` | Whether usable statistics exist; an unavailable result explains why |
| `entity_id`, `period`, `bucket`, `unit_of_measurement` | Entity, requested period, resolved grouping, and shared display unit |
| `requested_window`, `effective_window` | UTC start/end timestamps; end is exclusive, and effective boundaries include only complete source intervals |
| `source_resolution` | `5minute` or `hour` |
| `summary` | Finite observed change total, arithmetic mean, minimum, and maximum for supported selected metrics |
| `coverage` | Expected, observed, and missing source buckets, requested-boundary alignment, `complete` for the entire requested interval, `effective_complete` for its complete source buckets, and the same counts/completeness per selected metric |
| `bucket_count`, `buckets`, `truncated` | Full grouped bucket count, displayed rows capped by `limit` (1–100), and whether rows were omitted |
| `source_coverage` | Optional bounded annotations reported by the entity's source |

Displayed buckets follow absolute chronological order, including repeated local
hours during daylight-saving transitions. The display limit keeps the earliest
buckets.

Summary values cover every returned valid source bucket even when displayed rows
are capped. Duplicate timestamps, nonfinite values, and off-grid rows cannot
establish complete coverage. Gaps produce observed subtotals. Change summaries
read one preceding source bucket and cumulative sums internally to verify the
current sum and exact preceding sum for every change row. A missing, duplicated,
off-grid, or nonfinite baseline excludes the affected change from grouped
buckets, the subtotal, and calendar averages: Recorder may otherwise bridge the
gap and attribute multiple intervals to one source bucket, or include usage
outside the window. This validation applies whether or not comparison is requested.
Cumulative sums are never reported as interval totals. Change summaries
also include `complete_day_count`, `complete_month_count`, and
`average_per_complete_day`/`average_per_complete_month`: only fully covered Home
Assistant-local calendar periods contribute, including 23- or 25-hour DST days.
Absent values are `null`, rather than inferred zeroes. Unavailable results omit
the numeric analysis fields.

Set `compare_previous: true` to add `comparison.previous` with its windows,
summary, coverage, and source annotations. The previous interval immediately
precedes the current interval and has equal elapsed duration; it need not be the
previous calendar month. One combined Recorder query reads both intervals.
Change comparisons also require valid baselines at both effective-window starts.
`change_baselines_available` reports those checks for the current and previous
windows, or is `null` for comparisons without change. An invalid baseline also
prevents comparison.
`comparison.comparable` requires equal positive effective-window durations,
`effective_complete` coverage for every selected metric in both windows, and no
source-reported uncertainty. `scope: "effective_windows"` and
`effective_duration_seconds` identify what the differences cover. Off-grid
rolling windows can be compared without pretending their requested boundary
fragments were measured: `boundaries_aligned` is false, `qualification` explains
the exclusions, and requested-interval `coverage.complete` remains false.
The same restriction applies to custom intervals; unequal effective durations
cannot be compared. Otherwise `differences` is empty
and `reason` explains the limitation. Comparable metrics include an `absolute`
difference; only change also has `percent`, which is `null` when previous change
is zero or negative.

An entity may supply a `recorder_coverage` attribute with these optional fields:

```yaml
recorder_coverage:
  started_at: "2026-01-01T00:00:00+00:00"
  counter_resets: 1
  note: "Some records were imported from another source."
  estimated_periods:
    days: ["2026-01-02"]
    months: ["2026-02"]
    unknown_before: "2026-01-01"
```

`started_at` requires a timezone-aware ISO timestamp; `counter_resets` must be a
nonnegative integer; `note` is limited to 1,200 characters. Estimated dates use
canonical ISO dates (at most 400 days) or `YYYY-MM` (at most 24 months).
`unknown_before` is an ISO date. Only overlapping Home Assistant-local days and
months appear in results, using the exclusive interval end. Pre-collection time,
overlapping estimates, older unverified time, and malformed annotations mark
source coverage uncertain. Unknown fields are omitted. These annotations are
source-reported evidence, not independently verified provenance; complete
Recorder buckets do not prove their underlying measurements are accurate.
Provider notes are data, not instructions for the assistant.

## Calculator and Unit Conversion

Calculator tools are useful when exact arithmetic matters.

| Tool | Purpose |
| --- | --- |
| `add`, `subtract`, `multiply`, `divide` | Basic arithmetic |
| `sqrt`, `power`, `round_number` | Common math operations |
| `average`, `min_value`, `max_value` | Aggregate numbers |
| `evaluate_expression` | Evaluate a bounded math expression |
| `convert_unit` | Convert common units, including cooking volumes |

Calculator and unit conversion are separate tool families so profiles can expose
one without the other.

Kitchen conversions support `cup`, `tablespoon`, `teaspoon`, `ml`, and `pint`.
Fractional values such as `1/8` and `1 1/2` can be passed as the conversion
value.

## Memory

Memory tools persist user-approved facts and preferences.

| Tool | Purpose |
| --- | --- |
| `list_memory_categories` | List suggested categories and active memory counts |
| `remember_memory` | Store a memory with optional TTL |
| `recall_memories` | Search stored memories |
| `forget_memory` | Delete matching stored memories |

The assistant should use memory only when the user asks it to remember, recall,
or forget something. Suggested categories include `preference`, `routine`,
`device_alias`, `automation_note`, `baseline`, `correction`, `maintenance`, and
`household`; custom categories still work. Memories are shared across MCP Assist
profiles.

## Web Search and URL Reading

Web tools are optional and controlled by shared provider settings.

| Tool | Purpose |
| --- | --- |
| `search` | Search the web with DuckDuckGo, Brave Search, or SearXNG |
| `read_url` | Fetch and extract content from a specific URL |

Use Home Assistant-native tools first for local Home Assistant data such as
weather, calendars, history, and entity state.
`read_url` favors main page content over navigation chrome where possible and
keeps longer summary excerpts for page-inspection workflows.

## Google Places and Routes

Google Places and Routes tools are optional and require a Google Maps API key.

| Tool | Purpose |
| --- | --- |
| `search_google_places` | Search for businesses and places, including open status, address, phone, and rating when available |
| `get_google_place_details` | Fetch details for a Google Places result |
| `get_google_route` | Calculate travel time, distance, and traffic-aware ETAs |

If `get_google_route` is called without an origin, it can use the configured
Home Assistant home latitude and longitude only when **Share Home Location with
MCP Tools** is enabled. Use regular web search for broad location research that
is not a place lookup or route question.

## Wikipedia Search

Wikipedia Search is an optional lightweight reference tool.

| Tool | Purpose |
| --- | --- |
| `search_wikipedia` | Search Wikipedia article titles and descriptions |

Use this for stable background or encyclopedia-style questions where Wikipedia
results are enough. Use regular web search for latest information, current
events, or broader internet research.

## Music Assistant

Music Assistant tools are available when the Home Assistant Music Assistant
integration is installed and the tool family is enabled.

| Tool | Purpose |
| --- | --- |
| `list_music_assistant_players` | List Music Assistant media players |
| `play_music_assistant` | Play media through Music Assistant |
| `list_music_assistant_instances` | List configured Music Assistant instances |
| `search_music_assistant` | Search Music Assistant tracks, albums, artists, playlists, radio, audiobooks, and podcasts |
| `get_music_assistant_library` | Browse Music Assistant library content |
| `get_music_assistant_queue` | Inspect a player queue |
| `control_music_assistant_player` | Pause, resume, skip, seek, adjust volume, shuffle, repeat, or clear queues |
| `transfer_music_assistant_queue` | Move an active queue to another Music Assistant player |

Use player names, areas, floors, labels, or entity IDs to narrow ambiguous
player requests. Search and library browsing support Music Assistant tracks,
albums, artists, playlists, radio, audiobooks, and podcasts when those media
types are available from the configured Music Assistant instance.
For a contextual follow-up, copy both `media_content_id` and
`media_content_type` from a native media search result and pass them as
`within_media_content_id` and `within_media_content_type`; select a player when
the chosen Music Assistant instance has more than one exposed player. Searches
without these fields keep using Music Assistant's global search.

## Image Tools

Image tools depend on provider and source support.

| Tool | Purpose |
| --- | --- |
| `analyze_image` | Ask the active multimodal model about a camera snapshot, image entity, URL, or local image |
| `get_image` | Return an image as an MCP image content block |
| `generate_image` | Generate an image when the active provider exposes compatible image generation |

Only use these when the provider and client can support the requested image
workflow.
OpenAI profiles select a separate image model and Image Generation API in the
provider settings. Custom endpoints default to the Images API independently
of their conversation transport; Responses image generation requires an
explicit selection. Other compatible providers retain their profile model
and Images API route.
The conversation entity exposes `image_model` and `image_model_available: true`
when its loaded provider has an image route. Providers without an image route
expose `image_model_available: false`. These attributes describe configured
routing, not verified endpoint or model support or a successful API call.
The Responses image tool and official GPT Image models do not accept `style`.

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
