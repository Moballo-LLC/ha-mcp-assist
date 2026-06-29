# Usage Examples

These examples show the kind of requests MCP Assist is designed to handle and
the tools it may use behind the scenes. Exact tool calls vary by model,
configuration, exposed entities, and installed integrations.

## Basic Control

User:

```text
Turn on the kitchen lights.
```

Typical flow:

1. `discover_entities` finds exposed kitchen light entities.
2. `perform_action` calls the appropriate Home Assistant action.
3. The assistant reports the result.

## Basic Read

User:

```text
What is the temperature in the living room?
```

Typical flow:

1. `discover_entities` searches for living-room temperature sensors.
2. `get_entity_details` reads the exact state and unit.
3. The assistant answers with the current value.

## Multi-Turn Conversation

User:

```text
What lights are on?
```

Assistant:

```text
The kitchen and living room lights are on.
```

User:

```text
Turn off the kitchen one.
```

Typical flow:

1. The assistant keeps enough conversation state to understand "the kitchen
   one."
2. `perform_action` turns off the kitchen light.
3. `set_conversation_state` may mark whether a follow-up is expected.

## Leak Check

User:

```text
Do we have a leak?
```

Typical flow:

1. `get_index` shows that moisture sensors and water-flow sensors exist.
2. `discover_entities` searches for moisture or leak sensors.
3. `discover_entities` searches for water-flow entities if relevant.
4. `get_entity_details` reads each candidate.
5. The assistant summarizes what is wet, dry, or flowing.

Follow-up:

```text
Turn off the water main.
```

Typical flow:

1. `discover_entities` finds the water-main shutoff entity.
2. `perform_action` turns it off.
3. The assistant reports whether Home Assistant confirmed the action.

## History Questions

User:

```text
When was the front door last opened?
```

Typical flow:

1. `discover_entities` finds the front-door contact sensor.
2. `get_entity_history` with `mode: "last_event"` checks recorder history.
3. The assistant reports the timestamp in a useful form.

Other examples:

- "How many times was the garage door opened today?"
- "How long has the basement deadbolt been locked?"
- "What was the thermostat set to last night?"

## Weather Forecast

User:

```text
What will the weather be here tomorrow?
```

Typical flow:

1. `discover_entities` or the weather helper finds an exposed `weather.`
   entity.
2. `get_weather_forecast` asks Home Assistant for a supported forecast type.
3. The assistant summarizes the forecast.

If you have several weather entities, include context:

```text
What is the backyard weather station forecast for tomorrow?
```

## Calendar and Response-Service Reads

User:

```text
What is on the family calendar this afternoon?
```

Typical flow:

1. `discover_entities` finds exposed calendar entities.
2. `get_calendar_events` reads the relevant time range.
3. The assistant summarizes matching events.

## Memory

User:

```text
Remember that I prefer the bedroom thermostat at 68 degrees.
```

Typical flow:

1. `remember_memory` stores the preference with the configured TTL.
2. Future requests can use `recall_memories` when the preference is relevant.

Other examples:

- "Remember for 14 days that the dog gets medicine with dinner."
- "What do you remember about my coffee preferences?"
- "Forget that reminder about the spare key."

Memory is shared across MCP Assist profiles. See
[Security and Privacy](security-and-privacy.md#memory).

## Web Search and URL Reading

User:

```text
Search for the latest Home Assistant release notes.
```

Typical flow:

1. `search` queries the configured provider.
2. `read_url` may open a selected result if URL reading is enabled.
3. The assistant summarizes the relevant information.

For Home Assistant-local information, such as entity state, weather, calendars,
or history, prefer the native Home Assistant tools over web search.

## Google Places and Routes

User:

```text
Is the nearest hardware store open, and how long would it take to get there?
```

Typical flow:

1. `search_google_places` finds matching places and open status.
2. `get_google_route` calculates a traffic-aware ETA from Home Assistant home.
3. The assistant summarizes the best matching place, address, phone, rating,
   open status, and travel time.

Other examples:

- "What is the phone number for Pike Place Market?"
- "How long will it take to drive to the airport right now?"
- "When should I leave home to arrive at dinner by 6:30?"

## Music Assistant

User:

```text
Play Miles Davis in the kitchen.
```

Typical flow:

1. `list_music_assistant_players` or entity discovery resolves the kitchen
   player.
2. `search_music_assistant` finds matching media.
3. `play_music_assistant` starts playback.

Other examples:

- "What is playing in the living room?"
- "Show me albums by The National."
- "Play my dinner playlist downstairs."

## Image and Camera Questions

User:

```text
What is in the driveway?
```

Typical flow:

1. `discover_entities` finds an exposed driveway camera if needed.
2. `analyze_image` snapshots or reads the image source.
3. The active multimodal provider answers the image question.

Provider and client support vary for image workflows. See
[Tool Reference](tool-reference.md#image-tools).

## External Custom Tools

External custom tools are useful for installation-specific workflows:

```text
What is the status of the custom pool controller?
```

The exact tool depends on the package you install under
`<home-assistant-config>/mcp-assist-tools`. See
[External Custom Tools](custom-tools.md).
