# Configuration

MCP Assist configuration has two layers:

- **Conversation profiles** for provider, model, prompt, and behavior settings.
- **Shared MCP server settings** for capabilities used by every profile.

This guide explains the main settings and when to change them.

## Profile Settings

Profile settings are independent per conversation agent.

| Setting | What it controls |
| --- | --- |
| Profile Name | The Home Assistant conversation agent name |
| Server Type | The provider or local server family |
| Server URL / API Key | How MCP Assist reaches the provider |
| Model Name | Which model the profile uses |
| System Prompt | General assistant behavior |
| Technical Instructions | Tool-use rules and Home Assistant-specific guidance |
| Temperature | Response randomness |
| Max Response Tokens | Response length cap |
| Max History Messages | Multi-turn conversation depth |
| Response Mode | Whether the assistant should continue conversations |
| Timeout | Provider response timeout |
| Debug Mode | Extra logging for investigation |
| Tool Family Overrides | Optional per-profile narrowing of shared tools |

## Provider Connection Settings

### Local and Self-Hosted Providers

For LM Studio, llama.cpp, Ollama, Moltbot, and vLLM, configure the server URL.

| Provider | Common default URL |
| --- | --- |
| LM Studio | `http://localhost:1234` |
| llama.cpp | `http://localhost:8080` |
| Ollama | `http://localhost:11434` |
| Moltbot | `http://localhost:18789` |
| vLLM | `http://localhost:8000` |

If Home Assistant runs in a container, `localhost` means the Home Assistant
container, not necessarily the host machine. Use a reachable host IP or Docker
network name when needed.

### Cloud Providers

For OpenAI, Google Gemini, Anthropic Claude, and OpenRouter, configure an API
key and model name. Keep provider keys in Home Assistant secrets or another
safe local workflow where possible.

## Prompt Settings

MCP Assist has two prompt fields:

- **System Prompt**: The assistant's general role and response style.
- **Technical Instructions**: Rules for tool use, discovery, action execution,
  memory, and optional tools.

During setup and options editing, the prompt fields are prefilled with the
current effective prompt. If the text is left unchanged, MCP Assist treats the
profile as using the built-in default. That allows future releases to improve
the default prompt without locking your profile to an older copy.

Customize prompts when you need a stable site-specific behavior. Keep technical
instructions concise and avoid hardcoding entity IDs unless that is intentional.

## Conversation Settings

| Setting | Guidance |
| --- | --- |
| Temperature | Lower values are usually better for predictable device control |
| Max Response Tokens | Keep high enough for explanations, lower for voice-only profiles |
| Max History Messages | Increase for longer multi-turn context, decrease for smaller prompts |
| Max Tool Iterations | Raise only if legitimate requests need more tool calls |
| Clean Responses | Useful for voice output when models include extra formatting |
| Control Home Assistant | Disable for read-only profiles |

### Response Mode

- **None**: End the conversation after each response.
- **Smart**: Continue when the assistant expects a follow-up.
- **Always**: Prefer conversational follow-ups until the user ends the exchange.

Users can end a continuing conversation with configured end words such as "bye"
or "thanks".

## Shared MCP Server Settings

Shared settings apply to all profiles:

| Setting | What it controls |
| --- | --- |
| MCP Server Port | The HTTP/WebSocket MCP server port, default `8090` |
| Additional Allowed IPs/Ranges | Extra clients allowed to connect to the MCP server |
| Smart Entity Index | Compact Home Assistant structure index and gap-filling behavior |
| Max Entities Per Discovery | Upper bound for a single discovery result |
| Web Search Provider | None, DuckDuckGo, Brave Search, or SearXNG |
| Memory Retention | Default TTL, max TTL, and max stored memories |
| Shared Tool Families | Which optional built-in and external tools are exposed |

Because the MCP server is shared, changing these settings from one profile's
options flow affects every profile.

## Shared vs Per-Profile Tool Settings

Tool families have two levels of control:

1. Shared MCP server setting: whether the tool family is exposed at all.
2. Per-profile override: whether a specific profile may use a shared tool
   family.

Use shared settings to enable capabilities for the whole integration. Use
per-profile disables when one profile should be narrower, such as a read-only
assistant or a room-specific voice profile.

See [Tool Reference](tool-reference.md) for the tool families.

## Web Search Configuration

Web search is optional and disabled unless enabled in shared tool settings.

Providers:

- **None**: Search disabled.
- **DuckDuckGo**: No API key required.
- **Brave Search**: Requires a Brave Search API key.
- **SearXNG**: Requires the base URL of your SearXNG instance.

URL reading is controlled separately by the **Read URL** tool family, although
legacy web-search settings may still enable it for older profiles.

## Memory Configuration

The Memory tool family is optional. When enabled, the assistant can store,
recall, and forget user-approved memories.

Shared settings:

- **Default Memory TTL**: How long a new memory is kept when a tool call does
  not specify `ttl_days`.
- **Max Memory TTL**: The longest allowed retention period.
- **Max Stored Memories**: The maximum number of active stored memories.

Memories are shared across MCP Assist profiles. See
[Security and Privacy](security-and-privacy.md#memory).

## Weather Configuration

For Home Assistant-native weather answers:

- Keep the **Weather Forecast** tool family enabled.
- Expose at least one `weather.` entity to the conversation assistant.
- Make sure the weather integration supports at least one forecast type, such as
  daily, twice-daily, or hourly.
- Use room, area, label, or entity names to disambiguate when you have multiple
  weather entities.

MCP Assist falls back to a supported forecast type when a specific type is not
available.

## External Custom Tools

External custom tools are disabled by default. When enabled, MCP Assist loads
trusted Python tool packages from:

```text
<home-assistant-config>/mcp-assist-tools
```

Use this for installation-specific behavior that should not live in the
integration itself. See [External Custom Tools](custom-tools.md).

## Common Recipes

### Read-Only Assistant

- Disable **Control Home Assistant**.
- Leave read tools enabled, such as discovery, weather, recorder, and response
  service reads.
- Consider disabling web search and external custom tools.

### Fast Voice Profile

- Use a model that is already loaded or has fast cold starts.
- Keep **Max Response Tokens** modest.
- Keep **Max Tool Iterations** close to the default.
- Enable **Clean Responses**.
- Use **Response Mode: None** or **Smart**.

### Rich Research Profile

- Enable web search and URL reading.
- Use a model with reliable tool calling and enough context.
- Consider a higher timeout.
- Keep action controls enabled only if this profile should also control devices.
