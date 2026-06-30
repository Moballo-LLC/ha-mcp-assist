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
| Context Mode | Adaptive, standard, or light context sent to the model |
| Response Mode | Whether the assistant should continue conversations |
| Timeout | Provider response timeout |
| Debug Mode | Extra logging for investigation |
| Chat Log Mode | Opt-in stored conversation records for debugging |
| Tool Family Overrides | Optional per-profile narrowing of shared tools |

## Provider Connection Settings

### Local and Self-Hosted Providers

For LM Studio, llama.cpp, Ollama, and vLLM, configure the server URL.

| Provider | Common default URL |
| --- | --- |
| LM Studio | `http://localhost:1234` |
| llama.cpp | `http://localhost:8080` |
| Ollama | `http://localhost:11434` |
| vLLM | `http://localhost:8000` |

If Home Assistant runs in a container, `localhost` means the Home Assistant
container, not necessarily the host machine. Use a reachable host IP or Docker
network name when needed.

OpenClaw uses gateway settings instead of the generic server URL field. Configure
the gateway host, port, bearer token, and SSL setting from your OpenClaw gateway.
The default gateway port is `18789`.

### Cloud Providers

For OpenAI, Google Gemini, Anthropic Claude, and OpenRouter, configure an API
key and model name. Keep provider keys in Home Assistant secrets or another
safe local workflow where possible.

For official OpenAI profiles, MCP Assist sends a stable non-identifying
`prompt_cache_key` so OpenAI can route repeated profile/tool prefixes to its
prompt cache more effectively. OpenAI controls whether a specific request is
cacheable. MCP Assist also requests streaming usage metadata so Debug Mode can
show cached prompt-token counts when OpenAI returns them. OpenAI-compatible
local providers do not receive these OpenAI-only fields.

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
| Context Mode | Use **Adaptive** by default, **Standard** for full upfront tool context, or **Light** for the smallest local-model requests |
| Max Tool Iterations | Raise only if legitimate requests need more tool calls |
| Clean Responses | Useful for voice output when models include extra formatting |
| Control Home Assistant | Disable for read-only profiles |

Adaptive context mode is the default. It advertises the core Home Assistant
discovery/control tools plus two compact routing tools that let the model search
and load optional, built-in package, or external custom tool schemas only when a
request needs them. This keeps first-prompt overhead low without disabling those
capabilities. For obvious requests, Adaptive may preload a small number of
high-confidence optional tool schemas from the user's wording so the model can
avoid an extra schema-discovery turn.

Light context mode keeps the profile's prompts but skips MCP Assist's optional
tool-family prompt instructions, keeps at most two prior conversation turns, and
advertises only core Home Assistant discovery/control tools to the model. It is
useful for small Ollama or other local models with tight context windows. Use
Standard when you intentionally want the profile's full tool surface sent
upfront, including optional tools such as recorder history, weather, web search,
memory, Music Assistant, and external custom tools.

### Response Mode

- **None**: End the conversation after each response.
- **Smart**: Continue when the assistant expects a follow-up.
- **Always**: Prefer conversational follow-ups until the user ends the exchange.

Users can end a continuing conversation with configured end words such as "bye"
or "thanks".

### Debugging Settings

- **Debug Mode** writes operational metadata to the Home Assistant log,
  including provider status, request sizes, latency, selected tools, and
  argument counts.
- **Chat Log Mode** stores recent conversation records in Home Assistant
  storage. Records can include user text, assistant replies, tool names,
  arguments, results, and errors. Keep it off unless you are actively
  troubleshooting. See [Debugging](debugging.md).

## Shared MCP Server Settings

Shared settings apply to all profiles:

| Setting | What it controls |
| --- | --- |
| MCP Server Port | The HTTP/WebSocket MCP server port, default `8090` |
| Additional Allowed IPs/Ranges | Extra clients allowed to connect to the MCP server |
| MCP Bearer Token | Optional token external MCP clients must send as `Authorization: Bearer <token>`. Enter `FFFF` and save to generate a replacement token. |
| Smart Entity Index | Compact Home Assistant structure index and gap-filling behavior |
| Max Entities Per Discovery | Upper bound for a single discovery result |
| Context Sharing | Whether user and home-location context is included in prompts or tool-call metadata |
| Web Search Provider | None, DuckDuckGo, Brave Search, or SearXNG |
| LLM API Bridge | Optional allowlist for third-party Home Assistant LLM APIs |
| Shared Tool Families | Which optional built-in and external tools are exposed, with tool-specific settings shown beside the relevant tool |

Because the MCP server is shared, changing these settings from one profile's
options flow affects every profile.

Context sharing has separate prompt and tool-call controls. The prompt settings
let the model know the current Home Assistant user and configured home location
when available. The tool-call settings separately decide whether that user or
location metadata is passed to MCP tools, including external custom tools.

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
recall, categorize, and forget user-approved memories.

Shared settings shown with the Memory tool:

- **Default Memory TTL**: How long a new memory is kept when a tool call does
  not specify `ttl_days`.
- **Max Memory TTL**: The longest allowed retention period.
- **Max Stored Memories**: The maximum number of active stored memories.

Memory categories are labels for retrieval and cleanup. MCP Assist suggests
stable categories such as `preference`, `routine`, `device_alias`,
`automation_note`, `baseline`, `correction`, `maintenance`, and `household`, but
custom categories are still accepted for existing workflows.

Memories are shared across MCP Assist profiles. See
[Security and Privacy](security-and-privacy.md#memory).

## Third-Party LLM API Bridge

The LLM API Bridge is optional and disabled by default. When enabled, MCP Assist
can inspect and call tools exposed by other Home Assistant integrations that
register LLM APIs.

Shared setting shown with the LLM API Bridge tool:

- **LLM API Bridge**: Exposes bridge tools on the shared MCP server.
- **Allowed LLM API IDs**: Comma- or newline-separated API IDs MCP Assist may
  inspect or call, such as `llm_intents`.

The shared settings form shows currently registered third-party LLM APIs beside
the allowlist field. Copy the API ID from that list when enabling an installed
integration.

The built-in Home Assistant `assist` API remains on the separate Assist Bridge.
Do not add third-party API IDs unless you trust the integration that registered
them. See [Security and Privacy](security-and-privacy.md#third-party-llm-apis).

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

## Google Places and Routes

Google Places and Routes is disabled by default. Enable it when a profile should
be able to look up place details or calculate travel time.

Required shared setting shown with the Google Places and Routes tool:

- **Google Maps API Key**: A Google Maps Platform API key with the Places API
  and Routes API enabled.

To create a key:

1. In Google Cloud, use a project with billing enabled.
2. Enable [Places API (New)](https://developers.google.com/maps/documentation/places/web-service/get-api-key)
   and [Routes API](https://developers.google.com/maps/documentation/routes/get-api-key)
   for that project.
3. Open the [Google Maps Platform Credentials page](https://console.cloud.google.com/google/maps-apis/credentials).
4. Choose **Create credentials** -> **API key**.
5. Restrict the key before using it in production. For API restrictions, allow
   only the Places API and Routes API. For application restrictions, choose the
   restriction type that matches where Home Assistant sends requests from, such
   as server IP address restrictions for a fixed outbound IP.

When `get_google_route` is called without an origin, MCP Assist can use the Home
Assistant home latitude and longitude as the route origin only if **Share Home
Location with MCP Tools** is enabled. Otherwise the caller must provide an
origin.

## Reference Tools

Wikipedia Search is disabled by default. Enable it when a profile should be able
to search Wikipedia for lightweight background or encyclopedia-style reference
results without using the broader web-search provider.

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
