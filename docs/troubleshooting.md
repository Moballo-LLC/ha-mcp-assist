# Troubleshooting

Start with the symptom, then check logs under Home Assistant
**Settings** -> **System** -> **Logs**. Enable MCP Assist debug mode when you
need to see tool selection and provider behavior in more detail.

## Integration Will Not Start

Check:

- Home Assistant restarted after installation.
- `custom_components/mcp_assist` exists in the expected location.
- The MCP server port is not already in use.
- Home Assistant can bind to the configured port.
- The logs do not show missing Python requirements or import errors.

If the port is in use, change **MCP Server Port** in the shared settings.

## HACS Install Does Not Show HA MCP Assist

Check:

- The repository was added as category **Integration**.
- The repository is `Moballo-LLC/ha-mcp-assist`.
- HACS has refreshed its repository data.
- Home Assistant has restarted after installation.

HACS and Home Assistant display the integration as **HA MCP Assist**. The
installed integration domain remains `mcp_assist`.

## Provider Connection Fails

For local providers:

- Confirm the server process is running.
- Confirm the URL is reachable from Home Assistant, not just from your laptop.
- If Home Assistant runs in Docker, avoid assuming `localhost` reaches the host.
- Confirm the selected model is loaded or available.

For cloud providers:

- Confirm the API key is valid.
- Confirm the model name is accepted by the provider.
- Check account quota, billing, and rate limits.
- Check network connectivity from Home Assistant.

### Ollama Request Exceeds Context Window

If Ollama returns an error like `request (...) tokens exceed`, the prompt,
history, and advertised tool schemas are larger than the selected model's
context window.

Try:

- Set the profile's **Context Mode** to **Adaptive** if it was changed to
  Standard.
- Use **Context Mode: Light** for very small direct-control local models.
- Reduce **Max History Messages**.
- Disable optional tool families the profile does not need.
- Increase the Ollama **Context Window** setting if the model and hardware
  support it.
- Keep custom prompts short.

## Tools Are Not Being Called

Symptoms:

- The assistant says it acted, but nothing changed.
- Logs show no `discover_entities`, `get_entity_details`, or `perform_action`.
- The response is only narration.

Check:

- The selected model supports tool/function calling in your runtime.
- The provider integration is using a tool-capable API path.
- Technical Instructions have not been replaced with conflicting guidance.
- Max Tool Iterations is not set too low.
- Debug Mode is enabled while testing.

See [Model Compatibility](model-compatibility.md) for a structured test plan.

## Entity Is Not Found

Check:

- The entity is exposed to the conversation assistant under **Settings** ->
  **Voice Assistants** -> **Expose**.
- The entity has a clear friendly name.
- The entity belongs to the expected area, floor, or label.
- The request includes enough context when names are ambiguous.
- The entity domain is supported for the requested read or write action.

Use simple requests first:

```text
What entities do you know about in the kitchen?
```

## Action Does Not Execute

Check:

- **Control Home Assistant** is enabled for the profile.
- The target entity is exposed.
- The entity domain supports the requested action.
- Home Assistant service calls work outside MCP Assist.
- The assistant did not choose a read-only entity such as a sensor.

Read-only domains should be inspected with `get_entity_details`, not changed
with `perform_action`.

## Follow-Ups Do Not Work

Check:

- Response Mode is **Smart** or **Always**.
- Max History Messages is high enough for the exchange.
- The model is capable of resolving references such as "that one."
- The previous response identified the object clearly enough for the model to
  reuse it.

## Chat Logs Are Empty

Check:

- **Chat Log Mode** is enabled on the profile that handled the request.
- You ran a new request after enabling Chat Log Mode.
- The `mcp_assist.get_chat_logs` service is not filtered to the wrong profile
  entry ID or conversation ID.
- Logs were not cleared by `mcp_assist.clear_chat_logs`.

Chat logs are intentionally opt-in and recent. Use Debug Mode alongside Chat Log
Mode when you need lower-level provider or tool-loading details.

## Weather Forecast Fails

Check:

- A `weather.` entity exists.
- The weather entity is exposed to the conversation assistant.
- The **Weather Forecast** tool family is enabled in shared settings.
- The weather integration supports at least one forecast type.
- The request disambiguates when multiple weather entities exist.

MCP Assist can fall back between supported forecast types, but it cannot invent
forecasts from integrations that do not expose them.

## Web Search Fails

Check:

- The **Search** tool family is enabled.
- A search provider is selected.
- Brave Search has a valid API key when using Brave.
- SearXNG has a valid base URL when using SearXNG.
- The profile has not disabled web search with a per-profile override.

For page-reading failures:

- Confirm the **Read URL** tool family is enabled.
- Confirm the URL is reachable from Home Assistant.
- Check whether the site blocks automated fetches.

## Memory Does Not Work

Check:

- The **Memory** tool family is enabled.
- The profile has not disabled memory.
- The user explicitly asked to remember, recall, or forget something.
- Memory retention settings are not pruning the item earlier than expected.

Stored memories are shared across MCP Assist profiles.

## Recorder History Is Empty

Check:

- Home Assistant recorder is enabled.
- The entity is included in recorder history.
- The requested time range is available.
- The entity had state changes during that time.
- The **Recorder History** tool family is enabled.

## Music Assistant Tools Do Not Appear

Check:

- The Home Assistant Music Assistant integration is installed and configured.
- The **Music Assistant** tool family is enabled.
- The profile has not disabled Music Assistant.
- The target player is a Music Assistant player or can be resolved from the
  request.

## External Custom Tools Do Not Load

Check:

- **Custom Tools** is enabled in shared settings.
- The package lives under
  `<home-assistant-config>/mcp-assist-tools/<tool_id>/`.
- The manifest file is named `mcp_tool.json`.
- The manifest `id` matches the folder name.
- Tool names are namespaced with `<tool_id>_`.
- The package directory is not a symlink.
- Home Assistant logs do not show package load errors.

You can reload external tools with the Home Assistant service:

```text
mcp_assist.reload_external_custom_tools
```

See [External Custom Tools](custom-tools.md).

## Debug Checklist

When reporting an issue, include:

- Home Assistant version
- MCP Assist version
- Provider and model name
- Whether the provider is local, containerized, or cloud-hosted
- Relevant MCP Assist settings
- A simple request that reproduces the issue
- Home Assistant log excerpts with Debug Mode enabled
- A recent `mcp_assist.get_chat_logs` response when Chat Log Mode is enabled
- Whether the entity is exposed to the conversation assistant
