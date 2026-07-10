# Debugging MCP Assist

MCP Assist has two troubleshooting modes:

- **Debug Mode** writes more detailed entries to the Home Assistant log.
- **Chat Log Mode** persists recent conversation records so you can inspect what happened after the conversation is over.

Chat Log Mode is off by default. Enable it per profile from the profile's advanced settings.

## Debug Mode

Use Debug Mode when you need to inspect provider behavior, prompt construction,
tool selection, MCP requests, or tool execution failures in the Home Assistant
log.

Debug Mode is intended to log operational metadata such as request sizes,
latency, status codes, selected tools, and argument counts. It should not log
full prompts, user messages, provider payloads, tool arguments, or tool results.
Turn it off after troubleshooting to keep logs quiet.

When Debug Mode is enabled, the first model request in a conversation logs
metadata-only payload sizing, including provider, transport, payload bytes,
message character count, advertised tool count, and compact tool-schema bytes.
Providers that return prompt-cache usage, such as OpenAI, may also log
metadata-only cache counts such as input tokens, cached prompt tokens, and an
approximate cache hit percentage.

You can also inspect the shared MCP tool-surface overhead directly:

```text
http://<home-assistant-host>:8090/debug/prompt-overhead
```

That endpoint is guarded by the MCP server IP allowlist and returns only size
metadata, approximate token estimates, and top tool/package contributors. It
reports the Standard, Adaptive, and Light tool surfaces separately. It does not
return prompts, conversation history, raw tool descriptions, schemas, tool
arguments, or tool results.

To preview Adaptive mode for a specific kind of request, pass a short sample
query:

```text
http://<home-assistant-host>:8090/debug/prompt-overhead?query=weather%20tomorrow
```

The response includes a metadata-only Adaptive query projection with the tool
schemas that would be preloaded, their match scores, and the projected first
tool-schema size.

## What Chat Log Mode Records

When enabled, MCP Assist stores recent conversation records in Home Assistant storage. Each record includes:

- profile entry ID and profile name
- conversation ID
- provider type and model name
- user text and assistant reply
- whether the assistant kept the conversation open
- MCP tool names, arguments, results, and model-facing tool result text
- processing errors when a request fails

These logs can include sensitive home context, prompts, entity names, tool arguments, and tool results. Enable Chat Log Mode only while you are actively debugging, and clear logs when you are done.

Chat Log Mode is useful when a request succeeds or fails after several tool
calls and the normal Home Assistant log is too noisy to reconstruct the
conversation. It is not a replacement for Debug Mode when diagnosing startup,
provider connection, or custom-tool loading failures.

## Review Logs

Call the Home Assistant service:

```yaml
service: mcp_assist.get_chat_logs
data:
  limit: 10
```

Optional filters:

```yaml
service: mcp_assist.get_chat_logs
data:
  profile_entry_id: "01J..."
  conversation_id: "abc123"
```

The service returns a response payload with `count`, `projection`, and `logs`.
The default `full` projection preserves the original response shape, including
both raw MCP results and the compact content sent back to the model.

For a smaller metadata-only response:

```yaml
service: mcp_assist.get_chat_logs
data:
  limit: 10
  compact: true
```

`compact: true` is a shortcut for `projection: compact`. You can set
`projection` explicitly to one of:

- `full`: raw results and model-facing content
- `raw`: raw results without duplicated model-facing content
- `model`: model-facing content without raw results
- `compact`: tool names, timing, status, and argument keys without arguments or results

When both are set, `projection` takes precedence over `compact`.

## Clear Logs

Clear every stored chat log:

```yaml
service: mcp_assist.clear_chat_logs
```

Clear logs for one profile or conversation:

```yaml
service: mcp_assist.clear_chat_logs
data:
  profile_entry_id: "01J..."
```

```yaml
service: mcp_assist.clear_chat_logs
data:
  conversation_id: "abc123"
```
