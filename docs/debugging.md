# Debugging MCP Assist

MCP Assist has two troubleshooting modes:

- **Debug Mode** writes more detailed entries to the Home Assistant log.
- **Chat Log Mode** persists recent conversation records so you can inspect what happened after the conversation is over.

Chat Log Mode is off by default. Enable it per profile from the profile's advanced settings.

## Debug Mode

Use Debug Mode when you need to inspect provider behavior, prompt construction,
tool selection, MCP requests, or tool execution failures in the Home Assistant
log.

Turn it off after troubleshooting. Debug logs can include provider error
messages, entity names, URLs, tool arguments, and other local home context.

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

The service returns a response payload with `count` and `logs`.

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
