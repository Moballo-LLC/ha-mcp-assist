# Security and Privacy

MCP Assist can read Home Assistant state, call Home Assistant services, connect
to model providers, and optionally use web, memory, media, image, and custom
tool integrations. Configure it with the same care you would use for any Home
Assistant automation that can inspect or control your home.

## Entity Exposure

MCP Assist follows Home Assistant's conversation exposure controls. If an entity
is not exposed to the conversation assistant, MCP Assist should not discover or
control it through normal entity discovery.

Review exposure under:

```text
Settings -> Voice Assistants -> Expose
```

Recommendations:

- Expose only entities the assistant should see or control.
- Use clear names, areas, floors, and labels.
- Be more conservative with locks, garage doors, alarms, valves, and other
  safety-sensitive entities.
- Consider separate profiles for read-only and control-capable use cases.

## Action Control

The **Control Home Assistant** profile setting determines whether a profile may
perform write actions. Disable it for profiles that should only answer questions.

Even with control enabled, the model should use discovery and details before
acting when the target is ambiguous. Keep technical instructions aligned with
that expectation.

## Context Sharing

MCP Assist can include the current Home Assistant user and configured home
location in model prompts. It can also pass that context as tool-call metadata
when the shared MCP server settings allow it.

Recommendations:

- Include current user and home location in prompts only when they improve
  routing or personalization for your use case.
- Keep tool-call context disabled unless built-in or external tools need it.
- Treat tool-call context as part of the data available to external custom tool
  packages.
- Avoid sharing screenshots or logs that contain user, location, or entity
  details unless they are redacted.

## MCP Server Network Access

The MCP server listens on the configured port, default `8090`. Use
**Additional Allowed IPs/Ranges** only for clients that should be able to reach
the MCP server.

Recommendations:

- Do not expose the MCP server directly to the public internet.
- Prefer local networks, Docker networks, or trusted add-on networks.
- Keep allowed IP ranges narrow.
- Remove temporary ranges after testing.

## Provider API Keys

Cloud providers require API keys. Treat those keys as secrets:

- Do not commit API keys to Git.
- Prefer Home Assistant secrets or another local secret-management workflow.
- Rotate keys if they were pasted into logs, issues, or screenshots.
- Review provider billing and quota settings.

## Local Provider Privacy

Local models can reduce third-party data exposure, but check your actual setup:

- Home Assistant may still send requests over your local network.
- A local server may write prompts or responses to logs.
- Container networking can expose services more broadly than expected.
- Some local tools or frontends may have telemetry settings of their own.

## Web Search, URL Reading, and Maps

The Search, Read URL, and Google Places and Routes tool families can send
requests to external services or websites.

Recommendations:

- Keep web tools disabled unless a profile needs them.
- Prefer Home Assistant-native tools for Home Assistant data.
- Use provider-specific keys and quotas where applicable.
- Be aware that fetched pages may include untrusted content.
- Google Places can use Home Assistant home coordinates as a default search
  bias, and Google Routes defaults to home coordinates when no origin is
  provided. Enable them only for profiles where that location sharing is expected.

`read_url` is intended for page text extraction. It should not be used as a
general network scanner or a way to bypass Home Assistant network policy.

## Memory

When enabled, memory tools can store user-approved facts and preferences with a
TTL. Memories are shared across MCP Assist profiles.

Recommendations:

- Ask the assistant to remember only information you are comfortable storing in
  Home Assistant.
- Use TTLs for temporary facts.
- Use `forget_memory` for stale or sensitive stored facts.
- Keep max TTL and max memory count aligned with your retention expectations.

## External Custom Tools

External custom tools are Python code loaded from your Home Assistant config
directory. Once enabled, they run inside Home Assistant.

Only install or write packages you trust.

Safety boundaries:

- Disabled by default.
- Loaded only from `<home-assistant-config>/mcp-assist-tools`.
- Symlinked package directories are rejected.
- Tool names must be namespaced.
- Invalid packages are skipped instead of crashing the MCP server.

These boundaries do not make arbitrary Python code safe. They reduce accidental
misconfiguration and package collisions.

## Logs and Debug Mode

Debug Mode is useful while diagnosing tool calls, but logs can include provider
errors, tool arguments, entity names, URLs, and other details about your home.

Chat Log Mode stores recent conversation records in Home Assistant storage. It
can include user text, assistant replies, tool names, tool arguments, tool
results, and errors. See [Debugging](debugging.md) for review and clear
services.

Recommendations:

- Disable Debug Mode after troubleshooting.
- Disable Chat Log Mode after troubleshooting.
- Clear stored chat logs when they are no longer needed.
- Review logs before sharing them publicly.
- Redact API keys, hostnames, IPs, and sensitive entity names when needed.

## Reporting Security Issues

If you believe you found a security issue, avoid posting sensitive details in a
public issue. Use GitHub's private vulnerability reporting flow if available for
the repository, or contact the maintainers through an appropriate private
channel.
