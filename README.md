# HA MCP Assist for Home Assistant

Give your Home Assistant voice assistant a way to look things up instead of
forcing a model to read your whole house in every prompt.

HA MCP Assist is a conversation integration for Home Assistant that exposes
your home through MCP (Model Context Protocol) tools. The model can discover
areas, devices, entities, current states, history, weather, media, and optional
external data only when a request needs it. That keeps prompts smaller, makes
large Home Assistant installs easier to handle, and lets you choose between
local models and cloud providers without redesigning your assistant.

[![Open your Home Assistant instance and add this repository to HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Moballo-LLC&repository=ha-mcp-assist&category=integration)

## Why Use It

Most LLM-based Home Assistant assistants run into the same problem: the model
needs enough context to be useful, but sending every exposed entity and state on
every request gets expensive, slow, and unreliable as your home grows.

HA MCP Assist takes a different path:

- **Smaller prompts for bigger homes**: models get a compact view of your Home
  Assistant structure and use tools such as `get_index`, `discover_entities`,
  and `get_entity_details` to fetch the details they need.
- **Better fit for local models**: local runtimes such as LM Studio, llama.cpp,
  Ollama, OpenClaw, and vLLM can work with a more focused context instead of a
  giant entity dump.
- **Choice of providers**: use local models, OpenAI, Google Gemini, Anthropic
  Claude, OpenRouter, or multiple profiles for different jobs.
- **Useful home workflows**: control exposed entities, run scripts and
  automations, read recorder history, check calendars and weather, search the
  web, remember user-approved notes, and work with Music Assistant.
- **Home Assistant boundaries still matter**: MCP Assist follows Home
  Assistant's conversation exposure model, so the assistant only discovers and
  controls entities you expose.
- **Room to extend**: package external custom tools in the same documented MCP
  Assist format when your home or site needs something specific.

## What It Can Help With

Start with normal voice-assistant requests:

```text
Turn on the kitchen lights.
What is the temperature in the living room?
Are any upstairs windows open?
Run the bedtime script.
```

Then use the tool-based workflows when you want more context:

```text
Look at the last hour of garage activity and tell me if anything changed.
Do we have a leak, and is water still flowing?
What is on the family room Music Assistant queue?
Search for the latest Home Assistant release notes and summarize the important bits.
Remember that the guest room lamp is the one by the window.
```

The exact quality still depends on the model you choose. MCP Assist works best
with models that reliably support tool/function calling.

## How It Works

1. Home Assistant loads the `mcp_assist` integration.
2. MCP Assist starts a shared MCP server and one or more conversation profiles.
3. A profile sends the model concise instructions and available tool
   definitions.
4. The model calls tools to discover the right area, entity, service, history
   window, or optional data source.
5. MCP Assist performs reads or actions through Home Assistant APIs.
6. The model answers from the tool results instead of guessing from stale or
   incomplete context.

This is the same core idea from the original upstream MCP Assist project:
discover details on demand instead of stuffing the full entity list into every
request.

## Features

- Drop-in `mcp_assist` Home Assistant conversation integration.
- Smart Entity Index and on-demand entity discovery.
- Read and write actions for exposed Home Assistant entities, scripts,
  automations, devices, calendars, recorder history, and response services.
- Optional built-in tool families for calculator, unit conversion, weather
  forecasts, memory, web search, URL reading, Google Places and Routes,
  Wikipedia search, Music Assistant, images, and third-party LLM API bridges.
- External custom tool packages for site-specific MCP tools.
- Multiple conversation profiles with shared MCP server settings.
- Local and cloud provider support:
  - LM Studio
  - llama.cpp
  - Ollama
  - OpenClaw
  - vLLM
  - OpenAI
  - Google Gemini
  - Anthropic Claude
  - OpenRouter
- Debug mode, chat log mode, diagnostic exports, HACS packaging, and automated
  release validation.

## Quick Start

1. Add `Moballo-LLC/ha-mcp-assist` to HACS with the button above, or add it
   manually as an integration repository.
2. Install **HA MCP Assist** from HACS and restart Home Assistant.
3. Go to **Settings** -> **Devices & Services** -> **Add Integration**.
4. Search for **HA MCP Assist** and create a profile for your model provider.
5. Expose the entities the assistant may see or control under **Settings** ->
   **Voice Assistants**.
6. Set the new profile as a voice assistant and test a simple command.

For the complete setup walkthrough, see
[Getting Started](docs/getting-started.md).

## Requirements

- Home Assistant 2024.1+
- Python 3.11+
- One supported local or cloud model provider
- A model that can reliably use tool/function calling for the workflows you
  want to automate

See [Model Compatibility](docs/model-compatibility.md) before spending much time
tuning a small local model.

## Documentation

| Guide | Use it for |
| --- | --- |
| [Docs Home](docs/README.md) | Full documentation map and where new docs belong |
| [Getting Started](docs/getting-started.md) | Requirements, installation, first profile, and first commands |
| [Architecture](docs/architecture.md) | How MCP Assist reduces context size and discovers entities |
| [Configuration](docs/configuration.md) | Provider settings, prompts, shared settings, profiles, and tool toggles |
| [Tool Reference](docs/tool-reference.md) | Core MCP tools and optional built-in tool families |
| [Usage Examples](docs/examples.md) | Voice commands, follow-ups, history, weather, memory, and web examples |
| [Model Compatibility](docs/model-compatibility.md) | Choosing and testing models that support tool calling |
| [Troubleshooting](docs/troubleshooting.md) | Common setup, connection, and tool-calling issues |
| [Debugging](docs/debugging.md) | Debug mode, chat log mode, retention, and diagnostic exports |
| [Security and Privacy](docs/security-and-privacy.md) | Entity exposure, API keys, network access, memory, and custom tools |
| [External Custom Tools](docs/custom-tools.md) | Package format for site-specific MCP tool extensions |
| [Releases](docs/releases.md) | Version bumps, tags, HACS packages, and generated release notes |

## Project Lineage

This repository is the Moballo-maintained fork of
[mike-nott/mcp-assist](https://github.com/mike-nott/mcp-assist). Full credit to
the upstream project for the original MCP Assist integration and the core
on-demand discovery approach. This fork keeps the Home Assistant integration
domain as `mcp_assist` while publishing HACS releases from
[Moballo-LLC/ha-mcp-assist](https://github.com/Moballo-LLC/ha-mcp-assist).

## Releases

Release notes are generated by GitHub from merged PRs using
`.github/release.yml`; this repo does not maintain a separate `CHANGELOG.md`.
The release workflow validates the tag, builds `mcp_assist.zip`, and publishes
the GitHub release for HACS. See [Releases](docs/releases.md).

## Support

- [GitHub Issues](https://github.com/Moballo-LLC/ha-mcp-assist/issues)
- [GitHub Discussions](https://github.com/Moballo-LLC/ha-mcp-assist/discussions)
- [GitHub Sponsors: Jason-Morcos](https://github.com/sponsors/Jason-Morcos)
- [Home Assistant Community Forum](https://community.home-assistant.io/)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the
upstream and Moballo copyright notices.
