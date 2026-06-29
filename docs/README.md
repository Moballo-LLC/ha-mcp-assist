# MCP Assist Documentation

This directory holds the long-form documentation for HA MCP Assist. The README
at the repository root is the entry point; these guides carry the operational
detail.

## Start Here

| Guide | Best for |
| --- | --- |
| [Getting Started](getting-started.md) | Installing the integration, creating your first profile, and testing it |
| [Architecture](architecture.md) | Understanding the MCP server, entity discovery, and token reduction model |
| [Configuration](configuration.md) | Provider setup, prompts, shared settings, profiles, and tool toggles |
| [Tool Reference](tool-reference.md) | What each core and optional tool family does |
| [Usage Examples](examples.md) | Example requests and the tool flow behind them |
| [Model Compatibility](model-compatibility.md) | Choosing and validating models that can use tools well |
| [Troubleshooting](troubleshooting.md) | Common failures and practical checks |
| [Security and Privacy](security-and-privacy.md) | Exposure controls, API keys, network access, memory, and external tools |
| [External Custom Tools](custom-tools.md) | Building installation-specific MCP tool packages |
| [Releases](releases.md) | Versioning, tags, release notes, and HACS packages |

## Where New Documentation Belongs

- Keep the root `README.md` short: project identity, quick start, and links.
- Put setup and first-run material in `getting-started.md`.
- Put setting descriptions and configuration decisions in `configuration.md`.
- Put tool behavior, tool-family toggles, and capability boundaries in
  `tool-reference.md`.
- Put scenario-based requests in `examples.md`.
- Put debugging steps in `troubleshooting.md`.
- Put release workflow and maintainer process in `releases.md`.
- Put external package authoring details in `custom-tools.md`.

When a topic spans several guides, link between them instead of duplicating the
same long explanation in multiple places.

## Documentation Style

- Prefer concrete behavior over marketing language.
- Describe what is supported, what must be enabled, and what can fail.
- Keep examples realistic and Home Assistant-specific.
- Avoid implying that every model or every Home Assistant installation behaves
  the same way.
- Document compatibility and security tradeoffs near the feature that creates
  them.
