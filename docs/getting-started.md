# Getting Started

This guide walks through installing HA MCP Assist, creating a first conversation
profile, and testing that the model can use MCP tools.

## Requirements

- Home Assistant 2024.1+
- Python 3.11+
- One supported model provider:
  - Local or self-hosted: LM Studio, llama.cpp, Ollama, OpenClaw, or vLLM
  - Cloud: OpenAI, Google Gemini, Anthropic Claude, or OpenRouter
- A model that supports tool/function calling well enough for your intended
  workflows

For model selection notes, see [Model Compatibility](model-compatibility.md).

## Install with HACS

[![Open your Home Assistant instance and add this repository to HACS.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Moballo-LLC&repository=ha-mcp-assist&category=integration)

1. Click the badge above, or add `Moballo-LLC/ha-mcp-assist` manually as a HACS
   custom repository.
2. Select category **Integration**.
3. Install **HA MCP Assist** from HACS.
4. Restart Home Assistant.

## Manual Installation

1. Copy `custom_components/mcp_assist` into your Home Assistant
   `custom_components` directory.
2. Restart Home Assistant.
3. Keep manual installs updated yourself; HACS is the recommended path for most
   users.

## Add the Integration

1. Go to **Settings** -> **Devices & Services**.
2. Select **Add Integration**.
3. Search for **HA MCP Assist**.
4. Follow the setup flow for your provider.

HACS and Home Assistant show this project as **HA MCP Assist**. The installed
integration still uses the `mcp_assist` domain and `custom_components/mcp_assist`
folder so it stays a drop-in replacement for the upstream integration.

## Create Your First Profile

The setup flow creates a conversation profile. You can add more profiles later
for different providers, models, rooms, or behavior.

### Step 1: Profile and Server Type

- **Profile Name**: A friendly name, such as `Living Room Assistant`.
- **Server Type**: Choose the model provider.

Supported server types include:

- LM Studio
- llama.cpp
- Ollama
- OpenAI
- Google Gemini
- Anthropic Claude
- OpenRouter
- OpenClaw
- vLLM

### Step 2: Provider Connection

For local or OpenAI-compatible servers other than OpenClaw, enter the server URL:

| Provider | Common default |
| --- | --- |
| LM Studio | `http://localhost:1234` |
| llama.cpp | `http://localhost:8080` |
| Ollama | `http://localhost:11434` |
| vLLM | `http://localhost:8000` |

For OpenClaw, enter the gateway host, port, bearer token, and SSL setting shown
by your OpenClaw gateway. The default gateway port is `18789`.

For cloud providers, enter the provider API key.

### Step 3: Model and Prompts

- **Model Name**: Select an auto-loaded model or enter a model name manually.
- **System Prompt**: Review or customize the general assistant behavior.
- **Technical Instructions**: Review or customize tool-use behavior.

If you leave the prompt text effectively unchanged, MCP Assist continues using
the built-in prompt from the integration code. That lets future releases improve
the default prompt without treating your profile as customized.

### Step 4: Conversation and Advanced Settings

Common settings:

- **Temperature**: Response randomness.
- **Max Response Tokens**: Maximum response length.
- **Max History Messages**: Conversation memory depth.
- **Context Mode**: Use Standard for the full profile tool context, or Light
  for small local models that need a smaller request.
- **Response Mode**: None, Smart, or Always.
- **Control Home Assistant**: Whether the assistant can perform write actions.
- **Max Tool Iterations**: How many tool calls are allowed for one request.
- **Timeout**: Maximum provider response wait time.
- **Debug Mode**: Extra logging for troubleshooting.

Provider-specific settings may also appear, such as Ollama keep-alive and
context-window controls.

### Step 5: Shared MCP Server Settings

Shared settings apply to every profile because all profiles use the same MCP
server.

- **MCP Server Port**: Default `8090`.
- **Additional Allowed IPs/Ranges**: Optional allowlist for external MCP
  clients.
- **Discovery**: Smart Entity Index and max entities per discovery call.
- **Tools**: Shared optional tool families, such as weather forecast, web
  search, Wikipedia search, memory, Music Assistant, and external custom tools.

See [Configuration](configuration.md) for the full settings reference.

## Set as a Voice Assistant

1. Go to **Settings** -> **Voice Assistants**.
2. Select the MCP Assist profile you created.
3. Expose the entities you want the assistant to see or control.
4. Test a simple request.

## First Test Commands

Start with small requests that make it obvious whether tool calling works:

- "Turn on the kitchen lights."
- "What is the temperature in the living room?"
- "Are any lights on upstairs?"
- "What rooms do you know about?"

If the model replies as if it acted but no entity changes, go to
[Troubleshooting](troubleshooting.md#tools-are-not-being-called).

## Next Steps

- Learn how entity discovery works: [Architecture](architecture.md)
- Review available tools: [Tool Reference](tool-reference.md)
- Tune profile and shared settings: [Configuration](configuration.md)
- Check model behavior: [Model Compatibility](model-compatibility.md)
