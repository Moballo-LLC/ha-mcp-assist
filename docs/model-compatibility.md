# Model Compatibility

MCP Assist relies on tool/function calling. A model that writes good prose is
not automatically good at choosing tools, formatting arguments, reading tool
results, and deciding when to call another tool.

## What Tool Calling Requires

For Home Assistant control, the model needs to:

1. Understand the user's request.
2. Decide whether a tool is needed.
3. Choose the right tool.
4. Format tool arguments as valid JSON.
5. Interpret the tool result.
6. Continue with another tool call when needed.
7. Explain the result without claiming an action succeeded unless Home Assistant
   confirmed it.

## Provider Families

| Provider type | Notes |
| --- | --- |
| Local OpenAI-compatible servers | Good for privacy and local control; quality depends heavily on the model and server's tool-calling support |
| Ollama | Convenient local workflow; confirm the selected model and client path support tools reliably |
| LM Studio | Good local testing surface; model loading and server settings matter |
| llama.cpp / vLLM / OpenClaw | Useful self-hosted options when configured with compatible chat and tool-call behavior |
| Cloud providers | Often stronger tool-calling behavior, but involve provider cost, network dependency, and API-key handling |

## Model Size and Behavior

Smaller local models can work for simple requests, especially direct single-tool
actions. More complex requests usually need stronger tool-calling behavior:

- searching by area and domain
- reading several entities
- deciding between read and write actions
- using recorder history
- resolving ambiguous follow-ups
- combining Home Assistant data with optional web or memory tools

Treat model recommendations as starting points, not guarantees. Models,
quantizations, provider APIs, and local server behavior change over time.

### Small Context Local Models

If Ollama or another local provider rejects a request because it exceeds the
model context window, try **Context Mode: Light** on that conversation profile.
Light mode keeps at most two prior conversation turns, skips MCP Assist's
optional tool-family prompt instructions, and advertises only core Home
Assistant discovery/control tools to the model.

Light mode is best for direct local-control profiles. Use Standard mode for
profiles that need optional tools such as recorder history, weather, web search,
memory, Music Assistant, or external custom tools. If the model and hardware can
handle it, increasing the provider context-window setting can also help.

## Instruct vs Reasoning Models

Instruct models often respond quickly and can be a good fit for simple voice
commands:

- "Turn on the kitchen lights."
- "What is the living room temperature?"

Reasoning or thinking models are often better for multi-step requests, but may
be slower:

- "Check every room for open windows, then turn off lights in rooms where a
  window is open."
- "Look at the last hour of garage activity and tell me whether anything looks
  unusual."

Choose based on the profile's job. A fast room assistant and a broader research
assistant may need different models.

## Practical Starting Points

For local models, start with a model that has proven tool-call support in your
chosen runtime. Larger models usually perform better for multi-tool Home
Assistant workflows when your hardware can run them comfortably.

Examples that have historically been useful starting points:

- Qwen instruct models with tool-call support
- Larger local instruct models when latency is acceptable
- Cloud models from OpenAI, Anthropic, Google, or OpenRouter when cost and
  network dependency are acceptable

Always test the exact model, quantization, and server combination you plan to
use.

## How to Test a Model

Start with simple tool calls:

```text
Turn on the kitchen lights.
```

Check:

- Home Assistant logs show `discover_entities` and `perform_action`.
- The target device actually changes.
- The response does not claim success when the action failed.

Then test reads:

```text
What is the temperature in the living room?
```

Check:

- The model discovers the right sensor.
- It calls `get_entity_details`.
- It reports the exact state and unit.

Then test a multi-step request:

```text
Are any doors open, and if so, when did they last change?
```

Check:

- The model uses discovery before history tools.
- It handles empty or ambiguous results.
- It does not invent entity names.

## Signs a Model Is Not a Good Fit

- It says it turned something on but no `perform_action` call happened.
- It repeatedly calls the wrong tool.
- It invents entity IDs.
- It sends malformed JSON arguments.
- It ignores tool errors.
- It gives up after discovery even though details are needed.
- It needs very high tool-iteration limits for ordinary requests.

## Tuning Tips

- Lower temperature for more predictable control.
- Keep prompts concise and avoid conflicting instructions.
- Make sure entity names and areas in Home Assistant are clear.
- Expose only the entities this assistant should use.
- Increase max tool iterations only after confirming the model is making useful
  progress.
- Try the same model on a different runtime if tool calls appear malformed.

## Dynamic Model Switching

MCP Assist profiles can switch models in the configuration UI without restarting
Home Assistant. This makes it easier to compare:

- small vs large models
- quantization levels
- local vs cloud providers
- fast voice profiles vs richer reasoning profiles

When comparing models, use the same test requests and inspect Home Assistant
logs so you can separate response style from actual tool behavior.
