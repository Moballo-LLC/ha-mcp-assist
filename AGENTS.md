# Agent Guide

This file gives coding agents the project-specific context they need before
editing this repository. It applies to the whole repository.

## Project Identity

- This repository is `Moballo-LLC/ha-mcp-assist`, the Moballo-maintained fork of
  upstream `mike-nott/mcp-assist`.
- Keep upstream credit intact in user-facing docs and license text.
- HACS should show this repo as `HA MCP Assist`, which comes from
  `hacs.json`.
- The actual Home Assistant integration must remain a drop-in replacement for
  upstream MCP Assist:
  - Keep the integration folder at `custom_components/mcp_assist`.
  - Keep `custom_components/mcp_assist/manifest.json` domain as `mcp_assist`.
  - Keep the Home Assistant integration name as `MCP Assist` unless there is a
    deliberate migration plan.
- Do not move files out of `custom_components/mcp_assist` for release packaging.
  The release workflow zips that folder as `mcp_assist.zip`.

## Compatibility Policy

- Runtime support is intentionally lenient. Do not raise `hacs.json`
  `homeassistant` or manifest requirement lower bounds just because CI or
  Dependabot is newer.
- CI targets the current Home Assistant-era test harness. At the time this file
  was written, CI uses Python 3.14 and `pytest-homeassistant-custom-component`
  from `requirements_test.txt`.
- Do not add arbitrary forward caps such as `<2026.7.0` to runtime or test
  dependencies unless there is a proven breakage and a plan to remove the cap.
- Keep older Home Assistant compatibility shims when they exist in code, such as
  guarded imports for APIs that moved between HA versions.

## Dependency Surfaces

- Runtime dependencies live in
  `custom_components/mcp_assist/manifest.json`.
- `requirements_runtime.txt` mirrors the runtime dependency package names so
  Dependabot can see them. It may carry newer lower bounds than the manifest.
- `tests/test_manifest_dependencies.py` intentionally checks package names, not
  exact specifier equality. Do not change it back to exact equality.
- Test-only dependencies live in `requirements_test.txt`. They are not runtime
  support promises.
- Dependabot is configured in `.github/dependabot.yml` for root `pip` and
  GitHub Actions updates. Minor and patch updates are grouped separately.

## Release Process

- There is no `CHANGELOG.md`. Do not add one back unless the release process is
  intentionally redesigned.
- GitHub generated release notes are configured by `.github/release.yml`.
  Release notes come from merged PR titles and labels.
- To publish a release:
  1. Update `custom_components/mcp_assist/manifest.json` `version`.
  2. Push a matching `vX.Y.Z` tag.
  3. Let `.github/workflows/release.yml` validate the tag, build
     `dist/mcp_assist.zip`, and publish the GitHub release.
- The release workflow requires the tag version to match the manifest version.

## CI and Validation

CI runs:

- Tests: `.github/workflows/tests.yml`
- HACS validation: `.github/workflows/hacs.yml`
- Hassfest: `.github/workflows/hassfest.yml`
- Secret scan: `.github/workflows/secret-scan.yml`
- CodeQL: `.github/workflows/codeql.yml`
- Release packaging on `v*` tags: `.github/workflows/release.yml`

For local validation, prefer the repo's existing Python environment when
available:

```bash
/tmp/ha-mcp-assist-py314-venv/bin/python -m pytest -q tests
/tmp/ha-mcp-assist-py314-venv/bin/ruff check custom_components tests
/tmp/ha-mcp-assist-py314-venv/bin/python -m compileall -q custom_components tests
git diff --check
```

If that venv is not available, create a fresh virtual environment and install
`requirements_test.txt`. The system `python` on a developer machine may be too
old or may not have pytest installed.

For focused changes, run the most relevant subset first, then broaden when the
change touches shared behavior:

- Metadata/dependency changes: `tests/test_manifest_dependencies.py`
- Brand assets: `tests/test_brand_assets.py`
- Config flow: `tests/test_config_flow.py`
- MCP tool lists and behavior: `tests/test_mcp_server.py` and
  `tests/test_agent.py`
- Custom tool package loading: `tests/test_custom_tools_loader.py`
- Provider runtime headers and metadata: `tests/test_provider_runtime.py`

When reproducing CI's pytest artifact behavior, use:

```bash
mkdir -p test-results
pytest -ra --junitxml=test-results/pytest.xml tests
```

## Code Layout

- `custom_components/mcp_assist/__init__.py`: Home Assistant setup, unload, and
  service registration.
- `custom_components/mcp_assist/config_flow.py`: config and options flows.
- `custom_components/mcp_assist/agent.py`: conversation entity and provider
  orchestration.
- `custom_components/mcp_assist/mcp_server.py`: MCP server and core tool
  dispatch.
- `custom_components/mcp_assist/server_tools/`: focused server-side tool
  modules for calendar, recorder, response services, and weather.
- `custom_components/mcp_assist/custom_tools/`: built-in and external custom
  tool loading plus individual tool implementations.
- `custom_components/mcp_assist/custom_tools/packages/`: manifest-backed
  built-in tool packages.
- `custom_components/mcp_assist/discovery.py`: entity, device, area, and index
  discovery helpers.
- `custom_components/mcp_assist/index_manager.py`: smart entity index lifecycle.
- `custom_components/mcp_assist/memory_manager.py`: persisted memory storage.
- `custom_components/mcp_assist/strings.json` and
  `custom_components/mcp_assist/translations/`: Home Assistant UI strings.
- `custom_components/mcp_assist/brand/`: HACS/Home Assistant brand assets.
- `docs/README.md`: documentation hub and placement guide.
- `docs/getting-started.md`: installation and first-profile setup.
- `docs/architecture.md`: MCP server, entity discovery, and index concepts.
- `docs/configuration.md`: settings and profile/shared configuration reference.
- `docs/tool-reference.md`: built-in MCP tool and tool-family reference.
- `docs/examples.md`: scenario-based usage examples.
- `docs/model-compatibility.md`: model selection and tool-calling validation.
- `docs/troubleshooting.md`: common issue diagnosis.
- `docs/security-and-privacy.md`: exposure, network, memory, and custom-tool safety.
- `docs/custom-tools.md`: external custom tool package documentation.
- `docs/releases.md`: release process and generated release-note workflow.

## Coding Conventions

- Follow the existing style. This repo uses Ruff with `E` and `F` rules and a
  line length of 100.
- Keep comments sparse and useful. Prefer clear code over explanatory comments.
- Follow DRY where it reduces real duplication or clarifies shared behavior, but
  do not introduce broad abstractions just to avoid a small amount of local
  repetition.
- Use async Home Assistant APIs and avoid blocking I/O in runtime code. Existing
  loaders use executor helpers for disk reads where needed.
- Preserve user-facing behavior when refactoring. Many toggles have backward
  compatibility fallbacks.
- Avoid broad refactors when a targeted fix is enough.
- Do not commit generated caches, `__pycache__`, `.pyc`, `.pyo`, local venvs, or
  `test-results/`.

## Documentation Tone

- Keep docs and UI copy plainspoken, concrete, and human. Explain what MCP
  Assist does and where model support can vary without making the project sound
  magical, autonomous, or universally best.
- Prefer targeted wording fixes over broad terminology sweeps. Do not rename
  established terms such as "Smart Entity Index" or existing config labels just
  to make copy feel less hyped.
- When recommending models or features, describe observed fit and tradeoffs
  instead of using phrases like "frontier", "very best", or "super AI".

## Home Assistant and HACS Notes

- HACS metadata is in `hacs.json`. It controls repository display behavior, not
  the HA integration domain.
- Home Assistant integration metadata is in
  `custom_components/mcp_assist/manifest.json`.
- HACS requires the integration files under
  `custom_components/INTEGRATION_NAME`; for this repo that is
  `custom_components/mcp_assist`.
- `hacs.json` keeps the public Home Assistant floor lenient. Treat changes to
  that floor as compatibility decisions, not routine maintenance.
- Brand assets are tested for supported filenames, dimensions, PNG format, and
  RGBA mode.

## PR and Maintenance Guidance

- Keep PRs as small, atomic, and easy to review as possible. Prefer one
  behavior change, fix, or maintenance concern per PR.
- Cover behavior changes with automated tests. For low-risk docs or metadata
  changes, run the relevant parser/format checks and explain why broader tests
  are unnecessary.
- For Dependabot PRs, keep the PRs open and use Dependabot's own rebase flow
  when possible. Do not close or supersede them unless asked.
- For dependency bumps, validate compatibility before merging. Runtime manifest
  lower bounds should remain lenient unless there is a real runtime need.
- For release-related changes, update the release workflow, `docs/releases.md`,
  and the README together when the human process changes.
- For GitHub Actions changes, remember CodeQL analyzes both `actions` and
  `python`.
- For docs-only changes, still run `git diff --check` and any cheap parser
  checks that apply.

## Things Not To Do Casually

- Do not rename `mcp_assist`.
- Do not remove upstream attribution.
- Do not make `hacs.json` or manifest minimum versions stricter as part of a
  routine dependency update.
- Do not add a `CHANGELOG.md` entry for every PR.
- Do not reintroduce a release workflow that requires manual release notes.
- Do not change `requirements_runtime.txt` mirror behavior to exact-match the
  manifest requirement strings.
