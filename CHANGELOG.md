# Changelog

## 1.0.0 - Pending

MCP Assist 1.0.0 is the first Moballo-maintained release of the fork. It builds on the original `mike-nott/mcp-assist` project, preserves upstream credit and license history, and turns this repository into the supported HACS-facing home for future releases.

### Fork Stewardship

- Added clear README attribution to the upstream `mike-nott/mcp-assist` project while updating installation, support, documentation, issue, discussion, and OpenRouter referer links for `Moballo-LLC/ha-mcp-assist`.
- Added GitHub Sponsors metadata for `Jason-Morcos`.
- Updated the MIT license notice to retain the original upstream copyright and add Moballo-maintained change coverage.
- Replaced the inherited/upstream-looking artwork with a distinct MCP-mark-based MCP Assist icon and logo set, including light, dark, and 2x assets for Home Assistant and HACS surfaces.

### HACS and Releases

- Added HACS metadata for README rendering and US country metadata while keeping the public Home Assistant support floor lenient at `2024.1.0`.
- Added direct HACS validation, scheduled HACS validation, Hassfest validation, and release packaging automation.
- Added `info.md` and expanded `CHANGELOG.md` so the repository is ready for a future `hacs/default` submission after a real GitHub release is published.
- Added a tag/manual release workflow that builds and uploads `mcp_assist.zip` for HACS installs.
- Updated the README Add to HACS button for `Moballo-LLC/ha-mcp-assist`.

### Integration and Tool Runtime

- Split large MCP server behavior into focused server tool modules for calendar, recorder, response services, and weather.
- Added packaged custom-tool support for recorder history, response services, weather forecasts, and Music Assistant.
- Preserved packaged tool prompt guidance and tightened boundaries between provider setup, packaged tools, and runtime execution.
- Improved MCP tool cache performance and reduced repeated tool metadata work.
- Passed user and location context through to MCP tools while keeping tests free of real coordinates.

### Fixes and Compatibility

- Fixed merged provider settings regressions.
- Fixed recorder history count windows and added regression coverage.
- Kept runtime requirements lenient so older supported Home Assistant installs are not blocked by the fork metadata.
- Moved CI to the current Home Assistant 2026.6 / Python 3.14 test harness without adding an arbitrary cap below Home Assistant 2026.7.

### Maintenance and Security

- Added Dependabot coverage for GitHub Actions and Python dependency mirrors, with grouped minor and patch update PRs and major updates left visible.
- Added a runtime requirements mirror plus a sync test so Dependabot can see Home Assistant manifest dependencies without making the manifest requirements less lenient.
- Fixed the pytest workflow so the main test run always creates `test-results/pytest.xml` before artifact upload.
- Added brand asset validation tests.
- Added full-history Gitleaks secret scanning with a narrow allowlist for inherited Home Assistant config constant-name false positives.
