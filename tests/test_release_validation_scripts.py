"""Tests for release validation helper scripts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

import yaml


def test_release_candidate_ancestry_fetch_preserves_main_history() -> None:
    script = Path("scripts/verify_release_candidate.sh").read_text(encoding="utf-8")
    function_start = script.index("fetch_main_for_ancestry_check()")
    function_end = script.index('if [[ "${REQUIRE_MAIN_ANCESTOR:-0}" == "1" ]]')
    fetch_function = script[function_start:function_end]
    require_main_block = script[function_end : script.index('if [[ "${SKIP_LOCAL_VERIFY:-0}"')]

    assert "--depth" not in fetch_function
    assert "--unshallow" in fetch_function
    assert "+refs/heads/main:refs/remotes/origin/main" in fetch_function
    assert "fetch_main_for_ancestry_check" in require_main_block
    assert "git merge-base --is-ancestor HEAD origin/main" in require_main_block


def test_release_workflow_keeps_validation_jobs_read_only() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    assert workflow["permissions"] == {"contents": "read"}
    assert jobs["hacs"]["permissions"] == {"contents": "read"}
    assert jobs["hassfest"]["permissions"] == {"contents": "read"}
    assert jobs["package"]["permissions"] == {"contents": "write"}


def test_release_workflow_uploads_assets_before_publishing() -> None:
    """Immutable releases reject asset uploads once published."""
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["package"]["steps"]
    publish = next(step for step in steps if step.get("name") == "Publish GitHub release")
    script = publish["run"]

    assert "softprops/action-gh-release" not in publish.get("uses", "")
    assert "--draft" in script
    assert "dist/mcp_assist.zip" in script
    assert script.index("gh release create") < script.index("--draft=false")
    assert script.index("gh release upload") < script.index("--draft=false")


def test_test_workflow_runs_full_suite_with_version_matched_harnesses() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text(encoding="utf-8"))
    test_job = workflow["jobs"]["test"]
    matrix = test_job["strategy"]["matrix"]["include"]

    rows = {(item["homeassistant"], item["harness-version"]) for item in matrix}
    assert len(rows) == len(matrix) == 3
    assert all(version.startswith("2026.") and harness.startswith("0.13.") for version, harness in rows)
    current_harness = next(line.split("==", 1)[1] for line in Path("requirements_test.txt").read_text().splitlines() if line.startswith("pytest-homeassistant-custom-component=="))
    assert ("2026.10.0", current_harness) in rows
    install_step = next(step for step in test_job["steps"] if step.get("name") == "Install test dependencies")
    pytest_step = next(step for step in test_job["steps"] if step.get("name") == "Run full pytest suite")
    artifact_step = next(step for step in test_job["steps"] if step.get("name") == "Upload pytest results")

    assert "${{ matrix.harness-version }}" in install_step["run"]
    assert "${{ matrix.homeassistant }}" in install_step["run"]
    assert "scripts/verify_local.sh --pytest" in pytest_step["run"]
    assert artifact_step["with"]["path"] == "test-results/pytest.xml"
    assert artifact_step["with"]["name"] == "pytest-results-ha-${{ matrix.homeassistant }}"


def test_test_workflow_preserves_required_fail_closed_aggregate_status() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text(encoding="utf-8"))
    aggregate = workflow["jobs"]["test-status"]
    steps = aggregate["steps"]

    assert aggregate["name"] == "Test"
    assert aggregate["if"] == "always()"
    assert aggregate["needs"] == ["test"]
    assert aggregate["timeout-minutes"] > 0
    assert len(steps) == 1
    assert steps[0]["env"]["MATRIX_RESULT"] == "${{ needs.test.result }}"
    assert '"$MATRIX_RESULT" != "success"' in steps[0]["run"]
    assert "exit 1" in steps[0]["run"]

    for result, expected_returncode in (
        ("success", 0),
        ("failure", 1),
        ("cancelled", 1),
        ("skipped", 1),
    ):
        completed = subprocess.run(
            ["bash", "-c", steps[0]["run"]],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ | {"MATRIX_RESULT": result},
        )
        assert completed.returncode == expected_returncode


def _run_test_installer(*args: str, fail_version_check: bool = False) -> tuple[subprocess.CompletedProcess[str], str, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        fake_python = temp / "python"
        calls_file = temp / "calls"
        requirements_file = temp / "requirements"
        fake_python.write_text(
            "#!/bin/sh\n"
            'printf "%s\\n" "$*" >> "$CALLS_FILE"\n'
            'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
            '  shift 2\n'
            '  if [ "$1" = "install" ] && [ "$2" = "-r" ]; then cat "$3" > "$REQUIREMENTS_FILE"; fi\n'
            "fi\n"
            'if [ "$1" = "-" ]; then cat >/dev/null; if [ "${VERSION_CHECK_EXIT:-0}" = "0" ]; then printf "Verified Home Assistant %s with harness %s\\n" "$2" "$3"; fi; exit "${VERSION_CHECK_EXIT:-0}"; fi\n',
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = os.environ | {
            "CALLS_FILE": str(calls_file),
            "REQUIREMENTS_FILE": str(requirements_file),
            "VERSION_CHECK_EXIT": "1" if fail_version_check else "0",
        }
        result = subprocess.run(
            ["bash", "scripts/install_test_dependencies.sh", *args, str(fake_python)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        return result, calls_file.read_text() if calls_file.exists() else "", requirements_file.read_text() if requirements_file.exists() else ""


def test_versioned_test_installer_rejects_unknown_and_mismatched_pairs_before_pip() -> None:
    unsupported, unsupported_calls, _ = _run_test_installer("2026.7.0")
    mismatched, mismatched_calls, _ = _run_test_installer("2026.9.3", "0.13.371")

    assert unsupported.returncode == 2
    assert "Unsupported HA version" in unsupported.stderr
    assert unsupported_calls == ""
    assert mismatched.returncode == 2
    assert "does not match HA" in mismatched.stderr
    assert mismatched_calls == ""


def test_versioned_test_installer_filters_default_and_installs_matching_harness() -> None:
    result, calls, common_requirements = _run_test_installer("2026.9.3", "0.13.366")

    assert result.returncode == 0, result.stderr
    assert "pytest-homeassistant-custom-component==" not in common_requirements
    assert "pytest-homeassistant-custom-component==0.13.366" in calls
    assert "Verified Home Assistant 2026.9.3 with harness 0.13.366" in result.stdout


def test_versioned_test_installer_fails_when_harness_readback_fails() -> None:
    result, calls, _ = _run_test_installer("2026.8.3", "0.13.357", fail_version_check=True)

    assert result.returncode == 1
    assert "- 2026.8.3 0.13.357" in calls
