"""Tests for release validation helper scripts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap

import pytest
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


def test_release_workflow_installs_resolved_core_conversation_dependencies() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["package"]["steps"]
    resolve = next(step for step in steps if step.get("id") == "environment")
    install = next(step for step in steps if step.get("name") == "Install test dependencies")
    validate = next(step for step in steps if step.get("name") == "Validate release candidate")
    assert "test_environment.py resolve default" in resolve["run"]
    assert "--github-output" in resolve["run"]
    snapshot = '"$RUNNER_TEMP/ha-test-environment.json"'
    assert f"--output {snapshot}" in resolve["run"]
    assert f"scripts/install_test_dependencies.sh --environment {snapshot}" in install["run"]
    target_python = next(step for step in steps if
                         step.get("with", {}).get("python-version") ==
                         "${{ steps.environment.outputs.python }}")
    assert steps.index(resolve) < steps.index(target_python) < steps.index(install)
    assert steps.index(install) < steps.index(validate)
    assert validate["run"] == 'PYTHON="$(command -v python)" scripts/verify_release_candidate.sh'


def test_test_workflow_runs_full_suite_with_independent_latest_resolution() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
    job = workflow["jobs"]["test"]
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["homeassistant"] == ["2026.8.3", "2026.9.3", "latest"]
    assert not job.get("continue-on-error", False)
    steps = job["steps"]
    resolve = next(step for step in steps if step.get("id") == "environment")
    assert "test_environment.py resolve" in resolve["run"]
    assert "--github-output" in resolve["run"]
    assert "--output" in resolve["run"]
    assert any(step.get("with", {}).get("python-version") == "${{ steps.environment.outputs.python }}" for step in steps)
    assert any("install_test_dependencies.sh --environment" in step.get("run", "") for step in steps)
    assert any("verify_local.sh --pytest" in step.get("run", "") for step in steps)
    artifact = next(step for step in steps if "upload-artifact" in step.get("uses", ""))
    assert "test-results/pytest.xml" in artifact["with"]["path"]
    assert "test-results/environment.json" in artifact["with"]["path"]


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


def test_all_workflow_verifiers_use_the_selected_python() -> None:
    for filename in ("tests.yml", "release.yml"):
        workflow = yaml.safe_load((Path(".github/workflows") / filename).read_text())
        verifiers = [step["run"] for job in workflow["jobs"].values()
                     for step in job["steps"]
                     if "scripts/verify_" in step.get("run", "")]
        assert verifiers
        for run in verifiers:
            verify_lines = [line.strip() for line in run.splitlines() if "scripts/verify_" in line]
            assert all(line.startswith('PYTHON="$(command -v python)" ') for line in verify_lines)


@pytest.mark.parametrize("modern", [False, True])
def test_release_environment_setup_supports_existing_tags(tmp_path, modern) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/release.yml").read_text())
    steps = workflow["jobs"]["package"]["steps"]
    resolve = next(step for step in steps if step.get("id") == "environment")
    install = next(step for step in steps if step.get("name") == "Install test dependencies")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    if modern:
        (scripts / "test_environment.py").touch()
        installer = scripts / "install_test_dependencies.sh"
        installer.write_text('#!/bin/bash\nprintf "%s\\n" "installer $*" >> "$COMMAND_LOG"\n')
        installer.chmod(0o755)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text(textwrap.dedent("""\
        #!/bin/bash
        printf '%s\\n' "python $*" >> "$COMMAND_LOG"
        if [[ "$1" == "scripts/test_environment.py" ]]; then
          echo '{"homeassistant":"2026.10.1","harness":"0.13.372","python":"3.15"}' > "$RUNNER_TEMP/ha-test-environment.json"
          echo "python=3.15" >> "$GITHUB_OUTPUT"
        fi
    """))
    python.chmod(0o755)
    log = tmp_path / "commands.log"
    output = tmp_path / "outputs"
    environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(output),
                   "COMMAND_LOG": str(log)}
    for step in (resolve, install):
        subprocess.run(["bash", "-e", "-c", step["run"]], cwd=tmp_path,
                       env=environment, check=True, capture_output=True, text=True)
    if modern:
        assert output.read_text().strip() == "python=3.15"
        assert log.read_text().splitlines() == [
            f"python scripts/test_environment.py resolve default --output {tmp_path}/ha-test-environment.json --github-output {output}",
            f"installer --environment {tmp_path}/ha-test-environment.json",
        ]
    else:
        assert output.read_text().strip() == "python=3.14"
        assert log.read_text().splitlines() == [
            "python -m pip install --upgrade pip wheel",
            "python -m pip install -r requirements_test.txt",
        ]
