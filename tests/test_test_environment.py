"""Compatibility setup follows the selected release, including moving stable HA."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import test_environment as setup


def package_info(ha: str = "2026.10.1", python: str = ">=3.14") -> dict:
    return {"info": {"requires_dist": [f"homeassistant=={ha}"], "requires_python": python},
            "urls": [{"yanked": False}]}


def index(*versions: str) -> dict:
    return {"releases": {version: [{"yanked": False}] for version in versions}}


def test_latest_ignores_beta_yanked_and_mismatched_harnesses(monkeypatch) -> None:
    core = index("2026.10.1", "2026.11.0b0", "2026.10.2")
    core["releases"]["2026.10.2"][0]["yanked"] = True
    harness = index("0.13.372", "0.13.373", "0.13.374", "0.14.0b1")
    harness["releases"]["0.13.374"][0]["yanked"] = True
    responses = {
        ("homeassistant", None): core,
        ("homeassistant", "2026.10.1"): package_info(python=">=3.14.2"),
        (setup.HARNESS, None): harness,
        (setup.HARNESS, "0.13.373"): package_info("2026.11.0b0"),
        (setup.HARNESS, "0.13.372"): package_info(),
    }
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    assert setup.resolve_environment("latest") == {
        "homeassistant": "2026.10.1", "harness": "0.13.372", "python": "3.14",
    }


def test_latest_does_not_fall_back_when_harness_lags(monkeypatch) -> None:
    responses = {
        ("homeassistant", None): index("2026.11.0", "2026.10.1"),
        ("homeassistant", "2026.11.0"): package_info("2026.11.0"),
        (setup.HARNESS, None): index("0.13.372"),
        (setup.HARNESS, "0.13.372"): package_info(),
    }
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    with pytest.raises(ValueError, match="No published stable test harness matches Home Assistant 2026.11.0"):
        setup.resolve_environment("latest")


def test_latest_searches_past_thirty_harnesses_and_stops_at_match(monkeypatch) -> None:
    candidates = [f"0.13.{version}" for version in range(410, 374, -1)]
    harness = index(*candidates, "0.13.374", "0.13.411", "0.14.0b1")
    harness["releases"]["0.13.411"][0]["yanked"] = True
    match = "0.13.375"
    responses = {
        ("homeassistant", None): index("2026.10.1"),
        ("homeassistant", "2026.10.1"): package_info(python=">=3.14.2"),
        (setup.HARNESS, None): harness,
        **{(setup.HARNESS, version): package_info("2026.9.3") for version in candidates[:35]},
        (setup.HARNESS, match): package_info("2026.10.1", ">=3.14.1"),
    }
    lookup = Mock(side_effect=lambda package, version=None: responses[package, version])
    monkeypatch.setattr(setup, "fetch_metadata", lookup)

    assert setup.resolve_environment("latest") == {
        "homeassistant": "2026.10.1", "harness": match, "python": "3.14",
    }
    queried = [call.args[1] for call in lookup.call_args_list
               if call.args[0] == setup.HARNESS and len(call.args) > 1]
    assert queried == [*candidates[:35], match]
    assert "0.13.411" not in queried
    assert "0.13.374" not in queried


def test_latest_exhausts_more_than_thirty_harnesses_without_match(monkeypatch) -> None:
    candidates = [f"0.13.{version}" for version in range(410, 374, -1)]
    harness = index(*candidates, "0.13.411", "0.14.0b1")
    harness["releases"]["0.13.411"][0]["yanked"] = True
    responses = {
        ("homeassistant", None): index("2026.10.1"),
        ("homeassistant", "2026.10.1"): package_info(),
        (setup.HARNESS, None): harness,
        **{(setup.HARNESS, version): package_info("2026.9.3") for version in candidates},
    }
    lookup = Mock(side_effect=lambda package, version=None: responses[package, version])
    monkeypatch.setattr(setup, "fetch_metadata", lookup)

    with pytest.raises(ValueError, match="No published stable test harness matches Home Assistant 2026.10.1"):
        setup.resolve_environment("latest")
    queried = [call.args[1] for call in lookup.call_args_list
               if call.args[0] == setup.HARNESS and len(call.args) > 1]
    assert queried == candidates
    assert "0.13.411" not in queried


def test_default_uses_pinned_harness_and_future_python_minimum(monkeypatch, tmp_path) -> None:
    (tmp_path / "requirements_test.txt").write_text("pytest-homeassistant-custom-component==0.13.372\n")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    responses = {
        (setup.HARNESS, "0.13.372"): package_info("2026.11.0", ">=3.15"),
        ("homeassistant", "2026.11.0"): package_info("2026.11.0", ">=3.15.1"),
    }
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    assert setup.resolve_environment("default")["python"] == "3.15"


@pytest.mark.parametrize("target", ["preview", "2026.13.1", "2027.2.1b0"])
def test_unknown_target_and_malformed_expected_harness_fail_before_network(monkeypatch, target) -> None:
    fetch = Mock()
    monkeypatch.setattr(setup, "fetch_metadata", fetch)
    with pytest.raises(ValueError, match="Unknown test target"):
        setup.resolve_environment(target)
    with pytest.raises(ValueError, match="exact stable version"):
        setup.resolve_environment("2027.2.1", "1.2b1")
    fetch.assert_not_called()


@pytest.mark.parametrize("harness", ["1.2", "1.2.3.1"])
@pytest.mark.parametrize("target", ["default", "latest", "2027.2.1"])
def test_stable_harness_versions_keep_supported_release_shapes(monkeypatch, tmp_path, harness, target) -> None:
    (tmp_path / "requirements_test.txt").write_text(f"{setup.HARNESS}=={harness}\n")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    responses = {
        ("homeassistant", None): index("2027.2.1"),
        ("homeassistant", "2027.2.1"): package_info("2027.2.1"),
        (setup.HARNESS, None): index(harness),
        (setup.HARNESS, harness): package_info("2027.2.1"),
    }
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    expected = {"homeassistant": "2027.2.1", "harness": harness, "python": "3.14"}
    assert setup.resolve_environment(target) == expected
    assert setup.resolve_environment(target, harness) == expected
    assert setup.validate_environment(expected) == expected


def test_expected_harness_must_match_the_exact_home_assistant_pair(monkeypatch) -> None:
    responses = {
        ("homeassistant", "2027.2.1"): package_info("2027.2.1"),
        (setup.HARNESS, "1.2.3"): package_info("2027.1.0"),
    }
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    with pytest.raises(ValueError, match="Requested harness does not match Home Assistant 2027.2.1"):
        setup.resolve_environment("2027.2.1", "1.2.3")


def test_expected_harness_rejects_yanked_distribution(monkeypatch) -> None:
    responses = {
        ("homeassistant", "2027.2.1"): package_info("2027.2.1"),
        (setup.HARNESS, "1.2.3"): package_info("2027.2.1"),
    }
    responses[setup.HARNESS, "1.2.3"]["urls"] = [{"yanked": True}]
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    with pytest.raises(ValueError, match=f"No published non-yanked distribution for {setup.HARNESS}"):
        setup.resolve_environment("2027.2.1", "1.2.3")


@pytest.mark.parametrize("value", [
    {"homeassistant": "2026.10.1", "harness": "0.13.372", "python": "3.14", "private": "hidden"},
    {"homeassistant": "2026.11.0b0", "harness": "0.13.372", "python": "3.14"},
    {"homeassistant": "2026.10.1", "harness": "0.13.372\nunsafe=value", "python": "3.14"},
])
def test_environment_artifact_rejects_nonpublic_or_unstable_fields(value) -> None:
    with pytest.raises(ValueError):
        setup.validate_environment(value)


def test_install_keeps_pair_in_one_resolution_and_reads_selected_core_dependencies(monkeypatch, tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "requirements_test.txt").write_text("pytest-homeassistant-custom-component==0.13.999\nruff>=0.17\n")
    package = tmp_path / "homeassistant"
    manifest = package / "components" / "conversation" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"requirements": ["hassil==3.12.1", "gazetteer-matcher==1.1.0"]}))
    calls = []
    common = []

    def run(command, **kwargs):
        calls.append(command)
        if "-r" in command:
            common.append(Path(command[command.index("-r") + 1]).read_text())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(setup, "ROOT", root)
    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(package / "__init__.py")))
    monkeypatch.setattr(setup.importlib.metadata, "version", lambda _: "0.13.372")
    from homeassistant import const as core_const
    from pytest_homeassistant_custom_component import const as harness_const
    monkeypatch.setattr(core_const, "__version__", "2026.10.1")
    monkeypatch.setattr(harness_const, "MAJOR_VERSION", 2026)
    monkeypatch.setattr(harness_const, "MINOR_VERSION", 10)
    monkeypatch.setattr(harness_const, "PATCH_VERSION", 1)
    setup.install_environment({"homeassistant": "2026.10.1", "harness": "0.13.372", "python": "3.14"})
    assert "homeassistant==2026.10.1" in calls[0]
    assert "pytest-homeassistant-custom-component==0.13.372" in calls[0]
    assert setup.HARNESS not in common[0]
    assert calls[1][-2:] == ["hassil==3.12.1", "gazetteer-matcher==1.1.0"]
    assert calls[2][-1] == "check"


@pytest.mark.parametrize("target", ["2026.8.3", "2026.9.3", "2027.2.1", "default"])
@pytest.mark.parametrize("package", ["homeassistant", setup.HARNESS])
@pytest.mark.parametrize("files", [[], [{"yanked": True}]])
def test_pinned_environments_reject_missing_or_yanked_distributions(
    monkeypatch, tmp_path, target, package, files,
) -> None:
    (tmp_path / "requirements_test.txt").write_text(
        "pytest-homeassistant-custom-component==0.13.372\n"
    )
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    ha = "2026.10.1" if target == "default" else target
    harness = "0.13.372" if target == "default" else "1.2.3"
    responses = {
        ("homeassistant", ha): package_info(ha),
        (setup.HARNESS, harness): package_info(ha),
    }
    if target != "default":
        responses[(setup.HARNESS, None)] = index(harness)
    responses[package, ha if package == "homeassistant" else harness]["urls"] = files
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    with pytest.raises(ValueError, match=f"No published non-yanked distribution for {package}"):
        setup.resolve_environment(target)


def test_exact_target_selects_compatible_harness_and_python_metadata(monkeypatch) -> None:
    responses = {
        ("homeassistant", "2027.2.1"): package_info("2027.2.1", ">=3.14.2"),
        (setup.HARNESS, None): index("1.2.4", "1.2.3", "1.3.0b1"),
        (setup.HARNESS, "1.2.4"): package_info("2027.1.0"),
        (setup.HARNESS, "1.2.3"): package_info("2027.2.1", ">=3.15"),
    }
    responses[(setup.HARNESS, None)]["releases"]["1.2.4"][0]["yanked"] = True
    monkeypatch.setattr(setup, "fetch_metadata", lambda p, v=None: responses[p, v])
    assert setup.resolve_environment("2027.2.1") == {
        "homeassistant": "2027.2.1", "harness": "1.2.3", "python": "3.15",
    }
    assert setup.resolve_environment("2027.2.1", "1.2.3")["harness"] == "1.2.3"
    with pytest.raises(ValueError, match="Requested harness does not match Home Assistant"):
        setup.resolve_environment("2027.2.1", "1.2.4")
