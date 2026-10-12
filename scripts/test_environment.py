"""Resolve and install mutually compatible Home Assistant test environments."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"2026.8.3": "0.13.357", "2026.9.3": "0.13.366"}
STABLE = re.compile(r"\d+(?:\.\d+)+\Z")
HA_VERSION = re.compile(r"\d{4}\.\d{1,2}\.\d+\Z")
HARNESS = "pytest-homeassistant-custom-component"


def fetch_metadata(package: str, version: str | None = None) -> dict:
    suffix = f"/{version}" if version else ""
    url = f"https://pypi.org/pypi/{package}{suffix}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        raise ValueError(f"Public package metadata is unavailable for {package}") from exc


def stable_releases(metadata: dict) -> list[str]:
    versions = [version for version, files in metadata["releases"].items()
                if STABLE.fullmatch(version) and files and any(not f.get("yanked", False) for f in files)]
    return sorted(versions, key=lambda v: tuple(map(int, v.split("."))), reverse=True)


def harness_ha(metadata: dict) -> str:
    versions = set()
    for requirement in metadata["info"].get("requires_dist") or []:
        match = re.match(r"homeassistant\s*==\s*(\d{4}\.\d+\.\d+)(?:\s*;.*)?\Z", requirement, re.I)
        if match:
            versions.add(match[1])
    if len(versions) != 1:
        raise ValueError("The test harness must declare one exact stable Home Assistant version")
    return versions.pop()


def python_minor(*metadata: dict) -> str:
    bounds = []
    for item in metadata:
        requirement = item["info"].get("requires_python") or ""
        matches = re.findall(r"(?:^|,)\s*>=\s*(\d+)\.(\d+)", requirement)
        if not matches:
            raise ValueError("Package metadata must declare a Python minimum")
        bounds.extend((int(major), int(minor)) for major, minor in matches)
    major, minor = max(bounds)
    if major != 3:
        raise ValueError("Unsupported Python major in package metadata")
    return f"{major}.{minor}"


def validate_environment(value: dict) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"homeassistant", "harness", "python"} or not all(isinstance(v, str) for v in value.values()):
        raise ValueError("Invalid resolved test environment")
    if not HA_VERSION.fullmatch(value["homeassistant"]) or not STABLE.fullmatch(value["harness"]):
        raise ValueError("Test environments require exact stable package versions")
    if not re.fullmatch(r"3\.\d+", value["python"]):
        raise ValueError("Invalid resolved Python version")
    return value


def resolve_environment(target: str, expected_harness: str | None = None) -> dict[str, str]:
    if target in BASELINES:
        harness = BASELINES[target]
        if expected_harness is not None and expected_harness != harness:
            raise ValueError("Requested harness does not match the selected baseline")
        return validate_environment({"homeassistant": target, "harness": harness, "python": "3.14"})
    if target == "default":
        requirements = (ROOT / "requirements_test.txt").read_text()
        match = re.search(r"^pytest-homeassistant-custom-component==([0-9.]+)$", requirements, re.M)
        if not match:
            raise ValueError("Default test harness requires an exact pin")
        harness = match[1]
        if expected_harness is not None and expected_harness != harness:
            raise ValueError("Requested harness does not match the default pin")
        harness_metadata = fetch_metadata(HARNESS, harness)
        ha = harness_ha(harness_metadata)
        core_metadata = fetch_metadata("homeassistant", ha)
    elif target == "latest":
        core_index = fetch_metadata("homeassistant")
        versions = [v for v in stable_releases(core_index) if HA_VERSION.fullmatch(v)]
        if not versions:
            raise ValueError("No published stable Home Assistant release")
        ha = versions[0]
        core_metadata = fetch_metadata("homeassistant", ha)
        harness_index = fetch_metadata(HARNESS)
        for harness in stable_releases(harness_index):
            harness_metadata = fetch_metadata(HARNESS, harness)
            try:
                matched = harness_ha(harness_metadata) == ha
            except ValueError:
                continue
            if matched:
                break
        else:
            raise ValueError(f"No published stable test harness matches Home Assistant {ha}")
        if expected_harness is not None and expected_harness != harness:
            raise ValueError("Requested harness does not match latest stable Home Assistant")
    else:
        raise ValueError("Unknown test target; use a maintained baseline, default, or latest")
    return validate_environment({"homeassistant": ha, "harness": harness,
                                 "python": python_minor(core_metadata, harness_metadata)})


def install_environment(environment: dict) -> None:
    environment = validate_environment(environment)
    required_python = tuple(map(int, environment["python"].split(".")))
    if sys.version_info[:2] != required_python:
        raise ValueError(f"Use Python {environment['python']} for this resolved test environment")
    lines = (ROOT / "requirements_test.txt").read_text().splitlines()
    common = [line for line in lines if not line.startswith(HARNESS + "==")]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as requirements:
        requirements.write("\n".join(common) + "\n")
        requirements.flush()
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements.name,
                        f"homeassistant=={environment['homeassistant']}",
                        f"{HARNESS}=={environment['harness']}"], check=True)
    spec = importlib.util.find_spec("homeassistant")
    if spec is None or spec.origin is None:
        raise ValueError("Installed Home Assistant package was not found")
    manifest = Path(spec.origin).parent / "components" / "conversation" / "manifest.json"
    conversation_requirements = json.loads(manifest.read_text())["requirements"]
    if not isinstance(conversation_requirements, list) or not all(isinstance(r, str) for r in conversation_requirements):
        raise ValueError("Invalid Core conversation dependency manifest")
    if conversation_requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", *conversation_requirements], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    from homeassistant.const import __version__ as actual_ha
    from pytest_homeassistant_custom_component import const
    harness_ha_version = f"{const.MAJOR_VERSION}.{const.MINOR_VERSION}.{const.PATCH_VERSION}"
    if (actual_ha != environment["homeassistant"] or harness_ha_version != actual_ha
            or importlib.metadata.version(HARNESS) != environment["harness"]):
        raise ValueError("Installed HA and test harness do not match the resolved environment")
    print(f"Verified Home Assistant {actual_ha} with harness {environment['harness']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("target")
    resolve.add_argument("--expected-harness")
    resolve.add_argument("--output", type=Path)
    resolve.add_argument("--github-output", type=Path)
    install = subparsers.add_parser("install")
    install.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "install":
            install_environment(json.loads(args.environment.read_text()))
            return
        environment = resolve_environment(args.target, args.expected_harness)
        serialized = json.dumps(environment, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(serialized)
        if args.github_output:
            with args.github_output.open("a") as output:
                for key, value in environment.items():
                    output.write(f"{key}={value}\n")
        print(serialized, end="")
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"Test environment setup failed: {exc}\n")
    except OSError:
        parser.exit(2, "Test environment setup could not read or write an environment file\n")


if __name__ == "__main__":
    main()
