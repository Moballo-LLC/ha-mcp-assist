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
STABLE = re.compile(r"\d+(?:\.\d+)+\Z")
HA_VERSION = re.compile(r"\d{4}\.(?:[1-9]|1[0-2])\.\d+\Z")
HARNESS = "pytest-homeassistant-custom-component"
OFFICIAL_HARNESS = "pytest-homeassistant"


def harness_package(ha: str) -> str:
    """Prefer Core's matching published harness, retaining older release coverage."""
    published = stable_releases(fetch_metadata(OFFICIAL_HARNESS))
    return OFFICIAL_HARNESS if ha in published else HARNESS


def release_matrix() -> dict[str, list[str]]:
    """Test the newest stable patch from each of the three most recent release months."""
    selected = {}
    for version in stable_releases(fetch_metadata("homeassistant")):
        if HA_VERSION.fullmatch(version):
            month = tuple(map(int, version.split(".")[:2]))
            selected.setdefault(month, version)
            if len(selected) == 3:
                break
    if len(selected) != 3:
        raise ValueError("Need three published stable Home Assistant release months")
    return {"homeassistant": list(selected.values())}


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


def has_published_distribution(metadata: dict) -> bool:
    return any(not file.get("yanked", False) for file in metadata.get("urls") or [])


def harness_ha(metadata: dict) -> str:
    versions = set()
    for requirement in metadata["info"].get("requires_dist") or []:
        match = re.match(r"homeassistant\s*==\s*(\d{4}\.\d+\.\d+)(?:\s*;.*)?\Z", requirement, re.I)
        if match and HA_VERSION.fullmatch(match[1]):
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
    if not isinstance(value, dict) or set(value) not in ({"homeassistant", "harness", "python"},
            {"homeassistant", "harness", "python", "harness_package"}) or not all(isinstance(v, str) for v in value.values()):
        raise ValueError("Invalid resolved test environment")
    if not HA_VERSION.fullmatch(value["homeassistant"]) or not STABLE.fullmatch(value["harness"]):
        raise ValueError("Test environments require exact stable package versions")
    package = value.get("harness_package", HARNESS)
    if package not in (HARNESS, OFFICIAL_HARNESS):
        raise ValueError("Unknown test harness package")
    if package == OFFICIAL_HARNESS and value["harness"] != value["homeassistant"]:
        raise ValueError("Official harness version must match Home Assistant")
    if not re.fullmatch(r"3\.\d+", value["python"]):
        raise ValueError("Invalid resolved Python version")
    return value


def _select_harness(ha: str, expected_harness: str | None = None,
                    package: str = HARNESS) -> tuple[str, dict]:
    if package == OFFICIAL_HARNESS:
        if expected_harness is not None and expected_harness != ha:
            raise ValueError(f"Requested harness does not match Home Assistant {ha}")
        metadata = fetch_metadata(package, ha)
        if not has_published_distribution(metadata):
            raise ValueError(f"No published non-yanked distribution for {package}")
        if harness_ha(metadata) != ha:
            raise ValueError(f"Official harness does not match Home Assistant {ha}")
        return ha, metadata
    if expected_harness is not None:
        if not isinstance(expected_harness, str) or not STABLE.fullmatch(expected_harness):
            raise ValueError("Requested harness must be an exact stable version")
        metadata = fetch_metadata(HARNESS, expected_harness)
        if not has_published_distribution(metadata):
            raise ValueError(f"No published non-yanked distribution for {HARNESS}")
        if harness_ha(metadata) != ha:
            raise ValueError(f"Requested harness does not match Home Assistant {ha}")
        return expected_harness, metadata

    harness_index = fetch_metadata(HARNESS)
    unpublished_match = False
    for harness in stable_releases(harness_index):
        metadata = fetch_metadata(HARNESS, harness)
        try:
            matches = harness_ha(metadata) == ha
        except ValueError:
            continue
        if matches:
            if not has_published_distribution(metadata):
                unpublished_match = True
                continue
            return harness, metadata
    if unpublished_match:
        raise ValueError(f"No published non-yanked distribution for {HARNESS}")
    raise ValueError(f"No published stable test harness matches Home Assistant {ha}")


def resolve_environment(target: str, expected_harness: str | None = None) -> dict[str, str]:
    if expected_harness is not None and (
        not isinstance(expected_harness, str) or not STABLE.fullmatch(expected_harness)
    ):
        raise ValueError("Requested harness must be an exact stable version")
    if target in ("default", "latest"):
        core_index = fetch_metadata("homeassistant")
        versions = [v for v in stable_releases(core_index) if HA_VERSION.fullmatch(v)]
        if not versions:
            raise ValueError("No published stable Home Assistant release")
        ha = versions[0]
        core_metadata = fetch_metadata("homeassistant", ha)
        if not has_published_distribution(core_metadata):
            raise ValueError("No published non-yanked distribution for homeassistant")
        package = harness_package(ha)
        harness, harness_metadata = _select_harness(ha, package=package)
        if expected_harness is not None and expected_harness != harness:
            raise ValueError("Requested harness does not match latest stable Home Assistant")
    elif isinstance(target, str) and HA_VERSION.fullmatch(target):
        ha = target
        core_metadata = fetch_metadata("homeassistant", ha)
        if not has_published_distribution(core_metadata):
            raise ValueError("No published non-yanked distribution for homeassistant")
        package = harness_package(ha)
        harness, harness_metadata = _select_harness(ha, expected_harness, package)
    else:
        raise ValueError("Unknown test target; use an exact stable Home Assistant version, default, or latest")
    if not has_published_distribution(harness_metadata):
        raise ValueError(f"No published non-yanked distribution for {package}")
    if harness_ha(harness_metadata) != ha:
        raise ValueError("The selected test harness does not match Home Assistant")
    environment = {"homeassistant": ha, "harness": harness,
                   "python": python_minor(core_metadata, harness_metadata)}
    environment["harness_package"] = package
    return validate_environment(environment)


def install_environment(environment: dict) -> None:
    environment = validate_environment(environment)
    required_python = tuple(map(int, environment["python"].split(".")))
    if sys.version_info[:2] != required_python:
        raise ValueError(f"Use Python {environment['python']} for this resolved test environment")
    lines = (ROOT / "requirements_test.txt").read_text().splitlines()
    package = environment.get("harness_package", HARNESS)
    other_package = HARNESS if package == OFFICIAL_HARNESS else OFFICIAL_HARNESS
    common = [line for line in lines
              if not re.match(rf"^(?:{HARNESS}|{OFFICIAL_HARNESS})(?:[<>=!~;\s\[]|$)", line)]
    # The plugins supply overlapping fixtures and must never be loaded together.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", other_package],
                   check=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as requirements:
        requirements.write("\n".join(common) + "\n")
        requirements.flush()
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements.name,
                        f"homeassistant=={environment['homeassistant']}",
                        f"{package}=={environment['harness']}"], check=True)
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
    installed_metadata = {"info": {"requires_dist": importlib.metadata.requires(package)}}
    if (actual_ha != environment["homeassistant"] or harness_ha(installed_metadata) != actual_ha
            or importlib.metadata.version(package) != environment["harness"]):
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
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--github-output", type=Path)
    install = subparsers.add_parser("install")
    install.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "matrix":
            serialized = json.dumps(release_matrix(), separators=(",", ":"))
            if args.github_output:
                with args.github_output.open("a") as output:
                    output.write(f"matrix={serialized}\n")
            print(serialized)
            return
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
