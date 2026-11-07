"""Installs must use exact, hash-checked versions, so a fresh install can't differ from what was reviewed and tested."""

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).parent.parent
LOCKS = ["requirements.lock", "requirements-dev.lock"]
LOCKED_LINE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s;]+)(?:\s*;[^\\]*)?(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})+)$")


def entries(file_name: str) -> list[str]:
    """The file's requirements, with comments removed and continuation lines joined."""
    text = (ROOT / file_name).read_text()
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    return [entry.strip() for entry in re.sub(r"\\\n\s*", " ", "\n".join(lines)).splitlines() if entry.strip()]


def locked_versions(file_name: str) -> dict[str, str]:
    versions = {}
    for entry in entries(file_name):
        match = LOCKED_LINE.match(entry)
        assert match, f"{file_name}: not an exact, hashed requirement: {entry[:80]}"
        versions[canonicalize_name(match["name"])] = match["version"]
    return versions


def direct_requirements(file_name: str) -> list[Requirement]:
    return [Requirement(entry) for entry in entries(file_name) if not entry.startswith("-r")]


@pytest.mark.parametrize("lock", LOCKS)
def test_every_locked_package_has_an_exact_version_and_hash(lock):
    assert locked_versions(lock)


@pytest.mark.parametrize(
    "lock, inputs",
    [
        ("requirements.lock", ["requirements.txt"]),
        ("requirements-dev.lock", ["requirements.txt", "requirements-dev.txt"]),
    ],
)
def test_lock_satisfies_its_direct_requirements(lock, inputs):
    versions = locked_versions(lock)
    for input_file in inputs:
        for requirement in direct_requirements(input_file):
            name = canonicalize_name(requirement.name)
            assert name in versions, f"{lock} is missing {name}: regenerate it"
            assert requirement.specifier.contains(versions[name], prereleases=True), (
                f"{lock} has {name}=={versions[name]}, outside {requirement.specifier}: regenerate it"
            )


def test_dev_lock_matches_the_runtime_lock():
    runtime = locked_versions("requirements.lock")
    dev = locked_versions("requirements-dev.lock")
    assert {name: dev.get(name) for name in runtime} == runtime
