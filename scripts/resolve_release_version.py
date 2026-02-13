"""Resolve release version from GitHub ref and git tags.

Behavior:
- On tag refs (refs/tags/vX.Y.Z), use X.Y.Z.
- Otherwise, bump patch from latest v* tag.
- If no tags exist, default to 0.1.0.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def normalize_version(raw: str) -> str:
    """Normalize a semantic version or v-prefixed tag to MAJOR.MINOR.PATCH."""
    match = _VERSION_RE.match(raw.strip())
    if match is None:
        raise ValueError(f"Invalid version/tag format: {raw}")
    major, minor, patch = match.groups()
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def bump_patch(version: str) -> str:
    """Return the next patch version for the provided semantic version."""
    normalized = normalize_version(version)
    major, minor, patch = normalized.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def resolve_version(ref: str, ref_name: str, latest_tag: Optional[str]) -> str:
    """Resolve package version from GitHub ref metadata and latest release tag."""
    if ref.startswith("refs/tags/"):
        return normalize_version(ref_name)
    if latest_tag is None or latest_tag == "":
        return "0.1.0"
    return bump_patch(latest_tag)


def latest_version_tag() -> Optional[str]:
    """Return the latest `v*` git tag by semantic version order, if available."""
    result = subprocess.run(
        ["git", "tag", "-l", "v*", "--sort=-v:refname"],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return tags[0] if tags else None


def main() -> int:
    """Resolve and print the release version for workflow consumption."""
    ref = os.environ.get("GITHUB_REF", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    tag = latest_version_tag()
    print(resolve_version(ref, ref_name, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
