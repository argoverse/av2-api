"""Validate release tag matches rust/Cargo.toml version.

Behavior:
- On tag refs (refs/tags/vX.Y.Z or refs/tags/X.Y.Z), require exact match
  with the version declared in rust/Cargo.toml.
- On non-tag refs, validation is skipped.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_CARGO_VERSION_RE = re.compile(r'^version\s*=\s*"([^\"]+)"', flags=re.MULTILINE)


def tag_version_from_ref(ref: str) -> str | None:
    """Extract a version string from a tag ref, handling optional `v` prefix."""
    if not ref.startswith("refs/tags/"):
        return None
    tag = ref.removeprefix("refs/tags/")
    return tag[1:] if tag.startswith("v") else tag


def cargo_version(cargo_toml_path: str = "rust/Cargo.toml") -> str:
    """Read package version from the `[package]` section of Cargo.toml."""
    text = Path(cargo_toml_path).read_text(encoding="utf-8")
    match = _CARGO_VERSION_RE.search(text)
    if match is None:
        raise ValueError(f"Could not find version in {cargo_toml_path}")
    return match.group(1)


def validate_tag_alignment(ref: str, declared_cargo_version: str) -> tuple[bool, str]:
    """Validate that a tag-triggered release ref matches Cargo package version."""
    version_from_tag = tag_version_from_ref(ref)
    if version_from_tag is None:
        return True, "Not a tag build; skipping version alignment check."

    if declared_cargo_version != version_from_tag:
        return (
            False,
            "Tag/Cargo version mismatch: "
            f"tag={ref!r} -> {version_from_tag!r}, cargo={declared_cargo_version!r}. "
            "Bump rust/Cargo.toml version before tagging.",
        )

    return (
        True,
        f"Version aligned: tag {ref!r} matches rust/Cargo.toml version {declared_cargo_version!r}",
    )


def main() -> int:
    """Run validation using workflow env vars and return process exit code."""
    ref = os.environ.get("GITHUB_REF", "")
    declared = cargo_version()
    ok, message = validate_tag_alignment(ref, declared)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
