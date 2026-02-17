"""Unit tests for tag-version alignment validator."""

# ruff: noqa: D103

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "validate_tag_version.py"
    spec = importlib.util.spec_from_file_location("validate_tag_version", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tag_version_from_ref_with_v_prefix() -> None:
    mod = _load_module()
    assert mod.tag_version_from_ref("refs/tags/v1.2.3") == "1.2.3"


def test_tag_version_from_ref_without_v_prefix() -> None:
    mod = _load_module()
    assert mod.tag_version_from_ref("refs/tags/1.2.3") == "1.2.3"


def test_tag_version_from_ref_non_tag() -> None:
    mod = _load_module()
    assert mod.tag_version_from_ref("refs/heads/main") is None


def test_validate_tag_alignment_match() -> None:
    mod = _load_module()
    ok, _ = mod.validate_tag_alignment("refs/tags/v0.3.6", "0.3.6")
    assert ok


def test_validate_tag_alignment_mismatch() -> None:
    mod = _load_module()
    ok, message = mod.validate_tag_alignment("refs/tags/v0.3.6", "0.3.5")
    assert not ok
    assert "Tag/Cargo version mismatch" in message


def test_validate_tag_alignment_skips_non_tag() -> None:
    mod = _load_module()
    ok, message = mod.validate_tag_alignment("refs/heads/main", "0.3.6")
    assert ok
    assert "skipping" in message.lower()


def test_cargo_version_reads_file(tmp_path: Path) -> None:
    mod = _load_module()
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[package]\nname = "av2"\nversion = "9.8.7"\n', encoding="utf-8")
    assert mod.cargo_version(str(cargo)) == "9.8.7"


def test_cargo_version_missing_raises(tmp_path: Path) -> None:
    mod = _load_module()
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[package]\nname = "av2"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        mod.cargo_version(str(cargo))
