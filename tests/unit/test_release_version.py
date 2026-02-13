"""Unit tests for release version resolver script."""

# ruff: noqa: D103

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "resolve_release_version.py"
    spec = importlib.util.spec_from_file_location(
        "resolve_release_version", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_from_tag_ref() -> None:
    mod = _load_module()
    assert mod.resolve_version("refs/tags/v1.2.3", "v1.2.3", "v1.2.2") == "1.2.3"


def test_resolve_from_latest_tag_patch_bump() -> None:
    mod = _load_module()
    assert mod.resolve_version("refs/heads/main", "main", "v1.2.3") == "1.2.4"


def test_resolve_no_tags_defaults() -> None:
    mod = _load_module()
    assert mod.resolve_version("refs/heads/main", "main", None) == "0.1.0"


def test_normalize_version_accepts_v_prefix() -> None:
    mod = _load_module()
    assert mod.normalize_version("v01.02.003") == "1.2.3"
