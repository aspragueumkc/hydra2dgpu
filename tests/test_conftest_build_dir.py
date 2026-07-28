import unittest
"""Verify tests/conftest.py honors the HYDRA_BUILD_DIR env var.

When the env var is set, conftest prepends its value to sys.path so a
worktree can point at a sibling repo's build instead of rebuilding.

Skipped if no main-repo build is present (the path discovery needs it).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load_conftest():
    """Re-exec tests/conftest.py as a fresh module to test its top-level logic."""
    spec = importlib.util.spec_from_file_location(
        "conftest_under_test", "tests/conftest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_conftest_exposes_repo_root():
    """conftest.py exposes _REPO_ROOT after module load."""
    mod = _load_conftest()
    assert hasattr(mod, "_REPO_ROOT")
    assert os.path.isdir(mod._REPO_ROOT)


def test_hydra_build_dir_env_var_prepended(monkeypatch, tmp_path):
    """When HYDRA_BUILD_DIR is set, that dir is prepended to sys.path."""
    fake_build = str(tmp_path / "fake_build")
    os.makedirs(fake_build, exist_ok=True)
    monkeypatch.setenv("HYDRA_BUILD_DIR", fake_build)

    # Drop any cached conftest imports so it re-runs the top-level block.
    sys.modules.pop("conftest_under_test", None)
    # Ensure fake_build is NOT already in sys.path before conftest loads.
    sys.path = [p for p in sys.path if p != fake_build]

    mod = _load_conftest()
    # The env var's dir should now be on sys.path.
    assert fake_build in sys.path
    # And it should appear before the default build dir (prepended).
    assert sys.path.index(fake_build) < sys.path.index(mod._DEFAULT_BUILD_DIR) \
        or fake_build == mod._DEFAULT_BUILD_DIR


def test_default_build_dir_used_when_env_var_unset(monkeypatch):
    """When HYDRA_BUILD_DIR is unset, the default <repo>/build is used."""
    monkeypatch.delenv("HYDRA_BUILD_DIR", raising=False)
    sys.modules.pop("conftest_under_test", None)

    mod = _load_conftest()
    assert mod._HYDRA_BUILD_DIR == ""
    assert mod._DEFAULT_BUILD_DIR == os.path.join(mod._REPO_ROOT, "build")

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
