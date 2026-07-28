import unittest
"""Tests for the pure-Python BackendInstaller service."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qgis_plugin"))

from HYDRA2DGPU.installer import BackendInstaller


def test_env_dir_respects_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRA2DGPU_CACHE_DIR", str(tmp_path))
    inst = BackendInstaller(plugin_dir=".")
    assert inst.env_dir() == tmp_path / ".hydra2dgpu"


def test_wheel_name_format_includes_python_and_platform(monkeypatch):
    inst = BackendInstaller(plugin_dir=".")
    name = inst.wheel_name()
    assert name.startswith("hydra_swe2d-")
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert py_tag in name, f"expected {py_tag} in {name}"
    assert name.endswith((".whl",)), name
    assert any(p in name for p in ("manylinux", "win_amd64"))

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
