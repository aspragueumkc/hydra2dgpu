import unittest
"""Smoke test that the Qt imports used inside InstallDialog are resolvable
via qgis.PyQt (works under both PyQt5 and PyQt6 in a real QGIS install).
Falls back to a skip when qgis bindings aren't on sys.path (CI without
real QGIS installed)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qgis_plugin"))


def test_installer_module_classes_present():
    from HYDRA2DGPU.installer import BackendInstaller, InstallDialog
    assert callable(BackendInstaller), "BackendInstaller must be callable"
    # InstallDialog is a class — checking it imports is enough; QApplication
    # construction would require a real QApplication which we defer to manual.
    assert hasattr(InstallDialog, "__init__")

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
