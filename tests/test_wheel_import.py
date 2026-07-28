import unittest
"""Smoke test: the installed hydra-swe2d wheel is importable and exposes the
symbol surface that swe2d.runtime expects. Locally this loads the symlink
trick's compiled .so via PYTHONPATH; in CI (Phase 2.2 cibuildwheel matrix)
it loads the real installed wheel.

DEVIATION FROM PLAN (Task 2.4 Step 2): the plan assumed symbol names
``initialize``, ``step``, ``finalize``. The actual pybind11 bindings
(cpp/src/swe2d_bindings.cpp) expose:

    line 2880: m.def("swe2d_create_solver", ...)   # was "initialize"
    line 3085: m.def("swe2d_step",        ...)     # unchanged
    line 3268: m.def("swe2d_destroy",     ...)     # was "finalize"

``swe2d_step`` matches the plan; the other two were wrong. _REQUIRED_NAMES
below uses the real names. If the bindings ever gain a wrapper that
re-exports them as the shorter aliases, this test should switch back.
"""
from __future__ import annotations
import importlib

# Symbol surface ground-truthed from cpp/src/swe2d_bindings.cpp.
# Update here ONLY if the bindings change; do not silently change.
_REQUIRED_NAMES = ("swe2d_create_solver", "swe2d_step", "swe2d_destroy")


def test_hydra_swe2d_module_imports():
    import hydra_swe2d  # installed via wheel / symlink / PYTHONPATH
    for name in _REQUIRED_NAMES:
        assert hasattr(hydra_swe2d, name), (
            f"hydra_swe2d.{name} missing - bindings surface drifted"
        )


def test_hydra_swe2d_version_metadata():
    pkg = importlib.import_module("hydra_swe2d")
    assert hasattr(pkg, "__version__"), "hydra_swe2d must expose __version__"
    assert pkg.__version__ == "1.2.0", f"version mismatch: {pkg.__version__}"

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
