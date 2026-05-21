"""Compatibility shim for SWE2D workbench Qt module.

The implementation was moved to swe2d_workbench_qt_impl.py to keep this public
entrypoint lean while preserving existing import paths.
"""

try:
    from .swe2d_workbench_qt_impl import *  # type: ignore F401,F403
except Exception:
    from swe2d_workbench_qt_impl import *  # type: ignore F401,F403
