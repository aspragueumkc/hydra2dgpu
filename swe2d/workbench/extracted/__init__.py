"""Domain-organized extracted methods for SWE2D workbench decomposition.

Submodules are imported lazily to avoid circular-import recursion.
Each submodule does ``from swe2d_workbench_qt import *`` at module level
(to access Qt widgets, constants, and helper functions defined in the
monolith).  When the package ``__init__.py`` eagerly imported every
subpackage, any import of *one* submodule triggered all five, which in
turn triggered ``from swe2d_workbench_qt import *`` on the still-loading
monolith — hitting Python's recursion limit.

The fix: only import a submodule when it is actually accessed via
``swe2d.workbench.extracted.<name>``.
"""

# Lazy submodule access — no eager imports here.
__all__ = [
    "model_and_run_methods",
    "results_and_ui_methods",
    "results_export_methods",
    "studio_host_methods",
    "topology_and_io_methods",
]

_SUBMODULE_NAMES = set(__all__)

def __getattr__(name: str):
    if name in _SUBMODULE_NAMES:
        import importlib
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
