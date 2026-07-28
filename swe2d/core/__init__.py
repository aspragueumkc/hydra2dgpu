"""Core SWE2D execution modules.

This package contains the GUI-free core of SWE2D, including:
- RunContext: The canonical run specification (moved in Phase 2.1)
- Builder: RunContext construction from specs (moved in Phase 2.1)
- Executor: Pure Python execution logic
- Sink protocol: Callback interface for execution events

All public names are imported eagerly below.  None of them depend on PyQt5
or ``qgis.gui``; ``build_run_context`` may import ``qgis.core`` via the GPKG
I/O helpers, which is allowed per the CLI-first plan.
"""

from swe2d.core.builder import build_run_context, build_run_context_from_dict
from swe2d.core.executor import ComputeResult, SnapshotData, execute_run
from swe2d.core.run_context import RunContext
from swe2d.core.sink_protocol import PermutationResult, Sink

__all__ = [
    "ComputeResult",
    "PermutationResult",
    "RunContext",
    "Sink",
    "SnapshotData",
    "build_run_context",
    "build_run_context_from_dict",
    "execute_run",
]

