"""Workbench worker exports.

This package re-exports the worker classes that drive the GUI event loop.
The worker classes (``SimulationWorker``, ``PersistenceWorker``) and their
result containers (``ComputeResult``, ``SnapshotData``) all import
``qgis.PyQt.QtCore`` at module load time, so importing them eagerly pulls
Qt into any consumer — including the headless CLI path.

To keep the Qt dependency behind the workbench boundary:

* ``RunContext`` is a plain dataclass with no Qt dependency, so it is
  re-exported eagerly.  It is safe to import from CLI / runtime paths.
* The Qt-dependent workers and result containers are re-exported via
  ``__getattr__`` (PEP 562) — they are loaded only when an attribute
  is actually accessed, and only when the import is performed from a
  fully-qualified ``swe2d.workbench.workers.<name>`` reference.

Callers that need the Qt workers should import them explicitly from
their submodule, e.g.::

    from swe2d.workbench.workers.simulation_worker import SimulationWorker

That keeps the import boundary intact: the workers' ``__init__`` module
does NOT load Qt at package-import time.
"""

from __future__ import annotations

from typing import Any

from swe2d.core.run_context import RunContext

# Submodules that lazy-load Qt.  Listed here so static analysers and
# IDEs can resolve the names without triggering the import.
_LAZY_ATTRS = {
    "SimulationWorker": "swe2d.workbench.workers.simulation_worker",
    "PersistenceWorker": "swe2d.workbench.workers.persistence_worker",
}

# Result containers are Qt-free and live in the core executor; expose them
# lazily from the canonical location so the package import never pulls Qt.
_CORE_ATTRS = {
    "ComputeResult": "swe2d.core.executor",
    "SnapshotData": "swe2d.core.executor",
}

__all__ = [
    "RunContext",
    "SimulationWorker",
    "SnapshotData",
    "ComputeResult",
    "PersistenceWorker",
]


def __getattr__(name: str) -> Any:
    """Resolve Qt-dependent worker classes and core result containers on first access (PEP 562)."""
    target = _LAZY_ATTRS.get(name) or _CORE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'swe2d.workbench.workers' has no attribute {name!r}")
    module = __import__(target, fromlist=[name])
    value = getattr(module, name)
    # Cache so subsequent lookups don't pay the import cost again.
    globals()[name] = value
    return value
