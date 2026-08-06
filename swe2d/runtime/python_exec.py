"""Locate a real Python interpreter for subprocess workers.

Inside OSGeo4W QGIS on Windows, ``sys.executable`` and
``sys._base_executable`` point at the QGIS launcher (``qgis-ltr-bin.exe``),
NOT a Python interpreter. Spawning ``sys.executable`` therefore launches a
new QGIS GUI instead of running a Python script. This module resolves the
actual ``python.exe`` so subprocess workers (gmsh mesh generation, etc.) run
in a plain interpreter.

Qt-free; safe for the runtime/service layer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def real_python() -> Path:
    """Resolve a real ``python.exe`` to base subprocess workers on.

    Candidate order (Windows-oriented, harmless elsewhere):
      1. ``sys._base_executable`` when it is actually a python binary
      2. ``python.exe`` beside ``base_prefix`` / ``base_exec_prefix``
         (OSGeo4W: ``C:\\OSGeo4W\\apps\\Python312\\python.exe``)
      3. ``python.exe`` beside the launcher
      4. Fall back to ``sys.executable`` (correct on normal installs)
    """
    candidates: list[str] = []
    bex = getattr(sys, "_base_executable", "") or ""
    if bex and os.path.basename(bex).lower().startswith("python"):
        candidates.append(bex)
    for prefix in (
        getattr(sys, "base_prefix", "") or "",
        getattr(sys, "base_exec_prefix", "") or "",
    ):
        if prefix:
            candidates.append(os.path.join(prefix, "python.exe"))
    if sys.executable:
        candidates.append(os.path.join(os.path.dirname(sys.executable), "python.exe"))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return Path(cand)
    return Path(sys.executable)
