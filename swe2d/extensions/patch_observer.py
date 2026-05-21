#!/usr/bin/env python3
"""3D patch runtime observation seam for SWE2D workbench.

Phase 12 goal: extract in-method 3D observation helpers from `_on_run`
into a reusable module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class SWE2DThreeDPatchObserver:
    """Provides safe accessors for optional 3D patch runtime observations."""

    def __init__(self, backend: Any, runtime_enabled: bool):
        self._backend = backend
        self._runtime_enabled = bool(runtime_enabled)

    def get_patch_stats(self) -> Optional[Dict[str, object]]:
        if not self._runtime_enabled:
            return None
        if self._backend is None:
            return None
        try:
            if not self._backend.supports_3d_patch_observation():
                return None
            return dict(self._backend.get_3d_patch_stats())
        except Exception:
            return None

    def get_patch_vof(self) -> Optional[np.ndarray]:
        if not self._runtime_enabled:
            return None
        if self._backend is None:
            return None
        try:
            if not self._backend.supports_3d_patch_observation():
                return None
            return np.asarray(self._backend.get_3d_patch_vof(), dtype=np.float64).ravel()
        except Exception:
            return None

    def get_patch_velocity(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self._runtime_enabled:
            return None
        if self._backend is None:
            return None
        try:
            if not self._backend.supports_3d_patch_observation():
                return None
            u, v, w = self._backend.get_3d_patch_velocity()
            return (
                np.asarray(u, dtype=np.float64).ravel(),
                np.asarray(v, dtype=np.float64).ravel(),
                np.asarray(w, dtype=np.float64).ravel(),
            )
        except Exception:
            return None
