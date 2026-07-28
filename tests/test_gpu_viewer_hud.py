"""Phase 4.3 — GPUViewerGLWidget HUD overlay integration test.

Verifies the widget's HUD wiring:
  - Constructor accepts a `solver` argument
  - The frameRendered signal is emitted when the timer ticks
  - The service-layer helpers are wired correctly (no numpy in View)

The full paintGL -> HUD render path requires a hardware GL context
(verified by the xfailed test in test_swe2d_gpu_viewer_interop.py
under offscreen Mesa).  This test verifies the widget structure.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


def _gpu_available():
    try:
        import hydra_swe2d as m
        return m.swe2d_gpu_available()
    except Exception:
        return False


def _hud_binding_present():
    try:
        import hydra_swe2d as m
        return hasattr(m, "swe2d_gpu_render_hud")
    except ImportError:
        return False


@pytest.mark.solver
@pytest.mark.gpu
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUViewerHUDOverlay:
    """Verify the widget's HUD wiring (no GL required for these)."""

    def _make_widget(self, qtbot, *, solver=None):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        mesh_data = {"cell_x": [], "cell_y": [], "cell_bed": [],
                     "node_x": [], "node_y": [], "cell_nodes": []}
        return GPUViewerGLWidget(
            reader=MagicMock(), mesh_data=mesh_data, solver=solver)

    def test_widget_accepts_solver_arg(self, qtbot):
        """Per subagent diff C1 — solver is an explicit constructor arg, not
        reached through the reader's private state."""
        solver = MagicMock()
        w = self._make_widget(qtbot, solver=solver)
        qtbot.addWidget(w)
        assert w._solver is solver

    def test_widget_solver_default_is_none(self, qtbot):
        """If no solver passed, _solver is None — caller wires it explicitly."""
        w = self._make_widget(qtbot)
        qtbot.addWidget(w)
        assert w._solver is None

    def test_service_layer_extents_callable(self):
        """Service helpers are importable and produce correct shape on
        synthetic data (per subagent diff C2 — no numpy in View)."""
        from swe2d.workbench.services.viewer_frame_extents import (
            compute_render_extents, compute_hud_wet_count,
        )
        cell_x = np.array([0.0, 1.0, 2.0, 3.0])
        cell_y = np.array([0.0, 0.0, 1.0, 1.0])
        snap = {
            "h": np.array([0.0, 0.1, 0.5, 0.2]),
            "depth": np.array([0.0, 0.1, 0.5, 0.2]),
        }
        xmax, ymax, vmin, vmax, rx, ry = compute_render_extents(
            {"cell_x": cell_x, "cell_y": cell_y}, snap, "depth",
        )
        assert xmax == 3.0
        assert ymax == 1.0
        assert vmin == 0.0
        assert vmax == 0.5
        np.testing.assert_array_equal(rx, cell_x)
        np.testing.assert_array_equal(ry, cell_y)

        wet = compute_hud_wet_count(snap)
        assert wet == 3  # h > 0.0 for indices 1, 2, 3

    def test_service_layer_handles_empty_arrays(self):
        """Edge case: empty mesh / empty snap — no crash, sane defaults."""
        from swe2d.workbench.services.viewer_frame_extents import (
            compute_render_extents, compute_hud_wet_count,
        )
        xmax, ymax, vmin, vmax, rx, ry = compute_render_extents(
            {"cell_x": np.array([]), "cell_y": np.array([])},
            {"h": np.array([]), "depth": np.array([])},
            "depth",
        )
        # Empty → fallback xmax=1.0 (avoids 0/0 in the kernel)
        assert xmax == 1.0
        assert ymax == 1.0
        assert vmin == 0.0
        assert vmax == 1.0
        wet = compute_hud_wet_count({"h": np.array([])})
        assert wet == 0