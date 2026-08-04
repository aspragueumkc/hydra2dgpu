#!/usr/bin/env python3
"""Integration tests for overlay and auto-load behavior (real headless QGIS).

Validates the interaction between:
1. Snapshot orchestration (``RunController.on_snapshot`` →
   ``SWE2DWorkbenchStudioDialog._sync_snapshot_to_ui``) and overlay sync
2. Results-panel animation → timestep signal chain
3. ``OverlayController`` data sync and overlay-time update paths
4. High-perf overlay rendering with synthetic mesh data (real renderer)

These tests run against real QGIS (offscreen) via ``tests.qgis_real_env``.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np

# Ensure repo root is on sys.path so imports work in headless mode
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tests.qgis_real_env import ensure_qgis_app, requires_qgis

_RUN_CONTROLLER_PATH = os.path.join(
    _REPO_ROOT, "swe2d", "workbench", "controllers", "run_controller.py"
)
_STUDIO_DIALOG_PATH = os.path.join(
    _REPO_ROOT, "swe2d", "workbench", "studio_dialog.py"
)
_HIGH_PERF_VIEWER_PATH = os.path.join(
    _REPO_ROOT, "swe2d", "results", "high_perf_viewer.py"
)


def _get_function_source_ast(mod_path: str, func_name: str) -> ast.FunctionDef:
    """Parse *mod_path* and return the AST node for *func_name*."""
    with open(mod_path, "r") as f:
        tree = ast.parse(f.read(), filename=mod_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise NameError(f"Function '{func_name}' not found in {mod_path}")


def _ast_has_call(func_node: ast.FunctionDef, target_name: str) -> bool:
    """Return True if *func_node* contains a call to *target_name* anywhere."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            fn = node.func
            # Direct call: foo()
            if isinstance(fn, ast.Name) and fn.id == target_name:
                return True
            # Method call: self.foo() / view.foo()
            if isinstance(fn, ast.Attribute) and fn.attr == target_name:
                return True
    return False


def _build_overlay_controller() -> tuple:
    """Build an ``OverlayController`` over a mock view with real results data.

    Returns ``(controller, view)``.  The controller's
    ``refresh_high_perf_canvas_overlay`` is replaced with a MagicMock so
    tests can assert on refresh calls without a map canvas.
    """
    from swe2d.results.data import SWE2DResultsData
    from swe2d.workbench.controllers.overlay_controller import OverlayController

    view = MagicMock(name="overlay_view")
    view._results_data = SWE2DResultsData()
    view._log = MagicMock(name="log")
    ctrl = OverlayController(view)
    ctrl.refresh_high_perf_canvas_overlay = MagicMock(name="refresh_overlay")
    return ctrl, view


def _make_synthetic_timesteps(
    n_cells: int = 100,
    n_ts: int = 5,
    depth_range: Tuple[float, float] = (0.5, 2.0),
) -> List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Build a list of synthetic timesteps (t_s, h, hu, hv) for testing.

    Each timestep is a tuple of (time_in_seconds, depth_array,
    x-momentum_array, y-momentum_array).
    """
    timesteps = []
    for i in range(n_ts):
        t = float(i) * 60.0  # one minute per frame
        h = np.random.uniform(depth_range[0], depth_range[1], n_cells).astype(np.float64)
        hu = np.random.uniform(-0.5, 0.5, n_cells).astype(np.float64)
        hv = np.random.uniform(-0.5, 0.5, n_cells).astype(np.float64)
        timesteps.append((t, h, hu, hv))
    return timesteps


# =========================================================================
# Test classes
# =========================================================================

class TestAutoloadSourceAnalysis(unittest.TestCase):
    """AST-based verification that snapshot → UI sync hooks exist.

    These tests inspect the source code statically rather than executing
    the dialog, making them fast and suitable for regression detection.
    """

    def test_snapshot_syncs_overlay_to_ui(self):
        """REGRESSION TEST: snapshot triggers overlay sync via the controller.

        ``RunController.on_snapshot`` must delegate to the view's
        ``_sync_snapshot_to_ui``, which must sync the high-perf overlay.
        """
        func_node = _get_function_source_ast(_RUN_CONTROLLER_PATH, "on_snapshot")
        self.assertTrue(
            _ast_has_call(func_node, "_sync_snapshot_to_ui"),
            "RunController.on_snapshot is missing the _sync_snapshot_to_ui "
            "call — snapshot data is fetched from device but never synced "
            "to the UI.",
        )
        helper_node = _get_function_source_ast(_STUDIO_DIALOG_PATH, "_sync_snapshot_to_ui")
        self.assertTrue(
            _ast_has_call(helper_node, "_sync_high_perf_overlay_data"),
            "_sync_snapshot_to_ui is missing the _sync_high_perf_overlay_data "
            "call — snapshot data is persisted but never reaches the overlay.",
        )

    def test_snapshot_updates_overlay_time(self):
        """``_sync_snapshot_to_ui`` advances the overlay to the latest snapshot time."""
        helper_node = _get_function_source_ast(_STUDIO_DIALOG_PATH, "_sync_snapshot_to_ui")
        self.assertTrue(
            _ast_has_call(helper_node, "_update_high_perf_overlay_time"),
            "_sync_snapshot_to_ui is missing the _update_high_perf_overlay_time "
            "call — the overlay will not advance to the latest snapshot time.",
        )


@requires_qgis
class TestResultsSignalChain(unittest.TestCase):
    """Verify that the results panel → workbench overlay signal chain exists.

    The chain is:
        panel slider / animation
          → ResultsAnimationController.current_timestep_changed
            → dialog._on_results_panel_timestep_changed(t_s, frame_idx)
              → studio_results_panel.on_results_panel_timestep_changed
                → dialog._update_high_perf_overlay_time(t_s)
    """

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_results_data_has_data_source(self):
        """SWE2DResultsData exposes set_data_source."""
        from swe2d.results.data import SWE2DResultsData
        self.assertTrue(
            hasattr(SWE2DResultsData, "set_data_source"),
            "SWE2DResultsData is missing set_data_source method",
        )

    def test_studio_dialog_has_timestep_handler(self):
        """Studio dialog has the _on_results_panel_timestep_changed handler."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        self.assertTrue(
            hasattr(SWE2DWorkbenchStudioDialog, "_on_results_panel_timestep_changed"),
            "SWE2DWorkbenchStudioDialog missing _on_results_panel_timestep_changed",
        )

    def test_results_panel_wires_timestep_signal(self):
        """show_results_panel connects current_timestep_changed → handler."""
        from swe2d.workbench.views.studio_results_panel import show_results_panel
        src = inspect.getsource(show_results_panel)
        self.assertIn(
            "current_timestep_changed",
            src,
            "studio_results_panel.show_results_panel does not reference "
            "the current_timestep_changed signal",
        )
        self.assertIn(
            "_on_results_panel_timestep_changed",
            src,
            "studio_results_panel.show_results_panel does not reference "
            "the _on_results_panel_timestep_changed handler",
        )


@requires_qgis
class TestOverlayController(unittest.TestCase):
    """Test the OverlayController sync / update paths with a mock view."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self.ctrl, self.view = _build_overlay_controller()

    def test_update_overlay_time_calls_refresh(self):
        """``update_high_perf_overlay_time`` refreshes the canvas overlay."""
        self.ctrl.update_high_perf_overlay_time(5.0)
        self.ctrl.refresh_high_perf_canvas_overlay.assert_called_once_with(5.0)

    def test_update_overlay_time_validates_float(self):
        """Controller converts to float and still works."""
        self.ctrl.update_high_perf_overlay_time("5.0")
        self.ctrl.refresh_high_perf_canvas_overlay.assert_called_once()
        call_arg = self.ctrl.refresh_high_perf_canvas_overlay.call_args[0][0]
        self.assertAlmostEqual(float(call_arg), 5.0)

    def test_sync_overlay_data_clears_on_no_run_data(self):
        """``sync_high_perf_overlay_data`` empties arrays when no run data."""
        # data_source defaults to "none" and no run records exist → GPKG
        # path with no enabled run → arrays cleared, refresh(None) called.
        self.ctrl.sync_high_perf_overlay_data()

        data = self.view._results_data
        self.assertEqual(data.overlay_cell_x.size, 0)
        self.assertEqual(data.overlay_cell_y.size, 0)
        self.ctrl.refresh_high_perf_canvas_overlay.assert_called_once_with(None)

    def test_overlay_data_sync_populates_arrays_live(self):
        """Live-path sync fills cell/bed arrays from mesh_data."""
        data = self.view._results_data
        data.set_data_source("live")
        self.view._mesh_data = {
            "node_x": np.array([0, 1, 0, 1], dtype=np.float64),
            "node_y": np.array([0, 0, 1, 1], dtype=np.float64),
            "cell_nodes": np.array([0, 1, 3, 0, 3, 2], dtype=np.int32),
        }
        self.view._mesh_cell_centroids = lambda: (
            np.array([0.25, 0.75], dtype=np.float64),
            np.array([0.25, 0.75], dtype=np.float64),
        )
        self.view._mesh_cell_solver_bed = lambda: np.array(
            [10.0, 12.0], dtype=np.float64
        )

        self.ctrl.sync_high_perf_overlay_data()

        self.assertEqual(data.overlay_cell_x.size, 2)
        self.assertEqual(data.overlay_cell_y.size, 2)
        self.assertEqual(data.overlay_cell_bed.size, 2)
        self.assertAlmostEqual(float(data.overlay_cell_bed[0]), 10.0)
        self.assertAlmostEqual(float(data.overlay_cell_bed[1]), 12.0)
        self.ctrl.refresh_high_perf_canvas_overlay.assert_called_once_with(None)

    def test_mesh_fingerprint_stable(self):
        """Same mesh data produces same fingerprint."""
        from swe2d.workbench.controllers.overlay_controller import (
            mesh_fingerprint_from_arrays,
        )

        n1 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        n2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        cn = np.array([0, 1, 3, 0, 3, 2], dtype=np.int32)

        fp1 = mesh_fingerprint_from_arrays(n1, n2, cn)
        fp2 = mesh_fingerprint_from_arrays(n1, n2, cn)
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, "")  # Non-empty


@requires_qgis
class TestOverlayRendering(unittest.TestCase):
    """Test the high-perf overlay rendering pipeline with synthetic data.

    Uses the real ``render_unstructured_snapshot_image`` function against
    real (offscreen) QImage/QPainter.
    """

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        # Create a synthetic 2×2 quad mesh (4 cells → 8 triangles)
        self.cell_x = np.array([0.25, 0.75, 0.25, 0.75], dtype=np.float64)
        self.cell_y = np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float64)
        self.cell_bed = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        self.node_x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=np.float64)
        self.node_y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        # 8 triangles (fan from node 0 and node 3 for each quad)
        self.cell_nodes = np.array([
            [0, 4, 3], [0, 1, 4],
            [1, 5, 4], [3, 4, 7],
            [3, 7, 6], [4, 5, 8],
            [4, 8, 7], [0, 3, 6],
        ], dtype=np.int32).ravel()
        self.tri_to_cell = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

    def _make_timesteps(self, n_ts=3):
        """Create synthetic timesteps for the 4-cell mesh."""
        timesteps = []
        for i in range(n_ts):
            t = float(i) * 60.0
            h = np.array([0.5 + i * 0.1, 0.6 + i * 0.1,
                          0.4 + i * 0.1, 0.7 + i * 0.1], dtype=np.float64)
            hu = np.array([0.1 * i, 0.0, 0.0, 0.1 * i], dtype=np.float64)
            hv = np.array([0.0, 0.1 * i, 0.0, 0.0], dtype=np.float64)
            timesteps.append((t, h, hu, hv))
        return timesteps

    def test_render_returns_expected_keys(self):
        """Render result contains all expected keys."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="depth",
            resolution=(320, 240),
        )
        expected_keys = {
            "ok", "image", "extent", "frame_idx", "frame_count",
            "time_s", "n_cells", "vmin", "vmax", "render_ms", "backend",
            "message", "grid", "grid_mask", "computed_vmin", "computed_vmax",
            "field_key", "length_unit_name",
        }
        self.assertEqual(
            set(result.keys()),
            expected_keys,
            f"Result keys mismatch. Extra: {set(result.keys()) - expected_keys}. "
            f"Missing: {expected_keys - set(result.keys())}",
        )

    def test_render_depth_ok(self):
        """Depth rendering returns ok=True with valid image."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="depth",
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"Render failed: {result['message']}")
        self.assertFalse(result["image"].isNull(), "Rendered image is null")
        self.assertGreater(result["n_cells"], 0)
        self.assertGreater(result["vmax"], result["vmin"])
        self.assertGreater(result["render_ms"], 0.0)

    def test_render_wse(self):
        """WSE rendering works (depth + bed elevation)."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="wse",
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"WSE render failed: {result['message']}")

    def test_render_speed(self):
        """Speed rendering works."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="speed",
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"Speed render failed: {result['message']}")

    def test_render_froude(self):
        """Froude number rendering works."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="froude",
            gravity=9.81,
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"Froude render failed: {result['message']}")

    def test_render_courant(self):
        """Courant number rendering works."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="courant",
            courant_cell_size=0.5,
            courant_dt=10.0,
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"Courant render failed: {result['message']}")

    def test_render_shear_stress(self):
        """Shear stress rendering works."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="shear_stress",
            mannings_n=0.035,
            gravity=9.81,
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"Shear stress render failed: {result['message']}")

    def test_render_interpolates_timestep(self):
        """Render picks the nearest timestep to current_time_s."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        timesteps = self._make_timesteps(n_ts=5)  # t=0, 60, 120, 180, 240

        # Request t=119 → should land on t=120 (index 2)
        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=timesteps,
            current_time_s=119.0,
            field_key="depth",
            resolution=(320, 240),
        )
        self.assertTrue(result["ok"], f"Render failed: {result['message']}")
        self.assertEqual(result["frame_idx"], 2)
        self.assertAlmostEqual(result["time_s"], 120.0, places=5)

    def test_render_no_cells_returns_not_ok(self):
        """Render with empty arrays returns ok=False."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=np.empty(0, dtype=np.float64),
            cell_y=np.empty(0, dtype=np.float64),
            cell_bed=None,
            timesteps=[],
            current_time_s=0.0,
            resolution=(320, 240),
        )
        self.assertFalse(result["ok"])
        self.assertIn("message", result)

    def test_render_large_mesh_does_not_crash(self):
        """Render a moderately large synthetic mesh without crashing."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        n = 500
        np.random.seed(42)
        cell_x = np.random.uniform(0, 100, n).astype(np.float64)
        cell_y = np.random.uniform(0, 100, n).astype(np.float64)
        cell_bed = np.random.uniform(0, 5, n).astype(np.float64)
        timesteps = _make_synthetic_timesteps(
            n_cells=n, n_ts=3, depth_range=(0.5, 2.0),
        )

        # Build a simple per-cell triangle fan so the tri-fill rasterizer has
        # valid mesh data. Each cell becomes one triangle with distinct nodes.
        eps = 1.0
        node_x = np.empty(3 * n, dtype=np.float64)
        node_y = np.empty(3 * n, dtype=np.float64)
        node_x[0::3] = cell_x
        node_y[0::3] = cell_y
        node_x[1::3] = cell_x + eps
        node_y[1::3] = cell_y
        node_x[2::3] = cell_x
        node_y[2::3] = cell_y + eps
        cell_nodes = np.arange(3 * n, dtype=np.int32)
        tri_to_cell = np.arange(n, dtype=np.int32)

        result = render_unstructured_snapshot_image(
            cell_x=cell_x,
            cell_y=cell_y,
            cell_bed=cell_bed,
            node_x=node_x,
            node_y=node_y,
            cell_nodes=cell_nodes,
            tri_to_cell=tri_to_cell,
            timesteps=timesteps,
            current_time_s=60.0,
            field_key="depth",
            resolution=(640, 480),
        )
        self.assertTrue(result["ok"], f"Large mesh render failed: {result['message']}")
        self.assertEqual(result["n_cells"], n)

    def test_render_with_visible_extent(self):
        """Render with visible_extent_world clips to subset."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        # Mesh spans [0, 1] × [0, 1]; render only the bottom-left quadrant
        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(),
            current_time_s=60.0,
            field_key="depth",
            resolution=(320, 240),
            visible_extent_world=(0.0, 0.5, 0.0, 0.5),
        )
        self.assertTrue(result["ok"], f"Clipped render failed: {result['message']}")

    def test_render_show_arrows(self):
        """Render with velocity arrows does not crash."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=self._make_timesteps(n_ts=1),
            current_time_s=0.0,
            field_key="depth",
            resolution=(320, 240),
            show_velocity_arrows=True,
            arrow_stride_px=8,
            arrow_scale_px=4.0,
        )
        self.assertTrue(result["ok"], f"Arrows render failed: {result['message']}")

    def test_render_all_dry_returns_message(self):
        """Render with all-zero depths returns ok=False with message."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        # All depths = 0.0 (dry)
        timesteps = [
            (0.0,
             np.zeros(4, dtype=np.float64),
             np.zeros(4, dtype=np.float64),
             np.zeros(4, dtype=np.float64)),
        ]
        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            node_x=self.node_x,
            node_y=self.node_y,
            cell_nodes=self.cell_nodes,
            tri_to_cell=self.tri_to_cell,
            timesteps=timesteps,
            current_time_s=0.0,
            field_key="depth",
            resolution=(320, 240),
        )
        self.assertFalse(result["ok"])
        self.assertIn("No wetted values", result.get("message", ""))


@requires_qgis
class TestResultsPanelControls(unittest.TestCase):
    """Verify the results panel's animation controls work correctly.

    Tests the slider → animation controller → signal chain.
    """

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_slider_changes_frame(self):
        """Slider value change triggers animation index change."""
        from swe2d.results.animation import ResultsAnimationController

        # We test the animation controller directly since it's the
        # intermediary between the slider and the timestep signal.
        controller = ResultsAnimationController(None, fps=10.0)

        # Set up timesteps
        times = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        controller.set_timesteps(times)

        # Step forward
        controller.step_forward()
        self.assertEqual(controller.current_index, 1)

        # Step backward
        controller.step_backward()
        self.assertEqual(controller.current_index, 0)

        # Set index directly
        controller.set_index(3)
        self.assertEqual(controller.current_index, 3)

    def test_animation_controller_clamps_index(self):
        """Animation controller clamps index to valid range."""
        from swe2d.results.animation import ResultsAnimationController

        controller = ResultsAnimationController(None, fps=10.0)
        controller.set_timesteps(np.array([0.0, 10.0, 20.0], dtype=np.float64))

        controller.set_index(100)  # out of range
        self.assertEqual(controller.current_index, 2)

        controller.set_index(-100)  # out of range
        self.assertEqual(controller.current_index, 0)

    def test_animation_controller_signal_emitted(self):
        """Setting index emits current_timestep_changed."""
        from swe2d.results.animation import ResultsAnimationController

        signals = []
        controller = ResultsAnimationController(None, fps=10.0)
        controller.set_timesteps(np.array([0.0, 10.0, 20.0], dtype=np.float64))
        controller.current_timestep_changed.connect(
            lambda t, idx: signals.append((t, idx))
        )

        controller.set_index(1)
        self.assertEqual(len(signals), 1)
        self.assertAlmostEqual(signals[0][0], 10.0, places=9)
        self.assertEqual(signals[0][1], 1)

        controller.set_index(2)
        self.assertEqual(len(signals), 2)
        self.assertAlmostEqual(signals[0][0], 10.0, places=9)
        self.assertAlmostEqual(signals[1][0], 20.0, places=9)


class TestNoSilentFallbacks(unittest.TestCase):
    """Verify that overlay code paths don't have silent fallbacks.

    Per AGENTS.md: "NO SILENT FALLBACKS! A silent fallback/degradation is
    the biggest failure you can make in this repo."
    """

    def test_hydra_overlay_import_is_hard(self):
        """hydra_overlay must be a top-level hard import in high_perf_viewer.

        A try/except around the import would silently degrade rendering to
        a slower/incomplete path.  Verify the import is unconditional.
        """
        self.assertTrue(
            os.path.exists(_HIGH_PERF_VIEWER_PATH),
            f"swe2d/results/high_perf_viewer.py not found at {_HIGH_PERF_VIEWER_PATH}",
        )
        with open(_HIGH_PERF_VIEWER_PATH, "r") as f:
            tree = ast.parse(f.read(), filename=_HIGH_PERF_VIEWER_PATH)

        def _imports_hydra_overlay(node: ast.AST) -> bool:
            if isinstance(node, ast.Import):
                return any(a.name == "hydra_overlay" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                return node.module == "hydra_overlay"
            return False

        top_level_hard = any(
            _imports_hydra_overlay(node) for node in tree.body
        )
        self.assertTrue(
            top_level_hard,
            "high_perf_viewer.py has no top-level 'import hydra_overlay' — "
            "the renderer must fail loudly when the extension is missing.",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for sub in ast.walk(node):
                    if _imports_hydra_overlay(sub):
                        self.fail(
                            "hydra_overlay import is wrapped in try/except at "
                            f"line {sub.lineno} — this is a silent fallback!"
                        )


# =========================================================================
# Run with: python -m unittest -v tests.test_swe2d_overlay_and_autoload
# =========================================================================

if __name__ == "__main__":
    unittest.main()
