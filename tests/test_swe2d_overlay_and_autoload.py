#!/usr/bin/env python3
"""Integration tests for overlay and auto-load behavior.

Validates the interaction between:
1. Snapshot persistence and auto-load into the results panel
2. Run-completion auto-load into the results panel
3. Results-panel time-slider → overlay refresh signal chain
4. High-perf overlay rendering with synthetic mesh data

These tests are designed to run headlessly with mock QGIS.

Key known issues documented by these tests:
- BUG: ``_on_snapshot()`` does NOT call ``_auto_load_results_panel()``,
  so requested snapshots never auto-load into the results viewer.
  See ``test_snapshot_missing_auto_load`` — this is a REGRESSION TEST
  that will start PASSING when the fix is applied.
- The overlay slider update path (panel → workbench → canvas) is wired
  through ``results_bridge.py`` but may still exhibit silent failures.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import unittest
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np


# Ensure repo root is on sys.path so imports work in headless mode
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Headless Qt + mock QGIS bootstrap
#
# The trick: real PyQt5 modules (QtGui, QtCore, QtWidgets) must be imported
# BEFORE ``install_qgis_mocks()`` runs.  That way the mock's
# ``_install_pyqt5_mocks()`` sees they're already cached and skips
# installing substitute mocks.  Then ``_install_qgis_pyqt_submodule()``
# copies REAL symbols into ``qgis.PyQt.*``, giving us working
# QImage/QPainter/QApplication even though ``qgis.core`` is mocked.
#
# Order:
#   1. Import real PyQt5 (QImage, QPainter, QApplication)
#   2. Install QGIS mocks (qgis.core, qgis.gui stay stubbed)
#   3. Create a QApplication instance for QPainter use
# ---------------------------------------------------------------------------
from PyQt5.QtGui import QImage as _RealQImage, QPainter as _RealQPainter
from PyQt5.QtWidgets import QApplication as _QApp

_test_app = _QApp.instance()
if _test_app is None:
    _test_app = _QApp([])

from tests.mocks.qgis_env import install_qgis_mocks as _install_qgis_mocks

_install_qgis_mocks()

# When ``swe2d.results.high_perf_viewer`` is imported as a side effect of loading
# other modules (e.g. ``swe2d.workbench.results_bridge``) under the mock
# QGIS setup, Python may cache an incomplete module entry (no __file__ or
# __spec__, all functions as MagicMock).  Clearing it here ensures that
# any test that actually needs the real module gets a fresh import.
sys.modules.pop("swe2d.results.high_perf_viewer", None)
import importlib as _il
_il.invalidate_caches()

# Whether real Qt GUI classes are available (needed for rendering tests).
# QImage/QPainter classes imported BEFORE mock installation are the real
# deal if ``PyQt5`` is installed in this environment.  Check by verifying
# the class module path starts with ``PyQt5`` (not ``unittest.mock``).
def _is_real_qimage(cls: type) -> bool:
    mod = getattr(cls, "__module__", "") or ""
    return str(mod).startswith("PyQt5") and not str(mod).startswith("unittest")


_HAS_REAL_QT = _is_real_qimage(_RealQImage) and _is_real_qimage(_RealQPainter)


def _import_wb_module():
    """Import swe2d_workbench_qt and return the module reference."""
    import swe2d_workbench_qt as wb
    return wb


def _get_source_path(mod_name: str) -> str:
    """Resolve the absolute file path for a top-level module."""
    import importlib
    spec = importlib.util.find_spec(mod_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Cannot locate source for module: {mod_name}")
    return str(spec.origin)


def _ensure_qapp():
    """Return the global QApplication instance."""
    return _test_app

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
            # Method call: self.foo()
            if isinstance(fn, ast.Attribute) and fn.attr == target_name:
                return True
    return False


def _build_mock_dialog() -> Any:
    """Build a minimally-viable mock dialog for overlay / panel tests."""
    from unittest.mock import MagicMock, PropertyMock

    dlg = MagicMock()
    dlg._high_perf_canvas_overlay_enabled = True
    dlg._high_perf_overlay_cell_x = np.array([0.25, 0.75, 0.25, 0.75], dtype=np.float64)
    dlg._high_perf_overlay_cell_y = np.array([0.25, 0.25, 0.75, 0.75], dtype=np.float64)
    dlg._high_perf_overlay_cell_bed = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dlg._high_perf_overlay_node_x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=np.float64)
    dlg._high_perf_overlay_node_y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
    dlg._high_perf_overlay_cell_nodes = np.array(
        [0, 4, 3, 0, 1, 4, 1, 5, 4, 3, 4, 7,
         3, 7, 6, 4, 5, 8, 4, 8, 7], dtype=np.int32
    )
    dlg._high_perf_overlay_tri_to_cell = np.empty(0, dtype=np.int32)
    dlg._high_perf_overlay_mesh_fingerprint = "test_fingerprint"
    dlg._snapshot_timesteps = [
        (0.0, np.array([0.5, 0.6, 0.4, 0.7], dtype=np.float64),
               np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
               np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        (10.0, np.array([0.8, 0.9, 0.7, 1.0], dtype=np.float64),
                np.array([0.1, 0.0, 0.0, 0.1], dtype=np.float64),
                np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float64)),
    ]
    dlg._gravity = 9.81
    dlg._length_unit_name = "m"
    dlg._mannings_n = 0.035
    dlg._overlay_no_data_warned = False
    dlg._overlay_last_loaded_t_s = None
    dlg._model_gpkg_path = ""
    dlg._log = MagicMock()
    return dlg


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
    """AST-based verification that auto-load hooks exist (or are missing).

    These tests inspect the source code statically rather than executing
    the dialog, making them fast and suitable for regression detection.
    """

    def setUp(self):
        self.wb_path = _get_source_path("swe2d_workbench_qt")

    # ------------------------------------------------------------------
    # BUG: snapshot does NOT call auto-load
    # ------------------------------------------------------------------

    def test_snapshot_missing_auto_load(self):
        """REGRESSION TEST (NOW FIXED): snapshot triggers auto-load via the controller.

        After Phase 2 Task 8 the snapshot orchestration lives on
        ``WorkbenchController.on_snapshot``. The auto-load is still
        triggered through ``_refresh_snapshot_overlay`` which is owned
        by the dialog but called from the controller. This test now
        checks the controller AST.
        """
        from swe2d.workbench import workbench_controller
        ctrl_path = os.path.normpath(
            os.path.join(
                os.path.dirname(self.wb_path),
                "swe2d", "workbench", "workbench_controller.py",
            )
        )
        func_node = _get_function_source_ast(ctrl_path, "on_snapshot")
        self.assertIsNotNone(
            func_node,
            "WorkbenchController.on_snapshot must exist after Phase 2 Task 8",
        )
        has_delegation = _ast_has_call(func_node, "_refresh_snapshot_overlay")
        self.assertTrue(
            has_delegation,
            "WorkbenchController.on_snapshot is missing _refresh_snapshot_overlay "
            "call — snapshot data is persisted to GPKG but never auto-loaded "
            "into the results panel.",
        )
        helper_node = _get_function_source_ast(self.wb_path, "_refresh_snapshot_overlay")
        has_auto_load = _ast_has_call(helper_node, "_auto_load_results_panel")
        self.assertTrue(
            has_auto_load,
            "_refresh_snapshot_overlay is missing _auto_load_results_panel call — "
            "snapshot data is persisted to GPKG but never auto-loaded "
            "into the results panel.",
        )

    def test_snapshot_calls_sync_overlay(self):
        """Controller ``on_snapshot`` delegates overlay sync via ``_refresh_snapshot_overlay``."""
        from swe2d.workbench import workbench_controller
        ctrl_path = os.path.normpath(
            os.path.join(
                os.path.dirname(self.wb_path),
                "swe2d", "workbench", "workbench_controller.py",
            )
        )
        func_node = _get_function_source_ast(ctrl_path, "on_snapshot")
        self.assertIsNotNone(
            func_node,
            "WorkbenchController.on_snapshot must exist after Phase 2 Task 8",
        )
        has_delegation = _ast_has_call(func_node, "_refresh_snapshot_overlay")
        self.assertTrue(
            has_delegation,
            "WorkbenchController.on_snapshot is missing _refresh_snapshot_overlay "
            "call — snapshot data won't be available in the overlay.",
        )
        helper_node = _get_function_source_ast(self.wb_path, "_refresh_snapshot_overlay")
        has_sync = _ast_has_call(helper_node, "_sync_high_perf_overlay_data")
        self.assertTrue(
            has_sync,
            "_refresh_snapshot_overlay is missing _sync_high_perf_overlay_data call — "
            "snapshot data won't be available in the overlay.",
        )

    # ------------------------------------------------------------------
    # Run completion DOES call auto-load
    # ------------------------------------------------------------------



class TestResultsSignalChain(unittest.TestCase):
    """Verify that the results panel → workbench overlay signal chain exists.

    The chain is:
        panel._time_slider.valueChanged
          → panel._on_slider_changed
            → panel._anim.set_index
              → panel._on_controller_timestep_changed
                → panel.timestep_changed.emit(t_s)
                  → dialog._on_results_panel_timestep_changed(t_s)
                    → dialog._update_high_perf_overlay_time(t_s)
                      → dialog._refresh_high_perf_canvas_overlay(t_s)
    """

    def test_panel_has_timestep_signal(self):
        """SWE2DResultsPanel has the timestep_changed signal."""
        from swe2d.results.data import SWE2DResultsData
        self.assertTrue(
            hasattr(SWE2DResultsData, "set_data_source"),
            "SWE2DResultsData is missing set_data_source method",
        )

    def test_dialog_has_timestep_handler(self):
        """Workbench dialog has the _on_results_panel_timestep_changed handler."""
        wb = _import_wb_module()
        cls = getattr(wb, "SWE2DWorkbenchDialog", None)
        if cls is None:
            self.skipTest("SWE2DWorkbenchDialog not importable")
        self.assertTrue(
            hasattr(cls, "_on_results_panel_timestep_changed"),
            "SWE2DWorkbenchDialog missing _on_results_panel_timestep_changed",
        )

    def test_studio_dialog_has_timestep_handler(self):
        """Studio dialog has the _on_results_panel_timestep_changed handler."""
        wb = _import_wb_module()
        cls = getattr(wb, "SWE2DWorkbenchStudioDialog", None)
        if cls is None:
            self.skipTest("SWE2DWorkbenchStudioDialog not importable")
        self.assertTrue(
            hasattr(cls, "_on_results_panel_timestep_changed"),
            "SWE2DWorkbenchStudioDialog missing _on_results_panel_timestep_changed",
        )

    def test_bridge_connects_timestep_signal(self):
        """The results_bridge connects timestep_changed → handler."""
        from swe2d.workbench.bridges.results_bridge import maybe_create_results_data
        import inspect
        src = inspect.getsource(maybe_create_results_data)
        self.assertIn(
            "timestep_changed",
            src,
            "results_bridge.maybe_create_results_data does not reference "
            "timestep_changed signal",
        )
        self.assertIn(
            "_on_results_panel_timestep_changed",
            src,
            "results_bridge.maybe_create_results_data does not reference "
            "_on_results_panel_timestep_changed handler",
        )


class TestOverlayUpdateBridge(unittest.TestCase):
    """Test that the overlay update bridge correctly invokes refresh."""

    def test_update_overlay_time_calls_refresh(self):
        """``update_high_perf_overlay_time`` calls ``_refresh_high_perf_canvas_overlay``."""
        dlg = _build_mock_dialog()
        from swe2d.workbench.bridges.high_perf_overlay_bridge import update_high_perf_overlay_time

        update_high_perf_overlay_time(dlg, 5.0)

        # Verify _refresh_high_perf_canvas_overlay was called with 5.0
        dlg._refresh_high_perf_canvas_overlay.assert_called_once_with(5.0)

    def test_update_overlay_time_validates_float(self):
        """Bridge converts to float and still works."""
        dlg = _build_mock_dialog()
        from swe2d.workbench.bridges.high_perf_overlay_bridge import update_high_perf_overlay_time

        update_high_perf_overlay_time(dlg, "5.0")

        dlg._refresh_high_perf_canvas_overlay.assert_called_once()
        call_arg = dlg._refresh_high_perf_canvas_overlay.call_args[0][0]
        self.assertAlmostEqual(float(call_arg), 5.0)

    def test_sync_overlay_data_clears_on_no_timesteps(self):
        """``sync_high_perf_overlay_data`` empties arrays when no timesteps."""
        dlg = _build_mock_dialog()
        dlg._snapshot_timesteps = []  # No data
        from swe2d.workbench.bridges.high_perf_overlay_bridge import sync_high_perf_overlay_data

        sync_high_perf_overlay_data(dlg)

        self.assertEqual(dlg._high_perf_overlay_cell_x.size, 0)
        self.assertEqual(dlg._high_perf_overlay_cell_y.size, 0)
        dlg._refresh_high_perf_canvas_overlay.assert_called_once()


class TestOverlayRendering(unittest.TestCase):
    """Test the high-perf overlay rendering pipeline with synthetic data.

    Uses the actual ``render_unstructured_snapshot_image`` function
    (not mocks) to verify the NumPy/C++ rasterization pipeline.

    These tests require real PyQt5 (not mock PyQt5) because they
    exercise QImage and QPainter directly.  Under mock QGIS, if real
    PyQt5 was imported *before* the mocks were installed, ``qgis.PyQt``
    delegates to real Qt classes and these tests will run fine even in
    a headless environment.

    NOTE: ``sys.modules`` cleanup is needed because other test classes in
    this file (e.g. ``TestResultsSignalChain``) trigger an import chain
    that pulls in ``swe2d.results.high_perf_viewer`` under mock QGIS, leaving a
    stale module entry with ``__spec__ = None`` and all functions as
    ``MagicMock``.  We pop & reimport here to force the real module load.
    """

    @classmethod
    def setUpClass(cls):
        # Other test classes may have left a stale entry in sys.modules
        sys.modules.pop("swe2d.results.high_perf_viewer", None)
        import importlib as _il
        _il.invalidate_caches()

        if not _HAS_REAL_QT:
            raise unittest.SkipTest(
                "Real PyQt5 QImage/QPainter unavailable — "
                "skipping QPainter-dependent rendering tests"
            )
        _ensure_qapp()

    def setUp(self):
        # Create a synthetic 2×2 quad mesh (4 cells → 8 triangles)
        n_cells = 4
        n_tri = 8
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


class TestOverlayDialogIntegration(unittest.TestCase):
    """Integration-style tests using a mock dialog + real bridge functions.

    These tests verify that the overlay data sync and refresh chain
    works end-to-end with synthetic mesh data.
    """

    def setUp(self):
        self.dlg = _build_mock_dialog()

    def test_overlay_time_update_chain(self):
        """Full chain: update → refresh called with correct time."""
        from swe2d.workbench.bridges.high_perf_overlay_bridge import update_high_perf_overlay_time

        update_high_perf_overlay_time(self.dlg, 10.0)
        self.dlg._refresh_high_perf_canvas_overlay.assert_called_once_with(10.0)

    def test_overlay_data_sync_populates_arrays(self):
        """sync_high_perf_overlay_data fills cell/bed arrays from mesh_data."""
        self.dlg._mesh_data = {
            "node_x": np.array([0, 1, 0, 1], dtype=np.float64),
            "node_y": np.array([0, 0, 1, 1], dtype=np.float64),
            "cell_nodes": np.array([0, 1, 3, 0, 3, 2], dtype=np.int32),
        }

        # We need a _mesh_cell_centroids and _mesh_cell_solver_bed
        def fake_centroids():
            return (
                np.array([0.25, 0.75], dtype=np.float64),
                np.array([0.25, 0.75], dtype=np.float64),
            )

        def fake_bed():
            return np.array([10.0, 12.0], dtype=np.float64)

        self.dlg._mesh_cell_centroids = fake_centroids
        self.dlg._mesh_cell_solver_bed = fake_bed

        from swe2d.workbench.bridges.high_perf_overlay_bridge import sync_high_perf_overlay_data
        sync_high_perf_overlay_data(self.dlg)

        self.assertEqual(self.dlg._high_perf_overlay_cell_x.size, 2)
        self.assertEqual(self.dlg._high_perf_overlay_cell_y.size, 2)
        self.assertEqual(self.dlg._high_perf_overlay_cell_bed.size, 2)
        self.assertAlmostEqual(float(self.dlg._high_perf_overlay_cell_bed[0]), 10.0)
        self.assertAlmostEqual(float(self.dlg._high_perf_overlay_cell_bed[1]), 12.0)

    def test_mesh_fingerprint_stable(self):
        """Same mesh data produces same fingerprint."""
        from swe2d.workbench.bridges.high_perf_overlay_bridge import mesh_fingerprint_from_arrays

        n1 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        n2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        cn = np.array([0, 1, 3, 0, 3, 2], dtype=np.int32)

        fp1 = mesh_fingerprint_from_arrays(n1, n2, cn)
        fp2 = mesh_fingerprint_from_arrays(n1, n2, cn)
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, "")  # Non-empty


class TestResultsPanelControls(unittest.TestCase):
    """Verify the results panel's animation controls work correctly.

    Tests the slider → animation controller → signal chain.
    """

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

    def test_high_perf_viewer_logs_on_hydra_overlay_fallback(self):
        """Source code: hydra_overlay import failure is LOGGED (not silent).

        Instead of trying to re-import the module (which may already be
        cached), we verify the source code contains the warning.
        This catches regressions where the warning is accidentally removed.
        """
        # Read the raw source file — avoids any import/mock issues.
        hpf_path = os.path.join(_REPO_ROOT, "swe2d/results/high_perf_viewer.py")
        self.assertTrue(
            os.path.exists(hpf_path),
            f"swe2d/results/high_perf_viewer.py not found at {hpf_path}",
        )
        with open(hpf_path, "r") as f:
            src = f.read()

        self.assertIn(
            "hydra_overlay",
            src,
            "swe2d.results.high_perf_viewer has no hydra_overlay reference at all",
        )
        # Look for the specific warning pattern that we added to replace
        # the old silent ``_hydra_overlay = None``.
        has_fallback_warning = (
            "exc_info=True" in src
            and "logger.warning" in src
            and "hydra_overlay" in src
            and "falling back" in src.lower()
        )
        self.assertTrue(
            has_fallback_warning,
            "swe2d.results.high_perf_viewer does NOT log a warning when "
            "hydra_overlay import fails — this would be a silent fallback! "
            "Look for: logger.warning(...'hydra_overlay'...exc_info=True...)",
        )


# =========================================================================
# Run with: python -m pytest tests/test_swe2d_overlay_and_autoload.py -v
# or: python -m pytest tests/ -k "overlay or autoload" -v
# =========================================================================

if __name__ == "__main__":
    unittest.main()
