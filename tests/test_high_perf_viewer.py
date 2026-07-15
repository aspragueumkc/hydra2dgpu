#!/usr/bin/env python3
"""Tests for swe2d.results.high_perf_viewer — renderer overlay field rendering.

Requires real PyQt5 (QImage/QPainter) for rendering tests.  When QGIS mocks
are installed before real PyQt5 imports, ``qgis.PyQt`` delegates to real Qt
classes so these tests can run headlessly.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List, Tuple

import numpy as np

# Ensure repo root and build dir are on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Headless Qt + mock QGIS bootstrap — same pattern as test_swe2d_overlay_and_autoload
# ---------------------------------------------------------------------------
from PyQt5.QtGui import QImage as _RealQImage, QPainter as _RealQPainter
from PyQt5.QtWidgets import QApplication as _QApp

_test_app = _QApp.instance()
if _test_app is None:
    _test_app = _QApp([])

from tests.mocks.qgis_env import install_qgis_mocks as _install_qgis_mocks

_install_qgis_mocks()

# Clear any stale module entry left by other test classes
sys.modules.pop("swe2d.results.high_perf_viewer", None)
import importlib as _il
_il.invalidate_caches()


def _is_real_qimage(cls: type) -> bool:
    mod = getattr(cls, "__module__", "") or ""
    return str(mod).startswith("PyQt5") and not str(mod).startswith("unittest")


_HAS_REAL_QT = _is_real_qimage(_RealQImage) and _is_real_qimage(_RealQPainter)


def _make_timesteps(
    n_cells: int = 4,
    n_ts: int = 3,
    depth_range: Tuple[float, float] = (0.5, 1.5),
) -> List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Build synthetic timesteps (t_s, h, hu, hv) for testing."""
    timesteps = []
    for i in range(n_ts):
        t = float(i) * 60.0
        h = np.random.uniform(depth_range[0], depth_range[1], n_cells).astype(np.float64)
        hu = np.random.uniform(-0.2, 0.2, n_cells).astype(np.float64)
        hv = np.random.uniform(-0.2, 0.2, n_cells).astype(np.float64)
        timesteps.append((t, h, hu, hv))
    return timesteps


# =========================================================================
# Renderer tests — require real PyQt5
# =========================================================================

class TestRenderOverlayFields(unittest.TestCase):
    """Verify that render_unstructured_snapshot_image correctly handles
    rain/Manning/CN overlay cell arrays passed as kwargs."""

    @classmethod
    def setUpClass(cls):
        sys.modules.pop("swe2d.results.high_perf_viewer", None)
        _il.invalidate_caches()
        if not _HAS_REAL_QT:
            raise unittest.SkipTest(
                "Real PyQt5 QImage/QPainter unavailable — skipping rendering tests"
            )
        _test_app  # ensure QApplication exists

    def setUp(self):
        # 2-cell quad mesh (4 nodes, 2 triangles)
        self.n_cells = 2
        self.overlay_node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        self.overlay_node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        self.overlay_cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.cell_x = np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        self.cell_y = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
        self.cell_bed = np.array([0.0, 0.1], dtype=np.float64)
        h = np.ones(self.n_cells, dtype=np.float64)  # wet cells
        hu = np.zeros(self.n_cells, dtype=np.float64)
        hv = np.zeros(self.n_cells, dtype=np.float64)
        self.timesteps = [(0.0, h, hu, hv)]

    # ------------------------------------------------------------------
    # Test 1 — mannings_n overlay
    # ------------------------------------------------------------------
    def test_render_mannings_n_field(self):
        """Manning's n overlay field renders without error and computes vmin/vmax."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            timesteps=self.timesteps,
            current_time_s=0.0,
            field_key="mannings_n",
            overlay_cell_mannings_n=np.array([0.013, 0.015], dtype=np.float64),
            node_x=self.overlay_node_x,
            node_y=self.overlay_node_y,
            cell_nodes=self.overlay_cell_nodes,
            resolution=(160, 90),
        )
        self.assertTrue(
            result.get("ok", False) is not False,
            f"Render failed: {result.get('message', 'unknown')}",
        )
        self.assertTrue(
            np.isfinite(result.get("vmin", np.nan)),
            "vmin should be finite",
        )
        self.assertTrue(
            np.isfinite(result.get("vmax", np.nan)),
            "vmax should be finite",
        )

    # ------------------------------------------------------------------
    # Test 2 — curve_number overlay
    # ------------------------------------------------------------------
    def test_render_curve_number_field(self):
        """Curve Number overlay field renders without error and computes vmin/vmax."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        result = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            timesteps=self.timesteps,
            current_time_s=0.0,
            field_key="curve_number",
            overlay_cell_curve_number=np.array([60.0, 72.0], dtype=np.float64),
            node_x=self.overlay_node_x,
            node_y=self.overlay_node_y,
            cell_nodes=self.overlay_cell_nodes,
            resolution=(160, 90),
        )
        self.assertTrue(
            result.get("ok", False) is not False,
            f"Render failed: {result.get('message', 'unknown')}",
        )
        self.assertTrue(
            np.isfinite(result.get("vmin", np.nan)),
            "vmin should be finite",
        )
        self.assertTrue(
            np.isfinite(result.get("vmax", np.nan)),
            "vmax should be finite",
        )

    # ------------------------------------------------------------------
    # Test 3 — cumulative_rain and cumulative_excess overlay fields
    # ------------------------------------------------------------------
    def test_render_cumulative_rain_cumulative_excess_fields(self):
        """Both cumulative_rain and cumulative_excess overlay fields render without error."""
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        # Test cumulative_rain
        result_rain = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            timesteps=self.timesteps,
            current_time_s=0.0,
            field_key="cumulative_rain",
            overlay_cell_cumulative_rain=np.array([0.0, 5.0], dtype=np.float64),
            node_x=self.overlay_node_x,
            node_y=self.overlay_node_y,
            cell_nodes=self.overlay_cell_nodes,
            resolution=(160, 90),
        )
        self.assertTrue(
            result_rain.get("ok", False) is not False,
            f"cumulative_rain render failed: {result_rain.get('message', 'unknown')}",
        )
        self.assertTrue(np.isfinite(result_rain.get("vmin", np.nan)))
        self.assertTrue(np.isfinite(result_rain.get("vmax", np.nan)))

        # Test cumulative_excess
        result_excess = render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            timesteps=self.timesteps,
            current_time_s=0.0,
            field_key="cumulative_excess",
            overlay_cell_cumulative_excess=np.array([0.0, 2.0], dtype=np.float64),
            node_x=self.overlay_node_x,
            node_y=self.overlay_node_y,
            cell_nodes=self.overlay_cell_nodes,
            resolution=(160, 90),
        )
        self.assertTrue(
            result_excess.get("ok", False) is not False,
            f"cumulative_excess render failed: {result_excess.get('message', 'unknown')}",
        )
        self.assertTrue(np.isfinite(result_excess.get("vmin", np.nan)))
        self.assertTrue(np.isfinite(result_excess.get("vmax", np.nan)))


if __name__ == "__main__":
    unittest.main()
