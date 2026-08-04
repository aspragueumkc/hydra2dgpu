#!/usr/bin/env python3
"""Tests for swe2d.results.high_perf_viewer — renderer overlay field rendering.

Runs against the REAL renderer under real headless QGIS (offscreen
QImage/QPainter).  The previous mock replaced the whole
``swe2d.results.high_perf_viewer`` module with a fake returning a
``bytearray``; that fake is gone — these tests assert on the real
renderer output.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Tuple

import numpy as np

# Ensure repo root and build dir are on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tests.qgis_real_env import ensure_qgis_app, requires_qgis


def _assert_render_ok(
    case: unittest.TestCase,
    result: dict,
    resolution: Tuple[int, int] = (160, 90),
) -> None:
    """Assertions re-derived from the REAL renderer contract.

    render_unstructured_snapshot_image returns a dict with ok/image/vmin/
    vmax; on success image is a real QImage of the requested resolution
    (clamped to >= 32 px per side) with at least one opaque pixel, and
    vmax > vmin (the renderer forces vmax = vmin + 1.0 on degenerate
    ranges).
    """
    from qgis.PyQt.QtGui import QImage

    case.assertTrue(
        result.get("ok") is True,
        f"Render failed: {result.get('message', 'unknown')}",
    )
    image = result.get("image")
    case.assertIsInstance(image, QImage)
    case.assertFalse(image.isNull(), "rendered QImage must not be null")
    case.assertEqual(image.width(), resolution[0])
    case.assertEqual(image.height(), resolution[1])
    # Non-empty content: at least one pixel painted (alpha > 0)
    has_opaque = any(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, image.width(), 7)
        for y in range(0, image.height(), 7)
    )
    case.assertTrue(has_opaque, "rendered image has no opaque pixels")
    vmin = result.get("vmin", np.nan)
    vmax = result.get("vmax", np.nan)
    case.assertTrue(np.isfinite(vmin), "vmin should be finite")
    case.assertTrue(np.isfinite(vmax), "vmax should be finite")
    case.assertGreater(vmax, vmin, "renderer guarantees vmax > vmin on success")


# =========================================================================
# Renderer tests — real QImage rendering under offscreen Qt
# =========================================================================

@requires_qgis
class TestRenderOverlayFields(unittest.TestCase):
    """Verify that render_unstructured_snapshot_image correctly handles
    rain/Manning/CN overlay cell arrays passed as kwargs."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

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

    def _render(self, field_key: str, **overlay_kw) -> dict:
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        return render_unstructured_snapshot_image(
            cell_x=self.cell_x,
            cell_y=self.cell_y,
            cell_bed=self.cell_bed,
            timesteps=self.timesteps,
            current_time_s=0.0,
            field_key=field_key,
            node_x=self.overlay_node_x,
            node_y=self.overlay_node_y,
            cell_nodes=self.overlay_cell_nodes,
            resolution=(160, 90),
            **overlay_kw,
        )

    # ------------------------------------------------------------------
    # Test 1 — mannings_n overlay
    # ------------------------------------------------------------------
    def test_render_mannings_n_field(self):
        """Manning's n overlay field renders without error and computes vmin/vmax."""
        result = self._render(
            "mannings_n",
            overlay_cell_mannings_n=np.array([0.013, 0.015], dtype=np.float64),
        )
        _assert_render_ok(self, result)

    # ------------------------------------------------------------------
    # Test 2 — curve_number overlay
    # ------------------------------------------------------------------
    def test_render_curve_number_field(self):
        """Curve Number overlay field renders without error and computes vmin/vmax."""
        result = self._render(
            "curve_number",
            overlay_cell_curve_number=np.array([60.0, 72.0], dtype=np.float64),
        )
        _assert_render_ok(self, result)

    # ------------------------------------------------------------------
    # Test 3 — cumulative_rain and cumulative_excess overlay fields
    # ------------------------------------------------------------------
    def test_render_cumulative_rain_cumulative_excess_fields(self):
        """Both cumulative_rain and cumulative_excess overlay fields render without error."""
        result_rain = self._render(
            "cumulative_rain",
            overlay_cell_cumulative_rain=np.array([0.0, 5.0], dtype=np.float64),
        )
        _assert_render_ok(self, result_rain)

        result_excess = self._render(
            "cumulative_excess",
            overlay_cell_cumulative_excess=np.array([0.0, 2.0], dtype=np.float64),
        )
        _assert_render_ok(self, result_excess)


if __name__ == "__main__":
    unittest.main()
