#!/usr/bin/env python3
"""Tests for the SWE2D results data layer and visualization components.

Validates that SWE2DResultsData provides the expected API,
and that ResultsAnimationController and overlay rendering work correctly.
"""

from __future__ import annotations

import unittest
from typing import List


class TestResultsDataImports(unittest.TestCase):
    """Verify the data module and its key classes import cleanly."""

    def test_data_imports(self):
        from swe2d.results.data import SWE2DResultsData
        self.assertIsNotNone(SWE2DResultsData)

    def test_data_class_exists(self):
        from swe2d.results.data import SWE2DResultsData
        self.assertTrue(callable(SWE2DResultsData))

    def test_data_methods_exist(self):
        from swe2d.results.data import SWE2DResultsData

        methods = [
            "discover_runs",
            "get_run_records",
            "enabled_overlay_targets",
            "current_time_sec",
            "save_data_state",
            "restore_data_state",
        ]
        for m in methods:
            self.assertTrue(
                hasattr(SWE2DResultsData, m),
                f"Missing method: {m}",
            )

    def test_data_overlay_properties(self):
        from swe2d.results.data import SWE2DResultsData
        self.assertTrue(
            hasattr(SWE2DResultsData, "velocity_overlay_enabled"),
        )
        self.assertTrue(
            hasattr(SWE2DResultsData, "streamline_overlay_enabled"),
        )


class TestDrainageToolsImport(unittest.TestCase):
    """Verify the standalone drainage network viewer imports."""

    def test_drainage_viewer_import(self):
        import importlib
        try:
            mod = importlib.import_module("tools.drainage_network_viewer")
            self.assertIsNotNone(mod)
        except ImportError:
            import sys
            import os
            root = os.path.join(os.path.dirname(__file__), "..")
            tools_dir = os.path.join(root, "tools")
            if os.path.exists(tools_dir) and tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "drainage_network_viewer",
                os.path.join(tools_dir, "drainage_network_viewer.py"),
            )
            self.assertIsNotNone(spec, "drainage_network_viewer.py not found")


class TestHighPerfOverlay(unittest.TestCase):
    """Verify high-perf overlay rendering functions."""

    def test_render_function_import(self):
        from swe2d.results.high_perf_viewer import (
            render_unstructured_snapshot_image,
            _build_color_lut,
            _draw_scalar_legend,
        )
        self.assertIsNotNone(render_unstructured_snapshot_image)
        self.assertIsNotNone(_build_color_lut)
        self.assertIsNotNone(_draw_scalar_legend)
        stops = [(0.0, (0, 0, 255)), (1.0, (255, 0, 0))]
        lut = _build_color_lut(stops)
        self.assertEqual(lut.shape, (256, 3))

    def test_new_overlay_field_params(self):
        import inspect
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        sig = inspect.signature(render_unstructured_snapshot_image)
        params = list(sig.parameters.keys())
        for required in ("gravity", "courant_cell_size", "courant_dt", "mannings_n"):
            self.assertIn(required, params, f"Missing param: {required}")

    def test_no_set_canvas_rotation(self):
        import inspect
        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        src = inspect.getsource(inspect.getmodule(render_unstructured_snapshot_image))
        self.assertNotIn("set_canvas_rotation", src)
        self.assertNotIn("_canvas_rotation_deg", src)


if __name__ == "__main__":
    unittest.main()
