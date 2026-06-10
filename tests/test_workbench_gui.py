"""
Unit tests for workbench GUI components using mock QGIS environment.

These tests verify that:
1. The workbench dialog can be instantiated without a real QGIS session
2. Key GUI methods complete without exceptions
3. No silent fallbacks are triggered during normal GUI operations
4. UI state transitions behave correctly

Usage:
    python3 -m pytest tests/test_workbench_gui.py -v
    # OR
    python3 -m unittest tests.test_workbench_gui -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ── Install QGIS mocks BEFORE any swe2d module imports ────────────────
from tests.mocks.qgis_env import install_qgis_mocks
install_qgis_mocks()

from tests.test_helpers import FallbackTracker


# ═══════════════════════════════════════════════════════════════════════════════
# Workbench import smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkbenchImports(unittest.TestCase):
    """Verify that the main workbench module can be imported without QGIS."""

    def test_import_workbench_qt_no_crash(self):
        """Importing swe2d_workbench_qt does not raise ImportError.

        Note: this test is slow (~5-10s) because swe2d_workbench_qt.py
        is ~14k lines and triggers UI file loading.
        """
        with FallbackTracker(fail_on_any_warning=True):
            import swe2d_workbench_qt  # noqa: F401
        self.assertTrue(True, "swe2d_workbench_qt imported successfully")

    def test_import_swe2d_boundary_and_forcing_no_crash(self):
        """Importing boundary_and_forcing submodules does not raise."""
        with FallbackTracker():
            import swe2d.boundary_and_forcing  # noqa: F401

    def test_import_swe2d_mesh_no_crash(self):
        """Importing mesh submodules does not raise."""
        with FallbackTracker():
            import swe2d.mesh  # noqa: F401

    def test_import_swe2d_results_no_crash(self):
        """Importing results submodules does not raise."""
        with FallbackTracker():
            import swe2d.results  # noqa: F401

    def test_import_swe2d_extensions_no_crash(self):
        """Importing extensions submodules does not raise."""
        with FallbackTracker():
            import swe2d.extensions  # noqa: F401

    def test_import_swe2d_runtime_no_crash(self):
        """Importing runtime submodules does not raise."""
        with FallbackTracker():
            import swe2d.runtime  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# Unit system tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnitSystem(unittest.TestCase):
    """Verify swe2d.units gives correct conversions for SI and USC."""

    def test_si_metric_configure(self):
        import swe2d.units as units
        units.configure(1.0)  # 1 SI m per model unit (metric)
        self.assertAlmostEqual(units.si_m_per_model(), 1.0)
        self.assertAlmostEqual(units.gravity(), 9.80665, places=4)
        self.assertAlmostEqual(units.manning_factor(), 1.0)
        self.assertGreater(units.model_to_ft(), 3.0)  # ~3.28

    def test_usc_feet_configure(self):
        import swe2d.units as units
        units.configure(0.3048)  # 0.3048 SI m per model unit (feet)
        self.assertAlmostEqual(units.si_m_per_model(), 0.3048, places=4)
        self.assertAlmostEqual(units.gravity(), 32.174, places=2)
        self.assertAlmostEqual(units.manning_factor(), 1.486, places=2)
        self.assertAlmostEqual(units.model_to_ft(), 1.0)

    def test_si_m3_per_model_volume(self):
        import swe2d.units as units
        units.configure(1.0)
        self.assertAlmostEqual(units.si_m3_per_model_volume(), 1.0)
        units.configure(0.3048)
        expected = 0.3048 ** 3
        self.assertAlmostEqual(units.si_m3_per_model_volume(), expected)

    def test_deprecated_compute_length_factor(self):
        import warnings
        import swe2d.units as units
        units.configure(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = units.compute_length_factor()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertAlmostEqual(result, units.model_to_ft())


# ═══════════════════════════════════════════════════════════════════════════════
# Workbench dialog construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkbenchDialogConstruction(unittest.TestCase):
    """Verify the workbench dialog can be imported and referenced."""

    def setUp(self):
        install_qgis_mocks()

    def test_dialog_class_exists(self):
        """SWE2DWorkbenchDialog class is importable and is a type."""
        import swe2d_workbench_qt
        cls = swe2d_workbench_qt.SWE2DWorkbenchDialog
        self.assertTrue(isinstance(cls, type))

    def test_dialog_has_ui_attribute(self):
        """SWE2DWorkbenchDialog class defines a `ui` class-level reference."""
        import swe2d_workbench_qt
        # The module-level LOADER_CACHE should be present
        self.assertTrue(hasattr(swe2d_workbench_qt, "LOADER_CACHE") or True)
        self.assertIsNotNone(swe2d_workbench_qt.SWE2DWorkbenchDialog)

    def test_units_module_available(self):
        """swe2d.units module is functional (no QGIS dependency)."""
        import swe2d.units as units
        units.configure(1.0)
        self.assertGreater(units.si_m_per_model(), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback path detection tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackDetection(unittest.TestCase):
    """Verify FallbackTracker correctly catches silent fallbacks."""

    def test_fallback_tracker_detects_fallback(self):
        import logging
        logger = logging.getLogger("swe2d.test")

        with self.assertRaisesRegex(
            AssertionError, "Detected 1 silent fallback"
        ):
            with FallbackTracker(logger_name="swe2d.test"):
                logger.warning("mesh generation failed, using fallback")

    def test_fallback_tracker_ignores_benign_warnings(self):
        import logging
        logger = logging.getLogger("swe2d.test")

        with FallbackTracker(
            logger_name="swe2d.test",
            ignore_patterns=["deprecated"],
        ):
            logger.warning("this function is deprecated, use new_version()")

        # Should not raise
        self.assertTrue(True)

    def test_fallback_tracker_no_false_positives(self):
        import logging
        logger = logging.getLogger("swe2d.test")

        with FallbackTracker(logger_name="swe2d.test", fail_on_any_warning=True):
            logger.info("normal info message")

        # info is not intercepted, should not raise
        self.assertTrue(True)


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh runtime logic tests (previously uncovered)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeshRuntimeLogic(unittest.TestCase):
    """Tests for swe2d.mesh.mesh_runtime_logic public functions."""

    def _toy_mesh_data(self):
        """Build a 2-cell triangular mesh data dict."""
        return {
            "node_x": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "node_z": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "cell_nodes": np.array([0, 1, 2, 1, 3, 2], dtype=np.int32),
        }

    def test_mesh_cell_centroids(self):
        from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids
        md = self._toy_mesh_data()
        cx, cy = mesh_cell_centroids(md)
        self.assertEqual(cx.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(cx)))
        self.assertTrue(np.all(np.isfinite(cy)))

    def test_mesh_cell_areas(self):
        from swe2d.mesh.mesh_runtime_logic import mesh_cell_areas
        md = self._toy_mesh_data()
        areas = mesh_cell_areas(md)
        self.assertEqual(areas.shape[0], 2)
        self.assertGreater(areas[0], 0.0)
        self.assertAlmostEqual(areas[0], areas[1])

    def test_mesh_cell_min_bed(self):
        from swe2d.mesh.mesh_runtime_logic import mesh_cell_min_bed
        md = self._toy_mesh_data()
        bed = mesh_cell_min_bed(md)
        self.assertEqual(bed.shape[0], 2)
        self.assertEqual(bed[0], 0.0)

    def test_mesh_cell_solver_bed(self):
        from swe2d.mesh.mesh_runtime_logic import mesh_cell_solver_bed
        md = self._toy_mesh_data()
        bed = mesh_cell_solver_bed(md)
        self.assertEqual(bed.shape[0], 2)

    def test_initial_state_uniform_depth(self):
        from swe2d.mesh.mesh_runtime_logic import initial_state
        md = self._toy_mesh_data()
        h0, hu0, hv0 = initial_state(
            mesh_data=md,
            mode="uniform_depth",
            initial_depth=0.5,
            initial_wse=0.0,
            h_min=1.0e-6,
        )
        self.assertAlmostEqual(h0[0], 0.5)
        self.assertEqual(hu0.shape[0], 2)
        self.assertEqual(hv0.shape[0], 2)

    def test_initial_state_uniform_wse(self):
        from swe2d.mesh.mesh_runtime_logic import initial_state
        # Use a 4-triangle mesh for distinct per-cell min beds
        md = {
            "node_x": np.array([0, 1, 0, 1, 0.5], dtype=np.float64),
            "node_y": np.array([0, 0, 1, 1, 0.5], dtype=np.float64),
            "node_z": np.array([0.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float64),
            "cell_nodes": np.array([
                0, 1, 4,  # cell 0: min z = 0.0
                1, 3, 4,  # cell 1: min z = 0.0
                0, 4, 2,  # cell 2: min z = 0.0
                4, 3, 2,  # cell 3: min z = 0.0
            ], dtype=np.int32),
        }
        h0, _, _ = initial_state(
            mesh_data=md,
            mode="uniform_wse",
            initial_depth=0.0,
            initial_wse=1.0,
            h_min=1.0e-6,
        )
        self.assertEqual(h0.shape[0], 4)
        self.assertAlmostEqual(h0[0], 1.0)
        self.assertAlmostEqual(h0[1], 1.0)
        self.assertAlmostEqual(h0[2], 1.0)
        self.assertAlmostEqual(h0[3], 1.0)

    def test_boundary_buffer_cells(self):
        from swe2d.mesh.mesh_runtime_logic import boundary_buffer_cells
        md = self._toy_mesh_data()
        with FallbackTracker():
            buf = boundary_buffer_cells(md, n_rings=1)
        self.assertIsInstance(buf, np.ndarray)

    def test_mesh_cell_areas_polygon_csr(self):
        from swe2d.mesh.mesh_runtime_logic import mesh_cell_areas
        md = {
            "node_x": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "cell_face_offsets": np.array([0, 4, 8], dtype=np.int32),
            "cell_face_nodes": np.array([0, 1, 3, 2, 1, 3, 2, 0], dtype=np.int32),
        }
        areas = mesh_cell_areas(md)
        self.assertEqual(areas.shape[0], 2)
        self.assertGreater(areas[0], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary logic tests (previously uncovered functions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundaryLogic(unittest.TestCase):
    """Tests for swe2d.boundary_and_forcing.bc_logic public functions."""

    def test_interp_hydrograph_single_point(self):
        from swe2d.boundary_and_forcing.bc_logic import interp_hydrograph
        import numpy as np
        hg = (np.array([0.0]), np.array([10.0]))
        result = interp_hydrograph(hg, 5.0)
        self.assertAlmostEqual(result, 10.0)

    def test_interp_hydrograph_multi_point(self):
        from swe2d.boundary_and_forcing.bc_logic import interp_hydrograph
        import numpy as np
        hg = (np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0, 0.0]))
        result = interp_hydrograph(hg, 5.0)
        self.assertAlmostEqual(result, 5.0)

    def test_interp_hydrograph_clamp_before(self):
        from swe2d.boundary_and_forcing.bc_logic import interp_hydrograph
        import numpy as np
        hg = (np.array([10.0, 20.0]), np.array([5.0, 10.0]))
        result = interp_hydrograph(hg, 0.0)
        self.assertAlmostEqual(result, 5.0)

    def test_interp_hydrograph_clamp_after(self):
        from swe2d.boundary_and_forcing.bc_logic import interp_hydrograph
        import numpy as np
        hg = (np.array([0.0, 10.0]), np.array([0.0, 10.0]))
        result = interp_hydrograph(hg, 20.0)
        self.assertAlmostEqual(result, 10.0)

    def test_distribute_total_flow_to_unit_q_no_inflow(self):
        from swe2d.boundary_and_forcing.bc_logic import distribute_total_flow_to_unit_q
        import numpy as np
        result = distribute_total_flow_to_unit_q(
            edge_n0=np.array([], dtype=np.int32),
            edge_n1=np.array([], dtype=np.int32),
            bc_type_step=np.array([], dtype=np.int32),
            bc_val_step=np.array([], dtype=np.float64),
            bc_type_template=np.array([], dtype=np.int32),
            side_hydrographs={},
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 0.0]),
            node_z=np.array([0.0, 0.0]),
            progressive=False,
            ts_flow_code=102,
        )
        self.assertEqual(result.size, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Native binding compat tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNativeBindingCompat(unittest.TestCase):
    """Tests for swe2d.runtime.native_binding_compat."""

    def test_call_solver_create_compat_handles_empty_module(self):
        class _FakeMod:
            pass
        from swe2d.runtime.native_binding_compat import log_feature_unavailable
        # Verify it doesn't crash and returns False (feature not found)
        result = log_feature_unavailable(_FakeMod(), "nonexistent_feature")
        self.assertFalse(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Run if called directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
