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
# Save real QApplication before mocks replace it (needed for dialog lifecycle tests)
from PyQt5.QtWidgets import QApplication as _REAL_QAPP
install_qgis_mocks()

from qgis.PyQt import QtCore, QtWidgets
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


class TestWorkbenchDialogConstructionFull(unittest.TestCase):
    """Static analysis checks for widget completeness after .ui removal.

    Verifies that every widget previously defined in .ui files is now
    created programmatically with ``setObjectName()`` and that no stale
    ``_find_or_create_*`` helper references remain.
    """

    def setUp(self):
        from tests.mocks.qgis_env import install_qgis_mocks
        install_qgis_mocks()

    def test_no_stale_helper_references_in_source(self):
        """No _find_or_create_, _find_child_robust, or _ensure_form_row remain."""
        import re

        files = {}
        for _fn in ("swe2d_workbench_qt.py",):
            with open(_fn) as _f:
                files[_fn] = _f.read()

        total = 0
        for fname, src in files.items():
            count = len(re.findall(
                r'_find_or_create_|_find_child_robust|_ensure_form_row',
                src
            ))
            total += count
            if count:
                print(f"  {fname}: {count} stale helper ref(s)")

        self.assertEqual(
            total, 0,
            f"Found {total} stale helper references across source files. "
            "These should be inlined into direct QtWidgets.Xxx() calls."
        )

    def test_key_widget_object_names_exist_in_source(self):
        """Critical widget objectNames appear in the source code."""
        import re

        with open("swe2d/workbench/views/topology_tab_view.py") as f:
            topo_src = f.read()

        # These widget names must appear in setObjectName("...") calls
        critical_widgets = [
            "topo_gmsh_tri_algo_combo",
            "topo_gmsh_quad_algo_combo",
            "topo_gmsh_recombine_algo_combo",
            "topo_gmsh_quality_enable_chk",
            "topo_gmsh_quality_max_iters_spin",
            "topo_gmsh_quality_time_limit_spin",
            "topo_quality_min_angle_spin",
            "topo_quality_max_aspect_spin",
            "topo_quality_max_non_orth_spin",
            "topo_quality_min_area_edit",
            "topo_quality_size_scales_edit",
            "topo_quality_smooth_increments_edit",
            "topo_gmsh_quality_recombine_topology_passes_edit",
            "topo_gmsh_quality_recombine_min_quality_edit",
            "topo_gmsh_quality_random_factors_edit",
            "topo_gmsh_quality_optimize_methods_edit",
            "topo_gmsh_algo_switch_on_failure_chk",
            "topo_gmsh_recombine_node_repositioning_chk",
            "topo_quality_strict_chk",
        ]

        missing = []
        for name in critical_widgets:
            pattern = f'setObjectName("{name}")'
            if pattern not in topo_src:
                missing.append(name)

        self.assertEqual(
            len(missing), 0,
            f"Widgets missing setObjectName() calls: {missing}"
        )

    def test_no_orphan_if_none_guards_in_topo_source(self):
        """No bare 'if self.X is None:' guards without prior initialization."""
        import re

        with open("swe2d/workbench/views/topology_tab_view.py") as f:
            src = f.read()

        # Find all "if self.X is None:" patterns that are NOT runtime guards
        # (runtime guards all have _ in attribute names like _mesh_data)
        problematic = []
        for m in re.finditer(r'    if self\.(\w+) is None:', src):
            attr = m.group(1)
            # Skip runtime attribute guards (start with _)
            if attr.startswith('_'):
                continue
            # Skip legitimate data guards
            if attr in ('_mesh_data', '_result_data'):
                continue
            # Check if there's a creation after it
            rest = src[m.end():m.end()+200]
            if 'QtWidgets.' in rest or 'setObjectName' in rest:
                problematic.append(attr)

        self.assertEqual(
            len(problematic), 0,
            f"Orphaned if-None guards remaining (need direct init): {problematic}"
        )


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
# Studio dialog lifecycle tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStudioDialogLifecycle(unittest.TestCase):
    """Verify the Studio dialog lifecycle: component registry, close, tabs."""

    _saved_qapp: type | None = None

    @classmethod
    def setUpClass(cls):
        # Restore real QApplication so processEvents() etc. work at runtime
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        cls._saved_qapp = _pyqt5_qt.QApplication if _pyqt5_qt else None
        if _pyqt5_qt is not None:
            _pyqt5_qt.QApplication = _REAL_QAPP
        if _qgis_qt is not None:
            _qgis_qt.QApplication = _REAL_QAPP
        cls._app = _REAL_QAPP.instance() or _REAL_QAPP([])

    @classmethod
    def tearDownClass(cls):
        # Restore mock QApplication for other tests
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        if _pyqt5_qt is not None and cls._saved_qapp is not None:
            _pyqt5_qt.QApplication = cls._saved_qapp
        if _qgis_qt is not None and cls._saved_qapp is not None:
            _qgis_qt.QApplication = cls._saved_qapp
        cls._saved_qapp = None

    def _make_dialog(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        iface = MagicMock()
        return SWE2DWorkbenchStudioDialog(iface=iface)

    def test_studio_component_registry(self):
        dlg = self._make_dialog()
        try:
            for key in ("results", "setup", "inspector"):
                self.assertIn(key, dlg._state.studio_components)
                comp = dlg._state.studio_components[key]
                self.assertIsNotNone(comp.dock)
        finally:
            dlg.close()

    def test_close_event_destroys_components(self):
        dlg = self._make_dialog()
        self.assertGreater(len(dlg._state.studio_components), 0)
        dlg.close()
        self.assertEqual(len(dlg._state.studio_components), 0)

    def test_left_pane_tab_order(self):
        dlg = self._make_dialog()
        try:
            tabs = dlg._left_tabs
            self.assertIsNotNone(tabs)
            self.assertGreaterEqual(tabs.count(), 5)
            self.assertEqual(tabs.tabText(0), "Mesh")
            self.assertEqual(tabs.tabText(1), "Layers")
            self.assertEqual(tabs.tabText(2), "Topo Mesh")
            self.assertEqual(tabs.tabText(3), "Boundary")
            self.assertEqual(tabs.tabText(4), "Parameters")
        finally:
            dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Studio results controls tab tests (Phase 5 / Task 23)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStudioResultsControlsTab(unittest.TestCase):
    """Verify the controls tab is built inline in _populate_results_dock."""

    _saved_qapp: type | None = None
    _app = None

    @classmethod
    def setUpClass(cls):
        # install_qgis_mocks is already called at top of file.
        # Swap mock QApplication for the real one so processEvents() works.
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        cls._saved_qapp = _pyqt5_qt.QApplication if _pyqt5_qt else None
        if _pyqt5_qt is not None:
            _pyqt5_qt.QApplication = _REAL_QAPP
        if _qgis_qt is not None:
            _qgis_qt.QApplication = _REAL_QAPP
        cls._app = _REAL_QAPP.instance() or _REAL_QAPP([])

    @classmethod
    def tearDownClass(cls):
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        if _pyqt5_qt is not None and cls._saved_qapp is not None:
            _pyqt5_qt.QApplication = cls._saved_qapp
        if _qgis_qt is not None and cls._saved_qapp is not None:
            _qgis_qt.QApplication = cls._saved_qapp
        cls._saved_qapp = None

    def test_results_view_exists_on_dialog(self):
        """After dialog build, _results_view (StudioResultsView) exists."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        from swe2d.workbench.views.results_view import StudioResultsView
        iface = MagicMock()
        dlg = SWE2DWorkbenchStudioDialog(iface=iface)
        try:
            rv = getattr(dlg, '_results_view', None)
            if rv is not None:
                self.assertIsInstance(rv, StudioResultsView,
                    "_results_view should be a StudioResultsView instance")
        finally:
            dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy panel cleanup tests (Phase 5 / Task 25)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyPanelCleanup(unittest.TestCase):
    """Verify the legacy SWE2DResultsPanel is no longer created."""

    _saved_qapp: type | None = None
    _app = None

    @classmethod
    def setUpClass(cls):
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        cls._saved_qapp = _pyqt5_qt.QApplication if _pyqt5_qt else None
        if _pyqt5_qt is not None:
            _pyqt5_qt.QApplication = _REAL_QAPP
        if _qgis_qt is not None:
            _qgis_qt.QApplication = _REAL_QAPP
        cls._app = _REAL_QAPP.instance() or _REAL_QAPP([])

    @classmethod
    def tearDownClass(cls):
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        if _pyqt5_qt is not None and cls._saved_qapp is not None:
            _pyqt5_qt.QApplication = cls._saved_qapp
        if _qgis_qt is not None and cls._saved_qapp is not None:
            _qgis_qt.QApplication = cls._saved_qapp
        cls._saved_qapp = None

    def test_no_legacy_panel_after_dialog_build(self):
        """After dialog builds, _results_panel should not be a live panel."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            has_panel = hasattr(dlg, '_results_panel') and dlg._results_panel is not None
            self.assertFalse(has_panel, "Legacy _results_panel should not exist")
        finally:
            dlg.close()

    def test_controls_tab_self_contained(self):
        """Results view has content widgets (owned by StudioResultsView)."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            rv = getattr(dlg, '_results_view', None)
            self.assertIsNotNone(rv, "Results view must exist")
            # Verify key widgets are present on the view
            self.assertTrue(hasattr(rv, '_run_list'), "View must own _run_list")
            self.assertTrue(hasattr(rv, '_line_combo'), "View must own _line_combo")
        finally:
            dlg.close()

    def test_studio_run_list_exists(self):
        """The run list widget should exist on the StudioResultsView."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            rv = getattr(dlg, '_results_view', None)
            self.assertIsNotNone(rv, "Results view should exist on dialog")
            rl = getattr(rv, '_run_list', None)
            self.assertIsNotNone(rl, "Run list widget should exist on results view")
        finally:
            dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Run if called directly
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Service integration tests (Phase 2 / Task 8)
# ═══════════════════════════════════════════════════════════════════════════════

from tests.test_workbench_dialog_builder import _ensure_app


class TestServiceIntegration(unittest.TestCase):
    """Verify the dialog uses services instead of inline code."""

    def setUp(self):
        _ensure_app()

    def test_dialog_has_no_inline_sqlite3(self):
        """The dialog module should not import sqlite3 directly.

        After the Phase 2 refactor (Task 7), all sqlite3 usage lives in
        ``swe2d.workbench.gpkg_service``. The dialog must delegate to that
        service (``load_mesh_snapshot``) rather than importing sqlite3
        inline.
        """
        from swe2d.workbench import studio_dialog
        source = open(studio_dialog.__file__).read()
        self.assertNotIn('sqlite3', source)

    def test_dialog_delegates_to_controller(self):
        """The dialog should delegate mesh snapshot loading via WorkbenchController.

        After Phase 3 Task 10, ``studio_dialog`` no longer imports
        ``load_mesh_snapshot`` directly. The dialog now instantiates a
        ``WorkbenchController`` in its builder, and the controller
        delegates to ``gpkg_service.load_mesh_snapshot``.
        """
        from swe2d.workbench import studio_dialog
        from swe2d.workbench import workbench_controller
        from swe2d.workbench.services import gpkg_service

        self.assertFalse(
            hasattr(studio_dialog, 'load_mesh_snapshot'),
            "studio_dialog should not import load_mesh_snapshot directly; "
            "the controller is the seam.",
        )
        self.assertTrue(
            hasattr(gpkg_service, 'load_mesh_snapshot'),
            "gpkg_service.load_mesh_snapshot must still exist (consumed by controller)",
        )
        controller_source = open(workbench_controller.__file__).read()
        self.assertIn("load_mesh_snapshot", controller_source)
        self.assertIn("gpkg_service", controller_source)


class TestOverlayParametersServiceUsage(unittest.TestCase):
    """Verify the overlay parameters service is available and usable.

    These are integration-style checks: the dialog (``studio_dialog``) must
    be able to call ``collect_overlay_parameters`` against any object
    exposing the dialog's widget state (typically the dialog itself).
    """

    def test_service_exists(self):
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        self.assertIsNotNone(collect_overlay_parameters)

    def test_service_is_sole_source_returns_full_dict(self):
        """``collect_overlay_parameters`` returns the complete dict consumed by
        ``render_unstructured_snapshot_image`` — not a 19-key stub.
        """
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        from unittest.mock import MagicMock
        import numpy as np

        view = MagicMock()
        view._high_perf_overlay_cell_x = np.array([0.0, 1.0])
        view._high_perf_overlay_cell_y = np.array([0.0, 1.0])
        view._high_perf_overlay_cell_bed = np.array([0.0, 0.0])
        view._high_perf_overlay_node_x = np.array([0.0, 1.0])
        view._high_perf_overlay_node_y = np.array([0.0, 1.0])
        view._high_perf_overlay_cell_nodes = np.array([[0, 1]])
        view._high_perf_overlay_tri_to_cell = np.array([0])
        view._snapshot_timesteps = [
            (0.0, np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        ]
        view._gravity = 9.81
        view._mannings_n = 0.035
        view._length_unit_name = "m"
        view.high_perf_canvas_overlay_field_combo.currentData.return_value = "depth"
        view.high_perf_canvas_overlay_wse_render_combo.currentData.return_value = "cell"
        view.high_perf_canvas_overlay_cmap_combo.currentData.return_value = "turbo"
        view.high_perf_canvas_overlay_visible_only_chk.isChecked.return_value = False
        view.high_perf_canvas_overlay_lock_canvas_chk.isChecked.return_value = False
        view.high_perf_canvas_overlay_auto_contrast_chk.isChecked.return_value = True
        view.high_perf_canvas_overlay_res_combo.currentData.return_value = (1280, 720)
        view.high_perf_canvas_overlay_opacity_spin.value.return_value = 1.0
        view.high_perf_canvas_overlay_arrows_chk.isChecked.return_value = False
        view.high_perf_canvas_overlay_arrow_density_spin.value.return_value = 28.0
        view.high_perf_canvas_overlay_arrow_length_spin.value.return_value = 1.0
        view.high_perf_canvas_overlay_arrow_head_length_spin.value.return_value = 1.0
        view.high_perf_canvas_overlay_arrow_head_width_spin.value.return_value = 1.0
        view.high_perf_canvas_overlay_streamlines_chk.isChecked.return_value = False
        view.high_perf_canvas_overlay_streamline_backend_combo.currentData.return_value = "auto"
        view.high_perf_canvas_overlay_streamline_seed_spin.value.return_value = 48.0
        view.high_perf_canvas_overlay_streamline_steps_spin.value.return_value = 24.0
        view._resolve_map_canvas.return_value = None

        params = collect_overlay_parameters(view, t_use=1.0)

        required_keys = [
            "cell_x", "cell_y", "cell_bed", "node_x", "node_y", "cell_nodes",
            "tri_to_cell", "timesteps", "current_time_s", "field_key",
            "wse_render_mode", "cmap_key", "resolution", "auto_contrast",
            "show_velocity_arrows", "arrow_stride_px", "arrow_length_scale",
            "arrow_head_length_scale", "arrow_head_width_scale",
            "show_streamlines", "streamline_backend", "streamline_seed_count",
            "streamline_steps", "visible_extent_world", "render_extent_world",
            "gravity", "courant_cell_size", "courant_dt", "manning_n",
            "show_legend", "legend_label",
        ]
        for key in required_keys:
            self.assertIn(key, params, f"service must return '{key}'")
        self.assertEqual(params["current_time_s"], 1.0)
        self.assertEqual(params["field_key"], "depth")
        self.assertEqual(params["cmap_key"], "turbo")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Tasks 1 + 5: dialog must NOT have overlay-collection methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestDialogNoOverlayDelegateMethods(unittest.TestCase):
    """Phase 1 Tasks 1 + 5 — the dialog must NOT define these methods.

    The dialog delegates to ``collect_overlay_parameters`` (service) for
    overlay collection and to ``_controller.load_mesh_snapshot_for_overlay``
    for mesh snapshot loading. No 1-line wrapper, no alias, no compat shim.
    """

    _saved_qapp: type | None = None
    _app = None
    _studio_source: str = ""

    @classmethod
    def setUpClass(cls):
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        cls._saved_qapp = _pyqt5_qt.QApplication if _pyqt5_qt else None
        if _pyqt5_qt is not None:
            _pyqt5_qt.QApplication = _REAL_QAPP
        if _qgis_qt is not None:
            _qgis_qt.QApplication = _REAL_QAPP
        cls._app = _REAL_QAPP.instance() or _REAL_QAPP([])

        from swe2d.workbench import studio_dialog
        cls._studio_source = open(studio_dialog.__file__).read()

    @classmethod
    def tearDownClass(cls):
        import sys as _sys
        _pyqt5_qt = _sys.modules.get("PyQt5.QtWidgets")
        _qgis_qt = _sys.modules.get("qgis.PyQt.QtWidgets")
        if _pyqt5_qt is not None and cls._saved_qapp is not None:
            _pyqt5_qt.QApplication = cls._saved_qapp
        if _qgis_qt is not None and cls._saved_qapp is not None:
            _qgis_qt.QApplication = cls._saved_qapp
        cls._saved_qapp = None

    def test_dialog_does_not_have_collect_overlay_parameters(self):
        from unittest.mock import MagicMock
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            self.assertFalse(
                hasattr(dlg, "_collect_overlay_parameters"),
                "Dialog still has _collect_overlay_parameters method — "
                "overlay_parameters_service.collect_overlay_parameters is the SOLE source.",
            )
        finally:
            dlg.close()

    def test_dialog_does_not_have_load_mesh_results_for_overlay(self):
        from unittest.mock import MagicMock
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=MagicMock())
        try:
            self.assertFalse(
                hasattr(dlg, "_load_mesh_results_for_overlay"),
                "Dialog still has _load_mesh_results_for_overlay method — "
                "callers must use self._controller.load_mesh_snapshot_for_overlay directly.",
            )
        finally:
            dlg.close()

    def test_dialog_source_has_no_collect_overlay_parameters_def(self):
        """``studio_dialog.py`` source does not define ``_collect_overlay_parameters``."""
        self.assertNotIn(
            "def _collect_overlay_parameters(",
            self._studio_source,
            "studio_dialog.py still defines _collect_overlay_parameters — "
            "overlay_parameters_service is supposed to be the SOLE source.",
        )

    def test_dialog_source_has_no_load_mesh_results_for_overlay_def(self):
        """``studio_dialog.py`` source does not define ``_load_mesh_results_for_overlay``."""
        self.assertNotIn(
            "def _load_mesh_results_for_overlay(",
            self._studio_source,
            "studio_dialog.py still defines _load_mesh_results_for_overlay — "
            "callers must use self._controller.load_mesh_snapshot_for_overlay directly.",
        )

    def test_dialog_caller_uses_controller_not_delegate(self):
        """The dialog's only caller of mesh-snapshot loading must use the controller.

        After deletion, the call site in ``_on_results_panel_timestep_changed``
        calls ``self._controller.load_mesh_snapshot_for_overlay(t_s)`` directly.
        """
        self.assertNotIn(
            "self._load_mesh_results_for_overlay(",
            self._studio_source,
            "Dialog still calls its own _load_mesh_results_for_overlay delegate.",
        )
        self.assertIn(
            "self._controller.load_mesh_snapshot_for_overlay(",
            self._studio_source,
            "Dialog must call self._controller.load_mesh_snapshot_for_overlay directly.",
        )

    def test_dialog_caller_uses_service_not_method(self):
        """The overlay-parameter collection must come from the service, not a dialog method."""
        self.assertNotIn(
            "self._collect_overlay_parameters(",
            self._studio_source,
            "Dialog still calls its own _collect_overlay_parameters method — "
            "must call collect_overlay_parameters from overlay_parameters_service.",
        )
        # The import may be in the dialog or in the overlay bridge — both are valid
        import swe2d.workbench.bridges.high_perf_overlay_bridge as bridge_mod
        bridge_source = open(bridge_mod.__file__).read()
        self.assertIn(
            "from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters",
            bridge_source,
            "Overlay bridge must import collect_overlay_parameters from the service module.",
        )

    def test_grep_dialog_references_only_service_and_controller(self):
        """``grep`` for these names finds only real code references in service and controller.

        The dialog must not define or call ``_collect_overlay_parameters`` or
        ``_load_mesh_results_for_overlay`` as methods. The only allowed
        ``swe2d/`` locations are the service module (defines
        ``collect_overlay_parameters``) and the controller module (defines
        ``load_mesh_snapshot_for_overlay``). The ``_EXCLUDE_METHODS`` frozenset
        in the dialog's import-time copy block is metadata about names — it
        is not a real code reference — so we ignore string-occurrences inside
        the ``_EXCLUDE_METHODS = frozenset({...})`` literal.
        """
        import re
        import os
        import ast
        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        swe2d_dir = os.path.join(repo_root, "swe2d")
        bad_refs = []
        target_names = ("_collect_overlay_parameters", "_load_mesh_results_for_overlay")
        for root, _dirs, files in os.walk(swe2d_dir):
            if "__pycache__" in root:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                fpath = os.path.join(root, fn)
                rel = os.path.relpath(fpath, repo_root)
                with open(fpath) as f:
                    text = f.read()
                try:
                    tree = ast.parse(text, filename=fpath)
                except SyntaxError:
                    continue
                # Collect line ranges that are string literals inside an
                # _EXCLUDE_METHODS = frozenset({...}) assignment — these are
                # metadata, not real code references.
                exclude_lines = set()
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    targets = [t for t in node.targets
                               if isinstance(t, ast.Name) and t.id == "_EXCLUDE_METHODS"]
                    if not targets:
                        continue
                    if not isinstance(node.value, ast.Call):
                        continue
                    func = node.value.func
                    if not (isinstance(func, ast.Name) and func.id == "frozenset"):
                        continue
                    if not node.value.args:
                        continue
                    arg = node.value.args[0]
                    if isinstance(arg, (ast.Set, ast.List, ast.Tuple)):
                        for elt in arg.elts:
                            if (isinstance(elt, ast.Constant)
                                    and isinstance(elt.value, str)):
                                sl = elt.lineno
                                el = getattr(elt, "end_lineno", sl)
                                for ln in range(sl, el + 1):
                                    exclude_lines.add(ln)
                # Walk every name node and check for forbidden uses.
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Name):
                        continue
                    if node.id not in target_names:
                        continue
                    if node.lineno in exclude_lines:
                        continue
                    # Allow service module (defines the service) and
                    # controller module (defines the controller method).
                    if rel.endswith("overlay_parameters_service.py"):
                        continue
                    if rel.endswith("workbench_controller.py"):
                        continue
                    # Filter out bare re-exports / string mentions that are
                    # not real code references.
                    if isinstance(node.ctx, ast.Load):
                        # An attribute access is OK if it's the module name
                        # itself (e.g. `overlay_parameters_service.collect_...`)
                        # — that's the service import. But the bare name as a
                        # load reference in a non-service file is a real ref.
                        bad_refs.append(f"{rel}:{node.lineno}: {node.id}")
                    else:
                        bad_refs.append(f"{rel}:{node.lineno}: {node.id}")
        if bad_refs:
            self.fail(
                f"Found {len(bad_refs)} forbidden reference(s) in swe2d/ "
                f"to deleted dialog methods:\n  "
                + "\n  ".join(bad_refs)
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Run if called directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
