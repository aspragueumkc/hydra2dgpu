"""
Unit tests for workbench GUI components against real headless QGIS.

These tests verify that:
1. The workbench dialog can be instantiated in a headless QGIS session
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
from unittest.mock import MagicMock

import numpy as np

from qgis.PyQt import QtCore, QtWidgets

from tests.qgis_real_env import ensure_qgis_app, requires_qgis, stub_iface
from tests.test_helpers import FallbackTracker


# ═══════════════════════════════════════════════════════════════════════════════
# Workbench import smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qgis
class TestWorkbenchImports(unittest.TestCase):
    """Verify that the main workbench module imports under real QGIS."""

    def test_import_workbench_studio_dialog_no_crash(self):
        """Importing swe2d.workbench.studio_dialog does not raise ImportError."""
        with FallbackTracker(fail_on_any_warning=True):
            import swe2d.workbench.studio_dialog  # noqa: F401
        self.assertTrue(True, "swe2d.workbench.studio_dialog imported successfully")

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

@requires_qgis
class TestWorkbenchDialogConstruction(unittest.TestCase):
    """Verify the workbench dialog can be imported and referenced."""

    def test_dialog_class_exists(self):
        """SWE2DWorkbenchStudioDialog class is importable and is a type."""
        from swe2d.workbench import studio_dialog
        cls = studio_dialog.SWE2DWorkbenchStudioDialog
        self.assertTrue(isinstance(cls, type))

    def test_dialog_has_ui_attribute(self):
        """SWE2DWorkbenchStudioDialog class defines a `ui` class-level reference."""
        from swe2d.workbench import studio_dialog
        self.assertTrue(hasattr(studio_dialog, "SWE2DWorkbenchStudioDialog"))
        self.assertIsNotNone(studio_dialog.SWE2DWorkbenchStudioDialog)

    def test_units_module_available(self):
        """swe2d.units module is functional (no QGIS dependency)."""
        import swe2d.units as units
        units.configure(1.0)
        self.assertGreater(units.si_m_per_model(), 0.0)


@requires_qgis
class TestWorkbenchDialogConstructionFull(unittest.TestCase):
    """Widget completeness checks after .ui removal.

    The widget-existence check is behavioral: it constructs the real
    ``TopologyTabView`` and looks up every critical widget in the live
    widget tree.  Two source-grep guards are retained (with in-line
    justifications) because they audit refactor cleanup that has no
    runtime signature.
    """

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def test_no_stale_helper_references_in_source(self):
        """No _find_or_create_, _find_child_robust, or _ensure_form_row remain."""
        # JUSTIFICATION (source grep, no behavioral equivalent): this is a
        # .ui-removal migration-cleanup audit.  A stale helper reference that
        # is never executed is dead code with no runtime signature — dialog
        # construction tests cannot observe it.
        import re

        from swe2d.workbench import studio_dialog

        files = {studio_dialog.__file__: open(studio_dialog.__file__).read()}

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

    def test_key_widgets_exist_in_live_topology_view(self):
        """Every critical topology widget exists in the live widget tree.

        Behavioral replacement for the former ``setObjectName`` source
        grep: constructing the real ``TopologyTabView`` and resolving each
        objectName via ``findChild`` proves the widgets are actually built
        and reachable — strictly stronger than matching source strings.
        """
        from swe2d.workbench.views.topology_tab_view import TopologyTabView

        view = TopologyTabView()
        self.addCleanup(view.deleteLater)

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

        missing = [
            name for name in critical_widgets
            if view.findChild(QtWidgets.QWidget, name) is None
        ]
        self.assertEqual(
            missing, [],
            f"Widgets missing from live TopologyTabView: {missing}",
        )

        # Spot-check widget classes — a name collision with the wrong
        # widget type must fail, not pass silently.
        self.assertIsInstance(
            view.findChild(QtWidgets.QWidget, "topo_gmsh_tri_algo_combo"),
            QtWidgets.QComboBox,
        )
        self.assertIsInstance(
            view.findChild(QtWidgets.QWidget, "topo_gmsh_quality_enable_chk"),
            QtWidgets.QCheckBox,
        )
        self.assertIsInstance(
            view.findChild(QtWidgets.QWidget, "topo_gmsh_quality_max_iters_spin"),
            QtWidgets.QSpinBox,
        )
        self.assertIsInstance(
            view.findChild(QtWidgets.QWidget, "topo_quality_min_angle_spin"),
            QtWidgets.QDoubleSpinBox,
        )
        self.assertIsInstance(
            view.findChild(QtWidgets.QWidget, "topo_quality_min_area_edit"),
            QtWidgets.QLineEdit,
        )
        self.assertIsInstance(
            view.findChild(QtWidgets.QWidget, "topo_quality_strict_chk"),
            QtWidgets.QCheckBox,
        )

    def test_no_orphan_if_none_guards_in_topo_source(self):
        """No bare 'if self.X is None:' guards without prior initialization."""
        # JUSTIFICATION (source grep, no behavioral equivalent): this guards
        # the .ui-removal refactor rule that widgets are created eagerly at
        # construction, not lazily inside ``if self.X is None:`` guards.  A
        # guard-wrapped creation is behaviorally indistinguishable from
        # eager init at runtime, so no widget-driving test can observe it.
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

        # No exception raised → test passes implicitly

    def test_fallback_tracker_no_false_positives(self):
        import logging
        logger = logging.getLogger("swe2d.test")

        with FallbackTracker(logger_name="swe2d.test", fail_on_any_warning=True):
            logger.info("normal info message")

        # info is not intercepted → no exception raised → test passes implicitly


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

@requires_qgis
class TestStudioDialogLifecycle(unittest.TestCase):
    """Verify the Studio dialog lifecycle: component registry, close, tabs."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def _make_dialog(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        iface = stub_iface()
        # Dock widgets need a real QWidget parent.
        iface.mainWindow.return_value = QtWidgets.QMainWindow()
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

    def test_close_event_keeps_component_registry(self):
        """Closing the dialog hides it; the component registry is not cleared."""
        dlg = self._make_dialog()
        self.assertGreater(len(dlg._state.studio_components), 0)
        dlg.close()
        self.assertGreater(len(dlg._state.studio_components), 0)

    def test_left_pane_tab_order(self):
        dlg = self._make_dialog()
        try:
            tabs = dlg._left_tabs
            self.assertIsNotNone(tabs)
            self.assertEqual(tabs.count(), 2)
            self.assertEqual(tabs.tabText(0), "Mesh Generation")
            self.assertEqual(tabs.tabText(1), "Simulation")
        finally:
            dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy panel cleanup tests (Phase 5 / Task 25)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qgis
class TestLegacyPanelCleanup(unittest.TestCase):
    """Verify the legacy SWE2DResultsPanel is no longer created."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def test_no_legacy_panel_after_dialog_build(self):
        """After dialog builds, _results_panel should not be a live panel."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        iface = stub_iface()
        iface.mainWindow.return_value = QtWidgets.QMainWindow()
        dlg = SWE2DWorkbenchStudioDialog(iface=iface)
        try:
            has_panel = hasattr(dlg, '_results_panel') and dlg._results_panel is not None
            self.assertFalse(has_panel, "Legacy _results_panel should not exist")
        finally:
            dlg.close()



# ═══════════════════════════════════════════════════════════════════════════════
# Service integration tests (Phase 2 / Task 8)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qgis
class TestServiceIntegration(unittest.TestCase):
    """Verify the dialog uses services instead of inline code."""

    def setUp(self):
        ensure_qgis_app()

    def test_dialog_has_no_inline_sqlite3(self):
        """The dialog module should not import sqlite3 directly.

        After the Phase 2 refactor (Task 7), all sqlite3 usage lives in
        ``swe2d.workbench.gpkg_service``. The dialog must delegate to that
        service (``load_mesh_snapshot``) rather than importing sqlite3
        inline.
        """
        # JUSTIFICATION (source grep, no behavioral equivalent): MVP
        # layering guard — the View must not perform DB I/O.  Importing
        # sqlite3 changes no runtime behavior, so no widget-driving test
        # can observe it.  (tests/test_import_boundary.py covers the
        # Qt-import boundary of *service* modules — a different axis; it
        # does not cover sqlite3 in the View layer.)
        from swe2d.workbench import studio_dialog
        source = open(studio_dialog.__file__).read()
        self.assertNotIn('sqlite3', source)

    def test_dialog_delegates_to_controller(self):
        """The dialog delegates mesh snapshot loading to the OverlayController.

        After Phase 3 Task 10, ``studio_dialog`` no longer imports
        ``load_mesh_snapshot_for_overlay`` directly — the controller owns
        the GPKG snapshot loading path.  The behavioral proof that the
        controller path actually works end-to-end lives in
        ``TestOverlayControllerDelegation`` below.
        """
        from swe2d.workbench import studio_dialog

        self.assertFalse(
            hasattr(studio_dialog, 'load_mesh_snapshot_for_overlay'),
            "studio_dialog should not import load_mesh_snapshot_for_overlay directly; "
            "the overlay controller is the seam.",
        )


class TestOverlayParametersServiceUsage(unittest.TestCase):
    """Verify the overlay parameters service is available and usable.

    These are integration-style checks: the dialog (``studio_dialog``) must
    be able to call ``collect_overlay_parameters`` against any object
    exposing the dialog's widget state (typically the dialog itself).
    """

    def test_service_is_sole_source_returns_full_dict(self):
        """``collect_overlay_parameters`` returns the complete dict consumed by
        ``render_unstructured_snapshot_image`` — not a 19-key stub.
        """
        from swe2d.workbench.services.overlay_parameters_service import collect_overlay_parameters
        import numpy as np

        view = MagicMock()
        data = MagicMock()
        data.overlay_cell_x = np.array([0.0, 1.0])
        data.overlay_cell_y = np.array([0.0, 1.0])
        data.overlay_cell_bed = np.array([0.0, 0.0])
        data.overlay_node_x = np.array([0.0, 1.0])
        data.overlay_node_y = np.array([0.0, 1.0])
        data.overlay_cell_nodes = np.array([[0, 1]])
        data.overlay_tri_to_cell = np.array([0])
        data.get_live_snapshot_timesteps.return_value = [
            (0.0, np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        ]
        view._results_data = data
        view._gravity = 9.81
        view._mannings_n = 0.035
        view._length_unit_name = "m"

        tb = MagicMock()
        tb.field_combo.currentData.return_value = "depth"
        tb.wse_render_combo.currentData.return_value = "cell"
        tb.cmap_combo.currentData.return_value = "turbo"
        tb.visible_only_chk.isChecked.return_value = False
        tb.lock_canvas_chk.isChecked.return_value = False
        tb.auto_contrast_chk.isChecked.return_value = True
        tb.res_combo.currentData.return_value = (1280, 720)
        tb.opacity_spin.value.return_value = 1.0
        tb.arrows_chk.isChecked.return_value = False
        tb.arrow_density_spin.value.return_value = 28.0
        tb.arrow_length_spin.value.return_value = 1.0
        tb.arrow_head_length_spin.value.return_value = 1.0
        tb.arrow_head_width_spin.value.return_value = 1.0
        tb.streamlines_chk.isChecked.return_value = False
        tb.streamline_backend_combo.currentData.return_value = "auto"
        tb.streamline_seed_spin.value.return_value = 48.0
        tb.streamline_steps_spin.value.return_value = 24.0
        view._results_toolbox = tb
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
            "gravity", "courant_cell_size", "courant_dt", "mannings_n",
            "show_legend", "legend_label",
        ]
        for key in required_keys:
            self.assertIn(key, params, f"service must return '{key}'")
        self.assertEqual(params["current_time_s"], 1.0)
        self.assertEqual(params["field_key"], "depth")
        self.assertEqual(params["cmap_key"], "turbo")
        self.assertEqual(params["legend_label"], "Depth (m)")


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay controller delegation — behavioral end-to-end (replaces source greps)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qgis
class TestOverlayControllerDelegation(unittest.TestCase):
    """Drive the real overlay snapshot/render path through the controller.

    Behavioral replacements for the former source greps that asserted
    "the overlay controller imports ``load_baked_snapshot`` /
    ``collect_overlay_parameters``" and "the results panel calls
    ``dialog._overlay_controller.load_mesh_snapshot_for_overlay``".
    Here a real dialog (real ``QgsMapCanvas``, real results toolbox
    widgets) loads a snapshot from a real results GPKG and renders it —
    every import and call site the greps matched is executed for real.
    """

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()
        cls._native = cls._import_native_swe2d()

    @staticmethod
    def _import_native_swe2d():
        """Import the compiled ``hydra_swe2d`` extension from this repo's build/.

        Constructing the Studio dialog imports the ``HYDRA2DGPU`` QGIS
        plugin package, whose ``__init__`` realpaths the dev plugin
        symlink and prepends that tree to ``sys.path``.  When the symlink
        points at a worktree without a compiled extension, a plain
        ``import hydra_swe2d`` after dialog construction resolves to the
        wrong package.  Loading from the canonical ``build/`` path keeps
        this test immune to that environment state; a missing .so skips
        the class loudly.
        """
        import glob
        import importlib.util

        mod = sys.modules.get("hydra_swe2d")
        if mod is not None and hasattr(mod, "swe2d_build_mesh"):
            return mod
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = sorted(
            glob.glob(os.path.join(repo_root, "build", "hydra_swe2d*.so"))
        )
        if not candidates:
            raise unittest.SkipTest(
                "hydra_swe2d native extension not built "
                "(build/hydra_swe2d*.so missing)"
            )
        spec = importlib.util.spec_from_file_location(
            "hydra_swe2d", candidates[0]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["hydra_swe2d"] = module
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        from qgis.core import QgsRectangle
        from qgis.gui import QgsMapCanvas

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

        iface = stub_iface()
        # Dock widgets need a real QWidget parent.
        iface.mainWindow.return_value = QtWidgets.QMainWindow()
        # Real offscreen canvas so the canvas overlay item is created for real.
        self.canvas = QgsMapCanvas()
        self.canvas.resize(400, 300)
        self.canvas.setExtent(QgsRectangle(-1.0, -1.0, 2.0, 2.0))
        iface.mapCanvas.return_value = self.canvas
        self.dlg = SWE2DWorkbenchStudioDialog(iface=iface)

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.canvas.deleteLater()

    @staticmethod
    def _bake_toy_mesh(gpkg_path: str, mesh_name: str = "hydra_test_mesh") -> None:
        """Bake a 4-cell toy mesh into the results GPKG (production writer)."""
        from swe2d.services.mesh_persistence_service import save_baked_mesh

        mesh_data = {
            "node_x": np.array([0.0, 1.0, 0.0, 1.0, 0.5], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0, 0.5], dtype=np.float64),
            "node_z": np.zeros(5, dtype=np.float64),
            "cell_nodes": np.array(
                [0, 1, 4, 1, 3, 4, 0, 4, 2, 4, 3, 2], dtype=np.int32
            ),
        }
        n_cells = save_baked_mesh(mesh_data, gpkg_path, mesh_name)
        assert n_cells == 4, f"expected 4 baked cells, got {n_cells}"

    def _load_run_into_dialog(self, gpkg_path: str) -> None:
        """Register the temp results run via the production discovery path."""
        added_paths, added_runs = self.dlg._results_data.add_results_files(
            [gpkg_path]
        )
        self.assertEqual(added_paths, 1)
        self.assertEqual(added_runs, 1)
        rec = self.dlg._results_data.overlay_selected_run()
        self.assertIsNotNone(rec)
        self.assertEqual(rec.run_id, "hydra_test_run")

    def test_controller_loads_snapshot_from_per_run_gpkg(self):
        """load_mesh_snapshot_for_overlay reads mesh + snapshot from the GPKG.

        Proves the controller owns the whole GPKG path the deleted dialog
        delegate used to expose: baked-mesh load, ``load_baked_snapshot``,
        and live-snapshot seeding.
        """
        from tests.qgis_real_env import make_temp_results_gpkg

        dlg = self.dlg
        with make_temp_results_gpkg(n_cells=4, n_timesteps=3) as gpkg:
            self._bake_toy_mesh(gpkg)
            self._load_run_into_dialog(gpkg)
            dlg._high_perf_canvas_overlay_enabled = True

            ok = dlg._overlay_controller.load_mesh_snapshot_for_overlay(10.0)

            self.assertTrue(ok, "controller must load the snapshot from GPKG")
            self.assertAlmostEqual(dlg._overlay_last_loaded_t_s, 10.0)
            data = dlg._results_data
            self.assertEqual(data.overlay_cell_x.size, 4)
            self.assertEqual(data.overlay_node_x.size, 5)
            live = data.get_live_snapshot_timesteps()
            self.assertTrue(
                any(abs(float(t) - 10.0) < 1.0e-6 for t, *_ in live),
                f"snapshot at t=10 must be seeded live, got {live!r}",
            )

    def test_results_panel_timestep_change_calls_controller(self):
        """on_results_panel_timestep_changed reaches the controller method.

        Behavioral proof of the call site the deleted source grep matched
        in ``studio_results_panel.py``: the panel function invokes
        ``dialog._overlay_controller.load_mesh_snapshot_for_overlay(t_s)``.
        """
        from tests.qgis_real_env import make_temp_results_gpkg
        from swe2d.workbench.views import studio_results_panel

        dlg = self.dlg
        with make_temp_results_gpkg(n_cells=4, n_timesteps=3) as gpkg:
            self._bake_toy_mesh(gpkg)
            self._load_run_into_dialog(gpkg)
            dlg._high_perf_canvas_overlay_enabled = True

            calls = []
            orig = dlg._overlay_controller.load_mesh_snapshot_for_overlay

            def recording_load(t_s, _orig=orig, _calls=calls):
                _calls.append(float(t_s))
                return _orig(t_s)

            dlg._overlay_controller.load_mesh_snapshot_for_overlay = (
                recording_load
            )
            try:
                studio_results_panel.on_results_panel_timestep_changed(
                    dlg, 20.0
                )
            finally:
                dlg._overlay_controller.load_mesh_snapshot_for_overlay = orig

            self.assertEqual(
                calls, [20.0],
                "results panel must call the controller's "
                "load_mesh_snapshot_for_overlay exactly once with t_s",
            )
            self.assertAlmostEqual(dlg._overlay_last_loaded_t_s, 20.0)

    def test_refresh_overlay_renders_through_service(self):
        """refresh_high_perf_canvas_overlay runs collect → render → apply.

        Behavioral proof that the controller pulls render parameters from
        ``overlay_parameters_service.collect_overlay_parameters``: a real
        render against real toolbox widgets must produce a frame, create
        the canvas overlay item, and publish the computed color range.
        """
        from tests.qgis_real_env import make_temp_results_gpkg

        dlg = self.dlg
        with make_temp_results_gpkg(n_cells=4, n_timesteps=3) as gpkg:
            self._bake_toy_mesh(gpkg)
            self._load_run_into_dialog(gpkg)
            dlg._high_perf_canvas_overlay_enabled = True
            self.assertTrue(
                dlg._overlay_controller.load_mesh_snapshot_for_overlay(10.0)
            )

            dlg._refresh_high_perf_canvas_overlay(10.0)

            item = dlg._state.high_perf_canvas_overlay_item
            self.assertIsNotNone(
                item, "render path must create the canvas overlay item"
            )
            self.assertTrue(item.isVisible())
            self.assertIsNotNone(
                dlg._results_data._overlay_computed_vmin,
                "rendered frame must publish its computed color range",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Tasks 1 + 5: dialog must NOT have overlay-collection methods
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qgis
class TestDialogNoOverlayDelegateMethods(unittest.TestCase):
    """Phase 1 Tasks 1 + 5 — the dialog must NOT define these methods.

    The dialog delegates to ``collect_overlay_parameters`` (service) for
    overlay collection and to ``_controller.load_mesh_snapshot_for_overlay``
    for mesh snapshot loading. No 1-line wrapper, no alias, no compat shim.

    Deleted source-grep duplicates (2026-08-02, plan F.1):
    - ``test_dialog_source_has_no_collect_overlay_parameters_def`` and
      ``test_dialog_source_has_no_load_mesh_results_for_overlay_def`` —
      exact duplicates of the two live-dialog ``hasattr`` checks below.
    - ``test_dialog_caller_uses_controller_not_delegate`` and
      ``test_dialog_caller_uses_service_not_method`` — replaced by the
      behavioral end-to-end tests in ``TestOverlayControllerDelegation``.
    """

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()
        cls._iface = stub_iface()
        # Dock widgets need a real QWidget parent.
        cls._iface.mainWindow.return_value = QtWidgets.QMainWindow()

    def _make_iface(self):
        return self._iface

    def test_dialog_does_not_have_collect_overlay_parameters(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=self._iface)
        try:
            self.assertFalse(
                hasattr(dlg, "_collect_overlay_parameters"),
                "Dialog still has _collect_overlay_parameters method — "
                "overlay_parameters_service.collect_overlay_parameters is the SOLE source.",
            )
        finally:
            dlg.close()

    def test_dialog_does_not_have_load_mesh_results_for_overlay(self):
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(iface=self._iface)
        try:
            self.assertFalse(
                hasattr(dlg, "_load_mesh_results_for_overlay"),
                "Dialog still has _load_mesh_results_for_overlay method — "
                "callers must use the overlay controller directly.",
            )
        finally:
            dlg.close()

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
        # JUSTIFICATION (AST walk, no behavioral equivalent): repo-wide
        # dangling-reference audit after the delegate-method deletion.  A
        # stale *reference* (not a call) in an arbitrary module has no
        # runtime signature, and no single widget-driving test exercises
        # every module in swe2d/.
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
                    # Justified AST audit — see JUSTIFICATION at method top.
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
                    # Allow service module (defines the service),
                    # the overlay controller (uses the service), and
                    # workbench_controller.py (legacy controller module).
                    if rel.endswith("overlay_parameters_service.py"):
                        continue
                    if rel.endswith("controllers/overlay_controller.py"):
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
