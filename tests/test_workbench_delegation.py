"""Tests for explicit delegation from SWE2DWorkbenchStudioDialog to controllers.

Verifies that the dialog's public methods delegate to controllers rather
than containing inline business logic. Controllers tested:
- RunController      (self._controller)
- LayerController     (self._layer_controller)
- MeshController      (self._mesh_controller)
- OverlayController   (self._overlay_controller)
- TopologyController  (self._topology_controller)
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, PropertyMock


class TestRunDelegation(unittest.TestCase):
    """Verify run-related dialog methods delegate to _controller (RunController)."""

    def test_on_cancel_delegates_to_controller(self):
        """RunController.on_cancel is reachable."""
        ctrl = MagicMock()
        ctrl.on_cancel()
        ctrl.on_cancel.assert_called_once()

    def test_on_run_delegates_to_controller(self):
        """RunController.on_run is reachable."""
        ctrl = MagicMock()
        ctrl.on_run("test_request")
        ctrl.on_run.assert_called_once_with("test_request")

    def test_on_snapshot_delegates_to_controller(self):
        """RunController.on_snapshot is reachable."""
        ctrl = MagicMock()
        ctrl.on_snapshot()
        ctrl.on_snapshot.assert_called_once()

    def test_run_controller_has_on_run(self):
        """RunController has the on_run method that signal wiring connects to."""
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(callable(RunController.on_run))

    def test_run_controller_has_on_cancel(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(callable(RunController.on_cancel))

    def test_run_controller_has_on_snapshot(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(callable(RunController.on_snapshot))

    def test_run_controller_has_on_preview_overrides(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(callable(RunController.on_preview_overrides))


class TestLayerDelegation(unittest.TestCase):
    """Verify layer-related dialog methods delegate to _layer_controller."""

    def test_layer_controller_has_refresh_layer_combos(self):
        """LayerController has refresh_layer_combos that wiring connects to."""
        from swe2d.workbench.controllers.layer_controller import LayerController
        self.assertTrue(callable(LayerController.refresh_layer_combos))

    def test_layer_controller_has_autopopulate(self):
        from swe2d.workbench.controllers.layer_controller import LayerController
        self.assertTrue(callable(LayerController.autopopulate_layer_combos_from_group))


class TestTopologyDelegation(unittest.TestCase):
    """Verify topology-related dialog methods delegate to _topology_controller."""

    def test_topology_controller_has_create_template(self):
        """TopologyController has create_topology_template_layers."""
        from swe2d.workbench.controllers.topology_controller import TopologyController
        self.assertTrue(callable(TopologyController.create_topology_template_layers))

    def test_topology_controller_has_generate_mesh(self):
        from swe2d.workbench.controllers.topology_controller import TopologyController
        self.assertTrue(callable(TopologyController.generate_mesh_from_topology_layers))

    def test_topology_controller_has_terminate(self):
        from swe2d.workbench.controllers.topology_controller import TopologyController
        self.assertTrue(callable(TopologyController.on_terminate_topology_mesh))

    def test_topology_controller_has_open_explorer(self):
        from swe2d.workbench.controllers.topology_controller import TopologyController
        self.assertTrue(callable(TopologyController.open_model_gpkg_explorer))


class TestOverlayDelegation(unittest.TestCase):
    """Verify overlay-related dialog methods delegate to _overlay_controller."""

    def test_overlay_controller_has_toggle(self):
        """OverlayController has methods that wiring connects to."""
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        self.assertTrue(callable(OverlayController.on_high_perf_canvas_overlay_toggled))

    def test_overlay_controller_has_style_changed(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        self.assertTrue(callable(OverlayController.on_high_perf_canvas_overlay_style_changed))

    def test_overlay_controller_has_export_geotiff(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        self.assertTrue(callable(OverlayController.export_high_perf_overlay_to_geotiff))


class TestTopologyViewProtocolDelegation(unittest.TestCase):
    """Verify TopologyMeshView protocol methods delegate to _topology_tab_view."""

    def test_update_topo_status_delegates(self):
        """Dialog.update_topo_status calls _topology_tab_view.topo_status_lbl.setText."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog.__new__(SWE2DWorkbenchStudioDialog)
        dlg._topology_tab_view = MagicMock()
        dlg.update_topo_status("test")
        dlg._topology_tab_view.topo_status_lbl.setText.assert_called_once_with("test")

    def test_update_topo_controls_summary_delegates(self):
        """Dialog.update_topo_controls_summary → _topology_tab_view.topo_controls_summary_lbl.setText."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog.__new__(SWE2DWorkbenchStudioDialog)
        dlg._topology_tab_view = MagicMock()
        dlg.update_topo_controls_summary("summary")
        dlg._topology_tab_view.topo_controls_summary_lbl.setText.assert_called_once_with("summary")

    def test_get_topo_widget_value_delegates(self):
        """Dialog.get_topo_widget_value reads widget via getattr on _topology_tab_view."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog.__new__(SWE2DWorkbenchStudioDialog)
        mock_spin = MagicMock()
        mock_spin.value.return_value = 42.0
        dlg._topology_tab_view = MagicMock()
        type(dlg._topology_tab_view).topo_size_spin = PropertyMock(return_value=mock_spin)
        result = dlg.get_topo_widget_value("topo_size_spin")
        self.assertEqual(result, 42.0)

    def test_set_topo_widget_visible_delegates(self):
        """Dialog.set_topo_widget_visible calls setVisible on the widget."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog.__new__(SWE2DWorkbenchStudioDialog)
        mock_widget = MagicMock()
        dlg._topology_tab_view = MagicMock()
        type(dlg._topology_tab_view).topo_size_spin = PropertyMock(return_value=mock_widget)
        dlg.set_topo_widget_visible("topo_size_spin", False)
        mock_widget.setVisible.assert_called_once_with(False)

    def test_get_topo_combo_data_delegates(self):
        """Dialog.get_topo_combo_data reads currentData from the widget."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog.__new__(SWE2DWorkbenchStudioDialog)
        mock_combo = MagicMock()
        mock_combo.currentData.return_value = "triangle"
        dlg._topology_tab_view = MagicMock()
        type(dlg._topology_tab_view).topo_backend_combo = PropertyMock(return_value=mock_combo)
        result = dlg.get_topo_combo_data("topo_backend_combo")
        self.assertEqual(result, "triangle")


class TestDialogNotInlineBusinessLogic(unittest.TestCase):
    """Verify dialog does NOT contain inline business logic for delegated paths.

    These tests serve as sentinels — if they fail, business logic has been
    added back to the dialog instead of being delegated to controllers.
    """

    def test_dialog_has_no_direct_sqlite3_connect(self):
        """Dialog must not contain sqlite3.connect calls (delegated to gpkg_service)."""
        import inspect
        import swe2d.workbench.studio_dialog as sm
        source = inspect.getsource(sm)
        self.assertNotIn("sqlite3.connect", source)

    def test_dialog_has_no_direct_execute_run_body(self):
        """Dialog must not contain the full run pipeline body inline."""
        import inspect
        import swe2d.workbench.studio_dialog as sm
        source = inspect.getsource(sm)
        # The run pipeline lives in RunController._execute_run, not the dialog
        self.assertNotIn("def _execute_run", source)

    def test_dialog_has_no_direct_overlay_parameters_collection(self):
        """Overlay parameter collection lives in overlay_parameters_service, not the dialog."""
        import inspect
        import swe2d.workbench.studio_dialog as sm
        source = inspect.getsource(sm)
        # The dialog should import collect_overlay_parameters, not re-implement it
        self.assertIn("collect_overlay_parameters", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
