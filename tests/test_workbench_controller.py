"""Tests for WorkbenchController.

Controller methods are distributed across domain controllers:
- RunController (run pipeline: on_run, on_cancel, on_snapshot, on_preview_overrides)
- LayerController (layer combos: refresh_layer_combos)
- MeshController (mesh operations: import_mesh_from_layers, on_select_results_gpkg)
- OverlayController (high-perf overlay: load_mesh_snapshot_for_overlay)
- TopologyController (topology meshing)
"""
import unittest
from unittest.mock import MagicMock

from qgis.PyQt.QtWidgets import QApplication

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


class TestRunController(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_controller_imports(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertIsNotNone(RunController)

    def test_controller_holds_view(self):
        from swe2d.workbench.controllers.run_controller import RunController
        mock_view = MagicMock()
        ctrl = RunController(view=mock_view)
        self.assertIs(ctrl._view, mock_view)


class TestOverlayControllerLoadMeshSnapshot(unittest.TestCase):
    """load_mesh_snapshot_for_overlay lives in OverlayController."""

    def setUp(self):
        _ensure_app()

    def test_returns_false_when_no_panel(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        mock_view = MagicMock()
        mock_view._results_data = None
        ctrl = OverlayController(view=mock_view)
        result = ctrl.load_mesh_snapshot_for_overlay(t_s=1.0)
        self.assertFalse(result)

    def test_returns_false_when_no_run_ids(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        mock_view = MagicMock()
        mock_data = MagicMock()
        mock_data.overlay_selected_run.return_value = None
        mock_view._results_data = mock_data
        ctrl = OverlayController(view=mock_view)
        result = ctrl.load_mesh_snapshot_for_overlay(t_s=1.0)
        self.assertFalse(result)

    def test_returns_true_on_successful_load(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        import numpy as np
        from unittest.mock import patch

        mock_view = MagicMock()
        mock_view._high_perf_canvas_overlay_enabled = False
        mock_data = MagicMock()
        mock_rec = MagicMock()
        mock_rec.gpkg_path = "/tmp/test.gpkg"
        mock_rec.run_id = "run1"
        mock_data.overlay_selected_run.return_value = mock_rec
        mock_data.overlay_cell_x = np.array([1.0, 2.0])
        mock_data._live_times = np.array([1.0])
        mock_view._results_data = mock_data

        snapshot = {
            'h': np.array([1.0, 2.0]),
            'hu': np.array([0.1, 0.2]),
            'hv': np.array([0.0, 0.0]),
            't_s': 1.0,
            'cell_count': 2,
        }
        with patch('os.path.exists', return_value=True), \
             patch('swe2d.services.gpkg_persistence_service.load_baked_snapshot', return_value=snapshot):
            ctrl = OverlayController(view=mock_view)
            result = ctrl.load_mesh_snapshot_for_overlay(t_s=1.0)
            self.assertTrue(result)
            mock_data.set_data_source.assert_called_with("gpkg")

    def test_returns_false_when_snapshot_none(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        import numpy as np
        from unittest.mock import patch

        mock_view = MagicMock()
        mock_data = MagicMock()
        mock_rec = MagicMock()
        mock_rec.gpkg_path = "/tmp/test.gpkg"
        mock_rec.run_id = "run1"
        mock_data.overlay_selected_run.return_value = mock_rec
        mock_data.overlay_cell_x = np.array([1.0, 2.0])
        mock_view._results_data = mock_data

        with patch('os.path.exists', return_value=True), \
             patch('swe2d.services.gpkg_persistence_service.load_baked_snapshot', return_value=None):
            ctrl = OverlayController(view=mock_view)
            result = ctrl.load_mesh_snapshot_for_overlay(t_s=1.0)
            self.assertFalse(result)


class TestControllerOnCancel(unittest.TestCase):
    """on_cancel lives in RunController."""

    def setUp(self):
        _ensure_app()

    def test_on_cancel_method_exists(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(
            callable(getattr(RunController, "on_cancel", None)),
            "RunController must have on_cancel method",
        )

    def test_on_cancel_sets_view_flag(self):
        from swe2d.workbench.controllers.run_controller import RunController
        mock_view = MagicMock()
        mock_view._log = MagicMock()
        ctrl = RunController(view=mock_view)
        ctrl.on_cancel()
        self.assertTrue(mock_view._cancel_requested)
        mock_view._log.assert_called_once()

    def test_on_cancel_takes_no_args(self):
        from swe2d.workbench.controllers.run_controller import RunController
        import inspect
        sig = inspect.signature(RunController.on_cancel)
        params = [p for p in sig.parameters if p != "self"]
        self.assertEqual(params, [], "on_cancel must take no args besides self")


class TestControllerOnSnapshot(unittest.TestCase):
    """on_snapshot lives in RunController."""

    def setUp(self):
        _ensure_app()

    def test_on_snapshot_method_exists(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(
            callable(getattr(RunController, "on_snapshot", None)),
            "RunController must have on_snapshot method",
        )

    def test_on_snapshot_skips_when_no_mesh(self):
        from swe2d.workbench.controllers.run_controller import RunController
        mock_view = MagicMock()
        mock_view._mesh_data = None
        mock_view._results_data = None
        mock_view._log = MagicMock()
        ctrl = RunController(view=mock_view)
        ctrl.on_snapshot()
        mock_view._log.assert_not_called()

    def test_on_snapshot_calls_sync_snapshot_to_ui(self):
        from swe2d.workbench.controllers.run_controller import RunController
        from unittest.mock import patch
        mock_view = MagicMock()
        mock_view._mesh_data = {"node_x": MagicMock()}
        mock_view._log = MagicMock()
        mock_view._sync_snapshot_to_ui = MagicMock()

        ctrl = RunController(view=mock_view)
        ctrl.on_snapshot()
        mock_view._sync_snapshot_to_ui.assert_called_once()


class TestControllerOnSelectResultsGpkg(unittest.TestCase):
    """on_select_results_gpkg lives in MeshController."""

    def setUp(self):
        _ensure_app()

    def test_on_select_results_gpkg_method_exists(self):
        from swe2d.workbench.controllers.mesh_controller import MeshController
        self.assertTrue(
            callable(getattr(MeshController, "on_select_results_gpkg", None)),
            "MeshController must have on_select_results_gpkg method",
        )


class TestControllerImportMeshFromLayers(unittest.TestCase):
    """import_mesh_from_layers lives in MeshController."""

    def setUp(self):
        _ensure_app()

    def test_import_mesh_from_layers_method_exists(self):
        from swe2d.workbench.controllers.mesh_controller import MeshController
        self.assertTrue(
            callable(getattr(MeshController, "import_mesh_from_layers", None)),
            "MeshController must have import_mesh_from_layers method",
        )


class TestControllerOnRun(unittest.TestCase):
    """on_run lives in RunController with view reference."""

    def setUp(self):
        _ensure_app()
        from unittest.mock import patch
        self._patch = patch

    def test_on_run_method_exists(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(
            callable(getattr(RunController, "on_run", None)),
            "RunController must have on_run method",
        )

    def test_on_run_signature_uses_view_not_kwargs(self):
        """The 50+ kwarg explosion must be replaced by a view reference."""
        from swe2d.workbench.controllers.run_controller import RunController
        import inspect
        sig = inspect.signature(RunController.on_run)
        params = list(sig.parameters)
        # Must be self + optional request only; no widget-level kwargs.
        self.assertIn("self", params)
        self.assertLessEqual(
            len(params), 2,
            f"on_run must not take widget-level kwargs; got {params!r}",
        )


    def test_on_run_passes_view_as_wb(self):
        """The controller must call _build_run_context when starting a run."""
        from swe2d.workbench.controllers.run_controller import RunController
        patch = self._patch
        mock_view = MagicMock()
        mock_view._mesh_data = {"node_x": "mesh"}
        mock_view._log = MagicMock()
        ctrl = RunController(view=mock_view)
        with patch.object(
            ctrl, "_build_run_context"
        ) as mock_build, patch(
            "swe2d.workbench.controllers.run_controller.SimulationWorker"
        ):
            ctrl.on_run()
            mock_build.assert_called_once()

    def test_on_run_aborts_when_mesh_data_none(self):
        from swe2d.workbench.controllers.run_controller import RunController
        patch = self._patch
        mock_view = MagicMock()
        mock_view._mesh_data = None
        mock_view._log = MagicMock()
        ctrl = RunController(view=mock_view)
        with patch.object(
            ctrl, "_build_run_context"
        ) as mock_build:
            ctrl.on_run()
            mock_build.assert_not_called()
            log_message = mock_view._log.call_args[0][0]
            self.assertIn("mesh", log_message.lower())


class TestControllerRefreshLayerCombos(unittest.TestCase):
    """refresh_layer_combos lives in LayerController."""

    def setUp(self):
        _ensure_app()

    def test_method_exists(self):
        from swe2d.workbench.controllers.layer_controller import LayerController
        self.assertTrue(
            callable(getattr(LayerController, "refresh_layer_combos", None)),
            "LayerController must have refresh_layer_combos method",
        )


class TestControllerOnHighPerfCanvasOverlayToggled(unittest.TestCase):
    """on_high_perf_canvas_overlay_toggled lives in OverlayController."""

    def setUp(self):
        _ensure_app()

    def test_method_exists(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        self.assertTrue(
            callable(
                getattr(
                    OverlayController,
                    "on_high_perf_canvas_overlay_toggled",
                    None,
                )
            ),
            "OverlayController must have on_high_perf_canvas_overlay_toggled",
        )

    def test_disable_clears_overlay_state(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        mock_view = MagicMock()
        mock_view._state.high_perf_canvas_overlay_item = None
        mock_view._resolve_qgis_iface.return_value = MagicMock()
        ctrl = OverlayController(view=mock_view)
        ctrl.on_high_perf_canvas_overlay_toggled(False)
        self.assertFalse(mock_view._high_perf_canvas_overlay_enabled)

    def test_enable_triggers_refresh(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        import numpy as np
        mock_view = MagicMock()
        mock_view._results_data = None
        ctrl = OverlayController(view=mock_view)
        mock_data = ctrl._data
        mock_data._live_times = np.array([1.0])
        mock_data._live_h = np.array([0.5])
        mock_data._live_hu = np.array([0.1])
        mock_data._live_hv = np.array([0.0])
        mock_data.overlay_cell_x = np.array([1.0, 2.0])
        ctrl.on_high_perf_canvas_overlay_toggled(True)
        self.assertTrue(mock_view._high_perf_canvas_overlay_enabled)


class TestControllerOnHighPerfCanvasOverlayStyleChanged(unittest.TestCase):
    """on_high_perf_canvas_overlay_style_changed lives in OverlayController."""

    def setUp(self):
        _ensure_app()

    def test_method_exists(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        self.assertTrue(
            callable(
                getattr(
                    OverlayController,
                    "on_high_perf_canvas_overlay_style_changed",
                    None,
                )
            ),
            "OverlayController must have on_high_perf_canvas_overlay_style_changed",
        )

    def test_changes_calls_sync_overlay_widget_states(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        mock_view = MagicMock()
        mock_view._high_perf_canvas_overlay_enabled = False
        mock_view._snapshot_timesteps = []
        mock_data = MagicMock()
        mock_data.overlay_cell_x = type("a", (), {"size": 0})()
        mock_view._results_data = mock_data
        ctrl = OverlayController(view=mock_view)
        ctrl.on_high_perf_canvas_overlay_style_changed()
        mock_view.sync_overlay_widget_states.assert_called_once()


class TestControllerExportHighPerfOverlayToGeotiff(unittest.TestCase):
    """export_high_perf_overlay_to_geotiff lives in OverlayController."""

    def setUp(self):
        _ensure_app()

    def test_method_exists(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        self.assertTrue(
            callable(
                getattr(
                    OverlayController,
                    "export_high_perf_overlay_to_geotiff",
                    None,
                )
            ),
            "OverlayController must have export_high_perf_overlay_to_geotiff",
        )

    def test_aborts_when_no_overlay_data(self):
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        import numpy as np
        from unittest.mock import patch

        mock_view = MagicMock()
        mock_view._results_data = None
        ctrl = OverlayController(view=mock_view)
        mock_data = ctrl._data
        mock_data.overlay_cell_x = np.array([], dtype=np.float64)
        mock_view._results_data = mock_data
        mock_view._snapshot_timesteps = []
        ctrl.export_high_perf_overlay_to_geotiff()
        mock_view.show_warning_message.assert_called_once()


class TestControllerOnPreviewOverrides(unittest.TestCase):
    """on_preview_overrides lives in RunController."""

    def setUp(self):
        _ensure_app()

    def test_method_exists(self):
        from swe2d.workbench.controllers.run_controller import RunController
        self.assertTrue(
            callable(getattr(RunController, "on_preview_overrides", None)),
            "RunController must have on_preview_overrides method",
        )

    def test_aborts_after_failed_mesh_gen(self):
        from swe2d.workbench.controllers.run_controller import RunController
        mock_view = MagicMock()
        mock_view._mesh_data = None
        ctrl = RunController(view=mock_view)

        def _gen():
            mock_view._mesh_data = None
        mock_view._on_generate_mesh.side_effect = _gen
        ctrl.on_preview_overrides()
        mock_view._on_generate_mesh.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
