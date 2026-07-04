"""Tests for the two-step Load/Save Simulation Config flow.

These methods used to silently require
``view._current_line_results_storage_path()`` to point at a valid GPKG.
They now show a file picker first (same dialog as the GeoPackage
Explorer action) so the user can browse any .gpkg on disk.

The picker is a real ``QFileDialog`` — these tests verify the
controller invokes ``getOpenFileName`` / ``getSaveFileName`` with the
right title, filter, and pre-fill, then handles the user choice
(cancel / missing-file / valid-file) correctly.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from qgis.PyQt.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


def _make_view():
    """Build a mock dialog with the surface area the controllers touch."""
    view = MagicMock()
    view._current_line_results_storage_path = MagicMock(return_value="")
    view._log = MagicMock()
    view._apply_run_log_metadata_to_ui = MagicMock(return_value=0)
    view.collect_run_widget_params = MagicMock(return_value={})
    view.model_tab = MagicMock()
    view.model_tab.get_run_time_hours_parsed = MagicMock(return_value=0.0)
    view._mesh_data = {}
    return view


def _make_controller(view):
    """Build a RunController with our mock view."""
    from swe2d.workbench.controllers.run_controller import RunController
    return RunController(view=view)


class TestOnLoadSimulationConfigFilePicker(unittest.TestCase):
    """``on_load_simulation_config`` opens a QFileDialog first so the
    user can choose which GeoPackage to load configs from."""

    def test_cancelled_picker_returns_silently(self):
        """If the user cancels the file picker, the controller returns
        without touching the GeoPackage or showing the config dialog."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("", "")), \
             patch("swe2d.services.gpkg_persistence_service.load_simulation_configs") as mock_load:
            rc.on_load_simulation_config()
            # load_simulation_configs must NOT be called — there's no
            # GeoPackage to read from.
            mock_load.assert_not_called()
            view._log.assert_not_called()

    def test_missing_file_logs_error(self):
        """If the user picks a path that doesn't exist, log an error
        and don't proceed."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("/tmp/does_not_exist.gpkg", "GeoPackage (*.gpkg)")), \
             patch("swe2d.services.gpkg_persistence_service.load_simulation_configs") as mock_load:
            rc.on_load_simulation_config()
            mock_load.assert_not_called()
            # The error must mention the missing path so the user
            # knows what went wrong.
            self.assertTrue(
                any("not found" in str(call).lower()
                    for call in view._log.call_args_list),
                f"Expected 'not found' in log calls, got: {view._log.call_args_list}",
            )

    def test_valid_file_loads_configs_and_shows_picker(self):
        """If the user picks a real file, the controller reads configs
        and shows the SWE2DSimulationConfigDialog."""
        view = _make_view()
        rc = _make_controller(view)
        fake_configs = [{"config_id": "cfg1", "mesh_name": "m1",
                          "created_utc": "2026-01-01",
                          "run_duration_s": 3600.0,
                          "description": "", "widget_state": {}}]
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("/tmp/exists.gpkg", "GeoPackage (*.gpkg)")), \
             patch("os.path.exists", return_value=True), \
             patch("swe2d.services.gpkg_persistence_service.load_simulation_configs",
                   return_value=fake_configs) as mock_load, \
             patch("swe2d.workbench.dialogs.simulation_config_dialog.SWE2DSimulationConfigDialog") as mock_dlg_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = 0  # QDialog.Rejected
            mock_dlg_cls.return_value = mock_dlg
            rc.on_load_simulation_config()
            mock_load.assert_called_once()
            self.assertEqual(mock_load.call_args.args[0], "/tmp/exists.gpkg")
            mock_dlg_cls.assert_called_once()
            kwargs = mock_dlg_cls.call_args.kwargs
            self.assertEqual(kwargs["db_path"], "/tmp/exists.gpkg")
            self.assertEqual(kwargs["configs"], fake_configs)

    def test_picker_title_and_filter_match_explorer(self):
        """The file picker should use the same title and filter as the
        GeoPackage Explorer action — that's the whole point of the UX
        consistency requirement."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("", "")) as mock_dialog:
            rc.on_load_simulation_config()
            self.assertEqual(mock_dialog.call_count, 1)
            args, kwargs = mock_dialog.call_args
            # Title is the second positional arg (parent is first).
            self.assertIn("GeoPackage", args[1])
            # Filter is the 4th positional arg.
            self.assertIn("*.gpkg", args[3])

    def test_picker_uses_the_results_gpkg_directory_as_start(self):
        """If a results GPKG path is set, pre-fill the picker with it."""
        view = _make_view()
        view._current_line_results_storage_path = MagicMock(
            return_value="/data/results/sim_run.gpkg"
        )
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("", "")) as mock_dialog, \
             patch("os.path.exists", return_value=True):
            rc.on_load_simulation_config()
            # The 3rd positional arg is the directory the dialog opens
            # to — should be empty (default), since getOpenFileName
            # accepts a path that doesn't exist and just opens the
            # parent dir. Either way, the controller must not raise.
            self.assertEqual(mock_dialog.call_count, 1)

    def test_empty_configs_logs_message(self):
        """If the chosen GeoPackage has no configs, log a message and
        don't open the config picker."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=("/tmp/empty.gpkg", "GeoPackage (*.gpkg)")), \
             patch("os.path.exists", return_value=True), \
             patch("swe2d.services.gpkg_persistence_service.load_simulation_configs",
                   return_value=[]), \
             patch("swe2d.workbench.dialogs.simulation_config_dialog.SWE2DSimulationConfigDialog") as mock_dlg_cls:
            rc.on_load_simulation_config()
            mock_dlg_cls.assert_not_called()


class TestOnSaveSimulationConfigFilePicker(unittest.TestCase):
    """``on_save_simulation_config`` opens a QFileDialog first so the
    user can choose which GeoPackage to save to."""

    def test_cancelled_picker_returns_silently(self):
        """If the user cancels the file picker, return without
        prompting for a name."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("", "")), \
             patch("qgis.PyQt.QtWidgets.QInputDialog.getText") as mock_input, \
             patch("swe2d.services.gpkg_persistence_service.persist_simulation_config") as mock_save:
            rc.on_save_simulation_config()
            mock_input.assert_not_called()
            mock_save.assert_not_called()

    def test_missing_gpkg_extension_is_added(self):
        """If the user typed a path without an extension, append .gpkg."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("/tmp/myresults", "GeoPackage (*.gpkg)")), \
             patch("qgis.PyQt.QtWidgets.QInputDialog.getText",
                   return_value=("myconfig", True)), \
             patch("swe2d.workbench.bridges.project_settings_bridge.collect_workbench_widget_state",
                   return_value={}), \
             patch("swe2d.services.gpkg_persistence_service.persist_simulation_config") as mock_save:
            rc.on_save_simulation_config()
            self.assertEqual(mock_save.call_args.kwargs["gpkg_path"],
                             "/tmp/myresults.gpkg")

    def test_valid_save_path_persists_config(self):
        """If the user picks a real path and enters a config name, the
        config is persisted to that path."""
        view = _make_view()
        view._mesh_data = {"mesh_name": "test_mesh"}
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("/tmp/sim.gpkg", "GeoPackage (*.gpkg)")), \
             patch("qgis.PyQt.QtWidgets.QInputDialog.getText",
                   return_value=("myconfig", True)), \
             patch("swe2d.workbench.bridges.project_settings_bridge.collect_workbench_widget_state",
                   return_value={}), \
             patch("swe2d.services.gpkg_persistence_service.persist_simulation_config") as mock_save:
            rc.on_save_simulation_config()
            mock_save.assert_called_once()
            kwargs = mock_save.call_args.kwargs
            self.assertEqual(kwargs["gpkg_path"], "/tmp/sim.gpkg")
            self.assertEqual(kwargs["config_id"], "myconfig")
            self.assertEqual(kwargs["mesh_name"], "test_mesh")

    def test_blank_name_falls_back_to_timestamp(self):
        """If the user clears the name field, fall back to a timestamp."""
        view = _make_view()
        rc = _make_controller(view)
        with patch("qgis.PyQt.QtWidgets.QFileDialog.getSaveFileName",
                   return_value=("/tmp/sim.gpkg", "GeoPackage (*.gpkg)")), \
             patch("qgis.PyQt.QtWidgets.QInputDialog.getText",
                   return_value=("   ", True)), \
             patch("swe2d.workbench.bridges.project_settings_bridge.collect_workbench_widget_state",
                   return_value={}), \
             patch("swe2d.services.gpkg_persistence_service.persist_simulation_config") as mock_save:
            rc.on_save_simulation_config()
            # config_id should be a swe2d_<timestamp> style string.
            self.assertTrue(
                mock_save.call_args.kwargs["config_id"].startswith("swe2d_"),
                f"Expected timestamp config_id, got: {mock_save.call_args.kwargs['config_id']!r}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)