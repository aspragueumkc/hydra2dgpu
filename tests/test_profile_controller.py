"""tests/test_profile_controller.py

Verifies the ProfileController surfaces errors and never silently swallows
them. Uses a mock view to avoid the QApplication dependency.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from swe2d.workbench.controllers.profile_controller import ProfileController


def _make_minimal_gpkg(path: str) -> None:
    """Create a GPKG with just the tables the dialog reads at construction."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE swe2d_drainage_nodes ("
        "node_id TEXT, invert_elev REAL, rim_elev REAL, max_depth REAL)"
    )
    conn.execute(
        "CREATE TABLE swe2d_drainage_links ("
        "link_id TEXT, from_node TEXT, to_node TEXT, length REAL, "
        "inlet_invert_elev REAL, outlet_invert_elev REAL, "
        "link_shape TEXT, diameter REAL, rise REAL)"
    )
    conn.execute("INSERT INTO swe2d_drainage_nodes VALUES ('N1', 0.0, 5.0, 1.0)")
    conn.execute("INSERT INTO swe2d_drainage_nodes VALUES ('N2', 0.0, 5.0, 1.0)")
    conn.execute(
        "INSERT INTO swe2d_drainage_links VALUES "
        "('L1', 'N1', 'N2', 100.0, 0.0, 0.0, 'circular', 2.0, 0.0)"
    )
    conn.commit()
    conn.close()


class _MockView:
    """Stand-in for the studio dialog. Records log calls and answers the
    file-picker prompt."""

    def __init__(self, gpkg_path: str = "", picked_path: str = ""):
        self._gpkg_path = gpkg_path
        self._picked_path = picked_path
        self.logs: List[str] = []
        self.show_errors: List[Exception] = []

    def get_active_gpkg_path(self) -> str:
        return self._gpkg_path

    def get_active_run_id(self) -> str:
        return ""

    def get_qgis_iface(self):
        return None

    def get_open_file_name(self, *args, **kwargs) -> str:
        return self._picked_path

    def _log(self, msg: str) -> None:
        self.logs.append(msg)


class TestProfileControllerUsesActiveGpkg(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_minimal_gpkg(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_active_gpkg_path_used_when_file_exists(self):
        view = _MockView(gpkg_path=self.path)
        ctrl = ProfileController(view)
        with patch("swe2d.workbench.dialogs.network_profile_dialog.NetworkProfileDialog") as dlg:
            ctrl.open_network_profile_viewer()
            # Dialog was constructed with the active path
            dlg.assert_called_once()
            self.assertEqual(dlg.call_args.kwargs["gpkg_path"], self.path)
            # No file picker was used
            self.assertEqual(len(view.logs), 0)

    def test_no_active_path_falls_back_to_picker(self):
        view = _MockView(gpkg_path="", picked_path=self.path)
        ctrl = ProfileController(view)
        with patch("swe2d.workbench.dialogs.network_profile_dialog.NetworkProfileDialog") as dlg:
            ctrl.open_network_profile_viewer()
            dlg.assert_called_once()
            self.assertEqual(dlg.call_args.kwargs["gpkg_path"], self.path)
            # User was informed that no active GPKG was set
            self.assertTrue(any("No active" in m for m in view.logs))

    def test_picker_cancelled_returns_silently(self):
        view = _MockView(gpkg_path="", picked_path="")
        ctrl = ProfileController(view)
        with patch("swe2d.workbench.dialogs.network_profile_dialog.NetworkProfileDialog") as dlg:
            ctrl.open_network_profile_viewer()
            dlg.assert_not_called()

    def test_active_path_missing_prompts_picker(self):
        missing = "/nonexistent/path.gpkg"
        view = _MockView(gpkg_path=missing, picked_path=self.path)
        ctrl = ProfileController(view)
        with patch("swe2d.workbench.dialogs.network_profile_dialog.NetworkProfileDialog") as dlg:
            ctrl.open_network_profile_viewer()
            dlg.assert_called_once()
            self.assertEqual(dlg.call_args.kwargs["gpkg_path"], self.path)

    def test_picked_path_missing_shows_message_box(self):
        view = _MockView(gpkg_path="", picked_path="/nonexistent/path.gpkg")
        ctrl = ProfileController(view)
        with patch("swe2d.workbench.dialogs.network_profile_dialog.NetworkProfileDialog") as dlg, \
             patch("swe2d.workbench.controllers.profile_controller.QtWidgets.QMessageBox.warning") as mb:
            ctrl.open_network_profile_viewer()
            dlg.assert_not_called()
            mb.assert_called_once()

    def test_dialog_constructor_exception_is_surfaced(self):
        """Any exception in dialog construction must reach the user, not be swallowed."""
        view = _MockView(gpkg_path=self.path)
        ctrl = ProfileController(view)
        boom = RuntimeError("kaboom")
        with patch("swe2d.workbench.dialogs.network_profile_dialog.NetworkProfileDialog",
                   side_effect=boom), \
             patch("swe2d.workbench.controllers.profile_controller.QtWidgets.QMessageBox.critical") as mb:
            ctrl.open_network_profile_viewer()
            # A modal critical message box was shown
            mb.assert_called_once()
            # And a log line was emitted
            self.assertTrue(any("ERROR" in m for m in view.logs))
            # With the exception text
            self.assertTrue(any("kaboom" in m for m in view.logs))


if __name__ == "__main__":
    unittest.main()
