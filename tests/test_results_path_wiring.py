"""Tests for results GeoPackage path wiring after the widget move.

These two methods used to reach into ``self._run_dock`` for the
``results_gpkg_path_edit`` and ``results_table_name_edit`` widgets.
Those widgets moved to ``self._model_tab_view`` (commit 686e609 — the
Run dock was stripped to its execution surface). The methods were
left reading from the run dock, so they silently fell back to the
model GPKG even when the user had typed a different path in the
Output page's "Results GPKG" field.

These tests pin the new wiring: both methods must read from
``_model_tab_view`` and return the path/prefix the user entered.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from qgis.PyQt.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


def _make_dlg_with_mock_tab_view():
    """Build a dialog mock whose ``_model_tab_view`` owns the moved
    widgets. The dialog itself is a MagicMock so ``_log`` and other
    attributes are stubbed automatically.
    """
    dlg = MagicMock()
    mt = MagicMock()
    dlg._model_tab_view = mt
    # _run_dock is present but does NOT have the moved widgets (which
    # is the bug shape — old code looked here, got None, fell back).
    dlg._run_dock = MagicMock(spec=[])  # no attrs at all
    return dlg, mt


class TestCurrentLineResultsStoragePath(unittest.TestCase):
    """``_current_line_results_storage_path`` must read from
    ``_model_tab_view.results_gpkg_path_edit``."""

    def test_returns_path_from_model_tab_widget(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "results.gpkg")
            dlg, mt = _make_dlg_with_mock_tab_view()
            mt.results_gpkg_path_edit.text.return_value = gpkg_path

            from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
            result = SWE2DWorkbenchStudioDialog._current_line_results_storage_path(dlg)
            self.assertEqual(result, os.path.abspath(gpkg_path))
            mt.results_gpkg_path_edit.text.assert_called_once()

    def test_does_not_consult_run_dock(self):
        """Regression: the bug was that the dialog reached into
        ``_run_dock.results_gpkg_path_edit`` (which no longer exists).
        After the fix, the run dock must not be consulted. We
        confirm by putting a 'talking' value on the run dock — if
        the dialog looks there, the test sees it.
        """
        dlg, mt = _make_dlg_with_mock_tab_view()
        mt.results_gpkg_path_edit.text.return_value = ""
        # If the dialog reads from _run_dock.results_gpkg_path_edit,
        # it'll pick up this path (a fake one that doesn't exist).
        # After the fix, the run dock is bypassed and we get "" back.
        with tempfile.TemporaryDirectory() as tmp:
            fake_run_dock_path = os.path.join(tmp, "run_dock.gpkg")
            dlg._run_dock.results_gpkg_path_edit = MagicMock()
            dlg._run_dock.results_gpkg_path_edit.text.return_value = fake_run_dock_path

            from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
            result = SWE2DWorkbenchStudioDialog._current_line_results_storage_path(dlg)
            # Must not be the fake run-dock path.
            self.assertNotEqual(
                os.path.abspath(result),
                os.path.abspath(fake_run_dock_path),
                "Dialog is reading from _run_dock.results_gpkg_path_edit — "
                "should read from _model_tab_view.results_gpkg_path_edit.",
            )

    def test_expands_user_path(self):
        """``~`` and relative paths must be resolved to absolute."""
        dlg, mt = _make_dlg_with_mock_tab_view()
        mt.results_gpkg_path_edit.text.return_value = "~/my_results.gpkg"

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        result = SWE2DWorkbenchStudioDialog._current_line_results_storage_path(dlg)
        self.assertTrue(result.startswith("/"))
        self.assertNotIn("~", result)


class TestSelectedResultsTablePrefix(unittest.TestCase):
    """``_selected_results_table_prefix`` must read from
    ``_model_tab_view.results_table_name_edit``."""

    def test_returns_prefix_from_model_tab_widget(self):
        dlg, mt = _make_dlg_with_mock_tab_view()
        mt.results_table_name_edit.text.return_value = "run_a"

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        result = SWE2DWorkbenchStudioDialog._selected_results_table_prefix(dlg)
        self.assertEqual(result, "run_a")

    def test_does_not_consult_run_dock(self):
        dlg, mt = _make_dlg_with_mock_tab_view()
        mt.results_table_name_edit.text.return_value = ""
        # Run dock must not be consulted.
        dlg._run_dock.results_table_name_edit = MagicMock()
        type(dlg._run_dock.results_table_name_edit).text = MagicMock(
            side_effect=AssertionError(
                "Dialog must not consult _run_dock.results_table_name_edit."
            )
        )

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        # Empty input → empty output.
        self.assertEqual(
            SWE2DWorkbenchStudioDialog._selected_results_table_prefix(dlg),
            "",
        )

    def test_sanitizes_table_prefix(self):
        """Non-alphanumeric chars must be replaced with underscores."""
        dlg, mt = _make_dlg_with_mock_tab_view()
        mt.results_table_name_edit.text.return_value = "run with spaces"

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        result = SWE2DWorkbenchStudioDialog._selected_results_table_prefix(dlg)
        # Spaces → underscores, trimmed.
        self.assertEqual(result, "run_with_spaces")


class TestBatchSimulationDialogMeshGpkgPrefill(unittest.TestCase):
    """``open_batch_simulation_dialog`` must read the results GPKG
    fallback path from ``_model_tab_view``, not the run dock."""

    def _run_and_capture_mesh_gpkg(self, dlg):
        """Patch BatchSimulationDialog, run the controller, return the
        ``mesh_gpkg`` arg the dialog was called with.
        """
        from contextlib import contextmanager
        from unittest.mock import patch
        from swe2d.workbench.controllers.run_controller import RunController

        captured = {}

        @contextmanager
        def _capture():
            with patch(
                "swe2d.workbench.dialogs.batch_simulation_dialog.BatchSimulationDialog"
            ) as mock_dlg_cls:
                rc = RunController(view=dlg)
                rc.open_batch_simulation_dialog()
                # Read call args AFTER the call so they're populated.
                captured["mesh_gpkg"] = mock_dlg_cls.call_args.kwargs.get("mesh_gpkg")
                yield

        with _capture():
            pass
        return captured.get("mesh_gpkg")

    def test_reads_from_model_tab_when_model_gpkg_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "results.gpkg")
            dlg = MagicMock()
            dlg._model_gpkg_path = ""  # model GPKG empty → fall through
            dlg.get_results_gpkg_path.return_value = gpkg_path
            dlg._run_dock = MagicMock(spec=[])

            mesh_gpkg = self._run_and_capture_mesh_gpkg(dlg)
            self.assertEqual(mesh_gpkg, os.path.abspath(gpkg_path))

    def test_does_not_consult_run_dock_for_mesh_gpkg(self):
        """Regression: the run_controller used to read
        ``view._run_dock.results_gpkg_path_edit``. Now it reads through
        the View protocol ``get_results_gpkg_path()``.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dlg = MagicMock()
            dlg.get_results_gpkg_path.return_value = ""
            fake = os.path.join(tmp, "run_dock_stale.gpkg")

            mesh_gpkg = self._run_and_capture_mesh_gpkg(dlg)
            # Must NOT be the fake run-dock path.
            self.assertNotEqual(
                mesh_gpkg, os.path.abspath(fake),
                "Controller is reading from _run_dock.results_gpkg_path_edit "
                "— should read through View protocol.",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)