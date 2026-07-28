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

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])


def _make_batch_parent_widget(
    model_gpkg_path: str = "",
    mesh_data: dict | None = None,
    widget_state: dict | None = None,
    replay_payload: dict | None = None,
) -> QtWidgets.QWidget:
    """Build a real QWidget that can be passed as parent to
    ``BatchSimulationDialog`` while providing the mock protocol
    attributes (``collect_widget_state_for_save``,
    ``build_replay_payload``) that the dialog expects.

    A plain ``MagicMock`` cannot be passed because PyQt's C++ type
    checks reject it (``QDialog(parent)`` requires a real QWidget).
    """
    parent = QtWidgets.QWidget()
    parent._model_gpkg_path = str(model_gpkg_path or "")
    parent._mesh_data = dict(mesh_data or {})
    parent.collect_widget_state_for_save = MagicMock(
        return_value=dict(widget_state or {})
    )
    parent.build_replay_payload = MagicMock(
        return_value=dict(replay_payload or {})
    )
    return parent


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


class TestCollectDataSourceConfigFullPaths(unittest.TestCase):
    """``collect_data_source_config`` must always include the full GPKG path
    for every layer, even when the layer comes from the same GPKG as the
    model (``_model_gpkg_path``). Without this, the CLI replay JSON cannot
    resolve layers when the model GPKG is at a different path."""

    def test_model_gpkg_path_is_included(self):
        """Regression: the old code dropped the ``gpkg`` key when the
        layer's source GPKG matched ``_model_gpkg_path``. Every entry needs
        the full path so batch/CLI replays can resolve layers."""
        import sqlite3
        from unittest.mock import patch

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "model.gpkg")
            # Create a minimal GPKG with a gpkg_contents table
            conn = sqlite3.connect(gpkg_path)
            conn.execute(
                "CREATE TABLE gpkg_contents (table_name TEXT, identifier TEXT)"
            )
            conn.execute(
                "INSERT INTO gpkg_contents VALUES (?, ?)",
                ("bc_lines", "bc_lines"),
            )
            conn.commit()
            conn.close()

            dlg = MagicMock()
            dlg._model_gpkg_path = gpkg_path

            # Mock model tab view with bc_lines_layer_combo
            mt = MagicMock()
            bc_combo = MagicMock()
            bc_combo.currentData.return_value = "layer_abc"
            mt.bc_lines_layer_combo = bc_combo
            dlg._model_tab_view = mt

            # Mock layer that lives in the same GPKG as _model_gpkg_path
            layer = MagicMock()
            layer.source.return_value = f"{gpkg_path}|layername=bc_lines"
            layer.name.return_value = "bc_lines"

            with patch("qgis.core.QgsProject.instance") as mock_instance:
                mock_project = MagicMock()
                mock_instance.return_value = mock_project
                mock_project.mapLayer.return_value = layer

                result = SWE2DWorkbenchStudioDialog.collect_data_source_config(
                    dlg
                )

            self.assertIn("bc_lines", result)
            self.assertIn(
                "gpkg",
                result["bc_lines"],
                "gpkg key must be present even when layer GPKG "
                "matches _model_gpkg_path",
            )
            self.assertEqual(result["bc_lines"]["gpkg"], gpkg_path)
            self.assertEqual(result["bc_lines"]["table"], "bc_lines")


    def test_drainage_gpkg_path_is_included(self):
        """Regression: the drainage section built its dict inline (not via
        ``_dict_with_gpkg``) and conditionally omitted the ``gpkg`` key when
        it matched ``_model_gpkg_path``. Every entry needs the full path so
        batch/CLI replays can resolve layers."""
        import sqlite3
        from unittest.mock import patch

        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "model.gpkg")
            # Create a minimal GPKG with gpkg_contents for drainage tables
            conn = sqlite3.connect(gpkg_path)
            conn.execute(
                "CREATE TABLE gpkg_contents (table_name TEXT, identifier TEXT)"
            )
            for tbl in ("drain_nodes", "drain_links"):
                conn.execute(
                    "INSERT INTO gpkg_contents VALUES (?, ?)",
                    (tbl, tbl),
                )
            conn.commit()
            conn.close()

            dlg = MagicMock()
            dlg._model_gpkg_path = gpkg_path

            # Mock model tab view with drainage combo boxes
            mt = MagicMock()
            dn_combo = MagicMock()
            dn_combo.currentData.return_value = "layer_dn"
            mt.drain_nodes_layer_combo = dn_combo
            dl_combo = MagicMock()
            dl_combo.currentData.return_value = "layer_dl"
            mt.drain_links_layer_combo = dl_combo
            # Inlets combos return nothing — we want basic drainage test
            di_combo = MagicMock()
            di_combo.currentData.return_value = None
            mt.drain_inlets_layer_combo = di_combo
            ni_combo = MagicMock()
            ni_combo.currentData.return_value = None
            mt.drain_node_inlets_layer_combo = ni_combo
            dlg._model_tab_view = mt

            # Mock layers that live in the same GPKG as _model_gpkg_path
            dn_layer = MagicMock()
            dn_layer.source.return_value = f"{gpkg_path}|layername=drain_nodes"
            dn_layer.name.return_value = "drain_nodes"
            dl_layer = MagicMock()
            dl_layer.source.return_value = f"{gpkg_path}|layername=drain_links"
            dl_layer.name.return_value = "drain_links"

            def _map_layer_side_effect(lid: str):
                mapping = {
                    "layer_dn": dn_layer,
                    "layer_dl": dl_layer,
                }
                return mapping.get(lid)

            with patch("qgis.core.QgsProject.instance") as mock_instance:
                mock_project = MagicMock()
                mock_instance.return_value = mock_project
                mock_project.mapLayer.side_effect = _map_layer_side_effect

                result = SWE2DWorkbenchStudioDialog.collect_data_source_config(
                    dlg
                )

            self.assertIn("drainage", result)
            drainage = result["drainage"]
            self.assertIn(
                "gpkg",
                drainage,
                "gpkg key must be present even when layer GPKG "
                "matches _model_gpkg_path",
            )
            self.assertEqual(drainage["gpkg"], gpkg_path)
            self.assertEqual(drainage["nodes_layer"], "drain_nodes")
            self.assertEqual(drainage["links_layer"], "drain_links")


class TestBatchSimulationDialogMeshSelector(unittest.TestCase):
    """Batch Simulation Dialog mesh auto-population and snapshot payload."""

    def test_auto_populates_from_parent_model(self):
        """Auto-populate mesh from parent's ``_model_gpkg_path`` and
        ``_mesh_data`` when ``os.path.isfile`` returns True."""
        from swe2d.workbench.dialogs.batch_simulation_dialog import (
            BatchSimulationDialog,
        )

        parent = _make_batch_parent_widget(
            model_gpkg_path="/tmp/model.gpkg",
            mesh_data={"mesh_name": "mesh_001"},
        )

        with patch("os.path.isfile", return_value=True):
            dlg = BatchSimulationDialog(parent=parent, mesh_gpkg="/tmp/model.gpkg")
            self.assertEqual(dlg._mesh_gpkg, "/tmp/model.gpkg")
            self.assertEqual(dlg._mesh_name, "mesh_001")

    def test_snapshot_uses_build_replay_payload(self):
        """Snapshot calls parent's ``build_replay_payload`` with the correct
        kwargs and adds a row to the table."""
        from swe2d.workbench.dialogs.batch_simulation_dialog import (
            BatchSimulationDialog,
        )

        parent = _make_batch_parent_widget(
            model_gpkg_path="/tmp/model.gpkg",
            mesh_data={"mesh_name": "mesh_001"},
            widget_state={"widget": "state"},
            replay_payload={
                "schema_version": "swe2d-replay/1",
                "run_id": "run_1",
                "mesh": {"mesh_name": "mesh_001"},
            },
        )

        with patch("os.path.isfile", return_value=True):
            dlg = BatchSimulationDialog(parent=parent, mesh_gpkg="/tmp/model.gpkg")
            dlg._snapshot_current_setup()

        parent.build_replay_payload.assert_called_once()
        kwargs = parent.build_replay_payload.call_args.kwargs
        self.assertEqual(kwargs["mesh_name"], "mesh_001")
        self.assertEqual(kwargs["mesh_gpkg_path"], "/tmp/model.gpkg")
        self.assertTrue(kwargs["run_id"].startswith("swe2d_"))
        # 1 base row from _add_row + 1 snapshot row from _add_row_from_entry
        self.assertEqual(dlg._table.rowCount(), 2)

    def test_set_mesh_preserves_rich_mesh_dict(self):
        """``_set_row_mesh`` preserves existing mesh dict fields (e.g.
        ``gpkg_path``, ``crs_wkt``) when updating ``mesh_name``."""
        from swe2d.workbench.dialogs.batch_simulation_dialog import (
            BatchSimulationDialog,
            _COL_PARAMS,
        )

        parent = _make_batch_parent_widget(
            model_gpkg_path="/tmp/model.gpkg",
            mesh_data={"mesh_name": "mesh_001"},
            replay_payload={
                "schema_version": "swe2d-replay/1",
                "run_id": "run_1",
                "mesh": {
                    "mesh_name": "mesh_001",
                    "gpkg_path": "/tmp/model.gpkg",
                    "crs_wkt": "EPSG:4326",
                },
            },
        )

        with patch("os.path.isfile", return_value=True):
            dlg = BatchSimulationDialog(parent=parent, mesh_gpkg="/tmp/model.gpkg")
            dlg._snapshot_current_setup()
            # Row 0: from _add_row (empty base_params {})
            # Row 1: from snapshot (rich mesh dict) — target this one
            dlg._set_row_mesh(1, "mesh_002")
            params_item = dlg._table.item(1, _COL_PARAMS)
            params = json.loads(params_item.text())

        self.assertEqual(params["mesh"]["mesh_name"], "mesh_002")
        self.assertEqual(params["mesh"]["gpkg_path"], "/tmp/model.gpkg")
        self.assertEqual(params["mesh"]["crs_wkt"], "EPSG:4326")


if __name__ == "__main__":
    unittest.main(verbosity=2)