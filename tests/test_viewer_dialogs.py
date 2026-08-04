#!/usr/bin/env python3
"""Behavioral tests for the run-log and simulation-config viewer dialogs.

Covers (Task D.5 of docs/plans/2026-08-02-gui-test-coverage.md):
- ``swe2d/workbench/dialogs/run_log_viewer_dialog.py`` (SWE2DRunLogViewerDialog)
- ``swe2d/workbench/dialogs/simulation_config_viewer_dialog.py``
  (SWE2DSimulationConfigViewerDialog)

Pattern P2 (spec docs/specs/2026-08-02-gui-test-coverage-design.md §4):
real artifacts are written through the PRODUCTION storage paths
(``swe2d.results.run_log_storage.persist_run_log_to_geopackage`` and
``swe2d.services.gpkg_persistence_service.persist_simulation_config``,
with widget state collected via the production
``collect_workbench_widget_state`` from real widgets), read back through
the production loaders, and the dialogs are constructed against them.
Assertions compare displayed widget content against the stored artifact.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# Ensure repo root is on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path and os.path.isdir(_REPO_ROOT):
    sys.path.insert(0, _REPO_ROOT)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_temp_results_gpkg,
    requires_qgis,
)


def _write_run_logs(gpkg_path, logs):
    """Persist run logs through the production storage writer and read them
    back through the production loader (the same pair the workbench uses)."""
    from swe2d.results.run_log_storage import (
        load_run_logs_from_geopackage,
        persist_run_log_to_geopackage,
    )

    for log in logs:
        ok = persist_run_log_to_geopackage(gpkg_path=gpkg_path, **log)
        if not ok:
            raise RuntimeError(f"persist_run_log_to_geopackage failed for {log}")
    records = load_run_logs_from_geopackage(gpkg_path=gpkg_path)
    if len(records) != len(logs):
        raise RuntimeError(
            f"production loader returned {len(records)} records, "
            f"expected {len(logs)}"
        )
    return records


def _close_active_modal():
    """Accept the currently active modal dialog (e.g. QMessageBox).

    Scheduled via ``QTimer.singleShot(0, ...)`` before triggering an action
    that opens a modal message box; the timer fires inside the modal's
    nested event loop.
    """
    from qgis.PyQt import QtWidgets

    widget = QtWidgets.QApplication.activeModalWidget()
    if widget is not None:
        widget.accept()


def _table_row_map(table):
    """Return {widget_name: (value_text, type_text)} from the viewer table."""
    out = {}
    for row in range(table.rowCount()):
        name_item = table.item(row, 0)
        value_item = table.item(row, 1)
        type_item = table.item(row, 2)
        out[name_item.text()] = (
            value_item.text() if value_item is not None else "",
            type_item.text() if type_item is not None else "",
        )
    return out


@requires_qgis
class TestRunLogViewerDialog(unittest.TestCase):
    """SWE2DRunLogViewerDialog against a real persisted run log."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from qgis.PyQt import QtWidgets

        self._qt_widgets = QtWidgets
        self._dialogs = []

    def tearDown(self):
        delete_widgets_now(*self._dialogs)

    def _make_dialog(self, records, run_id, db_path, **kwargs):
        from swe2d.workbench.dialogs.run_log_viewer_dialog import (
            SWE2DRunLogViewerDialog,
        )

        dlg = SWE2DRunLogViewerDialog(records, run_id, db_path, **kwargs)
        self._dialogs.append(dlg)
        dlg.show()
        self._qt_widgets.QApplication.processEvents()
        return dlg

    def test_displays_stored_run_log_content(self):
        log_text = (
            "SENTINEL_ALPHA_HEADER\n"
            "[00:00:01] Solver initialised\n"
            "[00:00:02] SENTINEL_ALPHA_TAIL"
        )
        with make_temp_results_gpkg() as gpkg_path:
            records = _write_run_logs(
                gpkg_path,
                [
                    {
                        "run_id": "run_alpha",
                        "start_wallclock": "2026-08-02T10:00:00",
                        "end_wallclock": "2026-08-02T10:00:12",
                        "duration_s": 12.5,
                        "log_text": log_text,
                        "metadata": {"purpose": "alpha sentinel"},
                    }
                ],
            )
            dlg = self._make_dialog(records, "run_alpha", gpkg_path)

            self.assertEqual(dlg.run_combo.count(), 1)
            self.assertEqual(str(dlg.run_combo.currentData()), "run_alpha")
            self.assertIn("run_alpha", dlg.run_combo.currentText())

            meta = dlg.meta_lbl.text()
            self.assertIn("run_alpha", meta)
            self.assertIn("2026-08-02T10:00:00", meta)
            self.assertIn("2026-08-02T10:00:12", meta)
            self.assertIn("12.50", meta)

            displayed = dlg.text.toPlainText()
            self.assertEqual(displayed, log_text)
            self.assertIn("SENTINEL_ALPHA_HEADER", displayed)
            self.assertIn("SENTINEL_ALPHA_TAIL", displayed)

            self.assertTrue(
                grab_non_empty(dlg), "run log viewer dialog grabbed blank"
            )

    def test_switching_runs_updates_display(self):
        with make_temp_results_gpkg() as gpkg_path:
            records = _write_run_logs(
                gpkg_path,
                [
                    {
                        "run_id": "run_alpha",
                        "start_wallclock": "2026-08-02T10:00:00",
                        "end_wallclock": "2026-08-02T10:00:05",
                        "duration_s": 5.0,
                        "log_text": "SENTINEL_ALPHA_LOG",
                        "metadata": {"idx": 1},
                    },
                    {
                        "run_id": "run_beta",
                        "start_wallclock": "2026-08-02T11:00:00",
                        "end_wallclock": "2026-08-02T11:00:07",
                        "duration_s": 7.25,
                        "log_text": "SENTINEL_BETA_LOG",
                        "metadata": {"idx": 2},
                    },
                ],
            )
            self.assertEqual(len(records), 2)
            # Loader returns newest first (rowid DESC tiebreak).
            self.assertEqual(records[0]["run_id"], "run_beta")

            dlg = self._make_dialog(records, "run_alpha", gpkg_path)
            self.assertEqual(dlg.run_combo.count(), 2)
            # Constructor selects the requested run_id even though it is
            # not the first record.
            self.assertEqual(str(dlg.run_combo.currentData()), "run_alpha")
            self.assertEqual(dlg.text.toPlainText(), "SENTINEL_ALPHA_LOG")
            self.assertIn("run_alpha", dlg.meta_lbl.text())

            beta_index = dlg.run_combo.findData("run_beta")
            self.assertGreaterEqual(beta_index, 0)
            dlg.run_combo.setCurrentIndex(beta_index)
            self._qt_widgets.QApplication.processEvents()

            self.assertEqual(dlg.text.toPlainText(), "SENTINEL_BETA_LOG")
            meta = dlg.meta_lbl.text()
            self.assertIn("run_beta", meta)
            self.assertIn("7.25", meta)

    def test_apply_callback_receives_stored_metadata(self):
        from qgis.PyQt import QtCore

        metadata = {"widget_state": {"run_duration": 3600}, "origin": "sentinel"}
        received = []

        def _apply_cb(payload):
            received.append(payload)
            return 5

        with make_temp_results_gpkg() as gpkg_path:
            records = _write_run_logs(
                gpkg_path,
                [
                    {
                        "run_id": "run_meta",
                        "start_wallclock": "2026-08-02T09:00:00",
                        "end_wallclock": "2026-08-02T09:00:03",
                        "duration_s": 3.0,
                        "log_text": "SENTINEL_META_LOG",
                        "metadata": metadata,
                    }
                ],
            )
            dlg = self._make_dialog(
                records,
                "run_meta",
                gpkg_path,
                apply_run_settings_callback=_apply_cb,
            )
            self.assertIsNotNone(dlg._apply_btn)

            # The success path opens a modal QMessageBox; auto-accept it
            # from inside its nested event loop.
            QtCore.QTimer.singleShot(0, _close_active_modal)
            dlg._apply_btn.click()
            self._qt_widgets.QApplication.processEvents()

            self.assertEqual(received, [metadata])

    def test_no_callback_means_no_apply_button(self):
        with make_temp_results_gpkg() as gpkg_path:
            records = _write_run_logs(
                gpkg_path,
                [
                    {
                        "run_id": "run_plain",
                        "start_wallclock": "2026-08-02T08:00:00",
                        "end_wallclock": "2026-08-02T08:00:01",
                        "duration_s": 1.0,
                        "log_text": "SENTINEL_PLAIN_LOG",
                        "metadata": None,
                    }
                ],
            )
            dlg = self._make_dialog(records, "run_plain", gpkg_path)
            self.assertIsNone(dlg._apply_btn)


@requires_qgis
class TestSimulationConfigViewerDialog(unittest.TestCase):
    """SWE2DSimulationConfigViewerDialog against a real saved config."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from qgis.PyQt import QtWidgets

        self._qt_widgets = QtWidgets
        self._dialogs = []
        self._widgets = []

    def tearDown(self):
        delete_widgets_now(*self._dialogs, *self._widgets)

    def _collect_real_widget_state(self):
        """Build widget state via the production collector from real widgets."""
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.bridges.project_settings_bridge import (
            collect_workbench_widget_state,
        )

        holder = SimpleNamespace()

        run_duration_spin = QtWidgets.QSpinBox()
        run_duration_spin.setRange(0, 10_000_000)
        run_duration_spin.setValue(3600)
        holder.run_duration_spin = run_duration_spin

        cfl_spin = QtWidgets.QDoubleSpinBox()
        cfl_spin.setRange(0.0, 1.0)
        cfl_spin.setDecimals(2)
        cfl_spin.setSingleStep(0.05)
        cfl_spin.setValue(0.45)
        holder.cfl_spin = cfl_spin

        scheme_combo = QtWidgets.QComboBox()
        scheme_combo.addItem("First order", 0)
        scheme_combo.addItem("MUSCL", 1)
        scheme_combo.setCurrentIndex(1)
        holder.spatial_scheme_combo = scheme_combo

        rain_check = QtWidgets.QCheckBox("Enable rainfall")
        rain_check.setChecked(True)
        holder.rain_check = rain_check

        title_edit = QtWidgets.QLineEdit("sentinel config title")
        holder.title_edit = title_edit

        self._widgets.extend(
            [run_duration_spin, cfl_spin, scheme_combo, rain_check, title_edit]
        )

        attrs = [
            "run_duration_spin",
            "cfl_spin",
            "spatial_scheme_combo",
            "rain_check",
            "title_edit",
        ]
        return collect_workbench_widget_state(
            ui=holder, widget_attrs=attrs, qtwidgets_module=QtWidgets
        )

    def _persist_config(self, gpkg_path, config_id, mesh_name, duration_s,
                        widget_state):
        from swe2d.services.gpkg_persistence_service import (
            persist_simulation_config,
        )

        persist_simulation_config(
            gpkg_path,
            config_id,
            mesh_name,
            duration_s,
            widget_state,
            description=f"sentinel config {config_id}",
            params={"run_duration_s": duration_s},
        )

    def _make_dialog(self, gpkg_path):
        from swe2d.workbench.dialogs.simulation_config_viewer_dialog import (
            SWE2DSimulationConfigViewerDialog,
        )

        dlg = SWE2DSimulationConfigViewerDialog(gpkg_path)
        self._dialogs.append(dlg)
        dlg.show()
        self._qt_widgets.QApplication.processEvents()
        return dlg

    def test_displays_saved_config_widgets(self):
        with make_temp_results_gpkg() as gpkg_path:
            widget_state = self._collect_real_widget_state()
            expected_widgets = widget_state["widgets"]
            self._persist_config(
                gpkg_path, "cfg_sentinel_alpha", "hydra_test_mesh", 3600.0,
                widget_state,
            )

            dlg = self._make_dialog(gpkg_path)

            self.assertEqual(dlg.config_combo.count(), 1)
            self.assertIn("cfg_sentinel_alpha", dlg.config_combo.currentText())

            meta = dlg.meta_lbl.text()
            self.assertIn("cfg_sentinel_alpha", meta)
            self.assertIn("hydra_test_mesh", meta)
            self.assertIn("3600.0", meta)

            rows = _table_row_map(dlg.table)
            # Every stored widget entry is displayed (5 widgets plus the
            # combo's companion ``_text`` entry from the production collector).
            self.assertEqual(len(rows), len(expected_widgets))
            self.assertEqual(rows["run_duration_spin"], ("3600", "QSpinBox"))
            self.assertEqual(rows["cfl_spin"], ("0.45", "QDoubleSpinBox"))
            self.assertEqual(rows["spatial_scheme_combo"], ("1", "QComboBox"))
            self.assertEqual(
                rows["spatial_scheme_combo_text"], ("MUSCL", "QComboBox_text")
            )
            self.assertEqual(rows["rain_check"], ("True", "QCheckBox"))
            self.assertEqual(
                rows["title_edit"], ("sentinel config title", "QLineEdit")
            )

            self.assertTrue(
                grab_non_empty(dlg), "config viewer dialog grabbed blank"
            )

    def test_switching_configs_updates_table(self):
        with make_temp_results_gpkg() as gpkg_path:
            state_a = self._collect_real_widget_state()
            state_b = self._collect_real_widget_state()
            # Give config B a distinguishing value.
            state_b["widgets"]["run_duration_spin"]["value"] = 7200
            state_b["widgets"]["title_edit"]["value"] = "sentinel beta title"

            self._persist_config(
                gpkg_path, "cfg_sentinel_alpha", "mesh_alpha", 3600.0, state_a
            )
            self._persist_config(
                gpkg_path, "cfg_sentinel_beta", "mesh_beta", 7200.0, state_b
            )

            dlg = self._make_dialog(gpkg_path)
            self.assertEqual(dlg.config_combo.count(), 2)

            # ORDER BY created_utc DESC → the most recently saved config
            # (beta) is selected first.
            self.assertIn("cfg_sentinel_beta", dlg.config_combo.currentText())
            self.assertIn("mesh_beta", dlg.meta_lbl.text())
            rows = _table_row_map(dlg.table)
            self.assertEqual(rows["run_duration_spin"], ("7200", "QSpinBox"))
            self.assertEqual(
                rows["title_edit"], ("sentinel beta title", "QLineEdit")
            )

            dlg.config_combo.setCurrentIndex(1)
            self._qt_widgets.QApplication.processEvents()
            self.assertIn("cfg_sentinel_alpha", dlg.config_combo.currentText())
            self.assertIn("mesh_alpha", dlg.meta_lbl.text())
            rows = _table_row_map(dlg.table)
            self.assertEqual(rows["run_duration_spin"], ("3600", "QSpinBox"))
            self.assertEqual(
                rows["title_edit"], ("sentinel config title", "QLineEdit")
            )

    def test_gpkg_without_configs_shows_placeholder(self):
        with make_temp_results_gpkg() as gpkg_path:
            dlg = self._make_dialog(gpkg_path)
            self.assertEqual(dlg.config_combo.count(), 1)
            self.assertEqual(dlg.config_combo.currentText(), "(no configs found)")
            self.assertFalse(dlg.config_combo.isEnabled())
            self.assertEqual(dlg.table.rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
