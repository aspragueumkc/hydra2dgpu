"""Behavioral tests for swe2d/workbench/dialogs/hydrograph_editor.py.

Task A.3 of docs/plans/2026-08-02-gui-test-coverage.md, per
docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4 (patterns P2 + P3).

Characterization notes:

* ``HydrographEditorDialog`` validates the serialized table through the
  production ``parse_hydrograph_text`` service before accepting.  Invalid
  input must show a warning and leave the dialog open.
* ``_load_text`` silently drops chunks that contain neither "," nor "="
  (no error surfaced).  Also asserted as a characterization finding.
"""

from __future__ import annotations

import unittest
from unittest import mock

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    requires_qgis,
)


@requires_qgis
class TestHydrographEditorDialog(unittest.TestCase):
    """P2 dialog-workflow tests against real widgets, offscreen QGIS app."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        from qgis.PyQt import QtWidgets  # noqa: F401  (env sanity)

    def _make_dialog(self, side="west", initial_text=""):
        from swe2d.workbench.dialogs.hydrograph_editor import (
            HydrographEditorDialog,
        )

        dlg = HydrographEditorDialog(side, initial_text)
        self.addCleanup(self._destroy, dlg)
        return dlg

    @staticmethod
    def _destroy(dlg):
        delete_widgets_now(dlg)

    @staticmethod
    def _set_cell(dlg, row, col, text):
        from qgis.PyQt import QtWidgets

        dlg.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

    def _cell(self, dlg, row, col):
        item = dlg.table.item(row, col)
        return item.text() if item is not None else None

    # -- construction -----------------------------------------------------

    def test_dialog_constructs_with_initial_text(self):
        dlg = self._make_dialog("west", "0,0; 1:00,5.5")
        self.assertIn("West", dlg.windowTitle())
        # Two parsed rows, no extra empty row appended.
        self.assertEqual(dlg.table.rowCount(), 2)
        self.assertEqual(self._cell(dlg, 0, 0), "0")
        self.assertEqual(self._cell(dlg, 0, 1), "0")
        self.assertEqual(self._cell(dlg, 1, 0), "1:00")
        self.assertEqual(self._cell(dlg, 1, 1), "5.5")

    def test_dialog_constructs_empty_adds_one_blank_row(self):
        dlg = self._make_dialog("east")
        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertEqual(self._cell(dlg, 0, 0), "")
        self.assertEqual(self._cell(dlg, 0, 1), "")

    # -- hydrograph_text() round-trip -------------------------------------

    def test_hydrograph_text_round_trips_exactly(self):
        initial = "0,0; 0.5,12.25; 1:30,3.5"
        dlg = self._make_dialog("north", initial)
        # serialize → re-parse into a fresh dialog → identical text.
        text1 = dlg.hydrograph_text()
        dlg2 = self._make_dialog("north", text1)
        text2 = dlg2.hydrograph_text()
        self.assertEqual(text1, text2)
        self.assertEqual(text1, "0,0; 0.5,12.25; 1:30,3.5")

    def test_hydrograph_text_reflects_widget_edits(self):
        dlg = self._make_dialog("south", "0,0")
        self._set_cell(dlg, 0, 0, "2:00")
        self._set_cell(dlg, 0, 1, "42.5")
        self.assertEqual(dlg.hydrograph_text(), "2:00,42.5")

    def test_hydrograph_text_skips_blank_rows(self):
        dlg = self._make_dialog("south", "0,0")
        dlg._add_row("", "")  # blank row must not appear in output
        self.assertEqual(dlg.hydrograph_text(), "0,0")

    # -- row buttons -------------------------------------------------------

    def test_add_and_remove_row_buttons(self):
        dlg = self._make_dialog("west", "0,0")
        dlg.add_row_btn.click()
        self.assertEqual(dlg.table.rowCount(), 2)
        self._set_cell(dlg, 1, 0, "1.0")
        self._set_cell(dlg, 1, 1, "7.0")
        # Select row 1 and remove it via the button.
        dlg.table.selectRow(1)
        dlg.remove_row_btn.click()
        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertEqual(dlg.hydrograph_text(), "0,0")

    # -- P3: valid series through the production save/read path ------------

    def test_valid_series_accepted_and_readback_via_production_loader(self):
        """accept() → hydrograph_text() → production parse_hydrograph_text.

        The dialog itself persists nothing; its real save path is the
        caller reading hydrograph_text() after Accepted.  The production
        read path for that text is
        ``text_parser_service.parse_hydrograph_text``.
        """
        from qgis.PyQt.QtWidgets import QDialog

        from swe2d.workbench.services.text_parser_service import (
            parse_hydrograph_text,
        )

        dlg = self._make_dialog("west")
        self._set_cell(dlg, 0, 0, "0")
        self._set_cell(dlg, 0, 1, "0.0")
        dlg._add_row("1:00", "10.0")
        dlg._add_row("2:30", "4.5")

        dlg.accept()
        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)

        times_s, values = parse_hydrograph_text(dlg.hydrograph_text())
        self.assertEqual(list(times_s), [0.0, 3600.0, 9000.0])
        self.assertEqual(list(values), [0.0, 10.0, 4.5])

    # -- invalid input: visible dialog error -------------------------------

    def test_invalid_series_warns_and_stays_open(self):
        """Invalid table text is rejected by the dialog's accept path."""
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QDialog

        dlg = self._make_dialog("west")
        self._set_cell(dlg, 0, 0, "not-a-time")
        self._set_cell(dlg, 0, 1, "also-not-a-number")

        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            dlg.accept()

        warning.assert_called_once()
        self.assertIn("invalid", str(warning.call_args.args[-1]).lower())
        self.assertNotEqual(dlg.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(dlg.hydrograph_text(), "not-a-time,also-not-a-number")

    def test_load_text_silently_drops_separatorless_chunks(self):
        """FINDING: _load_text skips entries without ',' or '=' silently."""
        dlg = self._make_dialog("west", "0,0; garbage-chunk; 1.0,2.0")
        self.assertEqual(dlg.table.rowCount(), 2)
        self.assertEqual(dlg.hydrograph_text(), "0,0; 1.0,2.0")

    # -- render ------------------------------------------------------------

    def test_grab_non_empty(self):
        dlg = self._make_dialog("west", "0,0; 1:00,5.0")
        dlg.resize(560, 380)
        self.assertTrue(grab_non_empty(dlg))


if __name__ == "__main__":
    unittest.main()
