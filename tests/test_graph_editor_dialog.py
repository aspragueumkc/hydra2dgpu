#!/usr/bin/env python3
"""Behavioral tests for swe2d/workbench/views/graph_editor_dialog.py.

Covers GraphEditorDialog (graph CRUD against a real model GPKG),
_ColumnPicker (CSV column selection), and ApplyGraphDialog (apply a graph
id to a feature in a target layer) per the coverage standard in
docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4 (patterns P2 + P3).

Everything runs against the REAL dialog, REAL widgets, and a REAL model
GeoPackage built through the production creation path
(``tests.qgis_real_env.make_temp_model_gpkg``).  Mutations are verified by
reading back through the production read paths
(``graph_editor_service.load_graphs`` / the dialog's own ``_open_layer``) —
never by inspecting write calls.  No mocks of Qgs* types.
"""

from __future__ import annotations

import os
import sys
import unittest

# Ensure repo root and build dir are on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_temp_model_gpkg,
    requires_qgis,
)

from swe2d.workbench.services.graph_editor_service import (
    load_graphs,
    save_hydrograph,
    save_hyetograph,
)


_PENDING_MODAL_STATES = []


def _pump_events() -> None:
    """Process events and synchronously surface armed-modal failures."""
    import time

    from qgis.PyQt.QtWidgets import QApplication

    while _PENDING_MODAL_STATES:
        QApplication.processEvents()
        QApplication.sendPostedEvents()
        for state in list(_PENDING_MODAL_STATES):
            if not state["done"]:
                continue
            _PENDING_MODAL_STATES.remove(state)
            if state["error"] is not None:
                raise state["error"]
        if not _PENDING_MODAL_STATES:
            return
        now = time.monotonic()
        for state in _PENDING_MODAL_STATES:
            if now >= state["deadline"] and not state["done"]:
                state["error"] = RuntimeError(state["timeout_message"])
                state["done"] = True
        if _PENDING_MODAL_STATES:
            time.sleep(0.001)

    QApplication.processEvents()
    QApplication.sendPostedEvents()


def _when_modal(
    action,
    *,
    max_attempts: int = 200,
    deadline_ms: int = 5000,
) -> None:
    """Run ``action(widget)`` on the next active modal widget.

    Modal ``exec()`` loops run their own event loop, so a zero-delay
    ``QTimer`` fires inside them.  The timer callback records failures instead
    of raising across the Qt boundary; ``_pump_events`` re-raises them in the
    test thread, where unittest can report them normally.
    """
    import time

    from qgis.PyQt.QtCore import QTimer
    from qgis.PyQt.QtWidgets import QApplication

    timeout_message = (
        f"_when_modal: no modal widget appeared within {deadline_ms} ms "
        f"(limit {max_attempts} polls); test is hung on a missing "
        "exec()/QMessageBox — check the call site that should have "
        "produced it."
    )
    state = {
        "n": 0,
        "deadline": time.monotonic() + deadline_ms / 1000.0,
        "done": False,
        "error": None,
        "timeout_message": timeout_message,
    }
    _PENDING_MODAL_STATES.append(state)

    def _try() -> None:
        if state["done"]:
            return
        widget = QApplication.activeModalWidget()
        if widget is not None:
            try:
                action(widget)
            except BaseException as exc:
                state["error"] = exc
            finally:
                state["done"] = True
            return
        state["n"] += 1
        if state["n"] >= max_attempts or time.monotonic() >= state["deadline"]:
            state["error"] = RuntimeError(state["timeout_message"])
            state["done"] = True
            return
        QTimer.singleShot(10, _try)

    QTimer.singleShot(0, _try)


def _accept_modal(widget) -> None:
    widget.accept()


def _click_yes(widget) -> None:
    from qgis.PyQt.QtWidgets import QMessageBox

    button = widget.button(QMessageBox.Yes)
    assert button is not None, "expected a Yes button on the modal dialog"
    button.click()


def _click_close_save(widget) -> None:
    from qgis.PyQt.QtWidgets import QMessageBox

    button = widget.button(QMessageBox.Save)
    assert button is not None, "expected a Save button on the unsaved-changes modal"
    button.click()


def _click_close_discard(widget) -> None:
    from qgis.PyQt.QtWidgets import QMessageBox

    button = widget.button(QMessageBox.Discard)
    assert button is not None, "expected a Discard button on the unsaved-changes modal"
    button.click()


def _click_close_cancel(widget) -> None:
    from qgis.PyQt.QtWidgets import QMessageBox

    button = widget.button(QMessageBox.Cancel)
    assert button is not None, "expected a Cancel button on the unsaved-changes modal"
    button.click()


def _seed_graphs(gpkg_path: str) -> None:
    """Seed one hyetograph and one hydrograph via the production write path."""
    save_hyetograph(
        gpkg_path,
        "rain1",
        [(0.0, 5.0), (2.25, 1.0)],
        value_type="intensity",
        units="mm/hr",
    )
    save_hydrograph(
        gpkg_path,
        "inflow1",
        [(0.0, 0.0), (2.0, 5.0)],
        bc_type=102,
        description="test inflow",
    )


def _list_entries(dlg) -> list:
    """Return the (graph_type, gid) UserRole payloads in the graph list."""
    from qgis.PyQt.QtCore import Qt

    entries = []
    for i in range(dlg._graph_list.count()):
        data = dlg._graph_list.item(i).data(Qt.UserRole)
        if data is not None:
            entries.append(tuple(data))
    return entries


def _select_graph(dlg, graph_type: str, gid: str) -> None:
    """Select a graph entry in the dialog's list widget (drives the signal)."""
    from qgis.PyQt.QtCore import Qt

    for i in range(dlg._graph_list.count()):
        item = dlg._graph_list.item(i)
        if tuple(item.data(Qt.UserRole) or ()) == (graph_type, gid):
            dlg._graph_list.setCurrentItem(item)
            _pump_events()
            return
    raise AssertionError(f"graph {graph_type}/{gid!r} not in dialog list")


def _add_feature(gpkg_path: str, layer_name: str, wkt: str, attrs: dict) -> int:
    """Add one feature to a model-GPKG layer via a real QgsVectorLayer.

    Returns the committed feature id.
    """
    from qgis.core import QgsFeature, QgsGeometry

    from swe2d.workbench.views.graph_editor_dialog import _open_layer

    layer = _open_layer(gpkg_path, layer_name)
    assert layer is not None, f"could not open layer {layer_name}"
    feat = QgsFeature(layer.fields())
    feat.setGeometry(QgsGeometry.fromWkt(wkt))
    for name, value in attrs.items():
        feat[name] = value
    if not layer.startEditing():
        raise AssertionError(f"startEditing failed on {layer_name}")
    if not layer.addFeature(feat):
        layer.rollBack()
        raise AssertionError(f"addFeature failed on {layer_name}")
    if not layer.commitChanges():
        raise AssertionError(
            f"commitChanges failed on {layer_name}: {layer.commitErrors()}"
        )
    fid = next(iter(layer.getFeatures())).id()
    # Release the OGR handle before any sqlite3 access to the same GPKG.
    del layer
    return fid


@requires_qgis
class TestGraphEditorDialog(unittest.TestCase):
    """P2+P3 coverage of GraphEditorDialog against a real model GPKG."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        self._gpkg_ctx = make_temp_model_gpkg()
        self.gpkg_path = self._gpkg_ctx.__enter__()
        self._dialogs = []

    def tearDown(self) -> None:
        for dlg in self._dialogs:
            dlg._dirty = False  # avoid the unsaved-changes modal in cleanup
        delete_widgets_now(*self._dialogs)
        self._dialogs.clear()
        self._gpkg_ctx.__exit__(None, None, None)

    def _make_dialog(self):
        from swe2d.workbench.views.graph_editor_dialog import GraphEditorDialog

        dlg = GraphEditorDialog(self.gpkg_path)
        self._dialogs.append(dlg)
        return dlg

    # ── P2: construction + list population ───────────────────────────

    def test_constructs_and_lists_seeded_graphs(self) -> None:
        _seed_graphs(self.gpkg_path)
        dlg = self._make_dialog()

        self.assertEqual(
            _list_entries(dlg),
            [("hyetographs", "rain1"), ("hydrographs", "inflow1")],
        )

        dlg.show()
        _pump_events()
        self.assertTrue(
            grab_non_empty(dlg),
            "GraphEditorDialog rendered a blank image after show()",
        )

    # ── P2: selection drives the editor widgets ──────────────────────

    def test_selection_populates_editor_fields(self) -> None:
        _seed_graphs(self.gpkg_path)
        dlg = self._make_dialog()

        _select_graph(dlg, "hyetographs", "rain1")
        self.assertEqual(dlg._current_type, "hyetographs")
        self.assertEqual(dlg._current_id, "rain1")
        self.assertEqual(dlg._name_edit.text(), "rain1")
        self.assertEqual(dlg._type_label.text(), "hyetograph")
        self.assertEqual(dlg._vt_combo.currentText(), "intensity")
        self.assertEqual(dlg._units_combo.currentText(), "mm/hr")
        self.assertTrue(dlg._vt_combo.isVisibleTo(dlg))
        self.assertFalse(dlg._bc_combo.isVisibleTo(dlg))
        # Time rendering: 0.0 h -> "0:00", 2.25 h -> "2:15".
        self.assertEqual(dlg._table.item(0, 0).text(), "0:00")
        self.assertEqual(dlg._table.item(0, 1).text(), "5.0")
        self.assertEqual(dlg._table.item(1, 0).text(), "2:15")
        self.assertEqual(dlg._table.item(1, 1).text(), "1.0")

        _select_graph(dlg, "hydrographs", "inflow1")
        self.assertEqual(dlg._current_type, "hydrographs")
        self.assertEqual(dlg._name_edit.text(), "inflow1")
        self.assertEqual(dlg._type_label.text(), "hydrograph")
        self.assertEqual(dlg._bc_combo.currentData(), 102)
        self.assertEqual(dlg._desc_edit.text(), "test inflow")
        self.assertTrue(dlg._bc_combo.isVisibleTo(dlg))
        self.assertFalse(dlg._vt_combo.isVisibleTo(dlg))

    # ── P3: add a hyetograph through the UI, verify by readback ─────

    def test_add_hyetograph_end_to_end(self) -> None:
        from qgis.PyQt.QtWidgets import QDialogButtonBox

        dlg = self._make_dialog()

        # "+ New" opens a modal type picker; default selection is
        # "hyetograph".  Accept it on the modal's own event loop.
        _when_modal(_accept_modal)
        dlg._new_btn.click()
        _pump_events()
        self.assertEqual(dlg._current_type, "hyetographs")
        self.assertIsNone(dlg._current_id)

        dlg._name_edit.setText("storm_x")
        dlg._vt_combo.setCurrentText("cumulative")
        dlg._units_combo.setCurrentText("in/hr")
        # Blank-row table: type into the first two rows.  "1:30" exercises
        # the HH:MM time parser.
        dlg._table.item(0, 0).setText("0")
        dlg._table.item(0, 1).setText("12.5")
        dlg._table.item(1, 0).setText("1:30")
        dlg._table.item(1, 1).setText("3.0")

        bbox = dlg.findChild(QDialogButtonBox)
        self.assertIsNotNone(bbox, "dialog has no QDialogButtonBox")
        bbox.button(QDialogButtonBox.Save).click()
        _pump_events()

        # Readback through the production loader — this is the contract.
        graphs = load_graphs(self.gpkg_path)
        self.assertIn("storm_x", graphs["hyetographs"])
        saved = graphs["hyetographs"]["storm_x"]
        self.assertEqual(saved["data"], [(0.0, 12.5), (1.5, 3.0)])
        self.assertEqual(saved["value_type"], "cumulative")
        self.assertEqual(saved["units"], "in/hr")
        # The saved graph is selected in the list.
        self.assertIn(("hyetographs", "storm_x"), _list_entries(dlg))
        self.assertEqual(dlg._current_id, "storm_x")

    # ── P3: delete a graph through the UI, verify by readback ───────

    def test_delete_graph_end_to_end(self) -> None:
        _seed_graphs(self.gpkg_path)
        dlg = self._make_dialog()

        _select_graph(dlg, "hydrographs", "inflow1")
        _when_modal(_click_yes)
        dlg._delete_btn.click()
        _pump_events()

        graphs = load_graphs(self.gpkg_path)
        self.assertNotIn("inflow1", graphs["hydrographs"])
        self.assertIn("rain1", graphs["hyetographs"])
        self.assertNotIn(("hydrographs", "inflow1"), _list_entries(dlg))
        self.assertIsNone(dlg._current_id)

    # ── closeEvent — Save / Discard / Cancel dialog contract ───────

    def test_close_event_dirty_save_persists_changes(self) -> None:
        """Closing a dirty dialog and answering Save persists the pending
        edit (driven through the real Save modal that closeEvent shows)."""
        dlg = self._make_dialog()
        # "+ New" → hyetograph; type a name + one row; do NOT click Save.
        _when_modal(_accept_modal)
        dlg._new_btn.click()
        _pump_events()
        self.assertTrue(dlg._dirty, "+ New should leave the dialog dirty")
        dlg._name_edit.setText("dirty_close_save")
        dlg._table.item(0, 0).setText("0")
        dlg._table.item(0, 1).setText("4.2")

        # Close the dialog while dirty — answer "Save" on the modal.
        _when_modal(_click_close_save)
        dlg.close()
        _pump_events()

        # Save path writes through _on_save; readback via the production
        # loader proves the dirty edit was actually persisted.
        graphs = load_graphs(self.gpkg_path)
        self.assertIn("dirty_close_save", graphs["hyetographs"])
        self.assertEqual(
            graphs["hyetographs"]["dirty_close_save"]["data"], [(0.0, 4.2)]
        )
        # After Save, the dirty flag clears — no further modal would fire.
        self.assertFalse(dlg._dirty)

    def test_close_event_dirty_discard_drops_changes(self) -> None:
        """Closing a dirty dialog and answering Discard drops the edit;
        the dialog is no longer dirty and the change never persists."""
        dlg = self._make_dialog()
        _when_modal(_accept_modal)
        dlg._new_btn.click()
        _pump_events()
        self.assertTrue(dlg._dirty)
        dlg._name_edit.setText("dirty_close_discard")
        dlg._table.item(0, 0).setText("0")
        dlg._table.item(0, 1).setText("9.9")

        _when_modal(_click_close_discard)
        dlg.close()
        _pump_events()

        graphs = load_graphs(self.gpkg_path)
        self.assertNotIn(
            "dirty_close_discard", graphs["hyetographs"],
            "Discard must NOT write the pending edit to the model GPKG",
        )
        # _dirty stays True because Discard doesn't re-clear it, but the
        # dialog is gone — verify through the visible state.
        self.assertFalse(dlg.isVisible())

    def test_close_event_dirty_cancel_keeps_dialog_open(self) -> None:
        """Closing a dirty dialog and answering Cancel ignores the close
        event — the dialog stays visible and no state is persisted."""
        dlg = self._make_dialog()
        dlg.show()
        _when_modal(_accept_modal)
        dlg._new_btn.click()
        _pump_events()
        self.assertTrue(dlg._dirty)
        dlg._name_edit.setText("dirty_close_cancel")
        dlg._table.item(0, 0).setText("0")
        dlg._table.item(0, 1).setText("7.7")
        self.assertTrue(dlg.isVisible(), "dialog must be visible before close()")

        _when_modal(_click_close_cancel)
        dlg.close()
        _pump_events()

        # Dialog remains visible (close was ignored).
        self.assertTrue(dlg.isVisible())
        # State is intact — name and value unchanged.
        self.assertEqual(dlg._name_edit.text(), "dirty_close_cancel")
        self.assertEqual(dlg._table.item(0, 1).text(), "7.7")
        # Nothing was written to disk.
        graphs = load_graphs(self.gpkg_path)
        self.assertNotIn("dirty_close_cancel", graphs["hyetographs"])

    def test_close_event_clean_no_modal(self) -> None:
        """Closing a non-dirty dialog must not show any modal — closeEvent
        calls super() and the dialog hides cleanly."""
        _seed_graphs(self.gpkg_path)
        dlg = self._make_dialog()
        # No edits — dialog is clean.
        self.assertFalse(dlg._dirty)
        dlg.show()
        _pump_events()

        # With no modal helper armed, any QMessageBox raised during close
        # would block forever.  Use the deadline-bounded helper with a
        # very short wait and assert that NO modal appeared.  The helper
        # raises RuntimeError on deadline — which is the SUCCESS path
        # here (no modal appeared), so swallow it and rely on the flag.
        saw_modal = {"flag": False}

        def _capture(widget):
            saw_modal["flag"] = True
            widget.button(0x40000)  # QMessageBox.Save — click Save if it does appear
            widget.button(0x40000).click()

        try:
            _when_modal(_capture, deadline_ms=200)
            dlg.close()
            _pump_events()
        except RuntimeError:
            # Expected: no modal within the deadline — that's the
            # contract for a clean dialog.
            pass

        self.assertFalse(
            saw_modal["flag"],
            "closeEvent must not show a modal on a clean dialog",
        )
        self.assertFalse(dlg.isVisible())


@requires_qgis
class TestColumnPicker(unittest.TestCase):
    """_ColumnPicker: selected_columns() reflects the combo selection."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def test_selected_columns_defaults_and_override(self) -> None:
        from swe2d.workbench.views.graph_editor_dialog import _ColumnPicker

        picker = _ColumnPicker(["t_hours", "q_cms", "notes"])
        try:
            # Default: time = first column, value = second column.
            self.assertEqual(
                picker.selected_columns(), ("t_hours", "q_cms")
            )
            picker._time_combo.setCurrentText("notes")
            picker._value_combo.setCurrentText("q_cms")
            self.assertEqual(picker.selected_columns(), ("notes", "q_cms"))
        finally:
            delete_widgets_now(picker)

    def test_single_column_defaults_to_same(self) -> None:
        from swe2d.workbench.views.graph_editor_dialog import _ColumnPicker

        picker = _ColumnPicker(["only_col"])
        try:
            self.assertEqual(
                picker.selected_columns(), ("only_col", "only_col")
            )
        finally:
            delete_widgets_now(picker)


@requires_qgis
class TestApplyGraphDialog(unittest.TestCase):
    """ApplyGraphDialog: target contract + end-to-end apply with readback."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        self._gpkg_ctx = make_temp_model_gpkg()
        self.gpkg_path = self._gpkg_ctx.__enter__()
        _seed_graphs(self.gpkg_path)
        self._dialogs = []

    def tearDown(self) -> None:
        delete_widgets_now(*self._dialogs)
        self._dialogs.clear()
        self._gpkg_ctx.__exit__(None, None, None)

    def _make_apply_dialog(self, graph_type: str, graph_id: str):
        from swe2d.workbench.views.graph_editor_dialog import ApplyGraphDialog

        dlg = ApplyGraphDialog(self.gpkg_path, graph_type, graph_id)
        self._dialogs.append(dlg)
        return dlg

    def test_targets_match_apply_contract(self) -> None:
        from swe2d.workbench.views.graph_editor_dialog import _APPLY_TARGETS

        hyeto_dlg = self._make_apply_dialog("hyetographs", "rain1")
        hyeto_targets = [
            hyeto_dlg._layer_combo.itemData(i)
            for i in range(hyeto_dlg._layer_combo.count())
        ]
        self.assertEqual(hyeto_targets, _APPLY_TARGETS["hyetographs"])
        self.assertEqual(
            hyeto_targets, [("swe2d_rain_gages", "hyetograph_id")]
        )

        hydro_dlg = self._make_apply_dialog("hydrographs", "inflow1")
        hydro_targets = [
            hydro_dlg._layer_combo.itemData(i)
            for i in range(hydro_dlg._layer_combo.count())
        ]
        self.assertEqual(hydro_targets, _APPLY_TARGETS["hydrographs"])
        self.assertEqual(
            hydro_targets,
            [
                ("swe2d_bc_lines", "hydrograph_id"),
                ("swe2d_internal_flow_sources", "hydrograph_id"),
            ],
        )

    def test_apply_hyetograph_to_rain_gage_end_to_end(self) -> None:
        from swe2d.workbench.views.graph_editor_dialog import _open_layer

        fid = _add_feature(
            self.gpkg_path,
            "swe2d_rain_gages",
            "POINT (1 1)",
            {"gage_id": "G1", "name": "Gage One"},
        )
        dlg = self._make_apply_dialog("hyetographs", "rain1")

        # Feature table shows the seeded gage with an empty current value.
        self.assertEqual(dlg._feature_table.rowCount(), 1)
        self.assertEqual(dlg._feature_table.item(0, 0).text(), str(fid))
        self.assertEqual(dlg._feature_table.item(0, 1).text(), "")
        self.assertIn("gage_id=G1", dlg._feature_table.item(0, 2).text())

        dlg._feature_table.selectRow(0)
        _when_modal(_accept_modal)  # dismiss the success QMessageBox
        dlg._on_accept()
        _pump_events()

        # Readback through the dialog's own production layer reader.
        layer = _open_layer(self.gpkg_path, "swe2d_rain_gages")
        self.assertIsNotNone(layer)
        feat = layer.getFeature(fid)
        self.assertTrue(feat.isValid())
        self.assertEqual(feat["hyetograph_id"], "rain1")

    def test_apply_hydrograph_to_bc_line_end_to_end(self) -> None:
        from swe2d.workbench.views.graph_editor_dialog import _open_layer

        fid = _add_feature(
            self.gpkg_path,
            "swe2d_bc_lines",
            "LINESTRING (0 0, 1 1)",
            {"bc_type": 102},
        )
        dlg = self._make_apply_dialog("hydrographs", "inflow1")

        # First target is swe2d_bc_lines; the notes column shows bc_type.
        self.assertEqual(
            dlg._selected_target(), ("swe2d_bc_lines", "hydrograph_id")
        )
        self.assertEqual(dlg._feature_table.rowCount(), 1)
        self.assertIn("bc_type=102", dlg._feature_table.item(0, 2).text())

        dlg._feature_table.selectRow(0)
        _when_modal(_accept_modal)
        dlg._on_accept()
        _pump_events()

        layer = _open_layer(self.gpkg_path, "swe2d_bc_lines")
        self.assertIsNotNone(layer)
        feat = layer.getFeature(fid)
        self.assertTrue(feat.isValid())
        self.assertEqual(feat["hydrograph_id"], "inflow1")


if __name__ == "__main__":
    unittest.main()
