"""Behavioral tests for swe2d/workbench/dialogs/topo_attr_table_dialog.py.

Task A.4 of docs/plans/2026-08-02-gui-test-coverage.md, per
docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4 (patterns P2 + P3).

Characterization notes (read from production code, do not "fix" here):

* The dialog's only validation lives in ``_editor_value``: non-numeric text
  in an ``int``/``float`` cell raises ``ValueError``, which
  ``_save_and_accept`` catches, rolls the edit session back, and surfaces
  via ``QMessageBox.warning``.  The dialog stays open.  That is the real
  validation path and it is asserted below (the modal warning box is
  patched only to record the call — a Qt, not Qgs, type — so the test does
  not block offscreen).
* Error paths log through the module logger rather than calling an
  undefined ``_log`` attribute.  A missing field is used below to exercise
  the load-path logger without mocking QGIS objects.
* The "Remove Selected" tooltip promises feature deletion on save, so the
  mutation test below verifies that the removed feature is gone after save.
"""

from __future__ import annotations

import unittest
from unittest import mock

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_memory_layer,
    make_temp_model_gpkg,
    requires_qgis,
)

# Production field spec for the topology regions dialog, copied from
# SWE2DWorkbenchStudioDialog._open_topology_region_table
# (swe2d/workbench/studio_dialog.py).
def _region_field_specs():
    from swe2d.core.constants_service import CELL_TYPE_OPTIONS

    return [
        ("region_id", "Region ID", "int"),
        ("target_size", "Target Size", "float"),
        ("cell_type", "Cell Type", "enum", CELL_TYPE_OPTIONS),
        ("edge_len_1", "Edge Len 1", "float"),
        ("edge_len_2", "Edge Len 2", "float"),
        ("edge_len_3", "Edge Len 3", "float"),
        ("edge_len_4", "Edge Len 4", "float"),
    ]


def _region_qfields():
    from qgis.PyQt.QtCore import QVariant

    return [
        ("region_id", QVariant.Int),
        ("target_size", QVariant.Double),
        ("cell_type", QVariant.String),
        ("edge_len_1", QVariant.Double),
        ("edge_len_2", QVariant.Double),
        ("edge_len_3", QVariant.Double),
        ("edge_len_4", QVariant.Double),
    ]


_POLY = "Polygon ((0 0, 10 0, 10 10, 0 10, 0 0))"


def _make_region_layer(features=()):
    """Real memory layer matching the swe2d_topo_regions schema."""
    return make_memory_layer(
        geometry="Polygon",
        fields=_region_qfields(),
        features=features,
        name="swe2d_topo_regions",
    )


def _feature_by_attr(layer, field, value):
    for ft in layer.getFeatures():
        if ft[field] == value:
            return ft
    return None


@requires_qgis
class TestTopologyAttributeTableDialog(unittest.TestCase):
    """P2 dialog-workflow + P3 edit-persist tests, real widgets/layers."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        from qgis.PyQt import QtWidgets  # noqa: F401  (env sanity)

    def _make_dialog(self, layer, **kwargs):
        from swe2d.workbench.dialogs.topo_attr_table_dialog import (
            TopologyAttributeTableDialog,
        )

        dlg = TopologyAttributeTableDialog(
            layer,
            "Topology Region Controls",
            _region_field_specs(),
            sort_fields=["region_id"],
            **kwargs,
        )
        self.addCleanup(self._destroy, dlg)
        return dlg

    @staticmethod
    def _destroy(dlg):
        delete_widgets_now(dlg)

    @staticmethod
    def _set_cell(dlg, row, col, text):
        from qgis.PyQt import QtWidgets

        dlg.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

    @staticmethod
    def _click_ok(dlg):
        from qgis.PyQt import QtWidgets

        box = dlg.findChild(QtWidgets.QDialogButtonBox)
        assert box is not None, "dialog has no QDialogButtonBox"
        box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    # -- construction / population -----------------------------------------

    def test_constructs_and_populates_sorted_rows(self):
        from qgis.PyQt import QtWidgets

        layer = _make_region_layer(
            features=[
                (_POLY, (2, 8.0, "quadrilateral", 1.0, 2.0, 3.0, 4.0)),
                (_POLY, (1, 5.0, "triangular", 4.0, 3.0, 2.0, 1.0)),
            ]
        )
        dlg = self._make_dialog(layer)

        self.assertEqual(dlg.windowTitle(), "Topology Region Controls")
        headers = [
            dlg.table.horizontalHeaderItem(c).text()
            for c in range(dlg.table.columnCount())
        ]
        self.assertEqual(
            headers,
            ["Region ID", "Target Size", "Cell Type", "Edge Len 1",
             "Edge Len 2", "Edge Len 3", "Edge Len 4"],
        )
        # Two real feature rows, sorted ascending by region_id.
        self.assertEqual(dlg.table.rowCount(), 2)
        self.assertEqual(dlg.table.item(0, 0).text(), "1")
        self.assertEqual(dlg.table.item(1, 0).text(), "2")
        self.assertEqual(dlg.table.item(0, 1).text(), "5.0")
        self.assertEqual(dlg.table.item(1, 1).text(), "8.0")
        # Enum column uses a real combo seeded from the feature value.
        combo0 = dlg.table.cellWidget(0, 2)
        combo1 = dlg.table.cellWidget(1, 2)
        self.assertIsInstance(combo0, QtWidgets.QComboBox)
        self.assertEqual(combo0.currentData(), "triangular")
        self.assertEqual(combo1.currentData(), "quadrilateral")

    def test_reload_button_restores_layer_state(self):
        layer = _make_region_layer(
            features=[(_POLY, (1, 5.0, "triangular", 1.0, 1.0, 1.0, 1.0))]
        )
        dlg = self._make_dialog(layer)
        self._set_cell(dlg, 0, 1, "99.9")  # unsaved edit
        dlg.refresh_btn.click()
        self.assertEqual(dlg.table.item(0, 1).text(), "5.0")

    def test_add_and_remove_row_buttons(self):
        layer = _make_region_layer(
            features=[(_POLY, (1, 5.0, "triangular", 1.0, 1.0, 1.0, 1.0))]
        )
        dlg = self._make_dialog(layer)

        dlg.add_row_btn.click()
        self.assertEqual(dlg.table.rowCount(), 2)
        self.assertEqual(dlg._row_feature_ids, [1, -1])

        dlg.table.selectRow(1)
        dlg.remove_row_btn.click()
        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertEqual(dlg._row_feature_ids, [1])

    # -- P3: edit → real save path → production readback --------------------

    def test_edit_and_add_row_persist_to_layer(self):
        """Edit cells + add a row, click OK, re-read the layer itself."""
        from qgis.PyQt.QtWidgets import QDialog

        layer = _make_region_layer(
            features=[(_POLY, (1, 5.0, "triangular", 1.0, 2.0, 3.0, 4.0))]
        )
        dlg = self._make_dialog(layer)

        # Edit existing row: float cell + enum combo.
        self._set_cell(dlg, 0, 1, "12.5")
        dlg.table.cellWidget(0, 2).setCurrentText("cartesian")

        # Add a brand-new feature through the dialog's blank-row path.
        dlg.add_row_btn.click()
        self._set_cell(dlg, 1, 0, "7")
        self._set_cell(dlg, 1, 1, "2.5")

        self._click_ok(dlg)
        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)

        # Readback from the layer the dialog just committed to.
        ft1 = _feature_by_attr(layer, "region_id", 1)
        self.assertIsNotNone(ft1)
        self.assertAlmostEqual(float(ft1["target_size"]), 12.5)
        self.assertEqual(ft1["cell_type"], "cartesian")
        self.assertAlmostEqual(float(ft1["edge_len_1"]), 1.0)

        ft7 = _feature_by_attr(layer, "region_id", 7)
        self.assertIsNotNone(ft7, "added row never reached the layer")
        self.assertAlmostEqual(float(ft7["target_size"]), 2.5)

    def test_edit_persists_to_gpkg_readback_via_production_loader(self):
        """P3 end-to-end: real model GPKG, edit, save, reload via the
        production loader and assert the mutation survived the round-trip.
        """
        from qgis.core import QgsFeature, QgsGeometry

        from swe2d.workbench.services.model_gpkg_loader_service import (
            load_layers_from_gpkg,
        )

        with make_temp_model_gpkg() as gpkg_path:
            layers = load_layers_from_gpkg(gpkg_path)
            layer = layers["swe2d_topo_regions"]
            self.assertTrue(layer.isValid())

            # Seed one region feature at provider level (same as a user
            # digitizing a polygon and committing it in QGIS).  OGR layers
            # expose `fid` as field 0 and the regions schema also carries
            # `channel_generator_type` — supply all nine attributes.
            feat = QgsFeature(layer.fields())
            feat.setGeometry(QgsGeometry.fromWkt(_POLY))
            feat.setAttributes(
                [None, 1, 5.0, "triangular", None, 1.0, 2.0, 3.0, 4.0]
            )
            ok, _ = layer.dataProvider().addFeatures([feat])
            self.assertTrue(ok, "provider refused the seed feature")

            dlg = self._make_dialog(layer)
            self.assertEqual(dlg.table.rowCount(), 1)
            self._set_cell(dlg, 0, 1, "12.5")
            dlg.table.cellWidget(0, 2).setCurrentText("quadrilateral")
            self._click_ok(dlg)

            # Fresh read through the exact loader the workbench uses.
            reloaded = load_layers_from_gpkg(gpkg_path)["swe2d_topo_regions"]
            ft = _feature_by_attr(reloaded, "region_id", 1)
            self.assertIsNotNone(ft, "edited feature missing from GPKG reload")
            self.assertAlmostEqual(float(ft["target_size"]), 12.5)
            self.assertEqual(ft["cell_type"], "quadrilateral")

    # -- invalid input: the real validation path ----------------------------

    def test_invalid_value_warns_rolls_back_and_stays_open(self):
        """Non-numeric text in an int cell must NOT be silently accepted:
        _editor_value raises, _save_and_accept rolls back and warns, the
        dialog stays open, and the layer value is unchanged.
        """
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtWidgets import QDialog

        layer = _make_region_layer(
            features=[(_POLY, (1, 5.0, "triangular", 1.0, 2.0, 3.0, 4.0))]
        )
        dlg = self._make_dialog(layer)
        self._set_cell(dlg, 0, 0, "not-a-number")

        # Patch the modal warning box only to record the call (Qt type, not
        # Qgs) — otherwise the static exec would block the offscreen test.
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warn:
            self._click_ok(dlg)

        warn.assert_called_once()
        self.assertIn("Failed to save", str(warn.call_args.args[-1]))
        self.assertNotEqual(dlg.result(), QDialog.DialogCode.Accepted)

        ft = _feature_by_attr(layer, "region_id", 1)
        self.assertIsNotNone(ft, "rollback lost the original feature")
        self.assertAlmostEqual(float(ft["target_size"]), 5.0)

    def test_missing_field_logs_without_raising(self):
        """A missing source field is logged and still produces a row."""
        from qgis.PyQt.QtCore import QVariant
        from swe2d.workbench.dialogs.topo_attr_table_dialog import (
            TopologyAttributeTableDialog,
        )

        layer = make_memory_layer(
            geometry="Polygon",
            fields=[("region_id", QVariant.Int)],
            features=[(_POLY, (1,))],
            name="missing_field_source",
        )
        with self.assertLogs(
            "swe2d.workbench.dialogs.topo_attr_table_dialog", level="WARNING"
        ) as logs:
            dlg = TopologyAttributeTableDialog(
                layer,
                "Topology Region Controls",
                [("missing_field", "Missing Field", "float")],
            )
        self.addCleanup(self._destroy, dlg)

        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertTrue(
            any("Exception parsing feature value" in message for message in logs.output)
        )

    # -- removed rows are deleted on save -----------------------------------

    def test_removed_row_feature_deleted_on_save(self):
        """Removing a loaded row deletes that feature when saved."""
        from qgis.PyQt.QtWidgets import QDialog

        layer = _make_region_layer(
            features=[
                (_POLY, (1, 5.0, "triangular", 1.0, 2.0, 3.0, 4.0)),
                (_POLY, (2, 8.0, "triangular", 1.0, 2.0, 3.0, 4.0)),
            ]
        )
        dlg = self._make_dialog(layer)
        dlg.table.selectRow(1)  # region_id 2
        dlg.remove_row_btn.click()
        self.assertEqual(dlg.table.rowCount(), 1)

        self._click_ok(dlg)
        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(layer.featureCount(), 1)
        self.assertIsNone(_feature_by_attr(layer, "region_id", 2))

    # -- render --------------------------------------------------------------

    def test_grab_non_empty(self):
        layer = _make_region_layer(
            features=[(_POLY, (1, 5.0, "triangular", 1.0, 2.0, 3.0, 4.0))]
        )
        dlg = self._make_dialog(layer)
        dlg.show()
        self.assertTrue(grab_non_empty(dlg))


if __name__ == "__main__":
    unittest.main()
