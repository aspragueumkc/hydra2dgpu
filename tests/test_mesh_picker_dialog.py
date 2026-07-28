import os
import sqlite3
import tempfile
import unittest
from qgis.PyQt import QtWidgets

from swe2d.workbench.dialogs.mesh_picker_dialog import MeshPickerDialog


class TestMeshPickerDialog(unittest.TestCase):
    def test_lists_mesh_names_from_gpkg(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpkg = os.path.join(tmp, "test.gpkg")
            conn = sqlite3.connect(gpkg)
            conn.execute(
                "CREATE TABLE swe2d_baked_mesh (mesh_name TEXT, created_utc TEXT)"
            )
            conn.execute(
                "INSERT INTO swe2d_baked_mesh VALUES (?, ?)",
                ("mesh_a", "2026-01-01T00:00:00Z"),
            )
            conn.execute(
                "INSERT INTO swe2d_baked_mesh VALUES (?, ?)",
                ("mesh_b", "2026-01-02T00:00:00Z"),
            )
            conn.commit()
            conn.close()

            _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dlg = MeshPickerDialog(gpkg)
            names = [
                dlg._list.item(i).text() for i in range(dlg._list.count())
            ]
            self.assertEqual(names, ["mesh_b", "mesh_a"])

    def test_returns_selected_mesh_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpkg = os.path.join(tmp, "test.gpkg")
            conn = sqlite3.connect(gpkg)
            conn.execute(
                "CREATE TABLE swe2d_baked_mesh (mesh_name TEXT, created_utc TEXT)"
            )
            conn.execute(
                "INSERT INTO swe2d_baked_mesh VALUES (?, ?)",
                ("mesh_a", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
            conn.close()

            _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dlg = MeshPickerDialog(gpkg)
            dlg._list.setCurrentRow(0)
            dlg.accept()
            self.assertEqual(dlg.selected_mesh_name(), "mesh_a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
