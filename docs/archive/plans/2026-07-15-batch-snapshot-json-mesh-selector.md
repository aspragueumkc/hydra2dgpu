---
type: plan
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# Batch Snapshot JSON / Mesh Selector Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Batch Simulation dialog's *Snapshot Current Setup* reuse the same replay JSON builder as the main dialog's *Export Config to JSON*, replace the GPKG path + mesh combo with a single read-only display + `Select Mesh...` button, and ensure all layer data sources carry full GeoPackage paths.

**Architecture:** Controller delegation: the batch dialog (View) asks the parent `SWE2DStudioDialog` for a replay payload, which delegates to `RunController`. A new lightweight `MeshPickerDialog` handles GPKG mesh selection. The `collect_data_source_config` helper is updated to always include full GPKG paths.

**Tech Stack:** Python 3.12, PyQt5 via `qgis.PyQt`, sqlite3, `swe2d.runtime.run_context_builder.widget_state_to_flat_params`, unittest.

---

## File Structure

- `swe2d/workbench/controllers/run_controller.py` — add public `build_replay_payload` wrapper.
- `swe2d/workbench/studio_dialog.py` — add delegate method and update `_dict_with_gpkg` to always include full paths.
- `swe2d/workbench/dialogs/mesh_picker_dialog.py` — new lightweight mesh picker dialog.
- `swe2d/workbench/dialogs/batch_simulation_dialog.py` — replace UI and snapshot logic.
- `tests/test_results_path_wiring.py` — add/update tests for the new behavior.

---

### Task 1: Expose public `build_replay_payload` on `RunController`

**Files:**
- Modify: `swe2d/workbench/controllers/run_controller.py:825-872`

- [ ] **Step 1: Write the failing test**

Create a new test file (or append to an existing one) to verify the public method exists and delegates to the existing private method.

```python
# tests/test_run_controller_build_replay_payload.py
import unittest
from unittest.mock import MagicMock
from swe2d.workbench.controllers.run_controller import RunController


class TestRunControllerBuildReplayPayload(unittest.TestCase):
    def test_public_method_delegates_to_private_method(self):
        view = MagicMock()
        view._mesh_data = {}
        rc = RunController(view=view)
        rc._build_replay_payload = MagicMock(return_value={"ok": True})
        result = rc.build_replay_payload(
            widget_state={"a": 1},
            mesh_name="mesh",
            run_duration_s=3600.0,
            mesh_gpkg_path="/tmp/m.gpkg",
            run_id="run_1",
        )
        rc._build_replay_payload.assert_called_once_with(
            widget_state={"a": 1},
            mesh_name="mesh",
            run_duration_s=3600.0,
            mesh_gpkg_path="/tmp/m.gpkg",
            run_id="run_1",
        )
        self.assertEqual(result, {"ok": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_run_controller_build_replay_payload -v
```

Expected: FAIL with `AttributeError: 'RunController' object has no attribute 'build_replay_payload'`.

- [ ] **Step 3: Add the public wrapper method**

In `swe2d/workbench/controllers/run_controller.py`, immediately after `_build_replay_payload` (around line 871), add:

```python
    def build_replay_payload(
        self,
        widget_state: dict,
        mesh_name: str,
        run_duration_s: float,
        mesh_gpkg_path: str = "",
        run_id: str = "",
    ) -> dict:
        """Public wrapper for ``_build_replay_payload``.

        Allows child dialogs (e.g. Batch Simulation) to build the same
        ``swe2d-replay/1`` JSON payload used by the main JSON export.
        """
        return self._build_replay_payload(
            widget_state=widget_state,
            mesh_name=mesh_name,
            run_duration_s=run_duration_s,
            mesh_gpkg_path=mesh_gpkg_path,
            run_id=run_id,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_run_controller_build_replay_payload -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/controllers/run_controller.py tests/test_run_controller_build_replay_payload.py
git commit -m "feat: expose public build_replay_payload wrapper on RunController"
```

---

### Task 2: Add delegate method on `SWE2DStudioDialog`

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py:2666-2677`

- [ ] **Step 1: Add the delegate method**

Replace the existing `collect_widget_state_for_save` block with both methods:

```python
    def build_replay_payload(
        self,
        widget_state: dict,
        mesh_name: str,
        run_duration_s: float,
        mesh_gpkg_path: str = "",
        run_id: str = "",
    ) -> dict:
        """Build a CLI-replay JSON payload from the current widget state.

        Delegates to the workbench RunController so child dialogs (e.g.
        Batch Simulation) can produce the same JSON as Export Config to JSON.
        """
        ctrl = getattr(self, "_controller", None)
        if ctrl is not None:
            return ctrl.build_replay_payload(
                widget_state=widget_state,
                mesh_name=mesh_name,
                run_duration_s=run_duration_s,
                mesh_gpkg_path=mesh_gpkg_path,
                run_id=run_id,
            )
        return {}

    def collect_widget_state_for_save(self) -> dict:
        """Delegate to the workbench RunController so child dialogs (e.g. batch)
        can use the same widget-collection API as GUI save paths.

        Note: ``_run_controller`` is the runtime SWE2DRunController (no widget
        state methods); ``_controller`` is the workbench RunController.
        """
        ctrl = getattr(self, "_controller", None)
        if ctrl is not None:
            return ctrl.collect_widget_state_for_save()
        return {}
```

- [ ] **Step 2: Verify import/syntax**

```bash
mamba run -n qgis_stable python3 -c "import swe2d.workbench.studio_dialog"
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add swe2d/workbench/studio_dialog.py
git commit -m "feat: add studio_dialog delegate for build_replay_payload"
```

---

### Task 3: Make `collect_data_source_config` always include full GPKG paths

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py:2612-2617`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_results_path_wiring.py` (or a new file) a test that verifies the model GPKG path is included even when it matches `_model_gpkg_path`.

```python
# Append to tests/test_results_path_wiring.py
import tempfile
import os
import sqlite3
from unittest.mock import MagicMock, patch


class TestCollectDataSourceConfigFullPaths(unittest.TestCase):
    def test_model_gpkg_path_is_included(self):
        """Even when a layer comes from the same GPKG as the model, the full
        path must be stored so batch/CLI replays can resolve it."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        from qgis.core import QgsProject as _QgsProject

        with tempfile.TemporaryDirectory() as tmp:
            gpkg_path = os.path.join(tmp, "model.gpkg")
            # Create a minimal GPKG with gpkg_contents
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

            dlg = MagicMock(spec=SWE2DWorkbenchStudioDialog)
            dlg._model_gpkg_path = gpkg_path
            type(dlg)._model_gpkg_path = property(lambda self: gpkg_path)

            # Mock model tab view
            mt = MagicMock()
            bc_combo = MagicMock()
            bc_combo.currentData.return_value = "layer_id"
            mt.bc_lines_layer_combo = bc_combo
            dlg._model_tab_view = mt

            # Mock QgsProject mapLayer
            layer = MagicMock()
            layer.source.return_value = f"{gpkg_path}|layername=bc_lines"
            layer.name.return_value = "bc_lines"
            with patch.object(_QgsProject, "instance", return_value=MagicMock()) as mock_instance:
                mock_instance.return_value.mapLayer.return_value = layer
                # Bind the real method to the mock instance
                from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog as RealDlg
                result = RealDlg.collect_data_source_config(dlg)

            self.assertIn("bc_lines", result)
            self.assertEqual(result["bc_lines"]["gpkg"], gpkg_path)
            self.assertEqual(result["bc_lines"]["table"], "bc_lines")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_results_path_wiring.TestCollectDataSourceConfigFullPaths -v
```

Expected: FAIL with `AssertionError: 'gpkg' not found in {'table': 'bc_lines'}` (or similar).

- [ ] **Step 3: Update `_dict_with_gpkg` to always include the path**

In `swe2d/workbench/studio_dialog.py`, change:

```python
        def _dict_with_gpkg(table: str, gpkg_path: str, **extra) -> dict:
            d = {"table": table, **extra}
            mgp = getattr(self, "_model_gpkg_path", None)
            if gpkg_path and gpkg_path != mgp:
                d["gpkg"] = gpkg_path
            return d
```

to:

```python
        def _dict_with_gpkg(table: str, gpkg_path: str, **extra) -> dict:
            d = {"table": table, **extra}
            if gpkg_path:
                d["gpkg"] = gpkg_path
            return d
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_results_path_wiring.TestCollectDataSourceConfigFullPaths -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/studio_dialog.py tests/test_results_path_wiring.py
git commit -m "feat: always include full gpkg path in data_source_config"
```

---

### Task 4: Create the `MeshPickerDialog`

**Files:**
- Create: `swe2d/workbench/dialogs/mesh_picker_dialog.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mesh_picker_dialog.py
import os
import sqlite3
import tempfile
import unittest
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtCore import Qt
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

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            dlg = MeshPickerDialog(gpkg)
            dlg._list.setCurrentRow(0)
            dlg.accept()
            self.assertEqual(dlg.selected_mesh_name(), "mesh_a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_mesh_picker_dialog -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'swe2d.workbench.dialogs.mesh_picker_dialog'`.

- [ ] **Step 3: Create the dialog implementation**

Create `swe2d/workbench/dialogs/mesh_picker_dialog.py`:

```python
"""Lightweight mesh picker dialog for selecting a mesh from a GeoPackage."""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

from qgis.PyQt import QtWidgets


class MeshPickerDialog(QtWidgets.QDialog):
    """Dialog that lists mesh names from a GeoPackage and returns the selected one."""

    def __init__(self, gpkg_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Mesh")
        self.resize(400, 300)
        self._gpkg_path = str(gpkg_path or "")
        self._selected_mesh_name = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"GeoPackage: {self._gpkg_path}"))

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self._list, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_mesh_names()

    def _load_mesh_names(self) -> None:
        """Populate the list with mesh names from swe2d_baked_mesh."""
        if not self._gpkg_path or not os.path.isfile(self._gpkg_path):
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", "GeoPackage file not found."
            )
            return
        try:
            conn = sqlite3.connect(self._gpkg_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT mesh_name FROM swe2d_baked_mesh "
                "WHERE mesh_name IS NOT NULL AND mesh_name != '' "
                "ORDER BY created_utc DESC"
            )
            for row in cur.fetchall():
                self._list.addItem(str(row[0]))
            conn.close()
            if self._list.count() == 0:
                QtWidgets.QMessageBox.information(
                    self, "Select Mesh",
                    "No mesh names found in the selected GeoPackage."
                )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh",
                f"Failed to read mesh list:\n{exc}"
            )

    def _on_accept(self) -> None:
        item = self._list.currentItem()
        if item is None:
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", "Please select a mesh."
            )
            return
        self._selected_mesh_name = str(item.text())
        self.accept()

    def selected_mesh_name(self) -> str:
        """Return the selected mesh name (empty if dialog was cancelled)."""
        return self._selected_mesh_name

    def gpkg_path(self) -> str:
        """Return the GeoPackage path passed to the dialog."""
        return self._gpkg_path
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_mesh_picker_dialog -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/dialogs/mesh_picker_dialog.py tests/test_mesh_picker_dialog.py
git commit -m "feat: add lightweight mesh picker dialog"
```

---

### Task 5: Rewrite `BatchSimulationDialog` UI and snapshot logic

**Files:**
- Modify: `swe2d/workbench/dialogs/batch_simulation_dialog.py`

- [ ] **Step 1: Update constructor and internal state**

In the constructor (`__init__` around line 212-233), replace the GPKG-related attributes with mesh selection state:

```python
        self._base_params = dict(base_params or {})
        self._mesh_gpkg = str(mesh_gpkg)
        self._mesh_name = str(results_gpkg)  # temporary; will be overwritten by auto-populate
        self._param_sets: List[Dict[str, Any]] = []
```

Wait — `results_gpkg` is not the mesh name. The current constructor signature is `__init__(self, parent=None, base_params=None, mesh_gpkg: str = "", results_gpkg: str = "")`. We will need to derive the mesh name from the parent or leave it empty until the user selects. Change the constructor to set `_mesh_name = ""` initially and auto-populate later.

Also add the new widget attributes at the end of `_build_ui` (they will be created in the next step).

- [ ] **Step 2: Replace UI construction**

Replace the current GPKG and Mesh rows (lines 240-278) with a single mesh selection row:

```python
        # Mesh selection
        mesh_layout = QtWidgets.QHBoxLayout()
        mesh_layout.addWidget(QtWidgets.QLabel("Mesh:"))
        self._mesh_display_edit = QtWidgets.QLineEdit()
        self._mesh_display_edit.setReadOnly(True)
        self._mesh_display_edit.setPlaceholderText("Select a mesh...")
        mesh_layout.addWidget(self._mesh_display_edit, 1)
        self._select_mesh_btn = QtWidgets.QPushButton("Select Mesh...")
        self._select_mesh_btn.setToolTip(
            "Choose a GeoPackage and select a mesh from it."
        )
        self._select_mesh_btn.clicked.connect(self._select_mesh)
        mesh_layout.addWidget(self._select_mesh_btn)
        self._apply_mesh_all_btn = QtWidgets.QPushButton("Apply to All")
        self._apply_mesh_all_btn.setToolTip("Set the selected mesh name on ALL rows")
        self._apply_mesh_all_btn.setEnabled(False)
        self._apply_mesh_all_btn.clicked.connect(self._apply_mesh_to_all)
        mesh_layout.addWidget(self._apply_mesh_all_btn)
        layout.addLayout(mesh_layout)
```

Remove the `self._apply_mesh_all_btn` creation from the data toolbar section (lines 335-338) since it is now in the mesh row. Keep `Apply to Selected` in the data toolbar.

- [ ] **Step 3: Add the Select Mesh handler and remove obsolete methods**

Remove the old methods `_browse_gpkg`, `_on_gpkg_path_changed`, `_refresh_mesh_list`. Add the new `_select_mesh` method and a helper to update the display:

```python
    def _select_mesh(self):
        """Open a file picker, then a mesh picker, and store the result."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select GeoPackage",
            "",
            "GeoPackage (*.gpkg *.gpkgx);;All Files (*)",
        )
        if not path:
            return
        path = str(path).strip()
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(
                self, "Select Mesh", f"GeoPackage not found:\n{path}"
            )
            return
        from swe2d.workbench.dialogs.mesh_picker_dialog import MeshPickerDialog
        picker = MeshPickerDialog(path, parent=self)
        if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        mesh_name = picker.selected_mesh_name()
        if not mesh_name:
            return
        self._set_mesh(path, mesh_name)

    def _set_mesh(self, gpkg_path: str, mesh_name: str) -> None:
        """Store the selected mesh and update the read-only display."""
        self._mesh_gpkg = str(gpkg_path).strip()
        self._mesh_name = str(mesh_name).strip()
        display = f"{self._mesh_name}"
        if self._mesh_gpkg:
            display += f"  ({os.path.basename(self._mesh_gpkg)})"
        self._mesh_display_edit.setText(display)
        self._mesh_display_edit.setToolTip(
            f"{self._mesh_name}\n{self._mesh_gpkg}"
        )
        self._apply_mesh_all_btn.setEnabled(bool(self._mesh_name))
```

Update `_apply_mesh_to_selected` to use the stored mesh:

```python
    def _apply_mesh_to_selected(self):
        """Set the selected mesh name on all currently selected table rows."""
        mesh = self._mesh_name
        if not mesh:
            QtWidgets.QMessageBox.information(
                self, "Apply Mesh", "Select a mesh first."
            )
            return
        rows = set(i.row() for i in self._table.selectedIndexes())
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Apply Mesh", "Select rows in the table first."
            )
            return
        for r in rows:
            self._set_row_mesh(r, mesh)
```

Similarly update `_apply_mesh_to_all`:

```python
    def _apply_mesh_to_all(self):
        """Set the selected mesh name on ALL rows."""
        mesh = self._mesh_name
        if not mesh:
            return
        for r in range(self._table.rowCount()):
            self._set_row_mesh(r, mesh)
```

- [ ] **Step 4: Rewrite `_snapshot_current_setup` to use the controller payload builder**

Replace the entire `_snapshot_current_setup` method (lines 469-588) with:

```python
    def _snapshot_current_setup(self):
        """Read the parent dialog's current widget values and add a new row.

        Reuses the same ``build_replay_payload`` builder as the main dialog's
        Export Config to JSON action, so the snapshot format is identical.
        """
        import datetime
        parent = self.parent()
        if parent is None:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot", "No parent dialog to snapshot from."
            )
            return

        collect_fn = getattr(parent, "collect_widget_state_for_save", None)
        if collect_fn is None:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot",
                "Parent dialog does not implement collect_widget_state_for_save.",
            )
            return
        try:
            widget_state = collect_fn()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot Error", f"Failed to collect widget state:\n{exc}"
            )
            return
        if not isinstance(widget_state, dict):
            return

        run_id = datetime.datetime.now().astimezone().strftime(
            "swe2d_%Y%m%dT%H%M%S%z"
        )
        build_fn = getattr(parent, "build_replay_payload", None)
        if build_fn is None:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot",
                "Parent dialog does not implement build_replay_payload.",
            )
            return
        try:
            entry = build_fn(
                widget_state=widget_state,
                mesh_name=self._mesh_name,
                run_duration_s=0.0,  # builder will read from widget_state if present
                mesh_gpkg_path=self._mesh_gpkg,
                run_id=run_id,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Snapshot Error", f"Failed to build replay payload:\n{exc}"
            )
            return
        if not isinstance(entry, dict):
            return

        self._add_row_from_entry(entry)
        log = getattr(parent, "_log", None)
        if log:
            log("batch> snapshot added row from current setup")
```

Note: `run_duration_s=0.0` is passed because `_build_replay_payload` already computes `run_duration_s` from `widget_state_to_flat_params` and will override it if the widget state captures it. The explicit `run_duration_s` argument is only used if the widget state does not contain it, which is fine.

- [ ] **Step 5: Auto-populate the mesh selector on open**

After `_build_ui()` in the constructor, add:

```python
        # Auto-populate mesh selector from the current model if available
        parent_view = self.parent()
        if parent_view is not None:
            model_gpkg = str(getattr(parent_view, "_model_gpkg_path", "") or "")
            mesh_name = str(
                (getattr(parent_view, "_mesh_data", None) or {}).get("mesh_name", "")
                or ""
            )
            if model_gpkg and mesh_name and os.path.isfile(model_gpkg):
                self._set_mesh(model_gpkg, mesh_name)
```

This replaces the existing `self._refresh_mesh_list()` call in the constructor.

- [ ] **Step 6: Update `_run_batch` to use the stored GPKG path**

The `_run_batch` method already calls `gpkg = self._gpkg_path()`. Make sure `_gpkg_path` returns the stored mesh GPKG:

```python
    def _gpkg_path(self) -> str:
        return str(self._mesh_gpkg or "").strip()
```

- [ ] **Step 7: Run the existing batch tests**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_results_path_wiring -v
```

Expected: PASS (with possible updates in the next step).

- [ ] **Step 8: Commit**

```bash
git add swe2d/workbench/dialogs/batch_simulation_dialog.py
git commit -m "feat: rewrite batch dialog mesh selector and snapshot payload"
```

---

### Task 6: Update/add integration tests

**Files:**
- Modify: `tests/test_results_path_wiring.py`

- [ ] **Step 1: Add test for batch dialog auto-population and snapshot payload**

Add the following to `tests/test_results_path_wiring.py`:

```python
class TestBatchSimulationDialogMeshSelector(unittest.TestCase):
    def test_auto_populates_from_parent_model(self):
        from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog
        from qgis.PyQt import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        parent = MagicMock()
        parent._model_gpkg_path = "/tmp/model.gpkg"
        parent._mesh_data = {"mesh_name": "mesh_001"}
        parent.collect_widget_state_for_save.return_value = {}
        parent.build_replay_payload.return_value = {
            "schema_version": "swe2d-replay/1",
            "run_id": "run_1",
        }

        with patch("os.path.isfile", return_value=True):
            dlg = BatchSimulationDialog(parent=parent, mesh_gpkg="/tmp/model.gpkg")
            self.assertEqual(dlg._mesh_gpkg, "/tmp/model.gpkg")
            self.assertEqual(dlg._mesh_name, "mesh_001")

    def test_snapshot_uses_build_replay_payload(self):
        from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog
        from qgis.PyQt import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        parent = MagicMock()
        parent._model_gpkg_path = "/tmp/model.gpkg"
        parent._mesh_data = {"mesh_name": "mesh_001"}
        parent.collect_widget_state_for_save.return_value = {"widget": "state"}
        parent.build_replay_payload.return_value = {
            "schema_version": "swe2d-replay/1",
            "run_id": "run_1",
            "mesh": {"mesh_name": "mesh_001"},
        }

        with patch("os.path.isfile", return_value=True):
            dlg = BatchSimulationDialog(parent=parent, mesh_gpkg="/tmp/model.gpkg")
            dlg._snapshot_current_setup()

        parent.build_replay_payload.assert_called_once()
        kwargs = parent.build_replay_payload.call_args.kwargs
        self.assertEqual(kwargs["mesh_name"], "mesh_001")
        self.assertEqual(kwargs["mesh_gpkg_path"], "/tmp/model.gpkg")
        self.assertTrue(kwargs["run_id"].startswith("swe2d_"))
        self.assertEqual(dlg._table.rowCount(), 1)
```

- [ ] **Step 2: Run the updated tests**

```bash
mamba run -n qgis_stable python3 -m unittest tests.test_results_path_wiring -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_results_path_wiring.py
git commit -m "test: batch dialog mesh selector and snapshot payload"
```

---

### Task 7: Verification and cleanup

**Files:**
- All files changed above.

- [ ] **Step 1: Purge Python cache**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Run the relevant test suite**

```bash
mamba run -n qgis_stable python3 -m unittest \
    tests.test_run_controller_build_replay_payload \
    tests.test_mesh_picker_dialog \
    tests.test_results_path_wiring -v
```

Expected: PASS on all tests.

- [ ] **Step 3: Run architecture enforcement checks**

```bash
# No Qt in service layer
! grep -q 'from qgis\|from PyQt\|\.setEnabled\|\.setText\|\.setValue' swe2d/runtime/ swe2d/boundary_and_forcing/ && echo "PASS: service layer clean"

# No raw widget access in controller
! grep -q '\.setEnabled\|\.setText\|\.setValue\|\.isChecked' swe2d/workbench/controllers/run_controller.py && echo "PASS: controller clean"

# No numpy computation in View
! grep -q 'np\.min\|np\.max\|np\.vstack\|np\.argmin\|np\.where\|np\.hypot' swe2d/workbench/dialogs/batch_simulation_dialog.py swe2d/workbench/dialogs/mesh_picker_dialog.py 2>/dev/null && echo "PASS: view has no numpy computation"
```

Expected: All three print PASS.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: batch snapshot JSON parity and mesh selector redesign"
```

---

## Self-Review

- **Spec coverage:**
  - Snapshot uses same payload as Export JSON → Task 1, 2, 5 (Step 4).
  - Remove GPKG path line edit → Task 5 (Step 2).
  - Replace selector with read-only display + Select Mesh button → Task 4, 5 (Step 2, 3).
  - Full layer paths in data sources → Task 3.
  - Auto-populate from current model → Task 5 (Step 5).
  - Keep Apply to All / Apply to Selected → Task 5 (Step 2, 3).
- **Placeholder scan:** No TODO/TBD/similar placeholders found.
- **Type consistency:** `build_replay_payload` signature is identical across `RunController`, `SWE2DStudioDialog`, and the batch dialog call. `MeshPickerDialog` returns mesh name via `selected_mesh_name()` consistently.
