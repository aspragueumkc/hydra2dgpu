"""Dialog for picking and applying a saved simulation configuration."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from qgis.PyQt import QtWidgets, QtCore


class SWE2DSimulationConfigDialog(QtWidgets.QDialog):
    """Dialog with two tabs: Load from GPKG, Load from JSON."""

    def __init__(
        self,
        configs: List[Dict[str, Any]],
        db_path: str,
        parent=None,
        apply_callback: Optional[Callable[[Dict[str, Any]], int]] = None,
    ):
        super().__init__(parent)
        self._configs = list(configs)
        self._db_path = str(db_path)
        self._apply_callback = apply_callback
        self._selected_config: Optional[Dict[str, Any]] = None
        self._selected_json_path: Optional[str] = None
        self._loaded_json_state: Optional[Dict[str, Any]] = None

        self.setWindowTitle("Load Simulation Configuration")
        self.setMinimumSize(640, 420)
        self.setModal(True)

        main_layout = QtWidgets.QVBoxLayout(self)

        # ── Tab widget ─────────────────────────────────────────────────
        self._tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self._tabs)

        # ── GPKG tab ─────────────────────────────────────────────────
        gpkg_tab = QtWidgets.QWidget()
        gpkg_layout = QtWidgets.QVBoxLayout(gpkg_tab)

        gpkg_layout.addWidget(QtWidgets.QLabel(
            f"Saved configurations in: {self._db_path}"
        ))

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Config ID", "Mesh", "Created", "Duration (s)"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)

        for i, cfg in enumerate(self._configs):
            self._table.insertRow(i)
            self._table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(cfg.get("config_id", ""))))
            self._table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(cfg.get("mesh_name", ""))))
            self._table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(cfg.get("created_utc", ""))))
            dur = float(cfg.get("run_duration_s", 0.0))
            self._table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{dur:.1f}"))

        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        gpkg_layout.addWidget(self._table)

        self._desc_label = QtWidgets.QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #888; padding: 4px;")
        gpkg_layout.addWidget(self._desc_label)

        self._tabs.addTab(gpkg_tab, "Load from GeoPackage")

        # ── JSON tab ─────────────────────────────────────────────────
        json_tab = QtWidgets.QWidget()
        json_layout = QtWidgets.QVBoxLayout(json_tab)

        json_layout.addWidget(QtWidgets.QLabel(
            "Load a simulation configuration from a previously-exported JSON file."
        ))

        json_path_layout = QtWidgets.QHBoxLayout()
        self._json_path_edit = QtWidgets.QLineEdit()
        self._json_path_edit.setPlaceholderText("/path/to/config.json")
        self._json_path_edit.textChanged.connect(self._on_json_path_changed)
        json_browse_btn = QtWidgets.QPushButton("Browse...")
        json_browse_btn.clicked.connect(self._on_json_browse)
        json_path_layout.addWidget(self._json_path_edit, 1)
        json_path_layout.addWidget(json_browse_btn)
        json_layout.addLayout(json_path_layout)

        self._json_preview = QtWidgets.QLabel("No file selected")
        self._json_preview.setWordWrap(True)
        self._json_preview.setStyleSheet(
            "background: #f5f5f5; padding: 8px; border-radius: 4px; color: #555;"
        )
        self._json_preview.setMinimumHeight(80)
        json_layout.addWidget(QtWidgets.QLabel("Preview:"))
        json_layout.addWidget(self._json_preview)

        json_layout.addStretch(1)
        self._tabs.addTab(json_tab, "Load from JSON")

        # ── Buttons ──────────────────────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        self._apply_btn = QtWidgets.QPushButton("Apply Configuration")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self._apply_btn)
        btn_layout.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

        if self._configs:
            self._table.selectRow(0)

    # ── GPKG tab handlers ─────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            if 0 <= idx < len(self._configs):
                self._selected_config = self._configs[idx]
                self._selected_json_path = None
                self._loaded_json_state = None
                self._apply_btn.setEnabled(True)
                desc = self._selected_config.get("description", "")
                ws = self._selected_config.get("widget_state", {})
                n_widgets = len(ws) if isinstance(ws, dict) else 0
                self._desc_label.setText(
                    f"{desc}\n{n_widgets} widget parameters"
                    if desc
                    else f"{n_widgets} widget parameters"
                )
                return
        self._selected_config = None
        self._apply_btn.setEnabled(False)
        self._desc_label.setText("")

    def _on_apply(self) -> None:
        if self._tabs.currentIndex() == 0:
            self._apply_gpkg_config()
        else:
            self._apply_json_config()

    def _apply_gpkg_config(self) -> None:
        cfg = self._selected_config
        if cfg is None:
            return
        ws = cfg.get("widget_state", {})
        if not isinstance(ws, dict):
            return
        # Include params and units so the callback can fully restore the RunContext state
        metadata = {
            "workbench_widget_state": ws,
            "params": cfg.get("params", {}),
            "units": cfg.get("units", {}),
        }
        self._do_apply(cfg.get("config_id", ""), ws, metadata)

    def _apply_json_config(self) -> None:
        if self._loaded_json_state is None:
            return
        json_state = self._loaded_json_state
        # Extract the actual widget_state block from the replay JSON
        ws = json_state.get("widget_state", {})
        # Include params and units from the JSON state so the callback can
        # fully restore the RunContext state.
        metadata = {
            "workbench_widget_state": ws,
            "params": json_state.get("params", {}),
            "units": json_state.get("units", {}),
        }
        self._do_apply(
            json_state.get("run_id", os.path.basename(self._selected_json_path or "")),
            ws,
            metadata,
        )

    def _do_apply(self, config_id: str, ws: Dict, metadata: Dict) -> None:
        if self._apply_callback is not None:
            try:
                restored = self._apply_callback(metadata)
                parent = self.parent()
                if parent and hasattr(parent, "_log"):
                    parent._log(
                        f"Applied simulation config '{config_id}': "
                        f"{int(restored)} widgets restored."
                    )
            except Exception as exc:
                parent = self.parent()
                if parent and hasattr(parent, "_log"):
                    parent._log(f"[ERROR] Failed to apply config: {exc}")
        self.accept()

    # ── JSON tab handlers ─────────────────────────────────────────────

    def _on_json_path_changed(self, path: str) -> None:
        if not path or not os.path.isfile(path):
            self._json_preview.setText("No file selected")
            self._selected_json_path = None
            self._loaded_json_state = None
            self._apply_btn.setEnabled(False)
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._selected_json_path = path
            self._loaded_json_state = data
            run_id = data.get("run_id", data.get("id", "unknown"))
            mesh_info = data.get("mesh", {})
            mesh_name = mesh_info.get("mesh_name", "") if isinstance(mesh_info, dict) else ""
            params = data.get("params", {})
            widget_state = data.get("widget_state", data)
            n_ws = len(widget_state) if isinstance(widget_state, dict) else 0
            self._json_preview.setText(
                f"Run ID: {run_id}\n"
                f"Mesh: {mesh_name}\n"
                f"Params: {len(params)} keys\n"
                f"Widget state: {n_ws} entries"
            )
            self._apply_btn.setEnabled(True)
        except Exception as exc:
            self._json_preview.setText(f"Error reading file:\n{exc}")
            self._selected_json_path = None
            self._loaded_json_state = None
            self._apply_btn.setEnabled(False)

    def _on_json_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Simulation Configuration JSON",
            self._json_path_edit.text() or "",
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._json_path_edit.setText(path)
