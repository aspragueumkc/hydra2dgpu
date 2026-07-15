"""Dialog for saving a simulation configuration to GPKG or JSON."""

from __future__ import annotations

import datetime
import json
import os
from typing import Callable, Dict, List, Optional

from qgis.PyQt import QtCore, QtWidgets


class SaveConfigDialog(QtWidgets.QDialog):
    """Dialog shown after GPKG is selected — collects config name and save options.

    Parameters
    ----------
    gpkg_path : str
        Path to the GPKG file already selected by the user.
    existing_config_ids : list of str
        Config IDs already present in the GPKG (so the user can see
        what exists and avoid accidental overwrites).
    widget_state : dict
        The collected widget state to save.
    mesh_name : str
        Name of the associated mesh.
    run_duration_s : float
        Simulation run duration in seconds.
    save_callback : callable
        Called with (gpkg_path, config_id, widget_state, description, run_duration_s)
        when the user confirms the save.
    json_save_callback : callable
        Called with (json_path, widget_state) when the user saves as JSON.
    parent : QWidget, optional
    """

    def __init__(
        self,
        gpkg_path: str,
        existing_config_ids: List[str],
        widget_state: Dict,
        mesh_name: str,
        run_duration_s: float,
        save_callback: Callable,
        json_save_callback: Optional[Callable] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._gpkg_path = gpkg_path
        self._existing_ids = list(existing_config_ids)
        self._widget_state = widget_state
        self._mesh_name = mesh_name
        self._run_duration_s = run_duration_s
        self._save_callback = save_callback
        self._json_save_callback = json_save_callback
        self._overwrite_warned = False

        self.setWindowTitle("Save Simulation Configuration")
        self.setMinimumSize(520, 320)
        self.setModal(True)

        main_layout = QtWidgets.QVBoxLayout(self)

        # ── Header ──────────────────────────────────────────────────────
        main_layout.addWidget(QtWidgets.QLabel(f"Saving to: {self._gpkg_path}"))
        main_layout.addWidget(QtWidgets.QLabel(f"Mesh: {self._mesh_name}"))

        # ── Existing configs warning ────────────────────────────────────
        if self._existing_ids:
            first_five = ", ".join(self._existing_ids[:5])
            extra = f" ... (+{len(self._existing_ids)-5} more)" if len(self._existing_ids) > 5 else ""
            existing_label = QtWidgets.QLabel(
                f"Existing configs in this GPKG: {first_five}{extra})"
            )
            existing_label.setStyleSheet("color: #888; font-size: 11px;")
            main_layout.addWidget(existing_label)

        # ── Config ID input ─────────────────────────────────────────────
        config_id_layout = QtWidgets.QHBoxLayout()
        config_id_layout.addWidget(QtWidgets.QLabel("Configuration name:"))
        self._config_id_edit = QtWidgets.QLineEdit()
        self._config_id_edit.setPlaceholderText(
            "e.g. my_config or swe2d_20250714T120000"
        )
        self._config_id_edit.textChanged.connect(self._on_name_changed)
        config_id_layout.addWidget(self._config_id_edit, 1)
        main_layout.addLayout(config_id_layout)

        # Overwrite warning label (hidden initially)
        self._overwrite_warning = QtWidgets.QLabel(
            "⚠️  A config with this name already exists — saving will overwrite it."
        )
        self._overwrite_warning.setStyleSheet("color: #c0392b; font-weight: bold;")
        self._overwrite_warning.setVisible(False)
        main_layout.addWidget(self._overwrite_warning)

        # ── Description input ───────────────────────────────────────────
        desc_layout = QtWidgets.QHBoxLayout()
        desc_layout.addWidget(QtWidgets.QLabel("Description (optional):"))
        self._desc_edit = QtWidgets.QLineEdit()
        self._desc_edit.setPlaceholderText("Brief description of this configuration...")
        desc_layout.addWidget(self._desc_edit, 1)
        main_layout.addLayout(desc_layout)

        # ── Save format selector ───────────────────────────────────────
        format_group = QtWidgets.QGroupBox("Save format")
        format_layout = QtWidgets.QVBoxLayout(format_group)
        self._format_gpkg = QtWidgets.QRadioButton(
            f"Save to GeoPackage table (swe2d_simulation_configs)"
        )
        self._format_gpkg.setChecked(True)
        self._format_json = QtWidgets.QRadioButton("Export as JSON file")
        format_layout.addWidget(self._format_gpkg)
        format_layout.addWidget(self._format_json)
        main_layout.addWidget(format_group)

        # JSON path (hidden until JSON is selected)
        self._json_path_layout = QtWidgets.QHBoxLayout()
        self._json_path_layout.addWidget(QtWidgets.QLabel("JSON path:"))
        self._json_path_edit = QtWidgets.QLineEdit()
        self._json_browse_btn = QtWidgets.QPushButton("Browse...")
        self._json_browse_btn.clicked.connect(self._on_json_browse)
        self._json_path_layout.addWidget(self._json_path_edit, 1)
        self._json_path_layout.addWidget(self._json_browse_btn)
        self._json_path_widget = QtWidgets.QWidget()
        self._json_path_widget.setLayout(self._json_path_layout)
        self._json_path_widget.setVisible(False)
        main_layout.addWidget(self._json_path_widget)

        self._format_gpkg.toggled.connect(
            lambda checked: self._json_path_widget.setVisible(not checked)
        )
        self._format_json.toggled.connect(
            lambda checked: self._json_path_widget.setVisible(checked)
        )

        # ── Buttons ─────────────────────────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        self._save_btn = QtWidgets.QPushButton("Save")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self._save_btn)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

    def _on_name_changed(self, text: str) -> None:
        name = text.strip()
        exists = name in self._existing_ids
        self._overwrite_warning.setVisible(exists)
        self._save_btn.setEnabled(bool(name))

    def _on_json_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Configuration as JSON",
            self._json_path_edit.text() or "",
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            if not path.endswith(".json"):
                path += ".json"
            self._json_path_edit.setText(path)

    def _on_save(self) -> None:
        config_id = self._config_id_edit.text().strip()
        if not config_id:
            return
        description = self._desc_edit.text().strip()

        if self._format_json.isChecked():
            json_path = self._json_path_edit.text().strip()
            if not json_path:
                QtWidgets.QMessageBox.warning(
                    self, "JSON path required",
                    "Please enter or browse for a JSON file path."
                )
                return
            parent_dlg = self.parent()
            crs_wkt = ""
            if parent_dlg is not None:
                mesh_data = getattr(parent_dlg, "_mesh_data", None) or {}
                crs_wkt = str(mesh_data.get("crs_wkt", "") or "")
            payload = {
                "schema_version": "swe2d-replay/1",
                "run_id": config_id,
                "mesh": {
                    "gpkg_path": self._gpkg_path,
                    "mesh_name": self._mesh_name,
                    "crs_wkt": crs_wkt,
                },
                "params": {},
                "data_sources": self._widget_state.get("_data_sources", {}),
                "results": {},
                "units": {},
                "widget_state": self._widget_state,
            }
            try:
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                self.accept()
                if self._json_save_callback:
                    self._json_save_callback(json_path, self._widget_state)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, "Save failed", f"Could not write JSON file:\n{exc}"
                )
        else:
            # GPKG save — pass to caller
            self._save_callback(
                self._gpkg_path,
                config_id,
                self._widget_state,
                description,
                self._run_duration_s,
            )
            self.accept()
