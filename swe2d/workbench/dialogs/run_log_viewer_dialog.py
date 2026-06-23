#!/usr/bin/env python3
"""Viewer for saved SWE2D run logs stored in GeoPackage/SQLite."""

from __future__ import annotations

from typing import Dict, List, Optional

from qgis.PyQt import QtWidgets


class SWE2DRunLogViewerDialog(QtWidgets.QDialog):
    """Viewer for saved SWE2D run logs stored in GeoPackage/SQLite."""

    def __init__(
        self,
        records: List[Dict[str, object]],
        run_id: str,
        db_path: str,
        parent=None,
        apply_run_settings_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SWE2D Run Log Viewer")
        self.resize(900, 620)
        self._records = list(records)
        self._db_path = str(db_path)
        self._apply_run_settings_callback = apply_run_settings_callback

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"Source: {self._db_path}"))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Run:"))
        self.run_combo = QtWidgets.QComboBox()
        row.addWidget(self.run_combo)
        row.addStretch(1)
        root.addLayout(row)

        self.meta_lbl = QtWidgets.QLabel("")
        self.meta_lbl.setWordWrap(True)
        root.addWidget(self.meta_lbl)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        root.addWidget(self.text, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self._apply_btn = None
        if callable(self._apply_run_settings_callback):
            self._apply_btn = buttons.addButton(
                "Apply Inputs To UI",
                QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
            )
            self._apply_btn.clicked.connect(self._apply_selected_run_settings)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_run_combo()
        idx = self.run_combo.findData(str(run_id))
        if idx >= 0:
            self.run_combo.setCurrentIndex(idx)
        self.run_combo.currentIndexChanged.connect(self._refresh_view)
        self._refresh_view()

    def _selected_record(self) -> Optional[Dict[str, object]]:
        """Return the run record matching the currently selected run ID."""
        rid = str(self.run_combo.currentData() or "")
        for rec in self._records:
            if str(rec.get("run_id", "") or "") == rid:
                return rec
        return None

    def _populate_run_combo(self):
        """Populate the run combo box with labels from the loaded records."""
        self.run_combo.clear()
        for rec in self._records:
            rid = str(rec.get("run_id", "") or "")
            created = str(rec.get("created_utc", "") or "")
            dur = float(rec.get("duration_s", 0.0) or 0.0)
            label = f"{rid} ({created}, {dur:.2f}s)"
            self.run_combo.addItem(label, rid)

    def _refresh_view(self):
        """Update the metadata label and log text for the selected run."""
        rid = str(self.run_combo.currentData() or "")
        rec = self._selected_record()
        if rec is None:
            self.meta_lbl.setText("No run selected.")
            self.text.setPlainText("")
            return
        self.meta_lbl.setText(
            f"Run ID: {rid}\n"
            f"Start: {rec.get('start_wallclock', '')}\n"
            f"End: {rec.get('end_wallclock', '')}\n"
            f"Duration: {float(rec.get('duration_s', 0.0) or 0.0):.2f} s"
        )
        self.text.setPlainText(str(rec.get("log_text", "") or ""))

    def _apply_selected_run_settings(self) -> None:
        """Restore UI settings from the selected run's metadata."""
        if not callable(self._apply_run_settings_callback):
            return
        rec = self._selected_record()
        if rec is None:
            QtWidgets.QMessageBox.information(self, "Run Inputs", "No run record selected.")
            return
        metadata = rec.get("metadata")
        if not isinstance(metadata, dict):
            QtWidgets.QMessageBox.information(self, "Run Inputs", "Selected run has no metadata payload.")
            return
        try:
            restored = int(self._apply_run_settings_callback(metadata) or 0)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Run Inputs", f"Failed to apply saved inputs: {exc}")
            return
        if restored <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "Run Inputs",
                "Selected run metadata does not include restorable workbench inputs.",
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            "Run Inputs",
            f"Applied {restored} saved input setting(s) to the workbench UI.",
        )
