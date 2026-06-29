"""Dialog for picking and applying a saved simulation configuration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from qgis.PyQt import QtWidgets, QtCore


class SWE2DSimulationConfigDialog(QtWidgets.QDialog):
    """Dialog showing saved simulation configs. Apply restores widget state."""

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

        self.setWindowTitle("Load Simulation Configuration")
        self.setMinimumSize(600, 350)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        # Header
        layout.addWidget(QtWidgets.QLabel(
            f"Saved configurations in: {self._db_path}"
        ))

        # Table
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
        layout.addWidget(self._table)

        # Description / preview
        self._desc_label = QtWidgets.QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self._desc_label)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self._apply_btn = QtWidgets.QPushButton("Apply Configuration")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self._apply_btn)
        btn_layout.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        if self._configs:
            self._table.selectRow(0)

    def _on_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            if 0 <= idx < len(self._configs):
                self._selected_config = self._configs[idx]
                self._apply_btn.setEnabled(True)
                desc = self._selected_config.get("description", "")
                ws = self._selected_config.get("widget_state", {})
                n_widgets = len(ws) if isinstance(ws, dict) else 0
                self._desc_label.setText(
                    f"{desc}\n{n_widgets} widget parameters saved"
                    if desc
                    else f"{n_widgets} widget parameters"
                )
                return
        self._selected_config = None
        self._apply_btn.setEnabled(False)
        self._desc_label.setText("")

    def _on_apply(self) -> None:
        cfg = self._selected_config
        if cfg is None:
            return
        ws = cfg.get("widget_state", {})
        if not isinstance(ws, dict):
            return
        # The callback expects metadata with "workbench_widget_state" key containing
        # the version+widgets dict produced by collect_workbench_widget_state().
        # The saved widget_state IS that dict (has "version" and "widgets" keys).
        metadata = {"workbench_widget_state": ws}
        if self._apply_callback is not None:
            try:
                restored = self._apply_callback(metadata)
                if restored:
                    parent = self.parent()
                    if parent and hasattr(parent, "_log"):
                        parent._log(
                            f"Applied simulation config '{cfg.get('config_id', '')}': "
                            f"{int(restored)} widgets restored."
                        )
            except Exception as exc:
                parent = self.parent()
                if parent and hasattr(parent, "_log"):
                    parent._log(f"[ERROR] Failed to apply config: {exc}")
        self.accept()
