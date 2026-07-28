"""swe2d/workbench/dialogs/network_profile_dialog.py

Standalone Network Profile Viewer dialog. Ties together the chain editor,
matplotlib plot widget, time slider, and map tool.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.dialogs.network_profile_plot_widget import NetworkProfilePlotWidget
from swe2d.workbench.dialogs.profile_options_dialog import ProfileOptions, ProfileOptionsDialog
from swe2d.workbench.services.drainage_graph_service import (
    DrainageGraph,
    supplement_graph_from_coupling,
)
from swe2d.workbench.services.profile_pipeline_service import (
    ChainSpec,
    assemble_chain_profile,
)
from swe2d.workbench.views.profile_chain_widget import ProfileChainWidget

logger = logging.getLogger(__name__)


class NetworkProfileDialog(QtWidgets.QDialog):
    """Network Profile Viewer — main entry point."""

    def __init__(
        self,
        gpkg_path: str,
        run_id: Optional[str] = None,
        qgis_iface: object = None,
        parent=None,
    ):
        super().__init__(parent)
        self._gpkg_path = gpkg_path
        self._iface = qgis_iface
        self._map_tool = None
        self._previous_map_tool = None
        self._options = ProfileOptions()
        self._profile = None
        self._timestep_index = 0
        self._n_timesteps = 0

        self._run_id = run_id or self._prompt_run_id(self._gpkg_path)
        # Build graph entirely from coupling results (swe2d_baked_coupling).
        # No model topology tables are consulted.
        self._graph = supplement_graph_from_coupling(
            DrainageGraph(node_ids=[], link_ids=[]),
            self._gpkg_path, self._run_id,
        )

        self.setWindowTitle(f"Network Profile Viewer — {os.path.basename(gpkg_path)}")
        self.resize(1400, 800)
        self._build_ui()
        self._wire()
        self._populate_chain_widget()
        self._populate_timestep_slider()

    def _prompt_run_id(self, gpkg_path: str) -> Optional[str]:
        """Show a dialog to pick a run ID from available runs in the GPKG."""
        import sqlite3
        run_ids = []
        try:
            conn = sqlite3.connect(gpkg_path)
            try:
                for table in ("swe2d_run_logs", "swe2d_baked_pipe_cell_ts", "swe2d_baked_results", "swe2d_baked_coupling"):
                    cur = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    )
                    if cur.fetchone() is None:
                        continue
                    for row in conn.execute(f"SELECT DISTINCT run_id FROM {table}"):
                        rid = str(row[0]) if row[0] else ""
                        if rid and rid not in run_ids:
                            run_ids.append(rid)
            finally:
                conn.close()
        except Exception:
            return None
        if not run_ids:
            return None
        if len(run_ids) == 1:
            return run_ids[0]
        from qgis.PyQt.QtWidgets import QInputDialog
        item, ok = QInputDialog.getItem(
            self, "Select Run", "Choose a simulation run:", run_ids, False,
        )
        return item if ok and item else None

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # Left: chain editor
        self._chain_widget = ProfileChainWidget()
        self._chain_widget.setMinimumWidth(340)
        splitter.addWidget(self._chain_widget)

        # Right: matplotlib plot
        self._plot_widget = NetworkProfilePlotWidget()
        splitter.addWidget(self._plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Bottom: slider + variable + buttons
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("Time step:"))
        self._timestep_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._timestep_slider.setMinimum(0)
        self._timestep_slider.setMaximum(0)
        self._timestep_slider.setValue(0)
        bottom.addWidget(self._timestep_slider, stretch=1)
        self._timestep_lbl = QtWidgets.QLabel("0/0")
        bottom.addWidget(self._timestep_lbl)

        bottom.addSpacing(20)
        bottom.addWidget(QtWidgets.QLabel("Fill / overlay:"))
        self._variable_combo = QtWidgets.QComboBox()
        # depth/velocity/flow drive the per-cell water-fill colormap;
        # head draws a line overlay; —none— disables both.
        self._variable_combo.addItems(["—none—", "depth", "velocity", "flow", "head"])
        bottom.addWidget(self._variable_combo)

        bottom.addStretch(1)
        self._options_btn = QtWidgets.QPushButton("Options...")
        self._png_btn = QtWidgets.QPushButton("Export PNG")
        self._csv_btn = QtWidgets.QPushButton("Export CSV")
        self._close_btn = QtWidgets.QPushButton("Close")
        for w in (self._options_btn, self._png_btn, self._csv_btn, self._close_btn):
            bottom.addWidget(w)
        outer.addLayout(bottom)

    def _wire(self):
        self._chain_widget.chain_changed.connect(self._on_chain_changed)
        self._chain_widget.pick_requested.connect(self._on_pick_on_map)
        self._timestep_slider.valueChanged.connect(self._on_slider_change)
        self._variable_combo.currentTextChanged.connect(self._on_variable_change)
        self._options_btn.clicked.connect(self._on_options)
        self._png_btn.clicked.connect(self._on_export_png)
        self._csv_btn.clicked.connect(self._on_export_csv)
        self._close_btn.clicked.connect(self.accept)

    def _populate_chain_widget(self):
        self._chain_widget.set_context(self._gpkg_path, self._graph)

    def _populate_timestep_slider(self):
        if not self._run_id:
            return
        import sqlite3
        try:
            conn = sqlite3.connect(self._gpkg_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT MAX(n_timesteps) FROM swe2d_baked_pipe_cell_ts WHERE run_id = ?",
                    (self._run_id,),
                )
                row = cur.fetchone()
                max_ts = int(row[0]) if row and row[0] is not None else 0
            finally:
                conn.close()
        except Exception:
            max_ts = 0
        self._n_timesteps = max_ts
        self._timestep_slider.setMaximum(max(0, max_ts - 1))
        self._timestep_lbl.setText(f"0/{max_ts}")

    def _render(self):
        chain = self._chain_widget.get_chain()
        if chain.is_empty() or not self._run_id:
            self._profile = None
            self._plot_widget.draw_profile(_empty_profile())
            return
        try:
            self._profile = assemble_chain_profile(
                self._gpkg_path, self._run_id, chain, self._graph,
                self._timestep_index,
            )
        except Exception as exc:
            logger.exception("assemble_chain_profile failed")
            self._profile = None
            self._plot_widget.draw_profile(_empty_profile())
            QtWidgets.QMessageBox.warning(self, "Render Error", str(exc))
            return
        self._plot_widget.draw_profile(self._profile, variable=self._variable_combo.currentText())

    def _on_chain_changed(self, _chain):
        self._render()

    def _on_slider_change(self, idx):
        self._timestep_index = idx
        self._timestep_lbl.setText(f"{idx}/{self._n_timesteps}")
        self._render()

    def _on_variable_change(self, _name):
        self._render()

    def _on_options(self):
        dlg = ProfileOptionsDialog(self._options, parent=self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._options = dlg.get_options()
            self._plot_widget.set_options(self._options)
            self._render()

    def _on_export_png(self):
        if self._profile is None or self._profile.station_m.size == 0:
            QtWidgets.QMessageBox.information(self, "Export PNG", "No profile to export.")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Profile PNG", "", "PNG Files (*.png);;All Files (*)"
        )
        if not filepath:
            return
        self._plot_widget.export_png(filepath)

    def _on_export_csv(self):
        if self._profile is None or self._profile.station_m.size == 0:
            QtWidgets.QMessageBox.information(self, "Export CSV", "No profile to export.")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Profile CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        self._plot_widget.export_csv(filepath, self._profile)

    def _on_pick_on_map(self):
        if self._iface is None or self._iface.mapCanvas() is None:
            QtWidgets.QMessageBox.information(
                self, "Pick on Map",
                "Map canvas not available in this context. Use the chain editor instead.",
            )
            return
        try:
            from swe2d.workbench.views.network_profile_map_tool import NetworkProfileMapTool
            canvas = self._iface.mapCanvas()
            layers = self._iface.mapCanvas().layers() if hasattr(self._iface.mapCanvas(), "layers") else []
            drainage_layer = None
            for lyr in layers:
                try:
                    if lyr.name().endswith("drainage_links") or lyr.name() == "SWE2D_Drainage_Links":
                        drainage_layer = lyr
                        break
                except Exception:
                    continue
            if drainage_layer is None:
                QtWidgets.QMessageBox.information(
                    self, "Pick on Map",
                    "Drainage links layer not loaded. Use chain editor (Find Path / Add) instead.",
                )
                return
            self._previous_map_tool = canvas.mapTool()
            self._map_tool = NetworkProfileMapTool(canvas, drainage_layer, self._graph)
            self._map_tool.finished.connect(self._on_map_tool_finished)
            self._map_tool.chain_extended.connect(self._on_chain_extended_from_map)
            canvas.setMapTool(self._map_tool)
        except Exception as exc:
            logger.exception("Failed to activate map tool")
            QtWidgets.QMessageBox.warning(self, "Pick on Map", str(exc))

    def _on_chain_extended_from_map(self, chain: ChainSpec):
        self._chain_widget.set_chain(chain)

    def _on_map_tool_finished(self, chain: ChainSpec):
        self._chain_widget.set_chain(chain)
        if self._previous_map_tool is not None and self._iface is not None:
            self._iface.mapCanvas().setMapTool(self._previous_map_tool)
            self._previous_map_tool = None


def _empty_profile():
    import numpy as np
    from swe2d.workbench.services.profile_pipeline_service import ProfileArrays
    empty = np.zeros(0)
    return ProfileArrays(
        station_m=empty, invert_m=empty, crown_offset_m=empty, crown_m=empty,
        hgl_m=empty, depth_m=empty,
        velocity_ms=empty, flow_cms=empty,
        node_stations=[], node_ids=[], link_boundaries=[], crown_style="circular",
    )
