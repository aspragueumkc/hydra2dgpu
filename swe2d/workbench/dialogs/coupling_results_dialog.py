#!/usr/bin/env python3
"""Viewer for drainage/structure coupling time series stored in GeoPackage/SQLite."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.runtime.coupling import prepare_coupling_timeseries
from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

logger_wb = logging.getLogger(__name__)


class SWE2DCouplingResultsViewerDialog(QtWidgets.QDialog):
    """Viewer for drainage/structure coupling time series stored in GeoPackage/SQLite."""

    _BASE_COLUMNS = [
        ("t_s", "Time (s)"),
        ("component", "Component"),
        ("metric", "Metric"),
        ("object_id", "Object ID"),
        ("object_name", "Object Name"),
        ("value", "Value"),
    ]

    def __init__(
        self,
        records: List[Dict[str, object]],
        run_id: str,
        db_path: str,
        length_unit: str = "",
        flow_unit_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Drainage/Structure Results Viewer")
        self.resize(980, 620)

        self._records = list(records)
        self._run_id = str(run_id)
        self._db_path = str(db_path)
        self._length_unit = str(length_unit).strip() or "m"
        self._flow_unit = str(flow_unit_label).strip() or f"{self._length_unit}3/s"
        self._plot_canvas = None
        self._plot_fig = None

        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(f"Run ID: {self._run_id}\nSource: {self._db_path}")
        header.setWordWrap(True)
        root.addWidget(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Component:"))
        self.component_combo = QtWidgets.QComboBox()
        self.component_combo.setToolTip("Filter coupling records by component type (structure, drainage_node, etc.).")
        controls.addWidget(self.component_combo)
        controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.setToolTip("Filter coupling records by metric (flow, depth, velocity, etc.).")
        controls.addWidget(self.metric_combo)
        controls.addWidget(QtWidgets.QLabel("Object:"))
        self.object_combo = QtWidgets.QComboBox()
        self.object_combo.setToolTip("Filter coupling records by object ID.")
        controls.addWidget(self.object_combo)
        controls.addStretch(1)
        root.addLayout(controls)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(len(self._BASE_COLUMNS))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in self._BASE_COLUMNS])
        self.table.horizontalHeader().setStretchLastSection(True)

        self._have_mpl = False
        FigureCanvas, Figure, _ = try_import_matplotlib_qt()
        if FigureCanvas is not None and Figure is not None:
            self._have_mpl = True
            self._plot_fig = Figure(figsize=(6.8, 3.0), tight_layout=True)
            self._plot_canvas = FigureCanvas(self._plot_fig)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        table_host = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_host)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.table)
        split.addWidget(table_host)

        plot_host = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        if self._have_mpl:
            plot_layout.addWidget(self._plot_canvas)
        else:
            note = QtWidgets.QLabel("Matplotlib backend unavailable; table view only.")
            note.setWordWrap(True)
            plot_layout.addWidget(note)
        split.addWidget(plot_host)
        split.setSizes([220, 380])
        root.addWidget(split, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_component_combo()
        self._populate_metric_combo()
        self._populate_object_combo()
        self._refresh_table()
        self._refresh_plot()

        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.object_combo.currentIndexChanged.connect(self._refresh_table)
        self.object_combo.currentIndexChanged.connect(self._refresh_plot)

    def _unit_label_for_metric(self, metric: str) -> str:
        """Return the unit label string for a given coupling metric name."""
        m = str(metric or "")
        if m == "depth":
            return self._length_unit
        if m == "flow":
            return self._flow_unit
        if m == "source":
            return f"{self._length_unit}/s"
        # Unit-agnostic metric names (no suffix) — all flows are in model³/s,
        # all depths/lengths in model units, as displayed to the user.
        _FLOW_METRICS = {
            "inlet_control_flow", "outlet_control_flow", "orifice_cap",
            "manning_cap", "embankment_flow",
        }
        _LENGTH_METRICS = {
            "available_head_up", "tailwater_depth",
            "inlet_invert_elev", "outlet_invert_elev",
        }
        if m in _FLOW_METRICS or m.endswith("_cms"):
            return self._flow_unit
        if m in _LENGTH_METRICS or m.endswith("_m"):
            return self._length_unit
        if m.endswith("_mps"):
            return f"{self._length_unit}/s"
        return ""

    def _populate_component_combo(self):
        """Populate the component combo from available record components."""
        self.component_combo.clear()
        self.component_combo.addItem("All components", None)
        comps = sorted({str(r.get("component", "") or "") for r in self._records if r.get("component") is not None})
        for comp in comps:
            if comp:
                self.component_combo.addItem(comp, comp)

    def _populate_metric_combo(self):
        """Populate the metric combo filtered by the selected component."""
        selected_comp = self.component_combo.currentData()
        metrics = set()
        for rec in self._records:
            comp = str(rec.get("component", "") or "")
            if selected_comp is not None and comp != str(selected_comp):
                continue
            metric = str(rec.get("metric", "") or "")
            if metric:
                metrics.add(metric)
        current = self.metric_combo.currentData()
        self.metric_combo.clear()
        self.metric_combo.addItem("All metrics", None)
        for metric in sorted(metrics):
            unit = self._unit_label_for_metric(metric)
            label = metric if not unit else f"{metric} ({unit})"
            self.metric_combo.addItem(label, metric)
        if current is not None:
            idx = self.metric_combo.findData(current)
            if idx >= 0:
                self.metric_combo.setCurrentIndex(idx)

    def _populate_object_combo(self):
        """Populate the object combo filtered by selected component and metric."""
        selected_comp = self.component_combo.currentData()
        selected_metric = self.metric_combo.currentData()
        objects: Dict[str, str] = {}
        for rec in self._records:
            comp = str(rec.get("component", "") or "")
            metric = str(rec.get("metric", "") or "")
            if selected_comp is not None and comp != str(selected_comp):
                continue
            if selected_metric is not None and metric != str(selected_metric):
                continue
            oid = str(rec.get("object_id", "") or "")
            if not oid:
                continue
            objects[oid] = str(rec.get("object_name", "") or "")

        current = self.object_combo.currentData()
        self.object_combo.clear()
        self.object_combo.addItem("All objects", None)
        for oid in sorted(objects.keys()):
            name = objects[oid]
            label = oid if not name else f"{oid} - {name}"
            self.object_combo.addItem(label, oid)
        if current is not None:
            idx = self.object_combo.findData(current)
            if idx >= 0:
                self.object_combo.setCurrentIndex(idx)

    def _on_component_changed(self):
        """Refresh metric/object combos, table, and plot when component changes."""
        self._populate_metric_combo()
        self._populate_object_combo()
        self._refresh_table()
        self._refresh_plot()

    def _on_metric_changed(self):
        """Refresh object combo, table, and plot when metric changes."""
        self._populate_object_combo()
        self._refresh_table()
        self._refresh_plot()

    def _filtered_records(self) -> List[Dict[str, object]]:
        """Return records matching the current combo filter selections."""
        comp_sel = self.component_combo.currentData()
        metric_sel = self.metric_combo.currentData()
        obj_sel = self.object_combo.currentData()
        out: List[Dict[str, object]] = []
        for rec in self._records:
            comp = str(rec.get("component", "") or "")
            metric = str(rec.get("metric", "") or "")
            oid = str(rec.get("object_id", "") or "")
            if comp_sel is not None and comp != str(comp_sel):
                continue
            if metric_sel is not None and metric != str(metric_sel):
                continue
            if obj_sel is not None and oid != str(obj_sel):
                continue
            out.append(rec)
        return out

    def _refresh_table(self):
        """Reload the table widget with filtered and sorted coupling records."""
        rows = self._filtered_records()
        rows.sort(
            key=lambda r: (
                float(r.get("t_s", 0.0)),
                str(r.get("component", "") or ""),
                str(r.get("metric", "") or ""),
                str(r.get("object_id", "") or ""),
            )
        )
        self.table.setRowCount(len(rows))
        for r, rec in enumerate(rows):
            for c, (key, _) in enumerate(self._BASE_COLUMNS):
                val = rec.get(key)
                txt = f"{val:.6f}" if isinstance(val, float) else str(val)
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

    def _refresh_plot(self):
        """Replot coupling time series for the current filter selection."""
        if not self._have_mpl or self._plot_fig is None or self._plot_canvas is None:
            return
        rows = self._filtered_records()
        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        if not rows:
            ax.text(0.5, 0.5, "No coupling records for selected filter", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        grouped = prepare_coupling_timeseries(rows)
        if not grouped:
            ax.text(0.5, 0.5, "No numeric values to plot", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        for oid in sorted(grouped.keys()):
            entry = grouped[oid]
            x = entry["times"]
            y = entry["values"]
            label = oid if oid else "(unlabeled)"
            name = entry["name"]
            if name:
                label += f" ({name})"
            ax.plot(x, y, "-", linewidth=1.8, label=label)

        metric_sel = self.metric_combo.currentData()
        y_label = "Value"
        if metric_sel is not None:
            unit = self._unit_label_for_metric(str(metric_sel))
            y_label = str(metric_sel) if not unit else f"{metric_sel} ({unit})"

        ax.set_xlabel("Time (hr)")
        ax.set_ylabel(y_label)
        ax.set_title("Drainage/Structure coupling time series")
        if len(grouped) > 1:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()
