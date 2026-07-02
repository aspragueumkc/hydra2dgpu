"""PyQtGraph-based Time Series plot widget — drop-in for PlotViewWidget.

Replaces the matplotlib FigureCanvas for the Time Series tab with a
pyqtgraph PlotWidget, giving:
- Hardware-accelerated rendering (partial updates, no full redraws)
- Native zoom (scroll wheel) and pan (drag + right-drag)
- Hover crosshair with data value readout
- Smoother animation during temporal playback

MVP Architecture
----------------
This module is a **View** component.  It owns the pyqtgraph PlotWidget
and all Qt widgets.  Data loading is delegated to the data layer
(swe2d.results.queries) — no business logic here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt.QtCore import Qt

try:
    import pyqtgraph as pg
    _HAVE_PG = True
except ImportError:
    _HAVE_PG = False


# Unit helpers (mirrors results_render_service._unit_labels)
def _unit_labels(length_unit: str = "") -> dict:
    from swe2d import units as _u
    lu = str(length_unit or _u.length_unit_name() or "m").strip().lower()
    if lu == "ft":
        return {"len": "ft", "flow": "ft³/s", "vel": "ft/s"}
    return {"len": "m", "flow": "m³/s", "vel": "m/s"}


def _label_for_var(var_key: str, length_unit: str = "") -> str:
    u = _unit_labels(length_unit)
    table = {
        "flow_cms":      f"Flow ({u['flow']})",
        "depth_m":       f"Depth ({u['len']})",
        "wse_m":         f"WSE ({u['len']})",
        "velocity_ms":   f"Velocity ({u['vel']})",
    }
    return table.get(str(var_key), str(var_key))


def _var_from_label(label: str) -> str:
    """Reverse-lookup: given a display label, return the var key."""
    rev = {
        "flow_cms": "Flow",
        "depth_m": "Depth",
        "wse_m": "WSE",
        "velocity_ms": "Velocity",
    }
    for key, frag in rev.items():
        if frag in label:
            return key
    return "flow_cms"


# ── Colour conversion ────────────────────────────────────────────────

def _c2q(rgb: Tuple[int, int, int]) -> QtGui.QColor:
    """Convert an (R, G, B) tuple to a QColor."""
    return QtGui.QColor(*rgb)


# ── The widget ───────────────────────────────────────────────────────

_TIME_UNIT = "hr"


class PGTimeSeriesWidget(QtWidgets.QWidget):
    """pyqtgraph-based time-series plot for lines, structures, and drainage.

    Unified viewer for all time-series data.  The user selects an element
    type (Line, Structure, Drainage Node, Drainage Link) and an element ID,
    then plots the variable for all enabled runs.

    Protocol: set_data(), refresh(), selected_metric, selected_element_id.
    """

    _ELEMENT_TYPES = [
        ("Line", "line"),
        ("Mesh Cell", "mesh_cell"),
        ("Structure", "structure"),
        ("Drainage Node", "drainage_node"),
        ("Drainage Link", "drainage_link"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "Time Series"
        self._mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6
        self._selected_element_id: str = ""
        self._selected_metric: str = "flow_cms"

        # Cached plot data
        self._plot_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._plot_widget: Optional[pg.PlotWidget] = None
        self._plot_items: List[pg.PlotDataItem] = []
        self._vline: Optional[pg.InfiniteLine] = None
        self._hover_label: Optional[pg.TextItem] = None
        self._hover_vline: Optional[pg.InfiniteLine] = None
        self._hover_hline: Optional[pg.InfiniteLine] = None
        self._metric_combo: Optional[QtWidgets.QComboBox] = None
        self._element_type_combo: Optional[QtWidgets.QComboBox] = None
        self._element_id_combo: Optional[QtWidgets.QComboBox] = None
        self._table_widget: Optional[QtWidgets.QTableWidget] = None
        self.show_table_toggle: Optional[QtWidgets.QCheckBox] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build: element type/id, variable combo, pyqtgraph plot, data table."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not _HAVE_PG:
            label = QtWidgets.QLabel("pyqtgraph not available.\nInstall: conda install pyqtgraph")
            label.setWordWrap(True)
            root.addWidget(label)
            return

        # ── Top bar: element type, element ID, variable (row 1) ──
        top_bar = QtWidgets.QVBoxLayout()
        top_bar.setSpacing(2)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)

        def _make_combo(max_w: int = 120) -> QtWidgets.QComboBox:
            c = QtWidgets.QComboBox()
            c.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
            c.setMinimumWidth(60)
            c.setMaximumWidth(max_w)
            return c

        row1.addWidget(QtWidgets.QLabel("Type:"))
        self._element_type_combo = _make_combo(120)
        self._element_type_combo.setToolTip("Element type for time-series data: Line, Mesh Cell, Structure, or Drainage.")
        for label, key in self._ELEMENT_TYPES:
            self._element_type_combo.addItem(label, key)
        self._element_type_combo.currentIndexChanged.connect(self._on_element_type_changed)
        row1.addWidget(self._element_type_combo)
        row1.addSpacing(4)

        row1.addWidget(QtWidgets.QLabel("Elem:"))
        self._element_id_combo = _make_combo(140)
        self._element_id_combo.setToolTip("Select the specific element ID to plot.")
        self._element_id_combo.currentIndexChanged.connect(self._on_element_id_changed)
        row1.addWidget(self._element_id_combo)
        row1.addSpacing(4)

        row1.addWidget(QtWidgets.QLabel("Var:"))
        self._metric_combo = _make_combo(120)
        self._metric_combo.setToolTip("Variable to plot: flow, depth, WSE, velocity.")
        self._repopulate_combo_items()
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        row1.addWidget(self._metric_combo)

        row1.addStretch(1)
        top_bar.addLayout(row1)

        # ── Row 2: toggles, save, settings ──
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)

        self.show_table_toggle = QtWidgets.QCheckBox("Table")
        self.show_table_toggle.setChecked(False)
        self.show_table_toggle.setToolTip("Show/hide the time-series data table below the plot.")
        self.show_table_toggle.toggled.connect(self._on_table_toggle)
        row2.addWidget(self.show_table_toggle)

        row2.addStretch(1)

        # Save button
        self._save_btn = QtWidgets.QPushButton("💾")
        self._save_btn.setFixedSize(24, 24)
        self._save_btn.setToolTip("Save plot / data")
        self._save_menu = QtWidgets.QMenu(self._save_btn)
        self._save_menu.addAction("Save plot as PNG", self._save_plot_png)
        self._save_menu.addAction("Save plot as SVG", self._save_plot_svg)
        self._save_menu.addAction("Save plot as PDF / Print", self._save_plot_pdf)
        self._save_menu.addSeparator()
        self._save_menu.addAction("Save data as CSV", self._save_data_csv)
        self._save_btn.setMenu(self._save_menu)
        row2.addWidget(self._save_btn)

        # Settings button
        self._settings_btn = QtWidgets.QPushButton("⚙")
        self._settings_btn.setFixedSize(24, 24)
        self._settings_btn.setToolTip("Plot settings")
        self._settings_menu = QtWidgets.QMenu(self._settings_btn)
        self._settings_act_grid = self._settings_menu.addAction("Show grid")
        self._settings_act_grid.setCheckable(True)
        self._settings_act_grid.setChecked(True)
        self._settings_act_grid.toggled.connect(self._on_toggle_grid)
        self._settings_act_legend = self._settings_menu.addAction("Show legend")
        self._settings_act_legend.setCheckable(True)
        self._settings_act_legend.setChecked(True)
        self._settings_act_legend.toggled.connect(self._on_toggle_legend)
        self._settings_act_crosshair = self._settings_menu.addAction("Show crosshair")
        self._settings_act_crosshair.setCheckable(True)
        self._settings_act_crosshair.setChecked(True)
        self._settings_act_crosshair.toggled.connect(self._on_toggle_crosshair)
        self._settings_btn.setMenu(self._settings_menu)
        row2.addWidget(self._settings_btn)

        top_bar.addLayout(row2)
        root.addLayout(top_bar)

        # ── pyqtgraph plot ──
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(200)
        self._plot_widget.setBackground("white")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", f"Time ({_TIME_UNIT})")
        self._plot_widget.setLabel("left", "Value")
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.setMenuEnabled(False)

        self._hover_label = pg.TextItem("", anchor=(0, 1), color=(0, 0, 0))
        self._hover_label.setZValue(100)
        self._hover_label.setVisible(False)
        self._plot_widget.addItem(self._hover_label)

        self._hover_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(128, 128, 128), width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self._hover_vline.setVisible(False)
        self._hover_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(128, 128, 128), width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self._hover_hline.setVisible(False)
        self._plot_widget.addItem(self._hover_vline)
        self._plot_widget.addItem(self._hover_hline)

        proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )
        self._plot_widget.addLegend()
        root.addWidget(self._plot_widget, 1)

        # ── Data table ──
        self._table_widget = QtWidgets.QTableWidget()
        self._table_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_widget.setAlternatingRowColors(True)
        self._table_widget.horizontalHeader().setStretchLastSection(True)
        self._table_widget.setVisible(False)
        root.addWidget(self._table_widget)

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Return the plot mode (e.g., 'Time Series')."""
        return self._mode

    @property
    def canvas(self):
        """Return the pyqtgraph PlotWidget canvas."""
        return self._plot_widget

    @property
    def fig(self):
        """Return None (pyqtgraph uses its own figure internally)."""
        return None

    @property
    def selected_metric(self) -> str:
        """Return the currently selected metric key (e.g., 'flow_cms', 'depth_m')."""
        return self._selected_metric

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        """Set the selected metric and update the UI combo."""
        self._selected_metric = str(metric) if metric else "flow_cms"
        if self._metric_combo is not None:
            idx = self._metric_combo.findData(self._selected_metric)
            if idx >= 0:
                self._metric_combo.setCurrentIndex(idx)

    @property
    def selected_element_id(self) -> str:
        """Return the currently selected element ID (line ID, structure ID, etc.)."""
        return self._selected_element_id

    @selected_element_id.setter
    def selected_element_id(self, element_id: str) -> None:
        """Set the selected element ID and update the UI combo."""
        self._selected_element_id = str(element_id) if element_id else ""
        if self._element_id_combo is not None and element_id:
            idx = self._element_id_combo.findData(self._selected_element_id)
            if idx >= 0:
                self._element_id_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _repopulate_combo_items(self) -> None:
        """Populate/populate the variable combo with unit-agnostic labels.

        Uses _label_for_var which dynamically resolves the unit system
        (SI or USC) from swe2d.units — no hardcoded units in the UI.
        """
        if self._metric_combo is None:
            return
        prev_data = self._metric_combo.currentData()
        self._metric_combo.blockSignals(True)
        self._metric_combo.clear()
        for key in ("flow_cms", "depth_m", "wse_m", "velocity_ms"):
            self._metric_combo.addItem(_label_for_var(key), key)
        idx = self._metric_combo.findData(prev_data)
        if idx >= 0:
            self._metric_combo.setCurrentIndex(idx)
        self._metric_combo.blockSignals(False)

    def _on_element_type_changed(self) -> None:
        """Re-populate element ID combo and metric combo, then refresh."""
        self._populate_element_id_combo()
        self._repopulate_metric_for_etype()
        self.refresh()

    def _on_element_id_changed(self) -> None:
        """Update selected element ID and refresh."""
        self._selected_element_id = str(self._element_id_combo.currentData() or "")
        self.refresh()

    def _on_metric_changed(self) -> None:
        """Re-plot when the metric combo changes."""
        self._selected_metric = str(self._metric_combo.currentData() or "flow_cms")
        self.refresh()

    def _on_table_toggle(self, visible: bool) -> None:
        """Show/hide the data table."""
        if self._table_widget is not None:
            self._table_widget.setVisible(visible)
            if visible:
                self._populate_table()

    def _on_mouse_moved(self, evt) -> None:
        """Handle mouse hover on the plot — update crosshair and data label."""
        if self._plot_widget is None or not self._plot_items:
            return
        pos = evt[0]  # QPointF
        plot = self._plot_widget.plotItem
        vb = plot.vb
        if vb is None:
            return
        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()

        # Move crosshair
        self._hover_vline.setPos(mx)
        self._hover_hline.setPos(my)
        self._hover_vline.setVisible(True)
        self._hover_hline.setVisible(True)

        # Find closest data point across all curves
        closest_dist = float("inf")
        closest_text = ""
        for item in self._plot_items:
            x_data = item.xData
            y_data = item.yData
            if x_data is None or y_data is None or len(x_data) == 0:
                continue
            idx = np.argmin(np.abs(x_data - mx))
            dist = abs(x_data[idx] - mx)
            if dist < closest_dist:
                closest_dist = dist
                label = item.name() or "?"
                closest_text = f"{label}: ({x_data[idx]:.4g}, {y_data[idx]:.4g})"

        if closest_text:
            self._hover_label.setText(closest_text)
            self._hover_label.setVisible(True)
            # Position above the crosshair
            self._hover_label.setPos(mx, my)
        else:
            self._hover_label.setVisible(False)

    # ------------------------------------------------------------------
    # Element combo population
    # ------------------------------------------------------------------

    def _populate_element_id_combo(self) -> None:
        """Populate element ID combo based on selected element type."""
        if self._element_id_combo is None:
            return
        etype = str(self._element_type_combo.currentData() or "line")
        prev_data = self._element_id_combo.currentData()
        self._element_id_combo.blockSignals(True)
        self._element_id_combo.clear()

        data = self._result_data
        if etype == "line":
            if data is not None:
                line_ids = data.get_line_ids()
                for lid in line_ids:
                    self._element_id_combo.addItem(f"Line {lid}", lid)
        elif etype == "mesh_cell":
            # Populate with cell indices from live snapshots or mesh data
            n_cells = 0
            if data is not None:
                snaps = getattr(data, "_live_snapshot_timesteps", [])
                if snaps:
                    n_cells = int(getattr(snaps[0][1], "size", 0))
            if n_cells == 0 and self._mesh_data is not None:
                n_cells = int(getattr(self._mesh_data.get("cell_nodes"), "size", 0)) // 3
            for ci in range(n_cells):
                self._element_id_combo.addItem(f"Cell {ci}", ci)
        else:
            if data is not None:
                # Ensure coupling records are loaded before querying
                first_enabled = None
                for rec in getattr(data, "_run_records", []):
                    if rec.enabled:
                        first_enabled = rec
                        break
                if first_enabled is not None and data._coupling_run_id != str(first_enabled.run_id):
                    data.load_coupling_records(str(first_enabled.run_id))
                coupling = data.get_coupling_records()
                seen = set()
                for rec in coupling:
                    if str(rec.get("component", "") or "") != etype:
                        continue
                    oid = str(rec.get("object_id", "") or "")
                    if not oid or oid in seen:
                        continue
                    seen.add(oid)
                    oname = str(rec.get("object_name", "") or "")
                    lbl = f"{oname} ({oid})" if oname else oid
                    self._element_id_combo.addItem(lbl, oid)

        if prev_data is not None:
            idx = self._element_id_combo.findData(prev_data)
            if idx >= 0:
                self._element_id_combo.setCurrentIndex(idx)
        self._element_id_combo.blockSignals(False)
        self._selected_element_id = str(self._element_id_combo.currentData() or "")

    # ------------------------------------------------------------------
    # Data loading by element type
    # ------------------------------------------------------------------

    def _load_timeseries_for_type(
        self, run_rec, element_id, var_key: str, etype: str
    ) -> dict:
        """Load time-series data for a line (baked), or coupling records for structure/drainage."""
        from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
        import numpy as np

        data = self._result_data

        if etype == "line":
            try:
                lid = int(element_id)
            except (TypeError, ValueError):
                return {}
            raw = load_baked_line_timeseries(
                data, str(run_rec.run_id), lid
            ) if getattr(data, "_live_times", None) is not None and data._live_times.size > 0 else \
                load_baked_line_timeseries(
                    str(run_rec.gpkg_path), str(run_rec.run_id), lid
                )
            return raw if raw else {}

        # Coupling-based types
        eid = self._selected_element_id
        if not eid:
            return {}

        # Ensure coupling records are loaded for this run
        if data._coupling_run_id != str(run_rec.run_id):
            data.load_coupling_records(str(run_rec.run_id))

        records = data.get_coupling_records()
        if not records:
            return {}

        # Filter by component, object_id, and metric
        filtered = [
            r for r in records
            if str(r.get("component", "") or "") == etype
            and str(r.get("object_id", "") or "") == eid
            and str(r.get("metric", "") or "") == var_key
        ]
        if not filtered:
            return {}

        filtered.sort(key=lambda r: float(r.get("t_s", 0.0)))
        t_vals = np.array([float(r["t_s"]) for r in filtered], dtype=np.float64)
        v_vals = np.array([float(r["value"]) for r in filtered], dtype=np.float64)
        return {"t_s": t_vals, var_key: v_vals}

    # ------------------------------------------------------------------
    # Save / Export
    # ------------------------------------------------------------------

    def _save_plot_png(self) -> None:
        """Save the current plot as a PNG image via file dialog."""
        if self._plot_widget is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot as PNG", "", "PNG Image (*.png)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter
            exporter = ImageExporter(self._plot_widget.plotItem)
            exporter.export(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    def _save_plot_svg(self) -> None:
        """Save the current plot as an SVG vector graphic."""
        if self._plot_widget is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot as SVG", "", "SVG Image (*.svg)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import SVGExporter
            exporter = SVGExporter(self._plot_widget.plotItem)
            exporter.export(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    def _save_plot_pdf(self) -> None:
        """Save the current plot as PDF or send to a printer."""
        if self._plot_widget is None:
            return
        try:
            from pyqtgraph.exporters import PrintExporter
            exporter = PrintExporter(self._plot_widget.plotItem)
            exporter.export()  # shows native print dialog — user can pick PDF or physical printer
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    def _save_data_csv(self) -> None:
        """Save the currently plotted time-series data as CSV."""
        if not self._plot_items:
            QtWidgets.QMessageBox.information(
                self, "No Data", "No plot data to export."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data as CSV", "", "CSV File (*.csv)",
        )
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                # Header: time, name1, name2, ...
                names = [item.name() or f"Series_{i}" for i, item in enumerate(self._plot_items)]
                writer.writerow([f"Time ({_TIME_UNIT})"] + names)
                # Collect all x data (time) — use the first series as reference
                x_data = None
                y_series = []
                for item in self._plot_items:
                    if item.xData is not None and item.yData is not None:
                        if x_data is None:
                            x_data = item.xData
                        y_series.append(item.yData)
                    else:
                        y_series.append(np.array([]))
                if x_data is None:
                    return
                for i in range(len(x_data)):
                    row = [f"{x_data[i]:.6g}"]
                    for ys in y_series:
                        row.append(f"{ys[i]:.6g}" if i < len(ys) else "")
                    writer.writerow(row)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Customization toggles
    # ------------------------------------------------------------------

    def _on_toggle_grid(self, enabled: bool) -> None:
        """Toggle grid visibility."""
        if self._plot_widget is not None:
            self._plot_widget.showGrid(x=enabled, y=enabled, alpha=0.3)

    def _on_toggle_legend(self, enabled: bool) -> None:
        """Toggle legend visibility."""
        if self._plot_widget is not None:
            legend = self._plot_widget.plotItem.legend
            if legend is not None:
                legend.setVisible(enabled)

    def _on_toggle_crosshair(self, enabled: bool) -> None:
        """Toggle crosshair visibility."""
        self._hover_vline.setVisible(enabled)
        self._hover_hline.setVisible(enabled)
        if not enabled:
            self._hover_label.setVisible(False)

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Return the plot mode (e.g., 'Time Series')."""
        return self._mode

    @property
    def canvas(self):
        """Return the pyqtgraph PlotWidget canvas."""
        return self._plot_widget

    @property
    def fig(self):
        """Return None (pyqtgraph uses its own figure internally)."""
        return None

    @property
    def selected_metric(self) -> str:
        """Return the currently selected metric key (e.g., 'flow_cms', 'depth_m')."""
        return self._selected_metric

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        """Set the selected metric and update the UI combo."""
        self._selected_metric = str(metric) if metric else "flow_cms"
        if self._metric_combo is not None:
            idx = self._metric_combo.findData(self._selected_metric)
            if idx >= 0:
                self._metric_combo.setCurrentIndex(idx)

    @property
    def selected_element_id(self) -> str:
        """Return the currently selected element ID (line ID, structure ID, etc.)."""
        return self._selected_element_id

    @selected_element_id.setter
    def selected_element_id(self, element_id: str) -> None:
        """Set the selected element ID."""
        self._selected_element_id = str(element_id) if element_id else ""

    def set_data(
        self,
        mesh_data: Optional[Dict[str, np.ndarray]] = None,
        result_data: Any = None,
        h_min: float = 1.0e-6,
    ) -> None:
        """Set data sources and refresh."""
        if mesh_data is not None:
            self._mesh_data = mesh_data
        if result_data is not None:
            self._result_data = result_data
            self._repopulate_combo_items()
            self._populate_element_id_combo()
            self._populate_metric_combo()
            # Pre-load coupling records so structure/drainage element IDs
            # are available in the dropdown immediately.
            first_enabled = None
            for rec in getattr(result_data, "_run_records", []):
                if rec.enabled:
                    first_enabled = rec
                    break
            if first_enabled is not None and getattr(result_data, "_coupling_run_id", "") != str(first_enabled.run_id):
                result_data.load_coupling_records(str(first_enabled.run_id))
        self._h_min = float(h_min)

    def set_render_fn(self, fn) -> None:
        """No-op — pyqtgraph handles rendering directly."""

    def refresh(self) -> None:
        """Re-plot the time series with current element type, ID, and metric."""
        if not _HAVE_PG or self._result_data is None or self._plot_widget is None:
            return

        data = self._result_data
        etype = str(self._element_type_combo.currentData() or "line")
        element_id = self._selected_element_id
        var_key = self._selected_metric
        run_records = data.get_enabled_run_records()

        if not run_records:
            self._plot_widget.clear()
            self._plot_widget.plot([0], [0], pen=None)
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)
            return

        lu = getattr(data, "_length_unit", "")
        ylabel = _label_for_var(var_key, lu)
        self._plot_widget.setLabel("left", ylabel)

        self._plot_widget.clear()
        self._plot_items = []
        self._hover_vline.setVisible(False)
        self._hover_hline.setVisible(False)
        self._hover_label.setVisible(False)
        self._plot_widget.addItem(self._hover_label)
        self._plot_widget.addItem(self._hover_vline)
        self._plot_widget.addItem(self._hover_hline)

        plotted = 0
        for rec in run_records:
            eid = int(element_id) if etype == "line" and element_id else element_id
            raw = self._load_timeseries_for_type(rec, eid, var_key, etype)
            if not raw or var_key not in raw:
                continue
            t_hr = raw["t_s"] / 3600.0
            vals = raw[var_key]
            color = _c2q(rec.color)
            pen = pg.mkPen(color=color, width=1.6)
            item = self._plot_widget.plot(t_hr, vals, pen=pen, name=rec.display_label())
            self._plot_items.append(item)
            plotted += 1

        t_hr_now = getattr(data, "current_time_sec", 0.0) / 3600.0
        self._vline = pg.InfiniteLine(
            pos=t_hr_now, angle=90,
            pen=pg.mkPen(color=(128, 128, 128), width=0.9, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._vline.setZValue(50)
        self._plot_widget.addItem(self._vline)

        if not plotted:
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)

        self._plot_widget.plotItem.autoRange()

        if self._table_widget is not None and self._table_widget.isVisible():
            self._populate_table()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        """Fill the data table from the result data."""
        if self._table_widget is None:
            return
        data = self._result_data
        self._table_widget.setRowCount(0)
        self._table_widget.setColumnCount(0)
        if data is None:
            return

        records = []
        cols = []
        var_key = self._selected_metric
        etype = str(self._element_type_combo.currentData() or "line")
        element_id = self._selected_element_id

        for rec in data.get_enabled_run_records():
            eid = int(element_id) if etype == "line" and element_id else element_id
            raw = self._load_timeseries_for_type(rec, eid, var_key, etype)
            if raw and var_key in raw:
                cols = sorted(raw.keys())
                n = min(len(raw["t_s"]), 5000)
                for i in range(n):
                    records.append({k: raw[k][i] for k in cols})
                break

        if not records or not cols:
            return

        self._table_widget.setColumnCount(len(cols))
        self._table_widget.setHorizontalHeaderLabels(cols)
        n = min(len(records), 5000)
        self._table_widget.setRowCount(n)
        for i, r in enumerate(records[:n]):
            for j, c in enumerate(cols):
                val = r.get(c, "")
                if isinstance(val, str):
                    display = val
                else:
                    display = "" if val is None else f"{val:.6g}"
                self._table_widget.setItem(i, j, QtWidgets.QTableWidgetItem(display))

    # ------------------------------------------------------------------
    # Metric combo helpers
    # ------------------------------------------------------------------

    def _populate_metric_combo(self) -> None:
        """Populate metric combo from data coupling records."""
        self._repopulate_metric_for_etype()

    def _repopulate_metric_for_etype(self) -> None:
        """Populate the metric combo based on the selected element type.

        Line → TS metrics (flow_cms, depth_m, wse_m, velocity_ms).
        Coupling types → metrics from the coupling records (flow, depth, invert, etc.).
        """
        if self._metric_combo is None:
            return
        etype = str(self._element_type_combo.currentData() or "line")
        prev_data = self._metric_combo.currentData()

        self._metric_combo.blockSignals(True)
        self._metric_combo.clear()

        if etype == "line":
            # Standard TS metrics (unit-aware labels)
            for key in ("flow_cms", "depth_m", "wse_m", "velocity_ms"):
                self._metric_combo.addItem(_label_for_var(key), key)
        else:
            # Coupling-based types — collect unique metric values from records
            data = self._result_data
            coupling = data.get_coupling_records() if data is not None else []
            seen: set = set()
            for rec in coupling:
                comp = str(rec.get("component", "") or "")
                if comp != etype:
                    continue
                m = str(rec.get("metric", "") or "")
                if m and m not in seen:
                    seen.add(m)
            metrics = sorted(seen) or ["flow"]
            for m in metrics:
                self._metric_combo.addItem(m, m)

        # Restore previous selection if still valid
        idx = self._metric_combo.findData(prev_data)
        if idx >= 0:
            self._metric_combo.setCurrentIndex(idx)
        self._metric_combo.blockSignals(False)

        # Sync _selected_metric to the combo's actual current value
        # (the combo may have been repopulated with different items)
        self._selected_metric = str(self._metric_combo.currentData() or "flow_cms")
