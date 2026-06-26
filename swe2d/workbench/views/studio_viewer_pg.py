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
    """pyqtgraph-based time-series plot for the HYDRA2D View dock.

    Protocol matches PlotViewWidget: set_data(), refresh(), selected_metric.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "Time Series"
        self._mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6
        self._selected_element_id: str = ""
        self._selected_metric: str = "flow_cms"

        # Cached plot data — (x_hr, y) per run key, so we don't re-load on refresh
        self._plot_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._plot_widget: Optional[pg.PlotWidget] = None
        self._plot_items: List[pg.PlotDataItem] = []
        self._vline: Optional[pg.InfiniteLine] = None
        self._hover_label: Optional[pg.TextItem] = None
        self._hover_crosshair: Tuple[pg.InfiniteLine, pg.InfiniteLine] | None = None
        self._metric_combo: Optional[QtWidgets.QComboBox] = None
        self._table_widget: Optional[QtWidgets.QTableWidget] = None
        self.show_table_toggle: Optional[QtWidgets.QCheckBox] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the widget: combo bar, pyqtgraph plot, data table."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not _HAVE_PG:
            label = QtWidgets.QLabel(
                "pyqtgraph not available.\n"
                "Install: conda install pyqtgraph  or  pip install pyqtgraph"
            )
            label.setWordWrap(True)
            root.addWidget(label)
            return

        # Top bar: variable combo + table toggle
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addStretch(1)
        lbl = QtWidgets.QLabel("Variable:")
        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItem("Flow (m³/s)", "flow_cms")
        self._metric_combo.addItem("Depth (m)", "depth_m")
        self._metric_combo.addItem("WSE (m)", "wse_m")
        self._metric_combo.addItem("Velocity (m/s)", "velocity_ms")
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        top_bar.addWidget(lbl)
        top_bar.addWidget(self._metric_combo)
        top_bar.addSpacing(12)

        self.show_table_toggle = QtWidgets.QCheckBox("Show data table")
        self.show_table_toggle.setChecked(False)
        self.show_table_toggle.toggled.connect(self._on_table_toggle)
        top_bar.addWidget(self.show_table_toggle)
        root.addLayout(top_bar)

        # pyqtgraph plot
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(200)
        self._plot_widget.setBackground("white")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", f"Time ({_TIME_UNIT})")
        self._plot_widget.setLabel("left", "Flow (m³/s)")
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.setMenuEnabled(False)  # cleaner UX

        # Hover text item (hidden until mouse moves)
        self._hover_label = pg.TextItem("", anchor=(0, 1), color=(0, 0, 0))
        self._hover_label.setZValue(100)
        self._hover_label.setVisible(False)
        self._plot_widget.addItem(self._hover_label)

        # Crosshair lines (hidden)
        self._hover_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("0.5", width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self._hover_vline.setVisible(False)
        self._hover_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("0.5", width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self._hover_hline.setVisible(False)
        self._plot_widget.addItem(self._hover_vline)
        self._plot_widget.addItem(self._hover_hline)

        # Mouse hover proxy
        proxy = pg.SignalProxy(
            self._plot_widget.scene().sigMouseMoved,
            rateLimit=30,
            slot=self._on_mouse_moved,
        )

        # Legend
        self._plot_widget.addLegend()

        root.addWidget(self._plot_widget, 1)

        # Data table (hidden by default)
        self._table_widget = QtWidgets.QTableWidget()
        self._table_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_widget.setAlternatingRowColors(True)
        self._table_widget.horizontalHeader().setStretchLastSection(True)
        self._table_widget.setVisible(False)
        root.addWidget(self._table_widget)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

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
    # Public protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def canvas(self):
        """Compatibility — return the plot widget."""
        return self._plot_widget

    @property
    def fig(self):
        """Compatibility — return None (no matplotlib figure)."""
        return None

    @property
    def selected_metric(self) -> str:
        return self._selected_metric

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        self._selected_metric = str(metric) if metric else "flow_cms"
        if self._metric_combo is not None:
            idx = self._metric_combo.findData(self._selected_metric)
            if idx >= 0:
                self._metric_combo.setCurrentIndex(idx)

    @property
    def selected_element_id(self) -> str:
        return self._selected_element_id

    @selected_element_id.setter
    def selected_element_id(self, element_id: str) -> None:
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
            self._populate_metric_combo()
        self._h_min = float(h_min)

    def set_render_fn(self, fn) -> None:
        """No-op — pyqtgraph handles rendering directly."""

    def refresh(self) -> None:
        """Re-plot the time series with current data and metric."""
        if not _HAVE_PG or self._result_data is None or self._plot_widget is None:
            return

        from swe2d.results.queries import load_timeseries as _load_ts
        from swe2d.results.queries import load_timeseries_from_live as _load_ts_live

        data = self._result_data
        line_id = getattr(data, "_line_id", -1)
        var_key = self._selected_metric
        is_live = getattr(data, "data_source", "") == "live"
        run_records = data.get_enabled_run_records()

        if line_id < 0 or not run_records:
            self._plot_widget.clear()
            self._plot_widget.plot([0], [0], pen=None, symbol=None)  # force axes
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)
            return

        # Update axis label
        lu = getattr(data, "_length_unit", "")
        ylabel = _label_for_var(var_key, lu)
        self._plot_widget.setLabel("left", ylabel)

        # Clear old items (keep legend items by clearing and re-adding)
        self._plot_widget.clear()
        self._plot_items = []
        self._hover_vline.setVisible(False)
        self._hover_hline.setVisible(False)
        self._hover_label.setVisible(False)

        # Re-add hover / crosshair items (cleared by .clear())
        self._plot_widget.addItem(self._hover_label)
        self._plot_widget.addItem(self._hover_vline)
        self._plot_widget.addItem(self._hover_hline)

        plotted = 0
        for rec in run_records:
            raw = (
                _load_ts_live(data, str(rec.run_id), int(line_id))
                if is_live else
                _load_ts(str(rec.gpkg_path), str(rec.run_id), int(line_id))
            )
            if not raw or var_key not in raw:
                continue
            t_hr = raw["t_s"] / 3600.0
            vals = raw[var_key]
            color = _c2q(rec.color)
            pen = pg.mkPen(color=color, width=1.6)
            item = self._plot_widget.plot(
                t_hr, vals,
                pen=pen,
                name=rec.display_label(),
            )
            self._plot_items.append(item)
            plotted += 1

        # Vertical line at current time
        t_hr_now = getattr(data, "current_time_sec", 0.0) / 3600.0
        self._vline = pg.InfiniteLine(
            pos=t_hr_now, angle=90,
            pen=pg.mkPen("0.5", width=0.9, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._vline.setZValue(50)
        self._plot_widget.addItem(self._vline)

        if not plotted:
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)

        # Re-enable auto-range after data update
        self._plot_widget.plotItem.autoRange()

        # Re-populate table if visible
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
        line_id = getattr(data, "_line_id", -1)
        var_key = self._selected_metric
        from swe2d.results.queries import load_timeseries as _load_ts
        from swe2d.results.queries import load_timeseries_from_live as _load_ts_live
        is_live = getattr(data, "data_source", "") == "live"

        for rec in data.get_enabled_run_records():
            raw = (
                _load_ts_live(data, str(rec.run_id), int(line_id))
                if is_live else
                _load_ts(str(rec.gpkg_path), str(rec.run_id), int(line_id))
            )
            if raw and var_key in raw:
                n = len(raw["t_s"])
                for i in range(min(n, 5000)):
                    row = {"t_s": raw["t_s"][i], var_key: raw[var_key][i]}
                    records.append(row)
                cols = sorted(raw.keys())
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
                self._table_widget.setItem(
                    i, j,
                    QtWidgets.QTableWidgetItem("" if val is None else f"{val:.6g}"),
                )

    # ------------------------------------------------------------------
    # Metric combo helpers
    # ------------------------------------------------------------------

    def _populate_metric_combo(self) -> None:
        """Populate metric combo from data coupling records (Structure/Network mode)."""
        # Time Series has a fixed set of variables — no dynamic population needed
        pass
