"""PyQtGraph-based Profile plot widget — drop-in for matplotlib PlotViewWidget.

Replaces the matplotlib FigureCanvas for the Profile tab with a
pyqtgraph PlotWidget, giving:
- Hardware-accelerated rendering (partial updates, no full redraws)
- Native zoom (scroll wheel) and pan (drag + right-drag)
- Hover crosshair with data value readout
- Smoother animation during temporal playback

Renders:
  - WSE + Bed: bed fill, bed line, WSE line, fill-between, optional colormap shading
  - Depth / Velocity / EGL Error: single line along station
  - Structure annotations (vertical lines with flow labels)

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


# ── Unit helpers (mirrors results_render_service._unit_labels) ───────

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
        "station_m":     f"Station ({u['len']})",
        "bed_m":         f"Bed ({u['len']})",
        "egl_m":         f"EGL Error ({u['len']})",
    }
    return table.get(str(var_key), str(var_key))


# ── Colour conversion ────────────────────────────────────────────────

def _c2q(rgb: Tuple[int, int, int]) -> QtGui.QColor:
    """Convert an (R, G, B) tuple to a QColor."""
    return QtGui.QColor(*rgb)


def _c2q_alpha(rgb: Tuple[int, int, int], alpha: int) -> QtGui.QColor:
    """Convert an (R, G, B) tuple to a QColor with alpha."""
    c = QtGui.QColor(*rgb)
    c.setAlpha(alpha)
    return c


# ── Matplotlib colormap name -> brush generator ──────────────────────

def _cmap_brush(cmap_name: str, t: float) -> QtGui.QColor:
    """Return a QColor for a normalized position *t* in *cmap_name*.

    Uses a simple turbo-like fallback if matplotlib is unavailable.
    """
    try:
        from matplotlib import cm as mpl_cm, colors as mpl_colors
        cmap = mpl_cm.get_cmap(cmap_name)
        r, g, b, _ = cmap(float(t))
        return QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        # Turbo-like fallback
        t = float(np.clip(t, 0.0, 1.0))
        if t < 0.25:
            r, g, b = 48 + int(t * 8), 18 + int(t * 328), 59 + int(t * 644)
        elif t < 0.5:
            r, g, b = 50 + int((t - 0.25) * -36), 100 + int((t - 0.25) * 348), 220 + int((t - 0.25) * 64)
        elif t < 0.75:
            r, g, b = 41 + int((t - 0.5) * 332), 187 + int((t - 0.5) * 192), 236 + int((t - 0.5) * -596)
        else:
            r, g, b = 124 + int((t - 0.75) * 504), 234 + int((t - 0.75) * -116), 87 + int((t - 0.75) * -98)
        return QtGui.QColor(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


# ── The widget ───────────────────────────────────────────────────────

_TIME_UNIT = "hr"

_PROFILE_VAR_ITEMS = [
    ("WSE + Bed", "wse_bed"),
    ("Depth", "depth_m"),
    ("Velocity", "velocity_ms"),
    ("EGL Error", "egl_m"),
]

_FILL_ITEMS = [
    ("None", "none"),
    ("Depth", "depth_m"),
    ("Velocity", "velocity_ms"),
    ("Flow", "flow_cms"),
]

_CMAP_ITEMS = [
    ("Viridis", "viridis"),
    ("Turbo", "turbo"),
    ("Plasma", "plasma"),
    ("Inferno", "inferno"),
    ("Coolwarm", "coolwarm"),
]


class PGProfileWidget(QtWidgets.QWidget):
    """pyqtgraph-based longitudinal profile plot for sample-line cross-sections.

    Protocol: set_data(), refresh(), mode, selected_metric, selected_element_id.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "Profile"
        self._mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6

        # UI state
        self._line_id: int = -1
        self._prof_var_key: str = "wse_bed"
        self._prof_fill_key: str = "none"
        self._prof_cmap: str = "viridis"
        self._prof_show_structures: bool = True

        # Plot items
        self._plot_widget: Optional[pg.PlotWidget] = None
        self._plot_items: List[pg.PlotDataItem] = []
        self._fill_items: List[pg.FillBetweenItem] = []
        self._structure_items: List[pg.InfiniteLine] = []
        self._structure_labels: List[pg.TextItem] = []
        self._hover_label: Optional[pg.TextItem] = None
        self._hover_vline: Optional[pg.InfiniteLine] = None
        self._hover_hline: Optional[pg.InfiniteLine] = None

        # Combos
        self._line_combo: Optional[QtWidgets.QComboBox] = None
        self._var_combo: Optional[QtWidgets.QComboBox] = None
        self._fill_combo: Optional[QtWidgets.QComboBox] = None
        self._cmap_combo: Optional[QtWidgets.QComboBox] = None
        self._show_struct_chk: Optional[QtWidgets.QCheckBox] = None
        self._table_widget: Optional[QtWidgets.QTableWidget] = None
        self.show_table_toggle: Optional[QtWidgets.QCheckBox] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build: line selector, variable/fill/cmap combos, pyqtgraph plot, data table."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not _HAVE_PG:
            label = QtWidgets.QLabel(
                "pyqtgraph not available.\nInstall: conda install pyqtgraph"
            )
            label.setWordWrap(True)
            root.addWidget(label)
            return

        # ── Top bar ──
        top_bar = QtWidgets.QHBoxLayout()

        # Line selector
        top_bar.addWidget(QtWidgets.QLabel("Line:"))
        self._line_combo = QtWidgets.QComboBox()
        self._line_combo.currentIndexChanged.connect(self._on_line_changed)
        top_bar.addWidget(self._line_combo)
        top_bar.addSpacing(8)

        # Variable selector
        top_bar.addWidget(QtWidgets.QLabel("Var:"))
        self._var_combo = QtWidgets.QComboBox()
        for label, key in _PROFILE_VAR_ITEMS:
            self._var_combo.addItem(label, key)
        self._var_combo.currentIndexChanged.connect(self._on_var_changed)
        top_bar.addWidget(self._var_combo)
        top_bar.addSpacing(8)

        # Fill selector
        top_bar.addWidget(QtWidgets.QLabel("Fill:"))
        self._fill_combo = QtWidgets.QComboBox()
        for label, key in _FILL_ITEMS:
            self._fill_combo.addItem(label, key)
        self._fill_combo.currentIndexChanged.connect(self._on_fill_changed)
        top_bar.addWidget(self._fill_combo)
        top_bar.addSpacing(8)

        # Colormap selector
        top_bar.addWidget(QtWidgets.QLabel("Cmap:"))
        self._cmap_combo = QtWidgets.QComboBox()
        for label, key in _CMAP_ITEMS:
            self._cmap_combo.addItem(label, key)
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        top_bar.addWidget(self._cmap_combo)
        top_bar.addStretch(1)

        # Show structures toggle
        self._show_struct_chk = QtWidgets.QCheckBox("Structures")
        self._show_struct_chk.setChecked(True)
        self._show_struct_chk.toggled.connect(self._on_show_struct_changed)
        top_bar.addWidget(self._show_struct_chk)

        # Data table toggle
        self.show_table_toggle = QtWidgets.QCheckBox("Show data table")
        self.show_table_toggle.setChecked(False)
        self.show_table_toggle.toggled.connect(self._on_table_toggle)
        top_bar.addWidget(self.show_table_toggle)

        # Save button
        save_btn = QtWidgets.QPushButton("💾 Save")
        save_btn.setFixedHeight(24)
        save_menu = QtWidgets.QMenu(save_btn)
        save_menu.addAction("Save plot as PNG", self._save_plot_png)
        save_menu.addAction("Save plot as SVG", self._save_plot_svg)
        save_menu.addAction("Save plot as PDF / Print", self._save_plot_pdf)
        save_menu.addSeparator()
        save_menu.addAction("Save data as CSV", self._save_data_csv)
        save_btn.setMenu(save_menu)
        top_bar.addWidget(save_btn)

        root.addLayout(top_bar)

        # ── pyqtgraph plot ──
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(200)
        self._plot_widget.setBackground("white")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "Station (m)")
        self._plot_widget.setLabel("left", "Elevation (m)")
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.setMenuEnabled(False)

        # Crosshair
        self._hover_label = pg.TextItem("", anchor=(0, 1), color=(0, 0, 0))
        self._hover_label.setZValue(100)
        self._hover_label.setVisible(False)
        self._plot_widget.addItem(self._hover_label)

        self._hover_vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(color=(128, 128, 128), width=0.8,
                         style=QtCore.Qt.PenStyle.DashLine),
        )
        self._hover_vline.setVisible(False)
        self._hover_hline = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(color=(128, 128, 128), width=0.8,
                         style=QtCore.Qt.PenStyle.DashLine),
        )
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
        self._table_widget.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table_widget.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table_widget.setAlternatingRowColors(True)
        self._table_widget.horizontalHeader().setStretchLastSection(True)
        self._table_widget.setVisible(False)
        root.addWidget(self._table_widget)

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def canvas(self):
        return self._plot_widget

    @property
    def fig(self):
        return None

    @property
    def selected_metric(self) -> str:
        return self._prof_var_key

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        self._prof_var_key = str(metric) if metric else "wse_bed"
        if self._var_combo is not None:
            idx = self._var_combo.findData(self._prof_var_key)
            if idx >= 0:
                self._var_combo.setCurrentIndex(idx)

    @property
    def selected_element_id(self) -> str:
        return str(self._line_id)

    @selected_element_id.setter
    def selected_element_id(self, element_id: str) -> None:
        self._line_id = int(element_id) if element_id else -1
        if self._line_combo is not None and element_id:
            idx = self._line_combo.findData(self._line_id)
            if idx >= 0:
                self._line_combo.setCurrentIndex(idx)

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
            self._populate_line_combo()
        self._h_min = float(h_min)

    def set_render_fn(self, fn) -> None:
        """No-op — pyqtgraph handles rendering directly."""

    def refresh(self) -> None:
        """Re-render the profile with current line, variable, and display options."""
        if not _HAVE_PG or self._result_data is None or self._plot_widget is None:
            return

        data = self._result_data

        # Sync display options from result_data (set by toolbox controls)
        self._prof_fill_key = str(
            getattr(data, "prof_fill_key", self._prof_fill_key) or "none"
        )
        self._prof_cmap = str(
            getattr(data, "prof_cmap", self._prof_cmap) or "viridis"
        )
        self._prof_show_structures = bool(
            getattr(data, "prof_show_structures", self._prof_show_structures)
        )
        # Sync variable from result_data if set externally
        ext_var = getattr(data, "prof_var_key", None)
        if ext_var and ext_var != self._prof_var_key:
            self._prof_var_key = str(ext_var)
            if self._var_combo is not None:
                idx = self._var_combo.findData(self._prof_var_key)
                if idx >= 0:
                    self._var_combo.setCurrentIndex(idx)

        line_id = self._line_id
        t_sec = float(getattr(data, "current_time_sec", 0.0))
        run_records = data.get_enabled_run_records()
        var_key = self._prof_var_key
        fill_key = self._prof_fill_key
        cmap_name = self._prof_cmap
        show_structures = self._prof_show_structures
        use_fill_cmap = fill_key != "none"

        from swe2d.results.queries import (
            find_nearest_timestep,
            load_profile,
            load_profile_from_live,
            load_structure_flows_at_time,
        )
        from swe2d import units as _u

        is_live = getattr(data, "data_source", "") == "live"

        # Clear the plot
        self._plot_widget.clear()
        self._plot_items = []
        self._fill_items = []
        self._structure_items = []
        self._structure_labels = []
        self._hover_vline.setVisible(False)
        self._hover_hline.setVisible(False)
        self._hover_label.setVisible(False)
        self._plot_widget.addItem(self._hover_label)
        self._plot_widget.addItem(self._hover_vline)
        self._plot_widget.addItem(self._hover_hline)

        if not run_records or line_id < 0:
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)
            return

        # Set axis labels
        lu = getattr(data, "_length_unit", "")
        len_label = _unit_labels(lu)["len"]
        self._plot_widget.setLabel("bottom", f"Station ({len_label})")
        if var_key == "wse_bed":
            self._plot_widget.setLabel("left", f"Elevation ({len_label})")
        else:
            self._plot_widget.setLabel("left", _label_for_var(var_key, lu))

        plotted = 0
        bed_drawn = False
        structure_rows: List[Dict[str, Any]] = []
        fill_segments_data: List[Tuple[float, float, float, float, float]] = []
        # Each: (station_i, station_j, bed_i, wse_i, fill_val_mid)

        for rec in run_records:
            t = find_nearest_timestep(
                rec.gpkg_path, rec.run_id, line_id, t_sec
            )
            prof_data = (
                load_profile_from_live(data, str(rec.run_id), int(line_id), float(t))
                if is_live else
                load_profile(rec.gpkg_path, rec.run_id, line_id, t)
            )
            if not prof_data:
                continue

            color = _c2q(rec.color)
            run_color_t = rec.color  # keep as (R,G,B) tuple

            # Normalise station key (live uses "dist_m", GPKG uses "station_m")
            station = prof_data.get("station_m", prof_data.get("dist_m", np.empty(0)))
            if station.size == 0:
                continue

            if var_key == "wse_bed":
                wse = prof_data.get("wse_m", np.full_like(station, np.nan))
                bed = prof_data.get("bed_m", np.full_like(station, np.nan))
                depth = prof_data.get("depth_m", np.full_like(station, np.nan))
                wet_arr = prof_data.get("wet", np.ones_like(station))

                ok = np.isfinite(wse) & np.isfinite(bed)
                if not np.any(ok):
                    continue

                x_ok = station[ok]
                wse_ok = wse[ok]
                bed_ok = bed[ok]
                depth_ok = depth[ok]
                wet_ok = wet_arr[ok]
                wet_mask = np.where(
                    np.isfinite(wet_ok), wet_ok > 0.5, depth_ok > 1e-9
                )
                wse_phys = np.maximum(wse_ok, bed_ok)

                # "raw" render mode (always — no wet-mask clipping like matplotlib had)
                fill_mask = np.isfinite(wse_ok) & np.isfinite(bed_ok)
                wse_fill = wse_ok
                wse_plot_vals = wse_ok

                # ── Bed fill (below bed) ──
                if not bed_drawn and x_ok.size:
                    bed_min = float(np.min(bed_ok)) - 0.05 * max(
                        float(np.ptp(bed_ok)), 0.1
                    )
                    # Draw bed as a filled polygon to bed_min
                    bed_fill_x = np.concatenate([x_ok, x_ok[::-1]])
                    bed_fill_y = np.concatenate(
                        [np.full_like(bed_ok, bed_min), bed_ok[::-1]]
                    )
                    bed_fill_item = pg.PlotDataItem(
                        bed_fill_x, bed_fill_y,
                        fillLevel=bed_min,
                        brush=pg.mkBrush(QtGui.QColor(139, 115, 85, 128)),
                        pen=None,
                    )
                    self._plot_widget.addItem(bed_fill_item)
                    self._plot_items.append(bed_fill_item)

                    # Bed line
                    bed_line = pg.PlotDataItem(
                        x_ok, bed_ok,
                        pen=pg.mkPen(color=QtGui.QColor(92, 64, 51), width=0.9),
                    )
                    self._plot_widget.addItem(bed_line)
                    self._plot_items.append(bed_line)
                    bed_drawn = True

                # ── WSE-bed fill ──
                if use_fill_cmap:
                    # Collect fill metric for colormap shading
                    fill_metric = np.asarray(
                        prof_data.get(fill_key, np.full_like(station, np.nan)),
                        dtype=np.float64,
                    )
                    fill_ok = fill_metric[ok]
                    for i in range(len(x_ok) - 1):
                        if not (fill_mask[i] and fill_mask[i + 1]):
                            continue
                        if not (np.isfinite(fill_ok[i]) and np.isfinite(fill_ok[i + 1])):
                            continue
                        vmid = 0.5 * (float(fill_ok[i]) + float(fill_ok[i + 1]))
                        fill_segments_data.append(
                            (float(x_ok[i]), float(x_ok[i + 1]),
                             float(bed_ok[i]), float(wse_fill[i]), vmid)
                        )
                else:
                    # Single-color fill between bed and WSE
                    fill_curve_bed = pg.PlotDataItem(x_ok, bed_ok)
                    fill_curve_wse = pg.PlotDataItem(x_ok, wse_fill)
                    fill_item = pg.FillBetweenItem(
                        curve1=fill_curve_bed,
                        curve2=fill_curve_wse,
                        brush=pg.mkBrush(
                            run_color_t[0], run_color_t[1], run_color_t[2], 46
                        ),
                    )
                    self._plot_widget.addItem(fill_item)
                    self._fill_items.append(fill_item)

                # WSE line
                wse_plot_vals_plot = np.where(fill_mask, wse_plot_vals, np.nan)
                wse_line = pg.PlotDataItem(
                    x_ok, wse_plot_vals_plot,
                    pen=pg.mkPen(color=color, width=1.5),
                    name=f"{rec.display_label()} WSE",
                )
                self._plot_widget.addItem(wse_line)
                self._plot_items.append(wse_line)
                plotted += 1

            else:
                # Non-wse_bed modes: depth_m, velocity_ms, egl_m
                if var_key == "egl_m":
                    wse_arr = prof_data.get("wse_m")
                    vel_arr = prof_data.get("velocity_ms")
                    if wse_arr is None or vel_arr is None:
                        continue
                    y = np.asarray(wse_arr, dtype=np.float64) + (
                        np.asarray(vel_arr, dtype=np.float64) ** 2.0
                    ) / (2.0 * _u.gravity())
                else:
                    if var_key not in prof_data:
                        continue
                    y = np.asarray(prof_data[var_key], dtype=np.float64)
                ok = np.isfinite(station) & np.isfinite(y)
                if not np.any(ok):
                    continue
                line_item = pg.PlotDataItem(
                    station[ok], y[ok],
                    pen=pg.mkPen(color=color, width=1.5),
                    name=rec.display_label(),
                )
                self._plot_widget.addItem(line_item)
                self._plot_items.append(line_item)
                plotted += 1

            # ── Structure annotations ──
            if show_structures:
                try:
                    rows = load_structure_flows_at_time(
                        rec.gpkg_path, rec.run_id, t, t_tol=1.0
                    )
                    if rows:
                        placed_ids = {
                            str(r.get("object_id", "")) for r in structure_rows
                        }
                        for rr in rows:
                            sid = str(rr.get("object_id", ""))
                            if sid in placed_ids:
                                continue
                            structure_rows.append({
                                "run_label": rec.display_label(),
                                "object_id": sid,
                                "flow": float(rr.get("value", 0.0)),
                                "station": float("nan"),
                                "elev": float("nan"),
                                "placement": "unplaced",
                            })
                except Exception:
                    pass

        # ── Colormap fill segments ──
        if use_fill_cmap and fill_segments_data:
            vals_arr = np.asarray([f[4] for f in fill_segments_data], dtype=np.float64)
            finite = np.isfinite(vals_arr)
            if np.any(finite):
                vmin = float(np.nanmin(vals_arr[finite]))
                vmax = float(np.nanmax(vals_arr[finite]))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                for seg in fill_segments_data:
                    x0_s, x1_s, bed_s, wse_s, vmid = seg
                    t_norm = (vmid - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    seg_color = _cmap_brush(cmap_name, float(np.clip(t_norm, 0.0, 1.0)))
                    seg_bed = pg.PlotDataItem([x0_s, x1_s], [bed_s, bed_s])
                    seg_wse = pg.PlotDataItem([x0_s, x1_s], [wse_s, wse_s])
                    seg_fill = pg.FillBetweenItem(
                        curve1=seg_bed, curve2=seg_wse,
                        brush=pg.mkBrush(seg_color),
                    )
                    self._plot_widget.addItem(seg_fill)
                    self._fill_items.append(seg_fill)

        # ── Structure annotations on plot ──
        if plotted and show_structures and structure_rows:
            # Get current view range
            view_range = self._plot_widget.viewRange()
            x0_v, x1_v = view_range[0]
            y0_v, y1_v = view_range[1]
            y_span = max(y1_v - y0_v, 1.0e-6)
            for i, row in enumerate(structure_rows):
                xs = float(row.get("station", float("nan")))
                q_val = float(row.get("flow", 0.0))
                sid = str(row.get("object_id", ""))
                if not np.isfinite(xs):
                    continue
                vline = pg.InfiniteLine(
                    pos=xs, angle=90,
                    pen=pg.mkPen(color=(89, 89, 89), width=0.9,
                                 style=QtCore.Qt.PenStyle.DotLine),
                )
                vline.setZValue(2)
                self._plot_widget.addItem(vline)
                self._structure_items.append(vline)
                y_text = y1_v - 0.02 * y_span - 0.035 * y_span * (i % 3)
                label = pg.TextItem(
                    f"{sid} {q_val:.2f}",
                    anchor=(0.5, 1.0),
                    color=(89, 89, 89),
                )
                label.setPos(xs, y_text)
                label.setZValue(6)
                self._plot_widget.addItem(label)
                self._structure_labels.append(label)

        # ── Time title ──
        t_hr = t_sec / 3600.0
        # pyqtgraph doesn't have set_title on PlotWidget, use a label
        title_item = pg.LabelItem(
            f"t = {t_hr:.3f} {_TIME_UNIT}",
            color=(0, 0, 0),
        )
        # Place at top-left of the plot area

        if not plotted:
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)

        self._plot_widget.plotItem.autoRange()

        if self._table_widget is not None and self._table_widget.isVisible():
            self._populate_table()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_line_changed(self) -> None:
        """Line selector changed — update line ID and refresh."""
        if self._line_combo is None:
            return
        lid = self._line_combo.currentData()
        if lid is not None:
            self._line_id = int(lid)
            # Sync to result_data so external code can read it
            data = self._result_data
            if data is not None and hasattr(data, "set_line_id"):
                data.set_line_id(self._line_id)
        self.refresh()

    def _on_var_changed(self) -> None:
        """Variable combo changed — update var key and refresh."""
        if self._var_combo is None:
            return
        self._prof_var_key = str(self._var_combo.currentData() or "wse_bed")
        self.refresh()

    def _on_fill_changed(self) -> None:
        """Fill combo changed — update fill key and refresh."""
        if self._fill_combo is None:
            return
        self._prof_fill_key = str(self._fill_combo.currentData() or "none")
        self.refresh()

    def _on_cmap_changed(self) -> None:
        """Colormap combo changed — update cmap and refresh."""
        if self._cmap_combo is None:
            return
        self._prof_cmap = str(self._cmap_combo.currentData() or "viridis")
        self.refresh()

    def _on_show_struct_changed(self, checked: bool) -> None:
        """Structure visibility toggled — refresh."""
        self._prof_show_structures = bool(checked)
        self.refresh()

    def _on_table_toggle(self, visible: bool) -> None:
        """Show/hide the data table."""
        if self._table_widget is not None:
            self._table_widget.setVisible(visible)
            if visible:
                self._populate_table()

    def _on_mouse_moved(self, evt) -> None:
        """Handle mouse hover — update crosshair and data readout."""
        if self._plot_widget is None or not self._plot_items:
            return
        pos = evt[0]
        plot = self._plot_widget.plotItem
        vb = plot.vb
        if vb is None:
            return
        mouse_point = vb.mapSceneToView(pos)
        mx, my = mouse_point.x(), mouse_point.y()

        self._hover_vline.setPos(mx)
        self._hover_hline.setPos(my)
        self._hover_vline.setVisible(True)
        self._hover_hline.setVisible(True)

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
            self._hover_label.setPos(mx, my)
        else:
            self._hover_label.setVisible(False)

    # ------------------------------------------------------------------
    # Combo population
    # ------------------------------------------------------------------

    def _populate_line_combo(self) -> None:
        """Populate the line combo from result data line IDs."""
        if self._line_combo is None or self._result_data is None:
            return
        line_ids = self._result_data.get_line_ids()
        current = self._line_combo.currentData()
        self._line_combo.blockSignals(True)
        self._line_combo.clear()
        for lid in line_ids:
            self._line_combo.addItem(f"Line {lid}", lid)
        if current is not None:
            idx = self._line_combo.findData(current)
            if idx >= 0:
                self._line_combo.setCurrentIndex(idx)
        # Set initial line_id from first item if nothing selected
        if self._line_id < 0 and self._line_combo.count() > 0:
            self._line_id = int(self._line_combo.itemData(0))
            data = self._result_data
            if data is not None and hasattr(data, "set_line_id"):
                data.set_line_id(self._line_id)
        self._line_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Data table
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        """Fill the data table from profile data."""
        if self._table_widget is None or self._result_data is None:
            return
        data = self._result_data
        self._table_widget.setRowCount(0)
        self._table_widget.setColumnCount(0)

        line_id = self._line_id
        t_sec = float(getattr(data, "current_time_sec", 0.0))
        run_records = data.get_enabled_run_records()

        from swe2d.results.queries import (
            find_nearest_timestep,
            load_profile,
            load_profile_from_live,
        )
        is_live = getattr(data, "data_source", "") == "live"

        records: List[Dict[str, Any]] = []
        for rec in run_records:
            if line_id < 0:
                continue
            t = find_nearest_timestep(rec.gpkg_path, rec.run_id, line_id, t_sec)
            prof_data = (
                load_profile_from_live(data, str(rec.run_id), int(line_id), float(t))
                if is_live else
                load_profile(rec.gpkg_path, rec.run_id, line_id, t)
            )
            if not prof_data:
                continue
            station = prof_data.get("station_m", prof_data.get("dist_m", np.empty(0)))
            n = int(station.size)
            for i in range(n):
                row: Dict[str, Any] = {"run": rec.display_label()}
                for k, v in prof_data.items():
                    if isinstance(v, np.ndarray) and i < v.size:
                        row[k] = float(v[i])
                records.append(row)

        if not records:
            return

        cols = ["run"] + [k for k in records[0].keys() if k != "run"]
        self._table_widget.setColumnCount(len(cols))
        self._table_widget.setHorizontalHeaderLabels(cols)
        n = min(len(records), 5000)
        self._table_widget.setRowCount(n)
        for i, r in enumerate(records[:n]):
            for j, c in enumerate(cols):
                val = r.get(c, "")
                self._table_widget.setItem(
                    i, j, QtWidgets.QTableWidgetItem(
                        "" if val is None else f"{val:.4g}" if isinstance(val, float) else str(val)
                    )
                )

    # ------------------------------------------------------------------
    # Save / Export
    # ------------------------------------------------------------------

    def _save_plot_png(self) -> None:
        """Save as PNG."""
        if self._plot_widget is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot as PNG", "", "PNG Image (*.png)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter
            ImageExporter(self._plot_widget.plotItem).export(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    def _save_plot_svg(self) -> None:
        """Save as SVG."""
        if self._plot_widget is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot as SVG", "", "SVG Image (*.svg)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import SVGExporter
            SVGExporter(self._plot_widget.plotItem).export(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    def _save_plot_pdf(self) -> None:
        """Save as PDF / Print."""
        if self._plot_widget is None:
            return
        try:
            from pyqtgraph.exporters import PrintExporter
            PrintExporter(self._plot_widget.plotItem).export()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Error", str(exc))

    def _save_data_csv(self) -> None:
        """Save profile data as CSV."""
        if not self._plot_items:
            QtWidgets.QMessageBox.information(self, "No Data", "No plot data to export.")
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
                names = [item.name() or f"Series_{i}" for i, item in enumerate(self._plot_items)]
                writer.writerow(["Station (m)"] + names)
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
