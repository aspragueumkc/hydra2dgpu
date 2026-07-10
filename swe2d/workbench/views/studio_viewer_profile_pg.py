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

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt.QtCore import Qt

logger = logging.getLogger(__name__)

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


# ── Pure-numpy colormap LUTs (no matplotlib dependency) ──────────────

def _build_lut(stops):
    """Build a 256×3 uint8 LUT from (pos, (R,G,B)) stops via linear interpolation."""
    x = np.asarray([float(s[0]) for s in stops], dtype=np.float64)
    r = np.asarray([float(s[1][0]) for s in stops], dtype=np.float64)
    g = np.asarray([float(s[1][1]) for s in stops], dtype=np.float64)
    b = np.asarray([float(s[1][2]) for s in stops], dtype=np.float64)
    xi = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.clip(np.interp(xi, x, r), 0.0, 255.0).astype(np.uint8)
    lut[:, 1] = np.clip(np.interp(xi, x, g), 0.0, 255.0).astype(np.uint8)
    lut[:, 2] = np.clip(np.interp(xi, x, b), 0.0, 255.0).astype(np.uint8)
    return lut


# Colormap name → (256,3) uint8 LUT
_color_luts = {
    "viridis": _build_lut([
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]),
    "turbo": _build_lut([
        (0.00, (48, 18, 59)),
        (0.20, (50, 100, 220)),
        (0.40, (41, 187, 236)),
        (0.60, (124, 234, 87)),
        (0.80, (250, 205, 32)),
        (1.00, (180, 4, 38)),
    ]),
    "plasma": _build_lut([
        (0.00, (13, 8, 135)),
        (0.25, (126, 3, 167)),
        (0.50, (203, 71, 119)),
        (0.75, (248, 149, 64)),
        (1.00, (240, 249, 33)),
    ]),
    "inferno": _build_lut([
        (0.00, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.50, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.00, (252, 255, 164)),
    ]),
    "coolwarm": _build_lut([
        (0.00, (59, 76, 192)),
        (0.25, (101, 143, 222)),
        (0.50, (220, 220, 220)),
        (0.75, (222, 143, 101)),
        (1.00, (192, 76, 59)),
    ]),
}


def _cmap_color(cmap_name: str, t: float) -> Tuple[int, int, int]:
    """Return (R, G, B) for normalized position *t* in *cmap_name*.

    Pure numpy — no matplotlib dependency.  Falls back to grayscale.
    """
    lut = _color_luts.get(str(cmap_name).strip().lower())
    if lut is None:
        g = int(round(np.clip(t, 0.0, 1.0) * 255))
        return (g, g, g)
    idx = int(np.clip(round(t * 255.0), 0, 255))
    return (int(lut[idx, 0]), int(lut[idx, 1]), int(lut[idx, 2]))


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
    ("Flow", "flow_qn"),
]

_CMAP_ITEMS = [
    ("Viridis", "viridis"),
    ("Turbo", "turbo"),
    ("Plasma", "plasma"),
    ("Inferno", "inferno"),
    ("Coolwarm", "coolwarm"),
]


_ELEMENT_TYPES = [
    ("Line", "line"),
    ("Structure", "structure"),
    ("Drainage Node", "drainage_node"),
    ("Drainage Link", "drainage_link"),
]


class PGProfileWidget(QtWidgets.QWidget):
    """pyqtgraph-based plot widget supporting both profile and time-series views.

    Profile mode (element type = Line): cross-section station vs elevation.
    Time-series mode (element type = Structure/Drainage): coupling results.

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
        self._selected_element_id: str = ""
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
        self._vline: Optional[pg.InfiniteLine] = None

        # Combos
        self._etype_combo: Optional[QtWidgets.QComboBox] = None
        self._element_id_combo: Optional[QtWidgets.QComboBox] = None
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
        """Build: element type, element ID, var/fill/cmap combos, plot, table."""
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

        # ── Top bar (row 1: selectors, row 2: toggles) ──
        top_bar = QtWidgets.QVBoxLayout()
        top_bar.setSpacing(2)

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)

        def _make_combo(max_w: int = 100) -> QtWidgets.QComboBox:
            c = QtWidgets.QComboBox()
            c.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
            c.setMinimumWidth(60)
            c.setMaximumWidth(max_w)
            return c

        # Element type selector
        row1.addWidget(QtWidgets.QLabel("Type:"))
        self._etype_combo = _make_combo(120)
        self._etype_combo.setToolTip("Element type for profile data: Line, Structure, or Drainage.")
        for label, key in _ELEMENT_TYPES:
            self._etype_combo.addItem(label, key)
        self._etype_combo.currentIndexChanged.connect(self._on_etype_changed)
        row1.addWidget(self._etype_combo)
        row1.addSpacing(4)

        # Element ID selector (lines or coupling objects)
        row1.addWidget(QtWidgets.QLabel("Elem:"))
        self._element_id_combo = _make_combo(140)
        self._element_id_combo.setToolTip("Select the specific element ID to profile.")
        self._element_id_combo.currentIndexChanged.connect(self._on_element_id_changed)
        row1.addWidget(self._element_id_combo)
        row1.addSpacing(4)

        # Variable / Metric selector
        row1.addWidget(QtWidgets.QLabel("Var:"))
        self._var_combo = _make_combo(120)
        self._var_combo.setToolTip("Profile variable: WSE+Bed, Depth, Velocity, or EGL Error.")
        self._var_combo.currentIndexChanged.connect(self._on_var_changed)
        row1.addWidget(self._var_combo)
        row1.addStretch(1)
        top_bar.addLayout(row1)

        # ── Row 2: toggles ──
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)

        # Show structures toggle
        self._show_struct_chk = QtWidgets.QCheckBox("Struct")
        self._show_struct_chk.setChecked(True)
        self._show_struct_chk.setToolTip("Show structure annotations (flow labels) on the profile.")
        self._show_struct_chk.toggled.connect(self._on_show_struct_changed)
        row2.addWidget(self._show_struct_chk)

        # Data table toggle
        self.show_table_toggle = QtWidgets.QCheckBox("Table")
        self.show_table_toggle.setChecked(False)
        self.show_table_toggle.setToolTip("Show/hide the profile data table below the plot.")
        self.show_table_toggle.toggled.connect(self._on_table_toggle)
        row2.addWidget(self.show_table_toggle)
        row2.addSpacing(8)

        # Fill selector (profile only)
        row2.addWidget(QtWidgets.QLabel("Fill:"))
        self._fill_combo = _make_combo(100)
        self._fill_combo.setToolTip("Variable for color-filled profile shading: Depth, Velocity, or Flow.")
        for label, key in _FILL_ITEMS:
            self._fill_combo.addItem(label, key)
        self._fill_combo.currentIndexChanged.connect(self._on_fill_changed)
        row2.addWidget(self._fill_combo)
        row2.addSpacing(4)

        # Colormap selector (profile only)
        row2.addWidget(QtWidgets.QLabel("Cmap:"))
        self._cmap_combo = _make_combo(100)
        self._cmap_combo.setToolTip("Colormap used for profile fill shading.")
        for label, key in _CMAP_ITEMS:
            self._cmap_combo.addItem(label, key)
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        row2.addWidget(self._cmap_combo)
        row2.addSpacing(8)

        row2.addStretch(1)

        # Save button
        save_btn = QtWidgets.QPushButton("💾")
        save_btn.setFixedSize(24, 24)
        save_btn.setToolTip("Save plot / data")
        save_menu = QtWidgets.QMenu(save_btn)
        save_menu.addAction("Save plot as PNG", self._save_plot_png)
        save_menu.addAction("Save plot as SVG", self._save_plot_svg)
        save_menu.addAction("Save plot as PDF / Print", self._save_plot_pdf)
        save_menu.addSeparator()
        save_menu.addAction("Save data as CSV", self._save_data_csv)
        save_btn.setMenu(save_menu)
        row2.addWidget(save_btn)

        top_bar.addLayout(row2)
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
        """Return the plot mode (e.g., 'Profile')."""
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
        """Return the currently selected profile variable (e.g., 'wse_bed', 'depth_m')."""
        return self._prof_var_key

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        """Set the selected profile variable and update the UI combo."""
        self._prof_var_key = str(metric) if metric else "wse_bed"
        if self._var_combo is not None:
            idx = self._var_combo.findData(self._prof_var_key)
            if idx >= 0:
                self._var_combo.setCurrentIndex(idx)

    @property
    def selected_element_id(self) -> str:
        """Return the currently selected element ID (line ID, etc.)."""
        return self._selected_element_id

    @selected_element_id.setter
    def selected_element_id(self, element_id: str) -> None:
        """Set the selected element ID and update the UI combo."""
        self._selected_element_id = str(element_id) if element_id else ""
        if self._element_id_combo is not None and element_id:
            idx = self._element_id_combo.findData(self._selected_element_id)
            if idx >= 0:
                self._element_id_combo.setCurrentIndex(idx)

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
            self._populate_etype()
            # Pre-load coupling records
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
        """Re-render: profile for lines, time-series for coupling types."""
        if not _HAVE_PG or self._result_data is None or self._plot_widget is None:
            return

        data = self._result_data
        etype = str(self._etype_combo.currentData() or "line")
        t_sec = float(getattr(data, "current_time_sec", 0.0))
        run_records = data.get_enabled_run_records()
        var_key = self._prof_var_key

        from swe2d.services.gpkg_persistence_service import load_baked_line_profile, load_baked_line_timeseries
        from swe2d import units as _u

        # Clear the plot
        self._plot_widget.clear()
        self._plot_widget.addLegend()
        self._plot_items = []
        self._fill_items = []
        self._structure_items = []
        self._structure_labels = []
        self._vline = None
        self._hover_vline.setVisible(False)
        self._hover_hline.setVisible(False)
        self._hover_label.setVisible(False)
        self._plot_widget.addItem(self._hover_label)
        self._plot_widget.addItem(self._hover_vline)
        self._plot_widget.addItem(self._hover_hline)

        if not run_records:
            text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(text)
            self._plot_widget.plotItem.autoRange()
            return

        plotted = 0

        if etype == "line":
            # ── Profile rendering (station vs elevation) ──
            line_id = self._line_id
            fill_key = self._prof_fill_key
            cmap_name = self._prof_cmap
            show_structures = self._prof_show_structures
            use_fill_cmap = fill_key != "none"

            from swe2d.results.queries import (
                find_nearest_timestep, load_profile,
                load_structure_flows_at_time,
            )
            from swe2d import units as _u

            lu = getattr(data, "_length_unit", "")
            len_label = _unit_labels(lu)["len"]
            self._plot_widget.setLabel("bottom", f"Station ({len_label})")
            if var_key == "wse_bed":
                self._plot_widget.setLabel("left", f"Elevation ({len_label})")
            else:
                self._plot_widget.setLabel("left", _label_for_var(var_key, lu))

            if line_id < 0:
                text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
                self._plot_widget.addItem(text)
                self._plot_widget.plotItem.autoRange()
                return

            bed_drawn = False
            structure_rows: List[Dict[str, Any]] = []
            for rec in run_records:
                # Try live data first (for during-run viewing), then fall
                # back to GPKG (after run finalization).  The OLD matplotlib
                # code always used rec.gpkg_path — the live ternary added
                # during the pyqtgraph refactor had no fallback, so profiles
                # silently disappeared when _live_line_profile was empty.
                prof_data = {}
                if getattr(data, "_live_times", None) is not None and data._live_times.size > 0:
                    prof_data = load_baked_line_profile(
                        data, str(rec.run_id), int(line_id), float(t_sec),
                    )
                if not prof_data:
                    prof_data = load_baked_line_profile(
                        rec.gpkg_path, str(rec.run_id), int(line_id), float(t_sec),
                    )
                if not prof_data:
                    continue
                color = _c2q(rec.color)
                run_color_t = rec.color
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
                    wet_mask = np.where(np.isfinite(wet_ok), wet_ok > 0.5, depth_ok > 1e-9)
                    wse_phys = np.maximum(wse_ok, bed_ok)
                    fill_mask = np.isfinite(wse_ok) & np.isfinite(bed_ok)
                    wse_fill = wse_ok
                    wse_plot_vals = wse_ok

                    if not bed_drawn and x_ok.size:
                        bed_min = float(np.min(bed_ok)) - 0.05 * max(float(np.ptp(bed_ok)), 0.1)
                        bed_fill_x = np.concatenate([x_ok, x_ok[::-1]])
                        bed_fill_y = np.concatenate([np.full_like(bed_ok, bed_min), bed_ok[::-1]])
                        bed_fill_item = pg.PlotDataItem(bed_fill_x, bed_fill_y, fillLevel=bed_min,
                            brush=pg.mkBrush(QtGui.QColor(139, 115, 85, 128)), pen=None)
                        self._plot_widget.addItem(bed_fill_item)
                        self._plot_items.append(bed_fill_item)
                        bed_line = pg.PlotDataItem(x_ok, bed_ok, pen=pg.mkPen(color=QtGui.QColor(92, 64, 51), width=0.9))
                        self._plot_widget.addItem(bed_line)
                        self._plot_items.append(bed_line)
                        bed_drawn = True

                    fill_curve_bed = pg.PlotDataItem(x_ok, bed_ok)
                    fill_curve_wse = pg.PlotDataItem(x_ok, wse_fill)
                    fill_item = pg.FillBetweenItem(curve1=fill_curve_bed, curve2=fill_curve_wse,
                        brush=pg.mkBrush(run_color_t[0], run_color_t[1], run_color_t[2], 46))
                    self._plot_widget.addItem(fill_item)
                    self._fill_items.append(fill_item)

                    if use_fill_cmap:
                        fill_metric = np.asarray(prof_data.get(fill_key, np.full_like(station, np.nan)), dtype=np.float64)
                        fill_ok = fill_metric[ok]
                        seg_vals, seg_list = [], []
                        for i in range(len(x_ok) - 1):
                            if not (fill_mask[i] and fill_mask[i + 1]) or not (np.isfinite(fill_ok[i]) and np.isfinite(fill_ok[i + 1])):
                                continue
                            vmid = 0.5 * (float(fill_ok[i]) + float(fill_ok[i + 1]))
                            seg_list.append(i)
                            seg_vals.append(vmid)
                        if seg_vals:
                            sv = np.asarray(seg_vals, dtype=np.float64)
                            sv_min, sv_max = float(np.nanmin(sv)), float(np.nanmax(sv))
                            if sv_max <= sv_min:
                                sv_max = sv_min + 1.0
                            for idx, i in enumerate(seg_list):
                                vmid = seg_vals[idx]
                                t_norm = (vmid - sv_min) / (sv_max - sv_min)
                                rgb = _cmap_color(cmap_name, float(np.clip(t_norm, 0.0, 1.0)))
                                seg_bed = pg.PlotDataItem([float(x_ok[i]), float(x_ok[i + 1])], [float(bed_ok[i]), float(bed_ok[i + 1])])
                                seg_wse = pg.PlotDataItem([float(x_ok[i]), float(x_ok[i + 1])], [float(wse_fill[i]), float(wse_fill[i + 1])])
                                seg_fill = pg.FillBetweenItem(curve1=seg_bed, curve2=seg_wse, brush=pg.mkBrush(QtGui.QColor(*rgb)))
                                self._plot_widget.addItem(seg_fill)
                                self._fill_items.append(seg_fill)

                    wse_plot_vals_plot = np.where(fill_mask, wse_plot_vals, np.nan)
                    wse_line = pg.PlotDataItem(x_ok, wse_plot_vals_plot, pen=pg.mkPen(color=color, width=1.5), name=f"{rec.display_label()} WSE")
                    self._plot_widget.addItem(wse_line)
                    self._plot_items.append(wse_line)
                    plotted += 1
                else:
                    if var_key == "egl_m":
                        wse_arr = prof_data.get("wse_m")
                        vel_arr = prof_data.get("velocity_ms")
                        if wse_arr is None or vel_arr is None:
                            continue
                        y = np.asarray(wse_arr, dtype=np.float64) + (np.asarray(vel_arr, dtype=np.float64) ** 2.0) / (2.0 * _u.gravity())
                    else:
                        if var_key not in prof_data:
                            continue
                        y = np.asarray(prof_data[var_key], dtype=np.float64)
                    ok = np.isfinite(station) & np.isfinite(y)
                    if not np.any(ok):
                        continue
                    line_item = pg.PlotDataItem(station[ok], y[ok], pen=pg.mkPen(color=color, width=1.5), name=rec.display_label())
                    self._plot_widget.addItem(line_item)
                    self._plot_items.append(line_item)
                    plotted += 1

                if show_structures:
                    try:
                        # Use the live data layer for runs without a persisted
                        # GPKG yet (live runs carry an empty rec.gpkg_path).
                        flow_source = rec.gpkg_path or data
                        rows = load_structure_flows_at_time(flow_source, rec.run_id, t_sec)
                        if rows:
                            placed_ids = {str(r.get("object_id", "")) for r in structure_rows}
                            for rr in rows:
                                sid = str(rr.get("object_id", ""))
                                if sid in placed_ids:
                                    continue
                                structure_rows.append({"run_label": rec.display_label(), "object_id": sid,
                                    "flow": float(rr.get("value", 0.0)), "station": float("nan"),
                                    "elev": float("nan"), "placement": "unplaced"})
                    except Exception:
                        logger.warning("Failed to load structure flows for profile annotations", exc_info=True)

            if plotted and show_structures and structure_rows:
                view_range = self._plot_widget.viewRange()
                x0_v, x1_v = view_range[0]
                y0_v, y1_v = view_range[1]
                y_span = max(y1_v - y0_v, 1.0e-6)
                x_span = max(x1_v - x0_v, 1.0e-6)
                unplaced_count = 0
                for i, row in enumerate(structure_rows):
                    xs = float(row.get("station", float("nan")))
                    q_val = float(row.get("flow", 0.0))
                    sid = str(row.get("object_id", ""))
                    if np.isfinite(xs):
                        vline = pg.InfiniteLine(pos=xs, angle=90, pen=pg.mkPen(color=(89, 89, 89), width=0.9, style=QtCore.Qt.PenStyle.DotLine))
                        vline.setZValue(2)
                        self._plot_widget.addItem(vline)
                        self._structure_items.append(vline)
                        y_text = y1_v - 0.02 * y_span - 0.035 * y_span * (i % 3)
                        anchor = (0.5, 1.0)
                        label_x = xs
                    else:
                        # Unplaced structure: no geometry available (live runs or
                        # missing line/structure intersection).  Show as a text
                        # label along the top-right margin so it is still visible.
                        unplaced_count += 1
                        anchor = (1.0, 1.0)
                        label_x = x1_v - 0.02 * x_span
                        y_text = y1_v - 0.02 * y_span - 0.035 * y_span * ((unplaced_count - 1) % 3)
                    label = pg.TextItem(f"{sid} {q_val:.2f}", anchor=anchor, color=(89, 89, 89))
                    label.setPos(label_x, y_text)
                    label.setZValue(6)
                    self._plot_widget.addItem(label)
                    self._structure_labels.append(label)

            t_hr = t_sec / 3600.0
            if not plotted:
                text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
                self._plot_widget.addItem(text)
            self._plot_widget.plotItem.autoRange()

        else:
            # ── Time-series rendering for coupling types ──
            eid = self._selected_element_id
            if not eid:
                text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
                self._plot_widget.addItem(text)
                self._plot_widget.plotItem.autoRange()
                return

            lu = getattr(data, "_length_unit", "")
            self._plot_widget.setLabel("bottom", f"Time ({_TIME_UNIT})")
            self._plot_widget.setLabel("left", str(var_key))

            for rec in run_records:
                # Load coupling data for this run
                if data._coupling_run_id != str(rec.run_id):
                    data.load_coupling_records(str(rec.run_id))
                coupling = data.get_coupling_records()
                if not coupling:
                    continue
                filtered = [
                    r for r in coupling
                    if str(r.get("component", "") or "") == etype
                    and str(r.get("object_id", "") or "") == eid
                    and str(r.get("metric", "") or "") == var_key
                ]
                if not filtered:
                    continue
                filtered.sort(key=lambda r: float(r.get("t_s", 0.0)))
                t_vals = np.array([float(r["t_s"]) for r in filtered], dtype=np.float64)
                v_vals = np.array([float(r["value"]) for r in filtered], dtype=np.float64)
                t_hr = t_vals / 3600.0
                color = _c2q(rec.color)
                pen = pg.mkPen(color=color, width=1.6)
                item = self._plot_widget.plot(t_hr, v_vals, pen=pen, name=rec.display_label())
                self._plot_items.append(item)
                plotted += 1

            t_hr_now = t_sec / 3600.0
            self._vline = pg.InfiniteLine(pos=t_hr_now, angle=90,
                pen=pg.mkPen(color=(128, 128, 128), width=0.9, style=QtCore.Qt.PenStyle.DashLine))
            self._vline.setZValue(50)
            self._plot_widget.addItem(self._vline)

            if not plotted:
                text = pg.TextItem("No data", anchor=(0.5, 0.5), color=(128, 128, 128))
                self._plot_widget.addItem(text)
            self._plot_widget.plotItem.autoRange()

        if self._table_widget is not None and self._table_widget.isVisible():
            self._populate_table()

        # ── Time title ──
        t_hr = t_sec / 3600.0
        self._plot_widget.plotItem.autoRange()
        view_range = self._plot_widget.viewRange()
        if view_range and len(view_range) == 2:
            x0_v, x1_v = view_range[0]
            y0_v, y1_v = view_range[1]
            title_text = pg.TextItem(
                f"t = {t_hr:.3f} {_TIME_UNIT}",
                anchor=(0.0, 1.0),
                color=(0, 0, 0),
            )
            title_text.setPos(x0_v, y1_v)
            title_text.setZValue(10)
            self._plot_widget.addItem(title_text)

        if self._table_widget is not None and self._table_widget.isVisible():
            self._populate_table()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_etype_changed(self) -> None:
        """Element type changed — repopulate element ID combo and var combo."""
        self._populate_element_id_combo()
        self._repopulate_var_combo()
        self._show_profile_controls()
        self.refresh()

    def _on_element_id_changed(self) -> None:
        """Element ID changed — update selected element ID and refresh."""
        self._selected_element_id = str(self._element_id_combo.currentData() or "")
        data = self._result_data
        etype = str(self._etype_combo.currentData() or "line")
        # Keep the internal line_id in sync so refresh() renders the right line.
        if etype == "line" and self._selected_element_id:
            try:
                self._line_id = int(self._selected_element_id)
            except (ValueError, TypeError):
                pass
        if data is not None and etype == "line" and self._selected_element_id:
            try:
                data.set_line_id(int(self._selected_element_id))
            except (ValueError, TypeError):
                pass
        self.refresh()

    def _on_var_changed(self) -> None:
        """Variable/metric combo changed — update var key and refresh."""
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

    def _populate_etype(self) -> None:
        """Called on set_data — populate element ID combo for current etype."""
        self._populate_element_id_combo()
        self._repopulate_var_combo()
        self._show_profile_controls()

    def _populate_element_id_combo(self) -> None:
        """Populate element ID combo based on selected element type."""
        if self._element_id_combo is None or self._result_data is None:
            return
        etype = str(self._etype_combo.currentData() or "line")
        prev_data = self._element_id_combo.currentData()
        self._element_id_combo.blockSignals(True)
        self._element_id_combo.clear()

        data = self._result_data
        if etype == "line":
            line_ids = data.get_line_ids()
            for lid in line_ids:
                self._element_id_combo.addItem(f"Line {lid}", lid)
        else:
            # Coupling-based types — collect elements from ALL enabled runs
            # so the combo shows every selectable element, not just the first.
            data = self._result_data
            seen = set()
            for rec in getattr(data, "_run_records", []):
                if not rec.enabled:
                    continue
                if getattr(data, "_coupling_run_id", "") != str(rec.run_id):
                    data.load_coupling_records(str(rec.run_id))
                for row in data.get_coupling_records():
                    if str(row.get("component", "") or "") != etype:
                        continue
                    oid = str(row.get("object_id", "") or "")
                    if not oid or oid in seen:
                        continue
                    seen.add(oid)
                    oname = str(row.get("object_name", "") or "")
                    lbl = f"{oname} ({oid})" if oname else oid
                    self._element_id_combo.addItem(lbl, oid)

        if prev_data is not None:
            idx = self._element_id_combo.findData(prev_data)
            if idx >= 0:
                self._element_id_combo.setCurrentIndex(idx)
        self._element_id_combo.blockSignals(False)
        self._selected_element_id = str(self._element_id_combo.currentData() or "")
        # Sync line_id for line mode
        if etype == "line" and self._selected_element_id:
            try:
                self._line_id = int(self._selected_element_id)
            except (ValueError, TypeError):
                pass

    def _repopulate_var_combo(self) -> None:
        """Populate var/metric combo based on element type."""
        if self._var_combo is None:
            return
        etype = str(self._etype_combo.currentData() or "line")
        prev_data = self._var_combo.currentData()
        self._var_combo.blockSignals(True)
        self._var_combo.clear()

        if etype == "line":
            for label, key in _PROFILE_VAR_ITEMS:
                self._var_combo.addItem(label, key)
        else:
            # Coupling types — collect unique metrics from coupling records
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
                self._var_combo.addItem(m, m)

        idx = self._var_combo.findData(prev_data)
        if idx >= 0:
            self._var_combo.setCurrentIndex(idx)
        self._var_combo.blockSignals(False)
        self._prof_var_key = str(self._var_combo.currentData() or "wse_bed")

    def _show_profile_controls(self) -> None:
        """Show/hide profile-specific controls based on element type."""
        etype = str(self._etype_combo.currentData() or "line")
        is_profile = etype == "line"
        if self._fill_combo is not None:
            self._fill_combo.setVisible(is_profile)
        if self._cmap_combo is not None:
            self._cmap_combo.setVisible(is_profile)
        if self._show_struct_chk is not None:
            self._show_struct_chk.setVisible(is_profile)
        # Update axis labels for time-series mode
        if self._plot_widget is not None:
            if is_profile:
                self._plot_widget.setLabel("bottom", "Station (m)")
                self._plot_widget.setLabel("left", "Elevation (m)")
            else:
                self._plot_widget.setLabel("bottom", f"Time ({_TIME_UNIT})")
                self._plot_widget.setLabel("left", "Value")

    # ------------------------------------------------------------------
    # Data table
    # ------------------------------------------------------------------

    def _populate_table(self) -> None:
        """Fill the data table from current plot data."""
        if self._table_widget is None or self._result_data is None:
            return
        data = self._result_data
        etype = str(self._etype_combo.currentData() or "line")
        self._table_widget.setRowCount(0)
        self._table_widget.setColumnCount(0)

        if etype == "line":
            line_id = self._line_id
            t_sec = float(getattr(data, "current_time_sec", 0.0))
            run_records = data.get_enabled_run_records()

            from swe2d.services.gpkg_persistence_service import load_baked_line_profile

            records: List[Dict[str, Any]] = []
            for rec in run_records:
                if line_id < 0:
                    continue
                prof_data = {}
                if getattr(data, "_live_times", None) is not None and data._live_times.size > 0:
                    prof_data = load_baked_line_profile(
                        data, str(rec.run_id), int(line_id), float(t_sec),
                    )
                if not prof_data:
                    prof_data = load_baked_line_profile(
                        rec.gpkg_path, str(rec.run_id), int(line_id), float(t_sec),
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
        else:
            # Coupling data table
            records = data.get_coupling_records()
            if not records:
                return
            cols = ["t_s", "component", "metric", "object_id", "object_name", "value"]
            self._table_widget.setColumnCount(len(cols))
            self._table_widget.setHorizontalHeaderLabels(cols)
            n = min(len(records), 5000)
            self._table_widget.setRowCount(n)
            for i, r in enumerate(records[:n]):
                for j, c in enumerate(cols):
                    val = r.get(c, "")
                    self._table_widget.setItem(
                        i, j, QtWidgets.QTableWidgetItem("" if val is None else f"{val:.6g}" if isinstance(val, float) else str(val))
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
                etype = str(self._etype_combo.currentData() or "line")
                xlbl = "Station (m)" if etype == "line" else f"Time ({_TIME_UNIT})"
                writer.writerow([xlbl] + names)
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
