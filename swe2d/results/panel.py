"""swe2d_results_panel.py  — Sprint 1 (full-featured multi-run dockable panel)

Dockable results panel for the SWE2D workbench.

Features
--------
- Multi-run sidebar with color-coded checkboxes (multi-select)
- Time-Series tab: overlaid plots per run, variable selector, vertical time marker
- Profile tab: WSE+bed per run at current timestep (full wet/dry rendering)
- Metadata tab: per-run summary table
- Bottom animation bar: QSlider + Play/Pause + Step +/- + Speed + time label
- QGIS project persistence: selected runs, timestep, tab, variable

The panel is *additive*: the existing modal ``SWE2DLineResultsViewerDialog`` is
unchanged and still accessible.
"""

from __future__ import annotations

import dataclasses
import json
import os as _os
from typing import Dict, List, Set, Tuple

import numpy as np
try:
    from qgis.PyQt import QtCore, QtGui, QtWidgets
    from qgis.PyQt.QtCore import Qt, pyqtSignal
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, pyqtSignal

try:
    from .animation import ResultsAnimationController
except Exception:
    from swe2d.results.animation import ResultsAnimationController

try:
    from swe2d.results.db_utils import open_ro as _open_ro_shared, table_exists as _table_exists_shared
except Exception:
    from .db_utils import open_ro as _open_ro_shared, table_exists as _table_exists_shared

# ---------------------------------------------------------------------------
# Optional QGIS imports
# ---------------------------------------------------------------------------
try:
    from qgis.gui import QgsDockWidget as _QgsDockWidgetBase
    _BASE_DOCK: type = _QgsDockWidgetBase
except ImportError:
    _BASE_DOCK = QtWidgets.QDockWidget  # type: ignore[assignment,misc]

try:
    from qgis.core import QgsProject as _QgsProject, QgsVectorLayer, QgsFeatureRequest
    _HAVE_QGSPROJECT = True
except ImportError:
    _QgsProject = None  # type: ignore[assignment]
    QgsVectorLayer = None  # type: ignore[assignment]
    QgsFeatureRequest = None  # type: ignore[assignment]
    _HAVE_QGSPROJECT = False

# ---------------------------------------------------------------------------
# Optional matplotlib
# ---------------------------------------------------------------------------
_FigureCanvas = None
_Figure = None
_NavigationToolbar = None


def _try_import_matplotlib() -> bool:
    global _FigureCanvas, _Figure, _NavigationToolbar
    if _FigureCanvas is not None:
        return True
    for _backend in (
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_qtagg",
    ):
        try:
            import importlib as _importlib
            _mod = _importlib.import_module(_backend)
            from matplotlib.figure import Figure as _Figure_cls
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
            _FigureCanvas = _mod.FigureCanvasQTAgg
            _Figure = _Figure_cls
            _NavigationToolbar = NavigationToolbar2QT
            return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PANEL_COLORS: List[Tuple[int, int, int]] = [
    (31,  119, 180),
    (255, 127,  14),
    (44,  160,  44),
    (214,  39,  40),
    (148, 103, 189),
    (140,  86,  75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189,  34),
    (23,  190, 207),
]

_TS_VARIABLES = [
    ("Flow",             "flow_cms"),
    ("Depth",            "depth_m"),
    ("Velocity",         "velocity_ms"),
    ("Water Surface",    "wse_m"),
    ("Bed Elevation",    "bed_m"),
]

_PROF_VARIABLES = [
    ("WSE + Bed",        "wse_bed"),
    ("EGL",              "egl_m"),
    ("Depth",            "depth_m"),
    ("Velocity",         "velocity_ms"),
    ("Froude",           "fr"),
    ("Normal Flow",      "flow_qn"),
]

_PROFILE_FILL_OPTIONS = [
    ("None", "none"),
    ("Depth", "depth_m"),
    ("Velocity", "velocity_ms"),
    ("Froude", "fr"),
    ("Normal Flow", "flow_qn"),
]

_PROFILE_CMAP_OPTIONS = [
    ("Viridis", "viridis"),
    ("Plasma", "plasma"),
    ("Turbo", "turbo"),
    ("Inferno", "inferno"),
    ("Magma", "magma"),
    ("Cividis", "cividis"),
]

_PERSISTENCE_KEY = "swe2d_results_panel_state"
_PERSISTENCE_GROUP = "Backwater2DWorkbench"
_DEFAULT_FPS = 4.0


def _c2f(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _line_layer_uri(gpkg_path: str, layer_name: str) -> str:
    return f"{gpkg_path}|layername={layer_name}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunRecord:
    run_id: str
    gpkg_path: str
    color: Tuple[int, int, int]
    enabled: bool = True
    label: str = ""
    has_profile: bool = False

    def display_label(self) -> str:
        return self.label or self.run_id

    @property
    def key(self) -> str:
        return f"{self.gpkg_path}::{self.run_id}"


# ---------------------------------------------------------------------------
# Color-swatch list delegate
# ---------------------------------------------------------------------------

class _SwatchDelegate(QtWidgets.QStyledItemDelegate):
    _SW = 12
    _GAP = 3

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        rgb = index.data(Qt.UserRole + 1)
        if rgb is None:
            return
        r, g, b = rgb
        rect = option.rect
        sw, gap = self._SW, self._GAP
        sr = QtCore.QRect(
            rect.left() + gap,
            rect.top() + (rect.height() - sw) // 2,
            sw, sw,
        )
        painter.save()
        painter.setBrush(QtGui.QColor(r, g, b))
        painter.setPen(Qt.NoPen)
        painter.drawRect(sr)
        painter.restore()


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class SWE2DResultsPanel(_BASE_DOCK):  # type: ignore[valid-type,misc]
    """Full-featured dockable multi-run SWE2D results panel."""

    timestep_changed = pyqtSignal(float)
    velocity_overlay_changed = pyqtSignal()
    velocity_overlay_add_requested = pyqtSignal()

    def __init__(self, gpkg_path: str = "", iface=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SWE2D Results")
        self.setObjectName("SWE2DResultsPanel")
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable,
        )

        self._gpkg_path: str = str(gpkg_path or "")
        self._iface = iface

        self._run_records: List[RunRecord] = []
        self._manual_gpkg_paths: List[str] = []
        self._base_selected_run_keys: Set[str] = set()
        self._manual_selected_run_keys: Set[str] = set()
        self._line_id: int = -1
        self._current_t_sec: float = 0.0
        self._all_timesteps: np.ndarray = np.empty(0, dtype=np.float64)

        # Animation
        self._anim_fps: float = _DEFAULT_FPS
        self._anim_frame_idx: int = 0
        self._anim = ResultsAnimationController(self, fps=self._anim_fps)
        self._anim.current_timestep_changed.connect(self._on_controller_timestep_changed)
        self._anim.play_state_changed.connect(self._on_controller_play_state_changed)

        # matplotlib
        self._have_mpl: bool = False
        self._fig_ts = None
        self._ax_ts = None
        self._canvas_ts = None
        self._toolbar_ts = None
        self._fig_prof = None
        self._ax_prof = None
        self._canvas_prof = None
        self._toolbar_prof = None
        self._ts_vline = None
        self._prof_fill_cbar = None

        self._setup_ui()
        self._setup_matplotlib()

        if self._gpkg_path:
            self._discover_runs()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root_widget = QtWidgets.QWidget()
        self.setWidget(root_widget)
        root = QtWidgets.QVBoxLayout(root_widget)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)

        # Top bar
        top = QtWidgets.QHBoxLayout()
        self._gpkg_lbl = QtWidgets.QLabel()
        self._gpkg_lbl.setStyleSheet("color: gray; font-size: 9px;")
        self._gpkg_lbl.setMaximumWidth(320)
        refresh_btn = QtWidgets.QPushButton("\u21ba")
        refresh_btn.setFixedSize(22, 22)
        refresh_btn.setToolTip("Re-scan GPKG for new runs")
        refresh_btn.clicked.connect(self._discover_runs)
        add_btn = QtWidgets.QPushButton("+")
        add_btn.setFixedSize(22, 22)
        add_btn.setToolTip("Add results from one or more GeoPackages")
        add_btn.clicked.connect(self._add_results_files)
        top.addWidget(self._gpkg_lbl, 1)
        top.addWidget(add_btn)
        top.addWidget(refresh_btn)
        root.addLayout(top)

        # Body: sidebar + tabs
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(4)

        sidebar = QtWidgets.QVBoxLayout()
        sidebar.setSpacing(2)
        sidebar.addWidget(QtWidgets.QLabel("<b>Runs</b>"))
        self._run_list = QtWidgets.QListWidget()
        self._run_list.setFixedWidth(200)
        self._run_list.setAlternatingRowColors(True)
        self._run_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._run_list.setItemDelegate(_SwatchDelegate(self._run_list))
        self._run_list.itemChanged.connect(self._on_run_toggle)
        self._run_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._run_list.customContextMenuRequested.connect(self._show_run_list_context_menu)
        self._run_list.setToolTip(
            "Check/uncheck to toggle run visibility.\n"
            "Select row(s) then use Remove button or right-click to remove."
        )
        sidebar.addWidget(self._run_list)

        # Run list action buttons
        run_btn_row = QtWidgets.QHBoxLayout()
        run_btn_row.setSpacing(2)
        self._remove_runs_btn = QtWidgets.QPushButton("\u2212 Remove")
        self._remove_runs_btn.setFixedHeight(20)
        self._remove_runs_btn.setToolTip("Remove selected run(s) from the viewer")
        self._remove_runs_btn.clicked.connect(self._remove_selected_runs)
        _show_all_btn = QtWidgets.QPushButton("\u2713 All")
        _show_all_btn.setFixedHeight(20)
        _show_all_btn.setToolTip("Show all runs")
        _show_all_btn.clicked.connect(self._set_all_runs_visible)
        _hide_all_btn = QtWidgets.QPushButton("\u25a1 None")
        _hide_all_btn.setFixedHeight(20)
        _hide_all_btn.setToolTip("Hide all runs")
        _hide_all_btn.clicked.connect(self._set_all_runs_hidden)
        run_btn_row.addWidget(self._remove_runs_btn, 2)
        run_btn_row.addWidget(_show_all_btn, 1)
        run_btn_row.addWidget(_hide_all_btn, 1)
        sidebar.addLayout(run_btn_row)

        line_row = QtWidgets.QHBoxLayout()
        line_row.addWidget(QtWidgets.QLabel("Line:"))
        self._line_combo = QtWidgets.QComboBox()
        self._line_combo.setMinimumWidth(100)
        self._line_combo.currentIndexChanged.connect(self._on_line_changed)
        line_row.addWidget(self._line_combo, 1)
        sidebar.addLayout(line_row)

        var_row = QtWidgets.QHBoxLayout()
        var_row.addWidget(QtWidgets.QLabel("TS var:"))
        self._ts_var_combo = QtWidgets.QComboBox()
        for label, key in _TS_VARIABLES:
            self._ts_var_combo.addItem(label, key)
        self._ts_var_combo.currentIndexChanged.connect(self._on_ts_var_changed)
        var_row.addWidget(self._ts_var_combo, 1)
        sidebar.addLayout(var_row)

        pvar_row = QtWidgets.QHBoxLayout()
        pvar_row.addWidget(QtWidgets.QLabel("Prof:"))
        self._prof_var_combo = QtWidgets.QComboBox()
        for label, key in _PROF_VARIABLES:
            self._prof_var_combo.addItem(label, key)
        self._prof_var_combo.currentIndexChanged.connect(self._on_prof_var_changed)
        pvar_row.addWidget(self._prof_var_combo, 1)
        sidebar.addLayout(pvar_row)

        self._prof_fill_widget = QtWidgets.QWidget()
        prof_fill_row = QtWidgets.QHBoxLayout(self._prof_fill_widget)
        prof_fill_row.setContentsMargins(0, 0, 0, 0)
        self._prof_fill_lbl = QtWidgets.QLabel("Fill by:")
        self._prof_fill_combo = QtWidgets.QComboBox()
        for label, key in _PROFILE_FILL_OPTIONS:
            self._prof_fill_combo.addItem(label, key)
        self._prof_fill_combo.currentIndexChanged.connect(self._on_prof_fill_changed)
        prof_fill_row.addWidget(self._prof_fill_lbl)
        prof_fill_row.addWidget(self._prof_fill_combo, 1)
        sidebar.addWidget(self._prof_fill_widget)

        self._prof_wse_render_widget = QtWidgets.QWidget()
        prof_wse_row = QtWidgets.QHBoxLayout(self._prof_wse_render_widget)
        prof_wse_row.setContentsMargins(0, 0, 0, 0)
        self._prof_wse_render_lbl = QtWidgets.QLabel("WSE render:")
        self._prof_wse_render_combo = QtWidgets.QComboBox()
        self._prof_wse_render_combo.addItem("Clipped to bed (wet only)", "clipped")
        self._prof_wse_render_combo.addItem("Raw sampled", "raw")
        self._prof_wse_render_combo.currentIndexChanged.connect(self._on_prof_fill_changed)
        prof_wse_row.addWidget(self._prof_wse_render_lbl)
        prof_wse_row.addWidget(self._prof_wse_render_combo, 1)
        sidebar.addWidget(self._prof_wse_render_widget)

        self._prof_cmap_widget = QtWidgets.QWidget()
        prof_cmap_row = QtWidgets.QHBoxLayout(self._prof_cmap_widget)
        prof_cmap_row.setContentsMargins(0, 0, 0, 0)
        self._prof_cmap_lbl = QtWidgets.QLabel("Colormap:")
        self._prof_cmap_combo = QtWidgets.QComboBox()
        for label, key in _PROFILE_CMAP_OPTIONS:
            self._prof_cmap_combo.addItem(label, key)
        self._prof_cmap_combo.currentIndexChanged.connect(self._on_prof_fill_changed)
        prof_cmap_row.addWidget(self._prof_cmap_lbl)
        prof_cmap_row.addWidget(self._prof_cmap_combo, 1)
        sidebar.addWidget(self._prof_cmap_widget)

        self._show_structures_chk = QtWidgets.QCheckBox("Overlay structures")
        self._show_structures_chk.setChecked(True)
        self._show_structures_chk.toggled.connect(self._refresh_profile)
        sidebar.addWidget(self._show_structures_chk)

        self._show_velocity_chk = QtWidgets.QCheckBox("Velocity vectors")
        self._show_velocity_chk.setChecked(False)
        self._show_velocity_chk.toggled.connect(self._on_velocity_overlay_control_changed)
        sidebar.addWidget(self._show_velocity_chk)

        self._velocity_add_btn = QtWidgets.QPushButton("Add To Map Canvas...")
        self._velocity_add_btn.setToolTip("Manually select a GeoPackage layer to source velocity arrows from")
        self._velocity_add_btn.clicked.connect(self._on_velocity_overlay_add_requested)
        sidebar.addWidget(self._velocity_add_btn)

        vel_density_row = QtWidgets.QHBoxLayout()
        vel_density_row.addWidget(QtWidgets.QLabel("Density:"))
        self._vel_density_combo = QtWidgets.QComboBox()
        self._vel_density_combo.addItem("Full", 1)
        self._vel_density_combo.addItem("1/2", 2)
        self._vel_density_combo.addItem("1/4", 4)
        self._vel_density_combo.addItem("1/8", 8)
        self._vel_density_combo.setCurrentIndex(2)
        self._vel_density_combo.currentIndexChanged.connect(self._on_velocity_overlay_control_changed)
        vel_density_row.addWidget(self._vel_density_combo, 1)
        sidebar.addLayout(vel_density_row)

        vel_min_row = QtWidgets.QHBoxLayout()
        vel_min_row.addWidget(QtWidgets.QLabel("Min speed:"))
        self._vel_min_speed_spin = QtWidgets.QDoubleSpinBox()
        self._vel_min_speed_spin.setDecimals(3)
        self._vel_min_speed_spin.setRange(0.0, 100.0)
        self._vel_min_speed_spin.setValue(0.05)
        self._vel_min_speed_spin.setSingleStep(0.05)
        self._vel_min_speed_spin.setSuffix(" m/s")
        self._vel_min_speed_spin.valueChanged.connect(self._on_velocity_overlay_control_changed)
        vel_min_row.addWidget(self._vel_min_speed_spin, 1)
        sidebar.addLayout(vel_min_row)

        self._show_streamlines_chk = QtWidgets.QCheckBox("Streamline traces")
        self._show_streamlines_chk.setChecked(False)
        self._show_streamlines_chk.setToolTip("Render map-canvas streamline traces from the active velocity field")
        self._show_streamlines_chk.toggled.connect(self._on_velocity_overlay_control_changed)
        sidebar.addWidget(self._show_streamlines_chk)

        sl_seed_row = QtWidgets.QHBoxLayout()
        sl_seed_row.addWidget(QtWidgets.QLabel("Traces:"))
        self._streamline_seed_combo = QtWidgets.QComboBox()
        self._streamline_seed_combo.addItem("24", 24)
        self._streamline_seed_combo.addItem("48", 48)
        self._streamline_seed_combo.addItem("96", 96)
        self._streamline_seed_combo.addItem("160", 160)
        self._streamline_seed_combo.setCurrentIndex(1)
        self._streamline_seed_combo.currentIndexChanged.connect(self._on_velocity_overlay_control_changed)
        sl_seed_row.addWidget(self._streamline_seed_combo, 1)
        sidebar.addLayout(sl_seed_row)

        sl_steps_row = QtWidgets.QHBoxLayout()
        sl_steps_row.addWidget(QtWidgets.QLabel("Trace steps:"))
        self._streamline_steps_spin = QtWidgets.QSpinBox()
        self._streamline_steps_spin.setRange(4, 256)
        self._streamline_steps_spin.setValue(30)
        self._streamline_steps_spin.setSingleStep(2)
        self._streamline_steps_spin.valueChanged.connect(self._on_velocity_overlay_control_changed)
        sl_steps_row.addWidget(self._streamline_steps_spin, 1)
        sidebar.addLayout(sl_steps_row)

        sl_step_len_row = QtWidgets.QHBoxLayout()
        sl_step_len_row.addWidget(QtWidgets.QLabel("Step factor:"))
        self._streamline_step_scale_spin = QtWidgets.QDoubleSpinBox()
        self._streamline_step_scale_spin.setDecimals(2)
        self._streamline_step_scale_spin.setRange(0.1, 3.0)
        self._streamline_step_scale_spin.setValue(0.85)
        self._streamline_step_scale_spin.setSingleStep(0.05)
        self._streamline_step_scale_spin.valueChanged.connect(self._on_velocity_overlay_control_changed)
        sl_step_len_row.addWidget(self._streamline_step_scale_spin, 1)
        sidebar.addLayout(sl_step_len_row)

        self._run_count_lbl = QtWidgets.QLabel("")
        self._run_count_lbl.setStyleSheet("color: gray; font-size: 9px;")
        sidebar.addWidget(self._run_count_lbl)
        self._sync_profile_render_controls()
        sidebar.addStretch(1)
        body.addLayout(sidebar)

        self._tabs = QtWidgets.QTabWidget()
        self._ts_tab = QtWidgets.QWidget()
        self._prof_tab = QtWidgets.QWidget()
        self._meta_tab = self._build_meta_tab()
        self._tabs.addTab(self._ts_tab, "Time-Series")
        self._tabs.addTab(self._prof_tab, "Profile")
        self._tabs.addTab(self._meta_tab, "Metadata")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        body.addWidget(self._tabs, 1)
        root.addLayout(body, 1)

        # Animation bar
        root.addLayout(self._build_anim_bar())

        # Status bar
        self._status_lbl = QtWidgets.QLabel("No results loaded.")
        self._status_lbl.setStyleSheet("color: gray; font-size: 9px;")
        root.addWidget(self._status_lbl)

    def _build_meta_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        self._meta_table = QtWidgets.QTableWidget()
        self._meta_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self._meta_table.setAlternatingRowColors(True)
        self._meta_table.setColumnCount(5)
        self._meta_table.setHorizontalHeaderLabels(
            ["Run ID", "GPKG", "Timesteps", "Duration", "Profile?"]
        )
        self._meta_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._meta_table)
        return w

    def _build_anim_bar(self) -> QtWidgets.QHBoxLayout:
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)

        self._step_back_btn = QtWidgets.QPushButton("\u25c4")
        self._step_back_btn.setFixedSize(24, 22)
        self._step_back_btn.setToolTip("Step back one frame")
        self._step_back_btn.clicked.connect(self._on_step_back)

        self._play_btn = QtWidgets.QPushButton("\u25b6")
        self._play_btn.setFixedSize(24, 22)
        self._play_btn.setCheckable(True)
        self._play_btn.setToolTip("Play / Pause animation")
        self._play_btn.clicked.connect(self._on_play_pause)

        self._step_fwd_btn = QtWidgets.QPushButton("\u25b6|")
        self._step_fwd_btn.setFixedSize(28, 22)
        self._step_fwd_btn.setToolTip("Step forward one frame")
        self._step_fwd_btn.clicked.connect(self._on_step_fwd)

        self._time_slider = QtWidgets.QSlider(Qt.Horizontal)
        self._time_slider.setRange(0, 0)
        self._time_slider.setValue(0)
        self._time_slider.setTracking(True)
        self._time_slider.valueChanged.connect(self._on_slider_changed)

        self._time_lbl = QtWidgets.QLabel("T = 0.000 hr")
        self._time_lbl.setFixedWidth(100)
        self._time_lbl.setStyleSheet("font-size: 9px;")

        self._speed_combo = QtWidgets.QComboBox()
        for _spd_label in ("0.25\u00d7", "0.5\u00d7", "1\u00d7", "2\u00d7", "4\u00d7", "8\u00d7"):
            self._speed_combo.addItem(_spd_label)
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.setFixedWidth(56)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self._speed_combo.setToolTip("Playback speed")

        bar.addWidget(self._step_back_btn)
        bar.addWidget(self._play_btn)
        bar.addWidget(self._step_fwd_btn)
        bar.addWidget(self._time_slider, 1)
        bar.addWidget(self._time_lbl)
        bar.addWidget(self._speed_combo)
        return bar

    def _setup_matplotlib(self):
        self._have_mpl = _try_import_matplotlib()

        def _embed(tab_widget):
            layout = QtWidgets.QVBoxLayout(tab_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            if not self._have_mpl:
                layout.addWidget(QtWidgets.QLabel("matplotlib not available"))
                return None, None, None, None
            # Use constrained_layout instead of tight_layout for stability
            fig = _Figure(figsize=(6, 3.8), constrained_layout=True)
            ax = fig.add_subplot(111)
            canvas = _FigureCanvas(fig)
            canvas.setMinimumHeight(200)
            
            # Add navigation toolbar for zoom/pan controls
            toolbar = None
            if _NavigationToolbar:
                toolbar = _NavigationToolbar(canvas, tab_widget)
                toolbar.setIconSize(QtCore.QSize(16, 16))
                layout.addWidget(toolbar)
            
            layout.addWidget(canvas, 1)
            return fig, ax, canvas, toolbar

        self._fig_ts, self._ax_ts, self._canvas_ts, self._toolbar_ts = _embed(self._ts_tab)
        self._fig_prof, self._ax_prof, self._canvas_prof, self._toolbar_prof = _embed(self._prof_tab)

    # ------------------------------------------------------------------
    # Run discovery
    # ------------------------------------------------------------------

    def _next_color(self, index: int) -> Tuple[int, int, int]:
        return _PANEL_COLORS[index % len(_PANEL_COLORS)]

    def _collect_runs_from_gpkg(self, gpkg_path: str) -> List[RunRecord]:
        from swe2d.results.queries import discover_line_result_runs

        if not gpkg_path:
            return []
        runs = discover_line_result_runs(gpkg_path)
        out: List[RunRecord] = []
        gpkg_short = _os.path.basename(gpkg_path)
        for meta in runs:
            rid = str(meta.get("run_id", ""))
            if not rid:
                continue
            is_snapshot = rid.startswith("swe2d_snapshot_") or ("snapshot" in rid.lower())
            suffix = " [snapshot]" if is_snapshot else ""
            out.append(
                RunRecord(
                    run_id=rid,
                    gpkg_path=gpkg_path,
                    color=(0, 0, 0),
                    enabled=True,
                    has_profile=bool(meta.get("has_profile", False)),
                    label=f"{gpkg_short}:{rid}{suffix}",
                )
            )
        return out

    def _prompt_select_runs(self, gpkg_path: str, candidates: List[RunRecord]) -> List[RunRecord]:
        if not candidates:
            return []

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Select Results From { _os.path.basename(gpkg_path) }")
        dlg.resize(520, 420)
        lay = QtWidgets.QVBoxLayout(dlg)

        msg = QtWidgets.QLabel(
            "Choose individual result files/runs to add. Snapshot runs are listed too."
        )
        msg.setWordWrap(True)
        lay.addWidget(msg)

        run_list = QtWidgets.QListWidget()
        run_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for rec in candidates:
            item = QtWidgets.QListWidgetItem(rec.display_label())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, rec.key)
            item.setToolTip(f"Run: {rec.run_id}\nGPKG: {rec.gpkg_path}")
            run_list.addItem(item)
        lay.addWidget(run_list, 1)

        controls = QtWidgets.QHBoxLayout()
        sel_all_btn = QtWidgets.QPushButton("Select All")
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        controls.addWidget(sel_all_btn)
        controls.addWidget(clear_all_btn)
        controls.addStretch(1)
        lay.addLayout(controls)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        lay.addWidget(btn_box)

        def _set_all(state: int) -> None:
            for i in range(run_list.count()):
                run_list.item(i).setCheckState(state)

        sel_all_btn.clicked.connect(lambda: _set_all(Qt.Checked))
        clear_all_btn.clicked.connect(lambda: _set_all(Qt.Unchecked))
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return []

        selected: List[RunRecord] = []
        selected_keys: Set[str] = set()
        for i in range(run_list.count()):
            item = run_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            selected_keys.add(str(item.data(Qt.UserRole) or ""))

        if not selected_keys:
            return []
        for rec in candidates:
            if rec.key in selected_keys:
                selected.append(rec)
        return selected

    def _rebuild_run_list_widget(self) -> None:
        self._run_list.blockSignals(True)
        self._run_list.clear()
        for rec in self._run_records:
            item = QtWidgets.QListWidgetItem(rec.display_label())
            item.setCheckState(Qt.Checked if rec.enabled else Qt.Unchecked)
            item.setData(Qt.UserRole, rec.key)
            item.setData(Qt.UserRole + 1, rec.color)
            item.setToolTip(f"Run: {rec.run_id}\nGPKG: {rec.gpkg_path}")
            self._run_list.addItem(item)
        self._run_list.blockSignals(False)

    def _discover_runs(self):
        old_enabled = {rec.key for rec in self._run_records if rec.enabled}
        old_manual_paths = list(self._manual_gpkg_paths)

        self._gpkg_lbl.setText(
            _os.path.basename(self._gpkg_path) if self._gpkg_path else "(no GPKG)"
        )
        combined: List[RunRecord] = []
        seen: set = set()

        base_candidates = self._collect_runs_from_gpkg(self._gpkg_path)
        base_filter_keys = {
            k for k in self._base_selected_run_keys if k.startswith(f"{self._gpkg_path}::")
        }
        if base_filter_keys:
            filtered_base = [rec for rec in base_candidates if rec.key in base_filter_keys]
            if filtered_base:
                base_candidates = filtered_base
            else:
                # Stale base selection keys: clear and fall back to all discovered runs.
                self._base_selected_run_keys = {
                    k for k in self._base_selected_run_keys if not k.startswith(f"{self._gpkg_path}::")
                }

        for rec in base_candidates:
            if rec.key in seen:
                continue
            seen.add(rec.key)
            combined.append(rec)

        manual_paths: List[str] = []
        for gpkg in old_manual_paths:
            if not gpkg or gpkg == self._gpkg_path:
                continue
            if not _os.path.exists(gpkg):
                continue
            manual_paths.append(gpkg)
            candidates = self._collect_runs_from_gpkg(gpkg)
            gpkg_filter_keys = {
                k for k in self._manual_selected_run_keys if k.startswith(f"{gpkg}::")
            }
            if gpkg_filter_keys:
                filtered_candidates = [rec for rec in candidates if rec.key in gpkg_filter_keys]
                if filtered_candidates:
                    candidates = filtered_candidates
                else:
                    # Stale manual selection keys for this gpkg: clear and show all runs.
                    self._manual_selected_run_keys = {
                        k for k in self._manual_selected_run_keys if not k.startswith(f"{gpkg}::")
                    }

            for rec in candidates:
                if rec.key in seen:
                    continue
                seen.add(rec.key)
                combined.append(rec)

        for i, rec in enumerate(combined):
            rec.color = self._next_color(i)
            rec.enabled = rec.key in old_enabled if old_enabled else True

        self._manual_gpkg_paths = manual_paths
        self._run_records = combined
        self._rebuild_run_list_widget()
        self._run_count_lbl.setText(f"{len(self._run_records)} run(s)")
        self._refresh_line_combo()
        self._refresh_meta_table()

    def _add_results_files(self):
        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add SWE2D Results GeoPackage(s)",
            self._gpkg_path or "",
            "GeoPackage (*.gpkg)",
        )
        if not file_paths:
            return

        added_paths = 0
        added_runs = 0
        for fp in file_paths:
            gpkg = str(fp or "").strip()
            if not gpkg or not _os.path.exists(gpkg):
                continue
            candidates = self._collect_runs_from_gpkg(gpkg)
            if not candidates:
                continue
            selected = self._prompt_select_runs(gpkg, candidates)
            if not selected:
                continue
            if gpkg == self._gpkg_path:
                selected_keys = {rec.key for rec in selected}
                self._base_selected_run_keys = (
                    set() if len(selected_keys) >= len(candidates) else selected_keys
                )
                added_runs += len(selected)
                continue
            if gpkg not in self._manual_gpkg_paths:
                self._manual_gpkg_paths.append(gpkg)
                added_paths += 1
            for rec in selected:
                self._manual_selected_run_keys.add(rec.key)
            added_runs += len(selected)

        if added_paths <= 0 and added_runs <= 0:
            self._status_lbl.setText("No new results were added.")
            return
        self._discover_runs()
        self._status_lbl.setText(
            f"Added {added_runs} result run(s) from {added_paths} GeoPackage(s)."
        )

    # ------------------------------------------------------------------
    # Line combo
    # ------------------------------------------------------------------

    def _refresh_line_combo(self):
        from swe2d.results.queries import load_line_ids

        self._line_combo.blockSignals(True)
        self._line_combo.clear()

        seen: Dict[int, str] = {}
        for rec in self._run_records:
            if not rec.enabled:
                continue
            for lid, lname in load_line_ids(rec.gpkg_path, rec.run_id):
                if lid not in seen:
                    seen[lid] = lname

        if not seen:
            self._line_combo.addItem("(no results)", -1)
            self._line_id = -1
        else:
            for lid in sorted(seen.keys()):
                display = seen[lid] or f"Line {lid}"
                self._line_combo.addItem(display, lid)
            idx = self._line_combo.findData(self._line_id)
            if idx >= 0:
                self._line_combo.setCurrentIndex(idx)
            else:
                self._line_id = int(self._line_combo.currentData() or -1)

        self._line_combo.blockSignals(False)
        self._rebuild_timestep_union()
        self._refresh_plots()

    # ------------------------------------------------------------------
    # Timestep union / slider
    # ------------------------------------------------------------------

    def _rebuild_timestep_union(self):
        from swe2d.results.queries import load_timesteps

        if self._line_id < 0:
            self._all_timesteps = np.empty(0, dtype=np.float64)
            self._update_slider_range()
            return

        ts_sets: List[np.ndarray] = []
        for rec in self._run_records:
            if not rec.enabled:
                continue
            ts = load_timesteps(rec.gpkg_path, rec.run_id, self._line_id)
            if ts.size:
                ts_sets.append(ts)

        if ts_sets:
            self._all_timesteps = np.unique(np.concatenate(ts_sets))
        else:
            self._all_timesteps = np.empty(0, dtype=np.float64)

        self._update_slider_range()

    def _update_slider_range(self):
        n = self._all_timesteps.size
        self._anim_frame_idx = 0
        self._anim.set_timesteps(self._all_timesteps)
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, max(0, n - 1))
        self._time_slider.setValue(0)
        self._time_slider.blockSignals(False)
        if n:
            self._current_t_sec = float(self._all_timesteps[0])
        self._update_time_label()

    def _update_time_label(self):
        self._time_lbl.setText(f"T = {self._current_t_sec / 3600.0:.3f} hr")

    def _t_sec_to_frame_idx(self, t_sec: float) -> int:
        if self._all_timesteps.size == 0:
            return 0
        return int(np.argmin(np.abs(self._all_timesteps - float(t_sec))))

    def _frame_idx_to_t_sec(self, idx: int) -> float:
        n = self._all_timesteps.size
        if n == 0:
            return 0.0
        return float(self._all_timesteps[max(0, min(idx, n - 1))])

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_run_toggle(self, item: QtWidgets.QListWidgetItem):
        run_key = item.data(Qt.UserRole)
        enabled = item.checkState() == Qt.Checked
        for rec in self._run_records:
            if rec.key == run_key:
                rec.enabled = enabled
                break
        self._refresh_line_combo()
        self.velocity_overlay_changed.emit()

    def _remove_selected_runs(self):
        """Permanently remove selected run(s) from the viewer."""
        selected_keys = {
            item.data(Qt.UserRole)
            for item in self._run_list.selectedItems()
            if item.data(Qt.UserRole)
        }
        if not selected_keys:
            return
        self._run_records = [r for r in self._run_records if r.key not in selected_keys]
        self._base_selected_run_keys -= selected_keys
        self._manual_selected_run_keys -= selected_keys
        # Drop manual GPKG paths that no longer have any runs.
        remaining_manual = {r.gpkg_path for r in self._run_records if r.gpkg_path != self._gpkg_path}
        self._manual_gpkg_paths = [p for p in self._manual_gpkg_paths if p in remaining_manual]
        # Re-assign colors so the palette stays compact.
        for i, rec in enumerate(self._run_records):
            rec.color = self._next_color(i)
        self._rebuild_run_list_widget()
        self._run_count_lbl.setText(f"{len(self._run_records)} run(s)")
        self._refresh_line_combo()
        self._refresh_meta_table()
        self.velocity_overlay_changed.emit()

    def _set_all_runs_visible(self):
        """Check all runs (show in plots/tables)."""
        self._run_list.blockSignals(True)
        for i in range(self._run_list.count()):
            self._run_list.item(i).setCheckState(Qt.Checked)
        self._run_list.blockSignals(False)
        for rec in self._run_records:
            rec.enabled = True
        self._refresh_line_combo()
        self.velocity_overlay_changed.emit()

    def _set_all_runs_hidden(self):
        """Uncheck all runs (hide from plots/tables)."""
        self._run_list.blockSignals(True)
        for i in range(self._run_list.count()):
            self._run_list.item(i).setCheckState(Qt.Unchecked)
        self._run_list.blockSignals(False)
        for rec in self._run_records:
            rec.enabled = False
        self._refresh_line_combo()
        self.velocity_overlay_changed.emit()

    def _show_run_list_context_menu(self, pos: QtCore.QPoint):
        """Right-click context menu for the run list."""
        menu = QtWidgets.QMenu(self)
        if self._run_list.selectedItems():
            remove_act = menu.addAction("Remove selected run(s)")
            remove_act.triggered.connect(self._remove_selected_runs)
            menu.addSeparator()
        show_act = menu.addAction("\u2713 Show all")
        show_act.triggered.connect(self._set_all_runs_visible)
        hide_act = menu.addAction("\u25a1 Hide all")
        hide_act.triggered.connect(self._set_all_runs_hidden)
        menu.exec_(self._run_list.viewport().mapToGlobal(pos))

    def _on_line_changed(self, _index: int):
        lid = self._line_combo.currentData()
        if lid is not None and int(lid) != self._line_id:
            self._line_id = int(lid)
            self._rebuild_timestep_union()
            self._refresh_plots()

    def _on_ts_var_changed(self, _):
        if self._tabs.currentIndex() == 0:
            self._refresh_timeseries()

    def _on_prof_var_changed(self, _):
        self._sync_profile_render_controls()
        if self._tabs.currentIndex() == 1:
            self._refresh_profile()

    def _on_prof_fill_changed(self, _):
        if self._tabs.currentIndex() == 1:
            self._refresh_profile()

    def _sync_profile_render_controls(self):
        is_wse_bed = str(self._prof_var_combo.currentData() or "wse_bed") == "wse_bed"
        fill_enabled = is_wse_bed and str(self._prof_fill_combo.currentData() or "none") != "none"
        if hasattr(self, "_prof_fill_widget"):
            self._prof_fill_widget.setVisible(is_wse_bed)
        if hasattr(self, "_prof_wse_render_widget"):
            self._prof_wse_render_widget.setVisible(is_wse_bed)
        if hasattr(self, "_prof_cmap_widget"):
            self._prof_cmap_widget.setVisible(fill_enabled)

    def _on_tab_changed(self, index: int):
        if index == 0:
            self._refresh_timeseries()
        elif index == 1:
            self._refresh_profile()
        elif index == 2:
            self._refresh_meta_table()

    def _on_slider_changed(self, value: int):
        self._anim.set_index(int(value))

    def _on_play_pause(self, checked: bool):
        if checked:
            self._anim.play()
        else:
            self._anim.pause()

    def _on_step_back(self):
        self._anim.pause()
        self._anim.step_backward()

    def _on_step_fwd(self):
        self._anim.pause()
        self._anim.step_forward()

    def _on_speed_changed(self, index: int):
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        self._anim_fps = _DEFAULT_FPS * speeds[index]
        self._anim.set_frame_rate(self._anim_fps)

    def _on_controller_timestep_changed(self, t_sec: float, frame_idx: int):
        self._anim_frame_idx = int(frame_idx)
        self._current_t_sec = float(t_sec)
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(int(frame_idx))
        self._time_slider.blockSignals(False)
        self._update_time_label()
        tab = self._tabs.currentIndex()
        if tab == 0:
            self._update_ts_vline()
        elif tab == 1:
            self._refresh_profile()
        self.timestep_changed.emit(float(t_sec))

    def _on_velocity_overlay_control_changed(self, *_):
        self.velocity_overlay_changed.emit()

    def _on_velocity_overlay_add_requested(self):
        self.velocity_overlay_add_requested.emit()

    def _on_controller_play_state_changed(self, playing: bool):
        self._play_btn.blockSignals(True)
        self._play_btn.setChecked(bool(playing))
        self._play_btn.setText("\u23f8" if playing else "\u25b6")
        self._play_btn.blockSignals(False)

    def _set_frame(self, idx: int):
        self._anim.set_index(int(idx))

    # ------------------------------------------------------------------
    # Plot rendering
    # ------------------------------------------------------------------

    def _refresh_plots(self):
        if not self._have_mpl:
            return
        tab = self._tabs.currentIndex()
        if tab == 0:
            self._refresh_timeseries()
        elif tab == 1:
            self._refresh_profile()

    def _refresh_timeseries(self):
        if not self._have_mpl or self._ax_ts is None:
            return
        from swe2d.results.queries import load_timeseries

        var_key = str(self._ts_var_combo.currentData() or "flow_cms")
        var_label = self._ts_var_combo.currentText()

        self._ax_ts.cla()
        self._ts_vline = None
        plotted = 0

        for rec in self._run_records:
            if not rec.enabled or self._line_id < 0:
                continue
            data = load_timeseries(rec.gpkg_path, rec.run_id, self._line_id)
            if not data or var_key not in data:
                continue
            t_hr = data["t_s"] / 3600.0
            vals = data[var_key]
            self._ax_ts.plot(
                t_hr, vals,
                color=_c2f(rec.color), linewidth=1.6,
                label=rec.display_label(),
            )
            plotted += 1

        t_hr_now = self._current_t_sec / 3600.0
        self._ts_vline = self._ax_ts.axvline(
            x=t_hr_now, color="0.5", linewidth=0.9,
            linestyle="--", zorder=5,
        )

        self._ax_ts.set_xlabel("Time (hr)")
        self._ax_ts.set_ylabel(var_label)
        self._ax_ts.grid(True, alpha=0.3)
        if plotted:
            self._ax_ts.legend(fontsize=8, loc="best")
            self._status_lbl.setText(f"Time-series: {plotted} run(s).")
        else:
            self._ax_ts.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=self._ax_ts.transAxes, color="gray",
            )
        self._canvas_ts.draw_idle()

    def _update_ts_vline(self):
        """Move the vertical time marker cheaply without full redraw."""
        if not self._have_mpl or self._ax_ts is None or self._canvas_ts is None:
            return
        t_hr = self._current_t_sec / 3600.0
        if self._ts_vline is not None:
            try:
                self._ts_vline.set_xdata([t_hr, t_hr])
                self._canvas_ts.draw_idle()
                return
            except Exception:
                pass
        self._refresh_timeseries()

    def _refresh_profile(self):
        if not self._have_mpl or self._ax_prof is None:
            return
        from swe2d.results.queries import (
            find_nearest_timestep,
            load_profile,
            load_structure_flows_at_time,
        )

        mode = str(self._prof_var_combo.currentData() or "wse_bed")
        fill_key = str(self._prof_fill_combo.currentData() or "none") if hasattr(self, "_prof_fill_combo") else "none"
        render_mode = str(self._prof_wse_render_combo.currentData() or "clipped") if hasattr(self, "_prof_wse_render_combo") else "clipped"
        cmap_name = str(self._prof_cmap_combo.currentData() or "viridis") if hasattr(self, "_prof_cmap_combo") else "viridis"
        use_fill_cmap = mode == "wse_bed" and fill_key != "none"

        if self._prof_fill_cbar is not None:
            try:
                self._prof_fill_cbar.remove()
            except Exception:
                pass
            self._prof_fill_cbar = None

        self._ax_prof.cla()
        plotted = 0
        bed_drawn = False
        structure_rows: List[Dict[str, object]] = []
        line_name = str(self._line_combo.currentText() or "")
        fill_segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        fill_values: List[float] = []

        for rec in self._run_records:
            if not rec.enabled or self._line_id < 0:
                continue
            t = find_nearest_timestep(
                rec.gpkg_path, rec.run_id, self._line_id, self._current_t_sec
            )
            data = load_profile(rec.gpkg_path, rec.run_id, self._line_id, t)
            if not data:
                continue

            color = _c2f(rec.color)
            station = data.get("station_m", np.empty(0))

            if mode == "wse_bed":
                wse = data.get("wse_m", np.full_like(station, np.nan))
                bed = data.get("bed_m", np.full_like(station, np.nan))
                depth = data.get("depth_m", np.full_like(station, np.nan))
                wet = data.get("wet", np.ones_like(station))

                ok = np.isfinite(wse) & np.isfinite(bed)
                if not np.any(ok):
                    continue

                x_ok = station[ok]
                wse_ok = wse[ok]
                bed_ok = bed[ok]
                depth_ok = depth[ok]
                wet_ok = wet[ok]
                wet_mask = np.where(
                    np.isfinite(wet_ok), wet_ok > 0.5, depth_ok > 1e-9
                )
                wse_phys = np.maximum(wse_ok, bed_ok)

                if render_mode == "raw":
                    fill_mask = np.isfinite(wse_ok) & np.isfinite(bed_ok)
                    wse_fill = wse_ok
                    wse_plot = wse_ok
                else:
                    fill_mask = wet_mask
                    wse_fill = wse_phys
                    wse_plot = np.where(wet_mask, wse_phys, np.nan)

                if not bed_drawn and x_ok.size:
                    bed_min = float(np.min(bed_ok)) - 0.05 * max(float(np.ptp(bed_ok)), 0.1)
                    self._ax_prof.fill_between(
                        x_ok, bed_min, bed_ok,
                        color="#8B7355", alpha=0.5, zorder=1,
                    )
                    self._ax_prof.plot(
                        x_ok, bed_ok,
                        color="#5C4033", linewidth=0.9, zorder=2,
                    )
                    bed_drawn = True

                if use_fill_cmap:
                    fill_metric = np.asarray(
                        data.get(fill_key, np.full_like(station, np.nan)),
                        dtype=np.float64,
                    )
                    fill_ok = fill_metric[ok]
                    for i in range(len(x_ok) - 1):
                        if not (fill_mask[i] and fill_mask[i + 1]):
                            continue
                        if not (np.isfinite(fill_ok[i]) and np.isfinite(fill_ok[i + 1])):
                            continue
                        vmid = 0.5 * (float(fill_ok[i]) + float(fill_ok[i + 1]))
                        fill_values.append(vmid)
                        fill_segments.append(
                            (
                                x_ok[i : i + 2],
                                bed_ok[i : i + 2],
                                wse_fill[i : i + 2],
                                vmid,
                            )
                        )
                else:
                    self._ax_prof.fill_between(
                        x_ok, bed_ok, wse_fill,
                        where=fill_mask, interpolate=True,
                        color=color, alpha=0.18, zorder=3,
                    )
                self._ax_prof.plot(
                    x_ok, wse_plot,
                    color=color, linewidth=1.5, zorder=4,
                    label=f"{rec.display_label()} WSE",
                )
                plotted += 1

            else:
                if mode == "egl_m":
                    wse = data.get("wse_m")
                    vel = data.get("velocity_ms")
                    if wse is None or vel is None:
                        continue
                    y = np.asarray(wse, dtype=np.float64) + (
                        np.asarray(vel, dtype=np.float64) ** 2.0
                    ) / (2.0 * 9.81)
                else:
                    if mode not in data:
                        continue
                    y = data[mode]
                ok = np.isfinite(station) & np.isfinite(y)
                if not np.any(ok):
                    continue
                self._ax_prof.plot(
                    station[ok], y[ok],
                    color=color, linewidth=1.5,
                    label=rec.display_label(),
                )
                plotted += 1

            if self._show_structures_chk.isChecked():
                try:
                    rows = load_structure_flows_at_time(
                        rec.gpkg_path,
                        rec.run_id,
                        t,
                        t_tol=1.0,
                    )
                    structure_rows.extend(
                        self._resolve_structure_profile_overlays(
                            rec.gpkg_path,
                            rec.run_id,
                            rec.display_label(),
                            self._line_id,
                            line_name,
                            rows,
                        )
                    )
                    if not rows:
                        continue
                    placed_ids = {str(r.get("object_id", "")) for r in structure_rows}
                    for rr in rows:
                        sid = str(rr.get("object_id", ""))
                        if sid in placed_ids:
                            continue
                        structure_rows.append(
                            {
                                "run_label": rec.display_label(),
                                "object_id": sid,
                                "flow": float(rr.get("value", 0.0)),
                                "station": float("nan"),
                                "elev": float("nan"),
                                "placement": "unplaced",
                            }
                        )
                except Exception:
                    pass

        if use_fill_cmap and fill_segments and fill_values:
            try:
                from matplotlib import cm as mpl_cm, colors as mpl_colors

                vals = np.asarray(fill_values, dtype=np.float64)
                finite = np.isfinite(vals)
                if np.any(finite):
                    vmin = float(np.nanmin(vals[finite]))
                    vmax = float(np.nanmax(vals[finite]))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
                    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
                    cmap = mpl_cm.get_cmap(cmap_name)
                    for x_seg, bed_seg, wse_seg, vmid in fill_segments:
                        self._ax_prof.fill_between(
                            x_seg,
                            bed_seg,
                            wse_seg,
                            color=cmap(norm(vmid)),
                            alpha=0.85,
                            linewidth=0.0,
                            zorder=3,
                        )
                    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    self._prof_fill_cbar = self._fig_prof.colorbar(
                        sm,
                        ax=self._ax_prof,
                        label=self._prof_fill_combo.currentText(),
                    )
            except Exception:
                pass

        if plotted and self._show_structures_chk.isChecked() and structure_rows:
            x0, x1 = self._ax_prof.get_xlim()
            y0, y1 = self._ax_prof.get_ylim()
            placed = [r for r in structure_rows if np.isfinite(float(r.get("station", float("nan"))))]
            if np.isfinite(x0) and np.isfinite(x1) and x1 > x0 and np.isfinite(y0) and np.isfinite(y1) and placed:
                placed = sorted(placed, key=lambda r: float(r.get("station", 0.0)))[:12]
                y_span = max(y1 - y0, 1.0e-6)
                for i, row in enumerate(placed):
                    xs = float(row.get("station", 0.0))
                    elev = float(row.get("elev", float("nan")))
                    q_val = float(row.get("flow", 0.0))
                    sid = str(row.get("object_id", ""))
                    y_anchor = elev if np.isfinite(elev) else y1
                    y_anchor = min(max(y_anchor, y0 + 0.08 * y_span), y1 - 0.02 * y_span)
                    y_text = min(y1 - 0.02 * y_span, y_anchor + (0.04 + 0.035 * (i % 3)) * y_span)

                    self._ax_prof.axvline(
                        xs,
                        color="0.35",
                        linewidth=0.9,
                        linestyle=":",
                        alpha=0.5,
                        zorder=2,
                    )
                    if np.isfinite(elev):
                        self._ax_prof.plot(
                            [xs],
                            [elev],
                            marker="v",
                            markersize=4.0,
                            color="0.25",
                            zorder=6,
                        )
                    self._ax_prof.text(
                        xs,
                        y_text,
                        f"{sid} {q_val:.2f}",
                        fontsize=7,
                        rotation=90,
                        va="top",
                        ha="center",
                        color="0.35",
                        zorder=6,
                    )
                self._status_lbl.setText(
                    f"Profile: {plotted} run(s) at t={self._current_t_sec / 3600.0:.3f} hr; structures={len(placed)} placed."
                )

        self._ax_prof.set_xlabel("Station")
        var_label = self._prof_var_combo.currentText()
        self._ax_prof.set_ylabel(
            "Elevation" if mode == "wse_bed" else var_label
        )
        t_hr = self._current_t_sec / 3600.0
        self._ax_prof.set_title(f"t = {t_hr:.3f} hr", fontsize=9)
        self._ax_prof.grid(True, alpha=0.3)

        if plotted:
            self._ax_prof.legend(fontsize=8, loc="best")
            self._status_lbl.setText(
                f"Profile: {plotted} run(s) at t={t_hr:.3f} hr."
            )
        else:
            self._ax_prof.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=self._ax_prof.transAxes, color="gray",
            )
        self._canvas_prof.draw_idle()

    def _load_bound_layer_name(self, gpkg_path: str, role: str, default_name: str) -> str:
        if not gpkg_path:
            return str(default_name)
        conn = _open_ro_shared(gpkg_path)
        if conn is None:
            return str(default_name)
        try:
            if not _table_exists_shared(conn, "swe2d_layer_bindings"):
                return str(default_name)
            cur = conn.execute(
                "SELECT layer_name FROM swe2d_layer_bindings WHERE role = ?",
                (str(role),),
            )
            row = cur.fetchone()
            if row is None or not row[0]:
                return str(default_name)
            return str(row[0])
        except Exception:
            return str(default_name)
        finally:
            if conn is not None:
                conn.close()

    def _load_profile_line_geometry(self, gpkg_path: str, line_id: int, line_name: str):
        if QgsVectorLayer is None or not gpkg_path or line_id < 0:
            return None
        layer_name = "swe2d_sample_lines"
        line_layer = QgsVectorLayer(_line_layer_uri(gpkg_path, layer_name), layer_name, "ogr")
        if line_layer is None or not line_layer.isValid():
            return None
        fields = set(line_layer.fields().names())
        for ft in line_layer.getFeatures():
            try:
                if "line_id" in fields and int(ft["line_id"]) == int(line_id):
                    return ft.geometry()
            except Exception:
                pass
            try:
                if line_name and "name" in fields and str(ft["name"] or "") == line_name:
                    return ft.geometry()
            except Exception:
                pass
        return None

    def _resolve_structure_profile_overlays(
        self,
        gpkg_path: str,
        run_id: str,
        run_label: str,
        line_id: int,
        line_name: str,
        structure_rows: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        if QgsVectorLayer is None or not structure_rows:
            return []
        line_geom = self._load_profile_line_geometry(gpkg_path, line_id, line_name)
        if line_geom is None or line_geom.isEmpty():
            return []

        layer_name = self._load_bound_layer_name(gpkg_path, "hydraulic_structures", "swe2d_structures")
        layer = QgsVectorLayer(_line_layer_uri(gpkg_path, layer_name), layer_name, "ogr")
        if layer is None or not layer.isValid():
            return []

        fields = set(layer.fields().names())
        if "structure_id" not in fields:
            return []

        flow_by_id = {
            str(r.get("object_id", "")): float(r.get("value", 0.0))
            for r in structure_rows
            if str(r.get("object_id", ""))
        }
        overlays: List[Dict[str, object]] = []
        for ft in layer.getFeatures():
            sid = str(ft["structure_id"] or "")
            if sid not in flow_by_id:
                continue
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue

            station_m = float("nan")
            try:
                inter = geom.intersection(line_geom)
                if inter is not None and not inter.isEmpty():
                    station_m = float(line_geom.lineLocatePoint(inter.centroid()))
            except Exception:
                station_m = float("nan")

            if not np.isfinite(station_m):
                try:
                    centroid = geom.centroid()
                    if centroid is not None and not centroid.isEmpty():
                        nearest = line_geom.nearestPoint(centroid)
                        if nearest is not None and not nearest.isEmpty():
                            station_m = float(line_geom.lineLocatePoint(nearest))
                except Exception:
                    station_m = float("nan")

            crest = float("nan")
            try:
                if "crest_elev" in fields and ft["crest_elev"] not in (None, ""):
                    crest = float(ft["crest_elev"])
            except Exception:
                crest = float("nan")

            overlays.append(
                {
                    "run_id": str(run_id),
                    "run_label": str(run_label),
                    "object_id": sid,
                    "flow_cms": float(flow_by_id[sid]),
                    "station_m": station_m,
                    "elev_m": crest,
                    "placement": "geometry" if np.isfinite(station_m) else "fallback",
                }
            )
        return overlays

    def _refresh_meta_table(self):
        from swe2d.results.queries import load_timesteps, load_line_ids

        self._meta_table.setRowCount(len(self._run_records))
        for r, rec in enumerate(self._run_records):
            lines = load_line_ids(rec.gpkg_path, rec.run_id)
            if lines and self._line_id >= 0:
                ts = load_timesteps(rec.gpkg_path, rec.run_id, self._line_id)
                n_ts = ts.size
                duration = (
                    f"{(float(ts[-1]) - float(ts[0])) / 3600.0:.2f} hr"
                    if n_ts >= 2 else "\u2014"
                )
            else:
                n_ts, duration = 0, "\u2014"
            gpkg_short = _os.path.basename(rec.gpkg_path)
            vals = (
                rec.run_id,
                gpkg_short,
                str(n_ts),
                duration,
                "Yes" if rec.has_profile else "No",
            )
            for c, val in enumerate(vals):
                self._meta_table.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        self._meta_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Project persistence
    # ------------------------------------------------------------------

    def save_state(self):
        if not _HAVE_QGSPROJECT or _QgsProject is None:
            return
        state = {
            "run_keys_enabled": [r.key for r in self._run_records if r.enabled],
            "run_ids_enabled": [r.run_id for r in self._run_records if r.enabled],
            "manual_gpkg_paths": list(self._manual_gpkg_paths),
            "base_selected_run_keys": sorted(self._base_selected_run_keys),
            "manual_selected_run_keys": sorted(self._manual_selected_run_keys),
            "line_id": self._line_id,
            "t_sec": self._current_t_sec,
            "frame_idx": int(self._anim_frame_idx),
            "tab_index": self._tabs.currentIndex(),
            "ts_var": self._ts_var_combo.currentData(),
            "prof_var": self._prof_var_combo.currentData(),
            "prof_fill_var": self._prof_fill_combo.currentData(),
            "prof_wse_render_mode": self._prof_wse_render_combo.currentData(),
            "prof_fill_cmap": self._prof_cmap_combo.currentData(),
            "show_structures": bool(self._show_structures_chk.isChecked()),
            "show_velocity": bool(self._show_velocity_chk.isChecked()),
            "show_streamlines": bool(self._show_streamlines_chk.isChecked()),
            "velocity_density": int(self.velocity_density_stride()),
            "velocity_min_speed": float(self.velocity_min_speed()),
            "streamline_seed_count": int(self.streamline_seed_count()),
            "streamline_max_steps": int(self.streamline_max_steps()),
            "streamline_step_scale": float(self.streamline_step_scale()),
            "speed_index": int(self._speed_combo.currentIndex()),
            "is_playing": bool(self._anim.is_playing),
        }
        try:
            _QgsProject.instance().writeEntry(
                _PERSISTENCE_GROUP, _PERSISTENCE_KEY, json.dumps(state)
            )
        except Exception:
            pass

    def restore_state(self):
        if not _HAVE_QGSPROJECT or _QgsProject is None:
            return
        try:
            raw, _ = _QgsProject.instance().readEntry(
                _PERSISTENCE_GROUP, _PERSISTENCE_KEY, ""
            )
            if not raw:
                return
            state = json.loads(raw)
        except Exception:
            return

        self._manual_gpkg_paths = [
            str(p) for p in state.get("manual_gpkg_paths", [])
            if isinstance(p, str) and p and _os.path.exists(p) and p != self._gpkg_path
        ]
        self._base_selected_run_keys = {
            str(k) for k in state.get("base_selected_run_keys", [])
            if isinstance(k, str) and k
        }
        self._manual_selected_run_keys = {
            str(k) for k in state.get("manual_selected_run_keys", [])
            if isinstance(k, str) and k
        }
        self._discover_runs()

        enabled_keys = set(state.get("run_keys_enabled", []))
        enabled_ids = set(state.get("run_ids_enabled", []))
        for i in range(self._run_list.count()):
            item = self._run_list.item(i)
            run_key = str(item.data(Qt.UserRole) or "")
            rid = run_key.split("::", 1)[1] if "::" in run_key else run_key
            should_enable = (run_key in enabled_keys) if enabled_keys else (rid in enabled_ids)
            item.setCheckState(
                Qt.Checked if should_enable else Qt.Unchecked
            )

        lid = state.get("line_id", -1)
        idx = self._line_combo.findData(lid)
        if idx >= 0:
            self._line_combo.setCurrentIndex(idx)
            self._line_id = int(lid)

        t_sec = float(state.get("t_sec", 0.0))
        self._set_frame(self._t_sec_to_frame_idx(t_sec))

        self._tabs.setCurrentIndex(int(state.get("tab_index", 0)))

        ts_var = state.get("ts_var", "flow_cms")
        idx_v = self._ts_var_combo.findData(ts_var)
        if idx_v >= 0:
            self._ts_var_combo.setCurrentIndex(idx_v)

        pv = state.get("prof_var", "wse_bed")
        idx_pv = self._prof_var_combo.findData(pv)
        if idx_pv >= 0:
            self._prof_var_combo.setCurrentIndex(idx_pv)

        pfv = state.get("prof_fill_var", "none")
        idx_pfv = self._prof_fill_combo.findData(pfv)
        if idx_pfv >= 0:
            self._prof_fill_combo.setCurrentIndex(idx_pfv)

        prm = state.get("prof_wse_render_mode", "clipped")
        idx_prm = self._prof_wse_render_combo.findData(prm)
        if idx_prm >= 0:
            self._prof_wse_render_combo.setCurrentIndex(idx_prm)

        pcm = state.get("prof_fill_cmap", "viridis")
        idx_pcm = self._prof_cmap_combo.findData(pcm)
        if idx_pcm >= 0:
            self._prof_cmap_combo.setCurrentIndex(idx_pcm)

        self._sync_profile_render_controls()

        show_structures = bool(state.get("show_structures", True))
        self._show_structures_chk.setChecked(show_structures)

        show_velocity = bool(state.get("show_velocity", False))
        self._show_velocity_chk.setChecked(show_velocity)

        show_streamlines = bool(state.get("show_streamlines", False))
        self._show_streamlines_chk.setChecked(show_streamlines)

        density = int(state.get("velocity_density", 4))
        idx_den = self._vel_density_combo.findData(density)
        if idx_den >= 0:
            self._vel_density_combo.setCurrentIndex(idx_den)

        self._vel_min_speed_spin.setValue(float(state.get("velocity_min_speed", 0.05)))

        sl_count = int(state.get("streamline_seed_count", 48))
        idx_sl_count = self._streamline_seed_combo.findData(sl_count)
        if idx_sl_count >= 0:
            self._streamline_seed_combo.setCurrentIndex(idx_sl_count)

        self._streamline_steps_spin.setValue(int(state.get("streamline_max_steps", 30)))
        self._streamline_step_scale_spin.setValue(float(state.get("streamline_step_scale", 0.85)))

        speed_index = int(state.get("speed_index", 2))
        if 0 <= speed_index < self._speed_combo.count():
            self._speed_combo.setCurrentIndex(speed_index)

        frame_idx = state.get("frame_idx", None)
        if frame_idx is not None:
            self._set_frame(int(frame_idx))

        if bool(state.get("is_playing", False)):
            self._anim.play()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_gpkg_path(self, gpkg_path: str):
        new = str(gpkg_path or "")
        if new == self._gpkg_path:
            return
        self._gpkg_path = new
        self._discover_runs()

    def set_current_time(self, t_sec: float):
        self._set_frame(self._t_sec_to_frame_idx(float(t_sec)))

    def active_overlay_run_id(self) -> str:
        for rec in self._run_records:
            if rec.enabled:
                return str(rec.run_id)
        return ""

    def enabled_overlay_targets(self) -> List[Tuple[str, str]]:
        """Return enabled overlay targets as (gpkg_path, run_id) pairs."""
        out: List[Tuple[str, str]] = []
        for rec in self._run_records:
            if not rec.enabled:
                continue
            gpkg = str(rec.gpkg_path or "").strip()
            run_id = str(rec.run_id or "").strip()
            if not gpkg or not run_id:
                continue
            out.append((gpkg, run_id))
        return out

    def current_time_sec(self) -> float:
        return float(self._current_t_sec)

    def velocity_overlay_enabled(self) -> bool:
        return bool(self._show_velocity_chk.isChecked())

    def set_velocity_overlay_enabled(self, enabled: bool):
        self._show_velocity_chk.setChecked(bool(enabled))

    def velocity_density_stride(self) -> int:
        return int(self._vel_density_combo.currentData() or 1)

    def velocity_min_speed(self) -> float:
        return float(self._vel_min_speed_spin.value())

    def streamline_overlay_enabled(self) -> bool:
        return bool(self._show_streamlines_chk.isChecked())

    def set_streamline_overlay_enabled(self, enabled: bool):
        self._show_streamlines_chk.setChecked(bool(enabled))

    def streamline_seed_count(self) -> int:
        return int(self._streamline_seed_combo.currentData() or 48)

    def streamline_max_steps(self) -> int:
        return int(self._streamline_steps_spin.value())

    def streamline_step_scale(self) -> float:
        return float(self._streamline_step_scale_spin.value())

    def run_ids_for_gpkg(self, gpkg_path: str, enabled_only: bool = False) -> List[str]:
        gpkg_norm = str(gpkg_path or "").strip()
        out: List[str] = []
        seen: Set[str] = set()
        for rec in self._run_records:
            if str(rec.gpkg_path or "").strip() != gpkg_norm:
                continue
            if enabled_only and not rec.enabled:
                continue
            rid = str(rec.run_id or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
        return out

    def closeEvent(self, event):
        self._anim.pause()
        self.save_state()
        super().closeEvent(event)
