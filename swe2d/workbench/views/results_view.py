"""Results View — owns all widgets, matplotlib canvases, and signal handlers.

Calls SWE2DResultsData for data access.  No data logic lives here.
"""
from __future__ import annotations

import logging
import os as _os
from typing import Dict, List, Optional, Tuple

import numpy as np

from swe2d import units as _u
from swe2d.results.data import SWE2DResultsData
from swe2d import units as _u
from swe2d.workbench.services.results_render_service import (
    _TIME_UNIT,
    _TS_VARIABLES,
    _PROFILE_VARIABLES as _PROF_VARIABLES,
    _PROFILE_FILL_OPTIONS,
    _PROFILE_CMAP_OPTIONS,
    _SPEEDS,
    _ts_var_labels,
    _profile_var_labels,
    _profile_fill_labels,
    _label_for_var,
)

logger = logging.getLogger(__name__)

_TIME_UNIT = "hr"  # time is always in hours

try:
    from qgis.PyQt import QtCore, QtGui, QtWidgets
    from qgis.PyQt.QtCore import Qt
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt

try:
    import matplotlib
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas

    _NavigationToolbar = None
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar
    except ImportError:
        pass

    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Color-swatch list delegate
# ---------------------------------------------------------------------------

class _SwatchDelegate(QtWidgets.QStyledItemDelegate):
    _SW = 12
    _GAP = 3

    def paint(self, painter, option, index):
        """Paint a run list item with a color swatch badge."""
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
# Main view
# ---------------------------------------------------------------------------

class StudioResultsView:
    """Owns all results widgets.  Handlers call SWE2DResultsData API."""

    def __init__(self, data: SWE2DResultsData = None):
        self._data = data
        self._have_mpl = _HAVE_MPL

        # matplotlib state
        self._fig_ts = None
        self._ax_ts = None
        self._canvas_ts = None
        self._fig_prof = None
        self._ax_prof = None
        self._canvas_prof = None
        self._ts_vline = None
        self._prof_fill_cbar = None

        # Status callback (set by Studio after construction)
        self._log_fn = None

        self._build_ui()
        self._setup_matplotlib()

    def set_log_fn(self, fn):
        """Set the log callback function."""
        self._log_fn = fn

    def set_data(self, data):
        """Bind the data layer after construction (for lazy initialization)."""
        self._data = data
        self._refresh_unit_aware_labels()

    def _refresh_unit_aware_labels(self) -> None:
        """ponytail: rebuild the var combos with current unit labels."""
        length_unit = _u.length_unit_name()
        for combo, items in (
            (self._ts_var_combo, _ts_var_labels(length_unit)),
            (self._prof_var_combo, _profile_var_labels(length_unit)),
            (self._prof_fill_combo, _profile_fill_labels(length_unit)),
        ):
            if combo is None:
                continue
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for label, key in items:
                combo.addItem(label, key)
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _log(self, msg: str):
        """Forward a message to the log callback, if set."""
        if self._log_fn is not None:
            self._log_fn(msg)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the results panel with run list, variable combos, and animation bar."""
        self._root = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(self._root)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # Top bar (GPKG label, add, refresh)
        top = QtWidgets.QHBoxLayout()
        self._gpkg_lbl = QtWidgets.QLabel()
        self._gpkg_lbl.setStyleSheet("color: gray; font-size: 9px;")
        self._gpkg_lbl.setMaximumWidth(320)
        self._refresh_btn = QtWidgets.QPushButton("\u21ba")
        self._refresh_btn.setFixedSize(22, 22)
        self._refresh_btn.setToolTip("Re-scan GPKG for new runs")
        self._refresh_btn.clicked.connect(self._on_refresh)
        self._add_btn = QtWidgets.QPushButton("+")
        self._add_btn.setFixedSize(22, 22)
        self._add_btn.setToolTip("Add results from one or more GeoPackages")
        self._add_btn.clicked.connect(self._on_add_files)
        top.addWidget(self._gpkg_lbl, 1)
        top.addWidget(self._add_btn)
        top.addWidget(self._refresh_btn)
        root.addLayout(top)

        # Runs label
        root.addWidget(QtWidgets.QLabel("<b>Runs</b>"))

        # Run list
        self._run_list = QtWidgets.QListWidget()
        self._run_list.setAlternatingRowColors(True)
        self._run_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._run_list.setItemDelegate(_SwatchDelegate(self._run_list))
        self._run_list.itemChanged.connect(self._on_run_toggle)
        root.addWidget(self._run_list, 1)

        # Run list action buttons
        run_btn_row = QtWidgets.QHBoxLayout()
        run_btn_row.setSpacing(2)
        self._remove_runs_btn = QtWidgets.QPushButton("\u2212 Remove")
        self._remove_runs_btn.setFixedHeight(20)
        self._remove_runs_btn.clicked.connect(self._on_remove_runs)
        self._show_all_btn = QtWidgets.QPushButton("\u2713 All")
        self._show_all_btn.setFixedHeight(20)
        self._show_all_btn.clicked.connect(self._on_show_all)
        self._hide_all_btn = QtWidgets.QPushButton("\u25a1 None")
        self._hide_all_btn.setFixedHeight(20)
        self._hide_all_btn.clicked.connect(self._on_hide_all)
        run_btn_row.addWidget(self._remove_runs_btn, 2)
        run_btn_row.addWidget(self._show_all_btn, 1)
        run_btn_row.addWidget(self._hide_all_btn, 1)
        root.addLayout(run_btn_row)

        # Line combo
        line_row = QtWidgets.QHBoxLayout()
        line_row.addWidget(QtWidgets.QLabel("Line:"))
        self._line_combo = QtWidgets.QComboBox()
        self._line_combo.setMinimumWidth(100)
        self._line_combo.currentIndexChanged.connect(self._on_line_changed)
        line_row.addWidget(self._line_combo, 1)
        root.addLayout(line_row)

        # TS var combo
        var_row = QtWidgets.QHBoxLayout()
        var_row.addWidget(QtWidgets.QLabel("TS var:"))
        self._ts_var_combo = QtWidgets.QComboBox()
        self._ts_var_combo.currentIndexChanged.connect(self._on_ts_var_changed)
        var_row.addWidget(self._ts_var_combo, 1)
        root.addLayout(var_row)

        # Profile var combo
        pvar_row = QtWidgets.QHBoxLayout()
        pvar_row.addWidget(QtWidgets.QLabel("Prof:"))
        self._prof_var_combo = QtWidgets.QComboBox()
        self._prof_var_combo.currentIndexChanged.connect(self._on_prof_var_changed)
        pvar_row.addWidget(self._prof_var_combo, 1)
        root.addLayout(pvar_row)

        # Profile fill widget
        self._prof_fill_widget = QtWidgets.QWidget()
        prof_fill_row = QtWidgets.QHBoxLayout(self._prof_fill_widget)
        prof_fill_row.setContentsMargins(0, 0, 0, 0)
        self._prof_fill_lbl = QtWidgets.QLabel("Fill by:")
        self._prof_fill_combo = QtWidgets.QComboBox()
        self._prof_fill_combo.currentIndexChanged.connect(self._on_prof_fill_changed)
        prof_fill_row.addWidget(self._prof_fill_lbl)
        prof_fill_row.addWidget(self._prof_fill_combo, 1)
        root.addWidget(self._prof_fill_widget)

        # Profile WSE render widget
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
        root.addWidget(self._prof_wse_render_widget)

        # Profile colormap widget
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
        root.addWidget(self._prof_cmap_widget)

        # Show structures checkbox
        self._show_structures_chk = QtWidgets.QCheckBox("Overlay structures")
        self._show_structures_chk.setChecked(True)
        self._show_structures_chk.toggled.connect(self._on_show_structures_toggled)
        root.addWidget(self._show_structures_chk)

        # Run count label
        self._run_count_lbl = QtWidgets.QLabel("")
        self._run_count_lbl.setStyleSheet("color: gray; font-size: 9px;")
        root.addWidget(self._run_count_lbl)

        # Profile render visibility
        self._sync_profile_render_controls()

        # Animation bar
        self._build_anim_bar(root)

    def _build_anim_bar(self, parent_layout) -> None:
        """Build the animation playback controls bar."""
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

        self._time_lbl = QtWidgets.QLabel(f"T = 0.000 {_TIME_UNIT}")
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
        parent_layout.addLayout(bar)

    def _setup_matplotlib(self) -> None:
        """Create matplotlib figures and canvases for time-series and profile plots."""
        if not self._have_mpl:
            return

        # Time-series tab placeholder (will be embedded by Studio)
        self._ts_tab = QtWidgets.QWidget()
        layout_ts = QtWidgets.QVBoxLayout(self._ts_tab)
        layout_ts.setContentsMargins(0, 0, 0, 0)
        fig_ts = _Figure(figsize=(6, 3.8), constrained_layout=True)
        ax_ts = fig_ts.add_subplot(111)
        canvas_ts = _FigureCanvas(fig_ts)
        canvas_ts.setMinimumHeight(200)
        toolbar_ts = None
        if _NavigationToolbar:
            toolbar_ts = _NavigationToolbar(canvas_ts, self._ts_tab)
            toolbar_ts.setIconSize(QtCore.QSize(16, 16))
            layout_ts.addWidget(toolbar_ts)
        layout_ts.addWidget(canvas_ts, 1)
        self._fig_ts = fig_ts
        self._ax_ts = ax_ts
        self._canvas_ts = canvas_ts

        # Profile tab placeholder
        self._prof_tab = QtWidgets.QWidget()
        layout_prof = QtWidgets.QVBoxLayout(self._prof_tab)
        layout_prof.setContentsMargins(0, 0, 0, 0)
        fig_prof = _Figure(figsize=(6, 3.8), constrained_layout=True)
        ax_prof = fig_prof.add_subplot(111)
        canvas_prof = _FigureCanvas(fig_prof)
        canvas_prof.setMinimumHeight(200)
        toolbar_prof = None
        if _NavigationToolbar:
            toolbar_prof = _NavigationToolbar(canvas_prof, self._prof_tab)
            toolbar_prof.setIconSize(QtCore.QSize(16, 16))
            layout_prof.addWidget(toolbar_prof)
        layout_prof.addWidget(canvas_prof, 1)
        self._fig_prof = fig_prof
        self._ax_prof = ax_prof
        self._canvas_prof = canvas_prof

    # ------------------------------------------------------------------
    # Public: widget accessors for Studio to embed
    # ------------------------------------------------------------------

    @property
    def root_widget(self) -> QtWidgets.QWidget:
        """The root widget for embedding in the dialog."""
        return self._root

    @property
    def ts_plot_widget(self) -> QtWidgets.QWidget:
        """Full time-series plot widget (toolbar + canvas). For embedding."""
        return self._ts_tab

    @property
    def prof_plot_widget(self) -> QtWidgets.QWidget:
        """Full profile plot widget (toolbar + canvas). For embedding."""
        return self._prof_tab

    def set_visible_plot(self, mode: str) -> None:
        """Show/hide internal plot widgets based on view mode."""
        ts = self._ts_tab
        prof = self._prof_tab
        if ts is not None and prof is not None:
            ts.setVisible(mode == "Time-Series")
            prof.setVisible(mode != "Time-Series")

    # ------------------------------------------------------------------
    # Public: refresh
    # ------------------------------------------------------------------

    def refresh_all(self) -> None:
        """Full refresh: rebuild run list, line combo, and plots."""
        self._rebuild_run_list()
        self._refresh_line_combo()
        self._refresh_timeseries()
        self._refresh_profile()

    def rebuild_from_data(self) -> None:
        """Rebuild all widgets from data layer state."""
        if self._data is None:
            return
        self._rebuild_run_list()
        self._refresh_line_combo()
        self._sync_profile_render_controls()
        self._run_count_lbl.setText(f"{len(self._data.get_run_records())} run(s)")

    # ------------------------------------------------------------------
    # Handlers (call data API, update own widgets)
    # ------------------------------------------------------------------

    def _on_refresh(self) -> None:
        """Re-scan GPKG for new runs and rebuild the UI."""
        if self._data is None:
            return
        self._data.discover_runs()
        self._data._rebuild_timestep_union()
        self.rebuild_from_data()

    def _on_add_files(self) -> None:
        """Open file dialog to add results GeoPackage files."""
        if self._data is None:
            return
        file_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self._root,
            "Add SWE2D Results GeoPackage(s)",
            self._data.gpkg_path or "",
            "GeoPackage (*.gpkg)",
        )
        if not file_paths:
            return
        # Add the files first so we can scan for runs
        added_paths, _ = self._data.add_results_files(file_paths)
        if added_paths <= 0:
            return
        # Show run selection dialog
        from swe2d.workbench.dialogs.run_selection_dialog import RunSelectionDialog
        all_records = self._data.get_run_records()
        if not all_records:
            self.rebuild_from_data()
            return
        dlg = RunSelectionDialog(all_records, parent=self._root)
        dlg.exec()
        selected = dlg.selected_keys()
        # Disable runs the user didn't select
        for rec in all_records:
            if rec.key not in selected:
                self._data.toggle_run(rec.key, False)
        self.rebuild_from_data()

    def _on_run_toggle(self, item: QtWidgets.QListWidgetItem) -> None:
        """Toggle a run's visibility when its checkbox is changed."""
        if self._data is None:
            return
        run_key = item.data(Qt.UserRole)
        enabled = item.checkState() == Qt.Checked
        self._data.toggle_run(run_key, enabled)
        self._refresh_line_combo()

    def _on_remove_runs(self) -> None:
        """Remove selected runs from the data layer."""
        if self._data is None:
            return
        selected_keys = {
            item.data(Qt.UserRole)
            for item in self._run_list.selectedItems()
            if item.data(Qt.UserRole)
        }
        if not selected_keys:
            return
        self._data.remove_runs(selected_keys)
        self.rebuild_from_data()

    def _on_show_all(self) -> None:
        """Enable visibility for all runs."""
        if self._data is None:
            return
        self._data.set_all_runs_visible()
        self._rebuild_run_list()
        self._refresh_line_combo()

    def _on_hide_all(self) -> None:
        """Disable visibility for all runs."""
        if self._data is None:
            return
        self._data.set_all_runs_hidden()
        self._rebuild_run_list()
        self._refresh_line_combo()

    def _on_line_changed(self, _index: int) -> None:
        """Handle line combo change — update plots for the new line."""
        if self._data is None:
            return
        lid = self._line_combo.currentData()
        if lid is not None:
            # lid may be (line_id, name) tuple or plain int
            line_id = lid[0] if isinstance(lid, tuple) else lid
            self._data.set_line_id(int(line_id))
            self._refresh_timeseries()
            self._refresh_profile()

    def _on_ts_var_changed(self, _: int) -> None:
        """Handle time-series variable combo change — refresh the plot."""
        self._refresh_timeseries()

    def _on_prof_var_changed(self, _: int) -> None:
        """Handle profile variable combo change — sync controls and refresh."""
        self._sync_profile_render_controls()
        self._refresh_profile()

    def _on_prof_fill_changed(self, _: int) -> None:
        """Handle profile fill option change — refresh the profile plot."""
        self._refresh_profile()

    def _on_show_structures_toggled(self, _: bool) -> None:
        """Handle structures overlay toggle — refresh the profile plot."""
        self._refresh_profile()

    def _on_play_pause(self, checked: bool) -> None:
        """Toggle animation playback."""
        if self._data is None:
            return
        if checked:
            self._data.play()
        else:
            self._data.pause()

    def _on_step_back(self) -> None:
        """Step animation back one frame."""
        if self._data is None:
            return
        self._data.step_backward()

    def _on_step_fwd(self) -> None:
        """Step animation forward one frame."""
        if self._data is None:
            return
        self._data.step_forward()

    def _on_slider_changed(self, value: int) -> None:
        """Handle time slider change — seek to the given frame index."""
        if self._data is None:
            return
        self._data.set_index(int(value))

    def _on_speed_changed(self, index: int) -> None:
        """Handle speed combo change — update animation frame rate."""
        if self._data is None:
            return
        speed = _SPEEDS[max(0, min(index, len(_SPEEDS) - 1))]
        self._data.set_frame_rate(4.0 * speed)

    # ------------------------------------------------------------------
    # Animation sync (called by data layer signals)
    # ------------------------------------------------------------------

    def on_timestep_changed(self, t_sec: float, frame_idx: int) -> None:
        """Handle animation timestep change — update slider, time label, refresh plots."""
        if self._data is None:
            return
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(int(frame_idx))
        self._time_slider.blockSignals(False)
        self._time_lbl.setText(f"T = {t_sec / 3600.0:.3f} {_TIME_UNIT}")
        self._update_ts_vline()
        self._refresh_profile()

    def on_play_state_changed(self, playing: bool) -> None:
        """Handle animation play state change — sync play button."""
        self._play_btn.blockSignals(True)
        self._play_btn.setChecked(bool(playing))
        self._play_btn.setText("\u23f8" if playing else "\u25b6")
        self._play_btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Widget rebuilds
    # ------------------------------------------------------------------

    def _rebuild_run_list(self) -> None:
        """Clear and rebuild the run list from data layer records."""
        if self._data is None:
            return
        self._run_list.blockSignals(True)
        self._run_list.clear()
        for rec in self._data.get_run_records():
            item = QtWidgets.QListWidgetItem(rec.display_label())
            item.setCheckState(Qt.Checked if rec.enabled else Qt.Unchecked)
            item.setData(Qt.UserRole, rec.key)
            item.setData(Qt.UserRole + 1, rec.color)
            item.setToolTip(f"Run: {rec.run_id}\nGPKG: {rec.gpkg_path}")
            self._run_list.addItem(item)
        self._run_list.blockSignals(False)
        self._run_count_lbl.setText(f"{len(self._data.get_run_records())} run(s)")

    def _refresh_line_combo(self) -> None:
        """Rebuild the line combo from data layer line IDs."""
        if self._data is None:
            return
        self._line_combo.blockSignals(True)
        self._line_combo.clear()
        line_ids = self._data.get_line_ids()
        for lid in line_ids:
            # lid may be (line_id, name) tuple or plain int
            display = f"{lid[0]}: {lid[1]}" if isinstance(lid, tuple) else str(lid)
            self._line_combo.addItem(display, lid)
        self._line_combo.blockSignals(False)
        # Restore selection if possible
        current = self._data.line_id
        idx = self._line_combo.findData(current)
        if idx < 0:
            # Try matching just the int part of tuples
            for i in range(self._line_combo.count()):
                d = self._line_combo.itemData(i)
                if isinstance(d, tuple) and d[0] == current:
                    idx = i
                    break
        if idx >= 0:
            self._line_combo.setCurrentIndex(idx)
        elif line_ids:
            self._line_combo.setCurrentIndex(0)

    def _sync_profile_render_controls(self) -> None:
        """Show/hide profile fill, WSE render, and colormap widgets based on current mode."""
        is_wse_bed = str(self._prof_var_combo.currentData() or "wse_bed") == "wse_bed"
        fill_enabled = is_wse_bed and str(self._prof_fill_combo.currentData() or "none") != "none"
        self._prof_fill_widget.setVisible(is_wse_bed)
        self._prof_wse_render_widget.setVisible(is_wse_bed)
        self._prof_cmap_widget.setVisible(fill_enabled)

    def update_slider_range(self) -> None:
        """Update slider range from data layer timesteps."""
        if self._data is None:
            return
        n = self._data.frame_count
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, max(0, n - 1))
        self._time_slider.setValue(0)
        self._time_slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Matplotlib rendering
    # ------------------------------------------------------------------

    def _update_ts_vline(self) -> None:
        """Update the vertical line marker on the time-series plot."""
        if not self._have_mpl or self._ax_ts is None or self._canvas_ts is None:
            return
        from swe2d.workbench.services.results_render_service import update_vline
        self._ts_vline = update_vline(
            self._ax_ts, self._canvas_ts, self._ts_vline,
            self._data.current_time_sec,
        )

    def _refresh_timeseries(self) -> None:
        """Re-render the time-series plot with current data and settings."""
        if self._data is None or not self._have_mpl or self._ax_ts is None:
            return
        from swe2d.workbench.services.results_render_service import render_timeseries
        var_key = str(self._ts_var_combo.currentData() or "flow_cms")
        var_label = self._ts_var_combo.currentText()
        render_timeseries(
            ax=self._ax_ts,
            run_records=self._data.get_enabled_run_records(),
            line_id=self._data.line_id,
            var_key=var_key,
            var_label=var_label,
            current_time_sec=self._data.current_time_sec,
            load_timeseries_fn=self._data.load_timeseries,
            length_unit=_u.length_unit_name(),
        )
        self._ts_vline = None
        self._canvas_ts.draw_idle()

    def _refresh_profile(self) -> None:
        """Re-render the profile plot with current data and settings."""
        if self._data is None or not self._have_mpl or self._ax_prof is None:
            return
        from swe2d.workbench.services.results_render_service import render_profile
        from swe2d.results.queries import (
            find_nearest_timestep,
            load_profile,
            load_structure_flows_at_time,
        )
        from swe2d.results.structure_service import (
            load_bound_layer_name as _load_bound_layer_name_svc,
            load_line_geometry as _load_line_geometry_svc,
            resolve_structure_profile_overlays as _resolve_svc,
        )

        mode = str(self._prof_var_combo.currentData() or "wse_bed")
        fill_key = str(self._prof_fill_combo.currentData() or "none")
        render_mode = str(self._prof_wse_render_combo.currentData() or "clipped")
        cmap_name = str(self._prof_cmap_combo.currentData() or "viridis")
        use_fill_cmap = mode == "wse_bed" and fill_key != "none"

        _, self._prof_fill_cbar = render_profile(
            ax=self._ax_prof,
            fig=self._fig_prof,
            run_records=self._data.get_enabled_run_records(),
            line_id=self._data.line_id,
            t_sec=self._data.current_time_sec,
            mode=mode,
            fill_key=fill_key,
            render_mode=render_mode,
            cmap_name=cmap_name,
            use_fill_cmap=use_fill_cmap,
            show_structures=self._show_structures_chk.isChecked(),
            find_nearest_timestep_fn=find_nearest_timestep,
            load_profile_fn=load_profile,
            load_structure_flows_fn=load_structure_flows_at_time,
            load_bound_layer_name_fn=_load_bound_layer_name_svc,
            load_line_geometry_fn=_load_line_geometry_svc,
            resolve_structure_profile_overlays_fn=_resolve_svc,
            prof_fill_cbar=self._prof_fill_cbar,
            length_unit=_u.length_unit_name(),
        )
        self._canvas_prof.draw_idle()
