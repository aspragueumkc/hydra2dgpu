"""Results controls — 4-page toolbox for the "HYDRA2D Results" dock.

Follows the canonical _build_xxx_page(toolbox) pattern (MVP Rule 8).
Pages:
  1. Results & Overlay  — high-perf overlay field, cmap, opacity, arrows, etc.
  2. Line / Drainage    — line combo, profile controls, output save checkboxes
  3. Runs               — run list, add/refresh/remove, enable/disable
  4. Coupling           — metric/element filter for drainage/structure time series

Emits pyqtSignals for controller wiring (does NOT reach through dialog).
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt.QtCore import pyqtSignal

from swe2d.results.data import SWE2DResultsData


class _SwatchDelegate(QtWidgets.QStyledItemDelegate):
    _SW, _GAP = 12, 3

    def paint(self, painter, option, index):
        """Paint a run list item with a color swatch badge."""
        super().paint(painter, option, index)
        rgb = index.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        if rgb is None:
            return
        r, g, b = rgb
        rect = option.rect
        sr = QtCore.QRect(
            rect.left() + self._GAP,
            rect.top() + (rect.height() - self._SW) // 2,
            self._SW, self._SW,
        )
        painter.save()
        painter.setBrush(QtGui.QColor(r, g, b))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRect(sr)
        painter.restore()


class ResultsToolbox(QtWidgets.QWidget):
    """3-page QToolBox for HYDRA2D Results dock.  QWidget subclass for
    proper Qt parent ownership.  Emits signals for controller wiring."""

    # Signals — connected by the dialog to the controller
    overlay_toggled = pyqtSignal(bool)
    overlay_style_changed = pyqtSignal()
    overlay_export_geotiff = pyqtSignal()
    # Run list signals
    run_selection_changed = pyqtSignal()
    run_refresh_requested = pyqtSignal()
    run_add_requested = pyqtSignal()
    run_remove_requested = pyqtSignal()
    run_show_all = pyqtSignal()
    run_hide_all = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[SWE2DResultsData] = None
        self._toolbox: Optional[QtWidgets.QToolBox] = None
        self._overlay_refresh_callback: Optional[Callable[[], None]] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_data(self, data: SWE2DResultsData) -> None:
        """Bind the data layer and rebuild run list."""
        self._data = data
        self._rebuild_run_list()

    def set_overlay_refresh_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback invoked when overlay display parameters change."""
        self._overlay_refresh_callback = callback

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the 3-page toolbox (Overlay, Output, Runs)."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._toolbox = QtWidgets.QToolBox()
        self._toolbox.setObjectName("results_toolbox")
        self._build_overlay_page(self._toolbox)
        self._build_output_page(self._toolbox)
        self._build_runs_page(self._toolbox)
        self._toolbox.setCurrentIndex(0)
        layout.addWidget(self._toolbox, 1)

    # ------------------------------------------------------------------
    # Page 1: Results & Overlay
    # ------------------------------------------------------------------

    def _build_overlay_page(self, toolbox: QtWidgets.QToolBox) -> None:
        """Build the Results & Overlay page with field, colormap, and arrow controls."""
        page = QtWidgets.QWidget()
        page.setObjectName("results_overlay_page")
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)

        self.field_combo = self._add_combo(layout, "Field:")
        self.field_combo.setToolTip("Result field to render on the map canvas: Depth, Velocity, WSE, etc.")
        self.wse_render_combo = self._add_combo(layout, "WSE render:")
        self.wse_render_combo.setToolTip("Water surface elevation rendering mode: raw cell-centered or smoothed nodal.")
        self.cmap_combo = self._add_combo(layout, "Colormap:")
        self.cmap_combo.setToolTip("Color map used to render the selected field.")
        self.res_combo = self._add_combo(layout, "Resolution:")
        self.res_combo.setToolTip("Render resolution for the high-performance overlay canvas.")

        self.opacity_spin = self._add_spin(layout, "Opacity:", 2, 0.05, 1.0, 0.05, 0.65)
        self.opacity_spin.setToolTip("Overlay opacity: 0.05 (faint) to 1.0 (fully opaque).")
        self.auto_contrast_chk = self._add_chk(layout, "Auto contrast", True)
        self.auto_contrast_chk.setToolTip("Automatically adjust color range to data min/max. Uncheck to set custom range.")
        self.auto_contrast_chk.toggled.connect(self._on_auto_contrast_toggled)
        self.min_depth_spin = self._add_spin(
            layout, "Min depth threshold:", 6, 0.0, 100.0, 0.01, 1.0e-6)
        self.min_depth_spin.setToolTip("Cells with depth below this threshold are treated as dry in the overlay.")
        self.color_min_spin = self._add_spin(
            layout, "Color min:", 6, -1e12, 1e12, 0.01, 0.0)
        self.color_min_spin._no_persist = True
        self.color_min_spin.setToolTip("Minimum value for the color scale (manual mode). Only active when auto contrast is off.")
        self.color_max_spin = self._add_spin(
            layout, "Color max:", 6, -1e12, 1e12, 0.01, 1.0)
        self.color_max_spin._no_persist = True
        self.color_max_spin.setToolTip("Maximum value for the color scale (manual mode). Only active when auto contrast is off.")
        self.color_reset_btn = QtWidgets.QPushButton("↺ Reset")
        self.color_reset_btn.setToolTip("Reset color min/max to the actual data range.")
        self.color_reset_btn.setFixedWidth(70)
        self.color_reset_btn.clicked.connect(self._on_color_reset)
        layout.addRow("", self.color_reset_btn)
        self.lock_canvas_chk = self._add_chk(layout, "Lock canvas extent", True)
        self.lock_canvas_chk.setToolTip("Lock the map canvas extent to the overlay's bounding box.")
        self.visible_only_chk = self._add_chk(layout, "Visible cells only", True)
        self.visible_only_chk.setToolTip("Only render cells within the current map viewport for performance.")
        self.arrows_chk = self._add_chk(layout, "Show velocity arrows", True)
        self.arrows_chk.setToolTip("Display velocity direction arrows on the overlay.")

        self.arrow_density_spin = self._add_spin(
            layout, "Arrow spacing (px):", 0, 8, 80, 2, 28)
        self.arrow_density_spin.setToolTip("Spacing between velocity arrows in pixels. Lower = denser arrows.")
        self.arrow_length_spin = self._add_spin(
            layout, "Arrow length scale:", 2, 0.2, 3.0, 0.1, 1.0)
        self.arrow_length_spin.setToolTip("Velocity arrow length multiplier.")
        self.arrow_head_length_spin = self._add_spin(
            layout, "Arrow head length:", 2, 0.2, 3.0, 0.1, 1.0)
        self.arrow_head_length_spin.setToolTip("Velocity arrow head length.")
        self.arrow_head_width_spin = self._add_spin(
            layout, "Arrow head width:", 2, 0.2, 3.0, 0.1, 1.0)
        self.arrow_head_width_spin.setToolTip("Velocity arrow head width.")

        self.streamlines_chk = self._add_chk(layout, "Show streamlines", False)
        self.streamlines_chk.setToolTip("Display flow streamlines on the overlay.")
        self.streamline_backend_combo = self._add_combo(
            layout, "Streamline backend:")
        self.streamline_backend_combo.setToolTip("Streamline computation engine: Auto, CUDA (GPU), or CPU.")
        self.streamline_seed_spin = self._add_spin(
            layout, "Streamline seeds:", 0, 8, 256, 8, 48)
        self.streamline_seed_spin.setToolTip("Number of streamline seed points for flow tracing.")
        self.streamline_steps_spin = self._add_spin(
            layout, "Streamline steps:", 0, 4, 120, 2, 24)
        self.streamline_steps_spin.setToolTip("Maximum integration steps per streamline.")

        self.overlay_enabled_chk = QtWidgets.QCheckBox(
            "Enable high-performance overlay")
        self.overlay_enabled_chk.setToolTip(
            "Toggle the GPU-accelerated high-performance overlay on/off."
        )
        self.overlay_enabled_chk.toggled.connect(self.overlay_toggled.emit)
        layout.addRow(self.overlay_enabled_chk)

        self.export_btn = QtWidgets.QPushButton("Export Overlay to GeoTIFF...")
        self.export_btn.setToolTip("Export the current overlay view as a GeoTIFF raster.")
        self.export_btn.clicked.connect(self.overlay_export_geotiff.emit)
        layout.addRow(self.export_btn)

        self.export_res_spin = QtWidgets.QDoubleSpinBox()
        self.export_res_spin.setToolTip("Output pixel size in map units for GeoTIFF export.")
        self.export_res_spin.setValue(10.0)
        layout.addRow("GeoTIFF pixel size (map units):", self.export_res_spin)

        self._populate_overlay_combos()
        toolbox.addItem(page, "Results & Overlay")

    _SIGNAL_SELF = "overlay_style_changed"

    def _add_combo(self, layout, label):
        """Add a labeled combo that emits overlay_style_changed on change."""
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(self.overlay_style_changed.emit)
        layout.addRow(label, combo)
        return combo

    def _add_chk(self, layout, text, checked):
        """Add a checkbox that emits overlay_style_changed on toggle."""
        chk = QtWidgets.QCheckBox(text)
        chk.setChecked(checked)
        chk.toggled.connect(self.overlay_style_changed.emit)
        layout.addRow(chk)
        return chk

    def _add_spin(self, layout, label, decimals, lo, hi, step, val):
        """Add a labeled spin box that emits overlay_style_changed on value change."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals); spin.setRange(lo, hi)
        spin.setSingleStep(step); spin.setValue(val)
        spin.valueChanged.connect(self.overlay_style_changed.emit)
        layout.addRow(label, spin)
        return spin

    def _on_auto_contrast_toggled(self, checked: bool) -> None:
        """Enable/disable manual color min/max spinboxes and trigger overlay refresh."""
        for s in (getattr(self, "color_min_spin", None), getattr(self, "color_max_spin", None)):
            if s is not None:
                s.setEnabled(not bool(checked))
        if self._overlay_refresh_callback is not None:
            try:
                self._overlay_refresh_callback()
            except Exception:
                pass

    def _on_color_reset(self) -> None:
        """Reset color min/max to auto-contrast based on actual data range."""
        self.auto_contrast_chk.setChecked(True)

    def _on_field_changed(self, index: int) -> None:
        """Reset color auto-contrast when user switches the rendered field."""
        _ = index
        self.auto_contrast_chk.setChecked(True)

    def _populate_overlay_combos(self) -> None:
        """Fill field, colormap, WSE render, resolution, and streamline combos with defaults."""
        for combo, items in [
            (self.field_combo, [
                ("Depth", "depth"), ("Velocity", "speed"),
                ("Water Surface", "wse"), ("Froude Number", "froude"),
                ("Courant Number", "courant"), ("Shear Stress", "shear_stress"),
            ]),
            (self.cmap_combo, [
                ("Turbo", "turbo"), ("Viridis", "viridis"),
                ("Plasma", "plasma"), ("Magma", "magma"),
                ("Inferno", "inferno"), ("Cividis", "cividis"),
                ("Terrain", "terrain"), ("Ocean", "ocean"), ("Gray", "gray"),
            ]),
        ]:
            combo.clear()
            for label, key in items:
                combo.addItem(label, key)

        self.wse_render_combo.clear()
        self.wse_render_combo.addItem("Raw (cell-centered)", "cell")
        self.wse_render_combo.addItem("Smoothed (nodal eta)", "nodal")
        self.wse_render_combo.setCurrentIndex(0)

        # When user changes the field, reset color override to auto-range
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)

        self.res_combo.clear()
        for label, key in [
            ("640 x 360", (640, 360)), ("960 x 540", (960, 540)),
            ("1280 x 720", (1280, 720)), ("1920 x 1080", (1920, 1080)),
            ("2560 x 1440", (2560, 1440)), ("3200 x 1800", (3200, 1800)),
            ("3840 x 2160 (4K)", (3840, 2160)),
        ]:
            self.res_combo.addItem(label, key)
        self.res_combo.setCurrentIndex(2)

        self.streamline_backend_combo.clear()
        self.streamline_backend_combo.addItem("Auto (prefer compiled)", "auto")
        self.streamline_backend_combo.addItem("CUDA", "cuda")
        self.streamline_backend_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Page 2: Output configuration (save toggles)
    # ------------------------------------------------------------------

    def _build_output_page(self, toolbox: QtWidgets.QToolBox) -> None:
        """Build the Output page with save-to-GPKG checkboxes."""
        page = QtWidgets.QWidget()
        page.setObjectName("results_output_page")
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)

        self.extended_outputs_chk = QtWidgets.QCheckBox(
            "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)")
        self.extended_outputs_chk.setToolTip(
            "Include additional output fields beyond depth and velocity. "
            "Increases result file size."
        )
        self.extended_outputs_chk.setChecked(True)
        layout.addRow(self.extended_outputs_chk)

        self.save_mesh_chk = QtWidgets.QCheckBox("Save mesh results to GPKG")
        self.save_mesh_chk.setToolTip(
            "Save 2D mesh simulation results (depth, velocity, WSE) to the GeoPackage."
        )
        self.save_mesh_chk.setChecked(True)
        layout.addRow(self.save_mesh_chk)

        self.save_line_chk = QtWidgets.QCheckBox("Save line results to GPKG")
        self.save_line_chk.setToolTip(
            "Save sample line (cross-section) results to the GeoPackage."
        )
        self.save_line_chk.setChecked(True)
        layout.addRow(self.save_line_chk)

        self.save_coupling_chk = QtWidgets.QCheckBox("Save coupling results to GPKG")
        self.save_coupling_chk.setToolTip(
            "Save drainage/structure coupling time series results to the GeoPackage."
        )
        self.save_coupling_chk.setChecked(True)
        layout.addRow(self.save_coupling_chk)

        self.save_max_only_chk = QtWidgets.QCheckBox("Save max results only (skip interval snapshots)")
        self.save_max_only_chk.setToolTip(
            "Only save maximum-value results per cell. "
            "Skips interval snapshots to reduce file size."
        )
        self.save_max_only_chk.setChecked(False)
        layout.addRow(self.save_max_only_chk)

        self.save_log_chk = QtWidgets.QCheckBox("Save run log to GPKG")
        self.save_log_chk.setToolTip(
            "Save the solver run log (diagnostics, timesteps, errors) to the GeoPackage."
        )
        self.save_log_chk.setChecked(True)
        layout.addRow(self.save_log_chk)

        toolbox.addItem(page, "Output")

    # ------------------------------------------------------------------
    # Page 3: Runs
    # ------------------------------------------------------------------

    def _build_runs_page(self, toolbox: QtWidgets.QToolBox) -> None:
        """Build the Runs page with run list, add/refresh, and enable/disable buttons."""
        page = QtWidgets.QWidget()
        page.setObjectName("results_runs_page")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        top = QtWidgets.QHBoxLayout()
        self.gpkg_lbl = QtWidgets.QLabel()
        self.gpkg_lbl.setStyleSheet("color: gray; font-size: 9px;")
        self.gpkg_lbl.setMaximumWidth(320)
        self.gpkg_lbl.setToolTip("Currently loaded GeoPackage path.")
        self.refresh_btn = QtWidgets.QPushButton("\u21ba")
        self.refresh_btn.setFixedSize(22, 22)
        self.refresh_btn.setToolTip("Re-scan GPKG for new runs")
        self.refresh_btn.clicked.connect(self.run_refresh_requested.emit)
        self.add_btn = QtWidgets.QPushButton("+")
        self.add_btn.setFixedSize(22, 22)
        self.add_btn.setToolTip("Add results from GeoPackages")
        self.add_btn.clicked.connect(self.run_add_requested.emit)
        top.addWidget(self.gpkg_lbl, 1)
        top.addWidget(self.add_btn)
        top.addWidget(self.refresh_btn)
        layout.addLayout(top)

        layout.addWidget(QtWidgets.QLabel("<b>Runs</b>"))

        self.run_list = QtWidgets.QListWidget()
        self.run_list.setAlternatingRowColors(True)
        self.run_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.run_list.setItemDelegate(_SwatchDelegate(self.run_list))
        layout.addWidget(self.run_list, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(2)
        self.remove_btn = QtWidgets.QPushButton("\u2212 Remove")
        self.remove_btn.setFixedHeight(20)
        self.remove_btn.setToolTip("Remove selected runs from the list.")
        self.remove_btn.clicked.connect(self.run_remove_requested.emit)
        self.show_all_btn = QtWidgets.QPushButton("\u2713 All")
        self.show_all_btn.setFixedHeight(20)
        self.show_all_btn.setToolTip("Show (enable) all runs in the overlay.")
        self.show_all_btn.clicked.connect(self.run_show_all.emit)
        self.hide_all_btn = QtWidgets.QPushButton("\u25a1 None")
        self.hide_all_btn.setFixedHeight(20)
        self.hide_all_btn.setToolTip("Hide (disable) all runs in the overlay.")
        self.hide_all_btn.clicked.connect(self.run_hide_all.emit)
        btn_row.addWidget(self.remove_btn, 2)
        btn_row.addWidget(self.show_all_btn, 1)
        btn_row.addWidget(self.hide_all_btn, 1)
        layout.addLayout(btn_row)

        self.run_count_lbl = QtWidgets.QLabel("")
        self.run_count_lbl.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(self.run_count_lbl)

        toolbox.addItem(page, "Runs")

    # ------------------------------------------------------------------
    # Run list management
    # ------------------------------------------------------------------

    def _rebuild_run_list(self) -> None:
        """Clear and rebuild the run list from data layer records."""
        if self._data is None:
            return
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for rec in self._data.get_run_records():
            item = QtWidgets.QListWidgetItem(rec.display_label())
            item.setCheckState(
                QtCore.Qt.CheckState.Checked if rec.enabled
                else QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, rec.key)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, rec.color)
            item.setToolTip(f"Run: {rec.run_id}\nGPKG: {rec.gpkg_path}")
            self.run_list.addItem(item)
        self.run_list.blockSignals(False)
        self.run_list.itemChanged.connect(self._on_run_item_changed)
        self._update_run_count()

    def _on_run_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handle a run checkbox change — toggle visibility in data layer."""
        if self._data is None:
            return
        run_key = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        enabled = item.checkState() == QtCore.Qt.CheckState.Checked
        self._data.toggle_run(run_key, enabled)
        self.run_selection_changed.emit()

    def _update_run_count(self) -> None:
        """Update the run count label and GPKG path label from data."""
        if self._data is None:
            self.run_count_lbl.setText("")
            return
        enabled = len(self._data.get_enabled_run_records())
        total = len(self._data.get_run_records())
        self.run_count_lbl.setText(f"{enabled} / {total} runs enabled")
        if self._data.gpkg_path:
            self.gpkg_lbl.setText(f"GPKG: {self._data.gpkg_path}")

    @property
    def toolbox(self) -> QtWidgets.QToolBox:
        """The internal QToolBox widget."""
        return self._toolbox
