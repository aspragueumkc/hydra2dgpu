"""swe2d/workbench/dialogs/profile_options_dialog.py

SWMM-style 5-tab plot options dialog for the network profile viewer.
Stores options in a ProfileOptions dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from qgis.PyQt import QtCore, QtGui, QtWidgets


@dataclass
class ProfileOptions:
    """All user-configurable plot options.

    Axis labels are unit-aware: the value is a *template* — the plot
    widget substitutes the active length unit (e.g. ``"Distance"``
    becomes ``"Distance (m)"`` or ``"Distance (ft)"`` based on
    ``swe2d.units.length_unit_name()``).  Pass the empty string to use
    the default label, or a fully-spelled label like ``"Distance (m)"``
    to override the unit substitution.
    """
    water_color: str = "#3366CC"
    conduit_color: str = "#5A5A5A"
    invert_color: str = "#2A2A2A"
    crown_color: str = "#888888"
    ground_color: str = "#A0763D"
    ground_line_visible: bool = False
    conduits_only: bool = False
    thick_lines: bool = False
    x_label: str = "Distance"
    y_label: str = "Elevation"
    auto_scale: bool = True
    y_min: float = 0.0
    y_max: float = 10.0
    y_inc: float = 1.0
    node_labels_on_top_axis: bool = False
    node_labels_on_plot: bool = True
    arrow_length_px: int = 30
    font_size_pt: int = 8


class ProfileOptionsDialog(QtWidgets.QDialog):
    """5-tab options dialog: Colors / Styles / Axes / Vertical Scale / Node Labels."""

    def __init__(self, options: ProfileOptions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Network Profile Options")
        self.resize(560, 480)
        self._options = ProfileOptions(**asdict(options))
        self._build_ui()
        self._populate()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        self._tabs = QtWidgets.QTabWidget()
        root.addWidget(self._tabs)

        self._colors_tab = self._build_colors_tab()
        self._styles_tab = self._build_styles_tab()
        self._axes_tab = self._build_axes_tab()
        self._scale_tab = self._build_scale_tab()
        self._labels_tab = self._build_labels_tab()
        self._tabs.addTab(self._colors_tab, "Colors")
        self._tabs.addTab(self._styles_tab, "Styles")
        self._tabs.addTab(self._axes_tab, "Axes")
        self._tabs.addTab(self._scale_tab, "Vertical Scale")
        self._tabs.addTab(self._labels_tab, "Node Labels")

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self._restore_defaults)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root.addWidget(button_box)

    def _build_colors_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setObjectName("colors_page")
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        self._color_widgets = {}
        for label, key in [
            ("Water (HGL)", "water_color"),
            ("Conduit", "conduit_color"),
            ("Invert line", "invert_color"),
            ("Crown line", "crown_color"),
            ("Ground/rim line", "ground_color"),
        ]:
            btn = QtWidgets.QPushButton()
            btn.setProperty("color_key", key)
            btn.clicked.connect(lambda checked=False, k=key: self._pick_color(k))
            self._color_widgets[key] = btn
            layout.addRow(label, btn)
        return page

    def _build_styles_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setObjectName("styles_page")
        layout = QtWidgets.QVBoxLayout(page)
        self._ground_visible_chk = QtWidgets.QCheckBox("Display ground/rim line")
        self._conduits_only_chk = QtWidgets.QCheckBox("Display conduits only")
        self._thick_lines_chk = QtWidgets.QCheckBox("Use thick lines")
        for w in (self._ground_visible_chk, self._conduits_only_chk, self._thick_lines_chk):
            layout.addWidget(w)
        layout.addStretch(1)
        return page

    def _build_axes_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setObjectName("axes_page")
        layout = QtWidgets.QFormLayout(page)
        self._x_label_edit = QtWidgets.QLineEdit()
        self._y_label_edit = QtWidgets.QLineEdit()
        self._font_size_spin = QtWidgets.QSpinBox()
        self._font_size_spin.setRange(6, 24)
        layout.addRow("X axis label:", self._x_label_edit)
        layout.addRow("Y axis label:", self._y_label_edit)
        layout.addRow("Font size (pt):", self._font_size_spin)
        return page

    def _build_scale_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setObjectName("scale_page")
        layout = QtWidgets.QVBoxLayout(page)
        self._auto_scale_chk = QtWidgets.QCheckBox("Auto-scale Y axis")
        self._auto_scale_chk.toggled.connect(self._on_auto_scale_toggle)
        layout.addWidget(self._auto_scale_chk)
        manual = QtWidgets.QGroupBox("Manual Y range")
        manual_layout = QtWidgets.QFormLayout(manual)
        self._y_min_spin = QtWidgets.QDoubleSpinBox()
        self._y_min_spin.setRange(-1000, 10000)
        self._y_max_spin = QtWidgets.QDoubleSpinBox()
        self._y_max_spin.setRange(-1000, 10000)
        self._y_inc_spin = QtWidgets.QDoubleSpinBox()
        self._y_inc_spin.setRange(0.01, 1000)
        manual_layout.addRow("Y min:", self._y_min_spin)
        manual_layout.addRow("Y max:", self._y_max_spin)
        manual_layout.addRow("Y increment:", self._y_inc_spin)
        layout.addWidget(manual)
        return page

    def _build_labels_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setObjectName("labels_page")
        layout = QtWidgets.QFormLayout(page)
        self._labels_top_chk = QtWidgets.QCheckBox("Display on top axis")
        self._labels_plot_chk = QtWidgets.QCheckBox("Display on plot")
        self._arrow_spin = QtWidgets.QSpinBox()
        self._arrow_spin.setRange(0, 100)
        layout.addRow(self._labels_top_chk)
        layout.addRow(self._labels_plot_chk)
        layout.addRow("Arrow length (px):", self._arrow_spin)
        return page

    def _populate(self):
        for key, btn in self._color_widgets.items():
            c = getattr(self._options, key)
            btn.setStyleSheet(f"background-color: {c};")
            btn.setText(c)
        self._ground_visible_chk.setChecked(self._options.ground_line_visible)
        self._conduits_only_chk.setChecked(self._options.conduits_only)
        self._thick_lines_chk.setChecked(self._options.thick_lines)
        self._x_label_edit.setText(self._options.x_label)
        self._y_label_edit.setText(self._options.y_label)
        self._font_size_spin.setValue(self._options.font_size_pt)
        self._auto_scale_chk.setChecked(self._options.auto_scale)
        self._y_min_spin.setValue(self._options.y_min)
        self._y_max_spin.setValue(self._options.y_max)
        self._y_inc_spin.setValue(self._options.y_inc)
        self._on_auto_scale_toggle(self._options.auto_scale)
        self._labels_top_chk.setChecked(self._options.node_labels_on_top_axis)
        self._labels_plot_chk.setChecked(self._options.node_labels_on_plot)
        self._arrow_spin.setValue(self._options.arrow_length_px)

    def _on_auto_scale_toggle(self, checked: bool):
        for w in (self._y_min_spin, self._y_max_spin, self._y_inc_spin):
            w.setEnabled(not checked)

    def _pick_color(self, key: str):
        current = getattr(self._options, key)
        chosen = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(current), self, "Choose Color"
        )
        if chosen.isValid():
            new = chosen.name()
            setattr(self._options, key, new)
            self._color_widgets[key].setStyleSheet(f"background-color: {new};")
            self._color_widgets[key].setText(new)

    def _restore_defaults(self):
        defaults = ProfileOptions()
        self._options = defaults
        self._populate()

    def get_options(self) -> ProfileOptions:
        # Update from widgets first
        self._options.ground_line_visible = self._ground_visible_chk.isChecked()
        self._options.conduits_only = self._conduits_only_chk.isChecked()
        self._options.thick_lines = self._thick_lines_chk.isChecked()
        self._options.x_label = self._x_label_edit.text()
        self._options.y_label = self._y_label_edit.text()
        self._options.font_size_pt = self._font_size_spin.value()
        self._options.auto_scale = self._auto_scale_chk.isChecked()
        self._options.y_min = self._y_min_spin.value()
        self._options.y_max = self._y_max_spin.value()
        self._options.y_inc = self._y_inc_spin.value()
        self._options.node_labels_on_top_axis = self._labels_top_chk.isChecked()
        self._options.node_labels_on_plot = self._labels_plot_chk.isChecked()
        self._options.arrow_length_px = self._arrow_spin.value()
        return ProfileOptions(**asdict(self._options))
