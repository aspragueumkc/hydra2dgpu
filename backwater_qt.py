#!/usr/bin/env python3
"""backwater_qt.py

A Qt-based GUI wrapper for the backwater solver that reuses the computation
functions in `backwater2.py`.

This provides a minimal, modern GUI with: file open, New Model, Run, textual
results, and a matplotlib plot area.

This module is intended to run only inside QGIS as part of the plugin.
"""

import os
import sys
import json
import math
import importlib.util

from qgis.PyQt import QtWidgets, QtCore, QtGui
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox

# Import solver functions from the plugin-local `backwater2` module.
try:
    try:
        from . import backwater2 as _bwmod  # type: ignore
    except Exception:
        import backwater2 as _bwmod  # type: ignore

    load_input = _bwmod.load_input
    load_from_geopackage = getattr(_bwmod, 'load_from_geopackage', None)
    save_to_geopackage = getattr(_bwmod, 'save_to_geopackage', None)
    load_results_from_geopackage = getattr(_bwmod, 'load_results_from_geopackage', None)
    save_results_to_geopackage = getattr(_bwmod, 'save_results_to_geopackage', None)
    run_backwater = _bwmod.run_backwater
    ModelInput = _bwmod.ModelInput
    CrossSection = _bwmod.CrossSection
    HAVE_MPL = bool(getattr(_bwmod, 'HAVE_MPL', False))
except Exception:
    load_input = load_from_geopackage = save_to_geopackage = load_results_from_geopackage = save_results_to_geopackage = run_backwater = ModelInput = CrossSection = None
    HAVE_MPL = False

try:
    try:
        from . import ui_adapter as _ui_adapter  # type: ignore
    except Exception:
        import ui_adapter as _ui_adapter  # type: ignore
    ui_adapter = _ui_adapter
except Exception:
    ui_adapter = None


def _load_set_z_from_raster_expr_func():
    def _pick_callable(mod):
        for name in ('set_z_from_raster_expr_py', '_set_z_from_raster_impl', 'set_z_from_raster_expr'):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        return None

    try:
        from .expressions import vertices_z_from_raster as _mod  # type: ignore
        _fn = _pick_callable(_mod)
        if _fn is not None:
            return _fn
    except Exception:
        pass
    try:
        _expr_path = os.path.join(os.path.dirname(__file__), 'expressions', 'vertices_z_from_raster.py')
        _spec = importlib.util.spec_from_file_location('qgis_backwater_plugin.expressions.vertices_z_from_raster', _expr_path)
        if _spec is None or _spec.loader is None:
            return None
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _pick_callable(_mod)
    except Exception:
        return None


_SET_Z_FROM_RASTER_EXPR = _load_set_z_from_raster_expr_func()


# Small UI adapter wrappers: prefer plugin ui_adapter, fallback to Qt dialogs
def ui_info(parent, title, msg):
    if ui_adapter is not None:
        ui_adapter.info(msg, title, parent)
    else:
        QMessageBox.information(parent, title or 'Info', msg)


def ui_warning(parent, title, msg):
    if ui_adapter is not None:
        ui_adapter.warning(msg, title, parent)
    else:
        QMessageBox.warning(parent, title or 'Warning', msg)


def ui_critical(parent, title, msg):
    if ui_adapter is not None:
        ui_adapter.critical(msg, title, parent)
    else:
        QMessageBox.critical(parent, title or 'Error', msg)


def ui_get_open_filename(parent, caption, filter):
    if ui_adapter is not None:
        return ui_adapter.get_open_filename(parent, caption, filter)
    return QFileDialog.getOpenFileName(parent, caption, '', filter)


def ui_get_save_filename(parent, caption, filter):
    if ui_adapter is not None:
        return ui_adapter.get_save_filename(parent, caption, filter)
    return QFileDialog.getSaveFileName(parent, caption, '', filter)


class CanvasHolder(QtWidgets.QWidget):
    """A widget that hosts a FigureCanvas and implements Ctrl+wheel zoom by
    scaling the canvas widget size. Without Ctrl the wheel scrolls the
    containing QScrollArea normally.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = None
        self._base_size = None
        self._scale = 1.0
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(0,0,0,0)

    def set_canvas(self, canvas):
        # attach canvas and remember its base size for scaling
        self._canvas = canvas
        self.layout().addWidget(canvas)
        try:
            self._base_size = canvas.sizeHint()
        except Exception:
            self._base_size = canvas.size()

    def wheelEvent(self, event):
        # Ctrl+wheel => zoom canvas; otherwise default behavior (scroll)
        try:
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier and self._canvas is not None:
                delta = event.angleDelta().y()
                factor = 1.1 if delta > 0 else (1.0 / 1.1)
                self._scale *= factor
                if self._base_size is not None:
                    w = max(200, int(self._base_size.width() * self._scale))
                    h = max(200, int(self._base_size.height() * self._scale))
                    self._canvas.setFixedSize(w, h)
                event.accept()
                return
        except Exception:
            pass
        super().wheelEvent(event)


class CrossSectionPreview(QtWidgets.QWidget):
    """Lightweight cross-section preview that draws stations/elevations
    directly using QPainter so it doesn't depend on matplotlib.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._geom = []
        self._title = ''

    def set_geometry(self, geom):
        # geom: list of (station, elevation)
        try:
            self._geom = sorted([(float(s), float(z)) for s, z in geom], key=lambda p: p[0])
        except Exception:
            self._geom = []
        self.update()

    def set_title(self, t: str):
        self._title = str(t)
        self.update()

    def clear(self):
        self._geom = []
        self._title = ''
        self.update()

    def sizeHint(self):
        return QtCore.QSize(600, 200)

    def paintEvent(self, event):
        try:
            painter = QtGui.QPainter(self)
            rect = self.contentsRect()
            painter.fillRect(rect, self.palette().window())
            if not self._geom:
                painter.setPen(QtGui.QColor('gray'))
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, 'No geometry')
                painter.end()
                return

            # compute bounds
            sx = [p[0] for p in self._geom]
            sz = [p[1] for p in self._geom]
            minx, maxx = min(sx), max(sx)
            miny, maxy = min(sz), max(sz)
            if maxx - minx == 0:
                maxx = minx + 1.0
            if maxy - miny == 0:
                maxy = miny + 1.0

            # margins
            margin = 10
            left = rect.left() + 60
            right = rect.right() - margin
            top = rect.top() + margin + 20
            bottom = rect.bottom() - 30

            w = right - left
            h = bottom - top
            if w <= 0 or h <= 0:
                painter.end()
                return

            def tx(x):
                return left + (x - minx) / (maxx - minx) * w
            def ty(y):
                return bottom - (y - miny) / (maxy - miny) * h

            # build polygons for fill and polyline
            poly = QtGui.QPolygonF()
            for x, y in self._geom:
                poly.append(QtCore.QPointF(tx(x), ty(y)))

            # fill polygon down to baseline
            fill_poly = QtGui.QPolygonF(poly)
            fill_poly.append(QtCore.QPointF(tx(self._geom[-1][0]), ty(miny)))
            fill_poly.append(QtCore.QPointF(tx(self._geom[0][0]), ty(miny)))

            painter.setBrush(QtGui.QBrush(QtGui.QColor('#efefef')))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawPolygon(fill_poly)

            # draw polyline
            painter.setPen(QtGui.QPen(QtGui.QColor('#000000'), 1))
            painter.drawPolyline(poly)
            # draw points
            for pt in poly:
                painter.drawEllipse(pt, 2, 2)

            # draw labels
            painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.black))
            painter.drawText(rect.left()+2, rect.top()+12, self._title)
            painter.drawText(left, rect.bottom()-12, f"{minx:.2f}")
            painter.drawText(right-40, rect.bottom()-12, f"{maxx:.2f}")
            painter.drawText(rect.left()+2, bottom, f"{miny:.2f}")
            painter.drawText(rect.left()+2, top+10, f"{maxy:.2f}")
            painter.end()
        except Exception:
            try:
                painter.end()
            except Exception:
                pass


class _ReattachDockWidget(QtWidgets.QDockWidget):
    """Dock widget that calls back when user closes it so content can be reattached."""

    def __init__(self, title, on_close, parent=None):
        super().__init__(title, parent)
        self._on_close = on_close

    def closeEvent(self, event):
        try:
            if callable(self._on_close):
                self._on_close(self)
        except Exception:
            pass
        event.accept()

class BackwaterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Backwater — Qt GUI')
        self.resize(1100, 700)
        # GeoPackage form-first mode keeps all model edits in native QGIS forms.
        self.form_only_mode = True

        # Use splitters so user can resize sections; left pane = controls, right pane = plots
        main_layout = QtWidgets.QVBoxLayout(self)

        # Dock-level menu bar so the plugin has a native Backwater menu in QGIS dock mode.
        self.menu_bar = QtWidgets.QMenuBar(self)
        main_layout.setMenuBar(self.menu_bar)
        self.backwater_menu = self.menu_bar.addMenu('Backwater')

        # Top control row
        controls_row = QtWidgets.QHBoxLayout()
        self.input_path = QtWidgets.QLineEdit()
        self.input_path.setReadOnly(True)
        self.input_path.setPlaceholderText('No model GeoPackage loaded')
        browse_btn = QtWidgets.QPushButton('Select Model GeoPackage...')
        browse_btn.clicked.connect(self.on_menu_open_model)
        run_btn = QtWidgets.QPushButton('Run')
        run_btn.clicked.connect(self.on_run)

        controls_row.addWidget(QtWidgets.QLabel('Input GeoPackage:'))
        controls_row.addWidget(self.input_path)
        controls_row.addWidget(browse_btn)
        controls_row.addWidget(run_btn)
        main_layout.addLayout(controls_row)

        # Splitter between left (controls) and right (plots)
        self.horiz_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(self.horiz_split, stretch=1)

        self.left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)

        # Left-side tabs: Geometry | Boundary | Results
        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.currentChanged.connect(self.on_left_tab_changed)
        left_layout.addWidget(self.left_tabs)

        # Geometry tab (section properties, geometry editor controls)
        self.geom_tab = QtWidgets.QWidget()
        self.geom_tab_layout = QtWidgets.QVBoxLayout(self.geom_tab)
        self.left_tabs.addTab(self.geom_tab, 'Section Properties')

        # Boundary tab (downstream BC, flow overrides)
        self.boundary_tab = QtWidgets.QWidget()
        self.boundary_tab_layout = QtWidgets.QVBoxLayout(self.boundary_tab)
        self.left_tabs.addTab(self.boundary_tab, 'Boundary')

        # (Results tab will be populated with the tabular results later)

        # BC and flow go into the Boundary tab
        self.ds_bc = QtWidgets.QComboBox()
        self.ds_bc.addItems(['known_wse', 'normal_depth'])
        self.ds_val = QtWidgets.QLineEdit('0.0')
        self.flow_edit = QtWidgets.QLineEdit('500.0')
        self.boundary_tab_layout.addWidget(QtWidgets.QLabel('DS BC:'))
        self.boundary_tab_layout.addWidget(self.ds_bc)
        self.boundary_tab_layout.addWidget(QtWidgets.QLabel('DS value (WSE or S0):'))
        self.boundary_tab_layout.addWidget(self.ds_val)
        self.boundary_tab_layout.addWidget(QtWidgets.QLabel('Flow (cfs):'))
        self.boundary_tab_layout.addWidget(self.flow_edit)
        self.ds_bc.currentTextChanged.connect(lambda _value: self._mark_gpkg_dirty())
        self.ds_val.textChanged.connect(lambda _value: self._mark_gpkg_dirty())
        self.flow_edit.textChanged.connect(lambda _value: self._mark_gpkg_dirty())
        # Alpha and Sf method selectors for the GUI
        self.alpha_combo = QtWidgets.QComboBox()
        self.alpha_combo.addItems(['conveyance', 'area'])
        self.sf_combo = QtWidgets.QComboBox()
        self.sf_combo.addItems(['combined', 'avg'])
        # Solver implementation selector
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItems(['py', 'scipy'])
        self.boundary_tab_layout.addWidget(QtWidgets.QLabel('Alpha method:'))
        self.boundary_tab_layout.addWidget(self.alpha_combo)
        self.boundary_tab_layout.addWidget(QtWidgets.QLabel('Sf method:'))
        self.boundary_tab_layout.addWidget(self.sf_combo)
        self.boundary_tab_layout.addWidget(QtWidgets.QLabel('Solver:'))
        self.boundary_tab_layout.addWidget(self.solver_combo)

        # results widgets (tabular results will be added to the left tabs)
        # results_status_label will be created when the left Results tab is built

        self.horiz_split.addWidget(self.left_widget)

        # Right side: vertical split (top = plots, bottom = tabular IO)
        self.right_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top: plots container
        self.plots_container = QtWidgets.QWidget()
        self.plots_container_layout = QtWidgets.QVBoxLayout(self.plots_container)
        # Use tabs for top-right: Results Plot (scrollable subplots),
        # Cross-section preview, and a one-plot-at-a-time scroller.
        self.plots_tabs = QtWidgets.QTabWidget()
        # Results plot page (will host matplotlib canvas and toolbar)
        self.plot_page = QtWidgets.QWidget()
        self.plot_page_layout = QtWidgets.QVBoxLayout(self.plot_page)
        # Cross-section page for geometry editing preview
        self.cross_section_page = QtWidgets.QWidget()
        self.cross_section_layout = QtWidgets.QVBoxLayout(self.cross_section_page)
        # Plot scroller page (profile + one section plot at a time)
        self.plot_scroller_page = QtWidgets.QWidget()
        self.plot_scroller_layout = QtWidgets.QVBoxLayout(self.plot_scroller_page)
        scroller_controls = QtWidgets.QHBoxLayout()
        self.scroller_prev_btn = QtWidgets.QPushButton('Prev')
        self.scroller_next_btn = QtWidgets.QPushButton('Next')
        self.scroller_combo = QtWidgets.QComboBox()
        self.scroller_status = QtWidgets.QLabel('No plots available')
        self.scroller_prev_btn.clicked.connect(lambda: self._scroll_plot_step(-1))
        self.scroller_next_btn.clicked.connect(lambda: self._scroll_plot_step(1))
        self.scroller_combo.currentIndexChanged.connect(self._refresh_scroller_plot)
        scroller_controls.addWidget(self.scroller_prev_btn)
        scroller_controls.addWidget(self.scroller_next_btn)
        scroller_controls.addWidget(QtWidgets.QLabel('Plot:'))
        scroller_controls.addWidget(self.scroller_combo, stretch=1)
        scroller_controls.addWidget(self.scroller_status)
        self.plot_scroller_layout.addLayout(scroller_controls)
        self.scroller_plot_host = QtWidgets.QWidget()
        self.scroller_plot_host_layout = QtWidgets.QVBoxLayout(self.scroller_plot_host)
        self.scroller_plot_host_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_scroller_layout.addWidget(self.scroller_plot_host)
        self.plots_tabs.addTab(self.plot_page, 'Results Plot')
        self.plots_tabs.addTab(self.cross_section_page, 'cross-section plot')
        self.plots_tabs.addTab(self.plot_scroller_page, 'Plot Scroller')
        self.plots_container_layout.addWidget(self.plots_tabs)

        # Bottom: IO tabs (geometry table). Results table will be shown on the left Results tab.
        self.io_tabs = QtWidgets.QTabWidget()
        self.geom_page = QtWidgets.QWidget()
        self.geom_page_layout = QtWidgets.QVBoxLayout(self.geom_page)
        self.io_tabs.addTab(self.geom_page, 'Geometry Table')
        # create a results table (will be added to left tabs as the Results tab)
        self.results_table = QtWidgets.QTableWidget()

        # assemble right split and add to main splitter
        self.right_split.addWidget(self.plots_container)
        self.right_split.addWidget(self.io_tabs)
        self.horiz_split.addWidget(self.right_split)

        # Results page (moved to bottom-right IO tabs)
        try:
            self.results_page = QtWidgets.QWidget()
            self.results_page_layout = QtWidgets.QVBoxLayout(self.results_page)
            self.results_page_layout.addWidget(self.results_table)
            self.results_status_label = QtWidgets.QLabel('')
            self.results_status_label.setWordWrap(True)
            self.results_status_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            self.results_page_layout.addWidget(self.results_status_label)
            self.io_tabs.addTab(self.results_page, 'Results')
        except Exception:
            pass

        # Section selector and property editor (no 'Sections:' label)
        self.section_cb = QtWidgets.QComboBox()
        self.section_cb.currentIndexChanged.connect(self.on_section_change)
        # add section selector combobox to Geometry tab (no label)
        self.geom_tab_layout.addWidget(self.section_cb)

        prop_labels = [
            ('left_bank_station','Left bank'), ('right_bank_station','Right bank'),
            ('n_lob','n_lob'), ('n_ch','n_ch'), ('n_rob','n_rob'),
            ('contraction_coeff','Cc'), ('expansion_coeff','Ce'),
            # Clarify these are the downstream reach lengths (distance from this section
            # to the next upstream section) following HEC-RAS convention.
            ('L_lob_to_next','L_lob_to_next (to next upstream)'),
            ('L_ch_to_next','L_ch_to_next (to next upstream)'),
            ('L_rob_to_next','L_rob_to_next (to next upstream)'),
            # Culvert inlet control (FHWA HEC-5)
            ('culvert_code','Culvert code (0=none, 1-57=FHWA)'),
            ('culvert_shape','Shape (circular/rect)'),
            ('culvert_diameter','Diameter (ft)'),
            ('culvert_width','Width (ft)'),
            ('culvert_height','Height (ft)'),
            ('culvert_upstream_invert','Culvert upstream invert (ft)'),
            ('culvert_downstream_invert','Culvert downstream invert (ft)'),
            ('culvert_length','Culvert length (ft)'),
            ('culvert_weir_coeff','Weir coeff Cw (ft^0.5/s)'),
            ('culvert_weir_sta_left','Weir left station (ft)'),
            ('culvert_weir_sta_right','Weir right station (ft)')
        ]
        self.props = {}
        # Property editors in a form layout
        prop_form = QtWidgets.QFormLayout()
        for key, label in prop_labels:
            edit = QtWidgets.QLineEdit('0.0')
            prop_form.addRow(label + ':', edit)
            self.props[key] = edit
        # Read-only computed slope derived from culvert invert/length fields.
        self.culvert_slope_display = QtWidgets.QLineEdit('0.0')
        self.culvert_slope_display.setReadOnly(True)
        prop_form.addRow('Computed culvert slope (ft/ft):', self.culvert_slope_display)
        self.geom_tab_layout.addLayout(prop_form)

        self.apply_section_btn = QtWidgets.QPushButton('Apply Section Changes')
        self.apply_section_btn.clicked.connect(self.apply_section_changes)
        self.geom_tab_layout.addWidget(self.apply_section_btn)

        # Undo/Redo and layout buttons
        undo_btn = QtWidgets.QPushButton('Undo')
        redo_btn = QtWidgets.QPushButton('Redo')
        undo_btn.clicked.connect(self.undo)
        redo_btn.clicked.connect(self.redo)
        self.geom_tab_layout.addWidget(undo_btn)
        self.geom_tab_layout.addWidget(redo_btn)

        save_layout_btn = QtWidgets.QPushButton('Save Layout...')
        load_layout_btn = QtWidgets.QPushButton('Load Layout...')
        save_layout_btn.clicked.connect(self.save_layout_file)
        load_layout_btn.clicked.connect(self.load_layout_file)
        self.geom_tab_layout.addWidget(save_layout_btn)
        self.geom_tab_layout.addWidget(load_layout_btn)

        # Geometry editor (table)
        geom_label = QtWidgets.QLabel('Cross-section Geometry (station, elevation)')
        self.geom_page_layout.addWidget(geom_label)
        self.geom_table = QtWidgets.QTableWidget(0,2)
        self.geom_table.setHorizontalHeaderLabels(['Station','Elevation'])
        self.geom_table.horizontalHeader().setStretchLastSection(True)
        # Allow user to resize columns
        self.geom_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.geom_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.geom_page_layout.addWidget(self.geom_table)

        # preview updates only when user clicks 'Apply Geometry'

        geom_btns_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton('Add')
        add_btn.clicked.connect(self.geom_add_row)
        rem_btn = QtWidgets.QPushButton('Remove')
        rem_btn.clicked.connect(self.geom_remove_row)
        up_btn = QtWidgets.QPushButton('Up')
        up_btn.clicked.connect(lambda: self.geom_move(True))
        down_btn = QtWidgets.QPushButton('Down')
        down_btn.clicked.connect(lambda: self.geom_move(False))
        copy_btn = QtWidgets.QPushButton('Copy')
        copy_btn.clicked.connect(self.geom_copy_selected)
        paste_btn = QtWidgets.QPushButton('Paste')
        paste_btn.clicked.connect(self.geom_paste_clipboard)
        self.apply_geom_btn = QtWidgets.QPushButton('Apply Geometry')
        self.apply_geom_btn.clicked.connect(self.apply_geom_changes)
        for w in (add_btn, rem_btn, up_btn, down_btn, copy_btn, paste_btn, self.apply_geom_btn):
            geom_btns_layout.addWidget(w)
        self.geom_page_layout.addLayout(geom_btns_layout)
        self.geometry_edit_buttons = [add_btn, rem_btn, up_btn, down_btn, copy_btn, paste_btn, self.apply_geom_btn]

        terrain_layout = QtWidgets.QHBoxLayout()
        terrain_layout.addWidget(QtWidgets.QLabel('Terrain Raster'))
        self.terrain_raster_combo = QtWidgets.QComboBox()
        self.terrain_raster_combo.setToolTip('Select a loaded raster layer used to populate cross-section vertex Z values.')
        self.terrain_raster_combo.currentIndexChanged.connect(self._on_terrain_raster_combo_changed)
        terrain_layout.addWidget(self.terrain_raster_combo, 1)
        self.refresh_terrain_btn = QtWidgets.QPushButton('Refresh')
        self.refresh_terrain_btn.clicked.connect(self.refresh_terrain_raster_choices)
        terrain_layout.addWidget(self.refresh_terrain_btn)
        self.populate_z_btn = QtWidgets.QPushButton('Populate Z From Terrain')
        self.populate_z_btn.clicked.connect(self.on_populate_section_z_from_terrain)
        terrain_layout.addWidget(self.populate_z_btn)
        self.auto_populate_z_cb = QtWidgets.QCheckBox('Auto-populate on Apply Geometry')
        self.auto_populate_z_cb.setChecked(True)
        terrain_layout.addWidget(self.auto_populate_z_cb)
        self.geom_page_layout.addLayout(terrain_layout)

        # Detail small plot: will be shown in the Cross-section tab
        # Cross-section preview widget (uses Qt painting, updates live)
        self.detail_plot_widget = CrossSectionPreview()
        self.detail_plot_layout = QtWidgets.QVBoxLayout()
        self.detail_plot_layout.setContentsMargins(0,0,0,0)
        self.detail_plot_layout.addWidget(self.detail_plot_widget)
        self.cross_section_page.setLayout(self.cross_section_layout)
        # add detail plot to the cross-section tab
        try:
            self.cross_section_layout.addWidget(self.detail_plot_widget)
        except Exception:
            # fallback: keep in geom tab if cross-section page not yet available
            self.geom_tab_layout.addWidget(self.detail_plot_widget)

        # Save model/plot buttons
        self.save_model_btn = QtWidgets.QPushButton('Save Model...')
        self.save_model_btn.clicked.connect(self.on_save_model)
        self.save_plot_btn = QtWidgets.QPushButton('Save Plot...')
        self.save_plot_btn.clicked.connect(self.on_save_plot)
        self.save_plot_btn.setEnabled(False)
        # Save model belongs on Boundary tab; save plot in Results
        self.boundary_tab_layout.addWidget(self.save_model_btn)
        try:
            # place Save Plot button into bottom-right Results tab when available
            self.results_page_layout.addWidget(self.save_plot_btn)
        except Exception:
            # fallback: place on boundary tab
            try:
                self.boundary_tab_layout.addWidget(self.save_plot_btn)
            except Exception:
                pass
        # Plugin-only runtime uses native PyQGIS GeoPackage support.
        self.can_gpkg = True
        # Save-to-GeoPackage action for plugin workflows.
        self.save_gpkg_btn = QtWidgets.QPushButton('Save to GeoPackage...')
        self.save_gpkg_btn.clicked.connect(self.on_save_geopackage)
        self.save_gpkg_btn.setEnabled(self.can_gpkg)
        self.boundary_tab_layout.addWidget(self.save_gpkg_btn)
        # Small indicator for GeoPackage capability
        self.gpkg_label = QtWidgets.QLabel('GeoPackage: ' + ('available' if self.can_gpkg else 'missing'))
        self.boundary_tab_layout.addWidget(self.gpkg_label)
        self.toggle_gpkg_edit_btn = QtWidgets.QPushButton('Enable Layer Editing')
        self.toggle_gpkg_edit_btn.clicked.connect(self.on_toggle_geopackage_editing)
        self.toggle_gpkg_edit_btn.setEnabled(False)
        self.boundary_tab_layout.addWidget(self.toggle_gpkg_edit_btn)
        self.save_layer_edits_btn = QtWidgets.QPushButton('Save Layer Edits')
        self.save_layer_edits_btn.clicked.connect(self.on_save_layer_edits)
        self.save_layer_edits_btn.setEnabled(False)
        self.boundary_tab_layout.addWidget(self.save_layer_edits_btn)
        self.layer_edit_status_label = QtWidgets.QLabel('Layer editing: unavailable')
        self.boundary_tab_layout.addWidget(self.layer_edit_status_label)
        self.attribute_form_hint = QtWidgets.QLabel(
            'Model edits for GeoPackage-backed models are done through native QGIS attribute forms.'
        )
        self.attribute_form_hint.setWordWrap(True)
        self.boundary_tab_layout.addWidget(self.attribute_form_hint)
        # Load example into QGIS (only enabled when running inside QGIS)
        self.load_example_btn = QtWidgets.QPushButton('Load example into QGIS')
        self.load_example_btn.clicked.connect(self.load_example_into_qgis)
        # Detect whether we're running inside QGIS.
        has_iface = False
        try:
            import qgis.utils as _qutils
            if getattr(_qutils, 'iface', None) is not None:
                has_iface = True
        except Exception:
            pass
        self.load_example_btn.setEnabled(bool(has_iface))
        self.boundary_tab_layout.addWidget(self.load_example_btn)

        # In-memory model
        self.model = None
        self.results = None
        self.loaded_gpkg_path = ''
        self.gpkg_editing_enabled = False
        self.gpkg_dirty = False
        self._scroller_entries = []
        self._scroller_canvas = None
        self._dock_host_window = None
        self._detached_docks = {}
        self._tab_detach_config = {}
        self._cross_section_geometry_signal_connected = False
        self._handling_cross_section_geometry_change = False
        # Undo/redo stacks (store JSON strings)
        self.undo_stack = []
        self.redo_stack = []
        self._set_model_editing_enabled(True)
        self._create_backwater_menu()
        self._apply_form_only_ui_mode()
        self._update_geopackage_edit_state()
        self._refresh_scroller_choices()
        self.refresh_terrain_raster_choices()
        self._configure_detachable_tabs()

    def set_dock_host_window(self, host_window):
        """Set host QMainWindow used for detached panel docking/floating."""
        self._dock_host_window = host_window

    def _create_backwater_menu(self):
        menu = getattr(self, 'backwater_menu', None)
        if menu is None:
            return

        self.action_new_model = QtGui.QAction('Create Model GeoPackage...', self)
        self.action_new_model.triggered.connect(self.on_new_model)
        menu.addAction(self.action_new_model)

        self.action_open_model = QtGui.QAction('Load Model GeoPackage...', self)
        self.action_open_model.triggered.connect(self.on_menu_open_model)
        menu.addAction(self.action_open_model)

        self.action_save_model = QtGui.QAction('Save Model GeoPackage As...', self)
        self.action_save_model.triggered.connect(self.on_save_geopackage)
        menu.addAction(self.action_save_model)

        menu.addSeparator()

        self.action_toggle_layer_editing = QtGui.QAction('Enable Layer Editing', self)
        self.action_toggle_layer_editing.triggered.connect(self.on_toggle_geopackage_editing)
        menu.addAction(self.action_toggle_layer_editing)

        self.action_save_layer_edits = QtGui.QAction('Save Layer Edits', self)
        self.action_save_layer_edits.triggered.connect(self.on_save_layer_edits)
        menu.addAction(self.action_save_layer_edits)

        menu.addSeparator()

        self.action_run_model = QtGui.QAction('Run Model', self)
        self.action_run_model.setShortcut('F5')
        self.action_run_model.triggered.connect(self.on_run)
        menu.addAction(self.action_run_model)

        menu.addSeparator()

        self.action_open_results_plot = QtGui.QAction('Open Results Plot', self)
        self.action_open_results_plot.triggered.connect(self.open_results_plot)
        menu.addAction(self.action_open_results_plot)

        self.action_open_results_table = QtGui.QAction('Open Results Table', self)
        self.action_open_results_table.triggered.connect(self.open_results_table)
        menu.addAction(self.action_open_results_table)

    def _apply_form_only_ui_mode(self):
        if not getattr(self, 'form_only_mode', False):
            return

        # Hide legacy in-widget model editors; editing is done via QGIS attribute forms.
        try:
            if self.left_widget is not None:
                self.left_widget.setVisible(False)
        except Exception:
            pass
        try:
            self.horiz_split.setSizes([0, 1])
        except Exception:
            pass

        try:
            geom_idx = self.io_tabs.indexOf(self.geom_page)
            if geom_idx >= 0:
                self.io_tabs.removeTab(geom_idx)
        except Exception:
            pass

        try:
            cs_idx = self.plots_tabs.indexOf(self.cross_section_page)
            if cs_idx >= 0:
                self.plots_tabs.removeTab(cs_idx)
        except Exception:
            pass

        for widget in (
            getattr(self, 'save_model_btn', None),
            getattr(self, 'save_gpkg_btn', None),
            getattr(self, 'toggle_gpkg_edit_btn', None),
            getattr(self, 'save_layer_edits_btn', None),
            getattr(self, 'load_example_btn', None),
            getattr(self, 'attribute_form_hint', None),
            getattr(self, 'layer_edit_status_label', None),
            getattr(self, 'gpkg_label', None),
        ):
            try:
                if widget is not None:
                    widget.setVisible(False)
            except Exception:
                pass

    def open_results_plot(self):
        try:
            if self.plots_tabs is not None and self.plot_page is not None:
                self.plots_tabs.setCurrentWidget(self.plot_page)
        except Exception:
            pass

    def open_results_table(self):
        try:
            if self.io_tabs is not None and self.results_page is not None:
                self.io_tabs.setCurrentWidget(self.results_page)
        except Exception:
            pass

    def _configure_detachable_tabs(self):
        try:
            self._register_detachable_tab_widget(
                self.left_tabs,
                QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
                'left_tabs'
            )
            self._register_detachable_tab_widget(
                self.io_tabs,
                QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
                'io_tabs'
            )
            self._register_detachable_tab_widget(
                self.plots_tabs,
                QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
                'plots_tabs'
            )
        except Exception:
            pass

    def _register_detachable_tab_widget(self, tab_widget, default_area, widget_key):
        if tab_widget is None:
            return
        self._tab_detach_config[tab_widget] = {
            'default_area': default_area,
            'widget_key': widget_key,
        }
        bar = tab_widget.tabBar()
        bar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        bar.customContextMenuRequested.connect(
            lambda pos, tw=tab_widget: self._show_tab_detach_menu(tw, pos)
        )
        bar.setMovable(True)

    def _show_tab_detach_menu(self, tab_widget, pos):
        bar = tab_widget.tabBar()
        idx = bar.tabAt(pos)
        if idx < 0:
            return
        menu = QtWidgets.QMenu(self)
        tab_title = tab_widget.tabText(idx)
        detach_action = menu.addAction(f'Detach "{tab_title}" panel')
        action = menu.exec(bar.mapToGlobal(pos))
        if action == detach_action:
            self._detach_tab(tab_widget, idx)

    def _detach_tab(self, tab_widget, index):
        if tab_widget is None or index < 0:
            return
        config = self._tab_detach_config.get(tab_widget, {})
        default_area = config.get('default_area', QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        widget_key = config.get('widget_key', 'tabs')

        page = tab_widget.widget(index)
        if page is None:
            return
        title = tab_widget.tabText(index)
        tab_widget.removeTab(index)

        host = self._dock_host_window
        if host is None:
            host = self.window()

        dock = _ReattachDockWidget(title, self._reattach_dock_tab, host)
        dock.setObjectName(f'backwater_{widget_key}_{title}')
        dock.setWidget(page)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)

        if isinstance(host, QtWidgets.QMainWindow):
            host.addDockWidget(default_area, dock)
        dock.setFloating(True)
        dock.show()
        dock.raise_()

        self._detached_docks[dock] = {
            'tab_widget': tab_widget,
            'title': title,
            'widget_key': widget_key,
        }

    def _reattach_dock_tab(self, dock):
        meta = self._detached_docks.pop(dock, None)
        if not meta:
            try:
                dock.deleteLater()
            except Exception:
                pass
            return
        tab_widget = meta.get('tab_widget')
        title = meta.get('title', 'Panel')
        try:
            page = dock.widget()
            if page is not None and tab_widget is not None:
                dock.setWidget(None)
                tab_widget.addTab(page, title)
                tab_widget.setCurrentWidget(page)
        finally:
            try:
                dock.deleteLater()
            except Exception:
                pass

    def _show_about(self):
        ui_info(self, 'About', 'Backwater Qt GUI\nEnhanced UI with menus, toolbars, docks, and status bar')

    def set_view_mode(self, mode: str):
        """mode: 'geometry' | 'profile' | 'section'"""
        try:
            if mode == 'geometry':
                try:
                    # show geometry table in bottom IO tabs
                    self.io_tabs.setCurrentWidget(self.geom_page)
                except Exception:
                    pass
                try:
                    self.status_label.setText('View: Geometry Editor')
                except Exception:
                    pass
                return
            # ensure plot page is visible
            # top plots container hosts the plot page; nothing to switch
            # toggle axes visibility if plot exists
            if hasattr(self, '_plot_axes') and self._plot_axes is not None:
                prof_ax = self._plot_axes.get('profile_ax')
                secs = self._plot_axes.get('section_axes', [])
                if mode == 'profile':
                    try:
                        if prof_ax is not None:
                            prof_ax.set_visible(True)
                        for d in secs:
                            d['ax'].set_visible(False)
                    except Exception:
                        pass
                    try:
                        self.status_label.setText('View: Profile Plot')
                    except Exception:
                        pass
                elif mode == 'section':
                    try:
                        if prof_ax is not None:
                            prof_ax.set_visible(False)
                        for d in secs:
                            d['ax'].set_visible(True)
                    except Exception:
                        pass
                    try:
                        self.status_label.setText('View: Cross-section Plot')
                    except Exception:
                        pass
            else:
                try:
                    self.status_label.setText('View: Plot (no data)')
                except Exception:
                    pass
            try:
                if hasattr(self, '_plot_canvas') and self._plot_canvas is not None:
                    self._plot_canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    # --- geometry table helpers
    def geom_add_row(self):
        r = self.geom_table.rowCount()
        self.geom_table.insertRow(r)
        self.geom_table.setItem(r,0, QtWidgets.QTableWidgetItem('0.0'))
        self.geom_table.setItem(r,1, QtWidgets.QTableWidgetItem('0.0'))

    def geom_remove_row(self):
        sel = self.geom_table.currentRow()
        if sel >= 0:
            self.geom_table.removeRow(sel)

    def geom_copy_selected(self):
        """Copy selected geometry rows to the clipboard (tab-separated)."""
        rows = sorted(set(i.row() for i in self.geom_table.selectedItems()))
        if not rows:
            ui_info(self, 'Copy', 'No rows selected')
            return
        lines = []
        for r in rows:
            st = self.geom_table.item(r,0).text() if self.geom_table.item(r,0) else '0'
            z = self.geom_table.item(r,1).text() if self.geom_table.item(r,1) else '0'
            lines.append(f"{st}\t{z}")
        try:
            QtWidgets.QApplication.clipboard().setText('\n'.join(lines))
            ui_info(self, 'Copied', f'Copied {len(lines)} row(s)')
        except Exception:
            ui_warning(self, 'Copy failed', 'Could not copy to clipboard')

    def geom_paste_clipboard(self):
        """Paste tab-separated station,elevation rows from the clipboard into the geometry table.

        Existing selection's first row is used as the insert position; if nothing
        selected, rows are appended at the end. Non-numeric rows are skipped.
        """
        try:
            txt = QtWidgets.QApplication.clipboard().text()
        except Exception:
            ui_warning(self, 'Paste failed', 'Could not access clipboard')
            return
        if not txt:
            ui_info(self, 'Paste', 'Clipboard is empty')
            return
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            ui_info(self, 'Paste', 'No data to paste')
            return
        # Determine insert index: at current row or append
        sel = self.geom_table.currentRow()
        if sel is None or sel < 0:
            insert_at = self.geom_table.rowCount()
        else:
            insert_at = sel
        # push undo state
        try:
            self.push_undo()
        except Exception:
            pass
        added = 0
        for i, line in enumerate(lines):
            # support tab, comma or whitespace separation
            parts = line.split('\t') if '\t' in line else (line.split(',') if ',' in line else line.split())
            if not parts:
                continue
            try:
                st = float(parts[0])
                z = float(parts[1]) if len(parts) > 1 else 0.0
            except Exception:
                # skip malformed rows
                continue
            r = insert_at + added
            self.geom_table.insertRow(r)
            try:
                self.geom_table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{st:.3f}"))
                self.geom_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{z:.3f}"))
            except Exception:
                # best-effort insertion; ignore failures per-row
                pass
            added += 1
        if added:
            ui_info(self, 'Pasted', f'Pasted {added} row(s)')
        else:
            ui_warning(self, 'Paste', 'No valid rows found in clipboard')

    # --- undo / redo support (serialize model dict)
    def _model_to_dict(self):
        if self.model is None:
            return None
        out = {
            'flow_cfs': self.model.flow_cfs,
            'flow_change': self.model.flow_change,
            'boundary_condition': self.model.boundary_condition,
            'boundary_value': self.model.boundary_value,
            'sections': []
        }
        for xs in self.model.sections:
            out['sections'].append({
                'river_station': xs.river_station,
                'geometry': [[float(x), float(z)] for x,z in xs.geometry],
                'left_bank_station': xs.left_bank_station,
                'right_bank_station': xs.right_bank_station,
                'n_lob': xs.n_lob, 'n_ch': xs.n_ch, 'n_rob': xs.n_rob,
                'contraction_coeff': xs.contraction_coeff, 'expansion_coeff': xs.expansion_coeff,
                'L_lob_to_next': xs.L_lob_to_next, 'L_ch_to_next': xs.L_ch_to_next, 'L_rob_to_next': xs.L_rob_to_next,
                'culvert_code': xs.culvert_code,
                'culvert_shape': xs.culvert_shape,
                'culvert_diameter': xs.culvert_diameter,
                'culvert_width': xs.culvert_width,
                'culvert_height': xs.culvert_height,
                'culvert_upstream_invert': xs.culvert_upstream_invert,
                'culvert_downstream_invert': xs.culvert_downstream_invert,
                'culvert_length': xs.culvert_length,
                'culvert_weir_coeff': getattr(xs, 'culvert_weir_coeff', 3.0),
                'culvert_weir_sta_left': getattr(xs, 'culvert_weir_sta_left', 0.0),
                'culvert_weir_sta_right': getattr(xs, 'culvert_weir_sta_right', 0.0),
            })
        return out

    def _load_model_from_dict(self, d):
        if d is None:
            self.model = None
            return
        secs = []
        for s in d.get('sections', []):
            xs = CrossSection(
                river_station=s.get('river_station',''),
                geometry=[(float(x), float(z)) for x,z in s.get('geometry',[])],
                left_bank_station=float(s.get('left_bank_station',0.0)),
                right_bank_station=float(s.get('right_bank_station',0.0)),
                n_lob=float(s.get('n_lob',0.035)), n_ch=float(s.get('n_ch',0.035)), n_rob=float(s.get('n_rob',0.035)),
                contraction_coeff=float(s.get('contraction_coeff',0.1)), expansion_coeff=float(s.get('expansion_coeff',0.3)),
                L_lob_to_next=float(s.get('L_lob_to_next',0.0)), L_ch_to_next=float(s.get('L_ch_to_next',0.0)), L_rob_to_next=float(s.get('L_rob_to_next',0.0)),
                culvert_code=int(float(s.get('culvert_code', 0) or 0)),
                culvert_shape=(str(s.get('culvert_shape')).strip() if s.get('culvert_shape') is not None else None),
                culvert_diameter=float(s.get('culvert_diameter', 0.0) or 0.0),
                culvert_width=float(s.get('culvert_width', 0.0) or 0.0),
                culvert_height=float(s.get('culvert_height', 0.0) or 0.0),
                culvert_upstream_invert=float(s.get('culvert_upstream_invert', 0.0) or 0.0),
                culvert_downstream_invert=float(s.get('culvert_downstream_invert', 0.0) or 0.0),
                culvert_length=float(s.get('culvert_length', 0.0) or 0.0),
                culvert_weir_coeff=float(s.get('culvert_weir_coeff', 3.0) or 3.0),
                culvert_weir_sta_left=float(s.get('culvert_weir_sta_left', 0.0) or 0.0),
                culvert_weir_sta_right=float(s.get('culvert_weir_sta_right', 0.0) or 0.0)
            )
            secs.append(xs)
        self.model = ModelInput(
            flow_cfs=float(d.get('flow_cfs', 0.0)),
            flow_change=d.get('flow_change'),
            boundary_condition=d.get('boundary_condition','known_wse'),
            boundary_value=float(d.get('boundary_value',0.0)),
            sections=secs
        )
        self._sync_ui_from_model()
        # refresh UI
        self.section_cb.clear()
        self.section_cb.addItems([xs.river_station for xs in self.model.sections])
        if self.model.sections:
            self.section_cb.setCurrentIndex(0)
        self._refresh_scroller_choices()

    def push_undo(self):
        # push current model state onto undo stack
        d = self._model_to_dict()
        if d is None:
            return
        s = json.dumps(d)
        self.undo_stack.append(s)
        # clear redo on new action
        self.redo_stack.clear()

    def _clear_layout_widgets(self, layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)

    def _computed_culvert_slope(self, xs) -> float:
        """Compute slope from invert elevations and culvert length."""
        try:
            # Prefer model method when available.
            method = getattr(xs, 'culvert_slope', None)
            if callable(method):
                return float(method())
        except Exception:
            pass
        try:
            up = float(getattr(xs, 'culvert_upstream_invert', 0.0))
            dn = float(getattr(xs, 'culvert_downstream_invert', 0.0))
            length = float(getattr(xs, 'culvert_length', 0.0))
            if length <= 0.0:
                return 0.0
            return (up - dn) / length
        except Exception:
            return 0.0

    def _update_computed_culvert_slope_display(self, xs=None):
        try:
            if xs is None:
                idx = self.section_cb.currentIndex()
                if self.model is None or idx < 0 or idx >= len(self.model.sections):
                    self.culvert_slope_display.setText('0.000000')
                    return
                xs = self.model.sections[idx]
            slope = self._computed_culvert_slope(xs)
            self.culvert_slope_display.setText(f"{slope:.6f}")
        except Exception:
            try:
                self.culvert_slope_display.setText('0.000000')
            except Exception:
                pass

    def _refresh_scroller_choices(self):
        self._scroller_entries = []
        self.scroller_combo.blockSignals(True)
        self.scroller_combo.clear()

        if self.model is None:
            self.scroller_combo.blockSignals(False)
            self.scroller_status.setText('No model loaded')
            self.scroller_prev_btn.setEnabled(False)
            self.scroller_next_btn.setEnabled(False)
            self._clear_layout_widgets(self.scroller_plot_host_layout)
            return

        self._scroller_entries.append(('profile', None, 'Profile Plot'))
        for i, xs in enumerate(self.model.sections):
            self._scroller_entries.append(('section', i, f'Section: {xs.river_station}'))
        for _, _, label in self._scroller_entries:
            self.scroller_combo.addItem(label)

        self.scroller_combo.setCurrentIndex(0)
        self.scroller_combo.blockSignals(False)
        has_entries = len(self._scroller_entries) > 0
        self.scroller_prev_btn.setEnabled(has_entries)
        self.scroller_next_btn.setEnabled(has_entries)
        self._refresh_scroller_plot(0)

    def _scroll_plot_step(self, step):
        count = self.scroller_combo.count()
        if count <= 0:
            return
        current = self.scroller_combo.currentIndex()
        if current < 0:
            current = 0
        self.scroller_combo.setCurrentIndex((current + step) % count)

    def _refresh_scroller_plot(self, index):
        if not HAVE_MPL:
            self._clear_layout_widgets(self.scroller_plot_host_layout)
            self.scroller_plot_host_layout.addWidget(QtWidgets.QLabel('matplotlib not available'))
            return
        if not self._scroller_entries:
            self._clear_layout_widgets(self.scroller_plot_host_layout)
            self.scroller_plot_host_layout.addWidget(QtWidgets.QLabel('No plots available'))
            return
        if index is None or index < 0 or index >= len(self._scroller_entries):
            index = 0

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            import matplotlib.pyplot as plt
            import numpy as _np
        except Exception:
            self._clear_layout_widgets(self.scroller_plot_host_layout)
            self.scroller_plot_host_layout.addWidget(QtWidgets.QLabel('matplotlib backend unavailable'))
            return

        kind, section_idx, _label = self._scroller_entries[index]
        self.scroller_status.setText(f'{index + 1}/{len(self._scroller_entries)}')

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        if kind == 'profile':
            if self.results is None:
                ax.text(0.5, 0.5, 'Run solver to see profile plot', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            else:
                chainage = [0.0]
                for i in range(1, len(self.model.sections)):
                    prev = self.model.sections[i - 1]
                    try:
                        chainage.append(chainage[-1] + float(prev.L_ch_to_next))
                    except Exception:
                        chainage.append(chainage[-1] + 1.0)
                wse_vals = [s.wse for s in self.results]
                ax.plot(chainage[:len(wse_vals)], wse_vals, '-o', color='#1f77b4', lw=2, markersize=6)
                ax.set_xlabel('Chainage (ft)')
                ax.set_ylabel('Water Surface Elevation (ft)')
                ax.set_title('Water Surface Profile')
                ax.grid(True, linestyle='--', alpha=0.4)
        else:
            xs = self.model.sections[section_idx]
            geom = sorted(xs.geometry, key=lambda p: p[0])
            sx = _np.array([p[0] for p in geom]) if geom else _np.array([])
            sz = _np.array([p[1] for p in geom]) if geom else _np.array([])
            if len(sx) == 0:
                ax.text(0.5, 0.5, 'No geometry for section', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ax.plot(sx, sz, '-k', lw=1.5, label='Bed')
                ax.fill_between(sx, sz, min(sz) - 1.0, color='#efefef')
                try:
                    lb = float(xs.left_bank_station)
                    rb = float(xs.right_bank_station)
                    ax.axvspan(lb, rb, color='#fafafa', alpha=0.7)
                    ax.axvline(lb, color='0.6', linestyle='--')
                    ax.axvline(rb, color='0.6', linestyle='--')
                except Exception:
                    pass
                if self.results is not None and section_idx < len(self.results):
                    wse = self.results[section_idx].wse
                    ax.axhline(wse, color='#1f77b4', lw=1.5, label=f'WSE {wse:.3f}')
                    where_sub = sz < wse
                    if where_sub.any():
                        ax.fill_between(sx, sz, wse, where=where_sub, interpolate=True, color='#1f77b4', alpha=0.35)
                # Culvert opening overlay at section center station.
                try:
                    if getattr(xs, 'has_culvert', lambda: False)():
                        center_x = 0.5 * (float(sx.min()) + float(sx.max()))
                        y_full = max(
                            float(getattr(xs, 'culvert_height', 0.0) or 0.0),
                            float(getattr(xs, 'culvert_diameter', 0.0) or 0.0),
                        )
                        up_inv = float(getattr(xs, 'culvert_upstream_invert', 0.0) or 0.0)
                        dn_inv = float(getattr(xs, 'culvert_downstream_invert', 0.0) or 0.0)
                        if y_full > 0.0:
                            dx = max(0.02 * max(float(sx.max()) - float(sx.min()), 1.0), 0.1)
                            ax.plot([center_x - dx, center_x - dx], [up_inv, up_inv + y_full], '-', color='crimson', lw=2.0, label='Culvert opening (upstream)')
                            ax.plot([center_x + dx, center_x + dx], [dn_inv, dn_inv + y_full], ':', color='crimson', lw=2.0, label='Culvert opening (downstream)')
                except Exception:
                    pass
                ax.set_title(f'Cross Section {xs.river_station}')
                ax.set_xlabel('Station (ft)')
                ax.set_ylabel('Elevation (ft)')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc='best')

        fig.tight_layout()
        self._clear_layout_widgets(self.scroller_plot_host_layout)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self.plot_scroller_page)
        self.scroller_plot_host_layout.addWidget(toolbar)
        self.scroller_plot_host_layout.addWidget(canvas)
        self._scroller_canvas = canvas
        canvas.draw_idle()

    def undo(self):
        if len(self.undo_stack) < 2:
            ui_info(self, 'Undo', 'Nothing to undo')
            try:
                self.status_label.setText('Nothing to undo')
            except Exception:
                pass
            return
        cur = self.undo_stack.pop()
        self.redo_stack.append(cur)
        prev = self.undo_stack[-1]
        d = json.loads(prev)
        self._load_model_from_dict(d)
        ui_info(self, 'Undo', 'Reverted to previous model state')
        try:
            self.status_label.setText('Undone')
        except Exception:
            pass

    def redo(self):
        if not self.redo_stack:
            ui_info(self, 'Redo', 'Nothing to redo')
            try:
                self.status_label.setText('Nothing to redo')
            except Exception:
                pass
            return
        s = self.redo_stack.pop()
        self.undo_stack.append(s)
        d = json.loads(s)
        self._load_model_from_dict(d)
        ui_info(self, 'Redo', 'Redone model state')
        try:
            self.status_label.setText('Redone')
        except Exception:
            pass

    # Detach plot window
    def detach_plot(self):
        # Create a separate window containing the current plot widget contents
        try:
            self.detached_win = QtWidgets.QMainWindow(self)
            w = QtWidgets.QWidget()
            self.detached_win.setCentralWidget(w)
            l = QtWidgets.QVBoxLayout(w)
            # Move existing plot widgets into detached window by reparenting
            page = self.plots_container
            if page is not None and page.layout() is not None:
                layout = page.layout()
                for i in reversed(range(layout.count())):
                    item = layout.itemAt(i)
                    if item is None:
                        continue
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                        l.addWidget(widget)
            self.detached_win.setWindowTitle('Detached Plots')
            self.detached_win.resize(900,600)
            self.detached_win.show()
        except Exception:
            ui_warning(self, 'Detach', 'Could not detach plot')

    # --- section handling
    def on_section_change(self, idx):
        if self.model is None or idx < 0:
            return
        xs = self.model.sections[idx]
        # populate props
        for k, w in self.props.items():
            val = getattr(xs, k, 0.0)
            if k == 'culvert_shape':
                w.setText('' if val is None else str(val))
            else:
                w.setText(str(val))
        self._update_computed_culvert_slope_display(xs)
        # populate geom table
        self.geom_table.setRowCount(0)
        for st,z in sorted(xs.geometry, key=lambda p: p[0]):
            r = self.geom_table.rowCount()
            self.geom_table.insertRow(r)
            self.geom_table.setItem(r,0, QtWidgets.QTableWidgetItem(f"{st:.3f}"))
            self.geom_table.setItem(r,1, QtWidgets.QTableWidgetItem(f"{z:.3f}"))
        # detail plot
        self.plot_section_detail(idx)
        try:
            self.refresh_plot()
        except Exception:
            pass
        try:
            self.update_cross_section_tab_state()
        except Exception:
            pass

    def apply_section_changes(self):
        if self.model is None:
            ui_warning(self, 'No model', 'Load or create a model first')
            return
        idx = self.section_cb.currentIndex()
        if idx < 0:
            return
        # push undo before modifying
        self.push_undo()
        try:
            self.refresh_plot()
        except Exception:
            pass
        xs = self.model.sections[idx]
        for k,w in self.props.items():
            try:
                text_val = w.text().strip()
                # Handle culvert_shape as string, others as float
                if k == 'culvert_shape':
                    if not text_val or text_val.lower() in ('none', 'null', 'nan'):
                        setattr(xs, k, None)
                    else:
                        setattr(xs, k, text_val)
                elif k == 'culvert_code':
                    setattr(xs, k, int(float(text_val)))
                else:
                    setattr(xs, k, float(text_val))
            except Exception:
                pass
        self.results = None
        try:
            self.save_plot_btn.setEnabled(False)
        except Exception:
            pass
        try:
            self._write_cross_section_to_layer(idx, update_geometry=False)
            self._mark_gpkg_dirty()
        except Exception as exc:
            ui_critical(self, 'GeoPackage update failed', str(exc))
            return
        self._update_computed_culvert_slope_display(xs)
        ui_info(self, 'Applied', f'Changes applied to {xs.river_station}')

    def apply_geom_changes(self):
        if self.model is None:
            ui_warning(self, 'No model', 'Load or create a model first')
            return
        idx = self.section_cb.currentIndex()
        if idx < 0:
            return
        # push undo before modifying geometry
        self.push_undo()
        rows = []
        for r in range(self.geom_table.rowCount()):
            try:
                st = float(self.geom_table.item(r,0).text())
                z = float(self.geom_table.item(r,1).text())
            except Exception:
                ui_warning(self, 'Invalid', 'Station/elevation must be numeric')
                return
            rows.append((st,z))
        rows = sorted(rows, key=lambda p: p[0])
        self.model.sections[idx].geometry = [(float(x), float(z)) for x,z in rows]
        self.results = None
        try:
            self.save_plot_btn.setEnabled(False)
        except Exception:
            pass
        try:
            self._write_cross_section_to_layer(idx, update_geometry=True)
            self._mark_gpkg_dirty()
        except Exception as exc:
            ui_critical(self, 'GeoPackage update failed', str(exc))
            return

        used_terrain = False
        try:
            if self.auto_populate_z_cb.isChecked() and self._selected_terrain_raster_layer() is not None:
                used_terrain = self._populate_section_z_from_terrain(idx, announce=False)
        except Exception:
            used_terrain = False

        if used_terrain:
            ui_info(self, 'Applied', f'Geometry applied and Z values populated from terrain for {self.model.sections[idx].river_station}')
        else:
            ui_info(self, 'Applied', f'Geometry applied to {self.model.sections[idx].river_station}')
        # Refresh the cross-section preview (reads from the geometry table)
        try:
            self.plot_section_detail(idx)
        except Exception:
            pass
        try:
            # update cross-section tab availability
            self.update_cross_section_tab_state()
        except Exception:
            pass

    # --- plotting helpers
    def plot_section_detail(self, idx:int):
        # Read geometry directly from the geometry table so the preview reflects
        # live edits. The preview is only active when a model is loaded and the
        # solver has not yet been run (results is None).
        try:
            if self.model is None or getattr(self, 'results', None) is not None:
                try:
                    self.detail_plot_widget.clear()
                except Exception:
                    pass
                return
            geom = []
            for r in range(self.geom_table.rowCount()):
                try:
                    st_item = self.geom_table.item(r, 0)
                    z_item = self.geom_table.item(r, 1)
                    st = float(st_item.text()) if st_item is not None else None
                    z = float(z_item.text()) if z_item is not None else None
                    if st is None or z is None:
                        continue
                    geom.append((st, z))
                except Exception:
                    continue
            if not geom:
                try:
                    xs = self.model.sections[idx]
                    geom = sorted(xs.geometry, key=lambda p: p[0])
                except Exception:
                    geom = []
            self.detail_plot_widget.set_geometry(geom)
            try:
                self.detail_plot_widget.set_title(self.model.sections[idx].river_station)
            except Exception:
                pass
        except Exception:
            pass

    def on_left_tab_changed(self, idx:int):
        # Keep UI in sync when left tab changes by refreshing plots
        try:
            self.refresh_plot()
        except Exception:
            pass

    def populate_results_table(self):
        """Populate the left Results table from self.results and self.model."""
        try:
            import math as _math
            G_val = getattr(_bwmod, 'G', 32.174)
        except Exception:
            _math = None
            G_val = 32.174
        headers = ['Idx', 'Station', 'WSE (ft)', 'Depth (ft)', 'V (ft/s)', 'Alpha', 'Energy (ft)', 'K_total', 'A_total', 'Sf_total', 'Froude']
        self.results_table.clear()
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        n = 0
        if self.results is not None:
            n = len(self.results)
        self.results_table.setRowCount(n)
        for i in range(n):
            try:
                s = self.results[i]
                xs = self.model.sections[i] if (self.model and i < len(self.model.sections)) else None
                station = xs.river_station if xs is not None else str(i)
                wse = getattr(s, 'wse', 0.0)
                depth = getattr(s, 'depth_at_min', 0.0)
                v = getattr(s, 'V_t', 0.0)
                alpha = getattr(s, 'alpha', 0.0)
                Kt = getattr(s, 'K_t', 0.0)
                At = getattr(s, 'A_t', 0.0)
                Sf = getattr(s, 'Sf_total', 0.0)
                # prefer solver-computed Froude if available
                froude = getattr(s, 'Froude', 0.0)
                energy = wse + (alpha * (v ** 2)) / (2.0 * G_val)
                vals = [str(i), station, f"{wse:.3f}", f"{depth:.3f}", f"{v:.3f}", f"{alpha:.3f}", f"{energy:.3f}", f"{Kt:.3f}", f"{At:.3f}", f"{Sf:.6f}", f"{froude:.3f}"]
                for c, val in enumerate(vals):
                    item = QtWidgets.QTableWidgetItem(val)
                    item.setFlags(item.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)
                    self.results_table.setItem(i, c, item)
            except Exception:
                continue
        try:
            self.results_table.resizeColumnsToContents()
        except Exception:
            pass

    def refresh_plot(self):
        """Unified refresh: show profile plot when Results tab active, show section detail when Geometry active."""
        try:
            io_cur = self.io_tabs.currentWidget()
        except Exception:
            io_cur = None
        # If Results page active in bottom-right, refresh results canvas
        try:
            if io_cur is getattr(self, 'results_page', None):
                if hasattr(self, '_plot_canvas') and self._plot_canvas is not None:
                    try:
                        self._plot_canvas.draw_idle()
                    except Exception:
                        pass
                return
        except Exception:
            pass

        # Otherwise refresh cross-section detail for current selection
        try:
            idx = self.section_cb.currentIndex()
            if idx is not None and idx >= 0:
                self.plot_section_detail(idx)
        except Exception:
            pass

    def update_cross_section_tab_state(self):
        """Enable the cross-section preview tab only when a model is loaded and
        the solver has not been run (self.results is None)."""
        try:
            idx = self.plots_tabs.indexOf(self.cross_section_page)
            enabled = (self.model is not None) and (getattr(self, 'results', None) is None)
            if idx >= 0:
                self.plots_tabs.setTabEnabled(idx, bool(enabled))
        except Exception:
            pass

    def _set_model_editing_enabled(self, enabled: bool):
        for widget in self.props.values():
            widget.setReadOnly(not enabled)
        self.apply_section_btn.setEnabled(enabled)
        for button in getattr(self, 'geometry_edit_buttons', []):
            button.setEnabled(enabled)
        triggers = QtWidgets.QAbstractItemView.AllEditTriggers if enabled else QtWidgets.QAbstractItemView.NoEditTriggers
        self.geom_table.setEditTriggers(triggers)

    def refresh_terrain_raster_choices(self):
        combo = getattr(self, 'terrain_raster_combo', None)
        if combo is None:
            return
        selected_layer_id = combo.currentData()
        if not selected_layer_id:
            selected_layer_id = self._project_terrain_raster_id()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('(none)', '')
        try:
            from qgis.core import QgsProject, QgsRasterLayer
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsRasterLayer) and layer.isValid():
                    combo.addItem(layer.name(), layer.id())
        except Exception:
            pass
        if selected_layer_id:
            idx = combo.findData(selected_layer_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if combo.currentData() != self._project_terrain_raster_id():
            self._set_project_terrain_raster_id(combo.currentData() or '')
        combo.blockSignals(False)

    def _project_terrain_raster_id(self) -> str:
        try:
            from qgis.core import QgsExpressionContextUtils, QgsProject
            val = QgsExpressionContextUtils.projectScope(QgsProject.instance()).variable('backwater_terrain_raster_id')
            return str(val or '').strip()
        except Exception:
            return ''

    def _set_project_terrain_raster_id(self, layer_id: str):
        try:
            from qgis.core import QgsExpressionContextUtils, QgsProject
            QgsExpressionContextUtils.setProjectVariable(QgsProject.instance(), 'backwater_terrain_raster_id', str(layer_id or ''))
        except Exception:
            pass

    def _on_terrain_raster_combo_changed(self, _index: int):
        combo = getattr(self, 'terrain_raster_combo', None)
        if combo is None:
            return
        self._set_project_terrain_raster_id(combo.currentData() or '')

    def _selected_terrain_raster_layer(self):
        layer_id = getattr(self, 'terrain_raster_combo', None)
        if layer_id is None:
            return None
        selected_id = self.terrain_raster_combo.currentData()
        if not selected_id:
            return None
        try:
            from qgis.core import QgsProject, QgsRasterLayer
            layer = QgsProject.instance().mapLayer(selected_id)
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                return layer
        except Exception:
            pass
        return None

    @staticmethod
    def _profile_from_map_geometry(geometry):
        profile = []
        if geometry is None or geometry.isEmpty():
            return profile
        cumulative = 0.0
        last_xy = None
        for vertex in geometry.vertices():
            x_val = float(vertex.x())
            y_val = float(vertex.y())
            z_val = float(vertex.z()) if not math.isnan(float(vertex.z())) else 0.0
            if last_xy is None:
                cumulative = 0.0
            else:
                cumulative += math.hypot(x_val - last_xy[0], y_val - last_xy[1])
            profile.append((cumulative, z_val))
            last_xy = (x_val, y_val)
        return profile

    def _compute_centerline_chainage_for_geometry(self, geometry):
        if geometry is None or geometry.isEmpty():
            return None
        center_geom = self._get_loaded_centerline_geometry()
        if center_geom is None:
            return None
        try:
            locate_point = None
            try:
                crossing = geometry.intersection(center_geom)
            except Exception:
                crossing = None

            if crossing is not None and not crossing.isEmpty():
                locate_point = crossing.centroid()

            if locate_point is None or locate_point.isEmpty():
                try:
                    locate_point = center_geom.nearestPoint(geometry)
                except Exception:
                    locate_point = None

            if locate_point is None or locate_point.isEmpty():
                return None

            chainage = float(center_geom.lineLocatePoint(locate_point))
            if math.isnan(chainage) or chainage < 0.0:
                return None
            return chainage
        except Exception:
            return None

    def _compute_river_station_text_for_geometry(self, geometry):
        chainage = self._compute_centerline_chainage_for_geometry(geometry)
        if chainage is None:
            return None
        return f"{chainage:.3f}"

    def _call_set_z_from_raster_expr(self, source_geom, raster_layer):
        if _SET_Z_FROM_RASTER_EXPR is None:
            return None
        # Use the common positional subset so both expression-decorated and
        # plain helper call signatures are supported.
        return _SET_Z_FROM_RASTER_EXPR(source_geom, raster_layer, 1, True)

    def _cross_section_features_sorted_by_chainage(self, layer):
        feats = []
        if layer is None:
            return feats
        for feat in layer.getFeatures():
            geom = feat.geometry()
            chainage = self._compute_centerline_chainage_for_geometry(geom)
            if chainage is None:
                chainage = float(len(feats))
            feats.append((float(chainage), feat))
        feats.sort(key=lambda t: t[0])
        return [feat for _, feat in feats]

    def _find_cross_section_feature(self, layer, idx: int, xs):
        target_rs = str(getattr(xs, 'river_station', ''))
        for feat in layer.getFeatures():
            try:
                if str(feat['river_station']) == target_rs:
                    return feat
            except Exception:
                continue
        ordered = self._cross_section_features_sorted_by_chainage(layer)
        if 0 <= idx < len(ordered):
            return ordered[idx]
        return None

    def _on_cross_section_geometry_changed(self, fid, geometry):
        if self._handling_cross_section_geometry_change:
            return
        layer = self._get_gpkg_layer('cross_sections')
        if layer is None or not layer.isEditable():
            return
        try:
            river_idx = layer.fields().indexOf('river_station')
            self._handling_cross_section_geometry_change = True
            sampled_geom = geometry
            raster_layer = self._selected_terrain_raster_layer()
            if raster_layer is not None:
                try:
                    maybe_geom = self._call_set_z_from_raster_expr(geometry, raster_layer)
                    if maybe_geom is not None and not maybe_geom.isEmpty():
                        sampled_geom = maybe_geom
                        layer.changeGeometry(fid, sampled_geom)
                except Exception:
                    pass

            river_station_text = self._compute_river_station_text_for_geometry(sampled_geom)
            if river_idx != -1 and river_station_text is not None:
                layer.changeAttributeValue(fid, river_idx, river_station_text)
            self._mark_gpkg_dirty()
        finally:
            self._handling_cross_section_geometry_change = False

    def _connect_cross_section_layer_signals(self):
        if self._cross_section_geometry_signal_connected:
            return
        layer = self._get_gpkg_layer('cross_sections')
        if layer is None:
            return
        try:
            layer.geometryChanged.connect(self._on_cross_section_geometry_changed)
            self._cross_section_geometry_signal_connected = True
        except Exception:
            self._cross_section_geometry_signal_connected = False

    def _disconnect_cross_section_layer_signals(self):
        if not self._cross_section_geometry_signal_connected:
            return
        layer = self._get_gpkg_layer('cross_sections')
        if layer is None:
            self._cross_section_geometry_signal_connected = False
            return
        try:
            layer.geometryChanged.disconnect(self._on_cross_section_geometry_changed)
        except Exception:
            pass
        self._cross_section_geometry_signal_connected = False

    def _populate_section_z_from_terrain(self, idx: int, announce: bool = True):
        if _SET_Z_FROM_RASTER_EXPR is None:
            if announce:
                ui_warning(self, 'Terrain Z', 'Could not load expressions/vertices_z_from_raster.py.')
            return False
        if self.model is None or idx < 0 or idx >= len(self.model.sections):
            return False
        raster_layer = self._selected_terrain_raster_layer()
        if raster_layer is None:
            if announce:
                ui_warning(self, 'Terrain Z', 'Select a terrain raster layer first.')
            return False
        if not self.loaded_gpkg_path:
            if announce:
                ui_warning(self, 'Terrain Z', 'Terrain sampling requires a GeoPackage-backed model with mapped cross-section geometry.')
            return False

        layer = self._ensure_gpkg_layer_loaded('cross_sections')
        if layer is None:
            if announce:
                ui_warning(self, 'Terrain Z', 'Could not find the cross_sections layer in the loaded GeoPackage.')
            return False

        xs = self.model.sections[idx]
        target_rs = str(xs.river_station)
        feature = self._find_cross_section_feature(layer, idx, xs)

        if feature is None:
            if announce:
                ui_warning(self, 'Terrain Z', f'Could not find cross-section feature {target_rs} in layer.')
            return False

        source_geom = feature.geometry()
        if source_geom is None or source_geom.isEmpty():
            if announce:
                ui_warning(self, 'Terrain Z', f'Cross-section {target_rs} has no geometry to sample.')
            return False

        try:
            sampled_geom = self._call_set_z_from_raster_expr(source_geom, raster_layer)
        except Exception as exc:
            if announce:
                ui_warning(self, 'Terrain Z', f'Failed to sample raster elevations: {exc}')
            return False

        if sampled_geom is None or sampled_geom.isEmpty():
            if announce:
                ui_warning(self, 'Terrain Z', f'No sampled geometry returned for {target_rs}.')
            return False

        profile = self._profile_from_map_geometry(sampled_geom)
        if len(profile) < 2:
            if announce:
                ui_warning(self, 'Terrain Z', f'Sampled geometry for {target_rs} has too few vertices.')
            return False

        self.model.sections[idx].geometry = profile
        river_station_text = self._compute_river_station_text_for_geometry(sampled_geom)
        if river_station_text is not None:
            self.model.sections[idx].river_station = river_station_text
            try:
                self.section_cb.setItemText(idx, river_station_text)
            except Exception:
                pass

        try:
            if layer.isEditable():
                river_idx = layer.fields().indexOf('river_station')
                layer.changeGeometry(feature.id(), sampled_geom)
                if river_idx != -1 and river_station_text is not None:
                    layer.changeAttributeValue(feature.id(), river_idx, river_station_text)
                self._mark_gpkg_dirty()
        except Exception:
            pass

        if self.section_cb.currentIndex() == idx:
            self.on_section_change(idx)

        if announce:
            ui_info(self, 'Terrain Z', f'Updated Z values for {target_rs} from raster layer "{raster_layer.name()}".')
        return True

    def on_populate_section_z_from_terrain(self):
        if self.model is None:
            ui_warning(self, 'No model', 'Load or create a model first')
            return
        idx = self.section_cb.currentIndex()
        if idx < 0:
            return
        self.push_undo()
        self._populate_section_z_from_terrain(idx, announce=True)

    def _get_qgis_iface(self):
        try:
            import qgis.utils as _qutils
            if getattr(_qutils, 'iface', None) is not None:
                return _qutils.iface
        except Exception:
            pass
        if ui_adapter is not None:
            return getattr(ui_adapter, 'iface', None)
        return None

    def _normalize_path(self, path: str) -> str:
        return os.path.normcase(os.path.abspath(path)) if path else ''

    def _is_geopackage_path(self, path: str) -> bool:
        return bool(path) and str(path).lower().endswith('.gpkg')

    def _load_results_from_loaded_geopackage(self):
        if not self.loaded_gpkg_path or not callable(load_results_from_geopackage):
            self.results = None
            try:
                self.populate_results_table()
            except Exception:
                pass
            return 0

        try:
            persisted = load_results_from_geopackage(self.loaded_gpkg_path)
        except Exception:
            persisted = []

        if persisted:
            self.results = persisted
            try:
                self.save_plot_btn.setEnabled(bool(HAVE_MPL))
            except Exception:
                pass
        else:
            self.results = None
            try:
                self.save_plot_btn.setEnabled(False)
            except Exception:
                pass

        try:
            self.populate_results_table()
        except Exception:
            pass
        try:
            self.refresh_plot()
        except Exception:
            pass
        return len(persisted)

    def _iter_loaded_gpkg_layers(self):
        iface = self._get_qgis_iface()
        if iface is None or not self.loaded_gpkg_path:
            return []
        try:
            from qgis.core import QgsProject, QgsVectorLayer
        except Exception:
            return []
        target = self._normalize_path(self.loaded_gpkg_path)
        matches = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            source_path = self._normalize_path(str(layer.source()).split('|', 1)[0])
            if source_path == target:
                matches.append(layer)
        return matches

    def _layer_has_unsaved_edits(self, layer) -> bool:
        try:
            if not layer.isEditable():
                return False
        except Exception:
            return False
        try:
            return bool(layer.isModified())
        except Exception:
            return False

    def _sync_geopackage_edit_flags_from_layers(self):
        if not self.loaded_gpkg_path:
            self.gpkg_editing_enabled = False
            self.gpkg_dirty = False
            return

        layers = self._iter_loaded_gpkg_layers()
        if not layers:
            return

        any_editable = False
        any_dirty = False
        for layer in layers:
            try:
                editable = bool(layer.isEditable())
            except Exception:
                editable = False
            if not editable:
                continue
            any_editable = True
            if self._layer_has_unsaved_edits(layer):
                any_dirty = True

        self.gpkg_editing_enabled = any_editable
        self.gpkg_dirty = any_dirty

    def _iter_project_vector_layers_for_path(self, gpkg_path: str):
        if not gpkg_path:
            return []
        try:
            from qgis.core import QgsProject, QgsVectorLayer
        except Exception:
            return []
        target = self._normalize_path(gpkg_path)
        matches = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            source_path = self._normalize_path(str(layer.source()).split('|', 1)[0])
            if source_path == target:
                matches.append(layer)
        return matches

    def _remove_project_layers_for_path(self, gpkg_path: str):
        if not gpkg_path:
            return
        try:
            from qgis.core import QgsProject
        except Exception:
            return
        for layer in self._iter_project_vector_layers_for_path(gpkg_path):
            try:
                QgsProject.instance().removeMapLayer(layer.id())
            except Exception:
                pass

    def _read_gpkg_layer_authid(self, gpkg_path: str, layer_name: str) -> str:
        if not gpkg_path:
            return ''
        try:
            from qgis.core import QgsVectorLayer
            lyr = QgsVectorLayer(f"{gpkg_path}|layername={layer_name}", layer_name, 'ogr')
            if not lyr.isValid() or not lyr.crs().isValid():
                return ''
            return str(lyr.crs().authid() or '')
        except Exception:
            return ''

    def _get_gpkg_layer(self, layer_name: str):
        for layer in self._iter_loaded_gpkg_layers():
            source = str(layer.source())
            if layer.name() == layer_name or f'layername={layer_name}' in source:
                return layer
        return None

    def _ensure_gpkg_layer_loaded(self, layer_name: str):
        layer = self._get_gpkg_layer(layer_name)
        if layer is not None:
            return layer
        iface = self._get_qgis_iface()
        if iface is None or not self.loaded_gpkg_path:
            return None
        try:
            from qgis.core import QgsProject, QgsVectorLayer
        except Exception:
            return None
        uri = f"{self.loaded_gpkg_path}|layername={layer_name}"
        layer = QgsVectorLayer(uri, layer_name, 'ogr')
        if not layer.isValid():
            return None
        QgsProject.instance().addMapLayer(layer)
        return layer

    def _ensure_layer_fields(self, layer, fields):
        try:
            from qgis.core import QgsField
            from qgis.PyQt.QtCore import QVariant
        except Exception:
            return
        existing = {field.name() for field in layer.fields()}
        missing = []
        for name, variant_type in fields:
            if name not in existing:
                missing.append(QgsField(name, variant_type))
        if missing:
            layer.dataProvider().addAttributes(missing)
            layer.updateFields()

    def _sync_boundary_from_ui(self):
        if self.model is None:
            return False
        if getattr(self, 'form_only_mode', False):
            return True
        self.model.boundary_condition = self.ds_bc.currentText()
        try:
            self.model.boundary_value = float(self.ds_val.text())
            self.model.flow_cfs = float(self.flow_edit.text())
        except Exception:
            ui_warning(self, 'Invalid', 'Boundary value and flow must be numeric')
            return False
        return True

    def _sync_ui_from_model(self):
        if self.model is None:
            return
        try:
            self.ds_bc.setCurrentText(str(self.model.boundary_condition))
        except Exception:
            pass
        try:
            self.ds_val.setText(str(float(self.model.boundary_value)))
        except Exception:
            pass
        try:
            self.flow_edit.setText(str(float(self.model.flow_cfs)))
        except Exception:
            pass

    def _update_geopackage_edit_state(self):
        try:
            self._sync_geopackage_edit_flags_from_layers()
        except Exception:
            pass
        gpkg_loaded = bool(self.loaded_gpkg_path)
        qgis_available = self._get_qgis_iface() is not None
        # GeoPackage-backed models are edited through QGIS layer attribute forms,
        # not through the plugin's in-widget section editors.
        model_editable = False
        self._set_model_editing_enabled(model_editable)
        self.toggle_gpkg_edit_btn.setEnabled(gpkg_loaded and qgis_available)
        self.save_layer_edits_btn.setEnabled(gpkg_loaded and qgis_available and self.gpkg_editing_enabled)
        if not self.can_gpkg:
            self.gpkg_label.setText('GeoPackage: missing runtime support')
        elif gpkg_loaded:
            self.gpkg_label.setText(f'GeoPackage: {os.path.basename(self.loaded_gpkg_path)}')
        else:
            self.gpkg_label.setText('GeoPackage: available')
        if not gpkg_loaded:
            self.layer_edit_status_label.setText('Layer editing: no GeoPackage loaded')
            self.toggle_gpkg_edit_btn.setText('Enable Layer Editing')
        elif not qgis_available:
            self.layer_edit_status_label.setText('Layer editing: available only inside QGIS')
            self.toggle_gpkg_edit_btn.setText('Enable Layer Editing')
        elif self.gpkg_editing_enabled:
            suffix = ' (unsaved)' if self.gpkg_dirty else ''
            self.layer_edit_status_label.setText(f'Layer editing: enabled in QGIS attribute forms{suffix}')
            self.toggle_gpkg_edit_btn.setText('Disable Layer Editing')
        else:
            self.layer_edit_status_label.setText('Layer editing: read-only')
            self.toggle_gpkg_edit_btn.setText('Enable Layer Editing')

        try:
            if hasattr(self, 'action_toggle_layer_editing') and self.action_toggle_layer_editing is not None:
                self.action_toggle_layer_editing.setEnabled(gpkg_loaded and qgis_available)
                if self.gpkg_editing_enabled:
                    self.action_toggle_layer_editing.setText('Disable Layer Editing')
                else:
                    self.action_toggle_layer_editing.setText('Enable Layer Editing')
        except Exception:
            pass

        try:
            if hasattr(self, 'action_save_layer_edits') and self.action_save_layer_edits is not None:
                self.action_save_layer_edits.setEnabled(gpkg_loaded and qgis_available and self.gpkg_editing_enabled)
        except Exception:
            pass

        try:
            if hasattr(self, 'action_save_model') and self.action_save_model is not None:
                self.action_save_model.setEnabled(bool(self.can_gpkg and self.model is not None))
        except Exception:
            pass

        try:
            if hasattr(self, 'action_open_results_plot') and self.action_open_results_plot is not None:
                self.action_open_results_plot.setEnabled(True)
        except Exception:
            pass

        try:
            if hasattr(self, 'action_open_results_table') and self.action_open_results_table is not None:
                self.action_open_results_table.setEnabled(True)
        except Exception:
            pass

    def _forms_file_path(self, filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), 'forms', filename)

    def _clear_layer_constraints(self, layer):
        try:
            from qgis.core import QgsFieldConstraints
        except Exception:
            return
        for idx in range(layer.fields().count()):
            try:
                layer.setConstraintExpression(idx, '')
            except Exception:
                pass
            for constraint in (
                QgsFieldConstraints.ConstraintNotNull,
                QgsFieldConstraints.ConstraintExpression,
                QgsFieldConstraints.ConstraintUnique,
            ):
                try:
                    layer.removeFieldConstraint(idx, constraint)
                except Exception:
                    pass

    def _set_value_map_editor(self, layer, field_name: str, mapping: dict):
        try:
            from qgis.core import QgsEditorWidgetSetup
        except Exception:
            return
        idx = layer.fields().indexOf(field_name)
        if idx == -1:
            return
        try:
            layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup('ValueMap', {'map': mapping}))
        except Exception:
            pass

    def _configure_layer_custom_ui(self, layer, cfg, QgsEditFormConfig):
        layer_name = layer.name()
        if layer_name not in ('cross_sections', 'boundary_conditions'):
            return

        ui_name = 'cross_sections_form.ui' if layer_name == 'cross_sections' else 'boundary_conditions_form.ui'
        init_function = 'backwater_cross_sections_form_open' if layer_name == 'cross_sections' else 'backwater_boundary_form_open'
        ui_path = self._forms_file_path(ui_name)
        init_path = self._forms_file_path('backwater_form_init.py')

        if os.path.exists(ui_path):
            try:
                cfg.setUiForm(ui_path)
            except Exception:
                pass

        if os.path.exists(init_path):
            try:
                if hasattr(cfg, 'setInitFilePath'):
                    cfg.setInitFilePath(init_path)
                if hasattr(cfg, 'setInitFunction'):
                    cfg.setInitFunction(init_function)
                if hasattr(cfg, 'setInitCodeSource') and hasattr(QgsEditFormConfig, 'CodeSourceFile'):
                    cfg.setInitCodeSource(QgsEditFormConfig.CodeSourceFile)
            except Exception:
                pass

    def _configure_layer_field_widgets_and_constraints(self, layer):
        if layer is None:
            return

        # Explicitly remove plugin-added hard/soft constraints so form editing
        # remains unconstrained except for provider-level constraints.
        self._clear_layer_constraints(layer)

        lname = layer.name()
        if lname == 'cross_sections':
            self._set_value_map_editor(layer, 'culvert_shape', {
                '(none)': '',
                'circular': 'circular',
                'rect': 'rect',
            })

        elif lname == 'boundary_conditions':
            self._set_value_map_editor(layer, 'boundary_type', {
                'known_wse': 'known_wse',
                'normal_depth': 'normal_depth',
            })

    def _configure_attribute_forms_for_layer(self, layer):
        if layer is None:
            return
        try:
            from qgis.core import QgsDefaultValue, QgsEditFormConfig
        except Exception:
            return

        try:
            cfg = layer.editFormConfig()
            # Prefer drag-and-drop style form layout when available.
            layout_value = None
            for enum_name in ('DragAndDrop', 'TabLayout', 'GeneratedLayout'):
                if hasattr(QgsEditFormConfig, enum_name):
                    layout_value = getattr(QgsEditFormConfig, enum_name)
                    break
            if layout_value is not None and hasattr(cfg, 'setLayout'):
                cfg.setLayout(layout_value)

            self._configure_layer_custom_ui(layer, cfg, QgsEditFormConfig)

            if hasattr(layer, 'setEditFormConfig'):
                layer.setEditFormConfig(cfg)

            if layer.name() == 'cross_sections':
                centerline_id_idx = layer.fields().indexOf('centerline_id')
                if centerline_id_idx != -1 and hasattr(layer, 'setDefaultValueDefinition'):
                    layer.setDefaultValueDefinition(centerline_id_idx, QgsDefaultValue('1', True))

                contraction_idx = layer.fields().indexOf('contraction_coeff')
                if contraction_idx != -1 and hasattr(layer, 'setDefaultValueDefinition'):
                    layer.setDefaultValueDefinition(contraction_idx, QgsDefaultValue('0.1', True))

                expansion_idx = layer.fields().indexOf('expansion_coeff')
                if expansion_idx != -1 and hasattr(layer, 'setDefaultValueDefinition'):
                    layer.setDefaultValueDefinition(expansion_idx, QgsDefaultValue('0.3', True))

                river_idx = layer.fields().indexOf('river_station')
                if river_idx != -1 and hasattr(layer, 'setDefaultValueDefinition'):
                    expr = (
                        "with_variable('cl_geom', geometry(get_feature"
                        "('centerline','centerline_id',coalesce(\"centerline_id\",1))), "
                        "if (@cl_geom is null, NULL, "
                        "with_variable('xpt', intersection($geometry, @cl_geom), "
                        "with_variable('loc_pt', if(@xpt is null OR is_empty(@xpt), "
                        "closest_point(@cl_geom, $geometry), centroid(@xpt)), "
                        "with_variable('rs', line_locate_point(@cl_geom, @loc_pt), "
                        "if(@rs < 0, NULL, to_string(round(@rs, 3))))))))"
                    )
                    layer.setDefaultValueDefinition(river_idx, QgsDefaultValue(expr, True))



            self._configure_layer_field_widgets_and_constraints(layer)

            self._configure_layer_form_actions(layer)
        except Exception:
            pass

    def _layer_python_action_type(self):
        try:
            from qgis.core import Qgis
            return Qgis.AttributeActionType.GenericPython
        except Exception:
            try:
                from qgis.core import QgsAction
                return QgsAction.GenericPython
            except Exception:
                return None

    def _upsert_layer_python_action(self, layer, name: str, command: str):
        if layer is None:
            return
        action_type = self._layer_python_action_type()
        if action_type is None:
            return
        try:
            from qgis.core import QgsAction
            manager = layer.actions()
        except Exception:
            return

        try:
            for action in list(manager.actions()):
                try:
                    if str(action.name()) == str(name):
                        manager.removeAction(action.id())
                except Exception:
                    continue
        except Exception:
            pass

        scopes = {'Form', 'Feature', 'Canvas', 'Layer'}
        try:
            action_obj = QgsAction(action_type, name, command, False)
            try:
                action_obj.setActionScopes(scopes)
            except Exception:
                pass
            manager.addAction(action_obj)
        except TypeError:
            try:
                manager.addAction(action_type, name, command, False)
                for action in list(manager.actions()):
                    try:
                        if str(action.name()) == str(name):
                            try:
                                action.setActionScopes(scopes)
                            except Exception:
                                pass
                    except Exception:
                        continue
            except TypeError:
                manager.addAction(action_type, name, command)
        except Exception:
            pass

    def _cross_section_select_terrain_action_code(self) -> str:
        return """from qgis.core import QgsProject, QgsRasterLayer, QgsExpressionContextUtils, Qgis
from qgis.PyQt.QtWidgets import QInputDialog
from qgis.utils import iface

project = QgsProject.instance()
rasters = [lyr for lyr in project.mapLayers().values() if isinstance(lyr, QgsRasterLayer) and lyr.isValid()]
if not rasters:
    iface.messageBar().pushMessage('Backwater', 'No valid raster layers are loaded.', level=Qgis.Warning, duration=6)
else:
    labels = [f"{lyr.name()} ({lyr.id()[:8]})" for lyr in rasters]
    current = QgsExpressionContextUtils.projectScope(project).variable('backwater_terrain_raster_id')
    default_idx = 0
    for i, lyr in enumerate(rasters):
        if lyr.id() == str(current):
            default_idx = i
            break
    selected_label, ok = QInputDialog.getItem(None, 'Backwater Terrain Raster', 'Raster layer:', labels, default_idx, False)
    if ok and selected_label:
        selected = rasters[labels.index(selected_label)]
        QgsExpressionContextUtils.setProjectVariable(project, 'backwater_terrain_raster_id', selected.id())
        iface.messageBar().pushMessage('Backwater', f'Terrain raster set to {selected.name()}.', level=Qgis.Info, duration=5)
"""

    def _cross_section_update_z_action_code(self) -> str:
        return """import os
import importlib.util
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsExpressionContextUtils,
    QgsFeatureRequest,
    QgsProject,
    QgsRasterLayer,
)
from qgis.utils import iface

layer = iface.activeLayer()
if layer is None:
    iface.messageBar().pushMessage('Backwater', 'No active layer.', level=Qgis.Warning, duration=6)
    raise RuntimeError('No active layer')

fid = int([% $id %])
feat = None
if fid > -1:
    feat = next(layer.getFeatures(QgsFeatureRequest(fid)), None)

# Fallback for form/action contexts where $id can be null/temporary.
if feat is None:
    selected = layer.selectedFeatures()
    if selected:
        feat = selected[0]
        try:
            fid = int(feat.id())
        except Exception:
            fid = -1

# Final fallback by river_station expression placeholder.
if feat is None:
    rs_hint = '[% coalesce("river_station", "") %]'.strip()
    if rs_hint:
        for candidate in layer.getFeatures():
            try:
                if str(candidate['river_station']) == rs_hint:
                    feat = candidate
                    try:
                        fid = int(candidate.id())
                    except Exception:
                        fid = -1
                    break
            except Exception:
                continue

if feat is None:
    iface.messageBar().pushMessage('Backwater', f'Feature {fid} not found.', level=Qgis.Warning, duration=6)
    raise RuntimeError('Cross section feature not found')

project = QgsProject.instance()
raster_id = str(QgsExpressionContextUtils.projectScope(project).variable('backwater_terrain_raster_id') or '').strip()
raster_layer = project.mapLayer(raster_id) if raster_id else None
if not isinstance(raster_layer, QgsRasterLayer) or not raster_layer.isValid():
    rasters = [lyr for lyr in project.mapLayers().values() if isinstance(lyr, QgsRasterLayer) and lyr.isValid()]
    if not rasters:
        iface.messageBar().pushMessage('Backwater', 'No valid raster layers are loaded.', level=Qgis.Warning, duration=6)
        raise RuntimeError('No raster layer available')
    raster_layer = rasters[0]
    QgsExpressionContextUtils.setProjectVariable(project, 'backwater_terrain_raster_id', raster_layer.id())

plugin_dir = os.path.join(QgsApplication.qgisSettingsDirPath(), 'python', 'plugins', 'qgis-backwater-plugin')
expr_path = os.path.join(plugin_dir, 'expressions', 'vertices_z_from_raster.py')
spec = importlib.util.spec_from_file_location('backwater_vertices_z_from_raster', expr_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Could not import raster expression module: {expr_path}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
func = getattr(mod, 'set_z_from_raster_expr_py', None)
if not callable(func):
    func = getattr(mod, '_set_z_from_raster_impl', None)
if not callable(func):
    func = getattr(mod, 'set_z_from_raster_expr', None)
if not callable(func):
    raise RuntimeError('set_z_from_raster_expr callable not found')

sampled_geom = func(feat.geometry(), raster_layer, 1, True)
if sampled_geom is None or sampled_geom.isEmpty():
    raise RuntimeError('Raster sampling returned empty geometry')

if not layer.isEditable():
    layer.startEditing()

if fid > -1:
    layer.changeGeometry(fid, sampled_geom)
else:
    feat.setGeometry(sampled_geom)

river_idx = layer.fields().indexOf('river_station')
center_idx = layer.fields().indexOf('centerline_id')

def _set_attr(field_idx, value):
    if field_idx == -1:
        return
    if fid > -1:
        layer.changeAttributeValue(fid, field_idx, value)
    else:
        feat.setAttribute(field_idx, value)

if center_idx != -1:
    try:
        cur_center_id = feat[center_idx]
    except Exception:
        cur_center_id = None
    if cur_center_id is None:
        _set_attr(center_idx, 1)

center_layer = None
for lyr in project.mapLayers().values():
    if getattr(lyr, 'name', lambda: '')() == 'centerline':
        center_layer = lyr
        break
if center_layer is not None:
    center_feat = next(center_layer.getFeatures(), None)
    if center_feat is not None:
        center_geom = center_feat.geometry()
        if center_geom is not None and not center_geom.isEmpty():
            locate_geom = None
            try:
                crossing = sampled_geom.intersection(center_geom)
            except Exception:
                crossing = None
            if crossing is not None and not crossing.isEmpty():
                locate_geom = crossing.centroid()
            if locate_geom is None or locate_geom.isEmpty():
                try:
                    locate_geom = center_geom.nearestPoint(sampled_geom)
                except Exception:
                    locate_geom = None
            if locate_geom is None or locate_geom.isEmpty():
                chainage = -1.0
            else:
                chainage = float(center_geom.lineLocatePoint(locate_geom))
            if chainage >= 0 and river_idx != -1:
                _set_attr(river_idx, f"{chainage:.3f}")

if fid <= -1:
    if not layer.updateFeature(feat):
        raise RuntimeError('Could not update geometry for temporary feature')

iface.messageBar().pushMessage('Backwater', f'Updated geometry Z from raster {raster_layer.name()}.', level=Qgis.Success, duration=5)
"""

    def _boundary_run_model_action_code(self) -> str:
        return """from qgis.core import Qgis
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.utils import iface

dock = iface.mainWindow().findChild(QDockWidget, 'BackwaterMainDock')
if dock is None or dock.widget() is None or not hasattr(dock.widget(), 'on_run'):
    iface.messageBar().pushMessage(
        'Backwater',
        'Backwater dock is not open. Open the plugin panel, then run this action again.',
        level=Qgis.Warning,
        duration=7,
    )
else:
    try:
        dock.widget().on_run()
    except Exception as exc:
        iface.messageBar().pushMessage('Backwater', f'Run Model failed: {exc}', level=Qgis.Critical, duration=8)
        raise
"""

    def _configure_layer_form_actions(self, layer):
        if layer is None:
            return
        lname = layer.name()
        if lname == 'cross_sections':
            self._upsert_layer_python_action(layer, 'Backwater: Select Terrain Raster', self._cross_section_select_terrain_action_code())
            self._upsert_layer_python_action(layer, 'Backwater: Update Z From Terrain', self._cross_section_update_z_action_code())
        elif lname == 'boundary_conditions':
            self._upsert_layer_python_action(layer, 'Backwater: Run Model', self._boundary_run_model_action_code())

    def _configure_cross_section_centerline_join(self):
        try:
            from qgis.core import QgsVectorLayerJoinInfo
            from qgis.PyQt.QtCore import QVariant
        except Exception:
            return

        cross_layer = self._ensure_gpkg_layer_loaded('cross_sections')
        center_layer = self._ensure_gpkg_layer_loaded('centerline')
        if cross_layer is None or center_layer is None:
            return

        try:
            self._ensure_layer_fields(cross_layer, [('centerline_id', QVariant.Int)])
            self._ensure_layer_fields(center_layer, [('centerline_id', QVariant.Int)])
        except Exception:
            pass

        try:
            for j in list(cross_layer.vectorJoins()):
                if getattr(j, 'joinLayerId', lambda: '')() == center_layer.id() or getattr(j, 'joinLayer', lambda: None)() == center_layer:
                    cross_layer.removeJoin(j.joinLayerId())
        except Exception:
            pass

        try:
            join_info = QgsVectorLayerJoinInfo()
            join_info.setJoinLayerId(center_layer.id())
            join_info.setJoinLayer(center_layer)
            join_info.setJoinFieldName('centerline_id')
            join_info.setTargetFieldName('centerline_id')
            join_info.setUsingMemoryCache(True)
            join_info.setPrefix('cl_')
            cross_layer.addJoin(join_info)
        except Exception:
            pass

    def _persist_layer_form_style(self, layer):
        if layer is None or not self.loaded_gpkg_path:
            return
        try:
            if hasattr(layer, 'saveStyleToDatabase'):
                layer.saveStyleToDatabase('backwater_form', 'Backwater form defaults and joins', True, '')
        except Exception:
            pass

    def _configure_attribute_forms_for_loaded_layers(self):
        self._configure_cross_section_centerline_join()
        for layer_name in ('cross_sections', 'centerline', 'boundary_conditions'):
            layer = self._ensure_gpkg_layer_loaded(layer_name)
            self._configure_attribute_forms_for_layer(layer)
            self._persist_layer_form_style(layer)

    def _select_crs_authid_for_new_model(self):
        iface = self._get_qgis_iface()
        if iface is None:
            return 'EPSG:4326'

        default_authid = 'EPSG:4326'
        try:
            default_authid = iface.mapCanvas().mapSettings().destinationCrs().authid() or default_authid
        except Exception:
            pass

        try:
            from qgis.gui import QgsProjectionSelectionDialog
            dlg = QgsProjectionSelectionDialog(self)
            try:
                dlg.setCrs(iface.mapCanvas().mapSettings().destinationCrs())
            except Exception:
                pass
            if dlg.exec():
                crs = dlg.crs()
                if crs and crs.isValid():
                    return crs.authid() or default_authid
            return None
        except Exception:
            pass

        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Model Projection',
            'Projection (for example EPSG:26912):',
            QtWidgets.QLineEdit.Normal,
            default_authid,
        )
        if not ok:
            return None
        text = str(text).strip()
        return text or default_authid

    def _get_loaded_centerline_geometry(self):
        if not self.loaded_gpkg_path:
            return None
        layer = self._get_gpkg_layer('centerline')
        if layer is None:
            layer = self._ensure_gpkg_layer_loaded('centerline')
        if layer is not None:
            feat = next(layer.getFeatures(), None)
            if feat is not None:
                try:
                    return feat.geometry()
                except Exception:
                    pass
        return None

    def _mark_gpkg_dirty(self):
        if self.loaded_gpkg_path and self.gpkg_editing_enabled:
            self.gpkg_dirty = True
            self._update_geopackage_edit_state()

    def _interp_profile_elevation(self, profile, station: float) -> float:
        if not profile:
            return 0.0
        pts = sorted([(float(st), float(z)) for st, z in profile], key=lambda p: p[0])
        if station <= pts[0][0]:
            return float(pts[0][1])
        if station >= pts[-1][0]:
            return float(pts[-1][1])
        for i in range(1, len(pts)):
            s0, z0 = pts[i - 1]
            s1, z1 = pts[i]
            if s1 <= s0:
                continue
            if station <= s1:
                t = (station - s0) / (s1 - s0)
                return float(z0 + (z1 - z0) * t)
        return float(pts[-1][1])

    def _geometry_with_preserved_xy_updated_z(self, source_geometry, profile):
        try:
            from qgis.core import QgsGeometry, QgsPoint
        except Exception:
            return None

        try:
            vertices = [v for v in source_geometry.vertices()]
        except Exception:
            vertices = []

        if len(vertices) < 2:
            return None

        # Build chainage along existing XY so map coordinates remain untouched.
        chainage = [0.0]
        for i in range(1, len(vertices)):
            dx = float(vertices[i].x()) - float(vertices[i - 1].x())
            dy = float(vertices[i].y()) - float(vertices[i - 1].y())
            chainage.append(chainage[-1] + math.hypot(dx, dy))

        updated_points = []
        for v, station in zip(vertices, chainage):
            z = self._interp_profile_elevation(profile, float(station))
            updated_points.append(QgsPoint(float(v.x()), float(v.y()), float(z)))

        try:
            return QgsGeometry.fromPolyline(updated_points)
        except Exception:
            return None

    def _write_cross_section_to_layer(self, idx: int, update_geometry: bool = True):
        if not (self.loaded_gpkg_path and self.gpkg_editing_enabled):
            return
        layer = self._ensure_gpkg_layer_loaded('cross_sections')
        if layer is None:
            raise RuntimeError('cross_sections layer not available in project')
        try:
            from qgis.core import QgsFeature, QgsGeometry, QgsPointXY
            from qgis.PyQt.QtCore import QVariant
        except Exception as exc:
            raise RuntimeError(f'QGIS API unavailable: {exc}')

        self._ensure_layer_fields(layer, [
            ('centerline_id', QVariant.Int),
            ('river_station', QVariant.String),
            ('left_bank_station', QVariant.Double),
            ('right_bank_station', QVariant.Double),
            ('n_lob', QVariant.Double),
            ('n_ch', QVariant.Double),
            ('n_rob', QVariant.Double),
            ('contraction_coeff', QVariant.Double),
            ('expansion_coeff', QVariant.Double),
            ('L_lob_to_next', QVariant.Double),
            ('L_ch_to_next', QVariant.Double),
            ('L_rob_to_next', QVariant.Double),
            ('culvert_code', QVariant.Int),
            ('culvert_shape', QVariant.String),
            ('culvert_diameter', QVariant.Double),
            ('culvert_width', QVariant.Double),
            ('culvert_height', QVariant.Double),
            ('culvert_upstream_invert', QVariant.Double),
            ('culvert_downstream_invert', QVariant.Double),
            ('culvert_length', QVariant.Double),
            ('culvert_weir_coeff', QVariant.Double),
            ('culvert_weir_sta_left', QVariant.Double),
            ('culvert_weir_sta_right', QVariant.Double),
        ])

        xs = self.model.sections[idx]
        target_feature = self._find_cross_section_feature(layer, idx, xs)

        source_geom_for_station = None
        if target_feature is not None:
            source_geom_for_station = target_feature.geometry()
        elif update_geometry:
            try:
                from qgis.core import QgsGeometry, QgsPointXY
                source_geom_for_station = QgsGeometry.fromPolylineXY([QgsPointXY(float(st), float(z)) for st, z in xs.geometry])
            except Exception:
                source_geom_for_station = None

        river_station_text = self._compute_river_station_text_for_geometry(source_geom_for_station)
        if river_station_text is None:
            river_station_text = str(xs.river_station)
        else:
            xs.river_station = str(river_station_text)
            try:
                self.section_cb.setItemText(idx, str(river_station_text))
            except Exception:
                pass

        attrs = {
            'centerline_id': 1,
            'river_station': str(river_station_text),
            'left_bank_station': float(xs.left_bank_station),
            'right_bank_station': float(xs.right_bank_station),
            'n_lob': float(xs.n_lob),
            'n_ch': float(xs.n_ch),
            'n_rob': float(xs.n_rob),
            'contraction_coeff': float(xs.contraction_coeff),
            'expansion_coeff': float(xs.expansion_coeff),
            'L_lob_to_next': float(xs.L_lob_to_next),
            'L_ch_to_next': float(xs.L_ch_to_next),
            'L_rob_to_next': float(xs.L_rob_to_next),
            'culvert_code': int(xs.culvert_code),
            'culvert_shape': str(xs.culvert_shape) if xs.culvert_shape else '',
            'culvert_diameter': float(xs.culvert_diameter),
            'culvert_width': float(xs.culvert_width),
            'culvert_height': float(xs.culvert_height),
            'culvert_upstream_invert': float(xs.culvert_upstream_invert),
            'culvert_downstream_invert': float(xs.culvert_downstream_invert),
            'culvert_length': float(xs.culvert_length),
            'culvert_weir_coeff': float(getattr(xs, 'culvert_weir_coeff', 3.0) or 3.0),
            'culvert_weir_sta_left': float(getattr(xs, 'culvert_weir_sta_left', 0.0) or 0.0),
            'culvert_weir_sta_right': float(getattr(xs, 'culvert_weir_sta_right', 0.0) or 0.0),
        }
        if target_feature is None:
            if update_geometry:
                geometry = QgsGeometry.fromPolylineXY([QgsPointXY(float(st), float(z)) for st, z in xs.geometry])
            else:
                geometry = QgsGeometry()
            feature = QgsFeature(layer.fields())
            if update_geometry:
                feature.setGeometry(geometry)
            for name, value in attrs.items():
                if layer.fields().indexOf(name) != -1:
                    feature[name] = value
            layer.addFeature(feature)
            return

        fid = target_feature.id()
        for name, value in attrs.items():
            field_index = layer.fields().indexOf(name)
            if field_index != -1:
                layer.changeAttributeValue(fid, field_index, value)
        if update_geometry:
            source_geom = target_feature.geometry()
            geometry = self._geometry_with_preserved_xy_updated_z(source_geom, xs.geometry)
            if geometry is None:
                raise RuntimeError(
                    f'Could not safely update geometry for river_station {xs.river_station}; '
                    'existing geometry is invalid or unsupported.'
                )
            layer.changeGeometry(fid, geometry)
            mapped_station = self._compute_river_station_text_for_geometry(geometry)
            if mapped_station is not None:
                river_idx = layer.fields().indexOf('river_station')
                if river_idx != -1:
                    layer.changeAttributeValue(fid, river_idx, mapped_station)
                xs.river_station = mapped_station
                try:
                    self.section_cb.setItemText(idx, mapped_station)
                except Exception:
                    pass

    def _write_boundary_to_layer(self):
        if not (self.loaded_gpkg_path and self.gpkg_editing_enabled):
            return
        if not self._sync_boundary_from_ui():
            raise RuntimeError('Boundary values are invalid')
        layer = self._ensure_gpkg_layer_loaded('boundary_conditions')
        if layer is None:
            raise RuntimeError('boundary_conditions layer not available in project')
        try:
            from qgis.core import QgsFeature, QgsGeometry, QgsPointXY
            from qgis.PyQt.QtCore import QVariant
        except Exception as exc:
            raise RuntimeError(f'QGIS API unavailable: {exc}')

        self._ensure_layer_fields(layer, [
            ('flow_cfs', QVariant.Double),
            ('boundary_type', QVariant.String),
            ('boundary_value', QVariant.Double),
        ])

        features = list(layer.getFeatures())
        attrs = {
            'flow_cfs': float(self.model.flow_cfs),
            'boundary_type': str(self.model.boundary_condition),
            'boundary_value': float(self.model.boundary_value),
        }
        if features:
            fid = features[0].id()
            for name, value in attrs.items():
                field_index = layer.fields().indexOf(name)
                if field_index != -1:
                    layer.changeAttributeValue(fid, field_index, value)
            return

        feature = QgsFeature(layer.fields())
        try:
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(0.0, 0.0)))
        except Exception:
            pass
        for name, value in attrs.items():
            if layer.fields().indexOf(name) != -1:
                feature[name] = value
        layer.addFeature(feature)

    def _reload_model_from_loaded_geopackage(self):
        if not self.loaded_gpkg_path:
            return
        self.model = load_input(self.loaded_gpkg_path)
        self.section_cb.clear()
        self.section_cb.addItems([xs.river_station for xs in self.model.sections])
        if self.model.sections:
            self.section_cb.setCurrentIndex(0)
        self._load_results_from_loaded_geopackage()
        self._refresh_scroller_choices()

    def on_toggle_geopackage_editing(self):
        if not self.loaded_gpkg_path:
            ui_warning(self, 'GeoPackage required', 'Load a GeoPackage-backed model first.')
            return
        if self._get_qgis_iface() is None:
            ui_warning(self, 'QGIS required', 'Layer editing is only available when the plugin is running inside QGIS.')
            return
        layer_names = ('cross_sections', 'centerline', 'boundary_conditions')
        if not self.gpkg_editing_enabled:
            try:
                self._configure_cross_section_centerline_join()
                layers = []
                for layer_name in layer_names:
                    layer = self._ensure_gpkg_layer_loaded(layer_name)
                    if layer is None:
                        raise RuntimeError(f'{layer_name} layer could not be loaded from {self.loaded_gpkg_path}')
                    self._configure_attribute_forms_for_layer(layer)
                    if not layer.isEditable():
                        layer.startEditing()
                    layers.append(layer)
                self._connect_cross_section_layer_signals()
                self.gpkg_editing_enabled = True
                self.gpkg_dirty = False
                self._update_geopackage_edit_state()
                ui_info(self, 'Layer editing', 'GeoPackage layers are now editable. Use QGIS attribute forms to edit model features.')
            except Exception as exc:
                ui_critical(self, 'Layer editing', str(exc))
            return

        self._disconnect_cross_section_layer_signals()
        for layer_name in layer_names:
            layer = self._get_gpkg_layer(layer_name)
            if layer is not None and layer.isEditable():
                layer.rollBack()
        self.gpkg_editing_enabled = False
        self.gpkg_dirty = False
        try:
            self._reload_model_from_loaded_geopackage()
        except Exception:
            pass
        self._update_geopackage_edit_state()
        ui_info(self, 'Layer editing', 'Layer editing disabled. Unsaved layer edits were discarded.')

    def on_save_layer_edits(self):
        if not (self.loaded_gpkg_path and self.gpkg_editing_enabled):
            ui_warning(self, 'Layer editing', 'Enable GeoPackage layer editing first.')
            return
        try:
            if not getattr(self, 'form_only_mode', False):
                self._write_boundary_to_layer()
            self._disconnect_cross_section_layer_signals()
            for layer_name in ('cross_sections', 'centerline', 'boundary_conditions'):
                layer = self._get_gpkg_layer(layer_name)
                if layer is not None and layer.isEditable() and not layer.commitChanges():
                    errors = '; '.join(layer.commitErrors()) if hasattr(layer, 'commitErrors') else 'commit failed'
                    raise RuntimeError(f'{layer_name} commit failed: {errors}')
            self.gpkg_editing_enabled = False
            self.gpkg_dirty = False
            self._reload_model_from_loaded_geopackage()
            self._update_geopackage_edit_state()
            ui_info(self, 'Saved', f'Layer edits saved to {self.loaded_gpkg_path}')
        except Exception as exc:
            try:
                if self.gpkg_editing_enabled:
                    self._connect_cross_section_layer_signals()
            except Exception:
                pass
            ui_critical(self, 'Save layer edits', str(exc))

    # --- model actions
    def on_menu_open_model(self):
        filters = 'GeoPackage Files (*.gpkg)'
        p, _ = ui_get_open_filename(self, 'Open Model GeoPackage', filters)
        if not p:
            return
        self.input_path.setText(p)
        self.on_load()

    def on_browse(self):
        filters = 'GeoPackage Files (*.gpkg)'
        p, _ = ui_get_open_filename(self, 'Open Model', filters)
        if p:
            self.input_path.setText(p)

    def on_new_model(self):
        if not getattr(self, 'can_gpkg', False) or save_to_geopackage is None:
            ui_warning(self, 'Not available', 'Creating a model GeoPackage requires PyQGIS support.')
            return

        gpkg_path, _ = ui_get_save_filename(self, 'Create New Model GeoPackage', 'GeoPackage Files (*.gpkg)')
        if not gpkg_path:
            return
        if not gpkg_path.lower().endswith('.gpkg'):
            gpkg_path += '.gpkg'

        crs_authid = self._select_crs_authid_for_new_model()
        if not crs_authid:
            return

        base_flow = 500.0
        if not getattr(self, 'form_only_mode', False):
            try:
                base_flow = float(self.flow_edit.text())
            except Exception:
                base_flow = 500.0

        centerline_geom = None
        try:
            from qgis.core import QgsGeometry
            centerline_geom = QgsGeometry.fromWkt('LINESTRING (0 0, 20 0)')
        except Exception:
            centerline_geom = None
        if centerline_geom is None:
            ui_critical(self, 'Create model', 'Could not create centerline geometry via PyQGIS.')
            return

        xs0 = CrossSection(
            river_station='S_down',
            geometry=[(0.0, 100.0), (10.0, 99.5)],
            left_bank_station=2.0,
            right_bank_station=8.0,
            n_lob=0.035, n_ch=0.035, n_rob=0.035,
            contraction_coeff=0.1, expansion_coeff=0.3,
            L_lob_to_next=0.0, L_ch_to_next=0.0, L_rob_to_next=0.0
        )
        xs1 = CrossSection(
            river_station='S_up',
            geometry=[(10.0, 99.5), (20.0, 99.0)],
            left_bank_station=12.0,
            right_bank_station=18.0,
            n_lob=0.035, n_ch=0.035, n_rob=0.035,
            contraction_coeff=0.1, expansion_coeff=0.3,
            L_lob_to_next=0.0, L_ch_to_next=0.0, L_rob_to_next=0.0
        )
        model = ModelInput(
            flow_cfs=base_flow,
            flow_change=None,
            boundary_condition='known_wse',
            boundary_value=100.0,
            sections=[xs0, xs1]
        )

        # If this path is already loaded in the project, drop stale layers so
        # the newly written file CRS and schema are re-read from disk.
        self._remove_project_layers_for_path(gpkg_path)

        try:
            save_to_geopackage(
                gpkg_path,
                model,
                centerline_geom=centerline_geom,
                overwrite=True,
                crs_authid=crs_authid,
            )
        except Exception as exc:
            ui_critical(self, 'Create model', str(exc))
            return

        actual_authid = self._read_gpkg_layer_authid(gpkg_path, 'cross_sections')
        if actual_authid and str(actual_authid).upper() != str(crs_authid).upper():
            ui_warning(
                self,
                'CRS mismatch',
                f'Created GeoPackage layer CRS is {actual_authid}, but selected CRS was {crs_authid}. '
                'The file was written, but projection may differ from selection.'
            )

        self.input_path.setText(gpkg_path)
        self.on_load()
        self._configure_attribute_forms_for_loaded_layers()
        if self._get_qgis_iface() is not None and not self.gpkg_editing_enabled:
            self.on_toggle_geopackage_editing()
        ui_info(self, 'Created', f'Created new model GeoPackage at {gpkg_path} ({crs_authid}).')

    def on_load(self):
        # Ensure we can rebind module-level loader symbols if needed
        global load_from_geopackage, save_to_geopackage, load_results_from_geopackage, save_results_to_geopackage, load_input
        p = self.input_path.text().strip()
        if not p:
            ui_warning(self, 'Input required', 'Please choose an input GeoPackage file or create a new model')
            return
        if not self._is_geopackage_path(p):
            ui_warning(self, 'GeoPackage required', 'Only GeoPackage (*.gpkg) models are supported.')
            return
        self._remove_project_layers_for_path(p)
        try:
            self.model = load_input(p)
            self.loaded_gpkg_path = self._normalize_path(p)
            self.gpkg_editing_enabled = False
            self.gpkg_dirty = False
            self.results = None
            self._sync_ui_from_model()
            self.section_cb.clear()
            self.section_cb.addItems([xs.river_station for xs in self.model.sections])
            if self.model.sections:
                self.section_cb.setCurrentIndex(0)
            result_count = self._load_results_from_loaded_geopackage()
            ui_info(self, 'Loaded', f'Loaded model: {p}')
            if result_count > 0:
                ui_info(self, 'Results', f'Loaded {result_count} persisted result row(s) from model_results.')
            try:
                self.refresh_plot()
            except Exception:
                pass
                try:
                    self.update_cross_section_tab_state()
                except Exception:
                    pass
            # push initial loaded state
            self.push_undo()
            self._update_geopackage_edit_state()
            self._configure_attribute_forms_for_loaded_layers()
            self._refresh_scroller_choices()
            self.refresh_terrain_raster_choices()
        except Exception as e:
            ui_critical(self, 'Error', str(e))

    def on_run(self):
        try:
            self.status_label.setText('Running...')
        except Exception:
            pass

        try:
            self._sync_geopackage_edit_flags_from_layers()
            self._update_geopackage_edit_state()
        except Exception:
            pass

        if self.gpkg_editing_enabled and self.gpkg_dirty:
            ui_warning(self, 'Unsaved edits', 'Save GeoPackage layer edits before running the model.')
            return

        # ensure these names refer to the plugin-local solver module symbols
        global run_backwater, load_input, load_from_geopackage, save_to_geopackage, load_results_from_geopackage, save_results_to_geopackage, ModelInput, CrossSection, HAVE_MPL

        if self.model is None:
            p = self.input_path.text().strip()
            if not p:
                ui_warning(self, 'Input required', 'Please load a model or create a new one')
                return
            if not self._is_geopackage_path(p):
                ui_warning(self, 'GeoPackage required', 'Only GeoPackage (*.gpkg) models are supported.')
                return
            try:
                self.model = load_input(p)
                self.section_cb.clear()
                self.section_cb.addItems([xs.river_station for xs in self.model.sections])
            except Exception as e:
                ui_critical(self, 'Error', str(e))
                return

        # GeoPackage-only mode: always reload model from disk before run so
        # the solver uses the latest committed layer edits.
        try:
            if self.loaded_gpkg_path:
                self.model = load_input(self.loaded_gpkg_path)
                self.section_cb.clear()
                self.section_cb.addItems([xs.river_station for xs in self.model.sections])
        except Exception as e:
            ui_critical(self, 'Error', f'Failed to reload GeoPackage before run: {e}')
            return

        # Optional in-widget BC/flow override is disabled in form-only mode.
        if not getattr(self, 'form_only_mode', False):
            self.model.boundary_condition = self.ds_bc.currentText()
            try:
                self.model.boundary_value = float(self.ds_val.text())
            except Exception:
                ui_warning(self, 'Invalid', 'DS value must be numeric')
                return
            try:
                self.model.flow_cfs = float(self.flow_edit.text())
            except Exception:
                ui_warning(self, 'Invalid', 'Flow must be numeric')
                return

        # GUI validation: ensure known_wse downstream value is not below the downstream
        # section minimum bed elevation (non-physical). Provide a clear warning.
        try:
            if getattr(self.model, 'boundary_condition', '') == 'known_wse' and getattr(self.model, 'sections', None):
                ds_wse = float(self.model.boundary_value)
                try:
                    zmin_dn = min(z for _, z in self.model.sections[0].geometry)
                except Exception:
                    zmin_dn = None
                if zmin_dn is not None and ds_wse < zmin_dn:
                    ui_warning(self, 'Invalid DS WSE', f'Downstream WSE ({ds_wse}) is below minimum bed ({zmin_dn:.3f}). Set DS WSE >= {zmin_dn:.3f} or choose normal_depth.')
                    return
        except Exception:
            # If validation fails for unexpected reasons, don't block run here; let solver surface errors.
            pass

        # Ensure run_backwater is available; try to import/reload the plugin-local module if missing
        # Always reload backwater2 so that on-disk edits to the solver are
        # picked up without restarting QGIS.
        try:
            import importlib.util
            _mod_path = os.path.join(os.path.dirname(__file__), 'backwater2.py')
            _spec = importlib.util.spec_from_file_location('qgis_backwater_plugin.backwater2', _mod_path)
            if _spec is None or _spec.loader is None:
                raise ImportError(f'Could not create import spec for {_mod_path}')
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules['qgis_backwater_plugin.backwater2'] = _mod
            _spec.loader.exec_module(_mod)
            self._solver_module = _mod
            run_backwater = getattr(_mod, 'run_backwater', None)
            load_input = getattr(_mod, 'load_input', load_input)
            load_from_geopackage = getattr(_mod, 'load_from_geopackage', load_from_geopackage)
            save_to_geopackage = getattr(_mod, 'save_to_geopackage', save_to_geopackage)
            load_results_from_geopackage = getattr(_mod, 'load_results_from_geopackage', load_results_from_geopackage)
            save_results_to_geopackage = getattr(_mod, 'save_results_to_geopackage', save_results_to_geopackage)
            ModelInput = getattr(_mod, 'ModelInput', ModelInput)
            CrossSection = getattr(_mod, 'CrossSection', CrossSection)
            HAVE_MPL = getattr(_mod, 'HAVE_MPL', HAVE_MPL)
        except Exception:
            pass

        # Fallback if reload failed and run_backwater is still missing
        if not (('run_backwater' in globals() and callable(run_backwater))):
            try:
                import importlib
                _mod = importlib.import_module('qgis_backwater_plugin.backwater2')
                self._solver_module = _mod
                run_backwater = getattr(_mod, 'run_backwater', None)
                load_input = getattr(_mod, 'load_input', load_input)
                load_from_geopackage = getattr(_mod, 'load_from_geopackage', load_from_geopackage)
                save_to_geopackage = getattr(_mod, 'save_to_geopackage', save_to_geopackage)
                load_results_from_geopackage = getattr(_mod, 'load_results_from_geopackage', load_results_from_geopackage)
                save_results_to_geopackage = getattr(_mod, 'save_results_to_geopackage', save_results_to_geopackage)
                ModelInput = getattr(_mod, 'ModelInput', ModelInput)
                CrossSection = getattr(_mod, 'CrossSection', CrossSection)
                HAVE_MPL = getattr(_mod, 'HAVE_MPL', HAVE_MPL)
                # Propagate GUI method selections into the solver module
                try:
                    # self.alpha_combo / self.sf_combo exist on the Qt GUI
                    _alpha = str(self.alpha_combo.currentText()) if hasattr(self, 'alpha_combo') else None
                    _sf = str(self.sf_combo.currentText()) if hasattr(self, 'sf_combo') else None
                    if _alpha:
                        setattr(_mod, 'ALPHA_METHOD', _alpha)
                    if _sf:
                        setattr(_mod, 'SF_METHOD', _sf)
                except Exception:
                    pass
            except Exception as e:
                ui_warning(self, 'Import failed', f'Could not import qgis_backwater_plugin.backwater2: {e}')

        if not (('run_backwater' in globals() and callable(run_backwater))):
            ui_warning(self, 'Not available', 'Solver not available (backwater2.run_backwater missing)')
            return

        # If an in-memory model was loaded before the solver reload, culvert XS
        # objects may not carry newly added fields. Reload from disk when we
        # detect missing weir station attributes.
        try:
            need_model_reload = False
            if self.model is None:
                need_model_reload = True
            else:
                for _xs in getattr(self.model, 'sections', []):
                    if _xs.has_culvert() and (
                        not hasattr(_xs, 'culvert_weir_sta_left')
                        or not hasattr(_xs, 'culvert_weir_sta_right')
                    ):
                        need_model_reload = True
                        break

            if need_model_reload:
                p = self.input_path.text().strip()
                if p:
                    self.model = load_input(p)
                    self.section_cb.clear()
                    self.section_cb.addItems([xs.river_station for xs in self.model.sections])
        except Exception:
            pass

        try:
            # Ensure solver module sees current GUI method/solver selections
            try:
                _mod = getattr(self, '_solver_module', None)
                try:
                    _ensure_culvert = getattr(_mod, '_ensure_culvert_runtime', None)
                    if callable(_ensure_culvert):
                        _ensure_culvert()
                except Exception:
                    pass
                if hasattr(self, 'alpha_combo'):
                    setattr(_mod, 'ALPHA_METHOD', str(self.alpha_combo.currentText()))
                if hasattr(self, 'sf_combo'):
                    setattr(_mod, 'SF_METHOD', str(self.sf_combo.currentText()))
            except Exception:
                pass

            # Runtime debug trace: confirm the actual culvert weir inputs used.
            try:
                for _xs in getattr(self.model, 'sections', []):
                    if _xs.has_culvert():
                        print(
                            f"Culvert run inputs RS={_xs.river_station}: "
                            f"Cw={getattr(_xs, 'culvert_weir_coeff', None)}, "
                            f"sta_left={getattr(_xs, 'culvert_weir_sta_left', None)}, "
                            f"sta_right={getattr(_xs, 'culvert_weir_sta_right', None)}"
                        )
            except Exception:
                pass

            solver_choice = str(self.solver_combo.currentText()) if hasattr(self, 'solver_combo') else 'py'
            self.results = run_backwater(self.model, solver=solver_choice)

            if self.loaded_gpkg_path and callable(save_results_to_geopackage):
                try:
                    save_results_to_geopackage(
                        self.loaded_gpkg_path,
                        self.model,
                        self.results,
                        layer_name='model_results',
                        solver=solver_choice,
                    )
                except Exception as save_exc:
                    ui_warning(self, 'Results persistence', f'Run completed, but could not save model_results layer: {save_exc}')
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            ui_critical(self, 'Run error', f"{e}\n{tb}")
            return

        # update detail plot overlays
        # ensure current model state is on undo stack (so user can undo after run if they made changes beforehand)
        if not self.undo_stack:
            self.push_undo()

        # populate tabular results and show Results tab
        try:
            self.populate_results_table()
        except Exception:
            pass
        try:
            # show results tab in bottom-right IO tabs
            self.io_tabs.setCurrentWidget(self.results_page)
        except Exception:
            pass
        try:
            self._update_geopackage_edit_state()
        except Exception:
            pass
        # update small run summary in Results tab
        try:
            wse_vals = [s.wse for s in self.results if hasattr(s, 'wse')]
            run_lines = []
            if wse_vals:
                mn = min(wse_vals); mx = max(wse_vals)
                run_lines.append(f'Run: {len(wse_vals)} sections — WSE min {mn:.3f}, max {mx:.3f}')
            else:
                run_lines.append('Run complete — no WSE values')

            # Make diagnostics visible in GUI (stdout is often hidden in QGIS).
            try:
                run_lines.append(f"Solver: {solver_choice}")
                run_lines.append(f"Input: {self.input_path.text().strip()}")
                _solver_mod = getattr(self, '_solver_module', None)
                run_lines.append(f"Solver module: {getattr(_solver_mod, '__file__', 'unknown')}")
                try:
                    run_lines.append(f"Flow: {float(getattr(self.model, 'flow_cfs', float('nan'))):.3f} cfs")
                except Exception:
                    pass
                try:
                    run_lines.append(f"HAVE_CULVERT: {bool(getattr(_solver_mod, 'HAVE_CULVERT', False))}")
                except Exception:
                    pass
            except Exception:
                pass

            try:
                rs_diag = None
                for idx, _xs in enumerate(getattr(self.model, 'sections', [])):
                    if str(getattr(_xs, 'river_station', '')) == '89.3':
                        _wse = self.results[idx].wse if idx < len(self.results) else float('nan')
                        _geom = list(getattr(_xs, 'geometry', []) or [])
                        _zmin = min((float(z) for _x, z in _geom), default=float('nan'))
                        _zmax = max((float(z) for _x, z in _geom), default=float('nan'))
                        _z_crown = float(getattr(_xs, 'culvert_upstream_invert', 0.0) or 0.0) + max(
                            float(getattr(_xs, 'culvert_height', 0.0) or 0.0),
                            float(getattr(_xs, 'culvert_diameter', 0.0) or 0.0),
                        )
                        if _geom and _zmin > float(getattr(_xs, 'culvert_upstream_invert', 0.0) or 0.0):
                            _z_crown = _zmin

                        _qweir_80495 = float('nan')
                        _wse_crit = float('nan')
                        try:
                            _solver_mod = getattr(self, '_solver_module', None)
                            if _solver_mod is not None:
                                _qweir_80495 = float(_solver_mod.irregular_weir_flow_from_geometry(
                                    xs_culvert=_xs,
                                    headwater_wse=804.95,
                                    z_crown_inlet=_z_crown,
                                    Cw=float(getattr(_xs, 'culvert_weir_coeff', 3.0) or 3.0),
                                    sta_left=float(getattr(_xs, 'culvert_weir_sta_left', 0.0) or 0.0),
                                    sta_right=float(getattr(_xs, 'culvert_weir_sta_right', 0.0) or 0.0),
                                ))
                                _wse_crit = float(_solver_mod.solve_critical_depth(_xs, float(self.model.flow_cfs), z_guess=max(_z_crown, _zmin + 0.5)))
                        except Exception:
                            pass

                        rs_diag = (
                            f"RS 89.3 diag: has_culvert={_xs.has_culvert()}, "
                            f"Cw={getattr(_xs, 'culvert_weir_coeff', None)}, "
                            f"sta_left={getattr(_xs, 'culvert_weir_sta_left', None)}, "
                            f"sta_right={getattr(_xs, 'culvert_weir_sta_right', None)}, "
                            f"WSE={_wse:.3f}, zmin={_zmin:.3f}, zmax={_zmax:.3f}, "
                            f"z_crown={_z_crown:.3f}, wse_crit={_wse_crit:.3f}, qweir@804.95={_qweir_80495:.2f}"
                        )
                        break
                if rs_diag is not None:
                    run_lines.append(rs_diag)
            except Exception:
                pass

            self.results_status_label.setText('\n'.join(run_lines))
        except Exception:
            try:
                self.results_status_label.setText('Run complete')
            except Exception:
                pass

        # plotting
        if HAVE_MPL:
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                import matplotlib.pyplot as plt
                import numpy as _np

                n = len(self.model.sections)
                # compute chainage using channel reach lengths; fallback to cumulative section index
                chainage = [0.0]
                for i in range(1, n):
                    prev = self.model.sections[i-1]
                    try:
                        chainage.append(chainage[-1] + float(prev.L_ch_to_next))
                    except Exception:
                        chainage.append(chainage[-1] + 1.0)

                # Create figure: top row = WSE profile, below = per-section plots
                fig = plt.figure(figsize=(10, 2 + 1.2 * (n if n>0 else 1)))
                gs = fig.add_gridspec(n + 1, 1, height_ratios=[1] + [0.9] * n, hspace=0.25)

                # Top: WSE profile
                ax0 = fig.add_subplot(gs[0, 0])
                wse_vals = [s.wse for s in self.results]
                # compute channel-bottom profile (min bed elevation inside channel bounds)
                bottom_vals = []
                for i, xs in enumerate(self.model.sections):
                    try:
                        geom = sorted(xs.geometry, key=lambda p: p[0])
                        sx = _np.array([p[0] for p in geom])
                        sz = _np.array([p[1] for p in geom])
                        lb = float(xs.left_bank_station)
                        rb = float(xs.right_bank_station)
                        mask = (sx >= lb) & (sx <= rb)
                        if mask.any():
                            bottom_vals.append(float(sz[mask].min()))
                        else:
                            bottom_vals.append(float(sz.min()))
                    except Exception:
                        bottom_vals.append(0.0)

                ax0.plot(chainage[:len(wse_vals)], wse_vals, '-o', color='#1f77b4', lw=2, markersize=6)
                # Energy grade line (E = wse + alpha * V^2 / (2g))
                try:
                    G_val = getattr(_bwmod, 'G', 32.174)
                    energy_vals = [s.wse + (getattr(s, 'alpha', 0.0) * getattr(s, 'V_t', 0.0) ** 2) / (2.0 * G_val) for s in self.results]
                    ax0.plot(chainage[:len(energy_vals)], energy_vals, '--', color='orange', lw=1.8, label='Energy Grade')
                    ax0.legend()
                except Exception:
                    pass
                if bottom_vals:
                    try:
                        ax0.plot(chainage[:len(bottom_vals)], bottom_vals, '--', color='saddlebrown', lw=1.5, label='Channel bottom')
                        ax0.legend()
                    except Exception:
                        pass
                ax0.set_xlabel('Chainage (ft)')
                ax0.set_ylabel('Water Surface Elevation (ft)')
                ax0.grid(True, linestyle='--', alpha=0.5)
                ax0.set_title('Water Surface Profile')

                # Per-section plots
                ymin = min(min(z for _, z in xs.geometry) for xs in self.model.sections)
                for i, xs in enumerate(self.model.sections):
                    ax = fig.add_subplot(gs[i+1, 0])
                    geom = sorted(xs.geometry, key=lambda p: p[0])
                    sx = _np.array([p[0] for p in geom])
                    sz = _np.array([p[1] for p in geom])
                    # bed line
                    ax.plot(sx, sz, '-k', lw=1)
                    # ground fill
                    ax.fill_between(sx, sz, ymin - 2.0, color='#efefef')
                    # channel shading between left/right bank
                    try:
                        lb = float(xs.left_bank_station)
                        rb = float(xs.right_bank_station)
                        ax.axvspan(lb, rb, color='#fafafa', alpha=0.7)
                        ax.axvline(lb, color='0.6', linestyle='--')
                        ax.axvline(rb, color='0.6', linestyle='--')
                    except Exception:
                        pass
                    # water surface line and fill if available
                    if self.results is not None and i < len(self.results):
                        wse = self.results[i].wse
                        ax.axhline(wse, color='#1f77b4', linestyle='-', lw=1.5)
                        # fill submerged areas
                        where_sub = sz < wse
                        if where_sub.any():
                            ax.fill_between(sx, sz, wse, where=where_sub, interpolate=True, color='#1f77b4', alpha=0.35)
                    # Culvert opening overlay at section center station.
                    try:
                        if getattr(xs, 'has_culvert', lambda: False)() and len(sx) > 0:
                            center_x = 0.5 * (float(sx.min()) + float(sx.max()))
                            y_full = max(
                                float(getattr(xs, 'culvert_height', 0.0) or 0.0),
                                float(getattr(xs, 'culvert_diameter', 0.0) or 0.0),
                            )
                            up_inv = float(getattr(xs, 'culvert_upstream_invert', 0.0) or 0.0)
                            dn_inv = float(getattr(xs, 'culvert_downstream_invert', 0.0) or 0.0)
                            if y_full > 0.0:
                                dx = max(0.02 * max(float(sx.max()) - float(sx.min()), 1.0), 0.1)
                                ax.plot([center_x - dx, center_x - dx], [up_inv, up_inv + y_full], '-', color='crimson', lw=2.0)
                                ax.plot([center_x + dx, center_x + dx], [dn_inv, dn_inv + y_full], ':', color='crimson', lw=2.0)
                    except Exception:
                        pass
                    # labels and styling
                    # Add a visible label/title for each cross-section subplot
                    try:
                        ax.set_title(xs.river_station, loc='left', fontsize=9)
                    except Exception:
                        try:
                            ax.set_ylabel(xs.river_station, rotation=0, labelpad=50, va='center')
                        except Exception:
                            pass
                    ax.get_xaxis().set_visible(False)
                    ax.grid(False)

                # show x-axis labels on bottom plot
                if n > 0:
                    ax.set_xlabel('Station (ft)')
                    ax.get_xaxis().set_visible(True)

                fig.tight_layout()
                # clear old plot area (results plot page)
                for i in reversed(range(self.plot_page_layout.count())):
                    w = self.plot_page_layout.itemAt(i).widget()
                    if w is not None:
                        w.setParent(None)

                # Add matplotlib navigation toolbar for pan/zoom inside results page
                canvas = FigureCanvas(fig)
                try:
                    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
                    toolbar = NavigationToolbar(canvas, self.plot_page)
                    self.plot_page_layout.addWidget(toolbar)
                except Exception:
                    toolbar = None

                # Make the canvas scrollable by placing it inside a QScrollArea
                try:
                    scroll = QtWidgets.QScrollArea()
                    scroll.setWidgetResizable(True)
                    holder = CanvasHolder()
                    holder.set_canvas(canvas)
                    scroll.setWidget(holder)
                    self.plot_page_layout.addWidget(scroll)
                except Exception:
                    # fallback to direct add
                    self.plot_page_layout.addWidget(canvas)
                try:
                    canvas.draw()
                except Exception:
                    pass

                # store plot interactive state
                self._plot_canvas = canvas
                self._plot_fig = fig
                self._plot_axes = {'profile_ax': ax0, 'section_axes': []}
                # collect per-section axes data for interaction
                for i, xs in enumerate(self.model.sections):
                    ax = fig.axes[i+1]
                    geom = sorted(xs.geometry, key=lambda p: p[0])
                    sx = _np.array([p[0] for p in geom])
                    sz = _np.array([p[1] for p in geom])
                    self._plot_axes['section_axes'].append({'ax': ax, 'sx': sx, 'sz': sz, 'index': i})

                # data for profile hover
                self._profile_x = _np.array(chainage[:len(wse_vals)])
                self._profile_y = _np.array(wse_vals)

                # annotation for hover
                self._annot = ax0.annotate('', xy=(0,0), xytext=(15,15), textcoords='offset points',
                                           bbox=dict(boxstyle='round', fc='w'), arrowprops=dict(arrowstyle='->'))
                self._annot.set_visible(False)

                # connect events
                def on_motion(event):
                    vis = self._annot.get_visible()
                    if event.inaxes == ax0:
                        if event.xdata is None or event.ydata is None:
                            if vis:
                                self._annot.set_visible(False)
                                canvas.draw_idle()
                            return
                        # find nearest profile point
                        dx = self._profile_x - event.xdata
                        dy = self._profile_y - event.ydata
                        d2 = dx*dx + dy*dy
                        idx = int(_np.argmin(d2))
                        # convert distance to pixels threshold
                        xpix, ypix = ax0.transData.transform((self._profile_x[idx], self._profile_y[idx]))
                        dist = ((xpix - event.x) ** 2 + (ypix - event.y) ** 2) ** 0.5
                        if dist < 10:
                            self._annot.xy = (self._profile_x[idx], self._profile_y[idx])
                            self._annot.set_text(f'Idx {idx}\nChain: {self._profile_x[idx]:.2f}\nWSE: {self._profile_y[idx]:.3f}')
                            self._annot.set_visible(True)
                            canvas.draw_idle()
                        else:
                            if vis:
                                self._annot.set_visible(False)
                                canvas.draw_idle()
                    else:
                        if vis:
                            self._annot.set_visible(False)
                            canvas.draw_idle()

                def on_click(event):
                    # if clicked on profile near a marker, select that section
                    if event.inaxes == ax0 and event.xdata is not None and event.ydata is not None:
                        dx = self._profile_x - event.xdata
                        dy = self._profile_y - event.ydata
                        d2 = dx*dx + dy*dy
                        idx = int(_np.argmin(d2))
                        xpix, ypix = ax0.transData.transform((self._profile_x[idx], self._profile_y[idx]))
                        dist = ((xpix - event.x) ** 2 + (ypix - event.y) ** 2) ** 0.5
                        if dist < 10 and idx < len(self.model.sections):
                            # select section and show detail
                            self.section_cb.setCurrentIndex(idx)
                            self.on_section_change(idx)

                self._plot_cid_motion = canvas.mpl_connect('motion_notify_event', on_motion)
                self._plot_cid_click = canvas.mpl_connect('button_press_event', on_click)
                try:
                    wse_vals = [s.wse for s in self.results if hasattr(s, 'wse')]
                    if wse_vals:
                        mn = min(wse_vals); mx = max(wse_vals)
                        self.status_label.setText(f'Run complete — WSE min: {mn:.3f}, max: {mx:.3f}')
                    else:
                        self.status_label.setText('Run complete')
                except Exception:
                    try:
                        self.status_label.setText('Run complete')
                    except Exception:
                        pass
                try:
                    # keep view mode state; user can switch tabs manually
                    pass
                except Exception:
                    pass
                try:
                    # enable Save Plot button when a plot is available
                    self.save_plot_btn.setEnabled(True)
                except Exception:
                    pass
                try:
                    self.refresh_plot()
                except Exception:
                    pass
                    try:
                        # after a run, disable the cross-section preview (results take precedence)
                        self.update_cross_section_tab_state()
                    except Exception:
                        pass
                self._refresh_scroller_choices()
            except Exception as e:
                ui_warning(self, 'Plot', f'Plot failed: {e}')
                try:
                    self.status_label.setText('Plot failed')
                except Exception:
                    pass
                try:
                    self.save_plot_btn.setEnabled(False)
                except Exception:
                    pass
        else:
            ui_info(self, 'Plot', 'matplotlib not available; skipping plot')
            try:
                self.status_label.setText('Run complete (no plot)')
            except Exception:
                pass
            try:
                self.save_plot_btn.setEnabled(False)
            except Exception:
                pass

    def on_save_model(self):
        if self.model is None:
            ui_warning(self, 'No model', 'Load and edit a model before saving.')
            return
        if not getattr(self, 'can_gpkg', False):
            ui_warning(self, 'Not available', 'GeoPackage save requires PyQGIS support.')
            return
        p, _ = ui_get_save_filename(self, 'Save Model', 'GeoPackage Files (*.gpkg)')
        if not p:
            return
        if not getattr(self, 'form_only_mode', False):
            if not self._sync_boundary_from_ui():
                return
        centerline_geom = self._get_loaded_centerline_geometry()
        save_to_geopackage(p, self.model, centerline_geom=centerline_geom)
        self.loaded_gpkg_path = self._normalize_path(p)
        self.gpkg_editing_enabled = False
        self.gpkg_dirty = False
        self._update_geopackage_edit_state()
        ui_info(self, 'Saved', f'Model saved to {p}')

    # --- layout save/restore
    def save_layout_file(self):
        p, _ = ui_get_save_filename(self, 'Save Layout', 'JSON Files (*.json)')
        if not p:
            return
        geom = self.geometry()
        try:
            cfg = {
                'x': geom.x(), 'y': geom.y(), 'w': geom.width(), 'h': geom.height(),
                'splitter': self.horiz_split.sizes()
            }
            with open(p, 'w') as f:
                json.dump(cfg, f, indent=2)
            ui_info(self, 'Saved', f'Layout saved to {p}')
        except Exception as e:
            ui_critical(self, 'Save error', str(e))

    def load_layout_file(self):
        p, _ = ui_get_open_filename(self, 'Load Layout', 'JSON Files (*.json)')
        if not p:
            return
        try:
            with open(p, 'r') as f:
                cfg = json.load(f)
            self.setGeometry(cfg.get('x',100), cfg.get('y',100), cfg.get('w',1100), cfg.get('h',700))
            sizes = cfg.get('splitter')
            if sizes:
                try:
                    self.horiz_split.setSizes(sizes)
                except Exception:
                    pass
            ui_info(self, 'Loaded', f'Layout loaded from {p}')
        except Exception as e:
            ui_critical(self, 'Load error', str(e))

    def on_save_plot(self):
        if not HAVE_MPL:
            ui_warning(self, 'Plot not available', 'matplotlib is not installed')
            return
        p, _ = ui_get_save_filename(self, 'Save Plot', 'PNG Files (*.png)')
        if not p:
            return
        try:
            import matplotlib.pyplot as _plt
            _plt.savefig(p)
            ui_info(self, 'Saved', f'Plot saved to {p}')
        except Exception as e:
            ui_critical(self, 'Save error', str(e))

    def on_save_geopackage(self):
        if self.model is None:
            ui_warning(self, 'No model', 'Load and edit a model before saving.')
            return
        if not getattr(self, 'can_gpkg', False) or 'save_to_geopackage' not in globals() or save_to_geopackage is None:
            ui_warning(self, 'Not available', 'GeoPackage save routine not available (PyQGIS required).')
            return
        default_path = self.loaded_gpkg_path or ''
        p, _ = ui_get_save_filename(self, 'Save GeoPackage', 'GeoPackage Files (*.gpkg)')
        if not p:
            return
        try:
            if not getattr(self, 'form_only_mode', False):
                if not self._sync_boundary_from_ui():
                    return
            centerline_geom = self._get_loaded_centerline_geometry()
            save_to_geopackage(p, self.model, centerline_geom=centerline_geom)
            self.loaded_gpkg_path = self._normalize_path(p)
            self.gpkg_editing_enabled = False
            self.gpkg_dirty = False
            self._update_geopackage_edit_state()
            ui_info(self, 'Saved', f'Model saved to {p}')
        except Exception as e:
            ui_critical(self, 'Save error', str(e))

    def load_example_into_qgis(self):
        """Create simple example layers (cross_sections, centerline, boundary_conditions)
        and add them to the current QGIS project. Only works when running inside QGIS.
        """
        # Ensure running inside QGIS.
        has_iface = False
        try:
            import qgis.utils as _qutils
            if getattr(_qutils, 'iface', None) is not None:
                has_iface = True
        except Exception:
            pass
        if not has_iface:
            ui_warning(self, 'QGIS required', 'This action must be run from within QGIS.')
            return
        try:
            from qgis.core import QgsVectorLayer, QgsProject, QgsFeature, QgsGeometry
            from qgis.PyQt.QtCore import QVariant
        except Exception as e:
            ui_warning(self, 'QGIS API', f'Could not access QGIS API: {e}')
            return

        # Example cross-section geometries (LINESTRING Z)
        cs_wkts = [
            "LINESTRING Z (0 0 100, 10 0 99.5)",
            "LINESTRING Z (10 0 99.5, 20 0 99.0)"
        ]

        # Build attribute-rich example data matching GeoPackage schema
        examples = {
            'cross_sections': [],
            'centerline': [],
            'boundary_conditions': []
        }
        # Cross-sections with attributes
        examples['cross_sections'].append({
            'geometry_wkt': cs_wkts[0],
            'river_station': 'S_down',
            'left_bank_station': 2.0,
            'right_bank_station': 8.0,
            'n_lob': 0.035, 'n_ch': 0.035, 'n_rob': 0.035,
            'contraction_coeff': 0.1, 'expansion_coeff': 0.3,
            'L_lob_to_next': 10.0, 'L_ch_to_next': 10.0, 'L_rob_to_next': 10.0,
            'culvert_code': 0, 'culvert_shape': '',
            'culvert_diameter': 0.0, 'culvert_width': 0.0, 'culvert_height': 0.0,
            'culvert_upstream_invert': 0.0, 'culvert_downstream_invert': 0.0, 'culvert_length': 0.0,
            'culvert_weir_coeff': 3.0, 'culvert_weir_sta_left': 0.0, 'culvert_weir_sta_right': 0.0,
        })
        examples['cross_sections'].append({
            'geometry_wkt': cs_wkts[1],
            'river_station': 'S_up',
            'left_bank_station': 12.0,
            'right_bank_station': 18.0,
            'n_lob': 0.035, 'n_ch': 0.035, 'n_rob': 0.035,
            'contraction_coeff': 0.1, 'expansion_coeff': 0.3,
            'L_lob_to_next': 0.0, 'L_ch_to_next': 0.0, 'L_rob_to_next': 0.0,
            'culvert_code': 0, 'culvert_shape': '',
            'culvert_diameter': 0.0, 'culvert_width': 0.0, 'culvert_height': 0.0,
            'culvert_upstream_invert': 0.0, 'culvert_downstream_invert': 0.0, 'culvert_length': 0.0,
            'culvert_weir_coeff': 3.0, 'culvert_weir_sta_left': 0.0, 'culvert_weir_sta_right': 0.0,
        })

        # Centerline as simple LINESTRING
        examples['centerline'].append({'geometry_wkt': 'LINESTRING (0 0, 20 0)'})

        # Boundary conditions
        examples['boundary_conditions'].append({
            'geometry_wkt': 'POINT (0 0)',
            'flow_cfs': 500.0,
            'boundary_type': 'known_wse',
            'boundary_value': 100.0
        })

        # Ask user where to save the example GeoPackage
        gpkg_path, _ = ui_get_save_filename(self, 'Save Example GeoPackage', 'GeoPackage Files (*.gpkg)')
        if not gpkg_path:
            ui_info(self, 'Cancelled', 'Save cancelled')
            return

        wrote_ok = False
        try:
            from qgis.core import QgsFields, QgsField, QgsWkbTypes, QgsVectorFileWriter
            from qgis.PyQt.QtCore import QVariant
            transform_ctx = QgsProject.instance().transformContext()

            # cross_sections layer
            cs_fields = QgsFields()
            cs_fields.append(QgsField('river_station', QVariant.String))
            cs_fields.append(QgsField('left_bank_station', QVariant.Double))
            cs_fields.append(QgsField('right_bank_station', QVariant.Double))
            cs_fields.append(QgsField('n_lob', QVariant.Double))
            cs_fields.append(QgsField('n_ch', QVariant.Double))
            cs_fields.append(QgsField('n_rob', QVariant.Double))
            cs_fields.append(QgsField('contraction_coeff', QVariant.Double))
            cs_fields.append(QgsField('expansion_coeff', QVariant.Double))
            cs_fields.append(QgsField('L_lob_to_next', QVariant.Double))
            cs_fields.append(QgsField('L_ch_to_next', QVariant.Double))
            cs_fields.append(QgsField('L_rob_to_next', QVariant.Double))
            cs_fields.append(QgsField('culvert_code', QVariant.Int))
            cs_fields.append(QgsField('culvert_shape', QVariant.String))
            cs_fields.append(QgsField('culvert_diameter', QVariant.Double))
            cs_fields.append(QgsField('culvert_width', QVariant.Double))
            cs_fields.append(QgsField('culvert_height', QVariant.Double))
            cs_fields.append(QgsField('culvert_upstream_invert', QVariant.Double))
            cs_fields.append(QgsField('culvert_downstream_invert', QVariant.Double))
            cs_fields.append(QgsField('culvert_length', QVariant.Double))
            cs_fields.append(QgsField('culvert_weir_coeff', QVariant.Double))
            cs_fields.append(QgsField('culvert_weir_sta_left', QVariant.Double))
            cs_fields.append(QgsField('culvert_weir_sta_right', QVariant.Double))
            cs_layer = QgsVectorLayer('LineStringZ?crs=EPSG:4326', 'cross_sections', 'memory')
            cs_dp = cs_layer.dataProvider()
            cs_dp.addAttributes(cs_fields)
            cs_layer.updateFields()
            feats = []
            for r in examples['cross_sections']:
                f = QgsFeature()
                f.setFields(cs_layer.fields())
                try:
                    f.setGeometry(QgsGeometry.fromWkt(r['geometry_wkt']))
                except Exception:
                    pass
                f['river_station'] = r['river_station']
                f['left_bank_station'] = float(r['left_bank_station'])
                f['right_bank_station'] = float(r['right_bank_station'])
                f['n_lob'] = float(r['n_lob'])
                f['n_ch'] = float(r['n_ch'])
                f['n_rob'] = float(r['n_rob'])
                f['contraction_coeff'] = float(r['contraction_coeff'])
                f['expansion_coeff'] = float(r['expansion_coeff'])
                f['L_lob_to_next'] = float(r['L_lob_to_next'])
                f['L_ch_to_next'] = float(r['L_ch_to_next'])
                f['L_rob_to_next'] = float(r['L_rob_to_next'])
                f['culvert_code'] = int(r.get('culvert_code', 0))
                f['culvert_shape'] = str(r.get('culvert_shape', ''))
                f['culvert_diameter'] = float(r.get('culvert_diameter', 0.0))
                f['culvert_width'] = float(r.get('culvert_width', 0.0))
                f['culvert_height'] = float(r.get('culvert_height', 0.0))
                f['culvert_upstream_invert'] = float(r.get('culvert_upstream_invert', 0.0))
                f['culvert_downstream_invert'] = float(r.get('culvert_downstream_invert', 0.0))
                f['culvert_length'] = float(r.get('culvert_length', 0.0))
                f['culvert_weir_coeff'] = float(r.get('culvert_weir_coeff', 3.0))
                f['culvert_weir_sta_left'] = float(r.get('culvert_weir_sta_left', 0.0))
                f['culvert_weir_sta_right'] = float(r.get('culvert_weir_sta_right', 0.0))
                feats.append(f)
            cs_dp.addFeatures(feats)

            opts = QgsVectorFileWriter.SaveVectorOptions()
            opts.driverName = 'GPKG'
            opts.fileEncoding = 'UTF-8'
            opts.layerName = 'cross_sections'
            QgsVectorFileWriter.writeAsVectorFormatV2(cs_layer, gpkg_path, transform_ctx, opts)

            # centerline
            cl_layer = QgsVectorLayer('LineString?crs=EPSG:4326', 'centerline', 'memory')
            cl_dp = cl_layer.dataProvider()
            f = QgsFeature()
            try:
                f.setGeometry(QgsGeometry.fromWkt(examples['centerline'][0]['geometry_wkt']))
            except Exception:
                pass
            cl_dp.addFeatures([f])
            opts.layerName = 'centerline'
            QgsVectorFileWriter.writeAsVectorFormatV2(cl_layer, gpkg_path, transform_ctx, opts)

            # boundary conditions
            bd_layer = QgsVectorLayer('Point?crs=EPSG:4326', 'boundary_conditions', 'memory')
            bd_dp = bd_layer.dataProvider()
            bd_fields = QgsFields()
            bd_fields.append(QgsField('flow_cfs', QVariant.Double))
            bd_fields.append(QgsField('boundary_type', QVariant.String))
            bd_fields.append(QgsField('boundary_value', QVariant.Double))
            bd_dp.addAttributes(bd_fields)
            bd_layer.updateFields()
            bf = QgsFeature()
            bf.setFields(bd_layer.fields())
            try:
                bf.setGeometry(QgsGeometry.fromWkt(examples['boundary_conditions'][0]['geometry_wkt']))
            except Exception:
                pass
            bf['flow_cfs'] = float(examples['boundary_conditions'][0]['flow_cfs'])
            bf['boundary_type'] = examples['boundary_conditions'][0]['boundary_type']
            bf['boundary_value'] = float(examples['boundary_conditions'][0]['boundary_value'])
            bd_dp.addFeatures([bf])
            opts.layerName = 'boundary_conditions'
            QgsVectorFileWriter.writeAsVectorFormatV2(bd_layer, gpkg_path, transform_ctx, opts)

            wrote_ok = True
        except Exception as e:
            ui_warning(self, 'QGIS write failed', f'Could not write GeoPackage via QGIS API: {e}')

        # Add saved layers to project
        if wrote_ok:
            try:
                for lname in ('cross_sections', 'centerline', 'boundary_conditions'):
                    uri = f"{gpkg_path}|layername={lname}"
                    lyr = QgsVectorLayer(uri, lname, 'ogr')
                    if lyr.isValid():
                        QgsProject.instance().addMapLayer(lyr)
                ui_info(self, 'Loaded', f'Example GeoPackage saved to {gpkg_path} and added to project')
            except Exception:
                ui_info(self, 'Saved', f'Example GeoPackage saved to {gpkg_path} (could not add to project)')
        else:
            ui_warning(self, 'Failed', 'Could not write example GeoPackage')

def create_backwater_dockwidget(parent=None, title='Backwater'):
    """Create a QDockWidget containing the backwater UI suitable for adding
    to a QGIS main window via `addDockWidget()`.
    """
    dock = QtWidgets.QDockWidget(title, parent)
    dock.setObjectName('BackwaterMainDock')
    widget = BackwaterWidget(parent)
    dock.setWidget(widget)
    return dock
