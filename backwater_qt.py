#!/usr/bin/env python3
"""backwater_qt.py

A Qt-based GUI wrapper for the backwater solver that reuses the computation
functions in `backwater2.py`.

This provides a minimal, modern GUI with: file open, New Model, Run, textual
results, and a matplotlib plot area.

Requires: PyQt5 (or install PySide2 and adapt imports), matplotlib for plotting.
"""

import os
import sys
import json

# Try Qt imports
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    HAVE_QT = True
except Exception:
    try:
        from qgis.PyQt import QtWidgets, QtCore, QtGui
        from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
        HAVE_QT = True
    except Exception:
        QtWidgets = None
        QtCore = None
        QtGui = None
        QFileDialog = None
        QMessageBox = None
        HAVE_QT = False

if not HAVE_QT:
    print('PyQt5 not available. Install PyQt5 to use the Qt GUI.')
    if __name__ == '__main__':
        sys.exit(0)
    class _QtDummyNamespace:
        def __getattr__(self, name):
            return type(name, (), {})

    QtWidgets = _QtDummyNamespace()
    QtCore = _QtDummyNamespace()
    QtGui = _QtDummyNamespace()

# Import solver functions from the plugin-local `backwater2` module.
try:
    try:
        from . import backwater2 as _bwmod  # type: ignore
    except Exception:
        import backwater2 as _bwmod  # type: ignore

    load_input = _bwmod.load_input
    load_from_geopackage = getattr(_bwmod, 'load_from_geopackage', None)
    save_to_geopackage = getattr(_bwmod, 'save_to_geopackage', None)
    run_backwater = _bwmod.run_backwater
    ModelInput = _bwmod.ModelInput
    CrossSection = _bwmod.CrossSection
    _plot_results = _bwmod._plot_results
    HAVE_MPL = _bwmod.HAVE_MPL
except Exception:
    load_input = load_from_geopackage = save_to_geopackage = run_backwater = ModelInput = CrossSection = _plot_results = None
    HAVE_MPL = False

try:
    try:
        from . import ui_adapter as _ui_adapter  # type: ignore
    except Exception:
        import ui_adapter as _ui_adapter  # type: ignore
    ui_adapter = _ui_adapter
except Exception:
    ui_adapter = None


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
            if event.modifiers() & QtCore.Qt.ControlModifier and self._canvas is not None:
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
                painter.drawText(rect, QtCore.Qt.AlignCenter, 'No geometry')
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
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawPolygon(fill_poly)

            # draw polyline
            painter.setPen(QtGui.QPen(QtGui.QColor('#000000'), 1))
            painter.drawPolyline(poly)
            # draw points
            for pt in poly:
                painter.drawEllipse(pt, 2, 2)

            # draw labels
            painter.setPen(QtGui.QPen(QtCore.Qt.black))
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

class BackwaterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Backwater — Qt GUI')
        self.resize(1100, 700)

        # Use splitters so user can resize sections; left pane = controls, right pane = plots
        main_layout = QtWidgets.QVBoxLayout(self)

        # Top control row
        controls_row = QtWidgets.QHBoxLayout()
        self.input_path = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton('Browse...')
        browse_btn.clicked.connect(self.on_browse)
        new_btn = QtWidgets.QPushButton('New Model')
        new_btn.clicked.connect(self.on_new_model)
        load_btn = QtWidgets.QPushButton('Load')
        load_btn.clicked.connect(self.on_load)
        run_btn = QtWidgets.QPushButton('Run')
        run_btn.clicked.connect(self.on_run)
        detach_btn = QtWidgets.QPushButton('Detach Plot')
        detach_btn.clicked.connect(self.detach_plot)

        controls_row.addWidget(QtWidgets.QLabel('Input GeoPackage:'))
        controls_row.addWidget(self.input_path)
        controls_row.addWidget(browse_btn)
        controls_row.addWidget(new_btn)
        controls_row.addWidget(load_btn)
        controls_row.addWidget(run_btn)
        controls_row.addWidget(detach_btn)
        main_layout.addLayout(controls_row)

        # Splitter between left (controls) and right (plots)
        self.horiz_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.horiz_split, stretch=1)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

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

        self.horiz_split.addWidget(left_widget)

        # Right side: vertical split (top = plots, bottom = tabular IO)
        self.right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

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
                QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
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
        save_model_btn = QtWidgets.QPushButton('Save Model...')
        save_model_btn.clicked.connect(self.on_save_model)
        self.save_plot_btn = QtWidgets.QPushButton('Save Plot...')
        self.save_plot_btn.clicked.connect(self.on_save_plot)
        self.save_plot_btn.setEnabled(False)
        # Save model belongs on Boundary tab; save plot in Results
        self.boundary_tab_layout.addWidget(save_model_btn)
        try:
            # place Save Plot button into bottom-right Results tab when available
            self.results_page_layout.addWidget(self.save_plot_btn)
        except Exception:
            # fallback: place on boundary tab
            try:
                self.boundary_tab_layout.addWidget(self.save_plot_btn)
            except Exception:
                pass
        # Detect GeoPackage support at runtime (geopandas/fiona/shapely)
        try:
            import geopandas as _gpd  # type: ignore
            import fiona  # type: ignore
            from shapely.geometry import LineString  # type: ignore
            self.can_gpkg = True
        except Exception:
            self.can_gpkg = False
        # Add a Save-to-GeoPackage button (visible when embedded). Enable only if available.
        save_gpkg_btn = QtWidgets.QPushButton('Save to GeoPackage...')
        save_gpkg_btn.clicked.connect(self.on_save_geopackage)
        save_gpkg_btn.setEnabled(self.can_gpkg)
        self.boundary_tab_layout.addWidget(save_gpkg_btn)
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
        # Load example into QGIS (only enabled when running inside QGIS)
        load_example_btn = QtWidgets.QPushButton('Load example into QGIS')
        load_example_btn.clicked.connect(self.load_example_into_qgis)
        # Detect whether we're running inside QGIS: prefer qgis.utils.iface, fallback to ui_adapter.iface
        has_iface = False
        try:
            import qgis.utils as _qutils
            if getattr(_qutils, 'iface', None) is not None:
                has_iface = True
        except Exception:
            pass
        if not has_iface and ui_adapter is not None and getattr(ui_adapter, 'iface', None) is not None:
            has_iface = True
        load_example_btn.setEnabled(bool(has_iface))
        self.boundary_tab_layout.addWidget(load_example_btn)

        # In-memory model
        self.model = None
        self.results = None
        self.loaded_gpkg_path = ''
        self.gpkg_editing_enabled = False
        self.gpkg_dirty = False
        self._scroller_entries = []
        self._scroller_canvas = None
        # Undo/redo stacks (store JSON strings)
        self.undo_stack = []
        self.redo_stack = []
        self._set_model_editing_enabled(True)
        self._update_geopackage_edit_state()
        self._refresh_scroller_choices()

        # Note: menu/toolbar/status bar are created by the QMainWindow wrapper
        # when this widget is embedded in a MainWindow. This keeps the widget
        # usable standalone (embedded) without creating application chrome.

    def _create_menus_and_toolbars(self):
        # menu/tool creation moved to MainWindow wrapper; widget does not create app chrome
        return

    def _create_status_bar(self):
        # status bar is created by the MainWindow wrapper when embedded
        return

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
            self._write_cross_section_to_layer(idx)
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
            self._write_cross_section_to_layer(idx)
            self._mark_gpkg_dirty()
        except Exception as exc:
            ui_critical(self, 'GeoPackage update failed', str(exc))
            return
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
                    item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
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
        gpkg_loaded = bool(self.loaded_gpkg_path)
        qgis_available = self._get_qgis_iface() is not None
        model_editable = (not gpkg_loaded) or self.gpkg_editing_enabled or (not qgis_available)
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
            self.layer_edit_status_label.setText(f'Layer editing: enabled{suffix}')
            self.toggle_gpkg_edit_btn.setText('Disable Layer Editing')
        else:
            self.layer_edit_status_label.setText('Layer editing: read-only')
            self.toggle_gpkg_edit_btn.setText('Enable Layer Editing')

    def _mark_gpkg_dirty(self):
        if self.loaded_gpkg_path and self.gpkg_editing_enabled:
            self.gpkg_dirty = True
            self._update_geopackage_edit_state()

    def _write_cross_section_to_layer(self, idx: int):
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
        target_feature = None
        for feature in layer.getFeatures():
            try:
                if str(feature['river_station']) == str(xs.river_station):
                    target_feature = feature
                    break
            except Exception:
                continue

        geometry = QgsGeometry.fromPolylineXY([QgsPointXY(float(st), float(z)) for st, z in xs.geometry])
        attrs = {
            'river_station': str(xs.river_station),
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
            feature = QgsFeature(layer.fields())
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
        layer.changeGeometry(fid, geometry)

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
        self.results = None
        try:
            self.save_plot_btn.setEnabled(False)
        except Exception:
            pass
        self._refresh_scroller_choices()

    def on_toggle_geopackage_editing(self):
        if not self.loaded_gpkg_path:
            ui_warning(self, 'GeoPackage required', 'Load a GeoPackage-backed model first.')
            return
        if self._get_qgis_iface() is None:
            ui_warning(self, 'QGIS required', 'Layer editing is only available when the plugin is running inside QGIS.')
            return
        layer_names = ('cross_sections', 'boundary_conditions')
        if not self.gpkg_editing_enabled:
            try:
                layers = []
                for layer_name in layer_names:
                    layer = self._ensure_gpkg_layer_loaded(layer_name)
                    if layer is None:
                        raise RuntimeError(f'{layer_name} layer could not be loaded from {self.loaded_gpkg_path}')
                    if not layer.isEditable():
                        layer.startEditing()
                    layers.append(layer)
                self.gpkg_editing_enabled = True
                self.gpkg_dirty = False
                self._update_geopackage_edit_state()
                ui_info(self, 'Layer editing', 'GeoPackage layers are now editable.')
            except Exception as exc:
                ui_critical(self, 'Layer editing', str(exc))
            return

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
            self._write_boundary_to_layer()
            for layer_name in ('cross_sections', 'boundary_conditions'):
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
            ui_critical(self, 'Save layer edits', str(exc))

    # --- model actions
    def on_browse(self):
        filters = 'GeoPackage Files (*.gpkg);;All Files (*)'
        p, _ = ui_get_open_filename(self, 'Open Model', filters)
        if p:
            self.input_path.setText(p)

    def on_new_model(self):
        try:
            base_flow = float(self.flow_edit.text())
        except Exception:
            base_flow = 500.0
        xs0 = CrossSection(
            river_station='S_down',
            geometry=[(0.0, 100.0), (10.0, 99.5)],
            left_bank_station=2.0,
            right_bank_station=8.0,
            n_lob=0.035, n_ch=0.035, n_rob=0.035,
            contraction_coeff=0.1, expansion_coeff=0.3,
            L_lob_to_next=10.0, L_ch_to_next=10.0, L_rob_to_next=10.0
        )
        xs1 = CrossSection(
            river_station='S_up',
            geometry=[(10.0, 99.5), (20.0, 99.0)],
            left_bank_station=12.0,
            right_bank_station=18.0,
            n_lob=0.035, n_ch=0.035, n_rob=0.035,
            contraction_coeff=0.1, expansion_coeff=0.3,
            L_lob_to_next=10.0, L_ch_to_next=10.0, L_rob_to_next=10.0
        )
        self.model = ModelInput(
            flow_cfs=base_flow,
            flow_change=None,
            boundary_condition='known_wse',
            boundary_value=100.0,
            sections=[xs0, xs1]
        )
        self.loaded_gpkg_path = ''
        self.gpkg_editing_enabled = False
        self.gpkg_dirty = False
        self.input_path.setText('')
        self._sync_ui_from_model()
        self.section_cb.clear()
        self.section_cb.addItems([xs.river_station for xs in self.model.sections])
        self.section_cb.setCurrentIndex(0)
        self.results = None
        ui_info(self, 'Created', 'Created new minimal two-section model')
        # push initial state to undo stack
        self.push_undo()
        try:
            self.update_cross_section_tab_state()
        except Exception:
            pass
        self._update_geopackage_edit_state()
        self._refresh_scroller_choices()

    def on_load(self):
        # Ensure we can rebind module-level loader symbols if needed
        global load_from_geopackage, save_to_geopackage, load_input
        p = self.input_path.text().strip()
        if not p:
            ui_warning(self, 'Input required', 'Please choose an input GeoPackage file or create a new model')
            return
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
            ui_info(self, 'Loaded', f'Loaded model: {p}')
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
            self._refresh_scroller_choices()
        except Exception as e:
            ui_critical(self, 'Error', str(e))

    def on_run(self):
        try:
            self.status_label.setText('Running...')
        except Exception:
            pass

        # ensure these names refer to the plugin-local solver module symbols
        global run_backwater, load_input, load_from_geopackage, save_to_geopackage, ModelInput, CrossSection, _plot_results, HAVE_MPL

        if self.model is None:
            p = self.input_path.text().strip()
            if not p:
                ui_warning(self, 'Input required', 'Please load a model or create a new one')
                return
            try:
                self.model = load_input(p)
                self.section_cb.clear()
                self.section_cb.addItems([xs.river_station for xs in self.model.sections])
            except Exception as e:
                ui_critical(self, 'Error', str(e))
                return

        # override bc and flow
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
            ModelInput = getattr(_mod, 'ModelInput', ModelInput)
            CrossSection = getattr(_mod, 'CrossSection', CrossSection)
            _plot_results = getattr(_mod, '_plot_results', _plot_results)
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
                ModelInput = getattr(_mod, 'ModelInput', ModelInput)
                CrossSection = getattr(_mod, 'CrossSection', CrossSection)
                _plot_results = getattr(_mod, '_plot_results', _plot_results)
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
            ui_warning(self, 'Not available', 'GeoPackage save requires geopandas/fiona/shapely.')
            return
        p, _ = ui_get_save_filename(self, 'Save Model', 'GeoPackage Files (*.gpkg)')
        if not p:
            return
        if not self._sync_boundary_from_ui():
            return
        save_to_geopackage(p, self.model)
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
            ui_warning(self, 'Not available', 'GeoPackage save routine not available (requires geopandas/fiona/shapely).')
            return
        default_path = self.loaded_gpkg_path or ''
        p, _ = ui_get_save_filename(self, 'Save GeoPackage', 'GeoPackage Files (*.gpkg)')
        if not p:
            return
        try:
            if not self._sync_boundary_from_ui():
                return
            save_to_geopackage(p, self.model)
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
        # Ensure running inside QGIS (check qgis.utils.iface or ui_adapter.iface)
        has_iface = False
        try:
            import qgis.utils as _qutils
            if getattr(_qutils, 'iface', None) is not None:
                has_iface = True
        except Exception:
            pass
        if not has_iface:
            if ui_adapter is None or getattr(ui_adapter, 'iface', None) is None:
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

        # Try to use geopandas (if available) to write layers to the GeoPackage
        wrote_ok = False
        if self.can_gpkg:
            try:
                import geopandas as gpd
                from shapely import wkt
                # cross_sections
                rows = []
                for r in examples['cross_sections']:
                    geom = wkt.loads(r['geometry_wkt'])
                    rows.append({
                        'geometry': geom,
                        'river_station': r['river_station'],
                        'left_bank_station': float(r['left_bank_station']),
                        'right_bank_station': float(r['right_bank_station']),
                        'n_lob': float(r['n_lob']), 'n_ch': float(r['n_ch']), 'n_rob': float(r['n_rob']),
                        'contraction_coeff': float(r['contraction_coeff']),
                        'expansion_coeff': float(r['expansion_coeff']),
                        'L_lob_to_next': float(r['L_lob_to_next']),
                        'L_ch_to_next': float(r['L_ch_to_next']),
                        'L_rob_to_next': float(r['L_rob_to_next']),
                        'culvert_code': int(r.get('culvert_code', 0)),
                        'culvert_shape': r.get('culvert_shape', ''),
                        'culvert_diameter': float(r.get('culvert_diameter', 0.0)),
                        'culvert_width': float(r.get('culvert_width', 0.0)),
                        'culvert_height': float(r.get('culvert_height', 0.0)),
                        'culvert_upstream_invert': float(r.get('culvert_upstream_invert', 0.0)),
                        'culvert_downstream_invert': float(r.get('culvert_downstream_invert', 0.0)),
                        'culvert_length': float(r.get('culvert_length', 0.0)),
                        'culvert_weir_coeff': float(r.get('culvert_weir_coeff', 3.0)),
                        'culvert_weir_sta_left': float(r.get('culvert_weir_sta_left', 0.0)),
                        'culvert_weir_sta_right': float(r.get('culvert_weir_sta_right', 0.0)),
                    })
                gdf_cs = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
                gdf_cs.to_file(gpkg_path, layer='cross_sections', driver='GPKG')

                # centerline
                rows = []
                for r in examples['centerline']:
                    geom = wkt.loads(r['geometry_wkt'])
                    rows.append({'geometry': geom})
                gdf_cl = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
                gdf_cl.to_file(gpkg_path, layer='centerline', driver='GPKG', mode='a')

                # boundary conditions
                rows = []
                for r in examples['boundary_conditions']:
                    geom = wkt.loads(r['geometry_wkt'])
                    rows.append({'geometry': geom, 'flow_cfs': float(r['flow_cfs']), 'boundary_type': r['boundary_type'], 'boundary_value': float(r['boundary_value'])})
                gdf_bd = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
                gdf_bd.to_file(gpkg_path, layer='boundary_conditions', driver='GPKG', mode='a')

                wrote_ok = True
            except Exception as e:
                ui_warning(self, 'geopandas write failed', f'geopandas write failed: {e}')

        # If geopandas unavailable or failed, try QGIS writer API
        if not wrote_ok:
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


def main():
    if not HAVE_QT:
        print('PyQt5 not available. Install PyQt5 to use the Qt GUI.')
        return
    # When run standalone, create a QApplication and show the main window.
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True
    # Prefer the full MainWindow if present (menus/toolbars); otherwise use the
    # lightweight BackwaterWidget so the module can run even if class order
    # differs due to editing.
    if 'MainWindow' in globals():
        w = MainWindow()
    else:
        w = BackwaterWidget()
    w.show()
    if created_app:
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()


def create_backwater_widget(parent=None):
    """Create and return the BackwaterWidget instance for embedding in host apps.

    In QGIS plugin code you can call `create_backwater_widget()` and add the
    returned widget to a dock or layout. Do NOT call this from code that
    already creates a QApplication unless you intend to show the window.
    """
    if not HAVE_QT:
        raise RuntimeError('PyQt5 not available')
    w = BackwaterWidget(parent)
    if parent is not None:
        try:
            w.setParent(parent)
        except Exception:
            pass
    return w


def create_backwater_dockwidget(parent=None, title='Backwater'):
    """Create a QDockWidget containing the backwater UI suitable for adding
    to a QGIS main window via `addDockWidget()`.
    """
    if not HAVE_QT:
        raise RuntimeError('PyQt5 not available')
    dock = QtWidgets.QDockWidget(title, parent)
    # Use a MainWindow as the dock's widget to preserve menus/toolbars if desired.
    widget = BackwaterWidget(parent)
    dock.setWidget(widget)
    return dock


class MainWindow(QtWidgets.QMainWindow):
    """Thin QMainWindow wrapper that builds menus/toolbars/status bar and
    embeds the `BackwaterWidget` so the core UI can also be used as a plain
    widget (for QGIS plugin embedding).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Backwater — Qt GUI')
        self.resize(1100, 700)
        # central widget
        self.central = BackwaterWidget(self)
        self.setCentralWidget(self.central)
        # create status bar first so central widget can reference it
        self._create_status_bar()
        # let central widget update status_label attribute
        try:
            self.central.status_label = self.status_label
        except Exception:
            pass
        # create menus/toolbars and wire to central widget methods
        self._create_menus_and_toolbars()

    def _create_status_bar(self):
        sb = QtWidgets.QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QtWidgets.QLabel('Ready')
        sb.addPermanentWidget(self.status_label)

    def _create_menus_and_toolbars(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        new_act = QtWidgets.QAction('New', self)
        new_act.setShortcut('Ctrl+N')
        new_act.triggered.connect(self.central.on_new_model)
        file_menu.addAction(new_act)

        open_act = QtWidgets.QAction('Open...', self)
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self.central.on_load)
        file_menu.addAction(open_act)

        save_act = QtWidgets.QAction('Save...', self)
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self.central.on_save_model)
        file_menu.addAction(save_act)

        save_gpkg_act = QtWidgets.QAction('Save to GeoPackage...', self)
        save_gpkg_act.triggered.connect(self.central.on_save_geopackage)
        # Enable/disable based on runtime detection in the central widget
        try:
            enabled = bool(getattr(self.central, 'can_gpkg', False))
            save_gpkg_act.setEnabled(enabled)
            if not enabled:
                save_gpkg_act.setToolTip('Disabled: geopandas/fiona/shapely not installed')
        except Exception:
            pass
        file_menu.addAction(save_gpkg_act)

        file_menu.addSeparator()
        exit_act = QtWidgets.QAction('Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        undo_act = QtWidgets.QAction('Undo', self)
        undo_act.setShortcut('Ctrl+Z')
        undo_act.triggered.connect(self.central.undo)
        edit_menu.addAction(undo_act)

        redo_act = QtWidgets.QAction('Redo', self)
        redo_act.setShortcut('Ctrl+Y')
        redo_act.triggered.connect(self.central.redo)
        edit_menu.addAction(redo_act)

        # View menu
        view_menu = menubar.addMenu('View')
        detach_act = QtWidgets.QAction('Detach Plot', self)
        detach_act.triggered.connect(self.central.detach_plot)
        view_menu.addAction(detach_act)

        # Help menu
        help_menu = menubar.addMenu('Help')
        about_act = QtWidgets.QAction('About', self)
        about_act.triggered.connect(self.central._show_about)
        help_menu.addAction(about_act)

        # Main toolbar
        toolbar = self.addToolBar('Main')
        toolbar.setMovable(True)
        toolbar.addAction(new_act)
        toolbar.addAction(open_act)
        toolbar.addAction(save_act)
        toolbar.addSeparator()
        run_act = QtWidgets.QAction('Run', self)
        run_act.setShortcut('F5')
        run_act.triggered.connect(self.central.on_run)
        toolbar.addAction(run_act)
        toolbar.addSeparator()
        toolbar.addAction(undo_act)
        toolbar.addAction(redo_act)
        toolbar.addSeparator()
        toolbar.addAction(detach_act)
        toolbar.addSeparator()
        # View mode actions
        geom_view_act = QtWidgets.QAction('Geometry Editor', self)
        geom_view_act.setCheckable(True)
        geom_view_act.triggered.connect(lambda: self.central.set_view_mode('geometry'))
        profile_view_act = QtWidgets.QAction('Profile Plot', self)
        profile_view_act.setCheckable(True)
        profile_view_act.triggered.connect(lambda: self.central.set_view_mode('profile'))
        section_view_act = QtWidgets.QAction('Cross-section Plot', self)
        section_view_act.setCheckable(True)
        section_view_act.triggered.connect(lambda: self.central.set_view_mode('section'))
        toolbar.addAction(geom_view_act)
        toolbar.addAction(profile_view_act)
        toolbar.addAction(section_view_act)
        # action group so only one is checked
        ag = QtWidgets.QActionGroup(self)
        ag.addAction(geom_view_act); ag.addAction(profile_view_act); ag.addAction(section_view_act)
        geom_view_act.setChecked(True)

        # small toolbar for quick-save plot
        plot_tb = self.addToolBar('Plot')
        saveplot_act = QtWidgets.QAction('Save Plot', self)
        saveplot_act.triggered.connect(self.central.on_save_plot)
        plot_tb.addAction(saveplot_act)
