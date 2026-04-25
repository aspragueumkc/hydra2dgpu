"""QGIS plugin glue for Backwater solver UI.

This plugin creates a dockable UI by embedding the existing Backwater
widget. It also exposes a simple action to extract cross-section geometry
from a selected 3D polyline feature and populate the geometry editor.

Note: This code is intended to be loaded inside QGIS where `iface` is
available. When running outside QGIS it will not function.
"""
import os
import sys
from qgis.PyQt.QtWidgets import QAction, QDockWidget
from qgis.PyQt.QtCore import Qt
from qgis.core import Qgis


class BackwaterQgisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        # try to ensure workspace root is on path so we can import backwater modules
        root = os.path.abspath(os.path.join(self.plugin_dir, '..'))
        if root not in sys.path:
            sys.path.insert(0, root)

        # import the UI factory and utils. Prefer plugin-local modules, but
        # fall back to top-level `backwater_qt` when the package-local file
        # is not present (useful when plugin references shared workspace files).
        try:
            try:
                from .backwater_qt import create_backwater_dockwidget
            except Exception:
                # fallback to workspace-level backwater_qt
                try:
                    import backwater_qt as _bwqt
                    create_backwater_dockwidget = _bwqt.create_backwater_dockwidget
                except Exception:
                    create_backwater_dockwidget = None

            try:
                from .qgis_utils import extract_xs_from_line
            except Exception:
                extract_xs_from_line = None

            # ensure the plugin ui_adapter uses the QGIS iface
            try:
                from . import ui_adapter
                ui_adapter.iface = iface
            except Exception:
                try:
                    # fallback import
                    #from qgis_backwater_plugin import ui_adapter
                    import ui_adapter
                    ui_adapter.iface = iface
                except Exception:
                    pass

            self._create_dock = create_backwater_dockwidget
            self._extract_fn = extract_xs_from_line
        except Exception:
            # If imports fail in unexpected ways, allow plugin to load but
            # actions will report errors at runtime.
            self._create_dock = None
            self._extract_fn = None

        self.dock = None
        self.action = None
        self.extract_action = None

    def initGui(self):
        self.action = QAction('Backwater', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu('&Backwater', self.action)

        self.extract_action = QAction('Extract cross-section from selected line', self.iface.mainWindow())
        self.extract_action.triggered.connect(self.extract_from_selected)
        self.iface.addToolBarIcon(self.extract_action)
        self.iface.addPluginToMenu('&Backwater', self.extract_action)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu('&Backwater', self.action)
            self.iface.removeToolBarIcon(self.action)
        if self.extract_action:
            self.iface.removePluginMenu('&Backwater', self.extract_action)
            self.iface.removeToolBarIcon(self.extract_action)
        if self.dock:
            try:
                self.iface.removeDockWidget(self.dock)
            except Exception:
                pass

    def run(self):
        if not self._create_dock:
            self.iface.messageBar().pushMessage('Backwater', 'UI components not found', level=Qgis.Critical)
            return
        if not self.dock:
            self.dock = self._create_dock(parent=self.iface.mainWindow(), title='Backwater')
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.show()
        self.dock.raise_()

    def extract_from_selected(self):
        layer = self.iface.activeLayer()
        if layer is None:
            self.iface.messageBar().pushMessage('Backwater', 'No active layer', level=Qgis.Warning)
            return
        # get first selected feature
        sel = layer.selectedFeatureIds() if hasattr(layer, 'selectedFeatureIds') else [f.id() for f in layer.selectedFeatures()]
        if not sel:
            self.iface.messageBar().pushMessage('Backwater', 'No feature selected', level=Qgis.Warning)
            return
        fid = sel[0]
        feat = layer.getFeature(fid)
        if self._extract_fn is None:
            self.iface.messageBar().pushMessage('Backwater', 'Extraction function not available', level=Qgis.Critical)
            return
        geom = feat.geometry()
        pts = self._extract_fn(geom, samples=64)
        if not pts:
            self.iface.messageBar().pushMessage('Backwater', 'No geometry extracted', level=Qgis.Warning)
            return
        # ensure UI is shown
        if not self.dock:
            self.run()
        widget = self.dock.widget()
        if widget is None:
            self.iface.messageBar().pushMessage('Backwater', 'No widget available', level=Qgis.Critical)
            return
        # populate geometry table (geometry editor page)
        try:
            widget.geom_table.setRowCount(0)
            for st, z in pts:
                r = widget.geom_table.rowCount()
                widget.geom_table.insertRow(r)
                from qgis.PyQt.QtWidgets import QTableWidgetItem
                widget.geom_table.setItem(r, 0, QTableWidgetItem(f"{st:.3f}"))
                widget.geom_table.setItem(r, 1, QTableWidgetItem(f"{z:.3f}"))
            widget.set_view_mode('geometry')
            self.iface.messageBar().pushMessage('Backwater', f'Extracted {len(pts)} points', level=Qgis.Info)
        except Exception as e:
            self.iface.messageBar().pushMessage('Backwater', f'Populate failed: {e}', level=Qgis.Critical)
