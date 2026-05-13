"""swe2d_map_tools.py

Map tools used by the SWE2D workbench.
"""

from __future__ import annotations

from typing import List

from qgis.PyQt import QtCore
from qgis.PyQt.QtCore import Qt, pyqtSignal

try:
    from qgis.core import QgsGeometry, QgsPointXY, QgsWkbTypes
    from qgis.gui import QgsMapTool, QgsRubberBand
    _HAVE_QGIS_MAP_TOOL = True
except Exception:
    QgsGeometry = QgsPointXY = QgsWkbTypes = QgsMapTool = QgsRubberBand = None
    _HAVE_QGIS_MAP_TOOL = False


if _HAVE_QGIS_MAP_TOOL:

    class SWE2DLineDrawTool(QgsMapTool):
        """Interactive map tool to draw a polyline and emit finished geometry."""

        line_finished = pyqtSignal(object)

        def __init__(self, canvas):
            super().__init__(canvas)
            self._canvas = canvas
            self._points: List[QgsPointXY] = []
            self._rb = QgsRubberBand(canvas, QgsWkbTypes.GeometryType.LineGeometry)
            self._rb.setColor(QtCore.Qt.blue)
            self._rb.setWidth(2)

        def canvasPressEvent(self, event):
            if event.button() == Qt.RightButton:
                self._finish_line()
                return
            pt = self.toMapCoordinates(event.pos())
            self._points.append(QgsPointXY(pt.x(), pt.y()))
            self._redraw_rubberband()

        def canvasMoveEvent(self, event):
            if not self._points:
                return
            pt = self.toMapCoordinates(event.pos())
            self._redraw_rubberband(QgsPointXY(pt.x(), pt.y()))

        def canvasDoubleClickEvent(self, event):
            pt = self.toMapCoordinates(event.pos())
            self._points.append(QgsPointXY(pt.x(), pt.y()))
            self._finish_line()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self._points = []
                self._rb.reset(QgsWkbTypes.GeometryType.LineGeometry)

        def deactivate(self):
            self._rb.reset(QgsWkbTypes.GeometryType.LineGeometry)
            super().deactivate()

        def _redraw_rubberband(self, moving_pt=None):
            self._rb.reset(QgsWkbTypes.GeometryType.LineGeometry)
            pts = list(self._points)
            if moving_pt is not None:
                pts.append(moving_pt)
            for p in pts:
                self._rb.addPoint(p, False)
            self._rb.show()

        def _finish_line(self):
            if len(self._points) < 2:
                self._points = []
                self._rb.reset(QgsWkbTypes.GeometryType.LineGeometry)
                return
            geom = QgsGeometry.fromPolylineXY(self._points)
            self.line_finished.emit(geom)
            self._points = []
            self._rb.reset(QgsWkbTypes.GeometryType.LineGeometry)

else:

    class SWE2DLineDrawTool:  # pragma: no cover
        """Fallback placeholder when QGIS map tool APIs are unavailable."""

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("QGIS map tool APIs are unavailable in this runtime")
