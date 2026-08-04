"""swe2d/workbench/views/network_profile_map_tool.py

QgsMapTool that lets the user build a chain of drainage links by clicking
on the QGIS map canvas. Each click extends the chain downstream along the
network; orientation is auto-detected. Double-click / right-click / Escape
emits the finished chain.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from qgis.core import QgsFeature, QgsMapLayer
from qgis.gui import QgsMapTool, QgsMapToolIdentify, QgsMapToolIdentifyFeature
from qgis.PyQt import QtCore, QtGui, QtWidgets

from swe2d.workbench.services.drainage_graph_service import (
    DrainageGraph,
    link_orientation,
)
from swe2d.workbench.services.profile_pipeline_service import ChainSpec

logger = logging.getLogger(__name__)


class NetworkProfileMapTool(QgsMapTool):
    """Map tool: click on drainage links to extend the chain downstream."""

    chain_extended = QtCore.pyqtSignal(object)
    chain_cleared  = QtCore.pyqtSignal()
    pick_rejected  = QtCore.pyqtSignal(str, str)
    finished       = QtCore.pyqtSignal(object)

    def __init__(self, canvas, drainage_layer: QgsMapLayer, graph: DrainageGraph):
        # NOTE: do not pass `parent` to super().__init__ — the underlying
        # QGIS bindings (QgsMapTool) only accept (self, canvas). Passing a
        # third positional arg raises "too many arguments" at runtime.
        super().__init__(canvas)
        self._canvas = canvas
        self._layer = drainage_layer
        self._graph = graph
        self._chain: List[Tuple[str, bool]] = []
        self._last_downstream_node: Optional[str] = None
        self._identify = QgsMapToolIdentifyFeature(self._canvas, self._layer)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

    def _identify_link(self, event) -> Optional[QgsFeature]:
        # QgsMapToolIdentifyFeature has no ``identifyFeatureAt`` on QGIS 3.44;
        # ``identify(x, y, mode, layerType)`` is the real API and returns
        # IdentifyResult objects.  Fail loudly — a silent ``return None`` here
        # is what hid the missing-method bug.
        pos = event.pos()
        results = self._identify.identify(
            pos.x(),
            pos.y(),
            QgsMapToolIdentify.TopDownStopAtFirst,
            QgsMapToolIdentify.VectorLayer,
        )
        for result in results:
            if result.mLayer == self._layer:
                return result.mFeature
        return None

    def canvasReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._finish()
            return
        feat = self._identify_link(event)
        if feat is None:
            return
        link_id = str(feat.attribute("link_id") or feat.id())
        if link_id not in self._graph.from_node:
            self.pick_rejected.emit("link not in drainage network", link_id)
            return
        if not self._chain:
            # First click — start the chain. Determine upstream by lower out-degree.
            from_node = self._graph.from_node[link_id]
            to_node = self._graph.to_node[link_id]
            out_deg_from = len(self._graph.outgoing.get(from_node, []))
            out_deg_to = len(self._graph.outgoing.get(to_node, []))
            if out_deg_from <= out_deg_to:
                upstream, downstream = from_node, to_node
                reverse = False
            else:
                upstream, downstream = to_node, from_node
                reverse = True
            self._chain = [(link_id, reverse)]
            self._last_downstream_node = downstream
            self.chain_extended.emit(ChainSpec(link_specs=self._chain))
            return
        # Subsequent click — verify and extend
        last_link_id, _last_rev = self._chain[-1]
        if link_id == last_link_id:
            return  # same link ignored
        link_fn = self._graph.from_node[link_id]
        link_tn = self._graph.to_node[link_id]
        last_to = self._last_downstream_node or self._graph.to_node[last_link_id]
        if link_fn == last_to:
            reverse = False
            downstream = link_tn
        elif link_tn == last_to:
            reverse = True
            downstream = link_fn
        else:
            self.pick_rejected.emit(
                f"link {link_id} does not connect to last downstream node {last_to}",
                link_id,
            )
            return
        self._chain.append((link_id, reverse))
        self._last_downstream_node = downstream
        self.chain_extended.emit(ChainSpec(link_specs=self._chain))

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Return):
            self._finish()

    def _finish(self):
        chain = ChainSpec(link_specs=self._chain)
        self.finished.emit(chain)
        self.deactivate()

    def canvasDoubleClickEvent(self, event):
        self._finish()
