"""Behavioral tests for the workbench map tools (plan Task D.1).

Covers:
- ``swe2d/workbench/map_tools.py`` — ``SWE2DLineDrawTool``.  NOTE: the class
  IS present; it is defined inside ``if _HAVE_QGIS_MAP_TOOL:`` so a naive
  ``^class`` grep misses it.
- ``swe2d/workbench/views/network_profile_map_tool.py`` — ``NetworkProfileMapTool``.

Pattern P2 (spec docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4):
real offscreen ``QgsMapCanvas`` constructed directly, tools installed via
``canvas.setMapTool``, interaction synthesized with real ``QgsMapMouseEvent``
objects at real canvas pixel coordinates, effects asserted via rubber-band
geometry and ``QSignalSpy`` on the emitted signals.  No mocks.
"""

from __future__ import annotations

import unittest

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    make_memory_layer,
    requires_qgis,
)


def _map_mouse_event(canvas, qevent_type, pos, button):
    """Build a real QgsMapMouseEvent for *canvas* at pixel *pos*."""
    from qgis.gui import QgsMapMouseEvent

    return QgsMapMouseEvent(canvas, qevent_type, pos, button)


def _canvas_pixel(canvas, map_x, map_y):
    """Convert a map coordinate to the canvas pixel QPoint."""
    from qgis.PyQt.QtCore import QPoint
    from qgis.core import QgsPointXY

    p = canvas.getCoordinateTransform().transform(QgsPointXY(map_x, map_y))
    return QPoint(int(round(p.x())), int(round(p.y())))


def _expected_map(canvas, map_x, map_y):
    """Map coordinate the tool actually sees for a click at (map_x, map_y).

    The canvas transform is pixel-quantized; round-tripping the rounded
    pixel gives the exact coordinate under test, so assertions can be tight.
    """
    pixel = _canvas_pixel(canvas, map_x, map_y)
    return canvas.getCoordinateTransform().toMapCoordinates(pixel.x(), pixel.y())


def _key_event(key):
    from qgis.PyQt.QtCore import QEvent, Qt
    from qgis.PyQt.QtGui import QKeyEvent

    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


@requires_qgis
class TestSWE2DLineDrawTool(unittest.TestCase):
    """SWE2DLineDrawTool against a real offscreen QgsMapCanvas."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from qgis.core import QgsRectangle
        from qgis.gui import QgsMapCanvas

        from swe2d.workbench.map_tools import SWE2DLineDrawTool

        self.canvas = QgsMapCanvas()
        self.canvas.resize(400, 400)
        self.canvas.setExtent(QgsRectangle(0.0, 0.0, 100.0, 100.0))
        self.tool = SWE2DLineDrawTool(self.canvas)
        self.canvas.setMapTool(self.tool)

    def tearDown(self):
        self.canvas.setMapTool(None)
        delete_widgets_now(self.tool, self.canvas)

    def _press(self, map_x, map_y, button=None):
        from qgis.PyQt.QtCore import QEvent, Qt

        btn = button if button is not None else Qt.MouseButton.LeftButton
        event = _map_mouse_event(
            self.canvas, QEvent.Type.MouseButtonPress, _canvas_pixel(self.canvas, map_x, map_y), btn
        )
        self.tool.canvasPressEvent(event)

    def _double_click(self, map_x, map_y):
        from qgis.PyQt.QtCore import QEvent, Qt

        event = _map_mouse_event(
            self.canvas,
            QEvent.Type.MouseButtonDblClick,
            _canvas_pixel(self.canvas, map_x, map_y),
            Qt.MouseButton.LeftButton,
        )
        self.tool.canvasDoubleClickEvent(event)

    def test_left_clicks_build_rubberband_and_right_click_emits_geometry(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.line_finished)
        self._press(20.0, 20.0)
        self._press(60.0, 70.0)
        self.assertEqual(len(self.tool._points), 2)
        # Rubber band mirrors the collected vertices.
        self.assertEqual(self.tool._rb.numberOfVertices(), 2)

        self._press(0.0, 0.0, button=self._right_button())
        self.assertEqual(len(spy), 1)
        geom = spy[0][0]
        pts = geom.asPolyline()
        self.assertEqual(len(pts), 2)
        exp0 = _expected_map(self.canvas, 20.0, 20.0)
        exp1 = _expected_map(self.canvas, 60.0, 70.0)
        self.assertAlmostEqual(pts[0].x(), exp0.x(), places=6)
        self.assertAlmostEqual(pts[0].y(), exp0.y(), places=6)
        self.assertAlmostEqual(pts[1].x(), exp1.x(), places=6)
        self.assertAlmostEqual(pts[1].y(), exp1.y(), places=6)
        # Tool state resets after finishing.
        self.assertEqual(self.tool._points, [])
        self.assertEqual(self.tool._rb.numberOfVertices(), 0)

    @staticmethod
    def _right_button():
        from qgis.PyQt.QtCore import Qt

        return Qt.MouseButton.RightButton

    def test_move_event_extends_rubberband_preview(self):
        from qgis.PyQt.QtCore import QEvent, Qt

        # Move with no points is a documented no-op.
        event = _map_mouse_event(
            self.canvas,
            QEvent.Type.MouseMove,
            _canvas_pixel(self.canvas, 50.0, 50.0),
            Qt.MouseButton.NoButton,
        )
        self.tool.canvasMoveEvent(event)
        self.assertEqual(self.tool._rb.numberOfVertices(), 0)

        self._press(20.0, 20.0)
        event = _map_mouse_event(
            self.canvas,
            QEvent.Type.MouseMove,
            _canvas_pixel(self.canvas, 80.0, 80.0),
            Qt.MouseButton.NoButton,
        )
        self.tool.canvasMoveEvent(event)
        # One fixed vertex + one preview vertex.
        self.assertEqual(self.tool._rb.numberOfVertices(), 2)
        self.assertEqual(len(self.tool._points), 1)

    def test_double_click_adds_point_and_finishes(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.line_finished)
        self._press(10.0, 10.0)
        self._double_click(90.0, 90.0)
        self.assertEqual(len(spy), 1)
        pts = spy[0][0].asPolyline()
        self.assertEqual(len(pts), 2)
        exp1 = _expected_map(self.canvas, 90.0, 90.0)
        self.assertAlmostEqual(pts[1].x(), exp1.x(), places=6)
        self.assertEqual(self.tool._points, [])

    def test_right_click_with_fewer_than_two_points_emits_nothing(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.line_finished)
        self._press(20.0, 20.0)
        self._press(0.0, 0.0, button=self._right_button())
        self.assertEqual(len(spy), 0)
        self.assertEqual(self.tool._points, [])
        self.assertEqual(self.tool._rb.numberOfVertices(), 0)

    def test_escape_cancels_in_progress_line(self):
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.line_finished)
        self._press(20.0, 20.0)
        self._press(40.0, 40.0)
        self.assertEqual(self.tool._rb.numberOfVertices(), 2)

        self.tool.keyPressEvent(_key_event(Qt.Key.Key_Escape))
        self.assertEqual(self.tool._points, [])
        self.assertEqual(self.tool._rb.numberOfVertices(), 0)

        # A subsequent right-click must not resurrect the cancelled line.
        self._press(0.0, 0.0, button=self._right_button())
        self.assertEqual(len(spy), 0)

    def test_deactivate_resets_rubberband(self):
        self._press(20.0, 20.0)
        self._press(40.0, 40.0)
        self.assertEqual(self.tool._rb.numberOfVertices(), 2)
        self.tool.deactivate()
        self.assertEqual(self.tool._rb.numberOfVertices(), 0)


def _make_network_graph():
    """Real DrainageGraph: L1 n1->n2, L2 n2->n3, L3 n3->n4, L9 n8->n9 (detached).

    n2 has an extra outgoing stub (LB) so first-click orientation on L2
    exercises the out-degree tie-break (out_deg(from) > out_deg(to) -> reverse).
    """
    from swe2d.workbench.services.drainage_graph_service import DrainageGraph

    from_node = {"L1": "n1", "L2": "n2", "L3": "n3", "L9": "n8", "LB": "n2"}
    to_node = {"L1": "n2", "L2": "n3", "L3": "n4", "L9": "n9", "LB": "nB"}
    outgoing = {"n1": ["L1"], "n2": ["L2", "LB"], "n3": ["L3"], "n8": ["L9"]}
    incoming = {
        "n2": ["L1"],
        "n3": ["L2"],
        "n4": ["L3"],
        "n9": ["L9"],
        "nB": ["LB"],
    }
    both = {
        "n1": ["L1"],
        "n2": ["L1", "L2", "LB"],
        "n3": ["L2", "L3"],
        "n4": ["L3"],
        "n8": ["L9"],
        "n9": ["L9"],
        "nB": ["LB"],
    }
    return DrainageGraph(
        node_ids=["n1", "n2", "n3", "n4", "n8", "n9", "nB"],
        link_ids=["L1", "L2", "L3", "L9", "LB"],
        from_node=from_node,
        to_node=to_node,
        outgoing=outgoing,
        incoming=incoming,
        both=both,
    )


@requires_qgis
class TestNetworkProfileMapTool(unittest.TestCase):
    """NetworkProfileMapTool: click-to-chain against a real memory layer."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from qgis.PyQt.QtCore import QVariant
        from qgis.gui import QgsMapCanvas

        from swe2d.workbench.views.network_profile_map_tool import NetworkProfileMapTool

        # Geometry matches the logical graph: disjoint segments so identify
        # is unambiguous.  Midpoint click targets in comments.
        self.layer = make_memory_layer(
            "LineString",
            fields=[("link_id", QVariant.String)],
            features=[
                ("LineString (10 50, 28 50)", ("L1",)),   # click (19, 50)
                ("LineString (32 50, 48 50)", ("L2",)),   # click (40, 50)
                ("LineString (52 50, 68 50)", ("L3",)),   # click (60, 50)
                ("LineString (10 10, 28 10)", ("L9",)),   # click (19, 10)
                ("LineString (32 10, 48 10)", ("LB",)),   # click (40, 10)
                ("LineString (52 10, 68 10)", ("LX",)),   # click (60, 10) — not in graph
            ],
            name="drainage_links",
        )
        self.graph = _make_network_graph()

        self.canvas = QgsMapCanvas()
        self.canvas.resize(400, 400)
        self.canvas.setLayers([self.layer])
        self.canvas.setExtent(self.layer.extent())
        self.canvas.refresh()

        self.tool = NetworkProfileMapTool(self.canvas, self.layer, self.graph)
        self.canvas.setMapTool(self.tool)

    def tearDown(self):
        self.canvas.setMapTool(None)
        delete_widgets_now(self.tool, self.canvas)
        self.layer = None

    def _release(self, map_x, map_y, button=None):
        from qgis.PyQt.QtCore import QEvent, Qt

        btn = button if button is not None else Qt.MouseButton.LeftButton
        event = _map_mouse_event(
            self.canvas,
            QEvent.Type.MouseButtonRelease,
            _canvas_pixel(self.canvas, map_x, map_y),
            btn,
        )
        self.tool.canvasReleaseEvent(event)

    def test_first_click_starts_chain_forward(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.chain_extended)
        self._release(19.0, 50.0)  # L1: out_deg(n1)=1 <= out_deg(n2)=2
        self.assertEqual(len(spy), 1)
        chain = spy[0][0]
        self.assertEqual(chain.link_specs, [("L1", False)])

    def test_first_click_orientation_reverse_on_outdegree_tiebreak(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.chain_extended)
        self._release(40.0, 50.0)  # L2: out_deg(n2)=2 > out_deg(n3)=1 -> reverse
        self.assertEqual(len(spy), 1)
        self.assertEqual(spy[0][0].link_specs, [("L2", True)])

    def test_chain_extends_downstream_across_clicks(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.chain_extended)
        self._release(19.0, 50.0)  # L1
        self._release(40.0, 50.0)  # L2 connects at n2
        self._release(60.0, 50.0)  # L3 connects at n3
        self.assertEqual(len(spy), 3)
        final = spy[-1][0]
        self.assertEqual(
            final.link_specs, [("L1", False), ("L2", False), ("L3", False)]
        )

    def test_same_link_click_is_ignored(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.tool.chain_extended)
        self._release(19.0, 50.0)
        self._release(19.0, 50.0)  # same link again — no new emission
        self.assertEqual(len(spy), 1)

    def test_unconnected_link_emits_pick_rejected(self):
        from qgis.PyQt.QtTest import QSignalSpy

        ext = QSignalSpy(self.tool.chain_extended)
        rej = QSignalSpy(self.tool.pick_rejected)
        self._release(19.0, 50.0)  # L1 — chain starts, downstream = n2
        self._release(19.0, 10.0)  # L9 — in graph but detached from the chain
        self.assertEqual(len(ext), 1)
        self.assertEqual(len(rej), 1)
        reason, link_id = rej[0]
        self.assertIn("does not connect", reason)
        self.assertEqual(link_id, "L9")

    def test_link_not_in_graph_emits_pick_rejected(self):
        from qgis.PyQt.QtTest import QSignalSpy

        rej = QSignalSpy(self.tool.pick_rejected)
        self._release(60.0, 10.0)  # LX — geometry exists, not in DrainageGraph
        self.assertEqual(len(rej), 1)
        reason, link_id = rej[0]
        self.assertEqual(reason, "link not in drainage network")
        self.assertEqual(link_id, "LX")

    def test_click_on_empty_canvas_is_silent(self):
        from qgis.PyQt.QtTest import QSignalSpy

        ext = QSignalSpy(self.tool.chain_extended)
        rej = QSignalSpy(self.tool.pick_rejected)
        self._release(19.0, 30.0)  # between the two rows of links — no feature
        self.assertEqual(len(ext), 0)
        self.assertEqual(len(rej), 0)

    def test_right_click_finishes_and_emits_chain(self):
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QSignalSpy

        fin = QSignalSpy(self.tool.finished)
        self._release(19.0, 50.0)
        self._release(40.0, 50.0)
        self._release(0.0, 0.0, button=Qt.MouseButton.RightButton)
        self.assertEqual(len(fin), 1)
        self.assertEqual(fin[0][0].link_specs, [("L1", False), ("L2", False)])

    def test_escape_finishes_chain(self):
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QSignalSpy

        fin = QSignalSpy(self.tool.finished)
        self._release(19.0, 50.0)
        self.tool.keyPressEvent(_key_event(Qt.Key.Key_Escape))
        self.assertEqual(len(fin), 1)
        self.assertEqual(fin[0][0].link_specs, [("L1", False)])

    def test_double_click_finishes_chain(self):
        from qgis.PyQt.QtCore import QEvent, Qt
        from qgis.PyQt.QtTest import QSignalSpy

        fin = QSignalSpy(self.tool.finished)
        self._release(19.0, 50.0)
        event = _map_mouse_event(
            self.canvas,
            QEvent.Type.MouseButtonDblClick,
            _canvas_pixel(self.canvas, 19.0, 50.0),
            Qt.MouseButton.LeftButton,
        )
        self.tool.canvasDoubleClickEvent(event)
        self.assertEqual(len(fin), 1)
        self.assertEqual(fin[0][0].link_specs, [("L1", False)])


if __name__ == "__main__":
    unittest.main()
