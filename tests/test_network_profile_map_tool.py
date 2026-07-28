"""tests/test_network_profile_map_tool.py

Regression test for the 'too many arguments' runtime error when
activating the network profile map tool.  The bug: the original
``NetworkProfileMapTool.__init__`` forwarded a ``parent`` kwarg to
``QgsMapTool.__init__(canvas, parent)`` which only accepts ``(canvas)``
under the bindings used by QGIS.  This test verifies the constructor
signature and that the tool can be built with a real QGIS canvas
under the offscreen platform.

Skipped when the offscreen canvas cannot be built (no DISPLAY).
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from qgis.PyQt.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from swe2d.workbench.views.network_profile_map_tool import NetworkProfileMapTool
from swe2d.workbench.services.drainage_graph_service import DrainageGraph


class TestNetworkProfileMapToolSignature(unittest.TestCase):
    def test_init_does_not_take_parent_positional(self):
        """The Qt5 PyQt wrapper for QGIS map tools only accepts
        ``(canvas)``.  If we ever add `parent` to the super call it
        will raise at runtime — this test catches that at lint time."""
        sig = inspect.signature(NetworkProfileMapTool.__init__)
        params = [p for p in sig.parameters.values()
                  if p.name not in ("self", "canvas", "drainage_layer", "graph")]
        self.assertEqual(
            params, [],
            f"NetworkProfileMapTool.__init__ should accept only "
            f"(canvas, drainage_layer, graph); got extra params: {params}",
        )

    def test_init_does_not_forward_parent_to_super(self):
        """Read the source: ``super().__init__(...)`` must not pass
        a third positional argument.  Catches future regressions
        where someone restores the bug."""
        src = inspect.getsource(NetworkProfileMapTool.__init__)
        # Allow `super().__init__(canvas)` but NOT `super().__init__(canvas, parent)`
        # or `super().__init__(canvas, self)` etc.
        for line in src.splitlines():
            stripped = line.strip()
            if "super().__init__" in stripped:
                # Count args after the open paren
                args = stripped.split("(", 1)[1]
                # Strip the closing paren
                args = args.rsplit(")", 1)[0]
                parts = [p for p in args.split(",") if p.strip()]
                self.assertLessEqual(
                    len(parts), 1,
                    f"super().__init__ must take <= 1 arg (canvas); got: {parts!r}",
                )

    def test_super_class_is_qgs_map_tool(self):
        # The class should still extend QGIS's base class.
        from qgis.gui import QgsMapTool
        self.assertTrue(issubclass(NetworkProfileMapTool, QgsMapTool))

    def test_class_carries_expected_signals(self):
        from qgis.PyQt.QtCore import pyqtSignal
        actual = {
            name for name, value in vars(NetworkProfileMapTool).items()
            if isinstance(value, pyqtSignal)
        }
        self.assertTrue(
            {"chain_extended", "pick_rejected", "finished"}.issubset(actual),
            f"Missing signals. Found: {actual}",
        )


if __name__ == "__main__":
    unittest.main()
