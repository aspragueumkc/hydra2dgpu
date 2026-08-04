#!/usr/bin/env python3
"""Regression test for QGIS PyQt5 + pyqtgraph itemChange callbacks."""
from __future__ import annotations

import gc
import importlib.util
import sys
import unittest

from tests.qgis_real_env import delete_widgets_now, ensure_qgis_app, requires_qgis

_HAVE_PG = importlib.util.find_spec("pyqtgraph") is not None


@requires_qgis
@unittest.skipUnless(_HAVE_PG, "pyqtgraph required")
class TestPyQtGraphQVariantCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_item_change_callbacks_do_not_raise_qvariant_conversion_errors(self):
        """Removing real PG items must not invoke an uncaught callback error."""
        errors = []
        previous_hook = sys.excepthook

        def capture(exc_type, exc_value, exc_tb):
            errors.append((exc_type.__name__, str(exc_value)))

        sys.excepthook = capture
        plot = None
        try:
            # Install the capture hook before constructing/removing items;
            # PyQt5's virtual-method error path delegates to sys.excepthook.
            from swe2d.workbench.views import studio_viewer_pg as viewer_pg
            import pyqtgraph as pg

            self.assertTrue(viewer_pg._HAVE_PG)
            plot = pg.PlotWidget()
            plot.resize(320, 240)
            plot.show()
            plot.addLegend()
            plot.plot([0.0, 1.0, 2.0], [1.0, 2.0, 1.5], name="curve")
            plot.addItem(pg.InfiniteLine(pos=0.5, angle=90, movable=False))
            ensure_qgis_app().processEvents()
            plot.clear()
            ensure_qgis_app().processEvents()
        finally:
            if plot is not None:
                plot.close()
                delete_widgets_now(plot)
            gc.collect()
            sys.excepthook = previous_hook

        self.assertEqual(errors, [], f"uncaught Qt callback errors: {errors!r}")


if __name__ == "__main__":
    unittest.main()
