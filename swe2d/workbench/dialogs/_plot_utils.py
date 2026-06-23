#!/usr/bin/env python3
"""Shared matplotlib Qt integration utilities for workbench dialogs."""

from __future__ import annotations

import logging

logger_wb = logging.getLogger(__name__)


def try_import_matplotlib_qt():
    """Try to import matplotlib Qt backend, falling back from qt5agg to qtagg."""
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import matplotlib.tri as mtri
        return FigureCanvas, Figure, mtri
    except ImportError:
        logger_wb.warning("Exception in primary path — attempting fallback", exc_info=True)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import matplotlib.tri as mtri
            return FigureCanvas, Figure, mtri
        except ImportError:
            logger_wb.warning("Graceful degradation — Exception returned fallback value", exc_info=True)
            return None, None, None
