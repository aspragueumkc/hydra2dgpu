"""SWE2DStudioViewer — plot tab panel for the "HYDRA2D View" dock.

Owns a QTabWidget with tabs:
  Mesh — pyqtgraph mesh wireframe viewer
  Time Series — pyqtgraph line/structure/drainage time-series (unified)
  Profile — pyqtgraph longitudinal profile viewer
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from qgis.PyQt import QtCore, QtWidgets

from swe2d.workbench.views.studio_viewer_pg import PGTimeSeriesWidget, _HAVE_PG
from swe2d.workbench.views.studio_viewer_profile_pg import PGProfileWidget

_TAB_MODES = ["Mesh", "Time Series", "Profile"]


class SWE2DStudioViewer(QtWidgets.QWidget):
    """The entire HYDRA2D View panel — one widget, one dock, plot tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh_data: Optional[Dict[str, Any]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6

        self._tabs: QtWidgets.QTabWidget = None
        self._plot_widgets: Dict[str, Any] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the tab widget with plot tabs."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        for mode in _TAB_MODES:
            if mode == "Time Series" and _HAVE_PG:
                widget: Any = PGTimeSeriesWidget()
            elif mode == "Profile":
                widget = PGProfileWidget()
            else:
                # Use matplotlib PlotViewWidget for Mesh
                from swe2d.workbench.views.studio_viewer_plot import PlotViewWidget
                widget = PlotViewWidget(mode=mode)
            self._plot_widgets[mode] = widget
            self._tabs.addTab(widget, mode)

        # Allow the viewer panel to shrink horizontally — Qt often enforces
        # large minimum widths based on widget sizeHints (pyqtgraph plots,
        # combo boxes, labels, etc.).  Setting minimumWidth(0) on the tab
        # widget and all children removes those constraints.
        self._tabs.setMinimumWidth(0)
        for child in self._tabs.findChildren(QtWidgets.QWidget):
            child.setMinimumWidth(0)

        layout.addWidget(self._tabs, 1)

    def _on_tab_changed(self, idx: int) -> None:
        """Handle tab change — refresh the newly selected widget."""
        widget = self._tabs.widget(idx)
        if widget is None:
            return
        if hasattr(widget, "refresh"):
            widget.refresh()

    def set_mesh_data(self, mesh: Optional[Dict[str, Any]]) -> None:
        """Set mesh data on all plot widgets."""
        self._mesh_data = mesh
        for w in self._plot_widgets.values():
            w.set_data(mesh_data=mesh)

    def set_result_data(self, result: Any) -> None:
        """Set result data on all plot widgets."""
        self._result_data = result
        for w in self._plot_widgets.values():
            w.set_data(result_data=result)

    def set_h_min(self, h_min: float) -> None:
        """Set the minimum depth threshold on all plot widgets."""
        self._h_min = float(h_min)
        for w in self._plot_widgets.values():
            w.set_data(h_min=float(h_min))

    @property
    def current_widget(self):
        """Return the currently visible plot tab."""
        return self._tabs.currentWidget()

    def refresh(self) -> None:
        """Refresh all plot widgets."""
        for w in self._plot_widgets.values():
            if hasattr(w, "refresh"):
                try:
                    w.refresh()
                except Exception:
                    pass

    @property
    def tab_widget(self) -> QtWidgets.QTabWidget:
        """The internal QTabWidget."""
        return self._tabs

    @property
    def plot_widgets(self) -> Dict[str, Any]:
        """All plot widgets keyed by mode name."""
        return self._plot_widgets
