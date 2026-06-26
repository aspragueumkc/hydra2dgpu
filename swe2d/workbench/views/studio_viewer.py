"""SWE2DStudioViewer — plot tab panel for the "HYDRA2D View" dock.

Owns a QTabWidget with 5 tabs, each a PlotViewWidget that can optionally
show a coupling data table below the plot when the user toggles it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from qgis.PyQt import QtWidgets

from swe2d.workbench.views.studio_viewer_plot import PlotViewWidget
from swe2d.workbench.views.studio_viewer_pg import PGTimeSeriesWidget, _HAVE_PG

_TAB_MODES = ["Mesh", "Time Series", "Profile", "Structure", "Network"]


class SWE2DStudioViewer(QtWidgets.QWidget):
    """The entire HYDRA2D View panel — one widget, one dock, 5 plot tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh_data: Optional[Dict[str, Any]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6

        self._tabs: QtWidgets.QTabWidget = None
        self._plot_widgets: Dict[str, PlotViewWidget] = {}

        self._build_ui()
        self._register_default_renderers()

    def _build_ui(self) -> None:
        """Build the tab widget with 5 plot mode tabs."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        for mode in _TAB_MODES:
            if mode == "Time Series" and _HAVE_PG:
                widget = PGTimeSeriesWidget()
            else:
                widget = PlotViewWidget(mode=mode)
            self._plot_widgets[mode] = widget
            self._tabs.addTab(widget, mode)

        layout.addWidget(self._tabs, 1)

    def _on_tab_changed(self, idx: int) -> None:
        """Handle tab change — load coupling data for Structure/Network tabs and refresh."""
        widget = self._tabs.widget(idx)
        if widget is None:
            return
        mode = getattr(widget, "_mode", "")
        if mode in ("Structure", "Network") and self._result_data is not None:
            for rec in getattr(self._result_data, "_run_records", []):
                if rec.enabled and hasattr(rec, 'run_id'):
                    self._result_data.load_coupling_records(rec.run_id)
                    break
            widget._populate_metric_combo()
        widget.refresh()

    def _register_default_renderers(self) -> None:
        """Renderers are dispatched by swe2d.plotting.viewer_plots — no per-widget registration needed."""

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
        """Return the currently visible PlotViewWidget tab."""
        return self._tabs.currentWidget()

    def refresh(self) -> None:
        """Refresh the currently visible plot widget."""
        current = self.current_widget
        if current is not None and hasattr(current, "refresh"):
            current.refresh()

    @property
    def tab_widget(self) -> QtWidgets.QTabWidget:
        """The internal QTabWidget."""
        return self._tabs

    @property
    def plot_widgets(self) -> Dict[str, PlotViewWidget]:
        """All plot widgets keyed by mode name."""
        return self._plot_widgets
