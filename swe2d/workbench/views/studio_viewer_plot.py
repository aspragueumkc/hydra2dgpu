"""PlotViewWidget — a matplotlib plot + optional data table for one mode.

Owns a Figure + FigureCanvas + optional toolbar + optional QTableWidget.
The table lives below the plot in a QSplitter and is hidden by default.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from qgis.PyQt import QtCore, QtWidgets

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    _HAVE_MPL = True
    _NavigationToolbar = None
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar
    except ImportError as _e:

        try:

            dialog._log(f"[ERROR] ImportError in studio_viewer_plot.py: {_e}")

        except Exception:

            pass
except ImportError:
    _HAVE_MPL = False
    Figure = FigureCanvas = None  # type: ignore


class PlotViewWidget(QtWidgets.QWidget):
    """A matplotlib canvas with an optional data table.

    Usage:
        widget = PlotViewWidget("Time Series")
        widget.set_render_fn(my_render_function)
        widget.set_data(mesh_data={...}, result_data={...}, h_min=1e-6)
        widget.refresh()
        widget.show_table_toggle.setChecked(True)  # show the table
    """

    def __init__(self, mode: str = "Mesh", parent=None):
        super().__init__(parent)
        self._mode = str(mode)
        self._render_fn: Optional[Callable] = None
        self._mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6
        self._fig: Any = None
        self._canvas: Any = None
        self.show_table_toggle: Optional[QtWidgets.QCheckBox] = None
        self._table_widget: Optional[QtWidgets.QTableWidget] = None
        self._selected_metric: str = "flow"
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the plot widget with matplotlib canvas, toolbar, data table, and mode-specific selectors."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not _HAVE_MPL:
            label = QtWidgets.QLabel("Matplotlib Qt backend not available.")
            label.setWordWrap(True)
            root.addWidget(label)
            return

        self._fig = Figure(figsize=(6.4, 4.2), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(200)

        self._table_widget = QtWidgets.QTableWidget()
        self._table_widget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table_widget.setAlternatingRowColors(True)
        self._table_widget.horizontalHeader().setStretchLastSection(True)
        self._table_widget.setVisible(False)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        if _NavigationToolbar:
            toolbar = _NavigationToolbar(self._canvas, self)
            toolbar.setIconSize(QtCore.QSize(16, 16))
            wrapper = QtWidgets.QWidget()
            wl = QtWidgets.QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(0)
            wl.addWidget(toolbar)
            wl.addWidget(self._canvas, 1)
            splitter.addWidget(wrapper)
        else:
            splitter.addWidget(self._canvas)
        splitter.addWidget(self._table_widget)

        # Top bar — minimal (Mesh tab only; Time Series has its own pyqtgraph widget)
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addStretch(1)

        top_bar.addStretch(1)
        self.show_table_toggle = QtWidgets.QCheckBox("Show data table")
        self.show_table_toggle.setChecked(False)
        self.show_table_toggle.setToolTip("Show/hide the data table below the plot.")
        self.show_table_toggle.toggled.connect(self._on_table_toggle)
        top_bar.addWidget(self.show_table_toggle)

        root.addLayout(top_bar)
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # (Profile and Network mode-specific handlers removed — now use pyqtgraph)
    # ------------------------------------------------------------------

    @property
    def selected_metric(self) -> str:
        """The currently selected metric for the plot."""
        return self._selected_metric

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        """Set the selected metric by data value."""
        self._selected_metric = str(metric) if metric else "flow"

    def _on_table_toggle(self, visible: bool) -> None:
        """Show or hide the data table on toggle."""
        if self._table_widget is not None:
            self._table_widget.setVisible(visible)
            if visible:
                self._populate_table()



    def _populate_table(self) -> None:
        """Fill the data table from coupling records or GPKG results."""
        if self._table_widget is None:
            return
        data = self._result_data
        self._table_widget.setRowCount(0)
        self._table_widget.setColumnCount(0)
        if data is None:
            return

        records = []
        cols = []

        mode = self._mode
        if mode == "Mesh":
            return

        # Time Series data table (GPKG query)
        line_id = getattr(data, "_line_id", -1)
        for rec in getattr(data, "_run_records", []) or []:
            if not rec.enabled or not hasattr(rec, "run_id"):
                continue
            gpkg = getattr(rec, "gpkg_path", "")
            run_id = getattr(rec, "run_id", "")
            if not gpkg or not run_id:
                continue
            try:
                from swe2d.results.db_utils import get_table_contents, get_table_info
                tbl = "swe2d_baked_line_ts"
                info = get_table_info(gpkg, tbl)
                if info and "run_id" in info:
                    rows = get_table_contents(gpkg, tbl, limit=200)
                    for r in rows:
                        records.append(dict(zip(info, r)))
                    cols = info
                    break
            except Exception:
                pass

        if not records or not cols:
            return

        self._table_widget.setColumnCount(len(cols))
        self._table_widget.setHorizontalHeaderLabels(cols)
        n = min(len(records), 5000)
        self._table_widget.setRowCount(n)
        for i, r in enumerate(records[:n]):
            for j, c in enumerate(cols):
                val = r.get(c, "") if isinstance(r, dict) else r[j] if j < len(r) else ""
                self._table_widget.setItem(i, j, QtWidgets.QTableWidgetItem("" if val is None else str(val)))

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """The plot mode (Mesh)."""
        return self._mode

    @property
    def canvas(self):
        """The matplotlib FigureCanvas widget."""
        return self._canvas

    @property
    def fig(self):
        """The matplotlib Figure instance."""
        return self._fig

    def set_render_fn(self, fn: Callable) -> None:
        """Set the render function callback used by refresh()."""
        self._render_fn = fn

    def set_data(
        self,
        mesh_data: Optional[Dict[str, np.ndarray]] = None,
        result_data: Any = None,
        h_min: float = 1.0e-6,
    ) -> None:
        """Set mesh data, result data, and h_min for the current mode."""
        if mesh_data is not None:
            self._mesh_data = mesh_data
        if result_data is not None:
            self._result_data = result_data
            if self._table_widget is not None and self._table_widget.isVisible():
                self._populate_table()
        self._h_min = float(h_min)

    def refresh(self) -> None:
        """Re-render the plot and optionally repopulate the data table."""
        if not _HAVE_MPL or self._fig is None:
            return
        from swe2d.plotting.viewer_plots import render_viewer_figure
        render_viewer_figure(
            fig=self._fig,
            mesh_data=self._mesh_data,
            result_data=self._result_data,
            mode=self._mode,
            h_min=self._h_min,
        )
        self._canvas.draw_idle()
        if self._table_widget is not None and self._table_widget.isVisible():
            self._populate_table()
