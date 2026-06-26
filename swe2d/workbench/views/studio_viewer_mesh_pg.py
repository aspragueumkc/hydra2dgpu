"""PyQtGraph-based Mesh view widget — drop-in for PlotViewWidget("Mesh").

Renders a mesh wireframe (cell triangulation) with optional
depth/velocity color fills using pyqtgraph instead of matplotlib.

Protocol matches PGTimeSeriesWidget: set_data(), refresh(), etc.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from qgis.PyQt import QtCore, QtGui, QtWidgets
from qgis.PyQt.QtCore import Qt

try:
    import pyqtgraph as pg
    _HAVE_PG = True
except ImportError:
    _HAVE_PG = False


class PGMeshWidget(QtWidgets.QWidget):
    """pyqtgraph-based mesh viewer.

    Auto-detects render mode:
    - If node_z has meaningful (non-zero) elevations → fill by bed elevation
    - If node_z is missing or all zeros → wireframe only
    """

    # Terrain colormap (greens→browns→whites, like a hillshade)
    _TERRAIN_CMAP = [
        (0.0, (0, 100, 0)),
        (0.2, (120, 180, 50)),
        (0.4, (210, 190, 140)),
        (0.6, (180, 140, 100)),
        (0.8, (160, 120, 80)),
        (1.0, (240, 230, 210)),
    ]
    _DEPTH_CMAP = [
        (0.0, (240, 249, 232)),
        (0.25, (186, 228, 188)),
        (0.5, (123, 204, 196)),
        (0.75, (43, 140, 190)),
        (1.0, (8, 29, 88)),
    ]
    _VEL_CMAP = [
        (0.0, (255, 255, 255)),
        (0.25, (200, 200, 200)),
        (0.5, (128, 0, 128)),
        (0.75, (220, 50, 50)),
        (1.0, (255, 200, 50)),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "Mesh"
        self._mesh_data: Optional[Dict[str, Any]] = None
        self._result_data: Any = None
        self._h_min: float = 1.0e-6

        self._plot_widget: Optional[pg.PlotWidget] = None
        self._selected_element_id: str = ""
        self._selected_metric: str = "auto"  # auto → wireframe or bed_elevation
        self._metric_combo: Optional[QtWidgets.QComboBox] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build: top bar with render selector, pyqtgraph plot."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if not _HAVE_PG:
            root.addWidget(QtWidgets.QLabel("pyqtgraph not available."))
            return

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addStretch(1)
        lbl = QtWidgets.QLabel("Render:")
        self._metric_combo = QtWidgets.QComboBox()
        self._metric_combo.addItem("Auto (wireframe / bed elevation)", "auto")
        self._metric_combo.addItem("Wireframe only", "mesh")
        self._metric_combo.addItem("Bed elevation fill", "bed_elevation")
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        top_bar.addWidget(lbl)
        top_bar.addWidget(self._metric_combo)
        root.addLayout(top_bar)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(200)
        self._plot_widget.setBackground("white")
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.setLabel("bottom", "X")
        self._plot_widget.setLabel("left", "Y")
        root.addWidget(self._plot_widget, 1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_metric_changed(self) -> None:
        self._selected_metric = str(self._metric_combo.currentData() or "mesh")
        self.refresh()

    # ------------------------------------------------------------------
    # Public protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def selected_metric(self) -> str:
        return self._selected_metric

    @selected_metric.setter
    def selected_metric(self, metric: str) -> None:
        self._selected_metric = str(metric) if metric else "auto"
        idx = self._metric_combo.findData(self._selected_metric)
        if idx >= 0:
            self._metric_combo.setCurrentIndex(idx)

    @property
    def selected_element_id(self) -> str:
        return self._selected_element_id

    @selected_element_id.setter
    def selected_element_id(self, element_id: str) -> None:
        self._selected_element_id = str(element_id) if element_id else ""

    def set_data(
        self,
        mesh_data: Optional[Dict[str, np.ndarray]] = None,
        result_data: Any = None,
        h_min: float = 1.0e-6,
    ) -> None:
        if mesh_data is not None:
            self._mesh_data = mesh_data
        if result_data is not None:
            self._result_data = result_data
        self._h_min = float(h_min)

    def set_render_fn(self, fn) -> None:
        pass

    def _has_bed_elevation(self) -> bool:
        """Return True if node_z exists and has meaningful (non-zero) values."""
        md = self._mesh_data
        if md is None:
            return False
        nz = md.get("node_z")
        if nz is None or len(nz) == 0:
            return False
        try:
            arr = np.asarray(nz, dtype=np.float64).ravel()
            return bool(np.any(np.abs(arr) > 1e-12))
        except (TypeError, ValueError):
            return False

    def refresh(self) -> None:
        """Re-render the mesh view — auto-detect wireframe vs bed elevation."""
        if not _HAVE_PG or self._plot_widget is None:
            return
        self._plot_widget.clear()

        md = self._mesh_data
        if md is None:
            self._plot_widget.plot([0], [0], pen=None)
            t = pg.TextItem("No mesh data", anchor=(0.5, 0.5), color=(128, 128, 128))
            self._plot_widget.addItem(t)
            return

        nx = md.get("node_x")
        ny = md.get("node_y")
        if nx is None or ny is None or len(nx) == 0:
            return

        # Build triangulation — use cell_nodes if 2D (N,3), otherwise fan via face offsets
        cell_nodes = md.get("cell_nodes")
        off = md.get("cell_face_offsets")
        fv = md.get("cell_face_nodes")
        if cell_nodes is not None and hasattr(cell_nodes, "ndim") and cell_nodes.ndim == 2:
            tri = cell_nodes
        elif off is not None and fv is not None:
            tri = []
            for ci in range(len(off) - 1):
                f_start = int(off[ci])
                f_end = int(off[ci + 1])
                verts = [int(fv[j]) for j in range(f_start, f_end)]
                for k in range(1, len(verts) - 1):
                    tri.append([verts[0], verts[k], verts[k + 1]])
            tri = np.array(tri, dtype=np.int32)
        else:
            return

        # Guard: tri must be 2D (N,3) — if not, something went wrong in fan-out
        if not hasattr(tri, "ndim") or tri.ndim != 2 or tri.shape[1] != 3:
            return

        # Determine effective render mode
        mode = self._selected_metric
        if mode == "auto":
            do_fill = self._has_bed_elevation()
            mode = "bed_elevation" if do_fill else "mesh"

        if mode == "bed_elevation":
            nz = md.get("node_z")
            if nz is not None:
                vals = np.asarray(nz, dtype=np.float64).ravel()
                vmin, vmax = float(vals.min()), float(vals.max())
                if vmax - vmin < 1e-12:
                    vmax = vmin + 1.0
                cpos = np.array([c[0] for c in self._TERRAIN_CMAP])
                colors = np.array([c[1] for c in self._TERRAIN_CMAP], dtype=np.float64)
                norms = (vals - vmin) / (vmax - vmin)
                r = np.interp(norms, cpos, colors[:, 0])
                g = np.interp(norms, cpos, colors[:, 1])
                b = np.interp(norms, cpos, colors[:, 2])
                for i in range(min(tri.shape[0], 50000)):
                    idxs = tri[i]
                    if len(idxs) == 3:
                        ci = min(i, len(vals) - 1)
                        c = (int(r[ci]), int(g[ci]), int(b[ci]))
                        self._plot_widget.plot(
                            nx[list(idxs) + [idxs[0]]],
                            ny[list(idxs) + [idxs[0]]],
                            pen=pg.mkPen(color=c, width=0.5),
                            fillLevel=0,
                            brush=pg.mkBrush(c),
                        )

        # Wireframe overlay (always drawn)
        for i in range(min(tri.shape[0], 100000)):
            idxs = tri[i]
            nidxs = list(idxs) + [idxs[0]]
            pen_color = (100, 100, 100) if mode == "mesh" else (180, 180, 180)
            self._plot_widget.plot(
                nx[nidxs], ny[nidxs],
                pen=pg.mkPen(color=pen_color, width=0.3 if mode != "mesh" else 0.6),
            )

        self._plot_widget.plotItem.autoRange()

    def _populate_metric_combo(self) -> None:
        pass
