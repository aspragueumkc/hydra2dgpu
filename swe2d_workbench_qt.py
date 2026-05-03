#!/usr/bin/env python3
"""Qt workbench dialog for interactive 2D SWE setup and execution.

This module provides a focused GUI for:
- structured mesh generation
- side-based boundary condition assignment
- model parameter configuration
- solver execution with cancel/progress
- result visualization (mesh/depth/velocity)
"""

from __future__ import annotations

import concurrent.futures
import copy
import csv
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

try:
    from qgis.core import (
        QgsEditorWidgetSetup,
        QgsFieldConstraints,
        QgsFeature,
        QgsField,
        QgsGeometry,
        QgsProject,
        QgsPointXY,
        QgsRasterLayer,
        QgsUnitTypes,
        QgsVectorFileWriter,
        QgsVectorLayer,
        QgsWkbTypes,
    )
    from qgis.PyQt.QtCore import QVariant
    _HAVE_QGIS_CORE = True
except Exception:
    QgsEditorWidgetSetup = QgsFieldConstraints = None
    QgsFeature = QgsField = QgsGeometry = QgsPointXY = QgsProject = None
    QgsRasterLayer = QgsVectorLayer = QgsWkbTypes = None
    QgsUnitTypes = QgsVectorFileWriter = None
    QVariant = None
    _HAVE_QGIS_CORE = False

try:
    from swe2d_backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
except Exception:
    try:
        from .swe2d_backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
    except Exception:
        SWE2DBackend = None

        def swe2d_available() -> bool:
            return False

        def swe2d_gpu_available() -> bool:
            return False

try:
    import h5py as _h5py
    _HAVE_H5PY = True
except ImportError:
    _h5py = None
    _HAVE_H5PY = False

try:
    import netCDF4 as _netCDF4
    _HAVE_NETCDF4 = True
except ImportError:
    _netCDF4 = None
    _HAVE_NETCDF4 = False

try:
    from swe2d_meshing import conceptual_from_qgis_layers, generate_face_centric_mesh, _gmsh_available, _tqmesh_available
except Exception:
    try:
        from .swe2d_meshing import conceptual_from_qgis_layers, generate_face_centric_mesh, _gmsh_available, _tqmesh_available
    except Exception:
        conceptual_from_qgis_layers = None
        generate_face_centric_mesh = None
        def _gmsh_available() -> bool:
            return False
        def _tqmesh_available() -> bool:
            return False


def _try_import_matplotlib_qt():
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import matplotlib.tri as mtri
        return FigureCanvas, Figure, mtri
    except Exception:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import matplotlib.tri as mtri
            return FigureCanvas, Figure, mtri
        except Exception:
            return None, None, None


_BC_OPTIONS = [
    ("Wall (zero normal flux)", 1),
    ("Inflow Q (total discharge)", 2),
    ("Stage (prescribed WSE)", 3),
    ("Normal Depth (prescribed depth)", 6),
    ("Timeseries Flow Q", 102),
    ("Timeseries Stage", 103),
    ("Open (zero-gradient)", 4),
    ("Reflecting", 5),
]

_BC_TS_FLOW = 102
_BC_TS_STAGE = 103

_CELL_TYPE_OPTIONS = [
    "triangular",
    "quadrilateral",
    "cartesian",
    "empty",
]

_RECONSTRUCTION_OPTIONS = [
    ("First-order (baseline)",          0),
    ("MUSCL Fast (high-throughput)",     1),
    ("MUSCL MinMod (robust)",            2),
    ("MUSCL MC (less-diffusive TVD)",    3),
    ("MUSCL Van Leer (smooth TVD)",      4),
]

_BC_VALUE_MAP = {
    "Wall (zero normal flux)": 1,
    "Inflow Q (total discharge)": 2,
    "Stage (prescribed WSE)": 3,
    "Normal Depth (prescribed depth)": 6,
    "Timeseries Flow Q": 102,
    "Timeseries Stage": 103,
    "Open (zero-gradient)": 4,
    "Reflecting": 5,
}

_SWE2D_WORKBENCH_WINDOWS = []


def _run_topology_mesh_job(conceptual, backend_name: str):
    """Run heavy topology meshing work off the GUI thread/process."""
    # Use the already-imported function when available; fall back to local import
    # in subprocess contexts.
    gen = generate_face_centric_mesh
    if gen is None:
        try:
            from swe2d_meshing import generate_face_centric_mesh as gen  # type: ignore
        except Exception:
            from .swe2d_meshing import generate_face_centric_mesh as gen  # type: ignore
    return gen(conceptual, backend=backend_name)


def _clone_conceptual_without_constraints(conceptual):
    clone = copy.deepcopy(conceptual)
    clone.constraints = []
    return clone


class HydrographEditorDialog(QtWidgets.QDialog):
    def __init__(self, side: str, initial_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Hydrograph Editor - {side.capitalize()} Boundary")
        self.resize(560, 380)

        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Enter one point per row. Time accepts decimal hours or HH:MM(:SS). "
            "For Flow BCs, Value is total discharge Q."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Time", "Value (Q or stage)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row_btns = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("Add Row")
        self.remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        self.load_csv_btn = QtWidgets.QPushButton("Load CSV")
        self.save_csv_btn = QtWidgets.QPushButton("Save CSV")
        self.add_row_btn.clicked.connect(self._add_row)
        self.remove_row_btn.clicked.connect(self._remove_selected_rows)
        self.load_csv_btn.clicked.connect(self._load_csv)
        self.save_csv_btn.clicked.connect(self._save_csv)
        row_btns.addWidget(self.add_row_btn)
        row_btns.addWidget(self.remove_row_btn)
        row_btns.addStretch(1)
        row_btns.addWidget(self.load_csv_btn)
        row_btns.addWidget(self.save_csv_btn)
        root.addLayout(row_btns)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_text(initial_text)
        if self.table.rowCount() == 0:
            self._add_row()

    def _add_row(self, time_text: str = "", value_text: str = ""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(time_text)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(value_text)))

    def _remove_selected_rows(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _load_text(self, text: str):
        raw = str(text or "").strip()
        if not raw:
            return
        for chunk in raw.replace("\n", ";").split(";"):
            c = chunk.strip()
            if not c:
                continue
            if "," in c:
                a, b = c.split(",", 1)
            elif "=" in c:
                a, b = c.split("=", 1)
            else:
                continue
            self._add_row(a.strip(), b.strip())

    def _load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Hydrograph CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            rows = []
            with open(path, "r", encoding="utf-8", newline="") as f:
                for rec in csv.reader(f):
                    if len(rec) < 2:
                        continue
                    t = str(rec[0]).strip()
                    v = str(rec[1]).strip()
                    if not t or not v:
                        continue
                    if t.lower() in ("time", "hours"):
                        continue
                    rows.append((t, v))
            self.table.setRowCount(0)
            for t, v in rows:
                self._add_row(t, v)
            if self.table.rowCount() == 0:
                self._add_row()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Hydrograph CSV", f"Failed to load CSV: {exc}")

    def _save_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Hydrograph CSV", "hydrograph.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "value"])
                for r in range(self.table.rowCount()):
                    t_item = self.table.item(r, 0)
                    v_item = self.table.item(r, 1)
                    t = t_item.text().strip() if t_item else ""
                    v = v_item.text().strip() if v_item else ""
                    if not t and not v:
                        continue
                    w.writerow([t, v])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Hydrograph CSV", f"Failed to save CSV: {exc}")

    def hydrograph_text(self) -> str:
        parts = []
        for r in range(self.table.rowCount()):
            t_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            t = t_item.text().strip() if t_item else ""
            v = v_item.text().strip() if v_item else ""
            if not t and not v:
                continue
            parts.append(f"{t},{v}")
        return "; ".join(parts)


class SWE2DWorkbenchDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Workbench")
        self.resize(1160, 760)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        self._backend: Optional[SWE2DBackend] = None
        self._cancel_requested = False
        self._mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._result_data: Optional[Dict[str, np.ndarray]] = None
        self._snapshot_timesteps: List[Tuple] = []  # list of (time_s, h, hu, hv)
        self._mesh_nodes_layer_id: Optional[str] = None
        self._mesh_cells_layer_id: Optional[str] = None
        self._unit_system = "SI"
        self._length_unit_name = "m"
        self._gravity = 9.81
        self._topology_mesh_future: Optional[concurrent.futures.Future] = None
        self._topology_mesh_backend: Optional[str] = None
        self._topology_mesh_default_cell_type: Optional[str] = None
        self._topology_mesh_run_mode = "full"
        self._topology_mesh_auto_fallback_used = False
        self._topology_mesh_conceptual = None
        self._topology_mesh_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._topology_mesh_process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._topology_mesh_timer = QtCore.QTimer(self)
        self._topology_mesh_timer.setInterval(120)
        self._topology_mesh_timer.timeout.connect(self._poll_topology_mesh_future)
        self._topology_mesh_started_at: Optional[float] = None
        self._topology_mesh_poll_count = 0
        try:
            timeout_sec = float(os.environ.get("BACKWATER_TOPOLOGY_MESH_TIMEOUT_SEC", "300"))
        except Exception:
            timeout_sec = 300.0
        self._topology_mesh_timeout_sec = max(30.0, timeout_sec)

        FigureCanvas, Figure, mtri = _try_import_matplotlib_qt()
        self._FigureCanvas = FigureCanvas
        self._Figure = Figure
        self._mtri = mtri
        self._have_mpl = FigureCanvas is not None and Figure is not None and mtri is not None

        self._build_ui()
        self._update_unit_system_from_crs()
        self._log(
            f"2D bridge: {'available' if swe2d_available() else 'missing'} | "
            f"GPU: {'available' if swe2d_gpu_available() else 'cpu-only'}"
        )
        self._log(
            f"Meshing: Gmsh {'available' if _gmsh_available() else 'NOT INSTALLED — use Structured backend or: pip install gmsh'}"
        )

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            "Interactive 2D SWE workflow: generate mesh, assign side BCs, set model parameters, "
            "run, and visualize results."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(split, stretch=1)

        # Left pane: setup + run controls
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        mesh_group = QtWidgets.QGroupBox("Mesh Generation")
        mesh_form = QtWidgets.QFormLayout(mesh_group)
        self.nx_spin = QtWidgets.QSpinBox()
        self.nx_spin.setRange(2, 400)
        self.nx_spin.setValue(24)
        self.ny_spin = QtWidgets.QSpinBox()
        self.ny_spin.setRange(2, 400)
        self.ny_spin.setValue(14)
        self.lx_spin = QtWidgets.QDoubleSpinBox()
        self.lx_spin.setRange(1.0, 1.0e6)
        self.lx_spin.setDecimals(2)
        self.lx_spin.setValue(240.0)
        self.ly_spin = QtWidgets.QDoubleSpinBox()
        self.ly_spin.setRange(1.0, 1.0e6)
        self.ly_spin.setDecimals(2)
        self.ly_spin.setValue(120.0)
        self.bed_amp_spin = QtWidgets.QDoubleSpinBox()
        self.bed_amp_spin.setRange(0.0, 1.0e6)
        self.bed_amp_spin.setDecimals(3)
        self.bed_amp_spin.setValue(0.0)
        self.mesh_layout_combo = QtWidgets.QComboBox()
        self.mesh_layout_combo.addItem("Split triangles (2 cells / block)", "tri")
        self.mesh_layout_combo.addItem("Structured block quads (1 cell / block, faster)", "quad")
        self.mesh_layout_combo.setCurrentIndex(1)
        self.mesh_layout_combo.setToolTip(
            "Select generated cell layout.\n"
            "Structured block quads usually run faster for rectilinear domains."
        )
        self.generate_mesh_btn = QtWidgets.QPushButton("Generate Mesh")
        self.generate_mesh_btn.clicked.connect(self._on_generate_mesh)
        self.mesh_info_lbl = QtWidgets.QLabel("Mesh not generated")
        self.mesh_info_lbl.setWordWrap(True)
        mesh_form.addRow("Cells in X:", self.nx_spin)
        mesh_form.addRow("Cells in Y:", self.ny_spin)
        mesh_form.addRow("Length X:", self.lx_spin)
        mesh_form.addRow("Length Y:", self.ly_spin)
        mesh_form.addRow("Bed perturbation amplitude:", self.bed_amp_spin)
        mesh_form.addRow("Structured layout:", self.mesh_layout_combo)
        mesh_form.addRow(self.generate_mesh_btn)
        mesh_form.addRow(self.mesh_info_lbl)
        left_layout.addWidget(mesh_group)

        map_group = QtWidgets.QGroupBox("Map Layer Mesh + Terrain")
        map_layout = QtWidgets.QGridLayout(map_group)
        self.nodes_layer_combo = QtWidgets.QComboBox()
        self.cells_layer_combo = QtWidgets.QComboBox()
        self.terrain_layer_combo = QtWidgets.QComboBox()
        self.manning_layer_combo = QtWidgets.QComboBox()
        self.refresh_layers_btn = QtWidgets.QPushButton("Refresh Layers")
        self.refresh_layers_btn.clicked.connect(self._refresh_layer_combos)
        self.create_model_gpkg_btn = QtWidgets.QPushButton("Create 2D Model GeoPackage")
        self.create_model_gpkg_btn.clicked.connect(self._create_2d_model_geopackage)
        self.load_model_gpkg_btn = QtWidgets.QPushButton("Load 2D Model GeoPackage")
        self.load_model_gpkg_btn.clicked.connect(self._load_2d_model_geopackage)
        self.export_mesh_layers_btn = QtWidgets.QPushButton("Export Mesh To Map Layers")
        self.export_mesh_layers_btn.clicked.connect(self._export_mesh_to_layers)
        self.save_hdf5_btn = QtWidgets.QPushButton("Save Mesh To HEC-RAS HDF5")
        self.save_hdf5_btn.clicked.connect(self._export_mesh_to_hdf5)
        self.save_results_hdf5_btn = QtWidgets.QPushButton("Save Results To HEC-RAS HDF5")
        self.save_results_hdf5_btn.clicked.connect(self._export_results_to_hdf5)
        self.save_results_ugrid_btn = QtWidgets.QPushButton("Save Results To UGRID NetCDF")
        self.save_results_ugrid_btn.clicked.connect(self._export_results_to_ugrid)
        self.import_mesh_layers_btn = QtWidgets.QPushButton("Load Mesh From Selected Layers")
        self.import_mesh_layers_btn.clicked.connect(self._import_mesh_from_layers)
        self.terrain_to_nodes_btn = QtWidgets.QPushButton("Assign Node Z From Terrain")
        self.terrain_to_nodes_btn.clicked.connect(self._assign_node_z_from_terrain)
        self.pull_node_z_btn = QtWidgets.QPushButton("Pull Node Z From Nodes Layer")
        self.pull_node_z_btn.clicked.connect(self._pull_node_z_from_layer)
        self.layer_status_lbl = QtWidgets.QLabel("No layer-linked mesh yet")
        self.layer_status_lbl.setWordWrap(True)

        map_layout.addWidget(QtWidgets.QLabel("Nodes layer:"), 0, 0)
        map_layout.addWidget(self.nodes_layer_combo, 0, 1)
        map_layout.addWidget(QtWidgets.QLabel("Cells layer:"), 1, 0)
        map_layout.addWidget(self.cells_layer_combo, 1, 1)
        map_layout.addWidget(QtWidgets.QLabel("Terrain raster:"), 2, 0)
        map_layout.addWidget(self.terrain_layer_combo, 2, 1)
        map_layout.addWidget(QtWidgets.QLabel("Manning polygons:"), 3, 0)
        map_layout.addWidget(self.manning_layer_combo, 3, 1)
        map_layout.addWidget(self.refresh_layers_btn, 4, 0)
        map_layout.addWidget(self.create_model_gpkg_btn, 4, 1)
        map_layout.addWidget(self.load_model_gpkg_btn, 5, 0, 1, 2)
        map_layout.addWidget(self.export_mesh_layers_btn, 6, 0)
        map_layout.addWidget(self.save_hdf5_btn, 6, 1)
        map_layout.addWidget(self.save_results_hdf5_btn, 7, 0, 1, 2)
        map_layout.addWidget(self.save_results_ugrid_btn, 8, 0, 1, 2)
        map_layout.addWidget(self.import_mesh_layers_btn, 9, 0, 1, 2)
        map_layout.addWidget(self.terrain_to_nodes_btn, 10, 0, 1, 2)
        map_layout.addWidget(self.pull_node_z_btn, 11, 0, 1, 2)
        map_layout.addWidget(self.layer_status_lbl, 12, 0, 1, 2)
        left_layout.addWidget(map_group)

        topo_group = QtWidgets.QGroupBox("Topology Meshing (Face-centric)")
        topo_layout = QtWidgets.QGridLayout(topo_group)
        self.topo_nodes_combo = QtWidgets.QComboBox()
        self.topo_arcs_combo = QtWidgets.QComboBox()
        self.topo_regions_combo = QtWidgets.QComboBox()
        self.topo_constraints_combo = QtWidgets.QComboBox()
        self.topo_constraints_combo.addItem("(none)", None)
        self.topo_quad_edges_combo = QtWidgets.QComboBox()
        self.topo_quad_edges_combo.addItem("(none)", None)
        self.topo_backend_combo = QtWidgets.QComboBox()
        _gmsh_label = "Gmsh (recommended)" if _gmsh_available() else "Gmsh (install: pip install gmsh)"
        self.topo_backend_combo.addItem(_gmsh_label, "gmsh")
        self.topo_backend_combo.addItem("Structured (built-in fallback)", "structured")
        _tqmesh_label = "TQMesh (advancing-front, built-in)" if _tqmesh_available() else "TQMesh (build plugin to enable)"
        self.topo_backend_combo.addItem(_tqmesh_label, "tqmesh")
        self.topo_default_size_spin = QtWidgets.QDoubleSpinBox()
        self.topo_default_size_spin.setRange(0.01, 1.0e6)
        self.topo_default_size_spin.setDecimals(3)
        self.topo_default_size_spin.setValue(20.0)
        self.topo_default_cell_type_combo = QtWidgets.QComboBox()
        self.topo_default_cell_type_combo.addItems(["triangular", "quadrilateral", "cartesian", "empty"])
        self.topo_export_template_btn = QtWidgets.QPushButton("Create Topology Template Layers")
        self.topo_export_template_btn.clicked.connect(self._create_topology_template_layers)
        self.topo_generate_btn = QtWidgets.QPushButton("Generate Mesh From Topology Layers")
        self.topo_generate_btn.clicked.connect(self._generate_mesh_from_topology_layers)
        self.topo_status_lbl = QtWidgets.QLabel("Select regions layer and generate face-centric mesh")
        self.topo_status_lbl.setWordWrap(True)

        topo_layout.addWidget(QtWidgets.QLabel("Topology nodes layer:"), 0, 0)
        topo_layout.addWidget(self.topo_nodes_combo, 0, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Topology arcs layer:"), 1, 0)
        topo_layout.addWidget(self.topo_arcs_combo, 1, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Topology regions layer:"), 2, 0)
        topo_layout.addWidget(self.topo_regions_combo, 2, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Constraints layer:"), 3, 0)
        topo_layout.addWidget(self.topo_constraints_combo, 3, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Quad edges layer (TQMesh):"), 4, 0)
        topo_layout.addWidget(self.topo_quad_edges_combo, 4, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Meshing backend:"), 5, 0)
        topo_layout.addWidget(self.topo_backend_combo, 5, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Default target size:"), 6, 0)
        topo_layout.addWidget(self.topo_default_size_spin, 6, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Default cell type:"), 7, 0)
        topo_layout.addWidget(self.topo_default_cell_type_combo, 7, 1)
        topo_layout.addWidget(self.topo_export_template_btn, 8, 0, 1, 2)
        topo_layout.addWidget(self.topo_generate_btn, 9, 0, 1, 2)
        topo_layout.addWidget(self.topo_status_lbl, 10, 0, 1, 2)
        left_layout.addWidget(topo_group)

        bc_group = QtWidgets.QGroupBox("Boundary Conditions (side defaults + optional BC polyline overrides)")
        bc_grid = QtWidgets.QGridLayout(bc_group)
        bc_grid.addWidget(QtWidgets.QLabel("Side"), 0, 0)
        bc_grid.addWidget(QtWidgets.QLabel("Type"), 0, 1)
        bc_grid.addWidget(QtWidgets.QLabel("Value (Q_total for flow)"), 0, 2)
        bc_grid.addWidget(QtWidgets.QLabel("Hydrograph (hr,Q_total; hr,Q_total)"), 0, 3)
        bc_grid.addWidget(QtWidgets.QLabel("Editor"), 0, 4)
        bc_grid.addWidget(QtWidgets.QLabel("BC polyline layer override:"), 5, 0)
        self._bc_type_boxes: Dict[str, QtWidgets.QComboBox] = {}
        self._bc_value_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._bc_ts_edits: Dict[str, QtWidgets.QLineEdit] = {}
        for row, side in enumerate(("left", "right", "bottom", "top"), start=1):
            bc_grid.addWidget(QtWidgets.QLabel(side.capitalize()), row, 0)
            cb = QtWidgets.QComboBox()
            for label, code in _BC_OPTIONS:
                cb.addItem(label, code)
            if side == "left":
                cb.setCurrentIndex(1)  # inflow default
            elif side == "right":
                cb.setCurrentIndex(2)  # stage default
            else:
                cb.setCurrentIndex(0)  # wall default
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1.0e6, 1.0e6)
            spin.setDecimals(6)
            spin.setValue(0.0)
            if side == "left":
                spin.setValue(0.10)
            if side == "right":
                spin.setValue(1.00)
            ts_edit = QtWidgets.QLineEdit()
            ts_edit.setPlaceholderText("e.g. 0:00,10; 0:30,25; 1:00,40")
            edit_btn = QtWidgets.QPushButton("Edit...")
            edit_btn.clicked.connect(lambda _checked=False, s=side: self._open_hydrograph_editor(s))
            bc_grid.addWidget(cb, row, 1)
            bc_grid.addWidget(spin, row, 2)
            bc_grid.addWidget(ts_edit, row, 3)
            bc_grid.addWidget(edit_btn, row, 4)
            self._bc_type_boxes[side] = cb
            self._bc_value_spins[side] = spin
            self._bc_ts_edits[side] = ts_edit
        self.bc_lines_layer_combo = QtWidgets.QComboBox()
        self.bc_lines_layer_combo.addItem("(none)", None)
        bc_grid.addWidget(self.bc_lines_layer_combo, 5, 1, 1, 4)
        self.inflow_progressive_chk = QtWidgets.QCheckBox(
            "Flow BC: activate lowest-elevation boundary edges first as Q increases"
        )
        self.inflow_progressive_chk.setChecked(True)
        bc_grid.addWidget(self.inflow_progressive_chk, 6, 0, 1, 5)
        left_layout.addWidget(bc_group)

        param_group = QtWidgets.QGroupBox("Model Parameters")
        param_form = QtWidgets.QFormLayout(param_group)
        self.base_depth_spin = QtWidgets.QDoubleSpinBox()
        self.base_depth_spin.setRange(0.01, 1.0e6)
        self.base_depth_spin.setDecimals(4)
        self.base_depth_spin.setValue(1.0)
        self.bump_depth_spin = QtWidgets.QDoubleSpinBox()
        self.bump_depth_spin.setRange(0.0, 1.0e6)
        self.bump_depth_spin.setDecimals(4)
        self.bump_depth_spin.setValue(0.15)
        self.n_mann_spin = QtWidgets.QDoubleSpinBox()
        self.n_mann_spin.setRange(0.0, 1.0)
        self.n_mann_spin.setDecimals(5)
        self.n_mann_spin.setValue(0.020)
        self.cfl_spin = QtWidgets.QDoubleSpinBox()
        self.cfl_spin.setRange(0.01, 0.99)
        self.cfl_spin.setDecimals(3)
        self.cfl_spin.setValue(0.45)
        self.h_min_spin = QtWidgets.QDoubleSpinBox()
        self.h_min_spin.setRange(1.0e-9, 1.0)
        self.h_min_spin.setDecimals(8)
        self.h_min_spin.setValue(1.0e-6)
        self.dt_spin = QtWidgets.QDoubleSpinBox()
        self.dt_spin.setRange(1.0e-4, 1.0e6)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setValue(0.05)
        self.rain_rate_spin = QtWidgets.QDoubleSpinBox()
        self.rain_rate_spin.setRange(0.0, 2000.0)
        self.rain_rate_spin.setDecimals(3)
        self.rain_rate_spin.setValue(0.0)
        self.rain_rate_spin.setSuffix(" mm/hr")
        self.run_time_edit = QtWidgets.QLineEdit()
        self.run_time_edit.setPlaceholderText("decimal hours (e.g. 1.5) or HH:MM (e.g. 01:30)")
        self.run_time_edit.setText("1:00")
        self.internal_flow_layer_combo = QtWidgets.QComboBox()
        self.internal_flow_layer_combo.addItem("(none)", None)
        self.internal_flow_field_edit = QtWidgets.QLineEdit("q_cms")
        self.internal_flow_field_edit.setPlaceholderText("field name, e.g. q_cms")
        self.reconstruction_combo = QtWidgets.QComboBox()
        for label, value in _RECONSTRUCTION_OPTIONS:
            self.reconstruction_combo.addItem(label, int(value))
        self.reconstruction_combo.setCurrentIndex(1)
        self.reconstruction_combo.setToolTip(
            "Select spatial reconstruction for the native solver.\n"
            "All 2nd-order schemes use Green-Gauss gradient-based TVD reconstruction:\n"
            "  Superbee (MUSCL Fast)  — most aggressive TVD, sharpest fronts\n"
            "  MinMod                 — most conservative, most stable near dry fronts\n"
            "  MC                     — balanced monotonized-central (good default)\n"
            "  Van Leer               — smooth limiter, good for continuous waves\n"
            "Recommend: start with MUSCL MinMod; switch to MC or Van Leer once stable."
        )
        self.gpu_default_lbl = QtWidgets.QLabel("GPU is attempted by default when supported by the native backend.")
        self.gpu_default_lbl.setWordWrap(True)
        self.unit_system_lbl = QtWidgets.QLabel("Unit system: auto")
        self.unit_system_lbl.setWordWrap(True)
        param_form.addRow("Base depth:", self.base_depth_spin)
        param_form.addRow("Initial bump depth:", self.bump_depth_spin)
        param_form.addRow("Manning n:", self.n_mann_spin)
        param_form.addRow("CFL:", self.cfl_spin)
        param_form.addRow("h_min:", self.h_min_spin)
        param_form.addRow("Fixed dt:", self.dt_spin)
        param_form.addRow("Rain rate:", self.rain_rate_spin)
        param_form.addRow("Internal flow layer:", self.internal_flow_layer_combo)
        param_form.addRow("Internal flow field:", self.internal_flow_field_edit)
        param_form.addRow("Run duration (hr or HH:MM):", self.run_time_edit)
        param_form.addRow("Reconstruction:", self.reconstruction_combo)
        param_form.addRow(self.unit_system_lbl)
        param_form.addRow(self.gpu_default_lbl)
        left_layout.addWidget(param_group)

        run_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run 2D Model")
        self.run_btn.clicked.connect(self._on_run)
        self.preview_overrides_btn = QtWidgets.QPushButton("Preview Overrides")
        self.preview_overrides_btn.clicked.connect(self._on_preview_overrides)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.preview_overrides_btn)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        left_layout.addLayout(run_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        snap_row = QtWidgets.QHBoxLayout()
        snap_row.addWidget(QtWidgets.QLabel("Output interval (hr or HH:MM):"))
        self.output_interval_edit = QtWidgets.QLineEdit("00:30")
        self.output_interval_edit.setMaximumWidth(90)
        self.output_interval_edit.setToolTip(
            "Interval between captured result snapshots during a run.\n"
            "E.g. 00:30 captures every 30 minutes of simulation time."
        )
        snap_row.addWidget(self.output_interval_edit)
        self.snapshot_btn = QtWidgets.QPushButton("Take Snapshot")
        self.snapshot_btn.setToolTip(
            "Write all captured timesteps up to now to a temporary HEC-RAS HDF5 file.\n"
            "The file path is logged in the message panel."
        )
        self.snapshot_btn.clicked.connect(self._on_snapshot)
        snap_row.addWidget(self.snapshot_btn)
        left_layout.addLayout(snap_row)
        left_layout.addStretch(1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_scroll.setWidget(left)

        # Allow the left panel to shrink as narrow as the splitter permits.
        # Widgets keep their natural *preferred* size but no longer enforce a
        # large minimum width, so the user can drag the splitter very small.
        left.setMinimumWidth(0)
        left_scroll.setMinimumWidth(0)
        _sp_min = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        left.setSizePolicy(_sp_min)
        # Combo boxes size to their longest item by default; override so they
        # shrink with the panel instead of imposing a minimum panel width.
        for _cb in left.findChildren(QtWidgets.QComboBox):
            _cb.setMinimumContentsLength(0)
            _cb.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        # Buttons and spin boxes also contribute to minimum width; clear it.
        for _btn in left.findChildren(QtWidgets.QPushButton):
            _btn.setMinimumWidth(0)
        for _sp in left.findChildren(
            (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)  # type: ignore[arg-type]
        ):
            _sp.setMinimumWidth(0)

        split.addWidget(left_scroll)

        # Right pane: visualization + log
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        view_row = QtWidgets.QHBoxLayout()
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItems(["Mesh", "Depth", "Velocity magnitude"])
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)
        view_row.addWidget(QtWidgets.QLabel("View:"))
        view_row.addWidget(self.view_mode_combo)
        view_row.addStretch(1)
        right_layout.addLayout(view_row)

        if self._have_mpl:
            self._fig = self._Figure(figsize=(6.4, 4.2), tight_layout=True)
            self._canvas = self._FigureCanvas(self._fig)
            right_layout.addWidget(self._canvas, stretch=2)
        else:
            self._fig = None
            self._canvas = None
            no_plot = QtWidgets.QLabel("Matplotlib Qt backend not available; results shown in text log only.")
            no_plot.setWordWrap(True)
            right_layout.addWidget(no_plot)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        right_layout.addWidget(self.log_view, stretch=1)

        split.addWidget(right)
        split.setSizes([420, 740])

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._refresh_layer_combos()

    def _log(self, msg: str):
        self.log_view.appendPlainText(str(msg))
        QtWidgets.QApplication.processEvents()

    def closeEvent(self, event):
        self._topology_mesh_timer.stop()
        try:
            self._topology_mesh_thread_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            if self._topology_mesh_process_pool is not None:
                self._topology_mesh_process_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().closeEvent(event)

    def _set_topology_mesh_busy(self, busy: bool, status_msg: Optional[str] = None):
        try:
            self.topo_generate_btn.setEnabled(not busy)
        except Exception:
            pass
        if status_msg is not None:
            self.topo_status_lbl.setText(status_msg)
        if busy:
            self.progress_bar.setRange(0, 0)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def _format_elapsed(self, started_at: Optional[float]) -> str:
        if started_at is None:
            return "0.00s"
        return f"{max(0.0, time.perf_counter() - started_at):.2f}s"

    def _start_topology_mesh_async(
        self,
        conceptual,
        backend_name: str,
        default_cell_type: str,
        run_mode: str = "full",
    ):
        if self._topology_mesh_future is not None and not self._topology_mesh_future.done():
            self._log("Topology mesh is already running. Please wait for completion.")
            return

        self._topology_mesh_backend = backend_name
        self._topology_mesh_default_cell_type = default_cell_type
        self._topology_mesh_run_mode = run_mode
        self._topology_mesh_conceptual = conceptual
        if run_mode == "full":
            self._topology_mesh_auto_fallback_used = False
        self._topology_mesh_started_at = time.perf_counter()
        self._topology_mesh_poll_count = 0

        if backend_name == "gmsh":
            # Keep Gmsh in a separate process to avoid UI freezes from C++
            # meshing work and signal-handler constraints.
            if self._topology_mesh_process_pool is None:
                self._topology_mesh_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            executor = self._topology_mesh_process_pool
        else:
            executor = self._topology_mesh_thread_pool

        self._topology_mesh_future = executor.submit(_run_topology_mesh_job, conceptual, backend_name)
        status_msg = f"Meshing in progress with backend '{backend_name}'..."
        if run_mode == "fallback-no-constraints":
            status_msg = (
                f"Meshing fallback in progress with backend '{backend_name}' "
                "(constraints disabled)..."
            )
        self._set_topology_mesh_busy(True, status_msg)
        self._log(
            "mesh> start "
            f"backend={backend_name} default_cell_type={default_cell_type} "
            f"mode={run_mode} "
            f"timeout={self._topology_mesh_timeout_sec:.0f}s "
            f"elapsed={self._format_elapsed(self._topology_mesh_started_at)}"
        )
        self._topology_mesh_timer.start()

    def _poll_topology_mesh_future(self):
        fut = self._topology_mesh_future
        if fut is None:
            self._topology_mesh_timer.stop()
            self._set_topology_mesh_busy(False)
            return

        elapsed = 0.0
        if self._topology_mesh_started_at is not None:
            elapsed = max(0.0, time.perf_counter() - self._topology_mesh_started_at)

        if elapsed > self._topology_mesh_timeout_sec and not fut.done():
            backend_name = self._topology_mesh_backend or "unknown"
            default_cell_type = self._topology_mesh_default_cell_type or "triangular"
            run_mode = self._topology_mesh_run_mode
            conceptual = self._topology_mesh_conceptual
            can_retry_without_constraints = (
                backend_name == "gmsh"
                and run_mode == "full"
                and not self._topology_mesh_auto_fallback_used
                and conceptual is not None
                and bool(getattr(conceptual, "constraints", []))
            )
            self._topology_mesh_timer.stop()
            self._topology_mesh_future = None
            self._topology_mesh_started_at = None
            self._topology_mesh_poll_count = 0

            # For gmsh (process executor), terminate and recreate the pool to
            # ensure stuck native meshing work is not left running.
            if backend_name == "gmsh" and self._topology_mesh_process_pool is not None:
                try:
                    self._topology_mesh_process_pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._topology_mesh_process_pool = None

            self.topo_status_lbl.setText(
                f"Topology meshing timed out after {self._topology_mesh_timeout_sec:.0f}s "
                f"(backend '{backend_name}')."
            )
            self._log(
                "mesh> timeout "
                f"backend={backend_name} mode={run_mode} elapsed={elapsed:.2f}s "
                f"limit={self._topology_mesh_timeout_sec:.0f}s"
            )

            if can_retry_without_constraints:
                try:
                    fallback_conceptual = _clone_conceptual_without_constraints(conceptual)
                    self._topology_mesh_auto_fallback_used = True
                    self._log(
                        "mesh> fallback "
                        f"backend={backend_name} action=retry_without_constraints "
                        f"reason=timeout elapsed={elapsed:.2f}s"
                    )
                    self._start_topology_mesh_async(
                        fallback_conceptual,
                        backend_name,
                        default_cell_type,
                        run_mode="fallback-no-constraints",
                    )
                    return
                except Exception as exc:
                    self._log(f"mesh> fallback-fail backend={backend_name} error={exc}")

            self._set_topology_mesh_busy(False)
            return

        if not fut.done():
            self._topology_mesh_poll_count += 1
            # Emit lightweight runtime heartbeat at ~1 second cadence.
            if self._topology_mesh_poll_count % 8 == 0:
                spinner = "|/-\\"[(self._topology_mesh_poll_count // 8) % 4]
                self._log(
                    "mesh> run "
                    f"status={spinner} backend={self._topology_mesh_backend or 'unknown'} "
                    f"elapsed={self._format_elapsed(self._topology_mesh_started_at)}"
                )
            return

        self._topology_mesh_timer.stop()
        backend_name = self._topology_mesh_backend or "unknown"
        default_cell_type = self._topology_mesh_default_cell_type or "triangular"
        run_mode = self._topology_mesh_run_mode
        elapsed_str = self._format_elapsed(self._topology_mesh_started_at)
        self._topology_mesh_future = None
        self._topology_mesh_started_at = None
        self._topology_mesh_poll_count = 0

        try:
            mesh = fut.result()
            self._mesh_data = {
                "nx": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
                "ny": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
                "lx": np.array(max(float(np.max(mesh.node_x) - np.min(mesh.node_x)), 1.0)),
                "ly": np.array(max(float(np.max(mesh.node_y) - np.min(mesh.node_y)), 1.0)),
                "node_x": mesh.node_x,
                "node_y": mesh.node_y,
                "node_z": mesh.node_z,
                "cell_nodes": mesh.cell_nodes,
                "cell_face_offsets": mesh.cell_face_offsets,
                "cell_face_nodes": mesh.cell_face_nodes,
                "cell_type": mesh.cell_type,
                "region_id": mesh.region_id,
                "target_size": mesh.target_size,
            }
            n_faces = int(mesh.cell_face_offsets.size - 1)
            n_tris = int(mesh.cell_nodes.size // 3)
            self.mesh_info_lbl.setText(f"Topology mesh: nodes={mesh.node_x.size}, faces={n_faces}, plot_triangles={n_tris}")
            if run_mode == "fallback-no-constraints":
                self.topo_status_lbl.setText(
                    f"Generated {n_faces} computational faces using backend '{backend_name}' "
                    "after timeout fallback with constraints disabled. "
                    "Review/repair constraint polygons and regenerate when ready."
                )
            else:
                self.topo_status_lbl.setText(
                    f"Generated {n_faces} computational faces using backend '{backend_name}'. "
                    "Cell metadata (type/size/region) stored in mesh state."
                )
            self._log(
                "mesh> done "
                f"backend={backend_name} default_cell_type={default_cell_type} "
                f"mode={run_mode} "
                f"nodes={mesh.node_x.size} faces={n_faces} elapsed={elapsed_str}"
            )
            self._result_data = None
            self.view_mode_combo.setCurrentText("Mesh")
            self._refresh_plot()
        except NotImplementedError as exc:
            self.topo_status_lbl.setText(str(exc))
            self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        except RuntimeError as exc:
            self.topo_status_lbl.setText(str(exc))
            self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        except Exception as exc:
            self.topo_status_lbl.setText(f"Topology meshing failed: {exc}")
            self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        finally:
            self._set_topology_mesh_busy(False)

    def _set_value_map_editor(self, layer, field_name: str, mapping: dict):
        if QgsEditorWidgetSetup is None:
            return
        idx = layer.fields().indexOf(field_name)
        if idx < 0:
            return
        try:
            layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", {"map": mapping}))
        except Exception:
            pass

    def _set_expression_constraint(self, layer, field_name: str, expression: str):
        idx = layer.fields().indexOf(field_name)
        if idx < 0:
            return
        try:
            layer.setConstraintExpression(idx, expression, "")
        except Exception:
            pass
        if QgsFieldConstraints is None:
            return
        try:
            layer.setFieldConstraint(
                idx,
                QgsFieldConstraints.ConstraintExpression,
                QgsFieldConstraints.ConstraintStrengthHard,
            )
        except Exception:
            pass

    def _configure_swe2d_layer_editors(self, layer):
        if layer is None or not isinstance(layer, QgsVectorLayer):
            return
        lname = str(layer.name()).lower()

        try:
            from qgis.core import QgsEditFormConfig
            cfg = layer.editFormConfig()
            if hasattr(QgsEditFormConfig, "DragAndDrop") and hasattr(cfg, "setLayout"):
                cfg.setLayout(QgsEditFormConfig.DragAndDrop)
                layer.setEditFormConfig(cfg)
        except Exception:
            pass

        is_region = "topo_regions" in lname or lname.endswith("swe2d_topo_regions")
        is_constraint = "topo_constraints" in lname or lname.endswith("swe2d_topo_constraints")
        is_quad_edges = "topo_quad_edges" in lname or lname.endswith("swe2d_topo_quad_edges")
        is_bc_lines = "bc_lines" in lname
        is_manning = "manning" in lname

        if is_region or is_constraint:
            self._set_value_map_editor(
                layer,
                "cell_type",
                {s.capitalize(): s for s in _CELL_TYPE_OPTIONS},
            )
            allowed = ", ".join(f"'{s}'" for s in _CELL_TYPE_OPTIONS)
            self._set_expression_constraint(layer, "cell_type", f"\"cell_type\" IN ({allowed})")
            self._set_expression_constraint(layer, "target_size", '"target_size" > 0')
            for nm in ("edge_len_1", "edge_len_2", "edge_len_3", "edge_len_4"):
                self._set_expression_constraint(layer, nm, f'"{nm}" IS NULL OR "{nm}" > 0')

        if is_bc_lines:
            self._set_value_map_editor(layer, "bc_type", _BC_VALUE_MAP)
            self._set_expression_constraint(layer, "bc_type", '"bc_type" IN (1,2,3,4,5,6,102,103)')
            self._set_expression_constraint(layer, "priority", '"priority" >= 0')

        if is_quad_edges:
            self._set_expression_constraint(layer, "region_id", '"region_id" >= 0')
            self._set_expression_constraint(layer, "edge_id", '"edge_id" IN (1,2,3,4)')
            self._set_expression_constraint(layer, "target_size", '"target_size" IS NULL OR "target_size" > 0')
            self._set_expression_constraint(layer, "n_layers", '"n_layers" >= 0')
            self._set_expression_constraint(layer, "first_height", '"first_height" IS NULL OR "first_height" > 0')
            self._set_expression_constraint(layer, "growth_rate", '"growth_rate" IS NULL OR "growth_rate" > 0')

        if is_manning:
            self._set_expression_constraint(layer, "n_mann", '"n_mann" >= 0')
            self._set_expression_constraint(layer, "priority", '"priority" >= 0')

    def _detect_map_unit(self):
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return None
        try:
            crs = QgsProject.instance().crs()
            if crs is None or not crs.isValid() or QgsUnitTypes is None:
                return None
            return crs.mapUnits()
        except Exception:
            return None

    def _update_unit_system_from_crs(self):
        unit = self._detect_map_unit()
        unit_name = "m"
        sys_name = "SI"
        g = 9.81

        if QgsUnitTypes is not None and unit is not None:
            try:
                feet_candidates = {
                    getattr(QgsUnitTypes, "DistanceFeet", None),
                    getattr(QgsUnitTypes, "DistanceUSSurveyFeet", None),
                }
                unit_text = ""
                if hasattr(QgsUnitTypes, "toString"):
                    unit_text = str(QgsUnitTypes.toString(unit) or "").strip().lower()
                is_feet_like_text = (
                    "feet" in unit_text
                    or "foot" in unit_text
                    or "ft" in unit_text
                )

                if unit in feet_candidates or is_feet_like_text:
                    unit_name = "ft"
                    sys_name = "US Customary"
                    g = 32.174
                elif unit == getattr(QgsUnitTypes, "DistanceMeters", None):
                    unit_name = "m"
                    sys_name = "SI"
                    g = 9.81
                else:
                    # Fallback to SI for unknown map units.
                    unit_name = str(QgsUnitTypes.toString(unit)) if hasattr(QgsUnitTypes, "toString") else "m"
                    sys_name = "SI (fallback)"
                    g = 9.81
            except Exception:
                pass

        self._unit_system = sys_name
        self._length_unit_name = unit_name
        self._gravity = g
        if hasattr(self, "unit_system_lbl"):
            self.unit_system_lbl.setText(
                f"Unit system: {sys_name} (CRS length unit: {unit_name}, gravity={g:.3f})"
            )

    def _iter_project_layers(self):
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return []
        try:
            return list(QgsProject.instance().mapLayers().values())
        except Exception:
            return []

    def _combo_layer(self, combo: QtWidgets.QComboBox, expected_kind: str):
        idx = combo.currentIndex()
        if idx < 0:
            return None
        lid = combo.itemData(idx)
        if not lid:
            return None
        for lyr in self._iter_project_layers():
            try:
                if lyr.id() != lid:
                    continue
                if expected_kind == "vector" and isinstance(lyr, QgsVectorLayer):
                    return lyr
                if expected_kind == "raster" and isinstance(lyr, QgsRasterLayer):
                    return lyr
            except Exception:
                continue
        return None

    def _refresh_layer_combos(self):
        if not _HAVE_QGIS_CORE:
            self.layer_status_lbl.setText("QGIS layer API unavailable in this runtime")
            return

        keep_nodes = self.nodes_layer_combo.currentData()
        keep_cells = self.cells_layer_combo.currentData()
        keep_terrain = self.terrain_layer_combo.currentData()
        keep_manning = self.manning_layer_combo.currentData() if hasattr(self, "manning_layer_combo") else None
        keep_topo_nodes = self.topo_nodes_combo.currentData() if hasattr(self, "topo_nodes_combo") else None
        keep_topo_arcs = self.topo_arcs_combo.currentData() if hasattr(self, "topo_arcs_combo") else None
        keep_topo_regions = self.topo_regions_combo.currentData() if hasattr(self, "topo_regions_combo") else None
        keep_topo_constraints = self.topo_constraints_combo.currentData() if hasattr(self, "topo_constraints_combo") else None
        keep_topo_quad_edges = self.topo_quad_edges_combo.currentData() if hasattr(self, "topo_quad_edges_combo") else None
        keep_bc_lines = self.bc_lines_layer_combo.currentData() if hasattr(self, "bc_lines_layer_combo") else None
        keep_internal_flow = self.internal_flow_layer_combo.currentData() if hasattr(self, "internal_flow_layer_combo") else None

        self.nodes_layer_combo.clear()
        self.cells_layer_combo.clear()
        self.terrain_layer_combo.clear()
        if hasattr(self, "manning_layer_combo"):
            self.manning_layer_combo.clear()
            self.manning_layer_combo.addItem("(none)", None)
        if hasattr(self, "topo_nodes_combo"):
            self.topo_nodes_combo.clear()
        if hasattr(self, "topo_arcs_combo"):
            self.topo_arcs_combo.clear()
        if hasattr(self, "topo_regions_combo"):
            self.topo_regions_combo.clear()
        if hasattr(self, "topo_constraints_combo"):
            self.topo_constraints_combo.clear()
            self.topo_constraints_combo.addItem("(none)", None)
        if hasattr(self, "topo_quad_edges_combo"):
            self.topo_quad_edges_combo.clear()
            self.topo_quad_edges_combo.addItem("(none)", None)
        if hasattr(self, "bc_lines_layer_combo"):
            self.bc_lines_layer_combo.clear()
            self.bc_lines_layer_combo.addItem("(none)", None)
        if hasattr(self, "internal_flow_layer_combo"):
            self.internal_flow_layer_combo.clear()
            self.internal_flow_layer_combo.addItem("(none)", None)

        for lyr in self._iter_project_layers():
            try:
                if isinstance(lyr, QgsVectorLayer):
                    self._configure_swe2d_layer_editors(lyr)
                    if hasattr(self, "internal_flow_layer_combo"):
                        self.internal_flow_layer_combo.addItem(lyr.name(), lyr.id())
                    geom_type = lyr.geometryType()
                    if geom_type == QgsWkbTypes.GeometryType.PointGeometry:
                        self.nodes_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_nodes_combo"):
                            self.topo_nodes_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
                        self.cells_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "manning_layer_combo"):
                            self.manning_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_regions_combo"):
                            self.topo_regions_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_constraints_combo"):
                            self.topo_constraints_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.LineGeometry:
                        if hasattr(self, "topo_arcs_combo"):
                            self.topo_arcs_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_quad_edges_combo"):
                            self.topo_quad_edges_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "bc_lines_layer_combo"):
                            self.bc_lines_layer_combo.addItem(lyr.name(), lyr.id())
                elif isinstance(lyr, QgsRasterLayer):
                    self.terrain_layer_combo.addItem(lyr.name(), lyr.id())
            except Exception:
                continue

        # Keep hydrograph_layer dropdown current for BC line layers.
        hydro_layer_map = {}
        for lyr in self._iter_project_layers():
            if isinstance(lyr, QgsVectorLayer):
                hydro_layer_map[str(lyr.name())] = str(lyr.name())
        for lyr in self._iter_project_layers():
            if isinstance(lyr, QgsVectorLayer) and "bc_lines" in str(lyr.name()).lower():
                self._set_value_map_editor(lyr, "hydrograph_layer", hydro_layer_map)

        def _restore(combo, keep_id):
            if not keep_id:
                return
            idx = combo.findData(keep_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _restore(self.nodes_layer_combo, keep_nodes)
        _restore(self.cells_layer_combo, keep_cells)
        _restore(self.terrain_layer_combo, keep_terrain)
        if hasattr(self, "manning_layer_combo"):
            _restore(self.manning_layer_combo, keep_manning)
        if hasattr(self, "topo_nodes_combo"):
            _restore(self.topo_nodes_combo, keep_topo_nodes)
        if hasattr(self, "topo_arcs_combo"):
            _restore(self.topo_arcs_combo, keep_topo_arcs)
        if hasattr(self, "topo_regions_combo"):
            _restore(self.topo_regions_combo, keep_topo_regions)
        if hasattr(self, "topo_constraints_combo") and keep_topo_constraints is not None:
            _restore(self.topo_constraints_combo, keep_topo_constraints)
        if hasattr(self, "topo_quad_edges_combo") and keep_topo_quad_edges is not None:
            _restore(self.topo_quad_edges_combo, keep_topo_quad_edges)
        if hasattr(self, "bc_lines_layer_combo") and keep_bc_lines is not None:
            _restore(self.bc_lines_layer_combo, keep_bc_lines)
        if hasattr(self, "internal_flow_layer_combo") and keep_internal_flow is not None:
            _restore(self.internal_flow_layer_combo, keep_internal_flow)

        self._update_unit_system_from_crs()

    def _create_topology_template_layers(self):
        if not _HAVE_QGIS_CORE:
            self._log("QGIS layer API unavailable; cannot create topology layers.")
            return

        crs_auth = "EPSG:4326"
        try:
            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception:
            pass

        nodes = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=node_id:integer",
            "SWE2D_Topo_Nodes",
            "memory",
        )
        arcs = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer",
            "SWE2D_Topo_Arcs",
            "memory",
        )
        regions = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
            "SWE2D_Topo_Regions",
            "memory",
        )
        constraints = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=constraint_id:integer&field=target_size:double&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
            "SWE2D_Topo_Constraints",
            "memory",
        )
        quad_edges = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=region_id:integer&field=edge_id:integer&field=target_size:double&field=n_layers:integer&field=first_height:double&field=growth_rate:double",
            "SWE2D_Topo_Quad_Edges",
            "memory",
        )
        manning = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=zone_id:integer&field=n_mann:double&field=priority:integer",
            "SWE2D_Manning_Zones",
            "memory",
        )
        bc_lines = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=bc_type:integer&field=bc_value:double&field=priority:integer&field=hydrograph:string(1024)&field=hydrograph_id:string(64)&field=hydrograph_layer:string(128)",
            "SWE2D_BC_Lines",
            "memory",
        )
        hydro_tbl = QgsVectorLayer(
            "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)&field=Value:double&field=description:string(256)",
            "SWE2D_Hydrographs",
            "memory",
        )

        for lyr in (nodes, arcs, regions, constraints, quad_edges, manning, bc_lines, hydro_tbl):
            if lyr is not None and lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)
                if isinstance(lyr, QgsVectorLayer):
                    self._configure_swe2d_layer_editors(lyr)

        self._refresh_layer_combos()
        self.topo_status_lbl.setText(
            "Topology template layers created. Define regions (required), optional arcs/constraints, and optional TQMesh quad-edge lines, then generate mesh."
        )
        self._log("Created topology template layers: SWE2D_Topo_Nodes/Arcs/Regions/Constraints/Quad_Edges + SWE2D_Manning_Zones + SWE2D_BC_Lines + SWE2D_Hydrographs")

    def _write_memory_layer_to_gpkg(self, layer, path: str, layer_name: str, create_file: bool):
        if QgsVectorFileWriter is None:
            raise RuntimeError("QGIS vector writer is unavailable in this runtime")
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.layerName = layer_name
        opts.fileEncoding = "UTF-8"
        if hasattr(QgsVectorFileWriter, "CreateOrOverwriteFile"):
            opts.actionOnExistingFile = (
                QgsVectorFileWriter.CreateOrOverwriteFile if create_file else QgsVectorFileWriter.CreateOrOverwriteLayer
            )
        res = QgsVectorFileWriter.writeAsVectorFormatV2(
            layer,
            path,
            QgsProject.instance().transformContext(),
            opts,
        )
        if isinstance(res, tuple):
            err = res[0]
            msg = res[1] if len(res) > 1 else ""
        else:
            err = res
            msg = ""
        if err != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Failed writing layer '{layer_name}' to {path}: {msg}")

    def _create_2d_model_geopackage(self):
        if not _HAVE_QGIS_CORE:
            self._log("QGIS layer API unavailable; cannot create model GeoPackage.")
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Create 2D Model GeoPackage",
            "swe2d_model.gpkg",
            "GeoPackage (*.gpkg)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".gpkg"):
            out_path += ".gpkg"

        crs_auth = "EPSG:4326"
        try:
            crs = QgsProject.instance().crs()
            if crs is not None and crs.isValid():
                crs_auth = crs.authid() or crs_auth
        except Exception:
            pass

        nodes = QgsVectorLayer(f"Point?crs={crs_auth}&field=node_id:integer", "swe2d_topo_nodes", "memory")
        arcs = QgsVectorLayer(f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer", "swe2d_topo_arcs", "memory")
        regions = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
            "swe2d_topo_regions",
            "memory",
        )
        constraints = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=constraint_id:integer&field=target_size:double&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
            "swe2d_topo_constraints",
            "memory",
        )
        quad_edges = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=region_id:integer&field=edge_id:integer&field=target_size:double&field=n_layers:integer&field=first_height:double&field=growth_rate:double",
            "swe2d_topo_quad_edges",
            "memory",
        )
        manning = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=zone_id:integer&field=n_mann:double&field=priority:integer",
            "swe2d_manning_zones",
            "memory",
        )
        bc_lines = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=bc_type:integer&field=bc_value:double&field=priority:integer&field=hydrograph:string(1024)&field=hydrograph_id:string(64)&field=hydrograph_layer:string(128)",
            "swe2d_bc_lines",
            "memory",
        )
        hydro = QgsVectorLayer(
            "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)&field=Value:double&field=description:string(256)",
            "swe2d_hydrographs",
            "memory",
        )

        model_layers = [nodes, arcs, regions, constraints, quad_edges, manning, bc_lines, hydro]
        for lyr in model_layers:
            self._configure_swe2d_layer_editors(lyr)

        # Persist as a single GeoPackage file.
        for i, lyr in enumerate(model_layers):
            self._write_memory_layer_to_gpkg(lyr, out_path, lyr.name(), create_file=(i == 0))

        self._log(f"Created 2D model GeoPackage: {out_path}")
        self.layer_status_lbl.setText("2D model GeoPackage created.")
        self._load_2d_model_geopackage(path_override=out_path)

    def _load_2d_model_geopackage(self, path_override: Optional[str] = None):
        if not _HAVE_QGIS_CORE:
            return
        gpkg_path = path_override
        if not gpkg_path:
            gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load 2D Model GeoPackage",
                "",
                "GeoPackage (*.gpkg)",
            )
        if not gpkg_path:
            return

        layer_names = [
            "swe2d_topo_nodes",
            "swe2d_topo_arcs",
            "swe2d_topo_regions",
            "swe2d_topo_constraints",
            "swe2d_topo_quad_edges",
            "swe2d_manning_zones",
            "swe2d_bc_lines",
            "swe2d_hydrographs",
        ]
        loaded = 0
        for lname in layer_names:
            lyr = QgsVectorLayer(f"{gpkg_path}|layername={lname}", lname, "ogr")
            if lyr is not None and lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)
                self._configure_swe2d_layer_editors(lyr)
                loaded += 1

        self._refresh_layer_combos()
        self._log(f"Loaded 2D model GeoPackage: {gpkg_path} (layers loaded={loaded})")
        self.layer_status_lbl.setText(f"Loaded 2D model GeoPackage ({loaded} layers).")

    def _generate_mesh_from_topology_layers(self):
        if conceptual_from_qgis_layers is None or generate_face_centric_mesh is None:
            self._log("Meshing module unavailable. Could not import swe2d_meshing.")
            return
        if not _HAVE_QGIS_CORE:
            self._log("QGIS layer API unavailable; cannot read topology layers.")
            return

        nodes_layer = self._combo_layer(self.topo_nodes_combo, "vector")
        arcs_layer = self._combo_layer(self.topo_arcs_combo, "vector")
        regions_layer = self._combo_layer(self.topo_regions_combo, "vector")
        constraints_layer = self._combo_layer(self.topo_constraints_combo, "vector")
        quad_edges_layer = self._combo_layer(self.topo_quad_edges_combo, "vector")
        if self.topo_constraints_combo.currentData() is None:
            constraints_layer = None
        if self.topo_quad_edges_combo.currentData() is None:
            quad_edges_layer = None

        if regions_layer is None:
            self._log("Select a topology regions polygon layer first.")
            self.topo_status_lbl.setText("Missing required topology regions layer.")
            return

        default_size = float(self.topo_default_size_spin.value())
        default_cell_type = str(self.topo_default_cell_type_combo.currentText())
        backend_name = str(self.topo_backend_combo.currentData() or "structured")

        try:
            conceptual = conceptual_from_qgis_layers(
                nodes_layer=nodes_layer,
                arcs_layer=arcs_layer,
                regions_layer=regions_layer,
                constraints_layer=constraints_layer,
                quad_edges_layer=quad_edges_layer,
                default_size=default_size,
                default_cell_type=default_cell_type,
            )
            self._start_topology_mesh_async(conceptual, backend_name, default_cell_type)
        except NotImplementedError as exc:
            self.topo_status_lbl.setText(str(exc))
            self._log(f"Topology meshing backend not implemented: {exc}")
        except RuntimeError as exc:
            self.topo_status_lbl.setText(str(exc))
            self._log(f"Topology meshing runtime error: {exc}")
        except Exception as exc:
            self.topo_status_lbl.setText(f"Topology meshing failed: {exc}")
            self._log(f"Topology meshing error: {exc}")

    def _ensure_mesh_data(self):
        if self._mesh_data is None:
            self._on_generate_mesh()
        return self._mesh_data is not None

    def _export_mesh_to_layers(self):
        if not _HAVE_QGIS_CORE:
            self._log("QGIS layer API unavailable; cannot export mesh layers.")
            return
        if not self._ensure_mesh_data():
            return

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        node_z = self._mesh_data["node_z"]
        triangles = self._mesh_data["cell_nodes"].reshape((-1, 3))

        crs_auth = "EPSG:4326"
        try:
            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception:
            pass

        nodes_layer = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=node_id:integer&field=bed_z:double",
            "SWE2D_Mesh_Nodes",
            "memory",
        )
        cells_layer = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=cell_id:integer&field=n0:integer&field=n1:integer&field=n2:integer&field=cell_type:string(32)&field=region_id:integer&field=target_size:double",
            "SWE2D_Mesh_Cells",
            "memory",
        )
        if not nodes_layer.isValid() or not cells_layer.isValid():
            self._log("Failed to create memory layers for mesh export.")
            return

        node_feats = []
        for i in range(node_x.shape[0]):
            f = QgsFeature(nodes_layer.fields())
            f.setAttribute("node_id", int(i))
            f.setAttribute("bed_z", float(node_z[i]))
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(node_x[i]), float(node_y[i]))))
            node_feats.append(f)
        nodes_layer.dataProvider().addFeatures(node_feats)
        nodes_layer.updateExtents()

        cell_feats = []
        cell_type_meta = self._mesh_data.get("cell_type")
        region_meta = self._mesh_data.get("region_id")
        size_meta = self._mesh_data.get("target_size")
        for cid, tri in enumerate(triangles):
            n0, n1, n2 = [int(v) for v in tri]
            poly = [
                QgsPointXY(float(node_x[n0]), float(node_y[n0])),
                QgsPointXY(float(node_x[n1]), float(node_y[n1])),
                QgsPointXY(float(node_x[n2]), float(node_y[n2])),
                QgsPointXY(float(node_x[n0]), float(node_y[n0])),
            ]
            f = QgsFeature(cells_layer.fields())
            f.setAttribute("cell_id", int(cid))
            f.setAttribute("n0", n0)
            f.setAttribute("n1", n1)
            f.setAttribute("n2", n2)
            if cell_type_meta is not None and cid < len(cell_type_meta):
                f.setAttribute("cell_type", str(cell_type_meta[cid]))
            else:
                f.setAttribute("cell_type", "triangular")
            if region_meta is not None and cid < len(region_meta):
                f.setAttribute("region_id", int(region_meta[cid]))
            else:
                f.setAttribute("region_id", -1)
            if size_meta is not None and cid < len(size_meta):
                f.setAttribute("target_size", float(size_meta[cid]))
            else:
                f.setAttribute("target_size", 0.0)
            f.setGeometry(QgsGeometry.fromPolygonXY([poly]))
            cell_feats.append(f)
        cells_layer.dataProvider().addFeatures(cell_feats)
        cells_layer.updateExtents()

        QgsProject.instance().addMapLayer(nodes_layer)
        QgsProject.instance().addMapLayer(cells_layer)

        self._mesh_nodes_layer_id = nodes_layer.id()
        self._mesh_cells_layer_id = cells_layer.id()
        self._refresh_layer_combos()
        nodes_idx = self.nodes_layer_combo.findData(nodes_layer.id())
        cells_idx = self.cells_layer_combo.findData(cells_layer.id())
        if nodes_idx >= 0:
            self.nodes_layer_combo.setCurrentIndex(nodes_idx)
        if cells_idx >= 0:
            self.cells_layer_combo.setCurrentIndex(cells_idx)
        self.layer_status_lbl.setText("Mesh exported to map layers. Edit features, then click 'Load Mesh From Selected Layers'.")
        self._log("Mesh exported to SWE2D_Mesh_Nodes / SWE2D_Mesh_Cells layers.")

    def _import_mesh_from_layers(self):
        if not _HAVE_QGIS_CORE:
            return
        nodes_layer = self._combo_layer(self.nodes_layer_combo, "vector")
        cells_layer = self._combo_layer(self.cells_layer_combo, "vector")
        if nodes_layer is None or cells_layer is None:
            self._log("Select both nodes and cells vector layers.")
            return

        nodes_by_id: Dict[int, Tuple[float, float, float]] = {}
        auto_id = 0
        for ft in nodes_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.asPoint()
            nid = ft["node_id"] if "node_id" in nodes_layer.fields().names() else None
            if nid is None:
                nid = auto_id
                auto_id += 1
            try:
                nid_i = int(nid)
            except Exception:
                continue
            z = 0.0
            if "bed_z" in nodes_layer.fields().names():
                try:
                    z = float(ft["bed_z"])
                except Exception:
                    z = 0.0
            nodes_by_id[nid_i] = (float(pt.x()), float(pt.y()), z)

        if not nodes_by_id:
            self._log("No valid node features found in selected nodes layer.")
            return

        node_ids = sorted(nodes_by_id.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        node_x = np.array([nodes_by_id[nid][0] for nid in node_ids], dtype=np.float64)
        node_y = np.array([nodes_by_id[nid][1] for nid in node_ids], dtype=np.float64)
        node_z = np.array([nodes_by_id[nid][2] for nid in node_ids], dtype=np.float64)

        coord_to_idx = {
            (round(node_x[i], 9), round(node_y[i], 9)): i for i in range(node_x.shape[0])
        }

        cell_list: List[int] = []
        for ft in cells_layer.getFeatures():
            n0 = ft["n0"] if "n0" in cells_layer.fields().names() else None
            n1 = ft["n1"] if "n1" in cells_layer.fields().names() else None
            n2 = ft["n2"] if "n2" in cells_layer.fields().names() else None
            if n0 is not None and n1 is not None and n2 is not None:
                try:
                    tri_ids = [int(n0), int(n1), int(n2)]
                    tri_idx = [id_to_idx[t] for t in tri_ids]
                    cell_list.extend(tri_idx)
                    continue
                except Exception:
                    pass

            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            poly = geom.asPolygon()
            if not poly or not poly[0]:
                continue
            ring = poly[0]
            verts = []
            for p in ring[:-1]:
                key = (round(float(p.x()), 9), round(float(p.y()), 9))
                if key in coord_to_idx:
                    verts.append(coord_to_idx[key])
            uniq = []
            for vid in verts:
                if vid not in uniq:
                    uniq.append(vid)
            if len(uniq) >= 3:
                cell_list.extend(uniq[:3])

        if len(cell_list) < 3:
            self._log("No valid triangle cells found in selected cells layer.")
            return

        cell_nodes = np.array(cell_list, dtype=np.int32)
        if cell_nodes.size % 3 != 0:
            cell_nodes = cell_nodes[: (cell_nodes.size // 3) * 3]

        if node_x.size >= 2:
            lx = float(np.max(node_x) - np.min(node_x))
            ly = float(np.max(node_y) - np.min(node_y))
        else:
            lx, ly = 1.0, 1.0

        self._mesh_data = {
            "nx": np.array(max(2, int(round(np.sqrt(node_x.size))))),
            "ny": np.array(max(2, int(round(np.sqrt(node_x.size))))),
            "lx": np.array(max(lx, 1.0)),
            "ly": np.array(max(ly, 1.0)),
            "node_x": node_x,
            "node_y": node_y,
            "node_z": node_z,
            "cell_nodes": cell_nodes,
        }
        n_cells = int(cell_nodes.size // 3)
        self.mesh_info_lbl.setText(f"Loaded map mesh: nodes={node_x.size}, cells={n_cells}, triangles={n_cells}")
        self.layer_status_lbl.setText("Mesh loaded from selected map layers.")
        self._log(f"Imported mesh from map layers: nodes={node_x.size}, cells={n_cells}")
        self._result_data = None
        self.view_mode_combo.setCurrentText("Mesh")
        self._refresh_plot()

    def _assign_node_z_from_terrain(self):
        if not _HAVE_QGIS_CORE:
            return
        nodes_layer = self._combo_layer(self.nodes_layer_combo, "vector")
        raster_layer = self._combo_layer(self.terrain_layer_combo, "raster")
        if nodes_layer is None:
            self._log("Select a nodes point layer first.")
            return
        if raster_layer is None:
            self._log("Select a terrain raster layer first.")
            return

        field_names = nodes_layer.fields().names()
        if "bed_z" not in field_names:
            nodes_layer.dataProvider().addAttributes([QgsField("bed_z", QVariant.Double)])
            nodes_layer.updateFields()

        provider = raster_layer.dataProvider()
        z_idx = nodes_layer.fields().indexOf("bed_z")
        changed = {}
        sampled = 0
        for ft in nodes_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.asPoint()
            val, ok = provider.sample(QgsPointXY(pt.x(), pt.y()), 1)
            if ok:
                changed[ft.id()] = {z_idx: float(val)}
                sampled += 1

        if changed:
            nodes_layer.dataProvider().changeAttributeValues(changed)
            nodes_layer.triggerRepaint()
        self._log(f"Assigned terrain bed_z for {sampled} node features.")
        self.layer_status_lbl.setText("Terrain Z assigned to nodes layer bed_z field.")

    def _pull_node_z_from_layer(self):
        if self._mesh_data is None:
            self._log("Generate or import a mesh first.")
            return
        if not _HAVE_QGIS_CORE:
            return
        nodes_layer = self._combo_layer(self.nodes_layer_combo, "vector")
        if nodes_layer is None:
            self._log("Select a nodes layer first.")
            return
        if "node_id" not in nodes_layer.fields().names() or "bed_z" not in nodes_layer.fields().names():
            self._log("Nodes layer must contain node_id and bed_z fields.")
            return

        node_z = self._mesh_data["node_z"].copy()
        updated = 0
        for ft in nodes_layer.getFeatures():
            try:
                nid = int(ft["node_id"])
                z = float(ft["bed_z"])
            except Exception:
                continue
            if 0 <= nid < node_z.shape[0]:
                node_z[nid] = z
                updated += 1

        self._mesh_data["node_z"] = node_z
        self._result_data = None
        self._log(f"Pulled bed_z from nodes layer for {updated} nodes.")
        self.layer_status_lbl.setText("Mesh node bed elevations updated from nodes layer.")

    def _mesh_cell_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self._mesh_data is not None
        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]

        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            cx = np.zeros(offs.size - 1, dtype=np.float64)
            cy = np.zeros(offs.size - 1, dtype=np.float64)
            for i in range(offs.size - 1):
                s = int(offs[i])
                e = int(offs[i + 1])
                ids = faces[s:e]
                if ids.size == 0:
                    continue
                cx[i] = float(np.mean(node_x[ids]))
                cy[i] = float(np.mean(node_y[ids]))
            return cx, cy

        tris = self._mesh_data["cell_nodes"].reshape((-1, 3))
        return node_x[tris].mean(axis=1), node_y[tris].mean(axis=1)

    def _mesh_cell_areas(self) -> np.ndarray:
        assert self._mesh_data is not None
        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]

        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            area = np.zeros(offs.size - 1, dtype=np.float64)
            for i in range(offs.size - 1):
                s = int(offs[i])
                e = int(offs[i + 1])
                ids = faces[s:e]
                if ids.size < 3:
                    continue
                x = node_x[ids]
                y = node_y[ids]
                area[i] = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            return area

        tris = self._mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
        x0 = node_x[tris[:, 0]]
        y0 = node_y[tris[:, 0]]
        x1 = node_x[tris[:, 1]]
        y1 = node_y[tris[:, 1]]
        x2 = node_x[tris[:, 2]]
        y2 = node_y[tris[:, 2]]
        return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    def _build_internal_flow_source_cms(self) -> Optional[np.ndarray]:
        if self._mesh_data is None or not _HAVE_QGIS_CORE:
            return None
        if not hasattr(self, "internal_flow_layer_combo"):
            return None

        lyr = self._combo_layer(self.internal_flow_layer_combo, "vector")
        if lyr is None:
            return None

        field_name = str(self.internal_flow_field_edit.text() or "q_cms").strip()
        if not field_name:
            field_name = "q_cms"
        fields = set(lyr.fields().names())
        if field_name not in fields:
            for cand in ("q_cms", "flow_cms", "q", "flow"):
                if cand in fields:
                    field_name = cand
                    break
        if field_name not in fields:
            self._log(f"Internal flow layer '{lyr.name()}' missing flow field '{field_name}'; skipping internal sources.")
            return None

        cx, cy = self._mesh_cell_centroids()
        cell_q = np.zeros(cx.shape[0], dtype=np.float64)
        assigned = 0

        for ft in lyr.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                q_cms = float(ft[field_name])
            except Exception:
                continue
            if not np.isfinite(q_cms) or abs(q_cms) <= 0.0:
                continue

            try:
                wkb_type = int(geom.wkbType())
            except Exception:
                wkb_type = -1

            if QgsWkbTypes.geometryType(wkb_type) == QgsWkbTypes.GeometryType.PolygonGeometry:
                hit_ids = []
                for i in range(cx.shape[0]):
                    p = QgsGeometry.fromPointXY(QgsPointXY(float(cx[i]), float(cy[i])))
                    if geom.contains(p) or geom.intersects(p):
                        hit_ids.append(i)
                if not hit_ids:
                    continue
                share = q_cms / float(len(hit_ids))
                for idx in hit_ids:
                    cell_q[idx] += share
                assigned += 1
            else:
                rp = geom.centroid().asPoint() if not geom.centroid().isEmpty() else None
                if rp is None:
                    continue
                dx = cx - float(rp.x())
                dy = cy - float(rp.y())
                idx = int(np.argmin(dx * dx + dy * dy))
                cell_q[idx] += q_cms
                assigned += 1

        self._log(
            f"Internal flow sources mapped from layer '{lyr.name()}': features={assigned}, total_Q={float(np.sum(cell_q)):.6f} cms"
        )
        return cell_q

    def _apply_external_sources(
        self,
        backend: SWE2DBackend,
        dt_step: float,
        rain_rate_mps: float,
        cell_source_cms: Optional[np.ndarray],
    ) -> None:
        if dt_step <= 0.0:
            return
        if rain_rate_mps <= 0.0 and cell_source_cms is None:
            return

        h, hu, hv = backend.get_state()
        src = np.full(h.shape, float(rain_rate_mps), dtype=np.float64)
        if cell_source_cms is not None:
            area = self._mesh_cell_areas()
            safe_area = np.maximum(area, 1.0e-8)
            src += (cell_source_cms / safe_area)

        h = h + dt_step * src
        h = np.where(np.isfinite(h), h, 0.0)
        h = np.maximum(h, 0.0)
        dry = h < float(self.h_min_spin.value())
        hu = np.where(dry, 0.0, hu)
        hv = np.where(dry, 0.0, hv)
        backend.set_state(h, hu, hv)

    def _build_spatial_manning_array(self) -> Optional[np.ndarray]:
        if self._mesh_data is None or not _HAVE_QGIS_CORE:
            return None
        if not hasattr(self, "manning_layer_combo"):
            return None

        lyr = self._combo_layer(self.manning_layer_combo, "vector")
        if lyr is None:
            return None

        fields = set(lyr.fields().names())
        n_field = None
        for cand in ("n_mann", "manning_n", "manning", "n"):
            if cand in fields:
                n_field = cand
                break
        if n_field is None:
            self._log("Manning layer selected but no n_mann/manning_n/manning/n field found; using global n.")
            return None

        prio_field = "priority" if "priority" in fields else None
        cx, cy = self._mesh_cell_centroids()
        nvals = np.full(cx.shape[0], float(self.n_mann_spin.value()), dtype=np.float64)

        features = []
        for ft in lyr.getFeatures():
            g = ft.geometry()
            if g is None or g.isEmpty():
                continue
            try:
                n = float(ft[n_field])
            except Exception:
                continue
            pr = 0
            if prio_field is not None:
                try:
                    pr = int(ft[prio_field])
                except Exception:
                    pr = 0
            features.append((pr, g, n))

        if not features:
            return None

        features.sort(key=lambda x: x[0], reverse=True)
        applied = 0
        for i in range(cx.shape[0]):
            p = QgsGeometry.fromPointXY(QgsPointXY(float(cx[i]), float(cy[i])))
            for _, g, n in features:
                if g.contains(p) or g.intersects(p):
                    nvals[i] = n
                    applied += 1
                    break

        self._log(f"Spatial Manning applied to {applied}/{cx.shape[0]} cells from '{lyr.name()}'.")
        return nvals

    def _preview_spatial_manning(self) -> Tuple[Optional[np.ndarray], int, int, str]:
        if self._mesh_data is None or not _HAVE_QGIS_CORE or not hasattr(self, "manning_layer_combo"):
            return None, 0, 0, ""

        lyr = self._combo_layer(self.manning_layer_combo, "vector")
        if lyr is None:
            return None, 0, 0, ""

        fields = set(lyr.fields().names())
        n_field = None
        for cand in ("n_mann", "manning_n", "manning", "n"):
            if cand in fields:
                n_field = cand
                break
        if n_field is None:
            return None, 0, 0, lyr.name()

        prio_field = "priority" if "priority" in fields else None
        cx, cy = self._mesh_cell_centroids()
        nvals = np.full(cx.shape[0], float(self.n_mann_spin.value()), dtype=np.float64)

        features = []
        for ft in lyr.getFeatures():
            g = ft.geometry()
            if g is None or g.isEmpty():
                continue
            try:
                n = float(ft[n_field])
            except Exception:
                continue
            pr = 0
            if prio_field is not None:
                try:
                    pr = int(ft[prio_field])
                except Exception:
                    pr = 0
            features.append((pr, g, n))

        if not features:
            return None, 0, cx.shape[0], lyr.name()

        features.sort(key=lambda x: x[0], reverse=True)
        applied = 0
        for i in range(cx.shape[0]):
            p = QgsGeometry.fromPointXY(QgsPointXY(float(cx[i]), float(cy[i])))
            for _, g, n in features:
                if g.contains(p) or g.intersects(p):
                    nvals[i] = n
                    applied += 1
                    break

        return nvals, applied, cx.shape[0], lyr.name()

    def _bc_code_label(self, code: int) -> str:
        for label, opt_code in _BC_OPTIONS:
            if int(opt_code) == int(code):
                return label
        return str(code)

    def _on_preview_overrides(self):
        if self._mesh_data is None:
            self._on_generate_mesh()
        if self._mesh_data is None:
            return

        edge_n0, edge_n1 = self._mesh_boundary_edges()
        if edge_n0.size == 0:
            self._log("No boundary edges detected in mesh.")
            QtWidgets.QMessageBox.information(self, "Preview Overrides", "No boundary edges detected in mesh.")
            return

        bc_type_default, bc_val_default = self._default_bc_for_edges(edge_n0, edge_n1)
        bc_type_preview = bc_type_default.copy()
        bc_val_preview = bc_val_default.copy()
        bc_type_preview, bc_val_preview = self._apply_bc_layer_overrides(
            edge_n0, edge_n1, bc_type_preview, bc_val_preview
        )
        edge_hydrographs = self._collect_bc_layer_hydrographs(edge_n0, edge_n1)

        static_mask = (bc_type_preview != bc_type_default) | (~np.isclose(bc_val_preview, bc_val_default))
        static_count = int(np.count_nonzero(static_mask))
        static_type_counts: Dict[str, int] = {}
        if static_count:
            for code in np.unique(bc_type_preview[static_mask]):
                label = self._bc_code_label(int(code))
                static_type_counts[label] = int(np.count_nonzero(bc_type_preview[static_mask] == code))

        mann_arr, mann_applied, mann_total, mann_name = self._preview_spatial_manning()
        if mann_arr is not None and mann_total > 0:
            mann_range = f"{float(np.min(mann_arr)):.5f} to {float(np.max(mann_arr)):.5f}"
        else:
            mann_range = f"{float(self.n_mann_spin.value()):.5f}"

        bc_layer_name = "(none)"
        bc_layer = self._combo_layer(self.bc_lines_layer_combo, "vector") if hasattr(self, "bc_lines_layer_combo") else None
        if bc_layer is not None:
            bc_layer_name = bc_layer.name()

        manning_layer_name = mann_name or "(none)"
        summary_lines = [
            f"Boundary edges detected: {edge_n0.size}",
            f"BC layer: {bc_layer_name}",
            f"Static BC overrides applied: {static_count}",
            f"Timeseries BC edges applied: {len(edge_hydrographs)}",
            f"Manning layer: {manning_layer_name}",
            f"Manning cells affected: {mann_applied}/{mann_total}",
            f"Manning n range in solver input: {mann_range}",
        ]
        if static_type_counts:
            details = ", ".join(f"{label}={count}" for label, count in sorted(static_type_counts.items()))
            summary_lines.insert(3, f"Static BC types: {details}")

        summary = "\n".join(summary_lines)
        self._log("Override preview:\n" + summary.replace("\n", " | "))
        QtWidgets.QMessageBox.information(self, "Preview Overrides", summary)

    def _build_msh_elements(self) -> List[Tuple[int, Tuple[int, int], List[int], int]]:
        if self._mesh_data is None:
            return []

        # Build element list as (etype, tags, node_ids_1based, source_cell_id)
        elems: List[Tuple[int, Tuple[int, int], List[int], int]] = []
        region_meta = self._mesh_data.get("region_id")

        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            for i in range(offs.size - 1):
                s = int(offs[i])
                e = int(offs[i + 1])
                ids0 = [int(v) + 1 for v in faces[s:e]]
                rid = int(region_meta[i]) if region_meta is not None and i < len(region_meta) else 0
                if len(ids0) == 3:
                    elems.append((2, (rid, 0), ids0, i))
                elif len(ids0) == 4:
                    elems.append((3, (rid, 0), ids0, i))
                elif len(ids0) > 4:
                    # Fan triangulation for polygons beyond quads.
                    for k in range(1, len(ids0) - 1):
                        elems.append((2, (rid, 0), [ids0[0], ids0[k], ids0[k + 1]], i))
        else:
            tris = self._mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
            for i, tri in enumerate(tris):
                rid = int(region_meta[i]) if region_meta is not None and i < len(region_meta) else 0
                elems.append((2, (rid, 0), [int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1], i))

        return elems

    # ------------------------------------------------------------------
    # HEC-RAS HDF5 export (replaces legacy Gmsh .msh export)
    # QGIS can open the output as a Mesh layer via Layer > Add Layer >
    # Add Mesh Layer, selecting format "HEC-RAS 2D".
    # ------------------------------------------------------------------

    def _write_hecras_hdf5(self, path: str, timesteps=None):
        """Write a HEC-RAS 2D compatible HDF5 file readable by QGIS MDAL.

        Parameters
        ----------
        path : str
            Output .h5 file path.
        timesteps : list of (time_seconds, h, hu, hv) or None
            When supplied, results datasets are written; otherwise geometry only.
        """
        if not _HAVE_H5PY:
            raise RuntimeError("h5py is not installed.  Run: pip install h5py")
        if self._mesh_data is None:
            raise RuntimeError("No mesh data available")

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        node_z = self._mesh_data.get("node_z", np.zeros_like(node_x))

        # Build dense cell-vertex index array (HEC-RAS FacePoint Indexes,
        # -1 padded to maximum ring length).
        face_offsets = self._mesh_data.get("cell_face_offsets")
        face_nodes_arr = self._mesh_data.get("cell_face_nodes")
        cell_nodes_tri = self._mesh_data.get("cell_nodes")

        if face_offsets is not None and face_nodes_arr is not None:
            offsets = face_offsets.astype(np.int32)
            n_cells = int(offsets.size - 1)
            max_vp = int(max(offsets[i + 1] - offsets[i] for i in range(n_cells)))
            fp_idx = np.full((n_cells, max_vp), -1, dtype=np.int32)
            cell_cx = np.empty(n_cells, dtype=np.float64)
            cell_cy = np.empty(n_cells, dtype=np.float64)
            cell_min_z = np.empty(n_cells, dtype=np.float64)
            for i in range(n_cells):
                s, e = int(offsets[i]), int(offsets[i + 1])
                ring = face_nodes_arr[s:e].astype(np.int32)
                fp_idx[i, : e - s] = ring
                cell_cx[i] = float(np.mean(node_x[ring]))
                cell_cy[i] = float(np.mean(node_y[ring]))
                cell_min_z[i] = float(np.min(node_z[ring]))
        else:
            tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
            n_cells = tri.shape[0]
            fp_idx = tri
            cell_cx = np.mean(node_x[tri], axis=1)
            cell_cy = np.mean(node_y[tri], axis=1)
            cell_min_z = np.min(node_z[tri], axis=1)

        area_name = "Perimeter 1"

        with _h5py.File(path, "w") as f:
            f.attrs["File Type"] = np.bytes_(b"HEC-RAS Results")
            f.attrs["File Version"] = np.bytes_(b"HEC-RAS 7.0 April 2026")
            f.attrs["Units System"] = np.bytes_(
                b"US Customary" if self._unit_system == "US" else b"SI"
            )
            projection_wkt = 'LOCAL_CS["Unknown"]'
            if _HAVE_QGIS_CORE:
                try:
                    project_crs = QgsProject.instance().crs()
                    if project_crs is not None and project_crs.isValid():
                        projection_wkt = project_crs.toWkt()
                except Exception:
                    pass
            f.attrs["Projection"] = np.bytes_(projection_wkt.encode("utf-8"))

            # ---- Geometry ----
            geo = f.require_group("Geometry")
            geo.attrs["Complete Geometry"] = np.bytes_(b"True")
            geo.attrs["SI Units"] = np.bytes_(b"False" if self._unit_system == "US" else b"True")
            geo.attrs["Title"] = np.bytes_(b"Generated Geometry")
            geo.attrs["Version"] = np.bytes_(b"1.0")
            flow_areas_grp = geo.require_group("2D Flow Areas")

            # MDAL's HEC-RAS driver discovers 2D flow areas from
            # Geometry/2D Flow Areas/Attributes and expects the HEC-RAS 5.0.5+
            # field names, not the ad hoc top-level dataset used initially.
            attrs_dt = np.dtype(
                [
                    ("Name", "S16"),
                    ("Locked", np.uint8),
                    ("Mann", np.float32),
                    ("Multiple Face Mann n", np.uint8),
                    ("Composite LC", np.uint8),
                    ("Cell Vol Tol", np.float32),
                    ("Cell Min Area Fraction", np.float32),
                    ("Face Profile Tol", np.float32),
                    ("Face Area Tol", np.float32),
                    ("Face Conv Ratio", np.float32),
                    ("Laminar Depth", np.float32),
                    ("Min Face Length Ratio", np.float32),
                    ("Spacing dx", np.float32),
                    ("Spacing dy", np.float32),
                    ("Shift dx", np.float32),
                    ("Shift dy", np.float32),
                    ("Cell Count", np.int32),
                ]
            )
            flow_areas_grp.create_dataset(
                "Attributes",
                data=np.array(
                    [
                        (
                            area_name.encode(),
                            0,
                            np.float32(0.03),
                            0,
                            0,
                            np.float32(0.01),
                            np.float32(0.01),
                            np.float32(0.01),
                            np.float32(0.01),
                            np.float32(0.02),
                            np.float32(0.2),
                            np.float32(0.05),
                            np.float32(1.0),
                            np.float32(1.0),
                            np.float32(np.nan),
                            np.float32(np.nan),
                            n_cells,
                        )
                    ],
                    dtype=attrs_dt,
                ),
            )

            area_grp = flow_areas_grp.require_group(area_name)

            # Vertices ("FacePoints" in HEC-RAS 2D parlance)
            area_grp.create_dataset(
                "FacePoints Coordinate",
                data=np.column_stack([node_x, node_y]).astype(np.float64),
            )
            # Cell centroids
            area_grp.create_dataset(
                "Cells Center Coordinate",
                data=np.column_stack([cell_cx, cell_cy]).astype(np.float64),
            )
            # Minimum bed elevation per cell
            area_grp.create_dataset(
                "Cells Minimum Elevation",
                data=cell_min_z.astype(np.float32),
            )
            # Connectivity: nCells × maxVerts, -1 padded
            area_grp.create_dataset("Cells FacePoint Indexes", data=fp_idx)

            # ---- Results ----
            if timesteps:
                n_t = len(timesteps)
                times_hr = np.array([t / 3600.0 for t, *_ in timesteps], dtype=np.float32)

                ts_base = (
                    "Results/Unsteady/Output/Output Blocks/"
                    "Base Output/Unsteady Time Series"
                )
                ds_time = f.create_dataset(f"{ts_base}/Time", data=times_hr)
                ds_time.attrs["Number of actual Time Steps"] = np.array([n_t], dtype=np.int32)
                ds_time.attrs["Time"] = np.bytes_(b"Hours")

                # String time stamps (ddMONyyyy HH:MM:SS) — used by some MDAL versions
                stamps = []
                for t_s, *_ in timesteps:
                    total_min = int(t_s / 60)
                    hh, mm = divmod(total_min, 60)
                    stamps.append(f"01JAN2000 {hh:02d}:{mm:02d}:00".encode())
                f.create_dataset(
                    f"{ts_base}/Time Date Stamp",
                    data=np.array(stamps, dtype="S26"),
                )

                depth_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                wse_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                vel_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                vel_u_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                vel_v_arr = np.zeros((n_t, n_cells), dtype=np.float32)

                for ti, (_, h, hu, hv) in enumerate(timesteps):
                    h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                    hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                    hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                    hmag = np.maximum(h_f, 1e-12)
                    u = hu_f / hmag
                    v = hv_f / hmag
                    depth_arr[ti] = h_f.astype(np.float32)
                    wse_arr[ti] = (h_f + cell_min_z[:n_cells]).astype(np.float32)
                    vel_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                    vel_u_arr[ti] = u.astype(np.float32)
                    vel_v_arr[ti] = v.astype(np.float32)

                ar = f.require_group(f"{ts_base}/2D Flow Areas/{area_name}")
                ar.create_dataset("Depth", data=depth_arr)
                ar.create_dataset("Water Surface", data=wse_arr)
                ar.create_dataset("Cell Velocity - Magnitude", data=vel_arr)
                ar.create_dataset("Cell Velocity - X", data=vel_u_arr)
                ar.create_dataset("Cell Velocity - Y", data=vel_v_arr)
                # Alias names improve vector pairing across MDAL/QGIS versions.
                ar.create_dataset("Cell Velocity X", data=vel_u_arr)
                ar.create_dataset("Cell Velocity Y", data=vel_v_arr)
                ar.create_dataset("Velocity X", data=vel_u_arr)
                ar.create_dataset("Velocity Y", data=vel_v_arr)

                # MDAL's HEC-RAS reader expects Summary Output to exist when a
                # Results tree is present, even if most summary datasets are not.
                f.require_group(
                    "Results/Unsteady/Output/Output Blocks/"
                    f"Base Output/Summary Output/2D Flow Areas/{area_name}"
                )

    def _normalize_hecras_hdf_path(self, path: str) -> str:
        """MDAL's HEC-RAS driver is registered for .hdf, not .h5/.hdf5."""
        root, ext = os.path.splitext(path)
        if not ext:
            return f"{path}.hdf"
        if ext.lower() in {".h5", ".hdf5"}:
            return f"{root}.hdf"
        return path

    # ------------------------------------------------------------------
    # UGRID NetCDF export
    # ------------------------------------------------------------------
    def _write_ugrid_nc(self, path: str, timesteps=None):
        """Write a UGRID 1.0 NetCDF4 file readable by QGIS MDAL.

        The file follows the CF-1.8 + UGRID 1.0 conventions.  QGIS MDAL's
        UGRID driver natively pairs (velocity_u, velocity_v) into an arrow
        vector dataset without requiring any naming hacks.

        Parameters
        ----------
        path : str
            Output .nc file path.
        timesteps : list of (time_seconds, h, hu, hv) or None
            When supplied, result variables are written; otherwise topology only.
        """
        if not _HAVE_NETCDF4:
            raise RuntimeError("netCDF4 is not installed.  Run: pip install netCDF4")
        if self._mesh_data is None:
            raise RuntimeError("No mesh data available")

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        node_z = self._mesh_data.get("node_z", np.zeros_like(node_x))

        # Build face→node connectivity (zero-based, row per face, -1 padded)
        face_offsets = self._mesh_data.get("cell_face_offsets")
        face_nodes_arr = self._mesh_data.get("cell_face_nodes")
        cell_nodes_tri = self._mesh_data.get("cell_nodes")

        if face_offsets is not None and face_nodes_arr is not None:
            offsets = face_offsets.astype(np.int32)
            n_cells = int(offsets.size - 1)
            max_vp = int(max(offsets[i + 1] - offsets[i] for i in range(n_cells)))
            face_node = np.full((n_cells, max_vp), -1, dtype=np.int32)
            cell_cx = np.empty(n_cells, dtype=np.float64)
            cell_cy = np.empty(n_cells, dtype=np.float64)
            cell_min_z = np.empty(n_cells, dtype=np.float64)
            for i in range(n_cells):
                s, e = int(offsets[i]), int(offsets[i + 1])
                ring = face_nodes_arr[s:e].astype(np.int32)
                face_node[i, : e - s] = ring
                cell_cx[i] = float(np.mean(node_x[ring]))
                cell_cy[i] = float(np.mean(node_y[ring]))
                cell_min_z[i] = float(np.min(node_z[ring]))
        else:
            tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
            n_cells = tri.shape[0]
            max_vp = 3
            face_node = tri
            cell_cx = np.mean(node_x[tri], axis=1)
            cell_cy = np.mean(node_y[tri], axis=1)
            cell_min_z = np.min(node_z[tri], axis=1)

        n_nodes = int(node_x.size)

        # CRS info
        epsg_code = None
        crs_wkt = 'LOCAL_CS["Unknown"]'
        if _HAVE_QGIS_CORE:
            try:
                project_crs = QgsProject.instance().crs()
                if project_crs is not None and project_crs.isValid():
                    crs_wkt = project_crs.toWkt()
                    epsg_code = project_crs.postgisSrid() or None
            except Exception:
                pass

        with _netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            # Global attributes (CF + UGRID)
            ds.Conventions = "CF-1.8 UGRID-1.0"
            ds.title = "SWE2D backwater model results"
            ds.institution = "qgis-backwater-plugin"
            ds.history = "Created by swe2d_workbench_qt"
            ds.featureType = "mesh2D"

            # Dimensions
            ds.createDimension("node", n_nodes)
            ds.createDimension("face", n_cells)
            ds.createDimension("max_face_nodes", max_vp)
            if timesteps:
                ds.createDimension("time", len(timesteps))

            # ---- Mesh topology container variable ----
            mesh = ds.createVariable("mesh2d", "i4")
            mesh.cf_role = "mesh_topology"
            mesh.topology_dimension = 2
            mesh.node_coordinates = "node_x node_y"
            mesh.face_node_connectivity = "face_node"
            mesh.face_coordinates = "face_x face_y"

            # Node coordinates
            nx_var = ds.createVariable("node_x", "f8", ("node",))
            nx_var.standard_name = "projection_x_coordinate"
            nx_var.units = "m"
            nx_var.mesh = "mesh2d"
            nx_var.location = "node"
            nx_var.grid_mapping = "crs"
            nx_var[:] = node_x.astype(np.float64)

            ny_var = ds.createVariable("node_y", "f8", ("node",))
            ny_var.standard_name = "projection_y_coordinate"
            ny_var.units = "m"
            ny_var.mesh = "mesh2d"
            ny_var.location = "node"
            ny_var.grid_mapping = "crs"
            ny_var[:] = node_y.astype(np.float64)

            nz_var = ds.createVariable("node_z", "f8", ("node",))
            nz_var.standard_name = "altitude"
            nz_var.long_name = "bed elevation at node"
            nz_var.units = "m"
            nz_var.mesh = "mesh2d"
            nz_var.location = "node"
            nz_var.grid_mapping = "crs"
            nz_var[:] = node_z.astype(np.float64)

            # Face centroid coordinates
            fx_var = ds.createVariable("face_x", "f8", ("face",))
            fx_var.standard_name = "projection_x_coordinate"
            fx_var.units = "m"
            fx_var.mesh = "mesh2d"
            fx_var.location = "face"
            fx_var.grid_mapping = "crs"
            fx_var[:] = cell_cx.astype(np.float64)

            fy_var = ds.createVariable("face_y", "f8", ("face",))
            fy_var.standard_name = "projection_y_coordinate"
            fy_var.units = "m"
            fy_var.mesh = "mesh2d"
            fy_var.location = "face"
            fy_var.grid_mapping = "crs"
            fy_var[:] = cell_cy.astype(np.float64)

            # Face minimum bed elevation
            fz_var = ds.createVariable("face_z", "f8", ("face",))
            fz_var.long_name = "minimum bed elevation at face"
            fz_var.units = "m"
            fz_var.mesh = "mesh2d"
            fz_var.location = "face"
            fz_var.grid_mapping = "crs"
            fz_var[:] = cell_min_z.astype(np.float64)

            # Face→node connectivity (1-indexed as UGRID standard; -1 = fill)
            fn_var = ds.createVariable(
                "face_node", "i4", ("face", "max_face_nodes"),
                fill_value=-1,
            )
            fn_var.cf_role = "face_node_connectivity"
            fn_var.long_name = "face to node connectivity"
            fn_var.start_index = 0  # zero-based
            fn_var[:] = face_node

            # CRS variable
            crs_var = ds.createVariable("crs", "i4")
            crs_var.grid_mapping_name = "unknown"
            crs_var.crs_wkt = crs_wkt
            if epsg_code:
                crs_var.epsg_code = f"EPSG:{epsg_code}"

            # ---- Time-dependent results ----
            if timesteps:
                times_s = np.array([t for t, *_ in timesteps], dtype=np.float64)

                t_var = ds.createVariable("time", "f8", ("time",))
                t_var.standard_name = "time"
                t_var.long_name = "simulation time"
                t_var.units = "seconds since 2000-01-01 00:00:00"
                t_var.calendar = "proleptic_gregorian"
                t_var[:] = times_s

                depth_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                wse_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                vel_u_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                vel_v_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                vel_mag_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)

                for ti, (_, h, hu, hv) in enumerate(timesteps):
                    h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                    hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                    hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                    hmag = np.maximum(h_f, 1e-12)
                    u = hu_f / hmag
                    v = hv_f / hmag
                    depth_arr[ti] = h_f.astype(np.float32)
                    wse_arr[ti] = (h_f + cell_min_z[:n_cells]).astype(np.float32)
                    vel_u_arr[ti] = u.astype(np.float32)
                    vel_v_arr[ti] = v.astype(np.float32)
                    vel_mag_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)

                d_var = ds.createVariable(
                    "water_depth", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                d_var.standard_name = "water_depth"
                d_var.long_name = "water depth"
                d_var.units = "m"
                d_var.mesh = "mesh2d"
                d_var.location = "face"
                d_var.coordinates = "face_x face_y"
                d_var.grid_mapping = "crs"
                d_var[:] = depth_arr

                w_var = ds.createVariable(
                    "water_surface_elevation", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                w_var.standard_name = "water_surface_elevation"
                w_var.long_name = "water surface elevation"
                w_var.units = "m"
                w_var.mesh = "mesh2d"
                w_var.location = "face"
                w_var.coordinates = "face_x face_y"
                w_var.grid_mapping = "crs"
                w_var[:] = wse_arr

                # MDAL's UGRID driver infers vectors from component wording in
                # long_name, not just standard_name, on many QGIS builds.
                u_var = ds.createVariable(
                    "velocity_u", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                u_var.standard_name = "eastward_water_velocity"
                u_var.long_name = "eastward component of velocity"
                u_var.units = "m s-1"
                u_var.mesh = "mesh2d"
                u_var.location = "face"
                u_var.coordinates = "face_x face_y"
                u_var.grid_mapping = "crs"
                u_var[:] = vel_u_arr

                v_var = ds.createVariable(
                    "velocity_v", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                v_var.standard_name = "northward_water_velocity"
                v_var.long_name = "northward component of velocity"
                v_var.units = "m s-1"
                v_var.mesh = "mesh2d"
                v_var.location = "face"
                v_var.coordinates = "face_x face_y"
                v_var.grid_mapping = "crs"
                v_var[:] = vel_v_arr

                vm_var = ds.createVariable(
                    "velocity_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                vm_var.long_name = "velocity magnitude"
                vm_var.units = "m s-1"
                vm_var.mesh = "mesh2d"
                vm_var.location = "face"
                vm_var.coordinates = "face_x face_y"
                vm_var.grid_mapping = "crs"
                vm_var[:] = vel_mag_arr

    def _export_mesh_to_hdf5(self):
        if self._mesh_data is None:
            self._on_generate_mesh()
        if self._mesh_data is None:
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Mesh As HEC-RAS HDF5",
            "swe2d_mesh.hdf",
            "HEC-RAS HDF5 (*.hdf)",
        )
        if not out_path:
            return
        try:
            out_path = self._normalize_hecras_hdf_path(out_path)
            self._write_hecras_hdf5(out_path)
            n_nodes = int(self._mesh_data["node_x"].shape[0])
            self._log(f"Saved HEC-RAS HDF5 mesh: {out_path} (nodes={n_nodes})")
            self.layer_status_lbl.setText("Mesh saved to HEC-RAS HDF5.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "HDF5 Export", f"Export failed:\n{exc}")

    def _export_results_to_hdf5(self):
        if self._mesh_data is None or not self._snapshot_timesteps:
            self._log("Run the model first (snapshots must be captured) to export HDF5 results.")
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Results As HEC-RAS HDF5",
            "swe2d_results.hdf",
            "HEC-RAS HDF5 (*.hdf)",
        )
        if not out_path:
            return
        try:
            out_path = self._normalize_hecras_hdf_path(out_path)
            self._write_hecras_hdf5(out_path, timesteps=self._snapshot_timesteps)
            n_ts = len(self._snapshot_timesteps)
            self._log(f"Saved HEC-RAS HDF5 results: {out_path} ({n_ts} timesteps)")
            self.layer_status_lbl.setText(f"Results saved to HEC-RAS HDF5 ({n_ts} timesteps).")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "HDF5 Export", f"Export failed:\n{exc}")

    def _export_results_to_ugrid(self):
        if self._mesh_data is None or not self._snapshot_timesteps:
            self._log("Run the model first (snapshots must be captured) to export UGRID results.")
            return
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Results As UGRID NetCDF",
            "swe2d_results.nc",
            "UGRID NetCDF (*.nc)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".nc"):
            out_path += ".nc"
        try:
            self._write_ugrid_nc(out_path, timesteps=self._snapshot_timesteps)
            n_ts = len(self._snapshot_timesteps)
            self._log(f"Saved UGRID NetCDF results: {out_path} ({n_ts} timesteps)")
            self.layer_status_lbl.setText(f"Results saved to UGRID NetCDF ({n_ts} timesteps).")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "UGRID Export", f"Export failed:\n{exc}")

    def _on_snapshot(self):
        """Write all captured run timesteps to a temporary HEC-RAS HDF5 file."""
        if self._mesh_data is None or not self._snapshot_timesteps:
            self._log("No snapshot data available — run the model with an output interval set first.")
            return
        import os
        import tempfile
        snap_path = os.path.join(tempfile.gettempdir(), "swe2d_snapshot.hdf")
        try:
            self._write_hecras_hdf5(snap_path, timesteps=self._snapshot_timesteps)
            n_ts = len(self._snapshot_timesteps)
            last_t_hr = self._snapshot_timesteps[-1][0] / 3600.0
            self._log(
                f"Snapshot written → {snap_path}  "
                f"({n_ts} timestep(s), last t={last_t_hr:.3f} hr, "
                f"interval={self.output_interval_edit.text()})"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Snapshot", f"Snapshot failed:\n{exc}")

    def _structured_mesh(self, nx: int, ny: int, lx: float, ly: float, bed_amp: float, cell_layout: str = "quad"):
        xs = np.linspace(0.0, lx, nx + 1)
        ys = np.linspace(0.0, ly, ny + 1)
        Xg, Yg = np.meshgrid(xs, ys)
        node_x = Xg.ravel().copy()
        node_y = Yg.ravel().copy()
        if bed_amp > 0.0:
            node_z = bed_amp * np.sin(np.pi * node_x / max(lx, 1.0)) * np.cos(np.pi * node_y / max(ly, 1.0))
        else:
            node_z = np.zeros_like(node_x)

        stride = nx + 1
        cells: List[int] = []
        face_nodes: List[int] = []
        face_offsets: List[int] = [0]
        make_quads = str(cell_layout).strip().lower().startswith("quad")

        for j in range(ny):
            for i in range(nx):
                n00 = j * stride + i
                n10 = j * stride + i + 1
                n01 = (j + 1) * stride + i
                n11 = (j + 1) * stride + i + 1

                if make_quads:
                    # Polygon face ring (counter-clockwise) for solver path.
                    face_nodes.extend([n00, n10, n11, n01])
                    face_offsets.append(len(face_nodes))
                # Triangles are always kept for plotting and export compatibility.
                cells.extend([n00, n10, n11])
                cells.extend([n00, n11, n01])

        cell_nodes = np.array(cells, dtype=np.int32)
        if make_quads:
            return (
                node_x,
                node_y,
                node_z,
                cell_nodes,
                np.asarray(face_offsets, dtype=np.int32),
                np.asarray(face_nodes, dtype=np.int32),
            )
        return node_x, node_y, node_z, cell_nodes, None, None

    def _structured_side_edges(self, nx: int, ny: int):
        stride = nx + 1

        bottom_n0 = np.array([i for i in range(nx)], dtype=np.int32)
        bottom_n1 = np.array([i + 1 for i in range(nx)], dtype=np.int32)

        top_base = ny * stride
        top_n0 = np.array([top_base + i for i in range(nx)], dtype=np.int32)
        top_n1 = np.array([top_base + i + 1 for i in range(nx)], dtype=np.int32)

        left_n0 = np.array([j * stride for j in range(ny)], dtype=np.int32)
        left_n1 = np.array([(j + 1) * stride for j in range(ny)], dtype=np.int32)

        right_n0 = np.array([j * stride + nx for j in range(ny)], dtype=np.int32)
        right_n1 = np.array([(j + 1) * stride + nx for j in range(ny)], dtype=np.int32)

        return {
            "bottom": (bottom_n0, bottom_n1),
            "top": (top_n0, top_n1),
            "left": (left_n0, left_n1),
            "right": (right_n0, right_n1),
        }

    def _mesh_boundary_edges(self):
        if self._mesh_data is None:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Build a boundary edge set from mesh faces: edges seen once are on the boundary.
        edge_count: Dict[Tuple[int, int], int] = {}
        edge_oriented: Dict[Tuple[int, int], Tuple[int, int]] = {}

        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offsets = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            for i in range(offsets.size - 1):
                s = int(offsets[i])
                e = int(offsets[i + 1])
                poly = faces[s:e]
                if poly.size < 3:
                    continue
                for k in range(poly.size):
                    a = int(poly[k])
                    b = int(poly[(k + 1) % poly.size])
                    key = (a, b) if a < b else (b, a)
                    edge_count[key] = edge_count.get(key, 0) + 1
                    if key not in edge_oriented:
                        edge_oriented[key] = (a, b)
        else:
            tris = self._mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
            for tri in tris:
                a0, a1, a2 = int(tri[0]), int(tri[1]), int(tri[2])
                for a, b in ((a0, a1), (a1, a2), (a2, a0)):
                    key = (a, b) if a < b else (b, a)
                    edge_count[key] = edge_count.get(key, 0) + 1
                    if key not in edge_oriented:
                        edge_oriented[key] = (a, b)

        n0 = []
        n1 = []
        for key, cnt in edge_count.items():
            if cnt == 1:
                a, b = edge_oriented[key]
                n0.append(a)
                n1.append(b)

        if not n0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        return np.asarray(n0, dtype=np.int32), np.asarray(n1, dtype=np.int32)

    def _default_bc_for_edges(self, edge_n0: np.ndarray, edge_n1: np.ndarray):
        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        ymin = float(np.min(node_y))
        ymax = float(np.max(node_y))

        mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
        my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])

        d_left = np.abs(mx - xmin)
        d_right = np.abs(mx - xmax)
        d_bottom = np.abs(my - ymin)
        d_top = np.abs(my - ymax)
        d = np.vstack([d_left, d_right, d_bottom, d_top])
        side_idx = np.argmin(d, axis=0)
        side_names = ["left", "right", "bottom", "top"]

        bc_type = np.zeros(edge_n0.shape[0], dtype=np.int32)
        bc_val = np.zeros(edge_n0.shape[0], dtype=np.float64)
        for i, si in enumerate(side_idx):
            side = side_names[int(si)]
            bc_type[i] = int(self._bc_type_boxes[side].currentData())
            bc_val[i] = float(self._bc_value_spins[side].value())
        return bc_type, bc_val

    def _apply_bc_layer_overrides(self, edge_n0: np.ndarray, edge_n1: np.ndarray, bc_type: np.ndarray, bc_val: np.ndarray):
        if not _HAVE_QGIS_CORE:
            return bc_type, bc_val
        if not hasattr(self, "bc_lines_layer_combo"):
            return bc_type, bc_val

        bc_layer = self._combo_layer(self.bc_lines_layer_combo, "vector")
        if bc_layer is None:
            return bc_type, bc_val

        fields = set(bc_layer.fields().names())
        type_field = None
        for cand in ("bc_type", "type", "bc"):
            if cand in fields:
                type_field = cand
                break
        val_field = None
        for cand in ("bc_value", "value", "bc_val"):
            if cand in fields:
                val_field = cand
                break
        prio_field = "priority" if "priority" in fields else None

        if type_field is None:
            self._log("BC polyline layer selected but no bc_type/type field found; skipping overrides.")
            return bc_type, bc_val

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]

        features = []
        for ft in bc_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                t = int(ft[type_field])
            except Exception:
                continue
            v = 0.0
            if val_field is not None:
                try:
                    v = float(ft[val_field])
                except Exception:
                    v = 0.0
            pr = 0
            if prio_field is not None:
                try:
                    pr = int(ft[prio_field])
                except Exception:
                    pr = 0
            features.append((pr, geom, t, v))

        if not features:
            return bc_type, bc_val

        # Apply highest-priority BC feature per boundary edge.
        # Use midpoint-distance instead of intersects() to avoid corner contamination:
        # a bc_line endpoint at a mesh corner node technically "intersects" wall edges
        # that share that corner even though the lines are perpendicular.  Requiring
        # the edge midpoint to be within half the edge length of the bc_line means
        # only truly overlapping (collinear/parallel) edges are matched.
        features.sort(key=lambda x: x[0], reverse=True)
        applied = 0
        for i in range(edge_n0.size):
            x0 = float(node_x[edge_n0[i]]); y0 = float(node_y[edge_n0[i]])
            x1 = float(node_x[edge_n1[i]]); y1 = float(node_y[edge_n1[i]])
            tol = math.hypot(x1 - x0, y1 - y0) * 0.5
            mid = QgsGeometry.fromPointXY(QgsPointXY(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
            for _, g, t, v in features:
                if mid.distance(g) < tol:
                    changed = (int(bc_type[i]) != int(t)) or (not np.isclose(float(bc_val[i]), float(v)))
                    bc_type[i] = int(t)
                    bc_val[i] = float(v)
                    if changed:
                        applied += 1
                    break

        if applied:
            self._log(f"BC line static overrides applied to {applied}/{edge_n0.size} boundary edges from '{bc_layer.name()}'.")

        return bc_type, bc_val

    def _collect_bc_layer_hydrographs(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
        edge_hydro: Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]] = {}
        if not _HAVE_QGIS_CORE:
            return edge_hydro
        if not hasattr(self, "bc_lines_layer_combo"):
            return edge_hydro

        bc_layer = self._combo_layer(self.bc_lines_layer_combo, "vector")
        if bc_layer is None:
            return edge_hydro

        fields = set(bc_layer.fields().names())
        type_field = "bc_type" if "bc_type" in fields else ("type" if "type" in fields else None)
        if type_field is None:
            return edge_hydro
        prio_field = "priority" if "priority" in fields else None

        hydro_field = None
        for cand in ("hydrograph", "hydrograph_text", "hydro", "hg"):
            if cand in fields:
                hydro_field = cand
                break

        hgid_field = "hydrograph_id" if "hydrograph_id" in fields else None
        hlyr_field = "hydrograph_layer" if "hydrograph_layer" in fields else None
        hydro_lookup: Dict[str, str] = {}
        if hgid_field is not None:
            hydro_layers = [
                lyr for lyr in self._iter_project_layers()
                if isinstance(lyr, QgsVectorLayer) and str(lyr.name()).lower() in ("swe2d_hydrographs", "swe2d_hydrographs")
            ]
            if hydro_layers:
                hlyr = hydro_layers[0]
                hfields = set(hlyr.fields().names())
                if "hydrograph_id" in hfields and "hydrograph" in hfields:
                    for hft in hlyr.getFeatures():
                        hid = str(hft["hydrograph_id"] or "").strip()
                        htxt = str(hft["hydrograph"] or "").strip()
                        if hid and htxt:
                            hydro_lookup[hid] = htxt

        if hydro_field is None and hgid_field is None:
            return edge_hydro

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]

        features = []
        for ft in bc_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                t = int(ft[type_field])
            except Exception:
                continue
            if t not in (_BC_TS_FLOW, _BC_TS_STAGE):
                continue
            raw_h = str(ft[hydro_field] or "").strip() if hydro_field is not None else ""
            ref_layer = str(ft[hlyr_field] or "").strip() if hlyr_field is not None else ""
            if not raw_h and hgid_field is not None:
                hid = str(ft[hgid_field] or "").strip()
                if hid in hydro_lookup:
                    raw_h = hydro_lookup[hid]

            # If hydrograph field points to a map layer/table name or layer id,
            # load Time/Value records from that layer.
            if not raw_h and (ref_layer or (hydro_field is not None and str(ft[hydro_field] or "").strip())):
                layer_ref = ref_layer or str(ft[hydro_field] or "").strip()
                target_layer = None
                for lyr in self._iter_project_layers():
                    if not isinstance(lyr, QgsVectorLayer):
                        continue
                    try:
                        if lyr.id() == layer_ref or str(lyr.name()) == layer_ref:
                            target_layer = lyr
                            break
                    except Exception:
                        continue
                if target_layer is not None:
                    hid = str(ft[hgid_field] or "").strip() if hgid_field is not None else ""
                    hg_layer = self._hydrograph_from_layer(target_layer, hydrograph_id=hid, bc_type=t)
                    if hg_layer is not None:
                        pr = 0
                        if prio_field is not None:
                            try:
                                pr = int(ft[prio_field])
                            except Exception:
                                pr = 0
                        features.append((pr, geom, t, hg_layer))
                        continue

            if not raw_h:
                continue
            try:
                hg = self._parse_hydrograph_text(raw_h)
            except Exception:
                continue
            if hg is None:
                continue
            pr = 0
            if prio_field is not None:
                try:
                    pr = int(ft[prio_field])
                except Exception:
                    pr = 0
            features.append((pr, geom, t, hg))

        if not features:
            return edge_hydro

        features.sort(key=lambda x: x[0], reverse=True)
        for i in range(edge_n0.size):
            x0 = float(node_x[edge_n0[i]]); y0 = float(node_y[edge_n0[i]])
            x1 = float(node_x[edge_n1[i]]); y1 = float(node_y[edge_n1[i]])
            tol = math.hypot(x1 - x0, y1 - y0) * 0.5
            mid = QgsGeometry.fromPointXY(QgsPointXY(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
            for _pr, g, t, hg in features:
                if mid.distance(g) < tol:
                    edge_hydro[i] = (t, hg)
                    break

        if edge_hydro:
            self._log(f"BC line hydrographs applied to {len(edge_hydro)} boundary edges.")
        return edge_hydro

    def _collect_boundary_arrays(self):
        if self._mesh_data is None:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )

        edge_n0, edge_n1 = self._mesh_boundary_edges()
        if edge_n0.size == 0:
            self._log("No boundary edges detected in mesh.")
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )

        bc_type, bc_val = self._default_bc_for_edges(edge_n0, edge_n1)
        bc_type, bc_val = self._apply_bc_layer_overrides(edge_n0, edge_n1, bc_type, bc_val)
        return edge_n0, edge_n1, bc_type, bc_val

    def _parse_time_hours(self, token: str) -> float:
        t = str(token).strip()
        if not t:
            raise ValueError("empty time token")
        if ":" in t:
            parts = t.split(":")
            if len(parts) == 2:
                hh = float(parts[0])
                mm = float(parts[1])
                return hh + (mm / 60.0)
            if len(parts) == 3:
                hh = float(parts[0])
                mm = float(parts[1])
                ss = float(parts[2])
                return hh + (mm / 60.0) + (ss / 3600.0)
            raise ValueError(f"invalid HH:MM(:SS) token '{t}'")
        return float(t)

    def _parse_run_duration_seconds(self) -> float:
        hrs = self._parse_time_hours(self.run_time_edit.text())
        if hrs <= 0.0:
            raise ValueError("run duration must be > 0")
        return 3600.0 * hrs

    def _parse_hydrograph_text(self, text: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raw = str(text or "").strip()
        if not raw:
            return None

        pairs: List[Tuple[float, float]] = []
        chunks = raw.replace("\n", ";").split(";")
        for chunk in chunks:
            c = chunk.strip()
            if not c:
                continue
            if "," in c:
                a, b = c.split(",", 1)
            elif "=" in c:
                a, b = c.split("=", 1)
            else:
                raise ValueError(f"hydrograph entry '{c}' must use ',' or '=' between time and value")
            th = self._parse_time_hours(a.strip())
            vv = float(b.strip())
            pairs.append((th * 3600.0, vv))

        if not pairs:
            return None

        pairs.sort(key=lambda x: x[0])
        tsec = np.array([p[0] for p in pairs], dtype=np.float64)
        vals = np.array([p[1] for p in pairs], dtype=np.float64)

        uniq_t = []
        uniq_v = []
        for ti, vi in zip(tsec.tolist(), vals.tolist()):
            if uniq_t and abs(ti - uniq_t[-1]) < 1.0e-9:
                uniq_v[-1] = vi
            else:
                uniq_t.append(ti)
                uniq_v.append(vi)

        return np.asarray(uniq_t, dtype=np.float64), np.asarray(uniq_v, dtype=np.float64)

    def _open_hydrograph_editor(self, side: str):
        if side not in self._bc_ts_edits:
            return
        current = self._bc_ts_edits[side].text()
        dlg = HydrographEditorDialog(side=side, initial_text=current, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_text = dlg.hydrograph_text().strip()
        if new_text:
            try:
                self._parse_hydrograph_text(new_text)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Hydrograph",
                    f"Invalid hydrograph format for {side} side: {exc}",
                )
                return
        self._bc_ts_edits[side].setText(new_text)

    def _build_side_hydrographs(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        side_hg: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if not hasattr(self, "_bc_ts_edits"):
            return side_hg
        for side, edit in self._bc_ts_edits.items():
            hg = self._parse_hydrograph_text(edit.text())
            if hg is not None:
                side_hg[side] = hg
        return side_hg

    def _hydrograph_from_layer(self, layer, hydrograph_id: str = "", bc_type: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if layer is None or not isinstance(layer, QgsVectorLayer):
            return None
        fields = set(layer.fields().names())
        t_field = None
        for cand in ("Time", "time", "t", "hours"):
            if cand in fields:
                t_field = cand
                break
        v_field = None
        for cand in ("Value", "value", "val", "q", "stage"):
            if cand in fields:
                v_field = cand
                break
        if t_field is None or v_field is None:
            return None

        hid_field = "hydrograph_id" if "hydrograph_id" in fields else None
        bct_field = "bc_type" if "bc_type" in fields else None

        pairs: List[Tuple[float, float]] = []
        for ft in layer.getFeatures():
            if hid_field is not None and hydrograph_id:
                hid = str(ft[hid_field] or "").strip()
                if hid != hydrograph_id:
                    continue
            if bct_field is not None and bc_type is not None:
                try:
                    if int(ft[bct_field]) != int(bc_type):
                        continue
                except Exception:
                    pass
            try:
                th = self._parse_time_hours(str(ft[t_field]).strip())
                vv = float(ft[v_field])
            except Exception:
                continue
            pairs.append((th * 3600.0, vv))

        if not pairs:
            return None
        pairs.sort(key=lambda x: x[0])
        tsec = np.asarray([p[0] for p in pairs], dtype=np.float64)
        vals = np.asarray([p[1] for p in pairs], dtype=np.float64)
        return tsec, vals

    def _interp_hydrograph(self, hg: Tuple[np.ndarray, np.ndarray], t_sec: float) -> float:
        t, v = hg
        if t.size == 1:
            return float(v[0])
        if t_sec <= float(t[0]):
            return float(v[0])
        if t_sec >= float(t[-1]):
            return float(v[-1])
        return float(np.interp(t_sec, t, v))

    def _distribute_total_flow_to_unit_q(
        self,
        edge_n0: np.ndarray,
        edge_n1: np.ndarray,
        bc_type_step: np.ndarray,
        bc_val_step: np.ndarray,
        bc_type_template: np.ndarray,
        side_hydrographs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        edge_hydrographs: Optional[Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]] = None,
    ) -> np.ndarray:
        """Convert total discharge Q inputs into INFLOW_Q unit discharge q [L^2/T].

        Flow BC values are entered as total discharge for each BC source. This
        routine distributes each source over boundary-edge length to recover the
        solver's required unit discharge input, with optional progressive
        activation of lower-elevation boundary edges as Q rises.
        """
        if edge_n0.size == 0:
            return bc_val_step

        out_val = bc_val_step.astype(np.float64, copy=True)
        flow_idx = np.where(bc_type_step.astype(np.int32) == 2)[0]
        if flow_idx.size == 0:
            return out_val

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        node_z = self._mesh_data["node_z"]

        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        ymin = float(np.min(node_y))
        ymax = float(np.max(node_y))

        mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
        my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
        d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax), np.abs(my - ymin), np.abs(my - ymax)])
        side_idx = np.argmin(d, axis=0)
        side_names = ["left", "right", "bottom", "top"]

        edge_len = np.hypot(node_x[edge_n1] - node_x[edge_n0], node_y[edge_n1] - node_y[edge_n0])
        edge_z = 0.5 * (node_z[edge_n0] + node_z[edge_n1])

        progressive = True
        if hasattr(self, "inflow_progressive_chk") and self.inflow_progressive_chk is not None:
            try:
                progressive = bool(self.inflow_progressive_chk.isChecked())
            except Exception:
                progressive = True

        groups: Dict[Tuple, Dict[str, object]] = {}
        for i in flow_idx.tolist():
            side = side_names[int(side_idx[i])]

            peak_q = abs(float(out_val[i]))
            key: Tuple

            if edge_hydrographs is not None and i in edge_hydrographs and int(edge_hydrographs[i][0]) == _BC_TS_FLOW:
                hg = edge_hydrographs[i][1]
                try:
                    peak_q = float(np.max(np.abs(hg[1]))) if hg[1].size else abs(float(out_val[i]))
                except Exception:
                    peak_q = abs(float(out_val[i]))
                key = ("edge_hg", id(hg))
            elif int(bc_type_template[i]) == _BC_TS_FLOW:
                hg = side_hydrographs.get(side)
                if hg is not None:
                    try:
                        peak_q = float(np.max(np.abs(hg[1]))) if hg[1].size else abs(float(out_val[i]))
                    except Exception:
                        peak_q = abs(float(out_val[i]))
                key = ("side_hg", side)
            else:
                # For static flow entries, keep sources with different entered Q values separate.
                key = ("static", side, round(float(out_val[i]), 12))

            if key not in groups:
                groups[key] = {
                    "idx": [],
                    "peak_q": max(peak_q, 0.0),
                }
            groups[key]["idx"].append(i)
            groups[key]["peak_q"] = max(float(groups[key]["peak_q"]), max(peak_q, 0.0))

        eps = 1.0e-12
        for grp in groups.values():
            idx = np.asarray(grp["idx"], dtype=np.int32)
            if idx.size == 0:
                continue

            q_total = float(out_val[idx[0]])
            if abs(q_total) <= eps:
                out_val[idx] = 0.0
                continue

            g_len = edge_len[idx]
            g_z = edge_z[idx]
            total_len = float(np.sum(g_len))
            if total_len <= eps:
                out_val[idx] = 0.0
                continue

            if progressive:
                peak_q = max(float(grp["peak_q"]), abs(q_total))
                frac = min(1.0, abs(q_total) / max(peak_q, eps))
                target_len = frac * total_len
            else:
                target_len = total_len

            if target_len <= eps:
                out_val[idx] = 0.0
                continue

            order = np.argsort(g_z, kind="stable")
            idx_sorted = idx[order]
            len_sorted = g_len[order]
            csum = np.cumsum(len_sorted)
            n_active = int(np.searchsorted(csum, target_len, side="left") + 1)
            n_active = max(1, min(n_active, idx_sorted.size))
            active_idx = idx_sorted[:n_active]
            active_len = float(np.sum(edge_len[active_idx]))
            if active_len <= eps:
                out_val[idx] = 0.0
                continue

            q_unit = q_total / active_len
            out_val[idx] = 0.0
            out_val[active_idx] = q_unit

        return out_val

    def _apply_timeseries_bc_values(
        self,
        edge_n0: np.ndarray,
        edge_n1: np.ndarray,
        bc_type: np.ndarray,
        bc_val: np.ndarray,
        side_hydrographs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        t_sec: float,
        edge_hydrographs: Optional[Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if edge_n0.size == 0:
            return bc_type, bc_val

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        ymin = float(np.min(node_y))
        ymax = float(np.max(node_y))

        mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
        my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
        d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax), np.abs(my - ymin), np.abs(my - ymax)])
        side_idx = np.argmin(d, axis=0)
        side_names = ["left", "right", "bottom", "top"]

        out_type = bc_type.astype(np.int32, copy=True)
        out_val = bc_val.astype(np.float64, copy=True)
        for i in range(edge_n0.size):
            if edge_hydrographs is not None and i in edge_hydrographs:
                tcode, hg = edge_hydrographs[i]
                out_val[i] = self._interp_hydrograph(hg, t_sec)
                out_type[i] = 2 if int(tcode) == _BC_TS_FLOW else 3
                continue

            tcode = int(out_type[i])
            if tcode not in (_BC_TS_FLOW, _BC_TS_STAGE):
                continue
            side = side_names[int(side_idx[i])]
            if side not in side_hydrographs:
                continue
            out_val[i] = self._interp_hydrograph(side_hydrographs[side], t_sec)
            out_type[i] = 2 if tcode == _BC_TS_FLOW else 3
        return out_type, out_val

    def _on_generate_mesh(self):
        nx = int(self.nx_spin.value())
        ny = int(self.ny_spin.value())
        lx = float(self.lx_spin.value())
        ly = float(self.ly_spin.value())
        bed_amp = float(self.bed_amp_spin.value())
        layout_mode = str(self.mesh_layout_combo.currentData() or "quad")

        node_x, node_y, node_z, cell_nodes, face_offsets, face_nodes = self._structured_mesh(
            nx,
            ny,
            lx,
            ly,
            bed_amp,
            layout_mode,
        )
        self._mesh_data = {
            "nx": np.array(nx),
            "ny": np.array(ny),
            "lx": np.array(lx),
            "ly": np.array(ly),
            "cell_layout": np.array(layout_mode),
            "node_x": node_x,
            "node_y": node_y,
            "node_z": node_z,
            "cell_nodes": cell_nodes,
        }
        if face_offsets is not None and face_nodes is not None:
            self._mesh_data["cell_face_offsets"] = face_offsets
            self._mesh_data["cell_face_nodes"] = face_nodes
            n_cells = int(face_offsets.size - 1)
            n_tri = int(cell_nodes.shape[0] // 3)
            self.mesh_info_lbl.setText(
                f"Generated mesh: nodes={node_x.shape[0]}, cells={n_cells} (quads), plot-triangles={n_tri}"
            )
        else:
            n_cells = int(cell_nodes.shape[0] // 3)
            self.mesh_info_lbl.setText(
                f"Generated mesh: nodes={node_x.shape[0]}, cells={n_cells}, triangles={n_cells}"
            )
        self._log(
            f"Mesh generated: nx={nx}, ny={ny}, lx={lx:.2f}, ly={ly:.2f}, bed_amp={bed_amp:.4f}, layout={layout_mode}"
        )
        self._result_data = None
        self.view_mode_combo.setCurrentText("Mesh")
        self._refresh_plot()

    def _initial_state(self):
        assert self._mesh_data is not None
        cell_x, cell_y = self._mesh_cell_centroids()

        lx = float(self._mesh_data["lx"])
        ly = float(self._mesh_data["ly"])
        base_depth = float(self.base_depth_spin.value())
        bump = float(self.bump_depth_spin.value())

        cx0 = 0.5 * lx
        cy0 = 0.5 * ly
        sx = max(lx * 0.16, 1.0)
        sy = max(ly * 0.16, 1.0)
        h0 = base_depth + bump * np.exp(-(((cell_x - cx0) / sx) ** 2 + ((cell_y - cy0) / sy) ** 2))
        hu0 = np.zeros_like(h0)
        hv0 = np.zeros_like(h0)
        return h0, hu0, hv0

    def _on_cancel(self):
        self._cancel_requested = True
        self._log("Cancellation requested...")

    def _on_run(self):
        if self._mesh_data is None:
            self._on_generate_mesh()
        if self._mesh_data is None:
            return
        if not swe2d_available() or SWE2DBackend is None:
            QtWidgets.QMessageBox.critical(self, "2D SWE", "Native 2D backend is not available. Build backwater_swe2d first.")
            return

        self._cancel_requested = False
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        backend = None
        try:
            node_x = self._mesh_data["node_x"]
            node_y = self._mesh_data["node_y"]
            node_z = self._mesh_data["node_z"]
            cell_nodes = self._mesh_data["cell_nodes"]
            face_offsets = self._mesh_data.get("cell_face_offsets")
            face_nodes = self._mesh_data.get("cell_face_nodes")
            bc_n0, bc_n1, bc_tp, bc_vl = self._collect_boundary_arrays()
            side_hydrographs = self._build_side_hydrographs()
            edge_hydrographs = self._collect_bc_layer_hydrographs(bc_n0, bc_n1)
            h0, hu0, hv0 = self._initial_state()
            n_mann_cell = self._build_spatial_manning_array()
            self._update_unit_system_from_crs()

            run_duration_s = self._parse_run_duration_seconds()
            dt = float(self.dt_spin.value())
            reconstruction_mode = int(self.reconstruction_combo.currentData())
            reconstruction_name = self.reconstruction_combo.currentText().strip()
            rain_rate_mps = float(self.rain_rate_spin.value()) / 1000.0 / 3600.0
            cell_source_cms = self._build_internal_flow_source_cms()

            # Snapshot output interval — clamp to at least 1 s to avoid div-by-zero
            _oi_hr = self._parse_time_hours(self.output_interval_edit.text())
            output_interval_s = max(1.0, _oi_hr * 3600.0)
            self._snapshot_timesteps = []
            _next_snap_t = output_interval_s

            dynamic_bc = bool(np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs)
            if dynamic_bc:
                self._log("Timeseries BC mode active (flow/stage hydrographs).")

            self._log("Starting 2D run...")
            self._log(f"Reconstruction mode: {reconstruction_name}")
            if rain_rate_mps > 0.0:
                self._log(f"Rain-on-grid active: {float(self.rain_rate_spin.value()):.3f} mm/hr")
            if cell_source_cms is not None:
                self._log(f"Internal source/sink forcing active: total_Q={float(np.sum(cell_source_cms)):.6f} cms")
            backend = SWE2DBackend()

            bc_tp_init = bc_tp.copy()
            bc_vl_init = bc_vl.copy()
            if dynamic_bc:
                bc_tp_init, bc_vl_init = self._apply_timeseries_bc_values(
                    bc_n0, bc_n1, bc_tp_init, bc_vl_init, side_hydrographs, 0.0, edge_hydrographs
                )
            bc_vl_init = self._distribute_total_flow_to_unit_q(
                bc_n0,
                bc_n1,
                bc_tp_init,
                bc_vl_init,
                bc_tp,
                side_hydrographs,
                edge_hydrographs,
            )
            if face_offsets is not None and face_nodes is not None:
                backend.build_mesh(
                    node_x,
                    node_y,
                    node_z,
                    face_nodes,
                    bc_n0,
                    bc_n1,
                    bc_tp_init,
                    bc_vl_init,
                    face_offsets,
                )
            else:
                backend.build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp_init, bc_vl_init)
            backend.initialize(
                h0, hu0, hv0,
                g=float(self._gravity),
                n_mann=float(self.n_mann_spin.value()),
                n_mann_cell=n_mann_cell,
                cfl=float(self.cfl_spin.value()),
                h_min=float(self.h_min_spin.value()),
                dt_fixed=dt,
                dt_max=dt,
                spatial_discretization=reconstruction_mode,
            )

            last_diag = None
            t_accum = 0.0
            i = 0
            if dynamic_bc and not backend.supports_dynamic_boundary_update():
                raise RuntimeError("Native module does not support dynamic boundary updates. Rebuild backwater_swe2d.")

            while t_accum < run_duration_s:
                if self._cancel_requested:
                    break

                if dynamic_bc:
                    bc_tp_step, bc_vl_step = self._apply_timeseries_bc_values(
                        bc_n0, bc_n1, bc_tp, bc_vl, side_hydrographs, t_accum, edge_hydrographs
                    )
                    bc_vl_step = self._distribute_total_flow_to_unit_q(
                        bc_n0,
                        bc_n1,
                        bc_tp_step,
                        bc_vl_step,
                        bc_tp,
                        side_hydrographs,
                        edge_hydrographs,
                    )
                    backend.set_boundary_conditions(bc_n0, bc_n1, bc_tp_step, bc_vl_step)

                last_diag = backend.step(dt)
                self._apply_external_sources(
                    backend,
                    float(last_diag.get("dt", dt)),
                    rain_rate_mps,
                    cell_source_cms,
                )
                t_accum += float(last_diag.get("dt", dt))

                # Capture snapshot at each output interval boundary
                if t_accum >= _next_snap_t:
                    h_s, hu_s, hv_s = backend.get_state()
                    self._snapshot_timesteps.append(
                        (t_accum, h_s.copy(), hu_s.copy(), hv_s.copy())
                    )
                    _next_snap_t += output_interval_s

                pct = int(min(100.0, (t_accum / max(run_duration_s, 1.0e-9)) * 100.0))
                self.progress_bar.setValue(pct)
                i += 1
                if i == 1 or i % 10 == 0 or pct >= 100:
                    max_courant = float(last_diag.get("max_courant", float("nan")))
                    max_wse_res = float(
                        last_diag.get(
                            "max_depth_residual",
                            last_diag.get("max_wse_elev_error", float("nan")),
                        )
                    )
                    cmax_txt = f"{max_courant:.5f}" if np.isfinite(max_courant) and max_courant >= 0.0 else "n/a"
                    wse_res_txt = f"{max_wse_res:.6e}" if np.isfinite(max_wse_res) and max_wse_res >= 0.0 else "n/a"
                    self._log(
                        f"step={i} t={t_accum / 3600.0:.3f} hr / {run_duration_s / 3600.0:.3f} hr "
                        f"dt={float(last_diag.get('dt', 0.0)):.5f} "
                        f"gpu={bool(last_diag.get('gpu_active', False))} wet={last_diag.get('wet_cells', '?')} "
                        f"Cmax={cmax_txt} WSEres={wse_res_txt}"
                    )
                QtWidgets.QApplication.processEvents()

            h, hu, hv = backend.get_state()
            self._result_data = {
                "h": h,
                "hu": hu,
                "hv": hv,
                "gpu_active": np.array(bool(backend.gpu_active())),
                "last_mass_total": np.array(float(last_diag.get("mass_total", -1.0) if last_diag else -1.0)),
            }
            self._log("Run complete." if not self._cancel_requested else "Run canceled by user.")
            self._log(
                f"Depth range: {float(np.min(h)):.6f} .. {float(np.max(h)):.6f} | "
                f"Velocity mag max: {float(np.max(np.sqrt((hu / np.maximum(h, 1e-12)) ** 2 + (hv / np.maximum(h, 1e-12)) ** 2))):.6f}"
            )
            self._refresh_plot()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "2D SWE", f"Run failed: {exc}")
            self._log(f"Error: {exc}")
        finally:
            try:
                if backend is not None:
                    backend.destroy()
            except Exception:
                pass
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def _refresh_plot(self):
        if not self._have_mpl or self._mesh_data is None:
            return

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        triangles = self._mesh_data["cell_nodes"].reshape((-1, 3))

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        tri = self._mtri.Triangulation(node_x, node_y, triangles)

        mode = self.view_mode_combo.currentText()
        if mode == "Mesh" or self._result_data is None:
            ax.triplot(tri, color="black", linewidth=0.3)
            ax.set_title("Generated mesh")
        elif mode == "Depth":
            vals = self._result_data["h"]
            tpc = ax.tripcolor(tri, facecolors=vals, cmap="viridis", edgecolors="none")
            self._fig.colorbar(tpc, ax=ax, label="Depth")
            ax.set_title("Final depth")
        else:
            h = np.maximum(self._result_data["h"], 1.0e-12)
            hu = self._result_data["hu"]
            hv = self._result_data["hv"]
            vals = np.sqrt((hu / h) ** 2 + (hv / h) ** 2)
            tpc = ax.tripcolor(tri, facecolors=vals, cmap="plasma", edgecolors="none")
            self._fig.colorbar(tpc, ax=ax, label="Velocity magnitude")
            ax.set_title("Final velocity magnitude")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        self._canvas.draw_idle()


def launch_swe2d_workbench(parent=None):
    dlg = SWE2DWorkbenchDialog(parent)

    def _cleanup():
        try:
            _SWE2D_WORKBENCH_WINDOWS.remove(dlg)
        except ValueError:
            pass

    _SWE2D_WORKBENCH_WINDOWS.append(dlg)
    dlg.finished.connect(_cleanup)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
