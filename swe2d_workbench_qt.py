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
import datetime
import json
import math
import os
import sqlite3
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
    from swe2d_coupling import SWE2DCouplingController, pack_coupling_soa
    from swe2d_drainage_network import SWE2DUrbanDrainageModule
    from swe2d_extensions import (
        DrainageLink,
        DrainageNode,
        HydraulicStructure,
        HydraulicStructureConfig,
        InletExchange,
        PipeNetworkConfig,
        StructureType,
    )
    from swe2d_structures import SWE2DStructureModule
except Exception:
    try:
        from .swe2d_coupling import SWE2DCouplingController, pack_coupling_soa
        from .swe2d_drainage_network import SWE2DUrbanDrainageModule
        from .swe2d_extensions import (
            DrainageLink,
            DrainageNode,
            HydraulicStructure,
            HydraulicStructureConfig,
            InletExchange,
            PipeNetworkConfig,
            StructureType,
        )
        from .swe2d_structures import SWE2DStructureModule
    except Exception:
        SWE2DCouplingController = None
        pack_coupling_soa = None
        SWE2DUrbanDrainageModule = None
        SWE2DStructureModule = None
        DrainageLink = DrainageNode = HydraulicStructure = InletExchange = None
        PipeNetworkConfig = HydraulicStructureConfig = None
        StructureType = None

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

try:
    from rainfall_hydrology import (
        Gauge,
        ThiessenRainCNForcing,
        build_hyetograph,
        assign_cells_to_nearest_gauge,
        runoff_depth_mm_from_event_rain_mm,
        time_of_concentration_hours_velocity_method,
    )
except Exception:
    try:
        from .rainfall_hydrology import (
            Gauge,
            ThiessenRainCNForcing,
            build_hyetograph,
            assign_cells_to_nearest_gauge,
            runoff_depth_mm_from_event_rain_mm,
            time_of_concentration_hours_velocity_method,
        )
    except Exception:
        Gauge = ThiessenRainCNForcing = None
        build_hyetograph = assign_cells_to_nearest_gauge = None
        runoff_depth_mm_from_event_rain_mm = time_of_concentration_hours_velocity_method = None


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

_STRUCTURE_TYPE_VALUE_MAP = {
    "Weir": 1,
    "Culvert": 2,
    "Gate": 3,
    "Bridge": 4,
    "Pump": 5,
}

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

_MODEL_LAYER_BINDINGS = {
    "drainage_nodes": {
        "layer_name": "swe2d_drainage_nodes",
        "combo_attr": "drain_nodes_layer_combo",
        "geometry": "point",
        "required_fields": ("node_id", "invert_elev", "max_depth"),
    },
    "drainage_links": {
        "layer_name": "swe2d_drainage_links",
        "combo_attr": "drain_links_layer_combo",
        "geometry": "line",
        "required_fields": ("link_id", "from_node", "to_node"),
    },
    "drainage_inlets": {
        "layer_name": "swe2d_drainage_inlets",
        "combo_attr": "drain_inlets_layer_combo",
        "geometry": "point",
        "required_fields": ("inlet_id", "node_id", "crest_elev", "width_m", "coefficient"),
    },
    "hydraulic_structures": {
        "layer_name": "swe2d_structures",
        "combo_attr": "structures_layer_combo",
        "geometry": "line",
        "required_fields": ("structure_id", "structure_type", "crest_elev", "enabled"),
    },
}


def _run_topology_mesh_job(conceptual, backend_name: str, options: Optional[Dict[str, object]] = None):
    """Run heavy topology meshing work off the GUI thread/process."""
    # Use the already-imported function when available; fall back to local import
    # in subprocess contexts.
    gen = generate_face_centric_mesh
    if gen is None:
        try:
            from swe2d_meshing import generate_face_centric_mesh as gen  # type: ignore
        except Exception:
            from .swe2d_meshing import generate_face_centric_mesh as gen  # type: ignore
    return gen(conceptual, backend=backend_name, options=options)


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


class TopologyAttributeTableDialog(QtWidgets.QDialog):
    def __init__(self, layer, title: str, field_specs, sort_fields=None, note: str = "", parent=None):
        super().__init__(parent)
        self.layer = layer
        self.field_specs = list(field_specs)
        self.sort_fields = list(sort_fields or [])
        self._row_feature_ids: List[int] = []

        self.setWindowTitle(title)
        self.resize(920, 440)

        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            note
            or "Edit topology attributes here. Geometry remains edited in the map canvas or native QGIS layer tools."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, len(self.field_specs))
        self.table.setHorizontalHeaderLabels([spec[1] for spec in self.field_specs])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row_btns = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Reload From Layer")
        self.add_row_btn = QtWidgets.QPushButton("Add Row")
        self.remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        self.refresh_btn.clicked.connect(self._load_rows)
        self.add_row_btn.clicked.connect(self._add_blank_row)
        self.remove_row_btn.clicked.connect(self._remove_selected_rows)
        row_btns.addWidget(self.refresh_btn)
        row_btns.addWidget(self.add_row_btn)
        row_btns.addWidget(self.remove_row_btn)
        row_btns.addStretch(1)
        root.addLayout(row_btns)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_rows()

    def _sorted_features(self):
        feats = list(self.layer.getFeatures()) if self.layer is not None else []
        if not self.sort_fields:
            return feats

        def _key(ft):
            vals = []
            for name in self.sort_fields:
                try:
                    value = ft[name]
                except Exception:
                    value = None
                vals.append((value is None, value))
            return vals

        feats.sort(key=_key)
        return feats

    def _set_editor(self, row: int, col: int, value, spec):
        kind = spec[2]
        if kind == "enum":
            combo = QtWidgets.QComboBox()
            for option in spec[3]:
                combo.addItem(str(option), str(option))
            text = str(value or "").strip().lower()
            idx = combo.findData(text)
            combo.setCurrentIndex(max(0, idx))
            self.table.setCellWidget(row, col, combo)
            return
        item = QtWidgets.QTableWidgetItem("" if value in (None, "") else str(value))
        self.table.setItem(row, col, item)

    def _editor_value(self, row: int, col: int, spec):
        kind = spec[2]
        if kind == "enum":
            combo = self.table.cellWidget(row, col)
            return None if combo is None else combo.currentData()
        item = self.table.item(row, col)
        text = "" if item is None else str(item.text()).strip()
        if text == "":
            return None
        if kind == "int":
            return int(round(float(text)))
        if kind == "float":
            return float(text)
        return text

    def _load_rows(self):
        self.table.setRowCount(0)
        self._row_feature_ids = []
        for ft in self._sorted_features():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._row_feature_ids.append(int(ft.id()))
            for col, spec in enumerate(self.field_specs):
                field_name = spec[0]
                try:
                    value = ft[field_name]
                except Exception:
                    value = None
                self._set_editor(row, col, value, spec)

    def _add_blank_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._row_feature_ids.append(-1)
        for col, spec in enumerate(self.field_specs):
            self._set_editor(row, col, "", spec)

    def _remove_selected_rows(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            if 0 <= row < len(self._row_feature_ids):
                self._row_feature_ids.pop(row)

    def _save_and_accept(self):
        if self.layer is None:
            self.accept()
            return
        started_here = False
        try:
            if not self.layer.isEditable():
                if not self.layer.startEditing():
                    raise RuntimeError("Could not start layer editing session.")
                started_here = True

            field_idx = {spec[0]: self.layer.fields().indexOf(spec[0]) for spec in self.field_specs}
            provider = self.layer.dataProvider()

            for row, fid in enumerate(self._row_feature_ids):
                if fid < 0:
                    feat = QgsFeature(self.layer.fields())
                    if not provider.addFeatures([feat]):
                        raise RuntimeError("Failed to add new feature row to layer.")
                    fids = [f.id() for f in self.layer.getFeatures()]
                    fid = int(fids[-1]) if fids else -1
                    self._row_feature_ids[row] = fid
                for col, spec in enumerate(self.field_specs):
                    idx = field_idx.get(spec[0], -1)
                    if idx < 0:
                        continue
                    value = self._editor_value(row, col, spec)
                    self.layer.changeAttributeValue(fid, idx, value)

            if started_here and not self.layer.commitChanges():
                raise RuntimeError("Layer changes could not be committed.")
            self.layer.triggerRepaint()
            self.accept()
        except Exception as exc:
            if started_here:
                try:
                    self.layer.rollBack()
                except Exception:
                    pass
            QtWidgets.QMessageBox.warning(self, "Topology Editor", f"Failed to save layer edits: {exc}")


class SWE2DLineResultsViewerDialog(QtWidgets.QDialog):
    """Viewer for sampled SWE2D line results stored in GeoPackage/SQLite."""

    _BASE_COLUMNS = [
        ("t_s", "Time (s)"),
        ("line_id", "Line ID"),
        ("line_name", "Line Name"),
        ("depth_m", "Depth ({L})"),
        ("velocity_ms", "Velocity ({L}/s)"),
        ("wse_m", "Water Surface ({L})"),
        ("bed_m", "Bed ({L})"),
        ("flow_cms", "Flow ({Q})"),
    ]

    _PLOT_OPTIONS = [
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Water Surface", "wse_m"),
        ("Bed", "bed_m"),
        ("Flow", "flow_cms"),
    ]

    _PROFILE_OPTIONS = [
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Water Surface", "wse_m"),
        ("Bed", "bed_m"),
        ("Normal Flow", "flow_qn"),
        ("Froude", "fr"),
    ]

    _PROFILE_FILL_OPTIONS = [
        ("None", "none"),
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Froude", "fr"),
        ("Normal Flow", "flow_qn"),
    ]

    def __init__(
        self,
        ts_records: List[Dict[str, object]],
        profile_records: List[Dict[str, object]],
        run_id: str,
        db_path: str,
        length_unit: str = "m",
        flow_unit_label: str = "m3/s",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("2D Sample Line Results Viewer")
        self.resize(980, 620)

        self._ts_records = list(ts_records)
        self._profile_records = list(profile_records)
        self._run_id = str(run_id)
        self._db_path = str(db_path)
        l_unit = str(length_unit or "m")
        q_unit = str(flow_unit_label or "m3/s")
        self._columns = [(k, lbl.format(L=l_unit, Q=q_unit)) for k, lbl in self._BASE_COLUMNS]
        self._plot_canvas = None
        self._plot_fig = None

        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            f"Run ID: {self._run_id}\nSource: {self._db_path}"
        )
        header.setWordWrap(True)
        root.addWidget(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Line:"))
        self.line_combo = QtWidgets.QComboBox()
        controls.addWidget(self.line_combo)
        controls.addWidget(QtWidgets.QLabel("View:"))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItem("Time series", "time")
        self.view_mode_combo.addItem("Profile at timestep", "profile")
        self.view_mode_combo.addItem("WSE + Bed profile", "wse_bed")
        controls.addWidget(self.view_mode_combo)
        controls.addWidget(QtWidgets.QLabel("Variable:"))
        self.metric_combo = QtWidgets.QComboBox()
        for label, key in self._PLOT_OPTIONS:
            self.metric_combo.addItem(label, key)
        controls.addWidget(self.metric_combo)
        controls.addWidget(QtWidgets.QLabel("Profile variable:"))
        self.profile_metric_combo = QtWidgets.QComboBox()
        for label, key in self._PROFILE_OPTIONS:
            self.profile_metric_combo.addItem(label, key)
        controls.addWidget(self.profile_metric_combo)
        controls.addWidget(QtWidgets.QLabel("Timestep:"))
        self.time_combo = QtWidgets.QComboBox()
        controls.addWidget(self.time_combo)
        controls.addWidget(QtWidgets.QLabel("Fill by:"))
        self.fill_metric_combo = QtWidgets.QComboBox()
        for label, key in self._PROFILE_FILL_OPTIONS:
            self.fill_metric_combo.addItem(label, key)
        controls.addWidget(self.fill_metric_combo)
        controls.addStretch(1)
        root.addLayout(controls)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in self._columns])
        self.table.horizontalHeader().setStretchLastSection(True)

        self._have_mpl = False
        FigureCanvas, Figure, _ = _try_import_matplotlib_qt()
        if FigureCanvas is not None and Figure is not None:
            self._have_mpl = True
            self._plot_fig = Figure(figsize=(6.8, 3.0), tight_layout=True)
            self._plot_canvas = FigureCanvas(self._plot_fig)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        table_host = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_host)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.table)
        split.addWidget(table_host)

        plot_host = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        if self._have_mpl:
            plot_layout.addWidget(self._plot_canvas)
        else:
            note = QtWidgets.QLabel("Matplotlib backend unavailable; table view only.")
            note.setWordWrap(True)
            plot_layout.addWidget(note)
        split.addWidget(plot_host)
        split.setSizes([380, 220])
        root.addWidget(split, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_line_combo()
        self._populate_time_combo()
        self._sync_control_visibility()
        self._refresh_table()
        self._refresh_plot()

        self.line_combo.currentIndexChanged.connect(self._refresh_table)
        self.line_combo.currentIndexChanged.connect(self._refresh_plot)
        self.metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.profile_metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.time_combo.currentIndexChanged.connect(self._refresh_table)
        self.time_combo.currentIndexChanged.connect(self._refresh_plot)
        self.fill_metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.view_mode_combo.currentIndexChanged.connect(self._sync_control_visibility)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_table)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)

    def _line_filter(self):
        value = self.line_combo.currentData()
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _selected_time(self) -> Optional[float]:
        value = self.time_combo.currentData()
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _populate_line_combo(self):
        self.line_combo.clear()
        self.line_combo.addItem("All lines", None)
        by_line: Dict[int, str] = {}
        for rec in (self._ts_records + self._profile_records):
            try:
                lid = int(rec.get("line_id", -1))
            except Exception:
                continue
            lname = str(rec.get("line_name", "") or "")
            if lid not in by_line:
                by_line[lid] = lname
        for lid in sorted(by_line.keys()):
            label = f"{lid}"
            if by_line[lid]:
                label += f" - {by_line[lid]}"
            self.line_combo.addItem(label, lid)

    def _populate_time_combo(self):
        self.time_combo.clear()
        ts_vals = sorted({float(r.get("t_s", 0.0)) for r in self._profile_records})
        if not ts_vals:
            ts_vals = sorted({float(r.get("t_s", 0.0)) for r in self._ts_records})
        for t_s in ts_vals:
            self.time_combo.addItem(f"{t_s / 3600.0:.4f} hr", float(t_s))

    def _sync_control_visibility(self):
        mode = str(self.view_mode_combo.currentData())
        is_time = (mode == "time")
        is_profile = (mode == "profile")
        is_wse = (mode == "wse_bed")
        self.metric_combo.setVisible(is_time)
        self.profile_metric_combo.setVisible(is_profile)
        self.time_combo.setVisible(is_profile or is_wse)
        self.fill_metric_combo.setVisible(is_wse)

    def _filtered_records(self) -> List[Dict[str, object]]:
        lid = self._line_filter()
        if lid is None:
            return list(self._ts_records)
        out = []
        for rec in self._ts_records:
            try:
                if int(rec.get("line_id", -1)) == lid:
                    out.append(rec)
            except Exception:
                continue
        return out

    def _filtered_profile_records(self) -> List[Dict[str, object]]:
        lid = self._line_filter()
        t_sel = self._selected_time()
        out = []
        for rec in self._profile_records:
            try:
                if lid is not None and int(rec.get("line_id", -1)) != lid:
                    continue
                if t_sel is not None and abs(float(rec.get("t_s", 0.0)) - t_sel) > 1.0e-9:
                    continue
            except Exception:
                continue
            out.append(rec)
        return out

    def _refresh_table(self):
        mode = str(self.view_mode_combo.currentData())
        if mode == "time":
            rows = self._filtered_records()
            rows.sort(key=lambda r: (float(r.get("t_s", 0.0)), int(r.get("line_id", -1))))
            self.table.setColumnCount(len(self._columns))
            self.table.setHorizontalHeaderLabels([lbl for _, lbl in self._columns])
            self.table.setRowCount(len(rows))
            for r, rec in enumerate(rows):
                for c, (key, _) in enumerate(self._columns):
                    val = rec.get(key)
                    txt = f"{val:.6f}" if isinstance(val, float) else str(val)
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
            return

        rows = self._filtered_profile_records()
        rows.sort(key=lambda r: float(r.get("station_m", 0.0)))
        cols = [
            ("t_s", "Time (s)"),
            ("line_id", "Line ID"),
            ("line_name", "Line Name"),
            ("station_m", "Station ({})".format(self._columns[3][1].split("(")[-1].rstrip(")"))),
            ("depth_m", self._columns[3][1]),
            ("velocity_ms", self._columns[4][1]),
            ("wse_m", self._columns[5][1]),
            ("bed_m", self._columns[6][1]),
            ("flow_qn", "Normal Flow Density"),
            ("fr", "Froude"),
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in cols])
        self.table.setRowCount(len(rows))
        for r, rec in enumerate(rows):
            for c, (key, _) in enumerate(cols):
                val = rec.get(key)
                txt = f"{val:.6f}" if isinstance(val, float) else str(val)
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

    def _refresh_plot(self):
        if not self._have_mpl or self._plot_fig is None or self._plot_canvas is None:
            return
        mode = str(self.view_mode_combo.currentData())
        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        if mode == "time":
            rows = self._filtered_records()
            metric = str(self.metric_combo.currentData())
            if not rows:
                ax.text(0.5, 0.5, "No sampled line results", ha="center", va="center", transform=ax.transAxes)
                self._plot_canvas.draw_idle()
                return
            by_line: Dict[int, List[Tuple[float, float]]] = {}
            name_by_line: Dict[int, str] = {}
            for rec in rows:
                try:
                    lid = int(rec.get("line_id", -1))
                    ts = float(rec.get("t_s", 0.0))
                    vv = float(rec.get(metric, float("nan")))
                except Exception:
                    continue
                if not np.isfinite(vv):
                    continue
                by_line.setdefault(lid, []).append((ts, vv))
                name_by_line[lid] = str(rec.get("line_name", "") or "")
            if not by_line:
                ax.text(0.5, 0.5, "No numeric values to plot", ha="center", va="center", transform=ax.transAxes)
                self._plot_canvas.draw_idle()
                return
            for lid in sorted(by_line.keys()):
                pairs = sorted(by_line[lid], key=lambda x: x[0])
                t_hr = np.asarray([p[0] / 3600.0 for p in pairs], dtype=np.float64)
                vals = np.asarray([p[1] for p in pairs], dtype=np.float64)
                label = f"Line {lid}"
                if name_by_line.get(lid):
                    label += f" ({name_by_line[lid]})"
                ax.plot(t_hr, vals, "-", linewidth=1.8, label=label)
            ax.set_xlabel("Time (hr)")
            ax.set_ylabel(self.metric_combo.currentText())
            ax.set_title("Sample line time series")
            if len(by_line) > 1:
                ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            self._plot_canvas.draw_idle()
            return

        rows = self._filtered_profile_records()
        if not rows:
            ax.text(0.5, 0.5, "No profile records for selected line/timestep", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return
        rows = sorted(rows, key=lambda r: float(r.get("station_m", 0.0)))
        x = np.asarray([float(r.get("station_m", 0.0)) for r in rows], dtype=np.float64)
        line_name = str(rows[0].get("line_name", "") or "")
        line_id = int(rows[0].get("line_id", -1))
        t_s = float(rows[0].get("t_s", 0.0))

        if mode == "profile":
            metric = str(self.profile_metric_combo.currentData())
            y = np.asarray([float(r.get(metric, float("nan"))) for r in rows], dtype=np.float64)
            ok = np.isfinite(y)
            if np.any(ok):
                ax.plot(x[ok], y[ok], "-", linewidth=1.8)
            ax.set_xlabel("Station")
            ax.set_ylabel(self.profile_metric_combo.currentText())
            ax.set_title(f"Line {line_id} profile at t={t_s/3600.0:.4f} hr" + (f" ({line_name})" if line_name else ""))
            ax.grid(True, alpha=0.3)
            self._plot_canvas.draw_idle()
            return

        # WSE + bed profile, with optional color-ramped fill between bed and WSE.
        wse = np.asarray([float(r.get("wse_m", float("nan"))) for r in rows], dtype=np.float64)
        bed = np.asarray([float(r.get("bed_m", float("nan"))) for r in rows], dtype=np.float64)
        ok = np.isfinite(wse) & np.isfinite(bed)
        if not np.any(ok):
            ax.text(0.5, 0.5, "No WSE/bed values for selected line/timestep", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        x_ok = x[ok]
        wse_ok = wse[ok]
        bed_ok = bed[ok]
        fill_key = str(self.fill_metric_combo.currentData())

        if fill_key != "none":
            try:
                from matplotlib import cm as mpl_cm, colors as mpl_colors
                fill_vals = np.asarray([float(r.get(fill_key, float("nan"))) for r in rows], dtype=np.float64)[ok]
                finite = np.isfinite(fill_vals)
                if np.any(finite):
                    vmin = float(np.nanmin(fill_vals[finite]))
                    vmax = float(np.nanmax(fill_vals[finite]))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
                    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
                    cmap = mpl_cm.get_cmap("viridis")
                    for i in range(len(x_ok) - 1):
                        if not (np.isfinite(fill_vals[i]) and np.isfinite(fill_vals[i + 1])):
                            continue
                        c_mid = cmap(norm(0.5 * (fill_vals[i] + fill_vals[i + 1])))
                        ax.fill_between(
                            x_ok[i : i + 2],
                            bed_ok[i : i + 2],
                            wse_ok[i : i + 2],
                            color=c_mid,
                            alpha=0.85,
                            linewidth=0.0,
                        )
                    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    self._plot_fig.colorbar(sm, ax=ax, label=self.fill_metric_combo.currentText())
            except Exception:
                ax.fill_between(x_ok, bed_ok, wse_ok, color="tab:blue", alpha=0.18)
        else:
            ax.fill_between(x_ok, bed_ok, wse_ok, color="tab:blue", alpha=0.18)

        ax.plot(x_ok, bed_ok, "-", color="saddlebrown", linewidth=1.6, label="Bed")
        ax.plot(x_ok, wse_ok, "-", color="royalblue", linewidth=1.8, label="Water Surface")
        ax.set_xlabel("Station")
        ax.set_ylabel("Elevation")
        ax.set_title(f"Line {line_id} WSE + bed at t={t_s/3600.0:.4f} hr" + (f" ({line_name})" if line_name else ""))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()


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
        self._line_snapshot_rows: List[Dict[str, object]] = []
        self._line_snapshot_profile_rows: List[Dict[str, object]] = []
        self._line_results_latest_run_id: str = ""
        self._line_results_latest_db_path: str = ""
        self._model_gpkg_path: str = ""
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
        self._topology_mesh_options: Dict[str, object] = {}
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
        self.cn_layer_combo = QtWidgets.QComboBox()
        self.rain_gage_layer_combo = QtWidgets.QComboBox()
        self.hyetograph_layer_combo = QtWidgets.QComboBox()
        self.sample_lines_layer_combo = QtWidgets.QComboBox()
        self.drain_nodes_layer_combo = QtWidgets.QComboBox()
        self.drain_nodes_layer_combo.addItem("(none)", None)
        self.drain_links_layer_combo = QtWidgets.QComboBox()
        self.drain_links_layer_combo.addItem("(none)", None)
        self.drain_inlets_layer_combo = QtWidgets.QComboBox()
        self.drain_inlets_layer_combo.addItem("(none)", None)
        self.structures_layer_combo = QtWidgets.QComboBox()
        self.structures_layer_combo.addItem("(none)", None)
        self.refresh_layers_btn = QtWidgets.QPushButton("Refresh Layers")
        self.refresh_layers_btn.clicked.connect(self._refresh_layer_combos)
        self.create_model_gpkg_btn = QtWidgets.QPushButton("Create 2D Model GeoPackage")
        self.create_model_gpkg_btn.clicked.connect(self._create_2d_model_geopackage)
        self.create_lumped_gpkg_btn = QtWidgets.QPushButton("Create Lumped Hydro GeoPackage")
        self.create_lumped_gpkg_btn.clicked.connect(self._create_lumped_hydrology_geopackage)
        self.load_model_gpkg_btn = QtWidgets.QPushButton("Load 2D Model GeoPackage")
        self.load_model_gpkg_btn.clicked.connect(self._load_2d_model_geopackage)
        self.preview_coupling_btn = QtWidgets.QPushButton("Preview Drainage/Structure Coupling")
        self.preview_coupling_btn.clicked.connect(self._preview_coupling_configuration)
        self.export_mesh_layers_btn = QtWidgets.QPushButton("Export Mesh To Map Layers")
        self.export_mesh_layers_btn.clicked.connect(self._export_mesh_to_layers)
        self.save_hdf5_btn = QtWidgets.QPushButton("Save Mesh To HEC-RAS HDF5")
        self.save_hdf5_btn.clicked.connect(self._export_mesh_to_hdf5)
        self.save_results_hdf5_btn = QtWidgets.QPushButton("Save Results To HEC-RAS HDF5")
        self.save_results_hdf5_btn.clicked.connect(self._export_results_to_hdf5)
        self.save_results_ugrid_btn = QtWidgets.QPushButton("Save Results To UGRID NetCDF")
        self.save_results_ugrid_btn.clicked.connect(self._export_results_to_ugrid)
        self.extended_outputs_chk = QtWidgets.QCheckBox("Include extended outputs (momentum, qmag, wet mask, Fr, Manning)")
        self.extended_outputs_chk.setChecked(True)
        self.open_results_viewer_btn = QtWidgets.QPushButton("Open 2D Results Viewer")
        self.open_results_viewer_btn.clicked.connect(self._open_line_results_viewer)
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
        map_layout.addWidget(QtWidgets.QLabel("CN polygons:"), 4, 0)
        map_layout.addWidget(self.cn_layer_combo, 4, 1)
        map_layout.addWidget(QtWidgets.QLabel("Rain gages (points):"), 5, 0)
        map_layout.addWidget(self.rain_gage_layer_combo, 5, 1)
        map_layout.addWidget(QtWidgets.QLabel("Rain hyetographs (table):"), 6, 0)
        map_layout.addWidget(self.hyetograph_layer_combo, 6, 1)
        map_layout.addWidget(QtWidgets.QLabel("Sample lines layer:"), 7, 0)
        map_layout.addWidget(self.sample_lines_layer_combo, 7, 1)
        map_layout.addWidget(QtWidgets.QLabel("Drainage nodes layer:"), 8, 0)
        map_layout.addWidget(self.drain_nodes_layer_combo, 8, 1)
        map_layout.addWidget(QtWidgets.QLabel("Drainage links layer:"), 9, 0)
        map_layout.addWidget(self.drain_links_layer_combo, 9, 1)
        map_layout.addWidget(QtWidgets.QLabel("Drainage inlets layer:"), 10, 0)
        map_layout.addWidget(self.drain_inlets_layer_combo, 10, 1)
        map_layout.addWidget(QtWidgets.QLabel("Hydraulic structures layer:"), 11, 0)
        map_layout.addWidget(self.structures_layer_combo, 11, 1)
        map_layout.addWidget(self.refresh_layers_btn, 12, 0)
        map_layout.addWidget(self.create_model_gpkg_btn, 12, 1)
        map_layout.addWidget(self.create_lumped_gpkg_btn, 13, 0, 1, 2)
        map_layout.addWidget(self.load_model_gpkg_btn, 14, 0, 1, 2)
        map_layout.addWidget(self.preview_coupling_btn, 15, 0, 1, 2)
        map_layout.addWidget(self.export_mesh_layers_btn, 16, 0)
        map_layout.addWidget(self.save_hdf5_btn, 16, 1)
        map_layout.addWidget(self.save_results_hdf5_btn, 17, 0, 1, 2)
        map_layout.addWidget(self.save_results_ugrid_btn, 18, 0, 1, 2)
        map_layout.addWidget(self.extended_outputs_chk, 19, 0, 1, 2)
        map_layout.addWidget(self.open_results_viewer_btn, 20, 0, 1, 2)
        map_layout.addWidget(self.import_mesh_layers_btn, 21, 0, 1, 2)
        map_layout.addWidget(self.terrain_to_nodes_btn, 22, 0, 1, 2)
        map_layout.addWidget(self.pull_node_z_btn, 23, 0, 1, 2)
        map_layout.addWidget(self.layer_status_lbl, 24, 0, 1, 2)
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
        self.topo_quality_min_angle_spin = QtWidgets.QDoubleSpinBox()
        self.topo_quality_min_angle_spin.setRange(0.0, 89.0)
        self.topo_quality_min_angle_spin.setDecimals(1)
        self.topo_quality_min_angle_spin.setValue(5.0)
        self.topo_quality_max_aspect_spin = QtWidgets.QDoubleSpinBox()
        self.topo_quality_max_aspect_spin.setRange(1.0, 1.0e4)
        self.topo_quality_max_aspect_spin.setDecimals(2)
        self.topo_quality_max_aspect_spin.setValue(20.0)
        self.topo_quality_min_area_edit = QtWidgets.QLineEdit("1e-14")
        self.topo_quality_strict_chk = QtWidgets.QCheckBox("Strict quality acceptance")
        self.topo_quality_size_scales_edit = QtWidgets.QLineEdit("1.0")
        self.topo_quality_smooth_increments_edit = QtWidgets.QLineEdit("0")
        self.topo_gmsh_tri_algo_combo = QtWidgets.QComboBox()
        self.topo_gmsh_tri_algo_combo.addItem("Frontal-Delaunay (quality)", 6)
        self.topo_gmsh_tri_algo_combo.addItem("Delaunay (faster)", 5)
        self.topo_gmsh_quad_algo_combo = QtWidgets.QComboBox()
        self.topo_gmsh_quad_algo_combo.addItem("Frontal + Blossom recombine", 6)
        self.topo_gmsh_quad_algo_combo.addItem("Delaunay + Blossom recombine", 5)
        self.topo_gmsh_quad_algo_combo.addItem("Packing of Parallelograms", 9)
        self.topo_gmsh_smoothing_spin = QtWidgets.QSpinBox()
        self.topo_gmsh_smoothing_spin.setRange(0, 100)
        self.topo_gmsh_smoothing_spin.setValue(5)
        self.topo_gmsh_optimize_iters_spin = QtWidgets.QSpinBox()
        self.topo_gmsh_optimize_iters_spin.setRange(0, 100)
        self.topo_gmsh_optimize_iters_spin.setValue(3)
        self.topo_gmsh_recombine_algo_combo = QtWidgets.QComboBox()
        self.topo_gmsh_recombine_algo_combo.addItem("Simple", 0)
        self.topo_gmsh_recombine_algo_combo.addItem("Blossom", 1)
        self.topo_gmsh_recombine_algo_combo.addItem("Simple full-quad", 2)
        self.topo_gmsh_optimize_netgen_chk = QtWidgets.QCheckBox("Enable Netgen optimize")
        self.topo_gmsh_verbosity_spin = QtWidgets.QSpinBox()
        self.topo_gmsh_verbosity_spin.setRange(0, 10)
        self.topo_gmsh_verbosity_spin.setValue(1)
        gmsh_form_widget = QtWidgets.QWidget()
        gmsh_form = QtWidgets.QFormLayout(gmsh_form_widget)
        gmsh_form.setContentsMargins(0, 0, 0, 0)
        gmsh_form.addRow("Triangle algorithm:", self.topo_gmsh_tri_algo_combo)
        gmsh_form.addRow("Quadrilateral algorithm:", self.topo_gmsh_quad_algo_combo)
        gmsh_form.addRow("Recombine algorithm:", self.topo_gmsh_recombine_algo_combo)
        gmsh_form.addRow("Smoothing passes:", self.topo_gmsh_smoothing_spin)
        gmsh_form.addRow("Optimize iterations:", self.topo_gmsh_optimize_iters_spin)
        gmsh_form.addRow("Verbosity:", self.topo_gmsh_verbosity_spin)
        gmsh_form.addRow(self.topo_gmsh_optimize_netgen_chk)
        quality_form_widget = QtWidgets.QWidget()
        quality_form = QtWidgets.QFormLayout(quality_form_widget)
        quality_form.setContentsMargins(0, 0, 0, 0)
        quality_form.addRow("Min angle (deg):", self.topo_quality_min_angle_spin)
        quality_form.addRow("Max aspect ratio:", self.topo_quality_max_aspect_spin)
        quality_form.addRow("Min area / bbox area:", self.topo_quality_min_area_edit)
        quality_form.addRow("Retry size scales:", self.topo_quality_size_scales_edit)
        quality_form.addRow("Retry smooth increments:", self.topo_quality_smooth_increments_edit)
        quality_form.addRow(self.topo_quality_strict_chk)
        self.topo_validate_btn = QtWidgets.QPushButton("Summarize Layer Controls")
        self.topo_validate_btn.clicked.connect(self._update_topology_control_summary)
        self.topo_edit_regions_btn = QtWidgets.QPushButton("Edit Region Controls")
        self.topo_edit_regions_btn.clicked.connect(self._open_topology_region_table)
        self.topo_edit_quad_edges_btn = QtWidgets.QPushButton("Edit Transition Layers")
        self.topo_edit_quad_edges_btn.clicked.connect(self._open_topology_quad_edge_table)
        self.topo_controls_summary_lbl = QtWidgets.QLabel(
            "Topology-layer controls: use multiple region polygons for multiple blocks. "
            "Use region target_size + cell_type, edge_len_1..4 for cartesian/quadrilateral block spacing, "
            "and quad-edge n_layers / first_height / growth_rate for TQMesh transition layers."
        )
        self.topo_controls_summary_lbl.setWordWrap(True)
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
        topo_layout.addWidget(QtWidgets.QLabel("Quad edges / transition layers:"), 4, 0)
        topo_layout.addWidget(self.topo_quad_edges_combo, 4, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Meshing backend:"), 5, 0)
        topo_layout.addWidget(self.topo_backend_combo, 5, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Default target size:"), 6, 0)
        topo_layout.addWidget(self.topo_default_size_spin, 6, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Default cell type:"), 7, 0)
        topo_layout.addWidget(self.topo_default_cell_type_combo, 7, 1)
        topo_layout.addWidget(QtWidgets.QLabel("Gmsh advanced controls:"), 8, 0)
        topo_layout.addWidget(gmsh_form_widget, 8, 1)
        topo_layout.addWidget(QtWidgets.QLabel("TQMesh quality controls:"), 9, 0)
        topo_layout.addWidget(quality_form_widget, 9, 1)
        topo_layout.addWidget(self.topo_validate_btn, 10, 0, 1, 2)
        topo_layout.addWidget(self.topo_edit_regions_btn, 11, 0)
        topo_layout.addWidget(self.topo_edit_quad_edges_btn, 11, 1)
        topo_layout.addWidget(self.topo_controls_summary_lbl, 12, 0, 1, 2)
        topo_layout.addWidget(self.topo_export_template_btn, 13, 0, 1, 2)
        topo_layout.addWidget(self.topo_generate_btn, 14, 0, 1, 2)
        topo_layout.addWidget(self.topo_status_lbl, 15, 0, 1, 2)
        left_layout.addWidget(topo_group)

        self.topo_backend_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_regions_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_constraints_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_quad_edges_combo.currentIndexChanged.connect(self._update_topology_control_summary)

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
        self.initial_condition_combo = QtWidgets.QComboBox()
        self.initial_condition_combo.addItem("Dry start", "dry")
        self.initial_condition_combo.addItem("Uniform depth", "uniform_depth")
        self.initial_condition_combo.addItem("Uniform water surface elevation", "uniform_wse")
        self.initial_condition_combo.setCurrentIndex(0)
        self.initial_condition_combo.setToolTip(
            "Initial condition source used at run start.\n"
            "Dry start: h=0.\n"
            "Uniform depth: constant initial depth everywhere.\n"
            "Uniform WSE: depth = max(0, WSE - local bed)."
        )
        self.initial_depth_spin = QtWidgets.QDoubleSpinBox()
        self.initial_depth_spin.setRange(0.0, 1.0e6)
        self.initial_depth_spin.setDecimals(4)
        self.initial_depth_spin.setValue(0.0)
        self.initial_wse_spin = QtWidgets.QDoubleSpinBox()
        self.initial_wse_spin.setRange(-1.0e6, 1.0e6)
        self.initial_wse_spin.setDecimals(4)
        self.initial_wse_spin.setValue(0.0)
        self.adaptive_cfl_dt_chk = QtWidgets.QCheckBox("Enable variable timestep (CFL)")
        self.adaptive_cfl_dt_chk.setChecked(False)
        self.adaptive_cfl_dt_chk.setToolTip(
            "If enabled, runtime dt is selected from CFL each step.\n"
            "The dt field is used as dt_max (upper bound).\n"
            "If disabled, dt is fixed each step."
        )
        self.max_rel_depth_increase_spin = QtWidgets.QDoubleSpinBox()
        self.max_rel_depth_increase_spin.setRange(0.0, 1000.0)
        self.max_rel_depth_increase_spin.setDecimals(3)
        self.max_rel_depth_increase_spin.setValue(2.0)
        self.max_rel_depth_increase_spin.setToolTip(
            "Per-step depth growth limiter on GPU update:\n"
            "h_new <= h_old + factor * max(h_old, h_min).\n"
            "Lower values are more robust near advancing wet/dry fronts."
        )
        self.shallow_damping_depth_spin = QtWidgets.QDoubleSpinBox()
        self.shallow_damping_depth_spin.setRange(1.0e-8, 10.0)
        self.shallow_damping_depth_spin.setDecimals(6)
        self.shallow_damping_depth_spin.setValue(1.0e-4)
        self.shallow_damping_depth_spin.setToolTip(
            "Depth threshold for smooth momentum damping in shallow cells."
        )
        self.front_flux_damping_spin = QtWidgets.QDoubleSpinBox()
        self.front_flux_damping_spin.setRange(0.0, 1.0)
        self.front_flux_damping_spin.setDecimals(2)
        self.front_flux_damping_spin.setSingleStep(0.05)
        self.front_flux_damping_spin.setValue(0.5)
        self.front_flux_damping_spin.setToolTip(
            "Momentum-flux scale factor applied to edges on the wet/dry front.\n"
            "0.0 = fully damp momentum at the front (most stable, some diffusion).\n"
            "1.0 = no damping (default HLLC).\n"
            "0.5 is a good starting value for oscillating fronts."
        )
        self.active_set_hysteresis_chk = QtWidgets.QCheckBox("Enable")
        self.active_set_hysteresis_chk.setChecked(True)
        self.active_set_hysteresis_chk.setToolTip(
            "Keep cells active for one extra step after they dry below h_min.\n"
            "Prevents rapid oscillatory wet/dry switching at the advancing front.\n"
            "Has negligible performance overhead."
        )
        self.depth_cap_spin = QtWidgets.QDoubleSpinBox()
        self.depth_cap_spin.setRange(0.001, 1.0e7)
        self.depth_cap_spin.setDecimals(3)
        self.depth_cap_spin.setValue(1.0e6)
        self.depth_cap_spin.setToolTip("Absolute depth cap for robustness.")
        self.momentum_cap_min_speed_spin = QtWidgets.QDoubleSpinBox()
        self.momentum_cap_min_speed_spin.setRange(0.1, 1.0e4)
        self.momentum_cap_min_speed_spin.setDecimals(3)
        self.momentum_cap_min_speed_spin.setValue(50.0)
        self.momentum_cap_min_speed_spin.setToolTip(
            "Minimum speed floor used by momentum clipping."
        )
        self.momentum_cap_celerity_mult_spin = QtWidgets.QDoubleSpinBox()
        self.momentum_cap_celerity_mult_spin.setRange(0.1, 1000.0)
        self.momentum_cap_celerity_mult_spin.setDecimals(3)
        self.momentum_cap_celerity_mult_spin.setValue(20.0)
        self.momentum_cap_celerity_mult_spin.setToolTip(
            "Momentum clipping speed cap multiplier on sqrt(g*h)."
        )
        self.max_inv_area_spin = QtWidgets.QDoubleSpinBox()
        self.max_inv_area_spin.setRange(1.0, 1.0e12)
        self.max_inv_area_spin.setDecimals(1)
        self.max_inv_area_spin.setValue(1.0e6)
        self.max_inv_area_spin.setToolTip(
            "Cap on inverse cell area used in flux and update kernels."
        )
        self.cfl_lambda_cap_spin = QtWidgets.QDoubleSpinBox()
        self.cfl_lambda_cap_spin.setRange(1.0, 1.0e12)
        self.cfl_lambda_cap_spin.setDecimals(1)
        self.cfl_lambda_cap_spin.setValue(1.0e6)
        self.cfl_lambda_cap_spin.setToolTip(
            "Cap on local CFL lambda used in dt reduction and diagnostics."
        )
        self.rain_rate_spin = QtWidgets.QDoubleSpinBox()
        self.rain_rate_spin.setRange(0.0, 2000.0)
        self.rain_rate_spin.setDecimals(3)
        self.rain_rate_spin.setValue(0.0)
        self.rain_rate_spin.setSuffix(" mm/hr")
        self.cn_default_spin = QtWidgets.QDoubleSpinBox()
        self.cn_default_spin.setRange(1.0, 100.0)
        self.cn_default_spin.setDecimals(1)
        self.cn_default_spin.setValue(75.0)
        self.use_spatial_rain_cn_chk = QtWidgets.QCheckBox("Use Thiessen gage rainfall + CN infiltration when layers are available")
        self.use_spatial_rain_cn_chk.setChecked(True)
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
        self.degen_mode_combo = QtWidgets.QComboBox()
        for _label, _val in [
            ("None (max_inv_area cap)", 0),
            ("Skip (permanently inactive)", 1),
            ("Repair (neighbor-avg inv_area)", 2),
            ("Merge (redirect flux to owner)", 3),
        ]:
            self.degen_mode_combo.addItem(_label, int(_val))
        self.degen_mode_combo.setCurrentIndex(0)
        self.degen_mode_combo.setToolTip(
            "Degenerate cell handling mode (cells with area below 1/max_inv_area).\n"
            "None: existing max_inv_area cap in update kernel (default).\n"
            "Skip: permanently exclude degenerate cells from all flux/update.\n"
            "Repair: replace degenerate cell inv_area with neighbor average;\n"
            "  keeps them in physics with sane CFL contribution.\n"
            "Merge: redirect flux accumulation to largest non-degenerate neighbor."
        )
        self.coupling_loop_combo = QtWidgets.QComboBox()
        self.coupling_loop_combo.addItem("CPU coupling loop (reference)", "cpu")
        self.coupling_loop_combo.addItem("CUDA coupling loop (source assembly)", "cuda")
        self.coupling_loop_combo.setCurrentIndex(0)
        self.coupling_loop_combo.setToolTip(
            "Select coupling source assembly mode.\n"
            "CPU: Python reference path for drainage/structure source rates.\n"
            "CUDA: uses native CUDA kernel for per-cell source assembly when available;\n"
            "falls back to CPU reference automatically if CUDA binding/device is unavailable."
        )
        self.gpu_default_lbl = QtWidgets.QLabel("GPU is attempted by default when supported by the native backend.")
        self.gpu_default_lbl.setWordWrap(True)
        self.unit_system_lbl = QtWidgets.QLabel("Unit system: auto")
        self.unit_system_lbl.setWordWrap(True)
        param_form.addRow("Manning n:", self.n_mann_spin)
        param_form.addRow("CFL:", self.cfl_spin)
        param_form.addRow("h_min:", self.h_min_spin)
        param_form.addRow("Initial condition:", self.initial_condition_combo)
        param_form.addRow("Initial depth:", self.initial_depth_spin)
        param_form.addRow("Initial WSE:", self.initial_wse_spin)
        param_form.addRow("Variable timestep:", self.adaptive_cfl_dt_chk)
        param_form.addRow("dt (fixed or dt_max):", self.dt_spin)
        param_form.addRow("Max rel depth increase:", self.max_rel_depth_increase_spin)
        param_form.addRow("Shallow damping depth:", self.shallow_damping_depth_spin)
        param_form.addRow("Front flux damping:", self.front_flux_damping_spin)
        param_form.addRow("Active-set hysteresis:", self.active_set_hysteresis_chk)
        param_form.addRow("Depth cap:", self.depth_cap_spin)
        param_form.addRow("Momentum cap min speed:", self.momentum_cap_min_speed_spin)
        param_form.addRow("Momentum cap celerity mult:", self.momentum_cap_celerity_mult_spin)
        param_form.addRow("Max inv area:", self.max_inv_area_spin)
        param_form.addRow("CFL lambda cap:", self.cfl_lambda_cap_spin)
        param_form.addRow("Rain rate:", self.rain_rate_spin)
        param_form.addRow("Default CN:", self.cn_default_spin)
        param_form.addRow("Rain/CN forcing:", self.use_spatial_rain_cn_chk)
        param_form.addRow("Internal flow layer:", self.internal_flow_layer_combo)
        param_form.addRow("Internal flow field:", self.internal_flow_field_edit)
        param_form.addRow("Run duration (hr or HH:MM):", self.run_time_edit)
        param_form.addRow("Reconstruction:", self.reconstruction_combo)
        param_form.addRow("Degenerate cell mode:", self.degen_mode_combo)
        param_form.addRow("Coupling loop:", self.coupling_loop_combo)
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
        snap_row.addWidget(QtWidgets.QLabel("Line output interval:"))
        self.line_output_interval_edit = QtWidgets.QLineEdit("00:05")
        self.line_output_interval_edit.setMaximumWidth(90)
        self.line_output_interval_edit.setToolTip(
            "Interval for sampled line time-series output capture.\n"
            "Independent from mesh snapshot interval."
        )
        snap_row.addWidget(self.line_output_interval_edit)
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
        mesh_options: Optional[Dict[str, object]] = None,
        run_mode: str = "full",
    ):
        if self._topology_mesh_future is not None and not self._topology_mesh_future.done():
            self._log("Topology mesh is already running. Please wait for completion.")
            return

        self._topology_mesh_backend = backend_name
        self._topology_mesh_default_cell_type = default_cell_type
        self._topology_mesh_run_mode = run_mode
        self._topology_mesh_conceptual = conceptual
        self._topology_mesh_options = dict(mesh_options or {})
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

        self._topology_mesh_future = executor.submit(
            _run_topology_mesh_job,
            conceptual,
            backend_name,
            self._topology_mesh_options,
        )
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
                        self._topology_mesh_options,
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
        is_sample_lines = "sample_lines" in lname
        is_manning = "manning" in lname
        is_cn_zone = "cn_zones" in lname
        is_rain_gage = "rain_gages" in lname
        is_hyetograph = "hyetographs" in lname
        is_drain_nodes = "drainage_nodes" in lname
        is_drain_links = "drainage_links" in lname
        is_drain_inlets = "drainage_inlets" in lname
        is_structures = ("structures" in lname) and ("hydrographs" not in lname)

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

        if is_cn_zone:
            self._set_expression_constraint(layer, "cn", '"cn" >= 1 AND "cn" <= 100')
            self._set_expression_constraint(layer, "priority", '"priority" >= 0')

        if is_rain_gage:
            self._set_expression_constraint(layer, "gage_id", 'length(trim("gage_id")) > 0')
            self._set_expression_constraint(layer, "hyetograph_id", 'length(trim("hyetograph_id")) > 0')

        if is_hyetograph:
            self._set_expression_constraint(layer, "hydrograph_id", 'length(trim("hydrograph_id")) > 0')
            self._set_expression_constraint(layer, "hyetograph_id", 'length(trim("hyetograph_id")) > 0')
            self._set_expression_constraint(layer, "Time", 'length(trim("Time")) > 0')
            self._set_expression_constraint(layer, "Value", '"Value" >= 0')

        if is_sample_lines:
            self._set_expression_constraint(layer, "line_id", '"line_id" IS NULL OR "line_id" >= 0')
            self._set_expression_constraint(layer, "enabled", '"enabled" IS NULL OR "enabled" IN (0,1)')
            self._set_expression_constraint(layer, "priority", '"priority" IS NULL OR "priority" >= 0')

        if is_drain_nodes:
            self._set_expression_constraint(layer, "node_id", 'length(trim("node_id")) > 0')
            self._set_expression_constraint(layer, "max_depth", '"max_depth" > 0')
            self._set_expression_constraint(layer, "surface_area_m2", '"surface_area_m2" IS NULL OR "surface_area_m2" > 0')

        if is_drain_links:
            self._set_expression_constraint(layer, "link_id", 'length(trim("link_id")) > 0')
            self._set_expression_constraint(layer, "from_node", 'length(trim("from_node")) > 0')
            self._set_expression_constraint(layer, "to_node", 'length(trim("to_node")) > 0')
            self._set_expression_constraint(layer, "length_m", '"length_m" IS NULL OR "length_m" > 0')
            self._set_expression_constraint(layer, "roughness_n", '"roughness_n" IS NULL OR "roughness_n" > 0')
            self._set_expression_constraint(layer, "diameter_m", '"diameter_m" IS NULL OR "diameter_m" > 0')

        if is_drain_inlets:
            self._set_expression_constraint(layer, "inlet_id", 'length(trim("inlet_id")) > 0')
            self._set_expression_constraint(layer, "node_id", 'length(trim("node_id")) > 0')
            self._set_expression_constraint(layer, "width_m", '"width_m" IS NULL OR "width_m" > 0')
            self._set_expression_constraint(layer, "coefficient", '"coefficient" IS NULL OR "coefficient" > 0')

        if is_structures:
            self._set_value_map_editor(layer, "structure_type", _STRUCTURE_TYPE_VALUE_MAP)
            self._set_expression_constraint(layer, "structure_id", 'length(trim("structure_id")) > 0')
            self._set_expression_constraint(layer, "structure_type", '"structure_type" IN (1,2,3,4,5)')
            self._set_expression_constraint(layer, "enabled", '"enabled" IS NULL OR "enabled" IN (0,1)')

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

    def _is_us_customary_units(self) -> bool:
        return str(self._length_unit_name).strip().lower() == "ft"

    def _length_scale_si_to_model(self) -> float:
        # Solver state units follow CRS map units. Convert SI length to solver length.
        return 3.280839895013123 if self._is_us_customary_units() else 1.0

    def _rain_mm_to_model_depth(self) -> float:
        # 1 mm = 0.001 m, then convert SI meters to solver-space length units.
        return 1.0e-3 * self._length_scale_si_to_model()

    def _rain_rate_si_to_model(self, rain_rate_mps):
        return np.asarray(rain_rate_mps, dtype=np.float64) * self._length_scale_si_to_model()

    def _flow_si_to_model(self, flow_cms):
        # 1 m3/s -> ft3/s when running in US customary CRS units.
        q_scale = self._length_scale_si_to_model() ** 3
        return np.asarray(flow_cms, dtype=np.float64) * q_scale

    def _flow_unit_label(self) -> str:
        return "ft3/s" if self._is_us_customary_units() else "m3/s"

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
        keep_cn = self.cn_layer_combo.currentData() if hasattr(self, "cn_layer_combo") else None
        keep_rain_gages = self.rain_gage_layer_combo.currentData() if hasattr(self, "rain_gage_layer_combo") else None
        keep_hyetograph = self.hyetograph_layer_combo.currentData() if hasattr(self, "hyetograph_layer_combo") else None
        keep_topo_nodes = self.topo_nodes_combo.currentData() if hasattr(self, "topo_nodes_combo") else None
        keep_topo_arcs = self.topo_arcs_combo.currentData() if hasattr(self, "topo_arcs_combo") else None
        keep_topo_regions = self.topo_regions_combo.currentData() if hasattr(self, "topo_regions_combo") else None
        keep_topo_constraints = self.topo_constraints_combo.currentData() if hasattr(self, "topo_constraints_combo") else None
        keep_topo_quad_edges = self.topo_quad_edges_combo.currentData() if hasattr(self, "topo_quad_edges_combo") else None
        keep_bc_lines = self.bc_lines_layer_combo.currentData() if hasattr(self, "bc_lines_layer_combo") else None
        keep_internal_flow = self.internal_flow_layer_combo.currentData() if hasattr(self, "internal_flow_layer_combo") else None
        keep_sample_lines = self.sample_lines_layer_combo.currentData() if hasattr(self, "sample_lines_layer_combo") else None
        keep_drain_nodes = self.drain_nodes_layer_combo.currentData() if hasattr(self, "drain_nodes_layer_combo") else None
        keep_drain_links = self.drain_links_layer_combo.currentData() if hasattr(self, "drain_links_layer_combo") else None
        keep_drain_inlets = self.drain_inlets_layer_combo.currentData() if hasattr(self, "drain_inlets_layer_combo") else None
        keep_structures = self.structures_layer_combo.currentData() if hasattr(self, "structures_layer_combo") else None

        self.nodes_layer_combo.clear()
        self.cells_layer_combo.clear()
        self.terrain_layer_combo.clear()
        if hasattr(self, "manning_layer_combo"):
            self.manning_layer_combo.clear()
            self.manning_layer_combo.addItem("(none)", None)
        if hasattr(self, "cn_layer_combo"):
            self.cn_layer_combo.clear()
            self.cn_layer_combo.addItem("(none)", None)
        if hasattr(self, "rain_gage_layer_combo"):
            self.rain_gage_layer_combo.clear()
            self.rain_gage_layer_combo.addItem("(none)", None)
        if hasattr(self, "hyetograph_layer_combo"):
            self.hyetograph_layer_combo.clear()
            self.hyetograph_layer_combo.addItem("(none)", None)
        if hasattr(self, "sample_lines_layer_combo"):
            self.sample_lines_layer_combo.clear()
            self.sample_lines_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_nodes_layer_combo"):
            self.drain_nodes_layer_combo.clear()
            self.drain_nodes_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_links_layer_combo"):
            self.drain_links_layer_combo.clear()
            self.drain_links_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_inlets_layer_combo"):
            self.drain_inlets_layer_combo.clear()
            self.drain_inlets_layer_combo.addItem("(none)", None)
        if hasattr(self, "structures_layer_combo"):
            self.structures_layer_combo.clear()
            self.structures_layer_combo.addItem("(none)", None)
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
                        if hasattr(self, "rain_gage_layer_combo"):
                            self.rain_gage_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_nodes_combo"):
                            self.topo_nodes_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_nodes_layer_combo"):
                            self.drain_nodes_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_inlets_layer_combo"):
                            self.drain_inlets_layer_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
                        self.cells_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "manning_layer_combo"):
                            self.manning_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "cn_layer_combo"):
                            self.cn_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_regions_combo"):
                            self.topo_regions_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_constraints_combo"):
                            self.topo_constraints_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type in (
                        QgsWkbTypes.GeometryType.UnknownGeometry,
                        getattr(QgsWkbTypes.GeometryType, "NullGeometry", QgsWkbTypes.GeometryType.UnknownGeometry),
                    ):
                        if hasattr(self, "hyetograph_layer_combo"):
                            self.hyetograph_layer_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.LineGeometry:
                        if hasattr(self, "sample_lines_layer_combo"):
                            self.sample_lines_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_arcs_combo"):
                            self.topo_arcs_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_quad_edges_combo"):
                            self.topo_quad_edges_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "bc_lines_layer_combo"):
                            self.bc_lines_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_links_layer_combo"):
                            self.drain_links_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "structures_layer_combo"):
                            self.structures_layer_combo.addItem(lyr.name(), lyr.id())
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
        if hasattr(self, "cn_layer_combo"):
            _restore(self.cn_layer_combo, keep_cn)
        if hasattr(self, "rain_gage_layer_combo"):
            _restore(self.rain_gage_layer_combo, keep_rain_gages)
        if hasattr(self, "hyetograph_layer_combo"):
            _restore(self.hyetograph_layer_combo, keep_hyetograph)
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
        if hasattr(self, "sample_lines_layer_combo") and keep_sample_lines is not None:
            _restore(self.sample_lines_layer_combo, keep_sample_lines)
        if hasattr(self, "drain_nodes_layer_combo") and keep_drain_nodes is not None:
            _restore(self.drain_nodes_layer_combo, keep_drain_nodes)
        if hasattr(self, "drain_links_layer_combo") and keep_drain_links is not None:
            _restore(self.drain_links_layer_combo, keep_drain_links)
        if hasattr(self, "drain_inlets_layer_combo") and keep_drain_inlets is not None:
            _restore(self.drain_inlets_layer_combo, keep_drain_inlets)
        if hasattr(self, "structures_layer_combo") and keep_structures is not None:
            _restore(self.structures_layer_combo, keep_structures)

        self._update_unit_system_from_crs()
        self._update_topology_control_summary()

    def _parse_csv_number_list(self, text: str, cast=float):
        values = []
        for part in str(text or "").split(","):
            item = part.strip()
            if not item:
                continue
            number = float(item)
            values.append(cast(number) if cast is int else cast(number))
        return values

    def _build_topology_meshing_options(self) -> Dict[str, object]:
        return {
            "gmsh_tri_algorithm": int(self.topo_gmsh_tri_algo_combo.currentData() or 6),
            "gmsh_quad_algorithm": int(self.topo_gmsh_quad_algo_combo.currentData() or 6),
            "gmsh_recombination_algorithm": int(self.topo_gmsh_recombine_algo_combo.currentData() or 1),
            "gmsh_smoothing": int(self.topo_gmsh_smoothing_spin.value()),
            "gmsh_optimize_iters": int(self.topo_gmsh_optimize_iters_spin.value()),
            "gmsh_optimize_netgen": bool(self.topo_gmsh_optimize_netgen_chk.isChecked()),
            "gmsh_verbosity": int(self.topo_gmsh_verbosity_spin.value()),
            "tqmesh_min_angle_deg": float(self.topo_quality_min_angle_spin.value()),
            "tqmesh_max_aspect_ratio": float(self.topo_quality_max_aspect_spin.value()),
            "tqmesh_min_area_rel_bbox": float(self.topo_quality_min_area_edit.text().strip() or "0"),
            "tqmesh_quality_strict": bool(self.topo_quality_strict_chk.isChecked()),
            "tqmesh_size_scales": tuple(self._parse_csv_number_list(self.topo_quality_size_scales_edit.text(), float) or [1.0]),
            "tqmesh_smooth_increments": tuple(self._parse_csv_number_list(self.topo_quality_smooth_increments_edit.text(), int) or [0]),
        }

    def _open_topology_region_table(self):
        layer = self._combo_layer(self.topo_regions_combo, "vector")
        if layer is None:
            QtWidgets.QMessageBox.information(self, "Topology Editor", "Select a topology regions layer first.")
            return
        dlg = TopologyAttributeTableDialog(
            layer,
            "Topology Region Controls",
            [
                ("region_id", "Region ID", "int"),
                ("target_size", "Target Size", "float"),
                ("cell_type", "Cell Type", "enum", _CELL_TYPE_OPTIONS),
                ("edge_len_1", "Edge Len 1", "float"),
                ("edge_len_2", "Edge Len 2", "float"),
                ("edge_len_3", "Edge Len 3", "float"),
                ("edge_len_4", "Edge Len 4", "float"),
            ],
            sort_fields=["region_id"],
            note=(
                "Use one polygon per block. For structured/cartesian blocks, edge_len_1..4 define per-edge target spacing "
                "used by Gmsh and the structured fallback when the region has a complete four-edge topology definition."
            ),
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._update_topology_control_summary()

    def _open_topology_quad_edge_table(self):
        layer = self._combo_layer(self.topo_quad_edges_combo, "vector")
        if layer is None:
            QtWidgets.QMessageBox.information(self, "Topology Editor", "Select a quad-edge / transition-layer layer first.")
            return
        dlg = TopologyAttributeTableDialog(
            layer,
            "Topology Transition Layers",
            [
                ("region_id", "Region ID", "int"),
                ("edge_id", "Edge ID", "int"),
                ("target_size", "Target Size", "float"),
                ("n_layers", "N Layers", "int"),
                ("first_height", "First Height", "float"),
                ("growth_rate", "Growth Rate", "float"),
            ],
            sort_fields=["region_id", "edge_id"],
            note=(
                "Define one line per region edge for a complete four-edge structured block. n_layers / first_height / growth_rate "
                "control transition-layer packing inward from that edge."
            ),
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._update_topology_control_summary()

    def _update_topology_control_summary(self):
        if not hasattr(self, "topo_controls_summary_lbl"):
            return

        backend_name = str(self.topo_backend_combo.currentData() or "structured") if hasattr(self, "topo_backend_combo") else "structured"
        regions_layer = self._combo_layer(self.topo_regions_combo, "vector") if hasattr(self, "topo_regions_combo") else None
        quad_edges_layer = self._combo_layer(self.topo_quad_edges_combo, "vector") if hasattr(self, "topo_quad_edges_combo") else None

        if backend_name == "gmsh":
            backend_hint = (
                "Gmsh: use multiple region polygons for multiblock meshes. "
                "Set region cell_type to 'cartesian' or 'quadrilateral' and populate edge_len_1..4 "
                "for per-edge structured spacing. Opposite edges are matched automatically."
            )
        elif backend_name == "tqmesh":
            backend_hint = (
                "TQMesh: use multiple region polygons for blockwise target_size and cell_type. "
                "Use quad-edge lines with n_layers, first_height, and growth_rate for transition layers."
            )
        else:
            backend_hint = (
                "Structured fallback: honors per-region target_size and cell_type, "
                "but does not apply quad-edge transition layers or exact transfinite edge counts."
            )

        details: List[str] = []
        if regions_layer is not None:
            try:
                region_fields = set(regions_layer.fields().names())
                region_count = 0
                cartesian_count = 0
                size_values = set()
                missing_edge_lengths = 0
                for ft in regions_layer.getFeatures():
                    region_count += 1
                    ctype = str(ft["cell_type"]).strip().lower() if "cell_type" in region_fields and ft["cell_type"] not in (None, "") else ""
                    if ctype in {"cartesian", "quadrilateral"}:
                        cartesian_count += 1
                        edge_fields = [f"edge_len_{i}" for i in range(1, 5)]
                        edge_ok = True
                        for name in edge_fields:
                            if name not in region_fields or ft[name] in (None, ""):
                                edge_ok = False
                                break
                            try:
                                if float(ft[name]) <= 0.0:
                                    edge_ok = False
                                    break
                            except Exception:
                                edge_ok = False
                                break
                        if not edge_ok:
                            missing_edge_lengths += 1
                    if "target_size" in region_fields and ft["target_size"] not in (None, ""):
                        try:
                            size_values.add(round(float(ft["target_size"]), 6))
                        except Exception:
                            pass
                details.append(f"regions={region_count}")
                if cartesian_count > 0:
                    details.append(f"structured-block-regions={cartesian_count}")
                if len(size_values) > 1:
                    details.append(f"multi-block sizes={len(size_values)}")
                if missing_edge_lengths > 0:
                    details.append(f"structured regions missing edge_len_1..4={missing_edge_lengths}")
            except Exception:
                pass

        if quad_edges_layer is not None and getattr(self, "topo_quad_edges_combo", None) is not None and self.topo_quad_edges_combo.currentData() is not None:
            try:
                q_fields = set(quad_edges_layer.fields().names())
                edge_count = 0
                layered_edges = 0
                total_layers = 0
                for ft in quad_edges_layer.getFeatures():
                    edge_count += 1
                    if "n_layers" in q_fields and ft["n_layers"] not in (None, ""):
                        nl = max(0, int(ft["n_layers"]))
                        total_layers += nl
                        if nl > 0:
                            layered_edges += 1
                details.append(f"quad-edges={edge_count}")
                if layered_edges > 0:
                    details.append(f"transition-layer-edges={layered_edges}")
                    details.append(f"total-n_layers={total_layers}")
            except Exception:
                pass

        suffix = " | ".join(details)
        if suffix:
            self.topo_controls_summary_lbl.setText(f"{backend_hint} Current layers: {suffix}.")
        else:
            self.topo_controls_summary_lbl.setText(backend_hint)

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
        sample_lines = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=line_id:integer&field=name:string(128)&field=enabled:integer&field=priority:integer",
            "SWE2D_Sample_Lines",
            "memory",
        )
        drainage_nodes = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double&field=max_depth:double&field=node_type:string(32)&field=surface_area_m2:double",
            "SWE2D_Drainage_Nodes",
            "memory",
        )
        drainage_links = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)&field=to_node:string(64)&field=link_type:string(32)&field=length_m:double&field=roughness_n:double&field=diameter_m:double&field=max_flow_cms:double&field=cd:double",
            "SWE2D_Drainage_Links",
            "memory",
        )
        drainage_inlets = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=inlet_id:string(64)&field=node_id:string(64)&field=crest_elev:double&field=width_m:double&field=coefficient:double&field=max_capture_cms:double",
            "SWE2D_Drainage_Inlets",
            "memory",
        )
        structures = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width_m:double&field=height_m:double&field=diameter_m:double&field=length_m:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump_cms:double&field=max_flow_cms:double",
            "SWE2D_Structures",
            "memory",
        )
        hydro_tbl = QgsVectorLayer(
            "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)&field=Value:double&field=description:string(256)",
            "SWE2D_Hydrographs",
            "memory",
        )

        for lyr in (nodes, arcs, regions, constraints, quad_edges, manning, bc_lines, sample_lines, drainage_nodes, drainage_links, drainage_inlets, structures, hydro_tbl):
            if lyr is not None and lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)
                if isinstance(lyr, QgsVectorLayer):
                    self._configure_swe2d_layer_editors(lyr)

        self._refresh_layer_combos()
        self.topo_status_lbl.setText(
            "Topology template layers created. Define regions (required), optional arcs/constraints, and optional TQMesh quad-edge lines, then generate mesh."
        )
        self._log("Created topology template layers: SWE2D_Topo_Nodes/Arcs/Regions/Constraints/Quad_Edges + SWE2D_Manning_Zones + SWE2D_BC_Lines + SWE2D_Sample_Lines + SWE2D_Drainage_* + SWE2D_Structures + SWE2D_Hydrographs")

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
        sample_lines = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=line_id:integer&field=name:string(128)&field=enabled:integer&field=priority:integer",
            "swe2d_sample_lines",
            "memory",
        )
        rain_gages = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=gage_id:string(64)&field=name:string(128)&field=hyetograph_id:string(64)&field=units:string(32)&field=priority:integer",
            "swe2d_rain_gages",
            "memory",
        )
        cn_zones = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=zone_id:integer&field=cn:double&field=priority:integer",
            "swe2d_cn_zones",
            "memory",
        )
        hyetographs = QgsVectorLayer(
            "None?field=hyetograph_id:string(64)&field=Time:string(32)&field=Value:double&field=value_type:string(24)&field=units:string(24)&field=description:string(256)",
            "swe2d_hyetographs",
            "memory",
        )
        hydro = QgsVectorLayer(
            "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)&field=Value:double&field=description:string(256)",
            "swe2d_hydrographs",
            "memory",
        )
        drainage_nodes = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double&field=max_depth:double&field=node_type:string(32)&field=surface_area_m2:double",
            "swe2d_drainage_nodes",
            "memory",
        )
        drainage_links = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)&field=to_node:string(64)&field=link_type:string(32)&field=length_m:double&field=roughness_n:double&field=diameter_m:double&field=max_flow_cms:double&field=cd:double",
            "swe2d_drainage_links",
            "memory",
        )
        drainage_inlets = QgsVectorLayer(
            f"Point?crs={crs_auth}&field=inlet_id:string(64)&field=node_id:string(64)&field=crest_elev:double&field=width_m:double&field=coefficient:double&field=max_capture_cms:double",
            "swe2d_drainage_inlets",
            "memory",
        )
        structures = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width_m:double&field=height_m:double&field=diameter_m:double&field=length_m:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump_cms:double&field=max_flow_cms:double",
            "swe2d_structures",
            "memory",
        )

        model_layers = [
            nodes,
            arcs,
            regions,
            constraints,
            quad_edges,
            manning,
            bc_lines,
            sample_lines,
            rain_gages,
            cn_zones,
            hyetographs,
            hydro,
            drainage_nodes,
            drainage_links,
            drainage_inlets,
            structures,
        ]
        for lyr in model_layers:
            self._configure_swe2d_layer_editors(lyr)

        # Persist as a single GeoPackage file.
        for i, lyr in enumerate(model_layers):
            self._write_memory_layer_to_gpkg(lyr, out_path, lyr.name(), create_file=(i == 0))
        self._persist_model_layer_bindings(out_path)

        self._log(f"Created 2D model GeoPackage: {out_path}")
        self.layer_status_lbl.setText("2D model GeoPackage created.")
        self._load_2d_model_geopackage(path_override=out_path)

    def _create_lumped_hydrology_geopackage(self):
        if not _HAVE_QGIS_CORE:
            self._log("QGIS layer API unavailable; cannot create lumped hydrology GeoPackage.")
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Create Lumped Hydrology GeoPackage",
            "lumped_hydrology_model.gpkg",
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

        subbasins = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=sub_id:string(64)&field=name:string(128)&field=area_km2:double&field=cn:double&field=imperv_pct:double&field=tc_hr:double",
            "lumped_subbasins",
            "memory",
        )
        flow_paths = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=sub_id:string(64)&field=segment:string(32)&field=length_m:double&field=velocity_mps:double&field=slope:double",
            "lumped_flow_paths",
            "memory",
        )
        rain_events = QgsVectorLayer(
            "None?field=event_id:string(64)&field=Time:string(32)&field=Value:double&field=value_type:string(24)&field=units:string(24)&field=description:string(256)",
            "lumped_rain_events",
            "memory",
        )

        layers = [subbasins, flow_paths, rain_events]
        for i, lyr in enumerate(layers):
            self._write_memory_layer_to_gpkg(lyr, out_path, lyr.name(), create_file=(i == 0))

        self._log(
            "Created lumped hydrology GeoPackage with layers: "
            "lumped_subbasins, lumped_flow_paths, lumped_rain_events"
        )
        if runoff_depth_mm_from_event_rain_mm is not None and time_of_concentration_hours_velocity_method is not None:
            demo_q = runoff_depth_mm_from_event_rain_mm(50.0, 75.0)
            demo_tc = time_of_concentration_hours_velocity_method([(100.0, 0.5), (900.0, 1.2)])
            self._log(
                "Lumped helpers ready: "
                f"example runoff(50 mm, CN=75)={demo_q:.2f} mm, "
                f"example Tc={demo_tc:.2f} hr"
            )
        self.layer_status_lbl.setText("Lumped hydrology GeoPackage created.")

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
            "swe2d_sample_lines",
            "swe2d_rain_gages",
            "swe2d_cn_zones",
            "swe2d_hyetographs",
            "swe2d_hydrographs",
            "swe2d_drainage_nodes",
            "swe2d_drainage_links",
            "swe2d_drainage_inlets",
            "swe2d_structures",
        ]
        loaded = 0
        for lname in layer_names:
            lyr = QgsVectorLayer(f"{gpkg_path}|layername={lname}", lname, "ogr")
            if lyr is not None and lyr.isValid():
                QgsProject.instance().addMapLayer(lyr)
                self._configure_swe2d_layer_editors(lyr)
                loaded += 1

        self._refresh_layer_combos()
        self._model_gpkg_path = str(gpkg_path)
        schema_warnings = self._restore_model_layer_bindings(self._model_gpkg_path)
        self._log(f"Loaded 2D model GeoPackage: {gpkg_path} (layers loaded={loaded})")
        if schema_warnings:
            self._log("Coupling schema warnings: " + " | ".join(schema_warnings))
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
            mesh_options = self._build_topology_meshing_options()
            conceptual = conceptual_from_qgis_layers(
                nodes_layer=nodes_layer,
                arcs_layer=arcs_layer,
                regions_layer=regions_layer,
                constraints_layer=constraints_layer,
                quad_edges_layer=quad_edges_layer,
                default_size=default_size,
                default_cell_type=default_cell_type,
            )
            self._start_topology_mesh_async(conceptual, backend_name, default_cell_type, mesh_options)
        except ValueError as exc:
            self.topo_status_lbl.setText(f"Invalid topology mesh options: {exc}")
            self._log(f"Topology mesh option error: {exc}")
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

    def _mesh_cell_min_bed(self) -> np.ndarray:
        assert self._mesh_data is not None
        node_z = self._mesh_data["node_z"]
        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            out = np.zeros(offs.size - 1, dtype=np.float64)
            for i in range(offs.size - 1):
                s = int(offs[i])
                e = int(offs[i + 1])
                ids = faces[s:e]
                if ids.size:
                    out[i] = float(np.min(node_z[ids]))
            return out
        tri = self._mesh_data["cell_nodes"].reshape(-1, 3).astype(np.int32)
        return np.min(node_z[tri], axis=1).astype(np.float64)

    def _mesh_cell_polygons(self) -> List[QgsGeometry]:
        assert self._mesh_data is not None
        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        out: List[QgsGeometry] = []

        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            for i in range(offs.size - 1):
                s = int(offs[i])
                e = int(offs[i + 1])
                ids = faces[s:e]
                if ids.size < 3:
                    out.append(QgsGeometry())
                    continue
                ring = [QgsPointXY(float(node_x[n]), float(node_y[n])) for n in ids]
                ring.append(ring[0])
                out.append(QgsGeometry.fromPolygonXY([ring]))
            return out

        tris = self._mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
        for tri in tris:
            ring = [
                QgsPointXY(float(node_x[int(tri[0])]), float(node_y[int(tri[0])])),
                QgsPointXY(float(node_x[int(tri[1])]), float(node_y[int(tri[1])])),
                QgsPointXY(float(node_x[int(tri[2])]), float(node_y[int(tri[2])])),
            ]
            ring.append(ring[0])
            out.append(QgsGeometry.fromPolygonXY([ring]))
        return out

    def _build_line_sampling_map(self) -> List[Dict[str, object]]:
        if self._mesh_data is None or not _HAVE_QGIS_CORE:
            return []
        if not hasattr(self, "sample_lines_layer_combo"):
            return []
        line_layer = self._combo_layer(self.sample_lines_layer_combo, "vector")
        if line_layer is None:
            return []

        fields = set(line_layer.fields().names())
        id_field = "line_id" if "line_id" in fields else None
        name_field = "name" if "name" in fields else None
        enabled_field = "enabled" if "enabled" in fields else None

        cell_polys = self._mesh_cell_polygons()
        if not cell_polys:
            return []
        cell_bboxes = [g.boundingBox() if g is not None and not g.isEmpty() else None for g in cell_polys]

        sample_map: List[Dict[str, object]] = []
        for ft in line_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                if enabled_field is not None and int(ft[enabled_field]) <= 0:
                    continue
            except Exception:
                pass

            line_len = float(geom.length())
            if line_len <= 0.0:
                continue
            try:
                p0 = geom.interpolate(0.0).asPoint()
                p1 = geom.interpolate(max(0.0, line_len - 1.0e-9)).asPoint()
                dx = float(p1.x()) - float(p0.x())
                dy = float(p1.y()) - float(p0.y())
                mag = math.hypot(dx, dy)
                if mag <= 0.0:
                    continue
                tx = dx / mag
                ty = dy / mag
                nx = ty
                ny = -tx
            except Exception:
                continue

            try:
                line_id = int(ft[id_field]) if id_field is not None else int(ft.id())
            except Exception:
                line_id = int(ft.id())
            line_name = str(ft[name_field]) if name_field is not None and ft[name_field] not in (None, "") else ""

            line_bbox = geom.boundingBox()
            idx: List[int] = []
            lens: List[float] = []
            station_m: List[float] = []
            for ci, cell_geom in enumerate(cell_polys):
                bb = cell_bboxes[ci]
                if bb is None or not bb.intersects(line_bbox):
                    continue
                try:
                    inter = cell_geom.intersection(geom)
                except Exception:
                    continue
                if inter is None or inter.isEmpty():
                    continue
                seg_len = float(inter.length())
                if seg_len <= 0.0:
                    continue
                s_loc = float("nan")
                try:
                    cgeom = inter.centroid()
                    if cgeom is not None and not cgeom.isEmpty():
                        s_loc = float(geom.lineLocatePoint(cgeom))
                except Exception:
                    s_loc = float("nan")
                idx.append(ci)
                lens.append(seg_len)
                station_m.append(s_loc)

            if idx:
                ord_idx = np.argsort(np.nan_to_num(np.asarray(station_m, dtype=np.float64), nan=0.0))
                sample_map.append(
                    {
                        "line_id": int(line_id),
                        "line_name": line_name,
                        "normal_x": float(nx),
                        "normal_y": float(ny),
                        "cell_idx": np.asarray(idx, dtype=np.int32)[ord_idx],
                        "weights": np.asarray(lens, dtype=np.float64)[ord_idx],
                        "station_m": np.asarray(station_m, dtype=np.float64)[ord_idx],
                    }
                )

        if sample_map:
            self._log(f"Sample line mapping ready: {len(sample_map)} line(s).")
        return sample_map

    def _sample_line_metrics(
        self,
        sample_map: List[Dict[str, object]],
        t_s: float,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        cell_bed: np.ndarray,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        if not sample_map:
            return [], []
        out_ts: List[Dict[str, object]] = []
        out_prof: List[Dict[str, object]] = []
        g = float(self._gravity)
        h_min = float(self.h_min_spin.value()) if hasattr(self, "h_min_spin") else 1.0e-6
        for sm in sample_map:
            idx = sm["cell_idx"]
            w = sm["weights"]
            if idx.size == 0 or w.size == 0:
                continue
            hh = h[idx]
            huu = hu[idx]
            hvv = hv[idx]
            zb = cell_bed[idx]
            wet = (hh > h_min)
            safe_h = np.maximum(hh, 1.0e-12)
            vel = np.where(wet, np.sqrt((huu / safe_h) ** 2 + (hvv / safe_h) ** 2), 0.0)
            wsum = float(np.sum(w))
            if wsum <= 0.0:
                continue
            depth_m = float(np.sum(hh * w) / wsum)
            velocity_ms = float(np.sum(vel * w) / wsum)
            wse_m = float(np.sum((hh + zb) * w) / wsum)
            bed_m = float(np.sum(zb * w) / wsum)
            qn = huu * float(sm["normal_x"]) + hvv * float(sm["normal_y"])
            flow_cms = float(np.sum(qn * w))
            fr_arr = np.where(wet, vel / np.sqrt(np.maximum(g * hh, 1.0e-12)), 0.0)

            out_ts.append(
                {
                    "t_s": float(t_s),
                    "line_id": int(sm["line_id"]),
                    "line_name": str(sm.get("line_name", "") or ""),
                    "depth_m": depth_m,
                    "velocity_ms": velocity_ms,
                    "wse_m": wse_m,
                    "bed_m": bed_m,
                    "flow_cms": flow_cms,
                    "wet_frac": float(np.mean(wet.astype(np.float64))),
                    "fr": float(np.mean(fr_arr)),
                }
            )

            sta = np.asarray(sm.get("station_m", np.arange(idx.size, dtype=np.float64)), dtype=np.float64)
            if sta.size != idx.size:
                sta = np.linspace(0.0, float(idx.size - 1), idx.size, dtype=np.float64)
            for j in range(idx.size):
                out_prof.append(
                    {
                        "t_s": float(t_s),
                        "line_id": int(sm["line_id"]),
                        "line_name": str(sm.get("line_name", "") or ""),
                        "station_m": float(sta[j]),
                        "depth_m": float(hh[j]),
                        "velocity_ms": float(vel[j]),
                        "wse_m": float(hh[j] + zb[j]),
                        "bed_m": float(zb[j]),
                        "flow_qn": float(qn[j]),
                        "wet": int(bool(wet[j])),
                        "fr": float(fr_arr[j]),
                    }
                )
        return out_ts, out_prof

    def _current_line_results_storage_path(self) -> str:
        if self._model_gpkg_path and os.path.exists(self._model_gpkg_path):
            return self._model_gpkg_path
        if hasattr(self, "sample_lines_layer_combo"):
            lyr = self._combo_layer(self.sample_lines_layer_combo, "vector")
            if lyr is not None:
                try:
                    src = str(lyr.dataProvider().dataSourceUri())
                    gpkg = src.split("|", 1)[0]
                    if gpkg.lower().endswith(".gpkg") and os.path.exists(gpkg):
                        return gpkg
                except Exception:
                    pass
        import tempfile
        return os.path.join(tempfile.gettempdir(), "swe2d_line_results.gpkg")

    def _persist_line_results_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        rows: List[Dict[str, object]],
        mesh_interval_s: float,
        line_interval_s: float,
        profile_rows: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        if not gpkg_path or not rows:
            return
        profile_rows = list(profile_rows or [])
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_line_results_runs (
                    run_id TEXT PRIMARY KEY,
                    created_utc TEXT,
                    mesh_interval_s REAL,
                    line_interval_s REAL,
                    row_count INTEGER
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_line_results_ts (
                    run_id TEXT,
                    t_s REAL,
                    line_id INTEGER,
                    line_name TEXT,
                    depth_m REAL,
                    velocity_ms REAL,
                    wse_m REAL,
                    bed_m REAL,
                    flow_cms REAL,
                    wet_frac REAL,
                    fr REAL,
                    PRIMARY KEY (run_id, t_s, line_id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_swe2d_line_ts_run_line_t ON swe2d_line_results_ts(run_id, line_id, t_s)"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_line_results_profile (
                    run_id TEXT,
                    t_s REAL,
                    line_id INTEGER,
                    line_name TEXT,
                    station_m REAL,
                    depth_m REAL,
                    velocity_ms REAL,
                    wse_m REAL,
                    bed_m REAL,
                    flow_qn REAL,
                    wet INTEGER,
                    fr REAL,
                    PRIMARY KEY (run_id, t_s, line_id, station_m)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_swe2d_line_prof_run_line_t_s ON swe2d_line_results_profile(run_id, line_id, t_s, station_m)"
            )
            cur.execute("DELETE FROM swe2d_line_results_ts WHERE run_id = ?", (run_id,))
            cur.execute("DELETE FROM swe2d_line_results_profile WHERE run_id = ?", (run_id,))
            cur.execute(
                """
                INSERT OR REPLACE INTO swe2d_line_results_runs
                (run_id, created_utc, mesh_interval_s, line_interval_s, row_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                    float(mesh_interval_s),
                    float(line_interval_s),
                    int(len(rows)),
                ),
            )
            batch = [
                (
                    str(run_id),
                    float(r.get("t_s", 0.0)),
                    int(r.get("line_id", -1)),
                    str(r.get("line_name", "") or ""),
                    float(r.get("depth_m", float("nan"))),
                    float(r.get("velocity_ms", float("nan"))),
                    float(r.get("wse_m", float("nan"))),
                    float(r.get("bed_m", float("nan"))),
                    float(r.get("flow_cms", float("nan"))),
                    float(r.get("wet_frac", float("nan"))),
                    float(r.get("fr", float("nan"))),
                )
                for r in rows
            ]
            cur.executemany(
                """
                INSERT OR REPLACE INTO swe2d_line_results_ts
                (run_id, t_s, line_id, line_name, depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            if profile_rows:
                prof_batch = [
                    (
                        str(run_id),
                        float(r.get("t_s", 0.0)),
                        int(r.get("line_id", -1)),
                        str(r.get("line_name", "") or ""),
                        float(r.get("station_m", 0.0)),
                        float(r.get("depth_m", float("nan"))),
                        float(r.get("velocity_ms", float("nan"))),
                        float(r.get("wse_m", float("nan"))),
                        float(r.get("bed_m", float("nan"))),
                        float(r.get("flow_qn", float("nan"))),
                        int(r.get("wet", 0)),
                        float(r.get("fr", float("nan"))),
                    )
                    for r in profile_rows
                ]
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO swe2d_line_results_profile
                    (run_id, t_s, line_id, line_name, station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, wet, fr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    prof_batch,
                )
            conn.commit()
            self._line_results_latest_run_id = str(run_id)
            self._line_results_latest_db_path = str(gpkg_path)
            self._log(
                f"Stored sample line results in GeoPackage: {gpkg_path} "
                f"(run_id={run_id}, ts_rows={len(rows)}, profile_rows={len(profile_rows)})"
            )
        finally:
            conn.close()

    def _load_line_results_from_geopackage(
        self,
        gpkg_path: str,
        run_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]:
        if not gpkg_path or not os.path.exists(gpkg_path):
            return "", [], []
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_line_results_ts'"
            )
            if cur.fetchone() is None:
                return "", [], []

            chosen = str(run_id or "").strip()
            if not chosen:
                cur.execute(
                    """
                    SELECT run_id FROM swe2d_line_results_runs
                    ORDER BY datetime(created_utc) DESC, rowid DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row is None:
                    return "", [], []
                chosen = str(row[0])

            cur.execute(
                """
                SELECT t_s, line_id, line_name, depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr
                FROM swe2d_line_results_ts
                WHERE run_id = ?
                ORDER BY t_s ASC, line_id ASC
                """,
                (chosen,),
            )
            rows = []
            for t_s, line_id, line_name, depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr in cur.fetchall():
                rows.append(
                    {
                        "t_s": float(t_s),
                        "line_id": int(line_id),
                        "line_name": str(line_name or ""),
                        "depth_m": float(depth_m),
                        "velocity_ms": float(velocity_ms),
                        "wse_m": float(wse_m),
                        "bed_m": float(bed_m),
                        "flow_cms": float(flow_cms),
                        "wet_frac": float(wet_frac),
                        "fr": float(fr),
                    }
                )

            profile_rows: List[Dict[str, object]] = []
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_line_results_profile'"
            )
            if cur.fetchone() is not None:
                cur.execute(
                    """
                    SELECT t_s, line_id, line_name, station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, wet, fr
                    FROM swe2d_line_results_profile
                    WHERE run_id = ?
                    ORDER BY line_id ASC, t_s ASC, station_m ASC
                    """,
                    (chosen,),
                )
                for t_s, line_id, line_name, station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, wet, fr in cur.fetchall():
                    profile_rows.append(
                        {
                            "t_s": float(t_s),
                            "line_id": int(line_id),
                            "line_name": str(line_name or ""),
                            "station_m": float(station_m),
                            "depth_m": float(depth_m),
                            "velocity_ms": float(velocity_ms),
                            "wse_m": float(wse_m),
                            "bed_m": float(bed_m),
                            "flow_qn": float(flow_qn),
                            "wet": int(wet),
                            "fr": float(fr),
                        }
                    )

            return chosen, rows, profile_rows
        finally:
            conn.close()

    def _open_line_results_viewer(self):
        db_path = ""
        if self._line_results_latest_db_path and os.path.exists(self._line_results_latest_db_path):
            db_path = self._line_results_latest_db_path
        if not db_path:
            db_path = self._current_line_results_storage_path()
        if not db_path:
            self._log("No GeoPackage available for line results viewer.")
            return

        run_id = self._line_results_latest_run_id or None
        chosen, rows, profile_rows = self._load_line_results_from_geopackage(db_path, run_id=run_id)
        if not chosen or not rows:
            self._log("No sampled line results found in GeoPackage yet.")
            return
        dlg = SWE2DLineResultsViewerDialog(
            ts_records=rows,
            profile_records=profile_rows,
            run_id=chosen,
            db_path=db_path,
            length_unit=self._length_unit_name,
            flow_unit_label=self._flow_unit_label(),
            parent=self,
        )
        dlg.exec()

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
        rain_rate_model,
        cell_source_model: Optional[np.ndarray],
        coupled_source_rate: Optional[np.ndarray] = None,
    ) -> None:
        if dt_step <= 0.0:
            return
        if (
            np.all(np.asarray(rain_rate_model, dtype=np.float64) <= 0.0)
            and cell_source_model is None
            and coupled_source_rate is None
        ):
            return

        h, hu, hv = backend.get_state()
        rain_arr = np.asarray(rain_rate_model, dtype=np.float64)
        if rain_arr.ndim == 0:
            src = np.full(h.shape, float(rain_arr), dtype=np.float64)
        else:
            src = np.zeros(h.shape, dtype=np.float64)
            src[: min(src.shape[0], rain_arr.shape[0])] = rain_arr[: min(src.shape[0], rain_arr.shape[0])]
        if cell_source_model is not None:
            area = self._mesh_cell_areas()
            safe_area = np.maximum(area, 1.0e-8)
            src += (cell_source_model / safe_area)
        if coupled_source_rate is not None:
            csr = np.asarray(coupled_source_rate, dtype=np.float64)
            src[: min(src.shape[0], csr.shape[0])] += csr[: min(src.shape[0], csr.shape[0])]

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

    def _build_spatial_cn_array(self) -> np.ndarray:
        cx, cy = self._mesh_cell_centroids()
        cn_default = float(self.cn_default_spin.value()) if hasattr(self, "cn_default_spin") else 75.0
        cnvals = np.full(cx.shape[0], cn_default, dtype=np.float64)

        if self._mesh_data is None or not _HAVE_QGIS_CORE:
            return cnvals
        if not hasattr(self, "cn_layer_combo"):
            return cnvals

        lyr = self._combo_layer(self.cn_layer_combo, "vector")
        if lyr is None:
            return cnvals

        fields = set(lyr.fields().names())
        cn_field = None
        for cand in ("cn", "curve_number", "CN"):
            if cand in fields:
                cn_field = cand
                break
        if cn_field is None:
            self._log("CN layer selected but no cn/curve_number field found; using default CN.")
            return cnvals

        prio_field = "priority" if "priority" in fields else None
        features = []
        for ft in lyr.getFeatures():
            g = ft.geometry()
            if g is None or g.isEmpty():
                continue
            try:
                cn = float(ft[cn_field])
            except Exception:
                continue
            pr = 0
            if prio_field is not None:
                try:
                    pr = int(ft[prio_field])
                except Exception:
                    pr = 0
            features.append((pr, g, float(np.clip(cn, 1.0, 100.0))))

        if not features:
            return cnvals

        features.sort(key=lambda x: x[0], reverse=True)
        applied = 0
        for i in range(cx.shape[0]):
            p = QgsGeometry.fromPointXY(QgsPointXY(float(cx[i]), float(cy[i])))
            for _, g, cn in features:
                if g.contains(p) or g.intersects(p):
                    cnvals[i] = cn
                    applied += 1
                    break

        self._log(f"Spatial CN applied to {applied}/{cx.shape[0]} cells from '{lyr.name()}'.")
        return cnvals

    def _build_thiessen_rain_cn_forcing(self) -> Optional[ThiessenRainCNForcing]:
        if (
            self._mesh_data is None
            or not _HAVE_QGIS_CORE
            or ThiessenRainCNForcing is None
            or Gauge is None
            or build_hyetograph is None
            or assign_cells_to_nearest_gauge is None
        ):
            return None
        if not hasattr(self, "use_spatial_rain_cn_chk") or not bool(self.use_spatial_rain_cn_chk.isChecked()):
            return None
        if not hasattr(self, "rain_gage_layer_combo") or not hasattr(self, "hyetograph_layer_combo"):
            return None

        gage_layer = self._combo_layer(self.rain_gage_layer_combo, "vector")
        hyetograph_layer = self._combo_layer(self.hyetograph_layer_combo, "vector")
        if gage_layer is None or hyetograph_layer is None:
            return None

        gage_fields = set(gage_layer.fields().names())
        gid_field = "gage_id" if "gage_id" in gage_fields else None
        hyid_field = "hyetograph_id" if "hyetograph_id" in gage_fields else None
        if gid_field is None or hyid_field is None:
            self._log("Rain gage layer missing gage_id/hyetograph_id fields; skipping Thiessen rain forcing.")
            return None

        hy_fields = set(hyetograph_layer.fields().names())
        hy_id_field = "hyetograph_id" if "hyetograph_id" in hy_fields else None
        time_field = "Time" if "Time" in hy_fields else None
        value_field = "Value" if "Value" in hy_fields else None
        if hy_id_field is None or time_field is None or value_field is None:
            self._log("Hyetograph table missing hyetograph_id/Time/Value fields; skipping Thiessen rain forcing.")
            return None

        hy_rows_by_id: Dict[str, List[Dict[str, object]]] = {}
        for ft in hyetograph_layer.getFeatures():
            try:
                hy_id = str(ft[hy_id_field] or "").strip()
            except Exception:
                hy_id = ""
            if not hy_id:
                continue
            row = {
                "Time": ft[time_field],
                "Value": ft[value_field],
                "value_type": ft["value_type"] if "value_type" in hy_fields else "intensity",
                "units": ft["units"] if "units" in hy_fields else "mm/hr",
            }
            hy_rows_by_id.setdefault(hy_id, []).append(row)

        gauges: List[Gauge] = []
        hy_by_gauge_index: Dict[int, object] = {}
        for ft in gage_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                pt = geom.asPoint()
            except Exception:
                continue
            gauge_id = str(ft[gid_field] or "").strip()
            hy_id = str(ft[hyid_field] or "").strip()
            if not gauge_id or not hy_id:
                continue
            hy = build_hyetograph(hy_rows_by_id.get(hy_id, []))
            if hy is None:
                continue
            gauges.append(Gauge(gauge_id=gauge_id, x=float(pt.x()), y=float(pt.y()), hyetograph_id=hy_id))
            hy_by_gauge_index[len(gauges) - 1] = hy

        if not gauges:
            return None

        cell_x, cell_y = self._mesh_cell_centroids()
        cell_to_gauge = assign_cells_to_nearest_gauge(cell_x, cell_y, gauges)
        if cell_to_gauge is None:
            return None

        cnvals = self._build_spatial_cn_array()
        forcing = ThiessenRainCNForcing(
            cell_to_gauge=cell_to_gauge,
            gauge_hyetographs=hy_by_gauge_index,
            curve_number=cnvals,
            ia_ratio=0.2,
        )
        self._log(
            f"Thiessen rain/CN forcing active: gauges={len(gauges)}, "
            f"cells={cell_to_gauge.shape[0]}, cn_range=[{float(np.min(cnvals)):.1f}, {float(np.max(cnvals)):.1f}]"
        )
        return forcing

    def _nearest_cell_index_for_xy(self, x: float, y: float) -> int:
        cx, cy = self._mesh_cell_centroids()
        dx = cx - float(x)
        dy = cy - float(y)
        return int(np.argmin(dx * dx + dy * dy))

    def _build_pipe_network_config(self):
        if (
            self._mesh_data is None
            or not _HAVE_QGIS_CORE
            or PipeNetworkConfig is None
            or not hasattr(self, "drain_nodes_layer_combo")
        ):
            return None
        node_layer = self._combo_layer(self.drain_nodes_layer_combo, "vector")
        link_layer = self._combo_layer(self.drain_links_layer_combo, "vector") if hasattr(self, "drain_links_layer_combo") else None
        inlet_layer = self._combo_layer(self.drain_inlets_layer_combo, "vector") if hasattr(self, "drain_inlets_layer_combo") else None
        if node_layer is None or link_layer is None:
            return None

        node_fields = set(node_layer.fields().names())
        nodes: List[DrainageNode] = []
        for ft in node_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                pt = geom.asPoint()
            except Exception:
                continue
            node_id = str(ft["node_id"] if "node_id" in node_fields else ft.id()).strip()
            if not node_id:
                continue
            nodes.append(
                DrainageNode(
                    node_id=node_id,
                    x=float(pt.x()),
                    y=float(pt.y()),
                    invert_elev=float(ft["invert_elev"] if "invert_elev" in node_fields else 0.0),
                    max_depth=float(ft["max_depth"] if "max_depth" in node_fields else 2.0),
                    node_type=str(ft["node_type"] if "node_type" in node_fields else "junction").strip() or "junction",
                    metadata={
                        "surface_area_m2": float(ft["surface_area_m2"] if "surface_area_m2" in node_fields else 50.0)
                    },
                )
            )
        if not nodes:
            return None

        link_fields = set(link_layer.fields().names())
        links: List[DrainageLink] = []
        for ft in link_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            link_id = str(ft["link_id"] if "link_id" in link_fields else ft.id()).strip()
            from_node = str(ft["from_node"] if "from_node" in link_fields else "").strip()
            to_node = str(ft["to_node"] if "to_node" in link_fields else "").strip()
            if not link_id or not from_node or not to_node:
                continue
            links.append(
                DrainageLink(
                    link_id=link_id,
                    from_node_id=from_node,
                    to_node_id=to_node,
                    link_type=str(ft["link_type"] if "link_type" in link_fields else "conduit").strip() or "conduit",
                    length_m=float(ft["length_m"]) if "length_m" in link_fields and ft["length_m"] not in (None, "") else float(geom.length()),
                    roughness_n=float(ft["roughness_n"] if "roughness_n" in link_fields and ft["roughness_n"] not in (None, "") else 0.013),
                    diameter_m=float(ft["diameter_m"]) if "diameter_m" in link_fields and ft["diameter_m"] not in (None, "") else None,
                    max_flow_cms=float(ft["max_flow_cms"]) if "max_flow_cms" in link_fields and ft["max_flow_cms"] not in (None, "") else None,
                    metadata={
                        "cd": float(ft["cd"] if "cd" in link_fields and ft["cd"] not in (None, "") else 0.75)
                    },
                )
            )
        if not links:
            return None

        inlets: List[InletExchange] = []
        if inlet_layer is not None:
            inlet_fields = set(inlet_layer.fields().names())
            for ft in inlet_layer.getFeatures():
                geom = ft.geometry()
                if geom is None or geom.isEmpty():
                    continue
                try:
                    pt = geom.asPoint()
                except Exception:
                    try:
                        c = geom.centroid()
                        pt = c.asPoint() if c is not None and not c.isEmpty() else None
                    except Exception:
                        pt = None
                if pt is None:
                    continue
                node_id = str(ft["node_id"] if "node_id" in inlet_fields else "").strip()
                if not node_id:
                    continue
                inlets.append(
                    InletExchange(
                        inlet_id=str(ft["inlet_id"] if "inlet_id" in inlet_fields else ft.id()).strip(),
                        cell_id=self._nearest_cell_index_for_xy(float(pt.x()), float(pt.y())),
                        node_id=node_id,
                        crest_elev=float(ft["crest_elev"] if "crest_elev" in inlet_fields and ft["crest_elev"] not in (None, "") else 0.0),
                        width_m=float(ft["width_m"] if "width_m" in inlet_fields and ft["width_m"] not in (None, "") else 1.0),
                        coefficient=float(ft["coefficient"] if "coefficient" in inlet_fields and ft["coefficient"] not in (None, "") else 0.62),
                        max_capture_cms=float(ft["max_capture_cms"]) if "max_capture_cms" in inlet_fields and ft["max_capture_cms"] not in (None, "") else None,
                    )
                )

        self._log(f"Drainage coupling configured: nodes={len(nodes)}, links={len(links)}, inlets={len(inlets)}")
        return PipeNetworkConfig(enabled=True, nodes=nodes, links=links, inlets=inlets)

    def _build_hydraulic_structure_config(self):
        if (
            self._mesh_data is None
            or not _HAVE_QGIS_CORE
            or HydraulicStructureConfig is None
            or StructureType is None
            or not hasattr(self, "structures_layer_combo")
        ):
            return None
        layer = self._combo_layer(self.structures_layer_combo, "vector")
        if layer is None:
            return None
        fields = set(layer.fields().names())
        structures: List[HydraulicStructure] = []
        type_name_map = {
            "weir": StructureType.WEIR,
            "culvert": StructureType.CULVERT,
            "gate": StructureType.GATE,
            "bridge": StructureType.BRIDGE,
            "pump": StructureType.PUMP,
        }
        for ft in layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                if "enabled" in fields and int(ft["enabled"]) <= 0:
                    continue
            except Exception:
                pass
            try:
                p0 = geom.interpolate(0.0).asPoint()
                p1 = geom.interpolate(max(0.0, float(geom.length()) - 1.0e-9)).asPoint()
            except Exception:
                continue
            raw_type = ft["structure_type"] if "structure_type" in fields else 2
            if isinstance(raw_type, str):
                structure_type = type_name_map.get(raw_type.strip().lower(), StructureType.CULVERT)
            else:
                try:
                    structure_type = StructureType(int(raw_type))
                except Exception:
                    structure_type = StructureType.CULVERT
            metadata = {}
            for key in ("width_m", "height_m", "diameter_m", "length_m", "roughness_n", "coeff", "cd", "opening", "q_pump_cms", "max_flow_cms"):
                if key in fields and ft[key] not in (None, ""):
                    try:
                        metadata[key] = float(ft[key])
                    except Exception:
                        pass
            structures.append(
                HydraulicStructure(
                    structure_id=str(ft["structure_id"] if "structure_id" in fields else ft.id()).strip(),
                    structure_type=structure_type,
                    upstream_cell=self._nearest_cell_index_for_xy(float(p0.x()), float(p0.y())),
                    downstream_cell=self._nearest_cell_index_for_xy(float(p1.x()), float(p1.y())),
                    crest_elev=float(ft["crest_elev"] if "crest_elev" in fields and ft["crest_elev"] not in (None, "") else 0.0),
                    enabled=True,
                    metadata=metadata,
                )
            )
        if not structures:
            return None
        self._log(f"Hydraulic structures configured: count={len(structures)}")
        return HydraulicStructureConfig(enabled=True, structures=structures)

    def _combo_current_layer_name(self, combo) -> str:
        lyr = self._combo_layer(combo, "vector")
        if lyr is None:
            return ""
        try:
            return str(lyr.name() or "")
        except Exception:
            return ""

    def _set_combo_by_layer_name(self, combo, layer_name: str) -> bool:
        target = str(layer_name or "").strip().lower()
        if not target:
            return False
        for i in range(combo.count()):
            label = str(combo.itemText(i) or "").strip().lower()
            if label == target:
                combo.setCurrentIndex(i)
                return True
        return False

    def _missing_required_fields(self, layer, required_fields: Sequence[str]) -> List[str]:
        if layer is None:
            return list(required_fields)
        have = {str(n).strip().lower() for n in layer.fields().names()}
        return [f for f in required_fields if str(f).strip().lower() not in have]

    def _persist_model_layer_bindings(self, gpkg_path: str):
        if not gpkg_path or not os.path.exists(gpkg_path):
            return
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_model_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_layer_bindings (
                    role TEXT PRIMARY KEY,
                    layer_name TEXT NOT NULL,
                    geometry_type TEXT,
                    required_fields TEXT,
                    updated_utc TEXT NOT NULL
                )
                """
            )

            now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            cur.execute(
                "INSERT OR REPLACE INTO swe2d_model_metadata(key, value, updated_utc) VALUES (?, ?, ?)",
                ("swe2d_coupling_schema_version", "1", now),
            )
            cur.execute(
                "INSERT OR REPLACE INTO swe2d_model_metadata(key, value, updated_utc) VALUES (?, ?, ?)",
                ("swe2d_coupling_layer_roles", json.dumps(sorted(_MODEL_LAYER_BINDINGS.keys())), now),
            )

            for role, spec in _MODEL_LAYER_BINDINGS.items():
                combo_attr = str(spec.get("combo_attr", ""))
                combo = getattr(self, combo_attr, None)
                selected_name = self._combo_current_layer_name(combo) if combo is not None else ""
                layer_name = selected_name or str(spec.get("layer_name", ""))
                cur.execute(
                    """
                    INSERT OR REPLACE INTO swe2d_layer_bindings(
                        role, layer_name, geometry_type, required_fields, updated_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        role,
                        layer_name,
                        str(spec.get("geometry", "")),
                        ",".join(spec.get("required_fields", ())),
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _restore_model_layer_bindings(self, gpkg_path: str) -> List[str]:
        warnings: List[str] = []
        if not gpkg_path or not os.path.exists(gpkg_path):
            return warnings
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_layer_bindings'"
            )
            if cur.fetchone() is None:
                return warnings
            cur.execute("SELECT role, layer_name, required_fields FROM swe2d_layer_bindings")
            rows = cur.fetchall()
        finally:
            conn.close()

        for role, layer_name, req_csv in rows:
            spec = _MODEL_LAYER_BINDINGS.get(str(role))
            if spec is None:
                continue
            combo = getattr(self, str(spec.get("combo_attr", "")), None)
            if combo is None:
                continue
            if layer_name:
                self._set_combo_by_layer_name(combo, str(layer_name))
            layer = self._combo_layer(combo, "vector")
            req_fields = [s for s in str(req_csv or "").split(",") if s.strip()]
            if layer is None:
                warnings.append(f"{role}: bound layer not found ({layer_name})")
                continue
            missing = self._missing_required_fields(layer, req_fields)
            if missing:
                warnings.append(f"{role}: missing fields in {layer.name()}: {', '.join(missing)}")
        return warnings

    def _preview_coupling_configuration(self):
        if self._mesh_data is None:
            QtWidgets.QMessageBox.information(
                self,
                "Coupling Preview",
                "Generate or load a mesh first so cell-based coupling indices can be resolved.",
            )
            return
        pipe_cfg = self._build_pipe_network_config()
        struct_cfg = self._build_hydraulic_structure_config()
        if pipe_cfg is None and struct_cfg is None:
            QtWidgets.QMessageBox.information(
                self,
                "Coupling Preview",
                "No valid drainage or structure layers are configured.",
            )
            return

        lines: List[str] = []
        if pipe_cfg is not None:
            lines.append(
                f"Drainage network: nodes={len(pipe_cfg.nodes)}, links={len(pipe_cfg.links)}, inlets={len(pipe_cfg.inlets)}"
            )
        else:
            lines.append("Drainage network: not configured")

        if struct_cfg is not None:
            lines.append(f"Hydraulic structures: count={len(struct_cfg.structures)}")
        else:
            lines.append("Hydraulic structures: not configured")

        if pack_coupling_soa is not None:
            soa = pack_coupling_soa(
                n_cells=int(self._mesh_cell_areas().shape[0]),
                pipe_network=pipe_cfg,
                hydraulic_structures=struct_cfg,
            )
            if soa.drainage is not None:
                dn = soa.drainage
                invalid_links = int(np.sum((dn.link_from < 0) | (dn.link_to < 0)))
                invalid_inlets = int(np.sum((dn.inlet_cell < 0) | (dn.inlet_node < 0)))
                lines.append(
                    "Drainage SoA: "
                    f"nodes={dn.node_x.size}, links={dn.link_from.size}, inlets={dn.inlet_cell.size}, "
                    f"invalid_links={invalid_links}, invalid_inlets={invalid_inlets}"
                )
            if soa.structures is not None:
                ss = soa.structures
                invalid_struct = int(np.sum((ss.upstream_cell < 0) | (ss.downstream_cell < 0)))
                lines.append(
                    "Structures SoA: "
                    f"count={ss.structure_type.size}, invalid_cell_pairs={invalid_struct}"
                )

        QtWidgets.QMessageBox.information(self, "Coupling Preview", "\n".join(lines))

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

        include_extra = bool(getattr(self, "extended_outputs_chk", None) is None or self.extended_outputs_chk.isChecked())

        with _h5py.File(path, "w") as f:
            f.attrs["File Type"] = np.bytes_(b"HEC-RAS Results")
            f.attrs["File Version"] = np.bytes_(b"HEC-RAS 7.0 April 2026")
            f.attrs["Units System"] = np.bytes_(
                b"US Customary" if self._is_us_customary_units() else b"SI"
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
            geo.attrs["SI Units"] = np.bytes_(b"False" if self._is_us_customary_units() else b"True")
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
            if include_extra:
                if self._result_data is not None and "n_mann_cell" in self._result_data:
                    n_face = np.asarray(self._result_data["n_mann_cell"], dtype=np.float64)[:n_cells]
                else:
                    n_face = np.full(n_cells, float(self.n_mann_spin.value()), dtype=np.float64)
                area_grp.create_dataset("Cells Manning n", data=n_face.astype(np.float32))
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
                if include_extra:
                    mom_u_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                    mom_v_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                    qmag_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                    wet_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                    froude_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                    h_min = float(self.h_min_spin.value())
                    g = float(self._gravity)

                for ti, (_, h, hu, hv) in enumerate(timesteps):
                    h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                    hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                    hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                    wet = (h_f > h_min)
                    hmag = np.maximum(h_f, 1e-12)
                    u = np.where(wet, hu_f / hmag, 0.0)
                    v = np.where(wet, hv_f / hmag, 0.0)
                    depth_arr[ti] = h_f.astype(np.float32)
                    wse_arr[ti] = (h_f + cell_min_z[:n_cells]).astype(np.float32)
                    vel_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                    vel_u_arr[ti] = u.astype(np.float32)
                    vel_v_arr[ti] = v.astype(np.float32)
                    if include_extra:
                        mom_u_arr[ti] = hu_f.astype(np.float32)
                        mom_v_arr[ti] = hv_f.astype(np.float32)
                        qmag_arr[ti] = np.sqrt(hu_f ** 2 + hv_f ** 2).astype(np.float32)
                        wet_arr[ti] = wet.astype(np.float32)
                        froude_arr[ti] = np.where(wet, np.sqrt(u ** 2 + v ** 2) / np.sqrt(np.maximum(g * h_f, 1.0e-12)), 0.0).astype(np.float32)

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
                if include_extra:
                    ar.create_dataset("Cell Momentum - X", data=mom_u_arr)
                    ar.create_dataset("Cell Momentum - Y", data=mom_v_arr)
                    ar.create_dataset("Unit Discharge - Magnitude", data=qmag_arr)
                    ar.create_dataset("Wet Mask", data=wet_arr)
                    ar.create_dataset("Cell Froude Number", data=froude_arr)

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

        include_extra = bool(getattr(self, "extended_outputs_chk", None) is None or self.extended_outputs_chk.isChecked())

        with _netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            # Global attributes (CF + UGRID)
            ds.Conventions = "CF-1.8 UGRID-1.0"
            ds.title = "SWE2D backwater model results"
            ds.institution = "qgis-backwater-plugin"
            ds.history = "Created by swe2d_workbench_qt"
            ds.featureType = "mesh2D"
            len_unit = self._length_unit_name if self._length_unit_name else "m"
            vel_unit = f"{len_unit} s-1"
            mom_unit = f"{len_unit}2 s-1"
            manning_unit = "s ft-1/3" if self._is_us_customary_units() else "s m-1/3"

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
            nx_var.units = len_unit
            nx_var.mesh = "mesh2d"
            nx_var.location = "node"
            nx_var.grid_mapping = "crs"
            nx_var[:] = node_x.astype(np.float64)

            ny_var = ds.createVariable("node_y", "f8", ("node",))
            ny_var.standard_name = "projection_y_coordinate"
            ny_var.units = len_unit
            ny_var.mesh = "mesh2d"
            ny_var.location = "node"
            ny_var.grid_mapping = "crs"
            ny_var[:] = node_y.astype(np.float64)

            nz_var = ds.createVariable("node_z", "f8", ("node",))
            nz_var.standard_name = "altitude"
            nz_var.long_name = "bed elevation at node"
            nz_var.units = len_unit
            nz_var.mesh = "mesh2d"
            nz_var.location = "node"
            nz_var.grid_mapping = "crs"
            nz_var[:] = node_z.astype(np.float64)

            # Face centroid coordinates
            fx_var = ds.createVariable("face_x", "f8", ("face",))
            fx_var.standard_name = "projection_x_coordinate"
            fx_var.units = len_unit
            fx_var.mesh = "mesh2d"
            fx_var.location = "face"
            fx_var.grid_mapping = "crs"
            fx_var[:] = cell_cx.astype(np.float64)

            fy_var = ds.createVariable("face_y", "f8", ("face",))
            fy_var.standard_name = "projection_y_coordinate"
            fy_var.units = len_unit
            fy_var.mesh = "mesh2d"
            fy_var.location = "face"
            fy_var.grid_mapping = "crs"
            fy_var[:] = cell_cy.astype(np.float64)

            # Face minimum bed elevation
            fz_var = ds.createVariable("face_z", "f8", ("face",))
            fz_var.long_name = "minimum bed elevation at face"
            fz_var.units = len_unit
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
                if include_extra:
                    mom_u_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                    mom_v_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                    qmag_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                    wet_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                    froude_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                    h_min = float(self.h_min_spin.value())
                    g = float(self._gravity)

                for ti, (_, h, hu, hv) in enumerate(timesteps):
                    h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                    hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                    hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                    wet = (h_f > h_min)
                    hmag = np.maximum(h_f, 1e-12)
                    u = np.where(wet, hu_f / hmag, 0.0)
                    v = np.where(wet, hv_f / hmag, 0.0)
                    depth_arr[ti] = h_f.astype(np.float32)
                    wse_arr[ti] = (h_f + cell_min_z[:n_cells]).astype(np.float32)
                    vel_u_arr[ti] = u.astype(np.float32)
                    vel_v_arr[ti] = v.astype(np.float32)
                    vel_mag_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                    if include_extra:
                        mom_u_arr[ti] = hu_f.astype(np.float32)
                        mom_v_arr[ti] = hv_f.astype(np.float32)
                        qmag_arr[ti] = np.sqrt(hu_f ** 2 + hv_f ** 2).astype(np.float32)
                        wet_arr[ti] = wet.astype(np.float32)
                        froude_arr[ti] = np.where(wet, np.sqrt(u ** 2 + v ** 2) / np.sqrt(np.maximum(g * h_f, 1.0e-12)), 0.0).astype(np.float32)

                d_var = ds.createVariable(
                    "water_depth", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                d_var.standard_name = "water_depth"
                d_var.long_name = "water depth"
                d_var.units = len_unit
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
                w_var.units = len_unit
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
                u_var.units = vel_unit
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
                v_var.units = vel_unit
                v_var.mesh = "mesh2d"
                v_var.location = "face"
                v_var.coordinates = "face_x face_y"
                v_var.grid_mapping = "crs"
                v_var[:] = vel_v_arr

                vm_var = ds.createVariable(
                    "velocity_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                vm_var.long_name = "velocity magnitude"
                vm_var.units = vel_unit
                vm_var.mesh = "mesh2d"
                vm_var.location = "face"
                vm_var.coordinates = "face_x face_y"
                vm_var.grid_mapping = "crs"
                vm_var[:] = vel_mag_arr

                if include_extra:
                    mu_var = ds.createVariable(
                        "momentum_x", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                    )
                    mu_var.long_name = "x momentum per unit width"
                    mu_var.units = mom_unit
                    mu_var.mesh = "mesh2d"
                    mu_var.location = "face"
                    mu_var.coordinates = "face_x face_y"
                    mu_var.grid_mapping = "crs"
                    mu_var[:] = mom_u_arr

                    mv_var = ds.createVariable(
                        "momentum_y", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                    )
                    mv_var.long_name = "y momentum per unit width"
                    mv_var.units = mom_unit
                    mv_var.mesh = "mesh2d"
                    mv_var.location = "face"
                    mv_var.coordinates = "face_x face_y"
                    mv_var.grid_mapping = "crs"
                    mv_var[:] = mom_v_arr

                    qmag_var = ds.createVariable(
                        "unit_discharge_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                    )
                    qmag_var.long_name = "unit discharge magnitude"
                    qmag_var.units = mom_unit
                    qmag_var.mesh = "mesh2d"
                    qmag_var.location = "face"
                    qmag_var.coordinates = "face_x face_y"
                    qmag_var.grid_mapping = "crs"
                    qmag_var[:] = qmag_arr

                    wet_var = ds.createVariable(
                        "wet_mask", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                    )
                    wet_var.long_name = "wet mask"
                    wet_var.units = "1"
                    wet_var.mesh = "mesh2d"
                    wet_var.location = "face"
                    wet_var.coordinates = "face_x face_y"
                    wet_var.grid_mapping = "crs"
                    wet_var[:] = wet_arr

                    fr_var = ds.createVariable(
                        "froude_number", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                    )
                    fr_var.long_name = "Froude number"
                    fr_var.units = "1"
                    fr_var.mesh = "mesh2d"
                    fr_var.location = "face"
                    fr_var.coordinates = "face_x face_y"
                    fr_var.grid_mapping = "crs"
                    fr_var[:] = froude_arr

            if include_extra:
                if self._result_data is not None and "n_mann_cell" in self._result_data:
                    n_face = np.asarray(self._result_data["n_mann_cell"], dtype=np.float64)[:n_cells]
                else:
                    n_face = np.full(n_cells, float(self.n_mann_spin.value()), dtype=np.float64)
                n_var = ds.createVariable("manning_n_face", "f4", ("face",), fill_value=np.float32(-9999.0))
                n_var.long_name = "Manning roughness at face"
                n_var.units = manning_unit
                n_var.mesh = "mesh2d"
                n_var.location = "face"
                n_var.coordinates = "face_x face_y"
                n_var.grid_mapping = "crs"
                n_var[:] = n_face.astype(np.float32)

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

            if self._line_snapshot_rows:
                gpkg_results_path = self._current_line_results_storage_path()
                if gpkg_results_path:
                    snap_run_id = datetime.datetime.utcnow().strftime("swe2d_snapshot_%Y%m%dT%H%M%SZ")
                    mesh_interval_s = max(1.0, self._parse_time_hours(self.output_interval_edit.text()) * 3600.0)
                    line_interval_s = max(1.0, self._parse_time_hours(self.line_output_interval_edit.text()) * 3600.0)
                    self._persist_line_results_to_geopackage(
                        gpkg_results_path,
                        snap_run_id,
                        self._line_snapshot_rows,
                        profile_rows=self._line_snapshot_profile_rows,
                        mesh_interval_s=mesh_interval_s,
                        line_interval_s=line_interval_s,
                    )
                    self._log(
                        f"Sample line snapshot stored → {gpkg_results_path} "
                        f"(ts_rows={len(self._line_snapshot_rows)}, profile_rows={len(self._line_snapshot_profile_rows)}, run_id={snap_run_id})"
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

    def _inflow_adjacent_cells(self, bc_n0: np.ndarray, bc_n1: np.ndarray, bc_tp: np.ndarray) -> np.ndarray:
        """Return indices of cells that own at least one inflow boundary edge."""
        _INFLOW_TYPES = {2, 6, 102}  # INFLOW_Q, NORMAL_DEPTH, TS_FLOW
        inflow_mask = np.isin(bc_tp.astype(np.int32), list(_INFLOW_TYPES))
        if not np.any(inflow_mask):
            return np.empty(0, dtype=np.int32)

        inflow_n0 = set(int(v) for v in bc_n0[inflow_mask])
        inflow_n1 = set(int(v) for v in bc_n1[inflow_mask])
        inflow_nodes = inflow_n0 | inflow_n1

        hit: List[int] = []
        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            for ci in range(offs.size - 1):
                s = int(offs[ci])
                e = int(offs[ci + 1])
                poly = faces[s:e]
                for k in range(poly.size):
                    a = int(poly[k])
                    b = int(poly[(k + 1) % poly.size])
                    key_a = min(a, b)
                    key_b = max(a, b)
                    if key_a in inflow_nodes and key_b in inflow_nodes:
                        hit.append(ci)
                        break
        else:
            tris = self._mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
            for ci, tri in enumerate(tris):
                for k in range(3):
                    a = int(tri[k])
                    b = int(tri[(k + 1) % 3])
                    key_a = min(a, b)
                    key_b = max(a, b)
                    if key_a in inflow_nodes and key_b in inflow_nodes:
                        hit.append(ci)
                        break
        return np.asarray(hit, dtype=np.int32)

    def _initial_state(self, bc_n0: Optional[np.ndarray] = None, bc_n1: Optional[np.ndarray] = None, bc_tp: Optional[np.ndarray] = None):
        assert self._mesh_data is not None
        cell_x, _ = self._mesh_cell_centroids()
        h0 = np.zeros_like(cell_x, dtype=np.float64)
        mode = str(self.initial_condition_combo.currentData() if hasattr(self, "initial_condition_combo") else "dry")
        if mode == "uniform_depth":
            h0[:] = max(0.0, float(self.initial_depth_spin.value()))
        elif mode == "uniform_wse":
            bed = self._mesh_cell_min_bed().astype(np.float64)
            wse0 = float(self.initial_wse_spin.value())
            h0 = np.maximum(0.0, wse0 - bed)
        elif mode == "dry" and bc_n0 is not None and bc_n1 is not None and bc_tp is not None:
            # Dry start with inflow BCs: seed boundary-adjacent cells with a tiny depth
            # so that the Riemann solver at boundary edges has non-zero interior state to
            # work with.  Without this the explicit flux kernel sees h_L=0 and produces
            # zero inflow flux regardless of the prescribed BC.
            h_min_val = float(self.h_min_spin.value()) if hasattr(self, "h_min_spin") else 1.0e-6
            prime_depth = max(h_min_val * 100.0, 1.0e-4)
            adj = self._inflow_adjacent_cells(bc_n0, bc_n1, bc_tp)
            if adj.size > 0:
                h0[adj] = prime_depth
                self._log(
                    f"Dry start: primed {adj.size} inflow-adjacent cell(s) with h={prime_depth:.2e} m "
                    "to enable boundary-driven wetting."
                )
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
            h0, hu0, hv0 = self._initial_state(bc_n0=bc_n0, bc_n1=bc_n1, bc_tp=bc_tp)
            n_mann_cell = self._build_spatial_manning_array()
            self._update_unit_system_from_crs()

            run_duration_s = self._parse_run_duration_seconds()
            dt_cfg = float(self.dt_spin.value())
            adaptive_cfl_dt = bool(self.adaptive_cfl_dt_chk.isChecked())
            dt_fixed = -1.0 if adaptive_cfl_dt else dt_cfg
            dt_request = -1.0 if adaptive_cfl_dt else dt_cfg
            reconstruction_mode = int(self.reconstruction_combo.currentData())
            reconstruction_name = self.reconstruction_combo.currentText().strip()
            coupling_loop_mode = str(self.coupling_loop_combo.currentData() if hasattr(self, "coupling_loop_combo") else "cpu")
            rain_rate_model = self._rain_rate_si_to_model(float(self.rain_rate_spin.value()) / 1000.0 / 3600.0)
            cell_source_si = self._build_internal_flow_source_cms()
            cell_source_model = self._flow_si_to_model(cell_source_si) if cell_source_si is not None else None
            thiessen_forcing = self._build_thiessen_rain_cn_forcing()
            pipe_network_cfg = self._build_pipe_network_config()
            hydraulic_structures_cfg = self._build_hydraulic_structure_config()

            if self._model_gpkg_path and os.path.exists(self._model_gpkg_path):
                try:
                    self._persist_model_layer_bindings(self._model_gpkg_path)
                except Exception as exc:
                    self._log(f"Model coupling metadata persist warning: {exc}")

            coupling_soa = None
            if pack_coupling_soa is not None:
                coupling_soa = pack_coupling_soa(
                    n_cells=int(self._mesh_cell_areas().shape[0]),
                    pipe_network=pipe_network_cfg,
                    hydraulic_structures=hydraulic_structures_cfg,
                )
            coupling_controller = None
            if SWE2DCouplingController is not None and (pipe_network_cfg is not None or hydraulic_structures_cfg is not None):
                drainage_mod = SWE2DUrbanDrainageModule(pipe_network_cfg) if pipe_network_cfg is not None and SWE2DUrbanDrainageModule is not None else None
                if drainage_mod is not None:
                    drainage_mod.initialize()
                structures_mod = SWE2DStructureModule(hydraulic_structures_cfg) if hydraulic_structures_cfg is not None and SWE2DStructureModule is not None else None
                coupling_controller = SWE2DCouplingController(
                    cell_area_m2=self._mesh_cell_areas(),
                    cell_bed_m=self._mesh_cell_min_bed(),
                    drainage=drainage_mod,
                    structures=structures_mod,
                    coupling_loop=coupling_loop_mode,
                )
            rain_stats_acc = {"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0}

            # Snapshot output interval — clamp to at least 1 s to avoid div-by-zero
            _oi_hr = self._parse_time_hours(self.output_interval_edit.text())
            output_interval_s = max(1.0, _oi_hr * 3600.0)
            _line_oi_hr = self._parse_time_hours(self.line_output_interval_edit.text())
            line_output_interval_s = max(1.0, _line_oi_hr * 3600.0)
            self._snapshot_timesteps = []
            self._line_snapshot_rows = []
            self._line_snapshot_profile_rows = []
            _next_snap_t = output_interval_s
            _next_line_snap_t = line_output_interval_s
            sample_map = self._build_line_sampling_map()
            cell_min_z = self._mesh_cell_min_bed() if sample_map else None
            run_id = datetime.datetime.utcnow().strftime("swe2d_%Y%m%dT%H%M%SZ")

            dynamic_bc = bool(np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs)
            if dynamic_bc:
                self._log("Timeseries BC mode active (flow/stage hydrographs).")

            self._log("Starting 2D run...")
            self._log(f"Reconstruction mode: {reconstruction_name}")
            self._log(
                f"Output intervals: mesh={output_interval_s:.1f}s, sample-lines={line_output_interval_s:.1f}s"
            )
            self._log(
                "Stability controls: "
                f"max_rel_dh={float(self.max_rel_depth_increase_spin.value()):.3f}, "
                f"shallow_damp_h={float(self.shallow_damping_depth_spin.value()):.6e}, "
                f"depth_cap={float(self.depth_cap_spin.value()):.3f}, "
                f"mom_cap_min={float(self.momentum_cap_min_speed_spin.value()):.3f}, "
                f"mom_cap_mult={float(self.momentum_cap_celerity_mult_spin.value()):.3f}, "
                f"invA_cap={float(self.max_inv_area_spin.value()):.3e}, "
                f"lambda_cap={float(self.cfl_lambda_cap_spin.value()):.3e}"
            )
            if adaptive_cfl_dt:
                self._log(f"Timestep mode: variable CFL (dt_max={dt_cfg:.5f} s)")
            else:
                self._log(f"Timestep mode: fixed dt ({dt_cfg:.5f} s)")
            if float(np.asarray(rain_rate_model, dtype=np.float64)) > 0.0:
                self._log(
                    f"Rain-on-grid active: {float(self.rain_rate_spin.value()):.3f} mm/hr "
                    f"(applied as {float(np.asarray(rain_rate_model, dtype=np.float64)):.6e} {self._length_unit_name}/s)"
                )
            if thiessen_forcing is not None:
                self._log("Spatial rainfall forcing active: Thiessen nearest-gage interpolation + NRCS CN infiltration.")
            if cell_source_model is not None:
                self._log(
                    f"Internal source/sink forcing active: total_Q={float(np.sum(cell_source_model)):.6f} {self._flow_unit_label()}"
                )
            if coupling_controller is not None:
                self._log(
                    "Coupled drainage/structure forcing active: "
                    f"drainage={pipe_network_cfg is not None}, structures={hydraulic_structures_cfg is not None}, "
                    f"loop={coupling_loop_mode}"
                )
            if coupling_soa is not None:
                dn = coupling_soa.drainage
                ss = coupling_soa.structures
                if dn is not None:
                    bad_links = int(np.sum((dn.link_from < 0) | (dn.link_to < 0)))
                    bad_inlets = int(np.sum((dn.inlet_cell < 0) | (dn.inlet_node < 0)))
                    self._log(
                        "CUDA SoA pack (drainage): "
                        f"nodes={dn.node_x.size}, links={dn.link_from.size}, inlets={dn.inlet_cell.size}, "
                        f"invalid_links={bad_links}, invalid_inlets={bad_inlets}"
                    )
                if ss is not None:
                    bad_struct = int(np.sum((ss.upstream_cell < 0) | (ss.downstream_cell < 0)))
                    self._log(
                        "CUDA SoA pack (structures): "
                        f"count={ss.structure_type.size}, invalid_cell_pairs={bad_struct}"
                    )
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
                dt_fixed=dt_fixed,
                dt_max=dt_cfg,
                max_inv_area=float(self.max_inv_area_spin.value()),
                cfl_lambda_cap=float(self.cfl_lambda_cap_spin.value()),
                momentum_cap_min_speed=float(self.momentum_cap_min_speed_spin.value()),
                momentum_cap_celerity_mult=float(self.momentum_cap_celerity_mult_spin.value()),
                depth_cap=float(self.depth_cap_spin.value()),
                max_rel_depth_increase=float(self.max_rel_depth_increase_spin.value()),
                shallow_damping_depth=float(self.shallow_damping_depth_spin.value()),
                spatial_discretization=reconstruction_mode,
                degen_mode=int(self.degen_mode_combo.currentData()),
                front_flux_damping=float(self.front_flux_damping_spin.value()),
                active_set_hysteresis=bool(self.active_set_hysteresis_chk.isChecked()),
            )

            last_diag = None
            t_accum = 0.0
            i = 0
            if dynamic_bc and not backend.supports_dynamic_boundary_update():
                raise RuntimeError("Native module does not support dynamic boundary updates. Rebuild backwater_swe2d.")

            native_bc_forcing = False
            native_rain_cn_forcing = False

            if dynamic_bc and hasattr(backend, "set_boundary_hydrographs_native"):
                try:
                    progressive = True
                    if hasattr(self, "inflow_progressive_chk") and self.inflow_progressive_chk is not None:
                        progressive = bool(self.inflow_progressive_chk.isChecked())

                    node_x = self._mesh_data["node_x"]
                    node_y = self._mesh_data["node_y"]
                    xmin = float(np.min(node_x))
                    xmax = float(np.max(node_x))
                    ymin = float(np.min(node_y))
                    ymax = float(np.max(node_y))
                    mx = 0.5 * (node_x[bc_n0] + node_x[bc_n1])
                    my = 0.5 * (node_y[bc_n0] + node_y[bc_n1])
                    d = np.vstack([
                        np.abs(mx - xmin),
                        np.abs(mx - xmax),
                        np.abs(my - ymin),
                        np.abs(my - ymax),
                    ])
                    side_idx = np.argmin(d, axis=0)
                    side_names = ["left", "right", "bottom", "top"]
                    edge_len = np.hypot(node_x[bc_n1] - node_x[bc_n0], node_y[bc_n1] - node_y[bc_n0])

                    def _is_flow(tp_val: int) -> bool:
                        return int(tp_val) == int(_BC_INFLOW_Q)

                    any_flow_hg = False
                    edge_rows: List[int] = []
                    edge_types: List[int] = []
                    edge_hgs: List[Tuple[np.ndarray, np.ndarray]] = []

                    for bi in range(bc_n0.size):
                        hg_info = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
                        if hg_info is not None:
                            tp_i, hg_i = hg_info
                            edge_rows.append(int(bi))
                            edge_types.append(int(tp_i))
                            edge_hgs.append(hg_i)
                            any_flow_hg = any_flow_hg or _is_flow(int(tp_i))
                            continue
                        side = side_names[int(side_idx[bi])]
                        if side in side_hydrographs:
                            tp_i = int(bc_tp[bi])
                            edge_rows.append(int(bi))
                            edge_types.append(tp_i)
                            edge_hgs.append(side_hydrographs[side])
                            any_flow_hg = any_flow_hg or _is_flow(tp_i)

                    if edge_rows and not (progressive and any_flow_hg):
                        edge_index = np.empty(len(edge_rows), dtype=np.int32)
                        bc_type_native = np.asarray(edge_types, dtype=np.int32)
                        offsets = [0]
                        t_all: List[np.ndarray] = []
                        v_all: List[np.ndarray] = []

                        # Precompute fixed flow scaling by hydrograph source group.
                        flow_scale: Dict[int, float] = {}
                        for j, bi in enumerate(edge_rows):
                            key = -1000000 - bi
                            hg_info = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
                            if hg_info is None:
                                key = int(side_idx[bi])
                            if key not in flow_scale:
                                if key < 0:
                                    total_len = max(float(edge_len[bi]), 1.0e-9)
                                else:
                                    mask = side_idx == key
                                    total_len = max(float(np.sum(edge_len[mask])), 1.0e-9)
                                flow_scale[key] = total_len

                        for j, bi in enumerate(edge_rows):
                            a = int(bc_n0[bi])
                            b = int(bc_n1[bi])
                            keyn = (a, b) if a < b else (b, a)
                            edge_index[j] = int(backend._boundary_edge_index_by_nodes[keyn])
                            t_i, v_i = edge_hgs[j]
                            t_i = np.asarray(t_i, dtype=np.float64).ravel()
                            v_i = np.asarray(v_i, dtype=np.float64).ravel()
                            if int(bc_type_native[j]) == int(_BC_INFLOW_Q):
                                flow_key = -1000000 - bi
                                if not (edge_hydrographs and int(bi) in edge_hydrographs):
                                    flow_key = int(side_idx[bi])
                                v_i = v_i / max(flow_scale.get(flow_key, 1.0), 1.0e-9)
                            t_all.append(t_i)
                            v_all.append(v_i)
                            offsets.append(offsets[-1] + int(t_i.size))

                        time_s_native = np.concatenate(t_all).astype(np.float64, copy=False)
                        value_native = np.concatenate(v_all).astype(np.float64, copy=False)
                        offsets_native = np.asarray(offsets, dtype=np.int32)
                        backend.set_boundary_hydrographs_native(
                            edge_index=edge_index,
                            bc_type=bc_type_native,
                            offsets=offsets_native,
                            time_s=time_s_native,
                            value=value_native,
                        )
                        native_bc_forcing = True
                        self._log(f"Native BC hydrograph forcing configured for {len(edge_rows)} boundary edges.")
                    elif edge_rows and progressive and any_flow_hg:
                        self._log("Native BC hydrographs skipped: progressive inflow activation is enabled for flow hydrographs.")
                except Exception as exc:
                    self._log(f"Native BC hydrograph forcing unavailable: {exc}")

            if thiessen_forcing is not None and hasattr(backend, "set_rain_cn_forcing_native"):
                try:
                    cell_gage_idx = np.asarray(thiessen_forcing.cell_to_gauge, dtype=np.int32).ravel()
                    cn_arr = np.asarray(thiessen_forcing.cn_model.curve_number, dtype=np.float64).ravel()
                    ia_ratio = float(thiessen_forcing.cn_model.ia_ratio)
                    unique = np.unique(cell_gage_idx[cell_gage_idx >= 0])
                    if unique.size > 0:
                        gage_map = {int(g): i for i, g in enumerate(unique.tolist())}
                        remap = np.full(cell_gage_idx.shape, -1, dtype=np.int32)
                        for g, gi in gage_map.items():
                            remap[cell_gage_idx == g] = int(gi)
                        offsets = [0]
                        t_all = []
                        c_all = []
                        for g in unique.tolist():
                            hy = thiessen_forcing.gauge_hyetographs.get(int(g))
                            if hy is None:
                                t_all.append(np.asarray([0.0], dtype=np.float64))
                                c_all.append(np.asarray([0.0], dtype=np.float64))
                            else:
                                t_all.append(np.asarray(hy.times_s, dtype=np.float64).ravel())
                                c_all.append(np.asarray(hy.cumulative_mm, dtype=np.float64).ravel())
                            offsets.append(offsets[-1] + int(t_all[-1].size))
                        backend.set_rain_cn_forcing_native(
                            cell_gage_idx=remap,
                            gage_offsets=np.asarray(offsets, dtype=np.int32),
                            hg_time_s=np.concatenate(t_all).astype(np.float64, copy=False),
                            hg_cum_mm=np.concatenate(c_all).astype(np.float64, copy=False),
                            cn=cn_arr,
                            ia_ratio=ia_ratio,
                            mm_to_model_depth=float(self._rain_mm_to_model_depth()),
                        )
                        native_rain_cn_forcing = True
                        self._log("Native rain+CN forcing configured for GPU timestep evaluation.")
                except Exception as exc:
                    self._log(f"Native rain+CN forcing unavailable: {exc}")

            while t_accum < run_duration_s:
                if self._cancel_requested:
                    break

                if dynamic_bc and not native_bc_forcing:
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

                last_diag = backend.step(dt_request)
                dt_used = float(last_diag.get("dt", dt_cfg))
                coupled_source_rate = None
                if coupling_controller is not None:
                    h_c, hu_c, hv_c = backend.get_state()
                    coupled_source_rate = coupling_controller.compute_source_rates(t_accum, dt_used, h_c, hu_c, hv_c)
                rain_src = rain_rate_model
                if thiessen_forcing is not None and not native_rain_cn_forcing:
                    rain_src_si, rain_diag = thiessen_forcing.step_net_rainfall_mps(t_accum, t_accum + dt_used)
                    rain_src = self._rain_rate_si_to_model(rain_src_si)
                    rain_stats_acc["rain_mm"] += float(rain_diag.get("rain_mm_mean", 0.0))
                    rain_stats_acc["excess_mm"] += float(rain_diag.get("excess_mm_mean", 0.0))
                    rain_stats_acc["samples"] += 1
                elif native_rain_cn_forcing:
                    rain_src = 0.0
                self._apply_external_sources(
                    backend,
                    dt_used,
                    rain_src,
                    cell_source_model,
                    coupled_source_rate,
                )
                t_accum += dt_used

                # Capture snapshot at each output interval boundary
                need_mesh_snap = t_accum >= _next_snap_t
                need_line_snap = bool(sample_map) and t_accum >= _next_line_snap_t
                if need_mesh_snap or need_line_snap:
                    h_s, hu_s, hv_s = backend.get_state()

                if need_mesh_snap:
                    self._snapshot_timesteps.append(
                        (t_accum, h_s.copy(), hu_s.copy(), hv_s.copy())
                    )
                    _next_snap_t += output_interval_s

                if need_line_snap and cell_min_z is not None:
                    rows, profile_rows = self._sample_line_metrics(
                        sample_map,
                        t_accum,
                        h_s,
                        hu_s,
                        hv_s,
                        cell_min_z,
                    )
                    if rows:
                        self._line_snapshot_rows.extend(rows)
                    if profile_rows:
                        self._line_snapshot_profile_rows.extend(profile_rows)
                    _next_line_snap_t += line_output_interval_s

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
                    if coupling_controller is not None:
                        cdiag = coupling_controller.last_diag
                        self._log(
                            "  coupling: "
                            f"drain_qmax={cdiag.drainage_max_link_flow_cms:.4f} cms, "
                            f"drain_hmax={cdiag.drainage_max_node_depth_m:.4f}, "
                            f"struct_qsum={cdiag.structure_total_flow_cms:.4f} cms, "
                            f"src_range=[{cdiag.source_min_mps:.3e}, {cdiag.source_max_mps:.3e}]"
                        )
                QtWidgets.QApplication.processEvents()

            h, hu, hv = backend.get_state()
            self._result_data = {
                "h": h,
                "hu": hu,
                "hv": hv,
                "n_mann_cell": n_mann_cell.copy() if n_mann_cell is not None else np.full(h.shape, float(self.n_mann_spin.value()), dtype=np.float64),
                "gpu_active": np.array(bool(backend.gpu_active())),
                "last_mass_total": np.array(float(last_diag.get("mass_total", -1.0) if last_diag else -1.0)),
            }

            gpkg_results_path = self._current_line_results_storage_path()
            if gpkg_results_path and self._line_snapshot_rows:
                self._persist_line_results_to_geopackage(
                    gpkg_results_path,
                    run_id,
                    self._line_snapshot_rows,
                    profile_rows=self._line_snapshot_profile_rows,
                    mesh_interval_s=output_interval_s,
                    line_interval_s=line_output_interval_s,
                )
            if thiessen_forcing is not None and rain_stats_acc["samples"] > 0:
                avg_r = rain_stats_acc["rain_mm"] / rain_stats_acc["samples"]
                avg_e = rain_stats_acc["excess_mm"] / rain_stats_acc["samples"]
                self._log(
                    "Spatial rain/CN summary: "
                    f"mean rain={avg_r:.3f} mm/step, mean excess={avg_e:.3f} mm/step"
                )
            self._log("Run complete." if not self._cancel_requested else "Run canceled by user.")
            h_min = float(self.h_min_spin.value())
            wet = (h > h_min)
            safe_h = np.maximum(h, 1.0e-12)
            vel_mag = np.where(wet, np.sqrt((hu / safe_h) ** 2 + (hv / safe_h) ** 2), 0.0)
            self._log(
                f"Depth range: {float(np.min(h)):.6f} .. {float(np.max(h)):.6f} | "
                f"Velocity mag max (wet cells): {float(np.max(vel_mag)):.6f}"
            )
            if self._line_snapshot_rows:
                self._log(
                    f"Sample line rows captured: ts={len(self._line_snapshot_rows)}, "
                    f"profile={len(self._line_snapshot_profile_rows)}"
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
            h_raw = self._result_data["h"]
            h = np.maximum(h_raw, 1.0e-12)
            hu = self._result_data["hu"]
            hv = self._result_data["hv"]
            h_min = float(self.h_min_spin.value()) if hasattr(self, "h_min_spin") else 1.0e-6
            wet = (h_raw > h_min)
            vals = np.where(wet, np.sqrt((hu / h) ** 2 + (hv / h) ** 2), 0.0)
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
