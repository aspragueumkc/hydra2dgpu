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
import traceback
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
        DrainageSolverMode,
        DrainageLink,
        DrainageNode,
        GodunovSolverMode,
        HydraulicStructure,
        HydraulicStructureConfig,
        InletExchange,
        InletType,
        NodeInletAssignment,
        OutfallExchange,
        PipeEndExchange,
        PipeNetworkConfig,
        StructureType,
        SpatialDiscretization,
        TemporalScheme,
    )
    from swe2d_structures import SWE2DStructureModule
except Exception:
    try:
        from .swe2d_coupling import SWE2DCouplingController, pack_coupling_soa
        from .swe2d_drainage_network import SWE2DUrbanDrainageModule
        from .swe2d_extensions import (
            DrainageSolverMode,
            DrainageLink,
            DrainageNode,
            GodunovSolverMode,
            HydraulicStructure,
            HydraulicStructureConfig,
            InletExchange,
            InletType,
            NodeInletAssignment,
            OutfallExchange,
            PipeEndExchange,
            PipeNetworkConfig,
            StructureType,
            SpatialDiscretization,
            TemporalScheme,
        )
        from .swe2d_structures import SWE2DStructureModule
    except Exception:
        SWE2DCouplingController = None
        pack_coupling_soa = None
        SWE2DUrbanDrainageModule = None
        SWE2DStructureModule = None
        DrainageLink = DrainageNode = HydraulicStructure = InletExchange = OutfallExchange = None
        PipeEndExchange = None
        InletType = NodeInletAssignment = None
        PipeNetworkConfig = HydraulicStructureConfig = None
        DrainageSolverMode = None
        GodunovSolverMode = None
        SpatialDiscretization = None
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
        inspect_hyetograph_rows,
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
            inspect_hyetograph_rows,
            assign_cells_to_nearest_gauge,
            runoff_depth_mm_from_event_rain_mm,
            time_of_concentration_hours_velocity_method,
        )
    except Exception:
        Gauge = ThiessenRainCNForcing = None
        build_hyetograph = assign_cells_to_nearest_gauge = None
        inspect_hyetograph_rows = None
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
    ("Normal Depth (friction slope Sf)", 7),
    ("Timeseries Flow Q", 102),
    ("Timeseries Stage", 103),
    ("Open (zero-gradient)", 4),
    ("Reflecting", 5),
]

_BC_TS_FLOW = 102
_BC_TS_STAGE = 103
_BC_INFLOW_Q = 2

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

_TEMPORAL_ORDER_OPTIONS = [
    ("Euler (RK1, 1st-order)",           1),
    ("RK2 (Heun, 2nd-order, default)",   2),
    ("RK4 (classic, 4th-order)",         4),
]

_BC_VALUE_MAP = {
    "Wall (zero normal flux)": 1,
    "Inflow Q (total discharge)": 2,
    "Stage (prescribed WSE)": 3,
    "Normal Depth (prescribed depth)": 6,
    "Normal Depth (friction slope Sf)": 7,
    "Timeseries Flow Q": 102,
    "Timeseries Stage": 103,
    "Open (zero-gradient)": 4,
    "Reflecting": 5,
}

_DRAIN_NODE_TYPE_VALUE_MAP = {
    "Junction": "junction",
    "Outfall": "outfall",
    "Storage": "storage",
    "Inlet": "inlet",
}

_DRAIN_LINK_TYPE_VALUE_MAP = {
    "Conduit": "conduit",
    "Short lateral (simplified)": "lateral_simple",
    "Pump": "pump",
    "Weir": "weir",
    "Orifice": "orifice",
}

_DRAIN_LINK_SHAPE_VALUE_MAP = {
    "Circular": "circular",
    "Box": "box",
    "Pipe arch": "pipe_arch",
    "Custom area": "custom",
}

_RAIN_GAGE_UNITS_VALUE_MAP = {
    "mm/hr": "mm/hr",
    "in/hr": "in/hr",
    "mm": "mm",
    "in": "in",
}

_HYETOGRAPH_VALUE_TYPE_MAP = {
    "Intensity": "intensity",
    "Incremental depth": "incremental",
    "Cumulative depth": "cumulative",
}

_HYETOGRAPH_UNITS_VALUE_MAP = {
    "mm/hr": "mm/hr",
    "in/hr": "in/hr",
    "mm": "mm",
    "in": "in",
}

_SWE2D_WORKBENCH_WINDOWS = []

_MODEL_LAYER_BINDINGS = {
    "rain_gages": {
        "layer_name": "swe2d_rain_gages",
        "combo_attr": "rain_gage_layer_combo",
        "geometry": "point",
        "required_fields": ("gage_id", "hyetograph_id"),
    },
    "hyetographs": {
        "layer_name": "swe2d_hyetographs",
        "combo_attr": "hyetograph_layer_combo",
        "geometry": "table",
        "required_fields": ("hyetograph_id", "Time", "Value"),
    },
    "storm_areas": {
        "layer_name": "swe2d_storm_areas",
        "combo_attr": "storm_area_layer_combo",
        "geometry": "polygon",
        "required_fields": ("storm_id",),
    },
    "drainage_nodes": {
        "layer_name": "swe2d_drainage_nodes",
        "combo_attr": "drain_nodes_layer_combo",
        "geometry": "point",
        "required_fields": ("node_id", "invert_elev"),
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
        "geometry": "table",
        "required_fields": ("inlet_type_id", "weir_length", "coeff_weir", "coeff_orifice"),
    },
    "drainage_node_inlets": {
        "layer_name": "swe2d_drainage_node_inlets",
        "combo_attr": "drain_node_inlets_layer_combo",
        "geometry": "table",
        "required_fields": ("node_id", "inlet_type_id"),
    },
    "hydraulic_structures": {
        "layer_name": "swe2d_structures",
        "combo_attr": "structures_layer_combo",
        "geometry": "line",
        "required_fields": ("structure_id", "structure_type", "crest_elev", "enabled"),
    },
}

_MODEL_LAYER_BINDINGS_VERSION = 5


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
        self.wse_render_lbl = QtWidgets.QLabel("WSE render:")
        controls.addWidget(self.wse_render_lbl)
        self.wse_render_combo = QtWidgets.QComboBox()
        self.wse_render_combo.addItem("Clipped to bed (wet only)", "clipped")
        self.wse_render_combo.addItem("Raw sampled", "raw")
        controls.addWidget(self.wse_render_combo)
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
        self.wse_render_combo.currentIndexChanged.connect(self._refresh_plot)
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
        self.wse_render_lbl.setVisible(is_wse)
        self.wse_render_combo.setVisible(is_wse)

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

        # WSE + bed profile. Rendering is wet-aware and clips WSE to bed for
        # display so dry/near-dry samples do not produce misleading below-bed dips.
        wse = np.asarray([float(r.get("wse_m", float("nan"))) for r in rows], dtype=np.float64)
        bed = np.asarray([float(r.get("bed_m", float("nan"))) for r in rows], dtype=np.float64)
        depth = np.asarray([float(r.get("depth_m", float("nan"))) for r in rows], dtype=np.float64)
        wet = np.asarray([float(r.get("wet", float("nan"))) for r in rows], dtype=np.float64)
        ok = np.isfinite(wse) & np.isfinite(bed)
        if not np.any(ok):
            ax.text(0.5, 0.5, "No WSE/bed values for selected line/timestep", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        x_ok = x[ok]
        wse_ok = wse[ok]
        bed_ok = bed[ok]
        depth_ok = depth[ok]
        wet_ok_raw = wet[ok]
        wet_mask = np.where(np.isfinite(wet_ok_raw), wet_ok_raw > 0.5, depth_ok > 1.0e-9)

        render_mode = str(self.wse_render_combo.currentData()) if hasattr(self, "wse_render_combo") else "clipped"
        wse_phys = np.maximum(wse_ok, bed_ok)
        below_bed_count = int(np.sum(wse_ok < bed_ok))
        if render_mode == "raw":
            fill_mask = np.isfinite(wse_ok) & np.isfinite(bed_ok)
            wse_fill = wse_ok
            wse_plot = wse_ok
            render_note = f"Raw mode: {below_bed_count} sample(s) with WSE < bed"
        else:
            fill_mask = wet_mask
            wse_fill = wse_phys
            wse_plot = np.where(wet_mask, wse_phys, np.nan)
            render_note = f"Display note: clipped {below_bed_count} sample(s) where WSE < bed"
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
                        if not (fill_mask[i] and fill_mask[i + 1]):
                            continue
                        c_mid = cmap(norm(0.5 * (fill_vals[i] + fill_vals[i + 1])))
                        ax.fill_between(
                            x_ok[i : i + 2],
                            bed_ok[i : i + 2],
                            wse_fill[i : i + 2],
                            color=c_mid,
                            alpha=0.85,
                            linewidth=0.0,
                        )
                    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    self._plot_fig.colorbar(sm, ax=ax, label=self.fill_metric_combo.currentText())
            except Exception:
                ax.fill_between(x_ok, bed_ok, wse_fill, where=fill_mask, interpolate=True, color="tab:blue", alpha=0.18)
        else:
            ax.fill_between(x_ok, bed_ok, wse_fill, where=fill_mask, interpolate=True, color="tab:blue", alpha=0.18)

        ax.plot(x_ok, bed_ok, "-", color="saddlebrown", linewidth=1.6, label="Bed")
        ax.plot(x_ok, wse_plot, "-", color="royalblue", linewidth=1.8, label="Water Surface")
        if below_bed_count > 0:
            ax.text(
                0.01,
                0.99,
            render_note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="0.35",
            )
        ax.set_xlabel("Station")
        ax.set_ylabel("Elevation")
        ax.set_title(f"Line {line_id} WSE + bed at t={t_s/3600.0:.4f} hr" + (f" ({line_name})" if line_name else ""))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()


class SWE2DCouplingResultsViewerDialog(QtWidgets.QDialog):
    """Viewer for drainage/structure coupling time series stored in GeoPackage/SQLite."""

    _BASE_COLUMNS = [
        ("t_s", "Time (s)"),
        ("component", "Component"),
        ("metric", "Metric"),
        ("object_id", "Object ID"),
        ("object_name", "Object Name"),
        ("value", "Value"),
    ]

    def __init__(
        self,
        records: List[Dict[str, object]],
        run_id: str,
        db_path: str,
        length_unit: str = "m",
        flow_unit_label: str = "m3/s",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Drainage/Structure Results Viewer")
        self.resize(980, 620)

        self._records = list(records)
        self._run_id = str(run_id)
        self._db_path = str(db_path)
        self._length_unit = str(length_unit or "m")
        self._flow_unit = str(flow_unit_label or "m3/s")
        self._plot_canvas = None
        self._plot_fig = None

        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(f"Run ID: {self._run_id}\nSource: {self._db_path}")
        header.setWordWrap(True)
        root.addWidget(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Component:"))
        self.component_combo = QtWidgets.QComboBox()
        controls.addWidget(self.component_combo)
        controls.addWidget(QtWidgets.QLabel("Metric:"))
        self.metric_combo = QtWidgets.QComboBox()
        controls.addWidget(self.metric_combo)
        controls.addWidget(QtWidgets.QLabel("Object:"))
        self.object_combo = QtWidgets.QComboBox()
        controls.addWidget(self.object_combo)
        controls.addStretch(1)
        root.addLayout(controls)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(len(self._BASE_COLUMNS))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in self._BASE_COLUMNS])
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

        self._populate_component_combo()
        self._populate_metric_combo()
        self._populate_object_combo()
        self._refresh_table()
        self._refresh_plot()

        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        self.object_combo.currentIndexChanged.connect(self._refresh_table)
        self.object_combo.currentIndexChanged.connect(self._refresh_plot)

    def _unit_label_for_metric(self, metric: str) -> str:
        m = str(metric or "")
        if m == "depth":
            return self._length_unit
        if m == "flow":
            return self._flow_unit
        if m == "source":
            return f"{self._length_unit}/s"
        if m.endswith("_m"):
            return self._length_unit
        if m.endswith("_cms"):
            return self._flow_unit
        if m.endswith("_mps"):
            return f"{self._length_unit}/s"
        return ""

    def _populate_component_combo(self):
        self.component_combo.clear()
        self.component_combo.addItem("All components", None)
        comps = sorted({str(r.get("component", "") or "") for r in self._records if r.get("component") is not None})
        for comp in comps:
            if comp:
                self.component_combo.addItem(comp, comp)

    def _populate_metric_combo(self):
        selected_comp = self.component_combo.currentData()
        metrics = set()
        for rec in self._records:
            comp = str(rec.get("component", "") or "")
            if selected_comp is not None and comp != str(selected_comp):
                continue
            metric = str(rec.get("metric", "") or "")
            if metric:
                metrics.add(metric)
        current = self.metric_combo.currentData()
        self.metric_combo.clear()
        self.metric_combo.addItem("All metrics", None)
        for metric in sorted(metrics):
            unit = self._unit_label_for_metric(metric)
            label = metric if not unit else f"{metric} ({unit})"
            self.metric_combo.addItem(label, metric)
        if current is not None:
            idx = self.metric_combo.findData(current)
            if idx >= 0:
                self.metric_combo.setCurrentIndex(idx)

    def _populate_object_combo(self):
        selected_comp = self.component_combo.currentData()
        selected_metric = self.metric_combo.currentData()
        objects: Dict[str, str] = {}
        for rec in self._records:
            comp = str(rec.get("component", "") or "")
            metric = str(rec.get("metric", "") or "")
            if selected_comp is not None and comp != str(selected_comp):
                continue
            if selected_metric is not None and metric != str(selected_metric):
                continue
            oid = str(rec.get("object_id", "") or "")
            if not oid:
                continue
            objects[oid] = str(rec.get("object_name", "") or "")

        current = self.object_combo.currentData()
        self.object_combo.clear()
        self.object_combo.addItem("All objects", None)
        for oid in sorted(objects.keys()):
            name = objects[oid]
            label = oid if not name else f"{oid} - {name}"
            self.object_combo.addItem(label, oid)
        if current is not None:
            idx = self.object_combo.findData(current)
            if idx >= 0:
                self.object_combo.setCurrentIndex(idx)

    def _on_component_changed(self):
        self._populate_metric_combo()
        self._populate_object_combo()
        self._refresh_table()
        self._refresh_plot()

    def _on_metric_changed(self):
        self._populate_object_combo()
        self._refresh_table()
        self._refresh_plot()

    def _filtered_records(self) -> List[Dict[str, object]]:
        comp_sel = self.component_combo.currentData()
        metric_sel = self.metric_combo.currentData()
        obj_sel = self.object_combo.currentData()
        out: List[Dict[str, object]] = []
        for rec in self._records:
            comp = str(rec.get("component", "") or "")
            metric = str(rec.get("metric", "") or "")
            oid = str(rec.get("object_id", "") or "")
            if comp_sel is not None and comp != str(comp_sel):
                continue
            if metric_sel is not None and metric != str(metric_sel):
                continue
            if obj_sel is not None and oid != str(obj_sel):
                continue
            out.append(rec)
        return out

    def _refresh_table(self):
        rows = self._filtered_records()
        rows.sort(
            key=lambda r: (
                float(r.get("t_s", 0.0)),
                str(r.get("component", "") or ""),
                str(r.get("metric", "") or ""),
                str(r.get("object_id", "") or ""),
            )
        )
        self.table.setRowCount(len(rows))
        for r, rec in enumerate(rows):
            for c, (key, _) in enumerate(self._BASE_COLUMNS):
                val = rec.get(key)
                txt = f"{val:.6f}" if isinstance(val, float) else str(val)
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

    def _refresh_plot(self):
        if not self._have_mpl or self._plot_fig is None or self._plot_canvas is None:
            return
        rows = self._filtered_records()
        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        if not rows:
            ax.text(0.5, 0.5, "No coupling records for selected filter", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        by_object: Dict[str, List[Tuple[float, float]]] = {}
        object_names: Dict[str, str] = {}
        for rec in rows:
            try:
                t_s = float(rec.get("t_s", 0.0))
                value = float(rec.get("value", float("nan")))
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            oid = str(rec.get("object_id", "") or "")
            by_object.setdefault(oid, []).append((t_s, value))
            object_names[oid] = str(rec.get("object_name", "") or "")

        if not by_object:
            ax.text(0.5, 0.5, "No numeric values to plot", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        for oid in sorted(by_object.keys()):
            pairs = sorted(by_object[oid], key=lambda x: x[0])
            x = np.asarray([p[0] / 3600.0 for p in pairs], dtype=np.float64)
            y = np.asarray([p[1] for p in pairs], dtype=np.float64)
            label = oid if oid else "(unlabeled)"
            if object_names.get(oid):
                label += f" ({object_names[oid]})"
            ax.plot(x, y, "-", linewidth=1.8, label=label)

        metric_sel = self.metric_combo.currentData()
        y_label = "Value"
        if metric_sel is not None:
            unit = self._unit_label_for_metric(str(metric_sel))
            y_label = str(metric_sel) if not unit else f"{metric_sel} ({unit})"

        ax.set_xlabel("Time (hr)")
        ax.set_ylabel(y_label)
        ax.set_title("Drainage/Structure coupling time series")
        if len(by_object) > 1:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()


class SWE2DRunLogViewerDialog(QtWidgets.QDialog):
    """Viewer for saved SWE2D run logs stored in GeoPackage/SQLite."""

    def __init__(self, records: List[Dict[str, object]], run_id: str, db_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SWE2D Run Log Viewer")
        self.resize(900, 620)
        self._records = list(records)
        self._db_path = str(db_path)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"Source: {self._db_path}"))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Run:"))
        self.run_combo = QtWidgets.QComboBox()
        row.addWidget(self.run_combo)
        row.addStretch(1)
        root.addLayout(row)

        self.meta_lbl = QtWidgets.QLabel("")
        self.meta_lbl.setWordWrap(True)
        root.addWidget(self.meta_lbl)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        root.addWidget(self.text, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_run_combo()
        idx = self.run_combo.findData(str(run_id))
        if idx >= 0:
            self.run_combo.setCurrentIndex(idx)
        self.run_combo.currentIndexChanged.connect(self._refresh_view)
        self._refresh_view()

    def _populate_run_combo(self):
        self.run_combo.clear()
        for rec in self._records:
            rid = str(rec.get("run_id", "") or "")
            created = str(rec.get("created_utc", "") or "")
            dur = float(rec.get("duration_s", 0.0) or 0.0)
            label = f"{rid} ({created}, {dur:.2f}s)"
            self.run_combo.addItem(label, rid)

    def _refresh_view(self):
        rid = str(self.run_combo.currentData() or "")
        rec = None
        for r in self._records:
            if str(r.get("run_id", "") or "") == rid:
                rec = r
                break
        if rec is None:
            self.meta_lbl.setText("No run selected.")
            self.text.setPlainText("")
            return
        self.meta_lbl.setText(
            f"Run ID: {rid}\n"
            f"Start: {rec.get('start_wallclock', '')}\n"
            f"End: {rec.get('end_wallclock', '')}\n"
            f"Duration: {float(rec.get('duration_s', 0.0) or 0.0):.2f} s"
        )
        self.text.setPlainText(str(rec.get("log_text", "") or ""))


class SWE2DWorkbenchDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Workbench")
        self.resize(1160, 760)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self._iface = iface

        self._backend: Optional[SWE2DBackend] = None
        self._cancel_requested = False
        self._mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._result_data: Optional[Dict[str, np.ndarray]] = None
        self._snapshot_timesteps: List[Tuple] = []  # list of (time_s, h, hu, hv)
        self._line_snapshot_rows: List[Dict[str, object]] = []
        self._line_snapshot_profile_rows: List[Dict[str, object]] = []
        self._coupling_snapshot_rows: List[Dict[str, object]] = []
        self._line_results_latest_run_id: str = ""
        self._line_results_latest_db_path: str = ""
        self._coupling_results_latest_run_id: str = ""
        self._coupling_results_latest_db_path: str = ""
        self._run_log_latest_run_id: str = ""
        self._run_log_latest_db_path: str = ""
        self._runtime_log_lines: List[str] = []
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
        self._topology_mesh_active_timeout_sec = 0.0
        self._project_layer_state_blocked = False
        self._initial_layer_restore_pending = True
        try:
            timeout_sec = float(os.environ.get("BACKWATER_TOPOLOGY_MESH_TIMEOUT_SEC", "300"))
        except Exception:
            timeout_sec = 300.0
        self._topology_mesh_timeout_sec = max(30.0, timeout_sec)
        self._topology_mesh_active_timeout_sec = self._topology_mesh_timeout_sec

        FigureCanvas, Figure, mtri = _try_import_matplotlib_qt()
        self._FigureCanvas = FigureCanvas
        self._Figure = Figure
        self._mtri = mtri
        self._have_mpl = FigureCanvas is not None and Figure is not None and mtri is not None

        self._build_ui()
        self._connect_project_layer_state_signals()
        self._connect_project_workbench_state_signals()
        self._connect_project_save_state_signals()
        self._restore_project_layer_bindings()
        self._initial_layer_restore_pending = False
        self._persist_project_layer_bindings()
        # Note: workbench state restoration moved to showEvent() for more reliable timing
        self._update_unit_system_from_crs()
        self._log(
            f"2D bridge: {'available' if swe2d_available() else 'missing'} | "
            f"GPU: {'available' if swe2d_gpu_available() else 'cpu-only'}"
        )
        self._log(
            f"Meshing: Gmsh {'available' if _gmsh_available() else 'NOT INSTALLED — use Structured backend or: pip install gmsh'}"
        )
        # Sprint 0: dockable results panel (created lazily on first show)
        self._results_panel = None
        self._sample_line_draw_tool = None
        self._sample_line_prev_map_tool = None
        self._velocity_vector_builder = None
        self._velocity_vectors_layer_id: Optional[str] = None
        self._velocity_overlay_manual_gpkg_path: str = ""
        self._velocity_overlay_manual_run_id: str = ""
        self._velocity_overlay_manual_layer_name: str = ""
        self._velocity_overlay_manual_table_name: str = ""

    def _resolve_qgis_iface(self):
        """Resolve a usable QGIS iface object from dialog context/runtime."""
        if getattr(self, "_iface", None) is not None:
            return self._iface

        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "_get_qgis_iface") and callable(getattr(parent, "_get_qgis_iface")):
                try:
                    iface_obj = parent._get_qgis_iface()
                    if iface_obj is not None:
                        self._iface = iface_obj
                        return iface_obj
                except Exception:
                    pass
            if hasattr(parent, "iface"):
                try:
                    iface_obj = getattr(parent, "iface")
                    if iface_obj is not None:
                        self._iface = iface_obj
                        return iface_obj
                except Exception:
                    pass

        try:
            import qgis.utils as _qutils

            iface_obj = getattr(_qutils, "iface", None)
            if iface_obj is not None:
                self._iface = iface_obj
                return iface_obj
        except Exception:
            pass
        return None

    def _resolve_map_canvas(self):
        iface_obj = self._resolve_qgis_iface()
        if iface_obj is None or not hasattr(iface_obj, "mapCanvas"):
            return None
        try:
            return iface_obj.mapCanvas()
        except Exception:
            return None

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
        self.drain_node_inlets_layer_combo = QtWidgets.QComboBox()
        self.drain_node_inlets_layer_combo.addItem("(none)", None)
        self.structures_layer_combo = QtWidgets.QComboBox()
        self.structures_layer_combo.addItem("(none)", None)
        self.layer_group_combo = QtWidgets.QComboBox()
        self.layer_group_combo.addItem("(no group)", None)
        self.autopop_group_btn = QtWidgets.QPushButton("Autopopulate From Group")
        self.autopop_group_btn.clicked.connect(self._autopopulate_layer_combos_from_group)
        self.refresh_layers_btn = QtWidgets.QPushButton("Refresh Layers")
        self.refresh_layers_btn.clicked.connect(self._refresh_layer_combos)
        self.create_model_gpkg_btn = QtWidgets.QPushButton("Create 2D Model GeoPackage")
        self.create_model_gpkg_btn.clicked.connect(self._create_2d_model_geopackage)
        self.create_lumped_gpkg_btn = QtWidgets.QPushButton("Create Lumped Hydro GeoPackage")
        self.create_lumped_gpkg_btn.clicked.connect(self._create_lumped_hydrology_geopackage)
        self.load_model_gpkg_btn = QtWidgets.QPushButton("Load 2D Model GeoPackage")
        self.load_model_gpkg_btn.clicked.connect(self._load_2d_model_geopackage)
        self.migrate_model_gpkg_btn = QtWidgets.QPushButton("Update GeoPackage Schema")
        self.migrate_model_gpkg_btn.setToolTip(
            "Add any missing layers and columns to an existing 2D model GeoPackage "
            "so it matches the current schema."
        )
        self.migrate_model_gpkg_btn.clicked.connect(self._migrate_2d_model_geopackage)
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
        self.open_results_panel_btn = QtWidgets.QPushButton("Results Panel (multi-run)")
        self.open_results_panel_btn.setToolTip("Open the dockable multi-run results panel")
        self.open_results_panel_btn.clicked.connect(self._show_results_panel)
        self.draw_sample_line_btn = QtWidgets.QPushButton("Draw Sample Line On Map")
        self.draw_sample_line_btn.setToolTip("Draw a sample polyline directly on the map canvas")
        self.draw_sample_line_btn.clicked.connect(self._activate_sample_line_draw_tool)
        self.open_coupling_results_viewer_btn = QtWidgets.QPushButton("Open Drainage/Structure Results Viewer")
        self.open_coupling_results_viewer_btn.clicked.connect(self._open_coupling_results_viewer)
        self.open_run_log_viewer_btn = QtWidgets.QPushButton("Open Run Log Viewer")
        self.open_run_log_viewer_btn.clicked.connect(self._open_run_log_viewer)
        self.save_mesh_results_to_gpkg_chk = QtWidgets.QCheckBox("Save mesh snapshot results to GeoPackage")
        self.save_mesh_results_to_gpkg_chk.setChecked(True)
        self.save_line_results_to_gpkg_chk = QtWidgets.QCheckBox("Save sampled line results to GeoPackage")
        self.save_line_results_to_gpkg_chk.setChecked(True)
        self.save_coupling_results_to_gpkg_chk = QtWidgets.QCheckBox("Save drainage/structure results to GeoPackage")
        self.save_coupling_results_to_gpkg_chk.setChecked(True)
        self.save_run_log_to_gpkg_chk = QtWidgets.QCheckBox("Save run log to GeoPackage")
        self.save_run_log_to_gpkg_chk.setChecked(True)
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
        map_layout.addWidget(QtWidgets.QLabel("Drainage inlet types (table):"), 10, 0)
        map_layout.addWidget(self.drain_inlets_layer_combo, 10, 1)
        map_layout.addWidget(QtWidgets.QLabel("Drainage node-inlets (table):"), 11, 0)
        map_layout.addWidget(self.drain_node_inlets_layer_combo, 11, 1)
        map_layout.addWidget(QtWidgets.QLabel("Hydraulic structures layer:"), 12, 0)
        map_layout.addWidget(self.structures_layer_combo, 12, 1)
        map_layout.addWidget(QtWidgets.QLabel("Layer group:"), 13, 0)
        map_layout.addWidget(self.layer_group_combo, 13, 1)
        map_layout.addWidget(self.autopop_group_btn, 14, 0, 1, 2)
        map_layout.addWidget(self.refresh_layers_btn, 15, 0)
        map_layout.addWidget(self.create_model_gpkg_btn, 15, 1)
        map_layout.addWidget(self.create_lumped_gpkg_btn, 16, 0, 1, 2)
        map_layout.addWidget(self.load_model_gpkg_btn, 17, 0)
        map_layout.addWidget(self.migrate_model_gpkg_btn, 17, 1)
        map_layout.addWidget(self.preview_coupling_btn, 18, 0, 1, 2)
        map_layout.addWidget(self.export_mesh_layers_btn, 19, 0)
        map_layout.addWidget(self.save_hdf5_btn, 19, 1)
        map_layout.addWidget(self.save_results_hdf5_btn, 20, 0, 1, 2)
        map_layout.addWidget(self.save_results_ugrid_btn, 21, 0, 1, 2)
        map_layout.addWidget(self.extended_outputs_chk, 22, 0, 1, 2)
        map_layout.addWidget(self.save_mesh_results_to_gpkg_chk, 23, 0, 1, 2)
        map_layout.addWidget(self.save_line_results_to_gpkg_chk, 24, 0, 1, 2)
        map_layout.addWidget(self.save_coupling_results_to_gpkg_chk, 25, 0, 1, 2)
        map_layout.addWidget(self.save_run_log_to_gpkg_chk, 26, 0, 1, 2)
        map_layout.addWidget(self.open_results_viewer_btn, 27, 0, 1, 2)
        map_layout.addWidget(self.open_results_panel_btn, 28, 0, 1, 2)
        map_layout.addWidget(self.draw_sample_line_btn, 29, 0, 1, 2)
        map_layout.addWidget(self.open_coupling_results_viewer_btn, 30, 0, 1, 2)
        map_layout.addWidget(self.open_run_log_viewer_btn, 31, 0, 1, 2)
        map_layout.addWidget(self.import_mesh_layers_btn, 32, 0, 1, 2)
        map_layout.addWidget(self.terrain_to_nodes_btn, 33, 0, 1, 2)
        map_layout.addWidget(self.pull_node_z_btn, 34, 0, 1, 2)
        map_layout.addWidget(self.layer_status_lbl, 35, 0, 1, 2)
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
        self.topo_quality_max_non_orth_spin = QtWidgets.QDoubleSpinBox()
        self.topo_quality_max_non_orth_spin.setRange(1.0, 89.9)
        self.topo_quality_max_non_orth_spin.setDecimals(1)
        self.topo_quality_max_non_orth_spin.setValue(82.0)
        self.topo_quality_min_area_edit = QtWidgets.QLineEdit("1e-14")
        self.topo_quality_strict_chk = QtWidgets.QCheckBox("Strict quality acceptance")
        self.topo_quality_size_scales_edit = QtWidgets.QLineEdit("1.0,0.9,0.8,0.7")
        self.topo_quality_smooth_increments_edit = QtWidgets.QLineEdit("0,2,4,6")
        self.topo_gmsh_quality_enable_chk = QtWidgets.QCheckBox("Enable Gmsh iterative quality loop")
        self.topo_gmsh_quality_enable_chk.setChecked(False)
        self.topo_gmsh_quality_max_iters_spin = QtWidgets.QSpinBox()
        self.topo_gmsh_quality_max_iters_spin.setRange(1, 50)
        self.topo_gmsh_quality_max_iters_spin.setValue(6)
        self.topo_gmsh_quality_time_limit_spin = QtWidgets.QDoubleSpinBox()
        self.topo_gmsh_quality_time_limit_spin.setRange(1.0, 3600.0)
        self.topo_gmsh_quality_time_limit_spin.setDecimals(1)
        self.topo_gmsh_quality_time_limit_spin.setValue(60.0)
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
        quality_form.addRow(self.topo_gmsh_quality_enable_chk)
        quality_form.addRow("Gmsh max attempts:", self.topo_gmsh_quality_max_iters_spin)
        quality_form.addRow("Gmsh time budget (s):", self.topo_gmsh_quality_time_limit_spin)
        quality_form.addRow("Min angle (deg):", self.topo_quality_min_angle_spin)
        quality_form.addRow("Max aspect ratio:", self.topo_quality_max_aspect_spin)
        quality_form.addRow("Max non-orthogonality (deg):", self.topo_quality_max_non_orth_spin)
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
        self.topo_terminate_btn = QtWidgets.QPushButton("Terminate Mesh Run")
        self.topo_terminate_btn.setEnabled(False)
        self.topo_terminate_btn.clicked.connect(self._on_terminate_topology_mesh)
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
        topo_layout.addWidget(QtWidgets.QLabel("Quality controls (Gmsh + TQMesh):"), 9, 0)
        topo_layout.addWidget(quality_form_widget, 9, 1)
        topo_layout.addWidget(self.topo_validate_btn, 10, 0, 1, 2)
        topo_layout.addWidget(self.topo_edit_regions_btn, 11, 0)
        topo_layout.addWidget(self.topo_edit_quad_edges_btn, 11, 1)
        topo_layout.addWidget(self.topo_controls_summary_lbl, 12, 0, 1, 2)
        topo_layout.addWidget(self.topo_export_template_btn, 13, 0, 1, 2)
        topo_actions_row = QtWidgets.QWidget()
        topo_actions_layout = QtWidgets.QHBoxLayout(topo_actions_row)
        topo_actions_layout.setContentsMargins(0, 0, 0, 0)
        topo_actions_layout.addWidget(self.topo_generate_btn)
        topo_actions_layout.addWidget(self.topo_terminate_btn)
        topo_layout.addWidget(topo_actions_row, 14, 0, 1, 2)
        topo_layout.addWidget(self.topo_status_lbl, 15, 0, 1, 2)
        left_layout.addWidget(topo_group)

        self.topo_backend_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_regions_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_constraints_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_quad_edges_combo.currentIndexChanged.connect(self._update_topology_control_summary)
        self.topo_quality_min_angle_spin.valueChanged.connect(self._update_topology_control_summary)
        self.topo_quality_max_aspect_spin.valueChanged.connect(self._update_topology_control_summary)
        self.topo_quality_max_non_orth_spin.valueChanged.connect(self._update_topology_control_summary)
        self.topo_quality_min_area_edit.textChanged.connect(self._update_topology_control_summary)
        self.topo_quality_strict_chk.toggled.connect(self._update_topology_control_summary)
        self.topo_quality_size_scales_edit.textChanged.connect(self._update_topology_control_summary)
        self.topo_quality_smooth_increments_edit.textChanged.connect(self._update_topology_control_summary)
        self.topo_gmsh_quality_enable_chk.toggled.connect(self._update_topology_control_summary)
        self.topo_gmsh_quality_max_iters_spin.valueChanged.connect(self._update_topology_control_summary)
        self.topo_gmsh_quality_time_limit_spin.valueChanged.connect(self._update_topology_control_summary)

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
        self.gpu_diag_sync_interval_spin = QtWidgets.QSpinBox()
        self.gpu_diag_sync_interval_spin.setRange(1, 1000000)
        self.gpu_diag_sync_interval_spin.setValue(10)
        self.gpu_diag_sync_interval_spin.setToolTip(
            "GPU host diagnostic sync cadence in computational steps.\n"
            "1 = sync every step (freshest Cmax/WSEres runtime output).\n"
            "Higher values reduce host sync overhead but update diagnostics less often."
        )
        self.enable_cuda_graphs_chk = QtWidgets.QCheckBox("Enable")
        self.enable_cuda_graphs_chk.setChecked(False)
        self.enable_cuda_graphs_chk.setToolTip(
            "Enable CUDA graph capture/replay for the core GPU step kernel chain.\n"
            "Can reduce launch overhead and improve throughput on compatible runs."
        )
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
        self.max_source_depth_step_spin = QtWidgets.QDoubleSpinBox()
        self.max_source_depth_step_spin.setRange(0.0, 10.0)
        self.max_source_depth_step_spin.setDecimals(6)
        self.max_source_depth_step_spin.setValue(0.0)
        self.max_source_depth_step_spin.setToolTip(
            "Absolute cap on positive source-driven depth increase per step (model units).\n"
            "0 disables the cap. Useful for suppressing rain/CN impulse spikes."
        )
        self.max_source_rate_spin = QtWidgets.QDoubleSpinBox()
        self.max_source_rate_spin.setRange(0.0, 100.0)
        self.max_source_rate_spin.setDecimals(6)
        self.max_source_rate_spin.setValue(0.0)
        self.max_source_rate_spin.setToolTip(
            "Cap on positive net source rate (model units per second).\n"
            "0 disables the cap. Applies before per-step depth update."
        )
        self.extreme_rain_mode_chk = QtWidgets.QCheckBox("Enable")
        self.extreme_rain_mode_chk.setChecked(False)
        self.extreme_rain_mode_chk.setToolTip(
            "Adaptive source-CFL limiter for extreme rainfall/source events.\n"
            "When enabled, positive source terms are reduced using an equivalent\n"
            "substepping factor so dt*source remains bounded by beta*h_ref."
        )
        self.source_cfl_beta_spin = QtWidgets.QDoubleSpinBox()
        self.source_cfl_beta_spin.setRange(0.01, 2.0)
        self.source_cfl_beta_spin.setDecimals(3)
        self.source_cfl_beta_spin.setSingleStep(0.05)
        self.source_cfl_beta_spin.setValue(0.25)
        self.source_cfl_beta_spin.setToolTip(
            "Target source-CFL beta in dt*source <= beta*h_ref.\n"
            "Lower beta is more conservative."
        )
        self.source_max_substeps_spin = QtWidgets.QSpinBox()
        self.source_max_substeps_spin.setRange(1, 512)
        self.source_max_substeps_spin.setValue(16)
        self.source_max_substeps_spin.setToolTip(
            "Maximum equivalent source substeps used by adaptive source limiter."
        )
        self.source_true_subcycling_chk = QtWidgets.QCheckBox("Enable")
        self.source_true_subcycling_chk.setChecked(False)
        self.source_true_subcycling_chk.setToolTip(
            "Apply true source subcycling (real sub-iterations over dt) instead of\n"
            "equivalent one-shot source scaling."
        )
        self.source_imex_split_chk = QtWidgets.QCheckBox("Enable")
        self.source_imex_split_chk.setChecked(False)
        self.source_imex_split_chk.setToolTip(
            "IMEX-style split: apply flux update first, then source/friction subcycling.\n"
            "Most useful when true source subcycling is enabled."
        )
        self.source_stage_coupled_imex_rk2_chk = QtWidgets.QCheckBox("Enable")
        self.source_stage_coupled_imex_rk2_chk.setChecked(False)
        self.source_stage_coupled_imex_rk2_chk.setToolTip(
            "Stage-coupled IMEX-RK2 for external coupling sources (drainage/structures).\n"
            "Runs a predictor/corrector source update each step (GPU native injection path).\n"
            "Best for stiff coupling; costs extra compute per step."
        )
        self.shallow_damping_depth_spin = QtWidgets.QDoubleSpinBox()
        self.shallow_damping_depth_spin.setRange(1.0e-8, 10.0)
        self.shallow_damping_depth_spin.setDecimals(6)
        self.shallow_damping_depth_spin.setValue(1.0e-4)
        self.shallow_damping_depth_spin.setToolTip(
            "Depth threshold for smooth momentum damping in shallow cells."
        )
        self.shallow_front_recon_fallback_chk = QtWidgets.QCheckBox("Enable")
        self.shallow_front_recon_fallback_chk.setChecked(True)
        self.shallow_front_recon_fallback_chk.setToolTip(
            "If enabled, force first-order reconstruction on shallow wet/dry-front\n"
            "edge pairs to improve stability for higher-order schemes."
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
        self.ia_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.ia_ratio_spin.setRange(0.0, 1.0)
        self.ia_ratio_spin.setDecimals(3)
        self.ia_ratio_spin.setSingleStep(0.01)
        self.ia_ratio_spin.setValue(0.2)
        self.ia_ratio_spin.setToolTip(
            "Initial abstraction ratio (Ia/S) for SCS Curve Number losses.\n"
            "Typical default is 0.20."
        )
        self.use_spatial_rain_cn_chk = QtWidgets.QCheckBox("Use Thiessen gage rainfall when layers are available")
        self.use_spatial_rain_cn_chk.setChecked(True)
        self.infiltration_method_combo = QtWidgets.QComboBox()
        self.infiltration_method_combo.addItem("SCS Curve Number", "scs_cn")
        self.infiltration_method_combo.addItem("None (no infiltration)", "none")
        self.infiltration_method_combo.setToolTip(
            "Infiltration/loss method applied to rainfall before it enters the 2D surface as runoff.\n"
            "SCS Curve Number: NRCS CN abstraction (default).\n"
            "None: all rainfall becomes direct runoff — no abstraction."
        )
        self.storm_area_layer_combo = QtWidgets.QComboBox()
        self.storm_area_layer_combo.addItem("(none)", None)
        self.rain_boundary_buffer_rings_spin = QtWidgets.QSpinBox()
        self.rain_boundary_buffer_rings_spin.setRange(0, 10)
        self.rain_boundary_buffer_rings_spin.setValue(1)
        self.rain_boundary_buffer_rings_spin.setToolTip(
            "Boundary rain buffer rings (Thiessen + CN forcing).\n"
            "0: no exclusion. 1: exclude boundary cells.\n"
            "N>1: also exclude N-1 inward neighbor rings."
        )
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
        self.temporal_order_combo = QtWidgets.QComboBox()
        for label, value in _TEMPORAL_ORDER_OPTIONS:
            self.temporal_order_combo.addItem(label, int(value))
        self.temporal_order_combo.setCurrentIndex(1)  # Default RK2
        self.temporal_order_combo.setToolTip(
            "Select temporal integration scheme:\n"
            "  Euler (RK1)  — 1st-order, fastest, use for dry-bed or debugging\n"
            "  RK2 (Heun)   — 2nd-order (default), balanced stability & speed\n"
            "  RK4 (classic) — 4th-order, best for rain-on-grid scenarios, ~2x cost\n"
            "RK4 only available on GPU."
        )
        self.godunov_mode_combo = QtWidgets.QComboBox()
        self.godunov_mode_combo.addItem("Current GPU solver", int(GodunovSolverMode.CURRENT_GPU_STEP))
        self.godunov_mode_combo.addItem("Godunov rollout (2nd-order)", int(GodunovSolverMode.GODUNOV_ROLLOUT))
        self.godunov_mode_combo.setCurrentIndex(0)
        self.godunov_mode_combo.setToolTip(
            "Select the solver implementation used by the GPU path.\n"
            "Current GPU solver: existing production path.\n"
            "Godunov rollout: enables the second-order rollout configuration and\n"
            "keeps the native solver on the migration path for the new FVM mode."
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
        self.coupling_loop_combo.setCurrentIndex(1)
        self.coupling_loop_combo.setToolTip(
            "Select coupling source assembly mode.\n"
            "CPU: Python reference path for drainage/structure source rates.\n"
            "CUDA: uses native CUDA kernel for per-cell source assembly when available;\n"
            "falls back to CPU reference automatically if CUDA binding/device is unavailable."
        )
        self.drainage_solver_mode_combo = QtWidgets.QComboBox()
        self.drainage_solver_mode_combo.addItem("EGL (Bernoulli + minor losses)", int(0))
        self.drainage_solver_mode_combo.addItem("Diffusion wave", int(1))
        self.drainage_solver_mode_combo.addItem("Dynamic Saint-Venant", int(2))
        self.drainage_solver_mode_combo.setCurrentIndex(0)
        self.drainage_solver_mode_combo.setToolTip(
            "Drainage 1D equation set.\n"
            "EGL: Bernoulli + Manning + minor losses.\n"
            "Diffusion: slope-driven Manning flow.\n"
            "Dynamic: semi-implicit Saint-Venant momentum update."
        )
        self.drainage_backend_combo = QtWidgets.QComboBox()
        self.drainage_backend_combo.addItem("CPU drainage solver (reference)", "cpu")
        self.drainage_backend_combo.addItem("GPU drainage solver (CUDA)", "gpu")
        self.drainage_backend_combo.setCurrentIndex(1)
        self.drainage_backend_combo.setToolTip(
            "Select drainage network solver backend.\n"
            "CPU: Python reference implementation.\n"
            "GPU: native CUDA drainage solver for EGL/Diffusion/Dynamic modes;\n"
            "falls back to CPU path when CUDA drainage bindings are unavailable."
        )
        self.drainage_gpu_method_combo = QtWidgets.QComboBox()
        self.drainage_gpu_method_combo.addItem("Per-step GPU drainage (fast for sparse exchange)", "step")
        self.drainage_gpu_method_combo.addItem("Native iterative GPU drainage (batched substeps)", "iterative")
        self.drainage_gpu_method_combo.setCurrentIndex(0)
        self.drainage_gpu_method_combo.setToolTip(
            "Select GPU drainage coupling method when drainage backend is GPU.\n"
            "Per-step: calls the GPU drainage step once per substep/iteration from Python.\n"
            "Native iterative: runs substeps and implicit iterations in one native call.\n"
            "Use native iterative for dense/active drainage exchange; per-step can be faster\n"
            "when exchange is sparse or mostly inactive."
        )
        self.drainage_coupling_substeps_spin = QtWidgets.QSpinBox()
        self.drainage_coupling_substeps_spin.setRange(1, 256)
        self.drainage_coupling_substeps_spin.setValue(1)
        self.drainage_coupling_substeps_spin.setToolTip(
            "Fixed number of 1D drainage substeps taken per 2D coupling step.\n"
            "Increase this for stiff drainage networks or dynamic-wave runs."
        )
        self.drainage_max_coupling_substeps_spin = QtWidgets.QSpinBox()
        self.drainage_max_coupling_substeps_spin.setRange(1, 1024)
        self.drainage_max_coupling_substeps_spin.setValue(64)
        self.drainage_max_coupling_substeps_spin.setToolTip(
            "Maximum adaptive drainage substeps allowed when the 1D stability\n"
            "controller tightens the drainage timestep automatically."
        )
        self.drainage_head_deadband_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_head_deadband_spin.setRange(0.0, 10.0)
        self.drainage_head_deadband_spin.setDecimals(6)
        self.drainage_head_deadband_spin.setValue(1.0e-3)
        self.drainage_head_deadband_spin.setToolTip(
            "Head deadband used before drainage link and inlet exchange updates.\n"
            "Larger values reduce chatter near balanced states."
        )
        self.drainage_dynamic_relaxation_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_dynamic_relaxation_spin.setRange(0.0, 1.0)
        self.drainage_dynamic_relaxation_spin.setDecimals(3)
        self.drainage_dynamic_relaxation_spin.setSingleStep(0.05)
        self.drainage_dynamic_relaxation_spin.setValue(1.0)
        self.drainage_dynamic_relaxation_spin.setToolTip(
            "Dynamic-wave flow relaxation factor.\n"
            "1.0 keeps the full update; lower values damp oscillatory link-flow response."
        )
        self.drainage_adaptive_depth_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_adaptive_depth_fraction_spin.setRange(0.001, 1.0)
        self.drainage_adaptive_depth_fraction_spin.setDecimals(3)
        self.drainage_adaptive_depth_fraction_spin.setSingleStep(0.01)
        self.drainage_adaptive_depth_fraction_spin.setValue(0.2)
        self.drainage_adaptive_depth_fraction_spin.setToolTip(
            "Adaptive drainage substepping threshold based on fractional node-depth\n"
            "change per substep. Lower values are more conservative."
        )
        self.drainage_adaptive_wave_courant_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_adaptive_wave_courant_spin.setRange(0.001, 10.0)
        self.drainage_adaptive_wave_courant_spin.setDecimals(3)
        self.drainage_adaptive_wave_courant_spin.setSingleStep(0.05)
        self.drainage_adaptive_wave_courant_spin.setValue(0.5)
        self.drainage_adaptive_wave_courant_spin.setToolTip(
            "Adaptive drainage substepping target for dynamic-wave links based on\n"
            "wave Courant number. Lower values are more conservative."
        )
        self.drainage_implicit_iters_spin = QtWidgets.QSpinBox()
        self.drainage_implicit_iters_spin.setRange(1, 8)
        self.drainage_implicit_iters_spin.setValue(2)
        self.drainage_implicit_iters_spin.setToolTip(
            "Number of implicit predictor/corrector inner iterations per drainage substep\n"
            "(GPU path only). 1 = explicit single-pass; 2-4 gives better mass conservation\n"
            "at ~linear cost per extra iteration."
        )
        self.drainage_implicit_relax_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_implicit_relax_spin.setRange(0.1, 1.0)
        self.drainage_implicit_relax_spin.setDecimals(2)
        self.drainage_implicit_relax_spin.setSingleStep(0.05)
        self.drainage_implicit_relax_spin.setValue(0.5)
        self.drainage_implicit_relax_spin.setToolTip(
            "Relaxation factor for implicit coupling iterates (GPU path only).\n"
            "1.0 = no relaxation (full update); 0.5 damps oscillations between iterates."
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
        param_form.addRow("GPU diag sync (steps):", self.gpu_diag_sync_interval_spin)
        param_form.addRow("CUDA graph replay:", self.enable_cuda_graphs_chk)
        param_form.addRow("Max rel depth increase:", self.max_rel_depth_increase_spin)
        param_form.addRow("Max source dh/step:", self.max_source_depth_step_spin)
        param_form.addRow("Max source rate:", self.max_source_rate_spin)
        param_form.addRow("Extreme rain mode:", self.extreme_rain_mode_chk)
        param_form.addRow("Source CFL beta:", self.source_cfl_beta_spin)
        param_form.addRow("Source max substeps:", self.source_max_substeps_spin)
        param_form.addRow("True source subcycling:", self.source_true_subcycling_chk)
        param_form.addRow("IMEX source split:", self.source_imex_split_chk)
        param_form.addRow("Stage-coupled IMEX-RK2 sources:", self.source_stage_coupled_imex_rk2_chk)
        param_form.addRow("Shallow damping depth:", self.shallow_damping_depth_spin)
        param_form.addRow("Shallow-front recon fallback:", self.shallow_front_recon_fallback_chk)
        param_form.addRow("Front flux damping:", self.front_flux_damping_spin)
        param_form.addRow("Active-set hysteresis:", self.active_set_hysteresis_chk)
        param_form.addRow("Depth cap:", self.depth_cap_spin)
        param_form.addRow("Momentum cap min speed:", self.momentum_cap_min_speed_spin)
        param_form.addRow("Momentum cap celerity mult:", self.momentum_cap_celerity_mult_spin)
        param_form.addRow("Max inv area:", self.max_inv_area_spin)
        param_form.addRow("CFL lambda cap:", self.cfl_lambda_cap_spin)
        param_form.addRow("Rain rate:", self.rain_rate_spin)
        param_form.addRow("Default CN:", self.cn_default_spin)
        param_form.addRow("SCS Ia/S ratio:", self.ia_ratio_spin)
        param_form.addRow("Spatial rainfall:", self.use_spatial_rain_cn_chk)
        param_form.addRow("Infiltration method:", self.infiltration_method_combo)
        param_form.addRow("Storm area layer (optional):", self.storm_area_layer_combo)
        param_form.addRow("Rain boundary buffer rings:", self.rain_boundary_buffer_rings_spin)
        param_form.addRow("Internal flow layer:", self.internal_flow_layer_combo)
        param_form.addRow("Internal flow field:", self.internal_flow_field_edit)
        param_form.addRow("Run duration (hr or HH:MM):", self.run_time_edit)
        param_form.addRow("Reconstruction:", self.reconstruction_combo)
        param_form.addRow("Temporal order:", self.temporal_order_combo)
        param_form.addRow("GPU solver mode:", self.godunov_mode_combo)
        param_form.addRow("Degenerate cell mode:", self.degen_mode_combo)
        param_form.addRow("Coupling loop:", self.coupling_loop_combo)
        param_form.addRow("Drainage equation set:", self.drainage_solver_mode_combo)
        param_form.addRow("Drainage solver backend:", self.drainage_backend_combo)
        param_form.addRow("Drainage GPU method:", self.drainage_gpu_method_combo)
        param_form.addRow("Drainage substeps:", self.drainage_coupling_substeps_spin)
        param_form.addRow("Drainage max adaptive substeps:", self.drainage_max_coupling_substeps_spin)
        param_form.addRow("Drainage head deadband:", self.drainage_head_deadband_spin)
        param_form.addRow("Drainage dynamic relaxation:", self.drainage_dynamic_relaxation_spin)
        param_form.addRow("Drainage adaptive depth fraction:", self.drainage_adaptive_depth_fraction_spin)
        param_form.addRow("Drainage adaptive wave Courant:", self.drainage_adaptive_wave_courant_spin)
        param_form.addRow("Drainage implicit iterations (GPU):", self.drainage_implicit_iters_spin)
        param_form.addRow("Drainage implicit relaxation (GPU):", self.drainage_implicit_relax_spin)
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
        msg_txt = str(msg)
        self._runtime_log_lines.append(msg_txt)
        self.log_view.appendPlainText(msg_txt)
        QtWidgets.QApplication.processEvents()

    def _log_exception(self, context: str, exc: Exception) -> None:
        """Emit a concise error and full traceback into the runtime log pane."""
        self._log(f"{context}: {exc}")
        tb_txt = traceback.format_exc()
        if tb_txt:
            self._log("--- traceback begin ---")
            for ln in tb_txt.rstrip().splitlines():
                self._log(ln)
            self._log("--- traceback end ---")

    def _refresh_layer_group_combo(self):
        if not _HAVE_QGIS_CORE or QgsProject is None or not hasattr(self, "layer_group_combo"):
            return
        keep = self.layer_group_combo.currentData() if self.layer_group_combo.count() > 0 else None
        self.layer_group_combo.clear()
        self.layer_group_combo.addItem("(no group)", None)
        try:
            root = QgsProject.instance().layerTreeRoot()
        except Exception:
            root = None
        if root is None:
            return

        def _walk(node, prefix=""):
            out = []
            try:
                children = list(node.children())
            except Exception:
                children = []
            for ch in children:
                if hasattr(ch, "children"):
                    try:
                        nm = str(ch.name() or "")
                    except Exception:
                        nm = ""
                    if not nm:
                        continue
                    path = f"{prefix}/{nm}" if prefix else nm
                    out.append(path)
                    out.extend(_walk(ch, path))
            return out

        groups = sorted(set(_walk(root)))
        for gp in groups:
            self.layer_group_combo.addItem(gp, gp)
        if keep is not None:
            idx = self.layer_group_combo.findData(keep)
            if idx >= 0:
                self.layer_group_combo.setCurrentIndex(idx)

    def _group_layer_ids_by_path(self, group_path: str) -> set:
        ids = set()
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return ids
        target = str(group_path or "").strip()
        if not target:
            return ids
        try:
            root = QgsProject.instance().layerTreeRoot()
        except Exception:
            return ids

        parts = [p for p in target.split("/") if p]
        node = root
        for part in parts:
            nxt = None
            try:
                for ch in node.children():
                    if hasattr(ch, "children") and str(ch.name() or "") == part:
                        nxt = ch
                        break
            except Exception:
                nxt = None
            if nxt is None:
                return ids
            node = nxt

        def _collect(n):
            try:
                children = list(n.children())
            except Exception:
                children = []
            for ch in children:
                if hasattr(ch, "children"):
                    _collect(ch)
                elif hasattr(ch, "layer"):
                    try:
                        lyr = ch.layer()
                        if lyr is not None:
                            ids.add(str(lyr.id()))
                    except Exception:
                        pass

        _collect(node)
        return ids

    def _autopopulate_layer_combos_from_group(self):
        gp = self.layer_group_combo.currentData() if hasattr(self, "layer_group_combo") else None
        if not gp:
            self._log("Autopopulate skipped: select a layer group first.")
            return
        layer_ids = self._group_layer_ids_by_path(str(gp))
        if not layer_ids:
            self._log(f"Autopopulate: no layers found in group '{gp}'.")
            return

        defaults = {
            "nodes_layer_combo": ["swe2d_topo_nodes", "swe2d_mesh_nodes"],
            "cells_layer_combo": ["swe2d_mesh_cells", "swe2d_topo_regions"],
            "terrain_layer_combo": ["swe2d_terrain"],
            "manning_layer_combo": ["swe2d_manning_zones"],
            "cn_layer_combo": ["swe2d_cn_zones"],
            "rain_gage_layer_combo": ["swe2d_rain_gages"],
            "hyetograph_layer_combo": ["swe2d_hyetographs"],
            "sample_lines_layer_combo": ["swe2d_sample_lines"],
            "drain_nodes_layer_combo": ["swe2d_drainage_nodes"],
            "drain_links_layer_combo": ["swe2d_drainage_links"],
            "drain_inlets_layer_combo": ["swe2d_drainage_inlets"],
            "drain_node_inlets_layer_combo": ["swe2d_drainage_node_inlets"],
            "structures_layer_combo": ["swe2d_structures"],
            "bc_lines_layer_combo": ["swe2d_bc_lines"],
            "topo_nodes_combo": ["swe2d_topo_nodes"],
            "topo_arcs_combo": ["swe2d_topo_arcs"],
            "topo_regions_combo": ["swe2d_topo_regions"],
            "topo_constraints_combo": ["swe2d_topo_constraints"],
            "topo_quad_edges_combo": ["swe2d_topo_quad_edges"],
        }

        def _norm(s: str) -> str:
            return str(s or "").strip().lower()

        assigned = 0
        for combo_attr, names in defaults.items():
            combo = getattr(self, combo_attr, None)
            if combo is None:
                continue
            want = {_norm(nm) for nm in names}
            chosen = -1
            for i in range(combo.count()):
                lid = combo.itemData(i)
                if lid is None or str(lid) not in layer_ids:
                    continue
                txt = _norm(combo.itemText(i))
                if txt in want:
                    chosen = i
                    break
            if chosen >= 0:
                combo.setCurrentIndex(chosen)
                assigned += 1

        self._log(f"Autopopulate from group '{gp}': assigned {assigned} layer selectors.")

    def closeEvent(self, event):
        self._persist_project_layer_bindings()
        self._persist_project_workbench_state()
        self._terminate_topology_mesh_run(reason="dialog-close", update_status=False, emit_log=False)
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

    def showEvent(self, event):
        """Restore workbench state when dialog is shown; timing is more reliable than __init__."""
        super().showEvent(event)
        if not hasattr(self, "_workbench_state_restored_on_show"):
            self._workbench_state_restored_on_show = False
        if not self._workbench_state_restored_on_show:
            self._log("[DEBUG] showEvent: attempting to restore workbench state")
            self._restore_project_workbench_state()
            self._workbench_state_restored_on_show = True

    def _set_topology_mesh_busy(self, busy: bool, status_msg: Optional[str] = None):
        try:
            self.topo_generate_btn.setEnabled(not busy)
        except Exception:
            pass
        try:
            self.topo_terminate_btn.setEnabled(bool(busy))
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

    def _opt_bool(self, value: object, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _opt_float(self, value: object, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _effective_topology_timeout_sec(self, backend_name: str, mesh_options: Optional[Dict[str, object]]) -> float:
        base = float(self._topology_mesh_timeout_sec)
        if backend_name != "gmsh":
            return base
        opts = dict(mesh_options or {})
        gmsh_loop_enabled = self._opt_bool(opts.get("gmsh_quality_enable"), False)
        if not gmsh_loop_enabled:
            return base
        budget_s = max(1.0, self._opt_float(opts.get("gmsh_quality_time_limit_s"), 60.0))
        # Let Gmsh's internal quality-loop budget decide completion and keep
        # external watchdog comfortably above it to avoid preempting best-candidate return.
        return max(base, budget_s + 30.0)

    def _terminate_topology_mesh_run(
        self,
        reason: str,
        update_status: bool = True,
        emit_log: bool = True,
    ) -> bool:
        fut = self._topology_mesh_future
        if fut is None:
            return False

        backend_name = self._topology_mesh_backend or "unknown"
        run_mode = self._topology_mesh_run_mode
        elapsed_str = self._format_elapsed(self._topology_mesh_started_at)

        self._topology_mesh_timer.stop()
        self._topology_mesh_future = None
        self._topology_mesh_started_at = None
        self._topology_mesh_poll_count = 0

        try:
            fut.cancel()
        except Exception:
            pass

        if backend_name == "gmsh" and self._topology_mesh_process_pool is not None:
            try:
                self._topology_mesh_process_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._topology_mesh_process_pool = None

        if update_status:
            self.topo_status_lbl.setText(f"Topology meshing terminated by user (backend '{backend_name}').")
        if emit_log:
            self._log(
                "mesh> terminate "
                f"backend={backend_name} mode={run_mode} reason={reason} elapsed={elapsed_str}"
            )
        self._set_topology_mesh_busy(False)
        return True

    def _on_terminate_topology_mesh(self):
        if not self._terminate_topology_mesh_run(reason="user"):
            self._log("No topology mesh run is currently active.")

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
        self._topology_mesh_active_timeout_sec = self._effective_topology_timeout_sec(
            backend_name,
            self._topology_mesh_options,
        )

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
            f"timeout={self._topology_mesh_active_timeout_sec:.0f}s "
            f"elapsed={self._format_elapsed(self._topology_mesh_started_at)}"
        )
        if backend_name == "gmsh":
            gmsh_loop_enabled = self._opt_bool(self._topology_mesh_options.get("gmsh_quality_enable"), False)
            gmsh_budget = max(1.0, self._opt_float(self._topology_mesh_options.get("gmsh_quality_time_limit_s"), 60.0))
            self._log(
                "mesh> gmsh-quality "
                f"enabled={gmsh_loop_enabled} budget={gmsh_budget:.1f}s "
                f"effective-timeout={self._topology_mesh_active_timeout_sec:.1f}s"
            )
            if not gmsh_loop_enabled:
                self._log(
                    "mesh> note gmsh time budget is only enforced when 'Enable Gmsh iterative quality loop' is ON"
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

        if elapsed > self._topology_mesh_active_timeout_sec and not fut.done():
            backend_name = self._topology_mesh_backend or "unknown"
            run_mode = self._topology_mesh_run_mode
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
                f"Topology meshing timed out after {self._topology_mesh_active_timeout_sec:.0f}s "
                f"(backend '{backend_name}')."
            )
            self._log(
                "mesh> timeout "
                f"backend={backend_name} mode={run_mode} elapsed={elapsed:.2f}s "
                f"limit={self._topology_mesh_active_timeout_sec:.0f}s"
            )

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
        fallback_restarted = False

        try:
            mesh = fut.result()
            n_nodes = int(np.asarray(mesh.node_x).size)
            n_faces = max(0, int(np.asarray(mesh.cell_face_offsets).size) - 1)
            if n_nodes <= 0 or n_faces <= 0:
                raise RuntimeError(
                    f"Topology backend '{backend_name}' produced an empty mesh "
                    f"(nodes={n_nodes}, faces={n_faces})."
                )
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
                    "after automatic fallback with constraints disabled. "
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
            quality_summary = getattr(mesh, "quality_summary", None)
            if isinstance(quality_summary, dict):
                best_stats = quality_summary.get("best_stats", {})
                try:
                    self._log(
                        "mesh> gmsh-quality-summary "
                        f"attempts={int(quality_summary.get('attempts', 0))} "
                        f"strict={bool(quality_summary.get('strict_requested', False))} "
                        f"passed={bool(quality_summary.get('had_passing_candidate', False))} "
                        f"fail_cells(any/angle/aspect/area/non_orth)="
                        f"{int(float(best_stats.get('failed_any_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_min_angle_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_max_aspect_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_min_area_cells', 0.0)))}/"
                        f"{int(float(best_stats.get('failed_max_non_orth_cells', 0.0)))}"
                    )
                except Exception:
                    pass
            self._result_data = None
            self.view_mode_combo.setCurrentText("Mesh")
            self._refresh_plot()
        except NotImplementedError as exc:
            self.topo_status_lbl.setText(str(exc))
            self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        except RuntimeError as exc:
            err_txt = str(exc)
            err_l = err_txt.lower()
            empty_mesh_failure = ("empty mesh" in err_l) or ("non-empty mesh" in err_l)
            conceptual = self._topology_mesh_conceptual
            can_retry_without_constraints = (
                backend_name == "gmsh"
                and run_mode == "full"
                and not self._topology_mesh_auto_fallback_used
                and conceptual is not None
                and bool(getattr(conceptual, "constraints", []))
            )
            if empty_mesh_failure and can_retry_without_constraints:
                try:
                    fallback_conceptual = _clone_conceptual_without_constraints(conceptual)
                    self._topology_mesh_auto_fallback_used = True
                    self._log(
                        "mesh> fallback "
                        f"backend={backend_name} action=retry_without_constraints "
                        f"reason=empty-mesh elapsed={elapsed_str}"
                    )
                    self._start_topology_mesh_async(
                        fallback_conceptual,
                        backend_name,
                        default_cell_type,
                        self._topology_mesh_options,
                        run_mode="fallback-no-constraints",
                    )
                    fallback_restarted = True
                    return
                except Exception as fallback_exc:
                    self._log(
                        "mesh> fallback-fail "
                        f"backend={backend_name} elapsed={elapsed_str} error={fallback_exc}"
                    )
            self.topo_status_lbl.setText(err_txt)
            self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        except Exception as exc:
            self.topo_status_lbl.setText(f"Topology meshing failed: {exc}")
            self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
        finally:
            if not fallback_restarted:
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
        is_drain_node_inlets = "drainage_node_inlets" in lname
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
            self._set_expression_constraint(layer, "bc_type", '"bc_type" IN (1,2,3,4,5,6,7,102,103)')
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
            self._set_value_map_editor(layer, "units", _RAIN_GAGE_UNITS_VALUE_MAP)
            self._set_expression_constraint(layer, "gage_id", 'length(trim("gage_id")) > 0')
            self._set_expression_constraint(layer, "hyetograph_id", 'length(trim("hyetograph_id")) > 0')
            self._set_expression_constraint(layer, "units", '"units" IS NULL OR "units" IN (\'mm/hr\',\'in/hr\',\'mm\',\'in\')')

        if is_hyetograph:
            self._set_value_map_editor(layer, "value_type", _HYETOGRAPH_VALUE_TYPE_MAP)
            self._set_value_map_editor(layer, "units", _HYETOGRAPH_UNITS_VALUE_MAP)
            self._set_expression_constraint(layer, "hyetograph_id", 'length(trim("hyetograph_id")) > 0')
            self._set_expression_constraint(layer, "Time", 'length(trim("Time")) > 0')
            self._set_expression_constraint(layer, "Value", '"Value" >= 0')
            self._set_expression_constraint(layer, "value_type", '"value_type" IS NULL OR "value_type" IN (\'intensity\',\'incremental\',\'cumulative\')')
            self._set_expression_constraint(layer, "units", '"units" IS NULL OR "units" IN (\'mm/hr\',\'in/hr\',\'mm\',\'in\')')

        if is_sample_lines:
            self._set_expression_constraint(layer, "line_id", '"line_id" IS NULL OR "line_id" >= 0')
            self._set_expression_constraint(layer, "enabled", '"enabled" IS NULL OR "enabled" IN (0,1)')
            self._set_expression_constraint(layer, "priority", '"priority" IS NULL OR "priority" >= 0')

        if is_drain_nodes:
            node_field_names = set(layer.fields().names())
            self._set_value_map_editor(layer, "node_type", _DRAIN_NODE_TYPE_VALUE_MAP)
            self._set_expression_constraint(layer, "node_id", 'length(trim("node_id")) > 0')
            self._set_expression_constraint(layer, "node_type", '"node_type" IN (\'junction\',\'outfall\',\'storage\',\'inlet\')')
            self._set_expression_constraint(layer, "max_depth", '"max_depth" IS NULL OR "max_depth" > 0')
            self._set_expression_constraint(layer, "rim_elev", '"rim_elev" IS NULL OR "rim_elev" >= "invert_elev"')
            self._set_expression_constraint(layer, "crest_elev", '"crest_elev" IS NULL OR "crest_elev" >= "invert_elev"')
            self._set_expression_constraint(layer, "surface_area", '"surface_area" IS NULL OR "surface_area" > 0')
            if "outfall_area" in node_field_names:
                self._set_expression_constraint(layer, "outfall_area", '"outfall_area" IS NULL OR "outfall_area" > 0')
            if "zero_storage" in node_field_names:
                self._set_expression_constraint(layer, "zero_storage", '"zero_storage" IS NULL OR "zero_storage" IN (0,1)')

        if is_drain_links:
            self._set_value_map_editor(layer, "link_type", _DRAIN_LINK_TYPE_VALUE_MAP)
            self._set_value_map_editor(layer, "link_shape", _DRAIN_LINK_SHAPE_VALUE_MAP)
            self._set_expression_constraint(layer, "link_id", 'length(trim("link_id")) > 0')
            self._set_expression_constraint(layer, "from_node", 'length(trim("from_node")) > 0')
            self._set_expression_constraint(layer, "to_node", 'length(trim("to_node")) > 0')
            self._set_expression_constraint(layer, "link_type", '"link_type" IN (\'conduit\',\'lateral_simple\',\'pump\',\'weir\',\'orifice\')')
            self._set_expression_constraint(layer, "link_shape", '"link_shape" IS NULL OR "link_shape" IN (\'circular\',\'box\',\'pipe_arch\',\'custom\')')
            self._set_expression_constraint(layer, "length", '"length" IS NULL OR "length" > 0')
            self._set_expression_constraint(layer, "roughness_n", '"roughness_n" IS NULL OR "roughness_n" > 0')
            self._set_expression_constraint(layer, "diameter", '"diameter" IS NULL OR "diameter" > 0')
            self._set_expression_constraint(layer, "span", '"span" IS NULL OR "span" > 0')
            self._set_expression_constraint(layer, "rise", '"rise" IS NULL OR "rise" > 0')
            self._set_expression_constraint(layer, "area_m2", '"area_m2" IS NULL OR "area_m2" > 0')

        if is_drain_inlets:
            self._set_expression_constraint(layer, "inlet_type_id", 'length(trim("inlet_type_id")) > 0')
            self._set_expression_constraint(layer, "weir_length", '"weir_length" IS NULL OR "weir_length" > 0')
            self._set_expression_constraint(layer, "orifice_area", '"orifice_area" IS NULL OR "orifice_area" > 0')
            self._set_expression_constraint(layer, "coeff_weir", '"coeff_weir" IS NULL OR "coeff_weir" > 0')
            self._set_expression_constraint(layer, "coeff_orifice", '"coeff_orifice" IS NULL OR "coeff_orifice" > 0')
            self._set_expression_constraint(layer, "max_capture", '"max_capture" IS NULL OR "max_capture" > 0')

        if is_drain_node_inlets:
            self._set_expression_constraint(layer, "node_id", 'length(trim("node_id")) > 0')
            self._set_expression_constraint(layer, "inlet_type_id", 'length(trim("inlet_type_id")) > 0')
            self._set_expression_constraint(layer, "inlet_count", '"inlet_count" IS NULL OR "inlet_count" > 0')

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

        self._project_layer_state_blocked = True
        try:
            keep_nodes = self.nodes_layer_combo.currentData()
            keep_cells = self.cells_layer_combo.currentData()
            keep_terrain = self.terrain_layer_combo.currentData()
            keep_manning = self.manning_layer_combo.currentData() if hasattr(self, "manning_layer_combo") else None
            keep_cn = self.cn_layer_combo.currentData() if hasattr(self, "cn_layer_combo") else None
            keep_rain_gages = self.rain_gage_layer_combo.currentData() if hasattr(self, "rain_gage_layer_combo") else None
            keep_hyetograph = self.hyetograph_layer_combo.currentData() if hasattr(self, "hyetograph_layer_combo") else None
            keep_storm_area = self.storm_area_layer_combo.currentData() if hasattr(self, "storm_area_layer_combo") else None
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
            keep_drain_node_inlets = self.drain_node_inlets_layer_combo.currentData() if hasattr(self, "drain_node_inlets_layer_combo") else None
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
            if hasattr(self, "storm_area_layer_combo"):
                self.storm_area_layer_combo.clear()
                self.storm_area_layer_combo.addItem("(none)", None)
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
            if hasattr(self, "drain_node_inlets_layer_combo"):
                self.drain_node_inlets_layer_combo.clear()
                self.drain_node_inlets_layer_combo.addItem("(none)", None)
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
                        elif geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
                            self.cells_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "manning_layer_combo"):
                                self.manning_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "cn_layer_combo"):
                                self.cn_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "storm_area_layer_combo"):
                                self.storm_area_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "topo_regions_combo"):
                                self.topo_regions_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "topo_constraints_combo"):
                                self.topo_constraints_combo.addItem(lyr.name(), lyr.id())
                        elif geom_type in (
                            QgsWkbTypes.GeometryType.UnknownGeometry,
                            getattr(QgsWkbTypes.GeometryType, "NullGeometry", QgsWkbTypes.GeometryType.UnknownGeometry),
                        ):
                            lname = str(lyr.name() or "").lower()
                            if hasattr(self, "hyetograph_layer_combo") and "hyetograph" in lname:
                                self.hyetograph_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "drain_inlets_layer_combo") and "drainage_inlets" in lname:
                                self.drain_inlets_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "drain_node_inlets_layer_combo") and "drainage_node_inlets" in lname:
                                self.drain_node_inlets_layer_combo.addItem(lyr.name(), lyr.id())
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
            if hasattr(self, "storm_area_layer_combo"):
                _restore(self.storm_area_layer_combo, keep_storm_area)
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
            if hasattr(self, "drain_node_inlets_layer_combo") and keep_drain_node_inlets is not None:
                _restore(self.drain_node_inlets_layer_combo, keep_drain_node_inlets)
            if hasattr(self, "structures_layer_combo") and keep_structures is not None:
                _restore(self.structures_layer_combo, keep_structures)

            self._update_unit_system_from_crs()
            self._refresh_layer_group_combo()
            self._update_topology_control_summary()
        finally:
            self._project_layer_state_blocked = False

        self._restore_project_layer_bindings()
        self._persist_project_layer_bindings()

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
        size_scales = tuple(self._parse_csv_number_list(self.topo_quality_size_scales_edit.text(), float) or [1.0])
        smooth_increments = tuple(self._parse_csv_number_list(self.topo_quality_smooth_increments_edit.text(), int) or [0])
        return {
            "gmsh_tri_algorithm": int(self.topo_gmsh_tri_algo_combo.currentData() or 6),
            "gmsh_quad_algorithm": int(self.topo_gmsh_quad_algo_combo.currentData() or 6),
            "gmsh_recombination_algorithm": int(self.topo_gmsh_recombine_algo_combo.currentData() or 1),
            "gmsh_smoothing": int(self.topo_gmsh_smoothing_spin.value()),
            "gmsh_optimize_iters": int(self.topo_gmsh_optimize_iters_spin.value()),
            "gmsh_optimize_netgen": bool(self.topo_gmsh_optimize_netgen_chk.isChecked()),
            "gmsh_verbosity": int(self.topo_gmsh_verbosity_spin.value()),
            "gmsh_quality_enable": bool(self.topo_gmsh_quality_enable_chk.isChecked()),
            "gmsh_quality_max_iterations": int(self.topo_gmsh_quality_max_iters_spin.value()),
            "gmsh_quality_time_limit_s": float(self.topo_gmsh_quality_time_limit_spin.value()),
            "gmsh_min_angle_deg": float(self.topo_quality_min_angle_spin.value()),
            "gmsh_max_aspect_ratio": float(self.topo_quality_max_aspect_spin.value()),
            "gmsh_max_non_orth_deg": float(self.topo_quality_max_non_orth_spin.value()),
            "gmsh_min_area_rel_bbox": float(self.topo_quality_min_area_edit.text().strip() or "0"),
            "gmsh_quality_strict": bool(self.topo_quality_strict_chk.isChecked()),
            "gmsh_quality_size_scales": size_scales,
            "gmsh_quality_smooth_increments": smooth_increments,
            "tqmesh_min_angle_deg": float(self.topo_quality_min_angle_spin.value()),
            "tqmesh_max_aspect_ratio": float(self.topo_quality_max_aspect_spin.value()),
            "tqmesh_min_area_rel_bbox": float(self.topo_quality_min_area_edit.text().strip() or "0"),
            "tqmesh_quality_strict": bool(self.topo_quality_strict_chk.isChecked()),
            "tqmesh_size_scales": size_scales,
            "tqmesh_smooth_increments": smooth_increments,
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

        quality_hint = (
            " Quality UI: min angle >= {min_angle:.1f} deg, max aspect <= {max_aspect:.2f}, "
            "max non-orth <= {max_non_orth:.1f} deg, min area/bbox >= {min_area}, strict={strict}; "
            "retry scales={size_scales}, smooth increments={smooth_increments}; "
            "Gmsh loop={gmsh_loop}, attempts={attempts}, budget={budget:.1f}s."
        ).format(
            min_angle=float(self.topo_quality_min_angle_spin.value()) if hasattr(self, "topo_quality_min_angle_spin") else 0.0,
            max_aspect=float(self.topo_quality_max_aspect_spin.value()) if hasattr(self, "topo_quality_max_aspect_spin") else 0.0,
            max_non_orth=float(self.topo_quality_max_non_orth_spin.value()) if hasattr(self, "topo_quality_max_non_orth_spin") else 0.0,
            min_area=str(self.topo_quality_min_area_edit.text()).strip() if hasattr(self, "topo_quality_min_area_edit") else "0",
            strict="on" if getattr(self, "topo_quality_strict_chk", None) is not None and self.topo_quality_strict_chk.isChecked() else "off",
            size_scales=str(self.topo_quality_size_scales_edit.text()).strip() if hasattr(self, "topo_quality_size_scales_edit") else "1.0",
            smooth_increments=str(self.topo_quality_smooth_increments_edit.text()).strip() if hasattr(self, "topo_quality_smooth_increments_edit") else "0",
            gmsh_loop="on" if getattr(self, "topo_gmsh_quality_enable_chk", None) is not None and self.topo_gmsh_quality_enable_chk.isChecked() else "off",
            attempts=int(self.topo_gmsh_quality_max_iters_spin.value()) if hasattr(self, "topo_gmsh_quality_max_iters_spin") else 0,
            budget=float(self.topo_gmsh_quality_time_limit_spin.value()) if hasattr(self, "topo_gmsh_quality_time_limit_spin") else 0.0,
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
            self.topo_controls_summary_lbl.setText(f"{backend_hint}{quality_hint} Current layers: {suffix}.")
        else:
            self.topo_controls_summary_lbl.setText(f"{backend_hint}{quality_hint}")

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
            f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double&field=max_depth:double&field=rim_elev:double&field=crest_elev:double&field=node_type:string(32)&field=surface_area:double&field=outfall_area:double&field=zero_storage:integer",
            "SWE2D_Drainage_Nodes",
            "memory",
        )
        drainage_links = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)&field=to_node:string(64)&field=link_type:string(32)&field=link_shape:string(32)&field=length:double&field=roughness_n:double&field=diameter:double&field=span:double&field=rise:double&field=area_m2:double&field=equiv_diameter_m:double&field=max_flow:double&field=cd:double",
            "SWE2D_Drainage_Links",
            "memory",
        )
        drainage_inlets = QgsVectorLayer(
            "None?field=inlet_type_id:string(64)&field=name:string(128)&field=weir_length:double&field=orifice_area:double&field=coeff_weir:double&field=coeff_orifice:double&field=max_capture:double&field=description:string(256)",
            "SWE2D_Drainage_Inlets",
            "memory",
        )
        drainage_node_inlets = QgsVectorLayer(
            "None?field=node_id:string(64)&field=inlet_type_id:string(64)&field=inlet_count:double&field=crest_offset:double&field=description:string(256)",
            "SWE2D_Drainage_Node_Inlets",
            "memory",
        )
        structures = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width:double&field=height:double&field=diameter:double&field=length:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double",
            "SWE2D_Structures",
            "memory",
        )
        hydro_tbl = QgsVectorLayer(
            "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)&field=Value:double&field=description:string(256)",
            "SWE2D_Hydrographs",
            "memory",
        )

        for lyr in (nodes, arcs, regions, constraints, quad_edges, manning, bc_lines, sample_lines, drainage_nodes, drainage_links, drainage_inlets, drainage_node_inlets, structures, hydro_tbl):
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
        storm_areas = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=storm_id:integer&field=name:string(128)&field=priority:integer",
            "swe2d_storm_areas",
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
            f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double&field=max_depth:double&field=rim_elev:double&field=crest_elev:double&field=node_type:string(32)&field=surface_area:double&field=outfall_area:double&field=zero_storage:integer",
            "swe2d_drainage_nodes",
            "memory",
        )
        drainage_links = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)&field=to_node:string(64)&field=link_type:string(32)&field=link_shape:string(32)&field=length:double&field=roughness_n:double&field=diameter:double&field=span:double&field=rise:double&field=area_m2:double&field=equiv_diameter_m:double&field=max_flow:double&field=cd:double",
            "swe2d_drainage_links",
            "memory",
        )
        drainage_inlets = QgsVectorLayer(
            "None?field=inlet_type_id:string(64)&field=name:string(128)&field=weir_length:double&field=orifice_area:double&field=coeff_weir:double&field=coeff_orifice:double&field=max_capture:double&field=description:string(256)",
            "swe2d_drainage_inlets",
            "memory",
        )
        drainage_node_inlets = QgsVectorLayer(
            "None?field=node_id:string(64)&field=inlet_type_id:string(64)&field=inlet_count:double&field=crest_offset:double&field=description:string(256)",
            "swe2d_drainage_node_inlets",
            "memory",
        )
        structures = QgsVectorLayer(
            f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width:double&field=height:double&field=diameter:double&field=length:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double",
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
            storm_areas,
            cn_zones,
            hyetographs,
            hydro,
            drainage_nodes,
            drainage_links,
            drainage_inlets,
            drainage_node_inlets,
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

    def _migrate_2d_model_geopackage(self):
        """Add missing layers and columns to an existing 2D model GeoPackage."""
        if not _HAVE_QGIS_CORE:
            self._log("QGIS layer API unavailable; cannot migrate GeoPackage.")
            return

        gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select 2D Model GeoPackage to Update",
            "",
            "GeoPackage (*.gpkg)",
        )
        if not gpkg_path:
            return

        crs_auth = "EPSG:4326"
        try:
            crs = QgsProject.instance().crs()
            if crs is not None and crs.isValid():
                crs_auth = crs.authid() or crs_auth
        except Exception:
            pass

        # Canonical schema: list of (layer_name, memory_uri) pairs.
        # Geometry-less tables use "None?" as the URI prefix.
        layer_specs = [
            ("swe2d_topo_nodes",
             f"Point?crs={crs_auth}&field=node_id:integer"),
            ("swe2d_topo_arcs",
             f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer"),
            ("swe2d_topo_regions",
             f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double"
             "&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double"
             "&field=edge_len_3:double&field=edge_len_4:double"),
            ("swe2d_topo_constraints",
             f"Polygon?crs={crs_auth}&field=constraint_id:integer&field=target_size:double"
             "&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double"
             "&field=edge_len_3:double&field=edge_len_4:double"),
            ("swe2d_topo_quad_edges",
             f"LineString?crs={crs_auth}&field=region_id:integer&field=edge_id:integer"
             "&field=target_size:double&field=n_layers:integer&field=first_height:double"
             "&field=growth_rate:double"),
            ("swe2d_manning_zones",
             f"Polygon?crs={crs_auth}&field=zone_id:integer&field=n_mann:double&field=priority:integer"),
            ("swe2d_bc_lines",
             f"LineString?crs={crs_auth}&field=bc_type:integer&field=bc_value:double"
             "&field=priority:integer&field=hydrograph:string(1024)"
             "&field=hydrograph_id:string(64)&field=hydrograph_layer:string(128)"),
            ("swe2d_sample_lines",
             f"LineString?crs={crs_auth}&field=line_id:integer&field=name:string(128)"
             "&field=enabled:integer&field=priority:integer"),
            ("swe2d_rain_gages",
             f"Point?crs={crs_auth}&field=gage_id:string(64)&field=name:string(128)"
             "&field=hyetograph_id:string(64)&field=units:string(32)&field=priority:integer"),
            ("swe2d_storm_areas",
             f"Polygon?crs={crs_auth}&field=storm_id:integer&field=name:string(128)&field=priority:integer"),
            ("swe2d_cn_zones",
             f"Polygon?crs={crs_auth}&field=zone_id:integer&field=cn:double&field=priority:integer"),
            ("swe2d_hyetographs",
             "None?field=hyetograph_id:string(64)&field=Time:string(32)&field=Value:double"
             "&field=value_type:string(24)&field=units:string(24)&field=description:string(256)"),
            ("swe2d_hydrographs",
             "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)"
             "&field=Value:double&field=description:string(256)"),
            ("swe2d_drainage_nodes",
             f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double"
             "&field=max_depth:double&field=rim_elev:double&field=crest_elev:double"
             "&field=node_type:string(32)&field=surface_area:double"
             "&field=outfall_area:double&field=zero_storage:integer"),
            ("swe2d_drainage_links",
             f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)"
             "&field=to_node:string(64)&field=link_type:string(32)&field=link_shape:string(32)"
             "&field=length:double&field=roughness_n:double&field=diameter:double"
             "&field=span:double&field=rise:double&field=area_m2:double"
             "&field=equiv_diameter_m:double&field=max_flow:double&field=cd:double"),
            ("swe2d_drainage_inlets",
             "None?field=inlet_type_id:string(64)&field=name:string(128)"
             "&field=weir_length:double&field=orifice_area:double"
             "&field=coeff_weir:double&field=coeff_orifice:double"
             "&field=max_capture:double&field=description:string(256)"),
            ("swe2d_drainage_node_inlets",
             "None?field=node_id:string(64)&field=inlet_type_id:string(64)"
             "&field=inlet_count:double&field=crest_offset:double&field=description:string(256)"),
            ("swe2d_structures",
             f"LineString?crs={crs_auth}&field=structure_id:string(64)"
             "&field=structure_type:integer&field=crest_elev:double&field=enabled:integer"
             "&field=width:double&field=height:double&field=diameter:double"
             "&field=length:double&field=roughness_n:double&field=coeff:double"
             "&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double"),
        ]

        def _uri_fields(uri: str):
            """Parse field names and SQLite column types from a memory layer URI string."""
            fields = []
            for part in uri.split("&"):
                if not part.startswith("field="):
                    continue
                spec = part[len("field="):]
                if ":" not in spec:
                    continue
                fname, ftype_raw = spec.split(":", 1)
                ftype_lower = ftype_raw.lower()
                if ftype_lower.startswith("integer") or ftype_lower.startswith("int"):
                    sql_type = "INTEGER"
                elif ftype_lower.startswith("double") or ftype_lower.startswith("real"):
                    sql_type = "REAL"
                else:
                    sql_type = "TEXT"
                fields.append((fname, sql_type))
            return fields

        layers_added = []
        columns_added = []

        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            for layer_name, uri in layer_specs:
                # Check if the table already exists.
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (layer_name,),
                )
                exists = cur.fetchone() is not None

                if not exists:
                    # Write empty layer via QGIS driver (handles geometry and GPKG metadata).
                    mem_lyr = QgsVectorLayer(uri, layer_name, "memory")
                    if mem_lyr.isValid():
                        self._write_memory_layer_to_gpkg(
                            mem_lyr, gpkg_path, layer_name, create_file=False
                        )
                        layers_added.append(layer_name)
                else:
                    # Check for missing columns.
                    expected = _uri_fields(uri)
                    cur.execute(f"PRAGMA table_info(\"{layer_name}\")")
                    existing_cols = {row[1].lower() for row in cur.fetchall()}
                    for fname, sql_type in expected:
                        if fname.lower() not in existing_cols:
                            try:
                                cur.execute(
                                    f"ALTER TABLE \"{layer_name}\" ADD COLUMN \"{fname}\" {sql_type}"
                                )
                                columns_added.append(f"{layer_name}.{fname}")
                            except Exception as col_err:
                                self._log(
                                    f"[Migrate] Could not add column {layer_name}.{fname}: {col_err}"
                                )
            conn.commit()
        finally:
            conn.close()

        summary_parts = []
        if layers_added:
            summary_parts.append(f"Added {len(layers_added)} layer(s): {', '.join(layers_added)}")
        if columns_added:
            summary_parts.append(f"Added {len(columns_added)} column(s): {', '.join(columns_added)}")
        if not summary_parts:
            summary_parts.append("GeoPackage schema is already up to date — no changes needed.")

        summary = "; ".join(summary_parts)
        self._log(f"[Migrate] {summary}")
        self.layer_status_lbl.setText(f"GeoPackage updated: {summary}")

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
            "swe2d_storm_areas",
            "swe2d_cn_zones",
            "swe2d_hyetographs",
            "swe2d_hydrographs",
            "swe2d_drainage_nodes",
            "swe2d_drainage_links",
            "swe2d_drainage_inlets",
            "swe2d_drainage_node_inlets",
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

    # ------------------------------------------------------------------
    # Sprint 0: dockable multi-run results panel
    # ------------------------------------------------------------------

    def _maybe_create_results_panel(self):
        """Create the dockable results panel and register it with iface (hidden)."""
        try:
            try:
                from .swe2d_results_panel import SWE2DResultsPanel
            except ImportError:
                from swe2d_results_panel import SWE2DResultsPanel
        except ImportError:
            self._log("[Results Panel] swe2d_results_panel not found — panel unavailable.")
            return

        gpkg = self._model_gpkg_path or ""
        iface = getattr(self, "_iface", None)
        try:
            self._results_panel = SWE2DResultsPanel(
                gpkg_path=gpkg, iface=iface, parent=None
            )
            self._results_panel.setWindowTitle("SWE2D Results")
            try:
                self._results_panel.timestep_changed.disconnect(self._on_results_panel_timestep_changed)
            except Exception:
                pass
            self._results_panel.timestep_changed.connect(self._on_results_panel_timestep_changed)
            try:
                self._results_panel.velocity_overlay_changed.disconnect(self._on_results_panel_velocity_overlay_changed)
            except Exception:
                pass
            self._results_panel.velocity_overlay_changed.connect(self._on_results_panel_velocity_overlay_changed)
            try:
                self._results_panel.velocity_overlay_add_requested.disconnect(self._on_results_panel_velocity_overlay_add_requested)
            except Exception:
                pass
            self._results_panel.velocity_overlay_add_requested.connect(self._on_results_panel_velocity_overlay_add_requested)
            try:
                self._results_panel.restore_state()
            except Exception:
                pass
            if iface is not None:
                iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._results_panel)
                self._results_panel.hide()
        except Exception as exc:
            self._log(f"[Results Panel] Failed to create panel: {exc}")
            self._results_panel = None

    def _activate_sample_line_draw_tool(self):
        """Activate map-canvas tool for drawing one sample line."""
        canvas = self._resolve_map_canvas()
        if canvas is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Draw Sample Line",
                "Map canvas is not available. Open this dialog from a running QGIS session.",
            )
            return

        try:
            try:
                from .swe2d_map_tools import SWE2DLineDrawTool
            except ImportError:
                from swe2d_map_tools import SWE2DLineDrawTool
        except Exception as exc:
            self._log(f"[SampleLineTool] Could not load draw tool: {exc}")
            QtWidgets.QMessageBox.warning(
                self,
                "Draw Sample Line",
                "Could not load map draw tool. Ensure swe2d_map_tools.py is available.",
            )
            return

        if self._sample_line_draw_tool is None:
            self._sample_line_draw_tool = SWE2DLineDrawTool(canvas)

        try:
            self._sample_line_draw_tool.line_finished.disconnect(self._on_sample_line_drawn)
        except Exception:
            pass
        self._sample_line_draw_tool.line_finished.connect(self._on_sample_line_drawn)

        try:
            self._sample_line_prev_map_tool = canvas.mapTool()
        except Exception:
            self._sample_line_prev_map_tool = None

        canvas.setMapTool(self._sample_line_draw_tool)
        self._log("Sample line draw tool active: left-click to add vertices, right-click/double-click to finish.")

    def _next_sample_line_id(self, layer) -> int:
        next_id = 1
        if layer is None:
            return next_id
        fields = set(layer.fields().names())
        if "line_id" not in fields:
            return next_id
        for ft in layer.getFeatures():
            try:
                next_id = max(next_id, int(ft["line_id"]) + 1)
            except Exception:
                continue
        return next_id

    def _on_sample_line_drawn(self, geom):
        """Store drawn sample line into selected sample-lines layer."""
        if not _HAVE_QGIS_CORE or QgsFeature is None:
            return
        if geom is None or geom.isEmpty():
            return

        line_layer = self._combo_layer(self.sample_lines_layer_combo, "vector") if hasattr(self, "sample_lines_layer_combo") else None
        if line_layer is None:
            self._log("Drawn line ignored: no sample-lines layer selected.")
            QtWidgets.QMessageBox.warning(
                self,
                "Draw Sample Line",
                "Select a sample lines layer first.",
            )
            return

        feat = QgsFeature(line_layer.fields())
        feat.setGeometry(geom)
        fields = set(line_layer.fields().names())
        line_id = self._next_sample_line_id(line_layer)
        if "line_id" in fields:
            feat["line_id"] = int(line_id)
        if "name" in fields:
            feat["name"] = f"Line {line_id}"
        if "enabled" in fields:
            feat["enabled"] = 1
        if "priority" in fields:
            feat["priority"] = 0

        add_res = line_layer.dataProvider().addFeatures([feat])
        ok = bool(add_res[0]) if isinstance(add_res, tuple) else bool(add_res)
        if ok:
            line_layer.updateExtents()
            line_layer.triggerRepaint()
            iface = self._resolve_qgis_iface()
            if iface is not None and hasattr(iface, "mapCanvas"):
                try:
                    iface.mapCanvas().refresh()
                except Exception:
                    pass
            self._log(f"Sample line added: id={line_id}, length={float(geom.length()):.3f}")
            self._resample_latest_results_for_line(int(line_id))
        else:
            self._log("Failed to add drawn sample line feature to selected layer.")

        canvas = self._resolve_map_canvas()
        if canvas is not None and self._sample_line_prev_map_tool is not None:
            try:
                canvas.setMapTool(self._sample_line_prev_map_tool)
            except Exception:
                pass
        self._sample_line_prev_map_tool = None

    def _resample_latest_results_for_line(self, line_id: int):
        """Best-effort refresh of persisted line snapshots for a newly drawn line."""
        if line_id < 0:
            return
        try:
            sample_map = self._build_line_sampling_map()
        except Exception as exc:
            self._log(f"Sample line refresh skipped (map build failed): {exc}")
            return

        line_sample = None
        for sm in sample_map:
            try:
                if int(sm.get("line_id", -1)) == int(line_id):
                    line_sample = sm
                    break
            except Exception:
                continue
        if line_sample is None:
            self._log(f"Sample line refresh skipped: no intersecting cells for line {line_id}.")
            return

        result_data = self._result_data if isinstance(getattr(self, "_result_data", None), dict) else None
        if not result_data:
            self._log("Sample line refresh skipped: no in-memory result state is available yet.")
            return

        try:
            h = np.asarray(result_data.get("h"), dtype=np.float64).ravel()
            hu = np.asarray(result_data.get("hu"), dtype=np.float64).ravel()
            hv = np.asarray(result_data.get("hv"), dtype=np.float64).ravel()
            cell_bed = np.asarray(self._mesh_cell_min_bed(), dtype=np.float64).ravel()
        except Exception as exc:
            self._log(f"Sample line refresh skipped (state decode failed): {exc}")
            return

        if h.size == 0 or hu.size != h.size or hv.size != h.size or cell_bed.size != h.size:
            self._log("Sample line refresh skipped: state arrays are not aligned.")
            return

        db_path = self._line_results_latest_db_path if self._line_results_latest_db_path and os.path.exists(self._line_results_latest_db_path) else self._current_line_results_storage_path()
        run_id = str(self._line_results_latest_run_id or "").strip()
        if not db_path or not run_id or not os.path.exists(db_path):
            self._log("Sample line refresh deferred: no persisted run context available yet.")
            return

        chosen, rows, profile_rows = self._load_line_results_from_geopackage(db_path, run_id=run_id)
        if not chosen:
            self._log("Sample line refresh deferred: latest run rows are unavailable.")
            return

        t_latest = 0.0
        if rows:
            try:
                t_latest = max(float(r.get("t_s", 0.0)) for r in rows)
            except Exception:
                t_latest = 0.0

        new_rows, new_profile_rows = self._sample_line_metrics(
            [line_sample],
            t_latest,
            h,
            hu,
            hv,
            cell_bed,
        )
        if not new_rows and not new_profile_rows:
            self._log(f"Sample line refresh skipped: no metrics produced for line {line_id}.")
            return

        merged_rows = [r for r in rows if int(r.get("line_id", -1)) != int(line_id)]
        merged_rows.extend(new_rows)

        merged_profile_rows = [r for r in profile_rows if int(r.get("line_id", -1)) != int(line_id)]
        merged_profile_rows.extend(new_profile_rows)

        mesh_interval_s = max(1.0, self._parse_time_hours(self.output_interval_edit.text()) * 3600.0)
        line_interval_s = max(1.0, self._parse_time_hours(self.line_output_interval_edit.text()) * 3600.0)
        self._persist_line_results_to_geopackage(
            db_path,
            chosen,
            merged_rows,
            mesh_interval_s=mesh_interval_s,
            line_interval_s=line_interval_s,
            profile_rows=merged_profile_rows,
        )
        self._log(f"Sample line refresh complete: line {line_id} merged into run {chosen}.")

        panel = getattr(self, "_results_panel", None)
        if panel is not None:
            try:
                panel.set_gpkg_path(db_path)
            except Exception:
                pass

    def _show_results_panel(self):
        """Show (and lazily create) the dockable results panel."""
        if self._results_panel is None:
            self._maybe_create_results_panel()
        if self._results_panel is None:
            QtWidgets.QMessageBox.warning(
                self, "Results Panel",
                "Could not create results panel.\n"
                "Ensure swe2d_results_panel.py is in the plugin directory."
            )
            return
        gpkg = self._model_gpkg_path or ""
        if gpkg and gpkg != self._results_panel._gpkg_path:
            self._results_panel.set_gpkg_path(gpkg)
        self._results_panel.show()
        self._results_panel.raise_()
        self._refresh_velocity_vectors_overlay(self._results_panel.current_time_sec())

    def _on_results_panel_timestep_changed(self, t_s: float):
        self._refresh_velocity_vectors_overlay(float(t_s))

    def _on_results_panel_velocity_overlay_changed(self):
        panel = getattr(self, "_results_panel", None)
        t_s = panel.current_time_sec() if panel is not None else 0.0
        self._refresh_velocity_vectors_overlay(float(t_s))

    def _list_velocity_candidate_tables(self, gpkg_path: str) -> List[str]:
        gpkg_path = str(gpkg_path or "").strip()
        if not gpkg_path or not os.path.exists(gpkg_path):
            return []

        required_cols = {"run_id", "t_s", "cell_id", "h", "hu", "hv"}
        internal_prefixes = (
            "gpkg_",
            "rtree_",
            "sqlite_",
        )
        out: List[str] = []
        try:
            conn = sqlite3.connect(gpkg_path)
        except Exception:
            return out
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            table_rows = cur.fetchall()
            for row in table_rows:
                table_name = str(row[0] or "").strip()
                if not table_name:
                    continue
                lname = table_name.lower()
                if lname.startswith(internal_prefixes):
                    continue
                try:
                    cur.execute(f'PRAGMA table_info("{table_name}")')
                    cols = {str(r[1]).lower() for r in cur.fetchall()}
                except Exception:
                    continue
                if required_cols.issubset(cols):
                    out.append(table_name)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return out

    def _run_ids_for_velocity_table(self, gpkg_path: str, table_name: str) -> List[str]:
        gpkg_path = str(gpkg_path or "").strip()
        table_name = str(table_name or "").strip()
        if not gpkg_path or not table_name or not os.path.exists(gpkg_path):
            return []

        out: List[str] = []
        try:
            conn = sqlite3.connect(gpkg_path)
        except Exception:
            return out
        try:
            cur = conn.cursor()
            q_table = table_name.replace('"', '""')
            cur.execute(
                f'SELECT DISTINCT run_id FROM "{q_table}" WHERE run_id IS NOT NULL ORDER BY run_id'
            )
            out = [str(r[0]).strip() for r in cur.fetchall() if str(r[0]).strip()]
        except Exception:
            out = []
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return out

    def _pick_velocity_overlay_source(self) -> Tuple[str, str, str]:
        panel = getattr(self, "_results_panel", None)
        if panel is None:
            return "", "", ""

        start_path = self._velocity_overlay_manual_gpkg_path or self._gpkg_path or ""
        gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select SWE2D Velocity GeoPackage",
            start_path,
            "GeoPackage (*.gpkg)",
        )
        gpkg_path = str(gpkg_path or "").strip()
        if not gpkg_path:
            return "", "", ""

        table_choices = self._list_velocity_candidate_tables(gpkg_path)
        if not table_choices:
            QtWidgets.QMessageBox.warning(
                self,
                "Velocity Arrows",
                "No velocity-compatible tables were found in the selected GeoPackage.",
            )
            return "", "", ""

        default_table = table_choices[0]
        if self._velocity_overlay_manual_table_name in table_choices:
            default_table = self._velocity_overlay_manual_table_name

        table_name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Velocity Arrows",
            "Select table source:",
            table_choices,
            max(0, table_choices.index(default_table)),
            False,
        )
        if not ok:
            return "", "", ""
        table_name = str(table_name)

        run_ids: List[str] = []
        try:
            run_ids = self._run_ids_for_velocity_table(gpkg_path, table_name)
        except Exception:
            run_ids = []

        if not run_ids:
            active_run_id = str(panel.active_overlay_run_id() or "").strip()
            if active_run_id:
                run_ids = [active_run_id]

        if not run_ids:
            QtWidgets.QMessageBox.warning(
                self,
                "Velocity Arrows",
                "No run ids were found in the selected table for velocity rendering.",
            )
            return "", "", ""

        run_id = run_ids[0]
        if len(run_ids) > 1:
            default_run = run_id
            if self._velocity_overlay_manual_run_id in run_ids:
                default_run = self._velocity_overlay_manual_run_id
            chosen_run, run_ok = QtWidgets.QInputDialog.getItem(
                self,
                "Velocity Arrows",
                "Select run id:",
                run_ids,
                max(0, run_ids.index(default_run)),
                False,
            )
            if not run_ok:
                return "", "", ""
            run_id = str(chosen_run)

        return gpkg_path, run_id, table_name

    def _on_results_panel_velocity_overlay_add_requested(self):
        gpkg_path, run_id, table_name = self._pick_velocity_overlay_source()
        if not gpkg_path or not run_id:
            return

        self._velocity_overlay_manual_gpkg_path = gpkg_path
        self._velocity_overlay_manual_run_id = run_id
        self._velocity_overlay_manual_layer_name = table_name
        self._velocity_overlay_manual_table_name = table_name

        panel = getattr(self, "_results_panel", None)
        if panel is not None and hasattr(panel, "set_velocity_overlay_enabled"):
            try:
                panel.set_velocity_overlay_enabled(True)
            except Exception:
                pass

        t_s = panel.current_time_sec() if panel is not None else 0.0
        self._refresh_velocity_vectors_overlay(float(t_s))
        self._log(
            f"Velocity arrows source set: table='{table_name}', gpkg='{gpkg_path}', run_id='{run_id}'"
        )

    def _get_velocity_vector_builder(self):
        if self._velocity_vector_builder is not None:
            return self._velocity_vector_builder
        try:
            try:
                from .swe2d_velocity_layer import VelocityVectorBuilder
            except Exception:
                from swe2d_velocity_layer import VelocityVectorBuilder
            self._velocity_vector_builder = VelocityVectorBuilder(max_cache_entries=24)
        except Exception as exc:
            self._log(f"Velocity overlay unavailable: could not import builder ({exc})")
            self._velocity_vector_builder = None
        return self._velocity_vector_builder

    def _velocity_vectors_layer(self):
        if not _HAVE_QGIS_CORE or QgsProject is None or QgsVectorLayer is None:
            return None
        if self._velocity_vectors_layer_id:
            lyr = QgsProject.instance().mapLayer(self._velocity_vectors_layer_id)
            if lyr is not None and lyr.isValid():
                return lyr

        crs_auth = "EPSG:4326"
        try:
            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception:
            pass

        uri = (
            f"LineString?crs={crs_auth}"
            "&field=cell_id:integer"
            "&field=speed:double"
            "&field=u:double"
            "&field=v:double"
            "&field=angle_deg:double"
            "&field=color:string(16)"
            "&field=width:double"
        )
        lyr = QgsVectorLayer(uri, "SWE2D_Velocity_Vectors", "memory")
        if lyr is None or not lyr.isValid():
            return None
        QgsProject.instance().addMapLayer(lyr)
        self._velocity_vectors_layer_id = str(lyr.id())
        return lyr

    def _clear_velocity_vectors_layer(self):
        lyr = self._velocity_vectors_layer()
        if lyr is None:
            return
        try:
            dp = lyr.dataProvider()
            ids = [f.id() for f in lyr.getFeatures()]
            if ids:
                dp.deleteFeatures(ids)
            lyr.triggerRepaint()
        except Exception:
            pass

    def _refresh_velocity_vectors_overlay(self, t_s: float):
        panel = getattr(self, "_results_panel", None)
        if panel is None or not panel.velocity_overlay_enabled():
            self._clear_velocity_vectors_layer()
            return
        if self._mesh_data is None or not _HAVE_QGIS_CORE:
            self._clear_velocity_vectors_layer()
            return

        targets = []
        if self._velocity_overlay_manual_gpkg_path and self._velocity_overlay_manual_run_id:
            targets = [
                (
                    str(self._velocity_overlay_manual_gpkg_path),
                    str(self._velocity_overlay_manual_run_id),
                )
            ]
        else:
            if hasattr(panel, "enabled_overlay_targets"):
                try:
                    targets = list(panel.enabled_overlay_targets())
                except Exception:
                    targets = []
            if not targets:
                gpkg_path_fallback = str(self._model_gpkg_path or "")
                run_id_fallback = str(panel.active_overlay_run_id() or "")
                if gpkg_path_fallback and run_id_fallback:
                    targets = [(gpkg_path_fallback, run_id_fallback)]
        if not targets:
            self._clear_velocity_vectors_layer()
            return

        builder = self._get_velocity_vector_builder()
        if builder is None:
            self._clear_velocity_vectors_layer()
            return

        snap = None
        table_name = str(self._velocity_overlay_manual_table_name or "swe2d_mesh_results")
        for gpkg_path, run_id in targets:
            gpkg_path = str(gpkg_path or "")
            run_id = str(run_id or "")
            if not gpkg_path or not run_id or not os.path.exists(gpkg_path):
                continue
            snap = builder.load_snapshot(
                gpkg_path,
                run_id,
                float(t_s),
                t_tol=1.0,
                table_name=table_name,
            )
            if snap is not None:
                break
        if snap is None:
            self._clear_velocity_vectors_layer()
            return

        cx, cy = self._mesh_cell_centroids()
        n_cells = min(int(cx.size), int(cy.size))
        cell_xy = {i: (float(cx[i]), float(cy[i])) for i in range(n_cells)}
        stride = max(1, int(panel.velocity_density_stride()))
        min_speed = max(0.0, float(panel.velocity_min_speed()))
        vecs = builder.build_vectors(
            snapshot=snap,
            cell_xy=cell_xy,
            stride=stride,
            min_depth=1.0e-6,
            min_speed=min_speed,
        )

        lyr = self._velocity_vectors_layer()
        if lyr is None:
            return

        dp = lyr.dataProvider()
        old_ids = [f.id() for f in lyr.getFeatures()]
        if old_ids:
            dp.deleteFeatures(old_ids)

        if not vecs:
            lyr.triggerRepaint()
            return

        try:
            area = np.asarray(self._mesh_cell_areas(), dtype=np.float64)
            base_len = float(np.sqrt(max(float(np.nanmean(area)), 1.0e-9)))
        except Exception:
            base_len = 1.0
        base_len = max(base_len, 0.05)

        feats = []
        for v in vecs:
            speed = float(v.get("speed", 0.0))
            if speed <= 1.0e-12:
                continue
            style = builder.style_from_speed(speed)
            dir_u = float(v.get("u", 0.0)) / speed
            dir_v = float(v.get("v", 0.0)) / speed
            line_len = base_len * min(3.0, max(0.4, 0.7 + 0.8 * speed))

            x0 = float(v.get("x", 0.0))
            y0 = float(v.get("y", 0.0))
            x1 = x0 + dir_u * line_len
            y1 = y0 + dir_v * line_len

            feat = QgsFeature(lyr.fields())
            feat.setAttribute("cell_id", int(v.get("cell_id", -1)))
            feat.setAttribute("speed", speed)
            feat.setAttribute("u", float(v.get("u", 0.0)))
            feat.setAttribute("v", float(v.get("v", 0.0)))
            feat.setAttribute("angle_deg", float(v.get("angle_deg", 0.0)))
            feat.setAttribute("color", str(style.get("color", "#2c7bb6")))
            feat.setAttribute("width", float(style.get("width", 0.6)))
            feat.setGeometry(
                QgsGeometry.fromPolylineXY([
                    QgsPointXY(x0, y0),
                    QgsPointXY(x1, y1),
                ])
            )
            feats.append(feat)

        if feats:
            dp.addFeatures(feats)
            lyr.updateExtents()
        lyr.triggerRepaint()
        iface = getattr(self, "_iface", None)
        if iface is not None and hasattr(iface, "mapCanvas"):
            try:
                iface.mapCanvas().refresh()
            except Exception:
                pass

    # ------------------------------------------------------------------

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

    def _sample_coupling_object_metrics(
        self,
        coupling_controller,
        t_s: float,
        h: np.ndarray,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        if coupling_controller is None:
            return rows

        drainage_mod = getattr(coupling_controller, "drainage", None)
        if drainage_mod is not None:
            try:
                for node in drainage_mod.cfg.nodes:
                    node_id = str(node.node_id)
                    rows.append(
                        {
                            "t_s": float(t_s),
                            "component": "drainage_node",
                            "object_id": node_id,
                            "object_name": node_id,
                            "metric": "depth",
                            "value": float(drainage_mod.state.node_depth.get(node_id, 0.0)),
                        }
                    )
                for link in drainage_mod.cfg.links:
                    link_id = str(link.link_id)
                    from_node = str(link.from_node_id)
                    to_node = str(link.to_node_id)
                    rows.append(
                        {
                            "t_s": float(t_s),
                            "component": "drainage_link",
                            "object_id": link_id,
                            "object_name": f"{from_node}->{to_node}",
                            "metric": "flow",
                            "value": float(drainage_mod.state.link_flow.get(link_id, 0.0)),
                        }
                    )
            except Exception:
                pass

        structures_mod = getattr(coupling_controller, "structures", None)
        if structures_mod is not None:
            try:
                hh = np.ascontiguousarray(h, dtype=np.float64).ravel()
                cell_wse = hh + np.asarray(coupling_controller.cell_bed_m, dtype=np.float64).ravel()
                flows = list(structures_mod.structure_flows(cell_wse))
                for i, st in enumerate(structures_mod.cfg.structures):
                    sid = str(st.structure_id)
                    q = float(flows[i]) if i < len(flows) else 0.0
                    rows.append(
                        {
                            "t_s": float(t_s),
                            "component": "structure",
                            "object_id": sid,
                            "object_name": str(st.structure_type.name).lower(),
                            "metric": "flow",
                            "value": q,
                        }
                    )
            except Exception:
                pass

        return rows

    def _persist_coupling_results_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        rows: List[Dict[str, object]],
        interval_s: float,
    ) -> None:
        if not gpkg_path or not rows:
            return
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_coupling_results_runs (
                    run_id TEXT PRIMARY KEY,
                    created_utc TEXT,
                    interval_s REAL,
                    row_count INTEGER
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_coupling_results (
                    run_id TEXT,
                    t_s REAL,
                    component TEXT,
                    object_id TEXT,
                    object_name TEXT,
                    metric TEXT,
                    value REAL,
                    PRIMARY KEY (run_id, t_s, component, object_id, metric)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_swe2d_coupling_run_component_metric_obj_t "
                "ON swe2d_coupling_results(run_id, component, metric, object_id, t_s)"
            )
            cur.execute("DELETE FROM swe2d_coupling_results WHERE run_id = ?", (run_id,))
            cur.execute(
                """
                INSERT OR REPLACE INTO swe2d_coupling_results_runs
                (run_id, created_utc, interval_s, row_count)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                    float(interval_s),
                    int(len(rows)),
                ),
            )
            batch = [
                (
                    str(run_id),
                    float(r.get("t_s", 0.0)),
                    str(r.get("component", "") or ""),
                    str(r.get("object_id", "") or ""),
                    str(r.get("object_name", "") or ""),
                    str(r.get("metric", "") or ""),
                    float(r.get("value", float("nan"))),
                )
                for r in rows
            ]
            cur.executemany(
                """
                INSERT OR REPLACE INTO swe2d_coupling_results
                (run_id, t_s, component, object_id, object_name, metric, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
            self._coupling_results_latest_run_id = str(run_id)
            self._coupling_results_latest_db_path = str(gpkg_path)
            self._log(
                f"Stored coupling results in GeoPackage: {gpkg_path} "
                f"(run_id={run_id}, rows={len(rows)})"
            )
        finally:
            conn.close()

    def _load_coupling_results_from_geopackage(
        self,
        gpkg_path: str,
        run_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        if not gpkg_path or not os.path.exists(gpkg_path):
            return "", []
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_coupling_results'"
            )
            if cur.fetchone() is None:
                return "", []

            chosen = str(run_id or "").strip()
            if not chosen:
                cur.execute(
                    """
                    SELECT run_id FROM swe2d_coupling_results_runs
                    ORDER BY datetime(created_utc) DESC, rowid DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row is None:
                    return "", []
                chosen = str(row[0])

            cur.execute(
                """
                SELECT t_s, component, object_id, object_name, metric, value
                FROM swe2d_coupling_results
                WHERE run_id = ?
                ORDER BY t_s ASC, component ASC, metric ASC, object_id ASC
                """,
                (chosen,),
            )
            rows: List[Dict[str, object]] = []
            for t_s, component, object_id, object_name, metric, value in cur.fetchall():
                rows.append(
                    {
                        "t_s": float(t_s),
                        "component": str(component or ""),
                        "object_id": str(object_id or ""),
                        "object_name": str(object_name or ""),
                        "metric": str(metric or ""),
                        "value": float(value),
                    }
                )
            return chosen, rows
        finally:
            conn.close()

    def _open_coupling_results_viewer(self):
        db_path = ""
        if self._coupling_results_latest_db_path and os.path.exists(self._coupling_results_latest_db_path):
            db_path = self._coupling_results_latest_db_path
        if not db_path:
            db_path = self._current_line_results_storage_path()
        if not db_path:
            self._log("No GeoPackage available for coupling results viewer.")
            return

        run_id = self._coupling_results_latest_run_id or None
        chosen, rows = self._load_coupling_results_from_geopackage(db_path, run_id=run_id)
        if not chosen or not rows:
            self._log("No drainage/structure coupling results found in GeoPackage yet.")
            return

        dlg = SWE2DCouplingResultsViewerDialog(
            records=rows,
            run_id=chosen,
            db_path=db_path,
            length_unit=self._length_unit_name,
            flow_unit_label=self._flow_unit_label(),
            parent=self,
        )
        dlg.exec()

    def _persist_mesh_results_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        mesh_rows: List[Dict[str, object]],
        interval_s: float,
    ) -> None:
        if not gpkg_path or not mesh_rows:
            return
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_mesh_results_runs (
                    run_id TEXT PRIMARY KEY,
                    created_utc TEXT,
                    interval_s REAL,
                    row_count INTEGER
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_mesh_results (
                    run_id TEXT,
                    t_s REAL,
                    cell_id INTEGER,
                    h REAL,
                    hu REAL,
                    hv REAL,
                    PRIMARY KEY (run_id, t_s, cell_id)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_swe2d_mesh_results_run_t_cell "
                "ON swe2d_mesh_results(run_id, t_s, cell_id)"
            )
            cur.execute("DELETE FROM swe2d_mesh_results WHERE run_id = ?", (run_id,))
            cur.execute(
                """
                INSERT OR REPLACE INTO swe2d_mesh_results_runs
                (run_id, created_utc, interval_s, row_count)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                    float(interval_s),
                    int(len(mesh_rows)),
                ),
            )
            cur.executemany(
                """
                INSERT OR REPLACE INTO swe2d_mesh_results
                (run_id, t_s, cell_id, h, hu, hv)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(run_id),
                        float(r.get("t_s", 0.0)),
                        int(r.get("cell_id", -1)),
                        float(r.get("h", 0.0)),
                        float(r.get("hu", 0.0)),
                        float(r.get("hv", 0.0)),
                    )
                    for r in mesh_rows
                ],
            )
            conn.commit()
            self._log(
                f"Stored mesh snapshot results in GeoPackage: {gpkg_path} "
                f"(run_id={run_id}, rows={len(mesh_rows)})"
            )
        finally:
            conn.close()

    def _build_mesh_snapshot_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        if not getattr(self, "_snapshot_timesteps", None):
            return rows
        for snap in self._snapshot_timesteps:
            try:
                t_s, h, hu, hv = snap
                hh = np.asarray(h, dtype=np.float64).ravel()
                huu = np.asarray(hu, dtype=np.float64).ravel()
                hvv = np.asarray(hv, dtype=np.float64).ravel()
                n = min(hh.size, huu.size, hvv.size)
                ts_val = float(t_s)
                for ci in range(n):
                    rows.append(
                        {
                            "t_s": ts_val,
                            "cell_id": int(ci),
                            "h": float(hh[ci]),
                            "hu": float(huu[ci]),
                            "hv": float(hvv[ci]),
                        }
                    )
            except Exception:
                continue
        return rows

    def _persist_run_log_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        start_wallclock: str,
        end_wallclock: str,
        duration_s: float,
        log_text: str,
    ) -> None:
        if not gpkg_path or not run_id:
            return
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_run_logs (
                    run_id TEXT PRIMARY KEY,
                    created_utc TEXT,
                    start_wallclock TEXT,
                    end_wallclock TEXT,
                    duration_s REAL,
                    log_text TEXT
                )
                """
            )
            cur.execute(
                """
                INSERT OR REPLACE INTO swe2d_run_logs
                (run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                    str(start_wallclock or ""),
                    str(end_wallclock or ""),
                    float(duration_s),
                    str(log_text or ""),
                ),
            )
            conn.commit()
            self._run_log_latest_run_id = str(run_id)
            self._run_log_latest_db_path = str(gpkg_path)
            self._log(f"Stored run log in GeoPackage: {gpkg_path} (run_id={run_id})")
        finally:
            conn.close()

    def _load_run_logs_from_geopackage(
        self,
        gpkg_path: str,
    ) -> List[Dict[str, object]]:
        if not gpkg_path or not os.path.exists(gpkg_path):
            return []
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_run_logs'")
            if cur.fetchone() is None:
                return []
            cur.execute(
                """
                SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text
                FROM swe2d_run_logs
                ORDER BY datetime(created_utc) DESC, rowid DESC
                """
            )
            rows: List[Dict[str, object]] = []
            for run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text in cur.fetchall():
                rows.append(
                    {
                        "run_id": str(run_id or ""),
                        "created_utc": str(created_utc or ""),
                        "start_wallclock": str(start_wallclock or ""),
                        "end_wallclock": str(end_wallclock or ""),
                        "duration_s": float(duration_s or 0.0),
                        "log_text": str(log_text or ""),
                    }
                )
            return rows
        finally:
            conn.close()

    def _open_run_log_viewer(self):
        db_path = ""
        if self._run_log_latest_db_path and os.path.exists(self._run_log_latest_db_path):
            db_path = self._run_log_latest_db_path
        if not db_path:
            db_path = self._current_line_results_storage_path()
        if not db_path:
            self._log("No GeoPackage available for run log viewer.")
            return
        records = self._load_run_logs_from_geopackage(db_path)
        if not records:
            self._log("No saved run logs found in GeoPackage yet.")
            return
        dlg = SWE2DRunLogViewerDialog(
            records=records,
            run_id=self._run_log_latest_run_id,
            db_path=db_path,
            parent=self,
        )
        dlg.exec()

    def _build_internal_flow_source_cms(self) -> Optional[np.ndarray]:
        forcing = self._build_internal_flow_forcing()
        if forcing is None:
            return None
        return self._internal_flow_source_cms_at_time(forcing, 0.0)

    def _build_internal_flow_forcing(self) -> Optional[Dict[str, object]]:
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
                hlyr
                for hlyr in self._iter_project_layers()
                if isinstance(hlyr, QgsVectorLayer) and str(hlyr.name()).lower() in ("swe2d_hydrographs",)
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

        cx, cy = self._mesh_cell_centroids()
        base_q = np.zeros(cx.shape[0], dtype=np.float64)
        dynamic_terms: List[Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]] = []
        assigned = 0
        dynamic_assigned = 0

        for ft in lyr.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue

            q_cms = 0.0
            try:
                q_cms = float(ft[field_name])
            except Exception:
                q_cms = 0.0
            if not np.isfinite(q_cms):
                q_cms = 0.0

            hg = None
            raw_h = str(ft[hydro_field] or "").strip() if hydro_field is not None else ""
            ref_layer = str(ft[hlyr_field] or "").strip() if hlyr_field is not None else ""
            hid = str(ft[hgid_field] or "").strip() if hgid_field is not None else ""
            if not raw_h and hid and hid in hydro_lookup:
                raw_h = hydro_lookup[hid]

            if raw_h:
                try:
                    hg = self._parse_hydrograph_text(raw_h)
                except Exception:
                    hg = None

            if hg is None and (ref_layer or (hgid_field is not None and hid)):
                layer_ref = ref_layer or (str(ft[hydro_field] or "").strip() if hydro_field is not None else "")
                target_layer = None
                for hlyr in self._iter_project_layers():
                    if not isinstance(hlyr, QgsVectorLayer):
                        continue
                    if str(hlyr.name()) == layer_ref or str(hlyr.id()) == layer_ref:
                        target_layer = hlyr
                        break
                if target_layer is not None:
                    hg = self._hydrograph_from_layer(target_layer, hydrograph_id=hid, bc_type=None)

            if abs(q_cms) <= 0.0 and hg is None:
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
                idx_arr = np.asarray(hit_ids, dtype=np.int32)
                wt_arr = np.full(idx_arr.shape[0], 1.0 / float(idx_arr.shape[0]), dtype=np.float64)
            else:
                rp = geom.centroid().asPoint() if not geom.centroid().isEmpty() else None
                if rp is None:
                    continue
                dx = cx - float(rp.x())
                dy = cy - float(rp.y())
                idx = int(np.argmin(dx * dx + dy * dy))
                idx_arr = np.asarray([idx], dtype=np.int32)
                wt_arr = np.asarray([1.0], dtype=np.float64)

            if abs(q_cms) > 0.0:
                base_q[idx_arr] += q_cms * wt_arr
            if hg is not None:
                dynamic_terms.append((idx_arr, wt_arr, hg))
                dynamic_assigned += 1
            assigned += 1

        if assigned <= 0:
            return None

        self._log(
            f"Internal flow sources mapped from layer '{lyr.name()}': features={assigned}, "
            f"timeseries_features={dynamic_assigned}, static_total_Q={float(np.sum(base_q)):.6f} cms"
        )
        return {
            "base_q_cms": base_q,
            "dynamic_terms": dynamic_terms,
            "layer_name": str(lyr.name()),
        }

    def _internal_flow_source_cms_at_time(self, forcing: Optional[Dict[str, object]], t_sec: float) -> Optional[np.ndarray]:
        if forcing is None:
            return None
        base_q = forcing.get("base_q_cms")
        if base_q is None:
            return None

        cell_q = np.asarray(base_q, dtype=np.float64).copy()
        dynamic_terms = forcing.get("dynamic_terms", [])
        for idx_arr, wt_arr, hg in dynamic_terms:
            q_total = self._interp_hydrograph(hg, t_sec)
            cell_q[np.asarray(idx_arr, dtype=np.int32)] += q_total * np.asarray(wt_arr, dtype=np.float64)
        return cell_q

    def _apply_external_sources(
        self,
        backend: SWE2DBackend,
        dt_step: float,
        rain_rate_model,
        cell_source_model: Optional[np.ndarray],
        coupled_source_rate: Optional[np.ndarray] = None,
        prefer_native_injection: bool = False,
    ) -> None:
        if dt_step <= 0.0:
            if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
                try:
                    backend.set_external_sources_native(None)
                except Exception:
                    pass
            return

        no_external_sources = (
            np.all(np.asarray(rain_rate_model, dtype=np.float64) <= 0.0)
            and cell_source_model is None
            and coupled_source_rate is None
        )
        if no_external_sources:
            if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
                try:
                    backend.set_external_sources_native(None)
                except Exception:
                    pass
            return

        n_cells_raw = getattr(backend, "n_cells", 0)
        n_cells = int(n_cells_raw() if callable(n_cells_raw) else n_cells_raw)
        rain_arr = np.asarray(rain_rate_model, dtype=np.float64)
        if rain_arr.ndim == 0:
            src = np.full((n_cells,), float(rain_arr), dtype=np.float64)
        else:
            src = np.zeros((n_cells,), dtype=np.float64)
            src[: min(src.shape[0], rain_arr.shape[0])] = rain_arr[: min(src.shape[0], rain_arr.shape[0])]
        if cell_source_model is not None:
            area = self._mesh_cell_areas()
            safe_area = np.maximum(area, 1.0e-8)
            src += (cell_source_model / safe_area)
        if coupled_source_rate is not None:
            csr = np.asarray(coupled_source_rate, dtype=np.float64)
            src[: min(src.shape[0], csr.shape[0])] += csr[: min(src.shape[0], csr.shape[0])]

        src = np.where(np.isfinite(src), src, 0.0)

        # Optional positive source-rate cap for rain/CN/internal forcing spikes.
        src_cap_widget = getattr(self, "max_source_rate_spin", None)
        src_cap = float(src_cap_widget.value()) if src_cap_widget is not None else 0.0
        if src_cap > 0.0:
            src = np.where(src > src_cap, src_cap, src)
        if prefer_native_injection and hasattr(backend, "set_external_sources_native"):
            try:
                backend.set_external_sources_native(src)
                return
            except Exception:
                # Fallback to host-side source application if native injection fails.
                pass

        h, hu, hv = backend.get_state()
        h_prev = np.asarray(h, dtype=np.float64)

        dh = dt_step * src

        # Host-side source update limiter for robust dry-to-wet transitions.
        # Native solver applies these controls internally; keep the host-state
        # readback only on the fallback path so GPU runs can remain device-resident.
        max_rel_widget = getattr(self, "max_rel_depth_increase_spin", None)
        hmin_widget = getattr(self, "h_min_spin", None)
        h_min = float(hmin_widget.value()) if hmin_widget is not None else 1.0e-4
        max_rel = float(max_rel_widget.value()) if max_rel_widget is not None else 0.0
        if max_rel > 0.0:
            dh_pos_cap = np.maximum(h_prev, h_min) * max_rel
            dh = np.where(dh > dh_pos_cap, dh_pos_cap, dh)

        # Optional absolute cap on positive source-driven depth change per step.
        dh_cap_widget = getattr(self, "max_source_depth_step_spin", None)
        dh_cap = float(dh_cap_widget.value()) if dh_cap_widget is not None else 0.0
        if dh_cap > 0.0:
            dh = np.where(dh > dh_cap, dh_cap, dh)

        h = h_prev + dh
        h = np.where(np.isfinite(h), h, 0.0)
        h = np.maximum(h, 0.0)

        dry = h < h_min
        hu = np.where(dry, 0.0, hu)
        hv = np.where(dry, 0.0, hv)

        # Cells that just transitioned from dry to wet due to source terms should
        # not inherit stale momentum from prior dry-state numerical noise.
        newly_wet = (h_prev < h_min) & (~dry)
        hu = np.where(newly_wet, 0.0, hu)
        hv = np.where(newly_wet, 0.0, hv)

        # Apply shallow-water momentum damping on the fallback source path.
        shallow_widget = getattr(self, "shallow_damping_depth_spin", None)
        shallow_damp_h = float(shallow_widget.value()) if shallow_widget is not None else 0.0
        if shallow_damp_h > h_min:
            damp = np.clip(h / shallow_damp_h, 0.0, 1.0)
            hu = hu * damp
            hv = hv * damp
        
        # Momentum cap after rain/source application to prevent velocity blow-up.
        # When rain adds depth to cells with existing momentum, the effective velocity
        # can spike. Apply a cap based on depth-scaled wave celerity to prevent CFL explosion.
        gravity = 9.81
        min_speed_cap_widget = getattr(self, "momentum_cap_min_speed_spin", None)
        celerity_mult_widget = getattr(self, "momentum_cap_celerity_mult_spin", None)
        min_speed_cap = float(min_speed_cap_widget.value()) if min_speed_cap_widget is not None else 50.0
        celerity_mult = float(celerity_mult_widget.value()) if celerity_mult_widget is not None else 20.0
        
        abs_u = np.abs(hu) / np.maximum(h, 1.0e-12)
        abs_v = np.abs(hv) / np.maximum(h, 1.0e-12)
        abs_speed = np.sqrt(abs_u**2 + abs_v**2)
        wave_speed = np.sqrt(gravity * np.maximum(h, 1.0e-12))
        speed_cap = np.maximum(min_speed_cap, celerity_mult * wave_speed)
        clipped = abs_speed > speed_cap
        if np.any(clipped):
            scale = np.where(clipped, speed_cap / np.maximum(abs_speed, 1.0e-12), 1.0)
            hu = hu * scale
            hv = hv * scale
        
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

        hy_field_names = list(hyetograph_layer.fields().names())
        hy_fields = set(hy_field_names)
        hy_id_field = "hyetograph_id" if "hyetograph_id" in hy_fields else None

        time_field = None
        for cand in ("Time", "time", "Time_hr", "time_hr", "Time_min", "time_min", "minutes", "Minutes", "t_min"):
            if cand in hy_fields:
                time_field = cand
                break

        value_field = None
        for cand in (
            "Value",
            "value",
            "Rain",
            "rain",
            "rainfall",
            "Rainfall",
            "Incremental_Rainfall_in",
            "incremental_rainfall_in",
            "rain_in",
            "rain_mm",
        ):
            if cand in hy_fields:
                value_field = cand
                break

        if value_field is None:
            for name in hy_field_names:
                ln = str(name).lower()
                if "value" in ln or "rain" in ln or "hyeto" in ln:
                    value_field = name
                    break
        if hy_id_field is None or time_field is None or value_field is None:
            self._log("Hyetograph table missing hyetograph_id/Time/Value fields; skipping Thiessen rain forcing.")
            return None

        hy_rows_by_id: Dict[str, List[Dict[str, object]]] = {}
        time_field_l = str(time_field or "").lower()
        value_field_l = str(value_field or "").lower()

        inferred_value_type = "intensity"
        if "increment" in value_field_l or "depth" in value_field_l:
            inferred_value_type = "incremental_depth"
        elif "cum" in value_field_l:
            inferred_value_type = "cumulative_depth"

        inferred_units = "mm/hr"
        if "in/hr" in value_field_l:
            inferred_units = "in/hr"
        elif "mm/hr" in value_field_l:
            inferred_units = "mm/hr"
        elif "_in" in value_field_l or "inch" in value_field_l:
            inferred_units = "in"
        elif "_mm" in value_field_l:
            inferred_units = "mm"

        for ft in hyetograph_layer.getFeatures():
            try:
                hy_id = str(ft[hy_id_field] or "").strip()
            except Exception:
                hy_id = ""
            if not hy_id:
                continue

            time_value = ft[time_field]
            if "min" in time_field_l and isinstance(time_value, (int, float)):
                time_value = f"{float(time_value)} min"
            elif "hr" in time_field_l and isinstance(time_value, (int, float)):
                time_value = f"{float(time_value)} hr"

            row = {
                "Time": time_value,
                "Value": ft[value_field],
                "value_type": ft["value_type"] if "value_type" in hy_fields else inferred_value_type,
                "units": ft["units"] if "units" in hy_fields else inferred_units,
            }
            hy_rows_by_id.setdefault(hy_id, []).append(row)

        if inspect_hyetograph_rows is not None:
            for hy_id in sorted(hy_rows_by_id.keys()):
                rows = hy_rows_by_id.get(hy_id, [])
                diag = inspect_hyetograph_rows(rows)
                self._log(
                    "Hyetograph parse: "
                    f"id='{hy_id}', rows={int(diag.get('n_rows', 0))}, valid={int(diag.get('n_valid', 0))}, "
                    f"mode={diag.get('mode', 'unknown')}, units={diag.get('units', 'unknown')}, "
                    f"t=[{float(diag.get('t_start_s', 0.0)):.1f},{float(diag.get('t_end_s', 0.0)):.1f}] s, "
                    f"dt_med={float(diag.get('dt_median_s', 0.0)):.1f} s, "
                    f"total_depth={float(diag.get('total_depth_mm', 0.0)):.3f} mm"
                )
                for w in list(diag.get("warnings", [])):
                    self._log(f"Hyetograph parse warning (id='{hy_id}'): {w}")

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
        cell_to_gauge = np.asarray(cell_to_gauge, dtype=np.int32).copy()

        storm_area_layer = self._combo_layer(self.storm_area_layer_combo, "vector") if hasattr(self, "storm_area_layer_combo") else None
        if storm_area_layer is not None:
            in_storm = np.zeros(cell_to_gauge.shape[0], dtype=bool)
            for ft in storm_area_layer.getFeatures():
                geom = ft.geometry()
                if geom is None or geom.isEmpty():
                    continue
                try:
                    wkb_type = int(geom.wkbType())
                except Exception:
                    wkb_type = -1
                if QgsWkbTypes.geometryType(wkb_type) == QgsWkbTypes.GeometryType.PolygonGeometry:
                    for i in range(cell_x.shape[0]):
                        if in_storm[i]:
                            continue
                        p = QgsGeometry.fromPointXY(QgsPointXY(float(cell_x[i]), float(cell_y[i])))
                        if geom.contains(p) or geom.intersects(p):
                            in_storm[i] = True
                else:
                    rp = geom.centroid().asPoint() if not geom.centroid().isEmpty() else None
                    if rp is None:
                        continue
                    dx = cell_x - float(rp.x())
                    dy = cell_y - float(rp.y())
                    in_storm[int(np.argmin(dx * dx + dy * dy))] = True

            if np.any(in_storm):
                excluded_count = int(np.count_nonzero(~in_storm))
                cell_to_gauge[~in_storm] = -1
                self._log(
                    f"Thiessen storm-area mask active: included {int(np.count_nonzero(in_storm))} cell(s), "
                    f"excluded {excluded_count} outside '{storm_area_layer.name()}'."
                )
            else:
                cell_to_gauge[:] = -1
                self._log(
                    f"Thiessen storm-area mask active: no cell centroids intersected '{storm_area_layer.name()}'; "
                    "rainfall forcing disabled by mask."
                )

        # Exclude boundary-adjacent rings from rainfall source to reduce
        # compounding source/BC forcing at open or prescribed boundaries.
        boundary_exclusion_rings = int(self.rain_boundary_buffer_rings_spin.value()) if hasattr(self, "rain_boundary_buffer_rings_spin") else 1
        excluded = self._boundary_buffer_cells(boundary_exclusion_rings)
        if excluded.size > 0:
            cell_to_gauge[excluded] = -1
            self._log(
                f"Thiessen rain boundary buffer active: excluded {excluded.size} cell(s) "
                f"across {boundary_exclusion_rings} boundary ring(s)."
            )

        cnvals = self._build_spatial_cn_array()
        ia_ratio = float(self.ia_ratio_spin.value()) if hasattr(self, "ia_ratio_spin") else 0.2
        infiltration_method = "scs_cn"
        if hasattr(self, "infiltration_method_combo"):
            infiltration_method = self.infiltration_method_combo.currentData() or "scs_cn"
        forcing = ThiessenRainCNForcing(
            cell_to_gauge=cell_to_gauge,
            gauge_hyetographs=hy_by_gauge_index,
            curve_number=cnvals,
            ia_ratio=ia_ratio,
            infiltration_method=infiltration_method,
        )
        self._log(
            f"Thiessen rain forcing active: gauges={len(gauges)}, "
            f"cells={cell_to_gauge.shape[0]}, infiltration={infiltration_method}, Ia/S={ia_ratio:.3f}, "
            f"cn_range=[{float(np.min(cnvals)):.1f}, {float(np.max(cnvals)):.1f}]"
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
        node_inlet_layer = self._combo_layer(self.drain_node_inlets_layer_combo, "vector") if hasattr(self, "drain_node_inlets_layer_combo") else None
        if node_layer is None or link_layer is None:
            return None

        node_fields = set(node_layer.fields().names())
        nodes: List[DrainageNode] = []
        node_by_id: Dict[str, DrainageNode] = {}
        node_cell_by_id: Dict[str, int] = {}
        node_zero_storage_by_id: Dict[str, bool] = {}
        cell_min_bed = self._mesh_cell_min_bed()

        def _opt_float(value, fallback=None):
            if value in (None, ""):
                return fallback
            try:
                return float(value)
            except Exception:
                return fallback

        def _opt_bool(value, fallback=False):
            if value in (None, ""):
                return fallback
            if value is True:
                return True
            if value is False:
                return False
            sval = str(value).strip().lower()
            if sval in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if sval in {"0", "false", "f", "no", "n", "off"}:
                return False
            try:
                return float(value) != 0.0
            except Exception:
                return fallback

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
            x = float(pt.x())
            y = float(pt.y())
            invert = _opt_float(ft["invert_elev"] if "invert_elev" in node_fields else None, 0.0)
            node_type = str(ft["node_type"] if "node_type" in node_fields else "junction").strip().lower() or "junction"
            ci = self._nearest_cell_index_for_xy(x, y)
            bed_here = float(cell_min_bed[ci]) if ci >= 0 and ci < int(cell_min_bed.size) else invert
            rim = _opt_float(ft["rim_elev"] if "rim_elev" in node_fields else None, None)
            if rim is None:
                rim = max(invert, bed_here)
            max_depth = _opt_float(ft["max_depth"] if "max_depth" in node_fields else None, None)
            if max_depth is None:
                if node_type == "outfall":
                    max_depth = 10.0
                else:
                    max_depth = max(0.1, float(rim) - float(invert))
            crest = _opt_float(ft["crest_elev"] if "crest_elev" in node_fields else None, None)
            if crest is None:
                crest = float(invert if node_type == "outfall" else rim)

            node = DrainageNode(
                node_id=node_id,
                x=x,
                y=y,
                invert_elev=float(invert),
                max_depth=float(max_depth),
                crest_elev=float(crest),
                rim_elev=float(rim),
                node_type=node_type,
                metadata={
                    "surface_area": float(ft["surface_area"] if "surface_area" in node_fields and ft["surface_area"] not in (None, "") else 50.0),
                    "outfall_area_m2": float(ft["outfall_area"] if "outfall_area" in node_fields and ft["outfall_area"] not in (None, "") else 0.0),
                },
            )
            nodes.append(node)
            node_by_id[node_id] = node
            node_cell_by_id[node_id] = int(ci)
            node_zero_storage_by_id[node_id] = _opt_bool(ft["zero_storage"] if "zero_storage" in node_fields else None, False)
        if not nodes:
            return None

        link_fields = set(link_layer.fields().names())
        links: List[DrainageLink] = []
        links_missing_capacity: List[str] = []

        def _ellipse_perimeter(a: float, b: float) -> float:
            # Ramanujan approximation; stable and cheap for geometry-derived hydraulics.
            if a <= 0.0 or b <= 0.0:
                return 0.0
            return math.pi * (3.0 * (a + b) - math.sqrt(max(0.0, (3.0 * a + b) * (a + 3.0 * b))))

        for ft in link_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            link_id = str(ft["link_id"] if "link_id" in link_fields else ft.id()).strip()
            from_node = str(ft["from_node"] if "from_node" in link_fields else "").strip()
            to_node = str(ft["to_node"] if "to_node" in link_fields else "").strip()
            if not link_id or not from_node or not to_node:
                continue

            link_shape = str(ft["link_shape"] if "link_shape" in link_fields else "").strip().lower()
            if link_shape in ("", "none", "null"):
                link_shape = "circular"

            diameter_val = None
            for nm in ("diameter", "diameter_m", "equiv_diameter", "equiv_diameter_m"):
                if nm in link_fields and ft[nm] not in (None, ""):
                    try:
                        d_try = float(ft[nm])
                        if d_try > 0.0:
                            diameter_val = d_try
                            break
                    except Exception:
                        pass

            area_val = None
            for nm in ("area_m2", "area", "cross_area"):
                if nm in link_fields and ft[nm] not in (None, ""):
                    try:
                        a_try = float(ft[nm])
                        if a_try > 0.0:
                            area_val = a_try
                            break
                    except Exception:
                        pass

            span_val = None
            for nm in ("span", "span_m", "width", "width_m"):
                if nm in link_fields and ft[nm] not in (None, ""):
                    try:
                        s_try = float(ft[nm])
                        if s_try > 0.0:
                            span_val = s_try
                            break
                    except Exception:
                        pass

            rise_val = None
            for nm in ("rise", "rise_m", "height", "height_m"):
                if nm in link_fields and ft[nm] not in (None, ""):
                    try:
                        r_try = float(ft[nm])
                        if r_try > 0.0:
                            rise_val = r_try
                            break
                    except Exception:
                        pass

            equiv_d_val = None
            for nm in ("equiv_diameter_m", "equiv_diameter"):
                if nm in link_fields and ft[nm] not in (None, ""):
                    try:
                        eq_try = float(ft[nm])
                        if eq_try > 0.0:
                            equiv_d_val = eq_try
                            break
                    except Exception:
                        pass

            if (area_val is None or area_val <= 0.0):
                if link_shape == "circular" and diameter_val is not None and diameter_val > 0.0:
                    area_val = 0.25 * math.pi * float(diameter_val) * float(diameter_val)
                elif link_shape in ("box", "rectangular", "rect") and span_val is not None and rise_val is not None:
                    area_val = float(span_val) * float(rise_val)
                elif link_shape == "pipe_arch" and span_val is not None and rise_val is not None:
                    area_val = 0.25 * math.pi * float(span_val) * float(rise_val)

            if (equiv_d_val is None or equiv_d_val <= 0.0):
                if diameter_val is not None and diameter_val > 0.0:
                    equiv_d_val = float(diameter_val)
                elif area_val is not None and area_val > 0.0:
                    if link_shape in ("box", "rectangular", "rect") and span_val is not None and rise_val is not None:
                        perim = 2.0 * (float(span_val) + float(rise_val))
                        if perim > 0.0:
                            equiv_d_val = 4.0 * float(area_val) / perim
                    elif link_shape == "pipe_arch" and span_val is not None and rise_val is not None:
                        perim = _ellipse_perimeter(0.5 * float(span_val), 0.5 * float(rise_val))
                        if perim > 0.0:
                            equiv_d_val = 4.0 * float(area_val) / perim
                    if equiv_d_val is None or equiv_d_val <= 0.0:
                        equiv_d_val = math.sqrt(4.0 * float(area_val) / math.pi)

            if (diameter_val is None or diameter_val <= 0.0) and equiv_d_val is not None and equiv_d_val > 0.0:
                diameter_val = float(equiv_d_val)

            if (diameter_val is None or diameter_val <= 0.0) and (area_val is None or area_val <= 0.0) and (equiv_d_val is None or equiv_d_val <= 0.0):
                links_missing_capacity.append(link_id)

            links.append(
                DrainageLink(
                    link_id=link_id,
                    from_node_id=from_node,
                    to_node_id=to_node,
                    link_type=str(ft["link_type"] if "link_type" in link_fields else "conduit").strip() or "conduit",
                    length=float(ft["length"]) if "length" in link_fields and ft["length"] not in (None, "") else float(geom.length()),
                    roughness_n=float(ft["roughness_n"] if "roughness_n" in link_fields and ft["roughness_n"] not in (None, "") else 0.013),
                    diameter=diameter_val,
                    max_flow=float(ft["max_flow"]) if "max_flow" in link_fields and ft["max_flow"] not in (None, "") else None,
                    metadata={
                        "area_m2": float(area_val) if area_val is not None else 0.0,
                        "equiv_diameter_m": float(equiv_d_val) if equiv_d_val is not None else 0.0,
                        "cd": float(ft["cd"] if "cd" in link_fields and ft["cd"] not in (None, "") else 0.75),
                        "entry_loss_k": float(ft["entry_loss_k"] if "entry_loss_k" in link_fields and ft["entry_loss_k"] not in (None, "") else 0.5),
                        "exit_loss_k": float(ft["exit_loss_k"] if "exit_loss_k" in link_fields and ft["exit_loss_k"] not in (None, "") else 1.0),
                        "pipe_end_inlet_loss_k": float(
                            ft["pipe_end_inlet_loss_k"]
                            if "pipe_end_inlet_loss_k" in link_fields and ft["pipe_end_inlet_loss_k"] not in (None, "")
                            else (
                                ft["inlet_loss_k"]
                                if "inlet_loss_k" in link_fields and ft["inlet_loss_k"] not in (None, "")
                                else 0.5
                            )
                        ),
                        "pipe_end_outlet_loss_k": float(
                            ft["pipe_end_outlet_loss_k"]
                            if "pipe_end_outlet_loss_k" in link_fields and ft["pipe_end_outlet_loss_k"] not in (None, "")
                            else (
                                ft["outlet_loss_k"]
                                if "outlet_loss_k" in link_fields and ft["outlet_loss_k"] not in (None, "")
                                else 1.0
                            )
                        ),
                        "link_shape": link_shape,
                        "span_m": float(span_val) if span_val is not None else 0.0,
                        "rise_m": float(rise_val) if rise_val is not None else 0.0,
                    },
                )
            )
        if not links:
            return None

        if links_missing_capacity:
            preview = ", ".join(links_missing_capacity[:8])
            suffix = "" if len(links_missing_capacity) <= 8 else f", ... (+{len(links_missing_capacity) - 8} more)"
            self._log(
                "Drainage warning: link(s) missing hydraulic geometry (diameter/area/equiv_diameter/shape dimensions); "
                "link flow will stay zero for these IDs: "
                f"{preview}{suffix}"
            )

        inlets: List[InletExchange] = []
        inlet_types: List[InletType] = []
        node_inlets: List[NodeInletAssignment] = []
        inlet_types_by_id: Dict[str, InletType] = {}

        # New schema: tabular inlet-type catalog + node assignment table.
        if inlet_layer is not None:
            inlet_fields = set(inlet_layer.fields().names())
            has_new_inlet_schema = "inlet_type_id" in inlet_fields
            if has_new_inlet_schema:
                for ft in inlet_layer.getFeatures():
                    inlet_type_id = str(ft["inlet_type_id"] if "inlet_type_id" in inlet_fields else "").strip()
                    if not inlet_type_id:
                        continue
                    inlet_type = InletType(
                        inlet_type_id=inlet_type_id,
                        name=str(ft["name"] if "name" in inlet_fields and ft["name"] not in (None, "") else inlet_type_id),
                        length=float(ft["weir_length"] if "weir_length" in inlet_fields and ft["weir_length"] not in (None, "") else 1.0),
                        area=float(ft["orifice_area"] if "orifice_area" in inlet_fields and ft["orifice_area"] not in (None, "") else 0.0),
                        coeff_weir=float(ft["coeff_weir"] if "coeff_weir" in inlet_fields and ft["coeff_weir"] not in (None, "") else 1.70),
                        coeff_orifice=float(ft["coeff_orifice"] if "coeff_orifice" in inlet_fields and ft["coeff_orifice"] not in (None, "") else 0.62),
                        max_capture=float(ft["max_capture"]) if "max_capture" in inlet_fields and ft["max_capture"] not in (None, "") else None,
                    )
                    inlet_types.append(inlet_type)
                    inlet_types_by_id[inlet_type_id] = inlet_type

                if node_inlet_layer is not None:
                    assign_fields = set(node_inlet_layer.fields().names())
                    for ft in node_inlet_layer.getFeatures():
                        node_id = str(ft["node_id"] if "node_id" in assign_fields else "").strip()
                        inlet_type_id = str(ft["inlet_type_id"] if "inlet_type_id" in assign_fields else "").strip()
                        if not node_id or not inlet_type_id:
                            continue
                        node_inlets.append(
                            NodeInletAssignment(
                                node_id=node_id,
                                inlet_type_id=inlet_type_id,
                                multiplier=float(ft["inlet_count"] if "inlet_count" in assign_fields and ft["inlet_count"] not in (None, "") else 1.0),
                                crest_offset=float(ft["crest_offset"] if "crest_offset" in assign_fields and ft["crest_offset"] not in (None, "") else 0.0),
                            )
                        )

                for a in node_inlets:
                    if a.node_id not in node_by_id:
                        continue
                    it = inlet_types_by_id.get(a.inlet_type_id)
                    if it is None:
                        continue
                    node = node_by_id[a.node_id]
                    crest = float((node.crest_elev if node.crest_elev is not None else node.invert_elev) + a.crest_offset)
                    inlets.append(
                        InletExchange(
                            inlet_id=f"{a.node_id}:{a.inlet_type_id}",
                            cell_id=int(node_cell_by_id.get(a.node_id, self._nearest_cell_index_for_xy(node.x, node.y))),
                            node_id=a.node_id,
                            crest_elev=crest,
                            length=max(0.0, float(it.length)) * max(0.0, float(a.multiplier)),
                            area=max(0.0, float(it.area)) * max(0.0, float(a.multiplier)),
                            coeff_weir=max(0.0, float(it.coeff_weir)),
                            coeff_orifice=max(0.0, float(it.coeff_orifice)),
                            max_capture=it.max_capture,
                        )
                    )
            else:
                # Legacy schema: spatial inlets with node and geometry.
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
                            length=float(ft["width"] if "width" in inlet_fields and ft["width"] not in (None, "") else 1.0),
                            area=float(ft["area"] if "area" in inlet_fields and ft["area"] not in (None, "") else 0.0),
                            coeff_weir=float(ft["coeff_weir"] if "coeff_weir" in inlet_fields and ft["coeff_weir"] not in (None, "") else 1.70),
                            coeff_orifice=float(ft["coefficient"] if "coefficient" in inlet_fields and ft["coefficient"] not in (None, "") else 0.62),
                            max_capture=float(ft["max_capture"]) if "max_capture" in inlet_fields and ft["max_capture"] not in (None, "") else None,
                        )
                    )

        # Build outfall exchange objects for outfall-type nodes located within the mesh.
        # Prefer explicit outfall area on node features; fall back to connected-link
        # hydraulic capacity when area is not explicitly provided.
        outfalls: List[OutfallExchange] = []
        if OutfallExchange is not None:
            _node_connected_area: dict = {}
            _node_connected_diameter: dict = {}
            for lnk in links:
                area_lnk = float(lnk.metadata.get("area_m2", 0.0) or 0.0)
                d_lnk = float(lnk.diameter or 0.0)
                if d_lnk <= 0.0:
                    d_lnk = float(lnk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
                if d_lnk <= 0.0:
                    if area_lnk > 0.0:
                        d_lnk = math.sqrt(4.0 * area_lnk / math.pi)
                if area_lnk <= 0.0 and d_lnk > 0.0:
                    area_lnk = 0.25 * math.pi * d_lnk * d_lnk
                for nid in (lnk.from_node_id, lnk.to_node_id):
                    cur_a = float(_node_connected_area.get(nid, 0.0))
                    if area_lnk > cur_a:
                        _node_connected_area[nid] = area_lnk
                    cur = float(_node_connected_diameter.get(nid, 0.0))
                    if d_lnk > cur:
                        _node_connected_diameter[nid] = d_lnk

            outfalls_missing_capacity: List[str] = []
            for node in nodes:
                if str(node.node_type).strip().lower() != "outfall":
                    continue
                cell_id = self._nearest_cell_index_for_xy(float(node.x), float(node.y))
                area_outfall = max(0.0, float(node.metadata.get("outfall_area_m2", 0.0) or 0.0))
                if area_outfall <= 0.0:
                    area_outfall = max(0.0, float(_node_connected_area.get(node.node_id, 0.0) or 0.0))
                diameter = float(_node_connected_diameter.get(node.node_id, 0.0) or 0.0)
                if diameter <= 0.0 and area_outfall > 0.0:
                    diameter = math.sqrt(4.0 * area_outfall / math.pi)
                if area_outfall <= 0.0 and diameter <= 0.0:
                    outfalls_missing_capacity.append(str(node.node_id))
                outfalls.append(
                    OutfallExchange(
                        outfall_id=node.node_id,
                        cell_id=cell_id,
                        node_id=node.node_id,
                        invert_elev=float(node.invert_elev),
                        area_m2=area_outfall,
                        diameter=diameter,
                        coefficient=0.82,
                        max_flow=None,
                        zero_storage=bool(node_zero_storage_by_id.get(node.node_id, False)),
                    )
                )
            if outfalls_missing_capacity:
                preview = ", ".join(outfalls_missing_capacity[:8])
                suffix = "" if len(outfalls_missing_capacity) <= 8 else f", ... (+{len(outfalls_missing_capacity) - 8} more)"
                self._log(
                    "Drainage warning: outfall node(s) missing outfall_area and connected link capacity; "
                    f"outfall exchange will stay zero for IDs: {preview}{suffix}"
                )

        pipe_ends: List[PipeEndExchange] = []
        if PipeEndExchange is not None:
            pipe_end_link_types = {
                "pipe_end", "pipe-end", "daylighted_pipe", "daylighted", "daylight_pipe"
            }
            pipe_end_nodes = {
                str(n.node_id) for n in nodes
                if str(n.node_type).strip().lower() == "pipe_end"
            }
            assigned_pipe_end_nodes: set = set()
            for lnk in links:
                ltype = str(lnk.link_type or "").strip().lower()
                if (
                    ltype not in pipe_end_link_types
                    and str(lnk.from_node_id) not in pipe_end_nodes
                    and str(lnk.to_node_id) not in pipe_end_nodes
                ):
                    continue

                for nid in (str(lnk.from_node_id), str(lnk.to_node_id)):
                    if nid in assigned_pipe_end_nodes:
                        continue
                    node = node_by_id.get(nid)
                    if node is None:
                        continue
                    if str(node.node_type).strip().lower() != "pipe_end":
                        continue

                    cell_id = int(node_cell_by_id.get(nid, self._nearest_cell_index_for_xy(float(node.x), float(node.y))))
                    diameter = float(lnk.diameter or lnk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
                    area_pipe = max(0.0, float(lnk.metadata.get("area_m2", 0.0) or 0.0))
                    if area_pipe <= 0.0 and diameter > 0.0:
                        area_pipe = 0.25 * math.pi * diameter * diameter

                    pipe_ends.append(
                        PipeEndExchange(
                            pipe_end_id=f"pipe_end:{nid}",
                            cell_id=cell_id,
                            node_id=nid,
                            invert_elev=float(node.invert_elev),
                            diameter=diameter,
                            area_m2=area_pipe,
                            coefficient=float(lnk.metadata.get("cd", 0.82) or 0.82),
                            max_flow=lnk.max_flow,
                            inlet_loss_k=float(lnk.metadata.get("pipe_end_inlet_loss_k", lnk.metadata.get("entry_loss_k", 0.5)) or 0.5),
                            outlet_loss_k=float(lnk.metadata.get("pipe_end_outlet_loss_k", lnk.metadata.get("exit_loss_k", 1.0)) or 1.0),
                        )
                    )
                    assigned_pipe_end_nodes.add(nid)

        gravity = float(getattr(self, "_gravity", 9.81))
        solver_mode = int(self.drainage_solver_mode_combo.currentData() if hasattr(self, "drainage_solver_mode_combo") else 0)
        solver_mode_name = str(self.drainage_solver_mode_combo.currentText() if hasattr(self, "drainage_solver_mode_combo") else "EGL")
        self._log(
            f"Drainage coupling configured: nodes={len(nodes)}, links={len(links)}, "
            f"inlets={len(inlets)}, inlet_types={len(inlet_types)}, node_inlets={len(node_inlets)}, "
            f"outfalls={len(outfalls)}, pipe_ends={len(pipe_ends)}, gravity={gravity:.3f}, mode={solver_mode_name}, "
            f"substeps={int(self.drainage_coupling_substeps_spin.value()) if hasattr(self, 'drainage_coupling_substeps_spin') else 1}, "
            f"max_substeps={int(self.drainage_max_coupling_substeps_spin.value()) if hasattr(self, 'drainage_max_coupling_substeps_spin') else 64}, "
            f"gpu_method={str(self.drainage_gpu_method_combo.currentData()) if hasattr(self, 'drainage_gpu_method_combo') else 'step'}, "
            f"deadband={float(self.drainage_head_deadband_spin.value()) if hasattr(self, 'drainage_head_deadband_spin') else 1.0e-3:.4g}, "
            f"relax={float(self.drainage_dynamic_relaxation_spin.value()) if hasattr(self, 'drainage_dynamic_relaxation_spin') else 1.0:.3f}"
        )
        return PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlet_types=inlet_types,
            node_inlets=node_inlets,
            inlets=inlets,
            outfalls=outfalls,
            pipe_ends=pipe_ends,
            gravity=gravity,
            solver_mode=solver_mode,
            coupling_substeps=int(self.drainage_coupling_substeps_spin.value()) if hasattr(self, "drainage_coupling_substeps_spin") else 1,
            max_coupling_substeps=int(self.drainage_max_coupling_substeps_spin.value()) if hasattr(self, "drainage_max_coupling_substeps_spin") else 64,
            head_deadband_m=float(self.drainage_head_deadband_spin.value()) if hasattr(self, "drainage_head_deadband_spin") else 1.0e-3,
            dynamic_flow_relaxation=float(self.drainage_dynamic_relaxation_spin.value()) if hasattr(self, "drainage_dynamic_relaxation_spin") else 1.0,
            adaptive_depth_fraction=float(self.drainage_adaptive_depth_fraction_spin.value()) if hasattr(self, "drainage_adaptive_depth_fraction_spin") else 0.2,
            adaptive_wave_courant=float(self.drainage_adaptive_wave_courant_spin.value()) if hasattr(self, "drainage_adaptive_wave_courant_spin") else 0.5,
            implicit_coupling_iterations=int(self.drainage_implicit_iters_spin.value()) if hasattr(self, "drainage_implicit_iters_spin") else 2,
            implicit_coupling_relaxation=float(self.drainage_implicit_relax_spin.value()) if hasattr(self, "drainage_implicit_relax_spin") else 0.5,
        )

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
            for key in ("width", "height", "diameter", "length", "roughness_n", "coeff", "cd", "opening", "q_pump", "max_flow"):
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
        gravity = float(getattr(self, "_gravity", 9.81))
        self._log(f"Hydraulic structures configured: count={len(structures)}, gravity={gravity:.3f}")
        return HydraulicStructureConfig(enabled=True, structures=structures, gravity=gravity)

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

    def _set_combo_by_layer_id_or_name(self, combo: object, layer_id: object, layer_name: str) -> bool:
        """Set a combo box selection by layer id with name fallback.

        Args:
            combo: Target combo-like widget.
            layer_id: Saved layer id value.
            layer_name: Saved layer display name.

        Returns:
            True when selection was restored, otherwise False.
        """
        if combo is None:
            return False
        if layer_id not in (None, ""):
            idx = combo.findData(str(layer_id))
            if idx >= 0:
                combo.setCurrentIndex(idx)
                return True
        return self._set_combo_by_layer_name(combo, layer_name)

    def _project_layer_binding_specs(self) -> List[Tuple[str, object]]:
        """Return combo attributes that store project-layer bindings.

        Returns:
            List of `(attribute_name, combo_widget)` pairs.
        """
        specs: List[Tuple[str, object]] = []
        for attr_name in [
            "nodes_layer_combo",
            "cells_layer_combo",
            "terrain_layer_combo",
            "manning_layer_combo",
            "cn_layer_combo",
            "rain_gage_layer_combo",
            "hyetograph_layer_combo",
            "storm_area_layer_combo",
            "sample_lines_layer_combo",
            "drain_nodes_layer_combo",
            "drain_links_layer_combo",
            "drain_inlets_layer_combo",
            "drain_node_inlets_layer_combo",
            "structures_layer_combo",
            "bc_lines_layer_combo",
            "topo_nodes_combo",
            "topo_arcs_combo",
            "topo_regions_combo",
            "topo_constraints_combo",
            "topo_quad_edges_combo",
            "layer_group_combo",
        ]:
            combo = getattr(self, attr_name, None)
            if combo is not None:
                specs.append((attr_name, combo))
        return specs

    def _project_entry_read_text(self, key: str, default: str = "") -> str:
        """Read a text value from QGIS project settings.

        Args:
            key: Project entry key name.
            default: Fallback value when key is missing.

        Returns:
            Stored text value, or `default` if unavailable.
        """
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return str(default)
        try:
            result = QgsProject.instance().readEntry("Backwater2DWorkbench", key, str(default))
        except Exception:
            return str(default)
        if isinstance(result, tuple):
            return str(result[0] if result and result[0] not in (None, "") else default)
        return str(result if result not in (None, "") else default)

    def _persist_project_layer_bindings(self, *_args: object) -> None:
        """Persist current layer-combo selections into the QGIS project."""
        if self._project_layer_state_blocked or not _HAVE_QGIS_CORE or QgsProject is None:
            return
        if bool(getattr(self, "_initial_layer_restore_pending", False)):
            return
        payload = {"version": 1, "selectors": {}}
        for attr_name, combo in self._project_layer_binding_specs():
            idx = combo.currentIndex()
            label = str(combo.currentText() or "").strip() if idx >= 0 else ""
            layer_id = combo.currentData()
            layer_id = "" if layer_id in (None, "") else str(layer_id)
            payload["selectors"][attr_name] = {
                "layer_id": layer_id,
                "layer_name": label,
            }
        try:
            QgsProject.instance().writeEntry(
                "Backwater2DWorkbench",
                "layer_selector_state_json",
                json.dumps(payload, separators=(",", ":")),
            )
        except Exception:
            pass

    def _restore_project_layer_bindings(self) -> None:
        """Restore saved layer-combo selections from the QGIS project."""
        if self._project_layer_state_blocked or not _HAVE_QGIS_CORE or QgsProject is None:
            return
        raw = self._project_entry_read_text("layer_selector_state_json", "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            return
        selectors = payload.get("selectors", {}) if isinstance(payload, dict) else {}
        if not isinstance(selectors, dict):
            return

        self._project_layer_state_blocked = True
        try:
            for attr_name, combo in self._project_layer_binding_specs():
                saved = selectors.get(attr_name)
                if not isinstance(saved, dict):
                    continue
                self._set_combo_by_layer_id_or_name(
                    combo,
                    saved.get("layer_id"),
                    str(saved.get("layer_name") or ""),
                )
        finally:
            self._project_layer_state_blocked = False

    def _connect_project_layer_state_signals(self) -> None:
        """Connect layer combo change signals to persistence callback."""
        for _attr_name, combo in self._project_layer_binding_specs():
            try:
                combo.currentIndexChanged.connect(self._persist_project_layer_bindings)
            except Exception:
                pass

    def _connect_project_workbench_state_signals(self) -> None:
        """Connect workbench widget signals to state persistence callback."""
        widget_specs = [
            ("nx_spin", "valueChanged"),
            ("ny_spin", "valueChanged"),
            ("lx_spin", "valueChanged"),
            ("ly_spin", "valueChanged"),
            ("bed_amp_spin", "valueChanged"),
            ("mesh_layout_combo", "currentIndexChanged"),
            ("h_min_spin", "valueChanged"),
            ("initial_condition_combo", "currentIndexChanged"),
            ("initial_depth_spin", "valueChanged"),
            ("initial_wse_spin", "valueChanged"),
            ("adaptive_cfl_dt_chk", "toggled"),
            ("dt_spin", "valueChanged"),
            ("gpu_diag_sync_interval_spin", "valueChanged"),
            ("enable_cuda_graphs_chk", "toggled"),
            ("max_rel_depth_increase_spin", "valueChanged"),
            ("max_source_depth_step_spin", "valueChanged"),
            ("max_source_rate_spin", "valueChanged"),
            ("extreme_rain_mode_chk", "toggled"),
            ("source_cfl_beta_spin", "valueChanged"),
            ("source_max_substeps_spin", "valueChanged"),
            ("source_true_subcycling_chk", "toggled"),
            ("source_imex_split_chk", "toggled"),
            ("source_stage_coupled_imex_rk2_chk", "toggled"),
            ("shallow_damping_depth_spin", "valueChanged"),
            ("shallow_front_recon_fallback_chk", "toggled"),
            ("front_flux_damping_spin", "valueChanged"),
            ("active_set_hysteresis_chk", "toggled"),
            ("depth_cap_spin", "valueChanged"),
            ("momentum_cap_min_speed_spin", "valueChanged"),
            ("momentum_cap_celerity_mult_spin", "valueChanged"),
            ("max_inv_area_spin", "valueChanged"),
            ("cfl_lambda_cap_spin", "valueChanged"),
            ("rain_rate_spin", "valueChanged"),
            ("cn_default_spin", "valueChanged"),
            ("ia_ratio_spin", "valueChanged"),
            ("use_spatial_rain_cn_chk", "toggled"),
            ("infiltration_method_combo", "currentIndexChanged"),
            ("rain_boundary_buffer_rings_spin", "valueChanged"),
            ("internal_flow_field_edit", "editingFinished"),
            ("run_time_edit", "editingFinished"),
            ("output_interval_edit", "editingFinished"),
            ("line_output_interval_edit", "editingFinished"),
            ("reconstruction_combo", "currentIndexChanged"),
            ("temporal_order_combo", "currentIndexChanged"),
            ("degen_mode_combo", "currentIndexChanged"),
            ("coupling_loop_combo", "currentIndexChanged"),
            ("drainage_solver_mode_combo", "currentIndexChanged"),
            ("drainage_backend_combo", "currentIndexChanged"),
            ("drainage_gpu_method_combo", "currentIndexChanged"),
            ("drainage_coupling_substeps_spin", "valueChanged"),
            ("drainage_max_coupling_substeps_spin", "valueChanged"),
            ("drainage_head_deadband_spin", "valueChanged"),
            ("drainage_dynamic_relaxation_spin", "valueChanged"),
            ("drainage_adaptive_depth_fraction_spin", "valueChanged"),
            ("drainage_adaptive_wave_courant_spin", "valueChanged"),
            ("extended_outputs_chk", "toggled"),
            ("save_mesh_results_to_gpkg_chk", "toggled"),
            ("save_line_results_to_gpkg_chk", "toggled"),
            ("save_coupling_results_to_gpkg_chk", "toggled"),
            ("save_run_log_to_gpkg_chk", "toggled"),
            ("topo_backend_combo", "currentIndexChanged"),
            ("topo_default_size_spin", "valueChanged"),
            ("topo_default_cell_type_combo", "currentIndexChanged"),
            ("topo_quality_min_angle_spin", "valueChanged"),
            ("topo_quality_max_aspect_spin", "valueChanged"),
            ("topo_quality_max_non_orth_spin", "valueChanged"),
            ("topo_quality_min_area_edit", "editingFinished"),
            ("topo_quality_size_scales_edit", "editingFinished"),
            ("topo_quality_smooth_increments_edit", "editingFinished"),
            ("topo_quality_strict_chk", "toggled"),
            ("topo_gmsh_quality_enable_chk", "toggled"),
            ("topo_gmsh_quality_max_iters_spin", "valueChanged"),
            ("topo_gmsh_quality_time_limit_spin", "valueChanged"),
            ("topo_gmsh_tri_algo_combo", "currentIndexChanged"),
            ("topo_gmsh_quad_algo_combo", "currentIndexChanged"),
            ("topo_gmsh_recombine_algo_combo", "currentIndexChanged"),
            ("topo_gmsh_smoothing_spin", "valueChanged"),
            ("topo_gmsh_optimize_iters_spin", "valueChanged"),
            ("topo_gmsh_optimize_netgen_chk", "toggled"),
            ("topo_gmsh_verbosity_spin", "valueChanged"),
        ]

        for attr_name, signal_name in widget_specs:
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            try:
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    signal.connect(self._persist_project_workbench_state)
            except Exception:
                pass

    def _connect_project_save_state_signals(self) -> None:
        """Connect QGIS project read/save signals to workbench state sync."""
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return
        proj = QgsProject.instance()
        
        # Try multiple signal names for QGIS project save/load events
        signal_names = ["aboutToBeSaved", "writeProject", "projectSaved"]
        connected_count = 0
        
        for signal_name in signal_names:
            try:
                signal = getattr(proj, signal_name, None)
                if signal is not None:
                    signal.connect(self._persist_project_workbench_state)
                    connected_count += 1
            except Exception as e:
                self._log(f"[DEBUG] failed to connect signal {signal_name}: {e}")
        
        self._log(f"[DEBUG] connected {connected_count} project signals for workbench state persistence")
        for signal_name in ("readProject", "projectRead"):
            try:
                signal = getattr(proj, signal_name, None)
                if signal is not None:
                    signal.connect(self._restore_project_layer_bindings)
                    signal.connect(self._restore_project_workbench_state)
            except Exception:
                pass

    def _persist_project_workbench_state(self, *_args: object) -> None:
        """Persist workbench widget values to the active QGIS project.

        The method captures supported widget types (spin boxes, combo boxes,
        checkboxes, and line edits) and stores a compact JSON payload in the
        project under `Backwater2DWorkbench/workbench_state_json`.
        """
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return
        payload = {
            "version": 1,
            "widgets": {}
        }
        
        # Collect all widgets that should be persisted
        widget_attrs = [
            # Mesh generation
            "nx_spin", "ny_spin", "lx_spin", "ly_spin", "bed_amp_spin", "mesh_layout_combo",
            # Parameters
            "h_min_spin", "initial_condition_combo", "initial_depth_spin", "initial_wse_spin",
            "adaptive_cfl_dt_chk", "dt_spin", "gpu_diag_sync_interval_spin", "max_rel_depth_increase_spin",
            "enable_cuda_graphs_chk",
            "shallow_damping_depth_spin", "shallow_front_recon_fallback_chk",
            "front_flux_damping_spin", "active_set_hysteresis_chk",
            "depth_cap_spin", "momentum_cap_min_speed_spin", "momentum_cap_celerity_mult_spin",
            "max_inv_area_spin", "cfl_lambda_cap_spin", "rain_rate_spin", "cn_default_spin",
            "ia_ratio_spin",
            "max_source_depth_step_spin", "max_source_rate_spin", "extreme_rain_mode_chk",
            "source_cfl_beta_spin", "source_max_substeps_spin", "source_true_subcycling_chk",
            "source_imex_split_chk", "source_stage_coupled_imex_rk2_chk",
            "use_spatial_rain_cn_chk", "infiltration_method_combo", "rain_boundary_buffer_rings_spin", "internal_flow_field_edit",
            "run_time_edit", "output_interval_edit", "line_output_interval_edit",
            "reconstruction_combo", "temporal_order_combo", "degen_mode_combo", "coupling_loop_combo",
            "drainage_solver_mode_combo", "drainage_backend_combo", "drainage_gpu_method_combo", "drainage_coupling_substeps_spin",
            "drainage_max_coupling_substeps_spin", "drainage_head_deadband_spin",
            "drainage_dynamic_relaxation_spin", "drainage_adaptive_depth_fraction_spin",
            "drainage_adaptive_wave_courant_spin", "drainage_implicit_iters_spin",
            "drainage_implicit_relax_spin", "extended_outputs_chk",
            "save_mesh_results_to_gpkg_chk", "save_line_results_to_gpkg_chk",
            "save_coupling_results_to_gpkg_chk", "save_run_log_to_gpkg_chk",
            # Topology mesh
            "topo_backend_combo", "topo_default_size_spin", "topo_default_cell_type_combo",
            "topo_quality_min_angle_spin", "topo_quality_max_aspect_spin",
            "topo_quality_max_non_orth_spin", "topo_quality_min_area_edit",
            "topo_quality_size_scales_edit", "topo_quality_smooth_increments_edit",
            "topo_quality_strict_chk", "topo_gmsh_quality_enable_chk",
            "topo_gmsh_quality_max_iters_spin", "topo_gmsh_quality_time_limit_spin",
            "topo_gmsh_tri_algo_combo", "topo_gmsh_quad_algo_combo",
            "topo_gmsh_recombine_algo_combo", "topo_gmsh_smoothing_spin",
            "topo_gmsh_optimize_iters_spin", "topo_gmsh_optimize_netgen_chk",
            "topo_gmsh_verbosity_spin",
        ]
        
        for attr_name in widget_attrs:
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            
            value = None
            if isinstance(widget, QtWidgets.QSpinBox):
                value = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentData()
                if value is None:
                    value = widget.currentIndex()
            elif isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
            else:
                continue
            
            payload["widgets"][attr_name] = {
                "type": type(widget).__name__,
                "value": value
            }
        
        try:
            json_str = json.dumps(payload, separators=(",", ":"), default=str)
            QgsProject.instance().writeEntry(
                "Backwater2DWorkbench",
                "workbench_state_json",
                json_str,
            )
            self._log(f"[DEBUG] persist: saved {len(payload['widgets'])} widgets to project")
        except Exception as e:
            self._log(f"[DEBUG] persist: writeEntry failed: {e}")

    def _restore_project_workbench_state(self, *_args: object) -> None:
        """Restore persisted workbench widget values from QGIS project state."""
        if not _HAVE_QGIS_CORE or QgsProject is None:
            self._log("[DEBUG] restore: QGIS core not available")
            return
        
        raw = self._project_entry_read_text("workbench_state_json", "")
        if not raw:
            self._log("[DEBUG] restore: no saved workbench state found")
            return
        
        self._log(f"[DEBUG] restore: found {len(raw)} chars of state data")
        try:
            payload = json.loads(raw)
        except Exception as e:
            self._log(f"[DEBUG] restore: json parse failed: {e}")
            return
        
        widgets_data = payload.get("widgets", {}) if isinstance(payload, dict) else {}
        if not isinstance(widgets_data, dict):
            self._log("[DEBUG] restore: widgets not a dict")
            return
        
        self._log(f"[DEBUG] restore: restoring {len(widgets_data)} widget values")
        restored_count = 0
        for attr_name, widget_info in widgets_data.items():
            widget = getattr(self, attr_name, None)
            if widget is None or not isinstance(widget_info, dict):
                continue
            
            value = widget_info.get("value")
            if value is None:
                continue
            
            try:
                if isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                    restored_count += 1
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(value))
                    restored_count += 1
                elif isinstance(widget, QtWidgets.QComboBox):
                    # Try to find item by data first, then by index
                    found = False
                    for i in range(widget.count()):
                        if widget.itemData(i) == value:
                            widget.setCurrentIndex(i)
                            found = True
                            break
                    if not found:
                        # Fallback to index
                        try:
                            widget.setCurrentIndex(int(value))
                        except Exception:
                            pass
                    restored_count += 1
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(value))
                    restored_count += 1
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
                    restored_count += 1
            except Exception as e:
                self._log(f"[DEBUG] restore: failed to restore {attr_name}: {e}")
                continue
        
        self._log(f"[DEBUG] restore: successfully restored {restored_count} of {len(widgets_data)} widgets")

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
                ("swe2d_coupling_schema_version", str(_MODEL_LAYER_BINDINGS_VERSION), now),
            )
            cur.execute(
                "INSERT OR REPLACE INTO swe2d_model_metadata(key, value, updated_utc) VALUES (?, ?, ?)",
                ("swe2d_coupling_layer_roles", json.dumps(sorted(_MODEL_LAYER_BINDINGS.keys())), now),
            )

            explicit_roles: List[str] = []
            for role, spec in _MODEL_LAYER_BINDINGS.items():
                combo_attr = str(spec.get("combo_attr", ""))
                combo = getattr(self, combo_attr, None)
                selected_name = self._combo_current_layer_name(combo) if combo is not None else ""
                if selected_name:
                    explicit_roles.append(str(role))
                cur.execute(
                    """
                    INSERT OR REPLACE INTO swe2d_layer_bindings(
                        role, layer_name, geometry_type, required_fields, updated_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        role,
                        selected_name,
                        str(spec.get("geometry", "")),
                        ",".join(spec.get("required_fields", ())),
                        now,
                    ),
                )
            cur.execute(
                "INSERT OR REPLACE INTO swe2d_model_metadata(key, value, updated_utc) VALUES (?, ?, ?)",
                ("swe2d_explicit_layer_roles", json.dumps(sorted(explicit_roles)), now),
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
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_model_metadata'"
            )
            have_metadata = cur.fetchone() is not None
            schema_version = 0
            explicit_roles: Optional[set[str]] = None
            if have_metadata:
                try:
                    cur.execute(
                        "SELECT value FROM swe2d_model_metadata WHERE key='swe2d_coupling_schema_version'"
                    )
                    row = cur.fetchone()
                    if row and row[0] not in (None, ""):
                        schema_version = int(str(row[0]))
                except Exception:
                    schema_version = 0
                try:
                    cur.execute(
                        "SELECT value FROM swe2d_model_metadata WHERE key='swe2d_explicit_layer_roles'"
                    )
                    row = cur.fetchone()
                    if row and row[0] not in (None, ""):
                        parsed = json.loads(str(row[0]))
                        if isinstance(parsed, list):
                            explicit_roles = {str(v) for v in parsed if str(v).strip()}
                except Exception:
                    explicit_roles = None
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
            role_name = str(role)
            spec = _MODEL_LAYER_BINDINGS.get(role_name)
            if spec is None:
                continue
            combo = getattr(self, str(spec.get("combo_attr", "")), None)
            if combo is None:
                continue
            canonical_name = str(spec.get("layer_name", ""))
            if schema_version >= _MODEL_LAYER_BINDINGS_VERSION:
                if explicit_roles is not None and role_name not in explicit_roles:
                    combo.setCurrentIndex(0)
                    continue
            else:
                # Legacy GeoPackages could persist canonical optional layer names
                # even when the user left the selector on "(none)".
                if not layer_name or str(layer_name).strip() == canonical_name:
                    combo.setCurrentIndex(0)
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

        def _format_id_preview(ids: Sequence[str], limit: int = 10) -> str:
            vals = [str(v) for v in ids if str(v)]
            if not vals:
                return "(none)"
            if len(vals) <= limit:
                return ", ".join(vals)
            return ", ".join(vals[:limit]) + f", ... (+{len(vals) - limit} more)"

        if pipe_cfg is not None:
            lines.append(
                f"Drainage network: nodes={len(pipe_cfg.nodes)}, links={len(pipe_cfg.links)}, inlets={len(pipe_cfg.inlets)}"
            )

            node_by_id = {str(n.node_id): n for n in pipe_cfg.nodes}
            unknown_link_refs: List[str] = []
            unknown_inlet_refs: List[str] = []
            zero_capacity_links: List[str] = []
            near_zero_head_links: List[str] = []
            t0_probably_zero_links: List[str] = []

            for lk in pipe_cfg.links:
                lid = str(lk.link_id)
                n0 = node_by_id.get(str(lk.from_node_id))
                n1 = node_by_id.get(str(lk.to_node_id))
                if n0 is None or n1 is None:
                    unknown_link_refs.append(lid)
                    continue

                d = float(lk.diameter) if lk.diameter is not None else 0.0
                a = float(lk.metadata.get("area_m2", 0.0) or 0.0)
                eqd = float(lk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
                has_capacity = (d > 0.0) or (a > 0.0) or (eqd > 0.0)
                if not has_capacity:
                    zero_capacity_links.append(lid)

                dh0 = float(n0.invert_elev) - float(n1.invert_elev)
                near_zero_head = abs(dh0) <= 1.0e-4
                if near_zero_head:
                    near_zero_head_links.append(lid)

                if (not has_capacity) or near_zero_head:
                    t0_probably_zero_links.append(lid)

            for inlet in pipe_cfg.inlets:
                if str(inlet.node_id) not in node_by_id:
                    unknown_inlet_refs.append(str(inlet.inlet_id))

            lines.append("Coupling sanity report (drainage):")
            lines.append(f"- unknown link node refs: {len(unknown_link_refs)}")
            if unknown_link_refs:
                lines.append(f"  IDs: {_format_id_preview(unknown_link_refs)}")
            lines.append(f"- unknown inlet node refs: {len(unknown_inlet_refs)}")
            if unknown_inlet_refs:
                lines.append(f"  IDs: {_format_id_preview(unknown_inlet_refs)}")
            lines.append(f"- links with zero hydraulic capacity fields: {len(zero_capacity_links)}")
            if zero_capacity_links:
                lines.append(f"  IDs: {_format_id_preview(zero_capacity_links)}")
            lines.append(f"- links with near-zero initial head gradient (|dh0|<=1e-4): {len(near_zero_head_links)}")
            if near_zero_head_links:
                lines.append(f"  IDs: {_format_id_preview(near_zero_head_links)}")
            lines.append(f"- links likely zero-flow at t0 (capacity/head limits): {len(t0_probably_zero_links)}")
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

    def _collect_bc_layer_edge_groups(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> Dict[int, str]:
        """Return boundary-edge grouping labels from BC-line override features.

        Mapping key is boundary-edge row index in edge_n0/edge_n1 arrays.
        Values are labels like ``bc_line:<name>``. Only edges matched to BC-line
        features are included; callers should provide side-based fallback labels.
        """
        edge_groups: Dict[int, str] = {}
        if not _HAVE_QGIS_CORE:
            return edge_groups
        if not hasattr(self, "bc_lines_layer_combo"):
            return edge_groups

        bc_layer = self._combo_layer(self.bc_lines_layer_combo, "vector")
        if bc_layer is None:
            return edge_groups

        fields = set(bc_layer.fields().names())
        name_field = "name" if "name" in fields else None
        prio_field = "priority" if "priority" in fields else None

        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]

        features = []
        for ft in bc_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            pr = 0
            if prio_field is not None:
                try:
                    pr = int(ft[prio_field])
                except Exception:
                    pr = 0
            nm = ""
            if name_field is not None:
                try:
                    nm = str(ft[name_field] or "").strip()
                except Exception:
                    nm = ""
            if not nm:
                try:
                    nm = f"feature_{int(ft.id())}"
                except Exception:
                    nm = "feature"
            features.append((pr, geom, nm))

        if not features:
            return edge_groups

        features.sort(key=lambda x: x[0], reverse=True)
        for i in range(edge_n0.size):
            x0 = float(node_x[edge_n0[i]])
            y0 = float(node_y[edge_n0[i]])
            x1 = float(node_x[edge_n1[i]])
            y1 = float(node_y[edge_n1[i]])
            tol = math.hypot(x1 - x0, y1 - y0) * 0.5
            mid = QgsGeometry.fromPointXY(QgsPointXY(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
            for _pr, g, nm in features:
                if mid.distance(g) < tol:
                    edge_groups[i] = f"bc_line:{nm}"
                    break

        return edge_groups

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

    def _boundary_buffer_cells(self, n_rings: int) -> np.ndarray:
        """Return cell indices within n_rings of the mesh boundary.

        Ring 1 includes boundary cells. Ring 2 adds cells adjacent to ring 1, etc.
        """
        if self._mesh_data is None or int(n_rings) <= 0:
            return np.empty(0, dtype=np.int32)

        edge_cells: Dict[Tuple[int, int], List[int]] = {}
        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = self._mesh_data["cell_face_offsets"].astype(np.int32)
            faces = self._mesh_data["cell_face_nodes"].astype(np.int32)
            n_cells = int(offs.size - 1)
            for ci in range(n_cells):
                s = int(offs[ci])
                e = int(offs[ci + 1])
                poly = faces[s:e]
                for k in range(poly.size):
                    a = int(poly[k])
                    b = int(poly[(k + 1) % poly.size])
                    key = (min(a, b), max(a, b))
                    edge_cells.setdefault(key, []).append(ci)
        else:
            tris = self._mesh_data["cell_nodes"].reshape((-1, 3)).astype(np.int32)
            n_cells = int(tris.shape[0])
            for ci, tri in enumerate(tris):
                for k in range(3):
                    a = int(tri[k])
                    b = int(tri[(k + 1) % 3])
                    key = (min(a, b), max(a, b))
                    edge_cells.setdefault(key, []).append(ci)

        # Build adjacency and boundary seed set from edge ownership.
        neighbors: List[set] = [set() for _ in range(n_cells)]
        ring = set()
        for owners in edge_cells.values():
            if len(owners) == 1:
                ring.add(int(owners[0]))
            elif len(owners) == 2:
                c0 = int(owners[0])
                c1 = int(owners[1])
                neighbors[c0].add(c1)
                neighbors[c1].add(c0)

        if not ring:
            return np.empty(0, dtype=np.int32)

        selected = set(ring)
        for _ in range(1, int(n_rings)):
            nxt = set()
            for c in ring:
                nxt.update(neighbors[c])
            nxt.difference_update(selected)
            if not nxt:
                break
            selected.update(nxt)
            ring = nxt

        return np.asarray(sorted(selected), dtype=np.int32)

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
        run_id = ""
        run_wallclock_start = ""
        run_perf_start = time.perf_counter()
        run_log_start_idx = len(self._runtime_log_lines)
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
            edge_group_overrides = self._collect_bc_layer_edge_groups(bc_n0, bc_n1)
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
            temporal_order_value = int(self.temporal_order_combo.currentData())
            temporal_scheme = TemporalScheme(temporal_order_value)
            temporal_scheme_name = self.temporal_order_combo.currentText().strip()
            godunov_mode_value = int(self.godunov_mode_combo.currentData()) if hasattr(self, "godunov_mode_combo") else int(GodunovSolverMode.CURRENT_GPU_STEP)
            godunov_mode = GodunovSolverMode(godunov_mode_value)
            if godunov_mode == GodunovSolverMode.GODUNOV_ROLLOUT:
                promoted_temporal = max(temporal_order_value, int(TemporalScheme.SSP_RK2))
                promoted_reconstruction = max(reconstruction_mode, int(SpatialDiscretization.FV_MUSCL_MINMOD))
                if promoted_temporal != temporal_order_value:
                    self._log("Godunov rollout selected: promoting temporal integration to RK2.")
                    temporal_order_value = promoted_temporal
                    temporal_scheme = TemporalScheme(temporal_order_value)
                    temporal_scheme_name = self.temporal_order_combo.itemText(self.temporal_order_combo.findData(temporal_order_value)).strip() if self.temporal_order_combo.findData(temporal_order_value) >= 0 else temporal_scheme_name
                if promoted_reconstruction != reconstruction_mode:
                    self._log("Godunov rollout selected: promoting reconstruction to MUSCL MinMod.")
                    reconstruction_mode = promoted_reconstruction
                    reconstruction_name = self.reconstruction_combo.itemText(self.reconstruction_combo.findData(reconstruction_mode)).strip() if self.reconstruction_combo.findData(reconstruction_mode) >= 0 else reconstruction_name
            coupling_loop_mode = str(self.coupling_loop_combo.currentData() if hasattr(self, "coupling_loop_combo") else "cpu")
            drainage_solver_backend_mode = str(self.drainage_backend_combo.currentData() if hasattr(self, "drainage_backend_combo") else "cpu")
            drainage_gpu_method_mode = str(self.drainage_gpu_method_combo.currentData() if hasattr(self, "drainage_gpu_method_combo") else "step")
            cuda_graphs_enabled = bool(getattr(self, "enable_cuda_graphs_chk", None) and self.enable_cuda_graphs_chk.isChecked())
            if (
                cuda_graphs_enabled
                and int(temporal_order_value) >= 4
                and str(coupling_loop_mode).strip().lower() == "cuda"
                and str(drainage_solver_backend_mode).strip().lower() == "gpu"
            ):
                # Safety guard: avoid CUDA graph replay for RK4 with CUDA drainage/coupling.
                # This combination can trigger illegal memory access on some runs/devices.
                cuda_graphs_enabled = False
                self._log(
                    "CUDA graph replay auto-disabled for RK4 + CUDA drainage/coupling runtime "
                    "to avoid illegal memory access."
                )
            os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "1" if cuda_graphs_enabled else "0"
            rain_rate_model = self._rain_rate_si_to_model(float(self.rain_rate_spin.value()) / 1000.0 / 3600.0)
            internal_flow_forcing = self._build_internal_flow_forcing()
            cell_source_si = self._internal_flow_source_cms_at_time(internal_flow_forcing, 0.0)
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
                    drainage_solver_backend=drainage_solver_backend_mode,
                    drainage_gpu_method=drainage_gpu_method_mode,
                )
                # GPU-first runtime policy: for legacy saved projects that still
                # carry CPU coupling selections, opportunistically promote to
                # CUDA/GPU coupling when native bindings are available.
                force_cpu_coupling = os.environ.get("BACKWATER_SWE2D_FORCE_CPU_COUPLING", "").strip() == "1"
                if not force_cpu_coupling:
                    try:
                        native_mod = coupling_controller._native_cuda_module() if hasattr(coupling_controller, "_native_cuda_module") else None
                    except Exception:
                        native_mod = None
                    if native_mod is not None and str(coupling_loop_mode).strip().lower() == "cpu":
                        coupling_loop_mode = "cuda"
                        coupling_controller.coupling_loop = "cuda"
                        self._log("Coupling loop auto-promoted: CPU -> CUDA (native CUDA coupling available).")
                    if (
                        native_mod is not None
                        and str(drainage_solver_backend_mode).strip().lower() == "cpu"
                        and hasattr(native_mod, "swe2d_gpu_drainage_step")
                        and getattr(coupling_controller, "drainage", None) is not None
                    ):
                        drainage_solver_backend_mode = "gpu"
                        coupling_controller.drainage_solver_backend = "gpu"
                        self._log("Drainage backend auto-promoted: CPU -> GPU (native CUDA drainage available).")
            rain_stats_acc = {"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0}

            # Snapshot output interval — clamp to at least 1 s to avoid div-by-zero
            _oi_hr = self._parse_time_hours(self.output_interval_edit.text())
            output_interval_s = max(1.0, _oi_hr * 3600.0)
            _line_oi_hr = self._parse_time_hours(self.line_output_interval_edit.text())
            line_output_interval_s = max(1.0, _line_oi_hr * 3600.0)
            self._snapshot_timesteps = []
            self._line_snapshot_rows = []
            self._line_snapshot_profile_rows = []
            self._coupling_snapshot_rows = []
            _next_snap_t = output_interval_s
            _next_line_snap_t = line_output_interval_s
            _next_coupling_snap_t = line_output_interval_s
            sample_map = self._build_line_sampling_map()
            cell_min_z = self._mesh_cell_min_bed() if sample_map else None
            run_id = datetime.datetime.utcnow().strftime("swe2d_%Y%m%dT%H%M%SZ")
            run_wallclock_start = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")

            dynamic_bc = bool(np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs)
            if dynamic_bc:
                self._log("Timeseries BC mode active (flow/stage hydrographs).")

            self._log("Starting 2D run...")
            self._log(f"Run wallclock start: {run_wallclock_start}")
            self._log(f"Reconstruction mode: {reconstruction_name}")
            self._log(f"Temporal scheme: {temporal_scheme_name}")
            self._log(
                f"Output intervals: mesh={output_interval_s:.1f}s, sample-lines={line_output_interval_s:.1f}s"
            )
            self._log(
                "Stability controls: "
                f"max_rel_dh={float(self.max_rel_depth_increase_spin.value()):.3f}, "
                f"gpu_diag_sync_steps={int(self.gpu_diag_sync_interval_spin.value())}, "
                f"src_dh_step_cap={float(self.max_source_depth_step_spin.value()):.6e}, "
                f"src_rate_cap={float(self.max_source_rate_spin.value()):.6e}, "
                f"extreme_rain_mode={bool(self.extreme_rain_mode_chk.isChecked())}, "
                f"src_beta={float(self.source_cfl_beta_spin.value()):.3f}, "
                f"src_max_substeps={int(self.source_max_substeps_spin.value())}, "
                f"true_subcycling={bool(self.source_true_subcycling_chk.isChecked())}, "
                f"imex_split={bool(self.source_imex_split_chk.isChecked())}, "
                f"stage_coupled_imex_rk2={bool(getattr(self, 'source_stage_coupled_imex_rk2_chk', None) and self.source_stage_coupled_imex_rk2_chk.isChecked())}, "
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
                infil_method = str(getattr(thiessen_forcing, "infiltration_method", "scs_cn") or "scs_cn").lower().strip()
                infil_label = "NRCS CN infiltration"
                if infil_method == "none":
                    infil_label = "no infiltration (all rainfall to runoff)"
                self._log(
                    "Spatial rainfall forcing active: Thiessen nearest-gage interpolation + "
                    f"{infil_label}."
                )
            if cell_source_model is not None:
                self._log(
                    f"Internal source/sink forcing active: total_Q={float(np.sum(cell_source_model)):.6f} {self._flow_unit_label()}"
                )
            if internal_flow_forcing is not None:
                ts_count = int(len(internal_flow_forcing.get("dynamic_terms", [])))
                if ts_count > 0:
                    self._log(f"Internal flow time-series forcing active: features={ts_count}")
            if coupling_controller is not None:
                self._log(
                    "Coupled drainage/structure forcing active: "
                    f"drainage={pipe_network_cfg is not None}, structures={hydraulic_structures_cfg is not None}, "
                    f"loop={coupling_loop_mode}, drainage_backend={drainage_solver_backend_mode}, "
                    f"drainage_gpu_method={drainage_gpu_method_mode}"
                )
                coupling_runtime_mode = "cpu"
                if str(coupling_loop_mode).strip().lower() == "cuda":
                    try:
                        mod = coupling_controller._native_cuda_module() if hasattr(coupling_controller, "_native_cuda_module") else None
                    except Exception:
                        mod = None
                    if mod is not None:
                        coupling_runtime_mode = "cuda"
                    else:
                        coupling_runtime_mode = "cpu (cuda requested, fallback active)"
                self._log(f"Coupling runtime mode: {coupling_runtime_mode}")
            self._log(f"CUDA graph replay: {'enabled' if cuda_graphs_enabled else 'disabled'}")
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
            def _build_and_initialize_backend() -> SWE2DBackend:
                b = SWE2DBackend()

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
                    b.build_mesh(
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
                    b.build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp_init, bc_vl_init)

                b.initialize(
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
                    extreme_rain_mode=bool(self.extreme_rain_mode_chk.isChecked()),
                    source_cfl_beta=float(self.source_cfl_beta_spin.value()),
                    source_max_substeps=int(self.source_max_substeps_spin.value()),
                    source_rate_cap=float(self.max_source_rate_spin.value()),
                    source_depth_step_cap=float(self.max_source_depth_step_spin.value()),
                    source_true_subcycling=bool(self.source_true_subcycling_chk.isChecked()),
                    source_imex_split=bool(self.source_imex_split_chk.isChecked()),
                    enable_shallow_front_recon_fallback=bool(self.shallow_front_recon_fallback_chk.isChecked()),
                    gpu_diag_sync_interval_steps=int(self.gpu_diag_sync_interval_spin.value()),
                    spatial_discretization=reconstruction_mode,
                    temporal_scheme=temporal_scheme,
                    godunov_mode=godunov_mode,
                    degen_mode=int(self.degen_mode_combo.currentData()),
                    front_flux_damping=float(self.front_flux_damping_spin.value()),
                    active_set_hysteresis=bool(self.active_set_hysteresis_chk.isChecked()),
                )
                return b

            try:
                backend = _build_and_initialize_backend()
            except Exception as init_exc:
                err_l = str(init_exc).lower()
                is_illegal_mem = "illegal memory access" in err_l
                if cuda_graphs_enabled and is_illegal_mem:
                    self._log(
                        "CUDA solver init failed with illegal memory access while graph replay was enabled; "
                        "retrying once with CUDA graph replay disabled."
                    )
                    cuda_graphs_enabled = False
                    os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "0"
                    backend = _build_and_initialize_backend()
                    self._log("CUDA graph replay fallback at solver init succeeded.")
                else:
                    raise

            last_diag = None
            t_accum = 0.0
            i = 0
            last_valid_cmax = float("nan")
            last_valid_wse_res = float("nan")
            # Wall-clock throttle for QApplication.processEvents() – fire at most
            # every _PROCESS_EVENTS_INTERVAL_S seconds regardless of step count.
            # This prevents QGIS canvas repaints from dominating the loop when
            # solver steps are short (e.g. small meshes, fast GPU).
            _PROCESS_EVENTS_INTERVAL_S = 0.10  # 100 ms
            _last_process_events_wall = time.perf_counter()
            timing_totals_ms = {
                "wall": 0.0,
                "step": 0.0,
                "coupling": 0.0,
                "source": 0.0,
                "state": 0.0,
                "bc": 0.0,
                "ui": 0.0,
            }
            timing_samples = 0
            self._log("Step timing diagnostics enabled (ms): wall, step, coupling, source, state, bc, ui.")
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
                    payload = thiessen_forcing.build_native_preprocessed_payload()
                    cell_gage_idx = np.asarray(payload.get("cell_gage_idx"), dtype=np.int32).ravel()
                    gage_offsets = np.asarray(payload.get("gage_offsets"), dtype=np.int32).ravel()
                    hg_time_s = np.asarray(payload.get("hg_time_s"), dtype=np.float64).ravel()
                    hg_cum_mm = np.asarray(payload.get("hg_cum_mm"), dtype=np.float64).ravel()
                    cn_arr = np.asarray(payload.get("cn"), dtype=np.float64).ravel()
                    ia_ratio = float(np.asarray(payload.get("ia_ratio", [0.0]), dtype=np.float64).ravel()[0])

                    if cell_gage_idx.size > 0 and np.any(cell_gage_idx >= 0):
                        backend.set_rain_cn_forcing_native(
                            cell_gage_idx=cell_gage_idx,
                            gage_offsets=gage_offsets,
                            hg_time_s=hg_time_s,
                            hg_cum_mm=hg_cum_mm,
                            cn=cn_arr,
                            ia_ratio=ia_ratio,
                            mm_to_model_depth=float(self._rain_mm_to_model_depth()),
                        )
                        native_rain_cn_forcing = True
                        infil_method_native = str(getattr(thiessen_forcing, "infiltration_method", "scs_cn") or "scs_cn").lower().strip()
                        self._log(
                            "Native preprocessed rainfall-excess forcing configured for GPU timestep evaluation "
                            f"(infiltration={infil_method_native}, groups={max(0, int(gage_offsets.size) - 1)})."
                        )
                except Exception as exc:
                    self._log(f"Native rain+CN forcing unavailable: {exc}")

            native_source_injection_mode = hasattr(backend, "set_external_sources_native")
            if native_source_injection_mode:
                try:
                    backend.set_external_sources_native(None)
                    self._log("Native external source injection enabled (device-resident coupling path).")
                except Exception as exc:
                    native_source_injection_mode = False
                    self._log(f"Native external source injection unavailable: {exc}")

            area_model = np.asarray(self._mesh_cell_areas(), dtype=np.float64).ravel()
            n_area = int(area_model.size)
            h0_model = np.asarray(h0, dtype=np.float64).ravel()
            n_store = min(n_area, int(h0_model.size))
            storage_start_model = float(np.sum(h0_model[:n_store] * area_model[:n_store])) if n_store > 0 else 0.0
            source_budget_model = {
                "rain": 0.0,
                "cell": 0.0,
                "coupling": 0.0,
            }

            node_x_bc = self._mesh_data["node_x"]
            node_y_bc = self._mesh_data["node_y"]
            edge_len_bc = np.hypot(node_x_bc[bc_n1] - node_x_bc[bc_n0], node_y_bc[bc_n1] - node_y_bc[bc_n0]).astype(np.float64)
            xmin_bc = float(np.min(node_x_bc)) if node_x_bc.size else 0.0
            xmax_bc = float(np.max(node_x_bc)) if node_x_bc.size else 0.0
            ymin_bc = float(np.min(node_y_bc)) if node_y_bc.size else 0.0
            ymax_bc = float(np.max(node_y_bc)) if node_y_bc.size else 0.0
            mx_bc = 0.5 * (node_x_bc[bc_n0] + node_x_bc[bc_n1]) if bc_n0.size else np.empty(0, dtype=np.float64)
            my_bc = 0.5 * (node_y_bc[bc_n0] + node_y_bc[bc_n1]) if bc_n0.size else np.empty(0, dtype=np.float64)
            if bc_n0.size:
                d_bc = np.vstack([
                    np.abs(mx_bc - xmin_bc),
                    np.abs(mx_bc - xmax_bc),
                    np.abs(my_bc - ymin_bc),
                    np.abs(my_bc - ymax_bc),
                ])
                side_idx_bc = np.argmin(d_bc, axis=0)
            else:
                side_idx_bc = np.empty(0, dtype=np.int32)
            side_names_bc = ["left", "right", "bottom", "top"]
            edge_group_labels: List[str] = []
            for ei in range(int(bc_n0.size)):
                if ei in edge_group_overrides:
                    edge_group_labels.append(str(edge_group_overrides[ei]))
                else:
                    edge_group_labels.append(str(side_names_bc[int(side_idx_bc[ei])]))
            boundary_flux_budget_model: Dict[str, float] = {}

            def _accumulate_boundary_flux_volume_model(
                dt_apply_s: float,
                bc_type_local: np.ndarray,
                bc_val_local: np.ndarray,
            ) -> None:
                dt_apply = max(0.0, float(dt_apply_s))
                if dt_apply <= 0.0 or bc_n0.size == 0:
                    return
                bt = np.asarray(bc_type_local, dtype=np.int32).ravel()
                bv = np.asarray(bc_val_local, dtype=np.float64).ravel()
                n = min(int(bt.size), int(bv.size), int(edge_len_bc.size), len(edge_group_labels))
                if n <= 0:
                    return
                flow_mask = bt[:n] == int(_BC_INFLOW_Q)
                if not np.any(flow_mask):
                    return
                q_total = np.asarray(bv[:n], dtype=np.float64) * np.asarray(edge_len_bc[:n], dtype=np.float64)
                vol = q_total * dt_apply
                idx = np.nonzero(flow_mask)[0]
                for ii in idx.tolist():
                    grp = str(edge_group_labels[ii])
                    vv = float(vol[ii])
                    if not np.isfinite(vv):
                        continue
                    boundary_flux_budget_model[grp] = float(boundary_flux_budget_model.get(grp, 0.0) + vv)

            def _accumulate_source_volume_model(
                dt_apply_s: float,
                rain_rate_model_local,
                cell_source_model_local: Optional[np.ndarray],
                coupled_source_rate_local: Optional[np.ndarray],
            ) -> None:
                dt_apply = max(0.0, float(dt_apply_s))
                if dt_apply <= 0.0 or n_area <= 0:
                    return

                rain_arr = np.asarray(rain_rate_model_local, dtype=np.float64)
                if rain_arr.ndim == 0:
                    rain_vol = float(rain_arr) * float(np.sum(area_model)) * dt_apply
                else:
                    n = min(int(rain_arr.size), n_area)
                    rain_vol = float(np.sum(rain_arr[:n] * area_model[:n]) * dt_apply)
                if np.isfinite(rain_vol):
                    source_budget_model["rain"] += rain_vol

                if cell_source_model_local is not None:
                    cell_arr = np.asarray(cell_source_model_local, dtype=np.float64).ravel()
                    if cell_arr.size > 0:
                        n = min(int(cell_arr.size), n_area)
                        cell_vol = float(np.sum(cell_arr[:n]) * dt_apply)
                        if np.isfinite(cell_vol):
                            source_budget_model["cell"] += cell_vol

                if coupled_source_rate_local is not None:
                    cpl_arr = np.asarray(coupled_source_rate_local, dtype=np.float64).ravel()
                    if cpl_arr.size > 0:
                        n = min(int(cpl_arr.size), n_area)
                        cpl_vol = float(np.sum(cpl_arr[:n] * area_model[:n]) * dt_apply)
                        if np.isfinite(cpl_vol):
                            source_budget_model["coupling"] += cpl_vol

            stage_coupled_imex_requested = bool(
                hasattr(self, "source_stage_coupled_imex_rk2_chk")
                and self.source_stage_coupled_imex_rk2_chk.isChecked()
            )
            stage_coupled_imex_enabled = False
            if stage_coupled_imex_requested:
                stage_reasons: List[str] = []
                if coupling_controller is None:
                    stage_reasons.append("no coupling sources configured")
                if temporal_scheme != TemporalScheme.SSP_RK2:
                    stage_reasons.append("temporal scheme is not RK2")
                if not native_source_injection_mode:
                    stage_reasons.append("native source injection unavailable")
                if stage_reasons:
                    self._log(
                        "Stage-coupled IMEX-RK2 requested but disabled: "
                        + "; ".join(stage_reasons)
                    )
                else:
                    stage_coupled_imex_enabled = True
                    self._log("Stage-coupled IMEX-RK2 enabled for external coupling sources.")

            def _rain_source_for_window(t0_s: float, t1_s: float, accumulate: bool, mutate_state: bool) -> object:
                rain_src_local = rain_rate_model
                if thiessen_forcing is not None and not native_rain_cn_forcing:
                    rain_src_si_local, rain_diag_local = thiessen_forcing.step_net_rainfall_mps(
                        t0_s,
                        t1_s,
                        mutate_state=mutate_state,
                    )
                    rain_src_local = self._rain_rate_si_to_model(rain_src_si_local)
                    if accumulate:
                        rain_stats_acc["rain_mm"] += float(rain_diag_local.get("rain_mm_mean", 0.0))
                        rain_stats_acc["excess_mm"] += float(rain_diag_local.get("excess_mm_mean", 0.0))
                        rain_stats_acc["samples"] += 1
                elif native_rain_cn_forcing:
                    rain_src_local = 0.0
                return rain_src_local

            def _cell_source_model_at_time(t_s: float) -> Optional[np.ndarray]:
                src_si = self._internal_flow_source_cms_at_time(internal_flow_forcing, t_s)
                if src_si is None:
                    return None
                return self._flow_si_to_model(src_si)

            while t_accum < run_duration_s:
                if self._cancel_requested:
                    break

                step_wall_t0 = time.perf_counter()
                step_ms = 0.0
                coupling_ms = 0.0
                source_ms = 0.0
                state_ms = 0.0
                bc_ms = 0.0
                ui_ms = 0.0

                if dynamic_bc and not native_bc_forcing:
                    _t_bc0 = time.perf_counter()
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
                    bc_ms += (time.perf_counter() - _t_bc0) * 1000.0

                rain_src = rain_rate_model
                if stage_coupled_imex_enabled:
                    _t_state0 = time.perf_counter()
                    h0_c, hu0_c, hv0_c = backend.get_state()
                    state_ms += (time.perf_counter() - _t_state0) * 1000.0
                    dt_stage_guess = dt_cfg if dt_request <= 0.0 else float(dt_request)
                    cell_source_model_0 = _cell_source_model_at_time(t_accum)
                    _t_cpl0 = time.perf_counter()
                    coupled_source_rate_0 = coupling_controller.compute_source_rates(
                        t_accum,
                        dt_stage_guess,
                        h0_c,
                        hu0_c,
                        hv0_c,
                    )
                    coupling_ms += (time.perf_counter() - _t_cpl0) * 1000.0
                    rain_src_pred = _rain_source_for_window(
                        t_accum,
                        t_accum + dt_stage_guess,
                        accumulate=False,
                        mutate_state=False,
                    )
                    _t_src0 = time.perf_counter()
                    self._apply_external_sources(
                        backend,
                        dt_stage_guess,
                        rain_src_pred,
                        cell_source_model_0,
                        coupled_source_rate_0,
                        prefer_native_injection=native_source_injection_mode,
                    )
                    source_ms += (time.perf_counter() - _t_src0) * 1000.0
                    _t_step0 = time.perf_counter()
                    _diag_predict = backend.step(dt_request)
                    step_ms += (time.perf_counter() - _t_step0) * 1000.0
                    dt_used = float(_diag_predict.get("dt", dt_cfg))
                    _t_state1 = time.perf_counter()
                    h1_c, hu1_c, hv1_c = backend.get_state()
                    state_ms += (time.perf_counter() - _t_state1) * 1000.0
                    _t_cpl1 = time.perf_counter()
                    coupled_source_rate_1 = coupling_controller.compute_source_rates(
                        t_accum + dt_used,
                        dt_used,
                        h1_c,
                        hu1_c,
                        hv1_c,
                    )
                    coupling_ms += (time.perf_counter() - _t_cpl1) * 1000.0
                    coupled_source_rate = 0.5 * (
                        np.asarray(coupled_source_rate_0, dtype=np.float64)
                        + np.asarray(coupled_source_rate_1, dtype=np.float64)
                    )
                    backend.set_state(h0_c, hu0_c, hv0_c)
                    if dynamic_bc and not native_bc_forcing:
                        _t_bc1 = time.perf_counter()
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
                        bc_ms += (time.perf_counter() - _t_bc1) * 1000.0
                    rain_src = _rain_source_for_window(
                        t_accum,
                        t_accum + dt_used,
                        accumulate=True,
                        mutate_state=True,
                    )
                    cell_source_model_1 = _cell_source_model_at_time(t_accum + dt_used)
                    if cell_source_model_0 is None:
                        cell_source_model_stage = cell_source_model_1
                    elif cell_source_model_1 is None:
                        cell_source_model_stage = cell_source_model_0
                    else:
                        cell_source_model_stage = 0.5 * (
                            np.asarray(cell_source_model_0, dtype=np.float64)
                            + np.asarray(cell_source_model_1, dtype=np.float64)
                        )
                    _t_src1 = time.perf_counter()
                    _accumulate_source_volume_model(
                        dt_used,
                        rain_src,
                        cell_source_model_stage,
                        coupled_source_rate,
                    )
                    self._apply_external_sources(
                        backend,
                        dt_used,
                        rain_src,
                        cell_source_model_stage,
                        coupled_source_rate,
                        prefer_native_injection=native_source_injection_mode,
                    )
                    source_ms += (time.perf_counter() - _t_src1) * 1000.0
                    _t_step1 = time.perf_counter()
                    last_diag = backend.step(dt_used)
                    step_ms += (time.perf_counter() - _t_step1) * 1000.0
                else:
                    dt_source_guess = dt_cfg if dt_request <= 0.0 else float(dt_request)
                    cell_source_model_step = _cell_source_model_at_time(t_accum)
                    coupled_source_rate = None
                    if coupling_controller is not None:
                        _t_state2 = time.perf_counter()
                        h_c, hu_c, hv_c = backend.get_state()
                        state_ms += (time.perf_counter() - _t_state2) * 1000.0
                        _t_cpl2 = time.perf_counter()
                        coupled_source_rate = coupling_controller.compute_source_rates(
                            t_accum,
                            dt_source_guess,
                            h_c,
                            hu_c,
                            hv_c,
                        )
                        coupling_ms += (time.perf_counter() - _t_cpl2) * 1000.0
                    rain_src = _rain_source_for_window(
                        t_accum,
                        t_accum + dt_source_guess,
                        accumulate=True,
                        mutate_state=True,
                    )
                    _t_src2 = time.perf_counter()
                    _accumulate_source_volume_model(
                        dt_source_guess,
                        rain_src,
                        cell_source_model_step,
                        coupled_source_rate,
                    )
                    self._apply_external_sources(
                        backend,
                        dt_source_guess,
                        rain_src,
                        cell_source_model_step,
                        coupled_source_rate,
                        prefer_native_injection=native_source_injection_mode,
                    )
                    source_ms += (time.perf_counter() - _t_src2) * 1000.0
                    _t_step2 = time.perf_counter()
                    last_diag = backend.step(dt_request)
                    step_ms += (time.perf_counter() - _t_step2) * 1000.0
                    dt_used = float(last_diag.get("dt", dt_cfg))

                if bc_n0.size > 0:
                    if dynamic_bc:
                        bc_tp_flux, bc_vl_flux = self._apply_timeseries_bc_values(
                            bc_n0,
                            bc_n1,
                            bc_tp,
                            bc_vl,
                            side_hydrographs,
                            t_accum,
                            edge_hydrographs,
                        )
                        bc_vl_flux = self._distribute_total_flow_to_unit_q(
                            bc_n0,
                            bc_n1,
                            bc_tp_flux,
                            bc_vl_flux,
                            bc_tp,
                            side_hydrographs,
                            edge_hydrographs,
                        )
                    else:
                        bc_tp_flux = bc_tp
                        bc_vl_flux = bc_vl
                    _accumulate_boundary_flux_volume_model(dt_used, bc_tp_flux, bc_vl_flux)

                # Preserve most recent synchronized diagnostics so runtime log
                # fields remain meaningful between GPU host-sync intervals.
                step_cmax = float(last_diag.get("max_courant", float("nan")))
                if np.isfinite(step_cmax) and step_cmax >= 0.0:
                    last_valid_cmax = step_cmax
                step_wse_res = float(
                    last_diag.get(
                        "max_depth_residual",
                        last_diag.get("max_wse_elev_error", float("nan")),
                    )
                )
                if np.isfinite(step_wse_res) and step_wse_res >= 0.0:
                    last_valid_wse_res = step_wse_res
                t_accum += dt_used

                # Capture snapshot at each output interval boundary
                need_mesh_snap = t_accum >= _next_snap_t
                need_line_snap = bool(sample_map) and t_accum >= _next_line_snap_t
                need_coupling_snap = (coupling_controller is not None) and (t_accum >= _next_coupling_snap_t)
                if need_mesh_snap or need_line_snap or need_coupling_snap:
                    _t_state3 = time.perf_counter()
                    h_s, hu_s, hv_s = backend.get_state()
                    state_ms += (time.perf_counter() - _t_state3) * 1000.0

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

                if need_coupling_snap:
                    c_rows = self._sample_coupling_object_metrics(coupling_controller, t_accum, h_s)
                    if c_rows:
                        self._coupling_snapshot_rows.extend(c_rows)
                    _next_coupling_snap_t += line_output_interval_s

                _now_wall = time.perf_counter()
                if _now_wall - _last_process_events_wall >= _PROCESS_EVENTS_INTERVAL_S:
                    _t_ui0 = time.perf_counter()
                    QtWidgets.QApplication.processEvents()
                    ui_ms += (time.perf_counter() - _t_ui0) * 1000.0
                    _last_process_events_wall = _now_wall

                step_wall_ms = (time.perf_counter() - step_wall_t0) * 1000.0
                timing_totals_ms["wall"] += step_wall_ms
                timing_totals_ms["step"] += step_ms
                timing_totals_ms["coupling"] += coupling_ms
                timing_totals_ms["source"] += source_ms
                timing_totals_ms["state"] += state_ms
                timing_totals_ms["bc"] += bc_ms
                timing_totals_ms["ui"] += ui_ms
                timing_samples += 1

                pct = int(min(100.0, (t_accum / max(run_duration_s, 1.0e-9)) * 100.0))
                self.progress_bar.setValue(pct)
                i += 1
                if i == 1 or i % 10 == 0 or pct >= 100:
                    max_courant = last_valid_cmax
                    max_wse_res = last_valid_wse_res
                    cmax_txt = f"{max_courant:.5f}" if np.isfinite(max_courant) and max_courant >= 0.0 else "n/a"
                    wse_res_txt = f"{max_wse_res:.6e}" if np.isfinite(max_wse_res) and max_wse_res >= 0.0 else "n/a"
                    rain_diag_txt = ""
                    rain_arr_diag = np.asarray(rain_src, dtype=np.float64)
                    if np.any(rain_arr_diag > 0.0):
                        _t_state4 = time.perf_counter()
                        h_d, hu_d, hv_d = backend.get_state()
                        state_ms += (time.perf_counter() - _t_state4) * 1000.0
                        h_d = np.asarray(h_d, dtype=np.float64)
                        hu_d = np.asarray(hu_d, dtype=np.float64)
                        hv_d = np.asarray(hv_d, dtype=np.float64)
                        wet_mask = h_d > float(self.h_min_spin.value())
                        if np.any(wet_mask):
                            inv_h = 1.0 / np.maximum(h_d[wet_mask], 1.0e-12)
                            speed = np.sqrt((hu_d[wet_mask] * inv_h) ** 2 + (hv_d[wet_mask] * inv_h) ** 2)
                            umax = float(np.max(speed)) if speed.size else 0.0
                            hmin_wet = float(np.min(h_d[wet_mask]))
                            hmax = float(np.max(h_d)) if h_d.size else 0.0
                            rain_diag_txt = (
                                f" rain:umax={umax:.3e} {self._length_unit_name}/s"
                                f" hminWet={hmin_wet:.3e} {self._length_unit_name}"
                                f" hmax={hmax:.3e} {self._length_unit_name}"
                            )
                        else:
                            rain_diag_txt = " rain:all-dry"
                    self._log(
                        (
                        f"step={i} t={t_accum / 3600.0:.3f} hr / {run_duration_s / 3600.0:.3f} hr "
                        f"dt={float(last_diag.get('dt', 0.0)):.5f} "
                        f"gpu={bool(last_diag.get('gpu_active', False))} wet={last_diag.get('wet_cells', '?')} "
                        f"Cmax={cmax_txt} WSEres={wse_res_txt} "
                        f"graph_step={int(last_diag.get('gpu_graph_launches_step', 0))} "
                        f"graph_total={int(last_diag.get('gpu_graph_launches_total', 0))}"
                        f"{rain_diag_txt}"
                        )
                    )
                    if timing_samples > 0:
                        avg_wall = timing_totals_ms["wall"] / timing_samples
                        avg_step = timing_totals_ms["step"] / timing_samples
                        avg_cpl = timing_totals_ms["coupling"] / timing_samples
                        avg_src = timing_totals_ms["source"] / timing_samples
                        avg_state = timing_totals_ms["state"] / timing_samples
                        avg_bc = timing_totals_ms["bc"] / timing_samples
                        avg_ui = timing_totals_ms["ui"] / timing_samples
                        step_gpu_frac = 100.0 * step_ms / max(step_wall_ms, 1.0e-9)
                        avg_gpu_frac = 100.0 * avg_step / max(avg_wall, 1.0e-9)
                        other_ms = max(0.0, step_wall_ms - (step_ms + coupling_ms + source_ms + state_ms + bc_ms + ui_ms))
                        avg_other = max(0.0, avg_wall - (avg_step + avg_cpl + avg_src + avg_state + avg_bc + avg_ui))
                        self._log(
                            "  timing(ms): "
                            f"wall={step_wall_ms:.2f} step={step_ms:.2f} coupling={coupling_ms:.2f} "
                            f"source={source_ms:.2f} state={state_ms:.2f} bc={bc_ms:.2f} ui={ui_ms:.2f} other={other_ms:.2f} "
                            f"gpu_frac={step_gpu_frac:.1f}%"
                        )
                        self._log(
                            "  timing-avg(ms): "
                            f"wall={avg_wall:.2f} step={avg_step:.2f} coupling={avg_cpl:.2f} "
                            f"source={avg_src:.2f} state={avg_state:.2f} bc={avg_bc:.2f} ui={avg_ui:.2f} other={avg_other:.2f} "
                            f"gpu_frac={avg_gpu_frac:.1f}%"
                        )
                    if coupling_controller is not None:
                        cdiag = coupling_controller.last_diag
                        limiter_events = float(cdiag.component_sums.get("drainage_limiter_events", 0.0))
                        limiter_vol_m3 = float(cdiag.component_sums.get("drainage_limiter_volume_m3", 0.0))
                        drain_substeps = float(cdiag.component_sums.get("drainage_substeps_used", 1.0))
                        native_iterative = int(cdiag.component_sums.get("drainage_native_iterative", 0))
                        self._log(
                            "  coupling: "
                            f"drain_qmax={cdiag.drainage_max_link_flow:.4f} cms, "
                            f"drain_hmax={cdiag.drainage_max_node_depth:.4f}, "
                            f"struct_qsum={cdiag.structure_total_flow_cms:.4f} cms, "
                            f"src_range=[{cdiag.source_min_mps:.3e}, {cdiag.source_max_mps:.3e}], "
                            f"drain_substeps={drain_substeps:.0f}, "
                            f"native_iter={native_iterative}, "
                            f"limiter_events={limiter_events:.0f}, "
                            f"limiter_vol={limiter_vol_m3:.6f} m3"
                        )
            h, hu, hv = backend.get_state()
            if native_source_injection_mode:
                try:
                    backend.set_external_sources_native(None)
                except Exception:
                    pass
            self._result_data = {
                "h": h,
                "hu": hu,
                "hv": hv,
                "n_mann_cell": n_mann_cell.copy() if n_mann_cell is not None else np.full(h.shape, float(self.n_mann_spin.value()), dtype=np.float64),
                "gpu_active": np.array(bool(backend.gpu_active())),
                "last_mass_total": np.array(float(last_diag.get("mass_total", -1.0) if last_diag else -1.0)),
            }

            h_end_model = np.asarray(h, dtype=np.float64).ravel()
            n_store_end = min(n_area, int(h_end_model.size))
            storage_end_model = float(np.sum(h_end_model[:n_store_end] * area_model[:n_store_end])) if n_store_end > 0 else 0.0
            storage_delta_model = storage_end_model - storage_start_model
            source_total_model = (
                float(source_budget_model["rain"])
                + float(source_budget_model["cell"])
                + float(source_budget_model["coupling"])
            )
            implied_boundary_out_model = source_total_model - storage_delta_model
            avg_implied_boundary_q_model = implied_boundary_out_model / max(run_duration_s, 1.0e-12)

            vol_unit_label = f"{self._length_unit_name}3"
            vol_to_si = 1.0 / (self._length_scale_si_to_model() ** 3)
            self._log(
                "Mass balance (explicit sources/storage): "
                f"source_total={source_total_model:.6f} {vol_unit_label} "
                f"(rain={source_budget_model['rain']:.6f}, cell={source_budget_model['cell']:.6f}, "
                f"coupling={source_budget_model['coupling']:.6f}), "
                f"dStorage={storage_delta_model:.6f} {vol_unit_label}, "
                f"implied_net_boundary_out={implied_boundary_out_model:.6f} {vol_unit_label} "
                f"(avg={avg_implied_boundary_q_model:.6f} {self._flow_unit_label()})"
            )
            self._log(
                "Mass balance (SI reference): "
                f"source_total={source_total_model * vol_to_si:.6f} m3, "
                f"dStorage={storage_delta_model * vol_to_si:.6f} m3, "
                f"implied_net_boundary_out={implied_boundary_out_model * vol_to_si:.6f} m3"
            )
            if boundary_flux_budget_model:
                self._log("Boundary flux volume by group (from flow-type BC edges):")
                for grp, vol_model in sorted(boundary_flux_budget_model.items(), key=lambda kv: abs(float(kv[1])), reverse=True):
                    avg_q_model = float(vol_model) / max(run_duration_s, 1.0e-12)
                    self._log(
                        f"  {grp}: volume={float(vol_model):.6f} {vol_unit_label}, "
                        f"avg_q={avg_q_model:.6f} {self._flow_unit_label()}"
                    )

            gpkg_results_path = self._current_line_results_storage_path()
            if gpkg_results_path and bool(self.save_line_results_to_gpkg_chk.isChecked()) and self._line_snapshot_rows:
                self._persist_line_results_to_geopackage(
                    gpkg_results_path,
                    run_id,
                    self._line_snapshot_rows,
                    profile_rows=self._line_snapshot_profile_rows,
                    mesh_interval_s=output_interval_s,
                    line_interval_s=line_output_interval_s,
                )
            if gpkg_results_path and bool(self.save_coupling_results_to_gpkg_chk.isChecked()) and self._coupling_snapshot_rows:
                self._persist_coupling_results_to_geopackage(
                    gpkg_results_path,
                    run_id,
                    self._coupling_snapshot_rows,
                    interval_s=line_output_interval_s,
                )
            if gpkg_results_path and bool(self.save_mesh_results_to_gpkg_chk.isChecked()) and self._snapshot_timesteps:
                mesh_rows = self._build_mesh_snapshot_rows()
                if mesh_rows:
                    self._persist_mesh_results_to_geopackage(
                        gpkg_results_path,
                        run_id,
                        mesh_rows,
                        interval_s=output_interval_s,
                    )
            run_wallclock_end = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")
            run_duration_wallclock_s = max(0.0, time.perf_counter() - run_perf_start)
            self._log(f"Run wallclock end: {run_wallclock_end}")
            self._log(f"Run wallclock duration: {run_duration_wallclock_s:.3f} s")
            if gpkg_results_path and bool(self.save_run_log_to_gpkg_chk.isChecked()) and run_id:
                run_log_text = "\n".join(self._runtime_log_lines[run_log_start_idx:])
                self._persist_run_log_to_geopackage(
                    gpkg_results_path,
                    run_id,
                    run_wallclock_start,
                    run_wallclock_end,
                    run_duration_wallclock_s,
                    run_log_text,
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
            if self._coupling_snapshot_rows:
                self._log(f"Coupling rows captured: {len(self._coupling_snapshot_rows)}")
            self._refresh_plot()
        except Exception as exc:
            self._log_exception("Run failed", exc)
            QtWidgets.QMessageBox.critical(
                self,
                "2D SWE",
                "Run failed. Full traceback has been written to the runtime log pane.\n"
                f"Error: {exc}",
            )
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


def launch_swe2d_workbench(parent=None, iface=None):
    if iface is None and parent is not None:
        if hasattr(parent, "_get_qgis_iface") and callable(getattr(parent, "_get_qgis_iface")):
            try:
                iface = parent._get_qgis_iface()
            except Exception:
                iface = None
        if iface is None and hasattr(parent, "iface"):
            try:
                iface = getattr(parent, "iface")
            except Exception:
                iface = None
    if iface is None:
        try:
            import qgis.utils as _qutils

            iface = getattr(_qutils, "iface", None)
        except Exception:
            iface = None

    dlg = SWE2DWorkbenchDialog(parent, iface=iface)

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
