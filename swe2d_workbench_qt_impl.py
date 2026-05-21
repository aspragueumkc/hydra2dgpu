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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

try:
    from qgis.PyQt import uic as _qgis_uic
except Exception:
    _qgis_uic = None

from swe2d.workbench.seam_imports import (
    SWE2DBackendInitializer,
    SWE2DNativeBoundaryHydrographConfigurator,
    SWE2DRunController,
    SWE2DRunDataBuilder,
    SWE2DRunFinalizer,
    SWE2DRunLifecycle,
    SWE2DRunOptionsBuilder,
    SWE2DRunOrchestrator,
    SWE2DRunRequest,
    SWE2DRunSetupConfigurator,
    SWE2DRuntimeReporter,
    SWE2DRuntimeSourceManager,
    SWE2DRuntimeStepExecutor,
    SWE2DWorkbenchViewAdapter,
    wire_startup_run_components,
)

from swe2d.workbench.startup_bootstrap import bootstrap_startup_run_components
from swe2d.workbench.startup_state import initialize_workbench_startup_state
from swe2d.workbench.post_init import run_workbench_post_bootstrap_setup
from swe2d.workbench.results_bridge import (
    get_velocity_vector_builder as _get_velocity_vector_builder_bridge,
    maybe_create_results_panel as _maybe_create_results_panel_bridge,
)
from swe2d.workbench.high_perf_overlay_bridge import (
    destroy_high_perf_canvas_overlay_item as _destroy_high_perf_canvas_overlay_item_bridge,
    sync_high_perf_overlay_data as _sync_high_perf_overlay_data_bridge,
    update_high_perf_overlay_time as _update_high_perf_overlay_time_bridge,
)

try:
    from swe2d.extensions.patch_observer import SWE2DThreeDPatchObserver
except Exception:
    try:
        from .swe2d.extensions.patch_observer import SWE2DThreeDPatchObserver
    except Exception:
        SWE2DThreeDPatchObserver = None

from swe2d.boundary_and_forcing.bc_logic import (
    apply_timeseries_bc_values as _apply_timeseries_bc_values_logic,
    distribute_total_flow_to_unit_q as _distribute_total_flow_to_unit_q_logic,
    interp_hydrograph as _interp_hydrograph_logic,
)

from swe2d.boundary_and_forcing.runtime_source_logic import (
    apply_external_sources as _apply_external_sources_logic,
    internal_flow_source_cms_at_time as _internal_flow_source_cms_at_time_logic,
)

from swe2d.boundary_and_forcing.hydrograph_logic import (
    hydrograph_from_layer as _hydrograph_from_layer_logic,
    parse_hydrograph_text as _parse_hydrograph_text_logic,
    parse_time_hours as _parse_time_hours_logic,
)

try:
    from swe2d.boundary_and_forcing.internal_flow_qgis_adapter import (
        build_internal_flow_forcing_qgis as _build_internal_flow_forcing_qgis_logic,
    )
except Exception:
    _build_internal_flow_forcing_qgis_logic = None

from swe2d.mesh.mesh_runtime_logic import (
    boundary_buffer_cells as _boundary_buffer_cells_logic,
    inflow_adjacent_cells as _inflow_adjacent_cells_logic,
    initial_state as _initial_state_logic,
    mesh_cell_areas as _mesh_cell_areas_logic,
    mesh_cell_centroids as _mesh_cell_centroids_logic,
    mesh_cell_min_bed as _mesh_cell_min_bed_logic,
)

from swe2d.boundary_and_forcing.spatial_forcing_qgis_adapter import (
    build_spatial_cn_array_qgis as _build_spatial_cn_array_qgis_logic,
    build_spatial_manning_array_qgis as _build_spatial_manning_array_qgis_logic,
    build_thiessen_rain_cn_forcing_qgis as _build_thiessen_rain_cn_forcing_qgis_logic,
)

from swe2d.boundary_and_forcing.boundary_runtime_logic import (
    collect_boundary_arrays as _collect_boundary_arrays_logic,
    mesh_boundary_edges as _mesh_boundary_edges_logic,
)

from swe2d.boundary_and_forcing.boundary_qgis_adapter import (
    apply_bc_layer_overrides_qgis as _apply_bc_layer_overrides_qgis_logic,
    collect_bc_layer_edge_groups_qgis as _collect_bc_layer_edge_groups_qgis_logic,
    collect_bc_layer_hydrographs_qgis as _collect_bc_layer_hydrographs_qgis_logic,
)

from swe2d.extensions.patch_runtime_logic import (
    collect_3d_patch_env_overrides as _collect_3d_patch_env_overrides_logic,
    parse_optional_float_text as _parse_optional_float_text_logic,
)

from swe2d.extensions.patch_qgis_adapter import (
    sample_terrain_min_z_for_roi_qgis as _sample_terrain_min_z_for_roi_qgis_logic,
)

try:
    from swe2d_run_log_storage import (
        load_run_logs_from_geopackage as _load_run_logs_from_geopackage_logic,
        persist_run_log_to_geopackage as _persist_run_log_to_geopackage_logic,
    )
except Exception:
    from .swe2d_run_log_storage import (
        load_run_logs_from_geopackage as _load_run_logs_from_geopackage_logic,
        persist_run_log_to_geopackage as _persist_run_log_to_geopackage_logic,
    )

from swe2d.workbench.non_gui_runtime import (
    boundary_edge_owner_cells as _boundary_edge_owner_cells_runtime_logic,
    build_experimental_3d_interface_contract_arrays as _build_experimental_3d_interface_contract_arrays_runtime_logic,
    build_patch_spec_from_stats as _build_patch_spec_from_stats_runtime_logic,
    build_mesh_snapshot_rows as _build_mesh_snapshot_rows_logic,
    execute_run_timestep_loop as _execute_run_timestep_loop_runtime_logic,
    initialize_experimental_3d_patch_state as _initialize_experimental_3d_patch_state_runtime_logic,
    parse_obj_scale_value as _parse_obj_scale_value_runtime_logic,
    run_experimental_3d_obj_method_probe as _run_experimental_3d_obj_method_probe_runtime_logic,
    resolve_obj_model_path as _resolve_obj_model_path_runtime_logic,
    upload_experimental_3d_obj_geometry as _upload_experimental_3d_obj_geometry_runtime_logic,
    upload_experimental_3d_interface_contract as _upload_experimental_3d_interface_contract_runtime_logic,
)

from swe2d.workbench.non_gui_qgis import (
    build_patch_terrain_surface as _build_patch_terrain_surface_qgis_logic,
    infer_obj_path_from_layer_3d_renderer as _infer_obj_path_from_layer_3d_renderer_qgis_logic,
    parse_feature_float as _parse_feature_float_qgis_logic,
    resolve_layer_field_name as _resolve_layer_field_name_qgis_logic,
)

from swe2d.workbench.three_d_bc import (
    apply_3d_patch_face_bc_to_backend as _apply_3d_patch_face_bc_to_backend_logic,
    collect_3d_patch_env_overrides as _collect_3d_patch_env_overrides_delegate_logic,
    collect_3d_patch_face_bc_env_overrides as _collect_3d_patch_face_bc_env_overrides_logic,
    sync_experimental_3d_mode_widgets as _sync_experimental_3d_mode_widgets_logic,
    summarize_3d_patch_face_bc_modes as _summarize_3d_patch_face_bc_modes_logic,
)

try:
    from qgis.core import (
        QgsEditorWidgetSetup,
        QgsFieldConstraints,
        QgsFeature,
        QgsField,
        QgsGeometry,
        QgsMeshLayer,
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
    QgsMeshLayer = None
    QgsRasterLayer = QgsVectorLayer = QgsWkbTypes = None
    QgsUnitTypes = QgsVectorFileWriter = None
    QVariant = None
    _HAVE_QGIS_CORE = False

try:
    from swe2d.runtime.backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
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
    from swe3d_geometry_ingest import (
        PatchGridSpec,
        apply_instance_transform,
        build_static_geometry_tensors,
        load_obj_mesh,
        write_solid_voxels_obj,
        write_fluid_voxels_obj,
    )
except Exception:
    try:
        from .swe3d_geometry_ingest import (
            PatchGridSpec,
            apply_instance_transform,
            build_static_geometry_tensors,
            load_obj_mesh,
            write_solid_voxels_obj,
            write_fluid_voxels_obj,
        )
    except Exception:
        PatchGridSpec = None
        apply_instance_transform = None
        build_static_geometry_tensors = None
        load_obj_mesh = None
        write_solid_voxels_obj = None
        write_fluid_voxels_obj = None

# Import boundary/forcing/drainage/structures modules
from swe2d.runtime.coupling import SWE2DCouplingController, pack_coupling_soa
from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
from swe2d.extensions.extension_models import (
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
    SWE2DEquationSet,
    SWE2DThreeDCouplingMode,
    SWE2DThreeDSolverModel,
    StructureType,
    SpatialDiscretization,
    SolverModelOptions,
    TemporalScheme,
)
from swe2d.extensions.structures import SWE2DStructureModule




def _recover_optional_solver_imports() -> None:
    """Recover optional runtime imports independently after broad fallback paths.

    Some plugin environments can fail one import (for example coupling modules)
    while still having working solver enum bindings. Recovering these imports
    prevents silent runtime downgrades (e.g. requested 3D run falling back to 2D).
    """

    global SWE2DCouplingController
    global pack_coupling_soa
    global SWE2DUrbanDrainageModule
    global SWE2DStructureModule
    global DrainageSolverMode
    global DrainageLink
    global DrainageNode
    global GodunovSolverMode
    global HydraulicStructure
    global HydraulicStructureConfig
    global InletExchange
    global InletType
    global NodeInletAssignment
    global OutfallExchange
    global PipeEndExchange
    global PipeNetworkConfig
    global SWE2DEquationSet
    global SWE2DThreeDCouplingMode
    global SWE2DThreeDSolverModel
    global StructureType
    global SpatialDiscretization
    global SolverModelOptions
    global TemporalScheme

    if SWE2DCouplingController is None or pack_coupling_soa is None:
        try:
            from swe2d.runtime.coupling import SWE2DCouplingController as _Ctrl, pack_coupling_soa as _Pack
            SWE2DCouplingController = SWE2DCouplingController or _Ctrl
            pack_coupling_soa = pack_coupling_soa or _Pack
        except Exception:
            try:
                from .swe2d_coupling import SWE2DCouplingController as _Ctrl, pack_coupling_soa as _Pack
                SWE2DCouplingController = SWE2DCouplingController or _Ctrl
                pack_coupling_soa = pack_coupling_soa or _Pack
            except Exception:
                pass

    if SWE2DUrbanDrainageModule is None:
        try:
            from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule as _Drain
            SWE2DUrbanDrainageModule = SWE2DUrbanDrainageModule or _Drain
        except Exception:
            try:
                from .swe2d_drainage_network import SWE2DUrbanDrainageModule as _Drain
                SWE2DUrbanDrainageModule = SWE2DUrbanDrainageModule or _Drain
            except Exception:
                pass

    if SWE2DStructureModule is None:
        try:
            from swe2d.extensions.structures import SWE2DStructureModule as _StructMod
            SWE2DStructureModule = SWE2DStructureModule or _StructMod
        except Exception:
            try:
                from .swe2d_structures import SWE2DStructureModule as _StructMod
                SWE2DStructureModule = SWE2DStructureModule or _StructMod
            except Exception:
                pass

    need_ext = any(
        v is None
        for v in (
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
            SWE2DEquationSet,
            SWE2DThreeDCouplingMode,
            SWE2DThreeDSolverModel,
            StructureType,
            SpatialDiscretization,
            SolverModelOptions,
            TemporalScheme,
        )
    )
    if not need_ext:
        return

    ext = None
    try:
        import swe2d_extensions as ext
    except Exception:
        try:
            from . import swe2d_extensions as ext
        except Exception:
            ext = None
    if ext is None:
        return

    DrainageSolverMode = DrainageSolverMode or getattr(ext, "DrainageSolverMode", None)
    DrainageLink = DrainageLink or getattr(ext, "DrainageLink", None)
    DrainageNode = DrainageNode or getattr(ext, "DrainageNode", None)
    GodunovSolverMode = GodunovSolverMode or getattr(ext, "GodunovSolverMode", None)
    HydraulicStructure = HydraulicStructure or getattr(ext, "HydraulicStructure", None)
    HydraulicStructureConfig = HydraulicStructureConfig or getattr(ext, "HydraulicStructureConfig", None)
    InletExchange = InletExchange or getattr(ext, "InletExchange", None)
    InletType = InletType or getattr(ext, "InletType", None)
    NodeInletAssignment = NodeInletAssignment or getattr(ext, "NodeInletAssignment", None)
    OutfallExchange = OutfallExchange or getattr(ext, "OutfallExchange", None)
    PipeEndExchange = PipeEndExchange or getattr(ext, "PipeEndExchange", None)
    PipeNetworkConfig = PipeNetworkConfig or getattr(ext, "PipeNetworkConfig", None)
    SWE2DEquationSet = SWE2DEquationSet or getattr(ext, "SWE2DEquationSet", None)
    SWE2DThreeDCouplingMode = SWE2DThreeDCouplingMode or getattr(ext, "SWE2DThreeDCouplingMode", None)
    SWE2DThreeDSolverModel = SWE2DThreeDSolverModel or getattr(ext, "SWE2DThreeDSolverModel", None)
    StructureType = StructureType or getattr(ext, "StructureType", None)
    SpatialDiscretization = SpatialDiscretization or getattr(ext, "SpatialDiscretization", None)
    SolverModelOptions = SolverModelOptions or getattr(ext, "SolverModelOptions", None)
    TemporalScheme = TemporalScheme or getattr(ext, "TemporalScheme", None)


_recover_optional_solver_imports()

try:
    import h5py as _h5py
    _HAVE_H5PY = True
except ImportError:
    _h5py = None
    _HAVE_H5PY = False

_netCDF4 = None
_HAVE_NETCDF4 = False
_NETCDF4_IMPORT_ERROR = None


def _ensure_netcdf4_available() -> bool:
    global _netCDF4, _HAVE_NETCDF4, _NETCDF4_IMPORT_ERROR
    if _HAVE_NETCDF4:
        return True
    if _NETCDF4_IMPORT_ERROR is not None:
        return False
    try:
        import netCDF4 as _netcdf4_mod
        _netCDF4 = _netcdf4_mod
        _HAVE_NETCDF4 = True
        _NETCDF4_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _netCDF4 = None
        _HAVE_NETCDF4 = False
        _NETCDF4_IMPORT_ERROR = exc
        return False

try:
    from swe2d.mesh.meshing import conceptual_from_qgis_layers, generate_face_centric_mesh, _gmsh_available, _tqmesh_available
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
    ("WENO3-like (GPU experimental)",    5),
]

_TEMPORAL_ORDER_OPTIONS = [
    ("Euler (RK1, 1st-order)",           1),
    ("RK2 (Heun, 2nd-order, default)",   2),
    ("RK4 (classic, 4th-order)",         4),
    ("Graph-safe RK4 (true staged)",     5),
    ("Graph-safe RK5 (Cash-Karp)",       6),
]

_SWE3D_PATCH_FACES = (
    "XMIN",
    "XMAX",
    "YMIN",
    "YMAX",
    "ZMIN",
    "ZMAX",
)

_SWE3D_BC_MODE_OPTIONS = [
    ("Wall", 0),
    ("Inflow (U/V/W)", 1),
    ("Volumetric Inlet (Q)", 4),
    ("Outflow (zero-gradient)", 2),
    ("Free Surface", 3),
]

_SWE3D_BC_FIELD_DEFAULTS = {
    "q": 0.0,
    "u": 0.0,
    "v": 0.0,
    "w": 0.0,
    "vof": 1.0,
    "p": 0.0,
}

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
_SWE2D_WORKBENCH_DOCK = None
_SWE2D_WORKBENCH_DESIGNER_WINDOWS = []
_SWE2D_WORKBENCH_DESIGNER_DOCK = None
_SWE2D_WORKBENCH_STUDIO_WINDOWS = []
_SWE2D_WORKBENCH_STUDIO_DOCK = None
_SWE2D_WORKBENCH_SCENARIO_WINDOWS = []
_SWE2D_WORKBENCH_SCENARIO_DOCK = None
_SWE2D_STUDIO_HOST_TOOLBAR = None
_SWE2D_STUDIO_HOST_MENU = None
_SWE2D_STUDIO_COMPONENT_DOCKS = {}
_SWE2D_STUDIO_HOST_DIALOG = None

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
            from swe2d.mesh.meshing import generate_face_centric_mesh as gen  # type: ignore
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


class SWE2DDetachedRuntimeLogDialog(QtWidgets.QDialog):
    def __init__(self, initial_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Runtime Log")
        self.resize(920, 620)
        root = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(str(initial_text or ""))
        root.addWidget(self.text, stretch=1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    def append_text(self, msg: str) -> None:
        self.text.appendPlainText(str(msg))

    def set_text(self, text: str) -> None:
        self.text.setPlainText(str(text or ""))


class SWE2DDetachedMeshViewDialog(QtWidgets.QDialog):
    def __init__(self, render_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Mesh View")
        self.resize(980, 720)
        self._render_callback = render_callback

        root = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("View:"))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItem("Mesh", "mesh")
        self.view_mode_combo.addItem("Depth", "depth")
        self.view_mode_combo.addItem("Velocity magnitude", "velocity")
        header.addWidget(self.view_mode_combo)
        header.addStretch(1)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        FigureCanvas, Figure, mtri = _try_import_matplotlib_qt()
        self._have_mpl = FigureCanvas is not None and Figure is not None and mtri is not None
        self._fig = Figure(figsize=(6.4, 4.2), tight_layout=True) if self._have_mpl else None
        self._canvas = FigureCanvas(self._fig) if self._have_mpl else None
        if self._canvas is not None:
            root.addWidget(self._canvas, stretch=1)
        else:
            note = QtWidgets.QLabel("Matplotlib Qt backend not available; mesh view cannot be rendered in a separate window.")
            note.setWordWrap(True)
            root.addWidget(note, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh_btn.clicked.connect(self.refresh_view)
        self.view_mode_combo.currentIndexChanged.connect(self.refresh_view)
        self.refresh_view()

    def refresh_view(self) -> None:
        if not self._have_mpl or self._fig is None or self._canvas is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        mode = str(self.view_mode_combo.currentData() or "mesh")
        if callable(self._render_callback):
            try:
                self._render_callback(ax, mode)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Render failed: {exc}", ha="center", va="center", transform=ax.transAxes)
        self._canvas.draw_idle()


class SWE2DDetachedPanelDialog(QtWidgets.QDialog):
    """Generic detachable container with automatic reattach callback."""

    def __init__(self, title: str, content_widget: QtWidgets.QWidget, on_reattach=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Detached Panel"))
        self.resize(760, 620)
        self._on_reattach = on_reattach
        self._content_widget = content_widget
        self._reattached = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        if self._content_widget is not None:
            root.addWidget(self._content_widget, stretch=1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self._reattach_btn = btns.addButton("Reattach", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self._reattach_btn.clicked.connect(self._reattach_and_close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)

    def _reattach_and_close(self) -> None:
        self._reattach_once()
        self.close()

    def _reattach_once(self) -> None:
        if self._reattached:
            return
        self._reattached = True
        if callable(self._on_reattach):
            try:
                self._on_reattach()
            except Exception:
                pass

    def closeEvent(self, event):
        self._reattach_once()
        super().closeEvent(event)


class SWE3DPatchViewerDialog(QtWidgets.QDialog):
    """Quick-look viewer for stored 3D patch snapshots (VoF-focused MVP)."""

    def __init__(self, snapshots: List[Dict[str, object]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Patch Viewer (Experimental)")
        self.resize(940, 680)

        self._snapshots = sorted(
            [dict(s) for s in snapshots if isinstance(s, dict)],
            key=lambda s: float(s.get("t_s", 0.0) or 0.0),
        )

        root = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            "Experimental 3D patch QA view. "
            "Displays stored VoF snapshots and simple derived fields."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Snapshot:"))
        self.snapshot_combo = QtWidgets.QComboBox()
        controls.addWidget(self.snapshot_combo)
        controls.addWidget(QtWidgets.QLabel("Field:"))
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.addItem("VoF slice (XY)", "vof_slice")
        self.field_combo.addItem("Column fill depth", "column_depth")
        self.field_combo.addItem("Column fill fraction", "column_fraction")
        controls.addWidget(self.field_combo)
        controls.addWidget(QtWidgets.QLabel("Z index:"))
        self.z_spin = QtWidgets.QSpinBox()
        self.z_spin.setRange(0, 0)
        controls.addWidget(self.z_spin)
        controls.addStretch(1)
        root.addLayout(controls)

        self.stats_lbl = QtWidgets.QLabel("")
        self.stats_lbl.setWordWrap(True)
        root.addWidget(self.stats_lbl)

        self._have_mpl = False
        self._plot_fig = None
        self._plot_canvas = None
        FigureCanvas, Figure, _ = _try_import_matplotlib_qt()
        if FigureCanvas is not None and Figure is not None:
            self._have_mpl = True
            self._plot_fig = Figure(figsize=(7.8, 4.6), tight_layout=True)
            self._plot_canvas = FigureCanvas(self._plot_fig)
            root.addWidget(self._plot_canvas, stretch=1)
        else:
            note = QtWidgets.QLabel(
                "Matplotlib Qt backend unavailable; numeric summary only."
            )
            note.setWordWrap(True)
            root.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_snapshot_combo()
        self.snapshot_combo.currentIndexChanged.connect(self._refresh_controls)
        self.snapshot_combo.currentIndexChanged.connect(self._refresh_view)
        self.field_combo.currentIndexChanged.connect(self._refresh_controls)
        self.field_combo.currentIndexChanged.connect(self._refresh_view)
        self.z_spin.valueChanged.connect(self._refresh_view)

        self._refresh_controls()
        self._refresh_view()

    def _populate_snapshot_combo(self):
        self.snapshot_combo.clear()
        for i, snap in enumerate(self._snapshots):
            t_s = float(snap.get("t_s", 0.0) or 0.0)
            stats = dict(snap.get("stats", {}) or {})
            nx = int(stats.get("nx", 0) or 0)
            ny = int(stats.get("ny", 0) or 0)
            nz = int(stats.get("nz", 0) or 0)
            label = f"t={t_s/3600.0:.4f} hr (nx={nx}, ny={ny}, nz={nz})"
            self.snapshot_combo.addItem(label, i)

    def _current_snapshot(self) -> Optional[Dict[str, object]]:
        idx = self.snapshot_combo.currentData()
        if idx is None:
            idx = self.snapshot_combo.currentIndex()
        try:
            i = int(idx)
        except Exception:
            return None
        if i < 0 or i >= len(self._snapshots):
            return None
        return self._snapshots[i]

    def _reshape_vof(self, snap: Dict[str, object]) -> Tuple[Optional[np.ndarray], int, int, int, float]:
        stats = dict(snap.get("stats", {}) or {})
        vof = np.asarray(snap.get("vof", np.empty(0, dtype=np.float64)), dtype=np.float64).ravel()
        nx = max(0, int(stats.get("nx", 0) or 0))
        ny = max(0, int(stats.get("ny", 0) or 0))
        nz = max(0, int(stats.get("nz", 0) or 0))
        dz = float(stats.get("dz", 0.0) or 0.0)
        n_exp = nx * ny * nz
        if n_exp <= 0 or vof.size != n_exp:
            return None, nx, ny, nz, dz
        return vof.reshape((nz, ny, nx)), nx, ny, nz, dz

    def _refresh_controls(self):
        snap = self._current_snapshot()
        if snap is None:
            self.z_spin.setRange(0, 0)
            self.z_spin.setEnabled(False)
            return
        stats = dict(snap.get("stats", {}) or {})
        nz = max(1, int(stats.get("nz", 1) or 1))
        self.z_spin.setRange(0, max(0, nz - 1))
        want_slice = str(self.field_combo.currentData() or "") == "vof_slice"
        self.z_spin.setEnabled(bool(want_slice))

    def _refresh_view(self):
        snap = self._current_snapshot()
        if snap is None:
            self.stats_lbl.setText("No snapshot selected.")
            return

        arr3d, nx, ny, nz, dz = self._reshape_vof(snap)
        if arr3d is None:
            self.stats_lbl.setText(
                "Snapshot data is incomplete. "
                f"Expected nx*ny*nz={max(0, nx*ny*nz)} cells but data shape does not match."
            )
            return

        field = str(self.field_combo.currentData() or "vof_slice")
        if field == "column_depth":
            arr = np.sum(np.clip(arr3d, 0.0, 1.0), axis=0) * max(0.0, float(dz))
            title = "Column Fill Depth"
            cmap = "viridis"
            vmin = None
            vmax = None
        elif field == "column_fraction":
            arr = np.mean(np.clip(arr3d, 0.0, 1.0), axis=0)
            title = "Column Fill Fraction"
            cmap = "magma"
            vmin = 0.0
            vmax = 1.0
        else:
            z_idx = int(np.clip(self.z_spin.value(), 0, max(0, nz - 1)))
            arr = arr3d[z_idx, :, :]
            title = f"VoF Slice (z={z_idx}/{max(0, nz - 1)})"
            cmap = "cividis"
            vmin = 0.0
            vmax = 1.0

        t_s = float(snap.get("t_s", 0.0) or 0.0)
        txt = (
            f"t={t_s/3600.0:.4f} hr | "
            f"nx={nx}, ny={ny}, nz={nz}, dz={dz:.6g} | "
            f"min={float(np.nanmin(arr)):.6e}, max={float(np.nanmax(arr)):.6e}, "
            f"mean={float(np.nanmean(arr)):.6e}"
        )
        self.stats_lbl.setText(txt)

        if not self._have_mpl or self._plot_fig is None or self._plot_canvas is None:
            return

        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        im = ax.imshow(
            np.asarray(arr, dtype=np.float64),
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        self._plot_fig.colorbar(im, ax=ax)
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        ax.set_title(title)
        self._plot_canvas.draw_idle()


class SWE2DWorkbenchDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Workbench")
        self.resize(1160, 760)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self._iface = iface
        initialize_workbench_startup_state(
            self,
            qtcore_module=QtCore,
            concurrent_futures_module=concurrent.futures,
            try_import_matplotlib_qt=_try_import_matplotlib_qt,
        )

        self._build_ui()
        bootstrap_startup_run_components(
            self,
            wire_startup_run_components,
            view_adapter=SWE2DWorkbenchViewAdapter,
            run_orchestrator=SWE2DRunOrchestrator,
            run_request=SWE2DRunRequest,
            run_controller=SWE2DRunController,
            run_data_builder=SWE2DRunDataBuilder,
            run_options_builder=SWE2DRunOptionsBuilder,
            backend_initializer=SWE2DBackendInitializer,
            run_finalizer=SWE2DRunFinalizer,
            run_lifecycle=SWE2DRunLifecycle,
            swe2d_gpu_available=swe2d_gpu_available,
            temporal_scheme=TemporalScheme,
            spatial_discretization=SpatialDiscretization,
            godunov_solver_mode=GodunovSolverMode,
            solver_model_options=SolverModelOptions,
            swe2d_equation_set=SWE2DEquationSet,
            swe2d_3d_solver_model=SWE2DThreeDSolverModel,
            swe2d_3d_coupling_mode=SWE2DThreeDCouplingMode,
        )
        run_workbench_post_bootstrap_setup(
            self,
            swe2d_available_fn=swe2d_available,
            swe2d_gpu_available_fn=swe2d_gpu_available,
            gmsh_available_fn=_gmsh_available,
        )

    def _note_startup_component_missing(self, name: str, required_for_run: bool = False):
        self._log(f"Startup seam unavailable ({name}): import failed.")
        if required_for_run:
            self._startup_run_component_errors.append(name)

    def _init_startup_component(self, name: str, builder: Callable[[], object], required_for_run: bool = False):
        try:
            return builder()
        except Exception as exc:
            self._log(f"Startup seam unavailable ({name}): {exc}")
            if required_for_run:
                self._startup_run_component_errors.append(name)
            return None

    def _require_run_components(self, components: Sequence[Tuple[str, str]], context_label: str) -> bool:
        missing: List[str] = []
        for attr_name, label in components:
            if getattr(self, attr_name, None) is None:
                missing.append(label)
        if missing:
            self._log(
                f"{context_label} aborted: required run seams unavailable: "
                + ", ".join(missing)
            )
            return False
        return True

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

    def _forms_file_path(self, file_name: str) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "forms", str(file_name))

    def _build_mesh_tab_page(self) -> QtWidgets.QWidget:
        ui_path = self._forms_file_path("swe2d_mesh_tab.ui")
        mesh_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                mesh_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                mesh_tab_page = None
        if mesh_tab_page is None:
            mesh_tab_page = self._build_mesh_tab_page_fallback()
        self._bind_mesh_tab_controls(mesh_tab_page)
        return mesh_tab_page

    def _build_mesh_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        mesh_group = QtWidgets.QGroupBox("Mesh Generation")
        mesh_form = QtWidgets.QFormLayout(mesh_group)

        nx_spin = QtWidgets.QSpinBox()
        nx_spin.setObjectName("nx_spin")
        ny_spin = QtWidgets.QSpinBox()
        ny_spin.setObjectName("ny_spin")
        lx_spin = QtWidgets.QDoubleSpinBox()
        lx_spin.setObjectName("lx_spin")
        ly_spin = QtWidgets.QDoubleSpinBox()
        ly_spin.setObjectName("ly_spin")
        bed_amp_spin = QtWidgets.QDoubleSpinBox()
        bed_amp_spin.setObjectName("bed_amp_spin")
        mesh_layout_combo = QtWidgets.QComboBox()
        mesh_layout_combo.setObjectName("mesh_layout_combo")
        generate_mesh_btn = QtWidgets.QPushButton("Generate Mesh")
        generate_mesh_btn.setObjectName("generate_mesh_btn")
        mesh_info_lbl = QtWidgets.QLabel("Mesh not generated")
        mesh_info_lbl.setObjectName("mesh_info_lbl")

        mesh_form.addRow("Cells in X:", nx_spin)
        mesh_form.addRow("Cells in Y:", ny_spin)
        mesh_form.addRow("Length X:", lx_spin)
        mesh_form.addRow("Length Y:", ly_spin)
        mesh_form.addRow("Bed perturbation amplitude:", bed_amp_spin)
        mesh_form.addRow("Structured layout:", mesh_layout_combo)
        mesh_form.addRow(generate_mesh_btn)
        mesh_form.addRow(mesh_info_lbl)

        root_layout.addWidget(mesh_group)
        return root

    def _bind_mesh_tab_controls(self, mesh_tab_page: QtWidgets.QWidget) -> None:
        self.nx_spin = mesh_tab_page.findChild(QtWidgets.QSpinBox, "nx_spin")
        self.ny_spin = mesh_tab_page.findChild(QtWidgets.QSpinBox, "ny_spin")
        self.lx_spin = mesh_tab_page.findChild(QtWidgets.QDoubleSpinBox, "lx_spin")
        self.ly_spin = mesh_tab_page.findChild(QtWidgets.QDoubleSpinBox, "ly_spin")
        self.bed_amp_spin = mesh_tab_page.findChild(QtWidgets.QDoubleSpinBox, "bed_amp_spin")
        self.mesh_layout_combo = mesh_tab_page.findChild(QtWidgets.QComboBox, "mesh_layout_combo")
        self.generate_mesh_btn = mesh_tab_page.findChild(QtWidgets.QPushButton, "generate_mesh_btn")
        self.mesh_info_lbl = mesh_tab_page.findChild(QtWidgets.QLabel, "mesh_info_lbl")

        missing = []
        if self.nx_spin is None:
            missing.append("nx_spin")
        if self.ny_spin is None:
            missing.append("ny_spin")
        if self.lx_spin is None:
            missing.append("lx_spin")
        if self.ly_spin is None:
            missing.append("ly_spin")
        if self.bed_amp_spin is None:
            missing.append("bed_amp_spin")
        if self.mesh_layout_combo is None:
            missing.append("mesh_layout_combo")
        if self.generate_mesh_btn is None:
            missing.append("generate_mesh_btn")
        if self.mesh_info_lbl is None:
            missing.append("mesh_info_lbl")
        if missing:
            raise RuntimeError(f"Mesh tab UI missing controls: {', '.join(missing)}")

        self.nx_spin.setRange(2, 400)
        self.nx_spin.setValue(24)
        self.ny_spin.setRange(2, 400)
        self.ny_spin.setValue(14)

        self.lx_spin.setRange(1.0, 1.0e6)
        self.lx_spin.setDecimals(2)
        self.lx_spin.setValue(240.0)

        self.ly_spin.setRange(1.0, 1.0e6)
        self.ly_spin.setDecimals(2)
        self.ly_spin.setValue(120.0)

        self.bed_amp_spin.setRange(0.0, 1.0e6)
        self.bed_amp_spin.setDecimals(3)
        self.bed_amp_spin.setValue(0.0)

        self.mesh_layout_combo.clear()
        self.mesh_layout_combo.addItem("Split triangles (2 cells / block)", "tri")
        self.mesh_layout_combo.addItem("Structured block quads (1 cell / block, faster)", "quad")
        self.mesh_layout_combo.setCurrentIndex(1)
        self.mesh_layout_combo.setToolTip(
            "Select generated cell layout.\n"
            "Structured block quads usually run faster for rectilinear domains."
        )

        self.mesh_info_lbl.setWordWrap(True)
        if not str(self.mesh_info_lbl.text() or "").strip():
            self.mesh_info_lbl.setText("Mesh not generated")

        self.generate_mesh_btn.clicked.connect(self._on_generate_mesh)

    def _build_boundary_tab_page(self) -> QtWidgets.QWidget:
        ui_path = self._forms_file_path("swe2d_boundary_tab.ui")
        boundary_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                boundary_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                boundary_tab_page = None
        if boundary_tab_page is None:
            boundary_tab_page = self._build_boundary_tab_page_fallback()

        bc_grid = boundary_tab_page.findChild(QtWidgets.QGridLayout, "bc_grid")
        if bc_grid is None:
            raise RuntimeError("Boundary tab UI missing bc_grid layout")
        self._populate_boundary_tab_controls(bc_grid)
        return boundary_tab_page

    def _build_boundary_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        bc_group = QtWidgets.QGroupBox(
            "Boundary Conditions (side defaults + optional BC polyline overrides)"
        )
        bc_grid = QtWidgets.QGridLayout(bc_group)
        bc_grid.setObjectName("bc_grid")

        bc_grid.addWidget(QtWidgets.QLabel("Side"), 0, 0)
        bc_grid.addWidget(QtWidgets.QLabel("Type"), 0, 1)
        bc_grid.addWidget(QtWidgets.QLabel("Value (Q_total for flow)"), 0, 2)
        bc_grid.addWidget(QtWidgets.QLabel("Hydrograph (hr,Q_total; hr,Q_total)"), 0, 3)
        bc_grid.addWidget(QtWidgets.QLabel("Editor"), 0, 4)
        bc_grid.addWidget(QtWidgets.QLabel("BC polyline layer override:"), 5, 0)

        root_layout.addWidget(bc_group)
        return root

    def _populate_boundary_tab_controls(self, bc_grid: QtWidgets.QGridLayout) -> None:
        scope = bc_grid.parentWidget()

        def _find_or_create_combo(name: str) -> QtWidgets.QComboBox:
            w = scope.findChild(QtWidgets.QComboBox, name) if scope is not None else None
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            return w

        def _find_or_create_double_spin(name: str) -> QtWidgets.QDoubleSpinBox:
            w = scope.findChild(QtWidgets.QDoubleSpinBox, name) if scope is not None else None
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            return w

        def _find_or_create_line_edit(name: str) -> QtWidgets.QLineEdit:
            w = scope.findChild(QtWidgets.QLineEdit, name) if scope is not None else None
            if w is None:
                w = QtWidgets.QLineEdit()
                w.setObjectName(name)
            return w

        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = scope.findChild(QtWidgets.QPushButton, name) if scope is not None else None
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
            w = scope.findChild(QtWidgets.QCheckBox, name) if scope is not None else None
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            return w

        def _ensure_widget(widget: QtWidgets.QWidget, row: int, col: int, row_span: int = 1, col_span: int = 1) -> None:
            if bc_grid.indexOf(widget) >= 0:
                return
            bc_grid.addWidget(widget, row, col, row_span, col_span)

        self._bc_type_boxes = {}
        self._bc_value_spins = {}
        self._bc_ts_edits = {}
        for row, side in enumerate(("left", "right", "bottom", "top"), start=1):
            cb = _find_or_create_combo(f"{side}_bc_type_combo")
            cb.clear()
            for label, code in _BC_OPTIONS:
                cb.addItem(label, code)
            if side == "left":
                cb.setCurrentIndex(1)  # inflow default
            elif side == "right":
                cb.setCurrentIndex(2)  # stage default
            else:
                cb.setCurrentIndex(0)  # wall default

            spin = _find_or_create_double_spin(f"{side}_bc_value_spin")
            spin.setRange(-1.0e6, 1.0e6)
            spin.setDecimals(6)
            spin.setValue(0.0)
            if side == "left":
                spin.setValue(0.10)
            if side == "right":
                spin.setValue(1.00)

            ts_edit = _find_or_create_line_edit(f"{side}_bc_hydrograph_edit")
            ts_edit.setPlaceholderText("e.g. 0:00,10; 0:30,25; 1:00,40")

            edit_btn = _find_or_create_button(f"{side}_bc_editor_btn", "Edit...")
            try:
                edit_btn.clicked.disconnect()
            except Exception:
                pass
            edit_btn.clicked.connect(lambda _checked=False, s=side: self._open_hydrograph_editor(s))

            _ensure_widget(cb, row, 1)
            _ensure_widget(spin, row, 2)
            _ensure_widget(ts_edit, row, 3)
            _ensure_widget(edit_btn, row, 4)

            self._bc_type_boxes[side] = cb
            self._bc_value_spins[side] = spin
            self._bc_ts_edits[side] = ts_edit

        self.bc_lines_layer_combo = _find_or_create_combo("bc_lines_layer_combo")
        if self.bc_lines_layer_combo.count() == 0:
            self.bc_lines_layer_combo.addItem("(none)", None)
        _ensure_widget(self.bc_lines_layer_combo, 5, 1, 1, 4)

        self.inflow_progressive_chk = _find_or_create_check(
            "inflow_progressive_chk",
            "Flow BC: activate lowest-elevation boundary edges first as Q increases"
        )
        self.inflow_progressive_chk.setChecked(True)
        _ensure_widget(self.inflow_progressive_chk, 6, 0, 1, 5)

    def _build_map_tab_page(self) -> Tuple[QtWidgets.QWidget, QtWidgets.QGridLayout, QtWidgets.QGridLayout, QtWidgets.QGridLayout, QtWidgets.QGridLayout]:
        ui_path = self._forms_file_path("swe2d_map_tab.ui")
        map_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                map_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                map_tab_page = None
        if map_tab_page is None:
            map_tab_page = self._build_map_tab_page_fallback()

        map_data_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_data_layout")
        map_actions_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_actions_layout")
        map_results_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_results_layout")
        map_tools_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_tools_layout")
        if map_data_layout is None or map_actions_layout is None or map_results_layout is None or map_tools_layout is None:
            raise RuntimeError("Map tab UI missing one or more expected group layouts")
        return map_tab_page, map_data_layout, map_actions_layout, map_results_layout, map_tools_layout

    def _build_map_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        map_group = QtWidgets.QGroupBox("Map Layer Mesh + Terrain")
        map_layout = QtWidgets.QVBoxLayout(map_group)
        map_layout.setSpacing(6)

        map_data_group = QtWidgets.QGroupBox("Data layers")
        map_data_layout = QtWidgets.QGridLayout(map_data_group)
        map_data_layout.setObjectName("map_data_layout")

        map_actions_group = QtWidgets.QGroupBox("Model files and mesh actions")
        map_actions_layout = QtWidgets.QGridLayout(map_actions_group)
        map_actions_layout.setObjectName("map_actions_layout")

        map_results_group = QtWidgets.QGroupBox("Results and overlays")
        map_results_layout = QtWidgets.QGridLayout(map_results_group)
        map_results_layout.setObjectName("map_results_layout")

        map_tools_group = QtWidgets.QGroupBox("Utilities and 3D tools")
        map_tools_layout = QtWidgets.QGridLayout(map_tools_group)
        map_tools_layout.setObjectName("map_tools_layout")

        map_layout.addWidget(map_data_group)
        map_layout.addWidget(map_actions_group)
        map_layout.addWidget(map_results_group)
        map_layout.addWidget(map_tools_group)
        map_layout.addStretch(1)

        root_layout.addWidget(map_group)
        return root

    def _bind_map_tab_data_controls(self, map_tab_page: QtWidgets.QWidget, map_data_layout: QtWidgets.QGridLayout) -> None:
        def _find_or_create_combo(name: str) -> QtWidgets.QComboBox:
            w = map_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            return w

        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = map_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        def _ensure_labeled_widget(row: int, label_text: str, widget: QtWidgets.QWidget) -> None:
            if map_data_layout.indexOf(widget) >= 0:
                return
            map_data_layout.addWidget(QtWidgets.QLabel(label_text), row, 0)
            map_data_layout.addWidget(widget, row, 1)

        self.nodes_layer_combo = _find_or_create_combo("nodes_layer_combo")
        self.cells_layer_combo = _find_or_create_combo("cells_layer_combo")
        self.terrain_layer_combo = _find_or_create_combo("terrain_layer_combo")
        self.manning_layer_combo = _find_or_create_combo("manning_layer_combo")
        self.cn_layer_combo = _find_or_create_combo("cn_layer_combo")
        self.rain_gage_layer_combo = _find_or_create_combo("rain_gage_layer_combo")
        self.hyetograph_layer_combo = _find_or_create_combo("hyetograph_layer_combo")
        self.sample_lines_layer_combo = _find_or_create_combo("sample_lines_layer_combo")
        self.drain_nodes_layer_combo = _find_or_create_combo("drain_nodes_layer_combo")
        self.drain_links_layer_combo = _find_or_create_combo("drain_links_layer_combo")
        self.drain_inlets_layer_combo = _find_or_create_combo("drain_inlets_layer_combo")
        self.drain_node_inlets_layer_combo = _find_or_create_combo("drain_node_inlets_layer_combo")
        self.structures_layer_combo = _find_or_create_combo("structures_layer_combo")
        self.layer_group_combo = _find_or_create_combo("layer_group_combo")
        self.autopop_group_btn = _find_or_create_button("autopop_group_btn", "Autopopulate From Group")
        self.refresh_layers_btn = _find_or_create_button("refresh_layers_btn", "Refresh Layers")
        self.create_model_gpkg_btn = _find_or_create_button("create_model_gpkg_btn", "Create 2D Model GeoPackage")

        _ensure_labeled_widget(0, "Nodes layer:", self.nodes_layer_combo)
        _ensure_labeled_widget(1, "Cells layer:", self.cells_layer_combo)
        _ensure_labeled_widget(2, "Terrain raster:", self.terrain_layer_combo)
        _ensure_labeled_widget(3, "Manning polygons:", self.manning_layer_combo)
        _ensure_labeled_widget(4, "CN polygons:", self.cn_layer_combo)
        _ensure_labeled_widget(5, "Rain gages (points):", self.rain_gage_layer_combo)
        _ensure_labeled_widget(6, "Rain hyetographs (table):", self.hyetograph_layer_combo)
        _ensure_labeled_widget(7, "Sample lines layer:", self.sample_lines_layer_combo)
        _ensure_labeled_widget(8, "Drainage nodes layer:", self.drain_nodes_layer_combo)
        _ensure_labeled_widget(9, "Drainage links layer:", self.drain_links_layer_combo)
        _ensure_labeled_widget(10, "Drainage inlet types (table):", self.drain_inlets_layer_combo)
        _ensure_labeled_widget(11, "Drainage node-inlets (table):", self.drain_node_inlets_layer_combo)
        _ensure_labeled_widget(12, "Hydraulic structures layer:", self.structures_layer_combo)
        _ensure_labeled_widget(13, "Layer group:", self.layer_group_combo)

        if map_data_layout.indexOf(self.autopop_group_btn) < 0:
            map_data_layout.addWidget(self.autopop_group_btn, 14, 0, 1, 2)
        if map_data_layout.indexOf(self.refresh_layers_btn) < 0:
            map_data_layout.addWidget(self.refresh_layers_btn, 15, 0)
        if map_data_layout.indexOf(self.create_model_gpkg_btn) < 0:
            map_data_layout.addWidget(self.create_model_gpkg_btn, 15, 1)

        if self.drain_nodes_layer_combo.count() == 0:
            self.drain_nodes_layer_combo.addItem("(none)", None)
        if self.drain_links_layer_combo.count() == 0:
            self.drain_links_layer_combo.addItem("(none)", None)
        if self.drain_inlets_layer_combo.count() == 0:
            self.drain_inlets_layer_combo.addItem("(none)", None)
        if self.drain_node_inlets_layer_combo.count() == 0:
            self.drain_node_inlets_layer_combo.addItem("(none)", None)
        if self.structures_layer_combo.count() == 0:
            self.structures_layer_combo.addItem("(none)", None)
        if self.layer_group_combo.count() == 0:
            self.layer_group_combo.addItem("(no group)", None)

        try:
            self.autopop_group_btn.clicked.disconnect(self._autopopulate_layer_combos_from_group)
        except Exception:
            pass
        self.autopop_group_btn.clicked.connect(self._autopopulate_layer_combos_from_group)

        try:
            self.refresh_layers_btn.clicked.disconnect(self._refresh_layer_combos)
        except Exception:
            pass
        self.refresh_layers_btn.clicked.connect(self._refresh_layer_combos)

        try:
            self.create_model_gpkg_btn.clicked.disconnect(self._create_2d_model_geopackage)
        except Exception:
            pass
        self.create_model_gpkg_btn.clicked.connect(self._create_2d_model_geopackage)

    def _bind_map_tab_action_controls(self, map_tab_page: QtWidgets.QWidget, map_actions_layout: QtWidgets.QGridLayout) -> None:
        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = map_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        self.create_lumped_gpkg_btn = _find_or_create_button("create_lumped_gpkg_btn", "Create Lumped Hydro GeoPackage")
        self.load_model_gpkg_btn = _find_or_create_button("load_model_gpkg_btn", "Load 2D Model GeoPackage")
        self.migrate_model_gpkg_btn = _find_or_create_button("migrate_model_gpkg_btn", "Update GeoPackage Schema")
        self.preview_coupling_btn = _find_or_create_button("preview_coupling_btn", "Preview Drainage/Structure Coupling")
        self.export_mesh_layers_btn = _find_or_create_button("export_mesh_layers_btn", "Export Mesh To Map Layers")
        self.save_hdf5_btn = _find_or_create_button("save_hdf5_btn", "Save Mesh To HEC-RAS HDF5")
        self.save_results_hdf5_btn = _find_or_create_button("save_results_hdf5_btn", "Save Results To HEC-RAS HDF5")
        self.save_results_ugrid_btn = _find_or_create_button("save_results_ugrid_btn", "Save Results To UGRID NetCDF")
        self.import_mesh_layers_btn = _find_or_create_button("import_mesh_layers_btn", "Load Mesh From Selected Layers")
        self.terrain_to_nodes_btn = _find_or_create_button("terrain_to_nodes_btn", "Assign Node Z From Terrain")
        self.pull_node_z_btn = _find_or_create_button("pull_node_z_btn", "Pull Node Z From Nodes Layer")

        if map_actions_layout.indexOf(self.create_lumped_gpkg_btn) < 0:
            map_actions_layout.addWidget(self.create_lumped_gpkg_btn, 0, 0, 1, 2)
        if map_actions_layout.indexOf(self.load_model_gpkg_btn) < 0:
            map_actions_layout.addWidget(self.load_model_gpkg_btn, 1, 0)
        if map_actions_layout.indexOf(self.migrate_model_gpkg_btn) < 0:
            map_actions_layout.addWidget(self.migrate_model_gpkg_btn, 1, 1)
        if map_actions_layout.indexOf(self.preview_coupling_btn) < 0:
            map_actions_layout.addWidget(self.preview_coupling_btn, 2, 0, 1, 2)
        if map_actions_layout.indexOf(self.export_mesh_layers_btn) < 0:
            map_actions_layout.addWidget(self.export_mesh_layers_btn, 3, 0)
        if map_actions_layout.indexOf(self.save_hdf5_btn) < 0:
            map_actions_layout.addWidget(self.save_hdf5_btn, 3, 1)
        if map_actions_layout.indexOf(self.save_results_hdf5_btn) < 0:
            map_actions_layout.addWidget(self.save_results_hdf5_btn, 4, 0, 1, 2)
        if map_actions_layout.indexOf(self.save_results_ugrid_btn) < 0:
            map_actions_layout.addWidget(self.save_results_ugrid_btn, 5, 0, 1, 2)
        if map_actions_layout.indexOf(self.import_mesh_layers_btn) < 0:
            map_actions_layout.addWidget(self.import_mesh_layers_btn, 6, 0, 1, 2)
        if map_actions_layout.indexOf(self.terrain_to_nodes_btn) < 0:
            map_actions_layout.addWidget(self.terrain_to_nodes_btn, 7, 0, 1, 2)
        if map_actions_layout.indexOf(self.pull_node_z_btn) < 0:
            map_actions_layout.addWidget(self.pull_node_z_btn, 8, 0, 1, 2)

        self.migrate_model_gpkg_btn.setToolTip(
            "Add any missing layers and columns to an existing 2D model GeoPackage "
            "so it matches the current schema."
        )

        for btn, cb in (
            (self.create_lumped_gpkg_btn, self._create_lumped_hydrology_geopackage),
            (self.load_model_gpkg_btn, self._load_2d_model_geopackage),
            (self.migrate_model_gpkg_btn, self._migrate_2d_model_geopackage),
            (self.preview_coupling_btn, self._preview_coupling_configuration),
            (self.export_mesh_layers_btn, self._export_mesh_to_layers),
            (self.save_hdf5_btn, self._export_mesh_to_hdf5),
            (self.save_results_hdf5_btn, self._export_results_to_hdf5),
            (self.save_results_ugrid_btn, self._export_results_to_ugrid),
            (self.import_mesh_layers_btn, self._import_mesh_from_layers),
            (self.terrain_to_nodes_btn, self._assign_node_z_from_terrain),
            (self.pull_node_z_btn, self._pull_node_z_from_layer),
        ):
            try:
                btn.clicked.disconnect(cb)
            except Exception:
                pass
            btn.clicked.connect(cb)

    def _bind_map_tab_results_controls(self, map_tab_page: QtWidgets.QWidget, map_results_layout: QtWidgets.QGridLayout) -> None:
        def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
            w = map_tab_page.findChild(QtWidgets.QCheckBox, name)
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            return w

        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = map_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        def _find_or_create_combo(name: str) -> QtWidgets.QComboBox:
            w = map_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            return w

        def _find_or_create_double_spin(name: str) -> QtWidgets.QDoubleSpinBox:
            w = map_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            return w

        self.extended_outputs_chk = _find_or_create_check(
            "extended_outputs_chk",
            "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)",
        )
        self.save_mesh_results_to_gpkg_chk = _find_or_create_check(
            "save_mesh_results_to_gpkg_chk",
            "Save mesh snapshot results to GeoPackage",
        )
        self.save_line_results_to_gpkg_chk = _find_or_create_check(
            "save_line_results_to_gpkg_chk",
            "Save sampled line results to GeoPackage",
        )
        self.save_coupling_results_to_gpkg_chk = _find_or_create_check(
            "save_coupling_results_to_gpkg_chk",
            "Save drainage/structure results to GeoPackage",
        )
        self.save_run_log_to_gpkg_chk = _find_or_create_check(
            "save_run_log_to_gpkg_chk",
            "Save run log to GeoPackage",
        )
        self.open_results_viewer_btn = _find_or_create_button("open_results_viewer_btn", "Open 2D Results Viewer")
        self.open_results_panel_btn = _find_or_create_button("open_results_panel_btn", "Results Panel (multi-run)")
        self.high_perf_canvas_overlay_chk = _find_or_create_check(
            "high_perf_canvas_overlay_chk",
            "Show High-Perf Overlay On Map Canvas",
        )
        self.high_perf_canvas_overlay_field_combo = _find_or_create_combo("high_perf_canvas_overlay_field_combo")
        self.high_perf_canvas_overlay_cmap_combo = _find_or_create_combo("high_perf_canvas_overlay_cmap_combo")
        self.high_perf_canvas_overlay_lock_canvas_chk = _find_or_create_check(
            "high_perf_canvas_overlay_lock_canvas_chk",
            "Lock overlay resolution to current canvas size",
        )
        self.high_perf_canvas_overlay_res_combo = _find_or_create_combo("high_perf_canvas_overlay_res_combo")
        self.high_perf_canvas_overlay_auto_contrast_chk = _find_or_create_check(
            "high_perf_canvas_overlay_auto_contrast_chk",
            "Auto contrast",
        )
        self.high_perf_canvas_overlay_opacity_spin = _find_or_create_double_spin("high_perf_canvas_overlay_opacity_spin")

        if map_results_layout.indexOf(self.extended_outputs_chk) < 0:
            map_results_layout.addWidget(self.extended_outputs_chk, 0, 0, 1, 2)
        if map_results_layout.indexOf(self.save_mesh_results_to_gpkg_chk) < 0:
            map_results_layout.addWidget(self.save_mesh_results_to_gpkg_chk, 1, 0, 1, 2)
        if map_results_layout.indexOf(self.save_line_results_to_gpkg_chk) < 0:
            map_results_layout.addWidget(self.save_line_results_to_gpkg_chk, 2, 0, 1, 2)
        if map_results_layout.indexOf(self.save_coupling_results_to_gpkg_chk) < 0:
            map_results_layout.addWidget(self.save_coupling_results_to_gpkg_chk, 3, 0, 1, 2)
        if map_results_layout.indexOf(self.save_run_log_to_gpkg_chk) < 0:
            map_results_layout.addWidget(self.save_run_log_to_gpkg_chk, 4, 0, 1, 2)
        if map_results_layout.indexOf(self.open_results_viewer_btn) < 0:
            map_results_layout.addWidget(self.open_results_viewer_btn, 5, 0, 1, 2)
        if map_results_layout.indexOf(self.open_results_panel_btn) < 0:
            map_results_layout.addWidget(self.open_results_panel_btn, 6, 0, 1, 2)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_chk) < 0:
            map_results_layout.addWidget(self.high_perf_canvas_overlay_chk, 7, 0, 1, 2)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_field_combo) < 0:
            map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay field:"), 8, 0)
            map_results_layout.addWidget(self.high_perf_canvas_overlay_field_combo, 8, 1)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_cmap_combo) < 0:
            map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay colormap:"), 9, 0)
            map_results_layout.addWidget(self.high_perf_canvas_overlay_cmap_combo, 9, 1)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_lock_canvas_chk) < 0:
            map_results_layout.addWidget(self.high_perf_canvas_overlay_lock_canvas_chk, 10, 0, 1, 2)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_res_combo) < 0:
            map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay resolution:"), 11, 0)
            map_results_layout.addWidget(self.high_perf_canvas_overlay_res_combo, 11, 1)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_auto_contrast_chk) < 0:
            map_results_layout.addWidget(self.high_perf_canvas_overlay_auto_contrast_chk, 12, 0, 1, 2)
        if map_results_layout.indexOf(self.high_perf_canvas_overlay_opacity_spin) < 0:
            map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay opacity:"), 13, 0)
            map_results_layout.addWidget(self.high_perf_canvas_overlay_opacity_spin, 13, 1)

        self.extended_outputs_chk.setChecked(True)
        self.save_mesh_results_to_gpkg_chk.setChecked(True)
        self.save_line_results_to_gpkg_chk.setChecked(True)
        self.save_coupling_results_to_gpkg_chk.setChecked(True)
        self.save_run_log_to_gpkg_chk.setChecked(True)
        self.open_results_panel_btn.setToolTip("Open the dockable multi-run results panel")

        self.high_perf_canvas_overlay_chk.setChecked(False)
        self.high_perf_canvas_overlay_field_combo.clear()
        self.high_perf_canvas_overlay_field_combo.addItem("Depth", "depth")
        self.high_perf_canvas_overlay_field_combo.addItem("Velocity", "speed")
        self.high_perf_canvas_overlay_field_combo.addItem("Water Surface", "wse")
        self.high_perf_canvas_overlay_cmap_combo.clear()
        self.high_perf_canvas_overlay_cmap_combo.addItem("Turbo", "turbo")
        self.high_perf_canvas_overlay_cmap_combo.addItem("Viridis", "viridis")
        self.high_perf_canvas_overlay_cmap_combo.addItem("Plasma", "plasma")
        self.high_perf_canvas_overlay_cmap_combo.addItem("Gray", "gray")
        self.high_perf_canvas_overlay_res_combo.clear()
        self.high_perf_canvas_overlay_res_combo.addItem("640 x 360", (640, 360))
        self.high_perf_canvas_overlay_res_combo.addItem("960 x 540", (960, 540))
        self.high_perf_canvas_overlay_res_combo.addItem("1280 x 720", (1280, 720))
        self.high_perf_canvas_overlay_res_combo.addItem("1920 x 1080", (1920, 1080))
        self.high_perf_canvas_overlay_res_combo.setCurrentIndex(2)
        self.high_perf_canvas_overlay_lock_canvas_chk.setChecked(True)
        self.high_perf_canvas_overlay_auto_contrast_chk.setChecked(True)
        self.high_perf_canvas_overlay_opacity_spin.setDecimals(2)
        self.high_perf_canvas_overlay_opacity_spin.setRange(0.05, 1.0)
        self.high_perf_canvas_overlay_opacity_spin.setSingleStep(0.05)
        self.high_perf_canvas_overlay_opacity_spin.setValue(0.65)

        for sig_obj, cb in (
            (self.open_results_viewer_btn.clicked, self._open_line_results_viewer),
            (self.open_results_panel_btn.clicked, self._show_results_panel),
            (self.high_perf_canvas_overlay_chk.toggled, self._on_high_perf_canvas_overlay_toggled),
            (self.high_perf_canvas_overlay_field_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
            (self.high_perf_canvas_overlay_cmap_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
            (self.high_perf_canvas_overlay_lock_canvas_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
            (self.high_perf_canvas_overlay_res_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
            (self.high_perf_canvas_overlay_auto_contrast_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
            (self.high_perf_canvas_overlay_opacity_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        ):
            try:
                sig_obj.disconnect(cb)
            except Exception:
                pass
            sig_obj.connect(cb)

        self._on_high_perf_canvas_overlay_style_changed()

    def _bind_map_tab_tools_controls(self, map_tab_page: QtWidgets.QWidget, map_tools_layout: QtWidgets.QGridLayout) -> None:
        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = map_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        self.draw_sample_line_btn = _find_or_create_button("draw_sample_line_btn", "Draw Sample Line On Map")
        self.open_coupling_results_viewer_btn = _find_or_create_button(
            "open_coupling_results_viewer_btn", "Open Drainage/Structure Results Viewer"
        )
        self.open_run_log_viewer_btn = _find_or_create_button("open_run_log_viewer_btn", "Open Run Log Viewer")
        self.open_3d_patch_viewer_btn = _find_or_create_button("open_3d_patch_viewer_btn", "Open 3D Patch Viewer")
        self.publish_3d_patch_surface_btn = _find_or_create_button(
            "publish_3d_patch_surface_btn", "Publish Current 3D Surface To QGIS 3D"
        )
        self.layer_status_lbl = map_tab_page.findChild(QtWidgets.QLabel, "layer_status_lbl")
        if self.layer_status_lbl is None:
            self.layer_status_lbl = QtWidgets.QLabel("No layer-linked mesh yet")
            self.layer_status_lbl.setObjectName("layer_status_lbl")

        if map_tools_layout.indexOf(self.draw_sample_line_btn) < 0:
            map_tools_layout.addWidget(self.draw_sample_line_btn, 0, 0, 1, 2)
        if map_tools_layout.indexOf(self.open_coupling_results_viewer_btn) < 0:
            map_tools_layout.addWidget(self.open_coupling_results_viewer_btn, 1, 0, 1, 2)
        if map_tools_layout.indexOf(self.open_run_log_viewer_btn) < 0:
            map_tools_layout.addWidget(self.open_run_log_viewer_btn, 2, 0, 1, 2)
        if map_tools_layout.indexOf(self.layer_status_lbl) < 0:
            map_tools_layout.addWidget(self.layer_status_lbl, 3, 0, 1, 2)
        if map_tools_layout.indexOf(self.open_3d_patch_viewer_btn) < 0:
            map_tools_layout.addWidget(self.open_3d_patch_viewer_btn, 4, 0, 1, 2)
        if map_tools_layout.indexOf(self.publish_3d_patch_surface_btn) < 0:
            map_tools_layout.addWidget(self.publish_3d_patch_surface_btn, 5, 0, 1, 2)

        self.draw_sample_line_btn.setToolTip("Draw a sample polyline directly on the map canvas")
        self.open_3d_patch_viewer_btn.setToolTip(
            "Open experimental post-processing viewer for captured 3D patch snapshots."
        )
        self.publish_3d_patch_surface_btn.setToolTip(
            "Build/update a triangulated free-surface layer from the nearest captured 3D patch snapshot\n"
            "and push it into the current QGIS project for native 3D map viewing."
        )
        self.layer_status_lbl.setWordWrap(True)
        if not str(self.layer_status_lbl.text() or "").strip():
            self.layer_status_lbl.setText("No layer-linked mesh yet")

        for btn, cb in (
            (self.draw_sample_line_btn, self._activate_sample_line_draw_tool),
            (self.open_coupling_results_viewer_btn, self._open_coupling_results_viewer),
            (self.open_run_log_viewer_btn, self._open_run_log_viewer),
            (self.open_3d_patch_viewer_btn, self._open_3d_patch_viewer),
            (self.publish_3d_patch_surface_btn, self._publish_current_3d_surface_to_qgis_3d),
        ):
            try:
                btn.clicked.disconnect(cb)
            except Exception:
                pass
            btn.clicked.connect(cb)

    def _build_topology_tab_page(self) -> Tuple[QtWidgets.QWidget, QtWidgets.QGridLayout]:
        ui_path = self._forms_file_path("swe2d_topology_tab.ui")
        topology_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                topology_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                topology_tab_page = None
        if topology_tab_page is None:
            topology_tab_page = self._build_topology_tab_page_fallback()

        topo_layout = topology_tab_page.findChild(QtWidgets.QGridLayout, "topo_layout")
        if topo_layout is None:
            raise RuntimeError("Topology tab UI missing topo_layout")
        return topology_tab_page, topo_layout

    def _build_model_tab_page(self) -> Tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        ui_path = self._forms_file_path("swe2d_model_tab.ui")
        model_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                model_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                model_tab_page = None
        if model_tab_page is None:
            model_tab_page = self._build_model_tab_page_fallback()

        model_param_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_param_form")
        if model_param_form is None:
            raise RuntimeError("Model tab UI missing model_param_form")
        return model_tab_page, model_param_form

    def _build_model_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        model_group = QtWidgets.QGroupBox("Model Parameters")
        model_param_form = QtWidgets.QFormLayout(model_group)
        model_param_form.setObjectName("model_param_form")

        patch_3d_group = QtWidgets.QGroupBox("3D Patch")
        patch_3d_form = QtWidgets.QFormLayout(patch_3d_group)
        patch_3d_form.setObjectName("patch_3d_form")

        root_layout.addWidget(model_group)
        root_layout.addWidget(patch_3d_group)
        return root

    def _bind_model_tab_core_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
            if param_form.indexOf(widget) >= 0:
                return
            param_form.addRow(label, widget)

        def _find_or_create_double_spin(name: str, label: str) -> QtWidgets.QDoubleSpinBox:
            w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_spin(name: str, label: str) -> QtWidgets.QSpinBox:
            w = model_tab_page.findChild(QtWidgets.QSpinBox, name)
            if w is None:
                w = QtWidgets.QSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_check(name: str, label: str, text: str) -> QtWidgets.QCheckBox:
            w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            _ensure_row(label, w)
            return w

        def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
            w = model_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        self.n_mann_spin = _find_or_create_double_spin("n_mann_spin", "Manning n:")
        self.n_mann_spin.setRange(0.0, 1.0)
        self.n_mann_spin.setDecimals(5)
        self.n_mann_spin.setValue(0.020)

        self.cfl_spin = _find_or_create_double_spin("cfl_spin", "CFL:")
        self.cfl_spin.setRange(0.01, 0.99)
        self.cfl_spin.setDecimals(3)
        self.cfl_spin.setValue(0.45)

        self.h_min_spin = _find_or_create_double_spin("h_min_spin", "h_min:")
        self.h_min_spin.setRange(1.0e-9, 1.0)
        self.h_min_spin.setDecimals(8)
        self.h_min_spin.setValue(1.0e-6)

        self.initial_condition_combo = _find_or_create_combo("initial_condition_combo", "Initial condition:")
        prev_data = self.initial_condition_combo.currentData()
        prev_text = self.initial_condition_combo.currentText()
        self.initial_condition_combo.blockSignals(True)
        try:
            self.initial_condition_combo.clear()
            self.initial_condition_combo.addItem("Dry start", "dry")
            self.initial_condition_combo.addItem("Uniform depth", "uniform_depth")
            self.initial_condition_combo.addItem("Uniform water surface elevation", "uniform_wse")
            idx = self.initial_condition_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.initial_condition_combo.findText(prev_text)
            if idx < 0:
                idx = self.initial_condition_combo.findData("dry")
            if idx >= 0:
                self.initial_condition_combo.setCurrentIndex(idx)
        finally:
            self.initial_condition_combo.blockSignals(False)
        self.initial_condition_combo.setToolTip(
            "Initial condition source used at run start.\n"
            "Dry start: h=0.\n"
            "Uniform depth: constant initial depth everywhere.\n"
            "Uniform WSE: depth = max(0, WSE - local bed)."
        )

        self.initial_depth_spin = _find_or_create_double_spin("initial_depth_spin", "Initial depth:")
        self.initial_depth_spin.setRange(0.0, 1.0e6)
        self.initial_depth_spin.setDecimals(4)
        self.initial_depth_spin.setValue(0.0)

        self.initial_wse_spin = _find_or_create_double_spin("initial_wse_spin", "Initial WSE:")
        self.initial_wse_spin.setRange(-1.0e6, 1.0e6)
        self.initial_wse_spin.setDecimals(4)
        self.initial_wse_spin.setValue(0.0)

        self.adaptive_cfl_dt_chk = _find_or_create_check(
            "adaptive_cfl_dt_chk", "Variable timestep:", "Enable variable timestep (CFL)"
        )
        self.adaptive_cfl_dt_chk.setChecked(False)
        self.adaptive_cfl_dt_chk.setToolTip(
            "If enabled, runtime dt is selected from CFL each step.\n"
            "The dt field is used as dt_max (upper bound).\n"
            "If disabled, dt is fixed each step."
        )

        self.dt_spin = _find_or_create_double_spin("dt_spin", "dt (fixed or dt_max):")
        self.dt_spin.setRange(1.0e-4, 1.0e6)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setValue(0.05)

        self.gpu_diag_sync_interval_spin = _find_or_create_spin(
            "gpu_diag_sync_interval_spin", "GPU diag sync (steps):"
        )
        self.gpu_diag_sync_interval_spin.setRange(1, 1000000)
        self.gpu_diag_sync_interval_spin.setValue(10)
        self.gpu_diag_sync_interval_spin.setToolTip(
            "GPU host diagnostic sync cadence in computational steps.\n"
            "1 = sync every step (freshest Cmax/WSEres runtime output).\n"
            "Higher values reduce host sync overhead but update diagnostics less often."
        )

        self.enable_cuda_graphs_chk = _find_or_create_check(
            "enable_cuda_graphs_chk", "CUDA graph replay:", "Enable"
        )
        self.enable_cuda_graphs_chk.setChecked(False)
        self.enable_cuda_graphs_chk.setToolTip(
            "Enable CUDA graph capture/replay for the core GPU step kernel chain.\n"
            "Can reduce launch overhead and improve throughput on compatible runs."
        )

    def _bind_model_tab_hydrology_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
            if param_form.indexOf(widget) >= 0:
                return
            param_form.addRow(label, widget)

        def _find_or_create_double_spin(name: str, label: str) -> QtWidgets.QDoubleSpinBox:
            w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_spin(name: str, label: str) -> QtWidgets.QSpinBox:
            w = model_tab_page.findChild(QtWidgets.QSpinBox, name)
            if w is None:
                w = QtWidgets.QSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_check(name: str, label: str, text: str) -> QtWidgets.QCheckBox:
            w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            _ensure_row(label, w)
            return w

        def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
            w = model_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        self.max_rel_depth_increase_spin = _find_or_create_double_spin(
            "max_rel_depth_increase_spin", "Max rel depth increase:"
        )
        self.max_rel_depth_increase_spin.setRange(0.0, 1000.0)
        self.max_rel_depth_increase_spin.setDecimals(3)
        self.max_rel_depth_increase_spin.setValue(2.0)
        self.max_rel_depth_increase_spin.setToolTip(
            "Per-step depth growth limiter on GPU update:\n"
            "h_new <= h_old + factor * max(h_old, h_min).\n"
            "Lower values are more robust near advancing wet/dry fronts."
        )

        self.max_source_depth_step_spin = _find_or_create_double_spin(
            "max_source_depth_step_spin", "Max source dh/step:"
        )
        self.max_source_depth_step_spin.setRange(0.0, 10.0)
        self.max_source_depth_step_spin.setDecimals(6)
        self.max_source_depth_step_spin.setValue(0.0)
        self.max_source_depth_step_spin.setToolTip(
            "Absolute cap on positive source-driven depth increase per step (model units).\n"
            "0 disables the cap. Useful for suppressing rain/CN impulse spikes."
        )

        self.max_source_rate_spin = _find_or_create_double_spin("max_source_rate_spin", "Max source rate:")
        self.max_source_rate_spin.setRange(0.0, 100.0)
        self.max_source_rate_spin.setDecimals(6)
        self.max_source_rate_spin.setValue(0.0)
        self.max_source_rate_spin.setToolTip(
            "Cap on positive net source rate (model units per second).\n"
            "0 disables the cap. Applies before per-step depth update."
        )

        self.extreme_rain_mode_chk = _find_or_create_check("extreme_rain_mode_chk", "Extreme rain mode:", "Enable")
        self.extreme_rain_mode_chk.setChecked(False)
        self.extreme_rain_mode_chk.setToolTip(
            "Adaptive source-CFL limiter for extreme rainfall/source events.\n"
            "When enabled, positive source terms are reduced using an equivalent\n"
            "substepping factor so dt*source remains bounded by beta*h_ref."
        )

        self.source_cfl_beta_spin = _find_or_create_double_spin("source_cfl_beta_spin", "Source CFL beta:")
        self.source_cfl_beta_spin.setRange(0.01, 2.0)
        self.source_cfl_beta_spin.setDecimals(3)
        self.source_cfl_beta_spin.setSingleStep(0.05)
        self.source_cfl_beta_spin.setValue(0.25)
        self.source_cfl_beta_spin.setToolTip(
            "Target source-CFL beta in dt*source <= beta*h_ref.\n"
            "Lower beta is more conservative."
        )

        self.source_max_substeps_spin = _find_or_create_spin("source_max_substeps_spin", "Source max substeps:")
        self.source_max_substeps_spin.setRange(1, 512)
        self.source_max_substeps_spin.setValue(16)
        self.source_max_substeps_spin.setToolTip(
            "Maximum equivalent source substeps used by adaptive source limiter."
        )

        self.source_true_subcycling_chk = _find_or_create_check(
            "source_true_subcycling_chk", "True source subcycling:", "Enable"
        )
        self.source_true_subcycling_chk.setChecked(False)
        self.source_true_subcycling_chk.setToolTip(
            "Apply true source subcycling (real sub-iterations over dt) instead of\n"
            "equivalent one-shot source scaling."
        )

        self.source_imex_split_chk = _find_or_create_check("source_imex_split_chk", "IMEX source split:", "Enable")
        self.source_imex_split_chk.setChecked(False)
        self.source_imex_split_chk.setToolTip(
            "IMEX-style split: apply flux update first, then source/friction subcycling.\n"
            "Most useful when true source subcycling is enabled."
        )

        self.source_stage_coupled_imex_rk2_chk = _find_or_create_check(
            "source_stage_coupled_imex_rk2_chk", "Stage-coupled IMEX-RK2 sources:", "Enable"
        )
        self.source_stage_coupled_imex_rk2_chk.setChecked(False)
        self.source_stage_coupled_imex_rk2_chk.setToolTip(
            "Stage-coupled IMEX-RK2 for external coupling sources (drainage/structures).\n"
            "Runs a predictor/corrector source update each step (GPU native injection path).\n"
            "Best for stiff coupling; costs extra compute per step."
        )

        self.shallow_damping_depth_spin = _find_or_create_double_spin(
            "shallow_damping_depth_spin", "Shallow damping depth:"
        )
        self.shallow_damping_depth_spin.setRange(1.0e-8, 10.0)
        self.shallow_damping_depth_spin.setDecimals(6)
        self.shallow_damping_depth_spin.setValue(1.0e-4)
        self.shallow_damping_depth_spin.setToolTip(
            "Depth threshold for smooth momentum damping in shallow cells."
        )

        self.shallow_front_recon_fallback_chk = _find_or_create_check(
            "shallow_front_recon_fallback_chk", "Shallow-front recon fallback:", "Enable"
        )
        self.shallow_front_recon_fallback_chk.setChecked(True)
        self.shallow_front_recon_fallback_chk.setToolTip(
            "If enabled, force first-order reconstruction on shallow wet/dry-front\n"
            "edge pairs to improve stability for higher-order schemes."
        )

        self.front_flux_damping_spin = _find_or_create_double_spin("front_flux_damping_spin", "Front flux damping:")
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

        self.active_set_hysteresis_chk = _find_or_create_check(
            "active_set_hysteresis_chk", "Active-set hysteresis:", "Enable"
        )
        self.active_set_hysteresis_chk.setChecked(True)
        self.active_set_hysteresis_chk.setToolTip(
            "Keep cells active for one extra step after they dry below h_min.\n"
            "Prevents rapid oscillatory wet/dry switching at the advancing front.\n"
            "Has negligible performance overhead."
        )

        self.depth_cap_spin = _find_or_create_double_spin("depth_cap_spin", "Depth cap:")
        self.depth_cap_spin.setRange(0.001, 1.0e7)
        self.depth_cap_spin.setDecimals(3)
        self.depth_cap_spin.setValue(1.0e6)
        self.depth_cap_spin.setToolTip("Absolute depth cap for robustness.")

        self.momentum_cap_min_speed_spin = _find_or_create_double_spin(
            "momentum_cap_min_speed_spin", "Momentum cap min speed:"
        )
        self.momentum_cap_min_speed_spin.setRange(0.1, 1.0e4)
        self.momentum_cap_min_speed_spin.setDecimals(3)
        self.momentum_cap_min_speed_spin.setValue(50.0)
        self.momentum_cap_min_speed_spin.setToolTip(
            "Minimum speed floor used by momentum clipping."
        )

        self.momentum_cap_celerity_mult_spin = _find_or_create_double_spin(
            "momentum_cap_celerity_mult_spin", "Momentum cap celerity mult:"
        )
        self.momentum_cap_celerity_mult_spin.setRange(0.1, 1000.0)
        self.momentum_cap_celerity_mult_spin.setDecimals(3)
        self.momentum_cap_celerity_mult_spin.setValue(20.0)
        self.momentum_cap_celerity_mult_spin.setToolTip(
            "Momentum clipping speed cap multiplier on sqrt(g*h)."
        )

        self.max_inv_area_spin = _find_or_create_double_spin("max_inv_area_spin", "Max inv area:")
        self.max_inv_area_spin.setRange(1.0, 1.0e12)
        self.max_inv_area_spin.setDecimals(1)
        self.max_inv_area_spin.setValue(1.0e6)
        self.max_inv_area_spin.setToolTip(
            "Cap on inverse cell area used in flux and update kernels."
        )

        self.cfl_lambda_cap_spin = _find_or_create_double_spin("cfl_lambda_cap_spin", "CFL lambda cap:")
        self.cfl_lambda_cap_spin.setRange(1.0, 1.0e12)
        self.cfl_lambda_cap_spin.setDecimals(1)
        self.cfl_lambda_cap_spin.setValue(1.0e6)
        self.cfl_lambda_cap_spin.setToolTip(
            "Cap on local CFL lambda used in dt reduction and diagnostics."
        )

        self.rain_rate_spin = _find_or_create_double_spin("rain_rate_spin", "Rain rate:")
        self.rain_rate_spin.setRange(0.0, 2000.0)
        self.rain_rate_spin.setDecimals(3)
        self.rain_rate_spin.setValue(0.0)
        self.rain_rate_spin.setSuffix(" mm/hr")

        self.cn_default_spin = _find_or_create_double_spin("cn_default_spin", "Default CN:")
        self.cn_default_spin.setRange(1.0, 100.0)
        self.cn_default_spin.setDecimals(1)
        self.cn_default_spin.setValue(75.0)

        self.ia_ratio_spin = _find_or_create_double_spin("ia_ratio_spin", "SCS Ia/S ratio:")
        self.ia_ratio_spin.setRange(0.0, 1.0)
        self.ia_ratio_spin.setDecimals(3)
        self.ia_ratio_spin.setSingleStep(0.01)
        self.ia_ratio_spin.setValue(0.2)
        self.ia_ratio_spin.setToolTip(
            "Initial abstraction ratio (Ia/S) for SCS Curve Number losses.\n"
            "Typical default is 0.20."
        )

        self.use_spatial_rain_cn_chk = _find_or_create_check(
            "use_spatial_rain_cn_chk",
            "Spatial rainfall:",
            "Use Thiessen gage rainfall when layers are available",
        )
        self.use_spatial_rain_cn_chk.setChecked(True)

        self.infiltration_method_combo = _find_or_create_combo("infiltration_method_combo", "Infiltration method:")
        prev_data = self.infiltration_method_combo.currentData()
        prev_text = self.infiltration_method_combo.currentText()
        self.infiltration_method_combo.blockSignals(True)
        try:
            self.infiltration_method_combo.clear()
            self.infiltration_method_combo.addItem("SCS Curve Number", "scs_cn")
            self.infiltration_method_combo.addItem("None (no infiltration)", "none")
            idx = self.infiltration_method_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.infiltration_method_combo.findText(prev_text)
            if idx < 0:
                idx = self.infiltration_method_combo.findData("scs_cn")
            if idx >= 0:
                self.infiltration_method_combo.setCurrentIndex(idx)
        finally:
            self.infiltration_method_combo.blockSignals(False)
        self.infiltration_method_combo.setToolTip(
            "Infiltration/loss method applied to rainfall before it enters the 2D surface as runoff.\n"
            "SCS Curve Number: NRCS CN abstraction (default).\n"
            "None: all rainfall becomes direct runoff - no abstraction."
        )

        self.storm_area_layer_combo = _find_or_create_combo("storm_area_layer_combo", "Storm area layer (optional):")
        prev_data = self.storm_area_layer_combo.currentData()
        prev_text = self.storm_area_layer_combo.currentText()
        self.storm_area_layer_combo.blockSignals(True)
        try:
            self.storm_area_layer_combo.clear()
            self.storm_area_layer_combo.addItem("(none)", None)
            idx = self.storm_area_layer_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.storm_area_layer_combo.findText(prev_text)
            if idx >= 0:
                self.storm_area_layer_combo.setCurrentIndex(idx)
        finally:
            self.storm_area_layer_combo.blockSignals(False)

        self.rain_boundary_buffer_rings_spin = _find_or_create_spin(
            "rain_boundary_buffer_rings_spin", "Rain boundary buffer rings:"
        )
        self.rain_boundary_buffer_rings_spin.setRange(0, 10)
        self.rain_boundary_buffer_rings_spin.setValue(1)
        self.rain_boundary_buffer_rings_spin.setToolTip(
            "Boundary rain buffer rings (Thiessen + CN forcing).\n"
            "0: no exclusion. 1: exclude boundary cells.\n"
            "N>1: also exclude N-1 inward neighbor rings."
        )

    def _bind_model_tab_solver_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
            if param_form.indexOf(widget) >= 0:
                return
            param_form.addRow(label, widget)

        def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
            w = model_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_line_edit(name: str, label: str, text: str = "") -> QtWidgets.QLineEdit:
            w = model_tab_page.findChild(QtWidgets.QLineEdit, name)
            if w is None:
                w = QtWidgets.QLineEdit(text)
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        self.internal_flow_layer_combo = _find_or_create_combo("internal_flow_layer_combo", "Internal flow layer:")
        prev_data = self.internal_flow_layer_combo.currentData()
        prev_text = self.internal_flow_layer_combo.currentText()
        self.internal_flow_layer_combo.blockSignals(True)
        try:
            self.internal_flow_layer_combo.clear()
            self.internal_flow_layer_combo.addItem("(none)", None)
            idx = self.internal_flow_layer_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.internal_flow_layer_combo.findText(prev_text)
            if idx >= 0:
                self.internal_flow_layer_combo.setCurrentIndex(idx)
        finally:
            self.internal_flow_layer_combo.blockSignals(False)

        self.internal_flow_field_edit = _find_or_create_line_edit(
            "internal_flow_field_edit", "Internal flow field:", "q_cms"
        )
        if not str(self.internal_flow_field_edit.text() or "").strip():
            self.internal_flow_field_edit.setText("q_cms")
        self.internal_flow_field_edit.setPlaceholderText("field name, e.g. q_cms")

        self.run_time_edit = _find_or_create_line_edit("run_time_edit", "Run duration (hr or HH:MM):")
        self.run_time_edit.setPlaceholderText("decimal hours (e.g. 1.5) or HH:MM (e.g. 01:30)")
        if not str(self.run_time_edit.text() or "").strip():
            self.run_time_edit.setText("1:00")

        self.reconstruction_combo = _find_or_create_combo("reconstruction_combo", "Reconstruction:")
        prev_data = self.reconstruction_combo.currentData()
        prev_text = self.reconstruction_combo.currentText()
        self.reconstruction_combo.blockSignals(True)
        try:
            self.reconstruction_combo.clear()
            for label, value in _RECONSTRUCTION_OPTIONS:
                self.reconstruction_combo.addItem(label, int(value))
            idx = self.reconstruction_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.reconstruction_combo.findText(prev_text)
            if idx < 0:
                idx = min(1, max(0, self.reconstruction_combo.count() - 1))
            if idx >= 0:
                self.reconstruction_combo.setCurrentIndex(idx)
        finally:
            self.reconstruction_combo.blockSignals(False)
        self.reconstruction_combo.setToolTip(
            "Select spatial reconstruction for the native solver.\n"
            "All 2nd-order schemes use Green-Gauss gradient-based TVD reconstruction:\n"
            "  Superbee (MUSCL Fast)  - most aggressive TVD, sharpest fronts\n"
            "  MinMod                 - most conservative, most stable near dry fronts\n"
            "  MC                     - balanced monotonized-central (good default)\n"
            "  Van Leer               - smooth limiter, good for continuous waves\n"
            "Recommend: start with MUSCL MinMod; switch to MC or Van Leer once stable."
        )

        self.temporal_order_combo = _find_or_create_combo("temporal_order_combo", "Temporal discretization:")
        prev_data = self.temporal_order_combo.currentData()
        prev_text = self.temporal_order_combo.currentText()
        self.temporal_order_combo.blockSignals(True)
        try:
            self.temporal_order_combo.clear()
            for label, value in _TEMPORAL_ORDER_OPTIONS:
                self.temporal_order_combo.addItem(label, int(value))
            idx = self.temporal_order_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.temporal_order_combo.findText(prev_text)
            if idx < 0:
                idx = min(1, max(0, self.temporal_order_combo.count() - 1))
            if idx >= 0:
                self.temporal_order_combo.setCurrentIndex(idx)
        finally:
            self.temporal_order_combo.blockSignals(False)
        self.temporal_order_combo.setToolTip(
            "Select temporal integration scheme:\n"
            "  Euler (RK1)  - 1st-order, fastest, use for dry-bed or debugging\n"
            "  RK2 (Heun)   - 2nd-order (default), balanced stability and speed\n"
            "  RK4 (classic) - 4th-order composed path\n"
            "  Graph-safe RK4 - true staged RK4 with CUDA-graph-safe forcing\n"
            "  Graph-safe RK5 - Cash-Karp staged RK5 with CUDA-graph-safe forcing\n"
            "Higher-order schemes are GPU-oriented and may be auto-adjusted by runtime guards."
        )

        self.equation_set_combo = _find_or_create_combo("equation_set_combo", "Equation set:")
        prev_data = self.equation_set_combo.currentData()
        prev_text = self.equation_set_combo.currentText()
        self.equation_set_combo.blockSignals(True)
        try:
            self.equation_set_combo.clear()
            if SWE2DEquationSet is not None:
                self.equation_set_combo.addItem("Hydrostatic 2D (default)", int(SWE2DEquationSet.HYDROSTATIC_2D))
                self.equation_set_combo.addItem("Nonhydrostatic 2D", int(SWE2DEquationSet.NONHYDROSTATIC_2D))
            else:
                self.equation_set_combo.addItem("Hydrostatic 2D (default)", 0)
                self.equation_set_combo.addItem("Nonhydrostatic 2D", 1)
            idx = self.equation_set_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.equation_set_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.equation_set_combo.setCurrentIndex(idx)
        finally:
            self.equation_set_combo.blockSignals(False)
        self.equation_set_combo.setToolTip(
            "Choose the governing equation set for the 2D solver.\n"
            "Hydrostatic 2D keeps the existing shallow-water path.\n"
            "Nonhydrostatic 2D enables the pressure-correction solver and requires GPU."
        )

    def _bind_model_tab_3d_patch_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        patch_form = model_tab_page.findChild(QtWidgets.QFormLayout, "patch_3d_form") or param_form

        def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
            if patch_form.indexOf(widget) >= 0:
                return
            patch_form.addRow(label, widget)

        def _ensure_widget_row(widget: QtWidgets.QWidget) -> None:
            if patch_form.indexOf(widget) >= 0:
                return
            patch_form.addRow(widget)

        def _find_or_create_check(name: str, label: str, text: str) -> QtWidgets.QCheckBox:
            w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            _ensure_row(label, w)
            return w

        def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
            w = model_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_double_spin(name: str, label: str) -> QtWidgets.QDoubleSpinBox:
            w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_line_edit(name: str, label: str) -> QtWidgets.QLineEdit:
            w = model_tab_page.findChild(QtWidgets.QLineEdit, name)
            if w is None:
                w = QtWidgets.QLineEdit()
                w.setObjectName(name)
            _ensure_row(label, w)
            return w

        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = model_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            _ensure_widget_row(w)
            return w

        def _find_or_create_label(name: str, text: str) -> QtWidgets.QLabel:
            w = model_tab_page.findChild(QtWidgets.QLabel, name)
            if w is None:
                w = QtWidgets.QLabel(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            _ensure_widget_row(w)
            return w

        self.experimental_3d_mode_chk = _find_or_create_check(
            "experimental_3d_mode_chk", "3D patch execution mode:", "Run 3D patch solver (GPU)"
        )
        self.experimental_3d_mode_chk.setChecked(False)
        self.experimental_3d_mode_chk.setToolTip(
            "Experimental 3D patch solver mode for validation/smoke testing.\n"
            "Enables SINGLE_PHASE_FREE_SURFACE_VOF and optional 2D-3D coupling."
        )
        self._experimental_3d_mode_supported = bool(
            SolverModelOptions is not None
            and SWE2DThreeDSolverModel is not None
            and SWE2DThreeDCouplingMode is not None
        )

        self.experimental_3d_coupling_mode_combo = _find_or_create_combo(
            "experimental_3d_coupling_mode_combo", "3D patch coupling mode:"
        )
        prev_data = self.experimental_3d_coupling_mode_combo.currentData()
        prev_text = self.experimental_3d_coupling_mode_combo.currentText()
        self.experimental_3d_coupling_mode_combo.blockSignals(True)
        try:
            self.experimental_3d_coupling_mode_combo.clear()
            if SWE2DThreeDCouplingMode is not None:
                self.experimental_3d_coupling_mode_combo.addItem(
                    "Off (uncoupled)", int(SWE2DThreeDCouplingMode.OFF)
                )
                self.experimental_3d_coupling_mode_combo.addItem(
                    "One-way (2D -> 3D)", int(SWE2DThreeDCouplingMode.ONE_WAY_2D_TO_3D)
                )
                self.experimental_3d_coupling_mode_combo.addItem(
                    "Two-way (2D <-> 3D)", int(SWE2DThreeDCouplingMode.TWO_WAY_2D_3D)
                )
            else:
                self.experimental_3d_coupling_mode_combo.addItem("Off (uncoupled)", 0)
                self.experimental_3d_coupling_mode_combo.addItem("One-way (2D -> 3D)", 1)
                self.experimental_3d_coupling_mode_combo.addItem("Two-way (2D <-> 3D)", 2)
            idx = self.experimental_3d_coupling_mode_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.experimental_3d_coupling_mode_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.experimental_3d_coupling_mode_combo.setCurrentIndex(idx)
        finally:
            self.experimental_3d_coupling_mode_combo.blockSignals(False)
        self.experimental_3d_coupling_mode_combo.setToolTip(
            "Select 2D-3D exchange mode for the 3D patch runtime.\n"
            "When coupling is ON, the GUI auto-builds and uploads a boundary-edge interface contract."
        )

        self.experimental_3d_patch_face_len_x_spin = _find_or_create_double_spin(
            "experimental_3d_patch_face_len_x_spin", "3D patch target face length x:"
        )
        self.experimental_3d_patch_face_len_x_spin.setRange(1.0e-4, 1.0e6)
        self.experimental_3d_patch_face_len_x_spin.setDecimals(6)
        self.experimental_3d_patch_face_len_x_spin.setSingleStep(0.5)
        self.experimental_3d_patch_face_len_x_spin.setValue(5.0)
        self.experimental_3d_patch_face_len_x_spin.setToolTip(
            "Target x-face length for 3D patch cells (model units).\n"
            "Runtime resolves nx = ceil((xmax-xmin)/target_len_x)."
        )

        self.experimental_3d_patch_face_len_y_spin = _find_or_create_double_spin(
            "experimental_3d_patch_face_len_y_spin", "3D patch target face length y:"
        )
        self.experimental_3d_patch_face_len_y_spin.setRange(1.0e-4, 1.0e6)
        self.experimental_3d_patch_face_len_y_spin.setDecimals(6)
        self.experimental_3d_patch_face_len_y_spin.setSingleStep(0.5)
        self.experimental_3d_patch_face_len_y_spin.setValue(5.0)
        self.experimental_3d_patch_face_len_y_spin.setToolTip(
            "Target y-face length for 3D patch cells (model units).\n"
            "Runtime resolves ny = ceil((ymax-ymin)/target_len_y)."
        )

        self.experimental_3d_patch_face_len_z_spin = _find_or_create_double_spin(
            "experimental_3d_patch_face_len_z_spin", "3D patch target face length z:"
        )
        self.experimental_3d_patch_face_len_z_spin.setRange(1.0e-4, 1.0e6)
        self.experimental_3d_patch_face_len_z_spin.setDecimals(6)
        self.experimental_3d_patch_face_len_z_spin.setSingleStep(0.25)
        self.experimental_3d_patch_face_len_z_spin.setValue(2.0)
        self.experimental_3d_patch_face_len_z_spin.setToolTip(
            "Target z-face length for 3D patch cells (model units).\n"
            "Runtime resolves nz = ceil((zmax-zmin)/target_len_z)."
        )

        self.experimental_3d_patch_xmin_edit = _find_or_create_line_edit(
            "experimental_3d_patch_xmin_edit", "3D patch x min:"
        )
        self.experimental_3d_patch_xmax_edit = _find_or_create_line_edit(
            "experimental_3d_patch_xmax_edit", "3D patch x max:"
        )
        self.experimental_3d_patch_ymin_edit = _find_or_create_line_edit(
            "experimental_3d_patch_ymin_edit", "3D patch y min:"
        )
        self.experimental_3d_patch_ymax_edit = _find_or_create_line_edit(
            "experimental_3d_patch_ymax_edit", "3D patch y max:"
        )
        self.experimental_3d_patch_zmin_edit = _find_or_create_line_edit(
            "experimental_3d_patch_zmin_edit", "3D patch z min:"
        )
        self.experimental_3d_patch_zmax_edit = _find_or_create_line_edit(
            "experimental_3d_patch_zmax_edit", "3D patch z max:"
        )
        for _w in (
            self.experimental_3d_patch_xmin_edit,
            self.experimental_3d_patch_xmax_edit,
            self.experimental_3d_patch_ymin_edit,
            self.experimental_3d_patch_ymax_edit,
            self.experimental_3d_patch_zmin_edit,
            self.experimental_3d_patch_zmax_edit,
        ):
            _w.setPlaceholderText("auto from mesh")
        self.experimental_3d_patch_zmin_edit.setPlaceholderText("auto from terrain")

        self.experimental_3d_patch_set_roi_btn = _find_or_create_button(
            "experimental_3d_patch_set_roi_btn", "Set ROI From Current Mesh"
        )
        self.experimental_3d_patch_set_roi_btn.setToolTip(
            "Populate x/y/z min-max fields from the current 2D mesh extents.\n"
            "Used only when Experimental 3D patch mode is enabled."
        )
        try:
            self.experimental_3d_patch_set_roi_btn.clicked.disconnect(self._set_3d_patch_roi_from_mesh)
        except Exception:
            pass
        self.experimental_3d_patch_set_roi_btn.clicked.connect(self._set_3d_patch_roi_from_mesh)

        self.experimental_3d_patch_hint_lbl = _find_or_create_label(
            "experimental_3d_patch_hint_lbl",
            "3D patch ROI/resolution override (experimental): resolution is driven by target face lengths; "
            "leave min/max empty to auto-use mesh extents; z-min is terrain-driven when a DEM is available.",
        )
        self.experimental_3d_patch_hint_lbl.setWordWrap(True)

        self._experimental_3d_bc_widget_attrs = []
        self._experimental_3d_bc_signal_specs = []
        self.experimental_3d_patch_bc_widget = model_tab_page.findChild(
            QtWidgets.QWidget, "experimental_3d_patch_bc_widget"
        )
        if self.experimental_3d_patch_bc_widget is None:
            self.experimental_3d_patch_bc_widget = QtWidgets.QWidget()
            self.experimental_3d_patch_bc_widget.setObjectName("experimental_3d_patch_bc_widget")
        _ensure_row("3D patch face BCs:", self.experimental_3d_patch_bc_widget)

        existing_layout = self.experimental_3d_patch_bc_widget.layout()
        if isinstance(existing_layout, QtWidgets.QGridLayout):
            while existing_layout.count():
                item = existing_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            bc_grid = existing_layout
        else:
            bc_grid = QtWidgets.QGridLayout(self.experimental_3d_patch_bc_widget)
        bc_grid.setContentsMargins(0, 0, 0, 0)
        bc_grid.setHorizontalSpacing(4)
        bc_grid.setVerticalSpacing(2)

        bc_headers = ["Face", "Mode", "Q", "U", "V", "W", "VOF", "P"]
        for col, label in enumerate(bc_headers):
            hdr = QtWidgets.QLabel(label)
            hdr.setStyleSheet("font-weight: 600;")
            bc_grid.addWidget(hdr, 0, col)

        for row, face in enumerate(_SWE3D_PATCH_FACES, start=1):
            face_key = str(face).lower()
            bc_grid.addWidget(QtWidgets.QLabel(face), row, 0)

            mode_combo = QtWidgets.QComboBox()
            for mode_label, mode_value in _SWE3D_BC_MODE_OPTIONS:
                mode_combo.addItem(str(mode_label), int(mode_value))
            mode_combo.setCurrentIndex(0)
            mode_combo.setToolTip(
                "Boundary mode for this 3D patch face "
                "(0=Wall, 1=Inflow(U/V/W), 2=Outflow(zero-gradient), 3=Free Surface, 4=Volumetric Inlet(Q))."
            )
            mode_attr = f"experimental_3d_bc_{face_key}_mode_combo"
            setattr(self, mode_attr, mode_combo)
            self._experimental_3d_bc_widget_attrs.append(mode_attr)
            self._experimental_3d_bc_signal_specs.append((mode_attr, "currentIndexChanged"))
            bc_grid.addWidget(mode_combo, row, 1)

            for col, field_name in enumerate(("q", "u", "v", "w", "vof", "p"), start=2):
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(6)
                if field_name == "q":
                    spin.setRange(-1.0e9, 1.0e9)
                    spin.setSingleStep(1.0)
                elif field_name == "vof":
                    spin.setRange(0.0, 1.0)
                    spin.setSingleStep(0.05)
                elif field_name == "p":
                    spin.setRange(-1.0e9, 1.0e9)
                    spin.setSingleStep(1000.0)
                else:
                    spin.setRange(-1.0e6, 1.0e6)
                    spin.setSingleStep(0.1)
                spin.setValue(float(_SWE3D_BC_FIELD_DEFAULTS.get(field_name, 0.0)))
                spin.setMaximumWidth(100)
                if field_name == "q":
                    spin.setToolTip(
                        f"Prescribed volumetric flow rate Q [m^3/s] for {face} when mode=Volumetric Inlet (Q)."
                    )
                else:
                    spin.setToolTip(
                        f"Prescribed {field_name.upper()} state for {face} when mode uses boundary state input."
                    )
                field_attr = f"experimental_3d_bc_{face_key}_{field_name}_spin"
                setattr(self, field_attr, spin)
                self._experimental_3d_bc_widget_attrs.append(field_attr)
                self._experimental_3d_bc_signal_specs.append((field_attr, "valueChanged"))
                bc_grid.addWidget(spin, row, col)

        self.experimental_3d_patch_bc_hint_lbl = _find_or_create_label(
            "experimental_3d_patch_bc_hint_lbl",
            "3D face BCs map to BACKWATER_SWE3D_BC_<FACE>_<FIELD> env overrides; "
            "Outflow is zero-gradient, and Volumetric Inlet uses Q [m^3/s] for the face-normal inflow target.",
        )
        self.experimental_3d_patch_bc_hint_lbl.setWordWrap(True)

        if not self._experimental_3d_mode_supported:
            self.experimental_3d_mode_chk.setChecked(False)
            self.experimental_3d_mode_chk.setEnabled(False)
            self.experimental_3d_coupling_mode_combo.setEnabled(False)
            self.experimental_3d_mode_chk.setText("3D patch solver unavailable in this runtime")
            self.experimental_3d_mode_chk.setToolTip(
                "3D patch runtime enums (SolverModelOptions / SWE2DThreeD*) are unavailable.\n"
                "This session will run 2D only until the Python runtime imports swe2d_extensions fully."
            )

    def _bind_model_tab_3d_subgrid_drainage_controls(
        self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout
    ) -> None:
        patch_form = model_tab_page.findChild(QtWidgets.QFormLayout, "patch_3d_form") or param_form

        def _ensure_row(label: str, widget: QtWidgets.QWidget, target_form: Optional[QtWidgets.QFormLayout] = None) -> None:
            form = target_form or patch_form
            if form.indexOf(widget) >= 0:
                return
            form.addRow(label, widget)

        def _ensure_widget_row(widget: QtWidgets.QWidget, target_form: Optional[QtWidgets.QFormLayout] = None) -> None:
            form = target_form or patch_form
            if form.indexOf(widget) >= 0:
                return
            form.addRow(widget)

        def _find_or_create_check(
            name: str,
            label: str,
            text: str,
            target_form: Optional[QtWidgets.QFormLayout] = None,
        ) -> QtWidgets.QCheckBox:
            w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            _ensure_row(label, w, target_form)
            return w

        def _find_or_create_combo(
            name: str,
            label: str,
            target_form: Optional[QtWidgets.QFormLayout] = None,
        ) -> QtWidgets.QComboBox:
            w = model_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            _ensure_row(label, w, target_form)
            return w

        def _find_or_create_line_edit(
            name: str,
            label: str,
            text: str = "",
            target_form: Optional[QtWidgets.QFormLayout] = None,
        ) -> QtWidgets.QLineEdit:
            w = model_tab_page.findChild(QtWidgets.QLineEdit, name)
            if w is None:
                w = QtWidgets.QLineEdit(text)
                w.setObjectName(name)
            _ensure_row(label, w, target_form)
            return w

        def _find_or_create_double_spin(
            name: str,
            label: str,
            target_form: Optional[QtWidgets.QFormLayout] = None,
        ) -> QtWidgets.QDoubleSpinBox:
            w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w, target_form)
            return w

        def _find_or_create_spin(
            name: str,
            label: str,
            target_form: Optional[QtWidgets.QFormLayout] = None,
        ) -> QtWidgets.QSpinBox:
            w = model_tab_page.findChild(QtWidgets.QSpinBox, name)
            if w is None:
                w = QtWidgets.QSpinBox()
                w.setObjectName(name)
            _ensure_row(label, w, target_form)
            return w

        self.experimental_3d_obj_solids_chk = _find_or_create_check(
            "experimental_3d_obj_solids_chk", "3D sub-grid solids:", "Enable"
        )
        self.experimental_3d_obj_solids_chk.setChecked(True)
        self.experimental_3d_obj_solids_chk.setToolTip(
            "Upload static sub-grid geometry tensors (phi/ax/ay/az) before run start.\n"
            "Sources geometry from an OBJ instance point layer and optional terrain DEM solid fill."
        )

        self.experimental_3d_obj_method_combo = _find_or_create_combo(
            "experimental_3d_obj_method_combo", "3D sub-grid method:"
        )
        prev_data = self.experimental_3d_obj_method_combo.currentData()
        prev_text = self.experimental_3d_obj_method_combo.currentText()
        self.experimental_3d_obj_method_combo.blockSignals(True)
        try:
            self.experimental_3d_obj_method_combo.clear()
            self.experimental_3d_obj_method_combo.addItem("Fractional cut-cell (current)", "fractional_cutcell")
            self.experimental_3d_obj_method_combo.addItem("Porosity (Hirt-Nichols/FAVOR-like)", "favor1981_porosity")
            idx = self.experimental_3d_obj_method_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.experimental_3d_obj_method_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.experimental_3d_obj_method_combo.setCurrentIndex(idx)
        finally:
            self.experimental_3d_obj_method_combo.blockSignals(False)
        self.experimental_3d_obj_method_combo.setToolTip(
            "Static-obstacle tensor reconstruction method.\n"
            "Fractional cut-cell: current phi + pair-min face openness.\n"
            "Porosity/FAVOR-like: direct directional face-open sampling."
        )

        self.experimental_3d_obj_layer_combo = _find_or_create_combo(
            "experimental_3d_obj_layer_combo", "3D OBJ instances layer:"
        )
        prev_data = self.experimental_3d_obj_layer_combo.currentData()
        prev_text = self.experimental_3d_obj_layer_combo.currentText()
        self.experimental_3d_obj_layer_combo.blockSignals(True)
        try:
            self.experimental_3d_obj_layer_combo.clear()
            self.experimental_3d_obj_layer_combo.addItem("(none)", None)
            idx = self.experimental_3d_obj_layer_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.experimental_3d_obj_layer_combo.findText(prev_text)
            if idx >= 0:
                self.experimental_3d_obj_layer_combo.setCurrentIndex(idx)
        finally:
            self.experimental_3d_obj_layer_combo.blockSignals(False)

        self.experimental_3d_obj_path_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_path_field_edit", "3D OBJ path field:", "model_path"
        )
        if not str(self.experimental_3d_obj_path_field_edit.text() or "").strip():
            self.experimental_3d_obj_path_field_edit.setText("model_path")
        self.experimental_3d_obj_path_field_edit.setPlaceholderText("attribute with OBJ file path")

        self.experimental_3d_obj_default_path_edit = _find_or_create_line_edit(
            "experimental_3d_obj_default_path_edit", "3D OBJ fallback path:"
        )
        self.experimental_3d_obj_default_path_edit.setPlaceholderText("fallback OBJ path (optional)")

        self.experimental_3d_obj_scale_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_scale_field_edit", "3D OBJ scale field:", "scale"
        )
        if not str(self.experimental_3d_obj_scale_field_edit.text() or "").strip():
            self.experimental_3d_obj_scale_field_edit.setText("scale")
        self.experimental_3d_obj_scale_field_edit.setPlaceholderText("optional scale field (1 or sx,sy,sz)")

        self.experimental_3d_obj_yaw_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_yaw_field_edit", "3D OBJ yaw field:", "yaw_deg"
        )
        if not str(self.experimental_3d_obj_yaw_field_edit.text() or "").strip():
            self.experimental_3d_obj_yaw_field_edit.setText("yaw_deg")
        self.experimental_3d_obj_yaw_field_edit.setPlaceholderText("optional yaw field (degrees)")

        self.experimental_3d_obj_z_offset_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_z_offset_field_edit", "3D OBJ z-offset field:", "z_offset"
        )
        if not str(self.experimental_3d_obj_z_offset_field_edit.text() or "").strip():
            self.experimental_3d_obj_z_offset_field_edit.setText("z_offset")
        self.experimental_3d_obj_z_offset_field_edit.setPlaceholderText("optional per-instance z offset")

        self.experimental_3d_obj_inside_points_layer_combo = _find_or_create_combo(
            "experimental_3d_obj_inside_points_layer_combo", "3D OBJ outside-point layer:"
        )
        prev_data = self.experimental_3d_obj_inside_points_layer_combo.currentData()
        prev_text = self.experimental_3d_obj_inside_points_layer_combo.currentText()
        self.experimental_3d_obj_inside_points_layer_combo.blockSignals(True)
        try:
            self.experimental_3d_obj_inside_points_layer_combo.clear()
            self.experimental_3d_obj_inside_points_layer_combo.addItem("(none)", None)
            idx = self.experimental_3d_obj_inside_points_layer_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.experimental_3d_obj_inside_points_layer_combo.findText(prev_text)
            if idx >= 0:
                self.experimental_3d_obj_inside_points_layer_combo.setCurrentIndex(idx)
        finally:
            self.experimental_3d_obj_inside_points_layer_combo.blockSignals(False)

        self.experimental_3d_obj_instance_id_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_instance_id_field_edit", "3D OBJ instance id field:", "instance_id"
        )
        if not str(self.experimental_3d_obj_instance_id_field_edit.text() or "").strip():
            self.experimental_3d_obj_instance_id_field_edit.setText("instance_id")
        self.experimental_3d_obj_instance_id_field_edit.setPlaceholderText("optional OBJ instance id field")

        self.experimental_3d_obj_inside_id_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_inside_id_field_edit", "3D OBJ outside-point id field:", "instance_id"
        )
        if not str(self.experimental_3d_obj_inside_id_field_edit.text() or "").strip():
            self.experimental_3d_obj_inside_id_field_edit.setText("instance_id")
        self.experimental_3d_obj_inside_id_field_edit.setPlaceholderText("optional outside-point id field")

        self.experimental_3d_obj_inside_z_field_edit = _find_or_create_line_edit(
            "experimental_3d_obj_inside_z_field_edit", "3D OBJ outside-point z field:", "z"
        )
        if not str(self.experimental_3d_obj_inside_z_field_edit.text() or "").strip():
            self.experimental_3d_obj_inside_z_field_edit.setText("z")
        self.experimental_3d_obj_inside_z_field_edit.setPlaceholderText("optional outside-point z field")

        self.experimental_3d_obj_use_terrain_chk = _find_or_create_check(
            "experimental_3d_obj_use_terrain_chk", "3D terrain solid:", "Use terrain layer as bed solid"
        )
        self.experimental_3d_obj_use_terrain_chk.setChecked(True)
        self.experimental_3d_obj_use_terrain_chk.setToolTip(
            "Treat cells below sampled terrain DEM elevation as solid (phi=0)."
        )

        self.experimental_3d_obj_ab_compare_chk = _find_or_create_check(
            "experimental_3d_obj_ab_compare_chk", "3D A/B compare:", "A/B compare methods (startup probe)"
        )
        self.experimental_3d_obj_ab_compare_chk.setChecked(False)
        self.experimental_3d_obj_ab_compare_chk.setToolTip(
            "Run a short pre-run probe on temporary backends to compare fractional cut-cell and FAVOR-like methods.\n"
            "Logs mass drift proxy, max Courant, p_max_abs, and u_rms deltas before the main run starts."
        )

        self.experimental_3d_obj_ab_probe_steps_spin = _find_or_create_spin(
            "experimental_3d_obj_ab_probe_steps_spin", "3D A/B probe steps:"
        )
        self.experimental_3d_obj_ab_probe_steps_spin.setRange(1, 64)
        self.experimental_3d_obj_ab_probe_steps_spin.setValue(8)
        self.experimental_3d_obj_ab_probe_steps_spin.setToolTip(
            "Number of adaptive 3D probe steps used for each obstacle method in A/B compare mode."
        )

        self.experimental_3d_obj_export_obj_chk = _find_or_create_check(
            "experimental_3d_obj_export_obj_chk", "3D export voxel shell OBJ:", "Export voxelized solid shell OBJ"
        )
        self.experimental_3d_obj_export_obj_chk.setChecked(False)
        self.experimental_3d_obj_export_obj_chk.setToolTip(
            "Write the reconstructed solid representation (from phi thresholding) as an OBJ mesh for inspection."
        )

        self.experimental_3d_obj_export_obj_path_edit = _find_or_create_line_edit(
            "experimental_3d_obj_export_obj_path_edit", "3D solid OBJ export path:"
        )
        self.experimental_3d_obj_export_obj_path_edit.setPlaceholderText("optional OBJ output path (auto if empty)")

        self.experimental_3d_geom_sanitize_chk = _find_or_create_check(
            "experimental_3d_geom_sanitize_chk",
            "3D sanitize tensors:",
            "Sanitize upload tensors (clamp/snap tiny phi/area)",
        )
        self.experimental_3d_geom_sanitize_chk.setChecked(True)
        self.experimental_3d_geom_sanitize_chk.setToolTip(
            "Preprocess uploaded phi/ax/ay/az tensors for numerical robustness.\n"
            "Clamps all tensors to [0,1], snaps tiny phi cells to solid, and snaps tiny face-open areas to zero."
        )

        self.experimental_3d_geom_phi_snap_spin = _find_or_create_double_spin(
            "experimental_3d_geom_phi_snap_spin", "3D sanitize phi snap min:"
        )
        self.experimental_3d_geom_phi_snap_spin.setRange(0.0, 1.0)
        self.experimental_3d_geom_phi_snap_spin.setDecimals(6)
        self.experimental_3d_geom_phi_snap_spin.setSingleStep(0.001)
        self.experimental_3d_geom_phi_snap_spin.setValue(0.005)
        self.experimental_3d_geom_phi_snap_spin.setToolTip(
            "If phi < threshold, the cell is snapped to solid (phi=0) during geometry upload.\n"
            "Default is conservative to avoid over-sanitizing valid cut cells."
        )

        self.experimental_3d_geom_area_snap_spin = _find_or_create_double_spin(
            "experimental_3d_geom_area_snap_spin", "3D sanitize area snap min:"
        )
        self.experimental_3d_geom_area_snap_spin.setRange(0.0, 1.0)
        self.experimental_3d_geom_area_snap_spin.setDecimals(6)
        self.experimental_3d_geom_area_snap_spin.setSingleStep(0.001)
        self.experimental_3d_geom_area_snap_spin.setValue(0.01)
        self.experimental_3d_geom_area_snap_spin.setToolTip(
            "If ax/ay/az < threshold, the face-open fraction is snapped to zero during upload.\n"
            "Default is conservative and mainly targets sliver openings."
        )

        self.godunov_mode_combo = _find_or_create_combo(
            "godunov_mode_combo", "GPU solver mode:", param_form
        )
        prev_data = self.godunov_mode_combo.currentData()
        prev_text = self.godunov_mode_combo.currentText()
        self.godunov_mode_combo.blockSignals(True)
        try:
            self.godunov_mode_combo.clear()
            self.godunov_mode_combo.addItem("Current GPU solver", int(GodunovSolverMode.CURRENT_GPU_STEP))
            self.godunov_mode_combo.addItem("Godunov rollout (2nd-order)", int(GodunovSolverMode.GODUNOV_ROLLOUT))
            idx = self.godunov_mode_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.godunov_mode_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.godunov_mode_combo.setCurrentIndex(idx)
        finally:
            self.godunov_mode_combo.blockSignals(False)
        self.godunov_mode_combo.setToolTip(
            "Select the solver implementation used by the GPU path.\n"
            "Current GPU solver: existing production path.\n"
            "Godunov rollout: enables the second-order rollout configuration and\n"
            "keeps the native solver on the migration path for the new FVM mode."
        )

        self.degen_mode_combo = _find_or_create_combo("degen_mode_combo", "Degenerate cell mode:", param_form)
        prev_data = self.degen_mode_combo.currentData()
        prev_text = self.degen_mode_combo.currentText()
        self.degen_mode_combo.blockSignals(True)
        try:
            self.degen_mode_combo.clear()
            for _label, _val in [
                ("None (max_inv_area cap)", 0),
                ("Skip (permanently inactive)", 1),
                ("Repair (neighbor-avg inv_area)", 2),
                ("Merge (redirect flux to owner)", 3),
            ]:
                self.degen_mode_combo.addItem(_label, int(_val))
            idx = self.degen_mode_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.degen_mode_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.degen_mode_combo.setCurrentIndex(idx)
        finally:
            self.degen_mode_combo.blockSignals(False)
        self.degen_mode_combo.setToolTip(
            "Degenerate cell handling mode (cells with area below 1/max_inv_area).\n"
            "None: existing max_inv_area cap in update kernel (default).\n"
            "Skip: permanently exclude degenerate cells from all flux/update.\n"
            "Repair: replace degenerate cell inv_area with neighbor average;\n"
            "  keeps them in physics with sane CFL contribution.\n"
            "Merge: redirect flux accumulation to largest non-degenerate neighbor."
        )

        self.coupling_loop_combo = _find_or_create_combo("coupling_loop_combo", "Coupling loop:", param_form)
        prev_data = self.coupling_loop_combo.currentData()
        prev_text = self.coupling_loop_combo.currentText()
        self.coupling_loop_combo.blockSignals(True)
        try:
            self.coupling_loop_combo.clear()
            self.coupling_loop_combo.addItem("CPU coupling loop (reference)", "cpu")
            self.coupling_loop_combo.addItem("CUDA coupling loop (source assembly)", "cuda")
            idx = self.coupling_loop_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.coupling_loop_combo.findText(prev_text)
            if idx < 0:
                idx = min(1, max(0, self.coupling_loop_combo.count() - 1))
            if idx >= 0:
                self.coupling_loop_combo.setCurrentIndex(idx)
        finally:
            self.coupling_loop_combo.blockSignals(False)
        self.coupling_loop_combo.setToolTip(
            "Select coupling source assembly mode.\n"
            "CPU: Python reference path for drainage/structure source rates.\n"
            "CUDA: uses native CUDA kernel for per-cell source assembly when available;\n"
            "falls back to CPU reference automatically if CUDA binding/device is unavailable."
        )

        self.drainage_solver_mode_combo = _find_or_create_combo(
            "drainage_solver_mode_combo", "Drainage equation set:", param_form
        )
        prev_data = self.drainage_solver_mode_combo.currentData()
        prev_text = self.drainage_solver_mode_combo.currentText()
        self.drainage_solver_mode_combo.blockSignals(True)
        try:
            self.drainage_solver_mode_combo.clear()
            self.drainage_solver_mode_combo.addItem("EGL (Bernoulli + minor losses)", int(0))
            self.drainage_solver_mode_combo.addItem("Diffusion wave", int(1))
            self.drainage_solver_mode_combo.addItem("Dynamic Saint-Venant", int(2))
            idx = self.drainage_solver_mode_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.drainage_solver_mode_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.drainage_solver_mode_combo.setCurrentIndex(idx)
        finally:
            self.drainage_solver_mode_combo.blockSignals(False)
        self.drainage_solver_mode_combo.setToolTip(
            "Drainage 1D equation set.\n"
            "EGL: Bernoulli + Manning + minor losses.\n"
            "Diffusion: slope-driven Manning flow.\n"
            "Dynamic: semi-implicit Saint-Venant momentum update."
        )

        self.drainage_backend_combo = _find_or_create_combo(
            "drainage_backend_combo", "Drainage solver backend:", param_form
        )
        prev_data = self.drainage_backend_combo.currentData()
        prev_text = self.drainage_backend_combo.currentText()
        self.drainage_backend_combo.blockSignals(True)
        try:
            self.drainage_backend_combo.clear()
            self.drainage_backend_combo.addItem("CPU drainage solver (reference)", "cpu")
            self.drainage_backend_combo.addItem("GPU drainage solver (CUDA)", "gpu")
            idx = self.drainage_backend_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.drainage_backend_combo.findText(prev_text)
            if idx < 0:
                idx = min(1, max(0, self.drainage_backend_combo.count() - 1))
            if idx >= 0:
                self.drainage_backend_combo.setCurrentIndex(idx)
        finally:
            self.drainage_backend_combo.blockSignals(False)
        self.drainage_backend_combo.setToolTip(
            "Select drainage network solver backend.\n"
            "CPU: Python reference implementation.\n"
            "GPU: native CUDA drainage solver for EGL/Diffusion/Dynamic modes;\n"
            "falls back to CPU path when CUDA drainage bindings are unavailable."
        )

        self.drainage_gpu_method_combo = _find_or_create_combo(
            "drainage_gpu_method_combo", "Drainage GPU method:", param_form
        )
        prev_data = self.drainage_gpu_method_combo.currentData()
        prev_text = self.drainage_gpu_method_combo.currentText()
        self.drainage_gpu_method_combo.blockSignals(True)
        try:
            self.drainage_gpu_method_combo.clear()
            self.drainage_gpu_method_combo.addItem("Per-step GPU drainage (fast for sparse exchange)", "step")
            self.drainage_gpu_method_combo.addItem("Native iterative GPU drainage (batched substeps)", "iterative")
            idx = self.drainage_gpu_method_combo.findData(prev_data)
            if idx < 0 and prev_text:
                idx = self.drainage_gpu_method_combo.findText(prev_text)
            if idx < 0:
                idx = 0
            if idx >= 0:
                self.drainage_gpu_method_combo.setCurrentIndex(idx)
        finally:
            self.drainage_gpu_method_combo.blockSignals(False)
        self.drainage_gpu_method_combo.setToolTip(
            "Select GPU drainage coupling method when drainage backend is GPU.\n"
            "Per-step: calls the GPU drainage step once per substep/iteration from Python.\n"
            "Native iterative: runs substeps and implicit iterations in one native call.\n"
            "Use native iterative for dense/active drainage exchange; per-step can be faster\n"
            "when exchange is sparse or mostly inactive."
        )

        self.drainage_coupling_substeps_spin = _find_or_create_spin(
            "drainage_coupling_substeps_spin", "Drainage substeps:", param_form
        )
        self.drainage_coupling_substeps_spin.setRange(1, 256)
        self.drainage_coupling_substeps_spin.setValue(1)
        self.drainage_coupling_substeps_spin.setToolTip(
            "Fixed number of 1D drainage substeps taken per 2D coupling step.\n"
            "Increase this for stiff drainage networks or dynamic-wave runs."
        )

        self.drainage_max_coupling_substeps_spin = _find_or_create_spin(
            "drainage_max_coupling_substeps_spin", "Drainage max adaptive substeps:", param_form
        )
        self.drainage_max_coupling_substeps_spin.setRange(1, 1024)
        self.drainage_max_coupling_substeps_spin.setValue(64)
        self.drainage_max_coupling_substeps_spin.setToolTip(
            "Maximum adaptive drainage substeps allowed when the 1D stability\n"
            "controller tightens the drainage timestep automatically."
        )

        self.drainage_head_deadband_spin = _find_or_create_double_spin(
            "drainage_head_deadband_spin", "Drainage head deadband:", param_form
        )
        self.drainage_head_deadband_spin.setRange(0.0, 10.0)
        self.drainage_head_deadband_spin.setDecimals(6)
        self.drainage_head_deadband_spin.setValue(1.0e-3)
        self.drainage_head_deadband_spin.setToolTip(
            "Head deadband used before drainage link and inlet exchange updates.\n"
            "Larger values reduce chatter near balanced states."
        )

        self.drainage_dynamic_relaxation_spin = _find_or_create_double_spin(
            "drainage_dynamic_relaxation_spin", "Drainage dynamic relaxation:", param_form
        )
        self.drainage_dynamic_relaxation_spin.setRange(0.0, 1.0)
        self.drainage_dynamic_relaxation_spin.setDecimals(3)
        self.drainage_dynamic_relaxation_spin.setSingleStep(0.05)
        self.drainage_dynamic_relaxation_spin.setValue(1.0)
        self.drainage_dynamic_relaxation_spin.setToolTip(
            "Dynamic-wave flow relaxation factor.\n"
            "1.0 keeps the full update; lower values damp oscillatory link-flow response."
        )

        self.drainage_adaptive_depth_fraction_spin = _find_or_create_double_spin(
            "drainage_adaptive_depth_fraction_spin", "Drainage adaptive depth fraction:", param_form
        )
        self.drainage_adaptive_depth_fraction_spin.setRange(0.001, 1.0)
        self.drainage_adaptive_depth_fraction_spin.setDecimals(3)
        self.drainage_adaptive_depth_fraction_spin.setSingleStep(0.01)
        self.drainage_adaptive_depth_fraction_spin.setValue(0.2)
        self.drainage_adaptive_depth_fraction_spin.setToolTip(
            "Adaptive drainage substepping threshold based on fractional node-depth\n"
            "change per substep. Lower values are more conservative."
        )

        self.drainage_adaptive_wave_courant_spin = _find_or_create_double_spin(
            "drainage_adaptive_wave_courant_spin", "Drainage adaptive wave Courant:", param_form
        )
        self.drainage_adaptive_wave_courant_spin.setRange(0.001, 10.0)
        self.drainage_adaptive_wave_courant_spin.setDecimals(3)
        self.drainage_adaptive_wave_courant_spin.setSingleStep(0.05)
        self.drainage_adaptive_wave_courant_spin.setValue(0.5)
        self.drainage_adaptive_wave_courant_spin.setToolTip(
            "Adaptive drainage substepping target for dynamic-wave links based on\n"
            "wave Courant number. Lower values are more conservative."
        )

        self.drainage_implicit_iters_spin = _find_or_create_spin(
            "drainage_implicit_iters_spin", "Drainage implicit iterations (GPU):", param_form
        )
        self.drainage_implicit_iters_spin.setRange(1, 8)
        self.drainage_implicit_iters_spin.setValue(2)
        self.drainage_implicit_iters_spin.setToolTip(
            "Number of implicit predictor/corrector inner iterations per drainage substep\n"
            "(GPU path only). 1 = explicit single-pass; 2-4 gives better mass conservation\n"
            "at ~linear cost per extra iteration."
        )

        self.drainage_implicit_relax_spin = _find_or_create_double_spin(
            "drainage_implicit_relax_spin", "Drainage implicit relaxation (GPU):", param_form
        )
        self.drainage_implicit_relax_spin.setRange(0.1, 1.0)
        self.drainage_implicit_relax_spin.setDecimals(2)
        self.drainage_implicit_relax_spin.setSingleStep(0.05)
        self.drainage_implicit_relax_spin.setValue(0.5)
        self.drainage_implicit_relax_spin.setToolTip(
            "Relaxation factor for implicit coupling iterates (GPU path only).\n"
            "1.0 = no relaxation (full update); 0.5 damps oscillations between iterates."
        )

        self.gpu_default_lbl = model_tab_page.findChild(QtWidgets.QLabel, "gpu_default_lbl")
        if self.gpu_default_lbl is None:
            self.gpu_default_lbl = QtWidgets.QLabel(
                "GPU is attempted by default when supported by the native backend."
            )
            self.gpu_default_lbl.setObjectName("gpu_default_lbl")
            _ensure_widget_row(self.gpu_default_lbl, param_form)
        self.gpu_default_lbl.setWordWrap(True)

        self.unit_system_lbl = model_tab_page.findChild(QtWidgets.QLabel, "unit_system_lbl")
        if self.unit_system_lbl is None:
            self.unit_system_lbl = QtWidgets.QLabel("Unit system: auto")
            self.unit_system_lbl.setObjectName("unit_system_lbl")
            _ensure_widget_row(self.unit_system_lbl, param_form)
        self.unit_system_lbl.setWordWrap(True)

    def _build_topology_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        topo_group = QtWidgets.QGroupBox("Topology Meshing (Face-centric)")
        topo_layout = QtWidgets.QGridLayout(topo_group)
        topo_layout.setObjectName("topo_layout")

        root_layout.addWidget(topo_group)
        return root

    def _bind_topology_tab_static_controls(self, topology_tab_page: QtWidgets.QWidget, topo_layout: QtWidgets.QGridLayout) -> None:
        def _find_or_create_label(name: str, text: str) -> QtWidgets.QLabel:
            w = topology_tab_page.findChild(QtWidgets.QLabel, name)
            if w is None:
                w = QtWidgets.QLabel(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            return w

        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = topology_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        def _ensure(widget: QtWidgets.QWidget, row: int, col: int, row_span: int = 1, col_span: int = 1) -> None:
            if topo_layout.indexOf(widget) >= 0:
                return
            topo_layout.addWidget(widget, row, col, row_span, col_span)

        static_labels = [
            ("topo_nodes_lbl", "Topology nodes layer:", 0),
            ("topo_arcs_lbl", "Topology arcs layer:", 1),
            ("topo_regions_lbl", "Topology regions layer:", 2),
            ("topo_constraints_lbl", "Constraints layer:", 3),
            ("topo_quad_edges_lbl", "Quad edges / transition layers:", 4),
            ("topo_backend_lbl", "Meshing backend:", 5),
            ("topo_default_size_lbl", "Default target size:", 6),
            ("topo_default_cell_type_lbl", "Default cell type:", 7),
            ("topo_gmsh_controls_lbl", "Gmsh advanced controls:", 8),
            ("topo_quality_controls_lbl", "Quality controls (Gmsh + TQMesh):", 9),
        ]
        for name, text, row in static_labels:
            _ensure(_find_or_create_label(name, text), row, 0)

        self.topo_validate_btn = _find_or_create_button("topo_validate_btn", "Summarize Layer Controls")
        self.topo_edit_regions_btn = _find_or_create_button("topo_edit_regions_btn", "Edit Region Controls")
        self.topo_edit_quad_edges_btn = _find_or_create_button("topo_edit_quad_edges_btn", "Edit Transition Layers")
        self.topo_controls_summary_lbl = _find_or_create_label(
            "topo_controls_summary_lbl",
            "Topology-layer controls: use multiple region polygons for multiple blocks. "
            "Use region target_size + cell_type, edge_len_1..4 for cartesian/quadrilateral block spacing, "
            "use region interior rings or empty regions/constraints for holes, "
            "and quad-edge n_layers / first_height / growth_rate for TQMesh transition layers.",
        )
        self.topo_export_template_btn = _find_or_create_button("topo_export_template_btn", "Create Topology Template Layers")
        self.topo_generate_btn = _find_or_create_button("topo_generate_btn", "Generate Mesh From Topology Layers")
        self.topo_terminate_btn = _find_or_create_button("topo_terminate_btn", "Terminate Mesh Run")
        self.topo_status_lbl = _find_or_create_label(
            "topo_status_lbl", "Select regions layer and generate face-centric mesh"
        )

        _ensure(self.topo_validate_btn, 10, 0, 1, 2)
        _ensure(self.topo_edit_regions_btn, 11, 0)
        _ensure(self.topo_edit_quad_edges_btn, 11, 1)
        _ensure(self.topo_controls_summary_lbl, 12, 0, 1, 2)
        _ensure(self.topo_export_template_btn, 13, 0, 1, 2)
        _ensure(self.topo_generate_btn, 14, 0)
        _ensure(self.topo_terminate_btn, 14, 1)
        _ensure(self.topo_status_lbl, 15, 0, 1, 2)

        self.topo_controls_summary_lbl.setWordWrap(True)
        self.topo_status_lbl.setWordWrap(True)
        self.topo_terminate_btn.setEnabled(False)

        for btn, cb in (
            (self.topo_validate_btn, self._update_topology_control_summary),
            (self.topo_edit_regions_btn, self._open_topology_region_table),
            (self.topo_edit_quad_edges_btn, self._open_topology_quad_edge_table),
            (self.topo_export_template_btn, self._create_topology_template_layers),
            (self.topo_generate_btn, self._generate_mesh_from_topology_layers),
            (self.topo_terminate_btn, self._on_terminate_topology_mesh),
        ):
            try:
                btn.clicked.disconnect(cb)
            except Exception:
                pass
            btn.clicked.connect(cb)

    def _bind_topology_tab_dynamic_controls(self, topology_tab_page: QtWidgets.QWidget, topo_layout: QtWidgets.QGridLayout) -> None:
        def _ensure(widget: QtWidgets.QWidget, row: int, col: int, row_span: int = 1, col_span: int = 1) -> None:
            if topo_layout.indexOf(widget) >= 0:
                return
            topo_layout.addWidget(widget, row, col, row_span, col_span)

        def _find_or_create_combo(name: str, row: int) -> QtWidgets.QComboBox:
            w = topology_tab_page.findChild(QtWidgets.QComboBox, name)
            if w is None:
                w = QtWidgets.QComboBox()
                w.setObjectName(name)
            _ensure(w, row, 1)
            return w

        def _set_combo_items(
            combo: QtWidgets.QComboBox,
            items: List[Tuple[str, object]],
            default_data: Optional[object] = None,
        ) -> None:
            prev_data = combo.currentData()
            prev_text = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                for label, data in items:
                    combo.addItem(label, data)
                idx = -1
                if prev_data is not None:
                    idx = combo.findData(prev_data)
                if idx < 0 and default_data is not None:
                    idx = combo.findData(default_data)
                if idx < 0 and prev_text:
                    idx = combo.findText(prev_text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            finally:
                combo.blockSignals(False)

        def _find_or_create_double_spin(name: str) -> QtWidgets.QDoubleSpinBox:
            w = topology_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
            if w is None:
                w = QtWidgets.QDoubleSpinBox()
                w.setObjectName(name)
            return w

        def _find_or_create_spin(name: str) -> QtWidgets.QSpinBox:
            w = topology_tab_page.findChild(QtWidgets.QSpinBox, name)
            if w is None:
                w = QtWidgets.QSpinBox()
                w.setObjectName(name)
            return w

        def _find_or_create_line_edit(name: str, text: str) -> QtWidgets.QLineEdit:
            w = topology_tab_page.findChild(QtWidgets.QLineEdit, name)
            if w is None:
                w = QtWidgets.QLineEdit(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            return w

        def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
            w = topology_tab_page.findChild(QtWidgets.QCheckBox, name)
            if w is None:
                w = QtWidgets.QCheckBox(text)
                w.setObjectName(name)
            if not str(w.text() or "").strip():
                w.setText(text)
            return w

        def _find_or_create_form_container(name: str, row: int) -> QtWidgets.QFormLayout:
            container = topology_tab_page.findChild(QtWidgets.QWidget, name)
            if container is None:
                container = QtWidgets.QWidget()
                container.setObjectName(name)
            _ensure(container, row, 1)
            layout = container.layout()
            if not isinstance(layout, QtWidgets.QFormLayout):
                layout = QtWidgets.QFormLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            return layout

        def _reconnect(signal: object, callback: Callable[[], None]) -> None:
            try:
                signal.disconnect(callback)
            except Exception:
                pass
            signal.connect(callback)

        self.topo_nodes_combo = _find_or_create_combo("topo_nodes_combo", 0)
        self.topo_arcs_combo = _find_or_create_combo("topo_arcs_combo", 1)
        self.topo_regions_combo = _find_or_create_combo("topo_regions_combo", 2)
        self.topo_constraints_combo = _find_or_create_combo("topo_constraints_combo", 3)
        self.topo_quad_edges_combo = _find_or_create_combo("topo_quad_edges_combo", 4)
        self.topo_backend_combo = _find_or_create_combo("topo_backend_combo", 5)
        self.topo_default_size_spin = _find_or_create_double_spin("topo_default_size_spin")
        _ensure(self.topo_default_size_spin, 6, 1)
        self.topo_default_cell_type_combo = _find_or_create_combo("topo_default_cell_type_combo", 7)

        _set_combo_items(self.topo_constraints_combo, [("(none)", None)], default_data=None)
        _set_combo_items(self.topo_quad_edges_combo, [("(none)", None)], default_data=None)

        _gmsh_label = "Gmsh (recommended)" if _gmsh_available() else "Gmsh (install: pip install gmsh)"
        _tqmesh_label = "TQMesh (advancing-front, built-in)" if _tqmesh_available() else "TQMesh (build plugin to enable)"
        _set_combo_items(
            self.topo_backend_combo,
            [
                (_gmsh_label, "gmsh"),
                ("Structured (built-in fallback)", "structured"),
                (_tqmesh_label, "tqmesh"),
            ],
            default_data="gmsh",
        )
        _set_combo_items(
            self.topo_default_cell_type_combo,
            [
                ("triangular", "triangular"),
                ("quadrilateral", "quadrilateral"),
                ("cartesian", "cartesian"),
                ("empty", "empty"),
            ],
            default_data="triangular",
        )

        self.topo_default_size_spin.setRange(0.01, 1.0e6)
        self.topo_default_size_spin.setDecimals(3)
        self.topo_default_size_spin.setValue(20.0)

        gmsh_form = _find_or_create_form_container("topo_gmsh_controls_widget", 8)
        quality_form = _find_or_create_form_container("topo_quality_controls_widget", 9)

        self.topo_gmsh_tri_algo_combo = topology_tab_page.findChild(QtWidgets.QComboBox, "topo_gmsh_tri_algo_combo")
        if self.topo_gmsh_tri_algo_combo is None:
            self.topo_gmsh_tri_algo_combo = QtWidgets.QComboBox()
            self.topo_gmsh_tri_algo_combo.setObjectName("topo_gmsh_tri_algo_combo")
            gmsh_form.addRow("Triangle algorithm:", self.topo_gmsh_tri_algo_combo)
        _set_combo_items(
            self.topo_gmsh_tri_algo_combo,
            [
                ("Frontal-Delaunay (quality)", 6),
                ("Delaunay (faster)", 5),
            ],
            default_data=6,
        )

        self.topo_gmsh_quad_algo_combo = topology_tab_page.findChild(QtWidgets.QComboBox, "topo_gmsh_quad_algo_combo")
        if self.topo_gmsh_quad_algo_combo is None:
            self.topo_gmsh_quad_algo_combo = QtWidgets.QComboBox()
            self.topo_gmsh_quad_algo_combo.setObjectName("topo_gmsh_quad_algo_combo")
            gmsh_form.addRow("Quadrilateral algorithm:", self.topo_gmsh_quad_algo_combo)
        _set_combo_items(
            self.topo_gmsh_quad_algo_combo,
            [
                ("Frontal + Blossom recombine", 6),
                ("Delaunay + Blossom recombine", 5),
                ("Packing of Parallelograms", 9),
            ],
            default_data=6,
        )

        self.topo_gmsh_recombine_algo_combo = topology_tab_page.findChild(QtWidgets.QComboBox, "topo_gmsh_recombine_algo_combo")
        if self.topo_gmsh_recombine_algo_combo is None:
            self.topo_gmsh_recombine_algo_combo = QtWidgets.QComboBox()
            self.topo_gmsh_recombine_algo_combo.setObjectName("topo_gmsh_recombine_algo_combo")
            gmsh_form.addRow("Recombine algorithm:", self.topo_gmsh_recombine_algo_combo)
        _set_combo_items(
            self.topo_gmsh_recombine_algo_combo,
            [
                ("Simple", 0),
                ("Blossom", 1),
                ("Simple full-quad", 2),
            ],
            default_data=1,
        )

        self.topo_gmsh_smoothing_spin = _find_or_create_spin("topo_gmsh_smoothing_spin")
        self.topo_gmsh_smoothing_spin.setRange(0, 100)
        self.topo_gmsh_smoothing_spin.setValue(5)
        if self.topo_gmsh_smoothing_spin.parent() is None:
            gmsh_form.addRow("Smoothing passes:", self.topo_gmsh_smoothing_spin)

        self.topo_gmsh_optimize_iters_spin = _find_or_create_spin("topo_gmsh_optimize_iters_spin")
        self.topo_gmsh_optimize_iters_spin.setRange(0, 100)
        self.topo_gmsh_optimize_iters_spin.setValue(3)
        if self.topo_gmsh_optimize_iters_spin.parent() is None:
            gmsh_form.addRow("Optimize iterations:", self.topo_gmsh_optimize_iters_spin)

        self.topo_gmsh_verbosity_spin = _find_or_create_spin("topo_gmsh_verbosity_spin")
        self.topo_gmsh_verbosity_spin.setRange(0, 10)
        self.topo_gmsh_verbosity_spin.setValue(1)
        if self.topo_gmsh_verbosity_spin.parent() is None:
            gmsh_form.addRow("Verbosity:", self.topo_gmsh_verbosity_spin)

        self.topo_gmsh_optimize_netgen_chk = _find_or_create_check("topo_gmsh_optimize_netgen_chk", "Enable Netgen optimize")
        if self.topo_gmsh_optimize_netgen_chk.parent() is None:
            gmsh_form.addRow(self.topo_gmsh_optimize_netgen_chk)

        self.topo_gmsh_quality_enable_chk = _find_or_create_check(
            "topo_gmsh_quality_enable_chk", "Enable Gmsh iterative quality loop"
        )
        self.topo_gmsh_quality_enable_chk.setChecked(False)
        if self.topo_gmsh_quality_enable_chk.parent() is None:
            quality_form.addRow(self.topo_gmsh_quality_enable_chk)

        self.topo_gmsh_quality_max_iters_spin = _find_or_create_spin("topo_gmsh_quality_max_iters_spin")
        self.topo_gmsh_quality_max_iters_spin.setRange(1, 50)
        self.topo_gmsh_quality_max_iters_spin.setValue(6)
        if self.topo_gmsh_quality_max_iters_spin.parent() is None:
            quality_form.addRow("Gmsh max attempts:", self.topo_gmsh_quality_max_iters_spin)

        self.topo_gmsh_quality_time_limit_spin = _find_or_create_double_spin("topo_gmsh_quality_time_limit_spin")
        self.topo_gmsh_quality_time_limit_spin.setRange(1.0, 3600.0)
        self.topo_gmsh_quality_time_limit_spin.setDecimals(1)
        self.topo_gmsh_quality_time_limit_spin.setValue(60.0)
        if self.topo_gmsh_quality_time_limit_spin.parent() is None:
            quality_form.addRow("Gmsh time budget (s):", self.topo_gmsh_quality_time_limit_spin)

        self.topo_quality_min_angle_spin = _find_or_create_double_spin("topo_quality_min_angle_spin")
        self.topo_quality_min_angle_spin.setRange(0.0, 89.0)
        self.topo_quality_min_angle_spin.setDecimals(1)
        self.topo_quality_min_angle_spin.setValue(5.0)
        if self.topo_quality_min_angle_spin.parent() is None:
            quality_form.addRow("Min angle (deg):", self.topo_quality_min_angle_spin)

        self.topo_quality_max_aspect_spin = _find_or_create_double_spin("topo_quality_max_aspect_spin")
        self.topo_quality_max_aspect_spin.setRange(1.0, 1.0e4)
        self.topo_quality_max_aspect_spin.setDecimals(2)
        self.topo_quality_max_aspect_spin.setValue(20.0)
        if self.topo_quality_max_aspect_spin.parent() is None:
            quality_form.addRow("Max aspect ratio:", self.topo_quality_max_aspect_spin)

        self.topo_quality_max_non_orth_spin = _find_or_create_double_spin("topo_quality_max_non_orth_spin")
        self.topo_quality_max_non_orth_spin.setRange(1.0, 89.9)
        self.topo_quality_max_non_orth_spin.setDecimals(1)
        self.topo_quality_max_non_orth_spin.setValue(82.0)
        if self.topo_quality_max_non_orth_spin.parent() is None:
            quality_form.addRow("Max non-orthogonality (deg):", self.topo_quality_max_non_orth_spin)

        self.topo_quality_min_area_edit = _find_or_create_line_edit("topo_quality_min_area_edit", "1e-14")
        if self.topo_quality_min_area_edit.parent() is None:
            quality_form.addRow("Min area / bbox area:", self.topo_quality_min_area_edit)

        self.topo_quality_size_scales_edit = _find_or_create_line_edit("topo_quality_size_scales_edit", "1.0,0.9,0.8,0.7")
        if self.topo_quality_size_scales_edit.parent() is None:
            quality_form.addRow("Retry size scales:", self.topo_quality_size_scales_edit)

        self.topo_quality_smooth_increments_edit = _find_or_create_line_edit("topo_quality_smooth_increments_edit", "0,2,4,6")
        if self.topo_quality_smooth_increments_edit.parent() is None:
            quality_form.addRow("Retry smooth increments:", self.topo_quality_smooth_increments_edit)

        self.topo_quality_strict_chk = _find_or_create_check("topo_quality_strict_chk", "Strict quality acceptance")
        if self.topo_quality_strict_chk.parent() is None:
            quality_form.addRow(self.topo_quality_strict_chk)

        _reconnect(self.topo_backend_combo.currentIndexChanged, self._update_topology_control_summary)
        _reconnect(self.topo_regions_combo.currentIndexChanged, self._update_topology_control_summary)
        _reconnect(self.topo_constraints_combo.currentIndexChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quad_edges_combo.currentIndexChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quality_min_angle_spin.valueChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quality_max_aspect_spin.valueChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quality_max_non_orth_spin.valueChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quality_min_area_edit.textChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quality_strict_chk.toggled, self._update_topology_control_summary)
        _reconnect(self.topo_quality_size_scales_edit.textChanged, self._update_topology_control_summary)
        _reconnect(self.topo_quality_smooth_increments_edit.textChanged, self._update_topology_control_summary)
        _reconnect(self.topo_gmsh_quality_enable_chk.toggled, self._update_topology_control_summary)
        _reconnect(self.topo_gmsh_quality_max_iters_spin.valueChanged, self._update_topology_control_summary)
        _reconnect(self.topo_gmsh_quality_time_limit_spin.valueChanged, self._update_topology_control_summary)

    def _build_run_tab_page(self) -> QtWidgets.QWidget:
        ui_path = self._forms_file_path("swe2d_run_tab.ui")
        run_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                run_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                run_tab_page = None
        if run_tab_page is None:
            run_tab_page = self._build_run_tab_page_fallback()
        self._bind_run_tab_controls(run_tab_page)
        return run_tab_page

    def _build_run_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        run_group = QtWidgets.QGroupBox("Run / Output")
        run_group.setObjectName("run_group")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        run_layout.setObjectName("run_layout")

        run_row = QtWidgets.QHBoxLayout()
        run_row.setObjectName("run_row_layout")
        preview_overrides_btn = QtWidgets.QPushButton("Preview Overrides")
        preview_overrides_btn.setObjectName("preview_overrides_btn")
        run_btn = QtWidgets.QPushButton("Run 2D Model")
        run_btn.setObjectName("run_btn")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setObjectName("cancel_btn")
        run_row.addWidget(preview_overrides_btn)
        run_row.addWidget(run_btn)
        run_row.addWidget(cancel_btn)
        run_layout.addLayout(run_row)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setObjectName("progress_bar")
        progress_bar.setValue(0)
        run_layout.addWidget(progress_bar)

        snap_row = QtWidgets.QHBoxLayout()
        snap_row.setObjectName("run_snapshot_row_layout")
        output_interval_lbl = QtWidgets.QLabel("Output interval (hr or HH:MM):")
        output_interval_lbl.setObjectName("output_interval_lbl")
        output_interval_edit = QtWidgets.QLineEdit("00:30")
        output_interval_edit.setObjectName("output_interval_edit")
        line_output_interval_lbl = QtWidgets.QLabel("Line output interval:")
        line_output_interval_lbl.setObjectName("line_output_interval_lbl")
        line_output_interval_edit = QtWidgets.QLineEdit("00:05")
        line_output_interval_edit.setObjectName("line_output_interval_edit")
        snapshot_btn = QtWidgets.QPushButton("Take Snapshot")
        snapshot_btn.setObjectName("snapshot_btn")
        snap_row.addWidget(output_interval_lbl)
        snap_row.addWidget(output_interval_edit)
        snap_row.addWidget(line_output_interval_lbl)
        snap_row.addWidget(line_output_interval_edit)
        snap_row.addWidget(snapshot_btn)
        run_layout.addLayout(snap_row)

        root_layout.addWidget(run_group)
        return root

    def _bind_run_tab_controls(self, run_tab_page: QtWidgets.QWidget) -> None:
        def _ensure_root_layout() -> QtWidgets.QVBoxLayout:
            layout = run_tab_page.layout()
            if isinstance(layout, QtWidgets.QVBoxLayout):
                return layout
            layout = QtWidgets.QVBoxLayout(run_tab_page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            return layout

        def _ensure_run_group() -> QtWidgets.QGroupBox:
            run_group = run_tab_page.findChild(QtWidgets.QGroupBox, "run_group")
            if run_group is not None:
                return run_group
            run_group = QtWidgets.QGroupBox("Run / Output")
            run_group.setObjectName("run_group")
            root_layout = _ensure_root_layout()
            root_layout.addWidget(run_group)
            return run_group

        def _ensure_run_layout(run_group: QtWidgets.QGroupBox) -> QtWidgets.QVBoxLayout:
            layout = run_group.layout()
            if isinstance(layout, QtWidgets.QVBoxLayout):
                return layout
            layout = QtWidgets.QVBoxLayout(run_group)
            layout.setObjectName("run_layout")
            return layout

        run_group = _ensure_run_group()
        run_layout = _ensure_run_layout(run_group)

        run_row = run_tab_page.findChild(QtWidgets.QHBoxLayout, "run_row_layout")
        if run_row is None:
            run_row = QtWidgets.QHBoxLayout()
            run_row.setObjectName("run_row_layout")
            run_layout.insertLayout(0, run_row)

        self.preview_overrides_btn = run_tab_page.findChild(QtWidgets.QPushButton, "preview_overrides_btn")
        if self.preview_overrides_btn is None:
            self.preview_overrides_btn = QtWidgets.QPushButton("Preview Overrides")
            self.preview_overrides_btn.setObjectName("preview_overrides_btn")
        if run_row.indexOf(self.preview_overrides_btn) < 0:
            run_row.addWidget(self.preview_overrides_btn)

        self.run_btn = run_tab_page.findChild(QtWidgets.QPushButton, "run_btn")
        if self.run_btn is None:
            self.run_btn = QtWidgets.QPushButton("Run 2D Model")
            self.run_btn.setObjectName("run_btn")
        if run_row.indexOf(self.run_btn) < 0:
            run_row.addWidget(self.run_btn)

        self.cancel_btn = run_tab_page.findChild(QtWidgets.QPushButton, "cancel_btn")
        if self.cancel_btn is None:
            self.cancel_btn = QtWidgets.QPushButton("Cancel")
            self.cancel_btn.setObjectName("cancel_btn")
        if run_row.indexOf(self.cancel_btn) < 0:
            run_row.addWidget(self.cancel_btn)

        self.progress_bar = run_tab_page.findChild(QtWidgets.QProgressBar, "progress_bar")
        if self.progress_bar is None:
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setObjectName("progress_bar")
            run_layout.addWidget(self.progress_bar)
        elif run_layout.indexOf(self.progress_bar) < 0:
            run_layout.addWidget(self.progress_bar)

        snap_row = run_tab_page.findChild(QtWidgets.QHBoxLayout, "run_snapshot_row_layout")
        if snap_row is None:
            snap_row = QtWidgets.QHBoxLayout()
            snap_row.setObjectName("run_snapshot_row_layout")
            run_layout.addLayout(snap_row)

        output_interval_lbl = run_tab_page.findChild(QtWidgets.QLabel, "output_interval_lbl")
        if output_interval_lbl is None:
            output_interval_lbl = QtWidgets.QLabel("Output interval (hr or HH:MM):")
            output_interval_lbl.setObjectName("output_interval_lbl")
        if snap_row.indexOf(output_interval_lbl) < 0:
            snap_row.addWidget(output_interval_lbl)

        self.output_interval_edit = run_tab_page.findChild(QtWidgets.QLineEdit, "output_interval_edit")
        if self.output_interval_edit is None:
            self.output_interval_edit = QtWidgets.QLineEdit("00:30")
            self.output_interval_edit.setObjectName("output_interval_edit")
        if snap_row.indexOf(self.output_interval_edit) < 0:
            snap_row.addWidget(self.output_interval_edit)

        line_output_interval_lbl = run_tab_page.findChild(QtWidgets.QLabel, "line_output_interval_lbl")
        if line_output_interval_lbl is None:
            line_output_interval_lbl = QtWidgets.QLabel("Line output interval:")
            line_output_interval_lbl.setObjectName("line_output_interval_lbl")
        if snap_row.indexOf(line_output_interval_lbl) < 0:
            snap_row.addWidget(line_output_interval_lbl)

        self.line_output_interval_edit = run_tab_page.findChild(QtWidgets.QLineEdit, "line_output_interval_edit")
        if self.line_output_interval_edit is None:
            self.line_output_interval_edit = QtWidgets.QLineEdit("00:05")
            self.line_output_interval_edit.setObjectName("line_output_interval_edit")
        if snap_row.indexOf(self.line_output_interval_edit) < 0:
            snap_row.addWidget(self.line_output_interval_edit)

        self.snapshot_btn = run_tab_page.findChild(QtWidgets.QPushButton, "snapshot_btn")
        if self.snapshot_btn is None:
            self.snapshot_btn = QtWidgets.QPushButton("Take Snapshot")
            self.snapshot_btn.setObjectName("snapshot_btn")
        if snap_row.indexOf(self.snapshot_btn) < 0:
            snap_row.addWidget(self.snapshot_btn)

        self.cancel_btn.setEnabled(False)

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.output_interval_edit.setMaximumWidth(90)
        if not str(self.output_interval_edit.text() or "").strip():
            self.output_interval_edit.setText("00:30")
        self.output_interval_edit.setToolTip(
            "Interval between captured result snapshots during a run.\n"
            "E.g. 00:30 captures every 30 minutes of simulation time."
        )

        self.line_output_interval_edit.setMaximumWidth(90)
        if not str(self.line_output_interval_edit.text() or "").strip():
            self.line_output_interval_edit.setText("00:05")
        self.line_output_interval_edit.setToolTip(
            "Interval for sampled line time-series output capture.\n"
            "Independent from mesh snapshot interval."
        )

        self.snapshot_btn.setToolTip(
            "Write all captured timesteps up to now to a temporary HEC-RAS HDF5 file.\n"
            "The file path is logged in the message panel."
        )

        for btn, cb in (
            (self.run_btn, self._on_run_requested),
            (self.preview_overrides_btn, self._on_preview_overrides),
            (self.cancel_btn, self._on_cancel),
            (self.snapshot_btn, self._on_snapshot),
        ):
            try:
                btn.clicked.disconnect(cb)
            except Exception:
                pass
            btn.clicked.connect(cb)

        try:
            self.experimental_3d_mode_chk.toggled.disconnect(self._sync_experimental_3d_mode_widgets)
        except Exception:
            pass
        try:
            self.experimental_3d_mode_chk.toggled.connect(self._sync_experimental_3d_mode_widgets)
        except Exception:
            pass
        self._sync_experimental_3d_mode_widgets()

    def _build_right_pane(self) -> QtWidgets.QWidget:
        ui_path = self._forms_file_path("swe2d_right_pane.ui")
        right_pane = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                right_pane = _qgis_uic.loadUi(ui_path)
            except Exception:
                right_pane = None
        if right_pane is None:
            right_pane = self._build_right_pane_fallback()
        self._bind_right_pane_controls(right_pane)
        return right_pane

    def _build_right_pane_fallback(self) -> QtWidgets.QWidget:
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        view_row = QtWidgets.QHBoxLayout()
        view_row.setObjectName("view_row_layout")
        view_mode_lbl = QtWidgets.QLabel("View:")
        view_mode_lbl.setObjectName("view_mode_lbl")
        view_mode_combo = QtWidgets.QComboBox()
        view_mode_combo.setObjectName("view_mode_combo")
        view_row.addWidget(view_mode_lbl)
        view_row.addWidget(view_mode_combo)
        view_row.addStretch(1)
        right_layout.addLayout(view_row)

        popout_row = QtWidgets.QHBoxLayout()
        popout_row.setObjectName("popout_row_layout")
        detach_mesh_view_btn = QtWidgets.QPushButton("Detach Mesh View")
        detach_mesh_view_btn.setObjectName("detach_mesh_view_btn")
        detach_runtime_log_btn = QtWidgets.QPushButton("Detach Runtime Log")
        detach_runtime_log_btn.setObjectName("detach_runtime_log_btn")
        popout_row.addWidget(detach_mesh_view_btn)
        popout_row.addWidget(detach_runtime_log_btn)
        popout_row.addStretch(1)
        right_layout.addLayout(popout_row)

        right_vertical_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_vertical_split.setObjectName("right_vertical_split")

        right_plot_host = QtWidgets.QWidget()
        right_plot_host.setObjectName("right_plot_host")
        right_plot_host_layout = QtWidgets.QVBoxLayout(right_plot_host)
        right_plot_host_layout.setContentsMargins(0, 0, 0, 0)
        right_vertical_split.addWidget(right_plot_host)

        log_view = QtWidgets.QPlainTextEdit()
        log_view.setObjectName("log_view")
        right_vertical_split.addWidget(log_view)

        right_layout.addWidget(right_vertical_split, stretch=1)
        return right

    def _bind_right_pane_controls(self, right_pane: QtWidgets.QWidget) -> None:
        def _ensure_root_layout() -> QtWidgets.QVBoxLayout:
            layout = right_pane.layout()
            if isinstance(layout, QtWidgets.QVBoxLayout):
                return layout
            layout = QtWidgets.QVBoxLayout(right_pane)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            return layout

        right_layout = _ensure_root_layout()

        view_row = right_pane.findChild(QtWidgets.QHBoxLayout, "view_row_layout")
        if view_row is None:
            view_row = QtWidgets.QHBoxLayout()
            view_row.setObjectName("view_row_layout")
            right_layout.insertLayout(0, view_row)

        view_mode_lbl = right_pane.findChild(QtWidgets.QLabel, "view_mode_lbl")
        if view_mode_lbl is None:
            view_mode_lbl = QtWidgets.QLabel("View:")
            view_mode_lbl.setObjectName("view_mode_lbl")
        if view_row.indexOf(view_mode_lbl) < 0:
            view_row.addWidget(view_mode_lbl)

        self.view_mode_combo = right_pane.findChild(QtWidgets.QComboBox, "view_mode_combo")
        if self.view_mode_combo is None:
            self.view_mode_combo = QtWidgets.QComboBox()
            self.view_mode_combo.setObjectName("view_mode_combo")
        if view_row.indexOf(self.view_mode_combo) < 0:
            view_row.addWidget(self.view_mode_combo)
        if view_row.count() < 3:
            view_row.addStretch(1)

        prev_view_text = self.view_mode_combo.currentText()
        self.view_mode_combo.blockSignals(True)
        try:
            self.view_mode_combo.clear()
            self.view_mode_combo.addItems(["Mesh", "Depth", "Velocity magnitude"])
            idx = self.view_mode_combo.findText(prev_view_text)
            if idx < 0:
                idx = 0
            self.view_mode_combo.setCurrentIndex(idx)
        finally:
            self.view_mode_combo.blockSignals(False)
        try:
            self.view_mode_combo.currentIndexChanged.disconnect(self._refresh_plot)
        except Exception:
            pass
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)

        popout_row = right_pane.findChild(QtWidgets.QHBoxLayout, "popout_row_layout")
        if popout_row is None:
            popout_row = QtWidgets.QHBoxLayout()
            popout_row.setObjectName("popout_row_layout")
            right_layout.insertLayout(1, popout_row)

        self.detach_mesh_view_btn = right_pane.findChild(QtWidgets.QPushButton, "detach_mesh_view_btn")
        if self.detach_mesh_view_btn is None:
            self.detach_mesh_view_btn = QtWidgets.QPushButton("Detach Mesh View")
            self.detach_mesh_view_btn.setObjectName("detach_mesh_view_btn")
        if popout_row.indexOf(self.detach_mesh_view_btn) < 0:
            popout_row.addWidget(self.detach_mesh_view_btn)

        self.detach_runtime_log_btn = right_pane.findChild(QtWidgets.QPushButton, "detach_runtime_log_btn")
        if self.detach_runtime_log_btn is None:
            self.detach_runtime_log_btn = QtWidgets.QPushButton("Detach Runtime Log")
            self.detach_runtime_log_btn.setObjectName("detach_runtime_log_btn")
        if popout_row.indexOf(self.detach_runtime_log_btn) < 0:
            popout_row.addWidget(self.detach_runtime_log_btn)
        if popout_row.count() < 3:
            popout_row.addStretch(1)

        for btn, cb in (
            (self.detach_mesh_view_btn, self._open_detached_mesh_view),
            (self.detach_runtime_log_btn, self._open_detached_runtime_log),
        ):
            try:
                btn.clicked.disconnect(cb)
            except Exception:
                pass
            btn.clicked.connect(cb)

        self._right_vertical_split = right_pane.findChild(QtWidgets.QSplitter, "right_vertical_split")
        if self._right_vertical_split is None:
            self._right_vertical_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            self._right_vertical_split.setObjectName("right_vertical_split")
        self._right_vertical_split.setOrientation(QtCore.Qt.Orientation.Vertical)
        self._right_vertical_split.setChildrenCollapsible(False)
        if right_layout.indexOf(self._right_vertical_split) < 0:
            right_layout.addWidget(self._right_vertical_split, stretch=1)

        right_plot_host = right_pane.findChild(QtWidgets.QWidget, "right_plot_host")
        if right_plot_host is None:
            right_plot_host = QtWidgets.QWidget()
            right_plot_host.setObjectName("right_plot_host")
        if self._right_vertical_split.indexOf(right_plot_host) < 0:
            self._right_vertical_split.insertWidget(0, right_plot_host)

        plot_layout = right_plot_host.layout()
        if not isinstance(plot_layout, QtWidgets.QVBoxLayout):
            plot_layout = QtWidgets.QVBoxLayout(right_plot_host)
            plot_layout.setContentsMargins(0, 0, 0, 0)
        while plot_layout.count():
            item = plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if self._have_mpl:
            self._fig = self._Figure(figsize=(6.4, 4.2), tight_layout=True)
            self._canvas = self._FigureCanvas(self._fig)
            self._canvas.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            try:
                self._canvas.customContextMenuRequested.disconnect()
            except Exception:
                pass
            self._canvas.customContextMenuRequested.connect(
                lambda pos: self._show_panel_detach_menu("mesh", self._canvas.mapToGlobal(pos))
            )
            plot_layout.addWidget(self._canvas)
        else:
            self._fig = None
            self._canvas = None
            no_plot = QtWidgets.QLabel("Matplotlib Qt backend not available; results shown in text log only.")
            no_plot.setWordWrap(True)
            plot_layout.addWidget(no_plot)

        self.log_view = right_pane.findChild(QtWidgets.QPlainTextEdit, "log_view")
        if self.log_view is None:
            self.log_view = QtWidgets.QPlainTextEdit()
            self.log_view.setObjectName("log_view")
        if self._right_vertical_split.indexOf(self.log_view) < 0:
            self._right_vertical_split.addWidget(self.log_view)
        self.log_view.setReadOnly(True)
        self.log_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        try:
            self.log_view.customContextMenuRequested.disconnect()
        except Exception:
            pass
        self.log_view.customContextMenuRequested.connect(
            lambda pos: self._show_panel_detach_menu("log", self.log_view.mapToGlobal(pos))
        )
        self._right_vertical_split.setSizes([520, 220])

    def _build_workbench_shell(
        self,
    ) -> Tuple[
        QtWidgets.QWidget,
        QtWidgets.QLabel,
        QtWidgets.QSplitter,
        QtWidgets.QWidget,
        QtWidgets.QWidget,
        QtWidgets.QDialogButtonBox,
    ]:
        ui_path = self._forms_file_path("swe2d_workbench_shell.ui")
        shell = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                shell = _qgis_uic.loadUi(ui_path)
            except Exception:
                shell = None
        if shell is None:
            shell = self._build_workbench_shell_fallback()
        return self._bind_workbench_shell(shell)

    def _build_workbench_shell_fallback(self) -> QtWidgets.QWidget:
        shell = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(shell)

        header_lbl = QtWidgets.QLabel("Interactive 2D SWE workflow")
        header_lbl.setObjectName("header_lbl")
        header_lbl.setWordWrap(True)
        root_layout.addWidget(header_lbl)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setObjectName("main_splitter")
        left_host = QtWidgets.QWidget()
        left_host.setObjectName("left_host")
        right_host = QtWidgets.QWidget()
        right_host.setObjectName("right_host")
        main_splitter.addWidget(left_host)
        main_splitter.addWidget(right_host)
        root_layout.addWidget(main_splitter, stretch=1)

        bottom_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bottom_buttons.setObjectName("bottom_buttons")
        root_layout.addWidget(bottom_buttons)
        return shell

    def _bind_workbench_shell(
        self, shell: QtWidgets.QWidget
    ) -> Tuple[
        QtWidgets.QWidget,
        QtWidgets.QLabel,
        QtWidgets.QSplitter,
        QtWidgets.QWidget,
        QtWidgets.QWidget,
        QtWidgets.QDialogButtonBox,
    ]:
        def _ensure_root_layout() -> QtWidgets.QVBoxLayout:
            layout = shell.layout()
            if isinstance(layout, QtWidgets.QVBoxLayout):
                return layout
            return QtWidgets.QVBoxLayout(shell)

        root_layout = _ensure_root_layout()

        header_lbl = shell.findChild(QtWidgets.QLabel, "header_lbl")
        if header_lbl is None:
            header_lbl = QtWidgets.QLabel("Interactive 2D SWE workflow")
            header_lbl.setObjectName("header_lbl")
            root_layout.insertWidget(0, header_lbl)
        header_lbl.setWordWrap(True)

        main_splitter = shell.findChild(QtWidgets.QSplitter, "main_splitter")
        if main_splitter is None:
            main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            main_splitter.setObjectName("main_splitter")
            root_layout.addWidget(main_splitter, stretch=1)
        main_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        left_host = shell.findChild(QtWidgets.QWidget, "left_host")
        if left_host is None:
            left_host = QtWidgets.QWidget()
            left_host.setObjectName("left_host")
        if main_splitter.indexOf(left_host) < 0:
            main_splitter.insertWidget(0, left_host)

        right_host = shell.findChild(QtWidgets.QWidget, "right_host")
        if right_host is None:
            right_host = QtWidgets.QWidget()
            right_host.setObjectName("right_host")
        if main_splitter.indexOf(right_host) < 0:
            main_splitter.addWidget(right_host)

        bottom_buttons = shell.findChild(QtWidgets.QDialogButtonBox, "bottom_buttons")
        if bottom_buttons is None:
            bottom_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
            bottom_buttons.setObjectName("bottom_buttons")
            root_layout.addWidget(bottom_buttons)
        return shell, header_lbl, main_splitter, left_host, right_host, bottom_buttons

    def _compose_left_pane(self, left_host: QtWidgets.QWidget) -> QtWidgets.QWidget:
        left = left_host
        left_layout = left.layout()
        if not isinstance(left_layout, QtWidgets.QVBoxLayout):
            left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        while left_layout.count():
            item = left_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._left_tabs = QtWidgets.QTabWidget()
        self._left_tabs.setDocumentMode(True)
        left_layout.addWidget(self._left_tabs, stretch=1)

        mesh_tab_page = self._build_mesh_tab_page()
        self._left_tabs.addTab(self._wrap_left_tab_page(mesh_tab_page), "Mesh")

        (
            map_tab_page,
            map_data_layout,
            map_actions_layout,
            map_results_layout,
            map_tools_layout,
        ) = self._build_map_tab_page()
        self._bind_map_tab_data_controls(map_tab_page, map_data_layout)
        self._bind_map_tab_action_controls(map_tab_page, map_actions_layout)
        self._bind_map_tab_results_controls(map_tab_page, map_results_layout)
        self._bind_map_tab_tools_controls(map_tab_page, map_tools_layout)
        self._left_tabs.addTab(self._wrap_left_tab_page(map_tab_page), "Map")

        topology_tab_page, topo_layout = self._build_topology_tab_page()
        self._bind_topology_tab_static_controls(topology_tab_page, topo_layout)
        self._bind_topology_tab_dynamic_controls(topology_tab_page, topo_layout)
        self._left_tabs.addTab(self._wrap_left_tab_page(topology_tab_page), "Topology")

        boundary_tab_page = self._build_boundary_tab_page()
        self._left_tabs.addTab(self._wrap_left_tab_page(boundary_tab_page), "Boundary")

        model_tab_page, param_form = self._build_model_tab_page()
        self._bind_model_tab_core_controls(model_tab_page, param_form)
        self._bind_model_tab_hydrology_controls(model_tab_page, param_form)
        self._bind_model_tab_solver_controls(model_tab_page, param_form)
        self._bind_model_tab_3d_patch_controls(model_tab_page, param_form)
        self._bind_model_tab_3d_subgrid_drainage_controls(model_tab_page, param_form)
        self._left_tabs.addTab(self._wrap_left_tab_page(model_tab_page), "Model")

        run_tab_page = self._build_run_tab_page()
        self._left_tabs.addTab(self._wrap_left_tab_page(run_tab_page), "Run")

        # Allow the left panel to shrink as narrow as the splitter permits.
        # Widgets keep their natural *preferred* size but no longer enforce a
        # large minimum width, so the user can drag the splitter very small.
        left.setMinimumWidth(0)
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
        self._make_left_controls_compact(left)
        self._register_detachable_tab_widget(self._left_tabs)

        return left

    def _build_ui(self):
        root = self.layout()
        if not isinstance(root, QtWidgets.QVBoxLayout):
            root = QtWidgets.QVBoxLayout(self)
        while root.count():
            item = root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        shell, header, split, left_host, right_host, buttons = self._build_workbench_shell()
        root.addWidget(shell, stretch=1)

        header.setText(
            "Interactive 2D SWE workflow: generate mesh, assign side BCs, set model parameters, "
            "run, and visualize results."
        )
        header.setWordWrap(True)

        # Left pane: setup + run controls
        self._compose_left_pane(left_host)

        right = self._build_right_pane()
        right_layout = right_host.layout()
        if not isinstance(right_layout, QtWidgets.QVBoxLayout):
            right_layout = QtWidgets.QVBoxLayout(right_host)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        while right_layout.count():
            item = right_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        right_layout.addWidget(right, stretch=1)

        split.setSizes([420, 740])

        try:
            buttons.rejected.disconnect(self.reject)
        except Exception:
            pass
        try:
            buttons.accepted.disconnect(self.accept)
        except Exception:
            pass
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        self._refresh_layer_combos()

    def _wrap_left_tab_page(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        # Wrap in a container so content stays top-aligned when the pane is
        # taller than the content, rather than stretching the group box itself.
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(widget)
        vbox.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(container)
        return scroll

    def _make_left_controls_compact(self, parent_widget: QtWidgets.QWidget) -> None:
        for layout in parent_widget.findChildren(QtWidgets.QLayout):
            try:
                layout.setContentsMargins(4, 4, 4, 4)
            except Exception:
                pass
            try:
                if hasattr(layout, "setSpacing"):
                    layout.setSpacing(4)
            except Exception:
                pass
            if isinstance(layout, QtWidgets.QFormLayout):
                try:
                    layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
                    layout.setHorizontalSpacing(6)
                    layout.setVerticalSpacing(4)
                except Exception:
                    pass

    def _register_detachable_tab_widget(self, tab_widget: QtWidgets.QTabWidget) -> None:
        if tab_widget is None:
            return
        tab_widget.tabBar().setMovable(True)
        tab_widget.tabBar().setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        tab_widget.tabBar().customContextMenuRequested.connect(
            lambda pos, tw=tab_widget: self._show_tab_detach_menu(tw, pos)
        )

    def _show_tab_detach_menu(self, tab_widget: QtWidgets.QTabWidget, pos: QtCore.QPoint) -> None:
        bar = tab_widget.tabBar()
        idx = bar.tabAt(pos)
        if idx < 0:
            return
        title = str(tab_widget.tabText(idx) or "Tab")
        menu = QtWidgets.QMenu(self)
        detach_action = menu.addAction(f"Detach \"{title}\"")
        chosen = menu.exec(bar.mapToGlobal(pos))
        if chosen == detach_action:
            self._detach_tab(tab_widget, idx)

    def _detach_tab(self, tab_widget: QtWidgets.QTabWidget, index: int) -> None:
        if tab_widget is None or index < 0:
            return
        page = tab_widget.widget(index)
        if page is None:
            return
        title = str(tab_widget.tabText(index) or "Detached")
        icon = tab_widget.tabIcon(index)
        tip = str(tab_widget.tabToolTip(index) or "")
        insert_index = int(index)
        tab_widget.removeTab(index)
        # removeTab() hides the page widget; make it visible before placing in the detached dialog
        page.show()

        def _reattach() -> None:
            if page is None:
                return
            try:
                if page.parent() is not None:
                    page.setParent(None)
            except Exception:
                pass
            target_index = max(0, min(insert_index, tab_widget.count()))
            tab_widget.insertTab(target_index, page, icon, title)
            if tip:
                tab_widget.setTabToolTip(target_index, tip)
            tab_widget.setCurrentIndex(target_index)

        dlg = SWE2DDetachedPanelDialog(
            title=f"2D SWE - {title}",
            content_widget=page,
            on_reattach=_reattach,
            parent=self,
        )
        self._detached_panel_dialogs.append(dlg)

        def _cleanup(_result=None, dialog=dlg):
            try:
                if dialog in self._detached_panel_dialogs:
                    self._detached_panel_dialogs.remove(dialog)
            except Exception:
                pass

        dlg.finished.connect(_cleanup)
        dlg.show()

    def _show_panel_detach_menu(self, panel_kind: str, global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        if str(panel_kind) == "mesh":
            action = menu.addAction("Detach Mesh View")
            chosen = menu.exec(global_pos)
            if chosen == action:
                self._open_detached_mesh_view()
            return
        if str(panel_kind) == "log":
            action = menu.addAction("Detach Runtime Log")
            chosen = menu.exec(global_pos)
            if chosen == action:
                self._open_detached_runtime_log()
            return

    def _log(self, msg: str):
        msg_txt = str(msg)
        self._runtime_log_lines.append(msg_txt)
        self.log_view.appendPlainText(msg_txt)
        for dlg in list(getattr(self, "_runtime_log_detached_dialogs", [])):
            try:
                if dlg is not None:
                    dlg.append_text(msg_txt)
            except Exception:
                pass
        QtWidgets.QApplication.processEvents()

    def _render_workbench_mesh_view(self, ax, mode: str) -> None:
        if self._mesh_data is None:
            ax.text(0.5, 0.5, "No mesh loaded", ha="center", va="center", transform=ax.transAxes)
            return

        figure = getattr(ax, "figure", None)
        node_x = self._mesh_data["node_x"]
        node_y = self._mesh_data["node_y"]
        triangles = self._mesh_data["cell_nodes"].reshape((-1, 3))
        tri = self._mtri.Triangulation(node_x, node_y, triangles)

        if mode == "mesh" or self._result_data is None:
            ax.triplot(tri, color="black", linewidth=0.3)
            ax.set_title("Generated mesh")
            return

        if mode == "depth":
            vals = np.asarray(self._result_data["h"], dtype=np.float64)
            tpc = ax.tripcolor(tri, facecolors=vals, cmap="viridis", edgecolors="none")
            if figure is not None:
                figure.colorbar(tpc, ax=ax, label="Depth")
            ax.set_title("Final depth")
            return

        h_raw = np.asarray(self._result_data["h"], dtype=np.float64)
        h = np.maximum(h_raw, 1.0e-12)
        hu = np.asarray(self._result_data["hu"], dtype=np.float64)
        hv = np.asarray(self._result_data["hv"], dtype=np.float64)
        h_min = float(self.h_min_spin.value()) if hasattr(self, "h_min_spin") else 1.0e-6
        wet = (h_raw > h_min)
        vals = np.where(wet, np.sqrt((hu / h) ** 2 + (hv / h) ** 2), 0.0)
        tpc = ax.tripcolor(tri, facecolors=vals, cmap="plasma", edgecolors="none")
        if figure is not None:
            figure.colorbar(tpc, ax=ax, label="Velocity magnitude")
        ax.set_title("Final velocity magnitude")

    def _open_detached_runtime_log(self):
        text = "\n".join(self._runtime_log_lines)
        dlg = SWE2DDetachedRuntimeLogDialog(initial_text=text, parent=self)
        self._runtime_log_detached_dialog = dlg
        self._runtime_log_detached_dialogs.append(dlg)

        def _cleanup(_result=None, dialog=dlg):
            try:
                if dialog in self._runtime_log_detached_dialogs:
                    self._runtime_log_detached_dialogs.remove(dialog)
            except Exception:
                pass
            if self._runtime_log_detached_dialog is dialog:
                self._runtime_log_detached_dialog = None

        dlg.finished.connect(_cleanup)
        dlg.show()

    def _open_detached_mesh_view(self):
        dlg = SWE2DDetachedMeshViewDialog(render_callback=self._render_workbench_mesh_view, parent=self)
        self._mesh_view_detached_dialog = dlg
        self._mesh_view_detached_dialogs.append(dlg)

        def _cleanup(_result=None, dialog=dlg):
            try:
                if dialog in self._mesh_view_detached_dialogs:
                    self._mesh_view_detached_dialogs.remove(dialog)
            except Exception:
                pass
            if self._mesh_view_detached_dialog is dialog:
                self._mesh_view_detached_dialog = None

        dlg.finished.connect(_cleanup)
        dlg.show()

    def _is_experimental_3d_requested(self) -> bool:
        return bool(
            getattr(self, "experimental_3d_mode_chk", None)
            and self.experimental_3d_mode_chk.isChecked()
        )

    def _experimental_3d_selected_coupling_mode(self) -> int:
        if SWE2DThreeDCouplingMode is None:
            return 0
        combo = getattr(self, "experimental_3d_coupling_mode_combo", None)
        if combo is None:
            return int(SWE2DThreeDCouplingMode.OFF)
        try:
            value = combo.currentData()
            if value is None:
                value = combo.currentIndex()
            return int(SWE2DThreeDCouplingMode(int(value)))
        except Exception:
            return int(SWE2DThreeDCouplingMode.OFF)

    def _experimental_3d_bc_mode_label(self, mode_value: int) -> str:
        for label, value in _SWE3D_BC_MODE_OPTIONS:
            if int(value) == int(mode_value):
                return str(label)
        return f"mode{int(mode_value)}"

    def _collect_3d_patch_face_bc_env_overrides(self) -> Dict[str, str]:
        return _collect_3d_patch_face_bc_env_overrides_logic(
            ui=self,
            faces=_SWE3D_PATCH_FACES,
            field_defaults=_SWE3D_BC_FIELD_DEFAULTS,
        )

    def _summarize_3d_patch_face_bc_modes(self, overrides: Dict[str, str]) -> str:
        return _summarize_3d_patch_face_bc_modes_logic(
            overrides=overrides,
            faces=_SWE3D_PATCH_FACES,
            mode_label_callback=self._experimental_3d_bc_mode_label,
        )

    def _apply_3d_patch_face_bc_to_backend(self, backend: object) -> None:
        return _apply_3d_patch_face_bc_to_backend_logic(
            ui=self,
            backend=backend,
            faces=_SWE3D_PATCH_FACES,
            field_defaults=_SWE3D_BC_FIELD_DEFAULTS,
            coupling_mode_off=int(SWE2DThreeDCouplingMode.OFF),
            get_coupling_mode_callback=self._experimental_3d_selected_coupling_mode,
            log_callback=self._log,
        )

    def _sync_experimental_3d_mode_widgets(self, *_args: object) -> None:
        requested = self._is_experimental_3d_requested()
        supported = bool(getattr(self, "_experimental_3d_mode_supported", True))
        active = bool(requested and supported)
        _sync_experimental_3d_mode_widgets_logic(ui=self, active=active)

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
            "experimental_3d_obj_layer_combo": ["swe2d_obj_instances", "swe3d_obj_instances", "obj_instances"],
            "experimental_3d_obj_inside_points_layer_combo": [
                "swe2d_obj_inside_points",
                "swe3d_obj_inside_points",
                "obj_inside_points",
                "inside_points",
            ],
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
        for dlg in list(self._runtime_log_detached_dialogs) + list(self._mesh_view_detached_dialogs) + list(self._detached_panel_dialogs):
            try:
                if dlg is not None:
                    dlg.close()
            except Exception:
                pass
        try:
            self._destroy_high_perf_canvas_overlay_item()
        except Exception:
            pass
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
            keep_3d_obj_instances = self.experimental_3d_obj_layer_combo.currentData() if hasattr(self, "experimental_3d_obj_layer_combo") else None
            keep_3d_obj_inside_points = self.experimental_3d_obj_inside_points_layer_combo.currentData() if hasattr(self, "experimental_3d_obj_inside_points_layer_combo") else None

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
            if hasattr(self, "experimental_3d_obj_layer_combo"):
                self.experimental_3d_obj_layer_combo.clear()
                self.experimental_3d_obj_layer_combo.addItem("(none)", None)
            if hasattr(self, "experimental_3d_obj_inside_points_layer_combo"):
                self.experimental_3d_obj_inside_points_layer_combo.clear()
                self.experimental_3d_obj_inside_points_layer_combo.addItem("(none)", None)
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
                            if hasattr(self, "experimental_3d_obj_layer_combo"):
                                self.experimental_3d_obj_layer_combo.addItem(lyr.name(), lyr.id())
                            if hasattr(self, "experimental_3d_obj_inside_points_layer_combo"):
                                self.experimental_3d_obj_inside_points_layer_combo.addItem(lyr.name(), lyr.id())
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
            if hasattr(self, "experimental_3d_obj_layer_combo") and keep_3d_obj_instances is not None:
                _restore(self.experimental_3d_obj_layer_combo, keep_3d_obj_instances)
            if hasattr(self, "experimental_3d_obj_inside_points_layer_combo") and keep_3d_obj_inside_points is not None:
                _restore(self.experimental_3d_obj_inside_points_layer_combo, keep_3d_obj_inside_points)

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
                "used by Gmsh and the structured fallback when the region has a complete four-edge topology definition. "
                "Interior polygon rings are treated as hole cutouts. Regions with cell_type='empty' act as exclusion holes."
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
        constraints_layer = self._combo_layer(self.topo_constraints_combo, "vector") if hasattr(self, "topo_constraints_combo") else None
        quad_edges_layer = self._combo_layer(self.topo_quad_edges_combo, "vector") if hasattr(self, "topo_quad_edges_combo") else None

        if backend_name == "gmsh":
            backend_hint = (
                "Gmsh: use multiple region polygons for multiblock meshes. "
                "Set region cell_type to 'cartesian' or 'quadrilateral' and populate edge_len_1..4 "
                "for per-edge structured spacing. Opposite edges are matched automatically. "
                "Region interior rings plus empty regions/constraints are meshed as cutout holes."
            )
        elif backend_name == "tqmesh":
            backend_hint = (
                "TQMesh: use multiple region polygons for blockwise target_size and cell_type. "
                "Use quad-edge lines with n_layers, first_height, and growth_rate for transition layers. "
                "Region interior rings plus empty regions/constraints are meshed as cutout holes."
            )
        else:
            backend_hint = (
                "Structured fallback: honors per-region target_size and cell_type, "
                "supports cutout holes from region interior rings and empty zones, "
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
                empty_count = 0
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
                    if ctype == "empty":
                        empty_count += 1
                    if "target_size" in region_fields and ft["target_size"] not in (None, ""):
                        try:
                            size_values.add(round(float(ft["target_size"]), 6))
                        except Exception:
                            pass
                details.append(f"regions={region_count}")
                if cartesian_count > 0:
                    details.append(f"structured-block-regions={cartesian_count}")
                if empty_count > 0:
                    details.append(f"empty-regions={empty_count}")
                if len(size_values) > 1:
                    details.append(f"multi-block sizes={len(size_values)}")
                if missing_edge_lengths > 0:
                    details.append(f"structured regions missing edge_len_1..4={missing_edge_lengths}")
            except Exception:
                pass

        if constraints_layer is not None and getattr(self, "topo_constraints_combo", None) is not None and self.topo_constraints_combo.currentData() is not None:
            try:
                c_fields = set(constraints_layer.fields().names())
                constraint_count = 0
                empty_constraints = 0
                for ft in constraints_layer.getFeatures():
                    constraint_count += 1
                    ctype = str(ft["cell_type"]).strip().lower() if "cell_type" in c_fields and ft["cell_type"] not in (None, "") else ""
                    if ctype == "empty":
                        empty_constraints += 1
                details.append(f"constraints={constraint_count}")
                if empty_constraints > 0:
                    details.append(f"empty-constraints={empty_constraints}")
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
            "Topology template layers created. Define regions (required); use interior rings or cell_type='empty' zones for holes; "
            "add optional arcs/constraints and optional TQMesh quad-edge lines; then generate mesh."
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
        return _mesh_cell_centroids_logic(self._mesh_data)

    def _mesh_cell_areas(self) -> np.ndarray:
        assert self._mesh_data is not None
        return _mesh_cell_areas_logic(self._mesh_data)

    def _mesh_cell_min_bed(self) -> np.ndarray:
        assert self._mesh_data is not None
        return _mesh_cell_min_bed_logic(self._mesh_data)

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
        _maybe_create_results_panel_bridge(self)

    def _sync_high_perf_overlay_data(self):
        _sync_high_perf_overlay_data_bridge(self)

    def _update_high_perf_overlay_time(self, t_s: float):
        _update_high_perf_overlay_time_bridge(self, t_s)

    def _destroy_high_perf_canvas_overlay_item(self):
        _destroy_high_perf_canvas_overlay_item_bridge(self)

    def _ensure_high_perf_canvas_overlay_item(self):
        item = getattr(self, "_high_perf_canvas_overlay_item", None)
        if item is not None:
            return item

        canvas = self._resolve_map_canvas()
        if canvas is None:
            return None
        try:
            try:
                from .swe2d_high_perf_viewer import SWE2DHighPerfCanvasOverlayItem
            except ImportError:
                from swe2d_high_perf_viewer import SWE2DHighPerfCanvasOverlayItem

            item = SWE2DHighPerfCanvasOverlayItem(canvas)
            self._high_perf_canvas_overlay_item = item
            return item
        except Exception as exc:
            self._log(f"[HighPerf Overlay] could not create canvas item: {exc}")
            self._high_perf_canvas_overlay_item = None
            return None

    def _on_high_perf_canvas_overlay_toggled(self, checked: bool):
        self._high_perf_canvas_overlay_enabled = bool(checked)
        if not self._high_perf_canvas_overlay_enabled:
            item = getattr(self, "_high_perf_canvas_overlay_item", None)
            if item is not None:
                try:
                    item.clear()
                except Exception:
                    pass
            iface = self._resolve_qgis_iface()
            if iface is not None and hasattr(iface, "mapCanvas"):
                try:
                    iface.mapCanvas().refresh()
                except Exception:
                    pass
            return
        self._sync_high_perf_overlay_data()

    def _on_high_perf_canvas_overlay_style_changed(self, *_):
        lock_canvas = bool(self.high_perf_canvas_overlay_lock_canvas_chk.isChecked())
        self.high_perf_canvas_overlay_res_combo.setEnabled(not lock_canvas)
        if bool(getattr(self, "_high_perf_canvas_overlay_enabled", False)):
            self._refresh_high_perf_canvas_overlay(None)

    def _refresh_high_perf_canvas_overlay(self, t_s: Optional[float]):
        if not bool(getattr(self, "_high_perf_canvas_overlay_enabled", False)):
            return
        if self._high_perf_overlay_cell_x.size <= 0 or not self._snapshot_timesteps:
            item = getattr(self, "_high_perf_canvas_overlay_item", None)
            if item is not None:
                try:
                    item.clear()
                except Exception:
                    pass
            return

        item = self._ensure_high_perf_canvas_overlay_item()
        if item is None:
            return

        t_use = None
        if t_s is not None:
            t_use = float(t_s)
        elif self._results_panel is not None:
            try:
                t_use = float(self._results_panel.current_time_sec())
            except Exception:
                t_use = None
        if t_use is None:
            t_use = float(self._snapshot_timesteps[-1][0])

        try:
            try:
                from .swe2d_high_perf_viewer import render_unstructured_snapshot_image
            except ImportError:
                from swe2d_high_perf_viewer import render_unstructured_snapshot_image

            field_key = str(self.high_perf_canvas_overlay_field_combo.currentData() or "depth")
            cmap_key = str(self.high_perf_canvas_overlay_cmap_combo.currentData() or "turbo")
            auto_contrast = bool(self.high_perf_canvas_overlay_auto_contrast_chk.isChecked())
            lock_canvas = bool(self.high_perf_canvas_overlay_lock_canvas_chk.isChecked())
            if lock_canvas:
                canvas = self._resolve_map_canvas()
                if canvas is not None:
                    res = (max(64, int(canvas.width())), max(64, int(canvas.height())))
                else:
                    res = (1280, 720)
            else:
                raw_res = self.high_perf_canvas_overlay_res_combo.currentData()
                if isinstance(raw_res, tuple) and len(raw_res) == 2:
                    res = (max(64, int(raw_res[0])), max(64, int(raw_res[1])))
                else:
                    res = (1280, 720)
            opacity = float(self.high_perf_canvas_overlay_opacity_spin.value())
            frame = render_unstructured_snapshot_image(
                cell_x=self._high_perf_overlay_cell_x,
                cell_y=self._high_perf_overlay_cell_y,
                cell_bed=self._high_perf_overlay_cell_bed,
                timesteps=self._snapshot_timesteps,
                current_time_s=float(t_use),
                field_key=field_key,
                cmap_key=cmap_key,
                resolution=res,
                auto_contrast=auto_contrast,
            )
            if not bool(frame.get("ok", False)):
                try:
                    item.clear()
                except Exception:
                    pass
                return

            image = frame.get("image", None)
            extent = frame.get("extent", (0.0, 1.0, 0.0, 1.0))
            if image is None:
                try:
                    item.clear()
                except Exception:
                    pass
                return
            item.set_frame(image, extent, opacity=opacity)
            iface = self._resolve_qgis_iface()
            if iface is not None and hasattr(iface, "mapCanvas"):
                try:
                    iface.mapCanvas().refresh()
                except Exception:
                    pass
        except Exception as exc:
            self._log(f"[HighPerf Overlay] refresh failed: {exc}")

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

        if self._results_mesh_mode_enabled:
            self._ensure_results_mesh_layer_mode()
        self._results_panel.show()
        self._results_panel.raise_()
        self._refresh_results_map_overlays(self._results_panel.current_time_sec())

    def _results_mesh_layer(self):
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return None
        lid = str(getattr(self, "_results_mesh_layer_id", "") or "").strip()
        if not lid:
            return None
        try:
            lyr = QgsProject.instance().mapLayer(lid)
            if lyr is not None and lyr.isValid():
                return lyr
        except Exception:
            pass
        return None

    def _results_mesh_temp_nc_path(self) -> str:
        import tempfile

        stem = "swe2d_results"
        try:
            if self._model_gpkg_path:
                stem = os.path.splitext(os.path.basename(self._model_gpkg_path))[0] or stem
        except Exception:
            pass
        return os.path.join(tempfile.gettempdir(), f"{stem}_results_panel.nc")

    def _ensure_results_mesh_layer_mode(self) -> bool:
        """Ensure a QGIS mesh layer is available for snapshot-driven results viewing."""
        if not self._results_mesh_mode_enabled:
            return False
        if not _HAVE_QGIS_CORE or QgsProject is None or QgsMeshLayer is None:
            return False
        if self._mesh_data is None or not self._snapshot_timesteps:
            return False

        out_path = self._results_mesh_temp_nc_path()
        need_rewrite = (
            (not os.path.exists(out_path))
            or (int(self._results_mesh_snapshot_count) != int(len(self._snapshot_timesteps)))
        )
        if need_rewrite:
            try:
                self._write_ugrid_nc(out_path, timesteps=self._snapshot_timesteps)
                self._results_mesh_snapshot_count = int(len(self._snapshot_timesteps))
                self._results_mesh_source_path = str(out_path)
                self._log(
                    "Results mesh source updated for map mode: "
                    f"{out_path} (timesteps={self._results_mesh_snapshot_count})"
                )
            except Exception as exc:
                self._log(f"Results mesh-layer mode unavailable (UGRID export failed): {exc}")
                return False

        existing = self._results_mesh_layer()
        if existing is not None:
            try:
                src = str(existing.source() or "")
                if str(out_path) in src or src == str(out_path):
                    return True
            except Exception:
                pass
            try:
                QgsProject.instance().removeMapLayer(existing.id())
            except Exception:
                pass

        try:
            mesh_layer = QgsMeshLayer(str(out_path), "SWE2D_Results_Mesh", "mdal")
        except Exception as exc:
            self._log(f"Results mesh-layer mode unavailable (mesh layer create failed): {exc}")
            return False

        if mesh_layer is None or not mesh_layer.isValid():
            self._log(
                "Results mesh-layer mode unavailable: QGIS/MDAL failed to open generated UGRID file."
            )
            return False

        try:
            QgsProject.instance().addMapLayer(mesh_layer)
            self._results_mesh_layer_id = str(mesh_layer.id())
            self._log(
                "Results map mode active: mesh-layer-first (QGIS Mesh/MDAL). "
                f"Layer='{mesh_layer.name()}'"
            )
            return True
        except Exception as exc:
            self._log(f"Results mesh-layer mode unavailable (add layer failed): {exc}")
            return False

    def _on_results_panel_timestep_changed(self, t_s: float):
        self._refresh_results_map_overlays(float(t_s))
        self._update_high_perf_overlay_time(float(t_s))
        self._refresh_published_3d_surface_layer(float(t_s))

    def _on_results_panel_velocity_overlay_changed(self):
        panel = getattr(self, "_results_panel", None)
        t_s = panel.current_time_sec() if panel is not None else 0.0
        self._refresh_results_map_overlays(float(t_s))

    def _refresh_results_map_overlays(self, t_s: float):
        self._refresh_velocity_vectors_overlay(float(t_s))
        self._refresh_streamline_traces_overlay(float(t_s))

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

    def _velocity_data_support_for_run(self, gpkg_path: str, run_id: str, table_name: str) -> Dict[str, object]:
        """Inspect stored velocity data availability for a run.

        Returns availability flags so UI/runtime logs can explain which velocity
        source will be used: face-centered reconstruction (preferred) or
        cell-centered hu/hv fallback.
        """
        gpkg_path = str(gpkg_path or "").strip()
        run_id = str(run_id or "").strip()
        table_name = str(table_name or "swe2d_mesh_results").strip() or "swe2d_mesh_results"
        out = {
            "cell_rows": 0,
            "face_table": "",
            "face_rows": 0,
        }
        if not gpkg_path or not run_id or not os.path.exists(gpkg_path):
            return out

        def _quote_ident(name: str) -> str:
            return '"' + str(name or "").replace('"', '""') + '"'

        try:
            conn = sqlite3.connect(gpkg_path)
        except Exception:
            return out
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE run_id = ?",
                    (run_id,),
                )
                row = cur.fetchone()
                out["cell_rows"] = int(row[0]) if row and row[0] is not None else 0
            except Exception:
                out["cell_rows"] = 0

            for face_table in ("swe2d_face_flux_results", "swe2d_face_results", "swe2d_flux_faces"):
                try:
                    cur.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (face_table,),
                    )
                    if cur.fetchone() is None:
                        continue
                    cur.execute(
                        f"SELECT COUNT(*) FROM {_quote_ident(face_table)} WHERE run_id = ?",
                        (run_id,),
                    )
                    row = cur.fetchone()
                    n_face = int(row[0]) if row and row[0] is not None else 0
                    if n_face > 0:
                        out["face_table"] = face_table
                        out["face_rows"] = n_face
                        break
                except Exception:
                    continue
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

        start_path = self._velocity_overlay_manual_gpkg_path or self._model_gpkg_path or ""
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
        try:
            gpkg_path, run_id, table_name = self._pick_velocity_overlay_source()
            if not gpkg_path or not run_id:
                return

            source_key = f"{gpkg_path}::{table_name}::{run_id}"
            source = {
                "key": source_key,
                "gpkg_path": gpkg_path,
                "table_name": table_name,
                "run_id": run_id,
                "label": f"{os.path.basename(gpkg_path)}:{table_name}:{run_id}",
            }

            existing_idx = -1
            for i, rec in enumerate(self._velocity_overlay_sources):
                if str(rec.get("key", "")) == source_key:
                    existing_idx = i
                    break
            if existing_idx >= 0:
                self._velocity_overlay_sources[existing_idx] = source
            else:
                self._velocity_overlay_sources.append(source)

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
            self._refresh_results_map_overlays(float(t_s))

            support = self._velocity_data_support_for_run(gpkg_path, run_id, table_name)
            if int(support.get("face_rows", 0)) > 0:
                self._log(
                    "Velocity arrows source mode: face-centered reconstruction preferred "
                    f"(face_table={support.get('face_table')}, face_rows={int(support.get('face_rows', 0))}, "
                    f"cell_rows={int(support.get('cell_rows', 0))})."
                )
            else:
                self._log(
                    "Velocity arrows source mode: cell-centered hu/hv fallback only "
                    f"(no face flux rows for run_id={run_id}, cell_rows={int(support.get('cell_rows', 0))})."
                )
            self._log(
                f"Velocity arrows source added: table='{table_name}', gpkg='{gpkg_path}', run_id='{run_id}', total_sources={len(self._velocity_overlay_sources)}"
            )
        except Exception as exc:
            self._log(f"Velocity arrows source selection failed: {exc}")
            QtWidgets.QMessageBox.warning(
                self,
                "Velocity Arrows",
                f"Could not add velocity arrows source.\n\n{exc}",
            )

    def _velocity_source_color(self, source_key: str) -> str:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#17becf",
            "#bcbd22",
            "#8c564b",
            "#e377c2",
        ]
        idx = 0
        for i, rec in enumerate(self._velocity_overlay_sources):
            if str(rec.get("key", "")) == str(source_key):
                idx = i
                break
        return palette[idx % len(palette)]

    def _get_velocity_vector_builder(self):
        return _get_velocity_vector_builder_bridge(self)

    def _velocity_vectors_layer_for_source(self, source: Dict[str, str]):
        if not _HAVE_QGIS_CORE or QgsProject is None or QgsVectorLayer is None:
            return None

        source_key = str(source.get("key", ""))
        layer_id = self._velocity_overlay_layer_ids.get(source_key, "")
        if layer_id:
            lyr = QgsProject.instance().mapLayer(layer_id)
            if lyr is not None and lyr.isValid():
                return lyr

        crs_auth = "EPSG:4326"
        try:
            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception:
            pass

        run_id = str(source.get("run_id", "run"))
        table_name = str(source.get("table_name", "table"))
        layer_name = f"SWE2D_Velocity_{run_id}_{table_name}"
        uri = (
            f"LineString?crs={crs_auth}"
            "&field=cell_id:integer"
            "&field=speed:double"
            "&field=u:double"
            "&field=v:double"
            "&field=angle_deg:double"
            "&field=source:string(160)"
            "&field=color:string(16)"
            "&field=width:double"
        )
        lyr = QgsVectorLayer(uri, layer_name, "memory")
        if lyr is None or not lyr.isValid():
            return None
        QgsProject.instance().addMapLayer(lyr)
        self._velocity_overlay_layer_ids[source_key] = str(lyr.id())
        return lyr

    def _clear_velocity_vectors_layers(self):
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return
        for source_key, layer_id in list(self._velocity_overlay_layer_ids.items()):
            try:
                lyr = QgsProject.instance().mapLayer(layer_id)
                if lyr is None or not lyr.isValid():
                    continue
                dp = lyr.dataProvider()
                ids = [f.id() for f in lyr.getFeatures()]
                if ids:
                    dp.deleteFeatures(ids)
                self._velocity_overlay_feature_ids[source_key] = {}
                lyr.triggerRepaint()
            except Exception:
                continue

    def _streamline_traces_layer_for_source(self, source: Dict[str, str]):
        if not _HAVE_QGIS_CORE or QgsProject is None or QgsVectorLayer is None:
            return None

        source_key = str(source.get("key", ""))
        layer_id = self._streamline_overlay_layer_ids.get(source_key, "")
        if layer_id:
            lyr = QgsProject.instance().mapLayer(layer_id)
            if lyr is not None and lyr.isValid():
                return lyr

        crs_auth = "EPSG:4326"
        try:
            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception:
            pass

        run_id = str(source.get("run_id", "run"))
        table_name = str(source.get("table_name", "table"))
        layer_name = f"SWE2D_Streamlines_{run_id}_{table_name}"
        uri = (
            f"LineString?crs={crs_auth}"
            "&field=trace_id:integer"
            "&field=speed:double"
            "&field=length:double"
            "&field=source:string(160)"
            "&field=color:string(16)"
            "&field=width:double"
        )
        lyr = QgsVectorLayer(uri, layer_name, "memory")
        if lyr is None or not lyr.isValid():
            return None
        QgsProject.instance().addMapLayer(lyr)
        self._streamline_overlay_layer_ids[source_key] = str(lyr.id())
        return lyr

    def _clear_streamline_traces_layers(self):
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return
        for _source_key, layer_id in list(self._streamline_overlay_layer_ids.items()):
            try:
                lyr = QgsProject.instance().mapLayer(layer_id)
                if lyr is None or not lyr.isValid():
                    continue
                dp = lyr.dataProvider()
                ids = [f.id() for f in lyr.getFeatures()]
                if ids:
                    dp.deleteFeatures(ids)
                lyr.triggerRepaint()
            except Exception:
                continue

    def _mesh_cell_centers_for_gpkg(
        self,
        gpkg_path: str,
        run_id: str = "",
        table_name: str = "swe2d_mesh_results",
    ) -> Tuple[Dict[int, Tuple[float, float]], float]:
        gpkg_path = str(gpkg_path or "").strip()
        run_id = str(run_id or "").strip()
        table_name = str(table_name or "swe2d_mesh_results").strip() or "swe2d_mesh_results"
        cache_key = f"{gpkg_path}|{table_name}|{run_id}"
        if cache_key in self._velocity_cell_xy_cache:
            return (
                self._velocity_cell_xy_cache.get(cache_key, {}),
                float(self._velocity_base_len_cache.get(cache_key, 1.0)),
            )

        cell_xy: Dict[int, Tuple[float, float]] = {}
        base_len = 1.0
        mesh_layer_name = ""

        def _quote_ident(name: str) -> str:
            return '"' + str(name or "").replace('"', '""') + '"'

        expected_n_cells = 0
        candidate_layers: List[str] = [
            "swe2d_mesh_cells",
            "SWE2D_Mesh_Cells",
            "SWE2D_Mesh_Cells refined 2",
            "struct_SWE2D_Mesh_Cells",
            "smol_SWE2D_Mesh_Cells",
            "GMSH_SWE2D_Mesh_Cells",
        ]

        if gpkg_path and os.path.exists(gpkg_path):
            try:
                conn = sqlite3.connect(gpkg_path)
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND lower(name) LIKE '%mesh_cells%'"
                )
                for (nm,) in cur.fetchall():
                    nm = str(nm or "").strip()
                    if nm and nm not in candidate_layers:
                        candidate_layers.append(nm)

                if run_id:
                    cur.execute(
                        f"SELECT COUNT(DISTINCT cell_id) FROM {_quote_ident(table_name)} WHERE run_id = ?",
                        (run_id,),
                    )
                    row = cur.fetchone()
                    expected_n_cells = int(row[0]) if row and row[0] is not None else 0

                best_layer = ""
                best_score = None
                for lname in candidate_layers:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {_quote_ident(lname)}")
                        row = cur.fetchone()
                        n_cells = int(row[0]) if row and row[0] is not None else 0
                    except Exception:
                        continue
                    if n_cells <= 0:
                        continue
                    if expected_n_cells > 0:
                        score = abs(n_cells - expected_n_cells)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_layer = lname
                            if score == 0:
                                break
                    elif not best_layer:
                        best_layer = lname
                mesh_layer_name = best_layer
            except Exception:
                mesh_layer_name = ""
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        if _HAVE_QGIS_CORE and QgsVectorLayer is not None and gpkg_path and os.path.exists(gpkg_path):
            for lname in ([mesh_layer_name] if mesh_layer_name else []) + ["swe2d_mesh_cells", "SWE2D_Mesh_Cells"]:
                try:
                    lyr = QgsVectorLayer(f"{gpkg_path}|layername={lname}", lname, "ogr")
                    if lyr is None or not lyr.isValid():
                        continue
                    if lyr.fields().indexFromName("cell_id") < 0:
                        continue

                    areas = []
                    for ft in lyr.getFeatures():
                        try:
                            cid = int(ft["cell_id"])
                            geom = ft.geometry()
                            if geom is None or geom.isEmpty():
                                continue
                            cgeom = geom.centroid()
                            if cgeom is None or cgeom.isEmpty():
                                continue
                            pt = cgeom.asPoint()
                            cell_xy[cid] = (float(pt.x()), float(pt.y()))
                            try:
                                a = float(geom.area())
                                if a > 0.0:
                                    areas.append(a)
                            except Exception:
                                pass
                        except Exception:
                            continue

                    if cell_xy:
                        if areas:
                            base_len = max(0.05, float(np.sqrt(max(float(np.nanmean(np.asarray(areas))), 1.0e-9))))
                        if expected_n_cells > 0 and abs(int(len(cell_xy)) - int(expected_n_cells)) > 0:
                            self._log(
                                "Velocity overlay warning: selected mesh layer does not exactly match run cell count "
                                f"(run_id={run_id}, table={table_name}, expected={expected_n_cells}, got={len(cell_xy)}, layer={lname})."
                            )
                        break
                except Exception:
                    continue

        # Fallback for current active in-memory mesh if mesh layer was unavailable.
        if not cell_xy and self._mesh_data is not None:
            try:
                cx, cy = self._mesh_cell_centroids()
                n_cells = min(int(cx.size), int(cy.size))
                cell_xy = {i: (float(cx[i]), float(cy[i])) for i in range(n_cells)}
                area = np.asarray(self._mesh_cell_areas(), dtype=np.float64)
                base_len = max(0.05, float(np.sqrt(max(float(np.nanmean(area)), 1.0e-9))))
            except Exception:
                cell_xy = {}
                base_len = 1.0

        # Handle common 1-based cell_id schemas by also exposing shifted keys.
        if cell_xy and 0 not in cell_xy and 1 in cell_xy:
            shifted = {}
            for cid, xy in cell_xy.items():
                if cid > 0:
                    shifted[cid - 1] = xy
            cell_xy.update(shifted)

        self._velocity_cell_xy_cache[cache_key] = cell_xy
        self._velocity_base_len_cache[cache_key] = float(base_len)
        return cell_xy, float(base_len)

    def _refresh_velocity_vectors_overlay(self, t_s: float):
        self._velocity_overlay_refresh_token += 1
        refresh_token = int(self._velocity_overlay_refresh_token)
        frame_t0 = time.perf_counter()
        fetch_ms = 0.0
        build_ms = 0.0
        draw_ms = 0.0
        total_vectors = 0
        total_sources = 0
        panel = getattr(self, "_results_panel", None)
        if panel is None or not panel.velocity_overlay_enabled():
            self._clear_velocity_vectors_layers()
            return
        if not _HAVE_QGIS_CORE:
            self._clear_velocity_vectors_layers()
            return

        if not self._velocity_overlay_sources:
            self._clear_velocity_vectors_layers()
            return

        builder = self._get_velocity_vector_builder()
        if builder is None:
            self._clear_velocity_vectors_layers()
            return

        stride = max(1, int(panel.velocity_density_stride()))
        min_speed = max(0.0, float(panel.velocity_min_speed()))

        for source in list(self._velocity_overlay_sources):
            if refresh_token != self._velocity_overlay_refresh_token:
                return
            total_sources += 1
            gpkg_path = str(source.get("gpkg_path", "")).strip()
            run_id = str(source.get("run_id", "")).strip()
            table_name = str(source.get("table_name", "swe2d_mesh_results")).strip() or "swe2d_mesh_results"
            source_key = str(source.get("key", "")).strip()
            if not gpkg_path or not run_id or not source_key or not os.path.exists(gpkg_path):
                continue

            lyr = self._velocity_vectors_layer_for_source(source)
            if lyr is None:
                continue

            cell_to_fid = self._velocity_overlay_feature_ids.get(source_key)
            if cell_to_fid is None:
                cell_to_fid = {}
                self._velocity_overlay_feature_ids[source_key] = cell_to_fid

            dp = lyr.dataProvider()
            if not cell_to_fid:
                try:
                    idx_cell = lyr.fields().indexFromName("cell_id")
                    if idx_cell >= 0:
                        for f in lyr.getFeatures():
                            try:
                                cid = int(f["cell_id"])
                                cell_to_fid[cid] = int(f.id())
                            except Exception:
                                continue
                except Exception:
                    pass

            _tf0 = time.perf_counter()
            snap = builder.load_snapshot(
                gpkg_path,
                run_id,
                float(t_s),
                t_tol=1.0,
                table_name=table_name,
            )
            fetch_ms += (time.perf_counter() - _tf0) * 1000.0
            if snap is None:
                lyr.triggerRepaint()
                continue

            if not self._velocity_overlay_source_mode_logged.get(source_key, False):
                try:
                    support = self._velocity_data_support_for_run(gpkg_path, run_id, table_name)
                    if str(getattr(snap, "source", "")) == "face_flux_reconstruction":
                        self._log(
                            "Velocity rendering mode: using face-centered reconstruction "
                            f"(run_id={run_id}, table={table_name}, face_table={support.get('face_table')}, "
                            f"face_rows={int(support.get('face_rows', 0))}, cell_rows={int(support.get('cell_rows', 0))})."
                        )
                    else:
                        self._log(
                            "Velocity rendering mode: using cell-centered hu/hv "
                            f"(run_id={run_id}, table={table_name}, no usable face rows detected; "
                            f"cell_rows={int(support.get('cell_rows', 0))})."
                        )
                except Exception:
                    pass
                self._velocity_overlay_source_mode_logged[source_key] = True

            cell_xy, base_len = self._mesh_cell_centers_for_gpkg(
                gpkg_path,
                run_id=run_id,
                table_name=table_name,
            )
            if not cell_xy:
                lyr.triggerRepaint()
                continue

            _tb0 = time.perf_counter()
            vecs = builder.build_vectors(
                snapshot=snap,
                cell_xy=cell_xy,
                stride=stride,
                min_depth=1.0e-6,
                min_speed=min_speed,
            )
            build_ms += (time.perf_counter() - _tb0) * 1000.0
            if not vecs:
                existing = list(cell_to_fid.values())
                if existing:
                    dp.deleteFeatures(existing)
                    self._velocity_overlay_feature_ids[source_key] = {}
                lyr.triggerRepaint()
                continue
            total_vectors += int(len(vecs))

            source_color = self._velocity_source_color(source_key)
            idx_speed = lyr.fields().indexFromName("speed")
            idx_u = lyr.fields().indexFromName("u")
            idx_v = lyr.fields().indexFromName("v")
            idx_ang = lyr.fields().indexFromName("angle_deg")
            idx_src = lyr.fields().indexFromName("source")
            idx_color = lyr.fields().indexFromName("color")
            idx_width = lyr.fields().indexFromName("width")

            new_feats = []
            geom_updates = {}
            attr_updates = {}
            seen_cells = set()
            for v in vecs:
                speed = float(v.get("speed", 0.0))
                if speed <= 1.0e-12:
                    continue
                cid = int(v.get("cell_id", -1))
                if cid < 0:
                    continue
                seen_cells.add(cid)
                dir_u = float(v.get("u", 0.0)) / speed
                dir_v = float(v.get("v", 0.0)) / speed
                line_len = float(base_len) * min(6.0, max(1.0, 1.25 + 1.15 * speed))

                x0 = float(v.get("x", 0.0))
                y0 = float(v.get("y", 0.0))
                x1 = x0 + dir_u * line_len
                y1 = y0 + dir_v * line_len
                geom = QgsGeometry.fromPolylineXY([
                    QgsPointXY(x0, y0),
                    QgsPointXY(x1, y1),
                ])

                fid = cell_to_fid.get(cid)
                if fid is not None:
                    geom_updates[fid] = geom
                    updates = {}
                    if idx_speed >= 0:
                        updates[idx_speed] = speed
                    if idx_u >= 0:
                        updates[idx_u] = float(v.get("u", 0.0))
                    if idx_v >= 0:
                        updates[idx_v] = float(v.get("v", 0.0))
                    if idx_ang >= 0:
                        updates[idx_ang] = float(v.get("angle_deg", 0.0))
                    if idx_src >= 0:
                        updates[idx_src] = str(source.get("label", ""))
                    if idx_color >= 0:
                        updates[idx_color] = source_color
                    if idx_width >= 0:
                        updates[idx_width] = 0.8
                    if updates:
                        attr_updates[fid] = updates
                    continue

                feat = QgsFeature(lyr.fields())
                feat.setAttribute("cell_id", cid)
                feat.setAttribute("speed", speed)
                feat.setAttribute("u", float(v.get("u", 0.0)))
                feat.setAttribute("v", float(v.get("v", 0.0)))
                feat.setAttribute("angle_deg", float(v.get("angle_deg", 0.0)))
                feat.setAttribute("source", str(source.get("label", "")))
                feat.setAttribute("color", source_color)
                feat.setAttribute("width", 0.8)
                feat.setGeometry(geom)
                new_feats.append(feat)

            _td0 = time.perf_counter()
            if geom_updates:
                dp.changeGeometryValues(geom_updates)
            if attr_updates:
                dp.changeAttributeValues(attr_updates)
            if new_feats:
                ok, added = dp.addFeatures(new_feats)
                if ok:
                    for f in added:
                        try:
                            cid = int(f["cell_id"])
                            cell_to_fid[cid] = int(f.id())
                        except Exception:
                            continue

            stale_cells = [cid for cid in list(cell_to_fid.keys()) if cid not in seen_cells]
            if stale_cells:
                stale_fids = [cell_to_fid[cid] for cid in stale_cells if cid in cell_to_fid]
                if stale_fids:
                    dp.deleteFeatures(stale_fids)
                for cid in stale_cells:
                    cell_to_fid.pop(cid, None)

            if new_feats or stale_cells:
                lyr.updateExtents()
            lyr.triggerRepaint()
            draw_ms += (time.perf_counter() - _td0) * 1000.0

        iface = getattr(self, "_iface", None)
        if iface is not None and hasattr(iface, "mapCanvas"):
            try:
                iface.mapCanvas().refresh()
            except Exception:
                pass

        self._velocity_overlay_frame_counter += 1
        frame_ms = (time.perf_counter() - frame_t0) * 1000.0
        if (
            self._velocity_overlay_frame_counter % max(1, int(self._velocity_overlay_perf_log_every)) == 0
            or frame_ms > 80.0
        ):
            self._log(
                "Velocity overlay perf: "
                f"frame_ms={frame_ms:.1f}, fetch_ms={fetch_ms:.1f}, build_ms={build_ms:.1f}, draw_ms={draw_ms:.1f}, "
                f"sources={total_sources}, vectors={total_vectors}, stride={stride}"
            )

    def _refresh_streamline_traces_overlay(self, t_s: float):
        frame_t0 = time.perf_counter()
        fetch_ms = 0.0
        build_ms = 0.0
        draw_ms = 0.0
        total_traces = 0
        total_sources = 0

        panel = getattr(self, "_results_panel", None)
        if panel is None or not hasattr(panel, "streamline_overlay_enabled"):
            self._clear_streamline_traces_layers()
            return
        if not panel.streamline_overlay_enabled():
            self._clear_streamline_traces_layers()
            return
        if not _HAVE_QGIS_CORE:
            self._clear_streamline_traces_layers()
            return
        if not self._velocity_overlay_sources:
            self._clear_streamline_traces_layers()
            return

        builder = self._get_velocity_vector_builder()
        if builder is None:
            self._clear_streamline_traces_layers()
            return

        seed_count = 48
        max_steps = 30
        step_scale = 0.85
        try:
            seed_count = max(4, int(panel.streamline_seed_count()))
        except Exception:
            pass
        try:
            max_steps = max(4, int(panel.streamline_max_steps()))
        except Exception:
            pass
        try:
            step_scale = max(0.05, float(panel.streamline_step_scale()))
        except Exception:
            pass

        seed_stride = max(1, int(panel.velocity_density_stride()))
        min_speed = max(0.0, float(panel.velocity_min_speed()))

        for source in list(self._velocity_overlay_sources):
            total_sources += 1
            gpkg_path = str(source.get("gpkg_path", "")).strip()
            run_id = str(source.get("run_id", "")).strip()
            table_name = str(source.get("table_name", "swe2d_mesh_results")).strip() or "swe2d_mesh_results"
            source_key = str(source.get("key", "")).strip()
            if not gpkg_path or not run_id or not source_key or not os.path.exists(gpkg_path):
                continue

            lyr = self._streamline_traces_layer_for_source(source)
            if lyr is None:
                continue
            dp = lyr.dataProvider()

            existing_ids = [f.id() for f in lyr.getFeatures()]
            if existing_ids:
                try:
                    dp.deleteFeatures(existing_ids)
                except Exception:
                    pass

            _tf0 = time.perf_counter()
            snap = builder.load_snapshot(
                gpkg_path,
                run_id,
                float(t_s),
                t_tol=1.0,
                table_name=table_name,
            )
            fetch_ms += (time.perf_counter() - _tf0) * 1000.0
            if snap is None:
                lyr.triggerRepaint()
                continue

            cell_xy, _ = self._mesh_cell_centers_for_gpkg(
                gpkg_path,
                run_id=run_id,
                table_name=table_name,
            )
            if not cell_xy:
                lyr.triggerRepaint()
                continue

            _tb0 = time.perf_counter()
            traces = builder.build_streamline_traces(
                snapshot=snap,
                cell_xy=cell_xy,
                seed_count=seed_count,
                max_steps=max_steps,
                step_len_factor=step_scale,
                min_depth=1.0e-6,
                min_speed=min_speed,
                seed_stride=seed_stride,
            )
            build_ms += (time.perf_counter() - _tb0) * 1000.0
            if not traces:
                lyr.triggerRepaint()
                continue

            source_color = self._velocity_source_color(source_key)
            feats = []
            for tr in traces:
                pts = tr.get("points", [])
                if not isinstance(pts, list) or len(pts) < 2:
                    continue
                qpts = []
                for xy in pts:
                    try:
                        qpts.append(QgsPointXY(float(xy[0]), float(xy[1])))
                    except Exception:
                        continue
                if len(qpts) < 2:
                    continue

                mean_speed = float(tr.get("mean_speed", 0.0) or 0.0)
                style = builder.style_from_speed(mean_speed)
                feat = QgsFeature(lyr.fields())
                feat.setAttribute("trace_id", int(tr.get("trace_id", len(feats))))
                feat.setAttribute("speed", mean_speed)
                feat.setAttribute("length", float(tr.get("length", 0.0) or 0.0))
                feat.setAttribute("source", str(source.get("label", "")))
                feat.setAttribute("color", source_color)
                feat.setAttribute("width", float(style.get("width", 0.7) or 0.7))
                feat.setGeometry(QgsGeometry.fromPolylineXY(qpts))
                feats.append(feat)

            _td0 = time.perf_counter()
            if feats:
                try:
                    dp.addFeatures(feats)
                except Exception:
                    pass
                try:
                    lyr.updateExtents()
                except Exception:
                    pass
                total_traces += int(len(feats))

            lyr.triggerRepaint()
            draw_ms += (time.perf_counter() - _td0) * 1000.0

        iface = getattr(self, "_iface", None)
        if iface is not None and hasattr(iface, "mapCanvas"):
            try:
                iface.mapCanvas().refresh()
            except Exception:
                pass

        self._streamline_overlay_frame_counter += 1
        frame_ms = (time.perf_counter() - frame_t0) * 1000.0
        if (
            self._streamline_overlay_frame_counter % max(1, int(self._streamline_overlay_perf_log_every)) == 0
            or frame_ms > 100.0
        ):
            self._log(
                "Streamline overlay perf: "
                f"frame_ms={frame_ms:.1f}, fetch_ms={fetch_ms:.1f}, build_ms={build_ms:.1f}, draw_ms={draw_ms:.1f}, "
                f"sources={total_sources}, traces={total_traces}, seeds={seed_count}, steps={max_steps}"
            )

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
            support = self._velocity_data_support_for_run(gpkg_path, run_id, "swe2d_mesh_results")
            if int(support.get("face_rows", 0)) > 0:
                self._log(
                    "Velocity persistence check: both cell-centered and face-centered data are present "
                    f"(run_id={run_id}, cell_rows={int(support.get('cell_rows', 0))}, "
                    f"face_table={support.get('face_table')}, face_rows={int(support.get('face_rows', 0))})."
                )
            else:
                self._log(
                    "Velocity persistence check: only cell-centered h/hu/hv rows were stored for this run; "
                    "no face-centered flux rows were found in GeoPackage tables "
                    "(swe2d_face_flux_results / swe2d_face_results / swe2d_flux_faces)."
                )
            self._log(
                f"Stored mesh snapshot results in GeoPackage: {gpkg_path} "
                f"(run_id={run_id}, rows={len(mesh_rows)})"
            )
        finally:
            conn.close()

    def _build_mesh_snapshot_rows(self) -> List[Dict[str, object]]:
        return _build_mesh_snapshot_rows_logic(self._snapshot_timesteps)

    def _collect_run_log_metadata(self) -> Dict[str, object]:
        gate_cfg = dict(getattr(self, "_swe3d_geom_gate_last_config", {}) or {})
        gate_metrics = dict(getattr(self, "_swe3d_geom_gate_last_metrics", {}) or {})
        gate_violations = [str(v) for v in (getattr(self, "_swe3d_geom_gate_last_violations", []) or [])]

        if not gate_cfg:
            def _env_bool(name: str, default: bool) -> bool:
                raw = str(os.environ.get(name, "")).strip().lower()
                if not raw:
                    return bool(default)
                return raw not in ("0", "false", "no", "off")

            def _env_float(name: str, default: float) -> float:
                try:
                    return float(os.environ.get(name, str(default)))
                except Exception:
                    return float(default)

            def _env_int(name: str, default: int) -> int:
                try:
                    return int(os.environ.get(name, str(default)))
                except Exception:
                    return int(default)

            gate_cfg = {
                "strict": _env_bool("BACKWATER_SWE3D_GEOM_STRICT", False),
                "max_solid_fraction": max(0.0, min(1.0, _env_float("BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION", 0.98))),
                "max_seed_leak_fallbacks": max(0, _env_int("BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS", 0)),
            }

        metadata: Dict[str, object] = {
            "swe3d_geometry_gate": {
                "strict": bool(gate_cfg.get("strict", False)),
                "max_solid_fraction": float(gate_cfg.get("max_solid_fraction", 0.98)),
                "max_seed_leak_fallbacks": int(gate_cfg.get("max_seed_leak_fallbacks", 0)),
                "violation_count": int(len(gate_violations)),
            }
        }
        if gate_metrics:
            metadata["swe3d_geometry_gate"]["metrics"] = gate_metrics
        if gate_violations:
            metadata["swe3d_geometry_gate"]["violations"] = gate_violations
        return metadata

    def _persist_run_log_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        start_wallclock: str,
        end_wallclock: str,
        duration_s: float,
        log_text: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        ok = _persist_run_log_to_geopackage_logic(
            gpkg_path=gpkg_path,
            run_id=run_id,
            start_wallclock=start_wallclock,
            end_wallclock=end_wallclock,
            duration_s=duration_s,
            log_text=log_text,
            metadata=metadata,
        )
        if not ok:
            return
        self._run_log_latest_run_id = str(run_id)
        self._run_log_latest_db_path = str(gpkg_path)
        self._log(f"Stored run log in GeoPackage: {gpkg_path} (run_id={run_id})")

    def _load_run_logs_from_geopackage(
        self,
        gpkg_path: str,
    ) -> List[Dict[str, object]]:
        return _load_run_logs_from_geopackage_logic(gpkg_path=gpkg_path)

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

    def _open_3d_patch_viewer(self):
        snaps = list(getattr(self, "_three_d_patch_snapshots", []) or [])
        if not snaps:
            self._log(
                "No 3D patch snapshots available yet. "
                "Run with Experimental 3D patch mode enabled and capture mesh snapshots."
            )
            return
        dlg = SWE3DPatchViewerDialog(snaps, parent=self)
        dlg.exec()

    def _refresh_published_3d_surface_layer(self, t_s: float) -> None:
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return
        layer_id = str(getattr(self, "_three_d_patch_surface_layer_id", "") or "").strip()
        if not layer_id:
            return
        try:
            lyr = QgsProject.instance().mapLayer(layer_id)
        except Exception:
            lyr = None
        if lyr is None:
            self._three_d_patch_surface_layer_id = None
            return
        self._publish_current_3d_surface_to_qgis_3d(target_time_s=float(t_s), quiet=True)

    def _select_3d_patch_snapshot(self, target_time_s: Optional[float] = None) -> Optional[Dict[str, object]]:
        snaps = list(getattr(self, "_three_d_patch_snapshots", []) or [])
        if not snaps:
            return None

        t_use = target_time_s
        if t_use is None:
            panel = getattr(self, "_results_panel", None)
            if panel is not None and hasattr(panel, "current_time_sec"):
                try:
                    t_use = float(panel.current_time_sec())
                except Exception:
                    t_use = None

        if t_use is None:
            return snaps[-1]

        best = None
        best_dt = float("inf")
        for rec in snaps:
            try:
                t_rec = float(rec.get("t_s", 0.0))
            except Exception:
                t_rec = 0.0
            dt = abs(t_rec - float(t_use))
            if dt < best_dt:
                best_dt = dt
                best = rec
        return best if isinstance(best, dict) else snaps[-1]

    def _patch_spec_to_dict(self, spec: object) -> Optional[Dict[str, object]]:
        if spec is None:
            return None
        try:
            return {
                "nx": int(getattr(spec, "nx")),
                "ny": int(getattr(spec, "ny")),
                "nz": int(getattr(spec, "nz")),
                "dx": float(getattr(spec, "dx")),
                "dy": float(getattr(spec, "dy")),
                "dz": float(getattr(spec, "dz")),
                "origin_x": float(getattr(spec, "origin_x")),
                "origin_y": float(getattr(spec, "origin_y")),
                "origin_z": float(getattr(spec, "origin_z")),
            }
        except Exception:
            return None

    def _resolve_3d_patch_spec_for_snapshot(self, snapshot: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not isinstance(snapshot, dict):
            return None

        snap_spec = snapshot.get("patch_spec")
        if isinstance(snap_spec, dict):
            return dict(snap_spec)

        if isinstance(self._three_d_patch_last_spec, dict):
            return dict(self._three_d_patch_last_spec)

        stats = snapshot.get("stats")
        if isinstance(stats, dict):
            try:
                env = self._collect_3d_patch_env_overrides()
                spec = self._build_patch_spec_from_stats(stats, env)
                spec_dict = self._patch_spec_to_dict(spec)
                if isinstance(spec_dict, dict):
                    self._three_d_patch_last_spec = dict(spec_dict)
                    return dict(spec_dict)
            except Exception:
                pass
        return None

    def _compute_3d_patch_top_surface(
        self,
        snapshot: Dict[str, object],
        spec: Dict[str, object],
    ) -> Optional[np.ndarray]:
        if not isinstance(snapshot, dict) or not isinstance(spec, dict):
            return None

        nx = max(0, int(spec.get("nx", 0) or 0))
        ny = max(0, int(spec.get("ny", 0) or 0))
        nz = max(0, int(spec.get("nz", 0) or 0))
        dz = float(spec.get("dz", 0.0) or 0.0)
        oz = float(spec.get("origin_z", 0.0) or 0.0)
        if nx <= 0 or ny <= 0 or nz <= 0 or dz <= 0.0:
            return None

        vof = np.asarray(snapshot.get("vof", np.empty(0)), dtype=np.float64).ravel()
        expected = nx * ny * nz
        if expected <= 0 or vof.size != expected:
            return None

        vof_3d = np.reshape(vof, (nz, ny, nx), order="C")
        wet = vof_3d > 1.0e-6
        wet_any = np.any(wet, axis=0)
        out = np.full((ny, nx), np.nan, dtype=np.float64)
        if not np.any(wet_any):
            return out

        top_from_back = np.argmax(wet[::-1, :, :], axis=0)
        k_top = (nz - 1 - top_from_back).astype(np.int32)

        j_idx, i_idx = np.where(wet_any)
        k_idx = k_top[j_idx, i_idx]
        f_top = np.clip(vof_3d[k_idx, j_idx, i_idx], 0.0, 1.0)
        out[j_idx, i_idx] = oz + (k_idx.astype(np.float64) + f_top) * dz
        return out

    def _get_or_create_3d_patch_surface_layer(self):
        if not _HAVE_QGIS_CORE or QgsProject is None or QgsVectorLayer is None:
            return None

        proj = QgsProject.instance()
        layer_id = str(getattr(self, "_three_d_patch_surface_layer_id", "") or "").strip()
        if layer_id:
            try:
                lyr = proj.mapLayer(layer_id)
            except Exception:
                lyr = None
            if lyr is not None:
                return lyr

        for lyr in proj.mapLayers().values():
            try:
                if isinstance(lyr, QgsVectorLayer) and str(lyr.name()) == "SWE3D_Patch_FreeSurface":
                    self._three_d_patch_surface_layer_id = str(lyr.id())
                    return lyr
            except Exception:
                continue

        crs_auth = "EPSG:4326"
        try:
            proj_crs = proj.crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception:
            pass

        uri = (
            f"PolygonZ?crs={crs_auth}"
            "&field=tri_id:integer"
            "&field=t_s:double"
            "&field=z_mean:double"
        )
        lyr = QgsVectorLayer(uri, "SWE3D_Patch_FreeSurface", "memory")
        if lyr is None or not lyr.isValid():
            self._log("Could not create SWE3D_Patch_FreeSurface memory layer.")
            return None

        try:
            proj.addMapLayer(lyr)
            self._three_d_patch_surface_layer_id = str(lyr.id())
        except Exception as exc:
            self._log(f"Could not add SWE3D_Patch_FreeSurface layer to project: {exc}")
            return None

        self._apply_3d_renderer_to_patch_surface_layer(lyr)
        return lyr

    def _apply_3d_renderer_to_patch_surface_layer(self, layer) -> bool:
        if layer is None or not hasattr(layer, "setRenderer3D"):
            return False
        try:
            from qgis._3d import QgsPolygon3DSymbol, QgsVectorLayer3DRenderer
        except Exception:
            return False
        try:
            symbol = QgsPolygon3DSymbol()
            try:
                from qgis.core import Qgis

                if hasattr(symbol, "setAltitudeClamping") and hasattr(Qgis, "AltitudeClamping"):
                    symbol.setAltitudeClamping(Qgis.AltitudeClamping.Absolute)
            except Exception:
                pass
            renderer = QgsVectorLayer3DRenderer(symbol)
            layer.setRenderer3D(renderer)
            return True
        except Exception as exc:
            self._log(f"SWE3D free-surface 3D renderer setup warning: {exc}")
            return False

    def _update_3d_patch_surface_layer(
        self,
        layer,
        t_s: float,
        spec: Dict[str, object],
        z_top: np.ndarray,
    ) -> int:
        if layer is None:
            return 0
        if z_top is None or z_top.ndim != 2:
            return 0

        nx = max(0, int(spec.get("nx", 0) or 0))
        ny = max(0, int(spec.get("ny", 0) or 0))
        dx = float(spec.get("dx", 0.0) or 0.0)
        dy = float(spec.get("dy", 0.0) or 0.0)
        ox = float(spec.get("origin_x", 0.0) or 0.0)
        oy = float(spec.get("origin_y", 0.0) or 0.0)
        if nx <= 1 or ny <= 1 or dx <= 0.0 or dy <= 0.0:
            return 0
        if z_top.shape[0] != ny or z_top.shape[1] != nx:
            return 0

        provider = layer.dataProvider()
        if provider is None:
            raise RuntimeError("SWE3D surface layer has no data provider.")

        cleared = False
        if hasattr(provider, "truncate"):
            try:
                cleared = bool(provider.truncate())
            except Exception:
                cleared = False
        if not cleared:
            try:
                fids = [int(ft.id()) for ft in layer.getFeatures()]
                if fids:
                    provider.deleteFeatures(fids)
            except Exception:
                pass

        x_centers = ox + (np.arange(nx, dtype=np.float64) + 0.5) * dx
        y_centers = oy + (np.arange(ny, dtype=np.float64) + 0.5) * dy
        fields = layer.fields()
        feats: List[QgsFeature] = []
        tri_id = 0

        def _append_tri(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
            nonlocal tri_id
            if not (
                np.isfinite(z0)
                and np.isfinite(z1)
                and np.isfinite(z2)
                and np.isfinite(x0)
                and np.isfinite(y0)
                and np.isfinite(x1)
                and np.isfinite(y1)
                and np.isfinite(x2)
                and np.isfinite(y2)
            ):
                return

            wkt = (
                f"Polygon Z (({x0:.9g} {y0:.9g} {z0:.9g}, "
                f"{x1:.9g} {y1:.9g} {z1:.9g}, "
                f"{x2:.9g} {y2:.9g} {z2:.9g}, "
                f"{x0:.9g} {y0:.9g} {z0:.9g}))"
            )
            geom = QgsGeometry.fromWkt(wkt)
            if geom is None or geom.isEmpty():
                return

            tri_id += 1
            ft = QgsFeature(fields)
            ft.setGeometry(geom)
            ft["tri_id"] = int(tri_id)
            ft["t_s"] = float(t_s)
            ft["z_mean"] = float((z0 + z1 + z2) / 3.0)
            feats.append(ft)

        for j in range(ny - 1):
            y0 = float(y_centers[j])
            y1 = float(y_centers[j + 1])
            for i in range(nx - 1):
                x0 = float(x_centers[i])
                x1 = float(x_centers[i + 1])

                z00 = float(z_top[j, i])
                z10 = float(z_top[j, i + 1])
                z01 = float(z_top[j + 1, i])
                z11 = float(z_top[j + 1, i + 1])

                _append_tri(x0, y0, z00, x1, y0, z10, x1, y1, z11)
                _append_tri(x0, y0, z00, x1, y1, z11, x0, y1, z01)

        if feats:
            add_result = provider.addFeatures(feats)
            add_ok = bool(add_result[0]) if isinstance(add_result, tuple) else bool(add_result)
            if not add_ok:
                raise RuntimeError("Failed to write free-surface features to memory layer.")

        layer.updateExtents()
        layer.triggerRepaint()
        try:
            layer.setCustomProperty("swe2d/three_d_surface_time_s", float(t_s))
        except Exception:
            pass
        return int(tri_id)

    def _publish_current_3d_surface_to_qgis_3d(
        self,
        target_time_s: Optional[float] = None,
        quiet: bool = False,
    ) -> None:
        if not _HAVE_QGIS_CORE or QgsProject is None:
            if not quiet:
                self._log("QGIS core APIs are unavailable; cannot publish 3D free-surface layer.")
            return

        snapshot = self._select_3d_patch_snapshot(target_time_s=target_time_s)
        if snapshot is None:
            if not quiet:
                self._log("No 3D patch snapshots are available for map publication.")
            return

        spec = self._resolve_3d_patch_spec_for_snapshot(snapshot)
        if not isinstance(spec, dict):
            if not quiet:
                self._log(
                    "3D patch spec is unavailable for this snapshot. "
                    "Run Experimental 3D mode again so ROI/origin metadata is captured."
                )
            return

        z_top = self._compute_3d_patch_top_surface(snapshot, spec)
        if z_top is None:
            if not quiet:
                self._log("Could not compute 3D free-surface elevations from snapshot VOF data.")
            return

        layer = self._get_or_create_3d_patch_surface_layer()
        if layer is None:
            return

        renderer_ok = self._apply_3d_renderer_to_patch_surface_layer(layer)
        if not renderer_ok and not quiet:
            self._log(
                "QGIS 3D renderer API is unavailable in this runtime; "
                "surface layer was still published as a regular vector layer."
            )

        try:
            t_s = float(snapshot.get("t_s", 0.0))
        except Exception:
            t_s = 0.0

        try:
            n_tri = self._update_3d_patch_surface_layer(layer=layer, t_s=t_s, spec=spec, z_top=z_top)
        except Exception as exc:
            if not quiet:
                self._log(f"3D free-surface publish failed: {exc}")
            return

        if not quiet:
            wet_cols = int(np.count_nonzero(np.isfinite(z_top)))
            nx = int(spec.get("nx", 0) or 0)
            ny = int(spec.get("ny", 0) or 0)
            self._log(
                "Published 3D free-surface layer for QGIS 3D view: "
                f"time={t_s:.3f}s, wet_columns={wet_cols}/{max(1, nx * ny)}, triangles={n_tri}."
            )

        canvas = self._resolve_map_canvas()
        if canvas is not None:
            try:
                canvas.refresh()
            except Exception:
                pass

    def _build_internal_flow_source_cms(self) -> Optional[np.ndarray]:
        forcing = self._build_internal_flow_forcing()
        if forcing is None:
            return None
        return self._internal_flow_source_cms_at_time(forcing, 0.0)

    def _build_internal_flow_forcing(self) -> Optional[Dict[str, object]]:
        internal_flow_layer_combo = getattr(self, "internal_flow_layer_combo", None)
        field_edit = getattr(self, "internal_flow_field_edit", None)
        requested_field_name = str(field_edit.text() or "q_cms") if field_edit is not None else "q_cms"

        return _build_internal_flow_forcing_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            internal_flow_layer_combo=internal_flow_layer_combo,
            combo_layer_fn=self._combo_layer,
            requested_field_name=requested_field_name,
            iter_project_layers_fn=self._iter_project_layers,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            parse_hydrograph_text_fn=self._parse_hydrograph_text,
            hydrograph_from_layer_fn=self._hydrograph_from_layer,
            qgs_vector_layer_cls=QgsVectorLayer,
            qgs_wkb_types=QgsWkbTypes,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _internal_flow_source_cms_at_time(self, forcing: Optional[Dict[str, object]], t_sec: float) -> Optional[np.ndarray]:
        return _internal_flow_source_cms_at_time_logic(forcing, t_sec, self._interp_hydrograph)

    def _apply_external_sources(
        self,
        backend: SWE2DBackend,
        dt_step: float,
        rain_rate_model,
        cell_source_model: Optional[np.ndarray],
        coupled_source_rate: Optional[np.ndarray] = None,
        prefer_native_injection: bool = False,
    ) -> None:
        src_cap_widget = getattr(self, "max_source_rate_spin", None)
        max_rel_widget = getattr(self, "max_rel_depth_increase_spin", None)
        hmin_widget = getattr(self, "h_min_spin", None)
        dh_cap_widget = getattr(self, "max_source_depth_step_spin", None)
        shallow_widget = getattr(self, "shallow_damping_depth_spin", None)
        min_speed_cap_widget = getattr(self, "momentum_cap_min_speed_spin", None)
        celerity_mult_widget = getattr(self, "momentum_cap_celerity_mult_spin", None)

        mesh_cell_areas = self._mesh_cell_areas() if cell_source_model is not None else None

        _apply_external_sources_logic(
            backend=backend,
            dt_step=dt_step,
            rain_rate_model=rain_rate_model,
            cell_source_model=cell_source_model,
            coupled_source_rate=coupled_source_rate,
            prefer_native_injection=prefer_native_injection,
            mesh_cell_areas=mesh_cell_areas,
            max_source_rate=float(src_cap_widget.value()) if src_cap_widget is not None else 0.0,
            h_min=float(hmin_widget.value()) if hmin_widget is not None else 1.0e-4,
            max_rel_depth_increase=float(max_rel_widget.value()) if max_rel_widget is not None else 0.0,
            max_source_depth_step=float(dh_cap_widget.value()) if dh_cap_widget is not None else 0.0,
            shallow_damping_depth=float(shallow_widget.value()) if shallow_widget is not None else 0.0,
            momentum_cap_min_speed=float(min_speed_cap_widget.value()) if min_speed_cap_widget is not None else 50.0,
            momentum_cap_celerity_mult=float(celerity_mult_widget.value()) if celerity_mult_widget is not None else 20.0,
        )

    def _build_spatial_manning_array(self) -> Optional[np.ndarray]:
        return _build_spatial_manning_array_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            manning_layer_combo=getattr(self, "manning_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            default_n=float(self.n_mann_spin.value()) if hasattr(self, "n_mann_spin") else 0.03,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _build_spatial_cn_array(self) -> np.ndarray:
        return _build_spatial_cn_array_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            cn_layer_combo=getattr(self, "cn_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            default_cn=float(self.cn_default_spin.value()) if hasattr(self, "cn_default_spin") else 75.0,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _build_thiessen_rain_cn_forcing(self) -> Optional[ThiessenRainCNForcing]:
        use_spatial_rain_cn = bool(self.use_spatial_rain_cn_chk.isChecked()) if hasattr(self, "use_spatial_rain_cn_chk") else False
        ia_ratio = float(self.ia_ratio_spin.value()) if hasattr(self, "ia_ratio_spin") else 0.2
        infiltration_method = self.infiltration_method_combo.currentData() if hasattr(self, "infiltration_method_combo") else "scs_cn"
        return _build_thiessen_rain_cn_forcing_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            thiessen_rain_cn_forcing_cls=ThiessenRainCNForcing,
            gauge_cls=Gauge,
            build_hyetograph_fn=build_hyetograph,
            assign_cells_to_nearest_gauge_fn=assign_cells_to_nearest_gauge,
            inspect_hyetograph_rows_fn=inspect_hyetograph_rows,
            use_spatial_rain_cn=use_spatial_rain_cn,
            rain_gage_layer_combo=getattr(self, "rain_gage_layer_combo", None),
            hyetograph_layer_combo=getattr(self, "hyetograph_layer_combo", None),
            storm_area_layer_combo=getattr(self, "storm_area_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            boundary_buffer_cells_fn=self._boundary_buffer_cells,
            build_spatial_cn_array_fn=self._build_spatial_cn_array,
            ia_ratio=ia_ratio,
            infiltration_method=infiltration_method or "scs_cn",
            rain_boundary_buffer_rings=int(self.rain_boundary_buffer_rings_spin.value()) if hasattr(self, "rain_boundary_buffer_rings_spin") else 1,
            qgs_wkb_types=QgsWkbTypes,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

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
            "experimental_3d_obj_layer_combo",
            "experimental_3d_obj_inside_points_layer_combo",
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
            ("equation_set_combo", "currentIndexChanged"),
            ("experimental_3d_mode_chk", "toggled"),
            ("experimental_3d_coupling_mode_combo", "currentIndexChanged"),
            ("experimental_3d_patch_face_len_x_spin", "valueChanged"),
            ("experimental_3d_patch_face_len_y_spin", "valueChanged"),
            ("experimental_3d_patch_face_len_z_spin", "valueChanged"),
            ("experimental_3d_patch_xmin_edit", "editingFinished"),
            ("experimental_3d_patch_xmax_edit", "editingFinished"),
            ("experimental_3d_patch_ymin_edit", "editingFinished"),
            ("experimental_3d_patch_ymax_edit", "editingFinished"),
            ("experimental_3d_patch_zmin_edit", "editingFinished"),
            ("experimental_3d_patch_zmax_edit", "editingFinished"),
            ("experimental_3d_obj_solids_chk", "toggled"),
            ("experimental_3d_obj_method_combo", "currentIndexChanged"),
            ("experimental_3d_geom_sanitize_chk", "toggled"),
            ("experimental_3d_geom_phi_snap_spin", "valueChanged"),
            ("experimental_3d_geom_area_snap_spin", "valueChanged"),
            ("experimental_3d_obj_layer_combo", "currentIndexChanged"),
            ("experimental_3d_obj_path_field_edit", "editingFinished"),
            ("experimental_3d_obj_default_path_edit", "editingFinished"),
            ("experimental_3d_obj_scale_field_edit", "editingFinished"),
            ("experimental_3d_obj_yaw_field_edit", "editingFinished"),
            ("experimental_3d_obj_z_offset_field_edit", "editingFinished"),
            ("experimental_3d_obj_inside_points_layer_combo", "currentIndexChanged"),
            ("experimental_3d_obj_instance_id_field_edit", "editingFinished"),
            ("experimental_3d_obj_inside_id_field_edit", "editingFinished"),
            ("experimental_3d_obj_inside_z_field_edit", "editingFinished"),
            ("experimental_3d_obj_use_terrain_chk", "toggled"),
            ("experimental_3d_obj_ab_compare_chk", "toggled"),
            ("experimental_3d_obj_ab_probe_steps_spin", "valueChanged"),
            ("experimental_3d_obj_export_obj_chk", "toggled"),
            ("experimental_3d_obj_export_obj_path_edit", "editingFinished"),
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
        widget_specs.extend(list(getattr(self, "_experimental_3d_bc_signal_specs", []) or []))

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
            # Parameters"
            "n_mann_spin","cfl_spin",
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
            "reconstruction_combo", "temporal_order_combo", "equation_set_combo", "experimental_3d_mode_chk",
            "experimental_3d_coupling_mode_combo",
            "experimental_3d_patch_face_len_x_spin", "experimental_3d_patch_face_len_y_spin", "experimental_3d_patch_face_len_z_spin",
            "experimental_3d_patch_xmin_edit", "experimental_3d_patch_xmax_edit",
            "experimental_3d_patch_ymin_edit", "experimental_3d_patch_ymax_edit",
            "experimental_3d_patch_zmin_edit", "experimental_3d_patch_zmax_edit",
            "experimental_3d_obj_solids_chk", "experimental_3d_obj_method_combo", "experimental_3d_obj_layer_combo",
            "experimental_3d_geom_sanitize_chk", "experimental_3d_geom_phi_snap_spin", "experimental_3d_geom_area_snap_spin",
            "experimental_3d_obj_path_field_edit", "experimental_3d_obj_default_path_edit",
            "experimental_3d_obj_scale_field_edit", "experimental_3d_obj_yaw_field_edit",
            "experimental_3d_obj_z_offset_field_edit", "experimental_3d_obj_inside_points_layer_combo",
            "experimental_3d_obj_instance_id_field_edit", "experimental_3d_obj_inside_id_field_edit",
            "experimental_3d_obj_inside_z_field_edit", "experimental_3d_obj_use_terrain_chk",
            "experimental_3d_obj_ab_compare_chk", "experimental_3d_obj_ab_probe_steps_spin",
            "experimental_3d_obj_export_obj_chk", "experimental_3d_obj_export_obj_path_edit",
            "degen_mode_combo", "coupling_loop_combo",
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
        widget_attrs.extend(list(getattr(self, "_experimental_3d_bc_widget_attrs", []) or []))
        
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
        
            self._sync_experimental_3d_mode_widgets()
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
        if not _ensure_netcdf4_available():
            detail = ""
            if _NETCDF4_IMPORT_ERROR is not None:
                detail = f" Import error: {_NETCDF4_IMPORT_ERROR}"
            raise RuntimeError(
                "netCDF4 is unavailable (missing or binary-incompatible in current QGIS Python)."
                " Install a compatible netCDF4 build for this QGIS environment." + detail
            )
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
            ds.title = "SWE2D HYDRA model results"
            ds.institution = "qgis-hydra-plugin"
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
        return _mesh_boundary_edges_logic(self._mesh_data)

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
        return _apply_bc_layer_overrides_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            bc_type=bc_type,
            bc_val=bc_val,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _collect_bc_layer_hydrographs(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
        return _collect_bc_layer_hydrographs_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            iter_project_layers_fn=self._iter_project_layers,
            hydrograph_from_layer_fn=self._hydrograph_from_layer,
            parse_hydrograph_text_fn=self._parse_hydrograph_text,
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
            qgs_vector_layer_cls=QgsVectorLayer,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _collect_bc_layer_edge_groups(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> Dict[int, str]:
        return _collect_bc_layer_edge_groups_qgis_logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
        )

    def _collect_boundary_arrays(self):
        return _collect_boundary_arrays_logic(
            mesh_data=self._mesh_data,
            mesh_boundary_edges_fn=self._mesh_boundary_edges,
            default_bc_for_edges_fn=self._default_bc_for_edges,
            apply_bc_layer_overrides_fn=self._apply_bc_layer_overrides,
            log_fn=self._log,
        )

    def _parse_time_hours(self, token: str) -> float:
        return _parse_time_hours_logic(token)

    def _parse_run_duration_seconds(self) -> float:
        hrs = self._parse_time_hours(self.run_time_edit.text())
        if hrs <= 0.0:
            raise ValueError("run duration must be > 0")
        return 3600.0 * hrs

    def _parse_hydrograph_text(self, text: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return _parse_hydrograph_text_logic(text, parse_time_hours_fn=self._parse_time_hours)

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
        return _hydrograph_from_layer_logic(
            layer,
            hydrograph_id=hydrograph_id,
            bc_type=bc_type,
            parse_time_hours_fn=self._parse_time_hours,
            vector_layer_type=QgsVectorLayer,
        )

    def _interp_hydrograph(self, hg: Tuple[np.ndarray, np.ndarray], t_sec: float) -> float:
        return _interp_hydrograph_logic(hg, t_sec)

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
        progressive = True
        if hasattr(self, "inflow_progressive_chk") and self.inflow_progressive_chk is not None:
            try:
                progressive = bool(self.inflow_progressive_chk.isChecked())
            except Exception:
                progressive = True

        return _distribute_total_flow_to_unit_q_logic(
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            bc_type_step=bc_type_step,
            bc_val_step=bc_val_step,
            bc_type_template=bc_type_template,
            side_hydrographs=side_hydrographs,
            node_x=self._mesh_data["node_x"],
            node_y=self._mesh_data["node_y"],
            node_z=self._mesh_data["node_z"],
            progressive=progressive,
            ts_flow_code=_BC_TS_FLOW,
            edge_hydrographs=edge_hydrographs,
        )

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
        return _apply_timeseries_bc_values_logic(
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            bc_type=bc_type,
            bc_val=bc_val,
            side_hydrographs=side_hydrographs,
            node_x=self._mesh_data["node_x"],
            node_y=self._mesh_data["node_y"],
            t_sec=t_sec,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
            edge_hydrographs=edge_hydrographs,
        )

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
        return _inflow_adjacent_cells_logic(self._mesh_data, bc_n0, bc_n1, bc_tp)

    def _boundary_buffer_cells(self, n_rings: int) -> np.ndarray:
        """Return cell indices within n_rings of the mesh boundary."""
        return _boundary_buffer_cells_logic(self._mesh_data, n_rings)

    def _initial_state(self, bc_n0: Optional[np.ndarray] = None, bc_n1: Optional[np.ndarray] = None, bc_tp: Optional[np.ndarray] = None):
        assert self._mesh_data is not None
        mode = str(self.initial_condition_combo.currentData() if hasattr(self, "initial_condition_combo") else "dry")
        return _initial_state_logic(
            mesh_data=self._mesh_data,
            mode=mode,
            initial_depth=float(self.initial_depth_spin.value()) if hasattr(self, "initial_depth_spin") else 0.0,
            initial_wse=float(self.initial_wse_spin.value()) if hasattr(self, "initial_wse_spin") else 0.0,
            h_min=float(self.h_min_spin.value()) if hasattr(self, "h_min_spin") else 1.0e-6,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            log_fn=self._log,
        )

    def _set_3d_patch_roi_from_mesh(self):
        if self._mesh_data is None:
            self._log("3D patch ROI sync skipped: generate or import mesh first.")
            return
        try:
            node_x = np.asarray(self._mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
            node_y = np.asarray(self._mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
            node_z = np.asarray(self._mesh_data.get("node_z", np.empty(0)), dtype=np.float64).ravel()
            if node_x.size <= 0 or node_y.size <= 0:
                self._log("3D patch ROI sync skipped: mesh node coordinates are unavailable.")
                return
            xmin = float(np.min(node_x))
            xmax = float(np.max(node_x))
            ymin = float(np.min(node_y))
            ymax = float(np.max(node_y))
            target_len_x = (
                float(self.experimental_3d_patch_face_len_x_spin.value())
                if hasattr(self, "experimental_3d_patch_face_len_x_spin")
                else 5.0
            )
            target_len_y = (
                float(self.experimental_3d_patch_face_len_y_spin.value())
                if hasattr(self, "experimental_3d_patch_face_len_y_spin")
                else 5.0
            )
            nx_hint = max(8, min(256, int(math.ceil(max(xmax - xmin, 1.0e-9) / max(target_len_x, 1.0e-6)))))
            ny_hint = max(8, min(256, int(math.ceil(max(ymax - ymin, 1.0e-9) / max(target_len_y, 1.0e-6)))))
            terrain_zmin = self._sample_terrain_min_z_for_roi(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                nx_hint=nx_hint,
                ny_hint=ny_hint,
            )
            if node_z.size > 0:
                zmin = float(np.min(node_z))
                zmax = float(np.max(node_z))
                if zmax <= zmin:
                    zmax = zmin + 1.0
            else:
                zmin = 0.0
                zmax = 1.0

            if terrain_zmin is not None and np.isfinite(terrain_zmin):
                zmin = float(terrain_zmin)
                if zmax <= zmin:
                    zmax = zmin + 1.0

            self.experimental_3d_patch_xmin_edit.setText(f"{xmin:.9g}")
            self.experimental_3d_patch_xmax_edit.setText(f"{xmax:.9g}")
            self.experimental_3d_patch_ymin_edit.setText(f"{ymin:.9g}")
            self.experimental_3d_patch_ymax_edit.setText(f"{ymax:.9g}")
            self.experimental_3d_patch_zmin_edit.setText(f"{zmin:.9g}")
            self.experimental_3d_patch_zmax_edit.setText(f"{zmax:.9g}")
            self._log(
                "3D patch ROI set from mesh extents: "
                f"x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}], z=[{zmin:.3f}, {zmax:.3f}]"
            )
            if terrain_zmin is not None and np.isfinite(terrain_zmin):
                self._log(
                    "3D patch z-min source: terrain DEM minimum sampled in ROI "
                    f"(zmin={float(terrain_zmin):.3f})."
                )
        except Exception as exc:
            self._log(f"3D patch ROI sync failed: {exc}")

    def _edit_optional_float(self, edit: QtWidgets.QLineEdit) -> Optional[float]:
        return _parse_optional_float_text_logic(str(edit.text() or ""))

    def _sample_terrain_min_z_for_roi(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx_hint: int = 64,
        ny_hint: int = 64,
    ) -> Optional[float]:
        return _sample_terrain_min_z_for_roi_qgis_logic(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_pointxy_cls=QgsPointXY,
            terrain_layer_combo=getattr(self, "terrain_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            nx_hint=nx_hint,
            ny_hint=ny_hint,
        )

    def _collect_3d_patch_env_overrides(self) -> Dict[str, str]:
        if self._mesh_data is None:
            raise RuntimeError("Mesh is required before configuring 3D patch ROI/resolution.")
        return _collect_3d_patch_env_overrides_delegate_logic(
            ui=self,
            mesh_data=self._mesh_data,
            target_len_x=float(self.experimental_3d_patch_face_len_x_spin.value()),
            target_len_y=float(self.experimental_3d_patch_face_len_y_spin.value()),
            target_len_z=float(self.experimental_3d_patch_face_len_z_spin.value()),
            edit_optional_float_callback=self._edit_optional_float,
            sample_terrain_min_z_for_roi_callback=self._sample_terrain_min_z_for_roi,
            collect_patch_env_overrides_callback=_collect_3d_patch_env_overrides_logic,
            bed_manning_n=float(self.n_mann_spin.value()),
            log_callback=self._log,
            set_patch_zmin_text_callback=self.experimental_3d_patch_zmin_edit.setText,
            collect_face_bc_env_overrides_callback=self._collect_3d_patch_face_bc_env_overrides,
        )

    def _apply_env_overrides(self, overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
        previous: Dict[str, Optional[str]] = {}
        for key, value in (overrides or {}).items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        return previous

    def _restore_env_overrides(self, previous: Dict[str, Optional[str]]) -> None:
        for key, old in (previous or {}).items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(old)

    def _resolve_layer_field_name(self, layer: object, requested_name: str) -> str:
        return _resolve_layer_field_name_qgis_logic(layer, requested_name)

    def _parse_obj_scale_value(self, raw_value: object) -> Tuple[float, float, float]:
        return _parse_obj_scale_value_runtime_logic(raw_value)

    def _parse_feature_float(self, feature: object, field_name: str, default: float) -> float:
        return _parse_feature_float_qgis_logic(feature, field_name, default)

    def _resolve_obj_model_path(self, raw_path: str) -> str:
        project_file_path = ""
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                project_file_path = str(QgsProject.instance().fileName() or "").strip()
            except Exception:
                project_file_path = ""
        return _resolve_obj_model_path_runtime_logic(
            raw_path=raw_path,
            model_gpkg_path=str(self._model_gpkg_path or ""),
            project_file_path=project_file_path,
            module_dir=os.path.dirname(__file__),
            cwd=os.getcwd(),
        )

    def _infer_obj_path_from_layer_3d_renderer(self, layer: object) -> str:
        return _infer_obj_path_from_layer_3d_renderer_qgis_logic(layer)

    def _build_patch_terrain_surface(self, spec: object) -> Optional[np.ndarray]:
        if not _HAVE_QGIS_CORE or QgsPointXY is None:
            return None
        if not hasattr(self, "terrain_layer_combo"):
            return None
        raster_layer = self._combo_layer(self.terrain_layer_combo, "raster")
        return _build_patch_terrain_surface_qgis_logic(
            spec=spec,
            raster_layer=raster_layer,
            qgs_point_xy_cls=QgsPointXY,
        )

    def _build_patch_spec_from_stats(
        self,
        patch_stats: Dict[str, object],
        swe3d_env_overrides: Dict[str, str],
    ) -> Optional[object]:
        return _build_patch_spec_from_stats_runtime_logic(
            patch_stats=patch_stats,
            swe3d_env_overrides=swe3d_env_overrides,
            patch_grid_spec_cls=PatchGridSpec,
        )

    def _experimental_3d_selected_obstacle_method(self) -> str:
        obstacle_method = "fractional_cutcell"
        method_combo = getattr(self, "experimental_3d_obj_method_combo", None)
        if isinstance(method_combo, QtWidgets.QComboBox):
            try:
                raw_method = method_combo.currentData()
                if raw_method is None:
                    raw_method = method_combo.currentText()
                obstacle_method = str(raw_method or "fractional_cutcell").strip() or "fractional_cutcell"
            except Exception:
                obstacle_method = "fractional_cutcell"
        if obstacle_method not in ("fractional_cutcell", "favor1981_porosity"):
            obstacle_method = "fractional_cutcell"
        return obstacle_method

    def _experimental_3d_geometry_sanitize_options(self) -> Dict[str, object]:
        enabled = bool(
            getattr(self, "experimental_3d_geom_sanitize_chk", None)
            and self.experimental_3d_geom_sanitize_chk.isChecked()
        )

        phi_snap_min = 0.005
        area_snap_min = 0.01
        try:
            if hasattr(self, "experimental_3d_geom_phi_snap_spin"):
                phi_snap_min = float(self.experimental_3d_geom_phi_snap_spin.value())
        except Exception:
            phi_snap_min = 0.005
        try:
            if hasattr(self, "experimental_3d_geom_area_snap_spin"):
                area_snap_min = float(self.experimental_3d_geom_area_snap_spin.value())
        except Exception:
            area_snap_min = 0.01

        if not np.isfinite(phi_snap_min):
            phi_snap_min = 0.005
        if not np.isfinite(area_snap_min):
            area_snap_min = 0.01
        phi_snap_min = max(0.0, min(1.0, phi_snap_min))
        area_snap_min = max(0.0, min(1.0, area_snap_min))

        return {
            "sanitize": enabled,
            "phi_snap_min": phi_snap_min,
            "area_snap_min": area_snap_min,
        }

    def _resolve_experimental_3d_obj_export_path(self, obstacle_method: str) -> str:
        path_edit = getattr(self, "experimental_3d_obj_export_obj_path_edit", None)
        configured = ""
        if isinstance(path_edit, QtWidgets.QLineEdit):
            configured = str(path_edit.text() or "").strip()
        if configured:
            return os.path.abspath(os.path.expanduser(configured))

        base_dir = ""
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                base_dir = str(QgsProject.instance().homePath() or "").strip()
            except Exception:
                base_dir = ""
        if not base_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        safe_method = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in str(obstacle_method or "method"))
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, "swe3d_exports", f"swe3d_fluid_{safe_method}_{stamp}.obj")

    def _run_experimental_3d_obj_method_probe(
        self,
        backend_builder: Callable[[], object],
        method_name: str,
        phi: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        swe3d_env_overrides: Dict[str, str],
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        bc_vl: np.ndarray,
        probe_steps: int,
    ) -> Dict[str, float]:
        return _run_experimental_3d_obj_method_probe_runtime_logic(
            wb=self,
            backend_builder=backend_builder,
            method_name=method_name,
            phi=phi,
            ax=ax,
            ay=ay,
            az=az,
            swe3d_env_overrides=swe3d_env_overrides,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            probe_steps=probe_steps,
            coupling_mode_enum=SWE2DThreeDCouplingMode,
        )

    def _boundary_edge_owner_cells(
        self,
        edge_n0: np.ndarray,
        edge_n1: np.ndarray,
    ) -> np.ndarray:
        return _boundary_edge_owner_cells_runtime_logic(
            mesh_data=self._mesh_data,
            edge_n0=edge_n0,
            edge_n1=edge_n1,
        )

    def _build_experimental_3d_interface_contract_arrays(
        self,
        patch_stats: Dict[str, object],
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        return _build_experimental_3d_interface_contract_arrays_runtime_logic(
            wb=self,
            patch_stats=patch_stats,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_inflow_q=_BC_INFLOW_Q,
            bc_ts_flow=_BC_TS_FLOW,
            bc_ts_stage=_BC_TS_STAGE,
        )

    def _upload_experimental_3d_interface_contract(
        self,
        backend: object,
        patch_stats: Dict[str, object],
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        coupling_mode: int,
    ) -> None:
        _upload_experimental_3d_interface_contract_runtime_logic(
            wb=self,
            backend=backend,
            patch_stats=patch_stats,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            coupling_mode=coupling_mode,
            coupling_mode_enum=SWE2DThreeDCouplingMode,
        )

    def _initialize_experimental_3d_patch_state(
        self,
        backend: object,
        patch_stats: Dict[str, object],
        swe3d_env_overrides: Dict[str, str],
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        bc_vl: np.ndarray,
        log_notes: bool = True,
    ) -> None:
        _initialize_experimental_3d_patch_state_runtime_logic(
            wb=self,
            backend=backend,
            patch_stats=patch_stats,
            swe3d_env_overrides=swe3d_env_overrides,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            log_notes=bool(log_notes),
            bc_inflow_q=_BC_INFLOW_Q,
            bc_ts_flow=_BC_TS_FLOW,
            bc_ts_stage=_BC_TS_STAGE,
            coupling_mode_enum=SWE2DThreeDCouplingMode,
        )

    def _upload_experimental_3d_obj_geometry(
        self,
        backend: object,
        patch_stats: Dict[str, object],
        swe3d_env_overrides: Dict[str, str],
        backend_builder: Optional[Callable[[], object]] = None,
        bc_n0: Optional[np.ndarray] = None,
        bc_n1: Optional[np.ndarray] = None,
        bc_tp: Optional[np.ndarray] = None,
        bc_vl: Optional[np.ndarray] = None,
    ) -> None:
        _upload_experimental_3d_obj_geometry_runtime_logic(
            wb=self,
            backend=backend,
            patch_stats=patch_stats,
            swe3d_env_overrides=swe3d_env_overrides,
            backend_builder=backend_builder,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            patch_grid_spec_cls=PatchGridSpec,
            load_obj_mesh_fn=load_obj_mesh,
            apply_instance_transform_fn=apply_instance_transform,
            build_static_geometry_tensors_fn=build_static_geometry_tensors,
            write_solid_voxels_obj_fn=write_solid_voxels_obj,
            write_fluid_voxels_obj_fn=write_fluid_voxels_obj,
        )

    def _append_3d_patch_snapshot(self, t_s: float, stats: Dict[str, object], vof: np.ndarray) -> None:
        if not isinstance(stats, dict):
            return
        arr = np.asarray(vof, dtype=np.float64).ravel()
        nx = max(0, int(stats.get("nx", 0) or 0))
        ny = max(0, int(stats.get("ny", 0) or 0))
        nz = max(0, int(stats.get("nz", 0) or 0))
        expected = nx * ny * nz
        if expected <= 0 or arr.size != expected:
            return
        snap_spec = None
        if isinstance(self._three_d_patch_last_spec, dict):
            snap_spec = dict(self._three_d_patch_last_spec)
        self._three_d_patch_snapshots.append(
            {
                "t_s": float(t_s),
                "stats": dict(stats),
                "vof": arr.copy(),
                "patch_spec": snap_spec,
            }
        )
        max_keep = 48
        if len(self._three_d_patch_snapshots) > max_keep:
            self._three_d_patch_snapshots = self._three_d_patch_snapshots[-max_keep:]

    def _on_cancel(self):
        self._cancel_requested = True
        self._log("Cancellation requested...")

    def _build_run_request(self):
        if SWE2DRunRequest is None:
            return None

        if self._view_adapter is not None:
            try:
                run_duration_text = self._view_adapter.run_duration_text()
            except Exception:
                run_duration_text = ""
            try:
                output_interval_text = self._view_adapter.output_interval_text()
            except Exception:
                output_interval_text = ""
            try:
                line_output_interval_text = self._view_adapter.line_output_interval_text()
            except Exception:
                line_output_interval_text = ""
            try:
                adaptive_dt_enabled = self._view_adapter.adaptive_dt_enabled()
            except Exception:
                adaptive_dt_enabled = False
            try:
                requested_dt = self._view_adapter.requested_dt()
            except Exception:
                requested_dt = 0.0
        else:
            run_duration_text = str(self.run_time_edit.text()) if hasattr(self, "run_time_edit") else ""
            output_interval_text = str(self.output_interval_edit.text()) if hasattr(self, "output_interval_edit") else ""
            line_output_interval_text = str(self.line_output_interval_edit.text()) if hasattr(self, "line_output_interval_edit") else ""
            adaptive_dt_enabled = bool(self.adaptive_cfl_dt_chk.isChecked()) if hasattr(self, "adaptive_cfl_dt_chk") else False
            requested_dt = float(self.dt_spin.value()) if hasattr(self, "dt_spin") else 0.0

        try:
            return SWE2DRunRequest.from_ui_values(
                run_duration_text=run_duration_text,
                output_interval_text=output_interval_text,
                line_output_interval_text=line_output_interval_text,
                adaptive_dt_enabled=adaptive_dt_enabled,
                requested_dt=requested_dt,
            )
        except Exception:
            return None

    def _execute_run_request(self, request):
        self._last_run_request = request
        self._on_run()

    def _ensure_mesh_for_run_preflight(self):
        if self._mesh_data is None:
            self._on_generate_mesh()

    def _has_mesh_for_run_preflight(self) -> bool:
        return self._mesh_data is not None

    def _native_backend_ready_for_run_preflight(self) -> bool:
        return bool(swe2d_available() and SWE2DBackend is not None)

    def _show_backend_unavailable_for_run_preflight(self, message: str):
        QtWidgets.QMessageBox.critical(self, "2D SWE", str(message))

    def _on_run_requested(self):
        request = self._build_run_request()
        if request is None:
            self._log("Run request aborted: unable to capture run request from UI values.")
            QtWidgets.QMessageBox.critical(self, "2D SWE", "Run request could not be captured from current UI values.")
            return
        if not self._require_run_components(
            [
                ("_run_controller", "run controller"),
                ("_run_orchestrator", "run orchestrator"),
            ],
            context_label="Run request",
        ):
            return
        if not self._run_controller.run_preflight(request):
            return
        self._run_orchestrator.run(request)

    def _on_run(self):
        if self._mesh_data is None:
            self._log("Run aborted: mesh not available after preflight.")
            return
        if SWE2DBackend is None:
            self._log("Run aborted: native backend not available after preflight.")
            return
        if not self._require_run_components(
            [
                ("_run_data_builder", "run data builder"),
                ("_run_options_builder", "run options builder"),
                ("_backend_initializer", "backend initializer"),
                ("_run_finalizer", "run finalizer"),
                ("_run_lifecycle", "run lifecycle"),
            ],
            context_label="Run",
        ):
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
            run_input = self._run_data_builder.build()
            node_x = run_input.node_x
            node_y = run_input.node_y
            node_z = run_input.node_z
            cell_nodes = run_input.cell_nodes
            face_offsets = run_input.face_offsets
            face_nodes = run_input.face_nodes
            bc_n0 = run_input.bc_n0
            bc_n1 = run_input.bc_n1
            bc_tp = run_input.bc_tp
            bc_vl = run_input.bc_vl
            side_hydrographs = run_input.side_hydrographs
            edge_hydrographs = run_input.edge_hydrographs
            edge_group_overrides = run_input.edge_group_overrides
            h0 = run_input.h0
            hu0 = run_input.hu0
            hv0 = run_input.hv0
            n_mann_cell = run_input.n_mann_cell

            run_options = self._run_options_builder.build()
            run_duration_s = run_options.run_duration_s
            dt_cfg = run_options.dt_cfg
            adaptive_cfl_dt = run_options.adaptive_cfl_dt
            dt_fixed = run_options.dt_fixed
            dt_request = run_options.dt_request
            reconstruction_mode = run_options.reconstruction_mode
            reconstruction_name = run_options.reconstruction_name
            temporal_scheme = run_options.temporal_scheme
            temporal_scheme_name = run_options.temporal_scheme_name
            godunov_mode = run_options.godunov_mode
            coupling_loop_mode = run_options.coupling_loop_mode
            drainage_solver_backend_mode = run_options.drainage_solver_backend_mode
            drainage_gpu_method_mode = run_options.drainage_gpu_method_mode
            cuda_graphs_enabled = run_options.cuda_graphs_enabled
            experimental_3d_enabled = run_options.experimental_3d_enabled
            model_options = run_options.model_options
            swe3d_env_overrides = run_options.swe3d_env_overrides
            self._three_d_patch_snapshots = []
            self._three_d_patch_last_spec = None
            rain_rate_model = run_options.rain_rate_model
            internal_flow_forcing = run_options.internal_flow_forcing
            cell_source_model = run_options.cell_source_model
            thiessen_forcing = run_options.thiessen_forcing
            pipe_network_cfg = run_options.pipe_network_cfg
            hydraulic_structures_cfg = run_options.hydraulic_structures_cfg

            # Propagate locally-built drainage/structure configs into model_options
            # so that enable_pipe_network_module and enable_hydraulic_structures flags are set correctly.
            if model_options is not None:
                if pipe_network_cfg is not None:
                    model_options.pipe_network = pipe_network_cfg
                if hydraulic_structures_cfg is not None:
                    model_options.hydraulic_structures = hydraulic_structures_cfg

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

            run_mode_name = "2D"
            if model_options is not None and SWE2DThreeDSolverModel is not None:
                if int(model_options.three_d_solver_model) == int(SWE2DThreeDSolverModel.SINGLE_PHASE_FREE_SURFACE_VOF):
                    run_mode_name = "2D + Experimental 3D patch"

            coupling_mode_label = "off"
            if model_options is not None and SWE2DThreeDCouplingMode is not None:
                try:
                    cm = int(model_options.coupling_mode)
                    if cm == int(SWE2DThreeDCouplingMode.ONE_WAY_2D_TO_3D):
                        coupling_mode_label = "one-way (2D -> 3D)"
                    elif cm == int(SWE2DThreeDCouplingMode.TWO_WAY_2D_3D):
                        coupling_mode_label = "two-way (2D <-> 3D)"
                except Exception:
                    coupling_mode_label = "off"

            self._log("Starting 2D run...")
            if run_mode_name != "2D":
                self._log(f"Run mode: {run_mode_name} (coupling={coupling_mode_label}).")
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
                return self._backend_initializer.build_and_initialize(
                    backend_cls=SWE2DBackend,
                    swe3d_env_overrides=swe3d_env_overrides,
                    dynamic_bc=dynamic_bc,
                    node_x=node_x,
                    node_y=node_y,
                    node_z=node_z,
                    cell_nodes=cell_nodes,
                    face_offsets=face_offsets,
                    face_nodes=face_nodes,
                    bc_n0=bc_n0,
                    bc_n1=bc_n1,
                    bc_tp=bc_tp,
                    bc_vl=bc_vl,
                    side_hydrographs=side_hydrographs,
                    edge_hydrographs=edge_hydrographs,
                    h0=h0,
                    hu0=hu0,
                    hv0=hv0,
                    n_mann_cell=n_mann_cell,
                    dt_fixed=dt_fixed,
                    dt_max=dt_cfg,
                    model_options=model_options,
                    reconstruction_mode=reconstruction_mode,
                    temporal_scheme=temporal_scheme,
                    godunov_mode=godunov_mode,
                )

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

            experimental_3d_runtime = bool(
                model_options is not None
                and SWE2DThreeDSolverModel is not None
                and int(model_options.three_d_solver_model) == int(SWE2DThreeDSolverModel.SINGLE_PHASE_FREE_SURFACE_VOF)
            )
            if experimental_3d_enabled and not experimental_3d_runtime:
                raise RuntimeError(
                    "Experimental 3D mode was requested but solver model options did not activate 3D runtime."
                )
            if experimental_3d_runtime and not bool(backend.supports_3d_patch_observation()):
                raise RuntimeError(
                    "Experimental 3D mode requires native 3D patch observation APIs; "
                    "current native module does not expose them."
                )

            if SWE2DThreeDPatchObserver is None:
                raise RuntimeError("SWE2DThreeDPatchObserver seam is unavailable.")
            _three_d_observer = SWE2DThreeDPatchObserver(backend=backend, runtime_enabled=experimental_3d_runtime)
            _get_3d_patch_stats = _three_d_observer.get_patch_stats
            _get_3d_patch_vof = _three_d_observer.get_patch_vof

            if experimental_3d_runtime:
                try:
                    self._apply_3d_patch_face_bc_to_backend(backend)
                except Exception as exc:
                    self._log(f"3D face BC upload warning (continuing with env defaults): {exc}")
                stats0 = _get_3d_patch_stats()
                if stats0 is not None:
                    try:
                        spec0 = self._build_patch_spec_from_stats(stats0, swe3d_env_overrides)
                        spec0_dict = self._patch_spec_to_dict(spec0)
                        if isinstance(spec0_dict, dict):
                            self._three_d_patch_last_spec = dict(spec0_dict)
                    except Exception:
                        self._three_d_patch_last_spec = None
                    self._log(
                        "3D patch initialized: "
                        f"nx={int(stats0.get('nx', 0))} ny={int(stats0.get('ny', 0))} nz={int(stats0.get('nz', 0))} "
                        f"dx={float(stats0.get('dx', 0.0)):.3f} dy={float(stats0.get('dy', 0.0)):.3f} dz={float(stats0.get('dz', 0.0)):.3f} "
                        f"cells={int(stats0.get('n_cells', 0))}"
                    )
                    try:
                        self._upload_experimental_3d_obj_geometry(
                            backend=backend,
                            patch_stats=stats0,
                            swe3d_env_overrides=swe3d_env_overrides,
                            backend_builder=_build_and_initialize_backend,
                            bc_n0=bc_n0,
                            bc_n1=bc_n1,
                            bc_tp=bc_tp,
                            bc_vl=bc_vl,
                        )
                    except Exception as exc:
                        self._log(f"3D sub-grid preprocessing failed (run continues): {exc}")
                    try:
                        self._initialize_experimental_3d_patch_state(
                            backend=backend,
                            patch_stats=stats0,
                            swe3d_env_overrides=swe3d_env_overrides,
                            bc_n0=bc_n0,
                            bc_n1=bc_n1,
                            bc_tp=bc_tp,
                            bc_vl=bc_vl,
                        )
                    except Exception as exc:
                        self._log(f"3D patch initial-state seeding failed (run continues): {exc}")
                    try:
                        self._upload_experimental_3d_interface_contract(
                            backend=backend,
                            patch_stats=stats0,
                            bc_n0=bc_n0,
                            bc_n1=bc_n1,
                            bc_tp=bc_tp,
                            coupling_mode=int(model_options.coupling_mode) if model_options is not None else 0,
                        )
                    except Exception as exc:
                        raise RuntimeError(f"3D coupling contract setup failed: {exc}")
                else:
                    raise RuntimeError(
                        "Experimental 3D mode requested, but native 3D patch stats are unavailable after initialization."
                    )

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
                raise RuntimeError("Native module does not support dynamic boundary updates. Rebuild hydra_swe2d.")

            native_bc_forcing = False
            native_rain_cn_forcing = False
            if SWE2DRunSetupConfigurator is None:
                raise RuntimeError("SWE2DRunSetupConfigurator seam is unavailable.")
            if SWE2DNativeBoundaryHydrographConfigurator is None:
                raise RuntimeError("SWE2DNativeBoundaryHydrographConfigurator seam is unavailable.")
            run_setup_configurator = SWE2DRunSetupConfigurator()
            native_bc_cfg = SWE2DNativeBoundaryHydrographConfigurator()

            if dynamic_bc and hasattr(backend, "set_boundary_hydrographs_native"):
                try:
                    progressive = True
                    if hasattr(self, "inflow_progressive_chk") and self.inflow_progressive_chk is not None:
                        progressive = bool(self.inflow_progressive_chk.isChecked())
                    node_x = self._mesh_data["node_x"]
                    node_y = self._mesh_data["node_y"]
                    native_bc_res = native_bc_cfg.configure(
                        backend=backend,
                        bc_n0=bc_n0,
                        bc_n1=bc_n1,
                        bc_tp=bc_tp,
                        side_hydrographs=side_hydrographs,
                        edge_hydrographs=edge_hydrographs,
                        node_x=node_x,
                        node_y=node_y,
                        inflow_q_bc_type=int(_BC_INFLOW_Q),
                        progressive=progressive,
                    )
                    if bool(native_bc_res.get("native_bc_forcing", False)):
                        native_bc_forcing = True
                        self._log(
                            f"Native BC hydrograph forcing configured for {int(native_bc_res.get('configured_edges', 0))} boundary edges."
                        )
                    elif bool(native_bc_res.get("skipped_progressive", False)):
                        self._log("Native BC hydrographs skipped: progressive inflow activation is enabled for flow hydrographs.")
                except Exception as exc:
                    self._log(f"Native BC hydrograph forcing unavailable: {exc}")

            if thiessen_forcing is not None and hasattr(backend, "set_rain_cn_forcing_native"):
                try:
                    native_rain_res = run_setup_configurator.configure_native_rain_cn_forcing(
                        backend=backend,
                        thiessen_forcing=thiessen_forcing,
                        mm_to_model_depth=float(self._rain_mm_to_model_depth()),
                    )
                    if bool(native_rain_res.get("configured", False)):
                        native_rain_cn_forcing = True
                        self._log(
                            "Native preprocessed rainfall-excess forcing configured for GPU timestep evaluation "
                            f"(infiltration={str(native_rain_res.get('infiltration_method', 'scs_cn'))}, "
                            f"groups={int(native_rain_res.get('groups', 0))})."
                        )
                except Exception as exc:
                    self._log(f"Native rain+CN forcing unavailable: {exc}")

            native_source_injection_mode = hasattr(backend, "set_external_sources_native")
            if native_source_injection_mode:
                try:
                    native_src_res = run_setup_configurator.configure_native_source_injection(backend=backend)
                    native_source_injection_mode = bool(native_src_res.get("native_source_injection_mode", False))
                    if bool(native_src_res.get("configured", False)):
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

            if SWE2DRuntimeSourceManager is None:
                raise RuntimeError("SWE2DRuntimeSourceManager seam is unavailable.")
            runtime_source_manager = SWE2DRuntimeSourceManager(
                rain_rate_model=rain_rate_model,
                thiessen_forcing=thiessen_forcing,
                native_rain_cn_forcing=native_rain_cn_forcing,
                internal_flow_forcing=internal_flow_forcing,
                rain_stats_acc=rain_stats_acc,
                area_model=area_model,
                edge_len_bc=edge_len_bc,
                edge_group_labels=edge_group_labels,
                inflow_q_bc_type=int(_BC_INFLOW_Q),
                rain_rate_si_to_model_callback=self._rain_rate_si_to_model,
                internal_flow_source_cms_at_time_callback=self._internal_flow_source_cms_at_time,
                flow_si_to_model_callback=self._flow_si_to_model,
            )
            source_budget_model = runtime_source_manager.source_budget_model
            boundary_flux_budget_model = runtime_source_manager.boundary_flux_budget_model
            _accumulate_boundary_flux_volume_model = runtime_source_manager.accumulate_boundary_flux_volume_model
            _accumulate_source_volume_model = runtime_source_manager.accumulate_source_volume_model
            _rain_source_for_window = runtime_source_manager.rain_source_for_window
            _cell_source_model_at_time = runtime_source_manager.cell_source_model_at_time

            stage_coupled_imex_requested = bool(
                hasattr(self, "source_stage_coupled_imex_rk2_chk")
                and self.source_stage_coupled_imex_rk2_chk.isChecked()
            )
            stage_coupled_imex_enabled = False
            stage_res = run_setup_configurator.resolve_stage_coupled_imex(
                requested=stage_coupled_imex_requested,
                coupling_controller=coupling_controller,
                temporal_scheme=temporal_scheme,
                required_temporal_scheme=TemporalScheme.SSP_RK2,
                native_source_injection_mode=native_source_injection_mode,
            )
            stage_coupled_imex_enabled = bool(stage_res.get("enabled", False))
            stage_reasons = list(stage_res.get("reasons", []))
            if stage_coupled_imex_requested:
                if stage_reasons:
                    self._log(
                        "Stage-coupled IMEX-RK2 requested but disabled: "
                        + "; ".join(stage_reasons)
                    )
                else:
                    self._log("Stage-coupled IMEX-RK2 enabled for external coupling sources.")

            if SWE2DRuntimeStepExecutor is None:
                raise RuntimeError("SWE2DRuntimeStepExecutor seam is unavailable.")
            if SWE2DRuntimeReporter is None:
                raise RuntimeError("SWE2DRuntimeReporter seam is unavailable.")
            runtime_step_executor = SWE2DRuntimeStepExecutor()
            runtime_reporter = SWE2DRuntimeReporter()

            loop_result = _execute_run_timestep_loop_runtime_logic(
                wb=self,
                backend=backend,
                runtime_step_executor=runtime_step_executor,
                runtime_reporter=runtime_reporter,
                run_duration_s=run_duration_s,
                t_accum=t_accum,
                i=i,
                last_diag=last_diag,
                last_valid_cmax=last_valid_cmax,
                last_valid_wse_res=last_valid_wse_res,
                dt_cfg=dt_cfg,
                dt_request=dt_request,
                stage_coupled_imex_enabled=stage_coupled_imex_enabled,
                coupling_controller=coupling_controller,
                dynamic_bc=dynamic_bc,
                native_bc_forcing=native_bc_forcing,
                bc_n0=bc_n0,
                bc_n1=bc_n1,
                bc_tp=bc_tp,
                bc_vl=bc_vl,
                side_hydrographs=side_hydrographs,
                edge_hydrographs=edge_hydrographs,
                rain_source_for_window_callback=_rain_source_for_window,
                cell_source_model_at_time_callback=_cell_source_model_at_time,
                accumulate_source_volume_model_callback=_accumulate_source_volume_model,
                native_source_injection_mode=native_source_injection_mode,
                accumulate_boundary_flux_volume_model_callback=_accumulate_boundary_flux_volume_model,
                sample_map=sample_map,
                cell_min_z=cell_min_z,
                experimental_3d_runtime=experimental_3d_runtime,
                timing_totals_ms=timing_totals_ms,
                timing_samples=timing_samples,
                next_snap_t=_next_snap_t,
                next_line_snap_t=_next_line_snap_t,
                next_coupling_snap_t=_next_coupling_snap_t,
                output_interval_s=output_interval_s,
                line_output_interval_s=line_output_interval_s,
                process_events_interval_s=_PROCESS_EVENTS_INTERVAL_S,
                last_process_events_wall=_last_process_events_wall,
                process_events_callback=QtWidgets.QApplication.processEvents,
                get_3d_patch_stats_callback=_get_3d_patch_stats,
                get_3d_patch_vof_callback=_get_3d_patch_vof,
            )
            t_accum = float(loop_result.get("t_accum", t_accum))
            i = int(loop_result.get("i", i))
            last_diag = loop_result.get("last_diag", last_diag)
            last_valid_cmax = float(loop_result.get("last_valid_cmax", last_valid_cmax))
            last_valid_wse_res = float(loop_result.get("last_valid_wse_res", last_valid_wse_res))
            _next_snap_t = float(loop_result.get("next_snap_t", _next_snap_t))
            _next_line_snap_t = float(loop_result.get("next_line_snap_t", _next_line_snap_t))
            _next_coupling_snap_t = float(loop_result.get("next_coupling_snap_t", _next_coupling_snap_t))
            _last_process_events_wall = float(loop_result.get("last_process_events_wall", _last_process_events_wall))
            timing_samples = int(loop_result.get("timing_samples", timing_samples))
            h, hu, hv = backend.get_state()
            if experimental_3d_runtime and not self._three_d_patch_snapshots:
                s3 = _get_3d_patch_stats()
                v3 = _get_3d_patch_vof()
                if s3 is not None and v3 is not None:
                    self._append_3d_patch_snapshot(t_accum, s3, v3)
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

            self._run_finalizer.finalize_and_persist(
                h=h,
                hu=hu,
                hv=hv,
                n_area=n_area,
                area_model=area_model,
                storage_start_model=storage_start_model,
                source_budget_model=source_budget_model,
                run_duration_s=run_duration_s,
                boundary_flux_budget_model=boundary_flux_budget_model,
                run_id=run_id,
                output_interval_s=output_interval_s,
                line_output_interval_s=line_output_interval_s,
                run_perf_start=run_perf_start,
                run_wallclock_start=run_wallclock_start,
                run_log_start_idx=run_log_start_idx,
                thiessen_forcing=thiessen_forcing,
                rain_stats_acc=rain_stats_acc,
            )
        except Exception as exc:
            self._run_lifecycle.handle_run_failure(
                exc,
                lambda msg: QtWidgets.QMessageBox.critical(self, "2D SWE", msg),
            )
        finally:
            self._run_lifecycle.finalize_cleanup(backend)

    def _refresh_plot(self):
        if not self._have_mpl or self._mesh_data is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        mode = str(self.view_mode_combo.currentData() or "mesh")
        self._render_workbench_mesh_view(ax, mode)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        self._canvas.draw_idle()
        for dlg in list(self._mesh_view_detached_dialogs):
            try:
                if dlg is not None:
                    dlg.refresh_view()
            except Exception:
                pass


class SWE2DWorkbenchDesignerDialog(SWE2DWorkbenchDialog):
    """Parallel workbench shell whose full window frame is owned by a .ui file."""

    _DESIGNER_TAB_PAGES = (
        ("mesh_tab", "Mesh"),
        ("map_tab", "Map"),
        ("topology_tab", "Topology"),
        ("boundary_tab", "Boundary"),
        ("model_tab", "Model"),
        ("run_tab", "Run"),
    )

    def __init__(self, parent=None, iface=None):
        super().__init__(parent, iface=iface)
        self.setWindowTitle("2D SWE Workbench (Designer UI)")

    def _build_designer_workbench_shell(self) -> QtWidgets.QWidget:
        ui_path = self._forms_file_path("swe2d_workbench_designer.ui")
        shell = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                shell = _qgis_uic.loadUi(ui_path)
            except Exception:
                shell = None
        if shell is None:
            shell = self._build_designer_workbench_shell_fallback()
        return shell

    def _build_designer_workbench_shell_fallback(self) -> QtWidgets.QWidget:
        shell = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(shell)
        root_layout.setContentsMargins(0, 0, 0, 0)

        header_lbl = QtWidgets.QLabel("Interactive 2D SWE workflow")
        header_lbl.setObjectName("header_lbl")
        header_lbl.setWordWrap(True)
        root_layout.addWidget(header_lbl)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setObjectName("main_splitter")

        left_tabs = QtWidgets.QTabWidget()
        left_tabs.setObjectName("left_tabs")
        left_tabs.setDocumentMode(True)
        for page_name, title in self._DESIGNER_TAB_PAGES:
            page = QtWidgets.QWidget()
            page.setObjectName(page_name)
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            left_tabs.addTab(page, title)
        main_splitter.addWidget(left_tabs)

        right_pane_host = QtWidgets.QWidget()
        right_pane_host.setObjectName("right_pane_host")
        right_pane_host_layout = QtWidgets.QVBoxLayout(right_pane_host)
        right_pane_host_layout.setContentsMargins(0, 0, 0, 0)
        main_splitter.addWidget(right_pane_host)

        root_layout.addWidget(main_splitter, stretch=1)

        bottom_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bottom_buttons.setObjectName("bottom_buttons")
        root_layout.addWidget(bottom_buttons)
        return shell

    def _designer_host_widget(self, shell: QtWidgets.QWidget, object_name: str) -> QtWidgets.QWidget:
        host = shell.findChild(QtWidgets.QWidget, object_name)
        if host is None:
            raise RuntimeError(f"Designer workbench shell is missing {object_name}")
        return host

    def _mount_widget_in_host(self, host: QtWidgets.QWidget, widget: QtWidgets.QWidget) -> None:
        layout = host.layout()
        if not isinstance(layout, QtWidgets.QVBoxLayout):
            layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.deleteLater()
        layout.addWidget(widget, stretch=1)

    def _ensure_designer_tab_scroll(self, tab_page: QtWidgets.QWidget, scroll_object_name: str) -> None:
        scroll = tab_page.findChild(QtWidgets.QScrollArea, scroll_object_name)
        if scroll is not None:
            return

        page_layout = tab_page.layout()
        if not isinstance(page_layout, QtWidgets.QVBoxLayout):
            page_layout = QtWidgets.QVBoxLayout(tab_page)
        margins = page_layout.contentsMargins()
        spacing = page_layout.spacing()

        content = QtWidgets.QWidget(tab_page)
        content.setObjectName(f"{tab_page.objectName()}_scroll_content")
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(margins)
        content_layout.setSpacing(spacing)

        while page_layout.count():
            item = page_layout.takeAt(0)
            child_widget = item.widget()
            child_layout = item.layout()
            child_spacer = item.spacerItem()
            if child_widget is not None:
                content_layout.addWidget(child_widget)
            elif child_layout is not None:
                content_layout.addLayout(child_layout)
            elif child_spacer is not None:
                content_layout.addItem(child_spacer)

        scroll = QtWidgets.QScrollArea(tab_page)
        scroll.setObjectName(scroll_object_name)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        page_layout.addWidget(scroll, stretch=1)

    def _populate_designer_left_tabs(self, shell: QtWidgets.QWidget) -> None:
        self._left_tabs = shell.findChild(QtWidgets.QTabWidget, "left_tabs")
        if self._left_tabs is None:
            raise RuntimeError("Designer workbench shell is missing left_tabs")

        for tab_name in ("mesh_tab", "map_tab", "topology_tab", "boundary_tab", "model_tab", "run_tab"):
            self._ensure_designer_tab_scroll(
                self._designer_host_widget(shell, tab_name),
                f"{tab_name}_scroll",
            )

        mesh_tab_page = self._designer_host_widget(shell, "mesh_tab")
        self._bind_mesh_tab_controls(mesh_tab_page)

        map_tab_page = self._designer_host_widget(shell, "map_tab")
        map_data_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_data_layout")
        map_actions_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_actions_layout")
        map_results_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_results_layout")
        map_tools_layout = map_tab_page.findChild(QtWidgets.QGridLayout, "map_tools_layout")
        if (
            map_data_layout is None
            or map_actions_layout is None
            or map_results_layout is None
            or map_tools_layout is None
        ):
            raise RuntimeError("Designer workbench shell map tab is missing one or more expected layouts")
        self._bind_map_tab_data_controls(map_tab_page, map_data_layout)
        self._bind_map_tab_action_controls(map_tab_page, map_actions_layout)
        self._bind_map_tab_results_controls(map_tab_page, map_results_layout)
        self._bind_map_tab_tools_controls(map_tab_page, map_tools_layout)

        topology_tab_page = self._designer_host_widget(shell, "topology_tab")
        topo_layout = topology_tab_page.findChild(QtWidgets.QGridLayout, "topo_layout")
        if topo_layout is None:
            raise RuntimeError("Designer workbench shell topology tab is missing topo_layout")
        self._bind_topology_tab_static_controls(topology_tab_page, topo_layout)
        self._bind_topology_tab_dynamic_controls(topology_tab_page, topo_layout)

        boundary_tab_page = self._designer_host_widget(shell, "boundary_tab")
        bc_grid = boundary_tab_page.findChild(QtWidgets.QGridLayout, "bc_grid")
        if bc_grid is None:
            raise RuntimeError("Designer workbench shell boundary tab is missing bc_grid")
        self._populate_boundary_tab_controls(bc_grid)

        model_tab_page = self._designer_host_widget(shell, "model_tab")
        param_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_param_form")
        if param_form is None:
            raise RuntimeError("Designer workbench shell model tab is missing model_param_form")
        self._bind_model_tab_core_controls(model_tab_page, param_form)
        self._bind_model_tab_hydrology_controls(model_tab_page, param_form)
        self._bind_model_tab_solver_controls(model_tab_page, param_form)
        self._bind_model_tab_3d_patch_controls(model_tab_page, param_form)
        self._bind_model_tab_3d_subgrid_drainage_controls(model_tab_page, param_form)

        run_tab_page = self._designer_host_widget(shell, "run_tab")
        self._bind_run_tab_controls(run_tab_page)

        self._left_tabs.setMinimumWidth(0)
        for _cb in self._left_tabs.findChildren(QtWidgets.QComboBox):
            _cb.setMinimumContentsLength(0)
            _cb.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        for _btn in self._left_tabs.findChildren(QtWidgets.QPushButton):
            _btn.setMinimumWidth(0)
        for _sp in self._left_tabs.findChildren(
            (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)  # type: ignore[arg-type]
        ):
            _sp.setMinimumWidth(0)
        self._make_left_controls_compact(self._left_tabs)
        self._register_detachable_tab_widget(self._left_tabs)

    def _build_ui(self):
        root = self.layout()
        if not isinstance(root, QtWidgets.QVBoxLayout):
            root = QtWidgets.QVBoxLayout(self)
        while root.count():
            item = root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        shell = self._build_designer_workbench_shell()
        root.addWidget(shell, stretch=1)

        header = shell.findChild(QtWidgets.QLabel, "header_lbl")
        if header is not None:
            header.setText(
                "Interactive 2D SWE workflow: generate mesh, assign side BCs, set model parameters, "
                "run, and visualize results."
            )
            header.setWordWrap(True)

        self._populate_designer_left_tabs(shell)
        self._bind_right_pane_controls(self._designer_host_widget(shell, "right_pane_host"))

        split = shell.findChild(QtWidgets.QSplitter, "main_splitter")
        if split is not None:
            split.setSizes([420, 740])

        buttons = shell.findChild(QtWidgets.QDialogButtonBox, "bottom_buttons")
        if buttons is not None:
            try:
                buttons.rejected.disconnect(self.reject)
            except Exception:
                pass
            try:
                buttons.accepted.disconnect(self.accept)
            except Exception:
                pass
            buttons.rejected.connect(self.reject)
            buttons.accepted.connect(self.accept)

        self._refresh_layer_combos()


class SWE2DWorkbenchStudioDialog(SWE2DWorkbenchDialog):
    """Dock-inspired workspace layout with persistent side inspector."""

    def __init__(self, parent=None, iface=None):
        # Initialize handles before super(); base __init__ calls self._build_ui().
        self._studio_main_window = None
        self._studio_status_label = None
        self._studio_view_mode_combo = None
        self._studio_theme_combo = None
        self._studio_left_dock = None
        self._studio_inspector_dock = None
        self._studio_feature_flags = {
            "rainfall": True,
            "drainage": True,
            "structures": True,
        }
        super().__init__(parent, iface=iface)
        self.setWindowTitle("2D SWE Workbench (Studio)")

        # Last-resort recovery: if anything failed during _build_ui, build a
        # minimal visible panel instead of presenting a blank surface.
        if self._studio_main_window is None:
            root = self.layout()
            if not isinstance(root, QtWidgets.QVBoxLayout):
                root = QtWidgets.QVBoxLayout(self)
            while root.count():
                item = root.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            fallback = QtWidgets.QPlainTextEdit(self)
            fallback.setReadOnly(True)
            fallback.setPlainText(
                "Studio UI initialization did not complete.\n"
                "Please close and reopen this window. If this persists, share the QGIS log."
            )
            root.addWidget(fallback, 1)

    def _studio_project_scope_key(self) -> str:
        project_key = "default"
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                proj = QgsProject.instance()
                file_name = str(proj.fileName() or "").strip()
                if file_name:
                    project_key = file_name
                else:
                    project_key = str(proj.homePath() or "").strip() or project_key
            except Exception:
                pass
        safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in project_key)
        if not safe:
            safe = "default"
        return safe

    def _studio_layout_settings_keys(self) -> Tuple[str, str]:
        scope = self._studio_project_scope_key()
        base = f"Backwater2DWorkbenchStudio/v2/{scope}"
        return f"{base}/layout_state", f"{base}/layout_geometry"

    def _restore_studio_layout_state(self) -> None:
        if self._studio_main_window is None:
            return
        state_key, _geom_key = self._studio_layout_settings_keys()
        settings = QtCore.QSettings()
        try:
            state_raw = settings.value(state_key, "")
        except Exception:
            state_raw = ""

        restored = False
        if state_raw:
            try:
                state_bytes = QtCore.QByteArray.fromBase64(str(state_raw).encode("ascii"))
                restored = bool(self._studio_main_window.restoreState(state_bytes))
            except Exception:
                pass

        # Safety: always keep the core panes visible so Studio cannot reopen blank.
        try:
            center = self._studio_main_window.centralWidget()
            if center is not None:
                center.show()
        except Exception:
            pass
        try:
            if self._studio_left_dock is not None and not self._studio_left_dock.isVisible():
                self._studio_left_dock.show()
        except Exception:
            pass
        try:
            if self._studio_inspector_dock is not None and not self._studio_inspector_dock.isVisible():
                self._studio_inspector_dock.show()
        except Exception:
            pass

        if state_raw and not restored:
            self._studio_main_window.resize(1200, 760)

    def _save_studio_layout_state(self) -> None:
        if self._studio_main_window is None:
            return
        state_key, _geom_key = self._studio_layout_settings_keys()
        settings = QtCore.QSettings()
        try:
            state_b64 = bytes(self._studio_main_window.saveState().toBase64()).decode("ascii")
            settings.setValue(state_key, state_b64)
            settings.sync()
        except Exception:
            pass

    def _studio_mount_widget(self, host: QtWidgets.QWidget, widget: QtWidgets.QWidget) -> None:
        layout = host.layout()
        if not isinstance(layout, QtWidgets.QVBoxLayout):
            layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.deleteLater()
        layout.addWidget(widget, stretch=1)

    def _studio_select_tab(self, name: str) -> None:
        if not hasattr(self, "_left_tabs") or self._left_tabs is None:
            return
        target = str(name or "").strip().lower()
        for idx in range(self._left_tabs.count()):
            if str(self._left_tabs.tabText(idx) or "").strip().lower() == target:
                self._left_tabs.setCurrentIndex(idx)
                return

    def _studio_set_feature_enabled(self, feature: str, enabled: bool) -> None:
        key = str(feature or "").strip().lower()
        if key not in self._studio_feature_flags:
            return
        self._studio_feature_flags[key] = bool(enabled)
        self._studio_apply_feature_filters()

    def _studio_feature_keywords(self) -> Dict[str, Tuple[str, ...]]:
        return {
            "rainfall": ("rain", "gauge", "hyet", "storm", "runoff", "precip"),
            "drainage": ("drain", "node", "link", "inlet", "outfall", "pipe", "network"),
            "structures": ("structure", "culvert", "weir", "orifice", "gate", "spillway"),
        }

    def _studio_widget_text_blob(self, widget: QtWidgets.QWidget) -> str:
        parts = [str(widget.objectName() or "")]
        try:
            if hasattr(widget, "text") and callable(widget.text):
                parts.append(str(widget.text() or ""))
        except Exception:
            pass
        try:
            if hasattr(widget, "title") and callable(widget.title):
                parts.append(str(widget.title() or ""))
        except Exception:
            pass
        try:
            parts.append(str(widget.toolTip() or ""))
        except Exception:
            pass
        return " ".join(parts).lower()

    def _studio_apply_feature_filters(self) -> None:
        if not hasattr(self, "_left_tabs") or self._left_tabs is None:
            return
        keywords = self._studio_feature_keywords()
        for widget in self._left_tabs.findChildren(QtWidgets.QWidget):
            if widget is self._left_tabs:
                continue
            blob = self._studio_widget_text_blob(widget)
            matched = []
            for feature, words in keywords.items():
                if any(word in blob for word in words):
                    matched.append(feature)
            if not matched:
                continue
            visible = all(self._studio_feature_flags.get(feature, True) for feature in matched)
            try:
                widget.setVisible(visible)
            except Exception:
                pass

    def _studio_sync_view_mode(self, idx: int) -> None:
        if not hasattr(self, "view_mode_combo") or self.view_mode_combo is None:
            return
        if idx < 0:
            return
        try:
            self.view_mode_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def _studio_apply_visual_profile(self, profile: str) -> None:
        profile_key = str(profile or "").strip().lower()
        if self._studio_main_window is None:
            return
        if profile_key == "diagnostics":
            self._studio_main_window.setStyleSheet(
                "QMainWindow { background: #1f232a; }"
                "QDockWidget::title { background: #2d3640; color: #f2f4f8; padding: 4px; }"
                "QToolBar { background: #2b3139; border-bottom: 1px solid #3a424c; }"
                "QStatusBar { background: #2b3139; color: #e7edf5; }"
            )
        elif profile_key == "presentation":
            self._studio_main_window.setStyleSheet(
                "QMainWindow { background: #f2f5f8; }"
                "QDockWidget::title { background: #d9e2ec; color: #243b53; padding: 4px; }"
                "QToolBar { background: #e4ebf2; border-bottom: 1px solid #c9d4df; }"
                "QStatusBar { background: #e4ebf2; color: #243b53; }"
            )
        else:
            self._studio_main_window.setStyleSheet("")

    def _studio_update_status(self) -> None:
        if self._studio_status_label is None:
            return
        project_name = "(no project)"
        project_home = ""
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                proj = QgsProject.instance()
                project_name = str(proj.baseName() or "").strip() or "(unnamed project)"
                project_home = str(proj.homePath() or "").strip()
            except Exception:
                pass
        mode_txt = str(getattr(self, "_swe2d_workbench_host_mode", "window") or "window")
        detail = f"Project: {project_name}"
        if project_home:
            detail += f" | Home: {project_home}"
        detail += f" | Host mode: {mode_txt}"
        self._studio_status_label.setText(detail)

    def _build_ui(self):
        root = self.layout()
        if not isinstance(root, QtWidgets.QVBoxLayout):
            root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        while root.count():
            item = root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._studio_main_window = QtWidgets.QMainWindow(self)
        self._studio_main_window.setWindowFlags(QtCore.Qt.Widget)
        self._studio_main_window.setObjectName("SWE2DStudioMainWindow")
        self._studio_main_window.setDockOptions(
            QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.AnimatedDocks
        )
        root.addWidget(self._studio_main_window, stretch=1)

        toolbar = QtWidgets.QToolBar("CFD Workspace", self._studio_main_window)
        toolbar.setObjectName("SWE2DStudioToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._studio_main_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        act_mesh = toolbar.addAction("Mesh")
        act_model = toolbar.addAction("Model")
        act_run = toolbar.addAction("Run")
        act_map = toolbar.addAction("Map")
        toolbar.addSeparator()
        act_refresh = toolbar.addAction("Refresh Layers")
        act_snapshot = toolbar.addAction("Take Snapshot")
        toolbar.addSeparator()
        act_close = toolbar.addAction("Close")

        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" View: "))
        self._studio_view_mode_combo = QtWidgets.QComboBox()
        self._studio_view_mode_combo.addItems(["Mesh", "Depth", "Velocity magnitude", "Runtime Log"])
        toolbar.addWidget(self._studio_view_mode_combo)

        toolbar.addWidget(QtWidgets.QLabel(" Theme: "))
        self._studio_theme_combo = QtWidgets.QComboBox()
        self._studio_theme_combo.addItems(["Default", "Diagnostics", "Presentation"])
        toolbar.addWidget(self._studio_theme_combo)

        center_host = QtWidgets.QWidget()
        self._studio_mount_widget(center_host, self._build_right_pane())
        self._studio_main_window.setCentralWidget(center_host)

        self._studio_left_dock = QtWidgets.QDockWidget("Model Setup", self._studio_main_window)
        self._studio_left_dock.setObjectName("SWE2DStudioSetupDock")
        self._studio_left_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        left_host = QtWidgets.QWidget()
        self._compose_left_pane(left_host)
        self._studio_left_dock.setWidget(left_host)
        self._studio_main_window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._studio_left_dock)

        self._studio_inspector_dock = QtWidgets.QDockWidget("CFD Inspector", self._studio_main_window)
        self._studio_inspector_dock.setObjectName("SWE2DStudioInspectorDock")
        self._studio_inspector_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )

        inspector_tabs = QtWidgets.QTabWidget()
        inspector_tabs.setDocumentMode(True)

        tree_page = QtWidgets.QWidget()
        tree_layout = QtWidgets.QVBoxLayout(tree_page)
        tree_layout.setContentsMargins(6, 6, 6, 6)
        workspace_tree = QtWidgets.QTreeWidget()
        workspace_tree.setHeaderLabels(["Workspace Area", "Purpose"])
        root_item = QtWidgets.QTreeWidgetItem(["SWE2D CFD Studio", "QGIS-integrated workflow shell"])
        root_item.addChild(QtWidgets.QTreeWidgetItem(["Setup Dock", "Mesh/Boundary/Model tabs"]))
        root_item.addChild(QtWidgets.QTreeWidgetItem(["Central Workspace", "Runtime view and logs"]))
        root_item.addChild(QtWidgets.QTreeWidgetItem(["Inspector Dock", "QA checks and quick actions"]))
        workspace_tree.addTopLevelItem(root_item)
        workspace_tree.expandAll()
        tree_layout.addWidget(workspace_tree)
        inspector_tabs.addTab(tree_page, "Workspace")

        quick_page = QtWidgets.QWidget()
        quick_layout = QtWidgets.QVBoxLayout(quick_page)
        quick_layout.setContentsMargins(6, 6, 6, 6)
        tools_box = QtWidgets.QToolBox()

        nav_page = QtWidgets.QWidget()
        nav_layout = QtWidgets.QVBoxLayout(nav_page)
        cmd_mesh = QtWidgets.QCommandLinkButton("Open Mesh Setup", "Jump to grid generation and controls")
        cmd_model = QtWidgets.QCommandLinkButton("Open Model Setup", "Jump to solver and roughness settings")
        cmd_run = QtWidgets.QCommandLinkButton("Open Run Tab", "Jump to runtime and output controls")
        nav_layout.addWidget(cmd_mesh)
        nav_layout.addWidget(cmd_model)
        nav_layout.addWidget(cmd_run)
        nav_layout.addStretch(1)
        tools_box.addItem(nav_page, "Navigation")

        qa_page = QtWidgets.QWidget()
        qa_layout = QtWidgets.QVBoxLayout(qa_page)
        qa_hint = QtWidgets.QPlainTextEdit()
        qa_hint.setReadOnly(True)
        qa_hint.setPlainText(
            "CFD pre-run checks:\n"
            "1. Confirm mesh exists and BC sides are configured.\n"
            "2. Verify timestep mode and CFL consistency.\n"
            "3. Confirm output intervals and runtime duration.\n"
            "4. Enable 3D export toggles only when needed."
        )
        qa_layout.addWidget(qa_hint)
        qa_layout.addStretch(1)
        tools_box.addItem(qa_page, "Pre-run QA")

        quick_layout.addWidget(tools_box)
        inspector_tabs.addTab(quick_page, "Operations")

        self._studio_inspector_dock.setWidget(inspector_tabs)
        self._studio_main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._studio_inspector_dock)

        footer = QtWidgets.QStatusBar(self._studio_main_window)
        self._studio_main_window.setStatusBar(footer)
        self._studio_status_label = QtWidgets.QLabel("")
        footer.addPermanentWidget(self._studio_status_label, 1)
        self._studio_update_status()

        if hasattr(self, "view_mode_combo") and self.view_mode_combo is not None:
            try:
                self._studio_view_mode_combo.setCurrentIndex(max(0, int(self.view_mode_combo.currentIndex())))
            except Exception:
                pass

        act_mesh.triggered.connect(lambda: self._studio_select_tab("mesh"))
        act_model.triggered.connect(lambda: self._studio_select_tab("model"))
        act_run.triggered.connect(lambda: self._studio_select_tab("run"))
        act_map.triggered.connect(lambda: self._studio_select_tab("map"))
        act_refresh.triggered.connect(self._refresh_layer_combos)
        act_snapshot.triggered.connect(lambda: self.snapshot_btn.click() if hasattr(self, "snapshot_btn") and self.snapshot_btn is not None else None)
        act_close.triggered.connect(self.reject)

        cmd_mesh.clicked.connect(lambda: self._studio_select_tab("mesh"))
        cmd_model.clicked.connect(lambda: self._studio_select_tab("model"))
        cmd_run.clicked.connect(lambda: self._studio_select_tab("run"))

        self._studio_view_mode_combo.currentIndexChanged.connect(self._studio_sync_view_mode)
        self._studio_theme_combo.currentTextChanged.connect(self._studio_apply_visual_profile)

        self._restore_studio_layout_state()
        self._studio_apply_visual_profile("Default")
        self._studio_apply_feature_filters()
        self._refresh_layer_combos()

    def closeEvent(self, event):  # type: ignore[override]
        self._save_studio_layout_state()
        super().closeEvent(event)


class SWE2DWorkbenchScenarioDialog(SWE2DWorkbenchDialog):
    """Scenario-first shell with profile presets for rapid what-if runs."""

    _SCENARIO_PRESETS = {
        "Balanced": {
            "cfl": 0.45,
            "dt": 0.05,
            "n_mann": 0.020,
            "adaptive": False,
            "rain": 0.0,
            "output_interval": "00:30",
            "line_output_interval": "00:05",
        },
        "Stable": {
            "cfl": 0.30,
            "dt": 0.03,
            "n_mann": 0.025,
            "adaptive": False,
            "rain": 5.0,
            "output_interval": "00:15",
            "line_output_interval": "00:05",
        },
        "Fast": {
            "cfl": 0.75,
            "dt": 0.10,
            "n_mann": 0.018,
            "adaptive": True,
            "rain": 0.0,
            "output_interval": "01:00",
            "line_output_interval": "00:15",
        },
    }

    def __init__(self, parent=None, iface=None):
        super().__init__(parent, iface=iface)
        self.setWindowTitle("2D SWE Workbench (Scenario-first)")
        self._scenario_profile_combo = None

    def _apply_scenario_preset(self, preset_name: str) -> None:
        preset = self._SCENARIO_PRESETS.get(str(preset_name), None)
        if not isinstance(preset, dict):
            return

        if hasattr(self, "cfl_spin") and self.cfl_spin is not None:
            self.cfl_spin.setValue(float(preset["cfl"]))
        if hasattr(self, "dt_spin") and self.dt_spin is not None:
            self.dt_spin.setValue(float(preset["dt"]))
        if hasattr(self, "n_mann_spin") and self.n_mann_spin is not None:
            self.n_mann_spin.setValue(float(preset["n_mann"]))
        if hasattr(self, "adaptive_cfl_dt_chk") and self.adaptive_cfl_dt_chk is not None:
            self.adaptive_cfl_dt_chk.setChecked(bool(preset["adaptive"]))
        if hasattr(self, "rain_rate_spin") and self.rain_rate_spin is not None:
            self.rain_rate_spin.setValue(float(preset["rain"]))
        if hasattr(self, "output_interval_edit") and self.output_interval_edit is not None:
            self.output_interval_edit.setText(str(preset["output_interval"]))
        if hasattr(self, "line_output_interval_edit") and self.line_output_interval_edit is not None:
            self.line_output_interval_edit.setText(str(preset["line_output_interval"]))
        self._log(f"Scenario preset applied: {preset_name}")

    def _build_ui(self):
        root = self.layout()
        if not isinstance(root, QtWidgets.QVBoxLayout):
            root = QtWidgets.QVBoxLayout(self)
        while root.count():
            item = root.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        header = QtWidgets.QLabel(
            "Scenario-first mode: choose a run profile first, then tune details in tabs."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        scenario_group = QtWidgets.QGroupBox("Scenario Profiles")
        scenario_layout = QtWidgets.QHBoxLayout(scenario_group)
        scenario_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self._scenario_profile_combo = QtWidgets.QComboBox()
        self._scenario_profile_combo.addItems(["Balanced", "Stable", "Fast"])
        scenario_layout.addWidget(self._scenario_profile_combo)
        apply_btn = QtWidgets.QPushButton("Apply Preset")
        scenario_layout.addWidget(apply_btn)
        quick_balanced_btn = QtWidgets.QPushButton("Balanced")
        quick_stable_btn = QtWidgets.QPushButton("Stable")
        quick_fast_btn = QtWidgets.QPushButton("Fast")
        scenario_layout.addWidget(quick_balanced_btn)
        scenario_layout.addWidget(quick_stable_btn)
        scenario_layout.addWidget(quick_fast_btn)
        scenario_layout.addStretch(1)
        root.addWidget(scenario_group)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left_host = QtWidgets.QWidget()
        right_host = QtWidgets.QWidget()
        split.addWidget(left_host)
        split.addWidget(right_host)
        split.setSizes([430, 740])
        root.addWidget(split, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        root.addWidget(buttons)

        self._compose_left_pane(left_host)
        right_layout = QtWidgets.QVBoxLayout(right_host)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._build_right_pane(), stretch=1)

        apply_btn.clicked.connect(lambda: self._apply_scenario_preset(str(self._scenario_profile_combo.currentText())))
        quick_balanced_btn.clicked.connect(lambda: self._apply_scenario_preset("Balanced"))
        quick_stable_btn.clicked.connect(lambda: self._apply_scenario_preset("Stable"))
        quick_fast_btn.clicked.connect(lambda: self._apply_scenario_preset("Fast"))

        try:
            buttons.rejected.disconnect(self.reject)
        except Exception:
            pass
        try:
            buttons.accepted.disconnect(self.accept)
        except Exception:
            pass
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        self._refresh_layer_combos()


def _normalize_workbench_host_mode(host_mode: object) -> str:
    mode_txt = str(host_mode or "window").strip().lower()
    return "dock" if mode_txt in {"dock", "docked", "panel"} else "window"


def _resolve_workbench_iface(parent, iface):
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
    return iface


def _close_dialog_windows(window_store: List[QtWidgets.QDialog]) -> None:
    while window_store:
        dlg = window_store.pop()
        try:
            dlg.close()
        except Exception:
            pass


def _close_workbench_windows() -> None:
    # Close detached dialog windows when switching host mode.
    _close_dialog_windows(_SWE2D_WORKBENCH_WINDOWS)


def _close_workbench_designer_windows() -> None:
    _close_dialog_windows(_SWE2D_WORKBENCH_DESIGNER_WINDOWS)


def _close_workbench_studio_windows() -> None:
    _close_dialog_windows(_SWE2D_WORKBENCH_STUDIO_WINDOWS)


def _close_workbench_scenario_windows() -> None:
    _close_dialog_windows(_SWE2D_WORKBENCH_SCENARIO_WINDOWS)


def _remove_workbench_dock_instance(dock, iface_obj):
    if dock is None:
        return None
    try:
        widget = dock.widget()
        if widget is not None:
            try:
                widget.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        if iface_obj is not None and hasattr(iface_obj, "removeDockWidget"):
            iface_obj.removeDockWidget(dock)
    except Exception:
        pass
    try:
        dock.deleteLater()
    except Exception:
        pass
    return None


def _remove_workbench_dock(iface_obj) -> None:
    global _SWE2D_WORKBENCH_DOCK
    _SWE2D_WORKBENCH_DOCK = _remove_workbench_dock_instance(_SWE2D_WORKBENCH_DOCK, iface_obj)


def _remove_workbench_designer_dock(iface_obj) -> None:
    global _SWE2D_WORKBENCH_DESIGNER_DOCK
    _SWE2D_WORKBENCH_DESIGNER_DOCK = _remove_workbench_dock_instance(
        _SWE2D_WORKBENCH_DESIGNER_DOCK, iface_obj
    )


def _remove_workbench_studio_dock(iface_obj) -> None:
    global _SWE2D_WORKBENCH_STUDIO_DOCK, _SWE2D_STUDIO_COMPONENT_DOCKS, _SWE2D_STUDIO_HOST_DIALOG
    seen = set()

    for dock in [_SWE2D_WORKBENCH_STUDIO_DOCK] + list(_SWE2D_STUDIO_COMPONENT_DOCKS.values()):
        if dock is None:
            continue
        key = id(dock)
        if key in seen:
            continue
        seen.add(key)
        _remove_workbench_dock_instance(dock, iface_obj)

    _SWE2D_WORKBENCH_STUDIO_DOCK = None
    _SWE2D_STUDIO_COMPONENT_DOCKS = {}

    if _SWE2D_STUDIO_HOST_DIALOG is not None:
        try:
            _SWE2D_STUDIO_HOST_DIALOG.close()
        except Exception:
            pass
        try:
            _SWE2D_STUDIO_HOST_DIALOG.deleteLater()
        except Exception:
            pass
        _SWE2D_STUDIO_HOST_DIALOG = None

    _clear_studio_host_controls(iface_obj)


def _attach_host_dock_widget(iface_obj, host_window, dock: QtWidgets.QDockWidget, area) -> bool:
    attached = False
    try:
        if iface_obj is not None and hasattr(iface_obj, "addDockWidget"):
            iface_obj.addDockWidget(area, dock)
            attached = True
    except Exception:
        attached = False
    if not attached:
        try:
            if host_window is not None and hasattr(host_window, "addDockWidget"):
                host_window.addDockWidget(area, dock)
                attached = True
        except Exception:
            attached = False
    if not attached:
        try:
            dock.show()
        except Exception:
            pass
        return False
    try:
        dock.setFloating(False)
    except Exception:
        pass
    try:
        dock.show()
        dock.raise_()
    except Exception:
        pass
    return True


def _studio_take_dock_widget(studio_dock, fallback_text: str) -> QtWidgets.QWidget:
    widget = None
    try:
        widget = studio_dock.widget() if studio_dock is not None else None
    except Exception:
        widget = None

    if widget is None:
        fallback = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(fallback)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QtWidgets.QLabel(fallback_text)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addStretch(1)
        return fallback

    try:
        if studio_dock is not None:
            studio_dock.setWidget(QtWidgets.QWidget())
    except Exception:
        pass
    try:
        widget.setParent(None)
    except Exception:
        pass
    return widget


def _build_studio_component_docks(iface_obj, host_window, dlg) -> Dict[str, QtWidgets.QDockWidget]:
    component_docks: Dict[str, QtWidgets.QDockWidget] = {}

    setup_widget = _studio_take_dock_widget(
        getattr(dlg, "_studio_left_dock", None),
        "Model Setup panel is unavailable.",
    )
    inspector_widget = _studio_take_dock_widget(
        getattr(dlg, "_studio_inspector_dock", None),
        "CFD Inspector panel is unavailable.",
    )

    view_widget = None
    log_widget = getattr(dlg, "log_view", None)
    split = getattr(dlg, "_right_vertical_split", None)
    if split is not None and hasattr(split, "widget"):
        try:
            if split.count() > 0:
                view_widget = split.widget(0)
        except Exception:
            view_widget = None

    if view_widget is None:
        try:
            mw = getattr(dlg, "_studio_main_window", None)
            if mw is not None:
                view_widget = mw.centralWidget()
        except Exception:
            view_widget = None

    if view_widget is None:
        fallback = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(fallback)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QtWidgets.QLabel("View panel is unavailable.")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        lay.addStretch(1)
        view_widget = fallback

    if log_widget is None:
        log_widget = QtWidgets.QPlainTextEdit()
        log_widget.setReadOnly(True)
        log_widget.setPlainText("Runtime log panel initialized.")

    for w in (view_widget, log_widget):
        try:
            w.setParent(None)
        except Exception:
            pass

    def _mkdock(title: str, obj_name: str, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        dock = QtWidgets.QDockWidget(title, host_window)
        dock.setObjectName(obj_name)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable,
        )
        dock.setWidget(widget)
        return dock

    component_docks["setup"] = _mkdock(
        "SWE2D Studio - Model Setup",
        "SWE2DStudioSetupHostDock",
        setup_widget,
    )
    component_docks["view"] = _mkdock(
        "SWE2D Studio - View",
        "SWE2DStudioViewHostDock",
        view_widget,
    )
    component_docks["log"] = _mkdock(
        "SWE2D Studio - Runtime Log",
        "SWE2DStudioLogHostDock",
        log_widget,
    )
    component_docks["inspector"] = _mkdock(
        "SWE2D Studio - CFD Inspector",
        "SWE2DStudioInspectorHostDock",
        inspector_widget,
    )

    _attach_host_dock_widget(iface_obj, host_window, component_docks["setup"], QtCore.Qt.LeftDockWidgetArea)
    _attach_host_dock_widget(iface_obj, host_window, component_docks["view"], QtCore.Qt.RightDockWidgetArea)
    _attach_host_dock_widget(iface_obj, host_window, component_docks["inspector"], QtCore.Qt.RightDockWidgetArea)
    _attach_host_dock_widget(iface_obj, host_window, component_docks["log"], QtCore.Qt.BottomDockWidgetArea)

    try:
        if host_window is not None and hasattr(host_window, "tabifyDockWidget"):
            host_window.tabifyDockWidget(component_docks["view"], component_docks["inspector"])
    except Exception:
        pass

    return component_docks


def _remove_workbench_scenario_dock(iface_obj) -> None:
    global _SWE2D_WORKBENCH_SCENARIO_DOCK
    _SWE2D_WORKBENCH_SCENARIO_DOCK = _remove_workbench_dock_instance(
        _SWE2D_WORKBENCH_SCENARIO_DOCK, iface_obj
    )


def _studio_host_main_window(iface_obj, fallback_parent=None):
    host_window = None
    if iface_obj is not None and hasattr(iface_obj, "mainWindow"):
        try:
            host_window = iface_obj.mainWindow()
        except Exception:
            host_window = None
    if host_window is None:
        host_window = fallback_parent
    return host_window


def _clear_studio_host_controls(iface_obj, fallback_parent=None) -> None:
    global _SWE2D_STUDIO_HOST_TOOLBAR, _SWE2D_STUDIO_HOST_MENU

    host_window = _studio_host_main_window(iface_obj, fallback_parent)

    if _SWE2D_STUDIO_HOST_TOOLBAR is not None:
        try:
            if iface_obj is not None and hasattr(iface_obj, "mainWindow") and host_window is not None:
                host_window.removeToolBar(_SWE2D_STUDIO_HOST_TOOLBAR)
        except Exception:
            pass
        try:
            _SWE2D_STUDIO_HOST_TOOLBAR.deleteLater()
        except Exception:
            pass
        _SWE2D_STUDIO_HOST_TOOLBAR = None

    if _SWE2D_STUDIO_HOST_MENU is not None:
        try:
            act = _SWE2D_STUDIO_HOST_MENU.menuAction()
            parent = act.parentWidget()
            if parent is not None:
                parent.removeAction(act)
        except Exception:
            pass
        try:
            _SWE2D_STUDIO_HOST_MENU.deleteLater()
        except Exception:
            pass
        _SWE2D_STUDIO_HOST_MENU = None


def _install_studio_host_controls(
    iface_obj,
    dlg,
    fallback_parent=None,
    component_docks: Optional[Dict[str, QtWidgets.QDockWidget]] = None,
) -> None:
    global _SWE2D_STUDIO_HOST_TOOLBAR, _SWE2D_STUDIO_HOST_MENU

    host_window = _studio_host_main_window(iface_obj, fallback_parent)
    if host_window is None:
        return

    _clear_studio_host_controls(iface_obj, fallback_parent)
    component_docks = dict(component_docks or {})

    def _focus_panel(name: str) -> None:
        dock = component_docks.get(str(name or "").strip().lower())
        if dock is None:
            return
        try:
            dock.show()
            dock.raise_()
        except Exception:
            pass

    def _close_studio_panels() -> None:
        try:
            _remove_workbench_studio_dock(iface_obj)
        except Exception:
            pass

    menu_bar = None
    try:
        menu_bar = host_window.menuBar()
    except Exception:
        menu_bar = None

    if menu_bar is not None:
        menu = QtWidgets.QMenu("SWE2D Studio", menu_bar)
        menu.setObjectName("SWE2DStudioHostMenu")

        rainfall_act = menu.addAction("Enable Rainfall")
        rainfall_act.setCheckable(True)
        rainfall_act.setChecked(True)

        drainage_act = menu.addAction("Enable Drainage")
        drainage_act.setCheckable(True)
        drainage_act.setChecked(True)

        structures_act = menu.addAction("Enable Structures")
        structures_act.setCheckable(True)
        structures_act.setChecked(True)

        menu.addSeparator()
        menu.addAction("Focus Mesh", lambda: dlg._studio_select_tab("mesh"))
        menu.addAction("Focus Model", lambda: dlg._studio_select_tab("model"))
        menu.addAction("Focus Run", lambda: dlg._studio_select_tab("run"))
        menu.addAction("Focus Map", lambda: dlg._studio_select_tab("map"))

        if component_docks:
            menu.addSeparator()
            menu.addAction("Show Model Setup Panel", lambda: _focus_panel("setup"))
            menu.addAction("Show View Panel", lambda: _focus_panel("view"))
            menu.addAction("Show Runtime Log Panel", lambda: _focus_panel("log"))
            menu.addAction("Show CFD Inspector Panel", lambda: _focus_panel("inspector"))

        menu.addSeparator()
        menu.addAction("Close Studio Panels", _close_studio_panels)

        rainfall_act.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("rainfall", checked))
        drainage_act.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("drainage", checked))
        structures_act.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("structures", checked))

        menu_bar.addMenu(menu)
        _SWE2D_STUDIO_HOST_MENU = menu

    toolbar = QtWidgets.QToolBar("SWE2D Studio", host_window)
    toolbar.setObjectName("SWE2DStudioHostToolbar")
    toolbar.setMovable(True)
    toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

    if component_docks:
        show_setup = toolbar.addAction("Setup Panel")
        show_view = toolbar.addAction("View Panel")
        show_log = toolbar.addAction("Log Panel")
        show_inspector = toolbar.addAction("Inspector Panel")
        show_setup.triggered.connect(lambda: _focus_panel("setup"))
        show_view.triggered.connect(lambda: _focus_panel("view"))
        show_log.triggered.connect(lambda: _focus_panel("log"))
        show_inspector.triggered.connect(lambda: _focus_panel("inspector"))
        toolbar.addSeparator()

    act_mesh = toolbar.addAction("Mesh")
    act_model = toolbar.addAction("Model")
    act_run = toolbar.addAction("Run")
    act_map = toolbar.addAction("Map")
    toolbar.addSeparator()

    act_refresh = toolbar.addAction("Refresh")
    act_snapshot = toolbar.addAction("Snapshot")
    toolbar.addSeparator()

    rainfall_tb = toolbar.addAction("Rainfall")
    rainfall_tb.setCheckable(True)
    rainfall_tb.setChecked(True)

    drainage_tb = toolbar.addAction("Drainage")
    drainage_tb.setCheckable(True)
    drainage_tb.setChecked(True)

    structures_tb = toolbar.addAction("Structures")
    structures_tb.setCheckable(True)
    structures_tb.setChecked(True)

    act_mesh.triggered.connect(lambda: dlg._studio_select_tab("mesh"))
    act_model.triggered.connect(lambda: dlg._studio_select_tab("model"))
    act_run.triggered.connect(lambda: dlg._studio_select_tab("run"))
    act_map.triggered.connect(lambda: dlg._studio_select_tab("map"))
    act_refresh.triggered.connect(dlg._refresh_layer_combos)
    act_snapshot.triggered.connect(
        lambda: dlg.snapshot_btn.click()
        if hasattr(dlg, "snapshot_btn") and dlg.snapshot_btn is not None
        else None
    )

    rainfall_tb.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("rainfall", checked))
    drainage_tb.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("drainage", checked))
    structures_tb.toggled.connect(lambda checked: dlg._studio_set_feature_enabled("structures", checked))

    toolbar.addSeparator()
    toolbar.addWidget(QtWidgets.QLabel(" View: "))
    host_view_combo = QtWidgets.QComboBox(toolbar)
    host_view_combo.addItems(["Mesh", "Depth", "Velocity magnitude", "Runtime Log"])
    try:
        source_idx = int(getattr(dlg, "view_mode_combo", host_view_combo).currentIndex())
        host_view_combo.setCurrentIndex(max(0, min(source_idx, host_view_combo.count() - 1)))
    except Exception:
        pass
    host_view_combo.currentIndexChanged.connect(
        lambda idx: dlg.view_mode_combo.setCurrentIndex(idx)
        if hasattr(dlg, "view_mode_combo") and dlg.view_mode_combo is not None
        else None
    )
    toolbar.addWidget(host_view_combo)

    toolbar.addWidget(QtWidgets.QLabel(" Theme: "))
    host_theme_combo = QtWidgets.QComboBox(toolbar)
    host_theme_combo.addItems(["Default", "Diagnostics", "Presentation"])
    host_theme_combo.currentTextChanged.connect(dlg._studio_apply_visual_profile)
    toolbar.addWidget(host_theme_combo)

    toolbar.addSeparator()
    act_close = toolbar.addAction("Close Studio")
    act_close.triggered.connect(_close_studio_panels)

    try:
        host_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
        _SWE2D_STUDIO_HOST_TOOLBAR = toolbar
    except Exception:
        try:
            toolbar.deleteLater()
        except Exception:
            pass


def launch_swe2d_workbench(parent=None, iface=None, host_mode: str = "window"):
    global _SWE2D_WORKBENCH_DOCK
    iface = _resolve_workbench_iface(parent, iface)

    mode = _normalize_workbench_host_mode(host_mode)

    if mode == "dock":
        _close_workbench_windows()
        if _SWE2D_WORKBENCH_DOCK is not None:
            try:
                _SWE2D_WORKBENCH_DOCK.show()
                _SWE2D_WORKBENCH_DOCK.raise_()
            except Exception:
                pass
            return _SWE2D_WORKBENCH_DOCK

        host_window = None
        if iface is not None and hasattr(iface, "mainWindow"):
            try:
                host_window = iface.mainWindow()
            except Exception:
                host_window = None
        if host_window is None:
            host_window = parent

        dock = QtWidgets.QDockWidget("2D SWE Workbench", host_window)
        dock.setObjectName("SWE2DWorkbenchDock")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable,
        )
        dlg = SWE2DWorkbenchDialog(host_window, iface=iface)
        # QDialog carries Qt::Dialog window flags that force an independent OS
        # window, preventing embedding in a QDockWidget.  Resetting to
        # Qt::Widget removes those flags so the dialog content sits inline in
        # the dock panel exactly like any other embedded QWidget.
        dlg.setWindowFlags(QtCore.Qt.Widget)
        dock.setWidget(dlg)
        _SWE2D_WORKBENCH_DOCK = dock

        try:
            if iface is not None and hasattr(iface, "addDockWidget"):
                iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            else:
                dock.show()
        except Exception:
            dock.show()
        try:
            dock.raise_()
        except Exception:
            pass
        return dock

    _remove_workbench_dock(iface)

    for existing in list(_SWE2D_WORKBENCH_WINDOWS):
        try:
            if existing.isVisible():
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return existing
        except Exception:
            pass

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
    return dlg


def launch_swe2d_workbench_designer(parent=None, iface=None, host_mode: str = "window"):
    global _SWE2D_WORKBENCH_DESIGNER_DOCK
    iface = _resolve_workbench_iface(parent, iface)

    mode = _normalize_workbench_host_mode(host_mode)

    if mode == "dock":
        _close_workbench_designer_windows()
        if _SWE2D_WORKBENCH_DESIGNER_DOCK is not None:
            try:
                _SWE2D_WORKBENCH_DESIGNER_DOCK.show()
                _SWE2D_WORKBENCH_DESIGNER_DOCK.raise_()
            except Exception:
                pass
            return _SWE2D_WORKBENCH_DESIGNER_DOCK

        host_window = None
        if iface is not None and hasattr(iface, "mainWindow"):
            try:
                host_window = iface.mainWindow()
            except Exception:
                host_window = None
        if host_window is None:
            host_window = parent

        dock = QtWidgets.QDockWidget("2D SWE Workbench (Designer UI)", host_window)
        dock.setObjectName("SWE2DWorkbenchDesignerDock")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable,
        )
        dlg = SWE2DWorkbenchDesignerDialog(host_window, iface=iface)
        dlg.setWindowFlags(QtCore.Qt.Widget)
        dock.setWidget(dlg)
        _SWE2D_WORKBENCH_DESIGNER_DOCK = dock

        try:
            if iface is not None and hasattr(iface, "addDockWidget"):
                iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            else:
                dock.show()
        except Exception:
            dock.show()
        try:
            dock.raise_()
        except Exception:
            pass
        return dock

    _remove_workbench_designer_dock(iface)

    for existing in list(_SWE2D_WORKBENCH_DESIGNER_WINDOWS):
        try:
            if existing.isVisible():
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return existing
        except Exception:
            pass

    dlg = SWE2DWorkbenchDesignerDialog(parent, iface=iface)

    def _cleanup():
        try:
            _SWE2D_WORKBENCH_DESIGNER_WINDOWS.remove(dlg)
        except ValueError:
            pass

    _SWE2D_WORKBENCH_DESIGNER_WINDOWS.append(dlg)
    dlg.finished.connect(_cleanup)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


def launch_swe2d_workbench_studio(parent=None, iface=None, host_mode: str = "dock"):
    global _SWE2D_WORKBENCH_STUDIO_DOCK, _SWE2D_STUDIO_COMPONENT_DOCKS, _SWE2D_STUDIO_HOST_DIALOG
    iface = _resolve_workbench_iface(parent, iface)

    def _enforce_studio_shell_visible(dlg: "SWE2DWorkbenchStudioDialog") -> None:
        try:
            mw = getattr(dlg, "_studio_main_window", None)
            if mw is None:
                return
            try:
                # Ensure Studio's internal shell is embedded in the dock host,
                # never as a detached top-level window.
                if mw.isWindow():
                    mw.setWindowFlags(QtCore.Qt.Widget)
                    mw.setParent(dlg)
            except Exception:
                pass
            center = mw.centralWidget()
            if center is None:
                fallback = QtWidgets.QWidget()
                lay = QtWidgets.QVBoxLayout(fallback)
                lay.setContentsMargins(12, 12, 12, 12)
                msg = QtWidgets.QLabel(
                    "Studio workspace recovered from an invalid layout state.\n"
                    "Use the left Model Setup dock to continue."
                )
                msg.setWordWrap(True)
                lay.addWidget(msg)
                lay.addStretch(1)
                mw.setCentralWidget(fallback)
                center = fallback
            try:
                center.show()
            except Exception:
                pass
            left_dock = getattr(dlg, "_studio_left_dock", None)
            if left_dock is not None and not left_dock.isVisible():
                left_dock.show()
            inspector_dock = getattr(dlg, "_studio_inspector_dock", None)
            if inspector_dock is not None and not inspector_dock.isVisible():
                inspector_dock.show()
        except Exception:
            pass

    # Studio is intentionally QGIS-integrated: always mount as a docked panel.
    mode = "dock"

    if mode == "dock":
        _close_workbench_studio_windows()
        # Always rebuild in dock mode to avoid reusing a stale/blank cached dock.
        _remove_workbench_studio_dock(iface)

        host_window = None
        if iface is not None and hasattr(iface, "mainWindow"):
            try:
                host_window = iface.mainWindow()
            except Exception:
                host_window = None
        if host_window is None:
            host_window = parent

        dlg = SWE2DWorkbenchStudioDialog(host_window, iface=iface)
        dlg._swe2d_workbench_host_mode = mode
        _enforce_studio_shell_visible(dlg)
        dlg.setWindowFlags(QtCore.Qt.Widget)
        try:
            dlg.hide()
        except Exception:
            pass

        component_docks = _build_studio_component_docks(iface, host_window, dlg)
        _SWE2D_STUDIO_COMPONENT_DOCKS = component_docks
        _SWE2D_STUDIO_HOST_DIALOG = dlg

        _install_studio_host_controls(iface, dlg, host_window, component_docks=component_docks)
        try:
            dlg._studio_update_status()
        except Exception:
            pass

        _SWE2D_WORKBENCH_STUDIO_DOCK = component_docks.get("view")
        if _SWE2D_WORKBENCH_STUDIO_DOCK is not None:
            return _SWE2D_WORKBENCH_STUDIO_DOCK
        return dlg

    _remove_workbench_studio_dock(iface)

    for existing in list(_SWE2D_WORKBENCH_STUDIO_WINDOWS):
        try:
            if existing.isVisible():
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return existing
        except Exception:
            pass

    dlg = SWE2DWorkbenchStudioDialog(parent, iface=iface)
    _enforce_studio_shell_visible(dlg)
    _install_studio_host_controls(iface, dlg, parent)

    def _cleanup():
        try:
            _SWE2D_WORKBENCH_STUDIO_WINDOWS.remove(dlg)
        except ValueError:
            pass
        _clear_studio_host_controls(iface, parent)

    _SWE2D_WORKBENCH_STUDIO_WINDOWS.append(dlg)
    dlg.finished.connect(_cleanup)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


def launch_swe2d_workbench_scenario(parent=None, iface=None, host_mode: str = "window"):
    global _SWE2D_WORKBENCH_SCENARIO_DOCK
    iface = _resolve_workbench_iface(parent, iface)

    mode = _normalize_workbench_host_mode(host_mode)

    if mode == "dock":
        _close_workbench_scenario_windows()
        if _SWE2D_WORKBENCH_SCENARIO_DOCK is not None:
            try:
                _SWE2D_WORKBENCH_SCENARIO_DOCK.show()
                _SWE2D_WORKBENCH_SCENARIO_DOCK.raise_()
            except Exception:
                pass
            return _SWE2D_WORKBENCH_SCENARIO_DOCK

        host_window = None
        if iface is not None and hasattr(iface, "mainWindow"):
            try:
                host_window = iface.mainWindow()
            except Exception:
                host_window = None
        if host_window is None:
            host_window = parent

        dock = QtWidgets.QDockWidget("2D SWE Workbench (Scenario-first)", host_window)
        dock.setObjectName("SWE2DWorkbenchScenarioDock")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable,
        )
        dlg = SWE2DWorkbenchScenarioDialog(host_window, iface=iface)
        dlg.setWindowFlags(QtCore.Qt.Widget)
        dock.setWidget(dlg)
        _SWE2D_WORKBENCH_SCENARIO_DOCK = dock

        try:
            if iface is not None and hasattr(iface, "addDockWidget"):
                iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            else:
                dock.show()
        except Exception:
            dock.show()
        try:
            dock.raise_()
        except Exception:
            pass
        return dock

    _remove_workbench_scenario_dock(iface)

    for existing in list(_SWE2D_WORKBENCH_SCENARIO_WINDOWS):
        try:
            if existing.isVisible():
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return existing
        except Exception:
            pass

    dlg = SWE2DWorkbenchScenarioDialog(parent, iface=iface)

    def _cleanup():
        try:
            _SWE2D_WORKBENCH_SCENARIO_WINDOWS.remove(dlg)
        except ValueError:
            pass

    _SWE2D_WORKBENCH_SCENARIO_WINDOWS.append(dlg)
    dlg.finished.connect(_cleanup)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
