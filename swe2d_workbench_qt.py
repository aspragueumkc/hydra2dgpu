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
import multiprocessing
import os
import sys
import sqlite3
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qgis.PyQt import QtCore, QtGui, QtWidgets

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
    mesh_fingerprint_from_mesh_data as _mesh_fingerprint_from_mesh_data_bridge,
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
    _bc_side_classification,
    apply_timeseries_bc_values as _apply_timeseries_bc_values_logic,
    distribute_total_flow_to_unit_q as _distribute_total_flow_to_unit_q_logic,
    interp_hydrograph as _interp_hydrograph_logic,
    normalize_inflow_to_uniform_velocity as _normalize_inflow_to_uniform_velocity_logic,
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
    mesh_cell_solver_bed as _mesh_cell_solver_bed_logic,
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
from swe2d.workbench.project_settings import (
    LAYER_SELECTOR_STATE_KEY,
    WORKBENCH_STATE_KEY,
    build_layer_selector_state,
    collect_workbench_widget_state,
    load_project_json,
    parse_layer_selector_state,
    read_project_entry_text,
    restore_workbench_widget_state,
    write_project_json,
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
    from swe2d.mesh.meshing import (
        conceptual_from_qgis_layers,
        generate_face_centric_mesh,
        _gmsh_available,
        _tqmesh_available,
    )
except Exception:
    try:
        from .swe2d_meshing import (
            conceptual_from_qgis_layers,
            generate_face_centric_mesh,
            _gmsh_available,
            _tqmesh_available,
        )
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
    "channel_generator",
    "empty",
]

_RECONSTRUCTION_OPTIONS = [
    ("First-order (baseline)",          0),
    ("MUSCL Fast (high-throughput)",     1),
    ("MUSCL MinMod (robust)",            2),
    ("MUSCL MC (less-diffusive TVD)",    3),
    ("MUSCL Van Leer (smooth TVD)",      4),
    ("WENO3-like (GPU experimental)",    5),
    ("WENO5 (GPU, 3rd-order LSQ)",        6),
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
_SWE2D_WORKBENCH_STUDIO_WINDOWS = []
_SWE2D_WORKBENCH_STUDIO_DOCK = None
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
    options_local = dict(options or {})

    # For GUI parity with the headless harness, optionally load meshing from a
    # workspace root explicitly provided by the caller.
    preferred_root = str(options_local.get("workspace_module_root", "") or "").strip()
    preferred_meshing_py = ""
    if preferred_root:
        candidate = os.path.join(preferred_root, "swe2d", "mesh", "meshing.py")
        if os.path.isfile(candidate):
            preferred_meshing_py = os.path.abspath(candidate)

    workspace_first_loaded = False
    workspace_first_error = ""
    gen_origin = ""

    # Use the already-imported function when available; fall back to local import
    # in subprocess contexts.
    gen = None
    if preferred_meshing_py:
        try:
            import importlib

            preferred_root_abs = os.path.abspath(preferred_root)
            if preferred_root_abs not in sys.path:
                sys.path.insert(0, preferred_root_abs)

            # Force package reload so import resolution honors workspace-first
            # sys.path ordering in worker processes.
            stale = [name for name in list(sys.modules.keys()) if name == "swe2d" or name.startswith("swe2d.")]
            for name in stale:
                try:
                    sys.modules.pop(name, None)
                except Exception:
                    pass
            importlib.invalidate_caches()

            from swe2d.mesh.meshing import generate_face_centric_mesh as _workspace_gen  # type: ignore

            gen = _workspace_gen
            workspace_first_loaded = True
            try:
                gen_origin = str(inspect.getsourcefile(gen) or inspect.getfile(gen) or "")
            except Exception:
                gen_origin = ""
            if not gen_origin:
                try:
                    mod_obj = sys.modules.get(getattr(gen, "__module__", ""))
                    gen_origin = str(getattr(mod_obj, "__file__", "") or "")
                except Exception:
                    gen_origin = ""
            if gen_origin:
                gen_origin_abs = os.path.abspath(gen_origin)
                workspace_first_loaded = bool(
                    gen_origin_abs.startswith(preferred_root_abs + os.sep)
                    or gen_origin_abs == preferred_root_abs
                )
        except Exception as exc:
            workspace_first_error = str(exc)
            gen = None

    if gen is None:
        gen = generate_face_centric_mesh
    if gen is None:
        try:
            from swe2d.mesh.meshing import generate_face_centric_mesh as gen  # type: ignore
        except Exception:
            from .swe2d_meshing import generate_face_centric_mesh as gen  # type: ignore

    if not gen_origin:
        try:
            gen_origin = str(inspect.getsourcefile(gen) or inspect.getfile(gen) or "")
        except Exception:
            gen_origin = ""

    mesh = gen(conceptual, backend=backend_name, options=options_local)
    try:
        summary = dict(getattr(mesh, "quality_summary", {}) or {})
        summary["meshing_module_origin"] = str(gen_origin)
        summary["meshing_module_workspace_first_requested"] = bool(preferred_meshing_py)
        summary["meshing_module_workspace_first_loaded"] = bool(workspace_first_loaded)
        if workspace_first_error:
            summary["meshing_module_workspace_first_error"] = str(workspace_first_error)
        if preferred_root:
            summary["workspace_module_root"] = str(os.path.abspath(preferred_root))
        mesh.quality_summary = summary
    except Exception:
        pass
    return mesh


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


def _quote_sqlite_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


class SWE2DSQLiteTablePreviewDialog(QtWidgets.QDialog):
    """Simple SQLite/GeoPackage table preview dialog."""

    def __init__(self, gpkg_path: str, table_name: str, title: str = "Table Preview", parent=None):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Table Preview"))
        self.resize(980, 640)
        self._gpkg_path = str(gpkg_path or "")
        self._table_name = str(table_name or "")

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(f"Source: {self._gpkg_path}\nTable: {self._table_name}"))

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Limit:"))
        self.limit_spin = QtWidgets.QSpinBox()
        self.limit_spin.setRange(10, 5000)
        self.limit_spin.setValue(250)
        row.addWidget(self.limit_spin)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        row.addWidget(self.refresh_btn)
        row.addStretch(1)
        root.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh_btn.clicked.connect(self.refresh_table)
        self.limit_spin.valueChanged.connect(lambda _v: self.refresh_table())
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        if not self._gpkg_path or not self._table_name or not os.path.exists(self._gpkg_path):
            return
        conn = sqlite3.connect(self._gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({_quote_sqlite_ident(self._table_name)})")
            cols = [str(r[1]) for r in cur.fetchall()]
            if not cols:
                return
            self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)
            lim = int(self.limit_spin.value())
            cur.execute(
                f"SELECT * FROM {_quote_sqlite_ident(self._table_name)} LIMIT ?",
                (lim,),
            )
            rows = cur.fetchall()
            self.table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    self.table.setItem(i, j, QtWidgets.QTableWidgetItem("" if val is None else str(val)))
        finally:
            conn.close()


class SWE2DModelGeoPackageExplorerDialog(QtWidgets.QDialog):
    """GeoPackage table explorer for opening table-aware viewers and table management."""

    def __init__(
        self,
        gpkg_path: str,
        open_run_log_viewer: Callable[[], None],
        open_line_results_viewer: Callable[[], None],
        open_coupling_results_viewer: Callable[[], None],
        logger: Callable[[str], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Model GeoPackage Explorer")
        self.resize(980, 660)
        self._gpkg_path = str(gpkg_path or "")
        self._open_run_log_viewer = open_run_log_viewer
        self._open_line_results_viewer = open_line_results_viewer
        self._open_coupling_results_viewer = open_coupling_results_viewer
        self._log = logger if callable(logger) else (lambda _msg: None)

        root = QtWidgets.QVBoxLayout(self)
        self.source_lbl = QtWidgets.QLabel(f"GeoPackage: {self._gpkg_path}")
        self.source_lbl.setWordWrap(True)
        root.addWidget(self.source_lbl)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Table", "Rows", "Type", "Actions"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.open_btn = QtWidgets.QPushButton("Open Viewer")
        self.preview_btn = QtWidgets.QPushButton("Preview Table")
        self.rename_btn = QtWidgets.QPushButton("Rename Table")
        self.delete_btn = QtWidgets.QPushButton("Delete Table")
        for btn in (self.refresh_btn, self.open_btn, self.preview_btn, self.rename_btn, self.delete_btn):
            row.addWidget(btn)
        row.addStretch(1)
        root.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh_btn.clicked.connect(self.refresh_tables)
        self.open_btn.clicked.connect(self.open_selected)
        self.preview_btn.clicked.connect(self.preview_selected)
        self.rename_btn.clicked.connect(self.rename_selected)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.table.itemSelectionChanged.connect(self._sync_button_state)
        self.table.itemDoubleClicked.connect(lambda _item: self.open_selected())

        self.refresh_tables()

    def _selected_table(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 0)
        return "" if item is None else str(item.text() or "").strip()

    def _table_kind(self, name: str) -> str:
        t = str(name or "").strip().lower()
        if t == "swe2d_run_logs" or t.endswith("_swe2d_run_logs"):
            return "run_log"
        if t.startswith("swe2d_line_results") or "_swe2d_line_results" in t:
            return "line_results"
        if t.startswith("swe2d_coupling_results") or "_swe2d_coupling_results" in t:
            return "coupling_results"
        if t.startswith("swe2d_mesh_results") or t.endswith("_swe2d_mesh_results") or t in ("swe2d_face_flux_results", "swe2d_face_results", "swe2d_flux_faces"):
            return "mesh_results"
        if (
            t.startswith("swe2d_conservation")
            or t.startswith("swe2d_boundary_flux_forensics")
            or t.startswith("swe2d_source_budget_forensics")
            or "_swe2d_conservation" in t
            or "_swe2d_boundary_flux_forensics" in t
            or "_swe2d_source_budget_forensics" in t
        ):
            return "conservation"
        if t.startswith("gpkg_") or t.startswith("sqlite_") or t.startswith("rtree_"):
            return "system"
        return "table"

    def _is_mutable_model_table(self, name: str) -> bool:
        t = str(name or "").strip().lower()
        return t.startswith("swe2d_")

    def _sync_button_state(self):
        name = self._selected_table()
        has_sel = bool(name)
        self.open_btn.setEnabled(has_sel)
        self.preview_btn.setEnabled(has_sel)
        mutable = has_sel and self._is_mutable_model_table(name)
        self.rename_btn.setEnabled(mutable)
        self.delete_btn.setEnabled(mutable)

    def refresh_tables(self):
        self.table.setRowCount(0)
        if not self._gpkg_path or not os.path.exists(self._gpkg_path):
            self._sync_button_state()
            return
        conn = sqlite3.connect(self._gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            names = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            for name in names:
                if name.startswith("sqlite_"):
                    continue
                row_idx = self.table.rowCount()
                self.table.insertRow(row_idx)
                self.table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(name))
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {_quote_sqlite_ident(name)}")
                    row = cur.fetchone()
                    n_rows = int(row[0]) if row and row[0] is not None else 0
                except Exception:
                    n_rows = -1
                self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem("?" if n_rows < 0 else str(n_rows)))
                self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(self._table_kind(name)))
                actions = "open+preview"
                if self._is_mutable_model_table(name):
                    actions += "+rename+delete"
                self.table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(actions))
        finally:
            conn.close()
        self.table.resizeColumnsToContents()
        self._sync_button_state()

    def _open_preview(self, name: str, title: str):
        dlg = SWE2DSQLiteTablePreviewDialog(self._gpkg_path, name, title=title, parent=self)
        dlg.exec()

    def open_selected(self):
        name = self._selected_table()
        if not name:
            return
        kind = self._table_kind(name)
        if kind == "run_log":
            self._open_run_log_viewer()
            return
        if kind == "line_results":
            self._open_line_results_viewer()
            return
        if kind == "coupling_results":
            self._open_coupling_results_viewer()
            return
        if kind == "mesh_results":
            self._open_preview(name, title=f"Mesh Results Viewer - {name}")
            return
        self._open_preview(name, title=f"Table Viewer - {name}")

    def preview_selected(self):
        name = self._selected_table()
        if not name:
            return
        self._open_preview(name, title=f"Table Viewer - {name}")

    def rename_selected(self):
        old_name = self._selected_table()
        if not old_name:
            return
        if not self._is_mutable_model_table(old_name):
            QtWidgets.QMessageBox.warning(self, "Rename Table", "Only model tables (swe2d_*) can be renamed from this explorer.")
            return
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Table", "New table name:", text=old_name)
        if not ok:
            return
        new_name = str(new_name or "").strip()
        if not new_name or new_name == old_name:
            return
        conn = sqlite3.connect(self._gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (new_name,))
            if cur.fetchone() is not None:
                QtWidgets.QMessageBox.warning(self, "Rename Table", f"Table '{new_name}' already exists.")
                return
            cur.execute(f"ALTER TABLE {_quote_sqlite_ident(old_name)} RENAME TO {_quote_sqlite_ident(new_name)}")
            for meta_tbl in ("gpkg_contents", "gpkg_geometry_columns", "gpkg_extensions"):
                cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (meta_tbl,))
                if cur.fetchone() is None:
                    continue
                try:
                    cur.execute(
                        f"UPDATE {_quote_sqlite_ident(meta_tbl)} SET table_name=? WHERE table_name=?",
                        (new_name, old_name),
                    )
                except Exception:
                    pass
            conn.commit()
            self._log(f"GeoPackage explorer renamed table: {old_name} -> {new_name}")
            self.refresh_tables()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Table", f"Failed to rename table:\n{exc}")
        finally:
            conn.close()

    def delete_selected(self):
        name = self._selected_table()
        if not name:
            return
        if not self._is_mutable_model_table(name):
            QtWidgets.QMessageBox.warning(self, "Delete Table", "Only model tables (swe2d_*) can be deleted from this explorer.")
            return
        ans = QtWidgets.QMessageBox.question(
            self,
            "Delete Table",
            f"Delete table '{name}' from GeoPackage?\n\nThis cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if ans != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        conn = sqlite3.connect(self._gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {_quote_sqlite_ident(name)}")
            for meta_tbl in ("gpkg_contents", "gpkg_geometry_columns", "gpkg_extensions"):
                cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (meta_tbl,))
                if cur.fetchone() is None:
                    continue
                try:
                    cur.execute(
                        f"DELETE FROM {_quote_sqlite_ident(meta_tbl)} WHERE table_name=?",
                        (name,),
                    )
                except Exception:
                    pass

            # Clean up common GeoPackage rtree sidecars if present.
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"rtree_{name}_%",))
            for row in cur.fetchall():
                t = str(row[0]) if row and row[0] is not None else ""
                if t:
                    try:
                        cur.execute(f"DROP TABLE IF EXISTS {_quote_sqlite_ident(t)}")
                    except Exception:
                        pass

            conn.commit()
            self._log(f"GeoPackage explorer deleted table: {name}")
            self.refresh_tables()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete Table", f"Failed to delete table:\n{exc}")
        finally:
            conn.close()


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
        ("flow_cms", "Flow FV Face ({Q})"),
        ("flow_cell_cms", "Flow Cell ({Q})"),
    ]

    _PLOT_OPTIONS = [
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Water Surface", "wse_m"),
        ("Bed", "bed_m"),
        ("Flow FV Face", "flow_cms"),
        ("Flow Cell", "flow_cell_cms"),
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
        length_unit: str = "",
        flow_unit_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("2D Sample Line Results Viewer")
        self.resize(980, 620)

        self._ts_records = list(ts_records)
        self._profile_records = list(profile_records)
        self._run_id = str(run_id)
        self._db_path = str(db_path)
        self._length_unit = str(length_unit).strip() or "m"
        self._flow_unit = str(flow_unit_label).strip() or f"{self._length_unit}3/s"
        l_unit = self._length_unit
        q_unit = self._flow_unit
        self._columns = [(k, lbl.format(L=l_unit, Q=q_unit)) for k, lbl in self._BASE_COLUMNS]
        self._plot_canvas = None
        self._plot_fig = None
        self._mpl_motion_cid = None

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
        export_btn = buttons.addButton("Export Table CSV...", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        export_btn.clicked.connect(self._export_current_table_csv)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_line_combo()
        self._populate_time_combo()
        self._sync_control_visibility()
        self._refresh_table()
        self._refresh_plot()
        self._notify_parent_line_selection()

        self.line_combo.currentIndexChanged.connect(self._refresh_table)
        self.line_combo.currentIndexChanged.connect(self._refresh_plot)
        self.line_combo.currentIndexChanged.connect(self._notify_parent_line_selection)
        self.metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.profile_metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.time_combo.currentIndexChanged.connect(self._refresh_table)
        self.time_combo.currentIndexChanged.connect(self._refresh_plot)
        self.fill_metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.wse_render_combo.currentIndexChanged.connect(self._refresh_plot)
        self.view_mode_combo.currentIndexChanged.connect(self._sync_control_visibility)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_table)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)
        self.view_mode_combo.currentIndexChanged.connect(self._notify_parent_line_selection)
        self.finished.connect(self._notify_parent_closed)

        if self._have_mpl and self._plot_canvas is not None:
            try:
                self._mpl_motion_cid = self._plot_canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
            except Exception:
                self._mpl_motion_cid = None

    def _unit_label_for_metric(self, metric: str) -> str:
        m = str(metric or "")
        if m in ("depth_m", "wse_m", "bed_m", "station_m"):
            return self._length_unit
        if m == "velocity_ms":
            return f"{self._length_unit}/s"
        if m in ("flow_cms", "flow_cell_cms", "flow_fv_cms"):
            return self._flow_unit
        if m == "flow_qn":
            return f"{self._length_unit}^2/s"
        return ""

    def _label_with_unit(self, label: str, metric: str) -> str:
        unit = self._unit_label_for_metric(metric)
        return str(label) if not unit else f"{label} ({unit})"

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
        def _fmt(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.6f}" if np.isfinite(v) else ""
            return str(v)

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
                    txt = _fmt(val)
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
            return

        rows = self._filtered_profile_records()
        rows.sort(key=lambda r: float(r.get("station_m", 0.0)))
        cols = [
            ("t_s", "Time (s)"),
            ("line_id", "Line ID"),
            ("line_name", "Line Name"),
            ("station_m", self._label_with_unit("Station", "station_m")),
            ("depth_m", self._columns[3][1]),
            ("velocity_ms", self._columns[4][1]),
            ("wse_m", self._columns[5][1]),
            ("bed_m", self._columns[6][1]),
            ("flow_qn", self._label_with_unit("Normal Flow Density", "flow_qn")),
            ("fr", "Froude"),
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in cols])
        self.table.setRowCount(len(rows))
        for r, rec in enumerate(rows):
            for c, (key, _) in enumerate(cols):
                val = rec.get(key)
                txt = _fmt(val)
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
            ax.set_ylabel(self._label_with_unit(self.metric_combo.currentText(), metric))
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
            ax.set_xlabel(self._label_with_unit("Station", "station_m"))
            ax.set_ylabel(self._label_with_unit(self.profile_metric_combo.currentText(), metric))
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
        ax.set_xlabel(self._label_with_unit("Station", "station_m"))
        ax.set_ylabel(self._label_with_unit("Elevation", "wse_m"))
        ax.set_title(f"Line {line_id} WSE + bed at t={t_s/3600.0:.4f} hr" + (f" ({line_name})" if line_name else ""))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()

    def _table_rows_for_export(self) -> Tuple[List[str], List[List[str]]]:
        headers = [str(self.table.horizontalHeaderItem(i).text()) if self.table.horizontalHeaderItem(i) is not None else f"col_{i}" for i in range(self.table.columnCount())]
        rows: List[List[str]] = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(self.table.columnCount()):
                it = self.table.item(r, c)
                row.append(str(it.text()) if it is not None else "")
            rows.append(row)
        return headers, rows

    def _export_current_table_csv(self):
        default_name = f"line_results_{self._run_id}.csv" if self._run_id else "line_results.csv"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Current Table to CSV",
            default_name,
            "CSV files (*.csv)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".csv"):
            out_path += ".csv"
        headers, rows = self._table_rows_for_export()
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
            QtWidgets.QMessageBox.information(self, "Export CSV", f"Exported {len(rows)} row(s) to:\n{out_path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export CSV", f"Failed to export CSV:\n{exc}")

    def _notify_parent_line_selection(self, *_):
        self._notify_parent_hover_station(None)
        p = self.parent()
        if p is None or not hasattr(p, "_on_line_viewer_selection_changed"):
            return
        try:
            p._on_line_viewer_selection_changed(self._line_filter())
        except Exception:
            pass

    def _notify_parent_hover_station(self, station_m: Optional[float]):
        p = self.parent()
        if p is None or not hasattr(p, "_on_line_viewer_hover_station"):
            return
        try:
            p._on_line_viewer_hover_station(self._line_filter(), station_m)
        except Exception:
            pass

    def _notify_parent_closed(self, *_):
        self._notify_parent_hover_station(None)
        p = self.parent()
        if p is None or not hasattr(p, "_on_line_viewer_selection_changed"):
            return
        try:
            p._on_line_viewer_selection_changed(None)
        except Exception:
            pass

    def _on_plot_hover(self, event):
        mode = str(self.view_mode_combo.currentData())
        if mode not in ("profile", "wse_bed"):
            self._notify_parent_hover_station(None)
            return
        if event is None or event.inaxes is None or event.xdata is None:
            self._notify_parent_hover_station(None)
            return
        try:
            station_m = float(event.xdata)
        except Exception:
            self._notify_parent_hover_station(None)
            return
        if not np.isfinite(station_m):
            self._notify_parent_hover_station(None)
            return
        self._notify_parent_hover_station(station_m)


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
        length_unit: str = "",
        flow_unit_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Drainage/Structure Results Viewer")
        self.resize(980, 620)

        self._records = list(records)
        self._run_id = str(run_id)
        self._db_path = str(db_path)
        self._length_unit = str(length_unit).strip() or "m"
        self._flow_unit = str(flow_unit_label).strip() or f"{self._length_unit}3/s"
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
        # Unit-agnostic metric names (no suffix) — all flows are in model³/s,
        # all depths/lengths in model units, as displayed to the user.
        _FLOW_METRICS = {
            "inlet_control_flow", "outlet_control_flow", "orifice_cap",
            "manning_cap", "embankment_flow",
        }
        _LENGTH_METRICS = {
            "available_head_up", "tailwater_depth",
            "inlet_invert_elev", "outlet_invert_elev",
        }
        if m in _FLOW_METRICS or m.endswith("_cms"):
            return self._flow_unit
        if m in _LENGTH_METRICS or m.endswith("_m"):
            return self._length_unit
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

    def __init__(
        self,
        records: List[Dict[str, object]],
        run_id: str,
        db_path: str,
        parent=None,
        apply_run_settings_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SWE2D Run Log Viewer")
        self.resize(900, 620)
        self._records = list(records)
        self._db_path = str(db_path)
        self._apply_run_settings_callback = apply_run_settings_callback

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
        self._apply_btn = None
        if callable(self._apply_run_settings_callback):
            self._apply_btn = buttons.addButton(
                "Apply Inputs To UI",
                QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
            )
            self._apply_btn.clicked.connect(self._apply_selected_run_settings)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_run_combo()
        idx = self.run_combo.findData(str(run_id))
        if idx >= 0:
            self.run_combo.setCurrentIndex(idx)
        self.run_combo.currentIndexChanged.connect(self._refresh_view)
        self._refresh_view()

    def _selected_record(self) -> Optional[Dict[str, object]]:
        rid = str(self.run_combo.currentData() or "")
        for rec in self._records:
            if str(rec.get("run_id", "") or "") == rid:
                return rec
        return None

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
        rec = self._selected_record()
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

    def _apply_selected_run_settings(self) -> None:
        if not callable(self._apply_run_settings_callback):
            return
        rec = self._selected_record()
        if rec is None:
            QtWidgets.QMessageBox.information(self, "Run Inputs", "No run record selected.")
            return
        metadata = rec.get("metadata")
        if not isinstance(metadata, dict):
            QtWidgets.QMessageBox.information(self, "Run Inputs", "Selected run has no metadata payload.")
            return
        try:
            restored = int(self._apply_run_settings_callback(metadata) or 0)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Run Inputs", f"Failed to apply saved inputs: {exc}")
            return
        if restored <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "Run Inputs",
                "Selected run metadata does not include restorable workbench inputs.",
            )
            return
        QtWidgets.QMessageBox.information(
            self,
            "Run Inputs",
            f"Applied {restored} saved input setting(s) to the workbench UI.",
        )


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
        self.field_combo.addItem("U velocity slice (XY)", "u_slice")
        self.field_combo.addItem("V velocity slice (XY)", "v_slice")
        self.field_combo.addItem("Speed slice (XY)", "speed_slice")
        self.field_combo.addItem("Column fill depth", "column_depth")
        self.field_combo.addItem("Column fill fraction", "column_fraction")
        self.field_combo.addItem("Column lowest z (bed height)", "column_bed_z")
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

    def _resolve_bed_field(
        self,
        snap: Dict[str, object],
        nx: int,
        ny: int,
        oz: float,
    ) -> np.ndarray:
        bed_flat = np.asarray(snap.get("bed_z", np.empty(0, dtype=np.float64)), dtype=np.float64).ravel()
        nxy = max(0, int(nx) * int(ny))
        if nxy > 0 and bed_flat.size == nxy:
            return bed_flat.reshape((ny, nx))
        return np.full((ny, nx), float(oz), dtype=np.float64)

    def _reshape_optional_field(
        self,
        snap: Dict[str, object],
        key: str,
        nx: int,
        ny: int,
        nz: int,
    ) -> Optional[np.ndarray]:
        arr = np.asarray(snap.get(key, np.empty(0, dtype=np.float64)), dtype=np.float64).ravel()
        n_exp = max(0, int(nx) * int(ny) * int(nz))
        if n_exp <= 0 or arr.size != n_exp:
            return None
        return arr.reshape((nz, ny, nx))

    def _refresh_controls(self):
        snap = self._current_snapshot()
        if snap is None:
            self.z_spin.setRange(0, 0)
            self.z_spin.setEnabled(False)
            return
        stats = dict(snap.get("stats", {}) or {})
        nz = max(1, int(stats.get("nz", 1) or 1))
        self.z_spin.setRange(0, max(0, nz - 1))
        want_slice = str(self.field_combo.currentData() or "") in {"vof_slice", "u_slice", "v_slice", "speed_slice"}
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
        stats = dict(snap.get("stats", {}) or {})
        patch_spec = dict(snap.get("patch_spec", {}) or {})
        oz = float(stats.get("origin_z", patch_spec.get("origin_z", 0.0)) or 0.0)

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
        elif field in {"u_slice", "v_slice", "speed_slice"}:
            u3d = self._reshape_optional_field(snap, key="u", nx=nx, ny=ny, nz=nz)
            v3d = self._reshape_optional_field(snap, key="v", nx=nx, ny=ny, nz=nz)
            if u3d is None or v3d is None:
                self.stats_lbl.setText(
                    "Velocity slices unavailable for this snapshot. "
                    "Re-run with a native build exposing 3D patch velocity observation."
                )
                return
            z_idx = int(np.clip(self.z_spin.value(), 0, max(0, nz - 1)))
            if field == "u_slice":
                arr = u3d[z_idx, :, :]
                title = f"U Velocity Slice (z={z_idx}/{max(0, nz - 1)})"
                cmap = "coolwarm"
                vabs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
                vmin = -vabs if vabs > 0.0 else None
                vmax = vabs if vabs > 0.0 else None
            elif field == "v_slice":
                arr = v3d[z_idx, :, :]
                title = f"V Velocity Slice (z={z_idx}/{max(0, nz - 1)})"
                cmap = "coolwarm"
                vabs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
                vmin = -vabs if vabs > 0.0 else None
                vmax = vabs if vabs > 0.0 else None
            else:
                arr = np.sqrt(np.maximum(0.0, u3d[z_idx, :, :] ** 2 + v3d[z_idx, :, :] ** 2))
                title = f"Horizontal Speed Slice (z={z_idx}/{max(0, nz - 1)})"
                cmap = "plasma"
                vmin = 0.0
                vmax = None
        elif field == "column_bed_z":
            arr = self._resolve_bed_field(snap, nx=nx, ny=ny, oz=oz)
            title = "Column Lowest Z (Bed Height)"
            cmap = "terrain"
            vmin = None
            vmax = None
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

    def _find_child_or_raise(self, parent: QtWidgets.QWidget, widget_type: type, name: str) -> QtWidgets.QWidget:
        """Find a child widget by type and name, raising RuntimeError if missing.

        Replaces bare findChild() calls to fail fast on missing widgets
        instead of propagating None silently.
        """
        w = parent.findChild(widget_type, name)
        if w is None:
            raise RuntimeError(
                f"Required widget '{name}' of type {widget_type.__name__} "
                f"not found in {parent.objectName() or type(parent).__name__}"
            )
        return w

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

            # Expose boundary controls as direct attributes so the generic
            # workbench-state persistence auto-discovery (vars(self)) can
            # save/restore values across QGIS sessions.
            setattr(self, f"{side}_bc_type_combo", cb)
            setattr(self, f"{side}_bc_value_spin", spin)
            setattr(self, f"{side}_bc_hydrograph_edit", ts_edit)
            setattr(self, f"{side}_bc_editor_btn", edit_btn)

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

        self.uniform_inflow_velocity_chk = _find_or_create_check(
            "uniform_inflow_velocity_chk",
            "Inflow BC: uniform velocity across all boundary edges (distributes Q by depth)"
        )
        self.uniform_inflow_velocity_chk.setChecked(False)
        _ensure_widget(self.uniform_inflow_velocity_chk, 7, 0, 1, 5)

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

        # Keep labels explicit even when loaded from a .ui shell with older text.
        self.terrain_to_nodes_btn.setText("Assign Mesh Node Z From Terrain")
        self.pull_node_z_btn.setText("Pull Mesh Node Z From Nodes Layer")
        self.terrain_to_nodes_btn.setToolTip(
            "Sample the selected terrain raster directly at in-memory mesh nodes and update mesh node_z."
        )
        self.pull_node_z_btn.setToolTip(
            "Legacy workflow: read bed_z values from the selected nodes layer into in-memory mesh node_z."
        )

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
        from swe2d.workbench.monolith_methods import _bind_map_tab_results_controls as _logic
        return _logic(self, map_tab_page, map_results_layout)

    def _bind_map_tab_tools_controls(self, map_tab_page: QtWidgets.QWidget, map_tools_layout: QtWidgets.QGridLayout) -> None:
        def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
            w = map_tab_page.findChild(QtWidgets.QPushButton, name)
            if w is None:
                w = QtWidgets.QPushButton(text)
                w.setObjectName(name)
            return w

        self.draw_sample_line_btn = _find_or_create_button("draw_sample_line_btn", "Draw Sample Line On Map")
        self.open_model_gpkg_explorer_btn = _find_or_create_button(
            "open_model_gpkg_explorer_btn", "Open Model GeoPackage Explorer"
        )
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
        if map_tools_layout.indexOf(self.open_model_gpkg_explorer_btn) < 0:
            map_tools_layout.addWidget(self.open_model_gpkg_explorer_btn, 1, 0, 1, 2)
        if map_tools_layout.indexOf(self.open_coupling_results_viewer_btn) < 0:
            map_tools_layout.addWidget(self.open_coupling_results_viewer_btn, 2, 0, 1, 2)
        if map_tools_layout.indexOf(self.open_run_log_viewer_btn) < 0:
            map_tools_layout.addWidget(self.open_run_log_viewer_btn, 3, 0, 1, 2)
        if map_tools_layout.indexOf(self.layer_status_lbl) < 0:
            map_tools_layout.addWidget(self.layer_status_lbl, 4, 0, 1, 2)
        if map_tools_layout.indexOf(self.open_3d_patch_viewer_btn) < 0:
            map_tools_layout.addWidget(self.open_3d_patch_viewer_btn, 5, 0, 1, 2)
        if map_tools_layout.indexOf(self.publish_3d_patch_surface_btn) < 0:
            map_tools_layout.addWidget(self.publish_3d_patch_surface_btn, 6, 0, 1, 2)

        self.draw_sample_line_btn.setToolTip("Draw a sample polyline directly on the map canvas")
        self.open_model_gpkg_explorer_btn.setToolTip(
            "Browse model GeoPackage tables and open matching viewers; rename/delete model result tables."
        )
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
            (self.open_model_gpkg_explorer_btn, self._open_model_gpkg_explorer),
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

    def _build_model_tab_page(self) -> Tuple[QtWidgets.QWidget, QtWidgets.QFormLayout, QtWidgets.QFormLayout, QtWidgets.QFormLayout]:
        """Load the Model tab from swe2d_model_tab.ui.

        Returns (page_widget, solver_form, rain_form, drain_form).
        The .ui contains a QToolBox with three pages:
          - model_solver_form  (Solver Parameters)
          - model_rain_form    (Rain / Hydrology)
          - model_drain_form   (Structures & Drainage)

        Each form's interactive widgets are wired by a corresponding
        _bind_model_tab_*_controls() call in _compose_left_pane().
        If you add a new QToolBox page to the .ui, update this
        method to find and return the new form, then add its bind
        call in _compose_left_pane().  See docs/STUDIO_UI_ARCHITECTURE.md.
        """
        ui_path = self._forms_file_path("swe2d_model_tab.ui")
        model_tab_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                model_tab_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                model_tab_page = None
        if model_tab_page is None:
            model_tab_page = self._build_model_tab_page_fallback()

        solver_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_solver_form")
        rain_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_rain_form")
        drain_form = model_tab_page.findChild(QtWidgets.QFormLayout, "model_drain_form")
        if solver_form is None or rain_form is None or drain_form is None:
            raise RuntimeError("Model tab UI missing one or more form layouts")
        return model_tab_page, solver_form, rain_form, drain_form

    def _build_model_tab_page_fallback(self) -> QtWidgets.QWidget:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        toolbox = QtWidgets.QToolBox()
        toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )

        for name, label in [
            ("model_solver_page", "Solver Parameters"),
            ("model_rain_page", "Rain / Hydrology"),
            ("model_drain_page", "Structures & Drainage"),
        ]:
            page = QtWidgets.QWidget()
            page.setObjectName(name)
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            form = QtWidgets.QFormLayout()
            form.setObjectName(name.replace("_page", "_form"))
            page_layout.addLayout(form)
            toolbox.addItem(page, label)

        root_layout.addWidget(toolbox)
        return root

    def _build_3d_patch_tab_page(self) -> QtWidgets.QWidget:
        """Load the 3D Patch settings from its dedicated .ui file."""
        ui_path = self._forms_file_path("swe2d_3d_patch_tab.ui")
        patch_page = None
        if _qgis_uic is not None and os.path.exists(ui_path):
            try:
                patch_page = _qgis_uic.loadUi(ui_path)
            except Exception:
                patch_page = None
        if patch_page is None:
            patch_page = QtWidgets.QWidget()
            patch_layout = QtWidgets.QVBoxLayout(patch_page)
            patch_layout.setContentsMargins(0, 0, 0, 0)
            patch_form = QtWidgets.QFormLayout()
            patch_form.setObjectName("patch_3d_form")
            patch_layout.addLayout(patch_form)

        patch_form = patch_page.findChild(QtWidgets.QFormLayout, "patch_3d_form")
        if patch_form is not None:
            self._bind_model_tab_3d_patch_controls(patch_page, patch_form)
        return patch_page

    def _bind_model_tab_core_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        from swe2d.workbench.monolith_methods import _bind_model_tab_core_controls as _logic
        return _logic(self, model_tab_page, param_form)

    def _bind_model_tab_hydrology_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        from swe2d.workbench.monolith_methods import _bind_model_tab_hydrology_controls as _logic
        return _logic(self, model_tab_page, param_form)

    def _bind_model_tab_solver_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        from swe2d.workbench.monolith_methods import _bind_model_tab_solver_controls as _logic
        return _logic(self, model_tab_page, param_form)

    def _bind_model_tab_3d_patch_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
        from swe2d.workbench.monolith_methods import _bind_model_tab_3d_patch_controls as _logic
        try:
            _g = getattr(_logic, "__globals__", None)
            if isinstance(_g, dict):
                _g.setdefault("_SWE3D_PATCH_FACES", _SWE3D_PATCH_FACES)
                _g.setdefault("_SWE3D_BC_MODE_OPTIONS", _SWE3D_BC_MODE_OPTIONS)
                _g.setdefault("_SWE3D_BC_FIELD_DEFAULTS", _SWE3D_BC_FIELD_DEFAULTS)
        except Exception:
            pass
        return _logic(self, model_tab_page, param_form)

    def _bind_model_tab_3d_subgrid_drainage_controls(
        self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout,
        solver_form: Optional[QtWidgets.QFormLayout] = None,
    ) -> None:
        from swe2d.workbench.monolith_methods import _bind_model_tab_3d_subgrid_drainage_controls as _logic
        return _logic(self, model_tab_page, param_form, solver_form)

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
            ("topo_quality_controls_lbl", "Quality controls (Gmsh):", 9),
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
            "and quad-edge n_layers / first_height / growth_rate for Gmsh transition spacing.",
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
        from swe2d.workbench.monolith_methods import _bind_topology_tab_dynamic_controls as _logic
        return _logic(self, topology_tab_page, topo_layout)

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
        from swe2d.workbench.monolith_methods import _bind_run_tab_controls as _logic
        return _logic(self, run_tab_page)

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
        from swe2d.workbench.monolith_methods import _bind_right_pane_controls as _logic
        return _logic(self, right_pane)

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
        ui_path = self._forms_file_path("swe2d_workbench.ui")
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
        """Build the Studio left-pane QTabWidget with all setup tabs.

        This is THE canonical tab registry for the Studio UI.  Every tab
        that appears in the left pane is added here via:
            page = self._build_<name>_tab_page()
            self._left_tabs.addTab(self._wrap_left_tab_page(page), "Label")

        If you add a new tab page (new .ui file or new QToolBox page),
        add it here AFTER building and binding it.
        The legacy shell dialog has a parallel
        tab list in studio_build_ui() that must be kept in sync.
        See docs/STUDIO_UI_ARCHITECTURE.md for the full checklist.
        """
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

        model_tab_page, solver_form, rain_form, drain_form = self._build_model_tab_page()
        self._bind_model_tab_core_controls(model_tab_page, solver_form)
        self._bind_model_tab_hydrology_controls(model_tab_page, rain_form)
        self._bind_model_tab_solver_controls(model_tab_page, solver_form)
        self._bind_model_tab_3d_subgrid_drainage_controls(model_tab_page, drain_form, solver_form)
        self._left_tabs.addTab(self._wrap_left_tab_page(model_tab_page), "Model")

        patch_tab_page = self._build_3d_patch_tab_page()
        patch_scroll = self._wrap_left_tab_page(patch_tab_page)
        patch_scroll.setObjectName("patch_3d_tab_page")
        self._left_tabs.addTab(patch_scroll, "3D Patch")

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
        # Render [ERROR] messages in red using appendHtml.
        # Render [ERROR] messages in red using appendHtml if available.
        if msg_txt.startswith("[ERROR]"):
            try:
                self.log_view.appendHtml(
                    f'<span style="color:red;font-weight:bold;">{msg_txt}</span>')
            except Exception:
                self.log_view.appendPlainText(msg_txt)
        else:
            self.log_view.appendPlainText(msg_txt)
        for dlg in list(getattr(self, "_runtime_log_detached_dialogs", [])):
            try:
                if dlg is not None:
                    dlg.append_text(msg_txt)
            except Exception:
                pass
        # Avoid pumping the Qt event loop on every log line; the run loop
        # already performs throttled processEvents calls for UI responsiveness.
        now = time.perf_counter()
        last = float(getattr(self, "_last_log_process_events_wall", 0.0) or 0.0)
        if (now - last) >= 0.10:
            QtWidgets.QApplication.processEvents()
            self._last_log_process_events_wall = now

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

    def _apply_3d_patch_face_bc_to_backend(self, backend: object, quiet: bool = False) -> None:
        return _apply_3d_patch_face_bc_to_backend_logic(
            ui=self,
            backend=backend,
            faces=_SWE3D_PATCH_FACES,
            field_defaults=_SWE3D_BC_FIELD_DEFAULTS,
            coupling_mode_off=int(SWE2DThreeDCouplingMode.OFF),
            get_coupling_mode_callback=self._experimental_3d_selected_coupling_mode,
            log_callback=self._log,
            quiet=bool(quiet),
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
        opts = dict(mesh_options or {})
        timeout = base
        if backend_name == "gmsh":
            gmsh_loop_enabled = self._opt_bool(opts.get("gmsh_quality_enable"), False)
            if gmsh_loop_enabled:
                budget_s = max(1.0, self._opt_float(opts.get("gmsh_quality_time_limit_s"), 60.0))
                # When the iterative Gmsh quality loop is enabled, enforce timeout from
                # its configured budget rather than the generic topology timeout floor.
                # Keep a small grace window for candidate finalization/return plumbing.
                grace_s = max(0.0, self._opt_float(opts.get("gmsh_quality_timeout_grace_s"), 10.0))
                timeout = max(timeout, max(30.0, budget_s + grace_s))

        return timeout

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

        if backend_name in {"gmsh", "tqmesh"} and self._topology_mesh_process_pool is not None:
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
        cp_path = str(getattr(self, "_topology_mesh_checkpoint_path", "") or "").strip()
        if cp_path:
            try:
                os.remove(cp_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._topology_mesh_checkpoint_path = ""
        progress_path = str(getattr(self, "_topology_mesh_progress_path", "") or "").strip()
        if progress_path:
            try:
                os.remove(progress_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._topology_mesh_progress_path = ""
        self._topology_mesh_progress_last_seq = -1
        self._topology_mesh_progress_last_sig = ""
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
        self._topology_mesh_checkpoint_path = ""
        self._topology_mesh_progress_path = ""
        self._topology_mesh_progress_last_seq = -1
        self._topology_mesh_progress_last_sig = ""
        self._topology_mesh_progress = None
        if backend_name == "gmsh":
            cp_dir = os.path.join("/tmp", "qgis-live-bridge")
            cp_name = f"topology_mesh_checkpoint_{os.getpid()}_{int(time.time() * 1000)}.npz"
            self._topology_mesh_checkpoint_path = os.path.join(cp_dir, cp_name)
            self._topology_mesh_options["gmsh_quality_checkpoint_path"] = self._topology_mesh_checkpoint_path
            progress_name = f"topology_gmsh_progress_{os.getpid()}_{int(time.time() * 1000)}.json"
            self._topology_mesh_progress_path = os.path.join(cp_dir, progress_name)
            self._topology_mesh_options["gmsh_progress_path"] = self._topology_mesh_progress_path
            self._topology_mesh_options.setdefault("gmsh_progress_emit_interval_s", 0.75)
            try:
                os.makedirs(cp_dir, exist_ok=True)
            except Exception:
                pass
            try:
                os.remove(self._topology_mesh_checkpoint_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
            try:
                os.remove(self._topology_mesh_progress_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
        elif backend_name == "tqmesh":
            cp_dir = os.path.join("/tmp", "qgis-live-bridge")
            progress_name = f"topology_tqmesh_progress_{os.getpid()}_{int(time.time() * 1000)}.json"
            self._topology_mesh_progress_path = os.path.join(cp_dir, progress_name)
            self._topology_mesh_options["tqmesh_progress_path"] = self._topology_mesh_progress_path
            self._topology_mesh_options.setdefault("tqmesh_progress_emit_interval_s", 0.75)
            try:
                os.makedirs(cp_dir, exist_ok=True)
            except Exception:
                pass
            try:
                os.remove(self._topology_mesh_progress_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass
        if run_mode == "full":
            self._topology_mesh_auto_fallback_used = False
        self._topology_mesh_started_at = time.perf_counter()
        self._topology_mesh_poll_count = 0
        self._topology_mesh_active_timeout_sec = self._effective_topology_timeout_sec(
            backend_name,
            self._topology_mesh_options,
        )

        if backend_name in {"gmsh", "tqmesh"}:
            # Keep Gmsh in a separate process to avoid UI freezes from C++
            # meshing work and signal-handler constraints.
            if self._topology_mesh_process_pool is None:
                try:
                    mp_ctx = multiprocessing.get_context("spawn")
                    self._topology_mesh_process_pool = concurrent.futures.ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=mp_ctx,
                    )
                    self._log("mesh> worker-start-method method=spawn")
                except Exception as exc:
                    self._topology_mesh_process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
                    self._log(f"mesh> worker-start-method method=default reason={exc}")
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
        from swe2d.workbench.monolith_methods import _poll_topology_mesh_future as _logic
        return _logic(self)

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
        from swe2d.workbench.monolith_methods import _configure_swe2d_layer_editors as _logic
        return _logic(self, layer)

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
        from swe2d import units as _u
        unit = self._detect_map_unit()
        unit_name = "m"
        sys_name = "SI"
        scale = 1.0

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
                    scale = 0.3048  # si_m_per_model for ft
                elif unit == getattr(QgsUnitTypes, "DistanceMeters", None):
                    unit_name = "m"
                    sys_name = "SI"
                    scale = 1.0
                else:
                    # Fallback to SI for unknown map units.
                    unit_name = str(QgsUnitTypes.toString(unit)) if hasattr(QgsUnitTypes, "toString") else "m"
                    sys_name = "SI (fallback)"
                    scale = 1.0
            except Exception:
                pass

        _u.configure(scale)
        g = _u.gravity()
        k_mann = _u.manning_factor()

        self._unit_system = sys_name
        self._length_unit_name = unit_name
        self._gravity = g
        self._k_mann = k_mann
        if hasattr(self, "unit_system_lbl"):
            self.unit_system_lbl.setText(
                f"Unit system: {sys_name} (CRS length unit: {unit_name}, gravity={g:.3f})"
            )

    def _is_us_customary_units(self) -> bool:
        return str(self._length_unit_name).strip().lower() == "ft"

    def _length_scale_si_to_model(self) -> float:
        # _update_unit_system_from_crs already called configure().
        # Return model units per SI meter for SI→model conversions.
        from swe2d import units as _u
        return _u.model_per_si_m()

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
        from swe2d.workbench.monolith_methods import _refresh_layer_combos as _logic
        try:
            _g = getattr(_logic, "__globals__", None)
            if isinstance(_g, dict):
                _g.setdefault("_HAVE_QGIS_CORE", _HAVE_QGIS_CORE)
                _g.setdefault("QgsProject", QgsProject)
                _g.setdefault("QgsVectorLayer", QgsVectorLayer)
                _g.setdefault("QgsRasterLayer", QgsRasterLayer)
        except Exception:
            pass
        return _logic(self)

    def _parse_csv_number_list(self, text: str, cast=float):
        values = []
        for part in str(text or "").split(","):
            item = part.strip()
            if not item:
                continue
            number = float(item)
            values.append(cast(number) if cast is int else cast(number))
        return values

    def _parse_csv_text_list(self, text: str):
        values = []
        for part in str(text or "").replace(";", ",").split(","):
            item = part.strip()
            if item:
                values.append(item)
        return values

    def _build_topology_meshing_options(self) -> Dict[str, object]:
        size_scales = tuple(self._parse_csv_number_list(self.topo_quality_size_scales_edit.text(), float) or [1.0])
        smooth_increments = tuple(self._parse_csv_number_list(self.topo_quality_smooth_increments_edit.text(), int) or [0])
        recombine_topology_passes = tuple(
            self._parse_csv_number_list(self.topo_gmsh_quality_recombine_topology_passes_edit.text(), int) or [5]
        )
        recombine_min_quality = tuple(
            self._parse_csv_number_list(self.topo_gmsh_quality_recombine_min_quality_edit.text(), float) or [0.01]
        )
        random_factors = tuple(
            self._parse_csv_number_list(self.topo_gmsh_quality_random_factors_edit.text(), float) or [1.0e-9]
        )
        optimize_methods = tuple(
            self._parse_csv_text_list(self.topo_gmsh_quality_optimize_methods_edit.text()) or ["Laplace2D", "Relocate2D"]
        )
        hybrid_tri_method = "frontal_delaunay"
        hybrid_transition_width = 1.25
        hybrid_transition_outer = 2.5
        hybrid_overbank_grading = 4.0
        hybrid_constrained_snap_tol = 12.0
        hybrid_constrained_max_flips = 128
        hybrid_region_conformance_band = 0.55
        hybrid_arc_conformance_band = 0.45
        hybrid_strict_conformance_mode = False

        gmsh_quad_full_region_flow_align = False
        if hasattr(self, "topo_gmsh_quad_full_region_flow_align_chk"):
            gmsh_quad_full_region_flow_align = bool(self.topo_gmsh_quad_full_region_flow_align_chk.isChecked())

        gmsh_global_recombine = False
        if hasattr(self, "topo_gmsh_global_recombine_chk"):
            gmsh_global_recombine = bool(self.topo_gmsh_global_recombine_chk.isChecked())

        gmsh_interface_transition_enable = True
        if hasattr(self, "topo_gmsh_interface_transition_enable_chk"):
            gmsh_interface_transition_enable = bool(self.topo_gmsh_interface_transition_enable_chk.isChecked())

        gmsh_interface_transition_dist_factor = 2.5
        if hasattr(self, "topo_gmsh_interface_transition_dist_factor_spin"):
            gmsh_interface_transition_dist_factor = float(self.topo_gmsh_interface_transition_dist_factor_spin.value())

        gmsh_interface_transition_min_ratio = 1.25
        if hasattr(self, "topo_gmsh_interface_transition_min_ratio_spin"):
            gmsh_interface_transition_min_ratio = float(self.topo_gmsh_interface_transition_min_ratio_spin.value())

        gmsh_interface_conformance = False
        if hasattr(self, "topo_gmsh_interface_conformance_chk"):
            gmsh_interface_conformance = bool(self.topo_gmsh_interface_conformance_chk.isChecked())

        gmsh_transverse_interface_centroid_merge = False
        if hasattr(self, "topo_gmsh_transverse_interface_centroid_merge_chk"):
            gmsh_transverse_interface_centroid_merge = bool(
                self.topo_gmsh_transverse_interface_centroid_merge_chk.isChecked()
            )

        gmsh_interface_snap_tol = 1.0
        if hasattr(self, "topo_gmsh_interface_snap_tol_spin"):
            gmsh_interface_snap_tol = float(self.topo_gmsh_interface_snap_tol_spin.value())

        gmsh_interface_reject_near_unshared = True
        if hasattr(self, "topo_gmsh_interface_reject_near_unshared_chk"):
            gmsh_interface_reject_near_unshared = bool(
                self.topo_gmsh_interface_reject_near_unshared_chk.isChecked()
            )

        gmsh_interface_reject_tol = 1.0e-3
        if hasattr(self, "topo_gmsh_interface_reject_tol_spin"):
            gmsh_interface_reject_tol = float(self.topo_gmsh_interface_reject_tol_spin.value())

        return {
            "gmsh_tri_algorithm": int(self.topo_gmsh_tri_algo_combo.currentData() or 6),
            "gmsh_quad_algorithm": int(self.topo_gmsh_quad_algo_combo.currentData() or 6),
            "gmsh_recombination_algorithm": int(self.topo_gmsh_recombine_algo_combo.currentData() or 1),
            "gmsh_smoothing": int(self.topo_gmsh_smoothing_spin.value()),
            "gmsh_optimize_iters": int(self.topo_gmsh_optimize_iters_spin.value()),
            "gmsh_optimize_netgen": bool(self.topo_gmsh_optimize_netgen_chk.isChecked()),
            "gmsh_arc_mode": str(self.topo_gmsh_arc_mode_combo.currentData() or "hard_embed"),
            "gmsh_arc_soft_size_factor": float(self.topo_gmsh_arc_soft_size_factor_spin.value()),
            "gmsh_arc_soft_dist_factor": float(self.topo_gmsh_arc_soft_dist_factor_spin.value()),
            "gmsh_interface_transition_enable": gmsh_interface_transition_enable,
            "gmsh_interface_transition_dist_factor": gmsh_interface_transition_dist_factor,
            "gmsh_interface_transition_min_ratio": gmsh_interface_transition_min_ratio,
            "gmsh_interface_conformance": gmsh_interface_conformance,
            "gmsh_transverse_interface_centroid_merge": gmsh_transverse_interface_centroid_merge,
            "gmsh_interface_snap_tol": gmsh_interface_snap_tol,
            "gmsh_interface_reject_near_unshared": gmsh_interface_reject_near_unshared,
            "gmsh_interface_reject_tol": gmsh_interface_reject_tol,
            "gmsh_mesh_size_min": float(self.topo_gmsh_mesh_size_min_spin.value()),
            "gmsh_tolerance_edge_length": float(self.topo_gmsh_tolerance_edge_length_spin.value()),
            "gmsh_mesh_size_from_points": bool(self.topo_gmsh_mesh_size_from_points_chk.isChecked()),
            "gmsh_verbosity": int(self.topo_gmsh_verbosity_spin.value()),
            "gmsh_num_threads": int(self.topo_gmsh_num_threads_spin.value()) if hasattr(self, "topo_gmsh_num_threads_spin") else 1,
            "gmsh_max_num_threads_2d": int(self.topo_gmsh_max_num_threads_2d_spin.value()) if hasattr(self, "topo_gmsh_max_num_threads_2d_spin") else 0,
            "gmsh_quad_full_region_flow_align": gmsh_quad_full_region_flow_align,
            "gmsh_global_recombine": gmsh_global_recombine,
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
            "gmsh_quality_recombine_topology_passes": recombine_topology_passes,
            "gmsh_quality_recombine_minimum_quality": recombine_min_quality,
            "gmsh_quality_random_factors": random_factors,
            "gmsh_quality_optimize_methods": optimize_methods,
            "gmsh_algorithm_switch_on_failure": bool(self.topo_gmsh_algo_switch_on_failure_chk.isChecked()),
            "gmsh_quality_recombine_node_repositioning": bool(self.topo_gmsh_recombine_node_repositioning_chk.isChecked()),
            "tri_meshing_method": hybrid_tri_method,
            "transition_width_factor": hybrid_transition_width,
            "transition_outer_factor": hybrid_transition_outer,
            "overbank_grading_factor": hybrid_overbank_grading,
            "hybridcpp_constrained_edge_snap_tol": hybrid_constrained_snap_tol,
            "hybridcpp_constrained_edge_max_flips": hybrid_constrained_max_flips,
            "hybridcpp_region_conformance_band_factor": hybrid_region_conformance_band,
            "hybridcpp_arc_conformance_band_factor": hybrid_arc_conformance_band,
            "hybridcpp_strict_conformance_mode": hybrid_strict_conformance_mode,
            "post_opt_backend": "none",
        }

    def _infer_workspace_root_for_meshing(self) -> str:
        """Best-effort workspace root discovery for workspace-first meshing imports."""

        env_root = str(os.environ.get("QGIS_BACKWATER_WORKSPACE_ROOT", "") or "").strip()
        if env_root:
            env_root_abs = os.path.abspath(env_root)
            if os.path.isfile(os.path.join(env_root_abs, "swe2d", "mesh", "meshing.py")):
                return env_root_abs

        candidates: List[str] = []
        if self._model_gpkg_path:
            candidates.append(str(self._model_gpkg_path))

        try:
            reg_layer = self._combo_layer(self.topo_regions_combo, "vector")
            if reg_layer is not None and hasattr(reg_layer, "source"):
                candidates.append(str(reg_layer.source() or ""))
        except Exception:
            pass

        try:
            if _HAVE_QGIS_CORE and QgsProject is not None:
                proj_path = str(QgsProject.instance().fileName() or "").strip()
                if proj_path:
                    candidates.append(proj_path)
        except Exception:
            pass

        for raw in candidates:
            src = str(raw or "").strip()
            if not src:
                continue
            root = src.split("|", 1)[0].strip()
            if not root:
                continue
            probe = os.path.abspath(root)
            if os.path.isfile(probe):
                probe = os.path.dirname(probe)
            if not os.path.isdir(probe):
                continue

            here = probe
            while True:
                meshing_py = os.path.join(here, "swe2d", "mesh", "meshing.py")
                workbench_py = os.path.join(here, "swe2d_workbench_qt.py")
                if os.path.isfile(meshing_py) and os.path.isfile(workbench_py):
                    return os.path.abspath(here)
                parent = os.path.dirname(here)
                if parent == here:
                    break
                here = parent

        return ""

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
        from swe2d.workbench.monolith_methods import _update_topology_control_summary as _logic
        return _logic(self)

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
            f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer"
            "&field=region_id:integer&field=arc_role:string(24)"
            "&field=use_global_arc_ctrl:integer&field=arc_mode_override:string(24)"
            "&field=arc_soft_size_override:double&field=arc_soft_dist_override:double",
            "SWE2D_Topo_Arcs",
            "memory",
        )
        regions = QgsVectorLayer(
            f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double&field=cell_type:string(32)&field=channel_generator_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
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
            f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width:double&field=height:double&field=diameter:double&field=culvert_shape:string(32)&field=culvert_code:integer&field=culvert_rise:double&field=culvert_span:double&field=culvert_area_m2:double&field=culvert_barrels:integer&field=culvert_slope:double&field=inlet_invert_elev:double&field=outlet_invert_elev:double&field=entrance_loss_k:double&field=exit_loss_k:double&field=embankment_enabled:integer&field=embankment_crest_elev:double&field=embankment_overflow_width:double&field=embankment_weir_coeff:double&field=length:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double",
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
            "add optional arcs/constraints and optional quad-edge control lines; then generate mesh."
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
        from swe2d.workbench.monolith_methods import _create_2d_model_geopackage as _logic
        return _logic(self)

    def _migrate_2d_model_geopackage(self):
        from swe2d.workbench.monolith_methods import _migrate_2d_model_geopackage as _logic
        return _logic(self)

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

        self._reset_runtime_snapshot_overlay_cache("model GeoPackage loaded")

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
        if self.topo_nodes_combo.currentData() is None:
            nodes_layer = None
        if self.topo_arcs_combo.currentData() is None:
            arcs_layer = None
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
        backend_name = str(self.topo_backend_combo.currentData() or "gmsh")

        try:
            mesh_options = self._build_topology_meshing_options()
            if backend_name == "gmsh":
                workspace_root = self._infer_workspace_root_for_meshing()
                if workspace_root:
                    mesh_options["workspace_module_root"] = workspace_root
                    self._log(f"mesh> module-path workspace-root={workspace_root}")
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
            f"Polygon?crs={crs_auth}&field=cell_id:integer&field=n0:integer&field=n1:integer&field=n2:integer&field=n3:integer&field=node_ids:string(512)&field=cell_type:string(32)&field=region_id:integer&field=target_size:double",
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
        face_offsets = self._mesh_data.get("cell_face_offsets")
        face_nodes = self._mesh_data.get("cell_face_nodes")
        if face_offsets is not None and face_nodes is not None:
            offs = np.asarray(face_offsets, dtype=np.int32).ravel()
            nodes = np.asarray(face_nodes, dtype=np.int32).ravel()
            face_ids = [nodes[int(offs[i]) : int(offs[i + 1])].tolist() for i in range(max(0, int(offs.size) - 1))]
        else:
            face_ids = [tri.tolist() for tri in triangles]

        for cid, ids in enumerate(face_ids):
            ids_i = [int(v) for v in ids]
            if len(ids_i) < 3:
                continue
            poly = [QgsPointXY(float(node_x[nid]), float(node_y[nid])) for nid in ids_i]
            poly.append(poly[0])
            f = QgsFeature(cells_layer.fields())
            f.setAttribute("cell_id", int(cid))
            f.setAttribute("n0", int(ids_i[0]) if len(ids_i) > 0 else None)
            f.setAttribute("n1", int(ids_i[1]) if len(ids_i) > 1 else None)
            f.setAttribute("n2", int(ids_i[2]) if len(ids_i) > 2 else None)
            f.setAttribute("n3", int(ids_i[3]) if len(ids_i) > 3 else None)
            f.setAttribute("node_ids", ",".join(str(int(nid)) for nid in ids_i))
            if cell_type_meta is not None and cid < len(cell_type_meta):
                f.setAttribute("cell_type", str(cell_type_meta[cid]))
            else:
                f.setAttribute("cell_type", "quadrilateral" if len(ids_i) == 4 else "triangular")
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
        from swe2d.workbench.monolith_methods import _import_mesh_from_layers as _logic
        try:
            _g = getattr(_logic, "__globals__", None)
            if isinstance(_g, dict):
                _g["_HAVE_QGIS_CORE"] = _HAVE_QGIS_CORE
        except Exception:
            pass
        return _logic(self)

    def _assign_node_z_from_terrain(self):
        if not _HAVE_QGIS_CORE:
            return
        raster_layer = self._combo_layer(self.terrain_layer_combo, "raster")
        if raster_layer is None:
            self._log("Select a terrain raster layer first.")
            return

        provider = raster_layer.dataProvider()

        # Preferred path: sample directly onto in-memory mesh nodes so large
        # meshes do not require export/edit/re-import layer workflows.
        if self._mesh_data is not None:
            node_x = np.asarray(self._mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
            node_y = np.asarray(self._mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
            if node_x.size <= 0 or node_y.size <= 0:
                self._log("Mesh node coordinates are unavailable; regenerate or load a mesh first.")
                return
            n = int(min(node_x.size, node_y.size))
            node_z = np.asarray(self._mesh_data.get("node_z", np.zeros(n, dtype=np.float64)), dtype=np.float64).ravel()
            if node_z.size < n:
                node_z = np.pad(node_z, (0, n - node_z.size), mode="constant")
            elif node_z.size > n:
                node_z = node_z[:n]

            sampled = 0
            for i in range(n):
                val, ok = provider.sample(QgsPointXY(float(node_x[i]), float(node_y[i])), 1)
                if ok:
                    node_z[i] = float(val)
                    sampled += 1

            self._mesh_data["node_z"] = node_z.astype(np.float64, copy=False)
            self._result_data = None
            if hasattr(self, "_reset_runtime_snapshot_overlay_cache"):
                self._reset_runtime_snapshot_overlay_cache("terrain Z assigned to mesh nodes")
            self._log(f"Assigned terrain node_z for {sampled}/{n} mesh nodes (direct mesh update).")
            self.layer_status_lbl.setText("Terrain Z assigned directly to in-memory mesh nodes.")
            self._refresh_plot()
            return

        # Legacy fallback when operating on map layers without an active mesh.
        nodes_layer = self._combo_layer(self.nodes_layer_combo, "vector")
        if nodes_layer is None:
            self._log("Generate/load a mesh first, or select a nodes point layer for legacy workflow.")
            return

        field_names = nodes_layer.fields().names()
        if "bed_z" not in field_names:
            nodes_layer.dataProvider().addAttributes([QgsField("bed_z", QVariant.Double)])
            nodes_layer.updateFields()

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
        self._log(f"Assigned terrain bed_z for {sampled} node features (nodes layer workflow).")
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

    def _mesh_cell_solver_bed(self) -> np.ndarray:
        assert self._mesh_data is not None
        return _mesh_cell_solver_bed_logic(self._mesh_data)

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
        from swe2d.workbench.monolith_methods import _build_line_sampling_map as _logic
        smap = _logic(self)
        try:
            self._line_sampling_map_cache = list(smap or [])
            by_line: Dict[int, np.ndarray] = {}
            for sm in self._line_sampling_map_cache:
                try:
                    lid = int(sm.get("line_id", -1))
                except Exception:
                    continue
                seg = np.asarray(sm.get("flux_face_segments", np.empty((0, 4))), dtype=np.float64)
                if seg.ndim != 2 or seg.shape[1] != 4 or seg.size <= 0:
                    continue
                by_line[lid] = seg
            self._line_flux_face_segments_by_line = by_line
        except Exception:
            self._line_sampling_map_cache = []
            self._line_flux_face_segments_by_line = {}
        return smap

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
            huu_wet = np.where(wet, huu, 0.0)
            hvv_wet = np.where(wet, hvv, 0.0)
            uu = np.where(wet, huu_wet / safe_h, 0.0)
            vv = np.where(wet, hvv_wet / safe_h, 0.0)
            normal_v = uu * float(sm["normal_x"]) + vv * float(sm["normal_y"])
            # Normal unit discharge (m^2/s): qn = h * u_n.
            qn = np.where(wet, hh * normal_v, 0.0)
            flow_wx = np.asarray(sm.get("flow_wx", []), dtype=np.float64)
            flow_wy = np.asarray(sm.get("flow_wy", []), dtype=np.float64)
            flow_cell_cms = float("nan")
            if flow_wx.size == idx.size and flow_wy.size == idx.size:
                # Exact per-cell line-integral weights from local segment orientation.
                # Q = sum(h * (u dot n) * ds), where flow_wx/flow_wy carry n*ds.
                flow_cell_cms = float(np.sum(np.where(wet, hh * (uu * flow_wx + vv * flow_wy), 0.0)))
            else:
                # Fallback uses averaged segment lengths per sampled cell.
                flow_cell_cms = float(np.sum(qn * w))

            # Finite-volume face-based approximation: project sample-line normal
            # integral weights onto nearest mesh faces and use face-centered
            # momentum (averaged from adjacent cells).
            flow_fv_cms = float("nan")
            f_idx = np.asarray(sm.get("flux_face_idx", []), dtype=np.int32)
            f_wx = np.asarray(sm.get("flux_face_wx", []), dtype=np.float64)
            f_wy = np.asarray(sm.get("flux_face_wy", []), dtype=np.float64)
            f_c0 = np.asarray(sm.get("flux_face_c0", []), dtype=np.int32)
            f_c1 = np.asarray(sm.get("flux_face_c1", []), dtype=np.int32)
            if (
                f_idx.size > 0
                and f_wx.size == f_idx.size
                and f_wy.size == f_idx.size
                and f_c0.size == f_idx.size
                and f_c1.size == f_idx.size
            ):
                c0 = np.asarray(f_c0, dtype=np.int32)
                c1 = np.asarray(f_c1, dtype=np.int32)
                valid_c0 = (c0 >= 0) & (c0 < h.size)
                valid_c1 = (c1 >= 0) & (c1 < h.size)
                hu_f = np.zeros(f_idx.size, dtype=np.float64)
                hv_f = np.zeros(f_idx.size, dtype=np.float64)
                if np.any(valid_c0):
                    hu_f[valid_c0] = hu[c0[valid_c0]]
                    hv_f[valid_c0] = hv[c0[valid_c0]]
                both = valid_c0 & valid_c1
                if np.any(both):
                    hu_f[both] = 0.5 * (hu[c0[both]] + hu[c1[both]])
                    hv_f[both] = 0.5 * (hv[c0[both]] + hv[c1[both]])
                valid_face = valid_c0 | valid_c1
                if np.any(valid_face):
                    flow_fv_cms = float(np.sum((hu_f[valid_face] * f_wx[valid_face]) + (hv_f[valid_face] * f_wy[valid_face])))

            flow_cms = flow_fv_cms if np.isfinite(flow_fv_cms) else flow_cell_cms
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
                    "flow_cell_cms": flow_cell_cms,
                    "flow_fv_cms": flow_fv_cms,
                    "wet_frac": float(np.mean(wet.astype(np.float64))),
                    "fr": float(np.mean(fr_arr)),
                }
            )

            p_sta = np.asarray(sm.get("profile_station_m", np.empty(0, dtype=np.float64)), dtype=np.float64)
            p_idx = np.asarray(sm.get("profile_cell_idx", np.empty((0, 0), dtype=np.int32)), dtype=np.int32)
            p_w = np.asarray(sm.get("profile_cell_w", np.empty((0, 0), dtype=np.float64)), dtype=np.float64)

            use_hi_fidelity = (
                p_sta.ndim == 1
                and p_sta.size > 0
                and p_idx.ndim == 2
                and p_w.ndim == 2
                and p_idx.shape == p_w.shape
                and p_idx.shape[0] == p_sta.size
            )

            if use_hi_fidelity:
                valid = p_idx >= 0
                safe_idx = np.where(valid, p_idx, 0)
                ww = np.where(valid, p_w, 0.0)
                wsum = np.sum(ww, axis=1)
                good = np.isfinite(wsum) & (wsum > 0.0)

                h_nei = h[safe_idx]
                hu_nei = hu[safe_idx]
                hv_nei = hv[safe_idx]
                zb_nei = cell_bed[safe_idx]

                hh_p = np.where(good, np.sum(h_nei * ww, axis=1) / np.maximum(wsum, 1.0e-12), np.nan)
                huu_p = np.where(good, np.sum(hu_nei * ww, axis=1) / np.maximum(wsum, 1.0e-12), np.nan)
                hvv_p = np.where(good, np.sum(hv_nei * ww, axis=1) / np.maximum(wsum, 1.0e-12), np.nan)
                zb_p = np.where(good, np.sum(zb_nei * ww, axis=1) / np.maximum(wsum, 1.0e-12), np.nan)

                wet_p = good & np.isfinite(hh_p) & (hh_p > h_min)
                safe_h_p = np.maximum(hh_p, 1.0e-12)
                uu_p = np.where(wet_p, huu_p / safe_h_p, 0.0)
                vv_p = np.where(wet_p, hvv_p / safe_h_p, 0.0)
                vel_p = np.where(wet_p, np.sqrt(uu_p * uu_p + vv_p * vv_p), 0.0)
                qn_p = np.where(wet_p, hh_p * (uu_p * float(sm["normal_x"]) + vv_p * float(sm["normal_y"])), 0.0)
                fr_p = np.where(wet_p, vel_p / np.sqrt(np.maximum(g * hh_p, 1.0e-12)), 0.0)

                for j in range(p_sta.size):
                    if not np.isfinite(p_sta[j]):
                        continue
                    out_prof.append(
                        {
                            "t_s": float(t_s),
                            "line_id": int(sm["line_id"]),
                            "line_name": str(sm.get("line_name", "") or ""),
                            "station_m": float(p_sta[j]),
                            "depth_m": float(hh_p[j]) if np.isfinite(hh_p[j]) else float("nan"),
                            "velocity_ms": float(vel_p[j]) if np.isfinite(vel_p[j]) else float("nan"),
                            "wse_m": float(hh_p[j] + zb_p[j]) if np.isfinite(hh_p[j]) and np.isfinite(zb_p[j]) else float("nan"),
                            "bed_m": float(zb_p[j]) if np.isfinite(zb_p[j]) else float("nan"),
                            "flow_qn": float(qn_p[j]) if np.isfinite(qn_p[j]) else float("nan"),
                            "wet": int(bool(wet_p[j])),
                            "fr": float(fr_p[j]) if np.isfinite(fr_p[j]) else float("nan"),
                        }
                    )
            else:
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
        path_edit = getattr(self, "results_gpkg_path_edit", None)
        if path_edit is not None:
            override_raw = str(path_edit.text() or "").strip()
            if override_raw:
                override = os.path.abspath(os.path.expanduser(override_raw))
                parent_dir = os.path.dirname(override) or "."
                if os.path.isdir(parent_dir):
                    return override
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
        from swe2d.workbench.monolith_methods import _persist_line_results_to_geopackage as _logic
        return _logic(self, gpkg_path, run_id, rows, mesh_interval_s, line_interval_s, profile_rows)

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
            def _table_exists(name: str) -> bool:
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (str(name),),
                )
                return cur.fetchone() is not None

            ts_candidates = [
                self._results_table_name("swe2d_line_results_ts"),
                "swe2d_line_results_ts",
            ]
            ts_table = ""
            for cand in ts_candidates:
                if _table_exists(cand):
                    ts_table = str(cand)
                    break
            if not ts_table:
                return "", [], []

            runs_candidates = [
                self._results_table_name("swe2d_line_results_runs"),
                "swe2d_line_results_runs",
            ]
            runs_table = ""
            for cand in runs_candidates:
                if _table_exists(cand):
                    runs_table = str(cand)
                    break

            chosen = str(run_id or "").strip()
            if not chosen:
                if not runs_table:
                    return "", [], []
                q_runs = runs_table.replace('"', '""')
                cur.execute(
                    f"""
                    SELECT run_id FROM \"{q_runs}\"
                    ORDER BY datetime(created_utc) DESC, rowid DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row is None:
                    return "", [], []
                chosen = str(row[0])

            q_ts = ts_table.replace('"', '""')

            cur.execute(
                f"""
                SELECT t_s, line_id, line_name, depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr
                FROM \"{q_ts}\"
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
            profile_candidates = [
                self._results_table_name("swe2d_line_results_profile"),
                "swe2d_line_results_profile",
            ]
            profile_table = ""
            for cand in profile_candidates:
                if _table_exists(cand):
                    profile_table = str(cand)
                    break
            if profile_table:
                q_profile = profile_table.replace('"', '""')
                cur.execute(
                    f"""
                    SELECT t_s, line_id, line_name, station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, wet, fr
                    FROM \"{q_profile}\"
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

    def _current_mesh_fingerprint(self) -> str:
        return _mesh_fingerprint_from_mesh_data_bridge(getattr(self, "_mesh_data", {}) or {})

    def _reset_runtime_snapshot_overlay_cache(self, reason: str = "") -> None:
        self._snapshot_timesteps = []
        self._snapshot_mesh_fingerprint = ""
        self._line_snapshot_rows = []
        self._line_snapshot_profile_rows = []
        self._coupling_snapshot_rows = []
        self._high_perf_overlay_cell_x = np.empty(0, dtype=np.float64)
        self._high_perf_overlay_cell_y = np.empty(0, dtype=np.float64)
        self._high_perf_overlay_cell_bed = np.empty(0, dtype=np.float64)
        self._high_perf_overlay_node_x = np.empty(0, dtype=np.float64)
        self._high_perf_overlay_node_y = np.empty(0, dtype=np.float64)
        self._high_perf_overlay_cell_nodes = np.empty(0, dtype=np.int32)
        self._high_perf_overlay_mesh_fingerprint = ""
        item = getattr(self, "_high_perf_canvas_overlay_item", None)
        if item is not None:
            try:
                item.clear()
            except Exception:
                pass
        if reason:
            self._log(f"[HighPerf Overlay] Cleared snapshot/overlay cache: {reason}")

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
        field_key = str(self.high_perf_canvas_overlay_field_combo.currentData() or "depth") if hasattr(self, "high_perf_canvas_overlay_field_combo") else "depth"
        lock_canvas = bool(self.high_perf_canvas_overlay_lock_canvas_chk.isChecked())
        self.high_perf_canvas_overlay_res_combo.setEnabled(not lock_canvas)
        if hasattr(self, "high_perf_canvas_overlay_wse_render_combo"):
            self.high_perf_canvas_overlay_wse_render_combo.setEnabled(field_key == "wse")
        if hasattr(self, "high_perf_canvas_overlay_arrows_chk") and hasattr(self, "high_perf_canvas_overlay_arrow_density_spin"):
            arrows_on = bool(self.high_perf_canvas_overlay_arrows_chk.isChecked())
            self.high_perf_canvas_overlay_arrow_density_spin.setEnabled(arrows_on)
            if hasattr(self, "high_perf_canvas_overlay_arrow_length_spin"):
                self.high_perf_canvas_overlay_arrow_length_spin.setEnabled(arrows_on)
            if hasattr(self, "high_perf_canvas_overlay_arrow_head_length_spin"):
                self.high_perf_canvas_overlay_arrow_head_length_spin.setEnabled(arrows_on)
            if hasattr(self, "high_perf_canvas_overlay_arrow_head_width_spin"):
                self.high_perf_canvas_overlay_arrow_head_width_spin.setEnabled(arrows_on)
        if (
            hasattr(self, "high_perf_canvas_overlay_streamlines_chk")
            and hasattr(self, "high_perf_canvas_overlay_streamline_backend_combo")
            and hasattr(self, "high_perf_canvas_overlay_streamline_seed_spin")
            and hasattr(self, "high_perf_canvas_overlay_streamline_steps_spin")
        ):
            stream_on = bool(self.high_perf_canvas_overlay_streamlines_chk.isChecked())
            self.high_perf_canvas_overlay_streamline_backend_combo.setEnabled(stream_on)
            self.high_perf_canvas_overlay_streamline_seed_spin.setEnabled(stream_on)
            self.high_perf_canvas_overlay_streamline_steps_spin.setEnabled(stream_on)
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

        snap_fp = str(getattr(self, "_snapshot_mesh_fingerprint", "") or "")
        ov_fp = str(getattr(self, "_high_perf_overlay_mesh_fingerprint", "") or "")
        if snap_fp and ov_fp and snap_fp != ov_fp:
            self._log(
                "[HighPerf Overlay] Mesh fingerprint mismatch; render skipped to avoid wrong cell-index mapping."
            )
            try:
                item.clear()
            except Exception:
                pass
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
            wse_render_mode = (
                str(getattr(self, "high_perf_canvas_overlay_wse_render_combo", None).currentData() or "cell")
                if getattr(self, "high_perf_canvas_overlay_wse_render_combo", None) is not None
                else "cell"
            )
            cmap_key = str(self.high_perf_canvas_overlay_cmap_combo.currentData() or "turbo")
            auto_contrast = bool(self.high_perf_canvas_overlay_auto_contrast_chk.isChecked())
            lock_canvas = bool(self.high_perf_canvas_overlay_lock_canvas_chk.isChecked())
            visible_only = bool(
                getattr(self, "high_perf_canvas_overlay_visible_only_chk", None) is not None
                and self.high_perf_canvas_overlay_visible_only_chk.isChecked()
            )
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
            show_arrows = bool(getattr(self, "high_perf_canvas_overlay_arrows_chk", None) is not None and self.high_perf_canvas_overlay_arrows_chk.isChecked())
            arrow_stride_px = int(round(float(getattr(self, "high_perf_canvas_overlay_arrow_density_spin", None).value()))) if getattr(self, "high_perf_canvas_overlay_arrow_density_spin", None) is not None else 28
            arrow_length_scale = float(getattr(self, "high_perf_canvas_overlay_arrow_length_spin", None).value()) if getattr(self, "high_perf_canvas_overlay_arrow_length_spin", None) is not None else 1.0
            arrow_head_length_scale = float(getattr(self, "high_perf_canvas_overlay_arrow_head_length_spin", None).value()) if getattr(self, "high_perf_canvas_overlay_arrow_head_length_spin", None) is not None else 1.0
            arrow_head_width_scale = float(getattr(self, "high_perf_canvas_overlay_arrow_head_width_spin", None).value()) if getattr(self, "high_perf_canvas_overlay_arrow_head_width_spin", None) is not None else 1.0
            show_streamlines = bool(getattr(self, "high_perf_canvas_overlay_streamlines_chk", None) is not None and self.high_perf_canvas_overlay_streamlines_chk.isChecked())
            streamline_backend = str(getattr(self, "high_perf_canvas_overlay_streamline_backend_combo", None).currentData() or "auto") if getattr(self, "high_perf_canvas_overlay_streamline_backend_combo", None) is not None else "auto"
            streamline_seed_count = int(round(float(getattr(self, "high_perf_canvas_overlay_streamline_seed_spin", None).value()))) if getattr(self, "high_perf_canvas_overlay_streamline_seed_spin", None) is not None else 48
            streamline_steps = int(round(float(getattr(self, "high_perf_canvas_overlay_streamline_steps_spin", None).value()))) if getattr(self, "high_perf_canvas_overlay_streamline_steps_spin", None) is not None else 24
            visible_extent_world = None
            canvas = self._resolve_map_canvas()
            if canvas is not None and hasattr(canvas, "extent"):
                try:
                    ex = canvas.extent()
                    visible_extent_world = (
                        float(ex.xMinimum()),
                        float(ex.xMaximum()),
                        float(ex.yMinimum()),
                        float(ex.yMaximum()),
                    )
                except Exception:
                    visible_extent_world = None

            render_extent_world = visible_extent_world if (visible_only and visible_extent_world is not None) else None
            frame = render_unstructured_snapshot_image(
                cell_x=self._high_perf_overlay_cell_x,
                cell_y=self._high_perf_overlay_cell_y,
                cell_bed=self._high_perf_overlay_cell_bed,
                node_x=self._high_perf_overlay_node_x,
                node_y=self._high_perf_overlay_node_y,
                cell_nodes=self._high_perf_overlay_cell_nodes,
                tri_to_cell=getattr(self, "_high_perf_overlay_tri_to_cell", None),
                timesteps=self._snapshot_timesteps,
                current_time_s=float(t_use),
                field_key=field_key,
                wse_render_mode=wse_render_mode,
                cmap_key=cmap_key,
                resolution=res,
                auto_contrast=auto_contrast,
                show_velocity_arrows=show_arrows,
                arrow_stride_px=arrow_stride_px,
                arrow_length_scale=arrow_length_scale,
                arrow_head_length_scale=arrow_head_length_scale,
                arrow_head_width_scale=arrow_head_width_scale,
                show_streamlines=show_streamlines,
                streamline_backend=streamline_backend,
                streamline_seed_count=streamline_seed_count,
                streamline_steps=streamline_steps,
                visible_extent_world=visible_extent_world,
                render_extent_world=render_extent_world,
                show_legend=False,
                legend_label=(
                    f"Depth ({self._length_unit_name})" if field_key == "depth"
                    else (f"Velocity ({self._length_unit_name}/s)" if field_key == "speed"
                    else f"Water Surface ({self._length_unit_name})")
                ),
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
            try:
                item.set_legend(
                    enabled=True,
                    cmap_key=cmap_key,
                    vmin=float(frame.get("vmin", 0.0)),
                    vmax=float(frame.get("vmax", 1.0)),
                    label=(
                        f"Depth ({self._length_unit_name})" if field_key == "depth"
                        else (f"Velocity ({self._length_unit_name}/s)" if field_key == "speed"
                        else f"Water Surface ({self._length_unit_name})")
                    ),
                )
            except Exception:
                pass
            try:
                face_seg = np.asarray(getattr(self, "_line_viewer_face_segments_world", np.empty((0, 4))), dtype=np.float64)
                if hasattr(item, "set_face_segments"):
                    item.set_face_segments(face_seg)
            except Exception:
                pass
            try:
                hover_pt = getattr(self, "_line_viewer_hover_point_world", None)
                hover_station = getattr(self, "_line_viewer_hover_station_m", None)
                hover_label = ""
                if hover_station is not None and np.isfinite(float(hover_station)):
                    hover_label = f"Sta {float(hover_station):.2f} {self._length_unit_name}"
                if hasattr(item, "set_station_indicator"):
                    item.set_station_indicator(hover_pt, hover_label)
            except Exception:
                pass
            iface = self._resolve_qgis_iface()
            if iface is not None and hasattr(iface, "mapCanvas"):
                try:
                    iface.mapCanvas().refresh()
                except Exception:
                    pass
        except Exception as exc:
            self._log(f"[HighPerf Overlay] refresh failed: {exc}")

    def _export_high_perf_overlay_to_geotiff(self):
        if self._high_perf_overlay_cell_x.size <= 0 or not self._snapshot_timesteps:
            QtWidgets.QMessageBox.warning(
                self,
                "Export GeoTIFF",
                "No high-perf overlay data is available. "
                "Run a model with output intervals set, then enable the overlay.",
            )
            return

        start_dir = str(self._current_line_results_storage_path() or ".")
        if start_dir and os.path.exists(os.path.dirname(start_dir)):
            start_dir = os.path.dirname(start_dir)
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export High-Perf Overlay to GeoTIFF",
            start_dir,
            "GeoTIFF (*.tif *.tiff)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith((".tif", ".tiff")):
            out_path += ".tif"

        # -- gather overlay state -------------------------------------------------
        try:
            try:
                from .swe2d_high_perf_viewer import render_unstructured_snapshot_image
            except ImportError:
                from swe2d_high_perf_viewer import render_unstructured_snapshot_image
        except Exception as exc:
            self._log(f"[GeoTIFF Export] render import failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "Export GeoTIFF", f"Could not import overlay renderer:\n{exc}")
            return

        field_key = str(getattr(self, "high_perf_canvas_overlay_field_combo", None).currentData() or "depth")
        cmap_key = str(getattr(self, "high_perf_canvas_overlay_cmap_combo", None).currentData() or "turbo")
        wse_render_mode = (
            str(getattr(self, "high_perf_canvas_overlay_wse_render_combo", None).currentData() or "cell")
            if getattr(self, "high_perf_canvas_overlay_wse_render_combo", None) is not None
            else "cell"
        )
        auto_contrast = bool(
            getattr(self, "high_perf_canvas_overlay_auto_contrast_chk", None) is not None
            and self.high_perf_canvas_overlay_auto_contrast_chk.isChecked()
        )

        t_use = None
        if self._results_panel is not None:
            try:
                t_use = float(self._results_panel.current_time_sec())
            except Exception:
                t_use = None
        if t_use is None:
            t_use = float(self._snapshot_timesteps[-1][0])

        pixel_size = float(getattr(self, "high_perf_overlay_export_res_spin", None).value() or 10.0)
        pixel_size = max(1.0e-6, abs(pixel_size))

        # -- compute extent and pixel dimensions ----------------------------------
        cx = self._high_perf_overlay_cell_x
        cy = self._high_perf_overlay_cell_y
        x_min = float(np.nanmin(cx))
        x_max = float(np.nanmax(cx))
        y_min = float(np.nanmin(cy))
        y_max = float(np.nanmax(cy))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = 0.0, 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = 0.0, 1.0

        nx = max(32, int(np.ceil((x_max - x_min) / pixel_size)))
        ny = max(32, int(np.ceil((y_max - y_min) / pixel_size)))

        # -- render ---------------------------------------------------------------
        field_labels = {
            "depth": f"Depth ({self._length_unit_name})",
            "speed": f"Velocity ({self._length_unit_name}/s)",
            "wse": f"Water Surface ({self._length_unit_name})",
        }
        legend_label = field_labels.get(field_key, str(field_key))
        try:
            frame = render_unstructured_snapshot_image(
                cell_x=cx,
                cell_y=cy,
                cell_bed=self._high_perf_overlay_cell_bed,
                node_x=self._high_perf_overlay_node_x,
                node_y=self._high_perf_overlay_node_y,
                cell_nodes=self._high_perf_overlay_cell_nodes,
                tri_to_cell=getattr(self, "_high_perf_overlay_tri_to_cell", None),
                timesteps=self._snapshot_timesteps,
                current_time_s=float(t_use),
                field_key=field_key,
                wse_render_mode=wse_render_mode,
                cmap_key=cmap_key,
                resolution=(nx, ny),
                auto_contrast=auto_contrast,
                show_velocity_arrows=False,
                show_streamlines=False,
                render_extent_world=(x_min, x_max, y_min, y_max),
                show_legend=True,
                legend_label=legend_label,
            )
        except Exception as exc:
            self._log(f"[GeoTIFF Export] render error: {exc}")
            QtWidgets.QMessageBox.critical(self, "Export GeoTIFF", f"Overlay render failed:\n{exc}")
            return

        if not bool(frame.get("ok", False)):
            msg = str(frame.get("message", "unknown render error"))
            self._log(f"[GeoTIFF Export] empty frame: {msg}")
            QtWidgets.QMessageBox.warning(self, "Export GeoTIFF", f"Nothing rendered:\n{msg}")
            return

        # -- extract the raw scalar grid (data values, not RGB) ------------------
        scalar_grid = frame.get("grid")
        grid_mask = frame.get("grid_mask")
        if scalar_grid is None or grid_mask is None:
            self._log("[GeoTIFF Export] renderer did not return a scalar grid.")
            QtWidgets.QMessageBox.warning(
                self, "Export GeoTIFF",
                "The overlay renderer did not expose raw data values.\n"
                "A code update is required in swe2d_high_perf_viewer.",
            )
            return

        h_img, w = scalar_grid.shape

        # Mask out non-finite / out-of-mesh pixels → NaN
        grid_out = np.full((h_img, w), np.nan, dtype=np.float64)
        grid_out[grid_mask] = scalar_grid[grid_mask]

        # -- write via GDAL -------------------------------------------------------
        try:
            from osgeo import gdal, osr
        except ImportError:
            gdal = None
            osr = None

        crs_auth = "EPSG:4326"
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                proj_crs = QgsProject.instance().crs()
                if proj_crs is not None and proj_crs.isValid():
                    crs_auth = proj_crs.authid() or crs_auth
            except Exception:
                pass

        if gdal is not None:
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(out_path, w, h_img, 1, gdal.GDT_Float64)
            if ds is None:
                raise RuntimeError("GDAL could not create output dataset.")
            x_res = (x_max - x_min) / max(1, w)
            y_res = (y_max - y_min) / max(1, h_img)
            gt = (x_min, x_res, 0.0, y_max, 0.0, -y_res)
            ds.SetGeoTransform(gt)

            srs = osr.SpatialReference()
            srs.SetFromUserInput(crs_auth)
            ds.SetProjection(srs.ExportToWkt())

            band = ds.GetRasterBand(1)
            band.WriteArray(grid_out)
            band.SetNoDataValue(np.nan)
            band.SetDescription(str(field_key))
            ds.FlushCache()
            ds = None
        else:
            # Fallback: use PIL to write a single-band float TIFF (no CRS)
            try:
                from PIL import Image as _PILImage
            except ImportError:
                _PILImage = None
            if _PILImage is not None:
                # PIL can't write float TIFF natively → use GDAL-style temp
                raise RuntimeError("GDAL required for scalar GeoTIFF export.")
            else:
                raise RuntimeError("GDAL is not available. Cannot write GeoTIFF.")

        vmin = float(np.nanmin(grid_out))
        vmax = float(np.nanmax(grid_out))
        self._log(
            f"High-perf overlay exported to GeoTIFF: {out_path} "
            f"({w}x{h_img}, CRS={crs_auth}, field={field_key}, t={t_use / 3600.0:.3f} hr, "
            f"range=[{vmin:.6g}, {vmax:.6g}])"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Export GeoTIFF",
            f"Exported {w}x{h_img} single-band Float64 to:\n{out_path}\n"
            f"CRS: {crs_auth}\n"
            f"Field: {field_key}  Time: {t_use / 3600.0:.3f} hr\n"
            f"Pixel size: {pixel_size:.4f} map units\n"
            f"Value range: [{vmin:.6g}, {vmax:.6g}] {self._length_unit_name}",
        )

    def _on_line_viewer_selection_changed(self, line_id: Optional[int]) -> None:
        try:
            if not hasattr(self, "_line_flux_face_segments_by_line") or not getattr(self, "_line_flux_face_segments_by_line", {}):
                self._build_line_sampling_map()
            seg_map = getattr(self, "_line_flux_face_segments_by_line", {}) or {}
            lid = int(line_id) if line_id is not None else None
            if lid is None:
                self._line_viewer_face_segments_world = np.empty((0, 4), dtype=np.float64)
            else:
                self._line_viewer_face_segments_world = np.asarray(seg_map.get(lid, np.empty((0, 4))), dtype=np.float64)
        except Exception:
            self._line_viewer_face_segments_world = np.empty((0, 4), dtype=np.float64)
        self._refresh_high_perf_canvas_overlay(None)

    def _sample_line_world_point_at_station(self, line_id: int, station_m: float) -> Optional[Tuple[float, float]]:
        if not _HAVE_QGIS_CORE or not hasattr(self, "sample_lines_layer_combo"):
            return None
        line_layer = self._combo_layer(self.sample_lines_layer_combo, "vector")
        if line_layer is None:
            return None
        fields = set(line_layer.fields().names())
        id_field = "line_id" if "line_id" in fields else None
        if id_field is None:
            return None
        for ft in line_layer.getFeatures():
            try:
                if int(ft[id_field]) != int(line_id):
                    continue
            except Exception:
                continue
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                return None
            line_len = float(geom.length())
            if line_len <= 0.0:
                return None
            try:
                p0 = geom.interpolate(0.0).asPoint()
                p1 = geom.interpolate(max(0.0, line_len - 1.0e-9)).asPoint()
                start_key = (float(p0.x()), float(p0.y()))
                end_key = (float(p1.x()), float(p1.y()))
                orient_sign = 1.0 if end_key >= start_key else -1.0
            except Exception:
                orient_sign = 1.0
            s = float(station_m)
            s = max(0.0, min(float(line_len), s))
            raw_s = (float(line_len) - s) if orient_sign < 0.0 else s
            try:
                p = geom.interpolate(float(raw_s)).asPoint()
                return (float(p.x()), float(p.y()))
            except Exception:
                return None
        return None

    def _on_line_viewer_hover_station(self, line_id: Optional[int], station_m: Optional[float]) -> None:
        self._line_viewer_hover_station_m = None
        self._line_viewer_hover_point_world = None
        try:
            if line_id is None or station_m is None:
                self._refresh_high_perf_canvas_overlay(None)
                return
            s = float(station_m)
            if not np.isfinite(s):
                self._refresh_high_perf_canvas_overlay(None)
                return
            pt = self._sample_line_world_point_at_station(int(line_id), s)
            if pt is None:
                self._refresh_high_perf_canvas_overlay(None)
                return
            self._line_viewer_hover_station_m = float(s)
            self._line_viewer_hover_point_world = (float(pt[0]), float(pt[1]))
        except Exception:
            self._line_viewer_hover_station_m = None
            self._line_viewer_hover_point_world = None
        self._refresh_high_perf_canvas_overlay(None)

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
            cell_bed = np.asarray(self._mesh_cell_solver_bed(), dtype=np.float64).ravel()
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
                "Panel import or initialization failed.\n"
                "Check the plugin log for '[Results Panel]' details."
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
        if panel is not None and panel.velocity_overlay_enabled():
            self._auto_add_velocity_overlay_sources_from_panel()
        t_s = panel.current_time_sec() if panel is not None else 0.0
        self._refresh_results_map_overlays(float(t_s))

    def _auto_add_velocity_overlay_sources_from_panel(self) -> int:
        panel = getattr(self, "_results_panel", None)
        if panel is None or not hasattr(panel, "enabled_overlay_targets"):
            return 0
        if self._velocity_overlay_sources:
            return 0

        added = 0
        for gpkg_path, run_id in panel.enabled_overlay_targets():
            gpkg_path = str(gpkg_path or "").strip()
            run_id = str(run_id or "").strip()
            if not gpkg_path or not run_id or not os.path.exists(gpkg_path):
                continue

            table_choices = self._list_velocity_candidate_tables(gpkg_path)
            if not table_choices:
                continue

            ordered_tables: List[str] = []
            if "swe2d_mesh_results" in table_choices:
                ordered_tables.append("swe2d_mesh_results")
            for name in table_choices:
                if name not in ordered_tables:
                    ordered_tables.append(name)

            chosen_table = ""
            for table_name in ordered_tables:
                table_run_ids = self._run_ids_for_velocity_table(gpkg_path, table_name)
                if run_id in table_run_ids:
                    chosen_table = table_name
                    break
            if not chosen_table:
                continue

            source_key = f"{gpkg_path}::{chosen_table}::{run_id}"
            if any(str(rec.get("key", "")) == source_key for rec in self._velocity_overlay_sources):
                continue

            self._velocity_overlay_sources.append(
                {
                    "key": source_key,
                    "gpkg_path": gpkg_path,
                    "table_name": chosen_table,
                    "run_id": run_id,
                    "label": f"{os.path.basename(gpkg_path)}:{chosen_table}:{run_id}",
                }
            )
            added += 1

        if added:
            self._log(
                f"Velocity arrows auto-added {added} source(s) from enabled results-panel runs."
            )
        return added

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
        from swe2d.workbench.monolith_methods import _mesh_cell_centers_for_gpkg as _logic
        return _logic(self, gpkg_path, run_id, table_name)

    def _refresh_velocity_vectors_overlay(self, t_s: float):
        from swe2d.workbench.monolith_methods import _refresh_velocity_vectors_overlay as _logic
        return _logic(self, t_s)

    def _refresh_streamline_traces_overlay(self, t_s: float):
        from swe2d.workbench.monolith_methods import _refresh_streamline_traces_overlay as _logic
        return _logic(self, t_s)

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
        try:
            self._build_line_sampling_map()
        except Exception:
            pass
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

    def _open_model_gpkg_explorer(self):
        db_path = ""
        if self._model_gpkg_path and os.path.exists(self._model_gpkg_path):
            db_path = self._model_gpkg_path
        if not db_path and self._line_results_latest_db_path and os.path.exists(self._line_results_latest_db_path):
            db_path = self._line_results_latest_db_path
        if not db_path:
            db_path = self._current_line_results_storage_path()
        if not db_path or not os.path.exists(db_path):
            self._log("No model GeoPackage available for explorer.")
            QtWidgets.QMessageBox.warning(
                self,
                "Model GeoPackage Explorer",
                "No model GeoPackage was found. Load or create a model GeoPackage first.",
            )
            return

        dlg = SWE2DModelGeoPackageExplorerDialog(
            gpkg_path=db_path,
            open_run_log_viewer=self._open_run_log_viewer,
            open_line_results_viewer=self._open_line_results_viewer,
            open_coupling_results_viewer=self._open_coupling_results_viewer,
            logger=self._log,
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
                from swe2d.extensions.extension_models import StructureType
                from swe2d import units as _u

                hh = np.ascontiguousarray(h, dtype=np.float64).ravel()
                cell_wse = hh + np.asarray(coupling_controller.cell_bed, dtype=np.float64).ravel()
                details = list(structures_mod.structure_details(cell_wse))
                # Use the native path flows (which actually drove the simulation)
                # for the 'flow' metric, falling back to Python path if unavailable.
                native_flows = getattr(coupling_controller, "_last_native_structure_flows", None)

                # UI unit helpers: convert internal values → model units for display.
                # Culvert routines operate in USC (ft) and return flows in SI (m³/s = CMS).
                # Non-culvert kernel routines operate in USC and return CFS (ft³/s).
                # Python structure_details() also returns culvert flows in CMS
                # but non-culvert flows in model³/s.
                # We normalise everything to model³/s so the unit label matches the value.
                is_usc = self._is_us_customary_units()

                def _cms_to_model(v: float) -> float:
                    """Convert SI m³/s → model³/s (ft³/s for USC, m³/s for SI)."""
                    return float(self._flow_si_to_model(v))

                def _cfs_to_model(v: float) -> float:
                    """Convert USC ft³/s → model³/s."""
                    if is_usc:
                        return v  # already ft³/s = model³/s
                    # SI model: CFS → m³/s
                    return v / _u.USC_FT3_PER_SI_M3

                for i, st in enumerate(structures_mod.cfg.structures):
                    sid = str(st.structure_id)
                    detail = details[i] if i < len(details) else {}
                    stype = st.structure_type if hasattr(st, "structure_type") else StructureType.WEIR
                    base_name = str(stype.name).lower() if isinstance(stype, StructureType) else str(stype).lower()
                    is_culvert = (stype == StructureType.CULVERT)

                    # Native path flow: CMS for culverts, CFS for non-culvert types.
                    # Python fallback: CMS for culverts, model³/s for non-culvert.
                    if native_flows is not None and i < len(native_flows) and np.isfinite(native_flows[i]):
                        native_val = float(native_flows[i])
                        if is_culvert:
                            flow_val = _cms_to_model(native_val)
                        else:
                            flow_val = _cfs_to_model(native_val)
                    else:
                        flow_py = float(detail.get("flow", 0.0) or 0.0)
                        if is_culvert:
                            flow_val = _cms_to_model(flow_py)
                        else:
                            # Non-culvert Python flows are already in model³/s.
                            flow_val = flow_py

                    metric_map: Dict[str, float] = {
                        "flow": flow_val,
                    }
                    if is_culvert:
                        metric_map.update(
                            {
                                "inlet_control_flow": _cms_to_model(float(detail.get("inlet_control_flow", 0.0) or 0.0)),
                                "outlet_control_flow": _cms_to_model(float(detail.get("outlet_control_flow", 0.0) or 0.0)),
                                "orifice_cap": _cms_to_model(float(detail.get("orifice_cap", 0.0) or 0.0)),
                                "manning_cap": _cms_to_model(float(detail.get("manning_cap", 0.0) or 0.0)),
                                "embankment_flow": _cms_to_model(float(detail.get("embankment_flow", 0.0) or 0.0)),
                                # available_head_up and tailwater_depth are already in model units.
                                "available_head_up": float(detail.get("available_head_up", 0.0) or 0.0),
                                "tailwater_depth": float(detail.get("tailwater_depth", 0.0) or 0.0),
                            }
                        )
                    for metric_name, metric_value in metric_map.items():
                        rows.append(
                            {
                                "t_s": float(t_s),
                                "component": "structure",
                                "object_id": sid,
                                "object_name": f"{base_name}:{str(detail.get('control_mode', 'none') or 'none')}",
                                "metric": metric_name,
                                "value": float(metric_value),
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
        runs_table = self._results_table_name("swe2d_coupling_results_runs")
        data_table = self._results_table_name("swe2d_coupling_results")

        def _q(name: str) -> str:
            return '"' + str(name).replace('"', '""') + '"'

        q_runs = _q(runs_table)
        q_data = _q(data_table)
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {q_runs} (
                    run_id TEXT PRIMARY KEY,
                    created_utc TEXT,
                    interval_s REAL,
                    row_count INTEGER
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {q_data} (
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
                f"CREATE INDEX IF NOT EXISTS idx_{data_table}_run_component_metric_obj_t "
                f"ON {q_data}(run_id, component, metric, object_id, t_s)"
            )
            cur.execute(f"DELETE FROM {q_data} WHERE run_id = ?", (run_id,))
            cur.execute(
                f"""
                INSERT OR REPLACE INTO {q_runs}
                (run_id, created_utc, interval_s, row_count)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
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
                f"""
                INSERT OR REPLACE INTO {q_data}
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
        from swe2d.workbench.extracted.results_export_methods import load_coupling_results_from_geopackage

        return load_coupling_results_from_geopackage(self, gpkg_path, run_id)

    def _open_coupling_results_viewer(self):
        from swe2d.workbench.extracted.results_export_methods import open_coupling_results_viewer

        return open_coupling_results_viewer(self)

    def _persist_mesh_results_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        mesh_rows: List[Dict[str, object]],
        interval_s: float,
        table_name: str = "swe2d_mesh_results",
    ) -> None:
        from swe2d.workbench.extracted.results_export_methods import persist_mesh_results_to_geopackage

        return persist_mesh_results_to_geopackage(self, gpkg_path, run_id, mesh_rows, interval_s, table_name=table_name)

    def _persist_conservation_forensics_to_geopackage(
        self,
        gpkg_path: str,
        run_id: str,
        storage_rows: List[Dict[str, object]],
        boundary_rows: List[Dict[str, object]],
        summary: Dict[str, object],
        source_step_rows: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        from swe2d.workbench.extracted.results_export_methods import persist_conservation_forensics_to_geopackage

        return persist_conservation_forensics_to_geopackage(
            self,
            gpkg_path,
            run_id,
            storage_rows,
            boundary_rows,
            summary,
            source_step_rows=source_step_rows,
        )

    def _build_mesh_snapshot_rows(self) -> List[Dict[str, object]]:
        return _build_mesh_snapshot_rows_logic(self._snapshot_timesteps)

    def _collect_run_log_metadata(self) -> Dict[str, object]:
        from swe2d.workbench.extracted.results_export_methods import collect_run_log_metadata

        return collect_run_log_metadata(self)

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
            table_prefix=self._selected_results_table_prefix(),
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
        return _load_run_logs_from_geopackage_logic(
            gpkg_path=gpkg_path,
            table_prefix=self._selected_results_table_prefix(),
        )

    def _open_run_log_viewer(self):
        from swe2d.workbench.extracted.results_export_methods import open_run_log_viewer

        return open_run_log_viewer(self)

    def _open_3d_patch_viewer(self):
        snaps = list(getattr(self, "_three_d_patch_snapshots", []) or [])
        if not snaps:
            self._log(
                "No 3D patch snapshots available yet. "
                "Run with Experimental 3D patch mode enabled and capture mesh snapshots. "
                "Note: 'Snapshot written -> ... .hdf' is a 2D/mesh export and is not the source used by the 3D patch viewer."
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
        from swe2d.workbench.monolith_methods import _build_pipe_network_config as _logic
        return _logic(self)

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
        # ── Schema validation: warn if the layer is missing essential fields ──
        _essential_fields = {"structure_type", "crest_elev"}
        _missing_essential = _essential_fields - fields
        if _missing_essential:
            self._log(
                f"Structures layer '{layer.name()}' is missing required fields: "
                f"{', '.join(sorted(_missing_essential))}. "
                f"Coupling will be disabled. Select a valid structures layer."
            )
            return None
        _recommended_culvert_fields = {
            "culvert_code", "culvert_shape", "inlet_invert_elev",
            "outlet_invert_elev", "length", "roughness_n", "culvert_slope",
        }
        _missing_recommended = _recommended_culvert_fields - fields
        if _missing_recommended:
            self._log(
                f"Structures layer '{layer.name()}' missing culvert fields: "
                f"{', '.join(sorted(_missing_recommended))}. "
                f"Defaults will be used; results may be inaccurate."
            )
        # ── End schema validation ──
        structures: List[HydraulicStructure] = []
        type_name_map = {
            "weir": StructureType.WEIR,
            "culvert": StructureType.CULVERT,
            "gate": StructureType.GATE,
            "bridge": StructureType.BRIDGE,
            "pump": StructureType.PUMP,
        }
        stacked_required_fields = (
            "influence_width_m",
            "deck_soffit_elev",
            "deck_top_elev",
            "model_top_elev",
            "under_layers",
            "over_layers",
        )
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
            for key in (
                "width",
                "height",
                "diameter",
                "culvert_shape",
                "culvert_code",
                "culvert_rise",
                "culvert_span",
                "culvert_area_m2",
                "culvert_barrels",
                "culvert_slope",
                "inlet_invert_elev",
                "outlet_invert_elev",
                "entrance_loss_k",
                "exit_loss_k",
                "embankment_enabled",
                "embankment_crest_elev",
                "embankment_overflow_width",
                "embankment_weir_coeff",
                "length",
                "roughness_n",
                "coeff",
                "cd",
                "opening",
                "q_pump",
                "max_flow",
                "inlet_loss_k",
                "outlet_loss_k",
                "stacked_enabled",
                "influence_width_m",
                "upstream_buffer_m",
                "downstream_buffer_m",
                "deck_soffit_elev",
                "deck_top_elev",
                "model_top_elev",
                "under_layers",
                "over_layers",
                "pier_count",
                "pier_width",
            ):
                if key in fields and ft[key] not in (None, ""):
                    if key in ("culvert_shape",):
                        metadata[key] = str(ft[key])
                    else:
                        try:
                            metadata[key] = float(ft[key])
                        except Exception:
                            pass
            metadata["axis_x0"] = float(p0.x())
            metadata["axis_y0"] = float(p0.y())
            metadata["axis_x1"] = float(p1.x())
            metadata["axis_y1"] = float(p1.y())
            structure_id = str(ft["structure_id"] if "structure_id" in fields else ft.id()).strip()

            if structure_type == StructureType.BRIDGE and int(metadata.get("stacked_enabled", 0)) > 0:
                missing_schema = [k for k in stacked_required_fields if k not in fields]
                missing_values = [k for k in stacked_required_fields if k in fields and k not in metadata]
                if missing_schema or missing_values:
                    missing_all = ", ".join(missing_schema + missing_values)
                    self._log(
                        f"Bridge {structure_id}: stacked geometry disabled due to missing fields: {missing_all}"
                    )
                    metadata["stacked_enabled"] = 0.0
                else:
                    influence_width_m = float(metadata.get("influence_width_m", 0.0))
                    under_layers = int(max(1, round(float(metadata.get("under_layers", 1.0)))))
                    over_layers = int(max(1, round(float(metadata.get("over_layers", 1.0)))))
                    deck_soffit_elev = float(metadata.get("deck_soffit_elev", 0.0))
                    deck_top_elev = float(metadata.get("deck_top_elev", deck_soffit_elev + 0.1))
                    model_top_elev = float(metadata.get("model_top_elev", deck_top_elev + 0.1))
                    valid_stacked = (
                        influence_width_m > 0.0
                        and deck_top_elev > deck_soffit_elev
                        and model_top_elev > deck_top_elev
                    )
                    if not valid_stacked:
                        self._log(
                            f"Bridge {structure_id}: stacked geometry disabled due to invalid elevation/width values."
                        )
                        metadata["stacked_enabled"] = 0.0
                    else:
                        metadata["under_layers"] = float(under_layers)
                        metadata["over_layers"] = float(over_layers)
            structures.append(
                HydraulicStructure(
                    structure_id=structure_id,
                    structure_type=structure_type,
                    upstream_cell=self._nearest_cell_index_for_xy(float(p0.x()), float(p0.y())),
                    downstream_cell=self._nearest_cell_index_for_xy(float(p1.x()), float(p1.y())),
                    crest_elev=float(ft["crest_elev"] if "crest_elev" in fields and ft["crest_elev"] not in (None, "") else 0.0),
                    enabled=True,
                    metadata=metadata,
                )
            )
        if not structures:
            feature_count = layer.featureCount() if hasattr(layer, "featureCount") else -1
            if feature_count > 0:
                self._log(
                    f"Structures layer '{layer.name()}' has {feature_count} features "
                    f"but produced 0 valid structures. Check that geometries are valid "
                    f"and that 'enabled' is not 0."
                )
            return None
        from swe2d import units as _u
        gravity = float(getattr(self, "_gravity", _u.gravity()))
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
        return read_project_entry_text(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_project_cls=QgsProject,
            key=key,
            default=default,
        )

    def _persist_project_layer_bindings(self, *_args: object) -> None:
        """Persist current layer-combo selections into the QGIS project."""
        if self._project_layer_state_blocked or not _HAVE_QGIS_CORE or QgsProject is None:
            return
        if bool(getattr(self, "_initial_layer_restore_pending", False)):
            return
        payload = build_layer_selector_state(self._project_layer_binding_specs())
        write_project_json(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_project_cls=QgsProject,
            key=LAYER_SELECTOR_STATE_KEY,
            payload=payload,
        )

    def _restore_project_layer_bindings(self) -> None:
        """Restore saved layer-combo selections from the QGIS project."""
        if self._project_layer_state_blocked or not _HAVE_QGIS_CORE or QgsProject is None:
            return
        selectors = parse_layer_selector_state(
            load_project_json(
                have_qgis_core=_HAVE_QGIS_CORE,
                qgs_project_cls=QgsProject,
                key=LAYER_SELECTOR_STATE_KEY,
                default={},
            )
        )
        if not selectors:
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
        from swe2d.workbench.monolith_methods import _connect_project_workbench_state_signals as _logic
        return _logic(self)

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
        project under the centralized workbench-state key.
        """
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return

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
            "experimental_3d_projection_residual_sample_iters_spin",
            "experimental_3d_projection_divergence_gate_enable_chk",
            "experimental_3d_projection_divergence_ratio_target_spin",
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
            "degen_mode_combo", "solver_backend_combo", "solver_openmp_enabled_chk", "solver_cpu_threads_spin", "coupling_loop_combo",
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
            "topo_gmsh_verbosity_spin", "topo_gmsh_num_threads_spin",
            "topo_gmsh_max_num_threads_2d_spin", "topo_gmsh_global_recombine_chk",
            "topo_gmsh_interface_conformance_chk", "topo_gmsh_transverse_interface_centroid_merge_chk",
            "topo_gmsh_interface_snap_tol_spin", "topo_gmsh_interface_reject_near_unshared_chk",
            "topo_gmsh_interface_reject_tol_spin",
        ]
        widget_attrs.extend(list(getattr(self, "_experimental_3d_bc_widget_attrs", []) or []))

        # Include any additional supported widgets bound on the dialog so newly
        # added controls persist without requiring manual list maintenance.
        try:
            persistable_classes = (
                getattr(QtWidgets, "QSpinBox"),
                getattr(QtWidgets, "QDoubleSpinBox"),
                getattr(QtWidgets, "QComboBox"),
                getattr(QtWidgets, "QCheckBox"),
                getattr(QtWidgets, "QLineEdit"),
            )
            known_attrs = set(widget_attrs)
            for attr_name, widget in vars(self).items():
                if attr_name in known_attrs or attr_name.startswith("_"):
                    continue
                if isinstance(widget, persistable_classes):
                    widget_attrs.append(attr_name)
                    known_attrs.add(attr_name)
        except Exception:
            pass

        payload = collect_workbench_widget_state(
            ui=self,
            widget_attrs=widget_attrs,
            qtwidgets_module=QtWidgets,
        )
        if write_project_json(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_project_cls=QgsProject,
            key=WORKBENCH_STATE_KEY,
            payload=payload,
            log_callback=self._log,
        ):
            self._log(f"[DEBUG] persist: saved {len(payload['widgets'])} widgets to project")

    def _restore_project_workbench_state(self, *_args: object) -> None:
        """Restore persisted workbench widget values from QGIS project state."""
        if not _HAVE_QGIS_CORE or QgsProject is None:
            self._log("[DEBUG] restore: QGIS core not available")
            return

        payload = load_project_json(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_project_cls=QgsProject,
            key=WORKBENCH_STATE_KEY,
            default=None,
            log_callback=self._log,
        )
        if payload is None:
            self._log("[DEBUG] restore: no saved workbench state found")
            return

        widgets_data = payload.get("widgets", {}) if isinstance(payload, dict) else {}
        self._log(f"[DEBUG] restore: restoring {len(widgets_data)} widget values")

        restored_count = restore_workbench_widget_state(
            ui=self,
            widgets_data=widgets_data,
            qtwidgets_module=QtWidgets,
            log_callback=self._log,
        )

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

            now = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
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
        from swe2d.workbench.monolith_methods import _preview_coupling_configuration as _logic
        return _logic(self)

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
        from swe2d.workbench.monolith_methods import _write_hecras_hdf5 as _logic
        return _logic(self, path, timesteps)

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
        from swe2d.workbench.monolith_methods import _write_ugrid_nc as _logic
        return _logic(self, path, timesteps)

    def _export_mesh_to_hdf5(self):
        from swe2d.workbench.extracted.results_export_methods import export_mesh_to_hdf5

        return export_mesh_to_hdf5(self)

    def _export_results_to_hdf5(self):
        from swe2d.workbench.extracted.results_export_methods import export_results_to_hdf5

        return export_results_to_hdf5(self)

    def _export_results_to_ugrid(self):
        from swe2d.workbench.extracted.results_export_methods import export_results_to_ugrid

        return export_results_to_ugrid(self)

    def _on_snapshot(self):
        """Write captured 2D mesh timesteps to a temporary HEC-RAS HDF file.

        This export is for mesh/results interoperability. The experimental 3D
        patch viewer reads in-memory 3D VoF snapshots from
        `_three_d_patch_snapshots`, not this HDF file.
        """
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
            self._log(
                "Snapshot export note: this .hdf contains 2D mesh results. "
                "3D Patch Viewer uses in-memory 3D patch snapshots captured during run."
            )

            gpkg_results_path = self._current_line_results_storage_path()
            if gpkg_results_path:
                snap_run_id = datetime.datetime.now().astimezone().strftime("swe2d_snapshot_%Y%m%dT%H%M%S%z")
                mesh_interval_s = max(1.0, self._parse_time_hours(self.output_interval_edit.text()) * 3600.0)
                line_interval_s = max(1.0, self._parse_time_hours(self.line_output_interval_edit.text()) * 3600.0)

                if bool(getattr(self, "save_mesh_results_to_gpkg_chk", None) is None or self.save_mesh_results_to_gpkg_chk.isChecked()):
                    mesh_rows = self._build_mesh_snapshot_rows()
                    if mesh_rows:
                        mesh_table_name = "swe2d_mesh_results"
                        if hasattr(self, "_selected_mesh_results_table_name"):
                            try:
                                mesh_table_name = str(self._selected_mesh_results_table_name() or "swe2d_mesh_results")
                            except Exception:
                                mesh_table_name = "swe2d_mesh_results"
                        self._persist_mesh_results_to_geopackage(
                            gpkg_results_path,
                            snap_run_id,
                            mesh_rows,
                            interval_s=mesh_interval_s,
                            table_name=mesh_table_name,
                        )
                        self._log(
                            f"Mesh snapshot stored → {gpkg_results_path} "
                            f"(table={mesh_table_name}, rows={len(mesh_rows)}, run_id={snap_run_id})"
                        )

                if self._line_snapshot_rows:
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
        edge_groups: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        progressive = True
        if hasattr(self, "inflow_progressive_chk") and self.inflow_progressive_chk is not None:
            try:
                progressive = bool(self.inflow_progressive_chk.isChecked())
            except Exception:
                progressive = True

        if edge_groups is None:
            if hasattr(self, "_cached_edge_groups") and self._cached_edge_groups is not None:
                edge_groups = self._cached_edge_groups
            elif hasattr(self, "_collect_bc_layer_edge_groups"):
                try:
                    edge_groups = self._collect_bc_layer_edge_groups(edge_n0, edge_n1)
                    self._cached_edge_groups = edge_groups
                except Exception:
                    edge_groups = None

        # Cache mesh-constant BC geometry after first call.
        bc_cache = self.__dict__.setdefault("_bc_geom_cache", {})
        if "_side_idx" not in bc_cache or "_edge_len" not in bc_cache or "_edge_z" not in bc_cache:
            side_idx, _mx, _my, _xmin, _xmax, _ymin, _ymax = _bc_side_classification(
                edge_n0, edge_n1,
                self._mesh_data["node_x"],
                self._mesh_data["node_y"],
            )
            bc_cache["_side_idx"] = side_idx
            bc_cache["_edge_len"] = np.hypot(
                self._mesh_data["node_x"][edge_n1] - self._mesh_data["node_x"][edge_n0],
                self._mesh_data["node_y"][edge_n1] - self._mesh_data["node_y"][edge_n0],
            )
            bc_cache["_edge_z"] = 0.5 * (
                self._mesh_data["node_z"][edge_n0] + self._mesh_data["node_z"][edge_n1]
            )

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
            edge_groups=edge_groups,
            _side_idx=bc_cache.get("_side_idx"),
            _edge_len=bc_cache.get("_edge_len"),
            _edge_z=bc_cache.get("_edge_z"),
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
        # Cache mesh-constant BC geometry after first call.
        bc_cache = self.__dict__.setdefault("_bc_geom_cache", {})
        if "_side_idx" not in bc_cache:
            side_idx, _mx, _my, _xmin, _xmax, _ymin, _ymax = _bc_side_classification(
                edge_n0, edge_n1,
                self._mesh_data["node_x"],
                self._mesh_data["node_y"],
            )
            bc_cache["_side_idx"] = side_idx
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
            _side_idx=bc_cache["_side_idx"],
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

    def _append_3d_patch_snapshot(
        self,
        t_s: float,
        stats: Dict[str, object],
        vof: np.ndarray,
        u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(stats, dict):
            return
        arr = np.asarray(vof, dtype=np.float64).ravel()
        nx = max(0, int(stats.get("nx", 0) or 0))
        ny = max(0, int(stats.get("ny", 0) or 0))
        nz = max(0, int(stats.get("nz", 0) or 0))
        expected = nx * ny * nz
        if expected <= 0 or arr.size != expected:
            return
        u_arr = np.asarray(u, dtype=np.float64).ravel() if u is not None else np.empty(0, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64).ravel() if v is not None else np.empty(0, dtype=np.float64)
        w_arr = np.asarray(w, dtype=np.float64).ravel() if w is not None else np.empty(0, dtype=np.float64)
        if u_arr.size != expected:
            u_arr = np.empty(0, dtype=np.float64)
        if v_arr.size != expected:
            v_arr = np.empty(0, dtype=np.float64)
        if w_arr.size != expected:
            w_arr = np.empty(0, dtype=np.float64)
        snap_spec = None
        if isinstance(self._three_d_patch_last_spec, dict):
            snap_spec = dict(self._three_d_patch_last_spec)

        bed_flat = np.empty(0, dtype=np.float64)
        if isinstance(snap_spec, dict):
            try:
                nx_s = int(snap_spec.get("nx", 0) or 0)
                ny_s = int(snap_spec.get("ny", 0) or 0)
                dx_s = float(snap_spec.get("dx", 0.0) or 0.0)
                dy_s = float(snap_spec.get("dy", 0.0) or 0.0)
                ox_s = float(snap_spec.get("origin_x", 0.0) or 0.0)
                oy_s = float(snap_spec.get("origin_y", 0.0) or 0.0)
                oz_s = float(snap_spec.get("origin_z", 0.0) or 0.0)
                if nx_s > 0 and ny_s > 0 and dx_s > 0.0 and dy_s > 0.0:
                    class _PatchSpecTmp:
                        pass

                    spec_obj = _PatchSpecTmp()
                    spec_obj.nx = nx_s
                    spec_obj.ny = ny_s
                    spec_obj.dx = dx_s
                    spec_obj.dy = dy_s
                    spec_obj.origin_x = ox_s
                    spec_obj.origin_y = oy_s

                    terrain_surface = self._build_patch_terrain_surface(spec_obj)
                    if terrain_surface is not None:
                        bed_arr = np.where(np.isfinite(terrain_surface), terrain_surface, float(oz_s)).astype(np.float64)
                    else:
                        bed_arr = np.full((ny_s, nx_s), float(oz_s), dtype=np.float64)
                    bed_flat = bed_arr.ravel(order="C")
            except Exception:
                bed_flat = np.empty(0, dtype=np.float64)

        stats_rec = dict(stats)
        if "origin_z" not in stats_rec and isinstance(snap_spec, dict) and "origin_z" in snap_spec:
            stats_rec["origin_z"] = float(snap_spec.get("origin_z", 0.0) or 0.0)

        self._three_d_patch_snapshots.append(
            {
                "t_s": float(t_s),
                "stats": stats_rec,
                "vof": arr.copy(),
                "u": u_arr.copy(),
                "v": v_arr.copy(),
                "w": w_arr.copy(),
                "patch_spec": snap_spec,
                "bed_z": bed_flat.copy(),
            }
        )
        max_keep = 48
        if len(self._three_d_patch_snapshots) > max_keep:
            self._three_d_patch_snapshots = self._three_d_patch_snapshots[-max_keep:]

    def _on_cancel(self):
        self._cancel_requested = True
        self._log("Cancellation requested...")

    def _on_select_results_gpkg(self) -> None:
        default_path = str(self._current_line_results_storage_path() or "")
        if not os.path.exists(default_path):
            default_path = str(self._model_gpkg_path or "")
        gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Existing Results GeoPackage",
            default_path,
            "GeoPackage (*.gpkg)",
        )
        if not gpkg_path:
            return
        if not os.path.exists(gpkg_path):
            QtWidgets.QMessageBox.warning(
                self,
                "Results GeoPackage",
                "Please select an existing GeoPackage file.",
            )
            return
        if hasattr(self, "results_gpkg_path_edit") and self.results_gpkg_path_edit is not None:
            self.results_gpkg_path_edit.setText(str(gpkg_path))
        self._log(f"Results GeoPackage override set: {gpkg_path}")

    def _selected_results_table_prefix(self) -> str:
        raw = ""
        if hasattr(self, "results_table_name_edit") and self.results_table_name_edit is not None:
            raw = str(self.results_table_name_edit.text() or "").strip()
        if not raw:
            return ""
        cleaned_chars = []
        for ch in raw:
            if ch.isalnum() or ch == "_":
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append("_")
        cleaned = "".join(cleaned_chars).strip("_")
        if not cleaned:
            return ""
        if not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = f"p_{cleaned}"
        return cleaned

    def _results_table_name(self, base_name: str) -> str:
        base = str(base_name or "").strip() or "swe2d_mesh_results"
        prefix = str(self._selected_results_table_prefix() or "").strip("_")
        if not prefix:
            return base
        if base.startswith(prefix + "_"):
            return base
        return f"{prefix}_{base}"

    # Back-compat shim for older call sites.
    def _selected_mesh_results_table_name(self) -> str:
        return self._results_table_name("swe2d_mesh_results")

    def _apply_run_log_metadata_to_ui(self, metadata: Dict[str, object]) -> int:
        if not isinstance(metadata, dict):
            return 0
        state_payload = metadata.get("workbench_widget_state")
        if not isinstance(state_payload, dict):
            return 0
        widgets_data = state_payload.get("widgets")
        restored = restore_workbench_widget_state(
            ui=self,
            widgets_data=widgets_data,
            qtwidgets_module=QtWidgets,
            log_callback=self._log,
        )
        try:
            self._sync_experimental_3d_mode_widgets()
        except Exception:
            pass
        self._log(f"Applied run metadata settings: restored_widgets={int(restored)}")
        return int(restored)

    def _on_load_run_settings_from_results(self) -> None:
        db_path = str(self._current_line_results_storage_path() or "")
        if not db_path or not os.path.exists(db_path):
            self._log("Load run inputs skipped: results GeoPackage not found.")
            return
        records = self._load_run_logs_from_geopackage(db_path)
        if not records:
            self._log("Load run inputs skipped: no saved run logs found in selected GeoPackage.")
            return
        dlg = SWE2DRunLogViewerDialog(
            records=records,
            run_id=self._run_log_latest_run_id,
            db_path=db_path,
            parent=self,
            apply_run_settings_callback=self._apply_run_log_metadata_to_ui,
        )
        dlg.exec()

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
        self._on_run(request)

    def _ensure_mesh_for_run_preflight(self):
        if self._mesh_data is None:
            self._on_generate_mesh()

    def _has_mesh_for_run_preflight(self) -> bool:
        return self._mesh_data is not None

    def _native_backend_ready_for_run_preflight(self) -> bool:
        openmp_enabled = True
        try:
            if getattr(self, "solver_openmp_enabled_chk", None) is not None:
                openmp_enabled = bool(self.solver_openmp_enabled_chk.isChecked())
        except Exception:
            openmp_enabled = True
        try:
            return bool(swe2d_available(openmp_enabled=openmp_enabled) and SWE2DBackend is not None)
        except TypeError:
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

    def _on_run(self, request=None):
        from swe2d.workbench.monolith_methods import _on_run as _logic
        return _logic(self, request)

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
            "drainage_structures": True,
            "3d_patch": False,
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
        """Set a Studio feature flag and re-apply visibility filters.

        Valid feature keys are defined in self._studio_feature_flags.
        After updating the flag, calls _studio_apply_feature_filters() to
        immediately show/hide matching widgets and tabs.

        To add a new feature:
          1. Add the key to self._studio_feature_flags in the dialog __init__
          2. Add keyword entries in _studio_feature_keywords() below
          3. Add menu + toolbar toggles in _install_studio_host_controls()
        See docs/STUDIO_UI_ARCHITECTURE.md section C.
        """
        key = str(feature or "").strip().lower()
        if key not in self._studio_feature_flags:
            return
        self._studio_feature_flags[key] = bool(enabled)
        self._studio_apply_feature_filters()

    def _studio_feature_keywords(self) -> Dict[str, Tuple[str, ...]]:
        return {
            "rainfall": ("rain", "gauge", "hyet", "storm", "runoff", "precip"),
            "drainage_structures": (
                "drain", "node", "link", "inlet", "outfall", "pipe", "network",
                "structure", "culvert", "weir", "orifice", "gate", "spillway",
                "coupling",
            ),
            "3d_patch": ("3d_patch", "patch_3d", "swe3d"),
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
        # Sync tab page visibility: hide/show tabs whose page or content matches
        # a feature flag, so the tab bar entry disappears when the feature is off.
        tabs = self._left_tabs
        for i in range(tabs.count()):
            page = tabs.widget(i)
            if page is None:
                continue
            blob = self._studio_widget_text_blob(page)
            matched = []
            for feature, words in keywords.items():
                if any(word in blob for word in words):
                    matched.append(feature)
            if not matched:
                continue
            visible = all(self._studio_feature_flags.get(feature, True) for feature in matched)
            try:
                tabs.setTabVisible(i, visible)
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
        # Diagnostic: log that _build_ui was entered
        try:
            self._log("[Studio] _build_ui entered")
        except Exception:
            pass
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
        act_3d_patch = toolbar.addAction("3D Patch")
        act_3d_patch.setCheckable(True)
        act_3d_patch.setChecked(False)
        try:
            self._log("[Studio] 3D Patch toolbar action created")
        except Exception:
            pass
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

        # 3D Patch toggle — creates/hides a dock widget
        _SWE2D_3D_PATCH_DOCK_ATTR = "_swe2d_3d_patch_dock"

        def _toggle_3d_patch(checked: bool) -> None:
            try:
                self._log(f"[Studio] 3D Patch toggled: {checked}")
            except Exception:
                pass
            dock = getattr(self, _SWE2D_3D_PATCH_DOCK_ATTR, None)
            if checked and dock is None:
                patch_page = self._build_3d_patch_tab_page()
                dock = QtWidgets.QDockWidget("3D Patch Settings", self._studio_main_window)
                dock.setObjectName("SWE2D3DPatchDock")
                dock.setWidget(patch_page)
                dock.setFeatures(
                    QtWidgets.QDockWidget.DockWidgetMovable
                    | QtWidgets.QDockWidget.DockWidgetFloatable
                    | QtWidgets.QDockWidget.DockWidgetClosable
                )
                dock.visibilityChanged.connect(
                    lambda visible: act_3d_patch.setChecked(visible)
                )
                self._studio_main_window.addDockWidget(
                    QtCore.Qt.RightDockWidgetArea, dock
                )
                setattr(self, _SWE2D_3D_PATCH_DOCK_ATTR, dock)
            elif not checked and dock is not None:
                self._studio_main_window.removeDockWidget(dock)
                dock.deleteLater()
                setattr(self, _SWE2D_3D_PATCH_DOCK_ATTR, None)

        act_3d_patch.toggled.connect(_toggle_3d_patch)
        act_open_coupling_results = toolbar.addAction("Open Drainage/Structure Results")
        act_open_coupling_results.triggered.connect(
            lambda: self._open_coupling_results_viewer()
            if hasattr(self, "_open_coupling_results_viewer")
            else None
        )
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

        # ── Widget binding validation ─────────────────────────────────
        self._validate_widget_bindings()

    def _validate_widget_bindings(self) -> None:
        """Check that critical widgets have Python bindings.

        Logs warnings for missing optional widgets.  Raises RuntimeError
        for absolutely critical missing widgets (e.g. run button).
        """
        critical = {
            "run_btn": QtWidgets.QPushButton,
        }
        optional = {
            "cfl_spin": QtWidgets.QDoubleSpinBox,
            "dt_spin": QtWidgets.QDoubleSpinBox,
            "n_mann_spin": QtWidgets.QDoubleSpinBox,
            "view_mode_combo": QtWidgets.QComboBox,
            "snapshot_btn": QtWidgets.QPushButton,
        }
        missing_optional: List[str] = []

        for name, wtype in critical.items():
            w = self.findChild(wtype, name)
            if w is None:
                raise RuntimeError(
                    f"Critical widget '{name}' ({wtype.__name__}) has no Python "
                    f"binding in {type(self).__name__}.  Check that the widget "
                    f"objectName is correct and the .ui file is loaded."
                )

        for name, wtype in optional.items():
            w = self.findChild(wtype, name)
            if w is None:
                missing_optional.append(name)

        if missing_optional:
            self._log(
                f"[Studio] Optional widgets missing bindings: {', '.join(missing_optional)}"
            )

    def closeEvent(self, event):  # type: ignore[override]
        self._save_studio_layout_state()
        super().closeEvent(event)



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


def _close_workbench_studio_windows() -> None:
    _close_dialog_windows(_SWE2D_WORKBENCH_STUDIO_WINDOWS)


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


def _remove_workbench_studio_dock(iface_obj) -> None:
    from swe2d.workbench.extracted.studio_host_methods import _remove_workbench_studio_dock as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(iface_obj)


def _attach_host_dock_widget(iface_obj, host_window, dock: QtWidgets.QDockWidget, area) -> bool:
    from swe2d.workbench.extracted.studio_host_methods import _attach_host_dock_widget as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(iface_obj, host_window, dock, area)


def _studio_take_dock_widget(studio_dock, fallback_text: str) -> QtWidgets.QWidget:
    from swe2d.workbench.extracted.studio_host_methods import _studio_take_dock_widget as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(studio_dock, fallback_text)


def _build_studio_component_docks(iface_obj, host_window, dlg) -> Dict[str, QtWidgets.QDockWidget]:
    """Extract Studio component docks from the hidden dialog QMainWindow.

    The Studio dialog builds its full UI (toolbar, left tabs, right pane,
    log, inspector) inside a hidden QMainWindow.  This function tears down
    that window and re-parents its docks into the QGIS host window so the
    user sees native QGIS dock widgets.

    Returns a dict of dock references keyed by panel name.
    """
    from swe2d.workbench.extracted.studio_host_methods import _build_studio_component_docks as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(iface_obj, host_window, dlg)


def _studio_host_main_window(iface_obj, fallback_parent=None):
    from swe2d.workbench.extracted.studio_host_methods import _studio_host_main_window as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(iface_obj, fallback_parent)


def _clear_studio_host_controls(iface_obj, fallback_parent=None) -> None:
    from swe2d.workbench.extracted.studio_host_methods import _clear_studio_host_controls as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(iface_obj, fallback_parent)


def _install_studio_host_controls(
    iface_obj,
    dlg,
    fallback_parent=None,
    component_docks: Optional[Dict[str, QtWidgets.QDockWidget]] = None,
) -> None:
    """Install the SWE2D Studio menu and toolbar into the QGIS host window.

    Creates:
      - QGIS menu bar entry "SWE2D Studio" with feature toggles and focus actions
      - QGIS toolbar "SWE2D Studio" with tab-select, refresh, snapshot, and
        feature toggle checkable buttons

    Feature toggles (Rainfall, Drainage/Structures, 3D Patch) are wired to
    dlg._studio_set_feature_enabled().  When adding a new toggle, add both
    a menu action AND a toolbar button here.
    See docs/STUDIO_UI_ARCHITECTURE.md section C.
    """
    from swe2d.workbench.extracted.studio_host_methods import _install_studio_host_controls as _logic

    _prepare_studio_host_logic_globals(_logic)

    return _logic(iface_obj, dlg, fallback_parent, component_docks)


def _prepare_studio_host_logic_globals(_logic) -> None:
    try:
        _g = getattr(_logic, "__globals__", None)
        if not isinstance(_g, dict):
            return
        if _g.get("_SWE2D_WORKBENCH_STUDIO_DOCK") is None and _SWE2D_WORKBENCH_STUDIO_DOCK is not None:
            _g["_SWE2D_WORKBENCH_STUDIO_DOCK"] = _SWE2D_WORKBENCH_STUDIO_DOCK
        if not _g.get("_SWE2D_STUDIO_COMPONENT_DOCKS") and _SWE2D_STUDIO_COMPONENT_DOCKS:
            _g["_SWE2D_STUDIO_COMPONENT_DOCKS"] = _SWE2D_STUDIO_COMPONENT_DOCKS
        if _g.get("_SWE2D_STUDIO_HOST_DIALOG") is None and _SWE2D_STUDIO_HOST_DIALOG is not None:
            _g["_SWE2D_STUDIO_HOST_DIALOG"] = _SWE2D_STUDIO_HOST_DIALOG
        if _g.get("_SWE2D_STUDIO_HOST_TOOLBAR") is None and _SWE2D_STUDIO_HOST_TOOLBAR is not None:
            _g["_SWE2D_STUDIO_HOST_TOOLBAR"] = _SWE2D_STUDIO_HOST_TOOLBAR
        if _g.get("_SWE2D_STUDIO_HOST_MENU") is None and _SWE2D_STUDIO_HOST_MENU is not None:
            _g["_SWE2D_STUDIO_HOST_MENU"] = _SWE2D_STUDIO_HOST_MENU
        _g["_remove_workbench_dock_instance"] = _remove_workbench_dock_instance
    except Exception:
        pass


def launch_swe2d_workbench_studio(parent=None, iface=None, host_mode: str = "dock"):
    global _SWE2D_WORKBENCH_STUDIO_DOCK, _SWE2D_STUDIO_COMPONENT_DOCKS, _SWE2D_STUDIO_HOST_DIALOG
    from swe2d.workbench.extracted.studio_host_methods import enforce_studio_shell_visible as _enforce_studio_shell_visible

    iface = _resolve_workbench_iface(parent, iface)

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
        try:
            # Studio host keeps the controller dialog hidden, so the base
            # showEvent restore path may not run. Restore explicitly here.
            dlg._restore_project_workbench_state()
            dlg._workbench_state_restored_on_show = True
        except Exception:
            pass
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


