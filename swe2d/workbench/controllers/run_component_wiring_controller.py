#!/usr/bin/env python3
"""Startup wiring helper for SWE2D workbench run/runtime seam components."""

from __future__ import annotations

from typing import Any, Dict


def wire_startup_run_components(dialog: Any, ns: Dict[str, Any]) -> None:
    """Instantiate run/runtime seam components directly."""
    SWE2DRunOrchestrator = ns["SWE2DRunOrchestrator"]
    SWE2DRunRequest = ns["SWE2DRunRequest"]
    SWE2DRunController = ns["SWE2DRunController"]
    SWE2DRunDataBuilder = ns["SWE2DRunDataBuilder"]
    SWE2DRunOptionsBuilder = ns["SWE2DRunOptionsBuilder"]
    SWE2DBackendInitializer = ns["SWE2DBackendInitializer"]
    SWE2DRunFinalizer = ns["SWE2DRunFinalizer"]
    SWE2DRunLifecycle = ns["SWE2DRunLifecycle"]
    swe2d_gpu_available = ns["swe2d_gpu_available"]
    TemporalScheme = ns["TemporalScheme"]
    SpatialDiscretization = ns["SpatialDiscretization"]
    SolverModelOptions = ns["SolverModelOptions"]

    dialog._run_orchestrator = SWE2DRunOrchestrator(
        dialog._execute_run_request, dialog._log,
    )

    dialog._run_controller = SWE2DRunController(
        ensure_mesh_callback=dialog._ensure_mesh_for_run_preflight,
        has_mesh_callback=dialog._has_mesh_for_run_preflight,
        backend_ready_callback=dialog._backend_ready_for_run_preflight,
        backend_unavailable_callback=dialog._show_backend_unavailable_for_run_preflight,
        log_callback=dialog._log,
    )

    dialog._run_data_builder = SWE2DRunDataBuilder(
        get_mesh_data_callback=lambda: dialog._mesh_data,
        collect_boundary_arrays_callback=dialog._collect_boundary_arrays,
        build_side_hydrographs_callback=lambda: {},
        collect_bc_layer_hydrographs_callback=dialog._collect_bc_layer_hydrographs,
        collect_bc_layer_edge_groups_callback=dialog._collect_bc_layer_edge_groups,
        initial_state_callback=dialog._initial_state,
        build_spatial_manning_array_callback=dialog._build_spatial_manning_array,
        update_unit_system_callback=dialog._update_unit_system_from_crs,
    )

    dialog._run_options_builder = SWE2DRunOptionsBuilder(
        ui=dialog,
        log_callback=dialog._log,
        parse_run_duration_seconds_callback=dialog._parse_run_duration_seconds,
        rain_rate_si_to_model_callback=dialog._rain_rate_si_to_model,
        build_internal_flow_forcing_callback=dialog._build_internal_flow_forcing,
        internal_flow_source_cms_at_time_callback=dialog._internal_flow_source_cms_at_time,
        flow_si_to_model_callback=dialog._flow_si_to_model,
        build_thiessen_rain_cn_forcing_callback=dialog._build_thiessen_rain_cn_forcing,
        build_pipe_network_config_callback=dialog._build_pipe_network_config,
        build_hydraulic_structure_config_callback=dialog._build_hydraulic_structure_config,
        swe2d_gpu_available_callback=swe2d_gpu_available,
        temporal_scheme_enum=TemporalScheme,
        spatial_discretization_enum=SpatialDiscretization,
        solver_model_options_cls=SolverModelOptions,
    )

    dialog._backend_initializer = SWE2DBackendInitializer(
        apply_timeseries_bc_values_callback=dialog._apply_timeseries_bc_values,
        distribute_total_flow_to_unit_q_callback=dialog._distribute_total_flow_to_unit_q,
    )

    dialog._run_finalizer = SWE2DRunFinalizer(dialog)

    dialog._run_lifecycle = SWE2DRunLifecycle(dialog)
