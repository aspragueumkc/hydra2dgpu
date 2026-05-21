#!/usr/bin/env python3
"""Startup wiring helper for SWE2D workbench run/runtime seam components."""

from __future__ import annotations

from typing import Any, Dict


def wire_startup_run_components(dialog: Any, ns: Dict[str, Any]) -> None:
    """Instantiate optional run/runtime seam components with startup diagnostics."""
    SWE2DWorkbenchViewAdapter = ns.get("SWE2DWorkbenchViewAdapter")
    SWE2DRunOrchestrator = ns.get("SWE2DRunOrchestrator")
    SWE2DRunRequest = ns.get("SWE2DRunRequest")
    SWE2DRunController = ns.get("SWE2DRunController")
    SWE2DRunDataBuilder = ns.get("SWE2DRunDataBuilder")
    SWE2DRunOptionsBuilder = ns.get("SWE2DRunOptionsBuilder")
    SWE2DBackendInitializer = ns.get("SWE2DBackendInitializer")
    SWE2DRunFinalizer = ns.get("SWE2DRunFinalizer")
    SWE2DRunLifecycle = ns.get("SWE2DRunLifecycle")
    swe2d_gpu_available = ns.get("swe2d_gpu_available")
    TemporalScheme = ns.get("TemporalScheme")
    SpatialDiscretization = ns.get("SpatialDiscretization")
    GodunovSolverMode = ns.get("GodunovSolverMode")
    SolverModelOptions = ns.get("SolverModelOptions")
    SWE2DEquationSet = ns.get("SWE2DEquationSet")
    SWE2DThreeDSolverModel = ns.get("SWE2DThreeDSolverModel")
    SWE2DThreeDCouplingMode = ns.get("SWE2DThreeDCouplingMode")

    if SWE2DWorkbenchViewAdapter is not None:
        dialog._view_adapter = dialog._init_startup_component(
            "view adapter",
            lambda: SWE2DWorkbenchViewAdapter(dialog),
        )

    if SWE2DRunOrchestrator is not None and SWE2DRunRequest is not None:
        dialog._run_orchestrator = dialog._init_startup_component(
            "run orchestrator",
            lambda: SWE2DRunOrchestrator(dialog._execute_run_request, dialog._log),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("run orchestrator", required_for_run=True)

    if SWE2DRunController is not None:
        dialog._run_controller = dialog._init_startup_component(
            "run controller",
            lambda: SWE2DRunController(
                ensure_mesh_callback=dialog._ensure_mesh_for_run_preflight,
                has_mesh_callback=dialog._has_mesh_for_run_preflight,
                backend_ready_callback=dialog._native_backend_ready_for_run_preflight,
                backend_unavailable_callback=dialog._show_backend_unavailable_for_run_preflight,
                log_callback=dialog._log,
            ),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("run controller", required_for_run=True)

    if SWE2DRunDataBuilder is not None:
        dialog._run_data_builder = dialog._init_startup_component(
            "run data builder",
            lambda: SWE2DRunDataBuilder(
                get_mesh_data_callback=lambda: dialog._mesh_data,
                collect_boundary_arrays_callback=dialog._collect_boundary_arrays,
                build_side_hydrographs_callback=dialog._build_side_hydrographs,
                collect_bc_layer_hydrographs_callback=dialog._collect_bc_layer_hydrographs,
                collect_bc_layer_edge_groups_callback=dialog._collect_bc_layer_edge_groups,
                initial_state_callback=dialog._initial_state,
                build_spatial_manning_array_callback=dialog._build_spatial_manning_array,
                update_unit_system_callback=dialog._update_unit_system_from_crs,
            ),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("run data builder", required_for_run=True)

    if SWE2DRunOptionsBuilder is not None:
        dialog._run_options_builder = dialog._init_startup_component(
            "run options builder",
            lambda: SWE2DRunOptionsBuilder(
                ui=dialog,
                log_callback=dialog._log,
                parse_run_duration_seconds_callback=dialog._parse_run_duration_seconds,
                collect_3d_patch_env_overrides_callback=dialog._collect_3d_patch_env_overrides,
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
                godunov_solver_mode_enum=GodunovSolverMode,
                solver_model_options_cls=SolverModelOptions,
                swe2d_equation_set_enum=SWE2DEquationSet,
                swe2d_3d_solver_model_enum=SWE2DThreeDSolverModel,
                swe2d_3d_coupling_mode_enum=SWE2DThreeDCouplingMode,
            ),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("run options builder", required_for_run=True)

    if SWE2DBackendInitializer is not None:
        dialog._backend_initializer = dialog._init_startup_component(
            "backend initializer",
            lambda: SWE2DBackendInitializer(
                ui=dialog,
                apply_env_overrides_callback=dialog._apply_env_overrides,
                restore_env_overrides_callback=dialog._restore_env_overrides,
                apply_timeseries_bc_values_callback=dialog._apply_timeseries_bc_values,
                distribute_total_flow_to_unit_q_callback=dialog._distribute_total_flow_to_unit_q,
            ),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("backend initializer", required_for_run=True)

    if SWE2DRunFinalizer is not None:
        dialog._run_finalizer = dialog._init_startup_component(
            "run finalizer",
            lambda: SWE2DRunFinalizer(dialog),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("run finalizer", required_for_run=True)

    if SWE2DRunLifecycle is not None:
        dialog._run_lifecycle = dialog._init_startup_component(
            "run lifecycle",
            lambda: SWE2DRunLifecycle(dialog),
            required_for_run=True,
        )
    else:
        dialog._note_startup_component_missing("run lifecycle", required_for_run=True)

    if dialog._startup_run_component_errors:
        dialog._log(
            "Startup run seam readiness warning: "
            + ", ".join(sorted(set(dialog._startup_run_component_errors)))
        )
