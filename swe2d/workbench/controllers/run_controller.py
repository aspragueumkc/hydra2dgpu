"""WorkbenchController — mediates between Service Layer and View.

The Controller is the brain of the workbench:
- Receives requests from the View (user actions)
- Calls Service methods
- Pushes results back to the View (updates state, calls view methods)

The Controller does NOT contain business logic itself — it only
orchestrates. Business logic lives in services.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from swe2d.workbench.services import gpkg_service

if TYPE_CHECKING:
    from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog


# Module-level alias used by the snapshot orchestration below. This is kept
# because ``on_snapshot`` still calls into the HEC-RAS HDF5 export.


class RunController:
    """MVP domain controller for the 2D simulation run pipeline."""
    """Mediates between Service Layer and View (SWE2DWorkbenchStudioDialog).

    Holds a reference to the View (dialog). Methods are called either
    directly by the dialog or in response to View signals.
    """

    def __init__(self, view: "SWE2DWorkbenchStudioDialog"):
        self._view = view

    def on_run(self, request: Optional[Any] = None) -> Any:
        """Execute a 2D run.

        The controller reads all widget and service references directly
        from the view and executes the full simulation pipeline inline.
        No extracted-function indirection — the run pipeline body lives
        here as ``_execute_run``.

        Returns whatever the run pipeline returns, or None when the run
        is aborted (missing mesh, missing backend, etc.).
        """
        view = self._view
        if view._mesh_data is None:
            view._log("Run aborted: mesh not available after preflight.")
            return None
        return self._execute_run(view, request=request)

    _noop = staticmethod(lambda *a, **k: None)

    def _execute_run(self, view: Any, request: Optional[Any] = None) -> Any:
        """Full 2D simulation pipeline — inlined from extracted/_on_run.

        Reads all widget values, service references, and mesh data
        directly from ``view`` attributes.  No kwargs indirection:
        every variable is resolved at the top of this method.

        The algorithm flow is identical to the legacy ``_on_run``
        function in ``extracted/model_and_run_methods.py`` — only the
        data-access pattern has changed (direct attribute reads instead
        of 50+ kwargs).
        """
        # ── Resolve all view references (same as old _build_run_kwargs) ──
        log_fn = view._log
        mesh_data = view._mesh_data
        # Required seam components — AttributeError if missing (fail fast)
        run_data_builder = view._run_data_builder
        run_options_builder = view._run_options_builder
        backend_initializer = view._backend_initializer
        run_finalizer = view._run_finalizer
        run_lifecycle = view._run_lifecycle
        parse_time_hours_fn = view._parse_time_hours
        mesh_cell_areas_fn = view._mesh_cell_areas
        mesh_cell_min_bed_fn = view._mesh_cell_min_bed
        mesh_cell_centroids_fn = view._mesh_cell_centroids
        mesh_cell_solver_bed_fn = view._mesh_cell_solver_bed
        length_scale_si_to_model_fn = view._length_scale_si_to_model
        rain_mm_to_model_depth_fn = view._rain_mm_to_model_depth
        rain_rate_si_to_model_fn = view._rain_rate_si_to_model
        flow_si_to_model_fn = view._flow_si_to_model
        # Optional — fail fast if feature not configured
        build_line_sampling_map_fn = view._build_line_sampling_map
        model_gpkg_path = view._model_gpkg_path
        length_unit_name = view._length_unit_name
        internal_flow_source_cms_at_time_fn = view._internal_flow_source_cms_at_time
        last_run_request = view._last_run_request

        # Widget references — resolved through view protocol
        wp = view.collect_run_widget_params()
        n_mann_spin = wp["n_mann_spin"]
        h_min_spin = wp["h_min_spin"]
        cfl_lambda_cap_spin = wp["cfl_lambda_cap_spin"]
        gpu_diag_sync_interval_spin = wp["gpu_diag_sync_interval_spin"]
        max_rel_depth_increase_spin = wp["max_rel_depth_increase_spin"]
        max_source_depth_step_spin = wp["max_source_depth_step_spin"]
        max_source_rate_spin = wp["max_source_rate_spin"]
        extreme_rain_mode_chk = wp["extreme_rain_mode_chk"]
        source_cfl_beta_spin = wp["source_cfl_beta_spin"]
        source_max_substeps_spin = wp["source_max_substeps_spin"]
        source_true_subcycling_chk = wp["source_true_subcycling_chk"]
        source_imex_split_chk = wp["source_imex_split_chk"]
        shallow_damping_depth_spin = wp["shallow_damping_depth_spin"]
        depth_cap_spin = wp["depth_cap_spin"]
        momentum_cap_min_speed_spin = wp["momentum_cap_min_speed_spin"]
        momentum_cap_celerity_mult_spin = wp["momentum_cap_celerity_mult_spin"]
        max_inv_area_spin = wp["max_inv_area_spin"]
        rain_rate_spin = wp["rain_rate_spin"]
        output_interval_edit = wp["output_interval_edit"]
        line_output_interval_edit = wp["line_output_interval_edit"]
        tiny_mode_combo = wp["tiny_mode_combo"]
        tiny_wet_cell_threshold_spin = wp["tiny_wet_cell_threshold_spin"]
        source_stage_coupled_imex_rk2_chk = wp["source_stage_coupled_imex_rk2_chk"]
        inflow_progressive_chk = wp["inflow_progressive_chk"]

        # ── Imports that pull in Qt / swe2d_workbench_qt ────────────
        from qgis.PyQt import QtWidgets

        from swe2d import units as _u
        from swe2d.workbench.services.constants_service import (
            BC_INFLOW_Q as _BC_INFLOW_Q,
            BC_TS_FLOW as _BC_TS_FLOW,
            BC_TS_STAGE as _BC_TS_STAGE,
        )
        from swe2d.workbench.services.non_gui_runtime_service import execute_run_timestep_loop as _execute_run_timestep_loop_runtime_logic
        from swe2d.runtime.coupling import pack_coupling_soa
        from swe2d.runtime.backend import SWE2DBackend
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.runtime.native_bc_forcing import SWE2DNativeBoundaryHydrographConfigurator
        from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
        from swe2d.runtime.runtime_reporting import SWE2DRuntimeReporter
        from swe2d.runtime.runtime_sources import SWE2DRuntimeSourceManager
        from swe2d.runtime.runtime_step_executor import SWE2DRuntimeStepExecutor
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.extensions.extension_models import TemporalScheme

        # ── Begin _on_run body (identical algorithm, direct view reads) ──
        if request is None:
            request = last_run_request
        if mesh_data is None:
            log_fn("Run aborted: mesh not available after preflight.")
            return
        if SWE2DBackend is None:
            log_fn("Run aborted: native backend not available after preflight.")
            return

        view._cancel_requested = False
        view.set_run_button_enabled(False)
        view.set_cancel_button_enabled(True)
        view.set_run_progress(0)

        backend = None
        run_id = ""
        run_wallclock_start = ""
        run_perf_start = time.perf_counter()
        run_log_start_idx = len(view._runtime_log_lines)
        try:
            run_input = run_data_builder.build()
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

            run_options = run_options_builder.build(
                dt=wp["dt_spin"],
                adaptive_cfl_dt=wp["adaptive_cfl_dt_chk"],
                initial_dt=wp["initial_dt_spin"],
                reconstruction_mode=wp["reconstruction_combo"],
                reconstruction_name=wp["reconstruction_combo_text"],
                temporal_order_value=wp["temporal_order_combo"],
                temporal_scheme_name=wp["temporal_order_combo_text"],
                drainage_gpu_method=wp["drainage_gpu_method"],
                culvert_solver_mode=wp["culvert_solver_mode"],
                cuda_graphs_enabled=wp["enable_cuda_graphs_chk"],
                swe2d_perf_mode=wp["swe2d_perf_mode_chk"],
                rain_rate_mmhr=wp["rain_rate_spin"],
                bridge_coupling_mode=wp["bridge_coupling_mode"],
                culvert_face_flux=wp["culvert_face_flux_chk"],
            )
            run_duration_s = run_options.run_duration_s
            if request is not None:
                request_run_duration_text = getattr(request, "run_duration_text", None)
                if request_run_duration_text is not None and str(request_run_duration_text).strip():
                    try:
                        run_duration_s = max(0.0, parse_time_hours_fn(str(request_run_duration_text).strip()) * 3600.0)
                    except Exception:
                        log_fn("[WARNING] Unexpected error silently caught")
            dt_cfg = run_options.dt_cfg
            adaptive_cfl_dt = run_options.adaptive_cfl_dt
            dt_fixed = run_options.dt_fixed
            dt_request = run_options.dt_request
            initial_dt = getattr(run_options, "initial_dt", 0.0)
            reconstruction_mode = run_options.reconstruction_mode
            reconstruction_name = run_options.reconstruction_name
            temporal_scheme = run_options.temporal_scheme
            temporal_scheme_name = run_options.temporal_scheme_name
            solver_backend_mode = str(getattr(run_options, "solver_backend_mode", "gpu")).strip().lower()
            coupling_loop_mode = run_options.coupling_loop_mode
            drainage_solver_backend_mode = run_options.drainage_solver_backend_mode
            drainage_gpu_method_mode = run_options.drainage_gpu_method_mode
            culvert_solver_mode = getattr(run_options, "culvert_solver_mode", 0)
            cuda_graphs_enabled = run_options.cuda_graphs_enabled
            model_options = run_options.model_options
            rain_rate_model = run_options.rain_rate_model
            internal_flow_forcing = run_options.internal_flow_forcing
            cell_source_model = run_options.cell_source_model
            thiessen_forcing = run_options.thiessen_forcing
            pipe_network_cfg = run_options.pipe_network_cfg
            hydraulic_structures_cfg = run_options.hydraulic_structures_cfg
            bridge_stacked_plans = []

            try:
                from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime

                bridge_stacked_plans = build_bridge_stacked_plans_for_runtime(
                    mesh_data,
                    hydraulic_structures_cfg,
                    log_fn=log_fn,
                )
            except Exception as exc:
                log_fn(f"Bridge stacked-plan mapping warning: {exc}")

            # Propagate locally-built drainage/structure configs into model_options
            # so that enable_pipe_network_module and enable_hydraulic_structures flags are set correctly.
            if model_options is not None:
                if pipe_network_cfg is not None:
                    model_options.pipe_network = pipe_network_cfg
                if hydraulic_structures_cfg is not None:
                    model_options.hydraulic_structures = hydraulic_structures_cfg

            coupling_soa = None
            if pack_coupling_soa is not None:
                coupling_soa = pack_coupling_soa(
                    n_cells=int(mesh_cell_areas_fn().shape[0]),
                    pipe_network=pipe_network_cfg,
                    hydraulic_structures=hydraulic_structures_cfg,
                )
            coupling_controller = None
            if SWE2DCouplingController is not None and (pipe_network_cfg is not None or hydraulic_structures_cfg is not None):
                drainage_mod = SWE2DUrbanDrainageModule(pipe_network_cfg) if pipe_network_cfg is not None and SWE2DUrbanDrainageModule is not None else None
                if drainage_mod is not None:
                    drainage_mod.initialize()
                # Compute model-to-feet factor: for a foot model (length_scale=3.28)
                # this gives 1.0 (no conversion); for SI (length_scale=1.0) this gives 3.28.
                # _ls = model units per SI meter (e.g. 3.28 for feet, 1.0 for meters)
                # Convert to si_m_per_model (e.g. 0.3048 for feet, 1.0 for meters)
                # which the coupling controller and units.configure() expect.
                _ls = max(1.0e-6, float(length_scale_si_to_model_fn()))
                _si_m_per_model = 1.0 / _ls
                _model_to_ft = _u.USC_FT_PER_SI_M * _si_m_per_model
                structures_mod = SWE2DStructureModule(hydraulic_structures_cfg, model_to_ft=_model_to_ft) if hydraulic_structures_cfg is not None and SWE2DStructureModule is not None else None
                coupling_controller = SWE2DCouplingController(
                    cell_area=mesh_cell_areas_fn(),
                    cell_bed=mesh_cell_min_bed_fn(),
                    drainage=drainage_mod,
                    structures=structures_mod,
                    drainage_gpu_method=drainage_gpu_method_mode,
                    culvert_solver_mode=culvert_solver_mode,
                    bridge_cuda_coupling=bool(run_options.bridge_cuda_coupling),
                    bridge_stacked_coupling_mode=str(getattr(run_options, "bridge_stacked_coupling_mode", "phase3_spatial")),
                    length_scale_si_to_model=_si_m_per_model,
                    culvert_face_flux_mode=str(getattr(run_options, "culvert_face_flux_mode", "off")),
                    use_redistribution=bool(wp["use_redistribution_chk"]),
                    log_callback=log_fn,
                )
                setattr(coupling_controller, "bridge_stacked_plans", bridge_stacked_plans)
                # Provide cell centroids for influence-width redistribution.
                try:
                    cx, cy = mesh_cell_centroids_fn()
                    if hasattr(coupling_controller, "set_cell_centroids"):
                        coupling_controller.set_cell_centroids(cx, cy)
                    if hasattr(coupling_controller, "_build_redistribution_data"):
                        coupling_controller._build_redistribution_data()
                except Exception:
                    logger.warning("Unexpected error silently caught", exc_info=True)
                # GPU-only runtime: all coupling/drainage paths use CUDA.

            rain_stats_acc = {"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0}

            if request is not None:
                last_run_request = request

            def _parse_interval_text(text, default_widget_text):
                """Parse an output interval text (hours), falling back to widget value."""
                if text is not None and str(text).strip():
                    return parse_time_hours_fn(str(text).strip())
                return parse_time_hours_fn(str(default_widget_text or ""))

            request_output_interval_text = getattr(request, "output_interval_text", None) if request is not None else None
            request_line_output_interval_text = getattr(request, "line_output_interval_text", None) if request is not None else None

            _oi_hr = _parse_interval_text(
                request_output_interval_text,
                output_interval_edit,
            )
            output_interval_s = max(1.0, _oi_hr * 3600.0)
            _line_oi_hr = _parse_interval_text(
                request_line_output_interval_text,
                line_output_interval_edit,
            )
            line_output_interval_s = max(1.0, _line_oi_hr * 3600.0)
            results_data = getattr(view, "_results_data", None)
            if results_data is not None:
                results_data.clear_live_snapshots()
            # Initialise results panel + temporal dock so the user can
            # scrub through snapshots as they accumulate during the run.
            try:
                view._show_results_panel()
            except Exception:
                pass
            # Leave view._snapshot_mesh_fingerprint as empty string so the old
            # fingerprint guard in _refresh_high_perf_canvas_overlay never
            # blocks rendering (the guard has been removed anyway, but this
            # prevents stale fingerprint data from a future re-introduction).
            view._snapshot_mesh_fingerprint = ""
            run_span_s = max(float(run_duration_s), 1.0e-9)
            _next_snap_t = min(output_interval_s, run_span_s)
            _next_line_snap_t = min(line_output_interval_s, run_span_s)
            _next_coupling_snap_t = min(line_output_interval_s, run_span_s)
            sample_map = build_line_sampling_map_fn()
            cell_solver_z = mesh_cell_solver_bed_fn() if sample_map else None
            run_id = datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z")
            run_wallclock_start = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")

            dynamic_bc = bool(np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs)
            if dynamic_bc:
                log_fn("Timeseries BC mode active (flow/stage hydrographs).")

            log_fn("Starting 2D run...")
            log_fn(f"Run wallclock start: {run_wallclock_start}")
            log_fn(f"SWE2D solver backend: {solver_backend_mode}")
            log_fn("SWE2D solver: GPU-only mode (CUDA)")
            log_fn(f"Reconstruction mode: {reconstruction_name}")
            log_fn(f"Temporal scheme: {temporal_scheme_name}")
            log_fn(
                "SWE2D perf mode: "
                f"{'enabled' if bool(os.environ.get('BACKWATER_SWE2D_PERF_MODE', '0')) == '1' else 'disabled'}"
            )
            log_fn(
                "Tiny-mode config: "
                f"mode={int(tiny_mode_combo)}, "
                f"wet_cell_threshold={int(tiny_wet_cell_threshold_spin)}"
            )
            log_fn(
                f"Output intervals: mesh={output_interval_s:.1f}s, sample-lines={line_output_interval_s:.1f}s"
            )
            try:
                wp["gpu_diag_sync_interval_raw"]
            except Exception:
                logger.warning("Unexpected error silently caught", exc_info=True)
            log_fn(
                "Stability controls: "
                f"max_rel_dh={float(max_rel_depth_increase_spin):.3f}, "
                f"gpu_diag_sync_steps={int(gpu_diag_sync_interval_spin)}, "
                f"src_dh_step_cap={float(max_source_depth_step_spin):.6e}, "
                f"src_rate_cap={float(max_source_rate_spin):.6e}, "
                f"extreme_rain_mode={bool(extreme_rain_mode_chk)}, "
                f"src_beta={float(source_cfl_beta_spin):.3f}, "
                f"src_max_substeps={int(source_max_substeps_spin)}, "
                f"true_subcycling={bool(source_true_subcycling_chk)}, "
                f"imex_split={bool(source_imex_split_chk)}, "
                f"stage_coupled_imex_rk2={bool(source_stage_coupled_imex_rk2_chk)}, "
                f"shallow_damp_h={float(shallow_damping_depth_spin):.6e}, "
                f"depth_cap={float(depth_cap_spin):.3f}, "
                f"mom_cap_min={float(momentum_cap_min_speed_spin):.3f}, "
                f"mom_cap_mult={float(momentum_cap_celerity_mult_spin):.3f}, "
                f"invA_cap={float(max_inv_area_spin):.3e}, "
                f"lambda_cap={float(cfl_lambda_cap_spin):.3e}"
            )
            if adaptive_cfl_dt:
                log_fn(f"Timestep mode: variable CFL (dt_max={dt_cfg:.5f} s)")
            else:
                log_fn(f"Timestep mode: fixed dt ({dt_cfg:.5f} s)")
            if initial_dt > 0.0:
                log_fn(f"Initial dt override: {initial_dt:.5f} s (first step only)")
            if float(np.asarray(rain_rate_model, dtype=np.float64)) > 0.0:
                log_fn(
                    f"Rain-on-grid active: {float(rain_rate_spin):.3f} mm/hr "
                    f"(applied as {float(np.asarray(rain_rate_model, dtype=np.float64)):.6e} {length_unit_name}/s)"
                )
            if thiessen_forcing is not None:
                infil_method = str(getattr(thiessen_forcing, "infiltration_method", "scs_cn") or "scs_cn").lower().strip()
                infil_label = "NRCS CN infiltration"
                if infil_method == "none":
                    infil_label = "no infiltration (all rainfall to runoff)"
                log_fn(
                    "Spatial rainfall forcing active: Thiessen nearest-gage interpolation + "
                    f"{infil_label}."
                )
            if cell_source_model is not None:
                log_fn(
                    f"Internal source/sink forcing active: total_Q={float(np.sum(cell_source_model)):.6f} {'cfs' if str(length_unit_name).strip().lower() == 'ft' else 'cms'}"
                )
            if internal_flow_forcing is not None:
                ts_count = int(len(internal_flow_forcing.get("dynamic_terms", [])))
                if ts_count > 0:
                    log_fn(f"Internal flow time-series forcing active: features={ts_count}")
            if coupling_controller is not None:
                log_fn(
                    "Coupled drainage/structure forcing active: "
                    f"drainage={pipe_network_cfg is not None}, structures={hydraulic_structures_cfg is not None}, "
                    f"drainage_backend={drainage_solver_backend_mode}, "
                    f"drainage_gpu_method={drainage_gpu_method_mode}"
                )
                coupling_runtime_mode = "cuda"
                log_fn(f"Coupling runtime mode: {coupling_runtime_mode}")
                if bridge_stacked_plans:
                    total_bridge_cells = int(sum(int(p.selected_cells.size) for p in bridge_stacked_plans))
                    log_fn(
                        "Bridge stacked plans active: "
                        f"count={len(bridge_stacked_plans)}, selected_cells={total_bridge_cells}"
                    )
            log_fn(f"CUDA graph replay: {'enabled' if cuda_graphs_enabled else 'disabled'}")
            if coupling_soa is not None:
                dn = coupling_soa.drainage
                ss = coupling_soa.structures
                if dn is not None:
                    bad_links = int(np.sum((dn.link_from < 0) | (dn.link_to < 0)))
                    bad_inlets = int(np.sum((dn.inlet_cell < 0) | (dn.inlet_node < 0)))
                    log_fn(
                        "CUDA SoA pack (drainage): "
                        f"nodes={dn.node_x.size}, links={dn.link_from.size}, inlets={dn.inlet_cell.size}, "
                        f"invalid_links={bad_links}, invalid_inlets={bad_inlets}"
                    )
                if ss is not None:
                    bad_struct = int(np.sum((ss.upstream_cell < 0) | (ss.downstream_cell < 0)))
                    log_fn(
                        "CUDA SoA pack (structures): "
                        f"count={ss.structure_type.size}, invalid_cell_pairs={bad_struct}"
                    )

            def _build_and_initialize_backend() -> SWE2DBackend:
                """Build and initialize the SWE2D backend with current widget params."""
                return backend_initializer.build_and_initialize(
                    backend_cls=SWE2DBackend,
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
                    dt_initial=initial_dt,
                    reconstruction_mode=reconstruction_mode,
                    temporal_scheme=temporal_scheme,
                    gravity=wp["gravity"],
                    k_mann=wp["k_mann"],
                    n_mann=wp["n_mann_spin"],
                    cfl=wp["cfl_spin"],
                    h_min=wp["h_min_spin"],
                    max_inv_area=wp["max_inv_area_spin"],
                    cfl_lambda_cap=wp["cfl_lambda_cap_spin"],
                    momentum_cap_min_speed=wp["momentum_cap_min_speed_spin"],
                    momentum_cap_celerity_mult=wp["momentum_cap_celerity_mult_spin"],
                    depth_cap=wp["depth_cap_spin"],
                    max_rel_depth_increase=wp["max_rel_depth_increase_spin"],
                    shallow_damping_depth=wp["shallow_damping_depth_spin"],
                    extreme_rain_mode=wp["extreme_rain_mode_chk"],
                    source_cfl_beta=wp["source_cfl_beta_spin"],
                    source_max_substeps=wp["source_max_substeps_spin"],
                    source_rate_cap=wp["max_source_rate_spin"],
                    source_depth_step_cap=wp["max_source_depth_step_spin"],
                    source_true_subcycling=wp["source_true_subcycling_chk"],
                    source_imex_split=wp["source_imex_split_chk"],
                    enable_shallow_front_recon_fallback=False,
                    gpu_diag_sync_interval_steps=wp["gpu_diag_sync_interval_spin"],
                    tiny_mode=wp["tiny_mode_combo"],
                    tiny_wet_cell_threshold=wp["tiny_wet_cell_threshold_spin"],
                    degen_mode=wp["degen_mode"],
                    front_flux_damping=wp["front_flux_damping_spin"],
                    active_set_hysteresis=wp["active_set_hysteresis_chk"],
                )

            try:
                backend = _build_and_initialize_backend()
            except Exception as init_exc:
                err_l = str(init_exc).lower()
                is_illegal_mem = "illegal memory access" in err_l
                if cuda_graphs_enabled and is_illegal_mem:
                    log_fn(
                        "CUDA solver init failed with illegal memory access while graph replay was enabled; "
                        "retrying once with CUDA graph replay disabled."
                    )
                    cuda_graphs_enabled = False
                    os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "0"
                    backend = _build_and_initialize_backend()
                    log_fn("CUDA graph replay fallback at solver init succeeded.")
                else:
                    raise

            # Sync RCMK inverse permutation from backend to coupling controller.
            # The backend stores _inv_cell_perm after build_mesh (rebuilt above
            # via _build_and_initialize_backend).  Without this sync, structure
            # cell indices would be in original order while solver state is in
            # RCMK-renumbered order on the GPU.
            if coupling_controller is not None:
                _inv_perm = getattr(backend, "_inv_cell_perm", None)
                if _inv_perm is not None and _inv_perm.size > 0:
                    try:
                        coupling_controller._inv_cell_perm = np.asarray(_inv_perm, dtype=np.int32).copy()
                    except Exception as sync_exc:
                        log_fn(f"[WARNING] Failed to sync inv_cell_perm to coupling controller: {sync_exc}")

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
            perf_mode = str(os.environ.get("BACKWATER_SWE2D_PERF_MODE", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
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
            log_fn("Step timing diagnostics enabled (ms): wall, step, coupling, source, state, bc, ui.")
            if perf_mode:
                log_fn(
                    "SWE2D perf mode active: reduced runtime logging and disabled per-step source/boundary forensic accounting."
                )
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
                    if inflow_progressive_chk is not None:
                        progressive = bool(inflow_progressive_chk)
                    node_x_bc = mesh_data["node_x"]
                    node_y_bc = mesh_data["node_y"]
                    node_z_bc = mesh_data["node_z"]
                    native_bc_res = native_bc_cfg.configure(
                        backend=backend,
                        bc_n0=bc_n0,
                        bc_n1=bc_n1,
                        bc_tp=bc_tp,
                        side_hydrographs=side_hydrographs,
                        edge_hydrographs=edge_hydrographs,
                        node_x=node_x_bc,
                        node_y=node_y_bc,
                        node_z=node_z_bc,
                        inflow_q_bc_type=int(_BC_INFLOW_Q),
                        progressive=progressive,
                        ts_flow_code=int(_BC_TS_FLOW),
                        ts_stage_code=int(_BC_TS_STAGE),
                    )
                    if bool(native_bc_res.get("native_bc_forcing", False)):
                        native_bc_forcing = True
                        log_fn(
                            f"Native BC hydrograph forcing configured for {int(native_bc_res.get('configured_edges', 0))} boundary edges."
                        )
                        if bool(native_bc_res.get("progressive_uploaded", False)):
                            log_fn(
                                f"Progressive inflow data uploaded for {int(native_bc_res.get('n_prog_edges', 0))} edges."
                            )
                    elif bool(native_bc_res.get("skipped_progressive", False)):
                        log_fn("Native BC hydrographs skipped: progressive inflow activation is enabled for flow hydrographs.")
                except Exception as exc:
                    log_fn(f"Native BC hydrograph forcing unavailable: {exc}")

            if hasattr(backend, "set_rain_cn_forcing_native"):
                try:
                    if thiessen_forcing is not None:
                        native_rain_res = run_setup_configurator.configure_native_rain_cn_forcing(
                            backend=backend,
                            thiessen_forcing=thiessen_forcing,
                            mm_to_model_depth=float(rain_mm_to_model_depth_fn()),
                        )
                        infil_label = str(native_rain_res.get('infiltration_method', 'scs_cn'))
                    elif float(np.asarray(rain_rate_model, dtype=np.float64)) > 0.0:
                        native_rain_res = run_setup_configurator.configure_constant_rain_rate_native(
                            backend=backend,
                            rate_model_mps=float(np.asarray(rain_rate_model, dtype=np.float64)),
                            mm_to_model_depth=float(rain_mm_to_model_depth_fn()),
                        )
                        infil_label = "constant_rate"
                    else:
                        native_rain_res = {"configured": False}
                        infil_label = "none"
                    if bool(native_rain_res.get("configured", False)):
                        native_rain_cn_forcing = True
                        log_fn(
                            "Native rainfall forcing configured for GPU timestep evaluation "
                            f"(infiltration={infil_label}, "
                            f"groups={int(native_rain_res.get('groups', 0))})."
                        )
                except Exception as exc:
                    log_fn(f"[WARNING] Native rain forcing unavailable: {exc}")

            native_source_injection_mode = hasattr(backend, "set_external_sources_native")
            if native_source_injection_mode:
                try:
                    native_src_res = run_setup_configurator.configure_native_source_injection(backend=backend)
                    native_source_injection_mode = bool(native_src_res.get("native_source_injection_mode", False))
                    if bool(native_src_res.get("configured", False)):
                        log_fn("Native external source injection enabled (device-resident coupling path).")
                except Exception as exc:
                    native_source_injection_mode = False
                    log_fn(f"Native external source injection unavailable: {exc}")

            area_model = np.asarray(mesh_cell_areas_fn(), dtype=np.float64).ravel()
            n_area = int(area_model.size)
            h0_model = np.asarray(h0, dtype=np.float64).ravel()
            n_store = min(n_area, int(h0_model.size))
            storage_start_model = float(np.sum(h0_model[:n_store] * area_model[:n_store])) if n_store > 0 else 0.0
            source_budget_model = {
                "rain": 0.0,
                "cell": 0.0,
                "coupling": 0.0,
            }

            node_x_bc = mesh_data["node_x"]
            node_y_bc = mesh_data["node_y"]
            from swe2d.workbench.services.mesh_service import edge_lengths, mesh_bounds
            edge_len_bc = edge_lengths(node_x_bc, node_y_bc, bc_n0, bc_n1)
            xmin_bc, xmax_bc, ymin_bc, ymax_bc = mesh_bounds(node_x_bc, node_y_bc)
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
                rain_rate_si_to_model_callback=rain_rate_si_to_model_fn,
                internal_flow_source_cms_at_time_callback=internal_flow_source_cms_at_time_fn,
                flow_si_to_model_callback=flow_si_to_model_fn,
                enable_source_volume_accounting=(not perf_mode),
                enable_boundary_flux_accounting=(not perf_mode),
                record_source_step_rows=(not perf_mode),
                record_boundary_flux_step_rows=(not perf_mode),
            )
            source_budget_model = runtime_source_manager.source_budget_model
            source_step_rows_model = runtime_source_manager.source_step_rows_model
            boundary_flux_budget_model = runtime_source_manager.boundary_flux_budget_model
            boundary_flux_step_rows_model = runtime_source_manager.boundary_flux_step_rows_model
            _accumulate_boundary_flux_volume_model = runtime_source_manager.accumulate_boundary_flux_volume_model
            _accumulate_source_volume_model = runtime_source_manager.accumulate_source_volume_model
            _rain_source_for_window = runtime_source_manager.rain_source_for_window
            _cell_source_model_at_time = runtime_source_manager.cell_source_model_at_time

            stage_coupled_imex_requested = bool(
                source_stage_coupled_imex_rk2_chk is not None
                and source_stage_coupled_imex_rk2_chk
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
                    log_fn(
                        "Stage-coupled IMEX-RK2 requested but disabled: "
                        + "; ".join(stage_reasons)
                    )
                else:
                    log_fn("Stage-coupled IMEX-RK2 enabled for external coupling sources.")

            if SWE2DRuntimeStepExecutor is None:
                raise RuntimeError("SWE2DRuntimeStepExecutor seam is unavailable.")
            if SWE2DRuntimeReporter is None:
                raise RuntimeError("SWE2DRuntimeReporter seam is unavailable.")
            runtime_step_executor = SWE2DRuntimeStepExecutor()
            runtime_reporter = SWE2DRuntimeReporter()

            log_fn("The numbers they go UP! They go UP UP UP!!!") # FVM Loop start meme

            loop_result = _execute_run_timestep_loop_runtime_logic(
                wb=view,
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
                cell_solver_z=cell_solver_z,
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
                h_min=wp["h_min_spin"],
                uniform_inflow_velocity=view.get_uniform_inflow_velocity(),
                progress_callback=view.set_run_progress,
                perf_mode=perf_mode,
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
            sim_time_diff = float(t_accum) - float(run_duration_s)
            log_fn(
                "Runtime simulated-time check: "
                f"sim_t={float(t_accum):.6f}s, target={float(run_duration_s):.6f}s, "
                f"delta={sim_time_diff:.6e}s"
            )
            if native_source_injection_mode:
                try:
                    backend.set_external_sources_native(None)
                except Exception:
                    logger.warning("Unexpected error silently caught", exc_info=True)
            _result_data = {
                "h": h,
                "hu": hu,
                "hv": hv,
                "n_mann_cell": n_mann_cell.copy() if n_mann_cell is not None else np.full(h.shape, float(n_mann_spin), dtype=np.float64),
                "gpu_active": np.array(bool(backend.gpu_active())),
                "last_mass_total": np.array(float(last_diag.get("mass_total", -1.0) if last_diag else -1.0)),
            }

            # Sync live snapshot timesteps into the temporal dock slider
            # and overlay so they reflect the current state *before*
            # finalization (which persists to GPKG).
            results_data = getattr(view, "_results_data", None)
            if results_data is not None and hasattr(results_data, "set_live_snapshot_timesteps"):
                try:
                    live_ts = results_data.get_live_snapshot_timesteps()
                    results_data.set_live_snapshot_timesteps(
                        live_ts, t_sec=float(t_accum),
                    )
                except Exception as exc:
                    log_fn(f"[LiveSync] set_live_snapshot_timesteps failed: {exc}")
            try:
                view._sync_high_perf_overlay_data()
                live_ts = results_data.get_live_snapshot_timesteps() if results_data else []
                if live_ts:
                    view._update_high_perf_overlay_time(
                        float(live_ts[-1][0])
                    )
            except Exception as exc:
                log_fn(f"[LiveSync] overlay update failed: {exc}")

            # Push updated timesteps to the temporal dock slider
            temporal = getattr(view, "_temporal_dock", None)
            if temporal is not None and results_data is not None:
                try:
                    temporal.set_data(results_data)
                except Exception as exc:
                    log_fn(f"[LiveSync] temporal dock update failed: {exc}")

            # Feed mesh + result data to plot viewer
            try:
                view._refresh_plot()
            except Exception as exc:
                log_fn(f"[LiveSync] _refresh_plot failed: {exc}")

            save_mesh_results = bool(wp["save_mesh_results_to_gpkg_chk"])
            save_max_only = bool(wp.get("save_max_only_chk", False))
            max_results = backend.get_max_tracking() if (save_mesh_results or save_max_only) else None
            if save_max_only and max_results is not None:
                from swe2d.workbench.services.gpkg_persistence_service import (
                    persist_mesh_max_results_to_geopackage,
                )
                persist_mesh_max_results_to_geopackage(
                    wp.get("results_gpkg_path_edit", ""),
                    run_id,
                    max_results,
                    log_fn=log_fn,
                )
            run_finalizer.finalize_and_persist(
                h=h,
                hu=hu,
                hv=hv,
                final_sim_time_s=float(t_accum),
                n_area=n_area,
                area_model=area_model,
                storage_start_model=storage_start_model,
                source_budget_model=source_budget_model,
                source_step_rows_model=source_step_rows_model,
                run_duration_s=run_duration_s,
                boundary_flux_budget_model=boundary_flux_budget_model,
                boundary_flux_step_rows_model=boundary_flux_step_rows_model,
                run_id=run_id,
                output_interval_s=output_interval_s,
                line_output_interval_s=line_output_interval_s,
                run_perf_start=run_perf_start,
                run_wallclock_start=run_wallclock_start,
                run_log_start_idx=run_log_start_idx,
                thiessen_forcing=thiessen_forcing,
                rain_stats_acc=rain_stats_acc,
                save_line_results=wp["save_line_results_to_gpkg_chk"],
                save_coupling_results=wp["save_coupling_results_to_gpkg_chk"],
                save_mesh_results=save_mesh_results,
                save_run_log=wp["save_run_log_to_gpkg_chk"],
                h_min=wp["h_min_spin"],
            )

            return _result_data
        except Exception as exc:
            run_lifecycle.handle_run_failure(
                exc,
                lambda msg: QtWidgets.QMessageBox.critical(view, "2D SWE", msg),
            )
        finally:
            run_lifecycle.finalize_cleanup(backend)

    # ── Preflight and input collection ─────────────────────────────────

    def _preflight_validate_mesh(self) -> dict:
        """Validate mesh availability and backend readiness before a run.

        Returns a dict with ``mesh_data``, ``ok``, and ``message`` keys.
        """
        view = self._view
        mesh_data = getattr(view, "_mesh_data", None)
        if mesh_data is None:
            return {"mesh_data": None, "ok": False, "message": "Run aborted: no mesh loaded. Import mesh from map layers first."}
        from swe2d.runtime.backend import swe2d_gpu_available
        if not swe2d_gpu_available():
            return {"mesh_data": mesh_data, "ok": False, "message": "Run aborted: GPU backend not available."}
        return {"mesh_data": mesh_data, "ok": True, "message": ""}

    def _collect_bc_for_edges(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> dict:
        """Collect boundary condition arrays for the given boundary edges.

        Returns a dict with ``bc_type``, ``bc_val``, ``edge_hydrographs``,
        and ``edge_group_overrides`` keys.
        """
        view = self._view
        _, _, bc_type, bc_val = view._collect_boundary_arrays()
        edge_hydrographs = view._collect_bc_layer_hydrographs(edge_n0, edge_n1)
        edge_group_overrides = view._collect_bc_layer_edge_groups(edge_n0, edge_n1)
        return {
            "bc_type": bc_type,
            "bc_val": bc_val,
            "edge_hydrographs": edge_hydrographs,
            "edge_group_overrides": edge_group_overrides,
        }

    def _prepare_run_inputs(self) -> dict:
        """Prepare all inputs needed for a simulation run.

        Collects mesh data, boundary conditions, initial state, and
        spatially-varying forcing arrays.  Returns a dict that can be
        unpacked by the run pipeline.
        """
        view = self._view
        mesh_data = getattr(view, "_mesh_data", None)
        if mesh_data is None:
            return {}
        edge_n0, edge_n1 = view._mesh_boundary_edges()
        bc_result = self._collect_bc_for_edges(edge_n0, edge_n1)
        bc_tp = bc_result["bc_type"]
        bc_vl = bc_result["bc_val"]
        side_hydrographs = view._build_side_hydrographs()
        edge_hydrographs = bc_result["edge_hydrographs"]
        edge_group_overrides = bc_result["edge_group_overrides"]
        initial = view._initial_state(bc_n0=edge_n0, bc_n1=edge_n1, bc_tp=bc_tp)
        n_mann_cell = view._build_spatial_manning_array()
        return {
            "mesh_data": mesh_data,
            "edge_n0": edge_n0,
            "edge_n1": edge_n1,
            "bc_type": bc_tp,
            "bc_val": bc_vl,
            "side_hydrographs": side_hydrographs,
            "edge_hydrographs": edge_hydrographs,
            "edge_group_overrides": edge_group_overrides,
            "initial_state": initial,
            "n_mann_cell": n_mann_cell,
        }

    def _collect_simulation_settings(self) -> dict:
        """Collect simulation settings from UI widgets and return a plain dict.

        Includes unit-system information derived from the CRS.
        """
        view = self._view
        wp = view.collect_run_widget_params()
        wp["unit_system_name"] = getattr(view, "_unit_system", "")
        wp["length_unit_name"] = getattr(view, "_length_unit_name", "")
        wp["gravity"] = float(getattr(view, "_gravity", 9.81))
        wp["k_mann"] = float(getattr(view, "_k_mann", 1.0))
        return wp

    # ── Cancel orchestration ──────────────────────────────────────────
    def on_cancel(self) -> None:
        """Mark the current run as cancelled.

        The view owns the cancel flag; the controller just flips it and
        logs the request.
        """
        view = self._view
        view._cancel_requested = True
        view._log("Cancellation requested...")

    # ── Snapshot orchestration ─────────────────────────────────────────
    def on_snapshot(self) -> None:
        """Write captured 2D mesh timesteps to a temporary HEC-RAS HDF file.

        Behaviour matches the legacy ``_on_snapshot``:
        - Aborts (with a log line) when no mesh or no timesteps exist.
        - Writes a snapshot HDF to the system temp directory.
        - Persists the snapshot rows to the active results GeoPackage.
        - Refreshes the high-perf overlay so the new timesteps render.
        """
        from qgis.PyQt import QtWidgets  # local import keeps test runs lean
        import datetime
        import tempfile

        view = self._view
        results_data = getattr(view, "_results_data", None)
        _snapshots = results_data.get_live_snapshot_timesteps() if results_data else []
        if view._mesh_data is None and not _snapshots:
            view._log("No snapshot data available — run the model with an output interval set first.")
            return

        _coupling_rows = results_data.get_live_coupling_snapshot_rows() if results_data else []

        gpkg_results_path = view._current_line_results_storage_path()
        if gpkg_results_path:
            if not getattr(view, "_snapshot_run_id", None):
                view._snapshot_run_id = datetime.datetime.now().astimezone().strftime(
                    "swe2d_snapshot_%Y%m%dT%H%M%S%z"
                )
            snap_run_id = str(view._snapshot_run_id)
        else:
            snap_run_id = ""

        # HDF5 export (mesh only, optional — fails gracefully)
        if view._mesh_data is not None and _snapshots:
            try:
                wp = view.collect_run_widget_params()
                snap_path = os.path.join(tempfile.gettempdir(), "swe2d_snapshot.hdf")
                from swe2d.workbench.services.hecras_export_service import write_hecras_hdf5
                write_hecras_hdf5(
                    path=snap_path,
                    mesh_data=view._mesh_data,
                    length_unit_name=view._length_unit_name,
                    is_us_customary=view._is_us_customary_units(),
                    include_extra=bool(wp["extended_outputs_chk"]),
                    gravity=float(view._gravity),
                    h_min=float(wp["h_min_spin"]),
                    n_mann=float(wp["n_mann_spin"]),
                    timesteps=_snapshots,
                    log_fn=view._log,
                    result_data=view._result_data,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(view, "Snapshot", f"HDF5 write failed:\n{exc}")
                return

    def on_preview_overrides(self) -> None:
        """Compute and display a summary of BC and Manning overrides.

        Generates the mesh on demand, derives default and overridden
        boundary conditions, and presents a summary via QMessageBox.
        Aborts when no boundary edges are present.
        """
        from qgis.PyQt import QtWidgets
        import numpy as np

        view = self._view
        if view._mesh_data is None:
            view._on_generate_mesh()
        if view._mesh_data is None:
            return

        edge_n0, edge_n1 = view._mesh_boundary_edges()
        if edge_n0.size == 0:
            view._log("No boundary edges detected in mesh.")
            QtWidgets.QMessageBox.information(
                view, "Preview Overrides", "No boundary edges detected in mesh."
            )
            return

        bc_type_preview, bc_val_preview, _, _ = view._collect_boundary_arrays()
        bc_type_preview = bc_type_preview.copy()
        bc_val_preview = bc_val_preview.copy()
        edge_hydrographs = view._collect_bc_layer_hydrographs(edge_n0, edge_n1)

        static_mask = (bc_type_preview != bc_type_default) | (
            ~np.isclose(bc_val_preview, bc_val_default)
        )
        static_count = int(np.count_nonzero(static_mask))
        static_type_counts: Dict[str, int] = {}
        if static_count:
            for code in np.unique(bc_type_preview[static_mask]):
                label = view._bc_code_label(int(code))
                static_type_counts[label] = int(
                    np.count_nonzero(bc_type_preview[static_mask] == code)
                )

        mann_arr, mann_applied, mann_total, mann_name = view._preview_spatial_manning()
        if mann_arr is not None and mann_total > 0:
            mann_range = (
                f"{float(np.min(mann_arr)):.5f} to {float(np.max(mann_arr)):.5f}"
            )
        else:
            mann_range = f"{view.get_n_mann_value():.5f}"

        bc_layer_name = "(none)"
        bc_layer = None
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
            details = ", ".join(
                f"{label}={count}"
                for label, count in sorted(static_type_counts.items())
            )
            summary_lines.insert(3, f"Static BC types: {details}")

        summary = "\n".join(summary_lines)
        view._log("Override preview:\n" + summary.replace("\n", " | "))
        QtWidgets.QMessageBox.information(view, "Preview Overrides", summary)

    # ── Load run settings from results GeoPackage ─────────────────────
    def on_load_run_settings_from_results(self) -> None:
        """Open the run-log viewer for the active results GeoPackage.

        Aborts (with log) when the GeoPackage is missing or contains no
        saved run logs.
        """
        view = self._view
        db_path = str(view._current_line_results_storage_path() or "")
        if not db_path or not os.path.exists(db_path):
            view._log(
                "Load run inputs skipped: results GeoPackage not found."
            )
            return
        records = view._load_run_logs_from_geopackage(db_path)
        if not records:
            view._log(
                "Load run inputs skipped: no saved run logs found in selected GeoPackage."
            )
            return
        from swe2d.workbench.dialogs.run_log_viewer_dialog import SWE2DRunLogViewerDialog

        dlg = SWE2DRunLogViewerDialog(
            records=records,
            run_id=view._run_log_latest_run_id,
            db_path=db_path,
            parent=view,
            apply_run_settings_callback=view._apply_run_log_metadata_to_ui,
        )
        dlg.exec()

    def on_preview_coupling(self) -> None:
        """Compute and display a coupling configuration preview.

        Builds pipe network and hydraulic structure configs from widget state,
        validates them (unknown refs, zero capacity, near-zero head), and
        shows a summary via QMessageBox.
        """
        from qgis.PyQt import QtWidgets
        import numpy as np

        view = self._view
        if view._mesh_data is None:
            QtWidgets.QMessageBox.information(
                view,
                "Coupling Preview",
                "Generate or load a mesh first so cell-based coupling "
                "indices can be resolved.",
            )
            return

        pipe_cfg = view._build_pipe_network_config()
        struct_cfg = view._build_hydraulic_structure_config()

        if pipe_cfg is None and struct_cfg is None:
            QtWidgets.QMessageBox.information(
                view,
                "Coupling Preview",
                "No valid drainage or structure layers are configured.",
            )
            return

        lines: list[str] = []

        def _format_id_preview(ids, limit: int = 10) -> str:
            """Format a list of IDs as a comma-separated string, truncated at limit."""
            vals = [str(v) for v in ids if str(v)]
            if not vals:
                return "(none)"
            if len(vals) <= limit:
                return ", ".join(vals)
            return ", ".join(vals[:limit]) + f", ... (+{len(vals) - limit} more)"

        if pipe_cfg is not None:
            lines.append(
                f"Drainage network: nodes={len(pipe_cfg.nodes)}, "
                f"links={len(pipe_cfg.links)}, inlets={len(pipe_cfg.inlets)}"
            )

            node_by_id = {str(n.node_id): n for n in pipe_cfg.nodes}
            unknown_link_refs: list[str] = []
            unknown_inlet_refs: list[str] = []
            zero_capacity_links: list[str] = []
            near_zero_head_links: list[str] = []
            t0_probably_zero_links: list[str] = []

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
            lines.append(
                f"- unknown link node refs: {len(unknown_link_refs)}"
            )
            if unknown_link_refs:
                lines.append(
                    f"  IDs: {_format_id_preview(unknown_link_refs)}"
                )
            lines.append(
                f"- unknown inlet node refs: {len(unknown_inlet_refs)}"
            )
            if unknown_inlet_refs:
                lines.append(
                    f"  IDs: {_format_id_preview(unknown_inlet_refs)}"
                )
            lines.append(
                f"- links with zero hydraulic capacity fields: "
                f"{len(zero_capacity_links)}"
            )
            if zero_capacity_links:
                lines.append(
                    f"  IDs: {_format_id_preview(zero_capacity_links)}"
                )
            lines.append(
                f"- links with near-zero initial head gradient "
                f"(|dh0|<=1e-4): {len(near_zero_head_links)}"
            )
            if near_zero_head_links:
                lines.append(
                    f"  IDs: {_format_id_preview(near_zero_head_links)}"
                )
            lines.append(
                f"- links likely zero-flow at t0 "
                f"(capacity/head limits): {len(t0_probably_zero_links)}"
            )
        else:
            lines.append("Drainage network: not configured")

        if struct_cfg is not None:
            lines.append(
                f"Hydraulic structures: count={len(struct_cfg.structures)}"
            )
        else:
            lines.append("Hydraulic structures: not configured")

        try:
            from swe2d.runtime.coupling import pack_coupling_soa
        except ImportError:
            pack_coupling_soa = None

        if pack_coupling_soa is not None:
            try:
                soa = pack_coupling_soa(
                    n_cells=int(view._mesh_cell_areas().shape[0]),
                    pipe_network=pipe_cfg,
                    hydraulic_structures=struct_cfg,
                )
                if soa.drainage is not None:
                    dn = soa.drainage
                    invalid_links = int(
                        np.sum((dn.link_from < 0) | (dn.link_to < 0))
                    )
                    invalid_inlets = int(
                        np.sum((dn.inlet_cell < 0) | (dn.inlet_node < 0))
                    )
                    lines.append(
                        "Drainage SoA: "
                        f"nodes={dn.node_x.size}, "
                        f"links={dn.link_from.size}, "
                        f"inlets={dn.inlet_cell.size}, "
                        f"invalid_links={invalid_links}, "
                        f"invalid_inlets={invalid_inlets}"
                    )
                if soa.structures is not None:
                    ss = soa.structures
                    invalid_struct = int(
                        np.sum(
                            (ss.upstream_cell < 0)
                            | (ss.downstream_cell < 0)
                        )
                    )
                    lines.append(
                        "Structures SoA: "
                        f"count={ss.structure_type.size}, "
                        f"invalid_cell_pairs={invalid_struct}"
                    )
            except Exception as exc:
                lines.append(f"SoA packing failed: {exc}")

        QtWidgets.QMessageBox.information(
            view, "Coupling Preview", "\n".join(lines)
        )
