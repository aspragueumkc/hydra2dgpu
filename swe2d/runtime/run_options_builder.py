#!/usr/bin/env python3
"""Run options assembly builder for SWE2D workbench.

Phase 5 goal: extract solver/runtime option assembly from `_on_run`
into a focused, testable module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

from swe2d.extensions.extension_models import StructureType


@dataclass(frozen=True)
class SWE2DRunOptionsData:
    """Structured runtime options payload used by run execution."""

    run_duration_s: float
    dt_cfg: float
    adaptive_cfl_dt: bool
    dt_fixed: float
    dt_request: float
    initial_dt: float
    reconstruction_mode: int
    reconstruction_name: str
    temporal_order_value: int
    temporal_scheme: Any
    temporal_scheme_name: str
    solver_backend_mode: str
    openmp_enabled: bool
    godunov_mode: Any
    coupling_loop_mode: str
    drainage_solver_backend_mode: str
    drainage_gpu_method_mode: str
    culvert_solver_mode: int
    cuda_graphs_enabled: bool
    equation_set: Any
    experimental_3d_enabled: bool
    coupling_mode_3d: int
    model_options: Any
    swe3d_env_overrides: Dict[str, str]
    rain_rate_model: Any
    internal_flow_forcing: Any
    cell_source_si: Any
    cell_source_model: Any
    thiessen_forcing: Any
    pipe_network_cfg: Any
    hydraulic_structures_cfg: Any
    bridge_cuda_coupling: bool
    bridge_stacked_coupling_mode: str


class SWE2DRunOptionsBuilder:
    """Builds run options data by invoking dialog-provided callbacks."""

    def __init__(
        self,
        ui: Any,
        log_callback: Callable[[str], None],
        parse_run_duration_seconds_callback: Callable[[], float],
        collect_3d_patch_env_overrides_callback: Callable[[], Dict[str, str]],
        rain_rate_si_to_model_callback: Callable[[float], Any],
        build_internal_flow_forcing_callback: Callable[[], Any],
        internal_flow_source_cms_at_time_callback: Callable[[Any, float], Any],
        flow_si_to_model_callback: Callable[[Any], Any],
        build_thiessen_rain_cn_forcing_callback: Callable[[], Any],
        build_pipe_network_config_callback: Callable[[], Any],
        build_hydraulic_structure_config_callback: Callable[[], Any],
        swe2d_gpu_available_callback: Callable[[], bool],
        temporal_scheme_enum: Any,
        spatial_discretization_enum: Any,
        godunov_solver_mode_enum: Any,
        solver_model_options_cls: Any,
        swe2d_equation_set_enum: Any,
        swe2d_3d_solver_model_enum: Any,
        swe2d_3d_coupling_mode_enum: Any,
    ):
        self._ui = ui
        self._log = log_callback
        self._parse_run_duration_seconds = parse_run_duration_seconds_callback
        self._collect_3d_patch_env_overrides = collect_3d_patch_env_overrides_callback
        self._rain_rate_si_to_model = rain_rate_si_to_model_callback
        self._build_internal_flow_forcing = build_internal_flow_forcing_callback
        self._internal_flow_source_cms_at_time = internal_flow_source_cms_at_time_callback
        self._flow_si_to_model = flow_si_to_model_callback
        self._build_thiessen_rain_cn_forcing = build_thiessen_rain_cn_forcing_callback
        self._build_pipe_network_config = build_pipe_network_config_callback
        self._build_hydraulic_structure_config = build_hydraulic_structure_config_callback
        self._swe2d_gpu_available = swe2d_gpu_available_callback
        self._TemporalScheme = temporal_scheme_enum
        self._SpatialDiscretization = spatial_discretization_enum
        self._GodunovSolverMode = godunov_solver_mode_enum
        self._SolverModelOptions = solver_model_options_cls
        self._SWE2DEquationSet = swe2d_equation_set_enum
        self._SWE2DThreeDSolverModel = swe2d_3d_solver_model_enum
        self._SWE2DThreeDCouplingMode = swe2d_3d_coupling_mode_enum

    @staticmethod
    def _has_bridge_structures(hydraulic_structures_cfg: Any) -> bool:
        structures = getattr(hydraulic_structures_cfg, "structures", None)
        if not structures:
            return False
        for structure in structures:
            if int(getattr(structure, "structure_type", 0)) == int(StructureType.BRIDGE):
                return True
        return False

    def build(self) -> SWE2DRunOptionsData:
        run_duration_s = self._parse_run_duration_seconds()
        dt_cfg = float(self._ui.dt_spin.value())
        adaptive_cfl_dt = bool(self._ui.adaptive_cfl_dt_chk.isChecked())
        dt_fixed = -1.0 if adaptive_cfl_dt else dt_cfg
        dt_request = -1.0 if adaptive_cfl_dt else dt_cfg
        initial_dt = float(self._ui.initial_dt_spin.value()) if hasattr(self._ui, "initial_dt_spin") else 0.0

        reconstruction_mode = int(self._ui.reconstruction_combo.currentData())
        reconstruction_name = self._ui.reconstruction_combo.currentText().strip()
        temporal_order_value = int(self._ui.temporal_order_combo.currentData())
        temporal_scheme = self._TemporalScheme(temporal_order_value)
        temporal_scheme_name = self._ui.temporal_order_combo.currentText().strip()

        godunov_mode_value = int(self._ui.godunov_mode_combo.currentData()) if hasattr(self._ui, "godunov_mode_combo") else int(self._GodunovSolverMode.CURRENT_GPU_STEP)
        godunov_mode = self._GodunovSolverMode(godunov_mode_value)
        if godunov_mode == self._GodunovSolverMode.GODUNOV_ROLLOUT:
            promoted_temporal = max(temporal_order_value, int(self._TemporalScheme.SSP_RK2))
            promoted_reconstruction = max(reconstruction_mode, int(self._SpatialDiscretization.FV_MUSCL_MINMOD))
            if promoted_temporal != temporal_order_value:
                self._log("Godunov rollout selected: promoting temporal integration to RK2.")
                temporal_order_value = promoted_temporal
                temporal_scheme = self._TemporalScheme(temporal_order_value)
                idx_t = self._ui.temporal_order_combo.findData(temporal_order_value)
                if idx_t >= 0:
                    temporal_scheme_name = self._ui.temporal_order_combo.itemText(idx_t).strip()
            if promoted_reconstruction != reconstruction_mode:
                self._log("Godunov rollout selected: promoting reconstruction to MUSCL MinMod.")
                reconstruction_mode = promoted_reconstruction
                idx_r = self._ui.reconstruction_combo.findData(reconstruction_mode)
                if idx_r >= 0:
                    reconstruction_name = self._ui.reconstruction_combo.itemText(idx_r).strip()

        openmp_enabled = bool(
            not hasattr(self._ui, "solver_openmp_enabled_chk")
            or getattr(self._ui, "solver_openmp_enabled_chk", None) is None
            or self._ui.solver_openmp_enabled_chk.isChecked()
        )
        os.environ["BACKWATER_SWE2D_OPENMP"] = "1" if openmp_enabled else "0"

        def _gpu_available_for_selected_module() -> bool:
            if self._swe2d_gpu_available is None:
                return False
            try:
                return bool(self._swe2d_gpu_available(openmp_enabled=openmp_enabled))
            except TypeError:
                try:
                    return bool(self._swe2d_gpu_available())
                except Exception:
                    return False
            except Exception:
                return False

        solver_backend_mode = str(
            self._ui.solver_backend_combo.currentData()
            if hasattr(self._ui, "solver_backend_combo")
            else "gpu"
        ).strip().lower()
        if solver_backend_mode not in {"cpu", "gpu"}:
            solver_backend_mode = "gpu"
        if solver_backend_mode == "gpu" and not _gpu_available_for_selected_module():
            self._log("GPU solver backend selected but CUDA is unavailable; falling back to CPU solver backend.")
            solver_backend_mode = "cpu"

        coupling_loop_mode = str(self._ui.coupling_loop_combo.currentData() if hasattr(self._ui, "coupling_loop_combo") else "cpu")
        drainage_solver_backend_mode = str(self._ui.drainage_backend_combo.currentData() if hasattr(self._ui, "drainage_backend_combo") else "cpu")
        drainage_gpu_method_mode = str(self._ui.drainage_gpu_method_combo.currentData() if hasattr(self._ui, "drainage_gpu_method_combo") else "step")
        culvert_solver_mode = int(self._ui.culvert_solver_mode_combo.currentData() if hasattr(self._ui, "culvert_solver_mode_combo") else 0)

        if solver_backend_mode == "cpu":
            if str(coupling_loop_mode).strip().lower() != "cpu":
                self._log("CPU solver backend selected: forcing coupling loop to CPU mode.")
            if str(drainage_solver_backend_mode).strip().lower() != "cpu":
                self._log("CPU solver backend selected: forcing drainage backend to CPU mode.")
            coupling_loop_mode = "cpu"
            drainage_solver_backend_mode = "cpu"

        cuda_graphs_enabled = bool(getattr(self._ui, "enable_cuda_graphs_chk", None) and self._ui.enable_cuda_graphs_chk.isChecked())
        if (
            cuda_graphs_enabled
            and str(solver_backend_mode).strip().lower() == "gpu"
            and int(temporal_order_value) >= 4
            and str(coupling_loop_mode).strip().lower() == "cuda"
            and str(drainage_solver_backend_mode).strip().lower() == "gpu"
        ):
            cuda_graphs_enabled = True
            self._log(
                "CUDA graph replay auto-disabled for higher-order RK + CUDA drainage/coupling runtime "
                "to avoid illegal memory access."
            )
        os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "1" if cuda_graphs_enabled else "0"

        equation_set_value = int(self._ui.equation_set_combo.currentData()) if hasattr(self._ui, "equation_set_combo") else 0
        equation_set = self._SWE2DEquationSet(equation_set_value) if self._SWE2DEquationSet is not None else equation_set_value
        experimental_3d_enabled = bool(
            hasattr(self._ui, "experimental_3d_mode_chk")
            and self._ui.experimental_3d_mode_chk is not None
            and self._ui.experimental_3d_mode_chk.isChecked()
        )
        coupling_mode_3d = int(self._SWE2DThreeDCouplingMode.OFF) if self._SWE2DThreeDCouplingMode is not None else 0
        if experimental_3d_enabled:
            coupling_mode_3d = int(self._ui._experimental_3d_selected_coupling_mode())

        model_options = None
        if self._SolverModelOptions is None or self._SpatialDiscretization is None:
            if experimental_3d_enabled:
                raise RuntimeError(
                    "Experimental 3D mode requested, but solver enum bindings are unavailable "
                    "(SolverModelOptions/SpatialDiscretization import failure)."
                )
        else:
            model_options = self._SolverModelOptions(
                temporal_scheme=temporal_scheme,
                spatial_discretization=self._SpatialDiscretization(reconstruction_mode),
                godunov_mode=godunov_mode,
                equation_set=equation_set,
            )
            if experimental_3d_enabled:
                if self._SWE2DThreeDSolverModel is None or self._SWE2DThreeDCouplingMode is None:
                    raise RuntimeError("Experimental 3D mode requested but 3D enum support is unavailable in this build.")
                if str(solver_backend_mode).strip().lower() != "gpu":
                    raise RuntimeError("Experimental 3D mode requires the solver backend set to GPU.")
                if not _gpu_available_for_selected_module():
                    raise RuntimeError("Experimental 3D mode requires CUDA GPU availability.")
                if self._SWE2DEquationSet is not None and equation_set != self._SWE2DEquationSet.HYDROSTATIC_2D:
                    self._log(
                        "Experimental 3D mode overrides equation set to Hydrostatic 2D for scaffold validation."
                    )
                    equation_set = self._SWE2DEquationSet.HYDROSTATIC_2D
                    model_options.equation_set = equation_set
                model_options.three_d_solver_model = self._SWE2DThreeDSolverModel.SINGLE_PHASE_FREE_SURFACE_VOF
                model_options.coupling_mode = self._SWE2DThreeDCouplingMode(coupling_mode_3d)
                model_options.enforce_gpu_only_advanced_modes = True
                model_options.three_d_single_phase_free_surface = True

        swe3d_env_overrides: Dict[str, str] = {}
        if experimental_3d_enabled:
            swe3d_env_overrides = self._collect_3d_patch_env_overrides()
            self._log(
                "3D patch config override: "
                f"target_face_len=({swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_FACE_LEN_X')}, "
                f"{swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_FACE_LEN_Y')}, "
                f"{swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_FACE_LEN_Z')}) "
                f"nx={swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_NX')} "
                f"ny={swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_NY')} "
                f"nz={swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_NZ')} "
                f"origin=({swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_ORIGIN_X')}, "
                f"{swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_ORIGIN_Y')}, "
                f"{swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_ORIGIN_Z')}) "
                f"dxyz=({swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_DX')}, "
                f"{swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_DY')}, "
                f"{swe3d_env_overrides.get('BACKWATER_SWE3D_PATCH_DZ')})"
            )
            self._log(
                "3D patch physics config: "
                f"gravity_z_sign={swe3d_env_overrides.get('BACKWATER_SWE3D_GRAVITY_Z_SIGN')} "
                f"bed_drag={swe3d_env_overrides.get('BACKWATER_SWE3D_ENABLE_BED_DRAG')} "
                f"bed_manning_n={swe3d_env_overrides.get('BACKWATER_SWE3D_BED_MANNING_N')} "
                f"bed_drag_h_ref={swe3d_env_overrides.get('BACKWATER_SWE3D_BED_DRAG_HREF')} "
                f"bed_drag_layers={swe3d_env_overrides.get('BACKWATER_SWE3D_BED_DRAG_LAYERS')}"
            )
            self._log(
                "3D patch face BC modes: "
                f"{self._ui._summarize_3d_patch_face_bc_modes(swe3d_env_overrides)}"
            )

        swe2d_perf_mode_enabled = bool(
            getattr(self._ui, "swe2d_perf_mode_chk", None)
            and self._ui.swe2d_perf_mode_chk.isChecked()
        )
        swe3d_env_overrides["BACKWATER_SWE2D_PERF_MODE"] = "1" if swe2d_perf_mode_enabled else "0"

        rain_rate_model = self._rain_rate_si_to_model(float(self._ui.rain_rate_spin.value()) / 1000.0 / 3600.0)
        internal_flow_forcing = self._build_internal_flow_forcing()
        cell_source_si = self._internal_flow_source_cms_at_time(internal_flow_forcing, 0.0)
        cell_source_model = self._flow_si_to_model(cell_source_si) if cell_source_si is not None else None
        thiessen_forcing = self._build_thiessen_rain_cn_forcing()
        pipe_network_cfg = self._build_pipe_network_config()
        hydraulic_structures_cfg = self._build_hydraulic_structure_config()
        bridge_cuda_coupling = (
            str(solver_backend_mode).strip().lower() == "gpu"
            and _gpu_available_for_selected_module()
            and self._has_bridge_structures(hydraulic_structures_cfg)
        )
        bridge_stacked_coupling_mode = str(
            self._ui.bridge_stacked_coupling_mode_combo.currentData()
            if hasattr(self._ui, "bridge_stacked_coupling_mode_combo")
            else "phase3_spatial"
        ).strip().lower()
        if bridge_stacked_coupling_mode not in {"legacy_scalar", "phase3_spatial"}:
            bridge_stacked_coupling_mode = "phase3_spatial"
        if bridge_cuda_coupling:
            self._log(
                "Bridge coupling CUDA helper enabled for hydraulic structures with bridge entries "
                f"(stacked mode={bridge_stacked_coupling_mode})."
            )

        return SWE2DRunOptionsData(
            run_duration_s=run_duration_s,
            dt_cfg=dt_cfg,
            adaptive_cfl_dt=adaptive_cfl_dt,
            dt_fixed=dt_fixed,
            dt_request=dt_request,
            initial_dt=initial_dt,
            reconstruction_mode=reconstruction_mode,
            reconstruction_name=reconstruction_name,
            temporal_order_value=temporal_order_value,
            temporal_scheme=temporal_scheme,
            temporal_scheme_name=temporal_scheme_name,
            solver_backend_mode=solver_backend_mode,
            openmp_enabled=openmp_enabled,
            godunov_mode=godunov_mode,
            coupling_loop_mode=coupling_loop_mode,
            drainage_solver_backend_mode=drainage_solver_backend_mode,
            drainage_gpu_method_mode=drainage_gpu_method_mode,
            culvert_solver_mode=culvert_solver_mode,
            cuda_graphs_enabled=cuda_graphs_enabled,
            equation_set=equation_set,
            experimental_3d_enabled=experimental_3d_enabled,
            coupling_mode_3d=coupling_mode_3d,
            model_options=model_options,
            swe3d_env_overrides=swe3d_env_overrides,
            rain_rate_model=rain_rate_model,
            internal_flow_forcing=internal_flow_forcing,
            cell_source_si=cell_source_si,
            cell_source_model=cell_source_model,
            thiessen_forcing=thiessen_forcing,
            pipe_network_cfg=pipe_network_cfg,
            hydraulic_structures_cfg=hydraulic_structures_cfg,
            bridge_cuda_coupling=bridge_cuda_coupling,
            bridge_stacked_coupling_mode=bridge_stacked_coupling_mode,
        )
