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
    coupling_loop_mode: str
    drainage_solver_backend_mode: str
    drainage_gpu_method_mode: str
    culvert_solver_mode: int
    cuda_graphs_enabled: bool
    model_options: Any
    rain_rate_model: Any
    internal_flow_forcing: Any
    cell_source_si: Any
    cell_source_model: Any
    thiessen_forcing: Any
    pipe_network_cfg: Any
    hydraulic_structures_cfg: Any
    bridge_cuda_coupling: bool
    bridge_stacked_coupling_mode: str
    culvert_face_flux_mode: str


class SWE2DRunOptionsBuilder:
    """Builds run options data by invoking dialog-provided callbacks."""

    def __init__(
        self,
        ui: Any,
        log_callback: Callable[[str], None],
        parse_run_duration_seconds_callback: Callable[[], float],
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
        solver_model_options_cls: Any,
    ):
        self._ui = ui
        self._log = log_callback
        self._parse_run_duration_seconds = parse_run_duration_seconds_callback
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
        self._SolverModelOptions = solver_model_options_cls

    @staticmethod
    def _has_bridge_structures(hydraulic_structures_cfg: Any) -> bool:
        """has bridge structures."""
        structures = getattr(hydraulic_structures_cfg, "structures", None)
        if not structures:
            return False
        for structure in structures:
            if int(getattr(structure, "structure_type", 0)) == int(StructureType.BRIDGE):
                return True
        return False

    def build(
        self,
        *,
        dt: float,
        adaptive_cfl_dt: bool,
        initial_dt: float,
        dt_fixed: float = None,
        dt_request: float = None,
        reconstruction_mode: int,
        reconstruction_name: str,
        temporal_order_value: int,
        temporal_scheme_name: str,
        drainage_gpu_method: str = "step",
        culvert_solver_mode: int = 0,
        cuda_graphs_enabled: bool = False,
        swe2d_perf_mode: bool = False,
        rain_rate_mmhr: float = 0.0,
        bridge_coupling_mode: str = "phase3_spatial",
        culvert_face_flux: bool = False,
    ) -> SWE2DRunOptionsData:
        """build."""
        run_duration_s = self._parse_run_duration_seconds()
        dt_fixed = -1.0 if adaptive_cfl_dt else dt if dt_fixed is None else dt_fixed
        dt_request = -1.0 if adaptive_cfl_dt else dt if dt_request is None else dt_request
        temporal_scheme = self._TemporalScheme(temporal_order_value)
        def _gpu_available_for_selected_module() -> bool:
            """gpu available for selected module."""
            if self._swe2d_gpu_available is None:
                return False
            try:
                return bool(self._swe2d_gpu_available())
            except Exception as exc:
                self._log("[BACKEND] GPU availability check failed: " + str(exc))
                return False

        solver_backend_mode = "gpu"
        if not _gpu_available_for_selected_module():
            raise RuntimeError("CUDA GPU is required but unavailable or check failed.")

        coupling_loop_mode = "cuda"
        drainage_solver_backend_mode = "gpu"
        drainage_gpu_method_mode = drainage_gpu_method
        if (
            cuda_graphs_enabled
            and int(temporal_order_value) >= 4
            and str(coupling_loop_mode).strip().lower() == "cuda"
            and str(drainage_solver_backend_mode).strip().lower() == "gpu"
        ):
            cuda_graphs_enabled = False
        os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "1" if cuda_graphs_enabled else "0"

        model_options = None
        if self._SolverModelOptions is not None and self._SpatialDiscretization is not None:
            model_options = self._SolverModelOptions(
                temporal_scheme=temporal_scheme,
                spatial_discretization=self._SpatialDiscretization(reconstruction_mode),
            )

        swe2d_perf_mode_enabled = swe2d_perf_mode
        if swe2d_perf_mode_enabled:
            os.environ["BACKWATER_SWE2D_PERF_MODE"] = "1"
        else:
            os.environ["BACKWATER_SWE2D_PERF_MODE"] = "0"

        rain_rate_model = self._rain_rate_si_to_model(rain_rate_mmhr / 1000.0 / 3600.0)
        internal_flow_forcing = self._build_internal_flow_forcing()
        cell_source_si = self._internal_flow_source_cms_at_time(internal_flow_forcing, 0.0)
        cell_source_model = self._flow_si_to_model(cell_source_si) if cell_source_si is not None else None
        thiessen_forcing = self._build_thiessen_rain_cn_forcing()
        pipe_network_cfg = self._build_pipe_network_config()
        hydraulic_structures_cfg = self._build_hydraulic_structure_config()
        bridge_cuda_coupling = (
            _gpu_available_for_selected_module()
            and self._has_bridge_structures(hydraulic_structures_cfg)
        )
        bridge_stacked_coupling_mode = bridge_coupling_mode
        if bridge_stacked_coupling_mode not in {"legacy_scalar", "phase3_spatial"}:
            bridge_stacked_coupling_mode = "phase3_spatial"
        if bridge_cuda_coupling:
            self._log(
                "Bridge coupling CUDA helper enabled for hydraulic structures with bridge entries "
                f"(stacked mode={bridge_stacked_coupling_mode})."
            )

        culvert_face_flux_mode = "off"
        if culvert_face_flux and str(coupling_loop_mode).strip().lower() == "cuda":
            culvert_face_flux_mode = "face_flux"
            self._log(
                "Culvert face-based flux coupling enabled. "
                "Culvert flows will be applied as FVM face fluxes with "
                "momentum transfer."
            )

        return SWE2DRunOptionsData(
            run_duration_s=run_duration_s,
            dt_cfg=dt,
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
            coupling_loop_mode=coupling_loop_mode,
            drainage_solver_backend_mode=drainage_solver_backend_mode,
            drainage_gpu_method_mode=drainage_gpu_method_mode,
            culvert_solver_mode=culvert_solver_mode,
            cuda_graphs_enabled=cuda_graphs_enabled,
            model_options=model_options,
            rain_rate_model=rain_rate_model,
            internal_flow_forcing=internal_flow_forcing,
            cell_source_si=cell_source_si,
            cell_source_model=cell_source_model,
            thiessen_forcing=thiessen_forcing,
            pipe_network_cfg=pipe_network_cfg,
            hydraulic_structures_cfg=hydraulic_structures_cfg,
            bridge_cuda_coupling=bridge_cuda_coupling,
            bridge_stacked_coupling_mode=bridge_stacked_coupling_mode,
            culvert_face_flux_mode=culvert_face_flux_mode,
        )
