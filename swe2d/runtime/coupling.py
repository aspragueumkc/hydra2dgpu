"""Coupling orchestration for SWE2D surface, drainage network, and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
from swe2d.extensions.extension_models import (
    DrainageSolverMode,
    HydraulicStructureConfig,
    StructureType,
    PipeNetworkConfig,
    equivalent_circular_diameter_from_area,
)
from swe2d.runtime.bridge_stacked_runtime import (
    apply_bridge_stacked_phase3_source_weight,
    apply_bridge_stacked_source_weight,
)
from swe2d.extensions.structures import SWE2DStructureModule


@dataclass
class SWE2DCouplingDiagnostics:
    time_s: float = 0.0
    dt_s: float = 0.0
    drainage_max_node_depth: float = 0.0
    drainage_max_link_flow: float = 0.0
    structure_total_flow: float = 0.0
    source_sum: float = 0.0
    source_min: float = 0.0
    source_max: float = 0.0
    component_sums: Dict[str, float] = field(default_factory=dict)

    @property
    def drainage_max_node_depth_m(self) -> float:
        return self.drainage_max_node_depth

    @property
    def drainage_max_link_flow_cms(self) -> float:
        return self.drainage_max_link_flow

    @property
    def structure_total_flow_cms(self) -> float:
        return self.structure_total_flow

    @property
    def source_sum_mps(self) -> float:
        return self.source_sum

    @property
    def source_min_mps(self) -> float:
        return self.source_min

    @property
    def source_max_mps(self) -> float:
        return self.source_max

    @property
    def component_sums_mps(self) -> Dict[str, float]:
        return self.component_sums


@dataclass
class SWE2DDrainageSoA:
    node_x: np.ndarray
    node_y: np.ndarray
    node_invert_elev: np.ndarray
    node_max_depth: np.ndarray
    node_surface_area: np.ndarray
    link_from: np.ndarray
    link_to: np.ndarray
    link_length: np.ndarray
    link_roughness_n: np.ndarray
    link_diameter: np.ndarray
    link_max_flow: np.ndarray
    link_cd: np.ndarray
    inlet_cell: np.ndarray
    inlet_node: np.ndarray
    inlet_crest_elev: np.ndarray
    inlet_width: np.ndarray
    inlet_coefficient: np.ndarray
    inlet_max_capture: np.ndarray
    outfall_cell: np.ndarray
    outfall_node: np.ndarray
    outfall_invert_elev: np.ndarray
    outfall_diameter: np.ndarray
    outfall_coefficient: np.ndarray
    outfall_max_flow: np.ndarray
    outfall_zero_storage: np.ndarray
    pipe_end_cell: np.ndarray
    pipe_end_node: np.ndarray
    pipe_end_invert_elev: np.ndarray
    pipe_end_diameter: np.ndarray
    pipe_end_area: np.ndarray
    pipe_end_inlet_loss_k: np.ndarray
    pipe_end_outlet_loss_k: np.ndarray
    solver_mode: int = int(DrainageSolverMode.EGL)


@dataclass
class SWE2DStructuresSoA:
    structure_type: np.ndarray
    upstream_cell: np.ndarray
    downstream_cell: np.ndarray
    crest_elev: np.ndarray
    width: np.ndarray
    height: np.ndarray
    diameter: np.ndarray
    length: np.ndarray
    roughness_n: np.ndarray
    coeff: np.ndarray
    cd: np.ndarray
    opening: np.ndarray
    q_pump: np.ndarray
    max_flow: np.ndarray
    culvert_code: np.ndarray
    culvert_shape: np.ndarray
    culvert_rise: np.ndarray
    culvert_span: np.ndarray
    culvert_area_m2: np.ndarray
    culvert_barrels: np.ndarray
    culvert_slope: np.ndarray
    inlet_invert_elev: np.ndarray
    outlet_invert_elev: np.ndarray
    entrance_loss_k: np.ndarray
    exit_loss_k: np.ndarray
    embankment_enabled: np.ndarray
    embankment_crest_elev: np.ndarray
    embankment_overflow_width: np.ndarray
    embankment_weir_coeff: np.ndarray


@dataclass
class SWE2DCouplingSoA:
    n_cells: int
    drainage: Optional[SWE2DDrainageSoA] = None
    structures: Optional[SWE2DStructuresSoA] = None


def pack_pipe_network_soa(cfg: Optional[PipeNetworkConfig], n_cells: int) -> Optional[SWE2DDrainageSoA]:
    if cfg is None or not cfg.enabled:
        return None
    if not cfg.nodes:
        return None

    node_idx = {n.node_id: i for i, n in enumerate(cfg.nodes)}
    nn = len(cfg.nodes)
    nl = len(cfg.links)
    ni = len(cfg.inlets)
    no = len(cfg.outfalls)
    np_end = len(getattr(cfg, "pipe_ends", []))

    node_x = np.zeros(nn, dtype=np.float64)
    node_y = np.zeros(nn, dtype=np.float64)
    node_invert_elev = np.zeros(nn, dtype=np.float64)
    node_max_depth = np.zeros(nn, dtype=np.float64)
    node_surface_area = np.zeros(nn, dtype=np.float64)
    for i, nd in enumerate(cfg.nodes):
        node_x[i] = float(nd.x)
        node_y[i] = float(nd.y)
        node_invert_elev[i] = float(nd.invert_elev)
        node_max_depth[i] = float(nd.max_depth)
        node_surface_area[i] = float(nd.metadata.get("surface_area", nd.metadata.get("surface_area_m2", 50.0)))

    link_from = np.full(nl, -1, dtype=np.int32)
    link_to = np.full(nl, -1, dtype=np.int32)
    link_length = np.zeros(nl, dtype=np.float64)
    link_roughness_n = np.zeros(nl, dtype=np.float64)
    link_diameter = np.zeros(nl, dtype=np.float64)
    link_max_flow = np.full(nl, np.nan, dtype=np.float64)
    link_cd = np.zeros(nl, dtype=np.float64)
    for i, lk in enumerate(cfg.links):
        link_from[i] = int(node_idx.get(lk.from_node_id, -1))
        link_to[i] = int(node_idx.get(lk.to_node_id, -1))
        link_length[i] = float(lk.length)
        link_roughness_n[i] = float(lk.roughness_n)
        d_link = float(lk.diameter or 0.0)
        if d_link <= 0.0:
            area_link = float(lk.metadata.get("area_m2", 0.0) or 0.0)
            d_link = equivalent_circular_diameter_from_area(area_link)
        link_diameter[i] = d_link
        link_max_flow[i] = np.nan if lk.max_flow is None else float(lk.max_flow)
        link_cd[i] = float(lk.metadata.get("cd", 0.75))

    inlet_cell = np.full(ni, -1, dtype=np.int32)
    inlet_node = np.full(ni, -1, dtype=np.int32)
    inlet_crest_elev = np.zeros(ni, dtype=np.float64)
    inlet_width = np.zeros(ni, dtype=np.float64)
    inlet_coefficient = np.zeros(ni, dtype=np.float64)
    inlet_max_capture = np.full(ni, np.nan, dtype=np.float64)
    for i, it in enumerate(cfg.inlets):
        ci = int(it.cell_id)
        inlet_cell[i] = ci if 0 <= ci < int(n_cells) else -1
        inlet_node[i] = int(node_idx.get(it.node_id, -1))
        inlet_crest_elev[i] = float(it.crest_elev)
        # Keep support for both alias fields (width/coefficient) and canonical
        # fields (length/coeff_orifice) when aliases are intentionally unset.
        width_val = getattr(it, "width", None)
        if width_val is None:
            width_val = getattr(it, "length", 0.0)
        coeff_val = getattr(it, "coefficient", None)
        if coeff_val is None:
            coeff_val = getattr(it, "coeff_orifice", 0.0)
        inlet_width[i] = float(width_val)
        inlet_coefficient[i] = float(coeff_val)
        inlet_max_capture[i] = np.nan if it.max_capture is None else float(it.max_capture)

    outfall_cell = np.full(no, -1, dtype=np.int32)
    outfall_node = np.full(no, -1, dtype=np.int32)
    outfall_invert_elev = np.zeros(no, dtype=np.float64)
    outfall_diameter = np.zeros(no, dtype=np.float64)
    outfall_coefficient = np.zeros(no, dtype=np.float64)
    outfall_max_flow = np.full(no, np.nan, dtype=np.float64)
    outfall_zero_storage = np.zeros(no, dtype=np.int32)
    for i, ot in enumerate(cfg.outfalls):
        ci = int(ot.cell_id)
        outfall_cell[i] = ci if 0 <= ci < int(n_cells) else -1
        outfall_node[i] = int(node_idx.get(ot.node_id, -1))
        outfall_invert_elev[i] = float(ot.invert_elev)
        d_out = float(ot.diameter)
        if d_out <= 0.0:
            d_out = equivalent_circular_diameter_from_area(float(getattr(ot, "area_m2", 0.0) or 0.0))
        outfall_diameter[i] = d_out
        outfall_coefficient[i] = float(ot.coefficient)
        outfall_max_flow[i] = np.nan if ot.max_flow is None else float(ot.max_flow)
        outfall_zero_storage[i] = 1 if bool(getattr(ot, "zero_storage", False)) else 0

    pipe_end_cell = np.full(np_end, -1, dtype=np.int32)
    pipe_end_node = np.full(np_end, -1, dtype=np.int32)
    pipe_end_invert_elev = np.zeros(np_end, dtype=np.float64)
    pipe_end_diameter = np.zeros(np_end, dtype=np.float64)
    pipe_end_area = np.zeros(np_end, dtype=np.float64)
    pipe_end_inlet_loss_k = np.full(np_end, 0.5, dtype=np.float64)
    pipe_end_outlet_loss_k = np.full(np_end, 1.0, dtype=np.float64)
    for i, pe in enumerate(getattr(cfg, "pipe_ends", [])):
        ci = int(pe.cell_id)
        pipe_end_cell[i] = ci if 0 <= ci < int(n_cells) else -1
        pipe_end_node[i] = int(node_idx.get(pe.node_id, -1))
        pipe_end_invert_elev[i] = float(pe.invert_elev)
        pipe_end_diameter[i] = float(getattr(pe, "diameter", 0.0) or 0.0)
        pipe_end_area[i] = float(getattr(pe, "area_m2", 0.0) or 0.0)
        kin = getattr(pe, "inlet_loss_k", 0.5)
        kout = getattr(pe, "outlet_loss_k", 1.0)
        pipe_end_inlet_loss_k[i] = 0.5 if kin is None else float(kin)
        pipe_end_outlet_loss_k[i] = 1.0 if kout is None else float(kout)

    return SWE2DDrainageSoA(
        node_x=node_x,
        node_y=node_y,
        node_invert_elev=node_invert_elev,
        node_max_depth=node_max_depth,
        node_surface_area=node_surface_area,
        link_from=link_from,
        link_to=link_to,
        link_length=link_length,
        link_roughness_n=link_roughness_n,
        link_diameter=link_diameter,
        link_max_flow=link_max_flow,
        link_cd=link_cd,
        inlet_cell=inlet_cell,
        inlet_node=inlet_node,
        inlet_crest_elev=inlet_crest_elev,
        inlet_width=inlet_width,
        inlet_coefficient=inlet_coefficient,
        inlet_max_capture=inlet_max_capture,
        outfall_cell=outfall_cell,
        outfall_node=outfall_node,
        outfall_invert_elev=outfall_invert_elev,
        outfall_diameter=outfall_diameter,
        outfall_coefficient=outfall_coefficient,
        outfall_max_flow=outfall_max_flow,
        outfall_zero_storage=outfall_zero_storage,
        pipe_end_cell=pipe_end_cell,
        pipe_end_node=pipe_end_node,
        pipe_end_invert_elev=pipe_end_invert_elev,
        pipe_end_diameter=pipe_end_diameter,
        pipe_end_area=pipe_end_area,
        pipe_end_inlet_loss_k=pipe_end_inlet_loss_k,
        pipe_end_outlet_loss_k=pipe_end_outlet_loss_k,
        solver_mode=int(getattr(cfg, "solver_mode", DrainageSolverMode.EGL)),
    )


def pack_structures_soa(cfg: Optional[HydraulicStructureConfig], n_cells: int) -> Optional[SWE2DStructuresSoA]:
    if cfg is None or not cfg.enabled:
        return None
    if not cfg.structures:
        return None

    ns = len(cfg.structures)
    structure_type = np.zeros(ns, dtype=np.int32)
    upstream_cell = np.full(ns, -1, dtype=np.int32)
    downstream_cell = np.full(ns, -1, dtype=np.int32)
    crest_elev = np.zeros(ns, dtype=np.float64)
    width = np.zeros(ns, dtype=np.float64)
    height = np.zeros(ns, dtype=np.float64)
    diameter = np.zeros(ns, dtype=np.float64)
    length = np.zeros(ns, dtype=np.float64)
    roughness_n = np.zeros(ns, dtype=np.float64)
    coeff = np.zeros(ns, dtype=np.float64)
    cd = np.zeros(ns, dtype=np.float64)
    opening = np.zeros(ns, dtype=np.float64)
    q_pump = np.zeros(ns, dtype=np.float64)
    max_flow = np.full(ns, np.nan, dtype=np.float64)
    culvert_code = np.zeros(ns, dtype=np.int32)
    culvert_shape = np.zeros(ns, dtype=np.int32)
    culvert_rise = np.zeros(ns, dtype=np.float64)
    culvert_span = np.zeros(ns, dtype=np.float64)
    culvert_area_m2 = np.zeros(ns, dtype=np.float64)
    culvert_barrels = np.ones(ns, dtype=np.float64)
    culvert_slope = np.zeros(ns, dtype=np.float64)
    inlet_invert_elev = np.zeros(ns, dtype=np.float64)
    outlet_invert_elev = np.zeros(ns, dtype=np.float64)
    entrance_loss_k = np.zeros(ns, dtype=np.float64)
    exit_loss_k = np.zeros(ns, dtype=np.float64)
    embankment_enabled = np.zeros(ns, dtype=np.int32)
    embankment_crest_elev = np.zeros(ns, dtype=np.float64)
    embankment_overflow_width = np.zeros(ns, dtype=np.float64)
    embankment_weir_coeff = np.zeros(ns, dtype=np.float64)
    culvert_shape_map = {"circular": 0, "pipe": 0, "round": 0, "box": 1, "rect": 1, "rectangular": 1}

    for i, st in enumerate(cfg.structures):
        structure_type[i] = int(st.structure_type)
        iu = int(st.upstream_cell)
        idn = int(st.downstream_cell)
        upstream_cell[i] = iu if 0 <= iu < int(n_cells) else -1
        downstream_cell[i] = idn if 0 <= idn < int(n_cells) else -1
        crest_elev[i] = float(st.crest_elev)
        width[i] = float(st.metadata.get("width", 0.0))
        height[i] = float(st.metadata.get("height", 0.0))
        diameter[i] = float(st.metadata.get("diameter", 0.0))
        length[i] = float(st.metadata.get("length", 0.0))
        roughness_n[i] = float(st.metadata.get("roughness_n", 0.013))
        coeff[i] = float(st.metadata.get("coeff", 1.7))
        cd[i] = float(st.metadata.get("cd", 0.75))
        opening[i] = float(st.metadata.get("opening", 1.0))
        q_pump[i] = float(st.metadata.get("q_pump", 0.0))
        max_flow[i] = np.nan if st.metadata.get("max_flow") is None else float(st.metadata.get("max_flow"))
        culvert_code[i] = int(float(st.metadata.get("culvert_code", 1) or 1))
        culvert_shape[i] = int(culvert_shape_map.get(str(st.metadata.get("culvert_shape", "circular") or "circular").strip().lower(), 0))
        culvert_rise[i] = float(st.metadata.get("culvert_rise", st.metadata.get("height", st.metadata.get("diameter", 0.0))) or 0.0)
        culvert_span[i] = float(st.metadata.get("culvert_span", st.metadata.get("width", culvert_rise[i])) or 0.0)
        culvert_area_m2[i] = float(st.metadata.get("culvert_area_m2", st.metadata.get("area_m2", 0.0)) or 0.0)
        culvert_barrels[i] = float(st.metadata.get("culvert_barrels", 1.0) or 1.0)
        culvert_slope[i] = float(st.metadata.get("culvert_slope", 0.0) or 0.0)
        inlet_invert_elev[i] = float(st.metadata.get("inlet_invert_elev", st.crest_elev) or st.crest_elev)
        outlet_invert_elev[i] = float(st.metadata.get("outlet_invert_elev", inlet_invert_elev[i]) or inlet_invert_elev[i])
        entrance_loss_k[i] = float(st.metadata.get("entrance_loss_k", st.metadata.get("inlet_loss_k", 0.5)) or 0.5)
        exit_loss_k[i] = float(st.metadata.get("exit_loss_k", st.metadata.get("outlet_loss_k", 1.0)) or 1.0)
        embankment_enabled[i] = int(float(st.metadata.get("embankment_enabled", 0.0) or 0.0))
        embankment_crest_elev[i] = float(st.metadata.get("embankment_crest_elev", st.metadata.get("road_crest_elev", st.crest_elev)) or st.crest_elev)
        embankment_overflow_width[i] = float(st.metadata.get("embankment_overflow_width", st.metadata.get("road_overflow_width", st.metadata.get("width", 0.0))) or 0.0)
        embankment_weir_coeff[i] = float(st.metadata.get("embankment_weir_coeff", st.metadata.get("road_weir_coeff", 1.7)) or 1.7)

    return SWE2DStructuresSoA(
        structure_type=structure_type,
        upstream_cell=upstream_cell,
        downstream_cell=downstream_cell,
        crest_elev=crest_elev,
        width=width,
        height=height,
        diameter=diameter,
        length=length,
        roughness_n=roughness_n,
        coeff=coeff,
        cd=cd,
        opening=opening,
        q_pump=q_pump,
        max_flow=max_flow,
        culvert_code=culvert_code,
        culvert_shape=culvert_shape,
        culvert_rise=culvert_rise,
        culvert_span=culvert_span,
        culvert_area_m2=culvert_area_m2,
        culvert_barrels=culvert_barrels,
        culvert_slope=culvert_slope,
        inlet_invert_elev=inlet_invert_elev,
        outlet_invert_elev=outlet_invert_elev,
        entrance_loss_k=entrance_loss_k,
        exit_loss_k=exit_loss_k,
        embankment_enabled=embankment_enabled,
        embankment_crest_elev=embankment_crest_elev,
        embankment_overflow_width=embankment_overflow_width,
        embankment_weir_coeff=embankment_weir_coeff,
    )


def pack_coupling_soa(
    n_cells: int,
    pipe_network: Optional[PipeNetworkConfig] = None,
    hydraulic_structures: Optional[HydraulicStructureConfig] = None,
) -> SWE2DCouplingSoA:
    return SWE2DCouplingSoA(
        n_cells=int(n_cells),
        drainage=pack_pipe_network_soa(pipe_network, n_cells),
        structures=pack_structures_soa(hydraulic_structures, n_cells),
    )


class SWE2DCouplingController:
    """Combine optional drainage and structure modules into one source callback."""

    def __init__(
        self,
        cell_area: Optional[Sequence[float]] = None,
        cell_bed: Optional[Sequence[float]] = None,
        drainage: Optional[SWE2DUrbanDrainageModule] = None,
        structures: Optional[SWE2DStructureModule] = None,
        coupling_loop: str = "cpu",
        drainage_solver_backend: str = "cpu",
        drainage_gpu_method: str = "step",
        culvert_solver_mode: int = 0,
        bridge_cuda_coupling: bool = False,
        bridge_stacked_coupling_mode: str = "phase3_spatial",
        **legacy_kwargs,
    ):
        if cell_area is None:
            cell_area = legacy_kwargs.pop("cell_area_m2", None)
        if cell_bed is None:
            cell_bed = legacy_kwargs.pop("cell_bed_m", None)
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        if cell_area is None or cell_bed is None:
            raise ValueError("cell_area and cell_bed are required")

        self.cell_area = np.ascontiguousarray(cell_area, dtype=np.float64).ravel()
        self.cell_bed = np.ascontiguousarray(cell_bed, dtype=np.float64).ravel()
        if self.cell_area.size != self.cell_bed.size:
            raise ValueError("cell_area and cell_bed must have the same length")
        self.drainage = drainage
        self.structures = structures
        self.coupling_loop = str(coupling_loop or "cpu").strip().lower()
        if self.coupling_loop not in {"cpu", "cuda"}:
            raise ValueError("coupling_loop must be 'cpu' or 'cuda'")
        self.drainage_solver_backend = str(drainage_solver_backend or "cpu").strip().lower()
        if self.drainage_solver_backend not in {"cpu", "gpu"}:
            raise ValueError("drainage_solver_backend must be 'cpu' or 'gpu'")
        self.drainage_gpu_method = str(drainage_gpu_method or "step").strip().lower()
        if self.drainage_gpu_method not in {"step", "iterative"}:
            raise ValueError("drainage_gpu_method must be 'step' or 'iterative'")
        self.bridge_cuda_coupling = bool(bridge_cuda_coupling)
        self.bridge_stacked_coupling_mode = str(bridge_stacked_coupling_mode or "phase3_spatial").strip().lower()
        if self.bridge_stacked_coupling_mode not in {"legacy_scalar", "phase3_spatial"}:
            raise ValueError("bridge_stacked_coupling_mode must be 'legacy_scalar' or 'phase3_spatial'")
        self.culvert_solver_mode = int(culvert_solver_mode)
        if self.culvert_solver_mode not in {0, 1}:
            raise ValueError("culvert_solver_mode must be 0 or 1")
        self._culvert_solver_mode_applied = False
        self._persistent_coupling_preloaded = False
        self._drainage_soa = pack_pipe_network_soa(self.drainage.cfg, self.n_cells) if self.drainage is not None else None
        self._structures_soa = pack_structures_soa(self.structures.cfg, self.n_cells) if self.structures is not None else None
        self._gpu_node_depth: Optional[np.ndarray] = None
        self._gpu_link_flow: Optional[np.ndarray] = None
        self._gpu_drainage_static_args: Optional[Dict[str, np.ndarray]] = None
        self.last_diag = SWE2DCouplingDiagnostics()

    @property
    def cell_area_m2(self) -> np.ndarray:
        return self.cell_area

    @property
    def cell_bed_m(self) -> np.ndarray:
        return self.cell_bed

    def _native_cuda_module(self):
        try:
            import hydra_swe2d as mod  # type: ignore
        except Exception:
            return None
        if not hasattr(mod, "swe2d_gpu_compute_coupling_sources"):
            return None
        try:
            if not bool(mod.swe2d_gpu_available()):
                return None
        except Exception:
            return None
        return mod

    def _ensure_native_culvert_solver_mode(self, native_mod) -> None:
        if self._culvert_solver_mode_applied:
            return
        if not hasattr(native_mod, "swe2d_gpu_set_culvert_solver_mode"):
            return
        try:
            native_mod.swe2d_gpu_set_culvert_solver_mode(int(self.culvert_solver_mode))
        except Exception:
            pass
        self._culvert_solver_mode_applied = True

    def _ensure_persistent_coupling_preloaded(self, native_mod) -> None:
        if self._persistent_coupling_preloaded:
            return
        if not hasattr(native_mod, "swe2d_gpu_preload_structure_params"):
            return
        ssoa = self._structures_soa
        if ssoa is not None and int(len(ssoa.structure_type)) > 0:
            try:
                native_mod.swe2d_gpu_preload_structure_params(
                    np.asarray(ssoa.structure_type, dtype=np.int32),
                    np.asarray(ssoa.upstream_cell, dtype=np.int32),
                    np.asarray(ssoa.downstream_cell, dtype=np.int32),
                    np.asarray(ssoa.crest_elev, dtype=np.float64),
                    np.asarray(ssoa.width, dtype=np.float64),
                    np.asarray(ssoa.height, dtype=np.float64),
                    np.asarray(ssoa.diameter, dtype=np.float64),
                    np.asarray(ssoa.length, dtype=np.float64),
                    np.asarray(ssoa.roughness_n, dtype=np.float64),
                    np.asarray(ssoa.coeff, dtype=np.float64),
                    np.asarray(ssoa.cd, dtype=np.float64),
                    np.asarray(ssoa.opening, dtype=np.float64),
                    np.asarray(ssoa.q_pump, dtype=np.float64),
                    np.asarray(ssoa.max_flow, dtype=np.float64),
                    np.asarray(ssoa.culvert_code, dtype=np.int32),
                    np.asarray(ssoa.culvert_shape, dtype=np.int32),
                    np.asarray(ssoa.culvert_rise, dtype=np.float64),
                    np.asarray(ssoa.culvert_span, dtype=np.float64),
                    np.asarray(ssoa.culvert_area_m2, dtype=np.float64),
                    np.asarray(ssoa.culvert_barrels, dtype=np.float64),
                    np.asarray(ssoa.culvert_slope, dtype=np.float64),
                    np.asarray(ssoa.inlet_invert_elev, dtype=np.float64),
                    np.asarray(ssoa.outlet_invert_elev, dtype=np.float64),
                    np.asarray(ssoa.entrance_loss_k, dtype=np.float64),
                    np.asarray(ssoa.exit_loss_k, dtype=np.float64),
                    np.asarray(ssoa.embankment_enabled, dtype=np.int32),
                    np.asarray(ssoa.embankment_crest_elev, dtype=np.float64),
                    np.asarray(ssoa.embankment_overflow_width, dtype=np.float64),
                    np.asarray(ssoa.embankment_weir_coeff, dtype=np.float64),
                    float(getattr(self.structures.cfg, "gravity", 9.81)) if self.structures is not None else 9.81,
                )
            except Exception:
                return
        if hasattr(native_mod, "swe2d_gpu_preload_coupling_cell_area"):
            try:
                native_mod.swe2d_gpu_preload_coupling_cell_area(
                    np.asarray(self.cell_area, dtype=np.float64))
            except Exception:
                return
        self._persistent_coupling_preloaded = True

    def _ensure_gpu_drainage_state(self) -> None:
        if self.drainage is None:
            return
        cfg = self.drainage.cfg
        if self._gpu_node_depth is None or self._gpu_node_depth.size != len(cfg.nodes):
            self._gpu_node_depth = np.asarray(
                [float(self.drainage.state.node_depth.get(n.node_id, 0.0)) for n in cfg.nodes],
                dtype=np.float64,
            )
        if self._gpu_link_flow is None or self._gpu_link_flow.size != len(cfg.links):
            self._gpu_link_flow = np.asarray(
                [float(self.drainage.state.link_flow.get(l.link_id, 0.0)) for l in cfg.links],
                dtype=np.float64,
            )

    def _sync_gpu_state_back_to_drainage(self) -> None:
        if self.drainage is None or self._gpu_node_depth is None or self._gpu_link_flow is None:
            return
        for i, node in enumerate(self.drainage.cfg.nodes):
            self.drainage.state.node_depth[node.node_id] = float(self._gpu_node_depth[i])
        for i, link in enumerate(self.drainage.cfg.links):
            self.drainage.state.link_flow[link.link_id] = float(self._gpu_link_flow[i])

    def _ensure_gpu_drainage_static_args(self) -> Optional[Dict[str, np.ndarray]]:
        if self._drainage_soa is None:
            return None
        if self._gpu_drainage_static_args is not None:
            return self._gpu_drainage_static_args
        dsoa = self._drainage_soa
        self._gpu_drainage_static_args = {
            "cell_bed": np.ascontiguousarray(self.cell_bed, dtype=np.float64),
            "cell_area": np.ascontiguousarray(self.cell_area, dtype=np.float64),
            "node_invert_elev": np.ascontiguousarray(dsoa.node_invert_elev, dtype=np.float64),
            "node_max_depth": np.ascontiguousarray(dsoa.node_max_depth, dtype=np.float64),
            "node_surface_area": np.ascontiguousarray(dsoa.node_surface_area, dtype=np.float64),
            "link_from": np.ascontiguousarray(dsoa.link_from, dtype=np.int32),
            "link_to": np.ascontiguousarray(dsoa.link_to, dtype=np.int32),
            "link_length": np.ascontiguousarray(dsoa.link_length, dtype=np.float64),
            "link_roughness_n": np.ascontiguousarray(dsoa.link_roughness_n, dtype=np.float64),
            "link_diameter": np.ascontiguousarray(dsoa.link_diameter, dtype=np.float64),
            "link_max_flow": np.ascontiguousarray(dsoa.link_max_flow, dtype=np.float64),
            "inlet_cell": np.ascontiguousarray(dsoa.inlet_cell, dtype=np.int32),
            "inlet_node": np.ascontiguousarray(dsoa.inlet_node, dtype=np.int32),
            "inlet_crest_elev": np.ascontiguousarray(dsoa.inlet_crest_elev, dtype=np.float64),
            "inlet_width": np.ascontiguousarray(dsoa.inlet_width, dtype=np.float64),
            "inlet_coefficient": np.ascontiguousarray(dsoa.inlet_coefficient, dtype=np.float64),
            "inlet_max_capture": np.ascontiguousarray(dsoa.inlet_max_capture, dtype=np.float64),
            "outfall_cell": np.ascontiguousarray(dsoa.outfall_cell, dtype=np.int32),
            "outfall_node": np.ascontiguousarray(dsoa.outfall_node, dtype=np.int32),
            "outfall_invert_elev": np.ascontiguousarray(dsoa.outfall_invert_elev, dtype=np.float64),
            "outfall_diameter": np.ascontiguousarray(dsoa.outfall_diameter, dtype=np.float64),
            "outfall_coefficient": np.ascontiguousarray(dsoa.outfall_coefficient, dtype=np.float64),
            "outfall_max_flow": np.ascontiguousarray(dsoa.outfall_max_flow, dtype=np.float64),
            "outfall_zero_storage": np.ascontiguousarray(dsoa.outfall_zero_storage, dtype=np.int32),
            "pipe_end_cell": np.ascontiguousarray(dsoa.pipe_end_cell, dtype=np.int32),
            "pipe_end_node": np.ascontiguousarray(dsoa.pipe_end_node, dtype=np.int32),
            "pipe_end_invert_elev": np.ascontiguousarray(dsoa.pipe_end_invert_elev, dtype=np.float64),
            "pipe_end_diameter": np.ascontiguousarray(dsoa.pipe_end_diameter, dtype=np.float64),
            "pipe_end_area": np.ascontiguousarray(dsoa.pipe_end_area, dtype=np.float64),
            "pipe_end_inlet_loss_k": np.ascontiguousarray(dsoa.pipe_end_inlet_loss_k, dtype=np.float64),
            "pipe_end_outlet_loss_k": np.ascontiguousarray(dsoa.pipe_end_outlet_loss_k, dtype=np.float64),
        }
        return self._gpu_drainage_static_args

    def _bridge_structure_arrays(self, cell_wse: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if self.structures is None:
            return None
        bridge_indices = [
            i for i, st in enumerate(self.structures.cfg.structures)
            if st.enabled and st.structure_type == StructureType.BRIDGE
        ]
        if not bridge_indices:
            return None

        flows = np.asarray(self.structures.structure_flows(cell_wse), dtype=np.float64)
        return {
            "indices": np.asarray(bridge_indices, dtype=np.int32),
            "structure_id": np.asarray([str(self.structures.cfg.structures[i].structure_id) for i in bridge_indices], dtype=object),
            "upstream_cell": np.ascontiguousarray([int(self.structures.cfg.structures[i].upstream_cell) for i in bridge_indices], dtype=np.int32),
            "downstream_cell": np.ascontiguousarray([int(self.structures.cfg.structures[i].downstream_cell) for i in bridge_indices], dtype=np.int32),
            "flow_cms": np.ascontiguousarray(flows[bridge_indices], dtype=np.float64),
            "loss_k_upstream": np.ascontiguousarray([
                float(self.structures.cfg.structures[i].metadata.get("inlet_loss_k", self.structures.cfg.structures[i].metadata.get("coeff", 0.5)))
                for i in bridge_indices
            ], dtype=np.float64),
            "loss_k_downstream": np.ascontiguousarray([
                float(self.structures.cfg.structures[i].metadata.get("outlet_loss_k", self.structures.cfg.structures[i].metadata.get("coeff", 0.5)))
                for i in bridge_indices
            ], dtype=np.float64),
            "width_m": np.ascontiguousarray([
                float(self.structures.cfg.structures[i].metadata.get("width", 1.0))
                for i in bridge_indices
            ], dtype=np.float64),
        }

    def _native_structure_flows(self, native_mod, cell_wse: np.ndarray) -> Optional[np.ndarray]:
        ssoa = self._structures_soa
        if ssoa is None or not hasattr(native_mod, "swe2d_gpu_compute_structure_flows"):
            return None
        try:
            return np.asarray(
                native_mod.swe2d_gpu_compute_structure_flows(
                    np.asarray(cell_wse, dtype=np.float64),
                    np.asarray(self.cell_bed, dtype=np.float64),
                    np.asarray(ssoa.structure_type, dtype=np.int32),
                    np.asarray(ssoa.upstream_cell, dtype=np.int32),
                    np.asarray(ssoa.downstream_cell, dtype=np.int32),
                    np.asarray(ssoa.crest_elev, dtype=np.float64),
                    np.asarray(ssoa.width, dtype=np.float64),
                    np.asarray(ssoa.height, dtype=np.float64),
                    np.asarray(ssoa.diameter, dtype=np.float64),
                    np.asarray(ssoa.length, dtype=np.float64),
                    np.asarray(ssoa.roughness_n, dtype=np.float64),
                    np.asarray(ssoa.coeff, dtype=np.float64),
                    np.asarray(ssoa.cd, dtype=np.float64),
                    np.asarray(ssoa.opening, dtype=np.float64),
                    np.asarray(ssoa.q_pump, dtype=np.float64),
                    np.asarray(ssoa.max_flow, dtype=np.float64),
                    np.asarray(ssoa.culvert_code, dtype=np.int32),
                    np.asarray(ssoa.culvert_shape, dtype=np.int32),
                    np.asarray(ssoa.culvert_rise, dtype=np.float64),
                    np.asarray(ssoa.culvert_span, dtype=np.float64),
                    np.asarray(ssoa.culvert_area_m2, dtype=np.float64),
                    np.asarray(ssoa.culvert_barrels, dtype=np.float64),
                    np.asarray(ssoa.culvert_slope, dtype=np.float64),
                    np.asarray(ssoa.inlet_invert_elev, dtype=np.float64),
                    np.asarray(ssoa.outlet_invert_elev, dtype=np.float64),
                    np.asarray(ssoa.entrance_loss_k, dtype=np.float64),
                    np.asarray(ssoa.exit_loss_k, dtype=np.float64),
                    np.asarray(ssoa.embankment_enabled, dtype=np.int32),
                    np.asarray(ssoa.embankment_crest_elev, dtype=np.float64),
                    np.asarray(ssoa.embankment_overflow_width, dtype=np.float64),
                    np.asarray(ssoa.embankment_weir_coeff, dtype=np.float64),
                    float(getattr(self.structures.cfg, "gravity", 9.81)) if self.structures is not None else 9.81,
                ),
                dtype=np.float64,
            )
        except Exception:
            return None

    @property
    def n_cells(self) -> int:
        return int(self.cell_area.size)

    def source_rate_callback(self) -> Callable[[float, float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        return self.compute_source_rates

    def compute_source_rates(
        self,
        t_s: float,
        dt_s: float,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
    ) -> np.ndarray:
        _ = (hu, hv)
        hh = np.ascontiguousarray(h, dtype=np.float64).ravel()
        if hh.size != self.n_cells:
            raise ValueError("state size does not match coupling cell arrays")
        if self.coupling_loop == "cuda":
            mod = self._native_cuda_module()
            if mod is not None:
                self._ensure_native_culvert_solver_mode(mod)
                self._ensure_persistent_coupling_preloaded(mod)
                return self._compute_source_rates_cuda(mod, t_s, dt_s, hh)
        cell_wse = hh + self.cell_bed
        total = np.zeros(self.n_cells, dtype=np.float64)
        component_sums: Dict[str, float] = {}
        drainage_diag: Dict[str, float] = {}
        structure_diag: Dict[str, float] = {}

        if self.drainage is not None:
            has_pipe_end_routing = bool(getattr(self.drainage.cfg, "pipe_ends", []))
            if not has_pipe_end_routing:
                drainage_diag = self.drainage.solve_network_step(float(dt_s))
            src = np.asarray(
                self.drainage.surface_exchange_source_rate(
                    float(dt_s),
                    cell_wse,
                    self.cell_area,
                    cell_depth_m=hh,
                ),
                dtype=np.float64,
            )
            if has_pipe_end_routing:
                drainage_diag = dict(getattr(self.drainage, "_last_network_diag", {}))
            if src.size != self.n_cells:
                raise ValueError("drainage source-rate array size mismatch")
            total += src
            component_sums["drainage"] = float(np.sum(src))

        if self.structures is not None:
            src = np.asarray(
                self.structures.compute_cell_source_rate(float(dt_s), cell_wse, self.cell_area),
                dtype=np.float64,
            )
            if src.size != self.n_cells:
                raise ValueError("structure source-rate array size mismatch")
            total += src
            component_sums["structures"] = float(np.sum(src))
            structure_diag = self.structures.compute_structure_fluxes(float(dt_s), cell_wse)

        self.last_diag = SWE2DCouplingDiagnostics(
            time_s=float(t_s) + float(dt_s),
            dt_s=float(dt_s),
            drainage_max_node_depth=float(drainage_diag.get("max_node_depth", drainage_diag.get("max_node_depth_m", 0.0))),
            drainage_max_link_flow=float(drainage_diag.get("max_link_flow", drainage_diag.get("max_link_flow_cms", 0.0))),
            structure_total_flow=float(structure_diag.get("total_structure_flow", 0.0)),
            source_sum=float(np.sum(total)),
            source_min=float(np.min(total)) if total.size else 0.0,
            source_max=float(np.max(total)) if total.size else 0.0,
            component_sums=component_sums,
        )
        return total

    def _compute_source_rates_cuda(self, native_mod, t_s: float, dt_s: float, hh: np.ndarray) -> np.ndarray:
        cell_wse = hh + self.cell_bed
        drainage_diag: Dict[str, float] = {}
        structure_diag: Dict[str, float] = {}
        component_sums: Dict[str, float] = {}

        inlet_cell = np.empty(0, dtype=np.int32)
        inlet_flow = np.empty(0, dtype=np.float64)
        struct_up = np.empty(0, dtype=np.int32)
        struct_dn = np.empty(0, dtype=np.int32)
        struct_q = np.empty(0, dtype=np.float64)
        flows = np.empty(0, dtype=np.float64)

        if self.drainage is not None:
            q_cell = None
            if (
                self.drainage_solver_backend == "gpu"
                and self._drainage_soa is not None
                and hasattr(native_mod, "swe2d_gpu_drainage_step")
            ):
                self._ensure_gpu_drainage_state()
                dsoa = self._drainage_soa
                solver_mode = DrainageSolverMode(int(dsoa.solver_mode))
                base_substeps = max(1, int(getattr(self.drainage.cfg, "coupling_substeps", 1)))
                adaptive_substeps = 1
                if hasattr(self.drainage, "_adaptive_substep_count"):
                    adaptive_substeps = max(1, int(self.drainage._adaptive_substep_count(float(dt_s), solver_mode)))
                implicit_substeps = max(1, int(getattr(self.drainage.cfg, "implicit_coupling_iterations", 1)))
                n_substeps = max(base_substeps, adaptive_substeps, implicit_substeps)
                implicit_iters = max(1, int(getattr(self.drainage.cfg, "implicit_coupling_iterations", 1)))
                coupling_relax = float(getattr(self.drainage.cfg, "implicit_coupling_relaxation", 0.5))
                coupling_relax = min(1.0, max(0.0, coupling_relax))
                static_args = self._ensure_gpu_drainage_static_args()
                if static_args is None:
                    raise RuntimeError("GPU drainage static args are unavailable")
                hh64 = np.asarray(hh, dtype=np.float64)
                node_depth_state = np.asarray(self._gpu_node_depth, dtype=np.float64)
                link_flow_state = np.asarray(self._gpu_link_flow, dtype=np.float64)
                head_deadband = float(getattr(self.drainage.cfg, "head_deadband_m", 1.0e-3))

                # Fast-path for inactive exchange conditions.
                if static_args is not None:
                    inlet_idx = static_args["inlet_cell"]
                    inlet_crest = static_args["inlet_crest_elev"]
                    has_inlet_head = False
                    if inlet_idx.size > 0:
                        inlet_wse = np.asarray(cell_wse[inlet_idx], dtype=np.float64)
                        has_inlet_head = bool(np.any(inlet_wse > inlet_crest + head_deadband))
                    node_active = bool(np.any(node_depth_state > head_deadband))
                    link_active = bool(np.any(np.abs(link_flow_state) > 1.0e-10))
                    if (not has_inlet_head) and (not node_active) and (not link_active):
                        q_cell = np.zeros(self.n_cells, dtype=np.float64)
                        drainage_diag = {
                            "max_node_depth": 0.0,
                            "max_link_flow": 0.0,
                            "limiter_events": 0.0,
                            "limiter_volume_m3": 0.0,
                            "substeps_used": 0.0,
                            "implicit_iters_used": 0.0,
                            "inactive_fastpath": 1.0,
                        }
                        component_sums["drainage_native_iterative"] = 0.0
                        component_sums["drainage_inactive_fastpath"] = 1.0
                    else:
                        component_sums["drainage_inactive_fastpath"] = 0.0
                else:
                    component_sums["drainage_inactive_fastpath"] = 0.0

                if q_cell is None:
                    prefer_native_iterative = self.drainage_gpu_method == "iterative"
                    use_native_iterative = (
                        prefer_native_iterative
                        and hasattr(native_mod, "swe2d_gpu_drainage_step_iterative")
                        and os.environ.get("BACKWATER_SWE2D_DISABLE_NATIVE_ITERATIVE", "").strip() != "1"
                    )
                    # Enforce native iterative when requested (no fallback to Python loop).
                    if prefer_native_iterative and not use_native_iterative:
                        raise RuntimeError(
                            "GPU drainage method 'iterative' requested but native implementation unavailable. "
                            "Either switch to 'step' method or ensure CUDA bindings are available."
                        )
                    if use_native_iterative:
                        nd_out, lf_out, q_cell_step, diag = native_mod.swe2d_gpu_drainage_step_iterative(
                            static_args["cell_bed"],
                            static_args["cell_area"],
                            static_args["node_invert_elev"],
                            static_args["node_max_depth"],
                            static_args["node_surface_area"],
                            static_args["link_from"],
                            static_args["link_to"],
                            static_args["link_length"],
                            static_args["link_roughness_n"],
                            static_args["link_diameter"],
                            static_args["link_max_flow"],
                            static_args["inlet_cell"],
                            static_args["inlet_node"],
                            static_args["inlet_crest_elev"],
                            static_args["inlet_width"],
                            static_args["inlet_coefficient"],
                            static_args["inlet_max_capture"],
                            static_args["outfall_cell"],
                            static_args["outfall_node"],
                            static_args["outfall_invert_elev"],
                            static_args["outfall_diameter"],
                            static_args["outfall_coefficient"],
                            static_args["outfall_max_flow"],
                            static_args["outfall_zero_storage"],
                            static_args["pipe_end_cell"],
                            static_args["pipe_end_node"],
                            static_args["pipe_end_invert_elev"],
                            static_args["pipe_end_diameter"],
                            static_args["pipe_end_area"],
                            static_args["pipe_end_inlet_loss_k"],
                            static_args["pipe_end_outlet_loss_k"],
                            hh64,
                            node_depth_state,
                            link_flow_state,
                            float(dt_s),
                            float(getattr(self.drainage.cfg, "gravity", 9.81)),
                            int(dsoa.solver_mode),
                            float(getattr(self.drainage.cfg, "head_deadband_m", 1.0e-3)),
                            float(getattr(self.drainage.cfg, "dynamic_flow_relaxation", 1.0)),
                            int(n_substeps),
                            int(implicit_iters),
                            float(coupling_relax),
                        )
                        self._gpu_node_depth = np.asarray(nd_out, dtype=np.float64)
                        self._gpu_link_flow = np.asarray(lf_out, dtype=np.float64)
                        self._sync_gpu_state_back_to_drainage()
                        q_cell = np.asarray(q_cell_step, dtype=np.float64)
                        drainage_diag = {
                            "max_node_depth": float(diag.get("max_node_depth", 0.0)),
                            "max_link_flow": float(diag.get("max_link_flow", 0.0)),
                            "limiter_events": float(diag.get("limiter_events", 0.0)),
                            "limiter_volume_m3": float(diag.get("limiter_volume_m3", 0.0)),
                            "substeps_used": float(diag.get("substeps_used", n_substeps)),
                            "implicit_iters_used": float(diag.get("implicit_iters_used", max(1, n_substeps * implicit_iters))),
                            "inactive_fastpath": float(diag.get("inactive_fastpath", 0.0)),
                        }
                        component_sums["drainage_native_iterative"] = 1.0
                    else:
                        nd_out, lf_out, q_cell_step, diag = (
                            native_mod.swe2d_gpu_drainage_step(
                                cell_wse,
                                static_args["cell_area"],
                                static_args["node_invert_elev"],
                                static_args["node_max_depth"],
                                static_args["node_surface_area"],
                                static_args["link_from"],
                                static_args["link_to"],
                                static_args["link_length"],
                                static_args["link_roughness_n"],
                                static_args["link_diameter"],
                                static_args["link_max_flow"],
                                static_args["inlet_cell"],
                                static_args["inlet_node"],
                                static_args["inlet_crest_elev"],
                                static_args["inlet_width"],
                                static_args["inlet_coefficient"],
                                static_args["inlet_max_capture"],
                                static_args["outfall_cell"],
                                static_args["outfall_node"],
                                static_args["outfall_invert_elev"],
                                static_args["outfall_diameter"],
                                static_args["outfall_coefficient"],
                                static_args["outfall_max_flow"],
                                static_args["outfall_zero_storage"],
                                static_args["pipe_end_cell"],
                                static_args["pipe_end_node"],
                                static_args["pipe_end_invert_elev"],
                                static_args["pipe_end_diameter"],
                                static_args["pipe_end_area"],
                                static_args["pipe_end_inlet_loss_k"],
                                static_args["pipe_end_outlet_loss_k"],
                                hh64,
                                node_depth_state,
                                link_flow_state,
                                float(dt_s),
                                float(getattr(self.drainage.cfg, "gravity", 9.81)),
                                int(dsoa.solver_mode),
                                float(getattr(self.drainage.cfg, "head_deadband_m", 1.0e-3)),
                                float(getattr(self.drainage.cfg, "dynamic_flow_relaxation", 1.0)),
                            )
                        )
                        self._gpu_node_depth = np.asarray(nd_out, dtype=np.float64)
                        self._gpu_link_flow = np.asarray(lf_out, dtype=np.float64)
                        self._sync_gpu_state_back_to_drainage()
                        q_cell = np.asarray(q_cell_step, dtype=np.float64)
                        drainage_diag = {
                            "max_node_depth": float(diag.get("max_node_depth", 0.0)),
                            "max_link_flow": float(diag.get("max_link_flow", 0.0)),
                            "limiter_events": float(diag.get("limiter_events", 0.0)),
                            "limiter_volume_m3": float(diag.get("limiter_volume_m3", 0.0)),
                            "substeps_used": 1.0,
                            "implicit_iters_used": 0.0,
                            "inactive_fastpath": 0.0,
                        }
                        component_sums["drainage_native_iterative"] = 0.0
            else:
                has_pipe_end_routing = bool(getattr(self.drainage.cfg, "pipe_ends", []))
                if not has_pipe_end_routing:
                    drainage_step_diag = self.drainage.solve_network_step(float(dt_s))
                else:
                    drainage_step_diag = {}
                q_cell = np.asarray(
                    self.drainage.apply_surface_exchange(
                        float(dt_s),
                        cell_wse,
                        cell_area_m2=self.cell_area,
                        cell_depth_m=hh,
                    ),
                    dtype=np.float64,
                )
                if has_pipe_end_routing:
                    drainage_step_diag = dict(getattr(self.drainage, "_last_network_diag", {}))
                drainage_diag = {
                    "max_node_depth": float(drainage_step_diag.get("max_node_depth", 0.0)),
                    "max_link_flow": float(drainage_step_diag.get("max_link_flow", 0.0)),
                    "limiter_events": float(drainage_step_diag.get("limiter_events", 0.0)),
                    "limiter_volume_m3": float(drainage_step_diag.get("limiter_volume_m3", 0.0)),
                    "substeps_used": 1.0,
                }
                component_sums["drainage_native_iterative"] = 0.0

            if q_cell is None:
                raise RuntimeError(
                    "GPU drainage did not produce q_cell (None). "
                    "Check drainage_gpu_method and native CUDA drainage bindings."
                )
            nz = np.nonzero(np.abs(q_cell) > 0.0)[0]
            if nz.size > 0:
                inlet_cell = nz.astype(np.int32, copy=False)
                # Kernel convention: positive inlet flow removes surface water.
                inlet_flow = (-q_cell[nz]).astype(np.float64, copy=False)
            component_sums["drainage"] = float(np.sum(q_cell / np.maximum(self.cell_area, 1.0e-12)))
            component_sums["drainage_limiter_events"] = float(drainage_diag.get("limiter_events", 0.0))
            component_sums["drainage_limiter_volume_m3"] = float(drainage_diag.get("limiter_volume_m3", 0.0))
            component_sums["drainage_substeps_used"] = float(drainage_diag.get("substeps_used", 1.0))
            component_sums["drainage_implicit_iters_used"] = float(drainage_diag.get("implicit_iters_used", 0.0))
            component_sums["drainage_inactive_fastpath"] = float(drainage_diag.get("inactive_fastpath", 0.0))

        bridge_total = np.zeros(self.n_cells, dtype=np.float64)
        bridge_helper_used = False
        if self.structures is not None:
            bridge_plan_map = {
                str(plan.structure_id): plan for plan in getattr(self, "bridge_stacked_plans", []) or []
            }
            sts = list(self.structures.cfg.structures)
            use_persistent = (
                self._persistent_coupling_preloaded
                and hasattr(native_mod, "swe2d_gpu_compute_coupling_full_on_device")
            )
            use_fused = (not use_persistent
                         and hasattr(native_mod, "swe2d_gpu_compute_structure_and_coupling_sources"))
            if use_persistent:
                # Persistent device path: structure params preloaded on GPU.
                # Only cell_wse is transferred per step. Source rates are
                # written directly to dev->d_external_source_mps (no D2H).
                ssoa = self._structures_soa
                if ssoa is not None:
                    bridge_mask = np.asarray(
                        [st.structure_type == StructureType.BRIDGE for st in sts], dtype=bool
                    )
                    n_non_bridge = int(np.sum(~bridge_mask))
                    try:
                        native_mod.swe2d_gpu_compute_coupling_full_on_device(
                            np.asarray(cell_wse, dtype=np.float64),
                            n_non_bridge,
                        )
                        component_sums["structures_persistent_path"] = 1.0
                    except Exception:
                        component_sums["structures_persistent_path"] = 0.0
                    # Bridges still handled below
                    if bridge_mask.any():
                        bridge_flows = self._native_structure_flows(native_mod, cell_wse)
                        if bridge_flows is None:
                            bridge_flows = np.asarray(self.structures.structure_flows(cell_wse), dtype=np.float64)
                        if bridge_flows is not None and bridge_flows.size == len(sts):
                            use_bridge_cuda = self.bridge_cuda_coupling and hasattr(
                                native_mod, "swe2d_gpu_compute_bridge_coupling_sources")
                            if use_bridge_cuda:
                                bridge_arrays = self._bridge_structure_arrays(cell_wse)
                                if bridge_arrays is not None:
                                    bridge_helper_used = True
                                    for i in range(int(bridge_arrays["indices"].size)):
                                        src = np.asarray(
                                            native_mod.swe2d_gpu_compute_bridge_coupling_sources(
                                                np.asarray(self.cell_area, dtype=np.float64),
                                                np.asarray([int(bridge_arrays["upstream_cell"][i])], dtype=np.int32),
                                                np.asarray([int(bridge_arrays["downstream_cell"][i])], dtype=np.int32),
                                                np.asarray([float(bridge_arrays["flow_cms"][i])], dtype=np.float64),
                                                np.asarray([float(bridge_arrays["loss_k_upstream"][i])], dtype=np.float64),
                                                np.asarray([float(bridge_arrays["loss_k_downstream"][i])], dtype=np.float64),
                                                float(bridge_arrays["width_m"][i]), float(dt_s),
                                            ), dtype=np.float64)
                                        bridge_total += src
                total = np.zeros(self.n_cells, dtype=np.float64)
                flows = None
            elif use_fused and flows.size == 0:
                # Fused path: structure flows + coupling sources in one device-resident call.
                # Bridges handled separately below via the individual bridge helper.
                ssoa = self._structures_soa
                if ssoa is not None:
                    # Separate bridge vs non-bridge structures for the fused call
                    non_bridge_mask = np.asarray(
                        [st.structure_type != StructureType.BRIDGE for st in sts], dtype=bool
                    )
                    bridge_mask = ~non_bridge_mask
                    # Fused call for non-bridge structures + inlets
                    total = np.asarray(
                        native_mod.swe2d_gpu_compute_structure_and_coupling_sources(
                            np.asarray(self.cell_area, dtype=np.float64),
                            np.asarray(cell_wse, dtype=np.float64),
                            np.asarray(self.cell_bed, dtype=np.float64),
                            np.asarray(ssoa.structure_type[non_bridge_mask], dtype=np.int32),
                            np.asarray(ssoa.upstream_cell[non_bridge_mask], dtype=np.int32),
                            np.asarray(ssoa.downstream_cell[non_bridge_mask], dtype=np.int32),
                            np.asarray(ssoa.crest_elev[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.width[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.height[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.diameter[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.length[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.roughness_n[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.coeff[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.cd[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.opening[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.q_pump[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.max_flow[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.culvert_code[non_bridge_mask], dtype=np.int32),
                            np.asarray(ssoa.culvert_shape[non_bridge_mask], dtype=np.int32),
                            np.asarray(ssoa.culvert_rise[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.culvert_span[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.culvert_area_m2[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.culvert_barrels[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.culvert_slope[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.inlet_invert_elev[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.outlet_invert_elev[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.entrance_loss_k[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.exit_loss_k[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.embankment_enabled[non_bridge_mask], dtype=np.int32),
                            np.asarray(ssoa.embankment_crest_elev[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.embankment_overflow_width[non_bridge_mask], dtype=np.float64),
                            np.asarray(ssoa.embankment_weir_coeff[non_bridge_mask], dtype=np.float64),
                            float(getattr(self.structures.cfg, "gravity", 9.81)) if self.structures is not None else 9.81,
                            inlet_cell,
                            inlet_flow,
                        ),
                        dtype=np.float64,
                    )
                    component_sums["structures_native_helper"] = 1.0
                    component_sums["structures_fused_path"] = 1.0
                    # Handle bridges separately
                    if bridge_mask.any():
                        bridge_flow_indices = np.where(bridge_mask)[0]
                        bridge_flows = self._native_structure_flows(native_mod, cell_wse)
                        if bridge_flows is not None:
                            bridge_sts = [sts[i] for i in bridge_flow_indices]
                            bridge_q = bridge_flows[bridge_flow_indices.astype(np.int32)]
                        else:
                            bridge_q = np.asarray([], dtype=np.float64)
                        if len(bridge_q) > 0:
                            use_bridge_cuda = self.bridge_cuda_coupling and hasattr(
                                native_mod, "swe2d_gpu_compute_bridge_coupling_sources"
                            )
                            if use_bridge_cuda:
                                bridge_arrays = self._bridge_structure_arrays(cell_wse)
                                if bridge_arrays is not None:
                                    bridge_helper_used = True
                                    for i in range(int(bridge_arrays["indices"].size)):
                                        src = np.asarray(
                                            native_mod.swe2d_gpu_compute_bridge_coupling_sources(
                                                np.asarray(self.cell_area, dtype=np.float64),
                                                np.asarray([int(bridge_arrays["upstream_cell"][i])], dtype=np.int32),
                                                np.asarray([int(bridge_arrays["downstream_cell"][i])], dtype=np.int32),
                                                np.asarray([float(bridge_arrays["flow_cms"][i])], dtype=np.float64),
                                                np.asarray([float(bridge_arrays["loss_k_upstream"][i])], dtype=np.float64),
                                                np.asarray([float(bridge_arrays["loss_k_downstream"][i])], dtype=np.float64),
                                                float(bridge_arrays["width_m"][i]),
                                                float(dt_s),
                                            ),
                                            dtype=np.float64,
                                        )
                                        bridge_total += src
                flows = None  # Signal that we already handled structure flows
            else:
                flows = self._native_structure_flows(native_mod, cell_wse)
                if flows is None:
                    flows = np.asarray(self.structures.structure_flows(cell_wse), dtype=np.float64)
                    component_sums["structures_native_helper"] = 0.0
                else:
                    component_sums["structures_native_helper"] = 1.0
            if flows is not None and flows.size == len(sts) and flows.size > 0:
                use_bridge_cuda = self.bridge_cuda_coupling and hasattr(native_mod, "swe2d_gpu_compute_bridge_coupling_sources")
                if use_bridge_cuda:
                    non_bridge_mask = np.asarray([st.structure_type != StructureType.BRIDGE for st in sts], dtype=bool)
                    struct_up = np.asarray([int(st.upstream_cell) for st in sts if st.structure_type != StructureType.BRIDGE], dtype=np.int32)
                    struct_dn = np.asarray([int(st.downstream_cell) for st in sts if st.structure_type != StructureType.BRIDGE], dtype=np.int32)
                    struct_q = flows[non_bridge_mask]
                    bridge_arrays = self._bridge_structure_arrays(cell_wse)
                    if bridge_arrays is not None:
                        bridge_helper_used = True
                        for i in range(int(bridge_arrays["indices"].size)):
                            src = np.asarray(
                                native_mod.swe2d_gpu_compute_bridge_coupling_sources(
                                    np.asarray(self.cell_area, dtype=np.float64),
                                    np.asarray([int(bridge_arrays["upstream_cell"][i])], dtype=np.int32),
                                    np.asarray([int(bridge_arrays["downstream_cell"][i])], dtype=np.int32),
                                    np.asarray([float(bridge_arrays["flow_cms"][i])], dtype=np.float64),
                                    np.asarray([float(bridge_arrays["loss_k_upstream"][i])], dtype=np.float64),
                                    np.asarray([float(bridge_arrays["loss_k_downstream"][i])], dtype=np.float64),
                                    float(bridge_arrays["width_m"][i]),
                                    float(dt_s),
                                ),
                                dtype=np.float64,
                            )
                            if src.size != self.n_cells:
                                raise ValueError("bridge source-rate array size mismatch")
                            bridge_id = str(bridge_arrays["structure_id"][i])
                            bridge_plan = bridge_plan_map.get(bridge_id)
                            if bridge_plan is not None:
                                if self.bridge_stacked_coupling_mode == "legacy_scalar":
                                    src = apply_bridge_stacked_source_weight(src, bridge_plan)
                                else:
                                    src = apply_bridge_stacked_phase3_source_weight(
                                        src,
                                        bridge_plan,
                                        self.cell_area,
                                    )
                            bridge_total += src
                else:
                    struct_up = np.asarray([int(st.upstream_cell) for st in sts], dtype=np.int32)
                    struct_dn = np.asarray([int(st.downstream_cell) for st in sts], dtype=np.int32)
                    struct_q = flows
            structure_diag = self.structures.compute_structure_fluxes(float(dt_s), cell_wse)

        if flows is not None:
            total = np.asarray(
                native_mod.swe2d_gpu_compute_coupling_sources(
                    np.asarray(self.cell_area, dtype=np.float64),
                    inlet_cell,
                    inlet_flow,
                    struct_up,
                    struct_dn,
                    struct_q,
                ),
                dtype=np.float64,
            )
        if bridge_helper_used:
            total += bridge_total
            component_sums["bridges"] = float(np.sum(bridge_total))

        self.last_diag = SWE2DCouplingDiagnostics(
            time_s=float(t_s) + float(dt_s),
            dt_s=float(dt_s),
            drainage_max_node_depth=float(drainage_diag.get("max_node_depth", drainage_diag.get("max_node_depth_m", 0.0))),
            drainage_max_link_flow=float(drainage_diag.get("max_link_flow", drainage_diag.get("max_link_flow_cms", 0.0))),
            structure_total_flow=float(structure_diag.get("total_structure_flow", 0.0)),
            source_sum=float(np.sum(total)),
            source_min=float(np.min(total)) if total.size else 0.0,
            source_max=float(np.max(total)) if total.size else 0.0,
            component_sums=component_sums,
        )
        return total


__all__ = [
    "SWE2DCouplingDiagnostics",
    "SWE2DCouplingController",
    "SWE2DDrainageSoA",
    "SWE2DStructuresSoA",
    "SWE2DCouplingSoA",
    "pack_pipe_network_soa",
    "pack_structures_soa",
    "pack_coupling_soa",
]
