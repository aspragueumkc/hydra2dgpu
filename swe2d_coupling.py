"""Coupling orchestration for SWE2D surface, drainage network, and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from swe2d_drainage_network import SWE2DUrbanDrainageModule
from swe2d_extensions import HydraulicStructureConfig, PipeNetworkConfig
from swe2d_structures import SWE2DStructureModule


@dataclass
class SWE2DCouplingDiagnostics:
    time_s: float = 0.0
    dt_s: float = 0.0
    drainage_max_node_depth_m: float = 0.0
    drainage_max_link_flow_cms: float = 0.0
    structure_total_flow_cms: float = 0.0
    source_sum_mps: float = 0.0
    source_min_mps: float = 0.0
    source_max_mps: float = 0.0
    component_sums_mps: Dict[str, float] = field(default_factory=dict)


@dataclass
class SWE2DDrainageSoA:
    node_x: np.ndarray
    node_y: np.ndarray
    node_invert_elev: np.ndarray
    node_max_depth: np.ndarray
    node_surface_area_m2: np.ndarray
    link_from: np.ndarray
    link_to: np.ndarray
    link_length_m: np.ndarray
    link_roughness_n: np.ndarray
    link_diameter_m: np.ndarray
    link_max_flow_cms: np.ndarray
    link_cd: np.ndarray
    inlet_cell: np.ndarray
    inlet_node: np.ndarray
    inlet_crest_elev: np.ndarray
    inlet_width_m: np.ndarray
    inlet_coefficient: np.ndarray
    inlet_max_capture_cms: np.ndarray


@dataclass
class SWE2DStructuresSoA:
    structure_type: np.ndarray
    upstream_cell: np.ndarray
    downstream_cell: np.ndarray
    crest_elev: np.ndarray
    width_m: np.ndarray
    height_m: np.ndarray
    diameter_m: np.ndarray
    length_m: np.ndarray
    roughness_n: np.ndarray
    coeff: np.ndarray
    cd: np.ndarray
    opening: np.ndarray
    q_pump_cms: np.ndarray
    max_flow_cms: np.ndarray


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

    node_x = np.zeros(nn, dtype=np.float64)
    node_y = np.zeros(nn, dtype=np.float64)
    node_invert_elev = np.zeros(nn, dtype=np.float64)
    node_max_depth = np.zeros(nn, dtype=np.float64)
    node_surface_area_m2 = np.zeros(nn, dtype=np.float64)
    for i, nd in enumerate(cfg.nodes):
        node_x[i] = float(nd.x)
        node_y[i] = float(nd.y)
        node_invert_elev[i] = float(nd.invert_elev)
        node_max_depth[i] = float(nd.max_depth)
        node_surface_area_m2[i] = float(nd.metadata.get("surface_area_m2", 50.0))

    link_from = np.full(nl, -1, dtype=np.int32)
    link_to = np.full(nl, -1, dtype=np.int32)
    link_length_m = np.zeros(nl, dtype=np.float64)
    link_roughness_n = np.zeros(nl, dtype=np.float64)
    link_diameter_m = np.zeros(nl, dtype=np.float64)
    link_max_flow_cms = np.full(nl, np.nan, dtype=np.float64)
    link_cd = np.zeros(nl, dtype=np.float64)
    for i, lk in enumerate(cfg.links):
        link_from[i] = int(node_idx.get(lk.from_node_id, -1))
        link_to[i] = int(node_idx.get(lk.to_node_id, -1))
        link_length_m[i] = float(lk.length_m)
        link_roughness_n[i] = float(lk.roughness_n)
        link_diameter_m[i] = float(lk.diameter_m or 0.0)
        link_max_flow_cms[i] = np.nan if lk.max_flow_cms is None else float(lk.max_flow_cms)
        link_cd[i] = float(lk.metadata.get("cd", 0.75))

    inlet_cell = np.full(ni, -1, dtype=np.int32)
    inlet_node = np.full(ni, -1, dtype=np.int32)
    inlet_crest_elev = np.zeros(ni, dtype=np.float64)
    inlet_width_m = np.zeros(ni, dtype=np.float64)
    inlet_coefficient = np.zeros(ni, dtype=np.float64)
    inlet_max_capture_cms = np.full(ni, np.nan, dtype=np.float64)
    for i, it in enumerate(cfg.inlets):
        ci = int(it.cell_id)
        inlet_cell[i] = ci if 0 <= ci < int(n_cells) else -1
        inlet_node[i] = int(node_idx.get(it.node_id, -1))
        inlet_crest_elev[i] = float(it.crest_elev)
        inlet_width_m[i] = float(it.width_m)
        inlet_coefficient[i] = float(it.coefficient)
        inlet_max_capture_cms[i] = np.nan if it.max_capture_cms is None else float(it.max_capture_cms)

    return SWE2DDrainageSoA(
        node_x=node_x,
        node_y=node_y,
        node_invert_elev=node_invert_elev,
        node_max_depth=node_max_depth,
        node_surface_area_m2=node_surface_area_m2,
        link_from=link_from,
        link_to=link_to,
        link_length_m=link_length_m,
        link_roughness_n=link_roughness_n,
        link_diameter_m=link_diameter_m,
        link_max_flow_cms=link_max_flow_cms,
        link_cd=link_cd,
        inlet_cell=inlet_cell,
        inlet_node=inlet_node,
        inlet_crest_elev=inlet_crest_elev,
        inlet_width_m=inlet_width_m,
        inlet_coefficient=inlet_coefficient,
        inlet_max_capture_cms=inlet_max_capture_cms,
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
    width_m = np.zeros(ns, dtype=np.float64)
    height_m = np.zeros(ns, dtype=np.float64)
    diameter_m = np.zeros(ns, dtype=np.float64)
    length_m = np.zeros(ns, dtype=np.float64)
    roughness_n = np.zeros(ns, dtype=np.float64)
    coeff = np.zeros(ns, dtype=np.float64)
    cd = np.zeros(ns, dtype=np.float64)
    opening = np.zeros(ns, dtype=np.float64)
    q_pump_cms = np.zeros(ns, dtype=np.float64)
    max_flow_cms = np.full(ns, np.nan, dtype=np.float64)

    for i, st in enumerate(cfg.structures):
        structure_type[i] = int(st.structure_type)
        iu = int(st.upstream_cell)
        idn = int(st.downstream_cell)
        upstream_cell[i] = iu if 0 <= iu < int(n_cells) else -1
        downstream_cell[i] = idn if 0 <= idn < int(n_cells) else -1
        crest_elev[i] = float(st.crest_elev)
        width_m[i] = float(st.metadata.get("width_m", 0.0))
        height_m[i] = float(st.metadata.get("height_m", 0.0))
        diameter_m[i] = float(st.metadata.get("diameter_m", 0.0))
        length_m[i] = float(st.metadata.get("length_m", 0.0))
        roughness_n[i] = float(st.metadata.get("roughness_n", 0.013))
        coeff[i] = float(st.metadata.get("coeff", 1.7))
        cd[i] = float(st.metadata.get("cd", 0.75))
        opening[i] = float(st.metadata.get("opening", 1.0))
        q_pump_cms[i] = float(st.metadata.get("q_pump_cms", 0.0))
        max_flow_cms[i] = np.nan if st.metadata.get("max_flow_cms") is None else float(st.metadata.get("max_flow_cms"))

    return SWE2DStructuresSoA(
        structure_type=structure_type,
        upstream_cell=upstream_cell,
        downstream_cell=downstream_cell,
        crest_elev=crest_elev,
        width_m=width_m,
        height_m=height_m,
        diameter_m=diameter_m,
        length_m=length_m,
        roughness_n=roughness_n,
        coeff=coeff,
        cd=cd,
        opening=opening,
        q_pump_cms=q_pump_cms,
        max_flow_cms=max_flow_cms,
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
        cell_area_m2: Sequence[float],
        cell_bed_m: Sequence[float],
        drainage: Optional[SWE2DUrbanDrainageModule] = None,
        structures: Optional[SWE2DStructureModule] = None,
        coupling_loop: str = "cpu",
    ):
        self.cell_area_m2 = np.ascontiguousarray(cell_area_m2, dtype=np.float64).ravel()
        self.cell_bed_m = np.ascontiguousarray(cell_bed_m, dtype=np.float64).ravel()
        if self.cell_area_m2.size != self.cell_bed_m.size:
            raise ValueError("cell_area_m2 and cell_bed_m must have the same length")
        self.drainage = drainage
        self.structures = structures
        self.coupling_loop = str(coupling_loop or "cpu").strip().lower()
        if self.coupling_loop not in {"cpu", "cuda"}:
            raise ValueError("coupling_loop must be 'cpu' or 'cuda'")
        self.last_diag = SWE2DCouplingDiagnostics()

    def _native_cuda_module(self):
        try:
            import backwater_swe2d as mod  # type: ignore
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

    @property
    def n_cells(self) -> int:
        return int(self.cell_area_m2.size)

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
                return self._compute_source_rates_cuda(mod, t_s, dt_s, hh)
        cell_wse = hh + self.cell_bed_m
        total = np.zeros(self.n_cells, dtype=np.float64)
        component_sums: Dict[str, float] = {}
        drainage_diag: Dict[str, float] = {}
        structure_diag: Dict[str, float] = {}

        if self.drainage is not None:
            drainage_diag = self.drainage.solve_network_step(float(dt_s))
            src = np.asarray(
                self.drainage.surface_exchange_source_rate(float(dt_s), cell_wse, self.cell_area_m2),
                dtype=np.float64,
            )
            if src.size != self.n_cells:
                raise ValueError("drainage source-rate array size mismatch")
            total += src
            component_sums["drainage"] = float(np.sum(src))

        if self.structures is not None:
            src = np.asarray(
                self.structures.compute_cell_source_rate(float(dt_s), cell_wse, self.cell_area_m2),
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
            drainage_max_node_depth_m=float(drainage_diag.get("max_node_depth_m", 0.0)),
            drainage_max_link_flow_cms=float(drainage_diag.get("max_link_flow_cms", 0.0)),
            structure_total_flow_cms=float(structure_diag.get("total_structure_flow_cms", 0.0)),
            source_sum_mps=float(np.sum(total)),
            source_min_mps=float(np.min(total)) if total.size else 0.0,
            source_max_mps=float(np.max(total)) if total.size else 0.0,
            component_sums_mps=component_sums,
        )
        return total

    def _compute_source_rates_cuda(self, native_mod, t_s: float, dt_s: float, hh: np.ndarray) -> np.ndarray:
        cell_wse = hh + self.cell_bed_m
        drainage_diag: Dict[str, float] = {}
        structure_diag: Dict[str, float] = {}
        component_sums: Dict[str, float] = {}

        inlet_cell = np.empty(0, dtype=np.int32)
        inlet_flow_cms = np.empty(0, dtype=np.float64)
        struct_up = np.empty(0, dtype=np.int32)
        struct_dn = np.empty(0, dtype=np.int32)
        struct_q = np.empty(0, dtype=np.float64)

        if self.drainage is not None:
            drainage_diag = self.drainage.solve_network_step(float(dt_s))
            q_cell = np.asarray(self.drainage.apply_surface_exchange(float(dt_s), cell_wse), dtype=np.float64)
            nz = np.nonzero(np.abs(q_cell) > 0.0)[0]
            if nz.size > 0:
                inlet_cell = nz.astype(np.int32, copy=False)
                # Kernel convention: positive inlet flow removes surface water.
                inlet_flow_cms = (-q_cell[nz]).astype(np.float64, copy=False)
            component_sums["drainage"] = float(np.sum(q_cell / np.maximum(self.cell_area_m2, 1.0e-12)))

        if self.structures is not None:
            flows = np.asarray(self.structures.structure_flows_cms(cell_wse), dtype=np.float64)
            sts = list(self.structures.cfg.structures)
            if flows.size == len(sts) and flows.size > 0:
                struct_up = np.asarray([int(st.upstream_cell) for st in sts], dtype=np.int32)
                struct_dn = np.asarray([int(st.downstream_cell) for st in sts], dtype=np.int32)
                struct_q = flows
            structure_diag = self.structures.compute_structure_fluxes(float(dt_s), cell_wse)

        total = np.asarray(
            native_mod.swe2d_gpu_compute_coupling_sources(
                np.asarray(self.cell_area_m2, dtype=np.float64),
                inlet_cell,
                inlet_flow_cms,
                struct_up,
                struct_dn,
                struct_q,
            ),
            dtype=np.float64,
        )

        self.last_diag = SWE2DCouplingDiagnostics(
            time_s=float(t_s) + float(dt_s),
            dt_s=float(dt_s),
            drainage_max_node_depth_m=float(drainage_diag.get("max_node_depth_m", 0.0)),
            drainage_max_link_flow_cms=float(drainage_diag.get("max_link_flow_cms", 0.0)),
            structure_total_flow_cms=float(structure_diag.get("total_structure_flow_cms", 0.0)),
            source_sum_mps=float(np.sum(total)),
            source_min_mps=float(np.min(total)) if total.size else 0.0,
            source_max_mps=float(np.max(total)) if total.size else 0.0,
            component_sums_mps=component_sums,
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
