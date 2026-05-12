"""Coupling orchestration for SWE2D surface, drainage network, and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from swe2d_drainage_network import SWE2DUrbanDrainageModule
from swe2d_extensions import (
    DrainageSolverMode,
    HydraulicStructureConfig,
    PipeNetworkConfig,
    equivalent_circular_diameter_from_area,
)
from swe2d_structures import SWE2DStructureModule


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
        self._drainage_soa = pack_pipe_network_soa(self.drainage.cfg, self.n_cells) if self.drainage is not None else None
        self._gpu_node_depth: Optional[np.ndarray] = None
        self._gpu_link_flow: Optional[np.ndarray] = None
        self.last_diag = SWE2DCouplingDiagnostics()

    @property
    def cell_area_m2(self) -> np.ndarray:
        return self.cell_area

    @property
    def cell_bed_m(self) -> np.ndarray:
        return self.cell_bed

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
                return self._compute_source_rates_cuda(mod, t_s, dt_s, hh)
        cell_wse = hh + self.cell_bed
        total = np.zeros(self.n_cells, dtype=np.float64)
        component_sums: Dict[str, float] = {}
        drainage_diag: Dict[str, float] = {}
        structure_diag: Dict[str, float] = {}

        if self.drainage is not None:
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
                dt_sub = float(dt_s) / float(n_substeps)
                implicit_iters = max(1, int(getattr(self.drainage.cfg, "implicit_coupling_iterations", 1)))
                coupling_relax = float(getattr(self.drainage.cfg, "implicit_coupling_relaxation", 0.5))
                coupling_relax = min(1.0, max(0.0, coupling_relax))
                area_safe = np.maximum(self.cell_area, 1.0e-12)

                q_cell_acc = np.zeros(self.n_cells, dtype=np.float64)
                diag = {"max_node_depth": 0.0, "max_link_flow": 0.0, "limiter_events": 0.0, "limiter_volume_m3": 0.0}
                nd_state = np.asarray(self._gpu_node_depth, dtype=np.float64)
                lf_state = np.asarray(self._gpu_link_flow, dtype=np.float64)
                hh_sub = np.asarray(hh, dtype=np.float64).copy()
                for _ in range(n_substeps):
                    hh_iter = np.asarray(hh_sub, dtype=np.float64)
                    q_cell_step_last = np.zeros(self.n_cells, dtype=np.float64)
                    diag_step_last = {"max_node_depth": 0.0, "max_link_flow": 0.0, "limiter_events": 0.0, "limiter_volume_m3": 0.0}
                    for _ in range(implicit_iters):
                        wse_iter = hh_iter + self.cell_bed
                        nd_out, lf_out, q_cell_step, diag_step = native_mod.swe2d_gpu_drainage_step(
                            np.asarray(wse_iter, dtype=np.float64),
                            np.asarray(self.cell_area, dtype=np.float64),
                            np.asarray(dsoa.node_invert_elev, dtype=np.float64),
                            np.asarray(dsoa.node_max_depth, dtype=np.float64),
                            np.asarray(dsoa.node_surface_area, dtype=np.float64),
                            np.asarray(dsoa.link_from, dtype=np.int32),
                            np.asarray(dsoa.link_to, dtype=np.int32),
                            np.asarray(dsoa.link_length, dtype=np.float64),
                            np.asarray(dsoa.link_roughness_n, dtype=np.float64),
                            np.asarray(dsoa.link_diameter, dtype=np.float64),
                            np.asarray(dsoa.link_max_flow, dtype=np.float64),
                            np.asarray(dsoa.inlet_cell, dtype=np.int32),
                            np.asarray(dsoa.inlet_node, dtype=np.int32),
                            np.asarray(dsoa.inlet_crest_elev, dtype=np.float64),
                            np.asarray(dsoa.inlet_width, dtype=np.float64),
                            np.asarray(dsoa.inlet_coefficient, dtype=np.float64),
                            np.asarray(dsoa.inlet_max_capture, dtype=np.float64),
                            np.asarray(dsoa.outfall_cell, dtype=np.int32),
                            np.asarray(dsoa.outfall_node, dtype=np.int32),
                            np.asarray(dsoa.outfall_invert_elev, dtype=np.float64),
                            np.asarray(dsoa.outfall_diameter, dtype=np.float64),
                            np.asarray(dsoa.outfall_coefficient, dtype=np.float64),
                            np.asarray(dsoa.outfall_max_flow, dtype=np.float64),
                            np.asarray(dsoa.outfall_zero_storage, dtype=np.int32),
                            np.asarray(hh_iter, dtype=np.float64),
                            nd_state,
                            lf_state,
                            dt_sub,
                            float(getattr(self.drainage.cfg, "gravity", 9.81)),
                            int(dsoa.solver_mode),
                            float(getattr(self.drainage.cfg, "head_deadband_m", 1.0e-3)),
                            float(getattr(self.drainage.cfg, "dynamic_flow_relaxation", 1.0)),
                        )
                        nd_state = np.asarray(nd_out, dtype=np.float64)
                        lf_state = np.asarray(lf_out, dtype=np.float64)
                        q_cell_step_last = np.asarray(q_cell_step, dtype=np.float64)
                        diag_step_last = diag_step
                        # q_cell_step > 0 removes surface water; blend depth update for implicit coupling.
                        hh_target = np.maximum(hh_sub - q_cell_step_last * dt_sub / area_safe, 0.0)
                        hh_iter = (1.0 - coupling_relax) * hh_iter + coupling_relax * hh_target

                    q_cell_acc += q_cell_step_last
                    hh_sub = np.maximum(hh_sub - q_cell_step_last * dt_sub / area_safe, 0.0)
                    diag["max_node_depth"] = max(float(diag.get("max_node_depth", 0.0)), float(diag_step_last.get("max_node_depth", 0.0)))
                    diag["max_link_flow"] = max(float(diag.get("max_link_flow", 0.0)), float(diag_step_last.get("max_link_flow", 0.0)))
                    diag["limiter_events"] = float(diag.get("limiter_events", 0.0)) + float(diag_step_last.get("limiter_events", 0.0))
                    diag["limiter_volume_m3"] = float(diag.get("limiter_volume_m3", 0.0)) + float(diag_step_last.get("limiter_volume_m3", 0.0))

                self._gpu_node_depth = nd_state
                self._gpu_link_flow = lf_state
                self._sync_gpu_state_back_to_drainage()
                q_cell = np.asarray(q_cell_acc / float(n_substeps), dtype=np.float64)
                drainage_diag = {
                    "max_node_depth": float(diag.get("max_node_depth", 0.0)),
                    "max_link_flow": float(diag.get("max_link_flow", 0.0)),
                    "limiter_events": float(diag.get("limiter_events", 0.0)),
                    "limiter_volume_m3": float(diag.get("limiter_volume_m3", 0.0)),
                    "substeps_used": float(n_substeps),
                }
            else:
                drainage_step_diag = self.drainage.solve_network_step(float(dt_s))
                q_cell = np.asarray(
                    self.drainage.apply_surface_exchange(
                        float(dt_s),
                        cell_wse,
                        cell_area_m2=self.cell_area,
                        cell_depth_m=hh,
                    ),
                    dtype=np.float64,
                )
                drainage_diag = {
                    "max_node_depth": float(drainage_step_diag.get("max_node_depth", 0.0)),
                    "max_link_flow": float(drainage_step_diag.get("max_link_flow", 0.0)),
                    "limiter_events": float(drainage_step_diag.get("limiter_events", 0.0)),
                    "limiter_volume_m3": float(drainage_step_diag.get("limiter_volume_m3", 0.0)),
                    "substeps_used": 1.0,
                }

            nz = np.nonzero(np.abs(q_cell) > 0.0)[0]
            if nz.size > 0:
                inlet_cell = nz.astype(np.int32, copy=False)
                # Kernel convention: positive inlet flow removes surface water.
                inlet_flow = (-q_cell[nz]).astype(np.float64, copy=False)
            component_sums["drainage"] = float(np.sum(q_cell / np.maximum(self.cell_area, 1.0e-12)))
            component_sums["drainage_limiter_events"] = float(drainage_diag.get("limiter_events", 0.0))
            component_sums["drainage_limiter_volume_m3"] = float(drainage_diag.get("limiter_volume_m3", 0.0))
            component_sums["drainage_substeps_used"] = float(drainage_diag.get("substeps_used", 1.0))

        if self.structures is not None:
            flows = np.asarray(self.structures.structure_flows(cell_wse), dtype=np.float64)
            sts = list(self.structures.cfg.structures)
            if flows.size == len(sts) and flows.size > 0:
                struct_up = np.asarray([int(st.upstream_cell) for st in sts], dtype=np.int32)
                struct_dn = np.asarray([int(st.downstream_cell) for st in sts], dtype=np.int32)
                struct_q = flows
            structure_diag = self.structures.compute_structure_fluxes(float(dt_s), cell_wse)

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
