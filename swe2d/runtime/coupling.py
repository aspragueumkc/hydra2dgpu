"""Coupling orchestration for SWE2D surface, drainage network, and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
from swe2d.runtime.backend import load_swe2d_native_module
from swe2d import units as _u  # unit-system constants


def _meta_float(meta: dict, key: str, default: float) -> float:
    """meta float."""
    v = meta.get(key)
    if v is None:
        return float(default)
    return float(v)


@dataclass
class SWE2DCouplingDiagnostics:
    """Diagnostics snapshot from one coupling exchange step (model units)."""
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
        """Maximum drainage node depth in model units."""
        return self.drainage_max_node_depth

    @property
    def drainage_max_link_flow_cms(self) -> float:
        """Maximum drainage link flow rate in model units."""
        return self.drainage_max_link_flow

    @property
    def structure_total_flow_cms(self) -> float:
        """Total flow through all structures in model units."""
        return self.structure_total_flow

    @property
    def source_sum_mps(self) -> float:
        """Sum of per-cell source rates in model-length/s."""
        return self.source_sum

    @property
    def source_min_mps(self) -> float:
        """Minimum per-cell source rate."""
        return self.source_min

    @property
    def source_max_mps(self) -> float:
        """Maximum per-cell source rate."""
        return self.source_max

    @property
    def component_sums_mps(self) -> Dict[str, float]:
        """Per-component source sums keyed by component name."""
        return self.component_sums


@dataclass
class SWE2DDrainageSoA:
    """Structure-of-arrays layout for the 1D drainage network (GPU upload)."""
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
    """Structure-of-arrays layout for hydraulic structure parameters (GPU upload)."""
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
    culvert_area: np.ndarray  # L²  (in computation units)
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
class SWE2DCulvertFaceFluxSoA:
    """Face-based flux parameters for culvert structures.

    Only populated for structures where structure_type == CULVERT
    and the face_flux coupling mode is active.
    """
    # Index into the full structure arrays (culverts only)
    structure_index: np.ndarray       # [n_culvert_faces]
    # Face geometry (unit normal from upstream → downstream centroid)
    face_nx: np.ndarray              # [n_culvert_faces]
    face_ny: np.ndarray              # [n_culvert_faces]
    face_width: np.ndarray            # [n_culvert_faces] culvert face width L_s
    # Donor/receiver cell indices
    donor_cell: np.ndarray            # [n_culvert_faces] upstream cell
    receiver_cell: np.ndarray         # [n_culvert_faces] downstream cell
    # Invert elevation for depth limiting
    invert_elev: np.ndarray           # [n_culvert_faces]
    # Depth limiter safety factor (0..1, default 0.5)
    depth_safety_factor: np.ndarray   # [n_culvert_faces]
    # Donor-cell area for depth safety limiter
    donor_cell_area: np.ndarray       # [n_culvert_faces]


@dataclass
class SWE2DCouplingSoA:
    """Container for optional drainage and structures SoA data."""
    n_cells: int
    drainage: Optional[SWE2DDrainageSoA] = None
    structures: Optional[SWE2DStructuresSoA] = None


def pack_pipe_network_soa(cfg: Optional[PipeNetworkConfig], n_cells: int) -> Optional[SWE2DDrainageSoA]:
    """Pack a PipeNetworkConfig into flat SoA arrays for GPU consumption."""
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
        node_surface_area[i] = _meta_float(nd.metadata, "surface_area", _meta_float(nd.metadata, "surface_area_m2", 50.0))

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
            area_link = _meta_float(lk.metadata, "area_m2", 0.0)
            d_link = equivalent_circular_diameter_from_area(area_link)
        link_diameter[i] = d_link
        link_max_flow[i] = np.nan if lk.max_flow is None else float(lk.max_flow)
        link_cd[i] = _meta_float(lk.metadata, "cd", 0.75)

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


def pack_structures_soa(cfg: Optional[HydraulicStructureConfig], n_cells: int, model_to_ft: float = 1.0, cell_bed: Optional[np.ndarray] = None, log_fn: Optional[Callable[[str], None]] = None) -> Optional[SWE2DStructuresSoA]:
    """Pack hydraulic structure metadata into SoA arrays for GPU/CPU computation.

    Input metadata and output SoA arrays are in MODEL UNITS (meters for SI,
    feet for USC).  The kernel receives model units directly; its culvert path
    converts to feet internally using the caller-supplied model_to_ft parameter.

    The model_to_ft argument is retained for API compatibility but should
    always be 1.0 (no pre-conversion) with the new unit-agnostic kernel.
    """
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
    culvert_area = np.zeros(ns, dtype=np.float64)
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

    # Helper: read metadata value, return default only when key is missing
    # (NOT when value is 0.0 — zero is a legitimate value for invert

    for i, st in enumerate(cfg.structures):
        structure_type[i] = int(st.structure_type)
        iu = int(st.upstream_cell)
        idn = int(st.downstream_cell)
        upstream_cell[i] = iu if 0 <= iu < int(n_cells) else -1
        downstream_cell[i] = idn if 0 <= idn < int(n_cells) else -1
        # Geometry in model units — kernel converts to feet via model_to_ft.
        crest_elev[i] = _meta_float(st.metadata, "crest_elev", st.crest_elev)
        width[i] = _meta_float(st.metadata, "width", 0.0)
        height[i] = _meta_float(st.metadata, "height", 0.0)
        diameter[i] = _meta_float(st.metadata, "diameter", 0.0)
        length[i] = _meta_float(st.metadata, "length", 0.0)
        roughness_n[i] = _meta_float(st.metadata, "roughness_n", 0.013)
        coeff[i] = _meta_float(st.metadata, "coeff", 1.7)
        cd[i] = _meta_float(st.metadata, "cd", 0.75)
        opening[i] = _meta_float(st.metadata, "opening", 1.0)
        q_pump[i] = _meta_float(st.metadata, "q_pump", 0.0)
        max_flow[i] = np.nan if st.metadata.get("max_flow") is None else float(st.metadata.get("max_flow"))
        culvert_code[i] = int(_meta_float(st.metadata, "culvert_code", 1))
        culvert_shape[i] = int(culvert_shape_map.get(str(st.metadata.get("culvert_shape", "circular") or "circular").strip().lower(), 0))
        culvert_rise[i] = _meta_float(st.metadata, "culvert_rise", _meta_float(st.metadata, "height", _meta_float(st.metadata, "diameter", 0.0)))
        culvert_span[i] = _meta_float(st.metadata, "culvert_span", _meta_float(st.metadata, "width", culvert_rise[i]))
        culvert_area[i] = _meta_float(st.metadata, "culvert_area_m2", _meta_float(st.metadata, "area_m2", 0.0))  # L²
        culvert_barrels[i] = _meta_float(st.metadata, "culvert_barrels", 1.0)
        culvert_slope[i] = _meta_float(st.metadata, "culvert_slope", 0.0)
        raw_inlet = st.metadata.get("inlet_invert_elev")
        raw_outlet = st.metadata.get("outlet_invert_elev")
        if raw_inlet is not None:
            inlet_invert_elev[i] = float(raw_inlet)
        elif cell_bed is not None and 0 <= st.upstream_cell < len(cell_bed):
            inlet_invert_elev[i] = float(cell_bed[st.upstream_cell])
            if log_fn:
                log_fn(f"inlet_invert_elev not set for structure {st.structure_id} — defaulting to bed elevation {cell_bed[st.upstream_cell]:.3f}")
        else:
            inlet_invert_elev[i] = st.crest_elev
            if log_fn:
                log_fn(f"inlet_invert_elev not set for structure {st.structure_id} — defaulting to crest elevation {st.crest_elev:.3f}")
        if raw_outlet is not None:
            outlet_invert_elev[i] = float(raw_outlet)
        elif cell_bed is not None and 0 <= st.downstream_cell < len(cell_bed):
            outlet_invert_elev[i] = float(cell_bed[st.downstream_cell])
            if log_fn:
                log_fn(f"outlet_invert_elev not set for structure {st.structure_id} — defaulting to bed elevation {cell_bed[st.downstream_cell]:.3f}")
        else:
            outlet_invert_elev[i] = inlet_invert_elev[i]
        entrance_loss_k[i] = _meta_float(st.metadata, "entrance_loss_k", _meta_float(st.metadata, "inlet_loss_k", 0.5))
        exit_loss_k[i] = _meta_float(st.metadata, "exit_loss_k", _meta_float(st.metadata, "outlet_loss_k", 1.0))
        embankment_enabled[i] = int(_meta_float(st.metadata, "embankment_enabled", 0.0))
        embankment_crest_elev[i] = _meta_float(st.metadata, "embankment_crest_elev", _meta_float(st.metadata, "road_crest_elev", st.crest_elev))
        embankment_overflow_width[i] = _meta_float(st.metadata, "embankment_overflow_width", _meta_float(st.metadata, "road_overflow_width", _meta_float(st.metadata, "width", 0.0)))
        embankment_weir_coeff[i] = _meta_float(st.metadata, "embankment_weir_coeff", _meta_float(st.metadata, "road_weir_coeff", 1.7))

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
        culvert_area=culvert_area,
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
    cell_bed: Optional[np.ndarray] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> SWE2DCouplingSoA:
    """Pack both drainage and structures into a single SWE2DCouplingSoA."""
    return SWE2DCouplingSoA(
        n_cells=int(n_cells),
        drainage=pack_pipe_network_soa(pipe_network, n_cells),
        structures=pack_structures_soa(hydraulic_structures, n_cells, cell_bed=cell_bed, log_fn=log_fn),
    )


class SWE2DCouplingController:
    """Combine optional drainage and structure modules into one source callback."""

    def __init__(
        self,
        cell_area: Optional[Sequence[float]] = None,
        cell_bed: Optional[Sequence[float]] = None,
        drainage: Optional[SWE2DUrbanDrainageModule] = None,
        structures: Optional[SWE2DStructureModule] = None,
        drainage_gpu_method: str = "step",
        culvert_solver_mode: int = 0,
        bridge_cuda_coupling: bool = False,
        bridge_stacked_coupling_mode: str = "phase3_spatial",
        length_scale_si_to_model: float = 1.0,
        culvert_face_flux_mode: str = "face_flux",
        use_redistribution: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
        inv_cell_perm: Optional[np.ndarray] = None,
):
        """Coupling controller for SWE2D surface/drainage/structure exchange.

        Args:
            length_scale_si_to_model: SI meters per model unit (e.g. 0.3048
                for US-foot CRS, 1.0 for metric CRS).  Used to configure
                the unit system and compute model_to_ft for HDS-5 culverts.
        """
        if cell_area is None or cell_bed is None:
            raise ValueError("cell_area and cell_bed are required")

        self.cell_area = np.ascontiguousarray(cell_area, dtype=np.float64).ravel()
        self.cell_bed = np.ascontiguousarray(cell_bed, dtype=np.float64).ravel()
        self._log_callback: Optional[Callable[[str], None]] = None
        self._inv_cell_perm: Optional[np.ndarray] = inv_cell_perm
        # Optional cell centroid coordinates for influence-width redistribution.
        self._cell_cx: Optional[np.ndarray] = None
        self._cell_cy: Optional[np.ndarray] = None
        # Configure unit system from CRS-derived length scale.
        self._si_m_per_model = max(1.0e-6, float(length_scale_si_to_model))
        _u.configure(self._si_m_per_model)
        self._model_to_ft = _u.model_to_ft()
        self._gravity = _u.gravity()
        # Kernel works in model units.  HDS-5 culvert tables are the only
        # code path that converts to feet internally (via model_to_ft).
        if self.cell_area.size != self.cell_bed.size:
            raise ValueError("cell_area and cell_bed must have the same length")
        self.drainage = drainage
        self.structures = structures
        self.coupling_loop = "cuda"  # GPU-only
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
        # Face-based culvert flux coupling: "off" | "face_flux"
        raw_mode = str(culvert_face_flux_mode or "off").strip().lower()
        # Allow BACKWATER_DISABLE_FACE_FLUX env override for debugging
        if os.environ.get("BACKWATER_DISABLE_FACE_FLUX", "").strip() in ("1", "true", "yes"):
            raw_mode = "off"
        self.culvert_face_flux_mode = raw_mode
        if self.culvert_face_flux_mode not in {"off", "face_flux"}:
            raise ValueError("culvert_face_flux_mode must be 'off' or 'face_flux'")
        self._face_flux_soa: Optional[SWE2DCulvertFaceFluxSoA] = None
        self._enquiry_up_cell: Optional[np.ndarray] = None
        self._enquiry_dn_cell: Optional[np.ndarray] = None
        self._culvert_face_flux_preloaded = False
        self._culvert_table_n_hw = max(
            8, int(os.environ.get("BACKWATER_SWE2D_CULVERT_TABLE_N_HW", "32"))
        )
        self._culvert_table_n_tw = max(
            8, int(os.environ.get("BACKWATER_SWE2D_CULVERT_TABLE_N_TW", "16"))
        )
        self._culvert_table_uploaded = False
        self._culvert_solver_mode_applied = False
        self._persistent_coupling_preloaded = False
        # Set to True by apply_native_device_sources after it writes combined
        # sources to d_external_source_mps.  _compute_source_rates_cuda checks
        # this and short-circuits to avoid re-running GPU work.
        self._coupling_applied_this_timestep = False
        self._use_redistribution = bool(use_redistribution)
        self._drainage_soa = pack_pipe_network_soa(self.drainage.cfg, self.n_cells) if self.drainage is not None else None
        self._structures_soa = pack_structures_soa(self.structures.cfg, self.n_cells, model_to_ft=1.0, cell_bed=self.cell_bed, log_fn=self._log) if self.structures is not None else None
        self._structures_cfg = tuple(self.structures.cfg.structures) if self.structures is not None else tuple()
        self._structure_count = len(self._structures_cfg)
        self._last_structure_flows: Optional[np.ndarray] = None  # ponytail: per-element flows
        self._log(
            f"[COUPLING_INIT] coupling_loop={self.coupling_loop} "
            f"face_flux_mode={self.culvert_face_flux_mode} "
            f"n_structures={self._structure_count} "
            f"drainage={'yes' if self.drainage is not None else 'no'} "
            f"n_cells={self.n_cells} "
            f"model_to_ft={self._model_to_ft:.4f}"
        )
        if self._structure_count > 0:
            self._structure_bridge_mask = np.asarray(
                [st.structure_type == StructureType.BRIDGE for st in self._structures_cfg],
                dtype=bool,
            )
            self._structure_non_bridge_mask = ~self._structure_bridge_mask
            self._structure_bridge_indices = np.flatnonzero(self._structure_bridge_mask).astype(np.int32, copy=False)
            self._enabled_bridge_indices = np.asarray(
                [
                    i for i, st in enumerate(self._structures_cfg)
                    if st.enabled and st.structure_type == StructureType.BRIDGE
                ],
                dtype=np.int32,
            )
        else:
            self._structure_bridge_mask = np.zeros(0, dtype=bool)
            self._structure_non_bridge_mask = np.zeros(0, dtype=bool)
            self._structure_bridge_indices = np.zeros(0, dtype=np.int32)
            self._enabled_bridge_indices = np.zeros(0, dtype=np.int32)
        # Influence-width redistribution data (computed by _build_redistribution_data)
        self._redist_offsets: Optional[np.ndarray] = None
        self._redist_cell_idx: Optional[np.ndarray] = None
        self._redist_weights: Optional[np.ndarray] = None
        self._has_bridge_structures = bool(self._structure_bridge_indices.size > 0)
        self._has_enabled_bridge_structures = bool(self._enabled_bridge_indices.size > 0)
        self._n_non_bridge_structures = int(np.sum(self._structure_non_bridge_mask))
        self._native_cuda_mod_cache = None
        self._native_cuda_mod_checked = False
        self._gpu_node_depth: Optional[np.ndarray] = None
        self._gpu_link_flow: Optional[np.ndarray] = None
        self._gpu_drainage_static_args: Optional[Dict[str, np.ndarray]] = None
        self.last_diag = SWE2DCouplingDiagnostics()
        self._log_callback = log_callback

    def set_cell_centroids(self, cx: np.ndarray, cy: np.ndarray) -> None:
        """Provide cell centroid coordinates for influence-width redistribution."""
        self._cell_cx = np.ascontiguousarray(cx, dtype=np.float64).ravel()
        self._cell_cy = np.ascontiguousarray(cy, dtype=np.float64).ravel()

    def _build_redistribution_data(self) -> None:
        """Pre-compute redistribution weights for structures with influence_width.

        For each structure end, finds cells within a corridor of width
        `influence_width_m` perpendicular to the structure axis, centered
        on the structure endpoint.  Weights are uniform per cell.
        Results are stored as flat arrays with per-structure offsets.
        """
        if self.structures is None or self._cell_cx is None or self._cell_cy is None:
            self._redist_offsets = np.array([0], dtype=np.int32)
            self._redist_cell_idx = np.empty(0, dtype=np.int32)
            self._redist_weights = np.empty(0, dtype=np.float64)
            return

        n_cells = self._cell_cx.size
        offsets = [0]
        all_idx: list = []
        all_w: list = []

        if not self._use_redistribution:
            self._log("redistribution override: disabled via widget, using default values")
            self._redist_offsets = np.array([0], dtype=np.int32)
            self._redist_cell_idx = np.empty(0, dtype=np.int32)
            self._redist_weights = np.empty(0, dtype=np.float64)
            return

        for st in self.structures.cfg.structures:
            md = st.metadata
            use_redist = int(md.get("use_redistribution", 0))
            if use_redist == 0:
                self._log("redistribution override: false, using default values")
                offsets.append(offsets[-1])
                continue
            if use_redist != 1:
                self._log("redistribution override: false, using default values")
                offsets.append(offsets[-1])
                continue

            iw = _meta_float(md, "influence_width", 0.0)
            if iw <= 0.0:
                raise ValueError(
                    f"Redistribution override is enabled for structure {st.structure_id} "
                    f"but influence_width is missing or zero. Set use_redistribution to 0 "
                    f"in the GeoPackage if redistribution was not intended."
                )
            # Structure line endpoints from the feature geometry
            p0x = float(md.get("axis_x0", 0.0))
            p0y = float(md.get("axis_y0", 0.0))
            p1x = float(md.get("axis_x1", 0.0))
            p1y = float(md.get("axis_y1", 0.0))
            if p0x == 0.0 and p0y == 0.0 and p1x == 0.0 and p1y == 0.0:
                offsets.append(offsets[-1])
                continue

            # Structure axis and perpendicular normal
            dx = p1x - p0x
            dy = p1y - p0y
            length = max(1.0e-12, math.sqrt(dx * dx + dy * dy))
            nx = -dy / length  # perpendicular unit normal
            ny = dx / length
            # Streamwise direction
            sx = dx / length
            sy = dy / length

            half_iw = iw / 2.0

            # Find cells within the perpendicular corridor
            # centered on the structure line midpoint, extending half_iw
            # in both perpendicular directions.
            cx = self._cell_cx
            cy = self._cell_cy
            midx = (p0x + p1x) * 0.5
            midy = (p0y + p1y) * 0.5

            # Perpendicular distance from structure axis
            perp_dist = np.abs((cx - midx) * nx + (cy - midy) * ny)
            # Streamwise projection (to limit along-axis extent)
            along = (cx - midx) * sx + (cy - midy) * sy
            half_len = length * 0.5 + half_iw  # extend slightly beyond ends

            mask = (perp_dist <= half_iw) & (np.abs(along) <= half_len)
            sel_idx = np.flatnonzero(mask).astype(np.int32)

            if sel_idx.size == 0:
                offsets.append(offsets[-1])
                continue

            # Weights: for now uniform (all 1.0). Could be refined to
            # use distance-from-structure or cell-area weighting.
            sel_w = np.ones(sel_idx.size, dtype=np.float64)

            all_idx.append(sel_idx)
            all_w.append(sel_w)
            offsets.append(offsets[-1] + sel_idx.size)

        self._redist_offsets = np.array(offsets, dtype=np.int32)
        if all_idx:
            self._redist_cell_idx = np.concatenate(all_idx).astype(np.int32)
            self._redist_weights = np.concatenate(all_w).astype(np.float64)
        else:
            self._redist_cell_idx = np.empty(0, dtype=np.int32)
            self._redist_weights = np.empty(0, dtype=np.float64)

    def _log(self, msg: str) -> None:
        """Route a message to the runtime log callback, if any."""
        if callable(self._log_callback):
            self._log_callback(str(msg))

    # ── Face-based culvert flux coupling ──────────────────────────────────
    def _build_face_flux_soa(self) -> Optional[SWE2DCulvertFaceFluxSoA]:
        """Build SoA for face-based culvert flux coupling.

        Computes face normals from cell centroids, determines face widths,
        and packs invert elevations and depth safety factors for all active
        culvert structures.
        """
        if self.structures is None or self.culvert_face_flux_mode != "face_flux":
            return None
        if self._cell_cx is None or self._cell_cy is None:
            return None  # need cell centroids

        cfg = self.structures.cfg
        culvert_indices = [
            i for i, st in enumerate(cfg.structures)
            if st.structure_type == StructureType.CULVERT and st.enabled
        ]
        if not culvert_indices:
            return None

        n = len(culvert_indices)
        struct_idx = np.array(culvert_indices, dtype=np.int32)
        donor_cell = np.zeros(n, dtype=np.int32)
        receiver_cell = np.zeros(n, dtype=np.int32)
        face_nx = np.zeros(n, dtype=np.float64)
        face_ny = np.zeros(n, dtype=np.float64)
        face_width = np.zeros(n, dtype=np.float64)
        invert_elev = np.zeros(n, dtype=np.float64)
        depth_safety = np.full(n, 0.5, dtype=np.float64)  # default α = 0.5
        donor_cell_area = np.ones(n, dtype=np.float64)

        for j, i in enumerate(culvert_indices):
            st = cfg.structures[i]
            cu = int(st.upstream_cell)
            cd = int(st.downstream_cell)
            if cu < 0 or cd < 0 or cu >= self.n_cells or cd >= self.n_cells:
                continue

            # Face normal from upstream → downstream centroid
            dx = self._cell_cx[cd] - self._cell_cx[cu]
            dy = self._cell_cy[cd] - self._cell_cy[cu]
            length = max(1.0e-12, math.sqrt(dx * dx + dy * dy))
            face_nx[j] = dx / length
            face_ny[j] = dy / length

            donor_cell[j] = cu
            receiver_cell[j] = cd
            donor_cell_area[j] = float(self.cell_area[cu])

            # Face width: culvert_span for box, diameter for circular
            md = st.metadata
            fwo = _meta_float(md, "face_width_override", 0.0)
            if fwo > 0.0:
                face_width[j] = fwo
            else:
                shape = str(md.get("culvert_shape", "circular")).strip().lower()
                if shape in ("box", "rect", "rectangular"):
                    face_width[j] = _meta_float(md, "culvert_span", _meta_float(md, "width", 1.0))
                else:
                    face_width[j] = _meta_float(md, "diameter", _meta_float(md, "culvert_rise", 1.0))

            invert_elev[j] = _meta_float(md, "inlet_invert_elev", st.crest_elev)
            depth_safety[j] = _meta_float(md, "face_flux_depth_safety", 0.5)

        # ── Compute enquiry cells for total-energy driving head ──────────
        # For each culvert face, find cells offset from the face in the
        # outward-normal direction.  WSE + velocity head at these cells is
        # used as the driving head for the culvert solver, avoiding the
        # local drawdown singularity at the face cell.
        enquiry_up_cell = np.full(n, -1, dtype=np.int32)
        enquiry_dn_cell = np.full(n, -1, dtype=np.int32)
        enq_offset = float(self._structures_cfg[0].metadata.get(
            "enquiry_offset", 2.0)) if self._structure_count > 0 else 2.0
        cx = np.asarray(self._cell_cx, dtype=np.float64).ravel()
        cy = np.asarray(self._cell_cy, dtype=np.float64).ravel()
        for j in range(n):
            cu = int(donor_cell[j])
            cd = int(receiver_cell[j])
            if cu < 0 or cd >= cx.size:
                enquiry_up_cell[j] = cu
                enquiry_dn_cell[j] = cd
                continue
            # Upstream enquiry: offset opposite face normal from donor centroid
            nx = face_nx[j]
            ny = face_ny[j]
            cell_size = math.sqrt(max(self.cell_area[cu], self.cell_area[cd]))
            offset = enq_offset * cell_size
            enq_x = cx[cu] - nx * offset
            enq_y = cy[cu] - ny * offset
            dist2 = (cx - enq_x)**2 + (cy - enq_y)**2
            best = int(np.argmin(dist2))
            enquiry_up_cell[j] = best if 0 <= best < self.n_cells else cu
            # Downstream enquiry: offset along face normal from receiver centroid
            enq_x = cx[cd] + nx * offset
            enq_y = cy[cd] + ny * offset
            dist2 = (cx - enq_x)**2 + (cy - enq_y)**2
            best = int(np.argmin(dist2))
            enquiry_dn_cell[j] = best if 0 <= best < self.n_cells else cd

        return SWE2DCulvertFaceFluxSoA(
            structure_index=struct_idx,
            face_nx=face_nx,
            face_ny=face_ny,
            face_width=face_width,
            donor_cell=donor_cell,
            receiver_cell=receiver_cell,
            invert_elev=invert_elev,
            depth_safety_factor=depth_safety,
            donor_cell_area=donor_cell_area,
        ), enquiry_up_cell, enquiry_dn_cell

    def _ensure_culvert_face_flux_preloaded(self, native_mod) -> None:
        """Upload culvert face-flux geometry to GPU if not yet done."""
        if self._culvert_face_flux_preloaded:
            return
        if self.culvert_face_flux_mode != "face_flux":
            return
        if not hasattr(native_mod, "swe2d_gpu_upload_culvert_face_flux_params"):
            return

        # Build the SoA if not yet built
        if self._face_flux_soa is None:
            result = self._build_face_flux_soa()
            if result is None:
                self._face_flux_soa = None
                self._enquiry_up_cell = None
                self._enquiry_dn_cell = None
                self._log("[COUPLING_FF] _build_face_flux_soa returned None (no culvert structures with face data)")
            else:
                self._face_flux_soa, self._enquiry_up_cell, self._enquiry_dn_cell = result
                self._log(f"[COUPLING_FF] built face-flux SoA: {self._face_flux_soa.structure_index.size} face(s)")
        if self._face_flux_soa is None or self._face_flux_soa.structure_index.size == 0:
            self._culvert_face_flux_preloaded = True
            self._log("[COUPLING_FF] no face-flux faces — marked preloaded (no upload)")
            return

        ff = self._face_flux_soa
        self._log(
            f"[COUPLING_FF] uploading {ff.structure_index.size} face(s) to GPU, "
            f"n_cells={self.n_cells}"
        )
        # Build kwargs for upload, adding enquiry cells if available
        upload_kwargs = dict(
            culvert_struct_idx=np.ascontiguousarray(ff.structure_index, dtype=np.int32),
            face_nx=np.ascontiguousarray(ff.face_nx, dtype=np.float64),
            face_ny=np.ascontiguousarray(ff.face_ny, dtype=np.float64),
            face_width=np.ascontiguousarray(ff.face_width, dtype=np.float64),
            donor_cell=self._remap_cells_for_gpu(np.asarray(ff.donor_cell, dtype=np.int32)),
            receiver_cell=self._remap_cells_for_gpu(np.asarray(ff.receiver_cell, dtype=np.int32)),
            invert_elev=np.ascontiguousarray(ff.invert_elev, dtype=np.float64),
            depth_safety=np.ascontiguousarray(ff.depth_safety_factor, dtype=np.float64),
            donor_cell_area=np.ascontiguousarray(ff.donor_cell_area, dtype=np.float64),
            use_face_flux=True,
        )
        if self._enquiry_up_cell is not None and self._enquiry_dn_cell is not None:
            upload_kwargs["enquiry_up_cell"] = self._remap_cells_for_gpu(
                np.asarray(self._enquiry_up_cell, dtype=np.int32))
            upload_kwargs["enquiry_dn_cell"] = self._remap_cells_for_gpu(
                np.asarray(self._enquiry_dn_cell, dtype=np.int32))
        try:
            native_mod.swe2d_gpu_upload_culvert_face_flux_params(**upload_kwargs)
            self._culvert_face_flux_preloaded = True
        except Exception as exc:
            self._log(f"[WARNING] culvert face-flux upload failed: {exc}")
            # Do NOT set _culvert_face_flux_preloaded = True here.
            # The GPU device state may not be initialized yet on the first
            # call.  Leaving the flag False allows retry on the next step
            # once the solver has allocated its device buffers.

    def _apply_redistribution(
        self,
        source_rate: np.ndarray,
        structure_flows: np.ndarray,
        up_cells: np.ndarray,
        dn_cells: np.ndarray,
        native_mod=None,
    ) -> np.ndarray:
        """Apply influence-width redistribution via GPU kernel (no CPU fallback).

        Args:
            source_rate: Per-cell source rates [m/s], will be modified in-place.
            structure_flows: Per-structure flow [m³/s].
            up_cells: Original single upstream cell indices.
            dn_cells: Original single downstream cell indices.
            native_mod: Native CUDA module with
                ``swe2d_gpu_redistribute_structure_sources`` binding.

        Returns:
            The modified source_rate array (same object as input).

        Raises:
            RuntimeError: If the GPU redistribution function is unavailable
                or the kernel call fails.
        """
        if self._redist_offsets is None or self._redist_offsets.size <= 1:
            return source_rate  # no redistribution data

        n_struct = len(structure_flows)
        if n_struct == 0:
            return source_rate

        if not (native_mod is not None and hasattr(native_mod, "swe2d_gpu_redistribute_structure_sources")):
            raise RuntimeError(
                "GPU redistribution function swe2d_gpu_redistribute_structure_sources "
                "is unavailable — no CPU fallback."
            )
        try:
            return np.asarray(
                native_mod.swe2d_gpu_redistribute_structure_sources(
                    np.asarray(source_rate, dtype=np.float64, order='C'),
                    np.asarray(self._redist_offsets, dtype=np.int32, order='C'),
                    np.asarray(self._redist_cell_idx, dtype=np.int32, order='C'),
                    np.asarray(self._redist_weights, dtype=np.float64, order='C'),
                    np.asarray(structure_flows, dtype=np.float64, order='C'),
                    np.asarray(up_cells, dtype=np.int32, order='C'),
                    np.asarray(dn_cells, dtype=np.int32, order='C'),
                    np.asarray(self.cell_area, dtype=np.float64, order='C'),
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            raise RuntimeError(
                "GPU redistribution kernel failed — no CPU fallback. "
                f"Error: {exc}"
            )

    def _native_cuda_module(self):
        """Load and cache the native CUDA module (returns None if unavailable)."""
        if self._native_cuda_mod_checked:
            return self._native_cuda_mod_cache
        mod = load_swe2d_native_module()
        if mod is None:
            self._native_cuda_mod_checked = True
            self._native_cuda_mod_cache = None
            return None
        if not hasattr(mod, "swe2d_gpu_compute_coupling_sources"):
            self._native_cuda_mod_checked = True
            self._native_cuda_mod_cache = None
            return None
        if not bool(mod.swe2d_gpu_available()):
            self._native_cuda_mod_checked = True
            self._native_cuda_mod_cache = None
            return None
        self._native_cuda_mod_checked = True
        self._native_cuda_mod_cache = mod
        return mod

    def _ensure_native_culvert_solver_mode(self, native_mod) -> None:
        """Upload culvert lookup tables and set the GPU culvert solver mode."""
        if self._culvert_solver_mode_applied and self._culvert_solver_mode_applied == self.culvert_solver_mode:
            return
        if not hasattr(native_mod, "swe2d_gpu_set_culvert_solver_mode"):
            return
        if (
            self.culvert_solver_mode == 1
            and self._structures_soa is not None
            and hasattr(native_mod, "swe2d_gpu_build_culvert_tables")
        ):
            ssoa = self._structures_soa
            try:
                table_data, table_header = native_mod.swe2d_gpu_build_culvert_tables(
                    np.asarray(ssoa.culvert_code, dtype=np.int32),
                    np.asarray(ssoa.culvert_shape, dtype=np.int32),
                    np.asarray(ssoa.culvert_rise, dtype=np.float64),
                    np.asarray(ssoa.culvert_span, dtype=np.float64),
                    np.asarray(ssoa.diameter, dtype=np.float64),
                    np.asarray(ssoa.length, dtype=np.float64),
                    np.asarray(ssoa.roughness_n, dtype=np.float64),
                    np.asarray(ssoa.culvert_slope, dtype=np.float64),
                    np.asarray(ssoa.entrance_loss_k, dtype=np.float64),
                    np.asarray(ssoa.exit_loss_k, dtype=np.float64),
                    self._model_to_ft,
                    int(self._culvert_table_n_hw),
                    int(self._culvert_table_n_tw),
                )
                native_mod.swe2d_gpu_set_culvert_solver_mode(
                    1,
                    np.asarray(table_data, dtype=np.float64),
                    np.asarray(table_header, dtype=np.float64),
                    int(self._culvert_table_n_hw),
                    int(self._culvert_table_n_tw),
                )
                self._culvert_table_uploaded = True
                self._culvert_solver_mode_applied = 1
                return
            except Exception:
                self._culvert_table_uploaded = False
        try:
            # Fallback to mode 0 (direct secant solver) if table mode failed
            fallback_mode = 0
            native_mod.swe2d_gpu_set_culvert_solver_mode(fallback_mode)
        except Exception:
            self._log(f"[WARNING] Unexpected error silently caught")
        self._culvert_solver_mode_applied = 0




    def apply_native_device_sources(self, t_s: float, dt_s: float) -> bool:
        """DEPRECATED (Phase 6): C++ handles source freshness internally.
        Retained for non-GPU test host readback only.

        Attempt full on-device source update without host state fetch.

        Returns True when external sources were written on device.

        When redistribution is active and the persistent on-device function
        is available, redistribution is also applied on-device (eliminating
        all D2H/H2D transfers of the source array).

        Raises RuntimeError if the GPU path is required but unavailable
        (no Python fallback — all coupling must go through the GPU path).
        """
        _ = (t_s, dt_s)
        if self.structures is None and self.drainage is None:
            return False
        if self._has_enabled_bridge_structures:
            raise RuntimeError(
                "GPU coupling path does not support bridge structures. "
                "Disable bridge structures or rebuild with bridge GPU support."
            )

        native_mod = self._native_cuda_module()
        if native_mod is None:
            raise RuntimeError(
                "CUDA module not available for GPU coupling path. "
                "The Python coupling fallback has been removed — "
                "ensure the native hydra_swe2d module is built and importable."
            )

        # Need compute_coupling_full_on_device for the final on-device write.
        # Without it we cannot return True (no way to get sources to device).
        if not hasattr(native_mod, "swe2d_gpu_compute_coupling_full_on_device"):
            raise RuntimeError(
                "swe2d_gpu_compute_coupling_full_on_device not found in native module. "
                "Rebuild hydra_swe2d with the persistent GPU coupling path enabled."
            )

        self._ensure_native_culvert_solver_mode(native_mod)

        # ── Drainage: compute q_cell on-device via swe2d_gpu_drainage_step ──
        # cell_wse and cell_depth are passed as None so the C++ function computes
        # WSE = h + zb and copies h directly from device-resident state, avoiding
        # all D2H readback of h.
        d_drainage_q = False  # ponytail: set True for device-resident coupling
        if self.drainage is not None:
            # Fast-path: skip GPU drainage step when no water is in the system.
            # Mirrors the check in _compute_source_rates_cuda (lines 1557-1562).
            self._ensure_gpu_drainage_state()
            node_depth_state = np.asarray(self._gpu_node_depth, dtype=np.float64)
            link_flow_state = np.asarray(self._gpu_link_flow, dtype=np.float64)
            node_active = bool(np.any(node_depth_state > 1.0e-3))
            link_active = bool(np.any(np.abs(link_flow_state) > 1.0e-10))
            if self.structures is None and not node_active and not link_active:
                return False
            # Ensure persisted device buffer for q_cell
            if not getattr(self, "_drainage_q_ensured", False):
                if hasattr(native_mod, "swe2d_gpu_ensure_drainage_q_buf"):
                    native_mod.swe2d_gpu_ensure_drainage_q_buf(self.n_cells)
                    self._drainage_q_ensured = True
            # GPU drainage step — same call as _compute_source_rates_cuda.
            if (self._drainage_soa is not None
                and hasattr(native_mod, "swe2d_gpu_drainage_step")):
                self._ensure_gpu_drainage_state()
                dsoa = self._drainage_soa
                solver_mode = DrainageSolverMode(int(dsoa.solver_mode))
                static_args = self._ensure_gpu_drainage_static_args()
                if static_args is None:
                    return False
                node_depth_state = np.asarray(self._gpu_node_depth, dtype=np.float64)
                link_flow_state = np.asarray(self._gpu_link_flow, dtype=np.float64)
                g = float(getattr(self.drainage.cfg, "gravity", _u.gravity()))
                head_deadband = float(getattr(self.drainage.cfg, "head_deadband_m", 1.0e-3))
                dynamic_relax = float(getattr(self.drainage.cfg, "dynamic_flow_relaxation", 1.0))
                nd_out, lf_out, _diag = (
                    native_mod.swe2d_gpu_drainage_step(
                        None,  # cell_wse=None → compute on-device from h+zb
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
                        None,  # cell_depth=None → compute on-device from d_h
                        node_depth_state,
                        link_flow_state,
                        float(dt_s),
                        float(g),
                        int(dsoa.solver_mode),
                        float(head_deadband),
                        float(dynamic_relax),
                    )
                )
                self._gpu_node_depth = np.asarray(nd_out, dtype=np.float64)
                self._gpu_link_flow = np.asarray(lf_out, dtype=np.float64)
                self._sync_gpu_state_back_to_drainage()
            else:
                return False  # no GPU drainage backend available

            # Note: do NOT update component_sums here — the caller skips the
            # callback entirely when we return True, so there's no diagnostics
            # path that needs them from this function.
        else:
            # No drainage — ensure persistent preload for structures-only path.
            self._ensure_persistent_coupling_preloaded(native_mod)

        # ── Structures: run on-device coupling ──────────────────────────
        # The C++ compute_coupling_full_on_device reads WSE from device-resident
        # h+zb (cell_wse_host=None), computes structure flows on-device
        # (host_structure_flows=None), and writes the combined result
        # (structures + drainage inlets) directly to d_external_source_mps.
        self._ensure_persistent_coupling_preloaded(native_mod)
        if not self._persistent_coupling_preloaded:
            return False

        if self.culvert_face_flux_mode == "face_flux":
            was_preloaded = self._culvert_face_flux_preloaded
            self._ensure_culvert_face_flux_preloaded(native_mod)
            if self._culvert_face_flux_preloaded != was_preloaded:
                self._culvert_solver_mode_applied = -1  # not yet applied

        n_structures = int(self._structure_count) if self.structures is not None else 0
        if hasattr(native_mod, "swe2d_gpu_set_coupling_dt"):
            native_mod.swe2d_gpu_set_coupling_dt(float(dt_s))


        # cell_wse_host=None → GPU computes WSE from device h+zb
        # host_structure_flows=None → GPU computes flows on-device
        # Drainage q_cell is folded into d_external_source_mps via d_drainage_q
        native_mod.swe2d_gpu_compute_coupling_full_on_device(
            None,
            n_structures,
            None,
        )

        # (No device sync here — compute_coupling_full_on_device already has a
        # cudaStreamSynchronize on the solver stream which is sufficient to clear
        # any pending stream errors before the next graph capture.  Full
        # cudaDeviceSynchronize is expensive and unnecessary.)

        # Mark coupling as applied so _compute_source_rates_cuda can skip.
        self._coupling_applied_this_timestep = True

        # ── Invalidate the cached CUDA graph because dev->use_culvert_face_flux
        # changed from the pre-coupling state (false) to the post-coupling state
        # (true).  The graph signature includes use_culvert_face_flux, so the
        # solver will detect a cache miss and re-capture on the next step.
        # Without explicit invalidation, the solver tries to replay the OLD graph
        # which has use_culvert_face_flux=false baked in, leading to incorrect
        # kernel arguments for the face-flux path.
        if hasattr(native_mod, "swe2d_gpu_invalidate_graph_cache"):
            native_mod.swe2d_gpu_invalidate_graph_cache()

        # When face-flux mode is active, the culvert face flux is already
        # applied via d_ext_struct_flux_h which the update kernel reads
        # directly inside the source subcycling loop (line 1972 of
        # swe2d_gpu.cu).  Folding into d_external_source_mps would
        # double-count the culvert mass because the update kernel would
        # then receive it from both paths.
        # The fold is only needed in the non-face-flux fallback where
        # ext_struct_flux_h is never populated and the culvert mass must
        # travel through d_external_source_mps like other sources.
        if self.culvert_face_flux_mode != "face_flux":
            if hasattr(native_mod, "swe2d_gpu_fold_culvert_mass_to_source"):
                try:
                    native_mod.swe2d_gpu_fold_culvert_mass_to_source(int(self.n_cells))
                except Exception as exc:
                    self._log(
                        "[COUPLING] Failed to fold culvert mass to source; "
                        "redistribution may be incomplete. Error: " + str(exc)
                    )

        # ── Face-flux influence-width redistribution ─────────────────────
        # When face-flux is active and redistribution geometry exists,
        # spread the culvert mass flux from single donor/receiver cells
        # across the pre-computed corridor cells.  This prevents excessive
        # local drawdown (and spurious velocity spikes) at the culvert
        # inlet/outlet cells.  The redistribution reverses the single-cell
        # injection in d_ext_struct_flux_h and distributes Q across a
        # wider set of cells, then re-uploads to device.
        # ── Face-flux redistribution (GPU-only, no PCIe transfers) ─────
        # The GPU kernel operates directly on d_ext_struct_flux_h with zero
        # host readback.  Static geometry (face SOA + redistribution arrays)
        # is uploaded once via content-hash tracking in the C++ wrapper.
        if (self.culvert_face_flux_mode == "face_flux"
            and self._redist_offsets is not None
            and self._redist_offsets.size > 1
            and self._face_flux_soa is not None
            and self._face_flux_soa.structure_index.size > 0
            and hasattr(native_mod, "swe2d_gpu_redistribute_face_flux")):
            ff = self._face_flux_soa
            native_mod.swe2d_gpu_redistribute_face_flux(
                np.asarray(ff.structure_index, dtype=np.int32),
                np.asarray(ff.donor_cell, dtype=np.int32),
                np.asarray(ff.receiver_cell, dtype=np.int32),
                np.asarray(self._redist_offsets, dtype=np.int32),
                np.asarray(self._redist_cell_idx, dtype=np.int32),
                np.asarray(self._redist_weights, dtype=np.float64),
                int(self.n_cells),
            )

        # ── On-device redistribution ────────────────────────────────────
        # When the model has redistribution geometry and the persistent
        # on-device function is available, apply redistribution directly
        # on dev->d_external_source_mps with no host readback.
        if (self._redist_offsets is not None
            and self._redist_offsets.size > 1
            and hasattr(native_mod, "swe2d_gpu_redistribute_structure_sources_persistent")):
            ssoa = self._structures_soa
            non_bridge_mask = self._structure_non_bridge_mask
            if ssoa is not None and non_bridge_mask is not None and np.any(non_bridge_mask):
                nb_n = int(self._n_non_bridge_structures)
                if nb_n > 0 and hasattr(native_mod, "swe2d_gpu_readback_structure_flows"):
                    nb_flows = np.asarray(
                        native_mod.swe2d_gpu_readback_structure_flows(nb_n),
                        dtype=np.float64,
                    )
                else:
                    nb_flows = None
                if nb_flows is not None and nb_flows.size > 0:
                    self._last_structure_flows = nb_flows.copy()  # ponytail: store for callback
                    nb_up = np.asarray(ssoa.upstream_cell[non_bridge_mask], dtype=np.int32)
                    nb_dn = np.asarray(ssoa.downstream_cell[non_bridge_mask], dtype=np.int32)
                    try:
                        native_mod.swe2d_gpu_redistribute_structure_sources_persistent(
                            np.asarray(self._redist_offsets, dtype=np.int32, order='C'),
                            np.asarray(self._redist_cell_idx, dtype=np.int32, order='C'),
                            np.asarray(self._redist_weights, dtype=np.float64, order='C'),
                            np.asarray(nb_flows, dtype=np.float64, order='C'),
                            nb_up,
                            nb_dn,
                            int(self.n_cells),
                            float(_u.si_m_per_model()),
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "GPU on-device redistribution failed — no fallback. "
                            f"Error: {exc}"
                        )

        self.last_diag = SWE2DCouplingDiagnostics(
            time_s=float(t_s) + float(dt_s),
            dt_s=float(dt_s),
            component_sums={
                "structures_persistent_path": 1.0,
                "native_device_coupling": 1.0,
            },
        )
        return True

    def _ensure_persistent_coupling_preloaded(self, native_mod) -> None:
        """Upload structure and cell-area parameters to GPU for the persistent coupling path."""
        if self._persistent_coupling_preloaded:
            return
        if not hasattr(native_mod, "swe2d_gpu_preload_structure_params"):
            raise RuntimeError(
                "GPU structure preloading function swe2d_gpu_preload_structure_params "
                "is unavailable — required for persistent coupling path."
            )
        ssoa = self._structures_soa
        if ssoa is not None and int(len(ssoa.structure_type)) > 0:
            try:
                native_mod.swe2d_gpu_preload_structure_params(
                    np.asarray(ssoa.structure_type, dtype=np.int32),
                    self._remap_cells_for_gpu(np.asarray(ssoa.upstream_cell, dtype=np.int32)),
                    self._remap_cells_for_gpu(np.asarray(ssoa.downstream_cell, dtype=np.int32)),
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
                    np.asarray(ssoa.culvert_area, dtype=np.float64),
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
                    self._gravity,
                    self._model_to_ft,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to preload structure params on device — no CPU fallback. "
                    f"Error: {exc}"
                )
        if not hasattr(native_mod, "swe2d_gpu_preload_coupling_cell_area"):
            raise RuntimeError(
                "GPU cell area preloading function swe2d_gpu_preload_coupling_cell_area "
                "is unavailable — required for persistent coupling path."
            )
        try:
            cell_area_si = np.asarray(self.cell_area, dtype=np.float64) / _u.si_m2_per_model_area()
            native_mod.swe2d_gpu_preload_coupling_cell_area(cell_area_si)
        except Exception as exc:
            raise RuntimeError(
                "Failed to preload coupling cell area on device — no CPU fallback. "
                f"Error: {exc}"
            )
        self._persistent_coupling_preloaded = True

    def _remap_cells_for_gpu(self, cells: np.ndarray) -> np.ndarray:
        """Remap cell indices from original (pre-RCMK) to solver (RCMK) order.

        The C++ mesh builder applies RCMK renumbering to solver arrays
        (d_h, d_cell_zb, d_cell_area) but structure cell indices come from
        Python (original order).  This function remaps them so the GPU
        coupling kernel reads the correct cells.
        """
        if self._inv_cell_perm is None or self._inv_cell_perm.size == 0:
            return cells
        out = cells.copy()
        valid = (cells >= 0) & (cells < self._inv_cell_perm.size)
        if np.any(valid):
            out[valid] = self._inv_cell_perm[cells[valid]]
        return out

    def _ensure_gpu_drainage_state(self) -> None:
        """Initialise GPU-resident drainage node-depth and link-flow arrays from Python state."""
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
        """Copy GPU-resident drainage state back to the Python DrainageCouplingEngine."""
        if self.drainage is None or self._gpu_node_depth is None or self._gpu_link_flow is None:
            return
        for i, node in enumerate(self.drainage.cfg.nodes):
            self.drainage.state.node_depth[node.node_id] = float(self._gpu_node_depth[i])
        for i, link in enumerate(self.drainage.cfg.links):
            self.drainage.state.link_flow[link.link_id] = float(self._gpu_link_flow[i])

    def _ensure_gpu_drainage_static_args(self) -> Optional[Dict[str, np.ndarray]]:
        """Build and cache the static (geometry) argument dict for GPU drainage calls."""
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
            "inlet_cell": np.ascontiguousarray(self._remap_cells_for_gpu(np.asarray(dsoa.inlet_cell, dtype=np.int32)), dtype=np.int32),
            "inlet_node": np.ascontiguousarray(dsoa.inlet_node, dtype=np.int32),
            "inlet_crest_elev": np.ascontiguousarray(dsoa.inlet_crest_elev, dtype=np.float64),
            "inlet_width": np.ascontiguousarray(dsoa.inlet_width, dtype=np.float64),
            "inlet_coefficient": np.ascontiguousarray(dsoa.inlet_coefficient, dtype=np.float64),
            "inlet_max_capture": np.ascontiguousarray(dsoa.inlet_max_capture, dtype=np.float64),
            "outfall_cell": np.ascontiguousarray(self._remap_cells_for_gpu(np.asarray(dsoa.outfall_cell, dtype=np.int32)), dtype=np.int32),
            "outfall_node": np.ascontiguousarray(dsoa.outfall_node, dtype=np.int32),
            "outfall_invert_elev": np.ascontiguousarray(dsoa.outfall_invert_elev, dtype=np.float64),
            "outfall_diameter": np.ascontiguousarray(dsoa.outfall_diameter, dtype=np.float64),
            "outfall_coefficient": np.ascontiguousarray(dsoa.outfall_coefficient, dtype=np.float64),
            "outfall_max_flow": np.ascontiguousarray(dsoa.outfall_max_flow, dtype=np.float64),
            "outfall_zero_storage": np.ascontiguousarray(dsoa.outfall_zero_storage, dtype=np.int32),
            "pipe_end_cell": np.ascontiguousarray(self._remap_cells_for_gpu(np.asarray(dsoa.pipe_end_cell, dtype=np.int32)), dtype=np.int32),
            "pipe_end_node": np.ascontiguousarray(dsoa.pipe_end_node, dtype=np.int32),
            "pipe_end_invert_elev": np.ascontiguousarray(dsoa.pipe_end_invert_elev, dtype=np.float64),
            "pipe_end_diameter": np.ascontiguousarray(dsoa.pipe_end_diameter, dtype=np.float64),
            "pipe_end_area": np.ascontiguousarray(dsoa.pipe_end_area, dtype=np.float64),
            "pipe_end_inlet_loss_k": np.ascontiguousarray(dsoa.pipe_end_inlet_loss_k, dtype=np.float64),
            "pipe_end_outlet_loss_k": np.ascontiguousarray(dsoa.pipe_end_outlet_loss_k, dtype=np.float64),
        }
        return self._gpu_drainage_static_args

    def _bridge_structure_arrays(self, bridge_flow_values: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Extract enabled bridge structures into a dict of arrays for stacked coupling."""
        if self.structures is None:
            return None
        bridge_indices = self._enabled_bridge_indices
        if bridge_indices.size == 0:
            return None

        sts = self._structures_cfg
        return {
            "indices": np.asarray(bridge_indices, dtype=np.int32),
            "structure_id": np.asarray([str(sts[i].structure_id) for i in bridge_indices], dtype=object),
            "upstream_cell": np.ascontiguousarray([int(sts[i].upstream_cell) for i in bridge_indices], dtype=np.int32),
            "downstream_cell": np.ascontiguousarray([int(sts[i].downstream_cell) for i in bridge_indices], dtype=np.int32),
            "flow": np.ascontiguousarray(bridge_flow_values[bridge_indices], dtype=np.float64),
            "loss_k_upstream": np.ascontiguousarray([
                _meta_float(sts[i].metadata, "inlet_loss_k", _meta_float(sts[i].metadata, "coeff", 0.5))
                for i in bridge_indices
            ], dtype=np.float64),
            "loss_k_downstream": np.ascontiguousarray([
                _meta_float(sts[i].metadata, "outlet_loss_k", _meta_float(sts[i].metadata, "coeff", 0.5))
                for i in bridge_indices
            ], dtype=np.float64),
            "width_m": np.ascontiguousarray([
                _meta_float(sts[i].metadata, "width", 1.0)
                for i in bridge_indices
            ], dtype=np.float64),
        }

    

    @property
    def n_cells(self) -> int:
        """Number of 2D mesh cells in the domain."""
        return int(self.cell_area.size)

    def source_rate_callback(self) -> Callable[[float, float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Return the callable expected by the solver backend for source-rate computation."""
        return self.compute_source_rates

    def compute_source_rates(
        self,
        t_s: float,
        dt_s: float,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
    ) -> np.ndarray:
        """Compute source rates."""
        _ = (hu, hv)
        hh = np.ascontiguousarray(h, dtype=np.float64).ravel()
        if hh.size != self.n_cells:
            raise ValueError("state size does not match coupling cell arrays")
        mod = self._native_cuda_module()
        if mod is None:
            raise RuntimeError("CUDA coupling module unavailable")
        self._ensure_native_culvert_solver_mode(mod)
        self._ensure_persistent_coupling_preloaded(mod)
        return self._compute_source_rates_cuda(mod, t_s, dt_s, hh)

    def _compute_source_rates_cuda(self, native_mod, t_s: float, dt_s: float, hh: np.ndarray) -> np.ndarray:
        """GPU-path source-rate computation: drainage step + structure coupling + redistribution."""
        if self._coupling_applied_this_timestep:
            self._coupling_applied_this_timestep = False
            self.last_diag = SWE2DCouplingDiagnostics(
                time_s=float(t_s) + float(dt_s), dt_s=float(dt_s),
                drainage_max_node_depth=0.0, drainage_max_link_flow=0.0,
                structure_total_flow=0.0, source_sum=0.0,
                source_min=0.0, source_max=0.0, component_sums={},
            )
            return None

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
            if (self._drainage_soa is not None
                and hasattr(native_mod, "swe2d_gpu_drainage_step")):
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
                # hh64 still needed for GPU drainage calls that don't have
                # device-resident WSE yet; can be eliminated in a future pass.
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
                            float(getattr(self.drainage.cfg, "gravity", _u.gravity())),
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
                        nd_out, lf_out, diag = (
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
                                float(getattr(self.drainage.cfg, "gravity", _u.gravity())),
                                int(dsoa.solver_mode),
                                float(getattr(self.drainage.cfg, "head_deadband_m", 1.0e-3)),
                                float(getattr(self.drainage.cfg, "dynamic_flow_relaxation", 1.0)),
                            )
                        )
                        self._gpu_node_depth = np.asarray(nd_out, dtype=np.float64)
                        self._gpu_link_flow = np.asarray(lf_out, dtype=np.float64)
                        self._sync_gpu_state_back_to_drainage()
                        # ponytail: q_cell_out removed from C++; reconstruct for diag
                        node_delta = np.asarray(nd_out, dtype=np.float64) - np.asarray(node_depth_state, dtype=np.float64)
                        q_cell = np.zeros(self.n_cells, dtype=np.float64)
                        if dsoa is not None and dsoa.inlet_cell.size > 0:
                            for ci, ni in zip(dsoa.inlet_cell, dsoa.inlet_node):
                                ci, ni = int(ci), int(ni)
                                if 0 <= ci < self.n_cells and 0 <= ni < len(node_delta):
                                    q_cell[ci] -= node_delta[ni] * float(dsoa.node_surface_area[ni]) / (max(float(self.cell_area[ci]), 1e-12) * max(float(dt_s), 1e-12))
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
                # No GPU drainage backend available — network is empty/disabled,
                # or the native module lacks the CUDA binding.  Produce zero
                # exchange so the solver continues without drainage sources.
                q_cell = np.zeros(self.n_cells, dtype=np.float64)
                drainage_diag = {
                    "max_node_depth": 0.0,
                    "max_link_flow": 0.0,
                    "limiter_events": 0.0,
                    "limiter_volume_m3": 0.0,
                    "substeps_used": 0.0,
                    "inactive_fastpath": 1.0,
                }
                component_sums["drainage_native_iterative"] = 0.0

            component_sums["drainage"] = float(np.sum(q_cell / np.maximum(self.cell_area, 1.0e-12)))
            component_sums["drainage_limiter_events"] = float(drainage_diag.get("limiter_events", 0.0))
            component_sums["drainage_limiter_volume_m3"] = float(drainage_diag.get("limiter_volume_m3", 0.0))
            component_sums["drainage_substeps_used"] = float(drainage_diag.get("substeps_used", 1.0))
            component_sums["drainage_implicit_iters_used"] = float(drainage_diag.get("implicit_iters_used", 0.0))
            component_sums["drainage_inactive_fastpath"] = float(drainage_diag.get("inactive_fastpath", 0.0))

        # ── Face-flux preload (isolation fix: drainage bypasses
        # apply_native_device_sources, so we must preload here) ──────
        if (self.culvert_face_flux_mode == "face_flux" and self._structures_soa is not None
                and not self._culvert_face_flux_preloaded):
            if hasattr(native_mod, "swe2d_gpu_upload_culvert_face_flux_params"):
                self._ensure_culvert_face_flux_preloaded(native_mod)

        bridge_total = None
        bridge_helper_used = False
        if self.structures is not None:
            sts = self._structures_cfg
            ssoa = self._structures_soa
            non_bridge_mask = self._structure_non_bridge_mask
            use_persistent = (
                self._persistent_coupling_preloaded
                and hasattr(native_mod, "swe2d_gpu_compute_coupling_full_on_device")
            )

            if use_persistent:
                # Persistent device path: structure params preloaded on GPU.
                # All work is done on-device — WSE computed from device h+zb,
                # structure flows computed on GPU, drainage inlet sources
                # folded in via inlet_cell/inlet_flow.  Source rates written
                # directly to dev->d_external_source_mps — zero PCIe transfers
                # of the source array.  Handles ALL structure types including
                # bridges; bridge stacked coupling is applied below.
                if ssoa is not None:
                    try:
                        native_mod.swe2d_gpu_compute_coupling_full_on_device(
                            None,  # cell_wse=None → on-device WSE computation
                            int(len(sts)),  # all structures including bridges
                            None,  # host_structure_flows=None → GPU computes
                        )
                        component_sums["structures_persistent_path"] = 1.0
                    except Exception:
                        component_sums["structures_persistent_path"] = 0.0

                # ── Bridge stacked coupling (on-device sources) ─────────
                # The on-device kernel already wrote bridge sources to
                # d_external_source_mps at single-cell (upstream/downstream).
                # When bridge stacked coupling is active, read back the
                # per-bridge flows (tiny: n_bridge doubles), apply the
                # corridor redistribution in Python, and fold the corrected
                # bridge sources back to device.
                if self._has_bridge_structures and len(sts) > 0:
                    bridge_flows = np.asarray(
                        native_mod.swe2d_gpu_readback_structure_flows(len(sts)),
                        dtype=np.float64,
                    )
                    # Store ALL structure flows (including non-bridge/culvert)
                    # so the snapshot callback reads non-zero values.
                    self._last_structure_flows = bridge_flows.copy()
                    bridge_mask = self._structure_bridge_indices
                    if bridge_mask is not None and bridge_mask.size > 0:
                        bridge_q = bridge_flows[bridge_mask.astype(np.int32)]
                        bridge_arrays = self._bridge_structure_arrays(bridge_flows)
                        if bridge_arrays is not None and len(bridge_q) > 0:
                            bridge_helper_used = True
                            bridge_total = np.zeros(self.n_cells, dtype=np.float64)
                            bridge_plan_map = {
                                str(plan.structure_id): plan
                                for plan in getattr(self, "bridge_stacked_plans", []) or []
                            }
                            for i in range(int(bridge_arrays["indices"].size)):
                                src = np.asarray(
                                    native_mod.swe2d_gpu_compute_bridge_coupling_sources(
                                        np.asarray(self.cell_area, dtype=np.float64),
                                        np.asarray([int(bridge_arrays["upstream_cell"][i])], dtype=np.int32),
                                        np.asarray([int(bridge_arrays["downstream_cell"][i])], dtype=np.int32),
                                        np.asarray([float(bridge_q[i])], dtype=np.float64),
                                        np.asarray([float(bridge_arrays["loss_k_upstream"][i])], dtype=np.float64),
                                        np.asarray([float(bridge_arrays["loss_k_downstream"][i])], dtype=np.float64),
                                        float(bridge_arrays["width_m"][i]), float(dt_s),
                                    ), dtype=np.float64)
                                bridge_id = str(bridge_arrays["structure_id"][i])
                                bridge_plan = bridge_plan_map.get(bridge_id)
                                if bridge_plan is not None:
                                    if self.bridge_stacked_coupling_mode == "legacy_scalar":
                                        src = apply_bridge_stacked_source_weight(src, bridge_plan)
                                    else:
                                        src = apply_bridge_stacked_phase3_source_weight(
                                            src, bridge_plan, self.cell_area,
                                        )
                                bridge_total += src
                # ── Non-bridge (culvert/weir/orifice) flow readback ──────
                # The on-device kernel computed ALL structure flows.  When
                # bridges exist they were read back above.  For culvert-only
                # configurations (no bridges) read back here so the snapshot
                # callback sees non-zero flow values.
                if (not self._has_bridge_structures
                        and self._n_non_bridge_structures > 0
                        and hasattr(native_mod, "swe2d_gpu_readback_structure_flows")):
                    try:
                        _nb_all = np.asarray(
                            native_mod.swe2d_gpu_readback_structure_flows(int(len(sts))),
                            dtype=np.float64,
                        )
                        self._last_structure_flows = _nb_all.copy()
                    except Exception:
                        pass

                # On-device sources are already in d_external_source_mps.
                # Skip redistribution: compute_coupling_full_on_device already
                # wrote combined (structures + drainage + face-flux) sources
                # directly to device memory.  No host readback or redistribution
                # is needed — the solver step consumes from the same buffer.
                # Return None so the backend knows the GPU buffer is current.
                total = None
                # Signal that persistent path handled structures.
                flows = None
                flows = None

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
            total *= _u.si_m_per_model()  # m/s → model-length/s
            # Apply influence-width redistribution via GPU kernel.
            if struct_q.size > 0 and struct_up.size == struct_q.size:
                total = self._apply_redistribution(total, struct_q, struct_up, struct_dn, native_mod=native_mod)
        if bridge_helper_used:
            if total is None:
                # On-device sources are already in d_external_source_mps.
                # We need to add bridge sources (host-resident) to the
                # device array.  The simplest safe approach: read back,
                # add bridges, return host array for re-upload.
                total = np.asarray(
                    native_mod.swe2d_gpu_readback_coupling_sources(self.n_cells),
                    dtype=np.float64,
                )
                total *= _u.si_m_per_model()
            total += bridge_total
            component_sums["bridges"] = float(np.sum(bridge_total))

        # When the persistent CUDA path was used and there's no redistribution
        # (and no bridges), total is None (sources are already on GPU).  Build
        # a zero diagnostics array so diagnostic stats are well-defined, then
        # return None to signal to the caller that GPU sources are current.
        if total is None:
            diag_total = np.zeros(max(1, self.n_cells), dtype=np.float64)
        else:
            diag_total = total
        self.last_diag = SWE2DCouplingDiagnostics(
            time_s=float(t_s) + float(dt_s),
            dt_s=float(dt_s),
            drainage_max_node_depth=float(drainage_diag.get("max_node_depth", drainage_diag.get("max_node_depth_m", 0.0))) * _u.model_per_si_m(),
            drainage_max_link_flow=float(drainage_diag.get("max_link_flow", drainage_diag.get("max_link_flow_cms", 0.0))) / _u.si_m3_per_model_volume(),
            structure_total_flow=float(structure_diag.get("total_structure_flow", 0.0)) / _u.si_m3_per_model_volume(),
            source_sum=float(np.sum(diag_total)),
            source_min=float(np.min(diag_total)) if diag_total.size else 0.0,
            source_max=float(np.max(diag_total)) if diag_total.size else 0.0,
            component_sums=component_sums,
        )
        return total  # None signals GPU-resident sources to caller


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
