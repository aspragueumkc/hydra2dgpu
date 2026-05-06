"""
SWE2D extension scaffolding for upcoming multi-physics and urban drainage work.

This module intentionally provides data-model skeletons only. Runtime coupling
logic should be implemented in dedicated kernels/solvers and orchestrated by
the workbench/backend once physics is validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Dict, List, Optional, Sequence, Tuple


class SpatialDiscretization(IntEnum):
    FV_FIRST_ORDER    = 0
    FV_MUSCL_FAST     = 1
    FV_MUSCL_MINMOD   = 2
    FV_MUSCL_MC       = 3   # Monotonized-Central limiter (gradient-based TVD)
    FV_MUSCL_VAN_LEER = 4   # Van Leer smooth limiter (gradient-based TVD)
    # Backward-compatibility aliases
    FV_MUSCL = FV_MUSCL_FAST
    FV_WENO  = FV_MUSCL_MINMOD


class TemporalScheme(IntEnum):
    EULER_1ST = 1
    SSP_RK2 = 2
    SSP_RK3 = 3


class TurbulenceModel(IntEnum):
    NONE = 0
    SMAGORINSKY = 1
    K_EPSILON = 2
    K_OMEGA_SST = 3


class BedFrictionModel(IntEnum):
    MANNING = 0
    CHEZY = 1
    DARCY_WEISBACH = 2
    NIKURADSE = 3


@dataclass
class RainFieldConfig:
    enabled: bool = False
    default_mm_per_hr: float = 0.0
    raster_path: Optional[str] = None
    timeseries_id: Optional[str] = None
    infiltration_enabled: bool = False
    infiltration_model: str = "green_ampt"
    infiltration_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class RainSourceTermState:
    timestep_s: float = 0.0
    cell_rain_rate_m_per_s: Optional[Sequence[float]] = None
    cell_excess_rain_m_per_s: Optional[Sequence[float]] = None


@dataclass
class DrainageNode:
    node_id: str
    x: float
    y: float
    invert_elev: float
    max_depth: float
    node_type: str = "junction"  # junction, outfall, storage, inlet
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class DrainageLink:
    link_id: str
    from_node_id: str
    to_node_id: str
    link_type: str = "conduit"  # conduit, pump, weir, orifice
    length_m: float = 0.0
    roughness_n: float = 0.013
    diameter_m: Optional[float] = None
    max_flow_cms: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class InletExchange:
    inlet_id: str
    cell_id: int
    node_id: str
    crest_elev: float
    width_m: float
    coefficient: float = 0.62
    max_capture_cms: Optional[float] = None


@dataclass
class PipeNetworkConfig:
    enabled: bool = False
    nodes: List[DrainageNode] = field(default_factory=list)
    links: List[DrainageLink] = field(default_factory=list)
    inlets: List[InletExchange] = field(default_factory=list)
    use_swmm_reference_mode: bool = False
    target_cuda_port: bool = False
    swmm_input_path: Optional[str] = None


@dataclass
class PipeNetworkState:
    """Transient state for the 1D drainage network reference implementation."""

    node_depth_m: Dict[str, float] = field(default_factory=dict)
    link_flow_cms: Dict[str, float] = field(default_factory=dict)


@dataclass
class CouplingDiagnostics:
    """Lightweight diagnostics for one network/exchange step."""

    dt_s: float
    net_node_inflow_cms: float = 0.0
    total_capture_cms: float = 0.0
    total_surcharge_cms: float = 0.0
    max_node_depth_m: float = 0.0
    max_link_flow_cms: float = 0.0


class StructureType(IntEnum):
    WEIR = 1
    CULVERT = 2
    GATE = 3
    BRIDGE = 4
    PUMP = 5


@dataclass
class HydraulicStructure:
    structure_id: str
    structure_type: StructureType
    upstream_cell: int
    downstream_cell: int
    crest_elev: float
    enabled: bool = True
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class HydraulicStructureConfig:
    enabled: bool = False
    structures: List[HydraulicStructure] = field(default_factory=list)
    control_interval_s: float = 1.0
    controller_name: str = "none"


def circular_area_from_diameter(diameter_m: float) -> float:
    d = max(0.0, float(diameter_m))
    return 0.25 * math.pi * d * d


def circular_wet_perimeter_full(diameter_m: float) -> float:
    d = max(0.0, float(diameter_m))
    return math.pi * d


def compute_orifice_flow(
    head_up_m: float,
    head_down_m: float,
    area_m2: float,
    discharge_coeff: float = 0.62,
    g: float = 9.81,
    max_flow_cms: Optional[float] = None,
) -> float:
    """Return signed orifice flow from up to down based on head difference."""
    a = max(0.0, float(area_m2))
    if a <= 0.0:
        return 0.0
    dh = float(head_up_m) - float(head_down_m)
    if abs(dh) <= 1.0e-12:
        return 0.0
    q = float(discharge_coeff) * a * math.sqrt(max(0.0, 2.0 * float(g) * abs(dh)))
    if max_flow_cms is not None:
        q = min(q, max(0.0, float(max_flow_cms)))
    return q if dh >= 0.0 else -q


def compute_weir_flow(
    upstream_wse_m: float,
    downstream_wse_m: float,
    crest_elev_m: float,
    width_m: float,
    coeff: float = 1.7,
    max_flow_cms: Optional[float] = None,
) -> float:
    """Broad-crested style weir discharge using upstream head over crest."""
    b = max(0.0, float(width_m))
    if b <= 0.0:
        return 0.0
    hup = max(0.0, float(upstream_wse_m) - float(crest_elev_m))
    hdn = max(0.0, float(downstream_wse_m) - float(crest_elev_m))
    if hup <= 0.0 and hdn <= 0.0:
        return 0.0
    if float(upstream_wse_m) >= float(downstream_wse_m):
        h = hup
        sign = 1.0
    else:
        h = hdn
        sign = -1.0
    q = float(coeff) * b * (h ** 1.5)
    if max_flow_cms is not None:
        q = min(q, max(0.0, float(max_flow_cms)))
    return sign * q


def compute_pipe_manning_capacity_full(
    diameter_m: float,
    slope_m_per_m: float,
    roughness_n: float,
) -> float:
    """Full-flow Manning capacity for a circular conduit."""
    d = max(0.0, float(diameter_m))
    if d <= 0.0:
        return 0.0
    n = max(1.0e-6, float(roughness_n))
    s = max(0.0, float(slope_m_per_m))
    if s <= 0.0:
        return 0.0
    area = circular_area_from_diameter(d)
    wetted_perimeter = circular_wet_perimeter_full(d)
    if wetted_perimeter <= 0.0:
        return 0.0
    r_h = area / wetted_perimeter
    return (1.0 / n) * area * (r_h ** (2.0 / 3.0)) * math.sqrt(s)


def convert_cell_flows_to_depth_rates(
    cell_flow_cms: Sequence[float],
    cell_area_m2: Sequence[float],
) -> List[float]:
    """Convert per-cell volumetric source terms [m^3/s] to depth rates [m/s]."""
    n = min(len(cell_flow_cms), len(cell_area_m2))
    out = [0.0] * n
    for i in range(n):
        area = max(1.0e-12, float(cell_area_m2[i]))
        out[i] = float(cell_flow_cms[i]) / area
    return out


@dataclass
class SolverModelOptions:
    temporal_scheme: TemporalScheme = TemporalScheme.SSP_RK2
    spatial_discretization: SpatialDiscretization = SpatialDiscretization.FV_FIRST_ORDER
    turbulence_model: TurbulenceModel = TurbulenceModel.NONE
    bed_friction_model: BedFrictionModel = BedFrictionModel.MANNING
    rain: RainFieldConfig = field(default_factory=RainFieldConfig)
    pipe_network: PipeNetworkConfig = field(default_factory=PipeNetworkConfig)
    hydraulic_structures: HydraulicStructureConfig = field(default_factory=HydraulicStructureConfig)

    def to_native_dict(self) -> Dict[str, int]:
        """Pack core model-selection flags for native solver creation."""
        return {
            "temporal_order": int(self.temporal_scheme),
            "spatial_scheme": int(self.spatial_discretization),
            "turbulence_model": int(self.turbulence_model),
            "bed_friction_model": int(self.bed_friction_model),
            "enable_rain_module": bool(self.rain.enabled),
            "enable_pipe_network_module": bool(self.pipe_network.enabled),
            "enable_hydraulic_structures": bool(self.hydraulic_structures.enabled),
        }


class DrainageCouplingEngine:
    """Skeleton orchestrator for 2D surface <-> 1D pipe-network exchange."""

    def __init__(self, cfg: PipeNetworkConfig):
        self.cfg = cfg
        self.state = PipeNetworkState()
        self._node_index: Dict[str, int] = {}
        self._node_area_m2: Dict[str, float] = {}
        self._links_from: Dict[str, List[DrainageLink]] = {}
        self._links_to: Dict[str, List[DrainageLink]] = {}

    def initialize(self) -> None:
        self._node_index = {n.node_id: i for i, n in enumerate(self.cfg.nodes)}
        self._node_area_m2 = {
            n.node_id: max(1.0, float(n.metadata.get("surface_area_m2", 50.0)))
            for n in self.cfg.nodes
        }
        self._links_from = {n.node_id: [] for n in self.cfg.nodes}
        self._links_to = {n.node_id: [] for n in self.cfg.nodes}
        for lnk in self.cfg.links:
            if lnk.from_node_id in self._links_from:
                self._links_from[lnk.from_node_id].append(lnk)
            if lnk.to_node_id in self._links_to:
                self._links_to[lnk.to_node_id].append(lnk)
            self.state.link_flow_cms.setdefault(lnk.link_id, 0.0)
        for n in self.cfg.nodes:
            self.state.node_depth_m.setdefault(n.node_id, 0.0)
        return None

    def exchange_step(self, dt: float, cell_wse: Sequence[float]) -> Tuple[List[float], List[float]]:
        # TODO: return (surface_sink_cms_per_cell, surcharge_source_cms_per_cell).
        _ = (dt, cell_wse)
        return [], []


class RainfallSourceEngine:
    """Skeleton rainfall source module for cell-wise rain and infiltration."""

    def __init__(self, cfg: RainFieldConfig):
        self.cfg = cfg

    def sample_cell_rain(self, t_seconds: float, n_cells: int) -> List[float]:
        # TODO: support gauge interpolation, raster time slices, and IDF events.
        _ = t_seconds
        if n_cells <= 0:
            return []
        return [self.cfg.default_mm_per_hr / 1000.0 / 3600.0] * n_cells


class HydraulicStructureEngine:
    """Skeleton hydraulic-structure dispatcher for weirs/culverts/gates/pumps."""

    def __init__(self, cfg: HydraulicStructureConfig):
        self.cfg = cfg

    def compute_structure_fluxes(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
        # TODO: route to per-structure equations and control logic.
        _ = (dt, cell_wse)
        return {}
