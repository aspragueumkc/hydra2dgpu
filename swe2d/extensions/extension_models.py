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
    FV_WENO5          = 6   # WENO5 + LSQ 2-ring gradient (~3rd-order, GPU-first)
    # Backward-compatibility aliases
    FV_MUSCL = FV_MUSCL_FAST
    FV_WENO  = FV_MUSCL_MINMOD


class TemporalScheme(IntEnum):
    EULER_1ST = 1
    SSP_RK2 = 2
    SSP_RK3 = 3
    GRAPH_SAFE_RK4 = 5      # True RK4 stage path with graph-safe staged forcing
    GRAPH_SAFE_RK5 = 6      # Cash-Karp RK5 stage path with graph-safe staged forcing


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


class GodunovSolverMode(IntEnum):
    CURRENT_GPU_STEP = 0


class SWE2DEquationSet(IntEnum):
    HYDROSTATIC_2D = 0


class DrainageSolverMode(IntEnum):
    """Equation set for the 1D drainage-network solver.

    EGL       – Energy-grade-line (Bernoulli + Manning friction + minor losses).
                Models pressure-pipe flow; analogous to FHWA HEC-22 outlet-control
                equations. Good default for storm-drain systems.
    DIFFUSION – Diffusion-wave: slope-driven Manning flow using partial-flow
                circular-section hydraulic geometry. Better for partially-full
                gravity sewers and open-channel reaches.
    DYNAMIC   – Full 1D Saint-Venant with semi-implicit per-link momentum update.
                Captures surge, bore propagation, and backwater transients.
    """
    EGL       = 0
    DIFFUSION = 1
    DYNAMIC   = 2


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
    crest_elev: Optional[float] = None
    rim_elev: Optional[float] = None
    node_type: str = "junction"  # junction, outfall, storage, inlet, pipe_end
    # --- Outfall boundary-condition fields (only used when node_type == "outfall") ---
    # outfall_mode choices: "free" | "fixed_wse" | "stage_discharge"
    #   free             – node drains freely; depth reset to 0 each step unless
    #                      backwatered by a coupled 2D cell.
    #   fixed_wse        – tailwater fixed at outfall_fixed_wse [m]; node head is
    #                      clamped to that elevation.
    #   stage_discharge  – outflow rate from rating table [(wse_m, Q_m3s), ...]
    outfall_mode: str = "free"
    outfall_fixed_wse: float = 0.0
    outfall_rating_table: list = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class DrainageLink:
    link_id: str
    from_node_id: str
    to_node_id: str
    link_type: str = "conduit"  # conduit, pump, weir, orifice, culvert
    length: float = 0.0
    roughness_n: float = 0.013
    diameter: Optional[float] = None
    max_flow: Optional[float] = None
    # Culvert-specific fields (used when link_type == "culvert")
    culvert_shape: Optional[str] = None       # circular, box, rectangular, pipe_arch
    culvert_code: int = 1                      # FHWA culvert code
    culvert_rise: Optional[float] = None       # vertical dimension (model units)
    culvert_span: Optional[float] = None       # horizontal dimension (model units)
    inlet_invert_elev: Optional[float] = None  # upstream invert (model units)
    outlet_invert_elev: Optional[float] = None # downstream invert (model units)
    entrance_loss_k: float = 0.5               # Ke (FHWA entrance loss)
    exit_loss_k: float = 1.0                   # Kx (FHWA exit loss)
    barrel_count: int = 1                      # number of barrels
    cd: float = 0.75                           # orifice discharge coefficient
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class InletExchange:
    inlet_id: str
    cell_id: int
    node_id: str
    crest_elev: float
    length: float = 1.0
    area: float = 0.0
    coeff_weir: float = 1.70
    coeff_orifice: float = 0.62
    max_capture: Optional[float] = None
    width: Optional[float] = None
    coefficient: Optional[float] = None

    def __post_init__(self):
        if self.width is not None:
            self.length = float(self.width)
        if self.coefficient is not None:
            self.coeff_orifice = float(self.coefficient)


@dataclass
class InletType:
    inlet_type_id: str
    name: str = ""
    length: float = 1.0
    area: float = 0.0
    coeff_weir: float = 1.70
    coeff_orifice: float = 0.62
    max_capture: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class NodeInletAssignment:
    node_id: str
    inlet_type_id: str
    multiplier: float = 1.0
    crest_offset: float = 0.0


@dataclass
class OutfallExchange:
    """Coupling object for an outfall node located within the 2D mesh.

    Outfall nodes discharge to (or receive from) the co-located 2D surface
    cell using an orifice equation based on the pipe cross-section area.
    Exchange is two-way:
      - Surcharge: network head > surface WSE  -> source injected into 2D cell.
      - Backwater: surface WSE > network head  -> sink removed from 2D cell.
    """
    outfall_id: str
    cell_id: int
    node_id: str
    invert_elev: float
    area_m2: float = 0.0
    diameter: float = 0.0
    coefficient: float = 0.82  # pipe outlet loss coefficient
    max_flow: Optional[float] = None
    # When true, treat the outfall as a daylighted pipe end with no local 1D
    # storage at the node during 2D exchange.
    zero_storage: bool = False


@dataclass
class PipeEndExchange:
    """Coupling object for a daylighted pipe end located within the 2D mesh.

    A pipe end represents an open pipe terminus that exchanges flow directly
    with the co-located 2D surface cell — no inlet grate, no invert
    depression.  Exchange is two-way:

      - Outflow: pipe head > surface WSE  -> flow discharges into 2D cell.
      - Inflow:  surface WSE > pipe invert -> flow enters pipe from surface.

    Unlike OutfallExchange there is no local manhole storage bucket; the pipe
    cross-section IS the exchange area (``zero_storage`` is always True).
    Typical uses: culvert face, daylighted underdrain, storm-sewer stub-out.
    """
    pipe_end_id: str
    cell_id: int
    node_id: str
    invert_elev: float
    diameter: float = 0.0
    area_m2: float = 0.0
    coefficient: float = 0.82  # pipe outlet loss coefficient (FHWA Ke)
    max_flow: Optional[float] = None
    # Optional empirical minor-loss coefficients applied at daylighted ends
    # when converting surface head to effective boundary head for routed-link
    # coupling. Defaults follow common design heuristics.
    inlet_loss_k: Optional[float] = 0.5
    outlet_loss_k: Optional[float] = 1.0


@dataclass
class PipeNetworkConfig:
    enabled: bool = False
    nodes: List[DrainageNode] = field(default_factory=list)
    links: List[DrainageLink] = field(default_factory=list)
    inlet_types: List[InletType] = field(default_factory=list)
    node_inlets: List[NodeInletAssignment] = field(default_factory=list)
    inlets: List[InletExchange] = field(default_factory=list)
    outfalls: List[OutfallExchange] = field(default_factory=list)
    pipe_ends: List[PipeEndExchange] = field(default_factory=list)
    use_swmm_reference_mode: bool = False
    target_cuda_port: bool = False
    gravity: float = 9.81
    swmm_input_path: Optional[str] = None
    # 1D solver equation set (see DrainageSolverMode)
    solver_mode: DrainageSolverMode = DrainageSolverMode.EGL
    # Number of 1D network sub-steps taken per 2D coupling call.  Values > 1
    # allow the 1D solver to run at a finer dt than the 2D timestep, improving
    # stability for stiff networks without requiring GPU sub-stepping.
    coupling_substeps: int = 1
    # Maximum substeps allowed when the adaptive drainage timestep controller
    # tightens the 1D solve for stiff dynamic states.
    max_coupling_substeps: int = 64
    # Small head-difference deadband applied before link/inlet exchange updates
    # to suppress chatter around near-balanced states.
    head_deadband_m: float = 1.0e-3
    # Relaxation factor applied to dynamic-wave link flow updates.
    # 1.0 keeps the full update, lower values damp oscillatory responses.
    dynamic_flow_relaxation: float = 1.0
    # Adaptive substepping limit: allowable fractional node-depth change per
    # drainage substep for the 1D network solve.
    adaptive_depth_fraction: float = 0.2
    # Adaptive substepping limit: wave Courant target used for dynamic links.
    adaptive_wave_courant: float = 0.5
    # Optional extra inner iterations for a predictor/corrector style coupling
    # solve between 2D surface exchange and the 1D drainage network.
    implicit_coupling_iterations: int = 2
    # Relaxation factor used when blending coupling iterates. 0.5 is a safe
    # default for stiff rain/drainage coupling; 1.0 disables relaxation.
    implicit_coupling_relaxation: float = 0.5


@dataclass
class PipeNetworkState:
    """Transient state for the 1D drainage network reference implementation."""

    node_depth: Dict[str, float] = field(default_factory=dict)
    link_flow: Dict[str, float] = field(default_factory=dict)


@dataclass
class CouplingDiagnostics:
    """Lightweight diagnostics for one network/exchange step."""

    dt_s: float
    net_node_inflow: float = 0.0
    total_capture: float = 0.0
    total_surcharge: float = 0.0
    max_node_depth: float = 0.0
    max_link_flow: float = 0.0


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
    gravity: float = 9.81


def circular_area_from_diameter(diameter_m: float) -> float:
    d = max(0.0, float(diameter_m))
    return 0.25 * math.pi * d * d


def equivalent_circular_diameter_from_area(area_m2: float) -> float:
    a = max(0.0, float(area_m2))
    if a <= 0.0:
        return 0.0
    return math.sqrt(4.0 * a / math.pi)


def circular_wet_perimeter_full(diameter_m: float) -> float:
    d = max(0.0, float(diameter_m))
    return math.pi * d


def circular_section_from_depth(depth_m: float, diameter_m: float) -> Tuple[float, float]:
    """Return (flow_area_m2, wetted_perimeter_m) for a partially-filled circular pipe.

    Uses the standard central-angle formula::

        theta = 2 * arccos(1 - 2*y/D)
        A     = (D^2 / 8) * (theta - sin(theta))
        P     = (D / 2) * theta

    Returns full-pipe values when depth >= diameter; zeros when depth <= 0.
    """
    D = max(1.0e-9, float(diameter_m))
    y = max(0.0, min(float(depth_m), D))
    if y <= 0.0:
        return 0.0, 0.0
    if y >= D:
        return circular_area_from_diameter(D), circular_wet_perimeter_full(D)
    arg = max(-1.0, min(1.0, 1.0 - 2.0 * y / D))
    theta = 2.0 * math.acos(arg)
    area = (D * D / 8.0) * (theta - math.sin(theta))
    perimeter = 0.5 * D * theta
    return max(0.0, area), max(0.0, perimeter)


def compute_orifice_flow(
    head_up: float,
    head_down: float,
    area: float,
    discharge_coeff: float = 0.62,
    g: float = 9.81,
    max_flow: Optional[float] = None,
) -> float:
    """Return signed orifice flow from up to down based on head difference."""
    a = max(0.0, float(area))
    if a <= 0.0:
        return 0.0
    dh = float(head_up) - float(head_down)
    if abs(dh) <= 1.0e-12:
        return 0.0
    q = float(discharge_coeff) * a * math.sqrt(max(0.0, 2.0 * float(g) * abs(dh)))
    if max_flow is not None:
        q = min(q, max(0.0, float(max_flow)))
    return q if dh >= 0.0 else -q


def compute_weir_flow(
    upstream_wse: float,
    downstream_wse: float,
    crest_elev: float,
    width: float,
    coeff: float = 1.7,
    max_flow: Optional[float] = None,
) -> float:
    """Broad-crested style weir discharge using upstream head over crest."""
    b = max(0.0, float(width))
    if b <= 0.0:
        return 0.0
    hup = max(0.0, float(upstream_wse) - float(crest_elev))
    hdn = max(0.0, float(downstream_wse) - float(crest_elev))
    if hup <= 0.0 and hdn <= 0.0:
        return 0.0
    if float(upstream_wse) >= float(downstream_wse):
        h = hup
        sign = 1.0
    else:
        h = hdn
        sign = -1.0
    q = float(coeff) * b * (h ** 1.5)
    if max_flow is not None:
        q = min(q, max(0.0, float(max_flow)))
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
    cell_flow: Sequence[float],
    cell_area: Sequence[float],
) -> List[float]:
    """Convert per-cell volumetric source terms to depth rates."""
    n = min(len(cell_flow), len(cell_area))
    out = [0.0] * n
    for i in range(n):
        area = max(1.0e-12, float(cell_area[i]))
        out[i] = float(cell_flow[i]) / area
    return out


@dataclass
class SolverModelOptions:
    temporal_scheme: TemporalScheme = TemporalScheme.SSP_RK2
    spatial_discretization: SpatialDiscretization = SpatialDiscretization.FV_FIRST_ORDER
    godunov_mode: GodunovSolverMode = GodunovSolverMode.CURRENT_GPU_STEP
    turbulence_model: TurbulenceModel = TurbulenceModel.NONE
    bed_friction_model: BedFrictionModel = BedFrictionModel.MANNING
    equation_set: SWE2DEquationSet = SWE2DEquationSet.HYDROSTATIC_2D
    rain: RainFieldConfig = field(default_factory=RainFieldConfig)
    pipe_network: PipeNetworkConfig = field(default_factory=PipeNetworkConfig)
    hydraulic_structures: HydraulicStructureConfig = field(default_factory=HydraulicStructureConfig)

    def to_native_dict(self) -> Dict[str, int]:
        """Pack core model-selection flags for native solver creation."""
        return {
            "temporal_order": int(self.temporal_scheme),
            "spatial_scheme": int(self.spatial_discretization),
            "godunov_mode": int(self.godunov_mode),
            "turbulence_model": int(self.turbulence_model),
            "bed_friction_model": int(self.bed_friction_model),
            "equation_set": int(self.equation_set),
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
        self._node_area: Dict[str, float] = {}
        self._links_from: Dict[str, List[DrainageLink]] = {}
        self._links_to: Dict[str, List[DrainageLink]] = {}
        # Set of node_ids that have a corresponding OutfallExchange (2D-coupled).
        # Populated in initialize(); used to skip pure-1D outfall BC on coupled nodes.
        self._outfall_exchange_nodes: set = set()

    def initialize(self) -> None:
        self._node_index = {n.node_id: i for i, n in enumerate(self.cfg.nodes)}
        self._node_area = {
            n.node_id: max(
                1.0,
                float(n.metadata.get("surface_area", n.metadata.get("surface_area_m2", 50.0))),
            )
            for n in self.cfg.nodes
        }
        self._links_from = {n.node_id: [] for n in self.cfg.nodes}
        self._links_to = {n.node_id: [] for n in self.cfg.nodes}
        for lnk in self.cfg.links:
            if lnk.from_node_id in self._links_from:
                self._links_from[lnk.from_node_id].append(lnk)
            if lnk.to_node_id in self._links_to:
                self._links_to[lnk.to_node_id].append(lnk)
            self.state.link_flow.setdefault(lnk.link_id, 0.0)
        for n in self.cfg.nodes:
            self.state.node_depth.setdefault(n.node_id, 0.0)
        self._outfall_exchange_nodes = (
            {o.node_id for o in self.cfg.outfalls}
            | {p.node_id for p in getattr(self.cfg, "pipe_ends", [])}
        )
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
