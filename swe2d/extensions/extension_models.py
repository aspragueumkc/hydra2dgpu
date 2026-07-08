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
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SpatialDiscretization(IntEnum):
    """Spatial reconstruction scheme for the finite-volume Godunov solver."""
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
    """Time-integration method for the solver."""
    EULER_1ST = 1
    SSP_RK2 = 2
    SSP_RK3 = 3
    CLASSIC_RK4 = 4
    GRAPH_SAFE_RK4 = 5      # True RK4 stage path with graph-safe staged forcing
    GRAPH_SAFE_RK5 = 6      # Cash-Karp RK5 stage path with graph-safe staged forcing


class TurbulenceModel(IntEnum):
    """Sub-grid turbulence closure model."""
    NONE = 0
    SMAGORINSKY = 1
    K_EPSILON = 2
    K_OMEGA_SST = 3


class BedFrictionModel(IntEnum):
    """Bed roughness friction formulation."""
    MANNING = 0
    CHEZY = 1
    DARCY_WEISBACH = 2
    NIKURADSE = 3


class GodunovSolverMode(IntEnum):
    """Godunov solver operating mode (GPU-only)."""
    CURRENT_GPU_STEP = 0


class SWE2DEquationSet(IntEnum):
    """Shallow-water equation set variant."""
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
    """Configuration for the rainfall source module."""
    enabled: bool = False
    default_mm_per_hr: float = 0.0
    raster_path: Optional[str] = None
    timeseries_id: Optional[str] = None
    infiltration_enabled: bool = False
    infiltration_model: str = "green_ampt"
    infiltration_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class RainSourceTermState:
    """Transient rainfall source-term state (cell-wise rates)."""
    timestep_s: float = 0.0
    cell_rain_rate_m_per_s: Optional[Sequence[float]] = None
    cell_excess_rain_m_per_s: Optional[Sequence[float]] = None


@dataclass
class DrainageNode:
    """A node in the 1D drainage pipe network."""
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
    """A link (conduit/pipe/pump/weir/orifice/culvert) connecting two drainage nodes."""
    link_id: str
    from_node_id: str
    to_node_id: str
    link_type: str = "conduit"  # conduit, pump, weir, orifice, culvert
    length: float = 0.0
    roughness_n: float = 0.013
    diameter: Optional[float] = None
    link_shape: str = "circular"            # "circular" | "rectangular" | "elliptical"
    width: Optional[float] = None            # width(rect) / span(ellipse)
    height: Optional[float] = None           # height(rect) / rise(ellipse)
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
    max_cell_length: float = 0.0              # max cell length for 1D mesh refinement
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class InletExchange:
    """2D-surface-to-1D-network inlet exchange coupling object."""
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
        """post init"""
        if self.width is not None:
            self.length = float(self.width)
        if self.coefficient is not None:
            self.coeff_orifice = float(self.coefficient)


@dataclass
class InletType:
    """Reusable inlet geometry template (grate, curb-opening, etc.)."""
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
    """Assigns an InletType template to a specific drainage node."""
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
    """Top-level configuration for the 1D drainage pipe network."""
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
    pipe_solver_mode: str = "diffusion_wave"  # "diffusion_wave" | "fully_dynamic"
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
    """Hydraulic structure type identifier."""
    WEIR = 1
    CULVERT = 2
    GATE = 3
    BRIDGE = 4
    PUMP = 5


@dataclass
class HydraulicStructure:
    """A hydraulic structure (weir/culvert/gate/bridge/pump) connecting two 2D cells."""
    structure_id: str
    structure_type: StructureType
    upstream_cell: int
    downstream_cell: int
    crest_elev: float
    enabled: bool = True
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for CLI JSON round-trip (structure_type as lowercase string)."""
        return {
            "id": self.structure_id,
            "type": self.structure_type.name.lower(),
            "upstream_cell": self.upstream_cell,
            "downstream_cell": self.downstream_cell,
            "crest_elev": self.crest_elev,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass
class HydraulicStructureConfig:
    """Top-level configuration for hydraulic structures coupling."""
    enabled: bool = False
    structures: List[HydraulicStructure] = field(default_factory=list)
    control_interval_s: float = 1.0
    controller_name: str = "none"
    gravity: float = 9.81

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict matching build_structures_config_from_json input format.

        Note: ``gravity`` is omitted because the coupling controller derives it
        from the mesh CRS via ``_u.gravity()``. Serializing it would override
        the CRS-derived value with whatever was in the config.
        """
        return {
            "enabled": self.enabled,
            "control_interval_s": self.control_interval_s,
            "controller_name": self.controller_name,
            "structures": [s.to_dict() for s in self.structures],
        }


def circular_area_from_diameter(diameter_m: float) -> float:
    """Return cross-sectional area of a circle from its diameter."""
    d = max(0.0, float(diameter_m))
    return 0.25 * math.pi * d * d


def equivalent_circular_diameter_from_area(area_m2: float) -> float:
    """Return the diameter of a circle with the given cross-sectional area."""
    a = max(0.0, float(area_m2))
    if a <= 0.0:
        return 0.0
    return math.sqrt(4.0 * a / math.pi)


def circular_wet_perimeter_full(diameter_m: float) -> float:
    """Return the wetted perimeter of a full-flowing circular pipe."""
    d = max(0.0, float(diameter_m))
    return math.pi * d


@dataclass
class SolverModelOptions:
    """Aggregate solver model-selection options passed to the native backend."""
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


from swe2d.extensions.drainage_network import DrainageCouplingEngine  # noqa: F401

from swe2d.extensions.structures import HydraulicStructureEngine  # noqa: F401
