"""
SWE2D extension scaffolding for upcoming multi-physics and urban drainage work.

This module intentionally provides data-model skeletons only. Runtime coupling
logic should be implemented in dedicated kernels/solvers and orchestrated by
the workbench/backend once physics is validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple


class SpatialDiscretization(IntEnum):
    FV_FIRST_ORDER = 0
    FV_MUSCL_FAST = 1
    FV_MUSCL_MINMOD = 2
    DG_P0 = 3
    DG_P1 = 4
    # Backward-compatibility aliases while WENO is not yet implemented.
    FV_MUSCL = FV_MUSCL_FAST
    FV_WENO = FV_MUSCL_MINMOD


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

    def initialize(self) -> None:
        # TODO: parse SWMM input and build CPU/GPU sparse graph representation.
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
