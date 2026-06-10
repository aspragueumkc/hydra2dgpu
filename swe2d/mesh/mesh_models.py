from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConceptualNode:
    node_id: int
    x: float
    y: float


@dataclass
class ConceptualArc:
    arc_id: int
    node0: int = -1
    node1: int = -1
    region_id: int = -1
    arc_role: Optional[str] = None
    points_xy: Optional[List[Tuple[float, float]]] = None
    use_global_arc_ctrl: bool = True
    arc_mode_override: Optional[str] = None
    arc_soft_size_override: Optional[float] = None
    arc_soft_dist_override: Optional[float] = None


@dataclass
class ConceptualRegion:
    region_id: int
    ring_xy: List[Tuple[float, float]]
    default_size: float
    default_cell_type: str
    edge_lengths: Optional[List[float]] = None
    hole_rings: Optional[List[List[Tuple[float, float]]]] = None


@dataclass
class CellConstraint:
    constraint_id: int
    ring_xy: List[Tuple[float, float]]
    target_size: float
    cell_type: str


@dataclass
class QuadEdgeControl:
    region_id: int
    edge_id: int
    points_xy: List[Tuple[float, float]]
    target_size: Optional[float] = None
    n_layers: int = 0
    first_height: Optional[float] = None
    growth_rate: float = 1.0


@dataclass
class ConceptualModel:
    nodes: List[ConceptualNode]
    arcs: List[ConceptualArc]
    regions: List[ConceptualRegion]
    constraints: List[CellConstraint]
    quad_edges: List[QuadEdgeControl]


@dataclass
class MeshResult:
    node_x: np.ndarray
    node_y: np.ndarray
    node_z: np.ndarray
    cell_nodes: np.ndarray
    cell_face_offsets: np.ndarray
    cell_face_nodes: np.ndarray
    cell_type: np.ndarray
    region_id: np.ndarray
    target_size: np.ndarray
    quality_summary: Optional[Dict[str, object]] = None


_CELL_TYPES = {"triangular", "quadrilateral", "cartesian", "channel_generator", "empty"}


@dataclass
class _TQMeshQualityConfig:
    min_angle_deg: float
    max_aspect_ratio: float
    min_area_rel_bbox: float
    strict: bool
    size_scales: Tuple[float, ...]
    smooth_increments: Tuple[int, ...]


@dataclass
class _GmshQualityConfig:
    enabled: bool
    strict: bool
    min_angle_deg: float
    max_aspect_ratio: float
    min_area_rel_bbox: float
    max_non_orth_deg: float
    max_iterations: int
    time_limit_s: float
    size_scales: Tuple[float, ...]
    smooth_increments: Tuple[int, ...]
    recombine_topology_passes: Tuple[int, ...]
    recombine_min_quality: Tuple[float, ...]
    random_factors: Tuple[float, ...]
    optimize_methods: Tuple[str, ...]
    algorithm_switch_on_failure: bool
    recombine_node_repositioning: bool


class MeshingBackend:
    name = "base"

    def generate(self, model: "ConceptualModel") -> "MeshResult":
        raise NotImplementedError()


__all__ = [
    "CellConstraint",
    "ConceptualArc",
    "ConceptualModel",
    "ConceptualNode",
    "ConceptualRegion",
    "MeshResult",
    "QuadEdgeControl",
    "_CELL_TYPES",
    "_GmshQualityConfig",
    "_TQMeshQualityConfig",
]
