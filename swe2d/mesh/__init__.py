"""Mesh and mesh-runtime exports for incremental package migration."""

from swe2d.mesh.meshing import (
    CellConstraint,
    ConceptualArc,
    ConceptualModel,
    ConceptualNode,
    ConceptualRegion,
    MeshResult,
    MeshingBackend,
    QuadEdgeControl,
    StructuredFaceCentricBackend,
    conceptual_from_qgis_layers,
)
from swe2d.mesh.meshing import GmshBackend
from swe2d.mesh.bridge_stacked_mesh import (
    BridgeStackedGeometrySpec,
    BridgeStackedPlan,
    bridge_specs_from_structure_config,
    build_bridge_stacked_plan,
)
from swe2d.mesh.mesh_runtime_logic import (
    boundary_buffer_cells,
    inflow_adjacent_cells,
    initial_state,
    mesh_cell_areas,
    mesh_cell_centroids,
    mesh_cell_min_bed,
    mesh_cell_solver_bed,
)
from swe2d.boundary_and_forcing.boundary_runtime_logic import (
    collect_boundary_arrays,
    mesh_boundary_edges,
)

__all__ = [
    "BridgeStackedGeometrySpec",
    "BridgeStackedPlan",
    "CellConstraint",
    "ConceptualArc",
    "ConceptualModel",
    "ConceptualNode",
    "ConceptualRegion",
    "GmshBackend",
    "MeshResult",
    "MeshingBackend",
    "QuadEdgeControl",
    "StructuredFaceCentricBackend",
    "boundary_buffer_cells",
    "bridge_specs_from_structure_config",
    "build_bridge_stacked_plan",
    "collect_boundary_arrays",
    "conceptual_from_qgis_layers",
    "inflow_adjacent_cells",
    "initial_state",
    "mesh_boundary_edges",
    "mesh_cell_areas",
    "mesh_cell_centroids",
    "mesh_cell_min_bed",
    "mesh_cell_solver_bed",
]
