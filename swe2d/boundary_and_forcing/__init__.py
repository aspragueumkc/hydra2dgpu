"""Boundary/source forcing exports for incremental package migration."""

from swe2d.boundary_and_forcing.bc_logic import (
    apply_timeseries_bc_values,
    distribute_total_flow_to_unit_q,
    interp_hydrograph,
    normalize_inflow_to_uniform_velocity,
)
from swe2d.boundary_and_forcing.boundary_qgis_adapter import (
    apply_bc_layer_overrides_qgis,
    collect_bc_layer_edge_groups_qgis,
    collect_bc_layer_hydrographs_qgis,
)
from swe2d.boundary_and_forcing.boundary_runtime_logic import (
    collect_boundary_arrays,
    mesh_boundary_edges,
)
from swe2d.boundary_and_forcing.hydrograph_logic import (
    hydrograph_from_layer,
    parse_hydrograph_text,
    parse_time_hours,
)
from swe2d.boundary_and_forcing.internal_flow_logic import (
    build_hydrograph_lookup_from_features,
    build_internal_flow_forcing_from_features,
    first_matching_field,
    resolve_internal_flow_field_name,
)
from swe2d.boundary_and_forcing.internal_flow_qgis_adapter import (
    build_internal_flow_forcing_qgis,
)
from swe2d.boundary_and_forcing.internal_flow_qgis_geometry import (
    internal_flow_geom_to_indices_weights_qgis,
)
from swe2d.boundary_and_forcing.native_bc_forcing import (
    BoundaryHydrographConfigurator,
)
from swe2d.runtime.native_bc_forcing import SWE2DNativeBoundaryHydrographConfigurator
from swe2d.boundary_and_forcing.runtime_source_logic import (
    apply_external_sources,
    internal_flow_source_cms_at_time,
)
from swe2d.runtime.runtime_sources import SWE2DRuntimeSourceManager
from swe2d.boundary_and_forcing.spatial_forcing_qgis_adapter import (
    build_spatial_cn_array_qgis,
    build_spatial_manning_array_qgis,
    build_thiessen_rain_cn_forcing_qgis,
)

__all__ = [
    "SWE2DNativeBoundaryHydrographConfigurator",
    "SWE2DRuntimeSourceManager",
    "apply_bc_layer_overrides_qgis",
    "apply_external_sources",
    "apply_timeseries_bc_values",
    "build_hydrograph_lookup_from_features",
    "build_internal_flow_forcing_from_features",
    "build_internal_flow_forcing_qgis",
    "build_spatial_cn_array_qgis",
    "build_spatial_manning_array_qgis",
    "build_thiessen_rain_cn_forcing_qgis",
    "collect_bc_layer_edge_groups_qgis",
    "collect_bc_layer_hydrographs_qgis",
    "collect_boundary_arrays",
    "distribute_total_flow_to_unit_q",
    "first_matching_field",
    "hydrograph_from_layer",
    "interp_hydrograph",
    "internal_flow_geom_to_indices_weights_qgis",
    "internal_flow_source_cms_at_time",
    "mesh_boundary_edges",
    "normalize_inflow_to_uniform_velocity",
    "parse_hydrograph_text",
    "parse_time_hours",
    "resolve_internal_flow_field_name",
]
