"""CLI GeoPackage adapter (implementation moved to ``swe2d.core.gpkg_io``).

Phase 3.6: 7 uncalled re-exports were removed (``query_sample_lines_from_qgis``,
``query_bc_arrays``, ``build_pipe_network_config_from_gpkg``,
``build_initial_state_from_json``, ``read_drainage_config_from_gpkg``,
``load_and_configure_hydrographs``, ``load_hydrograph_edge_data``).
``build_thiessen_rain_cn_forcing_from_gpkg`` was kept — the CLI builder
calls it.  The thin re-export module now only exposes the gpkg_io helpers
that have active callers.
"""
from swe2d.core.gpkg_io import (
    build_hydraulic_structure_config_from_gpkg,
    build_internal_flow_forcing_from_gpkg,
    build_line_sampling_map_from_gpkg,
    build_thiessen_rain_cn_forcing_from_gpkg,
    collect_bc_layer_edge_groups_from_gpkg,
    collect_bc_layer_hydrographs_from_gpkg,
    query_mesh_from_gpkg,
)

__all__ = [
    "build_hydraulic_structure_config_from_gpkg",
    "build_internal_flow_forcing_from_gpkg",
    "build_line_sampling_map_from_gpkg",
    "build_thiessen_rain_cn_forcing_from_gpkg",
    "collect_bc_layer_edge_groups_from_gpkg",
    "collect_bc_layer_hydrographs_from_gpkg",
    "query_mesh_from_gpkg",
]
