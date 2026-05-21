from __future__ import annotations

"""Compatibility index for extracted SWE2D workbench methods.

Call sites in swe2d_workbench_qt import from this module; implementations are
organized into domain modules for maintainability.
"""

from swe2d.workbench.extracted.model_and_run_methods import (  # noqa: F401
    _bind_model_tab_3d_patch_controls,
    _bind_model_tab_3d_subgrid_drainage_controls,
    _bind_model_tab_core_controls,
    _bind_model_tab_hydrology_controls,
    _bind_model_tab_solver_controls,
    _bind_run_tab_controls,
    _connect_project_workbench_state_signals,
    _on_run,
    _preview_coupling_configuration,
)
from swe2d.workbench.extracted.results_and_ui_methods import (  # noqa: F401
    _bind_map_tab_results_controls,
    _bind_right_pane_controls,
    _build_line_sampling_map,
    _refresh_layer_combos,
    _refresh_streamline_traces_overlay,
    _refresh_velocity_vectors_overlay,
)
from swe2d.workbench.extracted.topology_and_io_methods import (  # noqa: F401
    _bind_topology_tab_dynamic_controls,
    _build_pipe_network_config,
    _configure_swe2d_layer_editors,
    _create_2d_model_geopackage,
    _import_mesh_from_layers,
    _mesh_cell_centers_for_gpkg,
    _migrate_2d_model_geopackage,
    _persist_line_results_to_geopackage,
    _poll_topology_mesh_future,
    _update_topology_control_summary,
    _write_hecras_hdf5,
    _write_ugrid_nc,
)
