"""Per audit: these symbols should be gone after Phase 6."""
import importlib
import pytest


_DEAD = [
    ("swe2d.services.gpkg_persistence_service", "collect_run_log_metadata"),
    ("swe2d.workbench.controllers.run_controller", "RunController._noop"),
    ("swe2d.workbench.controllers.topology_controller", "_opt_float"),
    ("swe2d.workbench.controllers.topology_controller", "_opt_bool"),
    ("swe2d.cli.gpkg_adapter", "apply_bc_overrides_from_gpkg"),
    ("swe2d.cli.gpkg_adapter", "_parse_linestring_coords"),
    ("swe2d.workbench.devtools.widget_walker", "iter_with_parents"),
    ("swe2d.extensions.extension_models", "RainfallSourceEngine"),
    ("swe2d.runtime.coupling", "SWE2DCouplingController.source_rate_callback"),
    ("swe2d.mesh", "meshing._gmsh_available"),
    ("swe2d.boundary_and_forcing.rainfall_hydrology",
     "runoff_depth_mm_from_event_rain_mm"),
    ("swe2d.boundary_and_forcing.rainfall_hydrology", "composite_curve_number"),
    ("swe2d.boundary_and_forcing.rainfall_hydrology",
     "time_of_concentration_hours_velocity_method"),
    ("swe2d.extensions.extension_models", "compute_orifice_flow"),
    ("swe2d.extensions.extension_models", "compute_weir_flow"),
    ("swe2d.extensions.extension_models",
     "compute_pipe_manning_capacity_full"),
    ("swe2d.extensions.extension_models", "circular_section_from_depth"),
    ("swe2d.extensions.extension_models", "convert_cell_flows_to_depth_rates"),
]


@pytest.mark.parametrize("module_name,symbol", _DEAD)
def test_dead_symbol_removed(module_name, symbol):
    """Module-level symbols and Class.attr should not be importable."""
    mod = importlib.import_module(module_name)
    parts = symbol.split(".")
    if len(parts) == 1:
        assert not hasattr(mod, symbol), f"{module_name}.{symbol} still exists"
    else:
        cls_name, attr_name = parts[0], parts[1]
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            assert not hasattr(cls, attr_name), (
                f"{module_name}.{symbol} still exists"
            )
