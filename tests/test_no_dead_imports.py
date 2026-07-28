import unittest
"""Per audit: these symbols should be gone after Phase 6.

Phase 3.6: the 8 dead ``swe2d.core.gpkg_io`` functions (which are
re-exported from ``swe2d.cli.gpkg_adapter``) are pinned here as dead.
"""
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
    # Phase 3.6: 7 dead gpkg_io functions re-exported from cli.gpkg_adapter.
    # (build_thiessen_rain_cn_forcing_from_gpkg is NOT dead — it's the
    # CLI builder's GPKG-loading shim that delegates to
    # build_thiessen_rain_cn_forcing_qgis.  builder.build_run_context calls
    # it for every spec that has a hyetograph block.)
    ("swe2d.cli.gpkg_adapter", "query_sample_lines_from_qgis"),
    ("swe2d.cli.gpkg_adapter", "query_bc_arrays"),
    ("swe2d.cli.gpkg_adapter", "build_pipe_network_config_from_gpkg"),
    ("swe2d.cli.gpkg_adapter", "build_initial_state_from_json"),
    ("swe2d.cli.gpkg_adapter", "read_drainage_config_from_gpkg"),
    ("swe2d.cli.gpkg_adapter", "load_and_configure_hydrographs"),
    ("swe2d.cli.gpkg_adapter", "load_hydrograph_edge_data"),
    # Also pinned from the source module so the test catches both
    # the re-export and the underlying definition.
    ("swe2d.core.gpkg_io", "query_sample_lines_from_qgis"),
    ("swe2d.core.gpkg_io", "query_bc_arrays"),
    ("swe2d.core.gpkg_io", "build_pipe_network_config_from_gpkg"),
    ("swe2d.core.gpkg_io", "build_initial_state_from_json"),
    ("swe2d.core.gpkg_io", "read_drainage_config_from_gpkg"),
    ("swe2d.core.gpkg_io", "load_and_configure_hydrographs"),
    ("swe2d.core.gpkg_io", "load_hydrograph_edge_data"),
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

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
