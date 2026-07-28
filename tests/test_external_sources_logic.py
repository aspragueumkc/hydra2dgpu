import numpy as np
from unittest.mock import MagicMock, patch


def test_apply_external_sources_logic_runs_without_qt():
    from swe2d.core.runtime_source_application_service import (
        _apply_external_sources_logic,
    )

    backend = MagicMock()
    rain_rate_model = 0.0
    cell_source_model = np.array([0.1, 0.2])
    coupled_source_rate = np.array([0.01, 0.02])
    mesh_cell_areas = np.array([10.0, 20.0])

    with patch(
        "swe2d.boundary_and_forcing.runtime_source_logic.apply_external_sources"
    ) as mock_apply:
        _apply_external_sources_logic(
            backend=backend,
            dt_step=1.0,
            rain_rate_model=rain_rate_model,
            cell_source_model=cell_source_model,
            coupled_source_rate=coupled_source_rate,
            mesh_cell_areas=mesh_cell_areas,
            max_source_rate=1.0,
            h_min=1e-4,
            max_rel_depth_increase=0.5,
            max_source_depth_step=0.1,
            shallow_damping_depth=0.0,
            momentum_cap_min_speed=0.0,
            momentum_cap_celerity_mult=0.0,
        )

    mock_apply.assert_called_once_with(
        backend=backend,
        dt_step=1.0,
        rain_rate_model=rain_rate_model,
        cell_source_model=cell_source_model,
        coupled_source_rate=coupled_source_rate,
        mesh_cell_areas=mesh_cell_areas,
        max_source_rate=1.0,
        h_min=1e-4,
        max_rel_depth_increase=0.5,
        max_source_depth_step=0.1,
        shallow_damping_depth=0.0,
        momentum_cap_min_speed=0.0,
        momentum_cap_celerity_mult=0.0,
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
