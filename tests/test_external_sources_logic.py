import numpy as np
from unittest.mock import MagicMock


def test_apply_external_sources_logic_runs_without_qt():
    from swe2d.workbench.studio_dialog import _apply_external_sources_logic
    backend = MagicMock()
    backend.cell_areas.return_value = np.array([1.0])
    _apply_external_sources_logic(
        backend=backend,
        dt_step=1.0,
        rain_rate_model=0.0,
        cell_source_model=None,
        coupled_source_rate=None,
        mesh_cell_areas=np.array([1.0]),
        max_source_rate=1.0,
        h_min=1e-4,
        max_rel_depth_increase=0.5,
        max_source_depth_step=0.1,
        shallow_damping_depth=0.0,
        momentum_cap_min_speed=0.0,
        momentum_cap_celerity_mult=0.0,
    )
