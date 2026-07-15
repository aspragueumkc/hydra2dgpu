"""Regression tests for Thiessen rain/CN forcing under RCMK cell permutation."""

import numpy as np

from swe2d.boundary_and_forcing.rainfall_hydrology import Hyetograph, ThiessenRainCNForcing
from swe2d.boundary_and_forcing.runtime_source_logic import permute_thiessen_forcing


def test_permute_thiessen_forcing_maps_to_solver_order():
    """Forcing built in original order must produce solver-order mapping after permutation."""
    n = 5
    cell_to_gauge = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    curve_number = np.array([80.0, 70.0, 60.0, 50.0, 90.0], dtype=np.float64)
    hy0 = Hyetograph(times_s=np.array([0.0, 3600.0]), cumulative_mm=np.array([0.0, 25.0]))
    hy1 = Hyetograph(times_s=np.array([0.0, 3600.0]), cumulative_mm=np.array([0.0, 10.0]))
    forcing = ThiessenRainCNForcing(
        cell_to_gauge=cell_to_gauge,
        gauge_hyetographs={0: hy0, 1: hy1},
        curve_number=curve_number,
        ia_ratio=0.2,
        infiltration_method="scs_cn",
    )

    # Solver permutation: new cell 0 was old cell 4, new cell 1 was old cell 3, ...
    cell_perm = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    permuted = permute_thiessen_forcing(forcing, cell_perm)

    assert permuted is not forcing
    np.testing.assert_array_equal(permuted.cell_to_gauge, cell_to_gauge[cell_perm])
    np.testing.assert_array_equal(permuted.curve_number, curve_number[cell_perm])
    assert permuted.ia_ratio == forcing.ia_ratio
    assert permuted.infiltration_method == forcing.infiltration_method
    assert permuted.gauge_hyetographs == forcing.gauge_hyetographs


def test_permute_thiessen_forcing_no_change_without_permutation():
    """Empty permutation must leave forcing unchanged."""
    cell_to_gauge = np.array([0, 0, 1], dtype=np.int32)
    forcing = ThiessenRainCNForcing(
        cell_to_gauge=cell_to_gauge,
        gauge_hyetographs={},
        curve_number=np.full(3, 80.0),
    )
    assert permute_thiessen_forcing(forcing, np.array([], dtype=np.int32)) is forcing
    assert permute_thiessen_forcing(forcing, None) is forcing
