"""Regression tests for internal-flow forcing under RCMK cell permutation."""

import numpy as np

from swe2d.boundary_and_forcing.runtime_source_logic import (
    internal_flow_source_cms_at_time,
    permute_internal_flow_forcing,
)


def test_permute_internal_flow_forcing_maps_to_solver_order():
    """Forcing built in original order must produce solver-order source after permutation."""
    n = 5
    base_q = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    idx = np.array([1, 3], dtype=np.int32)
    wt = np.array([0.5, 0.5], dtype=np.float64)
    hg = (np.array([0.0, 1.0]), np.array([10.0, 20.0]))
    forcing = {
        "base_q": base_q,
        "dynamic_terms": [(idx, wt, hg)],
        "layer_name": "test",
    }

    # Solver permutation: new cell 0 was old cell 4, new cell 1 was old cell 3, ...
    cell_perm = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    permuted = permute_internal_flow_forcing(forcing, cell_perm)

    # Base_q should be in solver order.
    expected_base_q = base_q[cell_perm]
    np.testing.assert_array_equal(permuted["base_q"], expected_base_q)

    # Dynamic indices should be mapped to solver indices via inv_perm.
    inv_perm = np.zeros(n, dtype=np.int32)
    inv_perm[cell_perm] = np.arange(n, dtype=np.int32)
    expected_idx = inv_perm[idx]
    actual_idx, actual_wt, actual_hg = permuted["dynamic_terms"][0]
    np.testing.assert_array_equal(actual_idx, expected_idx)
    np.testing.assert_array_equal(actual_wt, wt)
    assert actual_hg is hg

    # Source computed from the permuted forcing should be in solver order.
    def interp(h, t):
        return 10.0

    src = internal_flow_source_cms_at_time(permuted, 0.0, interp)
    expected_src = expected_base_q.copy()
    expected_src[expected_idx] += 10.0 * wt
    np.testing.assert_allclose(src, expected_src)


def test_permute_internal_flow_forcing_no_change_without_permutation():
    """Empty permutation must leave forcing unchanged."""
    base_q = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    forcing = {"base_q": base_q, "dynamic_terms": [], "layer_name": "test"}
    permuted = permute_internal_flow_forcing(forcing, np.array([], dtype=np.int32))
    assert permuted is forcing

    permuted = permute_internal_flow_forcing(forcing, None)
    assert permuted is forcing
