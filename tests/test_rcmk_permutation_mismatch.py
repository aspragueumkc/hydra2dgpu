"""Tests that sample_line_metrics works when mesh, sample_map, and h are
all in the same RCMK (solver) order.

The real-world scenario: _mesh_data.cell_nodes is reordered to RCMK after
GPU init (run_controller.py:603). sample_map and cell_solver_z are built
from that reordered _mesh_data. Snapshots from read_snapshots() are also
RCMK. Everything must be consistent — no mismatch between sample_map.cell_idx
and h ordering.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from swe2d.workbench.services.mesh_service import (
    sample_line_metrics,
    build_line_sampling_map,
)


def _make_mesh():
    """4-cell strip mesh."""
    node_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    cell_nodes = np.array([
        [0, 1, 5],
        [1, 2, 6],
        [2, 3, 7],
        [3, 4, 8],
    ], dtype=np.int32)
    return node_x, node_y, cell_nodes


def _make_line():
    return np.array([[0.2, 0.5], [3.8, 0.5]], dtype=np.float64)


def _apply_rcmk(node_x, node_y, cell_nodes, cp):
    """Reorder mesh to RCMK order (same as run_controller.py:603-639)."""
    new_cn = cell_nodes[cp]
    return node_x, node_y, new_cn


def test_rcmk_consistent_order_gives_correct_depth():
    """When sample_map, cell_solver_z, and h are all RCMK, depth is correct."""
    node_x, node_y, cell_nodes = _make_mesh()
    line_xy = _make_line()

    # Original order values
    bed_orig = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    h_orig = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    # RCMK permutation
    cp = np.array([3, 1, 2, 0], dtype=np.int32)

    # Reorder mesh to RCMK (like run_controller.py:603)
    _, _, cell_nodes_rcmk = _apply_rcmk(node_x, node_y, cell_nodes, cp)

    # Build sample_map from RCMK mesh (like run_controller.py after fix)
    node_coords = np.column_stack([node_x, node_y])
    sample_map_rcmk = build_line_sampling_map(node_coords, cell_nodes_rcmk, line_xy)
    assert sample_map_rcmk is not None

    # cell_solver_z from RCMK mesh
    bed_rcmk = bed_orig[cp]

    # h from GPU snapshots (also RCMK)
    h_rcmk = h_orig[cp]
    hu = np.zeros(4, dtype=np.float64)
    hv = np.zeros(4, dtype=np.float64)

    gravity = 9.81
    h_min = 0.01

    result = sample_line_metrics(
        h=h_rcmk, hu=hu, hv=hv, bed=bed_rcmk,
        node_coords=node_coords, cell_nodes=cell_nodes_rcmk, line_xy=line_xy,
        h_min=h_min, timestep_s=0.0, gravity=gravity, sample_map=sample_map_rcmk,
    )

    # With consistent RCMK ordering, depth should reflect h_rcmk values
    # at the correct spatial positions
    assert np.any(np.isfinite(result["depth_m"])), "No finite depth values"
    assert np.any(np.isfinite(result["wse_m"])), "No finite WSE values"

    # Now build sample_map from original mesh, use original h and bed
    sample_map_orig = build_line_sampling_map(node_coords, cell_nodes, line_xy)
    result_orig = sample_line_metrics(
        h=h_orig, hu=np.zeros(4), hv=np.zeros(4), bed=bed_orig,
        node_coords=node_coords, cell_nodes=cell_nodes, line_xy=line_xy,
        h_min=h_min, timestep_s=0.0, gravity=gravity, sample_map=sample_map_orig,
    )

    # Consistent RCMK and consistent original should give same depth/WSE
    # (just different cell ordering, same spatial result)
    valid = np.isfinite(result["depth_m"]) & np.isfinite(result_orig["depth_m"])
    assert valid.any(), "No overlapping valid stations"
    np.testing.assert_allclose(
        result["depth_m"][valid], result_orig["depth_m"][valid],
        rtol=0.1,
        err_msg="RCMK-consistent and original-consistent give different depth",
    )


def test_mismatched_order_gives_wrong_depth():
    """When sample_map is original but h is RCMK, depth is wrong."""
    node_x, node_y, cell_nodes = _make_mesh()
    line_xy = _make_line()
    node_coords = np.column_stack([node_x, node_y])

    bed_orig = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    h_orig = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    cp = np.array([3, 1, 2, 0], dtype=np.int32)

    # sample_map in ORIGINAL order (the bug — built before RCMK reorder)
    sample_map_orig = build_line_sampling_map(node_coords, cell_nodes, line_xy)
    assert sample_map_orig is not None

    # h in RCMK order (from GPU snapshots)
    h_rcmk = h_orig[cp]
    bed_rcmk = bed_orig[cp]

    gravity = 9.81
    h_min = 0.01

    # Correct: original order everywhere
    result_correct = sample_line_metrics(
        h=h_orig, hu=np.zeros(4), hv=np.zeros(4), bed=bed_orig,
        node_coords=node_coords, cell_nodes=cell_nodes, line_xy=line_xy,
        h_min=h_min, timestep_s=0.0, gravity=gravity, sample_map=sample_map_orig,
    )

    # Bug: original sample_map + RCMK h + original bed
    result_bug = sample_line_metrics(
        h=h_rcmk, hu=np.zeros(4), hv=np.zeros(4), bed=bed_orig,
        node_coords=node_coords, cell_nodes=cell_nodes, line_xy=line_xy,
        h_min=h_min, timestep_s=0.0, gravity=gravity, sample_map=sample_map_orig,
    )

    valid = np.isfinite(result_correct["wse_m"]) & np.isfinite(result_bug["wse_m"])
    assert valid.any(), "No valid stations"
    assert not np.allclose(result_correct["wse_m"][valid], result_bug["wse_m"][valid]), \
        "Mismatched order had no effect — test is broken"
