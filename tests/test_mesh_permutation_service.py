"""Test for RCMK cell-permutation extracted from run_controller._execute_run."""
import numpy as np
import pytest


def test_apply_cell_permutation_triangle_mesh():
    from swe2d.workbench.services.mesh_service import apply_cell_permutation
    mesh = {
        "cell_nodes": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    }
    perm = np.array([1, 0], dtype=np.int32)
    out = apply_cell_permutation(mesh, perm)
    # Triangle cells (2 per row of 3): swap the two triangles.
    np.testing.assert_array_equal(out["cell_nodes"], [3, 4, 5, 0, 1, 2])


def test_apply_cell_permutation_mixed_mesh():
    from swe2d.workbench.services.mesh_service import apply_cell_permutation
    mesh = {
        "cell_face_offsets": np.array([0, 3, 6], dtype=np.int32),
        "cell_face_nodes": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    }
    perm = np.array([1, 0], dtype=np.int32)
    out = apply_cell_permutation(mesh, perm)
    np.testing.assert_array_equal(out["cell_face_nodes"], [3, 4, 5, 0, 1, 2])
    np.testing.assert_array_equal(out["cell_face_offsets"], [0, 3, 6])


def test_apply_cell_permutation_no_perm_returns_unchanged():
    from swe2d.workbench.services.mesh_service import apply_cell_permutation
    mesh = {"cell_nodes": np.array([0, 1, 2], dtype=np.int32)}
    perm = np.array([0], dtype=np.int32)
    out = apply_cell_permutation(mesh, perm)
    np.testing.assert_array_equal(out["cell_nodes"], [0, 1, 2])
