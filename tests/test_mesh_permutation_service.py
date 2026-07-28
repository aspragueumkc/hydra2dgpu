import unittest
"""Test for RCMK cell-permutation extracted from run_controller._execute_run."""
import numpy as np
import pytest


def test_apply_cell_permutation_triangle_mesh():
    from swe2d.core.mesh_service import apply_cell_permutation
    mesh = {
        "cell_nodes": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    }
    perm = np.array([1, 0], dtype=np.int32)
    out = apply_cell_permutation(mesh, perm)
    # Triangle cells (2 per row of 3): swap the two triangles.
    np.testing.assert_array_equal(out["cell_nodes"], [3, 4, 5, 0, 1, 2])


def test_apply_cell_permutation_mixed_mesh():
    from swe2d.core.mesh_service import apply_cell_permutation
    mesh = {
        "cell_face_offsets": np.array([0, 3, 6], dtype=np.int32),
        "cell_face_nodes": np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    }
    perm = np.array([1, 0], dtype=np.int32)
    out = apply_cell_permutation(mesh, perm)
    np.testing.assert_array_equal(out["cell_face_nodes"], [3, 4, 5, 0, 1, 2])
    np.testing.assert_array_equal(out["cell_face_offsets"], [0, 3, 6])


def test_apply_cell_permutation_no_perm_returns_unchanged():
    from swe2d.core.mesh_service import apply_cell_permutation
    mesh = {"cell_nodes": np.array([0, 1, 2], dtype=np.int32)}
    perm = np.array([0], dtype=np.int32)
    out = apply_cell_permutation(mesh, perm)
    np.testing.assert_array_equal(out["cell_nodes"], [0, 1, 2])

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
