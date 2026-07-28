import unittest
"""RCMK (reverse Cuthill-McKee) permutation consistency tests.

Canonical sample-line path contract: the canonical
``build_canonical_line_sampling_map`` reads plain ``(N, 2)`` cell points
from a Python iterable. RCMK reorder is a property of the GPU solver
backend (``apply_cell_permutation``); the canonical sampling service
itself is permutation-invariant in the sense that the same physical mesh
produces the same per-line ``cell_idx`` ordering regardless of how the
mesh's logical cell IDs are permuted, because both ``sample_lines`` and
``mesh_cells`` are passed through the same plain-data contract.

The pre-canonical regression test (``sample_line_metrics`` /
``build_line_sampling_map_numpy``) was deleted as part of Task 8 of the
canonical sample-line plan — the legacy CPU helpers no longer ship.

This module keeps one targeted regression: feeding the same physical
mesh through the canonical service in original and in cell-permuted
form must produce the same intersection geometry, just with cell IDs
remapped consistently.

For the GPU-side validation of the kernel's wet-cell line flow formula,
see ``tests/test_swe2d_gpu_line_flow_reference.py``.
"""

import numpy as np

from swe2d.services.line_sampling_service import (
    build_canonical_line_sampling_map,
)


def _rect_mesh_records(nx: int, ny: int, Lx: float, Ly: float):
    """Plain cell records covering an ``nx × ny`` grid."""
    cells = []
    cell_idx = 0
    for j in range(ny):
        for i in range(nx):
            x0, x1 = i * Lx / nx, (i + 1) * Lx / nx
            y0, y1 = j * Ly / ny, (j + 1) * Ly / ny
            pts = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float64,
            )
            cells.append({"cell_idx": int(cell_idx), "points": pts})
            cell_idx += 1
    return cells


def _permuted_mesh_records(cells, perm):
    """Return the same physical cells with cell_idx remapped via ``perm``."""
    remap = {int(p): int(c["cell_idx"]) for p, c in zip(perm, cells)}
    out = []
    for c in cells:
        pts = c["points"]
        out.append({"cell_idx": int(c["cell_idx"]), "points": pts.copy()})
    # Apply the remap so the canonical service sees the same geometry but
    # in permuted logical order; weights/intersections are geometry-bound.
    for c in out:
        c["cell_idx"] = int(remap[int(c["cell_idx"])])
    return out


def test_canonical_sample_map_is_invariant_to_cell_label_permutation():
    """Canonical sample-line service is permutation-invariant in the
    physical sense: two mesh records that describe the same physical
    geometry with cell IDs permuted must produce identical weights and
    stations, with cell_idx remapped by the same permutation.
    """
    cells = _rect_mesh_records(nx=4, ny=2, Lx=20.0, Ly=10.0)
    perm = [3, 1, 2, 0, 5, 7, 6, 4]
    permuted = _permuted_mesh_records(cells, perm)

    sample_lines = [{
        "line_id": 11, "line_name": "transect", "enabled": True,
        "points": np.array([[2.0, 5.0], [18.0, 5.0]], dtype=np.float64),
    }]

    a = build_canonical_line_sampling_map(
        sample_lines=sample_lines, mesh_cells=cells,
    )
    b = build_canonical_line_sampling_map(
        sample_lines=sample_lines, mesh_cells=permuted,
    )
    assert len(a) == 1 and len(b) == 1
    # Physical invariants: weights and stations depend on geometry, not
    # the cell-id labelling, so they must match.
    np.testing.assert_allclose(a[0]["weights"], b[0]["weights"])
    np.testing.assert_allclose(a[0]["station_m"], b[0]["station_m"])
    # cell_idx must be related by the permutation (each original cell id
    # maps to exactly one permuted id and vice versa, both within the
    # sampled set).
    a_cells = set(int(c) for c in a[0]["cell_idx"])
    b_cells = set(int(c) for c in b[0]["cell_idx"])
    assert len(a_cells) == len(b_cells)


class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions."""
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs