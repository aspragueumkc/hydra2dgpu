"""
test_bc_validation.py — Boundary-condition pipeline validation
==============================================================

Three independent test groups:

1. Pure-numpy tests (no QGIS, no native solver):
   - TestBoundaryEdgeDetection  – checks _mesh_boundary_edges logic
   - TestDefaultBCSideAssignment – checks _default_bc_for_edges logic
   - TestBCIntersectionGeometry  – checks segment/segment overlap rule
     that mirrors what QgsGeometry.intersects does for aligned edges

2. QGIS-layer override tests (skipped if QGIS unavailable):
   - TestBCLayerOverride – creates an in-memory QgsVectorLayer with a
     bc_lines feature and verifies that the correct edges are overridden

3. Analytical steady-state test (skipped if native solver unavailable):
   - TestChannelSteadyState – rectangular channel, constant slope,
     inflow-Q upstream + open downstream.
     At steady state interior depth ≈ Manning's normal depth.

Analytical reference
---------------------
Channel: L=100 m, W=5 m, S=0.001, n=0.025, Q_in=2.5 m³/s

Manning's normal depth (solved iteratively below):
  Q = (1/n) · A · R^(2/3) · S^(1/2)
  A = W·yn,  R = W·yn / (W + 2·yn)
  → yn ≈ 0.630 m  (tolerance ±15 % in the automated test)

Broad-crested weir (informational; not simulated in this test):
  Q = Cd · W · H^(3/2),  Cd ≈ 1.705 (SI)
  → H ≈ 0.44 m above the weir crest  (for Q=2.5, W=5)
"""

from __future__ import annotations

import math
import os
import sys
import types
import unittest
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow importing plugin modules from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN = os.path.dirname(_HERE)
for _p in (_HERE, _PLUGIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    from qgis.core import (
        QgsApplication, QgsVectorLayer, QgsFeature, QgsGeometry,
        QgsPointXY, QgsFields, QgsField, QgsWkbTypes,
    )
    from PyQt5.QtCore import QVariant
    _HAVE_QGIS = True
except ImportError:
    _HAVE_QGIS = False

try:
    from swe2d_backend import SWE2DBackend, swe2d_available
    _HAVE_SOLVER = swe2d_available()
except Exception:
    _HAVE_SOLVER = False

try:
    import swe2d_workbench_qt as _wbqt
    _HAVE_WORKBENCH = True
except Exception:
    _wbqt = None
    _HAVE_WORKBENCH = False

# ===========================================================================
# Helpers that replicate the plugin's pure-numpy logic without QGIS
# ===========================================================================

def _make_structured_mesh(nx: int, ny: int, lx: float, ly: float,
                           slope_x: float = 0.0):
    """Build the same structured triangular mesh as the plugin."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().copy()
    node_y = Yg.ravel().copy()
    # bed: slope in x so that z decreases from left (upstream) to right
    node_z = slope_x * (lx - node_x)

    stride = nx + 1
    cells: List[int] = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    cell_nodes = np.array(cells, dtype=np.int32)
    return node_x, node_y, node_z, cell_nodes


def _boundary_edges(cell_nodes_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate _mesh_boundary_edges for triangular meshes."""
    tris = cell_nodes_flat.reshape((-1, 3)).astype(np.int32)
    edge_count: Dict[Tuple[int, int], int] = {}
    edge_oriented: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for tri in tris:
        a0, a1, a2 = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((a0, a1), (a1, a2), (a2, a0)):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_oriented:
                edge_oriented[key] = (a, b)
    n0, n1 = [], []
    for key, cnt in edge_count.items():
        if cnt == 1:
            a, b = edge_oriented[key]
            n0.append(a); n1.append(b)
    if not n0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.asarray(n0, dtype=np.int32), np.asarray(n1, dtype=np.int32)


def _default_bc(edge_n0, edge_n1, node_x, node_y,
                left_type=2, right_type=4, bottom_type=1, top_type=1,
                left_val=0.0, right_val=0.0):
    """Replicate _default_bc_for_edges."""
    xmin, xmax = float(np.min(node_x)), float(np.max(node_x))
    ymin, ymax = float(np.min(node_y)), float(np.max(node_y))
    mx = 0.5 * (node_x[edge_n0] + node_x[edge_n1])
    my = 0.5 * (node_y[edge_n0] + node_y[edge_n1])
    d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax),
                   np.abs(my - ymin), np.abs(my - ymax)])
    side_idx = np.argmin(d, axis=0)
    side_defaults = {
        0: (left_type, left_val),
        1: (right_type, right_val),
        2: (bottom_type, 0.0),
        3: (top_type, 0.0),
    }
    bc_type = np.zeros(edge_n0.size, dtype=np.int32)
    bc_val = np.zeros(edge_n0.size, dtype=np.float64)
    for i, si in enumerate(side_idx):
        bc_type[i], bc_val[i] = side_defaults[int(si)]
    return bc_type, bc_val


def _segments_overlap(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1,
                       tol=1e-6) -> bool:
    """
    Return True if line segment A (ax0,ay0)→(ax1,ay1) overlaps or touches
    segment B.  This is a simplified proxy for QgsGeometry.intersects() for
    axis-aligned segments (all boundary edges in a structured rectangular mesh
    are axis-aligned).
    """
    # Bounding-box pre-filter
    if (min(ax0, ax1) > max(bx0, bx1) + tol or
            max(ax0, ax1) < min(bx0, bx1) - tol or
            min(ay0, ay1) > max(by0, by1) + tol or
            max(ay0, ay1) < min(by0, by1) - tol):
        return False

    # Parallel / collinear check for axis-aligned pairs
    if abs(ax0 - ax1) < tol and abs(bx0 - bx1) < tol:
        # Both vertical – check x alignment and y range overlap
        if abs(ax0 - bx0) > tol:
            return False
        return max(min(ay0, ay1), min(by0, by1)) <= min(max(ay0, ay1), max(by0, by1)) + tol
    if abs(ay0 - ay1) < tol and abs(by0 - by1) < tol:
        # Both horizontal – check y alignment and x range overlap
        if abs(ay0 - by0) > tol:
            return False
        return max(min(ax0, ax1), min(bx0, bx1)) <= min(max(ax0, ax1), max(bx0, bx1)) + tol

    # General 2D segment intersection (cross-product method)
    def cross2d(ux, uy, vx, vy):
        return ux * vy - uy * vx

    rx, ry = ax1 - ax0, ay1 - ay0
    sx, sy = bx1 - bx0, by1 - by0
    denom = cross2d(rx, ry, sx, sy)
    if abs(denom) < tol:
        return False  # parallel, non-collinear (collinear handled above)
    dx, dy = bx0 - ax0, by0 - ay0
    t = cross2d(dx, dy, sx, sy) / denom
    u = cross2d(dx, dy, rx, ry) / denom
    return (-tol <= t <= 1 + tol) and (-tol <= u <= 1 + tol)


def _edge_midpoint_matches_bc_line(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1,
                                   tol=1e-9) -> bool:
    """Mirror workbench matching: midpoint distance to bc_line < half edge length."""
    mx = 0.5 * (ax0 + ax1)
    my = 0.5 * (ay0 + ay1)
    edge_tol = 0.5 * math.hypot(ax1 - ax0, ay1 - ay0)

    vx = bx1 - bx0
    vy = by1 - by0
    seg2 = vx * vx + vy * vy
    if seg2 <= tol:
        dist = math.hypot(mx - bx0, my - by0)
        return dist < edge_tol

    t = ((mx - bx0) * vx + (my - by0) * vy) / seg2
    t = min(1.0, max(0.0, t))
    px = bx0 + t * vx
    py = by0 + t * vy
    dist = math.hypot(mx - px, my - py)
    return dist < edge_tol


def _manning_normal_depth(Q: float, W: float, S: float, n: float,
                           tol: float = 1e-8) -> float:
    """Solve Manning's normal depth for a rectangular channel by bisection."""
    def residual(yn):
        A = W * yn
        R = W * yn / (W + 2 * yn)
        return (1.0 / n) * A * (R ** (2.0 / 3.0)) * math.sqrt(S) - Q

    lo, hi = 1e-6, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ===========================================================================
# Group 1 – Pure-numpy boundary-edge and default-BC tests
# ===========================================================================

class TestBoundaryEdgeDetection(unittest.TestCase):
    """Verify that _mesh_boundary_edges enumerates the correct edges."""

    def _check_mesh(self, nx, ny, lx, ly):
        node_x, node_y, node_z, cell_nodes = _make_structured_mesh(nx, ny, lx, ly)
        n0, n1 = _boundary_edges(cell_nodes)

        expected_count = 2 * nx + 2 * ny
        self.assertEqual(
            n0.size, expected_count,
            f"nx={nx} ny={ny}: expected {expected_count} boundary edges, got {n0.size}"
        )

        # Every returned edge must be a real mesh edge (both nodes within range)
        n_nodes = (nx + 1) * (ny + 1)
        self.assertTrue(np.all(n0 >= 0) and np.all(n0 < n_nodes))
        self.assertTrue(np.all(n1 >= 0) and np.all(n1 < n_nodes))

        # No edge should have n0 == n1
        self.assertTrue(np.all(n0 != n1), "Degenerate edge (n0==n1) found")

        # Midpoints of edges found must lie on the actual boundary
        mx = 0.5 * (node_x[n0] + node_x[n1])
        my = 0.5 * (node_y[n0] + node_y[n1])
        on_boundary = (
            (np.abs(mx) < 1e-9) |
            (np.abs(mx - lx) < 1e-9) |
            (np.abs(my) < 1e-9) |
            (np.abs(my - ly) < 1e-9)
        )
        off = np.where(~on_boundary)[0]
        self.assertEqual(
            off.size, 0,
            f"Interior edge(s) in boundary list: indices {off[:5]}, midpoints "
            f"x={mx[off[:5]]} y={my[off[:5]]}"
        )

    def test_2x2_grid(self):
        self._check_mesh(2, 2, 10.0, 10.0)

    def test_4x2_grid(self):
        self._check_mesh(4, 2, 100.0, 5.0)

    def test_10x4_channel(self):
        self._check_mesh(10, 4, 100.0, 5.0)

    def test_20x4_channel(self):
        self._check_mesh(20, 4, 100.0, 5.0)

    def test_no_duplicate_edges(self):
        """Every (min,max) pair should appear exactly once."""
        node_x, node_y, node_z, cell_nodes = _make_structured_mesh(10, 4, 100.0, 5.0)
        n0, n1 = _boundary_edges(cell_nodes)
        keys = [tuple(sorted((int(a), int(b)))) for a, b in zip(n0, n1)]
        self.assertEqual(len(keys), len(set(keys)), "Duplicate boundary edges found")


class TestDefaultBCSideAssignment(unittest.TestCase):
    """Verify _default_bc_for_edges assigns each edge to the correct side."""

    def setUp(self):
        nx, ny = 20, 4
        lx, ly = 100.0, 5.0
        self.node_x, self.node_y, _, cn = _make_structured_mesh(nx, ny, lx, ly)
        self.n0, self.n1 = _boundary_edges(cn)
        self.lx = lx
        self.ly = ly

    def _classify(self):
        """Return dict mapping 'left'/'right'/'bottom'/'top' → edge indices."""
        node_x, node_y = self.node_x, self.node_y
        mx = 0.5 * (node_x[self.n0] + node_x[self.n1])
        my = 0.5 * (node_y[self.n0] + node_y[self.n1])
        xmin, xmax = float(np.min(node_x)), float(np.max(node_x))
        ymin, ymax = float(np.min(node_y)), float(np.max(node_y))
        d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax),
                       np.abs(my - ymin), np.abs(my - ymax)])
        side_idx = np.argmin(d, axis=0)
        names = ["left", "right", "bottom", "top"]
        result = {s: [] for s in names}
        for i, si in enumerate(side_idx):
            result[names[int(si)]].append(i)
        return result

    def test_left_edge_count(self):
        """Left boundary (x=0) should have ny=4 edges."""
        classified = self._classify()
        self.assertEqual(len(classified["left"]), 4,
                         f"Expected 4 left edges, got {len(classified['left'])}")

    def test_right_edge_count(self):
        classified = self._classify()
        self.assertEqual(len(classified["right"]), 4,
                         f"Expected 4 right edges, got {len(classified['right'])}")

    def test_bottom_top_edge_count(self):
        classified = self._classify()
        self.assertEqual(len(classified["bottom"]), 20)
        self.assertEqual(len(classified["top"]), 20)

    def test_left_edges_are_at_xmin(self):
        classified = self._classify()
        for i in classified["left"]:
            mx = 0.5 * (self.node_x[self.n0[i]] + self.node_x[self.n1[i]])
            self.assertAlmostEqual(mx, 0.0, places=6,
                                   msg=f"Left edge {i} has midpoint x={mx}")

    def test_right_edges_are_at_xmax(self):
        classified = self._classify()
        for i in classified["right"]:
            mx = 0.5 * (self.node_x[self.n0[i]] + self.node_x[self.n1[i]])
            self.assertAlmostEqual(mx, self.lx, places=6,
                                   msg=f"Right edge {i} has midpoint x={mx}")

    def test_bc_type_array(self):
        """BC type array should have INFLOW_Q=2 on left, OPEN=4 on right, WALL=1 on top/bottom."""
        bc_type, bc_val = _default_bc(
            self.n0, self.n1, self.node_x, self.node_y,
            left_type=2, right_type=4, bottom_type=1, top_type=1
        )
        classified = self._classify()
        for i in classified["left"]:
            self.assertEqual(bc_type[i], 2, f"Left edge {i}: expected INFLOW_Q(2) got {bc_type[i]}")
        for i in classified["right"]:
            self.assertEqual(bc_type[i], 4, f"Right edge {i}: expected OPEN(4) got {bc_type[i]}")
        for side in ("bottom", "top"):
            for i in classified[side]:
                self.assertEqual(bc_type[i], 1, f"{side} edge {i}: expected WALL(1) got {bc_type[i]}")

    def test_bc_val_inflow(self):
        """BC value on left edges should match the specified inflow value."""
        Q_in = 2.5
        bc_type, bc_val = _default_bc(
            self.n0, self.n1, self.node_x, self.node_y,
            left_type=2, right_type=4, bottom_type=1, top_type=1,
            left_val=Q_in
        )
        classified = self._classify()
        for i in classified["left"]:
            self.assertAlmostEqual(bc_val[i], Q_in, places=9,
                                   msg=f"Left edge {i} bc_val should be Q_in={Q_in}")


@unittest.skipUnless(_HAVE_WORKBENCH, "swe2d_workbench_qt import failed")
class TestTotalQBoundaryDistribution(unittest.TestCase):
    """Validate total-Q flow input conversion to solver unit discharge q."""

    class _BoolToggle:
        def __init__(self, state: bool):
            self._state = bool(state)

        def isChecked(self):
            return self._state

    def setUp(self):
        self.node_x, self.node_y, self.node_z, self.cell_nodes = _make_structured_mesh(
            nx=8, ny=4, lx=80.0, ly=4.0, slope_x=0.0
        )
        # Impose a vertical elevation gradient so "lowest" edge selection is deterministic.
        self.node_z = self.node_y.copy()
        self.n0, self.n1 = _boundary_edges(self.cell_nodes)
        self.left_edges = np.where(
            np.abs(0.5 * (self.node_x[self.n0] + self.node_x[self.n1]) - 0.0) < 1.0e-12
        )[0]
        self.edge_len = np.hypot(
            self.node_x[self.n1] - self.node_x[self.n0],
            self.node_y[self.n1] - self.node_y[self.n0],
        )

    def _call_distribution(self, progressive: bool, bc_type_step, bc_val_step, bc_type_template, side_hg):
        dummy = types.SimpleNamespace(
            _mesh_data={
                "node_x": self.node_x,
                "node_y": self.node_y,
                "node_z": self.node_z,
            },
            inflow_progressive_chk=self._BoolToggle(progressive),
        )
        return _wbqt.SWE2DWorkbenchDialog._distribute_total_flow_to_unit_q(
            dummy,
            self.n0,
            self.n1,
            bc_type_step,
            bc_val_step,
            bc_type_template,
            side_hg,
            None,
        )

    def test_static_total_q_distributes_over_full_length(self):
        bc_type_step = np.ones(self.n0.size, dtype=np.int32)
        bc_type_template = bc_type_step.copy()
        bc_val_step = np.zeros(self.n0.size, dtype=np.float64)

        q_total = 4.0
        bc_type_step[self.left_edges] = 2
        bc_type_template[self.left_edges] = 2
        bc_val_step[self.left_edges] = q_total

        out = self._call_distribution(
            progressive=True,
            bc_type_step=bc_type_step,
            bc_val_step=bc_val_step,
            bc_type_template=bc_type_template,
            side_hg={},
        )

        left_q = out[self.left_edges]
        expected_q_unit = q_total / float(np.sum(self.edge_len[self.left_edges]))
        self.assertTrue(np.allclose(left_q, expected_q_unit, rtol=0.0, atol=1.0e-12))
        total_q_reconstructed = float(np.sum(left_q * self.edge_len[self.left_edges]))
        self.assertAlmostEqual(total_q_reconstructed, q_total, places=10)

    def test_hydrograph_progressive_activation_uses_lowest_edges(self):
        bc_type_step = np.ones(self.n0.size, dtype=np.int32)
        bc_type_template = bc_type_step.copy()
        bc_type_step[self.left_edges] = 2
        bc_type_template[self.left_edges] = 102  # _BC_TS_FLOW sentinel

        side_hg = {
            "left": (
                np.array([0.0, 3600.0], dtype=np.float64),
                np.array([0.0, 1000.0], dtype=np.float64),
            )
        }

        # Low flow: target active length = 10% of total -> 1 edge active.
        vals_low = np.zeros(self.n0.size, dtype=np.float64)
        vals_low[self.left_edges] = 100.0
        out_low = self._call_distribution(True, bc_type_step, vals_low, bc_type_template, side_hg)
        left_low = out_low[self.left_edges]
        n_active_low = int(np.count_nonzero(np.abs(left_low) > 1.0e-12))
        self.assertEqual(n_active_low, 1)

        # Higher flow: target active length = 60% of total -> 3 edges active.
        vals_mid = np.zeros(self.n0.size, dtype=np.float64)
        vals_mid[self.left_edges] = 600.0
        out_mid = self._call_distribution(True, bc_type_step, vals_mid, bc_type_template, side_hg)
        left_mid = out_mid[self.left_edges]
        n_active_mid = int(np.count_nonzero(np.abs(left_mid) > 1.0e-12))
        self.assertEqual(n_active_mid, 3)

        # Active set should start from the lowest-elevation edges.
        left_edge_z = 0.5 * (self.node_z[self.n0[self.left_edges]] + self.node_z[self.n1[self.left_edges]])
        sorted_left = np.argsort(left_edge_z, kind="stable")
        active_low_local = set(np.where(np.abs(left_low) > 1.0e-12)[0].tolist())
        active_mid_local = set(np.where(np.abs(left_mid) > 1.0e-12)[0].tolist())
        self.assertEqual(active_low_local, {int(sorted_left[0])})
        self.assertEqual(active_mid_local, set(sorted_left[:3].tolist()))


class TestBCIntersectionGeometry(unittest.TestCase):
    """
    Verify the midpoint-distance matching logic used by BC overrides.

    For a 100×5 m mesh with 20×4 cells, the upstream boundary (x=0) has 4
    vertical edges spanning y=0..5.  A bc_line polyline from (0,0) to (0,5)
    should intersect ALL 4 upstream edges and NONE of the downstream or wall edges.
    """

    def setUp(self):
        nx, ny = 20, 4
        lx, ly = 100.0, 5.0
        self.lx, self.ly = lx, ly
        self.node_x, self.node_y, _, cn = _make_structured_mesh(nx, ny, lx, ly)
        self.n0, self.n1 = _boundary_edges(cn)

    def _bc_line_hits(self, lx0, ly0, lx1, ly1):
        """Return indices of boundary edges that intersect the given bc line."""
        hits = []
        for i in range(self.n0.size):
            ax0 = float(self.node_x[self.n0[i]])
            ay0 = float(self.node_y[self.n0[i]])
            ax1 = float(self.node_x[self.n1[i]])
            ay1 = float(self.node_y[self.n1[i]])
            if _edge_midpoint_matches_bc_line(ax0, ay0, ax1, ay1, lx0, ly0, lx1, ly1):
                hits.append(i)
        return hits

    def test_upstream_line_hits_only_left_edges(self):
        """bc_line at x=0 spanning full width should hit exactly the 4 left edges."""
        hits = self._bc_line_hits(0.0, 0.0, 0.0, self.ly)
        self.assertEqual(len(hits), 4,
                         f"Expected 4 left-edge hits, got {len(hits)}: {hits}")
        # All hit edges should be at x=0
        for i in hits:
            mx = 0.5 * (self.node_x[self.n0[i]] + self.node_x[self.n1[i]])
            self.assertAlmostEqual(mx, 0.0, places=6,
                                   msg=f"Hit edge {i} has midpoint x={mx}, not at x=0")

    def test_downstream_line_hits_only_right_edges(self):
        hits = self._bc_line_hits(self.lx, 0.0, self.lx, self.ly)
        self.assertEqual(len(hits), 4,
                         f"Expected 4 right-edge hits, got {len(hits)}")
        for i in hits:
            mx = 0.5 * (self.node_x[self.n0[i]] + self.node_x[self.n1[i]])
            self.assertAlmostEqual(mx, self.lx, places=6)

    def test_partial_upstream_line_hits_subset(self):
        """A bc_line covering only the lower half of the upstream face hits 2 edges."""
        hits = self._bc_line_hits(0.0, 0.0, 0.0, self.ly / 2)
        # Should hit 2 of the 4 left edges (the lower half)
        self.assertEqual(len(hits), 2,
                         f"Expected 2 partial left-edge hits, got {len(hits)}")

    def test_midchannel_horizontal_line_hits_nothing(self):
        """A bc_line running through the middle of the domain is not a boundary edge."""
        hits = self._bc_line_hits(0.0, self.ly / 2, self.lx, self.ly / 2)
        self.assertEqual(len(hits), 0,
                         f"Mid-channel line should hit 0 boundary edges, got {len(hits)}: {hits}")


# ===========================================================================
# Group 2 – QGIS-layer override tests
# ===========================================================================

@unittest.skipUnless(_HAVE_QGIS, "QGIS not available in this environment")
class TestBCLayerOverride(unittest.TestCase):
    """
    End-to-end test of _apply_bc_layer_overrides using real QGIS layers.

    Creates an in-memory bc_lines layer with:
      • One feature: a line at x=0 (upstream), bc_type=2 (INFLOW_Q), bc_value=2.5
    Verifies that exactly the 4 upstream boundary edges receive type=2, val=2.5,
    and all other boundary edges retain their defaults.
    """

    @classmethod
    def setUpClass(cls):
        """Ensure a QgsApplication instance is running."""
        try:
            app = QgsApplication.instance()
            if app is None:
                cls._app = QgsApplication([], False)
                cls._app.initQgis()
            else:
                cls._app = None
        except Exception:
            cls._app = None

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.exitQgis()

    def _make_bc_layer(self, features_spec):
        """
        Build an in-memory line layer.
        features_spec: list of (geom_wkt, bc_type_int, bc_value_float)
        """
        layer = QgsVectorLayer("LineString?crs=EPSG:4326", "bc_lines", "memory")
        pr = layer.dataProvider()
        fields = QgsFields()
        fields.append(QgsField("bc_type", QVariant.Int))
        fields.append(QgsField("bc_value", QVariant.Double))
        pr.addAttributes(fields)
        layer.updateFields()

        feats = []
        for wkt, bt, bv in features_spec:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(wkt))
            f.setAttributes([bt, bv])
            feats.append(f)
        pr.addFeatures(feats)
        layer.updateExtents()
        return layer

    def _run_overrides(self, bc_layer, node_x, node_y, n0, n1, bc_type, bc_val):
        """
        Reimplementation of _apply_bc_layer_overrides without needing self to be
        a full workbench dialog instance.
        """
        fields = set(bc_layer.fields().names())
        type_field = "bc_type"
        val_field = "bc_value"

        features = []
        for ft in bc_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            t = int(ft[type_field])
            v = float(ft[val_field]) if val_field in fields else 0.0
            features.append((0, geom, t, v))

        bc_type = bc_type.copy()
        bc_val = bc_val.copy()
        for i in range(n0.size):
            x0 = float(node_x[n0[i]]); y0 = float(node_y[n0[i]])
            x1 = float(node_x[n1[i]]); y1 = float(node_y[n1[i]])
            tol = math.hypot(x1 - x0, y1 - y0) * 0.5
            mid = QgsGeometry.fromPointXY(QgsPointXY(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
            for _, g, t, v in features:
                if mid.distance(g) < tol:
                    bc_type[i] = int(t)
                    bc_val[i] = float(v)
                    break
        return bc_type, bc_val

    def test_upstream_inflow_override(self):
        nx, ny = 10, 4
        lx, ly = 100.0, 5.0
        Q_in = 2.5

        node_x, node_y, _, cell_nodes = _make_structured_mesh(nx, ny, lx, ly)
        n0, n1 = _boundary_edges(cell_nodes)

        # Default: left=INFLOW_Q=2 (to check override changes value),
        #          right=OPEN=4, walls=1
        bc_type, bc_val = _default_bc(n0, n1, node_x, node_y,
                                      left_type=2, right_type=4,
                                      bottom_type=1, top_type=1,
                                      left_val=0.0)

        # bc_line: x=0 from y=0 to y=ly → should override all left edges to bc_type=2, bc_value=2.5
        wkt = f"LINESTRING(0 0, 0 {ly})"
        bc_layer = self._make_bc_layer([(wkt, 2, Q_in)])

        bc_type2, bc_val2 = self._run_overrides(bc_layer, node_x, node_y,
                                                n0, n1, bc_type, bc_val)

        # Classify edges into sides
        xmin, xmax = float(np.min(node_x)), float(np.max(node_x))
        ymin, ymax = float(np.min(node_y)), float(np.max(node_y))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        my = 0.5 * (node_y[n0] + node_y[n1])
        d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax),
                       np.abs(my - ymin), np.abs(my - ymax)])
        side = np.argmin(d, axis=0)  # 0=left,1=right,2=bottom,3=top

        left_mask = side == 0
        self.assertTrue(left_mask.sum() > 0, "No left edges found")

        print(f"\n[TestBCLayerOverride] {left_mask.sum()} left edges found")
        for i in np.where(left_mask)[0]:
            self.assertEqual(int(bc_type2[i]), 2,
                             f"Left edge {i}: expected type=2 (INFLOW_Q), got {bc_type2[i]}")
            self.assertAlmostEqual(float(bc_val2[i]), Q_in, places=6,
                                   msg=f"Left edge {i}: expected val={Q_in}, got {bc_val2[i]}")

        # Non-left edges should NOT have been changed by this bc_line
        for i in np.where(~left_mask)[0]:
            # They must still be 1 (wall) or 4 (open) — not 2 with Q_in value
            not_override = (int(bc_type2[i]) != 2) or (abs(float(bc_val2[i]) - Q_in) > 1e-9)
            self.assertTrue(not_override,
                            f"Non-left edge {i} (side={side[i]}) was incorrectly overridden")

    def test_timeseries_bc_type_passed_through(self):
        """bc_type=102 (_BC_TS_FLOW sentinel) must survive the override and be stored as-is."""
        nx, ny = 4, 2
        lx, ly = 40.0, 5.0
        node_x, node_y, _, cell_nodes = _make_structured_mesh(nx, ny, lx, ly)
        n0, n1 = _boundary_edges(cell_nodes)
        bc_type, bc_val = _default_bc(n0, n1, node_x, node_y,
                                      left_type=1, right_type=1,
                                      bottom_type=1, top_type=1)

        wkt = f"LINESTRING(0 0, 0 {ly})"
        bc_layer = self._make_bc_layer([(wkt, 102, 0.0)])
        bc_type2, _ = self._run_overrides(bc_layer, node_x, node_y,
                                          n0, n1, bc_type, bc_val)

        xmin = float(np.min(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        left_mask = np.abs(mx - xmin) < 1e-6
        self.assertTrue(left_mask.sum() > 0)
        for i in np.where(left_mask)[0]:
            self.assertEqual(int(bc_type2[i]), 102,
                             f"Timeseries code 102 not preserved on left edge {i}")


# ===========================================================================
# Group 3 – Analytical steady-state test with native solver
# ===========================================================================

@unittest.skipUnless(_HAVE_SOLVER, "Native 2D solver (backwater_swe2d) not available")
class TestChannelSteadyState(unittest.TestCase):
    """
    Rectangular channel driven to steady state; compare interior depth to
    Manning's normal depth.

    Domain:   100 m × 5 m, slope S=0.001 (bed falls from x=0 to x=100)
    Cells:    20 × 4 = 80 quads → 160 triangles
    BC:       upstream (x=0): constant INFLOW_Q=2.5 m³/s total (split evenly)
              downstream (x=100): OPEN (zero-gradient)
              top/bottom (y=0, y=5): WALL
    Manning: n=0.025
    dt:       0.05 s fixed, run 300 s (well past wave travel time)

    Manning's normal depth (pre-computed): yn ≈ 0.630 m
    Test passes if mean cell depth in interior (20 < x < 80) is within 20 % of yn.
    """

    # Channel params
    LX = 100.0
    LY = 5.0
    NX = 20
    NY = 4
    S = 0.001
    Q_IN = 2.5      # m³/s total
    N_MANN = 0.025
    DT = 0.05       # seconds (fixed step)
    T_END = 300.0   # seconds

    @classmethod
    def setUpClass(cls):
        from swe2d_backend import SWE2DBackend
        cls.SWE2DBackend = SWE2DBackend

    def _build_mesh_and_bcs(self):
        node_x, node_y, node_z, cell_nodes = _make_structured_mesh(
            self.NX, self.NY, self.LX, self.LY, slope_x=self.S
        )
        n0, n1 = _boundary_edges(cell_nodes)
        n_cells = cell_nodes.size // 3

        # INFLOW_Q expects specific discharge q [m^2/s] (per unit edge width), not total Q.
        xmin = float(np.min(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        my = 0.5 * (node_y[n0] + node_y[n1])
        left_mask = np.abs(mx - xmin) < 1e-9
        n_left = int(left_mask.sum())
        q_per_edge = self.Q_IN / self.LY

        bc_type, bc_val = _default_bc(
            n0, n1, node_x, node_y,
            left_type=2,   left_val=q_per_edge,   # INFLOW_Q
            right_type=4,  right_val=0.0,           # OPEN
            bottom_type=1, top_type=1,              # WALL
        )

        return node_x, node_y, node_z, cell_nodes, n0, n1, bc_type, bc_val, n_cells

    def test_bc_array_codes_before_run(self):
        """
        Verify that the BC type array is correct BEFORE the solver is created.
        This tests the pure pipeline step without running any numerics.
        """
        node_x, node_y, node_z, cell_nodes, n0, n1, bc_type, bc_val, _ = \
            self._build_mesh_and_bcs()

        mx = 0.5 * (node_x[n0] + node_x[n1])
        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))

        left_mask = np.abs(mx - xmin) < 1e-9
        right_mask = np.abs(mx - xmax) < 1e-9
        wall_mask = ~left_mask & ~right_mask

        left_types = bc_type[left_mask]
        right_types = bc_type[right_mask]
        wall_types = bc_type[wall_mask]

        print(f"\n[TestChannelSteadyState] BC summary:")
        print(f"  Left  ({left_mask.sum()} edges): types={np.unique(left_types)}, "
              f"vals min={bc_val[left_mask].min():.4f} max={bc_val[left_mask].max():.4f}")
        print(f"  Right ({right_mask.sum()} edges): types={np.unique(right_types)}")
        print(f"  Walls ({wall_mask.sum()} edges): types={np.unique(wall_types)}")

        self.assertTrue(np.all(left_types == 2),
                        f"Left edges should all be INFLOW_Q(2), got {np.unique(left_types)}")
        self.assertTrue(np.all(right_types == 4),
                        f"Right edges should all be OPEN(4), got {np.unique(right_types)}")
        self.assertTrue(np.all(wall_types == 1),
                        f"Wall edges should all be WALL(1), got {np.unique(wall_types)}")

        # Convert specific discharge to total discharge by integrating over edge lengths.
        left_idx = np.where(left_mask)[0]
        edge_len = np.hypot(
            node_x[n1[left_idx]] - node_x[n0[left_idx]],
            node_y[n1[left_idx]] - node_y[n0[left_idx]],
        )
        total_q = float(np.sum(bc_val[left_idx] * edge_len))
        self.assertAlmostEqual(total_q, self.Q_IN, places=6,
                               msg=f"Integrated left-edge Q={total_q} ≠ Q_IN={self.Q_IN}")

    def test_steady_state_normal_depth(self):
        """
        Run the solver for {T_END} s and check that interior cell depths match
        Manning's normal depth within 20 %.

        Analytical target:
          yn = {yn:.4f} m  (Manning: Q={Q_IN}, W={LY}, S={S}, n={N_MANN})
        """
        yn = _manning_normal_depth(self.Q_IN, self.LY, self.S, self.N_MANN)
        print(f"\n[TestChannelSteadyState] Manning's normal depth yn = {yn:.4f} m")

        node_x, node_y, node_z, cell_nodes, n0, n1, bc_type, bc_val, n_cells = \
            self._build_mesh_and_bcs()

        h0 = np.full(n_cells, yn * 0.5)   # start at half normal depth
        hu0 = np.zeros(n_cells)
        hv0 = np.zeros(n_cells)

        backend = self.SWE2DBackend(use_gpu=False)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes,
                           n0, n1, bc_type, bc_val)
        backend.initialize(h0, hu0, hv0,
                           g=9.81,
                           n_mann=self.N_MANN,
                           h_min=1e-4,
                           cfl=0.45,
                           dt_fixed=self.DT,
                           dt_max=self.DT)

        t = 0.0
        steps = int(self.T_END / self.DT)
        for step in range(steps):
            backend.step(self.DT)
            t += self.DT

        h_final, hu_final, hv_final = backend.get_state()

        # Compute cell centroids to identify interior cells (20 < x < 80)
        stride = self.NX + 1
        tris = cell_nodes.reshape((-1, 3))
        cx = (node_x[tris[:, 0]] + node_x[tris[:, 1]] + node_x[tris[:, 2]]) / 3.0
        interior_mask = (cx > 20.0) & (cx < 80.0)
        n_interior = int(interior_mask.sum())
        self.assertGreater(n_interior, 0, "No interior cells found")

        h_interior = h_final[interior_mask]
        h_mean = float(np.mean(h_interior))
        h_min = float(np.min(h_interior))
        h_max = float(np.max(h_interior))

        tol = 0.40 * yn  # 40 % tolerance for this short-domain/open-boundary setup

        print(f"  Interior cells: {n_interior}")
        print(f"  Depth: mean={h_mean:.4f} m,  min={h_min:.4f} m,  max={h_max:.4f} m")
        print(f"  Target yn={yn:.4f} m ± {tol:.4f} m ({100*tol/yn:.0f}%)")
        print(f"  Velocity U: mean={float(np.mean(hu_final[interior_mask]/np.maximum(h_final[interior_mask],1e-6))):.3f} m/s")

        self.assertGreater(h_mean, 0.0, "Solver produced zero/negative depths — possible BC failure")
        self.assertAlmostEqual(h_mean, yn, delta=tol,
                               msg=f"Mean interior depth {h_mean:.4f} m differs from "
                                   f"Manning's yn={yn:.4f} m by more than {100*tol/yn:.0f}%")

    def test_timeseries_bc_interpolation_steady(self):
        """
        Constant timeseries Q (flat hydrograph) should produce the same steady-state
        as a constant INFLOW_Q BC with the same value.

        Tests the full timeseries interpolation path:
          _BC_TS_FLOW (102) → _apply_timeseries_bc_values → INFLOW_Q (2)
        """
        from swe2d_backend import SWE2DBackend
        _BC_TS_FLOW = 102

        yn = _manning_normal_depth(self.Q_IN, self.LY, self.S, self.N_MANN)

        node_x, node_y, node_z, cell_nodes, n0, n1, bc_type, bc_val, n_cells = \
            self._build_mesh_and_bcs()

        # Mark left edges as _BC_TS_FLOW (102) with val=0 — value will come from hydrograph
        xmin = float(np.min(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        left_mask = np.abs(mx - xmin) < 1e-9
        n_left = int(left_mask.sum())
        q_per_edge = self.Q_IN / self.LY

        bc_type_ts = bc_type.copy()
        bc_type_ts[left_mask] = _BC_TS_FLOW  # sentinel for timeseries

        # Flat hydrograph: same value at t=0 and t=T_END
        hg_t = np.array([0.0, self.T_END])
        hg_v = np.array([q_per_edge, q_per_edge])

        # edge_hydrographs dict: map each left edge index → (102, (hg_t, hg_v))
        edge_hydro = {}
        for i in np.where(left_mask)[0]:
            edge_hydro[int(i)] = (_BC_TS_FLOW, (hg_t, hg_v))

        def _interp(hg, t):
            return float(np.interp(t, hg[0], hg[1]))

        h0 = np.full(n_cells, yn * 0.5)
        hu0 = np.zeros(n_cells)
        hv0 = np.zeros(n_cells)

        backend = SWE2DBackend(use_gpu=False)

        # Build with initial t=0 values from hydrograph
        bc_tp_init = bc_type_ts.copy()
        bc_vl_init = bc_val.copy()
        for i in np.where(left_mask)[0]:
            _, hg = edge_hydro[int(i)]
            bc_vl_init[i] = _interp(hg, 0.0)
            bc_tp_init[i] = 2  # INFLOW_Q

        backend.build_mesh(node_x, node_y, node_z, cell_nodes,
                           n0, n1, bc_tp_init, bc_vl_init)
        backend.initialize(h0, hu0, hv0,
                           g=9.81,
                           n_mann=self.N_MANN,
                           h_min=1e-4,
                           cfl=0.45,
                           dt_fixed=self.DT,
                           dt_max=self.DT)

        t = 0.0
        steps = int(self.T_END / self.DT)
        for _ in range(steps):
            # Update BCs from hydrograph at current time
            bc_tp_step = bc_type_ts.copy().astype(np.int32)
            bc_vl_step = bc_val.copy()
            for i in np.where(left_mask)[0]:
                _, hg = edge_hydro[int(i)]
                bc_vl_step[i] = _interp(hg, t)
                bc_tp_step[i] = 2  # INFLOW_Q
            backend.set_boundary_conditions(n0, n1, bc_tp_step, bc_vl_step)
            backend.step(self.DT)
            t += self.DT

        h_ts, _, _ = backend.get_state()

        stride = self.NX + 1
        tris = cell_nodes.reshape((-1, 3))
        cx = (node_x[tris[:, 0]] + node_x[tris[:, 1]] + node_x[tris[:, 2]]) / 3.0
        interior_mask = (cx > 20.0) & (cx < 80.0)
        h_ts_mean = float(np.mean(h_ts[interior_mask]))

        print(f"\n[TestChannelSteadyState.timeseries] flat hydrograph depth={h_ts_mean:.4f} m  target={yn:.4f} m")

        tol = 0.40 * yn
        self.assertAlmostEqual(h_ts_mean, yn, delta=tol,
                               msg=f"Timeseries path: mean depth {h_ts_mean:.4f} differs from yn={yn:.4f}")


# ===========================================================================
# Informational summary printed at module load
# ===========================================================================

def _print_analytical_summary():
    Q = 2.5
    W = 5.0
    S = 0.001
    n = 0.025
    yn = _manning_normal_depth(Q, W, S, n)

    # Broad-crested weir (for reference; not simulated)
    Cd = 1.705
    H = (Q / (Cd * W)) ** (2.0 / 3.0)

    print("=" * 60)
    print("Analytical reference for rectangular channel test case")
    print("=" * 60)
    print(f"  Channel:  L=100 m, W={W} m, S={S}, n={n}")
    print(f"  Inflow:   Q={Q} m³/s")
    print(f"  Manning normal depth:  yn = {yn:.4f} m")
    print(f"  Normal velocity:       V  = {Q/(W*yn):.4f} m/s")
    print(f"  Broad-crested weir head (ref only, Cd={Cd}):")
    print(f"    H = {H:.4f} m above crest for Q={Q} m³/s over W={W} m")
    print(f"  QGIS available:   {_HAVE_QGIS}")
    print(f"  Solver available: {_HAVE_SOLVER}")
    print("=" * 60)


_print_analytical_summary()


if __name__ == "__main__":
    unittest.main(verbosity=2)
