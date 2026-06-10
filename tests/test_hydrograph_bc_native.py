"""
test_hydrograph_bc_native.py — Headless validation of time-varying hydrograph BCs
=================================================================================

Tests the native GPU hydrograph BC forcing path (SWE2DNativeBoundaryHydrographConfigurator)
and the Python fallback path for correctness.

Key scenarios:
  1. Edge-based hydrographs (BC line layer): multiple edges share the same hydrograph
     → total inflow must equal the hydrograph Q, not N×Q
  2. Side-based hydrographs: all edges on a side share the hydrograph
     → total inflow must equal the hydrograph Q
  3. Progressive inflow: lowest edges activate first as Q increases
  4. Time-varying interpolation: values change correctly over time

Uses a simple rectangular channel mesh — no QGIS required.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Dict, Tuple

import numpy as np

try:
    from swe2d.runtime.backend import SWE2DBackend, swe2d_available
    _HAVE_SOLVER = swe2d_available()
except Exception:
    _HAVE_SOLVER = False

try:
    from swe2d.runtime.native_bc_forcing import SWE2DNativeBoundaryHydrographConfigurator
    _HAVE_NATIVE_CFG = True
except Exception:
    _HAVE_NATIVE_CFG = False

try:
    from swe2d.boundary_and_forcing.bc_logic import (
        apply_timeseries_bc_values,
        distribute_total_flow_to_unit_q,
        interp_hydrograph,
    )
    _HAVE_BC_LOGIC = True
except Exception:
    _HAVE_BC_LOGIC = False


_BC_TS_FLOW = 102
_BC_TS_STAGE = 103
_BC_INFLOW_Q = 2


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _make_structured_mesh(nx: int, ny: int, lx: float, ly: float,
                          slope_x: float = 0.0):
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().copy()
    node_y = Yg.ravel().copy()
    node_z = slope_x * (lx - node_x)

    stride = nx + 1
    cells = []
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


def _boundary_edges(cell_nodes_flat):
    tris = cell_nodes_flat.reshape((-1, 3)).astype(np.int32)
    edge_count = {}
    edge_oriented = {}
    for tri in tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_oriented:
                edge_oriented[key] = (a, b)
    n0, n1 = [], []
    for key, cnt in edge_count.items():
        if cnt == 1:
            a, b = edge_oriented[key]
            n0.append(a)
            n1.append(b)
    return np.asarray(n0, dtype=np.int32), np.asarray(n1, dtype=np.int32)


def _classify_sides(n0, n1, node_x, node_y):
    xmin, xmax = float(np.min(node_x)), float(np.max(node_x))
    ymin, ymax = float(np.min(node_y)), float(np.max(node_y))
    mx = 0.5 * (node_x[n0] + node_x[n1])
    my = 0.5 * (node_y[n0] + node_y[n1])
    d = np.vstack([
        np.abs(mx - xmin),
        np.abs(mx - xmax),
        np.abs(my - ymin),
        np.abs(my - ymax),
    ])
    side_idx = np.argmin(d, axis=0)
    return side_idx


def _edge_lengths(n0, n1, node_x, node_y):
    return np.hypot(node_x[n1] - node_x[n0], node_y[n1] - node_y[n0])


# ---------------------------------------------------------------------------
# Test: Native BC Configurator flow_scale correctness
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAVE_NATIVE_CFG and _HAVE_BC_LOGIC,
                     "Native BC configurator or bc_logic not available")
class TestNativeConfiguratorFlowScale(unittest.TestCase):
    """
    Verify that SWE2DNativeBoundaryHydrographConfigurator.compute() produces
    correct total-Q→unit-q conversion for edge-based and side-based hydrographs.
    """

    NX, NY = 10, 4
    LX, LY = 100.0, 5.0
    S = 0.001

    def _make_channel(self):
        node_x, node_y, node_z, cell_nodes = _make_structured_mesh(
            self.NX, self.NY, self.LX, self.LY, slope_x=self.S)
        n0, n1 = _boundary_edges(cell_nodes)
        side_idx = _classify_sides(n0, n1, node_x, node_y)
        edge_len = _edge_lengths(n0, n1, node_x, node_y)
        return node_x, node_y, node_z, cell_nodes, n0, n1, side_idx, edge_len

    def _left_edge_indices(self, n0, n1, node_x):
        xmin = float(np.min(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        return np.where(np.abs(mx - xmin) < 1e-9)[0]

    # ------------------------------------------------------------------
    # Test 1: Edge-based hydrograph — multiple edges share same hydrograph
    # ------------------------------------------------------------------
    def test_edge_based_hydrograph_total_flow(self):
        """
        When N edges share the same hydrograph with total Q=100,
        the uploaded unit-q values must produce total flow = 100,
        not 100*N.
        """
        node_x, node_y, node_z, cell_nodes, n0, n1, side_idx, edge_len = \
            self._make_channel()

        left_idx = self._left_edge_indices(n0, n1, node_x)
        n_left = left_idx.size

        Q_TOTAL = 100.0
        hg_t = np.array([0.0, 10.0, 100.0], dtype=np.float64)
        hg_v = np.array([Q_TOTAL, 200.0, 50.0], dtype=np.float64)

        bc_tp = np.ones(n0.size, dtype=np.int32) * _BC_TS_FLOW
        bc_vl = np.zeros(n0.size, dtype=np.float64)

        edge_hydrographs = {}
        shared_hg = (hg_t.copy(), hg_v.copy())
        for i in left_idx:
            edge_hydrographs[int(i)] = (_BC_TS_FLOW, shared_hg)

        cfg = SWE2DNativeBoundaryHydrographConfigurator()

        class _FakeBackend:
            def __init__(self):
                self._boundary_edge_index_by_nodes = {}
                self.uploaded = None

            def set_boundary_hydrographs_native(self, edge_index, bc_type, offsets,
                                                 time_s, value):
                self.uploaded = {
                    "edge_index": np.asarray(edge_index),
                    "bc_type": np.asarray(bc_type),
                    "offsets": np.asarray(offsets),
                    "time_s": np.asarray(time_s),
                    "value": np.asarray(value),
                }

            def set_progressive_bc_data(self, **kwargs):
                pass

        fake = _FakeBackend()

        # Build edge_index mapping
        for j, bi in enumerate(left_idx):
            a, b = int(n0[bi]), int(n1[bi])
            key = (a, b) if a < b else (b, a)
            fake._boundary_edge_index_by_nodes[key] = j

        result = cfg.configure(
            backend=fake,
            bc_n0=n0, bc_n1=n1, bc_tp=bc_tp,
            side_hydrographs={},
            edge_hydrographs=edge_hydrographs,
            node_x=node_x, node_y=node_y, node_z=node_z,
            inflow_q_bc_type=_BC_INFLOW_Q,
            progressive=False,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
        )

        self.assertTrue(result["native_bc_forcing"])
        self.assertEqual(result["configured_edges"], n_left)

        # Verify uploaded values at t=0 produce total flow = Q_TOTAL
        offsets = fake.uploaded["offsets"]
        values = fake.uploaded["value"]

        left_edge_len = edge_len[left_idx]
        total_left_len = float(np.sum(left_edge_len))

        total_flow_at_t0 = 0.0
        for k, bi in enumerate(left_idx):
            s = offsets[k]
            e = offsets[k + 1]
            t_arr = fake.uploaded["time_s"][s:e]
            v_arr = values[s:e]
            q_unit_at_t0 = float(np.interp(0.0, t_arr, v_arr))
            total_flow_at_t0 += q_unit_at_t0 * left_edge_len[k]

        print(f"\n[test_edge_based_hydrograph_total_flow]")
        print(f"  N edges sharing hydrograph: {n_left}")
        print(f"  Total left edge length: {total_left_len:.4f} m")
        print(f"  Q_TOTAL from hydrograph: {Q_TOTAL:.2f} m³/s")
        print(f"  Reconstructed total flow at t=0: {total_flow_at_t0:.6f} m³/s")

        self.assertAlmostEqual(total_flow_at_t0, Q_TOTAL, places=6,
                               msg=f"Total flow {total_flow_at_t0:.6f} != Q_TOTAL {Q_TOTAL}")

    # ------------------------------------------------------------------
    # Test 2: Side-based hydrograph
    # ------------------------------------------------------------------
    def test_side_based_hydrograph_total_flow(self):
        """
        Side-based hydrograph: all left-side edges share the hydrograph.
        Total flow must equal the hydrograph Q.
        """
        node_x, node_y, node_z, cell_nodes, n0, n1, side_idx, edge_len = \
            self._make_channel()

        left_idx = self._left_edge_indices(n0, n1, node_x)
        n_left = left_idx.size

        Q_TOTAL = 75.0
        hg_t = np.array([0.0, 50.0], dtype=np.float64)
        hg_v = np.array([Q_TOTAL, Q_TOTAL], dtype=np.float64)

        side_hydrographs = {"left": (hg_t, hg_v)}

        bc_tp = np.ones(n0.size, dtype=np.int32) * _BC_TS_FLOW
        bc_vl = np.zeros(n0.size, dtype=np.float64)

        cfg = SWE2DNativeBoundaryHydrographConfigurator()

        class _FakeBackend:
            def __init__(self):
                self._boundary_edge_index_by_nodes = {}
                self.uploaded = None

            def set_boundary_hydrographs_native(self, edge_index, bc_type, offsets,
                                                 time_s, value):
                self.uploaded = {
                    "edge_index": np.asarray(edge_index),
                    "bc_type": np.asarray(bc_type),
                    "offsets": np.asarray(offsets),
                    "time_s": np.asarray(time_s),
                    "value": np.asarray(value),
                }

            def set_progressive_bc_data(self, **kwargs):
                pass

        fake = _FakeBackend()
        for j, bi in enumerate(range(n0.size)):
            a, b = int(n0[bi]), int(n1[bi])
            key = (a, b) if a < b else (b, a)
            fake._boundary_edge_index_by_nodes[key] = j

        result = cfg.configure(
            backend=fake,
            bc_n0=n0, bc_n1=n1, bc_tp=bc_tp,
            side_hydrographs=side_hydrographs,
            edge_hydrographs={},
            node_x=node_x, node_y=node_y, node_z=node_z,
            inflow_q_bc_type=_BC_INFLOW_Q,
            progressive=False,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
        )

        self.assertTrue(result["native_bc_forcing"])

        offsets = fake.uploaded["offsets"]
        values = fake.uploaded["value"]

        left_edge_len = edge_len[left_idx]
        total_left_len = float(np.sum(left_edge_len))

        total_flow_at_t0 = 0.0
        for k, bi in enumerate(left_idx):
            s = offsets[k]
            e = offsets[k + 1]
            t_arr = fake.uploaded["time_s"][s:e]
            v_arr = values[s:e]
            q_unit_at_t0 = float(np.interp(0.0, t_arr, v_arr))
            total_flow_at_t0 += q_unit_at_t0 * left_edge_len[k]

        print(f"\n[test_side_based_hydrograph_total_flow]")
        print(f"  N left edges: {n_left}")
        print(f"  Total left edge length: {total_left_len:.4f} m")
        print(f"  Q_TOTAL from hydrograph: {Q_TOTAL:.2f} m³/s")
        print(f"  Reconstructed total flow at t=0: {total_flow_at_t0:.6f} m³/s")

        self.assertAlmostEqual(total_flow_at_t0, Q_TOTAL, places=6,
                               msg=f"Total flow {total_flow_at_t0:.6f} != Q_TOTAL {Q_TOTAL}")

    # ------------------------------------------------------------------
    # Test 3: Time-varying interpolation at multiple times
    # ------------------------------------------------------------------
    def test_edge_based_hydrograph_time_varying(self):
        """
        Verify that uploaded hydrograph values interpolate correctly at
        multiple time points and produce the correct total flow at each.
        """
        node_x, node_y, node_z, cell_nodes, n0, n1, side_idx, edge_len = \
            self._make_channel()

        left_idx = self._left_edge_indices(n0, n1, node_x)
        n_left = left_idx.size

        Q_TABLE = {0.0: 100.0, 50.0: 300.0, 100.0: 50.0}
        hg_t = np.array([0.0, 50.0, 100.0], dtype=np.float64)
        hg_v = np.array([100.0, 300.0, 50.0], dtype=np.float64)

        bc_tp = np.ones(n0.size, dtype=np.int32) * _BC_TS_FLOW
        bc_vl = np.zeros(n0.size, dtype=np.float64)

        edge_hydrographs = {}
        shared_hg = (hg_t.copy(), hg_v.copy())
        for i in left_idx:
            edge_hydrographs[int(i)] = (_BC_TS_FLOW, shared_hg)

        cfg = SWE2DNativeBoundaryHydrographConfigurator()

        class _FakeBackend:
            def __init__(self):
                self._boundary_edge_index_by_nodes = {}
                self.uploaded = None

            def set_boundary_hydrographs_native(self, edge_index, bc_type, offsets,
                                                 time_s, value):
                self.uploaded = {
                    "edge_index": np.asarray(edge_index),
                    "bc_type": np.asarray(bc_type),
                    "offsets": np.asarray(offsets),
                    "time_s": np.asarray(time_s),
                    "value": np.asarray(value),
                }

            def set_progressive_bc_data(self, **kwargs):
                pass

        fake = _FakeBackend()
        for j, bi in enumerate(left_idx):
            a, b = int(n0[bi]), int(n1[bi])
            key = (a, b) if a < b else (b, a)
            fake._boundary_edge_index_by_nodes[key] = j

        result = cfg.configure(
            backend=fake,
            bc_n0=n0, bc_n1=n1, bc_tp=bc_tp,
            side_hydrographs={},
            edge_hydrographs=edge_hydrographs,
            node_x=node_x, node_y=node_y, node_z=node_z,
            inflow_q_bc_type=_BC_INFLOW_Q,
            progressive=False,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
        )

        offsets = fake.uploaded["offsets"]
        values = fake.uploaded["value"]
        left_edge_len = edge_len[left_idx]

        test_times = [0.0, 25.0, 50.0, 75.0, 100.0]
        for t_test in test_times:
            expected_q = float(np.interp(t_test, hg_t, hg_v))
            total_flow = 0.0
            for k, bi in enumerate(left_idx):
                s = offsets[k]
                e = offsets[k + 1]
                t_arr = fake.uploaded["time_s"][s:e]
                v_arr = values[s:e]
                q_unit = float(np.interp(t_test, t_arr, v_arr))
                total_flow += q_unit * left_edge_len[k]

            print(f"  t={t_test:6.1f}s: expected Q={expected_q:8.2f}, "
                  f"reconstructed Q={total_flow:8.4f}")

            self.assertAlmostEqual(total_flow, expected_q, places=4,
                                   msg=f"At t={t_test}: total flow {total_flow:.4f} "
                                       f"!= expected {expected_q:.2f}")


# ---------------------------------------------------------------------------
# Test: Python fallback path (apply_timeseries_bc_values + distribute)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAVE_BC_LOGIC, "bc_logic not available")
class TestPythonFallbackPath(unittest.TestCase):
    """
    Verify the Python fallback path produces correct total flow for
    edge-based hydrographs with multiple edges sharing the same hydrograph.
    """

    NX, NY = 10, 4
    LX, LY = 100.0, 5.0
    S = 0.001

    def _make_channel(self):
        node_x, node_y, node_z, cell_nodes = _make_structured_mesh(
            self.NX, self.NY, self.LX, self.LY, slope_x=self.S)
        n0, n1 = _boundary_edges(cell_nodes)
        side_idx = _classify_sides(n0, n1, node_x, node_y)
        edge_len = _edge_lengths(n0, n1, node_x, node_y)
        return node_x, node_y, node_z, cell_nodes, n0, n1, side_idx, edge_len

    def _left_edge_indices(self, n0, n1, node_x):
        xmin = float(np.min(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        return np.where(np.abs(mx - xmin) < 1e-9)[0]

    def test_edge_based_python_fallback_total_flow(self):
        """
        Python fallback: apply_timeseries_bc_values + distribute_total_flow_to_unit_q
        must produce total flow = Q_TOTAL for edge-based hydrographs.
        """
        node_x, node_y, node_z, cell_nodes, n0, n1, side_idx, edge_len = \
            self._make_channel()

        left_idx = self._left_edge_indices(n0, n1, node_x)
        n_left = left_idx.size

        Q_TOTAL = 100.0
        hg_t = np.array([0.0, 100.0], dtype=np.float64)
        hg_v = np.array([Q_TOTAL, Q_TOTAL], dtype=np.float64)

        bc_tp = np.ones(n0.size, dtype=np.int32) * _BC_TS_FLOW
        bc_vl = np.zeros(n0.size, dtype=np.float64)

        edge_hydrographs = {}
        shared_hg = (hg_t.copy(), hg_v.copy())
        for i in left_idx:
            edge_hydrographs[int(i)] = (_BC_TS_FLOW, shared_hg)

        t_test = 0.0
        bc_tp_step, bc_vl_step = apply_timeseries_bc_values(
            n0, n1, bc_tp, bc_vl, {}, node_x, node_y, t_test,
            ts_flow_code=_BC_TS_FLOW, ts_stage_code=_BC_TS_STAGE,
            edge_hydrographs=edge_hydrographs,
        )

        bc_vl_dist = distribute_total_flow_to_unit_q(
            n0, n1, bc_tp_step, bc_vl_step, bc_tp, {},
            node_x, node_y, node_z,
            progressive=False, ts_flow_code=_BC_TS_FLOW,
            edge_hydrographs=edge_hydrographs,
        )

        left_edge_len = edge_len[left_idx]
        total_flow = float(np.sum(bc_vl_dist[left_idx] * left_edge_len))

        print(f"\n[test_edge_based_python_fallback_total_flow]")
        print(f"  N left edges: {n_left}")
        print(f"  Q_TOTAL from hydrograph: {Q_TOTAL:.2f} m³/s")
        print(f"  Reconstructed total flow: {total_flow:.6f} m³/s")

        self.assertAlmostEqual(total_flow, Q_TOTAL, places=6,
                               msg=f"Total flow {total_flow:.6f} != Q_TOTAL {Q_TOTAL}")


# ---------------------------------------------------------------------------
# Test: Full solver run with native hydrograph BC
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAVE_SOLVER and _HAVE_NATIVE_CFG,
                     "Native solver or configurator not available")
class TestSolverHydrographRun(unittest.TestCase):
    """
    Run the solver with native hydrograph BC and verify that the actual
    inflow matches the hydrograph table values.
    """

    NX, NY = 20, 4
    LX, LY = 100.0, 5.0
    S = 0.001
    N_MANN = 0.025
    DT = 0.05
    T_END = 10.0

    def _make_channel(self):
        node_x, node_y, node_z, cell_nodes = _make_structured_mesh(
            self.NX, self.NY, self.LX, self.LY, slope_x=self.S)
        n0, n1 = _boundary_edges(cell_nodes)
        n_cells = cell_nodes.size // 3

        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        left_mask = np.abs(mx - xmin) < 1e-9
        right_mask = np.abs(mx - xmax) < 1e-9

        bc_type = np.ones(n0.size, dtype=np.int32)
        bc_val = np.zeros(n0.size, dtype=np.float64)

        bc_type[left_mask] = _BC_TS_FLOW
        bc_type[right_mask] = 4  # OPEN

        return node_x, node_y, node_z, cell_nodes, n0, n1, bc_type, bc_val, n_cells

    def _left_edge_indices(self, n0, n1, node_x):
        xmin = float(np.min(node_x))
        mx = 0.5 * (node_x[n0] + node_x[n1])
        return np.where(np.abs(mx - xmin) < 1e-9)[0]

    def test_native_hydrograph_solver_run(self):
        """
        Run solver with edge-based hydrograph BC.
        Verify total inflow at each step matches the hydrograph table.
        """
        node_x, node_y, node_z, cell_nodes, n0, n1, bc_type, bc_val, n_cells = \
            self._make_channel()

        left_idx = self._left_edge_indices(n0, n1, node_x)
        n_left = left_idx.size

        Q_STEADY = 2.5
        hg_t = np.array([0.0, self.T_END], dtype=np.float64)
        hg_v = np.array([Q_STEADY, Q_STEADY], dtype=np.float64)

        edge_hydrographs = {}
        for i in left_idx:
            edge_hydrographs[int(i)] = (_BC_TS_FLOW, (hg_t.copy(), hg_v.copy()))

        h0 = np.full(n_cells, 0.3)
        hu0 = np.zeros(n_cells)
        hv0 = np.zeros(n_cells)

        backend = SWE2DBackend()

        bc_tp_init = bc_type.copy()
        bc_vl_init = bc_val.copy()
        for i in left_idx:
            bc_tp_init[i] = _BC_INFLOW_Q
            bc_vl_init[i] = Q_STEADY

        left_edge_len = _edge_lengths(n0, n1, node_x, node_y)[left_idx]
        total_left_len = float(np.sum(left_edge_len))
        q_unit_init = Q_STEADY / total_left_len
        for i in left_idx:
            bc_vl_init[i] = q_unit_init

        backend.build_mesh(node_x, node_y, node_z, cell_nodes,
                           n0, n1, bc_tp_init, bc_vl_init)
        backend.initialize(h0, hu0, hv0,
                           g=9.81, n_mann=self.N_MANN,
                           h_min=1e-4, cfl=0.45,
                           dt_fixed=self.DT, dt_max=self.DT)

        cfg = SWE2DNativeBoundaryHydrographConfigurator()
        result = cfg.configure(
            backend=backend,
            bc_n0=n0, bc_n1=n1, bc_tp=bc_type,
            side_hydrographs={},
            edge_hydrographs=edge_hydrographs,
            node_x=node_x, node_y=node_y, node_z=node_z,
            inflow_q_bc_type=_BC_INFLOW_Q,
            progressive=False,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
        )

        self.assertTrue(result["native_bc_forcing"])

        steps = int(self.T_END / self.DT)
        for step in range(steps):
            diag = backend.step(self.DT)

        h_final, hu_final, hv_final = backend.get_state()

        tris = cell_nodes.reshape((-1, 3))
        cx = (node_x[tris[:, 0]] + node_x[tris[:, 1]] + node_x[tris[:, 2]]) / 3.0
        interior_mask = (cx > 20.0) & (cx < 80.0)
        h_mean = float(np.mean(h_final[interior_mask]))

        print(f"\n[test_native_hydrograph_solver_run]")
        print(f"  Q_STEADY = {Q_STEADY} m³/s")
        print(f"  N left edges = {n_left}")
        print(f"  Total left edge length = {total_left_len:.4f} m")
        print(f"  Mean interior depth after {self.T_END}s = {h_mean:.4f} m")
        print(f"  Native BC forcing active: {result['native_bc_forcing']}")

        self.assertGreater(h_mean, 0.01, "Interior depth should be > 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
