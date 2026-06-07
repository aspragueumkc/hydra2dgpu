import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except Exception:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return bool(mod.swe2d_gpu_available())
    except Exception:
        return False


def _build_closed_two_cell_mesh(mod):
    node_x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    node_z = np.zeros(4, dtype=np.float64)
    cell_nodes = np.array([0, 1, 3, 0, 3, 2], dtype=np.int32)
    bc_n0 = np.array([0, 2, 1, 3], dtype=np.int32)
    bc_n1 = np.array([2, 3, 0, 1], dtype=np.int32)
    bc_type = np.array([1, 1, 1, 1], dtype=np.int32)
    bc_val = np.zeros(4, dtype=np.float64)
    return mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_type, bc_val)


def _make_rect_channel_mesh(nx, ny, lx, ly):
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    xg, yg = np.meshgrid(xs, ys)
    node_x = xg.ravel().astype(np.float64)
    node_y = yg.ravel().astype(np.float64)
    node_z = np.zeros_like(node_x)
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
    bc_n0 = []
    bc_n1 = []
    bc_type = []
    bc_val = []
    for j in range(ny):
        bc_n0.append(j * stride)
        bc_n1.append((j + 1) * stride)
        bc_type.append(2)
        bc_val.append(0.0)
    for j in range(ny):
        bc_n0.append(j * stride + nx)
        bc_n1.append((j + 1) * stride + nx)
        bc_type.append(4)
        bc_val.append(0.0)
    return (
        node_x,
        node_y,
        node_z,
        np.array(cells, dtype=np.int32),
        np.array(bc_n0, dtype=np.int32),
        np.array(bc_n1, dtype=np.int32),
        np.array(bc_type, dtype=np.int32),
        np.array(bc_val, dtype=np.float64),
    )


def _cn_exact_depth(total_rain_mm, cn, ia_ratio=0.2, mm_to_model_depth=1.0e-3):
    cn_c = min(100.0, max(1.0, float(cn)))
    s_mm = max((25400.0 / cn_c) - 254.0, 0.0)
    ia = ia_ratio * s_mm
    p = float(total_rain_mm)
    if p <= ia:
        return 0.0
    pe = ((p - ia) * (p - ia)) / max(p + (1.0 - ia_ratio) * s_mm, 1.0e-12)
    return pe * mm_to_model_depth


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUHigherOrderGraphSafe(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def _run_single_cell_rain(self, temporal_order, dt_fixed):
        mesh = _build_closed_two_cell_mesh(self.mod)
        n_cells = self.mod.swe2d_mesh_info(mesh)["n_cells"]
        solver = self.mod.swe2d_create_solver(
            mesh,
            np.zeros(n_cells, dtype=np.float64),
            np.zeros(n_cells, dtype=np.float64),
            np.zeros(n_cells, dtype=np.float64),
            g=9.81,
            n_mann=0.03,
            h_min=1.0e-6,
            cfl=0.45,
            dt_max=dt_fixed,
            dt_fixed=dt_fixed,
            use_gpu=True,
            temporal_order=temporal_order,
            spatial_scheme=0,
        )
        try:
            t_end = 60.0
            n_samples = 4097
            time_s = np.linspace(0.0, t_end, n_samples, dtype=np.float64)
            rain_total_mm = 42.0
            rain_cum_mm = rain_total_mm * (time_s / t_end) ** 2
            self.mod.swe2d_solver_set_rain_cn_forcing(
                solver,
                np.zeros(n_cells, dtype=np.int32),
                np.array([0, n_samples], dtype=np.int32),
                time_s,
                rain_cum_mm,
                np.full(n_cells, 78.0, dtype=np.float64),
                0.2,
                1.0e-3,
            )

            diag = None
            n_steps = int(round(t_end / dt_fixed))
            for _ in range(n_steps):
                diag = self.mod.swe2d_step(solver, dt_fixed)
            h, _hu, _hv = self.mod.swe2d_get_state(solver)
            exact = _cn_exact_depth(rain_total_mm, 78.0)
            return float(np.max(np.abs(h - exact))), diag
        finally:
            self.mod.swe2d_destroy(solver)

    def test_rain_exact_depth_error_improves_with_higher_order(self):
        err_rk2, diag2 = self._run_single_cell_rain(2, 2.0)
        err_rk4g, diag5 = self._run_single_cell_rain(5, 2.0)
        err_rk5g, diag6 = self._run_single_cell_rain(6, 2.0)

        self.assertTrue(diag2["gpu_active"])
        self.assertTrue(diag5["gpu_active"])
        self.assertTrue(diag6["gpu_active"])
        self.assertTrue(np.isfinite(err_rk2))
        self.assertTrue(np.isfinite(err_rk4g))
        self.assertTrue(np.isfinite(err_rk5g))
        self.assertLess(err_rk4g, err_rk2)
        self.assertLess(err_rk5g, err_rk4g)

    def test_dynamic_hydrograph_keeps_graph_path_live(self):
        mesh_args = _make_rect_channel_mesh(24, 1, 48.0, 2.0)
        mesh = self.mod.swe2d_build_mesh(*mesh_args)
        edge_idx, n0, n1, _bc_type, _bc_val, _cell0 = self.mod.swe2d_boundary_edges(mesh)
        node_x = mesh_args[0]
        left_mask = np.isclose(node_x[n0], 0.0) & np.isclose(node_x[n1], 0.0)
        left_edges = np.asarray(edge_idx)[left_mask]
        self.assertEqual(left_edges.size, 1)

        time_s = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        q_ts = np.array([0.05, 0.15, 0.08, 0.18, 0.10], dtype=np.float64)
        offsets = np.array([0, time_s.size], dtype=np.int32)
        bc_type = np.array([2], dtype=np.int32)
        h0 = np.full(self.mod.swe2d_mesh_info(mesh)["n_cells"], 0.2, dtype=np.float64)
        hu0 = np.zeros_like(h0)
        hv0 = np.zeros_like(h0)

        for temporal_order in (5, 6):
            solver = self.mod.swe2d_create_solver(
                mesh,
                h0,
                hu0,
                hv0,
                g=9.81,
                n_mann=0.03,
                h_min=1.0e-6,
                cfl=0.45,
                dt_max=0.25,
                dt_fixed=0.25,
                use_gpu=True,
                temporal_order=temporal_order,
                spatial_scheme=0,
            )
            try:
                self.mod.swe2d_solver_set_boundary_hydrographs(
                    solver,
                    left_edges.astype(np.int32),
                    bc_type,
                    offsets,
                    time_s,
                    q_ts,
                )
                diag = None
                for _ in range(16):
                    diag = self.mod.swe2d_step(solver, 0.25)
                h, hu, hv = self.mod.swe2d_get_state(solver)
                self.assertTrue(diag["gpu_active"])
                self.assertTrue(np.isfinite(h).all())
                self.assertTrue(np.isfinite(hu).all())
                self.assertTrue(np.isfinite(hv).all())
                self.assertGreater(float(np.max(h)), 0.0)
            finally:
                self.mod.swe2d_destroy(solver)


if __name__ == "__main__":
    unittest.main(verbosity=2)
