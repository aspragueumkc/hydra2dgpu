"""
Gmsh-based compound-channel SWE2D validation with split element sizes.

This test builds an unstructured triangle mesh for the same synthetic
compound-channel setup used by test_swe2d_compound_channel, but with a
piecewise target mesh size:
- upstream half   (x <= Lx/2): 50 m
- downstream half (x >  Lx/2): 25 m
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_swe2d_compound_channel import compound_conveyance, solve_stage


def _load_module():
    try:
        import backwater_swe2d
        return backwater_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


def _gmsh_available():
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


def _make_gmsh_compound_mesh(Lx, Ly, size_upstream, size_downstream, split_x, bed_func):
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("compound_multiscale")

        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, size_downstream)
        p2 = gmsh.model.geo.addPoint(Lx, 0.0, 0.0, size_downstream)
        p3 = gmsh.model.geo.addPoint(Lx, Ly, 0.0, size_downstream)
        p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0, size_downstream)

        l_bottom = gmsh.model.geo.addLine(p1, p2)
        l_right = gmsh.model.geo.addLine(p2, p3)
        l_top = gmsh.model.geo.addLine(p3, p4)
        l_left = gmsh.model.geo.addLine(p4, p1)

        cl = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
        surf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        # Split-size control: coarse upstream, finer downstream.
        # Disable point/curvature/boundary-extension sizing so the background
        # field is the dominant source of target mesh size.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field, "VIn", float(size_upstream))
        gmsh.model.mesh.field.setNumber(field, "VOut", float(size_downstream))
        gmsh.model.mesh.field.setNumber(field, "XMin", 0.0)
        gmsh.model.mesh.field.setNumber(field, "XMax", float(split_x))
        gmsh.model.mesh.field.setNumber(field, "YMin", 0.0)
        gmsh.model.mesh.field.setNumber(field, "YMax", float(Ly))
        gmsh.model.mesh.field.setNumber(field, "Thickness", 1.0e-9)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        node_x = node_coords[:, 0].copy()
        node_y = node_coords[:, 1].copy()
        node_z = np.asarray(bed_func(node_x, node_y), dtype=np.float64)

        elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
        cell_nodes_list = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype != 2:
                continue
            tri_nodes = np.array(enodes, dtype=np.int64).reshape(-1, 3)
            for tri in tri_nodes:
                cell_nodes_list.extend([
                    tag_to_idx[int(tri[0])],
                    tag_to_idx[int(tri[1])],
                    tag_to_idx[int(tri[2])],
                ])

        cell_nodes = np.array(cell_nodes_list, dtype=np.int32)

    finally:
        gmsh.finalize()

    return node_x, node_y, node_z, cell_nodes


def _boundary_edges(cell_nodes):
    tris = cell_nodes.reshape(-1, 3)
    counts = {}
    for a, b, c in tris:
        for i, j in ((a, b), (b, c), (c, a)):
            e = (int(i), int(j)) if i < j else (int(j), int(i))
            counts[e] = counts.get(e, 0) + 1

    b0 = []
    b1 = []
    for (i, j), n in counts.items():
        if n == 1:
            b0.append(i)
            b1.append(j)
    return np.array(b0, dtype=np.int32), np.array(b1, dtype=np.int32)


def _build_boundary_conditions(node_x, node_y, bc_n0, bc_n1,
                               y_chan_left, y_chan_right,
                               q_main, q_fp):
    x_mid = 0.5 * (node_x[bc_n0] + node_x[bc_n1])
    y_mid = 0.5 * (node_y[bc_n0] + node_y[bc_n1])

    xmin = float(np.min(node_x))
    xmax = float(np.max(node_x))
    tol = 1.0e-8 * max(1.0, xmax - xmin)

    bc_type = np.full(bc_n0.size, 1, dtype=np.int32)  # WALL default
    bc_val = np.zeros(bc_n0.size, dtype=np.float64)

    left = np.abs(x_mid - xmin) <= tol
    right = np.abs(x_mid - xmax) <= tol

    left_main = left & (y_mid >= y_chan_left) & (y_mid <= y_chan_right)
    left_fp = left & ~left_main

    bc_type[left_main] = 2  # INFLOW_Q
    bc_val[left_main] = q_main

    bc_type[left_fp] = 2    # INFLOW_Q
    bc_val[left_fp] = q_fp

    bc_type[right] = 4      # OPEN
    bc_val[right] = 0.0

    return bc_type, bc_val


def _cell_centroids_from_nodes(node_x, node_y, cell_nodes):
    tris = cell_nodes.reshape(-1, 3)
    cx = (node_x[tris[:, 0]] + node_x[tris[:, 1]] + node_x[tris[:, 2]]) / 3.0
    cy = (node_y[tris[:, 0]] + node_y[tris[:, 1]] + node_y[tris[:, 2]]) / 3.0
    return cx, cy


@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestCompoundChannelGmshMultiscale(unittest.TestCase):
    LX = 2000.0
    LY = 200.0
    SPLIT_X = 1000.0
    H_UPSTREAM = 50.0
    H_DOWNSTREAM = 25.0

    W_MAIN = 20.0
    W_FP_EACH = 90.0
    H_BANK = 2.0
    S0 = 0.001
    N_MAIN = 0.030
    N_FP = 0.060
    Q_TOTAL = 100.0

    Y_LEFT_BANK = 90.0
    Y_RIGHT_BANK = 110.0

    N_STEPS = 200

    LONG_STEPS = 200000
    LONG_DT = 0.05

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

        H_ana = solve_stage(
            cls.Q_TOTAL, cls.S0, cls.W_MAIN, cls.W_FP_EACH, cls.H_BANK, cls.N_MAIN, cls.N_FP
        )
        ks = compound_conveyance(H_ana, cls.W_MAIN, cls.W_FP_EACH, cls.H_BANK, cls.N_MAIN, cls.N_FP)
        sqrt_s0 = np.sqrt(cls.S0)
        cls.q_main = ks["K_main"] * sqrt_s0 / cls.W_MAIN
        cls.q_fp = ks["K_fp"] * sqrt_s0 / (2.0 * cls.W_FP_EACH)
        cls.H_ana = H_ana

        def bed(x, y):
            z_slope = cls.S0 * (cls.LX - x)
            bank = np.where((y >= cls.Y_LEFT_BANK) & (y <= cls.Y_RIGHT_BANK), 0.0, cls.H_BANK)
            return z_slope + bank

        node_x, node_y, node_z, cell_nodes = _make_gmsh_compound_mesh(
            cls.LX,
            cls.LY,
            cls.H_UPSTREAM,
            cls.H_DOWNSTREAM,
            cls.SPLIT_X,
            bed,
        )

        bc_n0, bc_n1 = _boundary_edges(cell_nodes)
        bc_type, bc_val = _build_boundary_conditions(
            node_x,
            node_y,
            bc_n0,
            bc_n1,
            cls.Y_LEFT_BANK,
            cls.Y_RIGHT_BANK,
            cls.q_main,
            cls.q_fp,
        )

        cls.mesh = cls.mod.swe2d_build_mesh(
            node_x,
            node_y,
            node_z,
            cell_nodes,
            bc_n0,
            bc_n1,
            bc_type,
            bc_val,
        )

        cls.node_x = node_x
        cls.node_y = node_y
        cls.node_z = node_z
        cls.cell_nodes = cell_nodes

        cx, cy = _cell_centroids_from_nodes(node_x, node_y, cell_nodes)
        cls.cx = cx
        cls.cy = cy

        cls.n_mann_cell = np.where(
            (cy >= cls.Y_LEFT_BANK) & (cy <= cls.Y_RIGHT_BANK),
            cls.N_MAIN,
            cls.N_FP,
        ).astype(np.float64)

        z_bed_cell = bed(cx, cy)
        wse_cell = cls.S0 * (cls.LX - cx) + cls.H_ana
        cls.h0 = np.maximum(0.0, wse_cell - z_bed_cell).astype(np.float64)
        cls.hu0 = np.where(
            (cy >= cls.Y_LEFT_BANK) & (cy <= cls.Y_RIGHT_BANK),
            cls.q_main,
            cls.q_fp,
        ).astype(np.float64)
        cls.hv0 = np.zeros_like(cls.h0)

    def test_mesh_is_refined_downstream(self):
        tris = self.cell_nodes.reshape(-1, 3)
        x0 = self.node_x[tris[:, 0]]
        x1 = self.node_x[tris[:, 1]]
        x2 = self.node_x[tris[:, 2]]
        y0 = self.node_y[tris[:, 0]]
        y1 = self.node_y[tris[:, 1]]
        y2 = self.node_y[tris[:, 2]]
        area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))

        upstream = self.cx <= self.SPLIT_X
        downstream = self.cx > self.SPLIT_X

        self.assertTrue(upstream.any() and downstream.any())
        med_up = float(np.median(area[upstream]))
        med_dn = float(np.median(area[downstream]))

        # Halving target size should reduce typical triangle area significantly.
        self.assertGreater(med_up / max(med_dn, 1.0e-12), 2.0)

    def test_higher_order_schemes_smoke(self):
        for scheme in (1, 2, 3, 4):
            solver = self.mod.swe2d_create_solver(
                self.mesh,
                self.h0.copy(),
                self.hu0.copy(),
                self.hv0.copy(),
                n_mann_cell=self.n_mann_cell,
                n_mann=self.N_MAIN,
                cfl=0.45,
                dt_max=1.0,
                spatial_scheme=scheme,
                use_gpu=True,
            )

            last_diag = None
            for _ in range(self.N_STEPS):
                last_diag = self.mod.swe2d_step(solver, -1.0)

            h, hu, hv = self.mod.swe2d_get_state(solver)
            self.mod.swe2d_destroy(solver)

            self.assertTrue(last_diag["gpu_active"], f"GPU inactive for scheme {scheme}")
            self.assertTrue(np.isfinite(h).all(), f"Non-finite h for scheme {scheme}")
            self.assertTrue(np.isfinite(hu).all(), f"Non-finite hu for scheme {scheme}")
            self.assertTrue(np.isfinite(hv).all(), f"Non-finite hv for scheme {scheme}")
            self.assertLess(float(np.max(h)), 1.0e6, f"Depth blow-up for scheme {scheme}")

    @unittest.skipUnless(
        os.environ.get("BACKWATER_RUN_LONG_GMSH_COMPOUND", "0") == "1",
        "Set BACKWATER_RUN_LONG_GMSH_COMPOUND=1 to enable 200000-step long-run test",
    )
    def test_higher_order_schemes_long_run_dt005(self):
        summary = []
        for scheme in (1, 2, 3, 4):
            solver = self.mod.swe2d_create_solver(
                self.mesh,
                self.h0.copy(),
                self.hu0.copy(),
                self.hv0.copy(),
                n_mann_cell=self.n_mann_cell,
                n_mann=self.N_MAIN,
                cfl=0.45,
                dt_max=self.LONG_DT,
                dt_fixed=self.LONG_DT,
                spatial_scheme=scheme,
                use_gpu=True,
            )

            failed = False
            fail_reason = ""
            fail_step = self.LONG_STEPS
            last_diag = None

            for step in range(1, self.LONG_STEPS + 1):
                last_diag = self.mod.swe2d_step(solver, -1.0)
                h, hu, hv = self.mod.swe2d_get_state(solver)

                if (not np.isfinite(h).all()) or (not np.isfinite(hu).all()) or (not np.isfinite(hv).all()):
                    failed = True
                    fail_step = step
                    fail_reason = "non-finite state"
                    break

                if np.max(np.abs(h)) > 1.0e6 or np.max(np.abs(hu)) > 1.0e7 or np.max(np.abs(hv)) > 1.0e7:
                    failed = True
                    fail_step = step
                    fail_reason = "state magnitude blow-up"
                    break

            h, hu, hv = self.mod.swe2d_get_state(solver)
            self.mod.swe2d_destroy(solver)

            summary.append(
                {
                    "scheme": scheme,
                    "failed": failed,
                    "fail_step": fail_step,
                    "fail_reason": fail_reason,
                    "h_min": float(np.min(h)),
                    "h_max": float(np.max(h)),
                    "hu_abs_max": float(np.max(np.abs(hu))),
                    "hv_abs_max": float(np.max(np.abs(hv))),
                    "gpu_active": bool(last_diag["gpu_active"]) if last_diag is not None else False,
                }
            )

            self.assertFalse(
                failed,
                f"Scheme {scheme} failed at step {fail_step}: {fail_reason}",
            )

        print("\\n[gmsh-multiscale long-run summary]")
        for r in summary:
            print(
                f"  scheme {r['scheme']}: failed={r['failed']} "
                f"fail_step={r['fail_step'] if r['failed'] else 'n/a'} "
                f"h[min,max]=[{r['h_min']:.4e},{r['h_max']:.4e}] "
                f"|hu|max={r['hu_abs_max']:.4e} |hv|max={r['hv_abs_max']:.4e} "
                f"gpu_active={r['gpu_active']}"
            )


if __name__ == "__main__":
    unittest.main()
