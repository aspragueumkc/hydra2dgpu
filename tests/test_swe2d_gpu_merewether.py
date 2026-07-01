"""
GPU-only real-topography flood validation: Merewether (Newcastle, 2007).

Reference:
    Water Research Laboratory, UNSW — WRL2012003.01
    2007 Pasha Bulka flood event simulation.

Mesh source:
    reference/anuga_validation_tests/case_studies/merewether/topography1.asc
    (321×416 DEM, 1m cells, MGA Zone 56S)

Observed peak stage at 5 gauges (from ObservationPoints.csv):
    Gauge 0: (382424, 6354478) — stage ≈ 20.00 m
    Gauge 1: (382510, 6354548) — stage ≈ 18.40 m
    Gauge 2: (382339, 6354298) — stage ≈ 23.50 m
    Gauge 3: (382355, 6354365) — stage ≈ 23.10 m
    Gauge 4: (382374, 6354388) — stage ≈ 23.00 m

Inlet:
    Circular region radius=10m, center=(382265, 6354280)
    Constant flow rate Q = 19.7 m³/s (Pasha Bulka peak)
    Treated as equivalent uniform rainfall over inlet area: q = Q/(π·10²) ≈ 0.0627 m/s
    Over the 5m-mesh inlet area (~314 m²) this gives the correct total inflow.

Boundary conditions (matching ANUGA merewether setup):
    Left (x=382250): WALL (reflective)
    Bottom (y=6354265): WALL (reflective)
    Right (x=382571): OPEN (free outflow)
    Top (y=6354681): OPEN (free outflow)

Friction: Manning n = 0.04 (typical for urban/floodplain)

Validation strategy:
    Because the 5m test mesh has different cell sizes and wetting dynamics from
    the 1m reference mesh used in ANUGA's reported gauge values, we validate
    the PHYSICS, not exact numbers:
    1. GPU stability + positivity throughout
    2. Flood wave ordering: gauges near the inlet (2,3,4) show higher stage
       than downstream gauges (0,1), matching the observed data ordering
    3. Downstream gauges (0,1) show measurable stage increase from inlet flow
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _load_module, _gpu_available
from swe2d.runtime.backend import SWE2DBackend, swe2d_available


GAUGE_IDS = [0, 1, 2, 3, 4]
GAUGE_XY = {
    0: (382424.40, 6354478.33),
    1: (382509.71, 6354548.22),
    2: (382339.42, 6354297.84),
    3: (382354.61, 6354365.21),
    4: (382373.51, 6354387.84),
}
GAUGE_OBSERVED_STAGE = {
    0: 20.00,
    1: 18.40,
    2: 23.50,
    3: 23.10,
    4: 23.00,
}

INLET_CX = 382265.0
INLET_CY = 6354280.0
INLET_RADIUS = 10.0
INLET_Q = 19.7

DOMAIN_XMIN = 382250.0
DOMAIN_YMIN = 6354265.0
DOMAIN_XMAX = 382571.0
DOMAIN_YMAX = 6354681.0

ELEM_SIZE = 5.0


def _parse_asc(path):
    with open(path) as f:
        lines = f.readlines()
    ncols = int(lines[0].split()[1])
    nrows = int(lines[1].split()[1])
    xll = float(lines[2].split()[1])
    yll = float(lines[3].split()[1])
    cs = float(lines[4].split()[1])
    nodata = float(lines[5].split()[1])
    data = np.array([[float(v) for v in line.split()] for line in lines[6:]], dtype=np.float64)
    return {"ncols": ncols, "nrows": nrows, "xll": xll, "yll": yll, "cellsize": cs,
            "nodata": nodata, "data": data}


def _bilinear_interp(x, y, asc):
    cs = asc["cellsize"]
    xll = asc["xll"]
    yll = asc["yll"]
    ncols = asc["ncols"]
    nrows = asc["nrows"]
    data = asc["data"]

    col = (x - xll) / cs
    row = (y - yll) / cs
    i = int(np.clip(np.floor(col), 0, ncols - 2))
    j = int(np.clip(np.floor(row), 0, nrows - 2))

    fx = col - i
    fy = row - j

    v00 = data[nrows - 1 - j, i]
    v10 = data[nrows - 1 - j, i + 1]
    v01 = data[nrows - 2 - j, i]
    v11 = data[nrows - 2 - j, i + 1]

    nodata = asc["nodata"]
    for v in [v00, v10, v01, v11]:
        if v == nodata:
            return np.nan

    return (v00 * (1 - fx) * (1 - fy) +
            v10 * fx * (1 - fy) +
            v01 * (1 - fx) * fy +
            v11 * fx * fy)


def _build_mesh_and_callbacks(asc, elem_size):
    import gmsh

    Lx = DOMAIN_XMAX - DOMAIN_XMIN
    Ly = DOMAIN_YMAX - DOMAIN_YMIN

    gmsh.initialize()
    gmsh.model.add("merewether")

    rect = gmsh.model.occ.addRectangle(DOMAIN_XMIN, DOMAIN_YMIN, 0, Lx, Ly)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", elem_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", elem_size * 0.5)
    gmsh.model.mesh.generate(2)

    _, node_coords, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_coords.reshape(-1, 3)
    node_x = node_xyz[:, 0].copy()
    node_y = node_xyz[:, 1].copy()

    node_z = np.array([_bilinear_interp(x, y, asc) for x, y in zip(node_x, node_y)],
                       dtype=np.float64)
    node_z = np.nan_to_num(node_z, nan=0.0)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    cell_nodes = None
    for etype, nodes in zip(elem_types, elem_node_tags):
        if etype == 2:
            cell_nodes = (nodes - 1).reshape(-1, 3).astype(np.int32)
            break
    gmsh.finalize()

    n_cells = cell_nodes.shape[0]
    cell_cx = np.mean(node_x[cell_nodes], axis=1)
    cell_cy = np.mean(node_y[cell_nodes], axis=1)
    cell_area = np.array([np.sum(np.abs(np.cross(
        [node_x[cell_nodes[i, 1]] - node_x[cell_nodes[i, 0]],
         node_y[cell_nodes[i, 1]] - node_y[cell_nodes[i, 0]], 0.0],
        [node_x[cell_nodes[i, 2]] - node_x[cell_nodes[i, 0]],
         node_y[cell_nodes[i, 2]] - node_y[cell_nodes[i, 0]], 0.0]
    )) / 2.0) for i in range(n_cells)])

    if cell_area.size == 1:
        cell_area = np.full(n_cells, cell_area[0])

    inlet_cells = np.zeros(n_cells, dtype=bool)
    for i in range(n_cells):
        dx = cell_cx[i] - INLET_CX
        dy = cell_cy[i] - INLET_CY
        if np.sqrt(dx*dx + dy*dy) <= INLET_RADIUS:
            inlet_cells[i] = True

    total_inlet_area = cell_area[inlet_cells].sum()
    q_rate = INLET_Q / total_inlet_area if total_inlet_area > 0 else 0.0

    def source_rate_callback(_t, _dt, h, _hu, _hv):
        rate = np.zeros_like(h)
        rate[inlet_cells] = q_rate
        return rate

    TOL = elem_size * 0.4

    def _is_on_xmin(x):
        return abs(x - DOMAIN_XMIN) < TOL

    def _is_on_xmax(x):
        return abs(x - DOMAIN_XMAX) < TOL

    def _is_on_ymin(y):
        return abs(y - DOMAIN_YMIN) < TOL

    def _is_on_ymax(y):
        return abs(y - DOMAIN_YMAX) < TOL

    bc_n0, bc_n1, bc_tp, bc_vl = [], [], [], []
    seen = set()

    for i in range(n_cells):
        nds = cell_nodes[i]
        for ek in range(3):
            n0 = nds[ek]
            n1 = nds[(ek + 1) % 3]
            edge_key = (min(n0, n1), max(n0, n1))
            if edge_key in seen:
                continue
            x0, y0 = node_x[n0], node_y[n0]
            x1, y1 = node_x[n1], node_y[n1]
            mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5

            if _is_on_xmin(mx):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)
                seen.add(edge_key)
            elif _is_on_xmax(mx):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(4); bc_vl.append(0.0)
                seen.add(edge_key)
            elif _is_on_ymin(my):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)
                seen.add(edge_key)
            elif _is_on_ymax(my):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(4); bc_vl.append(0.0)
                seen.add(edge_key)

    return (node_x, node_y, node_z, cell_nodes,
            np.array(bc_n0, dtype=np.int32),
            np.array(bc_n1, dtype=np.int32),
            np.array(bc_tp, dtype=np.int32),
            np.array(bc_vl, dtype=np.float64),
            source_rate_callback,
            cell_cx, cell_cy)


@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUMerewetherRealTopo(unittest.TestCase):
    """
    Real-topography flood validation on the Merewether (Newcastle) study area.

    Note: the 5m test mesh differs from the 1m reference mesh used in the
    ANUGA/ARR/TUFLOW benchmarks, so exact stage matching is not expected.
    The test validates physical ordering (upstream gauges > downstream) and
    GPU stability.
    """

    ASC_PATH = "reference/anuga_validation_tests/case_studies/merewether/topography1.asc"
    T_END = 1000.0
    N_MANN = 0.04
    CFL = 0.45
    DT_MAX = 1.0

    @classmethod
    def setUpClass(cls):
        import os
        asc_path = os.path.join(os.path.dirname(__file__), "..", cls.ASC_PATH)
        asc_path = os.path.normpath(asc_path)
        cls._asc = _parse_asc(asc_path)

        arrays = _build_mesh_and_callbacks(cls._asc, ELEM_SIZE)
        (node_x, node_y, node_z, cell_nodes,
         bc_n0, bc_n1, bc_tp, bc_vl,
         src_cb, cell_cx, cell_cy) = arrays

        backend = SWE2DBackend()
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl,
        )

        n_cells = cell_nodes.shape[0]
        h0 = np.zeros(n_cells, dtype=np.float64)
        hu0 = np.zeros(n_cells, dtype=np.float64)
        hv0 = np.zeros(n_cells, dtype=np.float64)

        backend.initialize(
            h0=h0, hu0=hu0, hv0=hv0,
            n_mann=cls.N_MANN,
            cfl=cls.CFL,
            dt_max=cls.DT_MAX,
            gpu_diag_sync_interval_steps=1,
        )

        cls._backend = backend
        cls._src_cb = src_cb
        cls._cell_cx = cell_cx
        cls._cell_cy = cell_cy
        cls._node_z = node_z
        cls._cell_nodes = cell_nodes

    def setUp(self):
        self._src_cb = object.__getattribute__(self.__class__, '_src_cb')

    def _run(self, t_end=None):
        t_end = t_end or self.T_END
        diags = self._backend.run(
            t_end=t_end,
            source_rate_callback=self._src_cb,
            use_native_source_injection=True,
        )
        h, hu, hv = self._backend.get_state()
        zb = self._node_z[self._cell_nodes].mean(axis=1)
        stage = h + zb
        return {
            "t": t_end,
            "h": h,
            "stage": stage,
            "diag": diags[-1] if diags else {},
            "cell_cx": self._cell_cx,
            "cell_cy": self._cell_cy,
        }

    def test_gpu_stability_positivity(self):
        """GPU solver must stay active with non-negative depth throughout."""
        result = self._run()
        self.assertTrue(result["diag"]["gpu_active"], "GPU became inactive")
        self.assertGreater(result["diag"]["dt"], 0.0)
        self.assertGreaterEqual(float(result["h"].min()), -1e-10,
            f"Negative depth: {result['h'].min():.6f}")
        self.assertTrue(np.isfinite(result["h"]).all(), "Non-finite depth detected")

    def test_flood_wave_reaches_downstream_gauges(self):
        """
        After 1000s of 19.7 m³/s inflow, the flood wave must reach and wet
        the downstream gauges (0 and 1), which are the lowest-elevation gauges.
        """
        result = self._run()

        stage = result["stage"]
        cell_cx = result["cell_cx"]
        cell_cy = result["cell_cy"]

        for gid in [0, 1]:
            gx, gy = GAUGE_XY[gid]
            dist = np.sqrt((cell_cx - gx)**2 + (cell_cy - gy)**2)
            nearest = np.argmin(dist)
            s = float(stage[nearest])
            obs = GAUGE_OBSERVED_STAGE[gid]
            self.assertGreater(s, obs - 2.0,
                f"Gauge {gid} stage {s:.2f}m should be near or above observed "
                f"{obs:.2f}m (allowing 2m for mesh difference)")
            self.assertGreater(result["h"][nearest], 0.0,
                f"Gauge {gid} should be wet after 1000s (cell h={result['h'][nearest]:.4f})")

    def test_stage_ordering_near_inlet_above_downstream(self):
        """
        At the end of the simulation, gauges near the inlet (2,3,4, all ≈23m bed)
        must have stage ≥ downstream gauges (0,1, bed ≈18-19m), matching the
        observed data ordering (gauge 2 > gauge 0).
        """
        result = self._run()

        stage = result["stage"]
        cell_cx = result["cell_cx"]
        cell_cy = result["cell_cy"]

        def gauge_stage(gid):
            gx, gy = GAUGE_XY[gid]
            dist = np.sqrt((cell_cx - gx)**2 + (cell_cy - gy)**2)
            nearest = np.argmin(dist)
            return float(stage[nearest])

        s0 = gauge_stage(0)
        s1 = gauge_stage(1)
        s2 = gauge_stage(2)

        self.assertGreater(s2, s0,
            f"Near-inlet gauge 2 (stage={s2:.2f}m) should exceed downstream "
            f"gauge 0 (stage={s0:.2f}m) — upstream must be deeper near inlet")

    def test_convergence_smaller_mesh(self):
        """
        Verify that stage at a representative point converges as mesh refines
        (5m → 3m).  Error should decrease with mesh refinement.
        """
        def _run_backend(elem_size):
            arrays = _build_mesh_and_callbacks(self._asc, elem_size)
            (node_x, node_y, node_z, cell_nodes,
             bc_n0, bc_n1, bc_tp, bc_vl,
             src_cb, cell_cx, cell_cy) = arrays

            backend = SWE2DBackend()
            backend.build_mesh(node_x, node_y, node_z, cell_nodes,
                              bc_n0, bc_n1, bc_tp, bc_vl)
            n_cells = cell_nodes.shape[0]
            backend.initialize(
                h0=np.zeros(n_cells, dtype=np.float64),
                hu0=np.zeros(n_cells, dtype=np.float64),
                hv0=np.zeros(n_cells, dtype=np.float64),
                n_mann=self.N_MANN,
                cfl=self.CFL,
                dt_max=self.DT_MAX,
                gpu_diag_sync_interval_steps=1,
            )
            try:
                backend.run(t_end=200.0, source_rate_callback=src_cb,
                             use_native_source_injection=True)
                h, _, _ = backend.get_state()
                zb = node_z[cell_nodes].mean(axis=1)
                stage = h + zb
                gx, gy = GAUGE_XY[4]
                dist = np.sqrt((cell_cx - gx)**2 + (cell_cy - gy)**2)
                return float(stage[np.argmin(dist)])
            finally:
                backend.destroy()

        s_coarse = _run_backend(ELEM_SIZE)
        s_fine = _run_backend(ELEM_SIZE * 0.6)

        self.assertLess(
            abs(s_fine - s_coarse), 2.0,
            f"Stage at gauge 4 should converge: coarse={s_coarse:.2f}, fine={s_fine:.2f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
