import os
import sys
import unittest
from contextlib import contextmanager

import numpy as np

from swe2d.mesh.meshing import generate_face_centric_mesh  # noqa: E402
from tools.gmsh_topology_mesher import load_conceptual_model  # noqa: E402


def _polyline_distance(x: float, y: float, line):
    if len(line) < 2:
        return np.inf
    best = np.inf
    for i in range(len(line) - 1):
        ax, ay = line[i]
        bx, by = line[i + 1]
        vx = bx - ax
        vy = by - ay
        den = vx * vx + vy * vy
        if den <= 1.0e-18:
            continue
        t = ((x - ax) * vx + (y - ay) * vy) / den
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        qx = ax + t * vx
        qy = ay + t * vy
        d = float(np.hypot(x - qx, y - qy))
        if d < best:
            best = d
    return best


def _collect_region_boundary_lines(model):
    lines = []
    for reg in getattr(model, "regions", []):
        ring = [(float(x), float(y)) for (x, y) in (reg.ring_xy or [])]
        if len(ring) >= 3:
            ring = list(ring)
            ring.append(ring[0])
            lines.append(ring)
        for hole in (reg.hole_rings or []):
            h = [(float(x), float(y)) for (x, y) in (hole or [])]
            if len(h) >= 3:
                h = list(h)
                h.append(h[0])
                lines.append(h)
    return lines


def _collect_arc_lines(model):
    lines = []
    for arc in getattr(model, "arcs", []):
        pts = [(float(x), float(y)) for (x, y) in (arc.points_xy or [])]
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def _build_face_edge_set(offs, conn):
    edge_set = set()
    for face_idx in range(int(offs.size) - 1):
        ids = conn[int(offs[face_idx]): int(offs[face_idx + 1])]
        if ids.size < 2:
            continue
        for i in range(int(ids.size)):
            a = int(ids[i])
            b = int(ids[(i + 1) % ids.size])
            if a == b:
                continue
            edge_set.add((a, b) if a < b else (b, a))
    return edge_set


def _nearest_node_index(px, py, node_x, node_y, snap_tol):
    dx = node_x - float(px)
    dy = node_y - float(py)
    d2 = dx * dx + dy * dy
    if d2.size == 0:
        return -1
    i = int(np.argmin(d2))
    if float(d2[i]) > float(snap_tol * snap_tol):
        return -1
    return i


def _collect_constrained_edges_sampled(node_x, node_y, boundary_lines, snap_tol):
    snap_tol = max(1.0e-9, float(snap_tol))
    step_len = max(1.0e-9, 0.75 * snap_tol)
    edges = set()
    for line in boundary_lines:
        if len(line) < 2:
            continue
        for i in range(len(line) - 1):
            p0 = line[i]
            p1 = line[i + 1]
            n0 = _nearest_node_index(p0[0], p0[1], node_x, node_y, snap_tol)
            n1 = _nearest_node_index(p1[0], p1[1], node_x, node_y, snap_tol)
            if n0 < 0 or n1 < 0 or n0 == n1:
                continue
            prev = n0
            seg_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            n_samples = max(0, int(np.floor(seg_len / step_len)) - 1)
            for s in range(1, n_samples + 1):
                t = float(s) / float(n_samples + 1)
                sx = (1.0 - t) * p0[0] + t * p1[0]
                sy = (1.0 - t) * p0[1] + t * p1[1]
                ns = _nearest_node_index(sx, sy, node_x, node_y, snap_tol)
                if ns < 0 or ns == prev:
                    continue
                e = (prev, ns) if prev < ns else (ns, prev)
                edges.add(e)
                prev = ns
            if prev != n1:
                e = (prev, n1) if prev < n1 else (n1, prev)
                edges.add(e)
    return edges


def _constrained_edge_recovery_stats(mesh, model, snap_tol):
    node_x = np.asarray(mesh.node_x, dtype=np.float64)
    node_y = np.asarray(mesh.node_y, dtype=np.float64)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    conn = np.asarray(mesh.cell_face_nodes, dtype=np.int32)
    boundary_lines = _collect_region_boundary_lines(model)
    expected_edges = _collect_constrained_edges_sampled(node_x, node_y, boundary_lines, snap_tol)
    if not expected_edges:
        return 0, 0, 0.0
    mesh_edges = _build_face_edge_set(offs, conn)
    recovered = sum(1 for e in expected_edges if e in mesh_edges)
    ratio = float(recovered) / float(len(expected_edges))
    return recovered, len(expected_edges), ratio


def _constrained_edge_recovery_stats_for_lines(mesh, lines, snap_tol):
    node_x = np.asarray(mesh.node_x, dtype=np.float64)
    node_y = np.asarray(mesh.node_y, dtype=np.float64)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    conn = np.asarray(mesh.cell_face_nodes, dtype=np.int32)
    expected_edges = _collect_constrained_edges_sampled(node_x, node_y, lines, snap_tol)
    if not expected_edges:
        return 0, 0, 0.0
    mesh_edges = _build_face_edge_set(offs, conn)
    recovered = sum(1 for e in expected_edges if e in mesh_edges)
    ratio = float(recovered) / float(len(expected_edges))
    return recovered, len(expected_edges), ratio


@contextmanager
def _temporary_env(updates):
    old = {}
    sentinel = object()
    try:
        for k, v in updates.items():
            old[k] = os.environ.get(k, sentinel)
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestHybridCppChannelTransition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import hydra_hybridmesh  # noqa: F401
        except Exception:
            raise unittest.SkipTest("hydra_hybridmesh module is not built")

        cls.source = os.path.join(ROOT, "qgis_testing_project", "swe2d_model.gpkg")
        if not os.path.exists(cls.source):
            raise unittest.SkipTest("topology GeoPackage fixture not found")

        cls.model = load_conceptual_model(
            source=cls.source,
            arcs_layer="swe2d_topo_arcs",
            regions_layer="swe2d_topo_regions",
            constraints_layer="swe2d_topo_constraints",
            quad_edges_layer="swe2d_topo_quad_edges",
            default_size=20.0,
            default_cell_type="triangular",
        )

        cls.mesh = generate_face_centric_mesh(
            cls.model,
            backend="hybrid_cpp",
            options={
                "transition_width_factor": 1.25,
                "transition_outer_factor": 2.5,
                "overbank_grading_factor": 4.0,
            },
        )

        cls.node_x = np.asarray(cls.mesh.node_x, dtype=np.float64)
        cls.node_y = np.asarray(cls.mesh.node_y, dtype=np.float64)
        cls.offs = np.asarray(cls.mesh.cell_face_offsets, dtype=np.int32)
        cls.conn = np.asarray(cls.mesh.cell_face_nodes, dtype=np.int32)
        cls.region_id = np.asarray(cls.mesh.region_id, dtype=np.int32)
        cls.target_size = np.asarray(cls.mesh.target_size, dtype=np.float64)

        cls.face_nverts = np.asarray(
            [int(cls.offs[i + 1] - cls.offs[i]) for i in range(cls.offs.size - 1)],
            dtype=np.int32,
        )

        cx = np.zeros(cls.offs.size - 1, dtype=np.float64)
        cy = np.zeros(cls.offs.size - 1, dtype=np.float64)
        for i in range(cls.offs.size - 1):
            ids = cls.conn[int(cls.offs[i]): int(cls.offs[i + 1])]
            cx[i] = float(np.mean(cls.node_x[ids]))
            cy[i] = float(np.mean(cls.node_y[ids]))
        cls.centroid_x = cx
        cls.centroid_y = cy

        cls.channel_region_ids = {
            int(r.region_id)
            for r in cls.model.regions
            if str(r.default_cell_type).strip().lower() == "channel_generator"
        }
        cls.channel_target_size = {
            int(r.region_id): float(r.default_size)
            for r in cls.model.regions
            if str(r.default_cell_type).strip().lower() == "channel_generator"
        }

        cls.bank_lines = [
            [(float(x), float(y)) for (x, y) in arc.points_xy]
            for arc in cls.model.arcs
            if arc.points_xy
            and len(arc.points_xy) >= 2
            and str(arc.arc_role or "").strip().lower() in {"left_bank", "right_bank"}
        ]

        cls.dist_to_bank = np.asarray(
            [
                min(_polyline_distance(float(cx[i]), float(cy[i]), ln) for ln in cls.bank_lines)
                if cls.bank_lines
                else np.inf
                for i in range(cx.size)
            ],
            dtype=np.float64,
        )

    def test_expected_hybrid_counts_and_mixture(self):
        n_quads = int(np.sum(self.face_nverts == 4))
        n_tris = int(np.sum(self.face_nverts == 3))

        self.assertGreater(n_quads, 250)
        self.assertGreater(n_tris, 3000)
        self.assertGreater(n_tris, n_quads)

        channel_mask = np.isin(self.region_id, list(self.channel_region_ids))
        self.assertGreater(int(np.sum(channel_mask & (self.face_nverts == 4))), 100)
        self.assertGreater(int(np.sum(channel_mask & (self.face_nverts == 3))), 100)

    def test_transition_strip_and_overbank_grading_present(self):
        if not self.bank_lines:
            self.skipTest("bank arcs not found in topology fixture")

        channel_mask = np.isin(self.region_id, list(self.channel_region_ids))
        non_channel_tri = (~channel_mask) & (self.face_nverts == 3)

        base = 20.0
        if self.channel_target_size:
            base = float(np.mean(list(self.channel_target_size.values())))

        inner = 1.25 * base
        outer = 2.5 * base

        channel_tri_transition = channel_mask & (self.face_nverts == 3) & (self.dist_to_bank <= outer)
        channel_quad_core = channel_mask & (self.face_nverts == 4) & (self.dist_to_bank > inner)

        self.assertGreater(int(np.sum(channel_tri_transition)), 50)
        self.assertGreater(int(np.sum(channel_quad_core)), 50)

        d = self.dist_to_bank[non_channel_tri]
        ts = self.target_size[non_channel_tri]
        finite = np.isfinite(d)
        d = d[finite]
        ts = ts[finite]

        self.assertGreater(d.size, 50)
        q20 = float(np.quantile(d, 0.20))
        q80 = float(np.quantile(d, 0.80))

        near = ts[d <= q20]
        far = ts[d >= q80]
        self.assertGreater(near.size, 10)
        self.assertGreater(far.size, 10)
        self.assertLess(float(np.mean(near)), float(np.mean(far)))

    def test_channel_overbank_interfaces_share_edges(self):
        channel_regions = set(int(v) for v in self.channel_region_ids)
        self.assertGreater(len(channel_regions), 0)

        edge_regions = {}
        for face_idx in range(self.offs.size - 1):
            rid = int(self.region_id[face_idx])
            ids = self.conn[int(self.offs[face_idx]): int(self.offs[face_idx + 1])]
            if ids.size < 2:
                continue
            for i in range(ids.size):
                a = int(ids[i])
                b = int(ids[(i + 1) % ids.size])
                e = (a, b) if a < b else (b, a)
                edge_regions.setdefault(e, set()).add(rid)

        shared_channel_to_nonchannel = 0
        for rset in edge_regions.values():
            if len(rset) < 2:
                continue
            has_channel = any(r in channel_regions for r in rset)
            has_non_channel = any(r not in channel_regions for r in rset)
            if has_channel and has_non_channel:
                shared_channel_to_nonchannel += 1

        self.assertGreater(
            shared_channel_to_nonchannel,
            10,
            msg="Expected conforming shared edges between channel and overbank regions",
        )

    def test_runtime_triangle_meshing_methods(self):
        methods = [
            "frontal_delaunay",
            "direct_curvilinear_generation",
            "anisotropic_delaunay_refinement",
        ]
        tri_counts = {}
        for method in methods:
            mesh = generate_face_centric_mesh(
                self.model,
                backend="hybrid_cpp",
                options={
                    "tri_meshing_method": method,
                    "transition_width_factor": 1.25,
                    "transition_outer_factor": 2.5,
                    "overbank_grading_factor": 4.0,
                },
            )
            offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
            self.assertGreater(int(offs.size - 1), 100)
            face_sizes = np.asarray([int(offs[i + 1] - offs[i]) for i in range(offs.size - 1)], dtype=np.int32)
            tri_counts[method] = int(np.sum(face_sizes == 3))
            self.assertGreater(tri_counts[method], 100)

        self.assertGreaterEqual(
            tri_counts["anisotropic_delaunay_refinement"],
            tri_counts["frontal_delaunay"],
            msg="Anisotropic refinement should not reduce triangle count versus frontal baseline",
        )

    def test_constrained_edge_recovery_survives_quality_smoothing(self):
        snap_tol = 12.0
        base_options = {
            "tri_meshing_method": "advancing_front",
            "transition_width_factor": 1.25,
            "transition_outer_factor": 2.5,
            "overbank_grading_factor": 4.0,
            "hybridcpp_constrained_edge_snap_tol": snap_tol,
            "hybridcpp_constrained_edge_max_flips": 512,
        }

        env_stress = {
            "HYDRA_HYBRIDCPP_QUALITY_SWEEPS": "12",
            "HYDRA_HYBRIDCPP_QUALITY_RELAX": "0.60",
            "HYDRA_HYBRIDCPP_QUALITY_MAX_MOVE_FACTOR": "0.80",
        }

        with _temporary_env({**env_stress, "HYDRA_HYBRIDCPP_QUALITY_LOCK_TOL_FACTOR": "0.00"}):
            mesh_unlocked = generate_face_centric_mesh(
                self.model,
                backend="hybrid_cpp",
                options=dict(base_options),
            )

        with _temporary_env({**env_stress, "HYDRA_HYBRIDCPP_QUALITY_LOCK_TOL_FACTOR": "0.20"}):
            mesh_locked = generate_face_centric_mesh(
                self.model,
                backend="hybrid_cpp",
                options=dict(base_options),
            )

        rec_u, tot_u, ratio_u = _constrained_edge_recovery_stats(mesh_unlocked, self.model, snap_tol)
        rec_l, tot_l, ratio_l = _constrained_edge_recovery_stats(mesh_locked, self.model, snap_tol)

        self.assertGreater(tot_u, 100, msg="Need enough constrained edges in fixture for robust regression")
        self.assertEqual(tot_u, tot_l)

        self.assertGreater(
            ratio_l,
            0.60,
            msg=(
                "Locked smoothing should preserve a substantial fraction of constrained edges "
                f"(locked={ratio_l:.3f}, recovered={rec_l}/{tot_l})"
            ),
        )
        self.assertGreaterEqual(
            ratio_l,
            ratio_u,
            msg=(
                "Boundary lock tolerance should not reduce constrained-edge recovery under stress "
                f"(unlocked={ratio_u:.3f} [{rec_u}/{tot_u}], locked={ratio_l:.3f} [{rec_l}/{tot_l}])"
            ),
        )

    def test_faces_conform_to_region_and_arc_constraints(self):
        snap_tol = 12.0
        opts = {
            "tri_meshing_method": "advancing_front",
            "transition_width_factor": 1.25,
            "transition_outer_factor": 2.5,
            "overbank_grading_factor": 4.0,
            "hybridcpp_constrained_edge_snap_tol": snap_tol,
            "hybridcpp_constrained_edge_max_flips": 768,
        }
        env_cfg = {
            "HYDRA_HYBRIDCPP_REGION_CONFORMANCE_BAND_FACTOR": "0.90",
            "HYDRA_HYBRIDCPP_ARC_CONFORMANCE_BAND_FACTOR": "0.90",
            "HYDRA_HYBRIDCPP_QUALITY_LOCK_TOL_FACTOR": "0.20",
        }
        with _temporary_env(env_cfg):
            mesh = generate_face_centric_mesh(
                self.model,
                backend="hybrid_cpp",
                options=opts,
            )

        region_lines = _collect_region_boundary_lines(self.model)
        arc_lines = _collect_arc_lines(self.model)
        rr, rt, rratio = _constrained_edge_recovery_stats_for_lines(mesh, region_lines, snap_tol)
        ar, at, aratio = _constrained_edge_recovery_stats_for_lines(mesh, arc_lines, snap_tol)

        self.assertGreater(rt, 120, msg="Need enough region constrained edges for conformance regression")
        self.assertGreater(at, 80, msg="Need enough arc constrained edges for conformance regression")

        self.assertGreater(
            rratio,
            0.62,
            msg=f"Region boundary conformance ratio too low: {rratio:.3f} ({rr}/{rt})",
        )
        self.assertGreater(
            aratio,
            0.50,
            msg=f"Arc conformance ratio too low: {aratio:.3f} ({ar}/{at})",
        )


if __name__ == "__main__":
    unittest.main()
