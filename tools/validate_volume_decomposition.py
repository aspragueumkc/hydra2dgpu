"""Validate volume decomposition against Preissmann slot and SWMM.

Compares three surcharge approaches on a surcharging pipe network:
  1. Preissmann slot (existing prototype)
  2. Volume decomposition (excess → node)
  3. SWMM reference (via pyswmm)
"""
import numpy as np, math, sys, os, tempfile, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.test_1d_fvm_prototype import (
    PipeSection, hllc_flux, build_mesh, mc, H_MIN, GRAVITY,
    RECTANGULAR, CIRCULAR, ELLIPTICAL,
)

# ── Volume-decomposition node model ────────────────────────────────────────

class Node:
    """Storage node that receives surcharge volume from connected pipes."""
    def __init__(self, invert, surface_area, max_depth, rim_elev):
        self.invert = invert
        self.surface_area = surface_area
        self.max_depth = max_depth
        self.rim = rim_elev
        self.vol = 0.0  # accumulated surcharge volume (ft³)
        self.wse = invert

    def add_volume(self, dV):
        self.vol += dV
        self.wse = self.invert + self.vol / max(self.surface_area, 1e-12)

    def head_above_invert(self):
        return max(self.wse - self.invert, 0.0)

    def overflow_rate(self, g=GRAVITY):
        """Weir overflow when WSE > rim."""
        if self.wse <= self.rim:
            return 0.0
        H = self.wse - self.rim
        return 1.84 * math.sqrt(self.surface_area) * H ** 1.5


# ── Volume-decomposition Godunov step ─────────────────────────────────────

def vd_godunov_step(sec, A, Q, dx, nxf, dt, nodes, link_end_nodes):
    """RK2 step with volume decomposition: A never exceeds A_full.
    Excess volume is pushed to the node cell at each link end.
    """
    N = len(A)

    def rhs(AA, QQ):
        fA, fQ = np.zeros(N + 1), np.zeros(N + 1)
        for k in range(N + 1):
            fA[k], fQ[k] = hllc_flux(sec, AA[k], QQ[k], AA[k + 1], QQ[k + 1], nxf[k])
        dA = np.zeros(N)
        dQ = np.zeros(N)
        for c in range(N):
            dA[c] = -(fA[c + 1] - fA[c]) / dx[c]
            dQ[c] = -(fQ[c + 1] - fQ[c]) / dx[c]
        return dA, dQ, fA, fQ

    # Build ghost states including node connections
    A_ext = np.concatenate([[A[0]], A, [A[-1]]])
    Q_ext = np.concatenate([[Q[0]], Q, [Q[-1]]])

    dA1, dQ1, fA1, fQ1 = rhs(A_ext, Q_ext)
    As = A + dt * dA1[1:-1]
    Qs = Q + dt * dQ1[1:-1]

    # Clamp A at A_full, push excess to nodes (stage 1)
    for c in range(N):
        if As[c] > sec.A_full:
            excess = (As[c] - sec.A_full) * dx[c]
            As[c] = sec.A_full
            # Push to node at link end (simplified: both ends equally)
            for ni in link_end_nodes:
                nodes[ni].add_volume(excess * 0.5)

    A_ext2 = np.concatenate([[As[0]], As, [As[-1]]])
    Q_ext2 = np.concatenate([[Qs[0]], Qs, [Qs[-1]]])
    dA2, dQ2, fA2, fQ2 = rhs(A_ext2, Q_ext2)

    An = A + 0.5 * (dA1[1:-1] + dA2[1:-1]) * dt
    Qn = Q + 0.5 * (dQ1[1:-1] + dQ2[1:-1]) * dt

    # Clamp at A_full (stage 2)
    for c in range(N):
        if An[c] > sec.A_full:
            excess = (An[c] - sec.A_full) * dx[c]
            An[c] = sec.A_full
            for ni in link_end_nodes:
                nodes[ni].add_volume(excess * 0.5)

    An = np.maximum(An, 0.0)
    return An, Qn


# ── SWMM model builder ────────────────────────────────────────────────────

def build_swmm_model(inp_path, link_lengths, diameters, manning_n,
                     node_inverts, node_surface_areas, inflow_cfs):
    """Write a SWMM input file for a series link network."""
    with open(inp_path, 'w') as f:
        f.write(textwrap.dedent(f"""\
        [TITLE]
        Volume decomposition validation

        [OPTIONS]
        FLOW_UNITS CFS
        INFILTRATION HORTON
        FLOW_ROUTING DYNWAVE
        START_DATE 01/01/2023
        START_TIME 00:00:00
        REPORT_START_DATE 01/01/2023
        REPORT_START_TIME 00:00:00
        END_DATE 01/01/2023
        END_TIME 00:01:00
        SWEEP_START 01/01
        SWEEP_END 12/31
        DRY_DAYS 0
        REPORT_STEP 00:00:01
        WET_STEP 00:00:01
        DRY_STEP 00:00:01
        ROUTING_STEP 0:00:01
        ALLOW_PONDING YES
        INERTIAL_DAMPING PARTIAL
        SLOPE WEIGHTING 0.5
        VARIABLE_STEP 0.75
        LENGTHENING_STEP 0
        MIN_SURFAREA 0
        NORMAL_FLOW_LIMITED BOTH
        SKIP_STEADY_STATE NO
        FORCE_MAIN_EQUATION H-W
        LINK_OFFSETS DEPTH
        MIN_SLOPE 0

        [JUNCTIONS]
        """))

        nn = len(node_inverts)
        nl = len(link_lengths)
        for i in range(nn):
            sa = node_surface_areas[i] if i < len(node_surface_areas) else 10.0
            f.write(f"  J{i}  {node_inverts[i]:.2f}  0  {sa:.2f}\n")

        f.write("\n[OUTFALLS]\n")
        f.write(f"  J{nn - 1}  {node_inverts[-1]:.2f}  FREE\n")

        f.write("\n[CONDUITS]\n")
        for i in range(nl):
            f.write(f"  C{i}  J{i}  J{i + 1}  {link_lengths[i]:.2f}  "
                    f"{diameters[i]:.3f}  0.001  {manning_n:.4f}\n")

        f.write("\n[XSECTIONS]\n")
        for i in range(nl):
            f.write(f"  C{i}  CIRCULAR  {diameters[i]:.3f}  0  0\n")

        if inflow_cfs > 0:
            f.write("\n[FLOW]\n")
            f.write(f"  J0  {inflow_cfs:.2f}\n")

        f.write("\n[TIMESERIES]\n")
        f.write("  Q_in  0  0.0  0.01  {inflow_cfs:.2f}\n")
        f.write("\n[REPORT]\n")
        f.write("  INPUT YES\n")
        f.write("  CONTROLS NO\n")
        f.write("  SUBCATCHMENTS NO\n")
        f.write("  NODES ALL\n")
        f.write("  LINKS ALL\n")


# ── Run comparison ─────────────────────────────────────────────────────────

def run_comparison():
    print("=" * 72)
    print("Volume Decomposition Validation")
    print("=" * 72)

    # Network: 2 links, 3 nodes. Node0 inflow, Node1 manhole, Node2 outfall.
    # Link0: D=1.0 ft, L=100 ft, 5 cells
    # Link1: D=1.0 ft, L=100 ft, 5 cells
    L = 100.0
    N = 10  # 5 cells per link
    xc, dx, nxf = build_mesh(L, N)
    sec = PipeSection(CIRCULAR, 1.0, 1.0, slot_width=0.01)

    Q_in = 5.0  # CFS — exceeds capacity of 1ft pipe at any reasonable slope

    # ── Slot prototype ──
    A0 = sec.area(np.full(N, 0.1))
    Q0 = np.zeros(N)
    A, Q = A0.copy(), Q0.copy()
    t, dt = 0.0, 0.01
    slot_A_hist, slot_Q_hist = [], []
    for step in range(500):
        # Ghost BCs: inflow at left, free outfall at right
        A_ext = np.concatenate([[A0[0]], A, [A[-1]]])
        Q_ext = np.concatenate([[Q_in], Q, [Q[-1]]])
        fA, fQ = np.zeros(N + 1), np.zeros(N + 1)
        for k in range(N + 1):
            fA[k], fQ[k] = hllc_flux(sec, A_ext[k], Q_ext[k], A_ext[k + 1], Q_ext[k + 1], nxf[k])
        dA = np.array([-(fA[c + 1] - fA[c]) / dx[c] for c in range(N)])
        A = np.maximum(A + dt * dA, 0.0)
        t += dt
        if step % 50 == 0:
            slot_A_hist.append(A.copy())
            slot_Q_hist.append(fA.copy())

    # ── Volume decomposition prototype ──
    nodes = [
        Node(invert=10.0, surface_area=10.0, max_depth=5.0, rim_elev=15.0),  # J0
        Node(invert=9.5, surface_area=10.0, max_depth=5.0, rim_elev=14.5),   # J1 (manhole)
        Node(invert=9.0, surface_area=1e10, max_depth=10.0, rim_elev=19.0),  # J2 (outfall)
    ]
    # Link end nodes: link0 connects J0-J1, link1 connects J1-J2
    link_ends = [[0], [2]]
    A_vd = A0.copy()
    Q_vd = Q0.copy()
    t = 0.0
    vd_node_vol_hist = []
    for step in range(500):
        A_ext = np.concatenate([[A_vd[0]], A_vd, [A_vd[-1]]])
        Q_ext = np.concatenate([[Q_in], Q_vd, [Q_vd[-1]]])
        dA = np.zeros(N)
        fA = np.zeros(N + 1)
        for k in range(N + 1):
            fA[k], _ = hllc_flux(sec, A_ext[k], Q_ext[k], A_ext[k + 1], Q_ext[k + 1], nxf[k])
        for c in range(N):
            dA[c] = -(fA[c + 1] - fA[c]) / dx[c]
        A_vd = A_vd + dt * dA
        Q_vd = Q_vd + dt * dA * 0.0  # momentum skipped for simplicity

        # Volume decomposition: clamp at A_full, push excess to node
        for c in range(N):
            if A_vd[c] > sec.A_full:
                excess = (A_vd[c] - sec.A_full) * dx[c]
                A_vd[c] = sec.A_full
                # Link0 cells push to node 0, link1 cells to node 2
                ni = 0 if c < 5 else 2
                nodes[ni].add_volume(excess)

        A_vd = np.maximum(A_vd, 0.0)
        t += dt
        if step % 50 == 0:
            vd_node_vol_hist.append([n.vol for n in nodes])

    print(f"\nAfter 500 steps (t={t:.1f}s):")
    pipe_vol_slot = float(np.sum(A * dx))
    pipe_vol_vd = float(np.sum(A_vd * dx))
    node_vol_vd = float(sum(n.vol for n in nodes))
    print(f"  Slot:     pipe_vol={pipe_vol_slot:.2f}")
    print(f"  VolDec:   pipe_vol={pipe_vol_vd:.2f}  node_vol={node_vol_vd:.2f}  total={pipe_vol_vd+node_vol_vd:.2f}")
    print(f"  Inflow volume (5 CFS × {t:.1f}s) = {Q_in * t:.1f}")
    print(f"  Slot mass error: {Q_in * t - pipe_vol_slot:.2f}")
    print(f"  VolDec mass error: {Q_in * t - (pipe_vol_vd + node_vol_vd):.2f}")

    # ── SWMM reference ──
    try:
        from pyswmm import Simulation, Nodes, Links
        tmp = tempfile.NamedTemporaryFile(suffix='.inp', delete=False, mode='w')
        build_swmm_model(tmp.name, [100.0, 100.0], [1.0, 1.0], 0.013,
                         [10.0, 9.5, 9.0], [10.0, 10.0, 1e10], inflow_cfs=Q_in)
        tmp.close()
        with Simulation(tmp.name) as sim:
            swmm_node_depth = []
            for _ in range(500):
                sim.step_advance()
                j1 = Nodes(sim)['J1']
                swmm_node_depth.append(j1.depth)
        os.unlink(tmp.name)
        print(f"\n  SWMM J1 final depth: {swmm_node_depth[-1]:.2f} ft")
    except Exception as e:
        print(f"\n  SWMM comparison skipped: {e}")


if __name__ == "__main__":
    run_comparison()
