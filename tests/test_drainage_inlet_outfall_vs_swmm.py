"""Validate pipe1d solver against SWMM with an inlet node upstream
and an outfall node downstream (1-link network).

Topology:
    Upstream junction (J1) -- conduit C1 --> Downstream outfall (O2)

Both solvers are driven with the SAME lateral inflow (INLET_INFLOW_CMS)
at J1 and run to steady state. The comparison is on the steady-state
upstream depth:

* SWMM side: junction J1 with constant 0.5 m³/s lateral inflow,
  outfall O2, run to steady state, read J1 depth.

* hydra_swe2d side: drive the same geometry through
  ``SWE2DCouplingController``. Each step injects INLET_INFLOW_CMS into
  J1's node storage (depth += inflow * dt / surface_area) before the
  pipe step. Run to steady state, read J1 depth.

The test compares depths. If pipe1d's open-channel steady-state depth
matches SWMM within a factor of 2, pipe1d solves open-channel pipe flow
correctly enough to be a usable surrogate for SWMM in this regime.

This is a *new* test that exercises the coupling-controller path rather
than the standalone ``Pipe1DRunner`` used in
``tests/test_pipe1d_vs_swmm.py``.  It exists to detect regressions in
the ``SWE2DCouplingController`` -> pipe1d GPU code path specifically
when an outfall node is part of the network.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np


def _gpu_available() -> bool:
    try:
        import hydra_swe2d as m
        return bool(m.swe2d_gpu_available())
    except Exception:
        return False


def _has_swmm_toolkit() -> bool:
    return importlib.util.find_spec("swmm") is not None


# SI mesh + SI pipe1d units; consistent with the rest of the test suite
# and with ``_make_rect_mesh`` (which produces node coordinates in metres).
G_SI = 9.81
K_MANN_SI = 1.0          # 1.0 → Manning equation in SI form
H_MIN = 1.0e-4

PIPE_D = 1.0             # m
PIPE_L = 100.0           # m
PIPE_N = 0.013           # Manning n
INLET_INFLOW_CMS = 0.5   # m³/s steady lateral inflow at J1
J1_SURFACE_AREA = 50.0   # m² (matches SWMM default; bigger than 1.0
                         #  so depth rises slowly enough to reach steady
                         #  state in a reasonable number of steps)


def _build_swmm_results() -> tuple[float, float]:
    """Run the SWMM side to steady state and return ``(Q_swmm, d_swmm)``.

    ``Q_swmm`` is the mean conduit flow over the last 20 steps and
    ``d_swmm`` is the mean junction depth over the last 20 steps.
    """
    from tests.swmm_runner import SWMMRunner, make_drainage_inp

    inp = make_drainage_inp(
        junctions=[("J1", 0.0, 10.0)],
        outfalls=[("O2", -PIPE_L)],
        conduits=[("C1", "J1", "O2", PIPE_L, PIPE_N, PIPE_D)],
        xsections=[("C1", "CIRCULAR", PIPE_D)],
        inflows=[("J1", "TS1")],
        timeseries=[("TS1", 0, INLET_INFLOW_CMS),
                    ("TS1", 3600, INLET_INFLOW_CMS)],
        end_time="01:00:00",
        routing_step_s=30.0,
    )
    runner = SWMMRunner()
    _, nodes, links = runner.run(inp, max_steps=120)

    flows = [r.flow for r in links["C1"]]
    depths = [j.depth for j in nodes["J1"]]
    if not flows or not depths:
        raise RuntimeError("SWMM produced no output for inlet/outfall model")

    # Last 20 records = steady state (last 20 routing steps × 30 s = 10 min)
    swmm_q = float(np.mean(flows[-20:]))
    swmm_depth = float(np.mean(depths[-20:]))
    return swmm_q, swmm_depth


def _build_hydra_swe2d_runner():
    """Build a SWE2D + drainage coupling runner and return its handles."""
    from swe2d.extensions.drainage_network import (
        DrainageNode,
        DrainageLink,
        PipeNetworkConfig,
        SWE2DUrbanDrainageModule,
    )
    from swe2d.runtime.backend import SWE2DBackend
    from swe2d.runtime.coupling import SWE2DCouplingController
    from tests._swe2d_test_helpers import _make_rect_mesh

    backend = SWE2DBackend()
    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(2, 1, 20.0, 10.0)
    backend.build_mesh(
        node_x, node_y, node_z, cell_nodes,
        bc_edge_node0=np.empty(0, dtype=np.int32),
        bc_edge_node1=np.empty(0, dtype=np.int32),
        bc_edge_type=np.empty(0, dtype=np.int32),
        bc_edge_val=np.empty(0, dtype=np.float64),
    )
    n_cells = int(backend.n_cells)
    backend.initialize(
        h0=np.full(n_cells, 0.0, dtype=np.float64),
        hu0=np.zeros(n_cells, dtype=np.float64),
        hv0=np.zeros(n_cells, dtype=np.float64),
        h_min=H_MIN,
        g=G_SI,
        k_mann=K_MANN_SI,
        cfl=0.45,
        dt_max=0.5,
        dt_fixed=0.5,
    )

    nodes = [
        DrainageNode(
            node_id="J1", x=0.0, y=0.0,
            invert_elev=0.0, max_depth=10.0,
            node_type="junction",
            # Surface area used to convert lateral inflow (m³/s) into a
            # depth increment per timestep.  Matches SWMM junction
            # default storage so the dynamics are comparable.
            metadata={"surface_area": J1_SURFACE_AREA},
        ),
        DrainageNode(
            node_id="O2", x=PIPE_L, y=0.0,
            invert_elev=-PIPE_L, max_depth=10.0,
            node_type="outfall",
        ),
    ]
    links = [
        DrainageLink(
            link_id="C1",
            from_node_id="J1", to_node_id="O2",
            length=PIPE_L,
            roughness_n=PIPE_N,
            diameter=PIPE_D,
            entrance_loss_k=0.5,
            exit_loss_k=1.0,
            inlet_invert_elev=0.0,
            outlet_invert_elev=-PIPE_L,
            max_cell_length=25,  # PIPE_L/25 = 4 sub-cells
        ),
    ]
    cfg = PipeNetworkConfig(
        enabled=True,
        nodes=nodes,
        links=links,
        gravity=G_SI,
        head_deadband_m=1.0e-3,
        pipe_solver_mode="diffusion_wave",
        coupling_substeps=1,
    )
    drain_mod = SWE2DUrbanDrainageModule(cfg)
    drain_mod.initialize()

    cc = SWE2DCouplingController(
        cell_area=backend.cell_areas(),
        cell_bed=np.zeros(n_cells, dtype=np.float64),
        drainage=drain_mod,
        backend=backend,
        h_min=backend.h_min,
        length_scale_si_to_model=1.0,
    )
    return backend, cc, drain_mod


def _drive_pipe1d_with_inflow_to_steady(
    cc,
    *,
    inflow_m3s: float,
    surface_area_m2: float,
    n_steps: int = 5000,
    dt: float = 0.5,
) -> tuple[float, float]:
    """Inject ``inflow_m3s`` at node 0 every step and run the pipe1d
    solver.  Return ``(steady_depth_node0, steady_link_flow)``.

    Each iteration:
      1. Run one pipe1d step.
      2. Read back the post-step ``cell_depth`` for the upstream cell via
         the production coupling readback (which honours ``n_pipe_cells``).
      3. Add the inflow volume (``inflow_m3s * dt``) by incrementing
         the depth by ``inflow_m3s * dt / surface_area_m2``.
      4. Upload the new depth as the cell-depth boundary condition for
         the next step (per-cell schema; the legacy per-node ``node_depth``
         state has been collapsed in commit ce74f7d / F7).

    At steady state the conduit carries flow equal to the inflow and
    ``depth`` stabilises where the head differential drives exactly
    ``inflow_m3s`` through the conduit.
    """
    native_mod = cc._native_cuda_module()
    if native_mod is None:
        raise RuntimeError("native CUDA module not available")
    if not hasattr(native_mod, "swe2d_pipe1d_step"):
        raise RuntimeError("pipe1d step binding missing")
    if not hasattr(native_mod, "swe2d_pipe1d_upload_cell_h"):
        raise RuntimeError("pipe1d upload-cell-h binding missing")
    if not hasattr(native_mod, "swe2d_pipe1d_init_cell_area"):
        raise RuntimeError("pipe1d init-cell-area binding missing")

    dev_ptr = int(native_mod.swe2d_get_coupling_dev_ptr())
    # With one link and no subdivision the unified mesh has one pipe
    # cell; seed cell_h directly.
    cell_h = np.zeros(1, dtype=np.float64)

    # Initialise cell areas from a tiny starting depth so the first
    # step has a non-degenerate area.
    native_mod.swe2d_pipe1d_upload_cell_h(dev_ptr, cell_h)
    native_mod.swe2d_pipe1d_init_cell_area(dev_ptr, H_MIN)

    depth_increment = inflow_m3s * dt / surface_area_m2
    for _ in range(n_steps):
        # Step first: the kernel drains the upstream cell via mass
        # balance.  Then we add inflow and upload for the next step.
        native_mod.swe2d_pipe1d_step(
            dev_ptr, dt, "diffusion_wave",
            1, 2, 0.5,
            G_SI, K_MANN_SI, H_MIN,
        )
        rb = cc.readback_coupling_state()
        cell_depth_arr = rb.get("cell_depth") if rb is not None else None
        cur_depth = (
            float(cell_depth_arr[0])
            if cell_depth_arr is not None and cell_depth_arr.size > 0
            else 0.0
        )
        cell_h[0] = max(0.0, cur_depth + depth_increment)
        native_mod.swe2d_pipe1d_upload_cell_h(dev_ptr, cell_h)

    state = cc.readback_coupling_state()
    cell_depth_arr = state.get("cell_depth") if state is not None else None
    steady_depth = (
        float(cell_depth_arr[0])
        if cell_depth_arr is not None and cell_depth_arr.size > 0
        else 0.0
    )
    link_flow = float(np.mean(np.abs(state["link_q"]))) \
        if state is not None and "link_q" in state and state["link_q"].size > 0 \
        else 0.0
    return steady_depth, link_flow


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestDrainageInletOutfallVsSWMM(unittest.TestCase):
    """1-link network: inlet (junction) upstream, outfall downstream."""

    @unittest.skipUnless(_has_swmm_toolkit(), "swmm-toolkit not installed")
    def test_inlet_outfall_1_link_depth_matches_swmm(self):
        """With identical lateral inflow at J1, pipe1d's steady-state
        upstream depth matches SWMM within a factor of 2.

        If this test fails, pipe1d cannot solve open-channel pipe flow
        correctly enough to substitute for SWMM in this regime — a
        fundamental accuracy gap, not a regression in this PR.
        """
        swmm_q, swmm_depth = _build_swmm_results()

        self.assertGreater(swmm_q, 1.0e-3,
                           f"SWMM produced zero Q (depth={swmm_depth:.3f}); "
                           "test setup is degenerate")
        self.assertGreater(swmm_depth, 1.0e-3,
                           f"SWMM produced zero upstream depth; "
                           "test setup is degenerate")

        backend, cc, _drain_mod = _build_hydra_swe2d_runner()
        try:
            pipe_depth, pipe_q = _drive_pipe1d_with_inflow_to_steady(
                cc,
                inflow_m3s=INLET_INFLOW_CMS,
                surface_area_m2=J1_SURFACE_AREA,
                n_steps=5000,
                dt=0.5,
            )
        finally:
            try:
                cc._native_cuda_module()
            except Exception:
                pass

        # Both solvers must produce a non-trivial upstream depth.
        self.assertGreater(pipe_depth, 1.0e-3,
                           f"pipe1d produced no upstream depth "
                           f"(Q={pipe_q:.4f} m³/s)")
        self.assertGreater(pipe_q, 1.0e-3,
                           f"pipe1d produced no flow "
                           f"(depth={pipe_depth:.4f} m)")

        ratio = pipe_depth / max(1.0e-10, swmm_depth)
        # Within a factor of 2: open-channel pipe flow must agree on the
        # order of magnitude.  Tighten as pipe1d accuracy improves.
        self.assertLess(
            ratio, 2.0,
            msg=(f"pipe1d depth > 2x SWMM depth "
                 f"(pipe1d={pipe_depth:.4f} m, "
                 f"SWMM={swmm_depth:.4f} m, "
                 f"ratio={ratio:.3f}, "
                 f"pipe1d Q={pipe_q:.4f} m³/s, "
                 f"SWMM Q={swmm_q:.4f} m³/s, "
                 f"backend.h_min={backend.h_min}) — pipe1d is "
                 f"systematically over-predicting depth at this inflow"),
        )
        self.assertGreater(
            ratio, 0.5,
            msg=(f"pipe1d depth < 0.5x SWMM depth "
                 f"(pipe1d={pipe_depth:.4f} m, "
                 f"SWMM={swmm_depth:.4f} m, "
                 f"ratio={ratio:.3f}, "
                 f"pipe1d Q={pipe_q:.4f} m³/s, "
                 f"SWMM Q={swmm_q:.4f} m³/s, "
                 f"backend.h_min={backend.h_min}) — pipe1d is "
                 f"systematically under-predicting depth at this inflow"),
        )


if __name__ == "__main__":
    unittest.main()
