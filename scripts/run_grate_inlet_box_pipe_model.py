#!/usr/bin/env python3
"""
Pipe1D grate-inlet → box-conduit → pipe-end → rectangular channel.
Exercises the refactored unified face-flux coupling (coupling.py pattern).

Each "logical rect" = 2 triangular cells.  We have 5 rects:
  rect 0: pool at inlet (bed 924 ft)
  rect 1: channel start / pipe-end (bed ~914 ft)
  rect 2-4: rest of channel (slope 0.003)
Downstream edge of rect 4 = tide BC.

Run:  mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \\
          python3 scripts/run_grate_inlet_box_pipe_model.py
"""
from __future__ import annotations
import sys, math
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "build"))

G = 32.174; K_MANN = 1.486; H_MIN = 1e-4
N_RECT = 5                  # 1 pool + 4 channel
N_TRI = N_RECT * 2          # triangular cells for 2D solver
CHAN_LEN = 200.0            # ft
CHAN_WID = 25.0             # ft (rectangular)
SLOPE = 0.003
POOL_BED = 924.0
PE_BED = 914.0
DN_BED = PE_BED - CHAN_LEN * SLOPE
ROW = N_RECT + 1            # nodes per row

# Pipe: box 8x4, L=553.3 ft, n=0.015
N_SUB = int(math.ceil(553.3 / 50))

def main():
    import hydra_swe2d as mod
    if not mod.swe2d_gpu_available(): print("ERROR: no GPU"); return 1
    print("=== Model: Inlet → Box Conduit → Pipe-End → Channel ===\n")

    # ── 2-D rectangular mesh ──────────────────────────────────────────────
    dx = CHAN_LEN / (N_RECT - 1)  # channel part only
    xs = np.linspace(0, CHAN_LEN + 50, ROW)  # 0 .. 250, 6 nodes
    # bed at each column
    zb_col = np.array([POOL_BED] +
                      [PE_BED - SLOPE * i * dx for i in range(N_RECT - 1)] +
                      [DN_BED])
    # 2 rows of nodes → flat array
    nx = np.concatenate([xs, xs]).astype(np.float64)
    ny = np.concatenate([np.zeros(ROW), np.full(ROW, CHAN_WID)]).astype(np.float64)
    nz = np.concatenate([zb_col, zb_col]).astype(np.float64)
    # cells: 1 rect = 2 triangles (i,i+1,i+ROW+1) (i,i+ROW+1,i+ROW)
    tris = []
    for r in range(N_RECT):
        b0 = r; b1 = r+1; t0 = r+ROW; t1 = r+1+ROW
        tris += [b0, b1, t1, b0, t1, t0]
    cell_nodes = np.array(tris, dtype=np.int32)
    # downstream edge BC
    bc_n0 = np.array([ROW-1], dtype=np.int32)
    bc_n1 = np.array([2*ROW-1], dtype=np.int32)
    bc_ty = np.array([2], dtype=np.int32)
    bc_va = np.array([0.0], dtype=np.float64)
    mesh = mod.swe2d_build_mesh(nx, ny, nz, cell_nodes,
                                bc_n0, bc_n1, bc_ty, bc_va)
    info = mod.swe2d_mesh_info(mesh)
    assert info["n_cells"] == N_TRI, f"{info['n_cells']} != {N_TRI}"

    # ── GPU solver ────────────────────────────────────────────────────────
    h0 = np.full(N_TRI, 0.001, dtype=np.float64)
    solver = mod.swe2d_create_solver(mesh, h0, n_mann=0.035, cfl=0.45,
                                     dt_max=2.0, use_gpu=True,
                                     enable_hydraulic_structures=True)
    dev_ptr = int(mod.swe2d_get_coupling_dev_ptr())

    # initial 2D state: pool cell h≈7.8 ft (WSE=931.8), rest dry
    pool_h = 931.8 - POOL_BED
    h_init = np.array([pool_h, pool_h] + [0.001]*2 + [0.001]*2 + [0.001]*2 + [0.001]*2,
                      dtype=np.float64)
    mod.swe2d_set_state(solver, h_init,
                        np.zeros(N_TRI, dtype=np.float64),
                        np.zeros(N_TRI, dtype=np.float64))

    # ── Pipe1D mesh ───────────────────────────────────────────────────────
    mod.swe2d_build_unified_mesh(
        dev_ptr=dev_ptr, n_links=1,
        link_from=np.array([1], dtype=np.int32),
        link_to=np.array([2], dtype=np.int32),
        L=np.array([553.3], dtype=np.float64),
        D=np.array([8.0], dtype=np.float64),
        n_mann=np.array([0.015], dtype=np.float64),
        S0=np.zeros(1, dtype=np.float64),
        node_invert=np.array([924.0, 914.0], dtype=np.float64),
        mcl=50.0,
        n_pipe_ends=2,
        pipe_end_node_ids=np.array([1, 2], dtype=np.int32),
        link_shape_type=np.array([1], dtype=np.int32),
        link_width=np.array([8.0], dtype=np.float64),
        link_height=np.array([4.0], dtype=np.float64),
    )

    # ── Pipe-end → 2D cell coupling (node 1→tri 0, node 2→tri 2) ──────────
    mod.swe2d_pipe1d_upload_pipe_end_surface_faces(
        dev_ptr, np.array([0, 2], dtype=np.int32))

    # ── Drainage exchange params ──────────────────────────────────────────
    mod.swe2d_gpu_upload_drainage_exchange_params(
        inlet_cell=np.array([0], dtype=np.int32),
        inlet_node=np.array([1], dtype=np.int32),
        inlet_crest=np.array([924.0], dtype=np.float64),
        inlet_width=np.array([10.0], dtype=np.float64),
        inlet_cd=np.array([0.67], dtype=np.float64),
        inlet_qmax=np.array([1e6], dtype=np.float64),
        inlet_type=np.array([1], dtype=np.int32),
        inlet_grate_len=np.array([10/12], dtype=np.float64),
        inlet_grate_wid=np.array([5/12], dtype=np.float64),
        inlet_grate_kind=np.array([0], dtype=np.int32),
        inlet_grate_open=np.array([0.9], dtype=np.float64),
        inlet_curb_len=np.array([0.0], dtype=np.float64),
        inlet_curb_ht=np.array([0.0], dtype=np.float64),
        inlet_curb_throat=np.array([0], dtype=np.int32),
        inlet_slot_len=np.array([0.0], dtype=np.float64),
        inlet_slot_wid=np.array([0.0], dtype=np.float64),
        outfall_cell=np.array([N_TRI-1], dtype=np.int32),
        outfall_node=np.array([0], dtype=np.int32),
        outfall_invert=np.array([DN_BED], dtype=np.float64),
        outfall_diameter=np.array([1.0], dtype=np.float64),
        outfall_cd=np.array([1.0], dtype=np.float64),
        outfall_qmax=np.array([1e6], dtype=np.float64),
        outfall_zero_storage=np.array([1], dtype=np.int32),
        pipe_end_cell=np.array([2], dtype=np.int32),
        pipe_end_node=np.array([2], dtype=np.int32),
        pipe_end_invert=np.array([914.0], dtype=np.float64),
        pipe_end_diameter=np.array([8.0], dtype=np.float64),
        pipe_end_area=np.array([32.0], dtype=np.float64),
        pipe_end_kin=np.array([0.05], dtype=np.float64),
        pipe_end_kout=np.array([0.1], dtype=np.float64),
        node_max_depth=np.array([5.8, 10.0], dtype=np.float64),
    )
    mod.swe2d_pipe1d_upload_node_rim(dev_ptr, np.array([930.8, 0.0], dtype=np.float64))
    mod.swe2d_pipe1d_init_cell_area(dev_ptr, H_MIN)
    cell_areas = np.array([50*CHAN_WID]*2 + [(dx*CHAN_WID)]*2*4, dtype=np.float64)
    mod.swe2d_gpu_preload_coupling_cell_area(cell_areas)

    # ── Time-step loop ────────────────────────────────────────────────────
    print("Stepping …")
    dt = 1.0
    for k in range(120):
        mod.swe2d_gpu_set_coupling_dt(dt)
        mod.swe2d_pipe1d_step(dev_ptr, dt, "fully_dynamic", 1, 1, 1.0,
                              G, K_MANN, H_MIN, surcharge_method=0)
        mod.swe2d_gpu_compute_coupling_full_on_device(None, 0, None)
        mod.swe2d_step(solver, dt)
        if k % 30 == 0 or k == 119:
            rb = mod.swe2d_pipe1d_readback_cell_state(dev_ptr, N_SUB, 0, 0)
            h2d = mod.swe2d_readback_state(solver, N_TRI)
            q_max = float(np.max(np.abs(rb["cell_Q"])))
            print(f"  step {k:3d}  Q={q_max:.3f}  "
                  f"pipe h[N-1]={rb['cell_h'][-1]:.4f}  "
                  f"2D h[0]={h2d["h"][0]:.4f}  h[{N_TRI//2}]={h2d["h"][N_TRI//2]:.4f}")

    rb = mod.swe2d_pipe1d_readback_cell_state(dev_ptr, N_SUB, 0, 0)
    h2d = mod.swe2d_readback_state(solver, N_TRI)
    print(f"\nQ range: {np.min(rb['cell_Q']):.4f} – {np.max(rb['cell_Q']):.4f} ft³/s")
    print(f"Pipe h:  {np.min(rb['cell_h']):.4f} – {np.max(rb['cell_h']):.4f} ft")
    print(f"2D h[0]={h2d["h"][0]:.4f}  h[1]={h2d["h"][1]:.4f}  h[-1]={h2d["h"][-1]:.4f} ft")
    vol_channel = float(np.sum(h2d["h"][2:] * cell_areas[2:]))
    print(f"Channel water vol: {vol_channel:.2f} ft³")
    mod.swe2d_destroy(solver)
    return 0

if __name__ == "__main__":
    sys.exit(main())
