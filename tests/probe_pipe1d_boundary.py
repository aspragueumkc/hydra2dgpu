"""Probe the boundary face flux magnitude at node 1 over the first 100 timesteps.

Run with:  mamba run -n qgis_stable python3 tests/probe_pipe1d_boundary.py

If the boundary face flux F stays bounded (<100 cfs), the issue is interior.
If F spikes to thousands of cfs at first inflow, the boundary is the bottleneck.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'build')

import numpy as np
import hydra_swe2d as M

# Standard 10x5 box, 553 ft, 2% slope, US units
G_US = 32.174
K_MANN = 1.486
H_MIN = 1.0e-6  # near-zero so the pipe starts truly dry
DT = 0.001
N_STEPS = 100

# Build a 1-link mesh
n_links = 1
link_length = np.array([553.3], dtype=np.float64)
link_diameter = np.array([0.0], dtype=np.float64)
link_roughness = np.array([0.013], dtype=np.float64)
link_inlet_loss = np.array([0.0], dtype=np.float64)
link_outlet_loss = np.array([0.0], dtype=np.float64)
link_invert_in = np.array([925.0], dtype=np.float64)
link_invert_out = np.array([914.0], dtype=np.float64)
shape_type = np.array([1], dtype=np.int32)
link_width = np.array([10.0], dtype=np.float64)
link_height = np.array([5.0], dtype=np.float64)

max_cell_length = 553.3  # 1 cell only

M.swe2d_build_unified_mesh(
    dev_ptr=1,                                  # placeholder; see note below
    n_links=1,
    link_from=np.array([0], dtype=np.int32),
    link_to=np.array([1], dtype=np.int32),
    L=link_length,
    D=link_diameter,
    n_mann=link_roughness,
    S0=np.zeros(1, dtype=np.float64),
    node_invert=np.array([925.0, 914.0], dtype=np.float64),
    mcl=float(max_cell_length),
    link_shape_type=shape_type,
    link_width=link_width,
    link_height=link_height,
)

dev_ptr = 1  # placeholder; the build function returns a properly allocated device,
              # we just need to keep it alive for the duration of the test

# Initialize pipe at minimum depth (per-cell via the unified upload).
M.swe2d_pipe1d_upload_cell_h(
    dev_ptr, np.array([0.0], dtype=np.float64)  # 1 pipe cell
)
M.swe2d_pipe1d_init_cell_area(dev_ptr, H_MIN)

print(f"Probing pipe1D at dt = {DT:.4f}s for {N_STEPS} steps")
print(f"{'step':>5} {'cell_depth[0]':>14} {'cell_A[0]':>12} {'cell_Q_max':>12}")

max_F_log = []

import ctypes
for step in range(N_STEPS):
    # Step the pipe1d solver
    M.swe2d_pipe1d_step(
        dev_ptr, DT, "fully_dynamic",
        1, 2, 0.5, G_US, K_MANN, H_MIN,
        1,        # surcharge_method = SLOT
    )

    # Readback state (unified ``readback_cell_state`` binding).
    if step % 5 == 0 or step < 5:
        state = M.swe2d_pipe1d_readback_cell_state(dev_ptr, 1)
        cell_depth = np.array(state['cell_depth'])
        cell_q = np.array(state['cell_Q'])
        cell_a = np.array(state['cell_A'])
        print(f"{step:>5} {cell_depth[0]:>14.4f} "
              f"{cell_a[0]:>12.4f} {np.abs(cell_q).max():>12.4f}")

print()
print("If cell_A_max is at A_full (50) and stable, dt is fine.")
print("If cell_A_max grows past 50ft² or Q explodes to 10000+ cfs, dt is too big.")
