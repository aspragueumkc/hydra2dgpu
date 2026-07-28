"""Validate volume decomposition against SWMM and current open-channel clamp.

Tests: pipe network with inflow that overwhelms capacity → surcharge.
Compares three approaches:
  1. Current open-channel (surcharge_method=0) — clamps A, loses mass
  2. Volume decomposition (prototype in Python) — stores excess in nodes
  3. SWMM (via pyswmm) — industry reference with Preissmann slot

Network: 
  Node1(inflow) --Link1(circ, D=1ft, L=100ft)--> Node2(junction)
    --Link2(circ, D=1ft, L=100ft)--> Node3(outfall)
  Baseflow Q=0 → ramp to Q=5 cfs at t=10s → steady.
  Pipe A_full = π/4 ≈ 0.785 ft².  At Q=5 cfs, required area exceeds
  A_full for any reasonable slope → pipe pressurizes.
"""
import numpy as np, math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
from swe2d.runtime.backend import SWE2DBackend
import hydra_swe2d as _MOD

_G = 32.2  # ft/s²
_K = 1.486  # Manning's k for USC
_H = 1e-6

# ── Build pipe network ──
b = SWE2DBackend()
b.build_mesh(np.asarray([0.,1.,0.]), np.asarray([0.,0.,1.]), np.asarray([0.,0.,0.]), np.asarray([0,1,2]))
b.initialize(h0=np.asarray([0.1]), hu0=np.zeros(1), hv0=np.zeros(1), dt_fixed=0.05, dt_max=0.05)
dev = int(_MOD.swe2d_get_coupling_dev_ptr())

_MOD.swe2d_build_unified_mesh(dev_ptr=dev, n_links=2,
    link_from=np.array([0,1],dtype=np.int32), link_to=np.array([1,2],dtype=np.int32),
    L=np.array([100.,100.],dtype=np.float64), D=np.array([1.,1.],dtype=np.float64),
    n_mann=np.array([0.013,0.013],dtype=np.float64), S0=np.zeros(2,dtype=np.float64),
    node_invert=np.array([10.,9.5,9.],dtype=np.float64), mcl=50.,
    link_shape_type=np.zeros(2,dtype=np.int32), link_width=np.array([1.,1.],dtype=np.float64),
    link_height=np.array([1.,1.],dtype=np.float64),
    n_manhole_cells=1, manhole_node_ids=np.array([1],dtype=np.int32),
    manhole_invert=np.array([9.5],dtype=np.float64),
    manhole_surface_area=np.array([10.],dtype=np.float64),  # 10 ft² node area
    manhole_max_depth=np.array([5.],dtype=np.float64),
    manhole_rim=np.array([14.5],dtype=np.float64),
    manhole_diameter=np.array([3.57],dtype=np.float64),  # equiv to 10 ft²
    n_pipe_ends=1, pipe_end_node_ids=np.array([2],dtype=np.int32))
_MOD.swe2d_gpu_device_sync()

st = _MOD.swe2d_pipe1d_readback_cell_state(dev, 10)
nc = int(st['n_cells_all'])
print(f"Cells: n_pipe={st['n_pipe_cells']} n_manhole={st['n_manhole_cells']} n_all={nc}")

# ── Run with surcharge_method=0 (open-channel clamp) ──
h = np.zeros(nc, dtype=np.float64)
h[0:4] = 0.1  # link1 cells
h[4:8] = 0.1  # link2 cells
_MOD.swe2d_pipe1d_upload_cell_h(dev, h)
_MOD.swe2d_pipe1d_init_cell_area(dev, _H)

dt = 0.05
nsteps = 400  # 20 seconds
pipe_vol_history = []
node_vol_history = []
for step in range(nsteps):
    t = step * dt
    # Inflow at node0: ramp to 5 cfs
    Q_in = min(5.0, t * 0.5)
    # ... would need INLET_BC upload for prescribed Q, skip for now
    _MOD.swe2d_pipe1d_step(dev, dt, 'rk2', 1, 2, 0.5, _G, _K, _H,
                          surcharge_method=0, recon_method=1, theta=1.0)
    st = _MOD.swe2d_pipe1d_readback_cell_state(dev, nc)
    A = np.asarray(st['cell_A'])
    L = np.asarray(st['cell_length'])
    pipe_vol = float(np.sum(A[0:8] * L[0:8]))
    mh_A = float(A[8]) if nc > 8 else 0.0
    mh_L = float(L[8]) if nc > 8 else 1.0
    node_vol = mh_A * mh_L
    if step % 100 == 0:
        print(f"  t={t:.1f}s: pipe_vol={pipe_vol:.2f} node_vol={node_vol:.2f} total={pipe_vol+node_vol:.2f}")

b.destroy()
print("\nDone. Now compare with SWMM and volume decomposition prototype.")
