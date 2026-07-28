"""Inlet→pipe single step diagnostic with printf."""
import numpy as np, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
from swe2d.runtime.backend import SWE2DBackend
import hydra_swe2d as _MOD
_G, _K, _H = 9.80665, 1.0, 1e-10

b = SWE2DBackend()
b.build_mesh(np.asarray([0.,1.,0.]), np.asarray([0.,0.,1.]), np.asarray([0.,0.,0.]), np.asarray([0,1,2]))
b.initialize(h0=np.asarray([0.1]), hu0=np.zeros(1), hv0=np.zeros(1), dt_fixed=0.05, dt_max=0.05)
dev = int(_MOD.swe2d_get_coupling_dev_ptr())

_MOD.swe2d_build_unified_mesh(
    dev_ptr=dev, n_links=1,
    link_from=np.array([0], dtype=np.int32),
    link_to=np.array([1], dtype=np.int32),
    L=np.array([1.0], dtype=np.float64),
    D=np.array([0.5], dtype=np.float64),
    n_mann=np.array([0.013], dtype=np.float64),
    S0=np.zeros(1, dtype=np.float64),
    node_invert=np.array([10.0, 9.0], dtype=np.float64),
    mcl=10.0,
    link_shape_type=np.zeros(1, dtype=np.int32),
    link_width=np.array([0.5], dtype=np.float64),
    link_height=np.array([0.5], dtype=np.float64),
    n_inlet_cells=1,
    inlet_node_ids=np.array([0], dtype=np.int32),
    inlet_invert=np.array([10.0], dtype=np.float64),
    inlet_surface_area=np.array([1.0], dtype=np.float64),
    inlet_max_depth=np.array([3.0], dtype=np.float64),
    inlet_diameter=np.array([1.0], dtype=np.float64),
    inlet_cell_length=np.array([1.0], dtype=np.float64),
    inlet_cell_width=np.array([0.785], dtype=np.float64),
)
_MOD.swe2d_gpu_device_sync()

h = np.array([0.0, 2.0], dtype=np.float64)
_MOD.swe2d_pipe1d_upload_cell_h(dev, h)
_MOD.swe2d_pipe1d_init_cell_area(dev, _H)

st = _MOD.swe2d_pipe1d_readback_cell_state(dev, 2)
print(f"Init: pipe_A={st['cell_A'][0]:.4f} inlet_A={st['cell_A'][1]:.4f}")

_MOD.swe2d_pipe1d_step(dev, 0.5, 'rk2', 1, 2, 0.5, _G, _K, _H,
                       surcharge_method=1, recon_method=1, theta=1.0)

st = _MOD.swe2d_pipe1d_readback_cell_state(dev, 2)
print(f"Step: pipe_A={st['cell_A'][0]:.4f} inlet_A={st['cell_A'][1]:.4f}")
b.destroy()
