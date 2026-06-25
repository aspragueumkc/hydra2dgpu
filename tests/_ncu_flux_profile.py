"""NCU profile target: swe2d_flux_kernel on GMSH ~500K mesh."""
import sys, numpy as np
sys.path.insert(0, 'build')
from swe2d.runtime.backend import SWE2DBackend
from tests._swe2d_test_helpers import _make_gmsh_triangle_mesh

def _zb(x, y):
    return 10.0 - 0.005*x - 0.003*y

node_x, node_y, node_z, cell_nodes, _, _ = _make_gmsh_triangle_mesh(
    2000.0, 2000.0, 4.5, zb_func=_zb)
ncells = cell_nodes.size // 3

backend = SWE2DBackend()
backend.build_mesh(node_x, node_y, node_z, cell_nodes,
    bc_edge_node0=np.empty(0, dtype=np.int32),
    bc_edge_node1=np.empty(0, dtype=np.int32),
    bc_edge_type=np.empty(0, dtype=np.int32),
    bc_edge_val=np.empty(0, dtype=np.float64))

backend.initialize(h0=np.full(ncells, 0.05, dtype=np.float64),
    n_mann=0.035, h_min=1e-4, cfl=0.45, dt_max=0.5,
    gpu_diag_sync_interval_steps=1, spatial_discretization=4)
backend.step(-1.0)
backend.step(-1.0)
backend.step(-1.0)
backend.destroy()
print("done")
