import sys
sys.path.insert(0, '.')

try:
    import hydra_swe2d as mod
    print(f"GPU available: {mod.swe2d_gpu_available()}")
    
    # Simple 2x2 mesh
    node_x = [0.0, 1.0, 0.0, 1.0]
    node_y = [0.0, 0.0, 1.0, 1.0]
    node_z = [0.0, 0.0, 0.0, 0.0]
    cell_nodes = [0, 1, 2, 1, 3, 2]  # Two triangles
    
    import numpy as np
    mesh = mod.swe2d_build_mesh(
        np.array(node_x, dtype=np.float64),
        np.array(node_y, dtype=np.float64),
        np.array(node_z, dtype=np.float64),
        np.array(cell_nodes, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64))
    
    info = mod.swe2d_mesh_info(mesh)
    print(f"Mesh: {info}")
    
    # Initial state
    h0 = np.array([1.0, 1.0], dtype=np.float64)
    
    # CPU solver
    solver_cpu = mod.swe2d_create_solver(mesh, h0.copy(), use_gpu=False)
    for i in range(3):
        diag_cpu = mod.swe2d_step(solver_cpu, -1.0)
    h_cpu, hu_cpu, hv_cpu = mod.swe2d_get_state(solver_cpu)
    print(f"CPU h: {h_cpu}")
    print(f"CPU hu: {hu_cpu}")
    print(f"CPU diag (step 3): {diag_cpu}")
    mod.swe2d_destroy(solver_cpu)
    
    # GPU solver
    h0 = np.array([1.0, 1.0], dtype=np.float64)
    solver_gpu = mod.swe2d_create_solver(mesh, h0.copy(), use_gpu=True)
    for i in range(3):
        diag_gpu = mod.swe2d_step(solver_gpu, -1.0)
    h_gpu, hu_gpu, hv_gpu = mod.swe2d_get_state(solver_gpu)
    print(f"GPU h: {h_gpu}")
    print(f"GPU hu: {hu_gpu}")
    print(f"GPU diag (step 3): {diag_gpu}")
    mod.swe2d_destroy(solver_gpu)
    
    print(f"h diff: {np.max(np.abs(h_cpu - h_gpu))}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
