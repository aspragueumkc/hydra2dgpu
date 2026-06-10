---
name: mesh-quality-triage
description: Diagnose and fix Gmsh mesh quality failures
---

# Mesh Quality Triage

When a Gmsh mesh generation attempt fails quality checks, follow these steps:

## 1. Read the error

Identify which quality threshold failed from the error message:
- `min_angle_deg` — elements too skinny
- `max_aspect_ratio` — elements too elongated
- `min_area_rel_bbox` — elements too small (near-degenerate)
- `max_non_orth_deg` — elements too skewed

## 2. Adjust quality config

Edit the `options` dict passed to `generate_face_centric_mesh()`:

```python
options = {
    # Relax thresholds (if geometry is challenging):
    "gmsh_min_angle_deg": 15.0,       # default 18.0
    "gmsh_max_aspect_ratio": 15.0,    # default 12.0
    
    # Enable retry ladder:
    "gmsh_quality_enable": True,
    "gmsh_quality_max_iterations": 5,
    "gmsh_quality_time_limit_s": 120.0,
    
    # Expand the size scale ladder:
    "gmsh_quality_size_scales": "1.0,0.85,0.7,0.55",
    
    # Switch algorithm on failure:
    "gmsh_algorithm_switch_on_failure": True,
    
    # Increase smoothing:
    "gmsh_smoothing": 3,
    "gmsh_optimize_iters": 2,
}
```

## 3. Common fixes by failure mode

| Failure | Fix |
|---|---|
| **Min angle too low** | Enable quality loop, reduce `size_scales`, increase smoothing |
| **Aspect ratio too high** | Increase `gmsh_quality_recombine_topology_passes` |
| **Min area too small** | Increase `gmsh_tolerance_edge_length`, check for degenerate arcs |
| **Non-orthogonality** | Enable `gmsh_interface_conformance`, reduce interface size ratio |

## 4. Verify

Run `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest tests.test_swe2d_gpu_unstructured -v`

## 5. Environment variables

```bash
export BACKWATER_GMSH_QUALITY_ENABLE=1
export BACKWATER_GMSH_MIN_ANGLE_DEG=15.0
export BACKWATER_GMSH_QUALITY_MAX_ITERATIONS=10
export BACKWATER_GMSH_QUALITY_TIME_LIMIT_S=120
```
