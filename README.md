# HYDRA — QGIS Hydrodynamics & Runoff Application

Steady-flow 1D backwater modeling plugin for QGIS with GeoPackage-native model storage, culvert support, and integrated run/result tools.

## Current State

- Plugin-first architecture.
- Core solver is in `hydra_1d.py` (GUI-free, CLI-capable).
- QGIS dock widget and UI workflows are in `hydra_qt.py`.
- QGIS menu integration is in `hydra_plugin.py`.
- Model I/O is GeoPackage-only in the plugin workflow.

## Project Status

- Native backend work is active: the 1D unsteady path has C++ build scaffolding, backend selection, and partial parity/benchmark coverage, but it is not yet the fully validated default runtime.
- SWE2D now has a dedicated QGIS workbench with native solver hooks, interactive run/cancel controls, and topology-driven meshing from map layers.
- SWE2D development is now GPU-only in practice: CUDA is the main implementation path for numerics, validation, and performance tuning.
- CPU SWE2D remains only as a compatibility/debug fallback. It is not the target for new optimization work, and CPU/GPU numerical parity is no longer treated as a development goal.
- The topology meshing workflow now treats constraint layers as real local sizing inputs in the Gmsh backend, so refinement polygons can drive smaller cells inside a larger region instead of only acting as metadata.
- Topology imports now handle MultiPolygon regions and constraints, which matters when QGIS layers contain multipart features.
- The structured backend remains a coarse-grid path and does not provide true local h-refinement; use Gmsh when local refinement matters.
- The topology meshing workflow also supports TQMesh as a backend in the workbench, including an optional quad-edge layer for four-sided regions defined by explicit side polylines.
- TQMesh stability work is in place: the Python binding crash caused by incorrect quadtree scaling was fixed, triangular meshing is stable, and quadrilateral output is available through the sampled-side plus `tri2quad` path.
- The current TQMesh quadrilateral workflow is intentionally conservative: for full four-edge regions, explicit side polylines are sampled into the exterior boundary and then converted from triangles to quads, because applying quad-layer generation on all four sides at once was not robust.
- Latest GPU validation status: gmsh-based unstructured dam-break and lake-at-rest checks pass on CUDA for spatial schemes 0..4.

## SWE2D GPU Validation and Performance Testing

- Legacy coarse GPU/CPU regression envelope: `tests/test_swe2d_gpu.py` (maintenance-only; not the main acceptance gate)
- Dedicated GPU-only validation + throughput suite: `tests/test_swe2d_gpu_validation_perf.py`
- Dedicated GPU unstructured gmsh-based validation: `tests/test_swe2d_gpu_unstructured.py`

Run:

```bash
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured
```

Optional benchmark mode:

```bash
BACKWATER_RUN_GPU_PERF=1 PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf
```

Latest unstructured GPU status:

- Dam-break on gmsh-generated unstructured triangles passes on CUDA for schemes 0..4.
- Lake-at-rest on gmsh-generated unstructured triangles now passes on CUDA for schemes 0..4.

Topological meshing status:

- Constraint polygons now act as local size fields in the Gmsh backend, so they can refine a subregion inside a larger mesh.
- Multipart QGIS regions and constraints are accepted.
- Structured meshing is still useful for simple coarse tiling, but it is not a substitute for local refinement.

Implication for contributors and agents:

- Prioritize fixing and optimizing the CUDA solver.
- Do not spend SWE2D development effort chasing CPU/GPU parity unless a specific compatibility bug requires it.
- When working on meshing, prefer the Gmsh topology path for refinement-sensitive domains and treat TQMesh as an alternative backend for quad-oriented layouts.

## SWE2D Extension Skeletons

Scaffold modules were added to accelerate future multi-physics work:

- Rain-on-grid source skeleton: `swe2d_rainfall.py`
- Urban drainage/pipe network skeleton (SWMM-style coupling): `swe2d_drainage_network.py`
- Hydraulic structures skeleton (weirs/culverts/gates/bridges/pumps): `swe2d_structures.py`
- Shared enums/config models for numerics, turbulence, and friction: `swe2d_extensions.py`

Native and Python solver creation now also accept model-selection flags for:

- temporal scheme selection
- spatial discretization scheme selection
- turbulence model selection
- bed friction law selection
- enabling rain, pipe-network, and hydraulic-structure modules

Current higher-order reconstruction implementation status:

- `FV_MUSCL_FAST`: throughput-oriented pairwise linear edge reconstruction in CUDA
  with minimal limiting.
- `FV_MUSCL_MINMOD`: robust pair-bounded reconstruction in CUDA to reduce
  overshoot risk near strong gradients and wet/dry transitions.

Both schemes are now selectable in the GPU flux path through native solver
configuration (`spatial_scheme`) and are compatible with RK2 stepping.

Additional wet/dry-front stability controls are now available from the 2D
workbench and native API:

- Localized shallow-front reconstruction fallback (optional):
  `enable_shallow_front_recon_fallback` forces first-order reconstruction only
  on shallow edge pairs near advancing wet/dry fronts.
- `front_flux_damping` momentum-flux damping at wet/dry fronts.
- `active_set_hysteresis` to reduce oscillatory active-set switching.

## Key Features

- Create, load, save, and run backwater models directly from QGIS.
- Built-in main menu entries under **HYDRA** for common actions:
  - Open HYDRA Panel
  - Create New Model GeoPackage...
  - Load Model GeoPackage...
  - Save Model GeoPackage As...
  - Run Model
  - Open Results Plot
  - Open Results Table
  - Enable/Disable Layer Editing
  - Save Layer Edits
- New **Options** submenu in the **HYDRA** menu:
  - Solver: `py` or `scipy`
  - Alpha Method: `conveyance` or `area`
- Downstream boundary controls in UI (`known_wse`, `normal_depth`) plus DS value and flow inputs.
- Result persistence to `model_results` and reload on model open.
- Culvert fields integrated into cross-section schema and forms.
- Matplotlib plotting support in plugin context with local runtime detection.

## GeoPackage Model Layers

The plugin expects these core layers:

- `cross_sections`
- `centerline` (required)
- `boundary_conditions`

The plugin also writes/reads:

- `model_results`

## Editing Workflow

- Enable editing with **HYDRA > Enable/Disable Layer Editing**.
- Edit model data through QGIS attribute forms for loaded layers.
- Save edits with **HYDRA > Save Layer Edits** before running.
- If unsaved layer edits exist, model run is blocked until edits are saved.

Form behavior includes:

- Cross-section and boundary-condition custom forms from `forms/`.
- Cross-section actions for terrain selection and Z updates.
- Conditional culvert field visibility based on `culvert_code`.

## Repository Layout

- `hydra_1d.py`: hydraulic solver, GeoPackage I/O, CLI entry logic.
- `hydra_qt.py`: dock widget, run workflow, plotting/table UI.
- `hydra_plugin.py`: QGIS main menu and action wiring.
- `culvert_routine.py`: culvert hydraulics helpers.
- `forms/`: Qt Designer forms and form-init hooks.
- `expressions/`: QGIS expression helpers.
- `tests/`: solver and integration-oriented tests.
- `docs/`: design notes and hydraulic reference material.

## Development Notes

- In this environment, use `python3` for checks.
- Example syntax check:

```bash
python3 -m py_compile hydra_1d.py hydra_qt.py hydra_plugin.py
```

- Python docstring/type-hint coverage audit (project-owned files):

```bash
python3 tools/python_style_audit.py
```

- Audit continuation and staged wave handoff guide:

  - `docs/PYTHON_STYLE_AUDIT_WAVES.md`

## User Documentation

See `USER_GUIDE.md` for step-by-step usage in QGIS.

## Implementation Roadmap

For the current native-backend roadmap (1D C++ port + 2D SWE solver), see:

- `docs/IMPLEMENTATION_PLAN_6W_1D_CPP_AND_2D_SWE.md`
- `docs/TASK_BOARD_6W_SOLO_AND_AGENT.md`
- `docs/WEEK1_CPP_STARTER_PACK.md`
- `docs/NATIVE_CPP_BACKEND_BUILD.md`
