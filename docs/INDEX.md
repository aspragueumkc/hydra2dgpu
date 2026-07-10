# Documentation Index

Central entry point for all HYDRA 2D GPU documentation. Pick the guide that
matches your role and task.

---

## For Users

Start here if you're setting up or running simulations.

| Document | When to Read |
|----------|--------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Installation, Studio UI, running your first simulation |
| [CLI_GUIDE.md](CLI_GUIDE.md) | Headless runs, batch sweeps, CI/CD pipelines |
| [GMSH_MESHING_GUIDE.md](GMSH_MESHING_GUIDE.md) | Generating computational meshes from topology layers |
| [DRAINAGE_SOLVER_MODE_GUIDE.md](DRAINAGE_SOLVER_MODE_GUIDE.md) | Choosing EGL / Diffusion / Dynamic mode for 1D networks |
| [RAINFALL_CN_GUIDE.md](RAINFALL_CN_GUIDE.md) | Setting up rainfall, hyetographs, and CN infiltration |
| [GPKG_EXPLORER_GUIDE.md](GPKG_EXPLORER_GUIDE.md) | Inspecting and cleaning up model GeoPackages |
| [RESULTS_PATH_GUIDE.md](RESULTS_PATH_GUIDE.md) | Reading results, timeline, overlays, export |

## For Developers

Start here if you're contributing code or extending the plugin.

| Document | When to Read |
|----------|--------------|
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Architecture, module reference, style guide, test suite |
| [STUDIO_GUI_API.md](STUDIO_GUI_API.md) | Public protocols and types for the workbench UI |
| [UI_COMPONENT_GUIDE.md](UI_COMPONENT_GUIDE.md) | Adding docks, tabs, signals, feature toggles |
| [MODEL_GEOPACKAGE_SCHEMA.md](MODEL_GEOPACKAGE_SCHEMA.md) | Input GPKG tables and field definitions |
| [RESULTS_GEOPACKAGE_SCHEMA.md](RESULTS_GEOPACKAGE_SCHEMA.md) | Output GPKG tables, BLOB formats |
| [SOLVER_ORDER_AND_STENCIL.md](SOLVER_ORDER_AND_STENCIL.md) | Spatial/temporal order, stencil, non-orthogonality |
| [ADVANCED_SPATIAL_SCHEMES.md](ADVANCED_SPATIAL_SCHEMES.md) | Barth-Jespersen, true WENO3, MP5 — math, references, properties |

### Implementation Plans

Authoritative plans for ongoing and upcoming work. Each plan links to the technical guides it references.

| Document | Scope |
|----------|-------|
| [IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md) | Schemes 5/6/8: Barth-Jespersen, true WENO3, MP5 — kernel design, mesh-assembly extensions, rollout |
| [superpowers/specs/2026-07-10-advanced-spatial-schemes-design.md](superpowers/specs/2026-07-10-advanced-spatial-schemes-design.md) | Design spec: math, stencil tables, kernel interface, CFL enforcement |
| [superpowers/plans/2026-07-10-advanced-spatial-schemes.md](superpowers/plans/2026-07-10-advanced-spatial-schemes.md) | Implementation plan: rollout order, config migration, testing strategy |
| [IMPLEMENTATION_PLANS/2026-07-10-unified-run-controller.md](IMPLEMENTATION_PLANS/2026-07-10-unified-run-controller.md) | Single-kernel-entry refactor: collapse GUI/CLI into one `SWE2DRunController` + `ProgressSink` protocol; closes MP5 CFL clamping |
| [IMPLEMENTATION_PLANS/2026-07-10-unified-run-controller-revision-1.md](IMPLEMENTATION_PLANS/2026-07-10-unified-run-controller-revision-1.md) | Revision 1: GUI is source of truth, CLI replays it via JSON; replay JSON persisted to `swe2d_run_replays` table; round-trip test is the contract |
| [IMPLEMENTATION_PLANS/2026-07-10-unified-run-controller-revision-2.md](IMPLEMENTATION_PLANS/2026-07-10-unified-run-controller-revision-2.md) | Revision 2: consolidate three JSON producers (snapshot, save-config, CLI replay) + two legacy GPKG tables (`swe2d_simulation_configs`, `swe2d_run_logs`) into one canonical pair (to_replay_json / from_replay_json → `swe2d_run_replays` table) |

## For C++ / CUDA Engineers

C++ kernel internals and GPU solver architecture.

| Document | When to Read |
|----------|--------------|
| [cpp/ARCHITECTURE.md](cpp/ARCHITECTURE.md) | C++/CUDA module layout, build system, unit convention |
| [cpp/GPU_KERNEL_STRATEGY.md](cpp/GPU_KERNEL_STRATEGY.md) | Kernel launch hierarchy, SoA layout, graph caching |
| [cpp/COUPLING_KERNELS.md](cpp/COUPLING_KERNELS.md) | GPU coupling: surface ↔ drainage ↔ structures |
| [cpp/CULVERT_HDS5.md](cpp/CULVERT_HDS5.md) | FHWA HDS-5 culvert implementation |

## For Documentation Authors

| Document | When to Read |
|----------|--------------|
| [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md) | GPU solver deep-dive (coupling, rainfall, structures, drainage) |

---

## API Reference (Auto-Generated)

| Output | How to Build |
|--------|--------------|
| Python API | `cd docs && make api` (uses pdoc) |
| C++ / CUDA API | `cd docs && make cpp-api` (uses Doxygen) |

Generated docs land in `docs/_build/` and are not tracked in git.

---

## Repository Knowledge Graph

A pre-built knowledge graph of the entire codebase is in `graphify-out/`.
It is the fastest way to find modules, god nodes, and cross-file
relationships.

| Output | What's Inside |
|--------|---------------|
| `graphify-out/graph.html` | Interactive browser visualizer |
| `graphify-out/GRAPH_REPORT.md` | Audit report, community detection, suggested questions |
| `graphify-out/wiki/index.md` | 549 articles — one per community cluster |
| `graphify-out/graph.json` | Raw graph for GraphRAG / custom tooling |

Example queries (with the graph already built):

```bash
graphify query "How does the coupling controller reach the GPU solver?"
graphify path "SWE2DDeviceState" "SWE2DCouplingController"
graphify explain "KernelGraphCache"
```

---

## Quick Links

- **New user?** → [USER_GUIDE.md](USER_GUIDE.md)
- **New developer?** → [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **Headless / batch run?** → [CLI_GUIDE.md](CLI_GUIDE.md)
- **C++ kernel work?** → [cpp/ARCHITECTURE.md](cpp/ARCHITECTURE.md)
- **Schema question?** → [MODEL_GEOPACKAGE_SCHEMA.md](MODEL_GEOPACKAGE_SCHEMA.md)
- **Architecture deep-dive?** → [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md)