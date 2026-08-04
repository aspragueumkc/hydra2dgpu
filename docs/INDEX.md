# Documentation Index

Central entry point for all HYDRA 2D GPU documentation. Pick the guide that matches your role and task.

---

## For Users

Start here if you're setting up or running simulations.

| Document | When to Read |
|----------|--------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Installation, Studio UI, running your first simulation |
| [USER_GUIDE.md §12 Graph Editor](USER_GUIDE.md#12-graph-editor-hydrographs--hyetographs) | Authoring hydrographs and hyetographs in the Studio |
| [USER_GUIDE.md §13 CLI Quickstart](USER_GUIDE.md#13-cli-quickstart) | `swe2d-cli run` / `swe2d-cli batch` one-liners |
| [USER_GUIDE.md §14 Batch Runner Workflow](USER_GUIDE.md#14-batch-runner-workflow) | `batch.json` schema, MPS, status file |
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
| [RUN_SPEC_SCHEMA.md](RUN_SPEC_SCHEMA.md) | Canonical `swe2d-run/2` JSON input schema |

## Active Plans & Specs

Authoritative plans and specs currently in flight. Authoritative history (completed, superseded) lives in [docs/archive/INDEX.md](archive/INDEX.md).

### Active Plans

| Document | Created | Scope |
|----------|---------|-------|
| [CLI-First Refactor Implementation Plan](plans/2026-07-24-cli-first-refactor.md) | 2026-07-24 | Land canonical CLI pipeline (spec→RunContext→executor→results); GUI reduced to view |
| [MCP Phase 5 — Display Attach Implementation Plan](plans/2026-07-24-hydra-mcp-p5.md) | 2026-07-24 | MCP Phase 5 — display-attach to live QGIS via bridge token polling |
| [HYDRA MCP Phase 1 — Production Modeling Tools Plan](plans/2026-07-24-hydra-mcp-phase1.md) | 2026-07-24 | MCP Phase 1 — Tier A production modeling tools (model+mesh+BC+run+results) |
| [HYDRA MCP Server Implementation Plan](plans/2026-07-24-hydra-mcp-server.md) | 2026-07-24 | Ship full HYDRA MCP server (Tier A modeling + Tier B GUI + Tier C design) |
| [MCP Phase 3 — Behavioral GUI Testing Implementation Plan](plans/2026-07-24-mcp-phase3-behavioral-gui-testing.md) | 2026-07-24 | MCP Phase 3 — QTest-driven behavioral GUI tools (click/key/run/screenshot) |
| [HYDRA MCP Server Plan — Agent-Assisted Production Use, Testing & Design](plans/HYDRA_MCP_SERVER_PLAN.md) | 2026-07-24 | Original 6-phase MCP server plan (agent-assisted production use + GUI testing) |
| [Docs Lifecycle Migration Implementation Plan](plans/2026-07-25-docs-lifecycle-migration.md) | 2026-07-25 | THIS PLAN — restructure docs/{plans,specs}/ + docs/archive/ with frontmatter |

### Active Specs

| Document | Created | Scope |
|----------|---------|-------|
| [CLI-First Refactor — Design Spec](specs/2026-07-24-cli-first-refactor-design.md) | 2026-07-24 | CLI-first canonical pipeline design (single builder + executor) |
| [HYDRA MCP Server — Design Spec](specs/2026-07-24-hydra-mcp-server-design.md) | 2026-07-24 | Single MCP server design — Tier A modeling + Tier B GUI + Tier C design |
| [Docs Lifecycle Design — Active vs Complete/Superseded](specs/2026-07-25-docs-lifecycle-design.md) | 2026-07-25 | THIS SPEC — single docs lifecycle convention + YAML frontmatter metadata |
| [Agent Memory Architecture — Design Spec](specs/2026-07-26-agent-memory-architecture-design.md) | 2026-07-26 | Hybrid (canonical Git memory + local vector index) agent/LLM memory layer |

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

## Agent Memory

Curated, Git-tracked memory that any agent or human can read, diff, and review.
The CLI is `tools/memory.py`; the local vector index lives in `.memory/`
(gitignored, rebuildable). See the `hydra-agent-memory` skill.

| Document | When to Read |
|----------|--------------|
| [Memory Index](memory/INDEX.md) | Active topics, listed by file |
| [Agent Memory Architecture Spec](specs/2026-07-26-agent-memory-architecture-design.md) | Schema, capture flow, retrieval, lifecycle |

---

## Quick Links

- **New user?** → [USER_GUIDE.md](USER_GUIDE.md)
- **New developer?** → [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **Headless / batch run?** → [CLI_GUIDE.md](CLI_GUIDE.md)
- **C++ kernel work?** → [cpp/ARCHITECTURE.md](cpp/ARCHITECTURE.md)
- **Schema question?** → [MODEL_GEOPACKAGE_SCHEMA.md](MODEL_GEOPACKAGE_SCHEMA.md)
- **Architecture deep-dive?** → [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md)

---

## Historical Catalog

Completed and superseded plans, specs, audits, session logs, and reference notes are cataloged at [docs/archive/INDEX.md](archive/INDEX.md).
