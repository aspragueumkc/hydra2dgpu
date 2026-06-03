# MFEM Backend Plan for Region/Arc/Boundary-Conforming Hybrid Meshing

## 1. Executive Recommendation

Use MFEM first as a conformance and quality optimization backend layered after the current hybrid-cpp generator, not as a day-1 replacement generator.

Recommended near-term architecture:
- Keep existing topology-to-mesh generation in the current backend pipeline.
- Export generated mesh plus region/arc/boundary markers into an MFEM mesh representation.
- Run MFEM TMOP-based fitting/optimization to improve boundary and interface conformance while preserving element validity and quality.
- Re-import the optimized mesh into the plugin mesh contract.

Reasoning:
- MFEM provides strong mesh optimization, fitting, untangling, and high-order node movement capabilities.
- MFEM does not provide a direct drop-in replacement for your specific constrained 2D hybrid generation workflow (region polygons + arcs + explicit boundary role semantics) out of the box.
- This staged approach reduces delivery risk while still exploiting MFEM strengths quickly.

## 2. Current Plugin Baseline

Current backend architecture already supports multiple meshing backends and a stable solver mesh contract.

Relevant integration points:
- [swe2d/mesh/meshing.py](swe2d/mesh/meshing.py)
- [cpp/src/hybrid_mesh_bindings.cpp](cpp/src/hybrid_mesh_bindings.cpp)
- [swe2d/workbench/extracted/topology_and_io_methods.py](swe2d/workbench/extracted/topology_and_io_methods.py)
- [swe2d_workbench_qt.py](swe2d_workbench_qt.py)
- [tests/test_hybrid_cpp_channel_transition.py](tests/test_hybrid_cpp_channel_transition.py)

Existing backend model in [swe2d/mesh/meshing.py](swe2d/mesh/meshing.py) is already ideal for an optional MFEM post-process stage.

## 3. MFEM Capability Fit

MFEM strengths that map well to your objective:
- TMOP variational optimization for shape, size, and orientation control.
- Boundary/interface fitting via weakly enforced alignment terms.
- Untangling and quality regularization pathways.
- Mature handling of mesh attributes and mixed element workflows.

MFEM gaps for your exact workflow (as a direct generator replacement):
- No direct region/arc/boundary-conforming planar hybrid generator matching current QGIS conceptual model semantics.
- No direct equivalent to your current constrained-edge recovery and strict conformance logic tuned for channel-transition topology.

Conclusion:
- Best technical fit is MFEM-as-optimizer/fitter first.
- Full MFEM-native generation can be explored as an R&D branch after measurable success in the optimization stage.

## 4. Target Architecture

### 4.1 New Components

Add a native bridge module:
- `cpp/src/mfem_mesh_opt_bridge.cpp`
- `cpp/src/mfem_mesh_opt_bridge.hpp`
- `cpp/src/mfem_mesh_opt_bindings.cpp` (pybind entrypoints)

Add Python orchestrator wrapper:
- `swe2d/mesh/mfem_opt.py`

Backend selection surface:
- Extend backend/options flow in [swe2d/mesh/meshing.py](swe2d/mesh/meshing.py) to support a mode like:
  - `backend=hybrid_cpp` + `post_opt_backend=mfem_tmop`
  - or a single alias `backend=hybrid_cpp_mfem`

### 4.2 Data Flow

1. Conceptual model from QGIS layers.
2. Generate base mesh with existing `hybrid_cpp` backend.
3. Build MFEM mesh from plugin mesh arrays.
4. Build MFEM attribute markers:
- Region ids from polygon source.
- Boundary ids from boundary-condition lines.
- Arc/interface ids from arc roles (centerline, banks, breaklines).
5. Run TMOP quality + fitting objective.
6. Export optimized node coordinates and connectivity.
7. Re-run plugin invariants/repair checks.
8. Return same mesh contract to SWE2D solver.

### 4.3 Contract Stability

Do not change solver-facing mesh API.

Preserve existing `MeshResult` arrays:
- `node_x`, `node_y`, `node_z`
- `cell_face_offsets`, `cell_face_nodes`
- `cell_nodes`, `cell_type`, `region_id`, `target_size`

This keeps downstream SWE2D/GPU workflows unchanged.

## 5. Phased Implementation Plan

## Phase 0: Build and Dependency Spike (1 week)

Goals:
- Prove MFEM can be built and linked in your native backend build without destabilizing existing builds.
- Keep MFEM support optional via compile flag.

Tasks:
- Add CMake option `HYDRA_ENABLE_MFEM` default OFF.
- Add detection/link logic for MFEM + required dependencies.
- Build a minimal pybind function that round-trips a tiny mesh without modification.

Exit criteria:
- Plugin builds with and without MFEM.
- No regressions in existing meshing tests.

## Phase 1: Identity Bridge + Markers (1 week)

Goals:
- Implement deterministic mesh import/export between plugin arrays and MFEM mesh.
- Attach region/boundary/interface markers correctly.

Tasks:
- Implement converter functions:
  - plugin arrays -> MFEM mesh
  - MFEM mesh -> plugin arrays
- Add marker mapping table for region ids, BC boundaries, arc roles.
- Add strict checks for element orientation and duplicate nodes after round-trip.

Exit criteria:
- Round-trip is topology-preserving on representative meshes.
- Marker integrity tests pass.

## Phase 2: TMOP Quality Optimization (2 weeks)

Goals:
- Improve mesh quality with MFEM while keeping geometry conformance at least neutral.

Tasks:
- Implement TMOP objective with conservative defaults.
- Add guardrails:
  - max iterations
  - min determinant threshold
  - early stop criteria
  - rollback on degradation
- Provide user-exposed advanced options (hidden/advanced panel first).

Exit criteria:
- Quality metrics improve or remain neutral on benchmark suite.
- No increase in invalid/inverted elements.

## Phase 3: Boundary/Interface Fitting (2 weeks)

Goals:
- Actively improve conformity to boundaries and arc-defined interfaces.

Tasks:
- Add fitting term targeting:
  - boundary lines
  - selected arc roles (centerline/banks/breakline)
- Introduce per-marker fitting weights.
- Keep constrained-edge semantics by locking or strongly penalizing key boundary sets.

Exit criteria:
- Measured decrease in edge-to-constraint distance metrics.
- No material loss of quality compared to Phase 2.

## Phase 4: Productionization and UI Exposure (1 week)

Goals:
- Make MFEM path selectable and diagnosable in workbench.

Tasks:
- Add UI options under advanced meshing section:
  - enable MFEM post-opt
  - quality/fitting weights
  - iteration cap
  - strict rollback toggle
- Add run log diagnostics in existing run-log storage path.
- Document recommended presets.

Exit criteria:
- End-to-end run from QGIS UI with reproducible results.
- Default behavior unchanged when MFEM disabled.

## Phase 5: Optional R&D Branch: MFEM-Native Generator (time-boxed, 2-3 weeks)

Goals:
- Evaluate feasibility of replacing some generation stages with MFEM-native operations.

Tasks:
- Prototype planar region/arc generation semantics directly in MFEM workflow.
- Compare complexity and robustness versus existing hybrid-cpp generation.

Exit criteria:
- Decision memo: continue or stop.
- Continue only if clearly superior in reliability and maintenance burden.

## 6. Validation and Acceptance Criteria

Use a dedicated GPU-priority validation flow for SWE2D impact, while meshing validation remains geometry-focused.

Core meshing acceptance metrics:
- Conformance:
  - max and p95 distance of mesh edges to constrained boundaries/arcs
  - fraction of constrained segments represented by mesh edges
- Quality:
  - min angle
  - max aspect ratio
  - min scaled Jacobian
  - inverted element count
- Stability:
  - deterministic output under fixed seed/options
  - no regression in existing hybrid transition tests

Suggested test additions:
- `tests/test_mfem_roundtrip_markers.py`
- `tests/test_mfem_tmop_quality.py`
- `tests/test_mfem_boundary_interface_fitting.py`
- `tests/test_mfem_strict_rollback.py`

Existing regression anchor:
- [tests/test_hybrid_cpp_channel_transition.py](tests/test_hybrid_cpp_channel_transition.py)

## 7. Build and Packaging Strategy

Use optional compilation and soft runtime fallback:
- Build flag OFF by default for local/dev environments.
- If MFEM unavailable at runtime, continue with existing backend path and emit clear warning.
- Keep CI matrix with and without MFEM.

Recommended CMake controls:
- `HYDRA_ENABLE_MFEM`
- `HYDRA_MFEM_USE_MPI` (OFF initially)
- `HYDRA_MFEM_USE_CUDA` (OFF initially for meshing stage)

Initial recommendation is serial MFEM integration first, then parallel path only if mesh preprocessing time becomes material.

## 8. Risks and Mitigations

Risk: Build complexity and dependency burden.
Mitigation: Optional feature flag, isolated bridge target, CI split.

Risk: Topology drift during optimization (losing constrained-edge intent).
Mitigation: marker locking, penalty weighting, strict rollback.

Risk: Non-deterministic optimization behavior.
Mitigation: fixed seeds, bounded solver settings, deterministic export ordering.

Risk: Runtime cost in interactive QGIS workflows.
Mitigation: tiered presets (fast/default/strict), iteration caps, abort on wall-time budget.

## 9. Recommended Default Presets

Fast preset:
- low iteration cap
- quality objective only
- no interface fitting

Balanced preset (default):
- moderate quality + moderate boundary fitting
- strict rollback enabled

Strict conformance preset:
- stronger boundary/interface fitting weights
- higher iteration cap
- rollback enabled

## 10. Decision Gate

Proceed with Phases 0 through 4 as the implementation baseline.

Only pursue Phase 5 (MFEM-native generation) if all are true:
- Phase 4 demonstrates robust measurable gains in conformance/quality.
- Build/deploy overhead remains manageable.
- Native generation prototype shows clear advantage over current hybrid-cpp generator.

This gives you a low-risk path to leverage MFEM where it is strongest while preserving your existing proven constrained generation stack.