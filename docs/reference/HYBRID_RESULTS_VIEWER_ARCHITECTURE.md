# Hybrid Results Viewer Architecture (QGIS Canvas + GPU-Ready Viewer)

## 1. Goal

Deliver a hybrid results-viewing system with:

- A QGIS-native results map panel for immediate production use.
- A high-performance map-canvas overlay renderer that can evolve toward GPU-native rendering.
- Shared data/cache services so both viewer paths use one results pipeline.

This architecture must support:

- Time-varying mesh results (depth, WSE, velocity components, derived fields).
- Velocity vectors and streamline/tracer visualization.
- Multi-run comparison workflows.
- Large GeoPackage result sets without per-frame feature rebuild bottlenecks.

## 2. Constraints and Principles

- GPU-first direction for SWE2D remains unchanged.
- Results visualization should avoid repeated CPU-heavy geometry churn.
- Persisted results in GeoPackage/UGRID/HEC-RAS exports remain first-class inputs.
- Keep incremental compatibility with existing workbench UI.
- Preserve fallback behavior when advanced viewer dependencies are unavailable.

## 3. Top-Level Hybrid Design

### 3.1 Panel A: Results Map Panel (QGIS-native)

A plugin-owned panel with dedicated map canvas behavior and controls:

- Uses QGIS/MDAL mesh layers for scalar/vector rendering where possible.
- Supports standard QGIS interactions (pan/zoom/identify/select/CRS).
- Serves as default viewer for broad compatibility.

### 3.2 Panel B: High-Performance Canvas Overlay (primary high-perf path)

A custom rendering overlay designed for high frame-rate and future all-GPU features:

- Current backend: NumPy-rasterized unstructured fields drawn via a georeferenced `QgsMapCanvasItem`.
- Future backend: CUDA/VTK-m/OpenGL compute for streamline particles and interpolation.
- Render loop decoupled from QGIS per-feature vector layer update constraints.

### 3.3 Shared Core Services

Both panels consume the same shared services:

- Results index and metadata service.
- Time cursor and animation controller.
- Mesh/field cache manager.
- Derived field calculator (velocity magnitude, Froude, etc.).
- Diagnostics and profiling service.

## 4. Component Architecture

### 4.1 Results Data Access Layer

Responsibilities:

- Discover runs, timesteps, and available fields from GeoPackage and exported mesh formats.
- Provide query APIs for exact timestep retrieval (no repeated loose time scans).
- Support per-run mesh mapping and schema compatibility handling.

Proposed interface sketch:

- list_runs(source)
- list_timesteps(source, run_id)
- get_field(source, run_id, t_s, field_name)
- get_vector(source, run_id, t_s, u_field, v_field)
- get_mesh(source, run_id)

Implementation notes:

- Use SQLite indices on (run_id, t_s, cell_id).
- Keep provider-specific adapters (GeoPackage, UGRID, HDF) behind one API.

### 4.2 Results Cache Manager

Responsibilities:

- Multi-level cache:
  - L1: current/adjacent timesteps in RAM.
  - L2: memory-mapped arrays for recent runs.
  - L3: optional GPU buffer cache in high-performance overlay renderer.
- Frame-window prefetch around the active timestep.
- Cache invalidation on source/run/layer switch.

### 4.3 Time and Playback Controller

Responsibilities:

- Single source of truth for active run/time.
- Emits frame events to all panels.
- Debounces rapid scrubbing and drops stale render requests.
- Supports lockstep multi-run compare mode.

### 4.4 QGIS Results Map Renderer

Responsibilities:

- Default renderer path using Mesh Layer (MDAL) for depth/WSE/vector fields.
- Optional vector overlay path retained for diagnostics and custom glyph styling.
- View-extent culling for any feature-based overlays.

Performance rules:

- Never rebuild full feature layers per frame.
- Use batch updates where overlays remain necessary.
- Prefer mesh-native rendering to feature geometry churn.

### 4.5 High-Performance Viewer Renderer

Responsibilities:

- Render unstructured mesh scalars/vectors at high FPS.
- Streamline/tracer pathline rendering.
- Progressive level-of-detail and density controls.

Current status (MVP started):

- Implemented a high-performance map-canvas overlay using NumPy rasterization of unstructured cell snapshots (depth/speed/WSE) to bypass per-feature QGIS map updates for frame playback.
- Removed the separate high-performance dock panel path from the active UI; overlay controls (field/colormap/resolution/auto-contrast/opacity) are now the primary high-performance control surface.

Roadmap:

- Stage 1: CPU-fed VTK rendering.
- Stage 2: GPU-uploaded field buffers and persistent mesh VBOs.
- Stage 3: GPU particle advection/interpolation kernels.

### 4.6 Derived Field and Analysis Engine

Responsibilities:

- Compute derived fields (speed, Froude, shear proxies, etc.) once per frame.
- Keep deterministic outputs across both panels.
- Optional offload path for GPU computations in high-performance mode.

## 5. Streamline/Tracer Strategy

### 5.1 Near-term

- Enable streamlines through existing mesh-capable stack when available (MDAL/Crayfish path).
- Add basic seeding controls in panel UI (line/area/random/manual seed points).
- Add QGIS-native per-frame streamline trace overlay from cached velocity snapshots (implemented baseline).

### 5.2 Mid-term

- VTK-based stream tracer in dedicated panel using unstructured velocity field.
- Persistent seed sets and animated particle trails.

### 5.3 Long-term GPU

- Device-resident mesh + velocity field + tracer particles.
- CUDA/compute-shader advection and interpolation.
- Minimal CPU readback; frame composition fully GPU-side.

## 6. Data Model for Viewer State

Unified viewer state object:

- active_source
- active_run_id
- active_time_s
- active_scalar_field
- active_vector_field
- density_level
- min_speed
- streamline_enabled
- tracer_mode
- compare_mode
- panel_mode (qgis_native | high_perf)

Persist in project/workbench settings to restore session state.

## 7. Phased Implementation Plan

### Phase A: Stabilize QGIS-native Panel (short-term)

- Move default result visualization to mesh-layer path where available.
- Keep optimized vector overlay fallback for unsupported cases.
- Add viewport culling and frame-budget safeguards.

Deliverables:

- Results Map Panel MVP integrated into existing workbench.
- Performance telemetry (frame time, fetch time, draw time) in log.

### Phase B: Shared Core Services (short-term to mid-term)

- Introduce Results Data Access Layer and Cache Manager.
- Route current panel and overlay code through shared services.
- Add prefetch and exact-time indexing.

Deliverables:

- Unified API with adapters for GeoPackage + mesh exports.
- Deterministic run/time behavior across all viewers.

### Phase C: High-Performance Panel v1 (mid-term)

- Add optional dedicated viewer dock with VTK/OpenGL backend.
- Implement scalar/vector mesh rendering and synchronized time playback.

Deliverables:

- Experimental panel toggle in UI.
- Feature parity baseline: depth/WSE/velocity display + animation.

### Phase D: Streamline/Tracer + GPU Acceleration (mid-term to long-term)

- Add streamlines/pathlines and advanced seeding controls.
- Introduce GPU buffer residency and particle advection kernels.

Deliverables:

- High-density tracer rendering at interactive frame rates.
- Profiling-backed performance targets documented and validated.

## 8. Performance Targets

Initial targets for representative large runs:

- QGIS-native panel: responsive scrubbing and >8-12 FPS visual update at practical density settings.
- High-performance panel v1: >20 FPS for scalar mesh animation on large unstructured domains.
- High-performance GPU tracer mode: >30 FPS with configurable particle budgets.

## 9. Dependency and Integration Options

Primary options:

- QGIS Mesh Layer + MDAL for native path.
- Crayfish tools where available for advanced mesh visualization workflows.
- VTK/QVTK (or equivalent) for custom panel backend.

Fallback policy:

- If optional dependencies are missing, remain on QGIS-native path and disable unsupported controls with clear messaging.

## 10. Risks and Mitigations

Risk: fragmentation between native and high-performance panels.

- Mitigation: enforce shared core services and one viewer state model.

Risk: high complexity of GPU tracer implementation.

- Mitigation: stage through CPU/VTK baseline before GPU kernels.

Risk: large-memory pressure from multi-run caching.

- Mitigation: bounded cache budgets, LRU eviction, adaptive density/LOD.

## 11. Immediate Next Actions

1. Implement a new Results Map panel mode that prefers Mesh Layer (MDAL) rendering by default.
2. Introduce a shared Results Data Access service skeleton and redirect current overlay retrieval through it.
3. Add frame-time instrumentation around fetch/build/draw stages.
4. Draft High-Performance panel API contract (renderer abstraction + state sync hooks).

## 12. Definition of Done (Architecture Draft)

This draft is complete when:

- Panel responsibilities, shared services, and phased rollout are explicitly defined.
- The path from current overlay prototype to GPU-capable viewer is incremental and testable.
- Implementation can begin without re-deciding foundational architecture.
