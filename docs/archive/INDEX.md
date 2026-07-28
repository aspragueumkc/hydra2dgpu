# Archive Catalog

Historical record of plans, specs, audits, session logs, reviews, reports, and reference notes. The active catalog is in [docs/INDEX.md](../INDEX.md).

Entries are grouped by `status` (`complete` then `superseded`, with a small `active` section at the top for items intentionally archived while still active) and `type`. Each entry links to the canonical path.

---

## Active (in archive)

These files are in `docs/archive/` but are still `status: active`.

### Specs

- [Pipe1D–SWE2D Temporal Coupling: Single-Exchange Architecture](specs/PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md) — Single-exchange Pipe1D–SWE2D temporal coupling architecture synthesized from prior plans (completed 2026-07-22)

## Completed

### Plans

- [CLI-First Refactor Plan — HYDRA2DGPU](plans/CLI_FIRST_REFACTOR_PLAN.md) — Original 5-phase CLI-first refactor plan (Phase 0–4) — landed via Phase 1.A (completed 2026-07-25)
- [Storage→Pipe Interface Fix Plan](plans/2026-07-23-storage-pipe-interface-fix.md) — Replace HLLC at storage↔pipe with dedicated weir/orifice hydraulic control (completed 2026-07-25)
- [Drainage Module Finalization Plan](plans/2026-07-23-drainage-module-finalization.md) — Finalize drainage module — schema/models/tests/docs (11 tasks complete, 1 superseded) (completed 2026-07-25)
- [Unit-Agnostic TS/Profile Key Names — Implementation Plan](plans/2026-07-22-unit-agnostic-ts-keys.md) — Rename depth_m→depth, velocity_ms→velocity, wse_m→wse etc. across codebase (completed 2026-07-25)
- [Separate Structure and Drainage Coupling Flux Buffers](plans/2026-07-22-separate-struct-drain-flux.md) — Separate drainage and structure coupling flux buffers; re-enable dead struct path (completed 2026-07-25)
- [Pipe1D–SWE2D Temporal Coupling Fix](plans/2026-07-22-pipe1d-swe2d-temporal-coupling-fix.md) — Fix temporal integration asymmetry where pipe1d ran closed-system (race fix T7) (completed 2026-07-25)
- [Coupling Flux Buffer — Rename + Zero + Fix Structure Path](plans/2026-07-22-cpl-flux-buffer-fix.md) — Rename d_ext_struct_flux_* → d_ext_cpl_flux_* + zero buffer + re-enable structures (completed 2026-07-25)
- [Pipe1D Unification Implementation Plan](plans/2026-07-21-pipe1d-unification-audit-class4.md) — Close 9-failure pipe1D audit gate + wire HEC-22 inlet face end-to-end (completed 2026-07-25)
- [Pipe1D Failure Investigation — Theories and Strategies](plans/2026-07-21-pipe1d-failure-investigation.md) — Read-only investigation report — 8 remaining test failures grouped into 3 buckets (completed 2026-07-25)
- [Pipe1D Refactor — Phase 2.4/5 Cleanup (Delete legacy code paths)](plans/pipe1d_phase_2_4_cleanup_plan.md) — Delete legacy pipe1d kernels, host wrappers, bindings, and duplicate coupling paths (completed 2026-07-25)
- [Pipe1D Face-Indexed FVM Refactor Plan](plans/pipe1d_face_indexed_refactor_plan.md) — Pure face-indexed FVM pipe1d refactor (Phase 0–4); supersedes F1–F15 fixes (completed 2026-07-25)
- [Pipe1D Face-Indexed Refactor — Gap-Closure Follow-Up Plan (Parallelized)](plans/pipe1d_face_indexed_refactor_followup_plan.md) — Gap-closure follow-up (G1–G9) to face-indexed refactor; parallelized waves (completed 2026-07-25)
- [Plan — Delete compat shims in coupling.py (Phase 5a)](plans/coupling_compat_shim_removal_2026-07-20.md) — Phase 5a — delete Python-side compat shims in coupling.py, direct binding names (completed 2026-07-25)
- [Pipe1D CLI/GUI Parity Fix — 2026-07-20](plans/PLAN_pipe1d_cli_gui_parity_2026-07-20.md) — Make CLI build run contexts through the same functions and ordering as the GUI (completed 2026-07-25)
- [Network Profile Viewer — Implementation Plan](plans/2026-07-19-network-profile-viewer.md) — SWMM-style longitudinal profile viewer (dialog + map tool + chain editor) (completed 2026-07-25)
- [Pipe1d MUSCL-minmod + RK1 + Runtime Alpha Boost — Implementation Plan](plans/2026-07-18-pipe1d-muscl-rk1-alpha-plan.md) — Add MUSCL-minmod + RK1 + runtime friction alpha boost to pipe1d Godunov solver (completed 2026-07-25)
- [GeoPackage Explorer Enhanced Viewer — Implementation Plan](plans/2026-07-18-gpkg-explorer-enhanced-viewer.md) — Transform GPKG Explorer into full results viewer with blob deser + XY plots (completed 2026-07-25)
- [Pipe1D Slot Surcharge Stabilization — Implementation Plan](plans/pipe1d_slot_fix_plan.md) — Stabilize Preissmann slot surcharge (A-bounded growth, slot width floor, mom cap) (completed 2026-07-25)
- [Pipe1D Phase A — Implicit Pressure + Semi-Implicit Friction](plans/pipe1d_phase_a_implicit_plan.md) — Phase A semi-implicit pressure and friction changes, covered by the face-indexed refactor (completed 2026-07-25)
- [Pipe1D Godunov Rewrite — Fully-Dynamic Solver](plans/2026-07-17-pipe1d-godunov-rewrite.md) — Replace hybrid fully-dynamic solver with true 1D Godunov FVM (MUSCL+HLLE+RK2) (completed 2026-07-25)
- [Pipe1D SWMM Validation Plan](plans/2026-07-16-pipe1d-swmm-validation.md) — Catalog SWMM scenarios, build comparison harness + tolerance framework (completed 2026-07-25)
- [Pipe1D Known Gaps — Implementation Plan](plans/2026-07-16-pipe1d-known-gaps.md) — Wire BC kernels into swe2d_pipe1d_step and populate node-mapping arrays (completed 2026-07-25)
- [Non-Blocking Batch Runner Implementation Plan](plans/2026-07-16-nonblocking-batch-runner.md) — Move batch runner off QGIS main thread with modeless dialog and live progress (completed 2026-07-25)
- [True Dynamic-Wave Terms for pipe1d Implementation Plan](plans/2026-07-15-true-dynamic-wave-pipe1d.md) — Add convective (dq4) and Froude damping (sigma) to fully_dynamic pipe1d kernel (completed 2026-07-25)
- [Pipe1D Solver and Coupling Unit-Awareness Implementation Plan](plans/2026-07-15-pipe1d-unit-aware.md) — Make pipe1d solver + coupling kernels unit-aware (propagate k_mann, h_min, gravity) (completed 2026-07-25)
- [Pipe1D Solver Rewrite — Parallel-Track Test Plan](plans/2026-07-15-pipe1d-test-plan.md) — Test inventory for pipe1d solver rewrite — pure test-planning, no execution (completed 2026-07-25)
- [Pipe1D Solver Rewrite — Implementation Plan](plans/2026-07-15-pipe1d-solver-rewrite.md) — Per-link sub-cells rewrite; 14/22 steps completed then superseded by refactor (completed 2026-07-25)
- [Apply SWMM-style minor loss to diffusion_wave pipe1d kernel](plans/2026-07-15-diffusion-wave-swmm-loss.md) — Apply SWMM-style minor loss to diffusion-wave pipe1d kernel (completed 2026-07-25)
- [Batch Snapshot JSON / Mesh Selector Redesign Implementation Plan](plans/2026-07-15-batch-snapshot-json-mesh-selector.md) — Reuse replay JSON builder for Batch dialog; replace GPKG+mesh combo with picker (completed 2026-07-25)
- [runtime_GUI_edit_mode](plans/runtime_GUI_edit_mode.md) — Runtime GUI editor for SWE2D — AST-based source rewriting for Qt design (completed 2026-07-25)
- [Workbench Session Persistence — Plan](plans/WORKBENCH_PERSISTENCE_PLAN.md) — Workbench session persistence via QMainWindow.saveState/restoreState → QSettings (completed 2026-07-25)
- [Test Migration & Rename Plan: Drainage Pipe1D Tests](plans/TEST_MIGRATION_PLAN.md) — Drainage Pipe1D test rename + add swe2d_pipe1d_readback_node_state binding (completed 2026-07-25)
- [Rainfall Source Term Re-evaluation Optimization — Revised](plans/RAINFALL_SOURCE_OPTIMIZATION_PLAN.md) — Rainfall SCS-CN re-eval interval (default 60s) — constant source during interval (completed 2026-07-25)
- [1D Pipe Network Solver — Design & Implementation Plan](plans/PIPE1D_SOLVER_PLAN.md) — Original 1D pipe network solver design — Diffusion Wave + Fully Dynamic (completed 2026-07-25)
- [GUI Tooltip Audit & Implementation Plan](plans/GUI_TOOLTIP_AUDIT_PLAN.md) — GUI tooltip audit + phased implementation plan across 14 view files (completed 2026-07-25)
- [GPU-Direct Real-Time Viewer](plans/GPU_DIRECT_VIEWER.md) — GPU-direct real-time viewer — zero D2H via CUDA-OpenGL interop (completed 2026-07-25)
- [Drainage Solver Equation Parity Plan — SWMM Alignment](plans/DRAINAGE_EQUATION_PLAN.md) — Drainage solver equation parity — align 3 modes with SWMM equation sets (completed 2026-07-25)
- [Drainage Link Profile Step 1 Implementation Plan](plans/2026-07-14-drainage-link-profile-step1.md) — Persist per-cell pipe geometry from C++ readback; render sloped invert/crown (completed 2026-07-25)
- [Fix Pipe-Cell + Overlay Field Implementation (with SI/USC Rain Units)](plans/2026-07-13-fix-pipe-cell-overlay-implementation.md) — Fix broken pipe-cell + overlay implementation with unit-aware display conversions (completed 2026-07-25)
- [Parallel Execution Plan: Fix Pipe-Cell + Overlay Implementation](plans/2026-07-13-fix-pipe-cell-overlay-implementation-PARALLEL.md) — Parallel-execution variant of the 2026-07-13 fix plan (completed 2026-07-25)
- [Drainage Per-Pipe-Cell Persistence + Overlay Rain/CN/Manning Fields — Implementation Plan](plans/2026-07-13-drainage-pipe-cell-output-and-overlay-fields.md) — Persist per-pipe-cell timeseries + overlay rain/CN/Manning scalar fields (completed 2026-07-25)
- [Structure / Drainage Group Separation — Implementation Plan](plans/2026-07-12-structure-drainage-group-separation.md) — Restructure Model tab Structures & Drainage page into distinct groups (completed 2026-07-25)
- [Pipe1D Implicit Friction Implementation Plan](plans/2026-07-12-pipe1d-implicit-friction.md) — Implicit friction+minor loss in pipe1d kernels matching 2D SWE solver treatment (completed 2026-07-25)
- [Module Enable/Disable Toggles — Implementation Plan](plans/2026-07-12-module-enable-disable-toggles.md) — Add three enable/disable toggles (structures, drainage, rainfall) to Model tab (completed 2026-07-25)
- [RCMK Ordering Elimination Plan](plans/2026-07-10-rcmk-ordering-elimination.md) — Eliminate pre-RCMK vs post-RCMK ordering mismatches by reading from C++ handle (completed 2026-07-25)
- [GPKG Delete by Run ID Redesign — Implementation Plan](plans/2026-07-10-gpkg-delete-by-run-id.md) — Two-step wizard for GPKG Explorer multi-run delete-by-id UI (completed 2026-07-25)
- [Advanced Spatial Reconstruction Schemes — Implementation Plan](plans/2026-07-10-advanced-spatial-schemes.md) — Three new reconstruction schemes (Barth-Jespersen, WENO3, MP5) implementation plan (completed 2026-07-25)
- [1D Pipe Network Surcharge / Pressurized Flow — Volume Decomposition Design](plans/2026-07-09-1d-pipe-surcharge-volume-decomposition-design.md) — Volume-decomposition surcharge design — historical, superseded by face-indexed refactor (completed 2026-07-25)
- [Simulation Threading Implementation Plan](plans/2026-07-05-simulation-threading-plan.md) — Move simulation off QGIS main thread via SimulationWorker QThread (completed 2026-07-25)
- [SWE2D Tier 1 #5-10 + Tier 2 — Implementation Plan](plans/2026-07-04-swe2d-tier1-5-10-tier2.md) — Tier 1 #5–10 + Tier 2 audit finding closures (12 deferred items) (completed 2026-07-25)
- [SWE2D Structural Placement Fixes — Implementation Plan](plans/2026-07-04-swe2d-placement-fixes.md) — SWE2D structural placement fixes — strict MVP per STRUCTURAL_PLACEMENT_AUDIT (completed 2026-07-25)
- [Simultaneous Live + GPKG Results Viewing — Implementation Plan](plans/2026-07-04-live-gpkg-simultaneous.md) — Live+GPKG simultaneous viewing implementation — _live_run_id + overlay select (completed 2026-07-25)
- [Workbench Service Relocation Implementation Plan](plans/2026-07-03-workbench-service-relocation.md) — Move pure-Python services out of swe2d/workbench/services/ into shared layer (completed 2026-07-25)
- [SWE2D GUI/UX Improvements Implementation Plan](plans/2026-07-03-gui-ux-improvements.md) — SWE2D GUI/UX improvements implementation across workbench views (completed 2026-07-25)
- [Structures Attribute Form Cleanup Plan](plans/2026-06-25-structures-form.md) — Clean up QGIS structures attribute form — visibility by structure type, defaults (completed 2026-07-25)
- [Results Architecture Cleanup Plan](plans/2026-06-25-results-cleanup.md) — Move scattered results state into SWE2DResultsData, fix MVP violations (completed 2026-07-25)
- [In-Memory Results Viewing Plan](plans/2026-06-25-in-memory-results.md) — Eliminate GPKG writes during runs — viewers read from in-memory snapshots (completed 2026-07-25)
- [QGIS Batch UI Implementation Plan](plans/2026-06-24-qgis-batch-ui.md) — QGIS batch UI — QTableWidget parameter grid + subprocess monitor (completed 2026-07-25)
- [Mesh Persistence Implementation Plan](plans/2026-06-24-mesh-persistence.md) — Mesh persistence — save/load solver arrays via GPKG zlib BLOBs (completed 2026-07-25)
- [Headless Runner Implementation Plan](plans/2026-06-24-headless-runner.md) — Headless runner — `hydra run` CLI reading GPKG without QGIS (completed 2026-07-25)
- [MVP Hardening Implementation Plan](plans/2026-06-22-mvp-hardening.md) — MVP hardening — fix 3 architecture violations (missing protocol, callback, plotting) (completed 2026-07-25)

### Specs

- [Unit-Agnostic Line TS / Profile Key Names](specs/2026-07-22-unit-agnostic-ts-keys-design.md) — Unit-agnostic TS/profile key names (depth_m→depth etc.) design (completed 2026-07-25)
- [Pipe1D Unification — Close Audit Gates + Wire Class-4 Inlet](specs/2026-07-21-pipe1d-unification-audit-class4-design.md) — Close 9-failure pipe1D audit gate + wire class-4 inlet end-to-end (completed 2026-07-25)
- [Network Profile Viewer — Design Spec](specs/2026-07-19-network-profile-viewer-design.md) — NetworkProfileDialog + ProfileChainWidget + ProfilePlotWidget design (completed 2026-07-25)
- [Pipe1d MUSCL-minmod + RK1 + Runtime Alpha Boost](specs/2026-07-18-pipe1d-muscl-rk1-alpha-design.md) — MUSCL-minmod slope + RK1 + runtime alpha boost plumbing design (completed 2026-07-25)
- [GeoPackage Explorer Enhanced Viewer — Design Spec](specs/2026-07-18-gpkg-explorer-enhanced-viewer-design.md) — Enhanced GPKG Explorer design — array viewer, filter bar, inline plots (completed 2026-07-25)
- [Non-Blocking Batch Runner Design](specs/2026-07-16-nonblocking-batch-runner-design.md) — BatchManager service + BatchWorker thread + modeless dialog MVP (completed 2026-07-25)
- [Batch Simulation Snapshot / Mesh Selector Redesign](specs/2026-07-15-batch-snapshot-json-mesh-selector-design.md) — Batch dialog snapshot reuses replay JSON; mesh picker replaces combo (completed 2026-07-25)
- [On-Device Line Metrics — Kernel Interface Spec](specs/on-device-line-metrics-spec.md) — On-device line metrics kernel interface — zero Python per snapshot (completed 2026-07-25)
- [Fix RK2 Stale Coupling Sources + Implement Higher-Order Temporal Schemes](specs/TEMPORAL_SCHEME_FIX_SPEC.md) — RK2 stale coupling sources + higher-order temporal schemes (RK3/RK4/RK5) (completed 2026-07-25)
- [Results Persistence & Rendering Bug Fix Specification](specs/RESULTS_PERSISTENCE_BUGFIX_SPEC.md) — Results persistence/rendering bug fixes (Status: Completed 2026-07-01) (completed 2026-07-25)
- [Baked Mesh & Results — GPKG BLOB Storage Specification](specs/BAKED_MESH_RESULTS_SPEC.md) — Clean-sheet GPKG BLOB storage spec (no backward compat; old tables dead) (completed 2026-07-25)
- [HYDRA2DGPU QGIS Locator Integration — Design Specification](specs/2026-07-14-hydra2dgpu-qgis-locator-design.md) — QGIS Locator integration — register `hydra` prefix with 7 core actions (completed 2026-07-25)
- [Drainage Link Profile Viewer Enhancement](specs/2026-07-14-drainage-link-profile-design.md) — Approved — sloped invert/crown profile with velocity-shaded water fill (completed 2026-07-25)
- [Coupling Distribute Lines — Design Spec](specs/2026-07-11-coupling-distribute-lines-design.md) — Approved — unified coupling line mechanism for culverts/pipe-ends/outfalls (completed 2026-07-25)
- [GPKG Explorer: Delete by Run ID Redesign](specs/2026-07-10-gpkg-delete-by-run-id-redesign.md) — Two-step wizard redesign for GPKG Explorer delete-by-run-id flow (completed 2026-07-25)
- [Advanced Spatial Reconstruction Schemes — Design Spec](specs/2026-07-10-advanced-spatial-schemes-design.md) — Design spec for three new spatial reconstruction schemes (WENO3, MP5, BJ) (completed 2026-07-25)
- [Simulation Threading Design — HYDRA2DGPU Workbench](specs/2026-07-05-simulation-threading-design.md) — Simulation threading design (CUDA backend thread-affinity constraint) (completed 2026-07-25)
- [Simultaneous Live + GPKG Results Viewing](specs/2026-07-04-live-gpkg-simultaneous-design.md) — Live + GPKG results viewing simultaneously with overlay radio-select (completed 2026-07-25)
- [Batch Multi-Sim, CLI, and Mesh Persistence Design](specs/2026-06-24-batch-multisim-cli-design.md) — Batch multi-sim + CLI + mesh persistence design (completed 2026-07-25)

### Audits

- [Drainage Network Coupling Audit — 2026-07-22](audit/pipe1d_refactor_audit_2_2026-07-22_15-09.md) — Full drainage coupling data-flow audit identifying a stacked zero-transfer failure chain (completed 2026-07-25)
- [Pipe1D Refactor — Implementation Audit against Plan](audit/pipe1d_refactor_audit_2026-07-19.md) — Pipe1D refactor implementation audit against plan — coupling.py NOT ported (completed 2026-07-25)
- [Codebase Audit — 2026-07-19 (Pre-Release)](audit/CODEBASE_AUDIT_2026-07-19.md) — Pre-release codebase audit covering bugs, architecture, documentation, and hygiene (completed 2026-07-25)
- [Pipe1D Solver & Coupling Audit — 2026-07-17](audit/PIPE1D_AUDIT_2026-07-17.md) — Pipe1d solver and coupling audit of mass conservation and non-physical coupled results (completed 2026-07-25)
- [SWE2D Structural Placement Audit](audit/STRUCTURAL_PLACEMENT_AUDIT.md) — Structural placement audit — 8 parallel agents, MVP + logical-scope rules (completed 2026-07-25)
- [CLI/GPKG Adapter Audit — 2026-07-02](audit/CLI_GPKG_ADAPTER_AUDIT.md) — CLI/GPKG adapter audit — crash on line_output_interval + QGIS import violation (completed 2026-07-25)
- [Unit Handling Consistency Audit & Remediation Plan](audit/AUDIT_UNIT_HANDLING_CONSISTENCY.md) — Unit suffix audit — 1000+ variables with _m/_ms/_cms, phasing rename (completed 2026-07-25)
- [Implementation Gap Audit: Drainage Per-Pipe-Cell + Overlay Fields](audit/AUDIT_PIPE_CELL_OVERLAY_GAP.md) — Implementation gap audit — cc._dsoa bug + 1000× rain intensity + non-shape depth (completed 2026-07-25)

### Session Logs

- [Debug RunController sender error and plan audit](session/threading_implementation_debugging.md) — Debug RunController sender() AttributeError + audit plan (2026-07-05) (completed 2026-07-25)
- [GMSH path mesh renumbering through workbench?](session/session-ses_1007.md) — Session log — GMSH path mesh renumbering through workbench? (ncu CUDA path) (completed 2026-07-25)
- [New session - 2026-06-30T11:34:20.003Z](session/session-ses_0e7b%28higher-order-coupling-issues%29.md) — Session log — gmsh target_size constraints layer exploration (completed 2026-07-25)
- [Coupling kernels CUDA graph bug](session/session-ses_0e56.md) — Session log — Coupling kernels CUDA graph bug investigation (completed 2026-07-25)
- [HYDRA GUI/UX improvements implementation plan](session/session-ses_0d79.md) — Session log — HYDRA GUI/UX improvements implementation plan (completed 2026-07-25)
- [Agent Session Recovery Log](session/AGENT_SESSION_RECOVERY_LOG.reference.md) — Duplicate recovery log, kept for history (completed 2026-07-25)
- [Agent Session Recovery Log](session/AGENT_SESSION_RECOVERY_LOG.md) — Canonical rolling agent session recovery log (completed 2026-07-25)

### Reference

- [Verification Review — Post-Fix (refactor/cli-first @ 516e223)](review/VERIFICATION_REVIEW.md) — Post-fix verification review of critical corrections and regression tests (completed 2026-07-25)
- [Refactor Phase 1 Review](review/REFACTOR_PHASE1_REVIEW.md) — CLI-first Phase 1 review of canonical builder compliance and code quality (completed 2026-07-25)
- [Comprehensive Review — CLI-First Refactor (Phase 0–1)](review/COMPREHENSIVE_REVIEW.md) — Comprehensive CLI-first Phase 0–1 review of spec alignment, quality, and blockers (completed 2026-07-25)
- [CLI-First Refactor + HYDRA2DGPU MCP — Progress Report](reports/CLI_FIRST_MCP_PROGRESS_REPORT.md) — CLI-first and MCP progress report covering completed phases, fixes, and green tests (completed 2026-07-25)
- [Drainage Module Reference](references/drainage_module_reference.md) — Technical reference for drainage architecture, data model, GPU pipeline, and 2D coupling (completed 2026-07-25)
- [CLI/GUI Parity — Drift Prevention](references/CLI_GUI_PARITY_DRIFT.md) — Reference for enforced CLI/GUI parity and prevention of future runtime drift (completed 2026-07-25)
- [Unit-System Assumptions for this Repository](references/UNIT_ASSUMPTIONS_AND_USC_DEFAULT.md) — Repository unit-system assumptions and USC defaults for solver and drainage work (completed 2026-07-25)
- [Simulation Configuration Table](references/SIMULATION_CONFIG_TABLE.md) — `swe2d_simulation_configs` table — persist widget_state as JSON BLOB (completed 2026-07-25)
- [Qt GUI UX Improvement Recommendations](references/QT_GUI_UX_IMPROVEMENTS.md) — Qt GUI UX improvement recommendations — 10 easy-wins with impact/effort (completed 2026-07-25)
- [SWE2D GUI — In-Depth Analysis & Recommendations](references/GUI_UX_RECOMMENDATIONS.md) — SWE2D GUI in-depth analysis — naming, layout, QGroupBox recommendations (completed 2026-07-25)
- [Multi-Link Drainage Network Profile Viewer — Architecture Sketch](references/DRAINAGE_PROFILE_VIEWER_ARCH.md) — Multi-link drainage network profile viewer architecture sketch (completed 2026-07-25)
- [Boundary Condition Handling: GUI vs CLI Paths](references/BC_HANDLING_GUI_VS_CLI.md) — Boundary condition handling — GUI vs CLI paths investigation (completed 2026-07-25)
- [HYDRA2D Model Setup Panel, Setup Tab Redesign](references/ALS Notes/Setup Tab Depreciation.md) — Setup-tab redesign notes (load layers, mesh setup relocation, utilities drop) (completed 2026-07-25)
- [Inlet Prescribed-Flow BC + Node Storage Fix](references/INLET_PRESCRIBED_FLOW_BC.md) — Inlet prescribed-flow BC + node storage fix (✅ Implemented 2026-07-12) (completed 2026-07-25)
- [GPU Kernel Strategy](cpp/GPU_KERNEL_STRATEGY.md) — GPU kernel design philosophy — SoA layout, edge-centric flux, graph caching (completed 2026-07-25)
- [HDS-5 Culvert Implementation](cpp/CULVERT_HDS5.md) — HDS-5 culvert implementation — 5 flow regimes + USC unit conversion (completed 2026-07-25)
- [Coupling Kernels (Surface ↔ Drainage ↔ Structures)](cpp/COUPLING_KERNELS.md) — Coupling kernels overview — 3-way SWE↔drainage↔structures GPU exchange (completed 2026-07-25)
- [C++ / CUDA Architecture](cpp/ARCHITECTURE.md) — C++/CUDA module layout, build system, target mapping (completed 2026-07-25)

## Superseded

### Plans

- [Pipe-end Face-Flux Coupling Plan](plans/pipe_end_face_flux_plan.md) — Replace source-term pipe-end coupling with mass/momentum-conserving face-flux (completed 2026-07-25)
  - superseded_by: [PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md](specs/PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md)
- [Pipe1D Unified Semi-Implicit Solver — Implementation Plan](plans/pipe1d_casulli_hu_plan.md) — Casulli/Hu semi-implicit θ-method replacement for RK2 Godunov + slot solver (completed 2026-07-25)
  - superseded_by: [pipe1d_face_indexed_refactor_plan.md](plans/pipe1d_face_indexed_refactor_plan.md)
- [Pipe1D Mass-Conservation & Physics Fixes Implementation Plan](plans/2026-07-17-pipe1d-mass-conservation-fixes.md) — F1–F15 mass-conservation fix plan, superseded by face-indexed refactor (completed 2026-07-25)
  - superseded_by: [pipe1d_face_indexed_refactor_plan.md](plans/pipe1d_face_indexed_refactor_plan.md)

### Specs

- [Pipe1D Solver Rewrite — Technical Specification](specs/2026-07-15-pipe1d-solver-rewrite-spec.md) — SWMM-aligned pipe1d rewrite (sub-cells, sub-cell states, RK2/Picard) (completed 2026-07-25)
  - superseded_by: [PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md](specs/PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md)

## Unparsed

Files without parseable frontmatter; not part of the catalog. Likely the migration working draft itself.

- `docs/archive/.migration/inventory.md`
