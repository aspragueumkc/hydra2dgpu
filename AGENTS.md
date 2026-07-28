# HYDRA — GPU-Accelerated 2D SWE Plugin for QGIS

HYDRA is a QGIS plugin for 2D shallow water equation (SWE) modeling: a CUDA-accelerated
finite-volume (FVM) solver on unstructured meshes, coupled with 1D SWMM-style urban
drainage, hydraulic structures (weirs, culverts, gates, bridges, pumps), and
rainfall/infiltration. The GUI is a PyQt5 "Studio" workbench (dock-based, MVP
architecture) embedded in the QGIS map canvas.

**Hard requirement:** NVIDIA CUDA GPU (Compute Capability ≥ 7.5). There is no CPU
fallback path. Linux and Windows only.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   # add -DSWE2D_STATE_FP32=ON for experimental mixed precision
make -j$(nproc)                       # outputs hydra_*.so into build/
```

Python deps: `pip install -r requirements.txt` into the QGIS Python env.
**Never** pip-install `qgis`, `PyQt5`, or `osgeo` — they come from QGIS itself.

## Test

**Pre-claim check (run before claiming any work is done):**

```bash
# Fast-fail set: ~20s for the tight loop, ~60s with HYDRA_MCP_INTEGRATION=1.
# Catches the 12 LLM failure modes catalogued in
# docs/audit/2026-07-26-test-infrastructure-audit.md §4.
mamba run -n qgis_stable bash tools/fast_fail.sh
```

**GPU-touching work** (CUDA kernels, drainage coupling, pipe-cell paths): before
claiming such work is done, also run:

```bash
# Compute Sanitizer (memcheck) on the dambreak test
python3 tools/run_compute_sanitizer.py --tool memcheck --test dambreak
# GPU validation suite (primary acceptance gate)
PYTHONPATH="$PWD:$PWD/build" mamba run -n qgis_stable python3 -m unittest -v \
  tests.test_swe2d_gpu_validation_perf \
  tests.test_swe2d_gpu_unstructured \
  tests.test_swe2d_gpu_dambreak
```

The fast-fail set does NOT include GPU tests because the hosted CI runner has
no GPU — GPU tests are local-only and agent-claimed.

- Validation priority: `tests/test_swe2d_gpu_validation_perf.py` (primary) →
  `tests/test_swe2d_gpu_unstructured.py` → `tests/test_swe2d_gpu.py` (informational only,
  do not drive design from it).
- GPU memory/race checking: `python tools/run_compute_sanitizer.py --tool memcheck --test dambreak`
- Full test list: `.github/workflows/test.yml`.
- See `docs/specs/2026-07-26-test-discipline-design.md` for the discipline spec.

## Layout

```
swe2d/                   Python package (solver API, extensions, workbench)
  runtime/               Backend creation and GPU interface
  extensions/            Drainage, structures, rainfall modules
  boundary_and_forcing/  BC sampling and hydrograph handling
  mesh/                  Mesh I/O and topology
  results/               Result queries, export, run management
  workbench/             QGIS workbench (views, controllers, dialogs)
cpp/src/                 CUDA/C++ solver, mesh, numerics, pybind11 bindings
tests/                   Solver validation and GPU performance tests
tools/                   Build helpers and dev utilities (incl. tools/memory.py)
docs/                    Design notes, guides, Doxygen (start at docs/INDEX.md)
  memory/                Curated agent memory (active/, review-pending/, superseded/)
```

- **Memory:** Curated agent memory lives in `docs/memory/active/`. Use
  `/remember <text>` to capture and `/recall <query>` to prepend the top hits
  (≤ 4000 tokens) to the next message. The CLI `tools/memory.py` is the only
  writer of `docs/memory/`. See the `hydra-agent-memory` skill.
- Current rolling session log: `docs/archive/session/AGENT_SESSION_RECOVERY_LOG.md`
  (status: complete as of 2026-07-25). Start a new rolling log under
  `docs/session/` (gitignored working notes) for in-flight captures; only
  curated lessons graduate to `docs/memory/active/`.

## Conventions — read before editing

- **Units:** Never assume SI or USC. All conversions go through `swe2d.units`
  (`_u.configure(scale)`, `_u.gravity()`, `_cms_to_model()`, etc.).
- **Computation source of truth:** If a GPU kernel computes a value internally, that
  value must come from the kernel (add a device buffer + D2H readback) — never
  re-compute it in Python. See `.agents/computation-source-truth.md`.
- **No premature backwards compatibility:** No compat shims/fallbacks for API shapes
  that never shipped. Fix callers instead. See
  `.opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md`.
- **PyQt5 widget liveness:** SIP wrappers outlive their C++ QObject. Guard accesses
  with the `objectName()` liveness-check pattern (see the `pyqt5-desktop-patterns`
  skill). Use `safe_disconnect()` for signal handlers.

## Docs lifecycle (active vs complete/superseded)

- Active plans live under `docs/plans/`. Active specs (including designs) live
  under `docs/specs/`.
- Completed and superseded documents move to `docs/archive/{plans,specs}/`.
- Lifecycle metadata lives in YAML frontmatter at the top of each file:

  ```
  ---
  type: plan | spec | guide | audit | session-log | reference
  status: active | complete | superseded
  created: YYYY-MM-DD
  completed: YYYY-MM-DD        # when not active
  superseded_by: path/to/...   # when superseded
  progress:                    # plans only; see spec §6.1
    total: N
    done: M
    current: "K. Step title"
    blockers: []
    last_updated: YYYY-MM-DD
  ---
  ```

- This project overrides the superpowers skill default of
  `docs/superpowers/{plans,specs}/`. Do not recreate that folder.
- Agents must NOT mark a document `complete` or `superseded` automatically.
  After verification passes, ask:

  > All steps in `<path>` have passed verification. May I mark it `complete`
  > and move it to `docs/archive/<type>/`?

  Wait for explicit user approval before editing/moving.
- When finishing a plan step, update both the task checkbox AND the
  frontmatter `progress:` block in the same edit.
- Agents do NOT scan `docs/archive/` for context. Read archived files only
  when the user names them or asks for history.

## Agent resources

- **Skills** (`.agents/skills/`): `fvm-cfd-solver-patterns` (mesh/solver/CUDA/BC/coupling
  patterns, validation workflow), `hydra2dgpu-studio-ui` (Studio UI architecture,
  signal safety, state persistence, CLI/GUI parity), `hydra-agent-memory`
  (capture and recall via `tools/memory.py`), `subagent-driven-development`,
  `frontend-design`, `skills-discovery`.
- **Reference notes:** `.agents/studio-gui-api.md` (Studio MVP public API contract),
  `.agents/computation-source-truth.md`, `.agents/codebase-audit.md`.
- **Docs:** `docs/INDEX.md` (index by audience), `docs/DEVELOPER_GUIDE.md`,
  `docs/SWE2D_GPU_ARCHITECTURE_REPORT.md`, `docs/MODEL_GEOPACKAGE_SCHEMA.md`,
  `docs/RESULTS_GEOPACKAGE_SCHEMA.md`.
- **Knowledge graph:** `graphify-out/GRAPH_REPORT.md` (pre-built codebase graph).
- **MCP:** `.kimi-code/mcp.json` configures `context7` for up-to-date library docs
  (PyQt5/QGIS, CUDA, pybind11, Gmsh). GitHub operations: use the authenticated `gh`
  CLI via Bash. GeoPackage/SQLite inspection: use the `sqlite3` CLI via Bash.
