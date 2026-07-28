---
type: plan
status: complete
created: 2026-07-16
completed: 2026-07-25
---

# Pipe1D SWMM Validation Plan

> **Spec:** `docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md` §2.17 (originally
> out of scope; explicitly added per user direction 2026-07-16).
> **Existing infrastructure:** `pyswmm 2.1.0` + `swmm-toolkit 0.17.0` installed in `qgis_stable`;
> `tests/swmm_runner.py`, `tests/pipe1d_runner.py`, and several `test_*_vs_swmm.py` tests
> already exercise the comparison pathway.
> **Strategy:** Mostly serial — catalog canonical SWMM scenarios, build automated comparison
> harness around existing runners, define tolerance framework, run validation matrix, generate
> regression baseline. Subagents work on existing test files and a new `tests/swmm_validation/`
> subdirectory. Orchestrator owns builds; SWMM runs via `swmm-toolkit` (no subprocess).
> **Model policy:** Per AGENT_SELECTION.md — `cpp-pro` base for `coding`/`test`,
> `python-pro` base for Python harness work, no `flash` (this is mostly Python test
> infrastructure, not refactor).
> **Out of scope:** SWMM source-code patching, full EPA test-suite conformance,
> real-world calibration. The goal is *agreement with SWMM* (we are an SWMM-aligned
> solver), not *deviation from SWMM*.

---

## 1. Selectable step dicts

| # | action (routing keyword in **bold**) | type | phase |
|---|--------------------------------------|------|-------|
| V1 | **Catalog** canonical SWMM 5.x reference scenarios (Example1-Example8 + surcharge + rating curve + tidal); locate .inp files or synthesize equivalents from SWMM's `tests/outfile/` fixtures | docs | 1 |
| V2 | Build **synthetic-equivalent test scenario generator** in `tests/swmm_validation/scenario_factory.py` — converts a SWMM network description to pipe1D equivalent config (shared network builder, single source of truth) | coding | 2 |
| V3 | Implement **comparison harness** in `tests/swmm_validation/compare.py` — runs both solvers side-by-side, extracts WSE/Q per node/link per timestep, computes per-metric error and tolerance pass/fail | coding | 2 |
| V4 | Define **tolerance framework** in `tests/swmm_validation/tolerances.py` — per-regime (open-channel, surcharge, transition) pass criteria: relative WSE error ≤ 5%, relative Q error ≤ 10% for steady-state; tighter for pressurised | docs | 3 |
| V5 | Run **steady-state validation matrix** (no rainfall, fixed inflows): compare Manning's normal depth, surcharge head, outfall backwater, pump rating curves — single value per scenario | test | 4 |
| V6 | Run **dynamic validation matrix** (synthetic storm hyetograph): compare peak WSE, time-to-peak, recession tail across 4-8 scenarios covering open-channel → surcharge → outfall backwater transitions | test | 5 |
| V7 | **Generate validation report** as `tests/swmm_validation/REPORT.md` — table of scenarios × metrics × pass/fail, with debug artefact paths for failures; CI-friendly machine-readable JSON alongside | coding | 6 |
| V8 | **Lock regression baseline** — record current SWMM-vs-pipe1D deltas as `tests/swmm_validation/baseline.json`; add a CI guard test that fails if any metric regresses > 1.5× the baseline | test | 6 |

---

## 2. Pre-computed agent + model columns

| #  | type     | agent                  | model        |
|----|----------|------------------------|--------------|
| V1 | docs     | `python-pro` (base)    | base/default |
| V2 | coding   | `python-pro` (base)    | base/default |
| V3 | coding   | `python-pro` (base)    | base/default |
| V4 | docs     | `python-pro` (base)    | base/default |
| V5 | test     | `python-pro` (base)    | base/default |
| V6 | test     | `python-pro` (base)    | base/default |
| V7 | coding   | `python-pro` (base)    | base/default |
| V8 | test     | `python-pro` (base)    | base/default |

No flash (refactor) tasks — this plan is greenfield Python test infrastructure.

---

## 3. Execution graph

```
[V1: catalog SWMM scenarios]
   └─→ [V2: scenario factory (uses V1's inventory)]
         └─→ [V4: tolerances (parallel with V3)]
         └─→ [V3: comparison harness]
               └─→ [V5: steady-state validation matrix]
                     └─→ [V6: dynamic validation matrix]
                           └─→ [V7: REPORT.md generation]
                                 └─→ [V8: lock baseline + CI guard]
```

V3 and V4 can run in parallel (V3 is the harness code, V4 is the tolerance definitions document).
All other steps are strictly serial — each step's output feeds the next.

---

## 4. Per-phase notes

### V1 — Catalog canonical SWMM scenarios

**Goal:** identify 8-12 reference scenarios that exercise the pipe1D feature space.

**Source:** EPA SWMM 5.x user-guide examples (Example1 through Example8 ship in the
SWMM installer; the canonical GitHub repo `epa-swmm/Stormwater-Management-Model` has them
under `examples/`). The local reference checkout at `reference/Stormwater-Management-Model-develop/`
has the source but not the example `.inp` files.

**Approach:**
1. Download SWMM 5.x example suite from EPA SWMM website or GitHub release
2. Place canonical `.inp` files under `reference/swmm_canonical/ExampleN.inp`
3. For each, write a short description: topology, hydraulics regime, expected behaviour
4. Tag each scenario by which pipe1D spec sections it exercises:
   - Open-channel flow (§2.3)
   - Surcharge / pressurised (§2.5 Preissmann slot)
   - Outfall: free / normal_depth / fixed_wse / rating_curve / tabular (§2.8)
   - Junction surcharge overflow (§2.10)
   - Local losses at faces (§2.12)
   - Subcritical vs supercritical (§2.6 regime override)

**Target matrix (8 scenarios minimum):**

| # | Scenario | Regimes exercised |
|---|----------|---------------------|
| 1 | Single pipe, half-full steady flow | Open-channel Manning's |
| 2 | Single pipe, surcharge from downstream boundary | Pressurised + Preissmann |
| 3 | 2-pipe network, free outfall | Open-channel + free outfall (§2.8) |
| 4 | 2-pipe network, fixed-WSE outfall | Open-channel + fixed-WSE outfall |
| 5 | 2-pipe network, rating-curve outfall | Open-channel + rating outfall |
| 6 | Junction surcharge overflow to 2D | §2.10 weir/orifice |
| 7 | Steep slope → supercritical → checkNormalFlow | §2.6 regime override |
| 8 | Multi-pipe storm network with synthetic hyetograph | Dynamic wave + surcharge |

### V2 — Synthetic-equivalent scenario factory

**Goal:** A single source of truth for network topology that both SWMM and pipe1D
can consume. Eliminates the current pattern of hand-built `make_drainage_inp` for each
test, with separate equivalent pipe1D configs.

**New file:** `tests/swmm_validation/scenario_factory.py`

**API:**
```python
def build_scenario(name: str) -> ScenarioBundle:
    """Returns a named scenario: SWMM .inp file + pipe1D equivalent config."""
    ...

@dataclass
class ScenarioBundle:
    name: str
    swmm_inp_path: Path          # absolute path to .inp
    pipe1d_config: Pipe1DConfig  # matching pipe1D setup
    duration_s: float
    hydrology: HydrologySpec      # rainfall + dry-weather flows
    expected_regimes: list[str]  # ["open_channel", "surcharge", ...]
```

The factory reads the canonical .inp from `reference/swmm_canonical/`, parses the network
using `pyswmm` or simple regex, and constructs a matching `Pipe1DConfig`. This is a
non-trivial parser; budget ~200-300 lines.

### V3 — Comparison harness

**New file:** `tests/swmm_validation/compare.py`

**API:**
```python
def run_comparison(
    scenario: ScenarioBundle,
    tolerances: ToleranceSpec,
    workdir: Path,
) -> ComparisonResult:
    """Run SWMM and pipe1D side-by-side, return per-node/per-link time-series + errors."""

@dataclass
class ComparisonResult:
    scenario: str
    node_timeseries: dict[int, list[float]]   # node_id → WSE over time
    link_timeseries: dict[int, list[float]]   # link_id → Q over time
    node_errors: dict[int, ErrorStats]        # node_id → RMSE, max-abs, max-rel
    link_errors: dict[int, ErrorStats]
    pass_fail: dict[str, bool]                # "open_channel": True, "surcharge": False, ...
```

The harness:
1. Runs SWMM via `swmm-toolkit` (already installed) → reads back `node_results`, `link_results`
2. Runs pipe1D via `tests/pipe1d_runner.py` (existing) → reads back from device
3. Aligns time series (both at same dt; SWMM at variable dt, pipe1D at fixed dt; resample if needed)
4. Computes per-node and per-link error statistics
5. Compares against tolerances per regime (regime determined by SWMM's reported surcharge state)

**Output:** `ComparisonResult` plus JSON dump to `tests/swmm_validation/runs/<scenario>/result.json`

### V4 — Tolerance framework

**New file:** `tests/swmm_validation/tolerances.py` (Python constants + helpers)

| Regime | Metric | Tolerance | Notes |
|--------|--------|-----------|-------|
| Open-channel (steady) | depth RMSE | ≤ 5% of pipe diameter | relative to local yFull |
| Open-channel (steady) | discharge RMSE | ≤ 10% | relative to mean Q |
| Surcharge (pressurised) | WSE RMSE | ≤ 2% of pipe crown height | tight for pressurised regime |
| Surcharge | Q RMSE | ≤ 5% | orifice equation sensitive to slot width |
| Transition (open→surcharge) | Front position timing | ≤ 1 timestep | bore-tracking accuracy |
| Outfall backwater (fixed WSE) | node depth RMSE | ≤ 1% | tailwater should propagate upstream |
| Junction surcharge overflow | 2D inflow mass | ≤ 5% | volume conservation check |
| Regime override (supercritical) | Q match | ≤ 5% | cap should match SWMM's checkNormalFlow |
| Pump | Q match | ≤ 2% | pump curves are deterministic |

Plus a regime detector: given (depth, d_full, d_crown), classify as `open_channel` /
`transition` / `surcharged`.

### V5 — Steady-state validation matrix

**New file:** `tests/test_swmm_validation_steady.py`

Runs V3 on 4-6 scenarios with constant inflows (no rain). Expected runtime ~5 minutes
across all scenarios.

Pass criteria: ALL scenarios pass all open-channel tolerances. Surcharge scenarios
are exempt if SWMM uses a slot and we don't (or vice versa) — flag in report.

### V6 — Dynamic validation matrix

**New file:** `tests/test_swmm_validation_dynamic.py`

Runs V3 on 4-8 scenarios with synthetic hyetographs (5-minute uniform + SCS Type II).
Expected runtime ~15 minutes.

Pass criteria: peak WSE within tolerance, time-to-peak within 1 timestep, recession tail
shape similar (KS test or RMSE). Focus scenarios that exercise surcharge transitions.

### V7 — Validation report generation

**New file:** `tests/swmm_validation/REPORT.md` (generated, not hand-written)

Markdown table:
```
| Scenario | Regime | SWMM depth (m) | pipe1D depth (m) | RMSE | Max-rel | Pass |
|----------|--------|----------------|------------------|------|---------|------|
| Example1 | open    | 0.42           | 0.41             | 0.003| 1.2%    | ✓    |
| Example2 | surcharge | 2.15        | 2.18             | 0.04 | 2.1%    | ✓    |
| ...
```

Plus a JSON sidecar `tests/swmm_validation/REPORT.json` for CI consumption.

### V8 — Regression baseline lock

**New file:** `tests/swmm_validation/baseline.json` — frozen snapshot of current errors.

**New file:** `tests/test_swmm_validation_baseline.py` — fails if any scenario's metrics
exceed `1.5 × baseline` for that scenario.

CI integration: `tests/test_swmm_validation_baseline.py` runs in the existing test
suite; CI fails if regression detected. Operator must explicitly update baseline when
intentional changes ship.

---

## 5. Routing keywords (per task)

- V1: `python`, `test`, `validate`
- V2: `python`, `test`, `validate`
- V3: `python`, `test`, `validate`
- V4: `python`, `test`, `validate`
- V5: `python`, `test`, `validate`
- V6: `python`, `test`, `validate`
- V7: `python`, `test`, `validate`
- V8: `python`, `test`, `validate`

---

## 6. Superpowers workflow

- **TDD discipline:** V2's scenario factory and V3's harness should be developed with
  unit tests in `tests/swmm_validation/test_factory.py` and `tests/swmm_validation/test_harness.py`
  before they are exercised on real scenarios.
- **Spec-compliance review:** V7 (REPORT.md) is reviewed against spec §2.17 ("validation
  is the final acceptance check") by a fresh subagent.
- **Cross-review rule:** V5 and V6 results are reviewed by a different subagent from the
  one that built them.
- **No skip-discipline:** every scenario's pass/fail must be reported. If SWMM cannot
  run on a scenario (e.g., licensing or runtime error), the report must say so explicitly,
  not silently drop the case.

---

## 7. Cross-review rule

Each scenario's pass/fail is determined by the harness (`V3`), not the implementer of
the harness. The implementer reports **what was run**, the reviewer (fresh subagent)
reports **whether the comparison was done correctly**.

---

## 8. Machine-readable JSON block

```json
{
  "spec": "docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md",
  "parent_plan": "docs/archive/plans/2026-07-15-pipe1d-solver-rewrite.md",
  "gaps_plan": "docs/archive/plans/2026-07-16-pipe1d-known-gaps.md",
  "strategy": "serial — catalog → factory → harness → tolerances → matrix → report → baseline",
  "model_policy": "base/default for all tasks (Python test infrastructure)",
  "review_model": "base/default",
  "depends_on": {
    "pyswmm": "2.1.0 (installed in qgis_stable)",
    "swmm-toolkit": "0.17.0 (installed in qgis_stable)",
    "swmm_canonical_examples": "needs download from EPA SWMM 5.x installer or GitHub release"
  },
  "steps": [
    {"id": "V1", "action": "Catalog canonical SWMM 5.x reference scenarios (Example1-8 + surcharge + rating + tidal) under reference/swmm_canonical/ and tag by regimes exercised", "type": "docs", "phase": 1, "agent": "python-pro", "model": "base/default", "depends_on": []},
    {"id": "V2", "action": "Build synthetic-equivalent scenario factory tests/swmm_validation/scenario_factory.py converting SWMM .inp files to pipe1D configs (shared network builder, single source of truth)", "type": "coding", "phase": 2, "agent": "python-pro", "model": "base/default", "depends_on": ["V1"]},
    {"id": "V3", "action": "Implement comparison harness tests/swmm_validation/compare.py running both solvers side-by-side via swmm-toolkit and tests/pipe1d_runner.py; extracts WSE/Q per node/link per timestep; computes per-metric error and tolerance pass/fail", "type": "coding", "phase": 2, "agent": "python-pro", "model": "base/default", "depends_on": ["V1"]},
    {"id": "V4", "action": "Define tolerance framework tests/swmm_validation/tolerances.py with per-regime pass criteria (open-channel, surcharge, transition, fixed WSE outfall, junction overflow, supercritical, pump)", "type": "docs", "phase": 3, "agent": "python-pro", "model": "base/default", "depends_on": []},
    {"id": "V5", "action": "Run steady-state validation matrix tests/test_swmm_validation_steady.py across 4-6 scenarios with constant inflows (no rain)", "type": "test", "phase": 4, "agent": "python-pro", "model": "base/default", "depends_on": ["V2", "V3", "V4"]},
    {"id": "V6", "action": "Run dynamic validation matrix tests/test_swmm_validation_dynamic.py across 4-8 scenarios with synthetic hyetographs (5-minute uniform + SCS Type II)", "type": "test", "phase": 5, "agent": "python-pro", "model": "base/default", "depends_on": ["V2", "V3", "V4"]},
    {"id": "V7", "action": "Generate validation report tests/swmm_validation/REPORT.md (markdown table) and REPORT.json (CI-readable) with per-scenario pass/fail and debug artefact paths", "type": "coding", "phase": 6, "agent": "python-pro", "model": "base/default", "depends_on": ["V5", "V6"]},
    {"id": "V8", "action": "Lock regression baseline tests/swmm_validation/baseline.json and add CI guard test_swmm_validation_baseline.py that fails if any metric regresses > 1.5x the baseline", "type": "test", "phase": 6, "agent": "python-pro", "model": "base/default", "depends_on": ["V7"]}
  ],
  "review_template": "superpowers:requesting-code-review",
  "out_of_scope": [
    "SWMM source-code patching",
    "EPA test-suite full conformance certification",
    "real-world calibration against measured data",
    "modifying the SWMM solver itself"
  ]
}
```

---

## 9. Things to confirm before starting

- [ ] User has the canonical SWMM 5.x example `.inp` files, or confirms V1 should download them
- [ ] User accepts the tolerance framework as a starting point (V4 may need iteration)
- [ ] `pyswmm 2.1.0` and `swmm-toolkit 0.17.0` confirmed installed in `qgis_stable` env
- [ ] `tests/pipe1d_runner.py` and `tests/swmm_runner.py` are stable (no concurrent refactor)
- [ ] `reference/Stormwater-Management-Model-develop/` is available for solver reference (already verified)

---

## 10. Estimated effort

| Step | Subagent invocations | Wall time |
|------|---------------------|-----------|
| V1   | 1                    | 30 min — download + catalog |
| V2   | 1                    | 2-4 hours — .inp parser + factory |
| V3   | 1                    | 2-3 hours — comparison harness |
| V4   | 1                    | 1 hour — tolerance docs |
| V5   | 1                    | 30 min run + analysis |
| V6   | 1                    | 1-2 hours run + analysis |
| V7   | 1                    | 30 min — report generation |
| V8   | 1                    | 30 min — baseline + CI guard |

**Total: ~10-15 hours of work**, mostly Python test infrastructure. Each step is
gated on the previous — no skipping.

---

## 11. Open questions for the user

1. **SWMM canonical examples source**: download from EPA SWMM website, or pull from
   a GitHub release of `epa-swmm/Stormwater-Management-Model`? The reference checkout at
   `reference/Stormwater-Management-Model-develop/` has source code but not the example
   `.inp` files.

2. **Tolerance strictness**: the proposed tolerances (5% depth, 10% Q) are starting
   points. The user may have specific in-house acceptance criteria from prior SWMM
   calibration work. Confirm before V4 starts.

3. **CI integration**: should `test_swmm_validation_baseline.py` run in the standard CI
   suite (slow but automated) or only on a dedicated nightly run (fast CI, slower
   validation)?

4. **Existing test overlap**: there are already `test_pipe1d_vs_swmm.py`,
   `test_drainage_inlet_outfall_vs_swmm.py`, and `test_swe2d_pipe1d_surcharge.py`. The
   V2/V3 harness should supersede these or run alongside them. User's call.