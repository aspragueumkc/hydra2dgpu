---
type: spec
status: complete
created: 2026-07-25
completed: 2026-07-25
---

# Agent Environment Cleanup & Standardization — Design Spec

> **Status:** Active spec, 2026-07-25.
> **Scope:** Audit + cleanup of `.opencode/rules/`, `.agents/skills/`, `.opencode/skills/`, `.agents/*.md`, the 6 base agent files in `.opencode/agents/`, and `opencode.json`. Adds three new repo-specific skills. Deletes one skill (merged into another). Mechanical sync of 48 model-variant agent files. Single implementation plan covers audit, edits, skill creation, and verification.

## 1. Problem

The repo's agent-facing environment has drifted since the docs-lifecycle migration, the CLI-first refactor, and the HYDRA MCP server plans landed. The known issues:

1. **Phantom skill references** — `AGENTS.md` lists `qgis-plugin-conventions` and a `.commandcode/` mirror directory that don't exist on disk.
2. **Wrong skill paths** — `gpu-test-diagnostics` and `mesh-quality-triage` are at `.opencode/skills/*.md`, but the rules reference them inconsistently.
3. **Duplicate AGENTS.md content** — `.opencode/AGENTS.md` (217 lines, auto-injected as system prompt) and `.opencode/rules/AGENTS.md` (238 lines, NOT currently loaded) are near-duplicates with a 21-line delta (one extra Git Safety section in the rules version).
4. **Duplicated 12 software-engineering principles** — the same 12-principles block appears in both `.opencode/rules/AGENTS.md` and `.opencode/rules/PLANNING.md`.
5. **`opencode.json` instructions array is broken** — references three rules that don't exist (`UNIT_SYSTEM.md`, `TEST_PRIORITY.md`, `STUDIO_UI.md`) and omits two that do (`AGENTS.md`, `NO_PREMATURE_BACKWARDS_COMPAT.md`).
6. **No skill for the MCP server, Studio UI parity, or GPKG schemas** — three areas with active specs (`docs/specs/2026-07-24-hydra-mcp-server-design.md`, `docs/specs/2026-07-24-cli-first-refactor-design.md`, `docs/MODEL_GEOPACKAGE_SCHEMA.md`, `docs/RESULTS_GEOPACKAGE_SCHEMA.md`) but no consolidated agent guidance.
7. **The existing `pyqt5-desktop-patterns` skill overlaps the new Studio UI / CLI-GUI parity territory** that the refactor plan now owns.

## 2. Goal

Produce a single, clean, contradiction-free agent environment with three new repo-specific skills covering the MCP server, the Studio UI / CLI-GUI parity boundary, and the model/results GeoPackage schemas. Apply audit-driven edits to the rules, skill tables, and `opencode.json`. Merge the existing PyQt5 skill into the new Studio UI skill.

**Non-goals:**
- No audit or rewrite of the 48 model-variant agent files beyond a mechanical sync (mirroring the corresponding base agent body, preserving only the variant's `model:` line).
- No reformat of the three `.opencode/skills/*.md` single-file skills into folder form — they're small reference sheets, no scripts to attach.
- No rewriting of the repo-root `AGENTS.md` (the 122-line project-overview file). It's the user-facing AGENTS.md per opencode convention; it documents the project correctly.
- No new skills beyond the three named.
- No plugin (plan_dispatcher.ts, superpowers, ponytail) edits — out of scope.

## 3. Target Architecture

```
project root
├── AGENTS.md                              (122 lines, opencode-convention auto-load, project overview)
├── .opencode/
│   ├── opencode.json                      (instructions array: add AGENTS.md + NO_PREMATURE_BACKWARDS_COMPAT.md,
│   │                                       drop phantom UNIT_SYSTEM.md / TEST_PRIORITY.md / STUDIO_UI.md)
│   ├── agents/
│   │   ├── {base agents}.md               (×6: python-pro, cpp-pro, debugger, performance-engineer,
│   │   │                                   build-engineer, test-automator — audit + rewire)
│   │   └── {base}_*.md                    (×48 model variants — mechanical mirror of base body,
│   │                                       preserving the variant's `model:` line)
│   ├── plugins/                           (untouched)
│   ├── skills/                            (untouched; path references in AGENTS.md updated)
│   └── rules/
│       ├── AGENTS.md                      (canonical agent-facing rules; loaded via instructions)
│       ├── AGENT_SELECTION.md
│       ├── CACHE_DISCIPLINE.md
│       ├── ENVIRONMENT.md
│       ├── GIT_SAFETY.md
│       ├── MVP_ARCHITECTURE.md            (verify studio_dialog.py path; cross-link new studio-ui skill)
│       ├── NO_PREMATURE_BACKWARDS_COMPAT.md
│       ├── PLANNING.md                    (dedupe 12-principles block)
│       └── SESSION_DOCUMENTATION.md
└── .agents/
    ├── skills/
    │   ├── frontend-design/               (unchanged)
    │   ├── fvm-cfd-solver-patterns/       (unchanged)
    │   ├── pyqt5-desktop-patterns/        (DELETED — content merged into studio-ui)
    │   ├── skills-discovery/              (unchanged)
    │   ├── subagent-driven-development/   (unchanged)
    │   ├── hydra-mcp-server/              (NEW)
    │   ├── hydra2dgpu-studio-ui/          (NEW — absorbs pyqt5-desktop-patterns)
    │   └── hydra2dgpu-gpkg-schema-expert/ (NEW)
    ├── codebase-audit.md                  (unchanged reference note)
    ├── computation-source-truth.md        (unchanged reference note)
    └── studio-gui-api.md                  (unchanged reference note)
```

## 4. New Skills — Specs

Each skill follows the same skeleton pattern:
- One-line trigger paragraph at the top
- "When to load" reminder
- Skim/summary of the source-of-truth doc
- Critical rules inline (7–10 max)
- Cross-link to related skills
- Source-of-truth doc reference at the top

### 4.1 — `hydra-mcp-server`

**Trigger:** Use when launching a live QGIS session, driving the Studio GUI from an agent, running a parameter sweep, verifying the GUI, or applying a widget design patch.

**Source of truth:** `docs/specs/2026-07-24-hydra-mcp-server-design.md`

**Skeleton:**
- Tier map (A: modeling/results; B: live GUI; C: design)
- Seven critical rules:
  1. Workspace-relative paths only (reject `..`, symlinks, escape)
  2. Fail-fast errors — no catch+swallow
  3. GUI thread safety — dispatch long work to `SimulationWorker`, return via Qt signals
  4. `design_apply_patch` is gated — always preview → ask user → apply
  5. Bridge auth — per-session random token in a 0600 file, same-machine only (`QLocalSocket`)
  6. Frame-length cap — reject oversized frames before decode; do not raise the cap
  7. Subprocess lifecycle — QGIS/Xvfb tracked and reaped by `gui_close`
- Session lifecycle (offscreen / xvfb / display)
- Common mistakes to avoid (5 items)
- Related skills cross-link: studio-ui, gpkg-schema-expert

### 4.2 — `hydra2dgpu-studio-ui` (merged from `pyqt5-desktop-patterns`)

**Trigger:** Use for any Studio UI work — adding tabs, wiring widgets, dock lifecycle, signal safety, structural changes, or anything touching the Studio CLI/GUI parity boundary.

**Source of truth:** `docs/specs/2026-07-24-cli-first-refactor-design.md` + absorbed content from `pyqt5-desktop-patterns/SKILL.md`.

**Skeleton:**
- The two-layer mental model: core (GUI-free, may import `qgis.core`) vs view (Qt widgets)
- Seven CLI/GUI parity rules:
  1. One canonical builder — `swe2d/core/builder.py::build_run_context`
  2. One defaults table — `_DEFAULTS`, every constructor reaches it
  3. Fail-fast validation at every level (top + nested unknown keys raise)
  4. No silent absorbs — no `try/except Exception → warning + None`
  5. CLI claims are precise — "requires qgis.core; no QGIS GUI, iface, or display needed"
  6. No view-bound callables ride RunContext — Qt signals only
  7. No re-export shims, no deprecation period
- Six widget-lifecycle rules (absorbed from pyqt5-desktop-patterns):
  1. Liveness guard via `try: _ = widget.objectName() except RuntimeError`
  2. Wrap `isinstance` in try/except (deleted C++ QObject)
  3. `safe_disconnect` (`swe2d_workbench_qt`) for signal disconnect
  4. Delete empty parent shells after extracting children
  5. `QTimer.stop()` + `deleteLater()` before destroying parent
  6. `.ui` files are source of truth; run `tools/ui_bind_sync.py` after every edit
- StudioComponent registry (absorbed)
- Feature flags (absorbed)
- State persistence (absorbed)
- Structural changes checklist (6 steps)
- Key file locations table (11 entries)
- Common mistakes to avoid (5 items)

### 4.3 — `hydra2dgpu-gpkg-schema-expert`

**Trigger:** Use when reading or writing any model GeoPackage or results GeoPackage — listing meshes, inspecting layers, configuring BCs/rainfall/drainage/structures, querying run results, comparing two runs.

**Source of truth:** `docs/MODEL_GEOPACKAGE_SCHEMA.md` + `docs/RESULTS_GEOPACKAGE_SCHEMA.md`.

**Skeleton:**
- Two files, two roles (model GPKG = user-authored input; results GPKG = solver output)
- 18-table skim for model GPKG (one row per table with geometry + purpose)
- 5-table skim for results GPKG (one row per table with row granularity + BLOB storage)
- Seven critical rules:
  1. CRS is the project CRS, not always 4326
  2. Units are model units, not SI
  3. BLOB layout is fixed and versioned
  4. Dry cells are omitted in mesh results
  5. No FK enforcement — application code maintains referential integrity
  6. Layers are created empty — populate via QGIS digitizing or `*_configure` MCP tools
  7. `swe2d_*` table prefix is reserved
- Common queries (5–8 snippets) — listed in implementation plan
- MCP tools cross-link (Tier A + Tier B that touch the GPKG)
- Common mistakes to avoid (5 items)

## 5. Cleanup Edits

| # | File | Edit | Reason |
|---|---|---|---|
| E1' | `.opencode/opencode.json` | Add `.opencode/rules/AGENTS.md` and `.opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md` to the `instructions` array. Remove `.opencode/rules/UNIT_SYSTEM.md`, `.opencode/rules/TEST_PRIORITY.md`, `.opencode/rules/STUDIO_UI.md` (don't exist). | Phantom rules + missing rules |
| E2 | `.opencode/AGENTS.md` | Delete (now an orphan — canonical content lives at `.opencode/rules/AGENTS.md` and is loaded via `instructions`) | Unification per E1' |
| E3 | `.opencode/rules/AGENTS.md` | Update skill table: drop `qgis-plugin-conventions` row, point `gpu-test-diagnostics` / `mesh-quality-triage` at `.opencode/skills/*.md` (current location), add the three new skills with their trigger summaries, remove `.commandcode/` mirror paragraph (directory doesn't exist). Keep the 12-principles block. | Audit findings |
| E4 | `.opencode/rules/PLANNING.md` | Delete the 12-principles block (now lives only in AGENTS.md). | Deduplication |
| E5 | `.opencode/rules/MVP_ARCHITECTURE.md` | Verify `studio_dialog.py` is still the right path (audit step). If renamed per CLI-first refactor, update references. Add a one-line cross-link to `hydra2dgpu-studio-ui` skill for "add a tab / wire a widget / structural changes" cases. | Verification + routing |
| E6 | `.opencode/agents/{python-pro,cpp-pro,debugger,performance-engineer,build-engineer,test-automator}.md` (6 base agents) | For each: (a) add the three new skill names to "Available Skills" / "Skills to consult" section if present, (b) verify tool list still matches `opencode.json`, (c) update any stale skill-table entries that reference `qgis-plugin-conventions` or `.commandcode/` | Audit findings |
| E7 | `.opencode/agents/{python-pro,cpp-pro,debugger,performance-engineer,build-engineer,test-automator}_*.md` (48 model variants) | **Mechanical sync**: mirror the body of the corresponding base agent into each variant, preserving only the variant's `model:` line. No content reading needed (per user direction). | Trivial delta |
| E8 | `.agents/skills/pyqt5-desktop-patterns/` | Delete folder after `hydra2dgpu-studio-ui/SKILL.md` is written and verified. | Merge |
| E9 | `.agents/skills/hydra-mcp-server/SKILL.md` | Create per §4.1. | New skill |
| E10 | `.agents/skills/hydra2dgpu-studio-ui/SKILL.md` | Create per §4.2, absorbing pyqt5-desktop-patterns content. | New skill (merged) |
| E11 | `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md` | Create per §4.3. | New skill |
| E12 | `.opencode/rules/GIT_SAFETY.md` | No content changes; verify only. | Audit sanity |
| E13 | `.opencode/rules/CACHE_DISCIPLINE.md` | No content changes; verify only. | Audit sanity |
| E14 | `.opencode/rules/ENVIRONMENT.md` | Verify mamba env name + build dir paths still correct. | Audit sanity |
| E15 | `.opencode/rules/SESSION_DOCUMENTATION.md` | Verify `docs/AGENT_SESSION_RECOVERY_LOG.md` exists and is the rolling log. | Audit sanity |
| E16 | `.opencode/rules/AGENT_SELECTION.md` | No content changes; verify only. | Audit sanity |
| E17 | `.opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md` | No content changes; verify only. (Now loaded via E1'.) | Audit sanity |

## 6. Verification Gate

```bash
# 1. Audit confirms the cleanup
! test -f .opencode/AGENTS.md || echo "FAIL: E2 didn't delete .opencode/AGENTS.md"
! grep -q "qgis-plugin-conventions" .opencode/rules/AGENTS.md || echo "FAIL: E3 phantom skill row"
! grep -q "\.commandcode/" .opencode/rules/AGENTS.md || echo "FAIL: E3 ghost references"
! grep -q "twelve fundamental software engineer principles" .opencode/rules/PLANNING.md \
  || echo "FAIL: E4 dedup didn't remove principles block"
! test -d .agents/skills/pyqt5-desktop-patterns && echo "FAIL: E8 didn't delete old skill"
test -f .agents/skills/hydra-mcp-server/SKILL.md || echo "FAIL: E9 new skill missing"
test -f .agents/skills/hydra2dgpu-studio-ui/SKILL.md || echo "FAIL: E10 new skill missing"
test -f .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md || echo "FAIL: E11 new skill missing"

# 2. opencode.json instructions are sane
python3 -c "
import json, os
cfg = json.load(open('.opencode/opencode.json'))
for p in cfg.get('instructions', []):
    if not os.path.exists(p):
        print(f'FAIL: phantom instruction {p}')
        exit(1)
if not any('AGENTS.md' in p for p in cfg.get('instructions', [])):
    print('FAIL: AGENTS.md not in instructions')
    exit(1)
print('PASS: opencode.json instructions array sane')
"

# 3. Always-on checks (per .opencode/rules/PLANNING.md)
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v \
  tests.test_workbench_gui \
  tests.test_workbench_persistence \
  tests.test_workbench_controller

# 4. Each new SKILL.md is loadable (non-empty, parseable)
for f in .agents/skills/{hydra-mcp-server,hydra2dgpu-studio-ui,hydra2dgpu-gpkg-schema-expert}/SKILL.md; do
  test -s "$f" || { echo "FAIL: $f empty"; exit 1; }
done

# 5. Variant agents mirror base agent body
diff <(cat .opencode/agents/python-pro.md) \
     <(cat .opencode/agents/python-pro_kimi-for-coding_k3.md | sed '/^model:/d') \
     | grep -q '^>' && echo "WARN: variant differs from base beyond model: line"

# 6. Per-skill smoke check — each new skill's canonical name is present
for skill_name in "hydra-mcp-server" "hydra2dgpu-studio-ui" "hydra2dgpu-gpkg-schema-expert"; do
  test -f ".agents/skills/$skill_name/SKILL.md" || { echo "FAIL: $skill_name not in any skill"; exit 1; }
done
echo "PASS block 6"
```

## 7. Out of Scope (defer)

- Audit of the 48 model-variant agents beyond the mechanical sync.
- Reformatting the three `.opencode/skills/*.md` files into `.agents/skills/<name>/SKILL.md` form.
- Rewriting the repo-root `AGENTS.md` from scratch as a thin manifest.
- Adding new skills beyond the three named.
- Plugin edits (`plan_dispatcher.ts`, superpowers, ponytail).
- The `studio_dialog.py` filename confirmation (E5) — verify in audit, don't pre-judge.
- The CLI-first refactor itself — this spec only consumes its design output; the refactor work lives in `docs/plans/2026-07-24-cli-first-refactor.md`.

## 8. Risks

- **Edit to `opencode.json` may break session config.** Mitigation: edit is small (add 2, remove 3 entries); verification gate runs the always-on tests; if a session breaks, revert `opencode.json` from git.
- **Skill merge loses pyqt5-desktop-patterns content.** Mitigation: the absorbed content is enumerated in §4.2 with seven rule line items + absorbed sections (registry, feature flags, persistence); the implementation step will diff-check against the original.
- **Mechanical sync of 48 model-variant agents fails silently.** Mitigation: variant count is verified before/after; the diff check (verification gate #5) flags any body that drifts beyond the `model:` line.

## 9. Parallel Subagent-Driven Execution Plan

The 17 edits decompose into 17 subagent tasks. Independence analysis groups them into 3 parallel batches and 3 sequential phases. All tasks follow the superpowers `subagent-driven-development` workflow: fresh subagent per task, two-stage review (spec compliance → code quality) per task. Parallel dispatch uses the `dispatching-parallel-agents` pattern only when tasks touch disjoint files with no shared state.

### 9.1 — Independence analysis

| Edit | File(s) touched | Depends on | Parallel-safe with |
|---|---|---|---|
| E1' | `.opencode/opencode.json` | — | every other task |
| E2 | (delete `.opencode/AGENTS.md`) | E1' | E8 |
| E3 | `.opencode/rules/AGENTS.md` | — | every other task |
| E4 | `.opencode/rules/PLANNING.md` | — | every other task |
| E5 | `.opencode/rules/MVP_ARCHITECTURE.md` | E10 (cross-link) | tasks not editing MVP_ARCHITECTURE |
| E6 (×6) | 6 base agent files | — | all other E6 tasks |
| E7 (×48) | 48 model-variant files | E6 (base bodies must be final first) | — |
| E8 | (delete `.agents/skills/pyqt5-desktop-patterns/`) | E10 | E2 |
| E9 | `.agents/skills/hydra-mcp-server/SKILL.md` (create) | — | E10, E11 |
| E10 | `.agents/skills/hydra2dgpu-studio-ui/SKILL.md` (create) | — | E9, E11 |
| E11 | `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md` (create) | — | E9, E10 |

**No two tasks edit the same file.** All Phase 1 tasks create or edit disjoint files. Phase 3 tasks each edit a different base agent file. The Phase 2 deletes target different paths.

### 9.2 — Phase breakdown

```
Phase 0  ─ Audit                       ─ 1 task   (sequential prerequisite)
Phase 1  ─ Parallel batch A            ─ 7 tasks  (parallel: skill writes + rule/config edits)
Phase 2  ─ Parallel batch B            ─ 2 tasks  (parallel: orphan deletes)
Phase 3  ─ Parallel batch C            ─ 6 tasks  (parallel: 6 base agent rewires)
Phase 4  ─ Mechanical sync             ─ 1 task   (sequential: 48 model variants mirror bases)
Phase 5  ─ Verification gate           ─ 1 task   (sequential: runs spec §6 shell blocks)
```

Wall-clock vs. sequential estimate: ~6 phases × max(per-phase time) vs. 17 tasks in series. Roughly 3× speedup with the parallel batches; saves the bulk of the time on Phase 1 (3 skill writes that are the longest individual tasks).

### 9.3 — Phase 0: Audit (1 task, sequential)

| Task | Description | Agent | Model |
|---|---|---|---|
| 0.1 | Read all 9 rule files, 3 `.opencode/skills/*.md`, 5 `.agents/skills/*/SKILL.md`, 3 `.agents/*.md` reference notes, the 6 base agents, `opencode.json`, and the two referenced schemas (`docs/MODEL_GEOPACKAGE_SCHEMA.md`, `docs/RESULTS_GEOPACKAGE_SCHEMA.md`). Verify each of E1'–E17 still matches the file's current content. Flag any drift from this spec's assumptions (especially E5 `studio_dialog.py` rename check, E12–E17 verify-only items). Produce a 1-page "audit findings" report. | `general` (explore) | standard |

If 0.1 finds material drift (a rule file has been substantively rewritten, a skill has been added since this spec was drafted, etc.), the spec needs revision **before** Phase 1 starts. If 0.1 finds only minor drift (typo fixes, single-line clarifications), Phase 1 tasks incorporate those without re-spec.

### 9.4 — Phase 1: Parallel batch A (7 tasks, parallel)

All seven tasks touch disjoint files. Dispatch using `dispatching-parallel-agents` after independence is verified (per §9.1 table).

| Task | Description | Agent | Model | Source |
|---|---|---|---|---|
| 1.1 | Write `.agents/skills/hydra-mcp-server/SKILL.md` (E9) | `python-pro` | standard | spec §4.1 + `docs/specs/2026-07-24-hydra-mcp-server-design.md` |
| 1.2 | Write `.agents/skills/hydra2dgpu-studio-ui/SKILL.md` (E10), absorbing `.agents/skills/pyqt5-desktop-patterns/SKILL.md` | `python-pro` | standard | spec §4.2 + both source docs |
| 1.3 | Write `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md` (E11) | `python-pro` | standard | spec §4.3 + both schema docs |
| 1.4 | Edit `.opencode/opencode.json` instructions array (E1') | `build-engineer` | cheap (mechanical) | spec §5 E1' |
| 1.5 | Edit `.opencode/rules/AGENTS.md` skill table (E3) | `python-pro` | cheap (mechanical) | spec §5 E3 |
| 1.6 | Edit `.opencode/rules/PLANNING.md` (E4 — delete 12-principles block) | `python-pro` | cheap (mechanical) | spec §5 E4 |
| 1.7 | Edit `.opencode/rules/MVP_ARCHITECTURE.md` (E5 — verify `studio_dialog.py` path, add cross-link to new studio-ui skill) | `python-pro` | cheap (mechanical) | spec §5 E5 |

**Per-task prompt** uses the superpowers `implementer-prompt.md` template with full task text + context. **Per-task review** uses the `spec-reviewer-prompt.md` and `code-quality-reviewer-prompt.md` templates. Each task produces a commit; subagent reports DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT.

For task 1.2 specifically, the implementer prompt includes: "After writing the new SKILL.md, `diff` it against the seven widget-lifecycle rules + the StudioComponent registry / feature flags / state persistence sections of the old pyqt5-desktop-patterns/SKILL.md. Confirm every absorbed item is present in the new file before reporting DONE."

### 9.5 — Phase 2: Parallel batch B (2 tasks, parallel)

Both tasks are deletes on disjoint paths with different Phase 1 prerequisites. They are independent of each other.

| Task | Description | Agent | Model | Depends on |
|---|---|---|---|---|
| 2.1 | Delete `.opencode/AGENTS.md` (E2) | `build-engineer` | cheap | 1.4 done |
| 2.2 | Delete `.agents/skills/pyqt5-desktop-patterns/` (E8) | `build-engineer` | cheap | 1.2 done |

Review per task: spec compliance only (no code quality on a deletion). The spec reviewer confirms the path no longer exists and that no other file references it.

### 9.6 — Phase 3: Parallel batch C (6 tasks, parallel)

Each task edits a disjoint base agent file. Dispatch using `dispatching-parallel-agents` after independence is verified.

| Task | File | Agent | Model |
|---|---|---|---|
| 3.1 | `.opencode/agents/python-pro.md` | `python-pro` | cheap |
| 3.2 | `.opencode/agents/cpp-pro.md` | `python-pro` | cheap |
| 3.3 | `.opencode/agents/debugger.md` | `python-pro` | cheap |
| 3.4 | `.opencode/agents/performance-engineer.md` | `python-pro` | cheap |
| 3.5 | `.opencode/agents/build-engineer.md` | `build-engineer` | cheap |
| 3.6 | `.opencode/agents/test-automator.md` | `test-automator` | cheap |

**Per-task work** (E6): (a) add the three new skill names to "Available Skills" / "Skills to consult" section if the file has one, (b) verify tool list still matches `opencode.json`, (c) update any stale entries referencing `qgis-plugin-conventions` or `.commandcode/`.

Review per task: spec compliance (did the agent file reference the three new skills and drop the stale ones) + code quality (no accidental body changes).

### 9.7 — Phase 4: Mechanical sync (1 task, sequential)

| Task | Description | Agent | Model | Depends on |
|---|---|---|---|---|
| 4.1 | Mirror the body of each base agent into its 8 model variants, preserving only the variant's `model:` line (E7). 48 files in total (6 bases × 8 variants each, varying). | `build-engineer` | cheap | Phase 3 done |

Implementer prompt explicitly notes: "Read each base agent once. For each of its model variants, copy the base body verbatim, then overwrite the `model:` line with the variant's existing `model:` line. Do NOT read the variant's existing body — it is the source of truth only for the model line. Use `diff` after to confirm the only difference between each variant and its base is the `model:` line."

Review per task: spec compliance (variant count matches, no body drift beyond `model:` line) + code quality (sparse — this is a pure mechanical mirror).

### 9.8 — Phase 5: Verification gate (1 task, sequential)

| Task | Description | Agent | Model | Depends on |
|---|---|---|---|---|
| 5.1 | Run all 6 shell blocks from spec §6 in sequence. Report pass/fail for each. On failure, identify which Phase 1–4 task introduced the regression. | `build-engineer` | cheap | all prior phases |

The `__pycache__` purge + workbench test run is GPU-free but slow (~minutes). The verification task commits any pre-existing `__pycache__` purge artifacts and reports the test summary.

### 9.9 — Cross-review rule

Per `.opencode/rules/PLANNING.md`: "Every code change produced by one subagent must be reviewed by a different subagent before the phase is marked complete." Implementation:

- For Phase 1 and Phase 3 parallel batches: the spec reviewer and code-quality reviewer are **different subagents** from the implementer (already required by the two-stage review pattern).
- For Phase 2 deletes: a single spec-compliance reviewer (one review) covers the deletion; the same subagent dispatching discipline applies.
- For Phase 4 mechanical sync: spec reviewer + code-quality reviewer are two distinct subagent dispatches.
- Cross-review between batches: Phase 5's verification subagent reads Phase 1–4 commit logs and flags any commit authored without a corresponding review commit (the superpowers review pattern produces paired commits).

### 9.10 — Failure handling

| Failure | Action |
|---|---|
| Task 0.1 finds material drift from this spec | Halt. Update spec. Restart from Phase 0. |
| One Phase 1 task reports BLOCKED | Re-dispatch with more capable model (per `subagent-driven-development` skill). Do not block other Phase 1 tasks. |
| Phase 1 spec review fails (item missing from skill) | Implementer (same subagent) fixes; review again. Do not proceed to Phase 2 for that dependency edge. |
| Phase 4 mechanical sync miscounts variants | Abort before committing; the implementer reports BLOCKED with the count discrepancy. |
| Phase 5 verification gate fails | Re-run individual shell blocks to localize; assign a fix subagent to the offending phase. Do not auto-revert. |

### 9.11 — Wall-clock estimate

| Phase | Tasks | Parallelism | Est. wall-clock |
|---|---|---|---|
| 0 | 1 | 1× | ~5 min |
| 1 | 7 | 7× (longest: skill write ~10 min) | ~10 min |
| 2 | 2 | 2× | ~2 min |
| 3 | 6 | 6× | ~5 min |
| 4 | 1 | 1× | ~5 min |
| 5 | 1 | 1× | ~5 min |
| Reviews | per task (spec + quality) | mostly sequential per task | embedded in phase time |
| **Total** | 18 tasks | — | **~32 min** |

Sequential equivalent: ~17 task-times × ~6 min avg ≈ **~100 min**. Roughly 3× speedup from the three parallel batches.

## 10. Related Documentation

- `docs/specs/2026-07-24-hydra-mcp-server-design.md` — source of truth for §4.1
- `docs/specs/2026-07-24-cli-first-refactor-design.md` — source of truth for §4.2
- `docs/MODEL_GEOPACKAGE_SCHEMA.md` — source of truth for §4.3
- `docs/RESULTS_GEOPACKAGE_SCHEMA.md` — source of truth for §4.3
- `docs/specs/2026-07-25-docs-lifecycle-design.md` — docs folder conventions used throughout
- `.agents/skills/pyqt5-desktop-patterns/SKILL.md` — content to absorb and delete (per §4.2)
- `.opencode/opencode.json` — modified per E1'
- `.opencode/rules/AGENTS.md` — modified per E3
- `.opencode/rules/PLANNING.md` — modified per E4
- `.opencode/rules/MVP_ARCHITECTURE.md` — modified per E5
- `.agents/skills/subagent-driven-development/SKILL.md` — execution framework (project-local)
- `~/.cache/opencode/.../skills/subagent-driven-development/{implementer,spec-reviewer,code-quality-reviewer}-prompt.md` — prompt templates used by the orchestrator
- `~/.cache/opencode/.../skills/dispatching-parallel-agents/SKILL.md` — parallel dispatch technique