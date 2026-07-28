---
type: plan
status: complete
created: 2026-07-25
completed: 2026-07-25
progress:
  total: 18
  done: 17
  current: null
  blockers: ["Block 3: test_returns_true_on_successful_load pre-existing failure (workbench_controller, fails on public-sanitize merge base too — not a regression of this plan; investigate separately)"]
  last_updated: 2026-07-25
---

# Agent Environment Cleanup & Standardization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and clean up `.opencode/rules/`, `.agents/skills/`, `.opencode/skills/`, `.opencode/agents/`, and `opencode.json`. Add three new repo-specific skills (`hydra-mcp-server`, `hydra2dgpu-studio-ui`, `hydra2dgpu-gpkg-schema-expert`). Merge `pyqt5-desktop-patterns` into the new Studio UI skill. Mechanically sync 48 model-variant agent files. Land in one plan with parallel subagent dispatch.

**Architecture:** Three parallel batches (skill writes + rule edits, then orphan deletes, then base-agent rewires) sandwiched between a sequential audit (Phase 0), a sequential mechanical sync (Phase 4), and a sequential verification gate (Phase 5). Each task runs as a fresh subagent with two-stage review (spec compliance → code quality) per the `subagent-driven-development` skill. Independence verified per spec §9.1: no two tasks edit the same file.

**Tech Stack:** PyQt5 (Studio UI), opencode (agent config), QGIS plugin Python packaging, superpowers skills (brainstorming → writing-plans → subagent-driven-development → dispatching-parallel-agents → requesting-code-review).

**Spec:** `docs/specs/2026-07-25-agent-env-cleanup-design.md` (400 lines, 10 sections). All edits E1'–E17 are defined there with full context. This plan operationalizes them.

---

## Selector-Consumable Step Table

The plan_dispatcher.ts plugin reads this section to assign each task to an agent and model. Each row is one of the 18 subagent dispatches.

| Phase | Task ID | Action | Type | Agent | Model | Routing keywords | Depends on |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | audit the agent env state to confirm spec assumptions | docs | general | standard | read, audit, refactor | — |
| 1 | 1.1 | refactor the agent env to write hydra-mcp-server skill SKILL.md | docs | python-pro | standard | python, refactor, write | 0.1 |
| 1 | 1.2 | refactor the agent env to write hydra2dgpu-studio-ui skill SKILL.md (absorbing pyqt5) | docs | python-pro | standard | python, pyqt5, refactor | 0.1 |
| 1 | 1.3 | refactor the agent env to write hydra2dgpu-gpkg-schema-expert skill SKILL.md | docs | python-pro | standard | python, refactor | 0.1 |
| 1 | 1.4 | refactor the agent env to edit opencode.json instructions array | refactor | build-engineer | cheap | refactor, edit, json | 0.1 |
| 1 | 1.5 | refactor the agent env to edit rules/AGENTS.md skill table | refactor | python-pro | cheap | python, refactor, edit | 0.1 |
| 1 | 1.6 | refactor the agent env to delete the duplicated 12-principles block in rules/PLANNING.md | refactor | python-pro | cheap | python, refactor, edit | 0.1 |
| 1 | 1.7 | refactor the agent env to edit rules/MVP_ARCHITECTURE.md (verify studio_dialog.py path + cross-link) | refactor | python-pro | cheap | python, refactor, edit | 0.1, 1.2 (cross-link target) |
| 2 | 2.1 | refactor the agent env to delete orphan .opencode/AGENTS.md | refactor | build-engineer | cheap | refactor, delete | 1.4 |
| 2 | 2.2 | refactor the agent env to delete old pyqt5-desktop-patterns skill folder | refactor | build-engineer | cheap | refactor, delete | 1.2 |
| 3 | 3.1 | refactor the agent env to update base agent python-pro.md skill references | refactor | python-pro | cheap | python, refactor, edit | 1.1, 1.2, 1.3, 2.2 |
| 3 | 3.2 | refactor the agent env to update base agent cpp-pro.md skill references | refactor | python-pro | cheap | python, refactor, edit | 1.1, 1.2, 1.3 |
| 3 | 3.3 | refactor the agent env to update base agent debugger.md skill references | refactor | python-pro | cheap | python, refactor, edit | 1.1, 1.2, 1.3 |
| 3 | 3.4 | refactor the agent env to update base agent performance-engineer.md skill references | refactor | python-pro | cheap | python, refactor, edit | 1.1, 1.2, 1.3 |
| 3 | 3.5 | refactor the agent env to update base agent build-engineer.md skill references | refactor | build-engineer | cheap | refactor, edit | 1.1, 1.2, 1.3 |
| 3 | 3.6 | refactor the agent env to update base agent test-automator.md skill references | refactor | test-automator | cheap | test, refactor, edit | 1.1, 1.2, 1.3 |
| 4 | 4.1 | refactor the agent env to mechanically sync 48 model-variant agent files from base bodies | refactor | build-engineer | cheap | refactor, sync, mirror | 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 |
| 5 | 5.1 | test the agent env cleanup with the spec §6 verification gate | test | build-engineer | cheap | test, validate, verify | 4.1 |

Total tasks: **18**. Parallel batches: Phase 1 (7 tasks), Phase 2 (2 tasks), Phase 3 (6 tasks). Sequential phases: Phase 0, Phase 4, Phase 5.

---

## Machine-Readable JSON Block (§8 pattern)

This block is consumed by `.opencode/plugins/plan_dispatcher.ts` to compute agent+model filenames at dispatch time.

```json
{
  "spec": "docs/specs/2026-07-25-agent-env-cleanup-design.md",
  "dispatch_table": {
    "0.1": {
      "model": "default",
      "skills": ["skills-discovery"]
    },
    "1.1": {
      "model": "standard",
      "skills": ["hydra-mcp-server"]
    },
    "1.2": {
      "model": "standard",
      "skills": ["hydra2dgpu-studio-ui", "pyqt5-desktop-patterns"]
    },
    "1.3": {
      "model": "standard",
      "skills": ["hydra2dgpu-gpkg-schema-expert"]
    },
    "1.4": {
      "model": "default",
      "skills": []
    },
    "1.5": {
      "model": "default",
      "skills": []
    },
    "1.6": {
      "model": "default",
      "skills": []
    },
    "1.7": {
      "model": "default",
      "skills": ["hydra2dgpu-studio-ui"]
    },
    "2.1": {
      "model": "default",
      "skills": []
    },
    "2.2": {
      "model": "default",
      "skills": []
    },
    "3.1": {
      "model": "default",
      "skills": ["hydra-mcp-server", "hydra2dgpu-studio-ui", "hydra2dgpu-gpkg-schema-expert"]
    },
    "3.2": {
      "model": "default",
      "skills": ["hydra-mcp-server", "hydra2dgpu-studio-ui", "hydra2dgpu-gpkg-schema-expert"]
    },
    "3.3": {
      "model": "default",
      "skills": ["hydra-mcp-server", "hydra2dgpu-studio-ui", "hydra2dgpu-gpkg-schema-expert"]
    },
    "3.4": {
      "model": "default",
      "skills": ["hydra-mcp-server", "hydra2dgpu-studio-ui", "hydra2dgpu-gpkg-schema-expert"]
    },
    "3.5": {
      "model": "default",
      "skills": ["hydra-mcp-server", "hydra2dgpu-studio-ui", "hydra2dgpu-gpkg-schema-expert"]
    },
    "3.6": {
      "model": "default",
      "skills": ["hydra-mcp-server", "hydra2dgpu-studio-ui", "hydra2dgpu-gpkg-schema-expert"]
    },
    "4.1": {
      "model": "default",
      "skills": []
    },
    "5.1": {
      "model": "default",
      "skills": []
    }
  },
  "phases": [
    {"id": "0", "name": "Audit", "tasks": ["0.1"], "parallel": false},
    {"id": "1", "name": "Parallel batch A — skill writes + rule edits", "tasks": ["1.1","1.2","1.3","1.4","1.5","1.6","1.7"], "parallel": true},
    {"id": "2", "name": "Parallel batch B — orphan deletes", "tasks": ["2.1","2.2"], "parallel": true},
    {"id": "3", "name": "Parallel batch C — base agent rewires", "tasks": ["3.1","3.2","3.3","3.4","3.5","3.6"], "parallel": true},
    {"id": "4", "name": "Mechanical sync — model variants", "tasks": ["4.1"], "parallel": false},
    {"id": "5", "name": "Verification gate", "tasks": ["5.1"], "parallel": false}
  ]
}
```

---

## Superpowers Workflow

Each task follows the per-task loop defined by `superpowers:subagent-driven-development`:

1. **Dispatch implementer subagent** using the prompt template at `~/.cache/opencode/packages/superpowers@git+https://github.com/obra/superpowers.git/node_modules/superpowers/skills/subagent-driven-development/implementer-prompt.md`. Each task's prompt inlines the full task text + context (no `cat`-of-plan).
2. **Implementer asks questions** if any; orchestrator answers; subagent implements, runs verification, commits, self-reviews.
3. **Dispatch spec-compliance reviewer** using `spec-reviewer-prompt.md` from the same path. Verifier reads the actual code (not the implementer's report) against the spec.
4. **If spec review fails** → implementer (same subagent type) fixes → spec review re-runs.
5. **Dispatch code-quality reviewer** using `code-quality-reviewer-prompt.md`. Only after spec compliance is ✅.
6. **If quality review fails** → implementer fixes → quality review re-runs.
7. **Mark task complete** in this plan's progress block: increment `progress.done`, set `progress.current` to the next pending task, set `progress.last_updated`.

For parallel batches (Phases 1, 2, 3): use `superpowers:dispatching-parallel-agents`. Independence is pre-verified in spec §9.1 — all parallel tasks touch disjoint files. Do not re-dispatch as parallel if a task reports BLOCKED (sequential fallback per the skill).

Cross-phase blocking rules (per spec §9.10):
- Phase 1 task 1.7 (MVP_ARCHITECTURE cross-link) blocks if Phase 1 task 1.2 (studio-ui skill) is not yet committed; the cross-link target must exist.
- Phase 2 deletes block if their Phase 1 prerequisite task has not committed.
- Phase 3 base-agent rewires require all three Phase 1 skills to exist (1.1, 1.2, 1.3 committed).
- Phase 4 sync requires Phase 3 commits to exist.
- Phase 5 verification requires Phase 4 commit.

Skill loading order per task is encoded in the `dispatch_table.skills` array — the orchestrator pre-loads each skill's content into the subagent's context.

---

## File Structure

Files created (3):
- `.agents/skills/hydra-mcp-server/SKILL.md`
- `.agents/skills/hydra2dgpu-studio-ui/SKILL.md`
- `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md`

Files modified (9):
- `.opencode/opencode.json` (E1')
- `.opencode/rules/AGENTS.md` (E3)
- `.opencode/rules/PLANNING.md` (E4)
- `.opencode/rules/MVP_ARCHITECTURE.md` (E5)
- `.opencode/agents/python-pro.md` (E6)
- `.opencode/agents/cpp-pro.md` (E6)
- `.opencode/agents/debugger.md` (E6)
- `.opencode/agents/performance-engineer.md` (E6)
- `.opencode/agents/build-engineer.md` (E6)
- `.opencode/agents/test-automator.md` (E6)
- 48 model-variant agent files (E7, mechanical)

Files deleted (2):
- `.opencode/AGENTS.md` (E2)
- `.agents/skills/pyqt5-desktop-patterns/` (E8 — the whole folder)

Files read but not modified (audit only):
- `.opencode/rules/{AGENT_SELECTION,CACHE_DISCIPLINE,ENVIRONMENT,GIT_SAFETY,NO_PREMATURE_BACKWARDS_COMPAT,SESSION_DOCUMENTATION}.md`
- `.agents/{codebase-audit,computation-source-truth,studio-gui-api}.md`
- `.opencode/skills/{gpu-test-diagnostics,mesh-quality-triage,sanitize-for-public-push}.md`
- `docs/specs/2026-07-24-hydra-mcp-server-design.md`
- `docs/specs/2026-07-24-cli-first-refactor-design.md`
- `docs/MODEL_GEOPACKAGE_SCHEMA.md`
- `docs/RESULTS_GEOPACKAGE_SCHEMA.md`

---

## Phase 0: Audit (1 task)

### Task 0.1: Audit agent env state

**Files:** Read-only — no edits.

- [ ] **Step 1: Read the spec to confirm scope**

Read: `docs/specs/2026-07-25-agent-env-cleanup-design.md`

Confirm sections §1 (Problem), §5 (Cleanup Edits E1'–E17), §9 (Parallel Execution Plan) all match the current state of the codebase.

- [ ] **Step 2: Verify each rule file in `.opencode/rules/`**

For each of `AGENTS.md`, `AGENT_SELECTION.md`, `CACHE_DISCIPLINE.md`, `ENVIRONMENT.md`, `GIT_SAFETY.md`, `MVP_ARCHITECTURE.md`, `NO_PREMATURE_BACKWARDS_COMPAT.md`, `PLANNING.md`, `SESSION_DOCUMENTATION.md`:

```bash
test -f .opencode/rules/<NAME>.md && wc -l .opencode/rules/<NAME>.md
```

Expected: each file exists and is non-empty. Record line count.

- [ ] **Step 3: Verify the three single-file skills at `.opencode/skills/`**

```bash
test -f .opencode/skills/gpu-test-diagnostics.md && echo "PASS gpu"
test -f .opencode/skills/mesh-quality-triage.md && echo "PASS mesh"
test -f .opencode/skills/sanitize-for-public-push.md && echo "PASS sanitize"
```

Expected: all three PASS. (The opencode.json `instructions` array does not load these — they're loaded via a different mechanism. This audit step just confirms presence.)

- [ ] **Step 4: Verify the five folder skills at `.agents/skills/`**

```bash
for s in frontend-design fvm-cfd-solver-patterns pyqt5-desktop-patterns skills-discovery subagent-driven-development; do
  test -f .agents/skills/$s/SKILL.md && echo "PASS $s"
done
```

Expected: five PASS lines.

- [ ] **Step 5: Verify the three reference notes at `.agents/`**

```bash
for f in codebase-audit.md computation-source-truth.md studio-gui-api.md; do
  test -f .agents/$f && echo "PASS $f"
done
```

Expected: three PASS lines.

- [ ] **Step 6: Verify the six base agent files**

```bash
for a in python-pro cpp-pro debugger performance-engineer build-engineer test-automator; do
  test -f .opencode/agents/$a.md && echo "PASS $a"
done
```

Expected: six PASS lines.

- [ ] **Step 7: Count model-variant agent files**

```bash
ls .opencode/agents/ | grep -v '^[a-z-]*\.md$' | wc -l
```

Expected: 48 (six base names × eight model variants each, though the exact count varies per base).

Record the actual count for use in Task 4.1.

- [ ] **Step 8: Verify E5's `studio_dialog.py` path assumption**

```bash
test -f swe2d/workbench/studio_dialog.py && echo "PATH_OK studio_dialog.py"
ls swe2d/workbench/studio_dialog.py 2>&1 | head -1
```

If `studio_dialog.py` has been renamed (per CLI-first refactor Phase 3.5: "delete the ~120 solver-adjacent dialog methods"), find the new file and record the path. This becomes input to Task 1.7.

- [ ] **Step 9: Verify E12–E17 verify-only items**

For each of `GIT_SAFETY.md`, `CACHE_DISCIPLINE.md`, `ENVIRONMENT.md`, `SESSION_DOCUMENTATION.md`, `AGENT_SELECTION.md`, `NO_PREMATURE_BACKWARDS_COMPAT.md`:

```bash
test -s .opencode/rules/<NAME>.md && echo "OK <NAME>"
```

Expected: six OK lines.

- [ ] **Step 10: Verify source-of-truth docs exist**

```bash
test -f docs/specs/2026-07-24-hydra-mcp-server-design.md && echo "OK mcp spec"
test -f docs/specs/2026-07-24-cli-first-refactor-design.md && echo "OK cli-first spec"
test -f docs/MODEL_GEOPACKAGE_SCHEMA.md && echo "OK model schema"
test -f docs/RESULTS_GEOPACKAGE_SCHEMA.md && echo "OK results schema"
```

Expected: four OK lines.

- [ ] **Step 11: Verify the spec-frontmatter convention**

```bash
head -5 docs/specs/2026-07-25-agent-env-cleanup-design.md | grep -q "^---$" && echo "OK frontmatter"
```

Expected: OK frontmatter (the spec has YAML frontmatter per docs-lifecycle design).

- [ ] **Step 12: Produce audit report**

Print to stdout:

```text
=== AUDIT REPORT ===
Rule files: <count of files found, expected 9>
Skills at .opencode/skills/: <count, expected 3>
Skills at .agents/skills/: <count, expected 5 (pyqt5-desktop-patterns to be deleted in Phase 2 task 2.2)>
Reference notes at .agents/: <count, expected 3>
Base agents: <count, expected 6>
Model variants: <count, expected 48>
studio_dialog.py status: <PATH_OK or RENAMED to path>
Source-of-truth docs: <4 of 4 OK, or list missing>
Frontmatter: <OK or MISSING>
```

- [ ] **Step 13: Report DONE / DONE_WITH_CONCERNS / BLOCKED**

- DONE: all 11 verification steps pass and audit report shows no surprise
- DONE_WITH_CONCERNS: minor drift (a file renamed, a line count different) — record in the report
- BLOCKED: a required source-of-truth doc is missing, or a section of the spec cannot be verified (escalate to orchestrator)

This task does NOT commit. It produces only stdout output.

---

## Phase 1: Parallel batch A (7 tasks)

All seven tasks below touch disjoint files (per spec §9.1). Dispatch in parallel using `dispatching-parallel-agents`. Each task is one subagent invocation.

### Task 1.1: Write `hydra-mcp-server` skill

**Files:**
- Create: `.agents/skills/hydra-mcp-server/SKILL.md`

- [ ] **Step 1: Read the source-of-truth spec**

Read: `docs/specs/2026-07-24-hydra-mcp-server-design.md`

Pay attention to §4 (Tool Catalog), §5 (Session Modes), §6 (Safety & Distribution). The skill is a skeleton — extract the Tier A/B/C tool names, the seven critical rules, and the session-lifecycle summary. Do not paste the full spec verbatim.

- [ ] **Step 2: Write `.agents/skills/hydra-mcp-server/SKILL.md`**

Use the structure defined in spec §4.1:

```markdown
---
name: hydra-mcp-server
description: HYDRA MCP server — production modeling tools (Tier A), live GUI tools (Tier B), and design tools (Tier C). Use when launching a live QGIS session, driving the Studio GUI from an agent, running a parameter sweep, verifying the GUI, or applying a widget design patch.
---

# HYDRA MCP Server

> Source of truth: docs/specs/2026-07-24-hydra-mcp-server-design.md
> This skill is a working summary. Read the spec for tool-level detail.

## When to load
[trigger paragraph from spec §4.1]

## Tier map (skim)
- **Tier A (production, no GUI):** [list all 19 Tier A tool names]
- **Tier B (live GUI):** [list all 13 Tier B tool names]
- **Tier C (design):** [list all 4 Tier C tool names]

## Critical rules (must follow)
1. Workspace-relative paths only (reject `..`, symlinks, escape)
2. Fail-fast errors — no catch+swallow
3. GUI thread safety — dispatch long work to `SimulationWorker`, return via Qt signals
4. `design_apply_patch` is gated — always preview → ask user → apply
5. Bridge auth — per-session random token in a 0600 file, same-machine only (`QLocalSocket`)
6. Frame-length cap — reject oversized frames before decode; do not raise the cap
7. Subprocess lifecycle — QGIS/Xvfb tracked and reaped by `gui_close`

## Session lifecycle (Tier B only)
[paragraph referencing spec §5 — offscreen / xvfb / display modes]

## Common mistakes to avoid
- Calling `gui_get_value` on a widget whose subclass doesn't match the type-discriminator
  → returns "wrong class" error. Use `gui_find_widget_by_path` first to confirm the path.
- Forgetting to `gui_close` → orphaned Xvfb. Always close at the end of a session.
- Treating `model_inspect` as a mutating tool — it is read-only and safe to allowlist.
- Treating `run_start` as synchronous — it is async. Poll `run_status(job_id)`.
- "Drive the canvas with mouse-path simulation" — not implemented; a non-goal
  (spec §4 non-goals).

## Related skills
- `hydra2dgpu-studio-ui` for Studio MVP architecture and CLI/GUI parity rules.
- `hydra2dgpu-gpkg-schema-expert` for the GPKG tables Tier A tools read/write.
```

- [ ] **Step 3: Verify the file is well-formed**

```bash
test -s .agents/skills/hydra-mcp-server/SKILL.md && echo "PASS size"
head -3 .agents/skills/hydra-mcp-server/SKILL.md | grep -q "^---$" && echo "PASS frontmatter"
grep -q "Workspace-relative paths" .agents/skills/hydra-mcp-server/SKILL.md && echo "PASS rule 1"
grep -q "design_apply_patch" .agents/skills/hydra-mcp-server/SKILL.md && echo "PASS rule 4"
grep -q "Tier A" .agents/skills/hydra-mcp-server/SKILL.md && echo "PASS tier A"
grep -q "Tier B" .agents/skills/hydra-mcp-server/SKILL.md && echo "PASS tier B"
grep -q "Tier C" .agents/skills/hydra-mcp-server/SKILL.md && echo "PASS tier C"
```

Expected: seven PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/hydra-mcp-server/SKILL.md
git commit -m "feat(skills): add hydra-mcp-server SKILL.md skeleton"
```

---

### Task 1.2: Write `hydra2dgpu-studio-ui` skill (absorbing pyqt5)

**Files:**
- Create: `.agents/skills/hydra2dgpu-studio-ui/SKILL.md`
- (Deletion handled in Task 2.2)

- [ ] **Step 1: Read both source docs**

Read:
- `docs/specs/2026-07-24-cli-first-refactor-design.md` (sections §3, §4, §5, §7)
- `.agents/skills/pyqt5-desktop-patterns/SKILL.md` (entire file — this is the content to absorb)

- [ ] **Step 2: Diff the source-of-truth docs to enumerate absorbed content**

Per spec §4.2, the new skill must contain:
- 7 CLI/GUI parity rules
- 6 widget-lifecycle rules (absorbed from pyqt5)
- StudioComponent registry section (absorbed)
- Feature flags section (absorbed)
- State persistence section (absorbed)
- Structural changes checklist
- Key file locations table

Build the new SKILL.md by combining these. Do not omit any absorbed item.

- [ ] **Step 3: Write `.agents/skills/hydra2dgpu-studio-ui/SKILL.md`**

```markdown
---
name: hydra2dgpu-studio-ui
description: HYDRA2DGPU Studio UI — CLI/GUI parity, MVP layering, dock lifecycle, signal safety, widget liveness. Use for any Studio UI work — adding tabs, wiring widgets, structural changes, or anything touching the Studio CLI/GUI parity boundary.
---

# HYDRA2DGPU Studio UI

> Source of truth: docs/specs/2026-07-24-cli-first-refactor-design.md (CLI/GUI parity)
> Absorbs: the prior pyqt5-desktop-patterns skill (dock lifecycle, signals, persistence).

## When to load
[trigger paragraph from spec §4.2]

## The two-layer mental model
- **Core (`swe2d/core/`):** GUI-free pipeline. May import `qgis.core`. Must NOT import
  `qgis.gui`, `qgis.utils.iface`, `qgis.PyQt`, or `PyQt5`. Run spec → RunContext →
  executor → persisted results.
- **View (`swe2d/workbench/`):** Qt widgets, docks, dialogs. Reduced to (a) serialize
  widget state into the same run spec, (b) render results. May import anything Qt.

The run spec schema is `swe2d-run/2`. The GUI is one input path. The CLI is another.
They MUST produce byte-equal specs for the same project — proven by
`tests/test_run_context_parity.py` and `tests/test_cli_gui_replay_parity.py`, not by
inspection.

## CLI/GUI parity — critical rules
1. One canonical builder — `swe2d/core/builder.py::build_run_context`
2. One defaults table — `_DEFAULTS`, every constructor reaches it
3. Fail-fast validation at every level (top + nested unknown keys raise)
4. No silent absorbs — no `try/except Exception → warning + None`
5. CLI claims are precise — "requires qgis.core; no QGIS GUI, iface, or display needed"
6. No view-bound callables ride RunContext — Qt signals only
7. No re-export shims, no deprecation period

## Widget lifecycle — critical rules (carried from pyqt5-desktop-patterns)
1. Liveness guard — `try: _ = widget.objectName() except RuntimeError: widget = None`
2. `isinstance` can raise — wrap in try/except (deleted C++ QObject)
3. `safe_disconnect` (`swe2d_workbench_qt`) for signal disconnect
4. Delete empty parent shells after extracting children
5. `QTimer.stop()` + `deleteLater()` before destroying parent
6. `.ui` files are source of truth; run `tools/ui_bind_sync.py` after every edit

## StudioComponent registry
[absorb verbatim from pyqt5-desktop-patterns/SKILL.md sections "StudioComponent Registry API" through "Left-pane tabs"]

## Feature flags
[absorb verbatim from pyqt5-desktop-patterns/SKILL.md section "Feature Flags"]

## State persistence
[absorb verbatim from pyqt5-desktop-patterns/SKILL.md section "Workbench State Persistence", including the auto-discovery pitfalls]

## Structural changes checklist
1. Update `.ui` file (Qt Designer)
2. Update py bindings
3. Run `tools/ui_bind_sync.py` (missing + orphan check)
4. If a new CLI flag should mirror this widget, add it to the run spec schema
   (`swe2d-run/2`); both GUI and CLI routes reach the same builder key
5. If a feature flag is added, update the 3-file set: dict + keyword function in
   `SWE2DWorkbenchStudioDialog`, menu/toolbar actions in `studio_host_methods.py`
6. Verify the parity test (`tests/test_run_context_parity.py`) still passes

## Key file locations
| File | Purpose |
|------|---------|
| `swe2d/core/builder.py` | Canonical run-context builder |
| `swe2d/core/executor.py` | GUI-free executor with sink protocol |
| `swe2d/core/run_context.py` | RunContext dataclass |
| `swe2d/cli/` | Headless CLI entry points |
| `swe2d/workbench/studio_dialog.py` | Main dialog (view layer) |
| `swe2d/workbench/views/*.py` | Sub-views |
| `swe2d/workbench/controllers/*.py` | Controllers (orchestration) |
| `swe2d/workbench/studio_component.py` | `StudioComponent` dataclass + tab registry |
| `swe2d/workbench/extracted/` | Seam modules extracted from monolith |
| `swe2d/workbench/project_settings.py` | Widget state persist/restore |
| `swe2d/workbench/services/*` | Pure-Python logic that belongs in `swe2d/core/` |
| `swe2d/workbench/workers/simulation_worker.py` | QThread shell around `core.executor` |

## Common mistakes to avoid
- Adding a widget binding in `studio_dialog.py` without updating the canonical builder
- Reading widget state directly from `studio_dialog` in a service module
- Adding a CLI-only or GUI-only code path without a parity gate
- Using `dataclasses.replace(...)` on `RunContext` — the controller "flip" antipattern
- Catching `Exception` broadly to make tests pass
```

- [ ] **Step 4: Verify every absorbed item is present**

```bash
diff <(grep -E '^## ' .agents/skills/pyqt5-desktop-patterns/SKILL.md | sed 's/^## //') \
     <(grep -E '^## ' .agents/skills/hydra2dgpu-studio-ui/SKILL.md | sed 's/^## //')
```

Expected: the new file's section headers are a SUPERSET of the old file's headers (every old section is represented in the new file, possibly renamed). Empty diff is also acceptable if all old sections are folded into new section names — verify by spot-checking each old header maps to a new section.

```bash
test -s .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS size"
head -3 .agents/skills/hydra2dgpu-studio-ui/SKILL.md | grep -q "^---$" && echo "PASS frontmatter"
grep -q "One canonical builder" .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS parity rule 1"
grep -q "Liveness guard" .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS lifecycle rule 1"
grep -q "StudioComponent" .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS registry"
grep -q "Feature flags" .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS feature flags"
grep -q "State persistence" .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS persistence"
grep -q "Structural changes checklist" .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS checklist"
```

Expected: nine PASS lines (size, frontmatter, 2 rules, 3 absorbed sections, checklist).

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/hydra2dgpu-studio-ui/SKILL.md
git commit -m "feat(skills): add hydra2dgpu-studio-ui SKILL.md (absorbs pyqt5-desktop-patterns)"
```

---

### Task 1.3: Write `hydra2dgpu-gpkg-schema-expert` skill

**Files:**
- Create: `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md`

- [ ] **Step 1: Read both schema docs**

Read:
- `docs/MODEL_GEOPACKAGE_SCHEMA.md` (entire file)
- `docs/RESULTS_GEOPACKAGE_SCHEMA.md` (entire file)

- [ ] **Step 2: Write `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md`**

Use the structure defined in spec §4.3:

```markdown
---
name: hydra2dgpu-gpkg-schema-expert
description: HYDRA2DGPU model & results GeoPackage schemas — 18 model tables, 5 results tables, BLOB layout, FK rules, common queries. Use when reading or writing any model GeoPackage or results GeoPackage.
---

# HYDRA2DGPU GeoPackage Schema Expert

> Source of truth: docs/MODEL_GEOPACKAGE_SCHEMA.md + docs/RESULTS_GEOPACKAGE_SCHEMA.md
> This skill is a working summary. Read the schema docs for column-level detail.

## When to load
[trigger paragraph from spec §4.3]

## Two files, two roles
- **Model GeoPackage** — user-authored input (mesh topology, BCs, rainfall, drainage, structures)
- **Results GeoPackage** — solver output (snapshot arrays, line timeseries, run logs)

Both are valid OGC GeoPackages (SQLite + spatial metadata). Both share the same
`gpkg_contents`, `gpkg_geometry_columns`, `spatial_ref_sys` metadata tables.

## Model GPKG — 18 domain tables (skim)

| # | Table | Geometry | What it stores |
|---|---|---|---|
| 1 | `swe2d_topo_nodes` | POINT | Seed nodes for mesh gen |
| 2 | `swe2d_topo_arcs` | LINESTRING | Topological arcs / breaklines |
| 3 | `swe2d_topo_regions` | POLYGON | Mesh refinement regions |
| 4 | `swe2d_topo_constraints` | POLYGON | Hard constraints (holes, refinements) |
| 5 | `swe2d_topo_quad_edges` | LINESTRING | Quad-boundary-layer edges |
| 6 | `swe2d_manning_zones` | POLYGON | Manning's n roughness |
| 7 | `swe2d_bc_lines` | LINESTRING | Boundary conditions |
| 8 | `swe2d_sample_lines` | LINESTRING | Sample/monitoring lines |
| 9 | `swe2d_rain_gages` | POINT | Rain gauge locations |
| 10 | `swe2d_storm_areas` | POLYGON | Storm/rainfall polygons |
| 11 | `swe2d_cn_zones` | POLYGON | Curve Number zones |
| 12 | `swe2d_hyetographs` | (attr) | Rainfall intensity time series |
| 13 | `swe2d_hydrographs` | (attr) | BC flow/stage time series |
| 14 | `swe2d_drainage_nodes` | POINT | Drainage manholes/junctions |
| 15 | `swe2d_drainage_links` | LINESTRING | Drainage pipes/channels |
| 16 | `swe2d_drainage_inlets` | (attr) | Inlet type definitions |
| 17 | `swe2d_drainage_node_inlets` | (attr) | Node→inlet assignments |
| 18 | `swe2d_structures` | LINESTRING | Weirs/orifices/culverts/bridges/pumps/dams |

## Results GPKG — 5 result tables (skim)

| Table | Row granularity | Storage format |
|---|---|---|
| `swe2d_baked_results` | one per run | BLOB (`np.ndarray.tobytes()`); `times_blob`, `h_blob`, `hu_blob`, `hv_blob`, `max_*_blob` |
| `swe2d_baked_line_ts` | one per (run, line) | BLOB; mean depth/velocity/WSE/bed/flow/wet_frac/Fr per timestep |
| `swe2d_baked_line_profiles` | one per (run, line) | BLOB; 2D time × station profiles |
| `swe2d_baked_coupling_ts` | one per (run, element) | BLOB; coupling flow time series |
| `swe2d_runs` / `swe2d_run_logs` | metadata per run | text |

## Critical rules (must follow)
1. CRS is the project CRS, not always 4326
2. Units are model units, not SI
3. BLOB layout is fixed and versioned — don't reshape
4. Dry cells are omitted in mesh results
5. No FK enforcement — application code maintains referential integrity
6. Layers are created empty — populate via QGIS digitizing or `*_configure` MCP tools
7. `swe2d_*` table prefix is reserved

## Common queries

```sql
-- List all baked meshes in a model GPKG
SELECT mesh_name, n_cells, n_nodes FROM swe2d_baked_meshes ORDER BY mesh_name;

-- Dump BCs for a given mesh
SELECT bc_type, bc_value, hydrograph_id FROM swe2d_bc_lines ORDER BY bc_type;

-- Verify drainage FK chain
SELECT l.link_id, l.from_node, l.to_node, n1.node_id AS from_exists, n2.node_id AS to_exists
FROM swe2d_drainage_links l
LEFT JOIN swe2d_drainage_nodes n1 ON l.from_node = n1.node_id
LEFT JOIN swe2d_drainage_nodes n2 ON l.to_node = n2.node_id;

-- Get max-h envelope for a run
SELECT run_id, mesh_name, n_cells, n_timesteps FROM swe2d_baked_results WHERE run_id = ?;

-- Diff two runs by field
SELECT a.run_id, b.run_id, a.times_blob, b.times_blob
FROM swe2d_baked_results a, swe2d_baked_results b
WHERE a.run_id = ? AND b.run_id = ?;

-- List runs in a results GPKG
SELECT run_id, mesh_name, created_utc FROM swe2d_runs ORDER BY created_utc DESC;

-- Count dry cells in a snapshot (Python-side)
import numpy as np
h_blob = gpkg_read('swe2d_baked_results', 'h_blob', 'run_id = ?', [run_id])
dry_count = np.sum(h_blob == 0)  # for a single timestep slice
```

## MCP tools that touch the GPKG
- Tier A: `model_create`, `model_inspect`, `mesh_bake`, `terrain_assign`,
  `bc_configure`, `rainfall_configure`, `drainage_configure`, `structures_configure`,
  `run_list`, `results_query`, `results_timeseries`, `results_compare`.
- See `hydra-mcp-server` skill for the tool catalog.

## Common mistakes to avoid
- Hardcoding `srs_id = 4326` — always join `gpkg_geometry_columns` and `gpkg_contents`
- Assuming `h_blob` is `(n_timesteps × n_cells)` — check `n_cells` from row metadata
- Reading `swe2d_baked_results` rows by `run_id` is a PK lookup — fast
- Writing a result row without the matching `swe2d_runs` metadata row
- Treating `bc_value` as SI — it is in model units (feet/cfs for USC, meters/cms for SI)
```

- [ ] **Step 3: Verify the file**

```bash
test -s .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md && echo "PASS size"
head -3 .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md | grep -q "^---$" && echo "PASS frontmatter"
grep -q "swe2d_topo_nodes" .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md && echo "PASS model table 1"
grep -q "swe2d_structures" .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md && echo "PASS model table 18"
grep -q "swe2d_baked_results" .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md && echo "PASS results table"
grep -q "BLOB layout is fixed" .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md && echo "PASS rule 3"
grep -q "No FK enforcement" .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md && echo "PASS rule 5"
```

Expected: nine PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md
git commit -m "feat(skills): add hydra2dgpu-gpkg-schema-expert SKILL.md skeleton"
```

---

### Task 1.4: Edit `opencode.json` instructions array

**Files:**
- Modify: `.opencode/opencode.json`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/opencode.json
```

Confirm the `instructions` array shape matches spec §5 E1' (currently 10 entries, three of which point at non-existent files).

- [ ] **Step 2: Replace the `instructions` array**

Use the `edit` tool to replace the existing array with the corrected list. The diff:

```diff
   "instructions": [
+    ".opencode/rules/AGENTS.md",
     ".opencode/rules/UNIT_SYSTEM.md",
-    ".opencode/rules/UNIT_SYSTEM.md",
     ".opencode/rules/GIT_SAFETY.md",
-    ".opencode/rules/TEST_PRIORITY.md",
+    ".opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md",
     ".opencode/rules/CACHE_DISCIPLINE.md",
-    ".opencode/rules/STUDIO_UI.md",
     ".opencode/rules/SESSION_DOCUMENTATION.md",
     ".opencode/rules/ENVIRONMENT.md",
     ".opencode/rules/AGENT_SELECTION.md",
     ".opencode/rules/PLANNING.md",
     ".opencode/rules/MVP_ARCHITECTURE.md"
   ],
```

The corrected array (10 entries, all pointing at files that exist):

```json
"instructions": [
  ".opencode/rules/AGENTS.md",
  ".opencode/rules/GIT_SAFETY.md",
  ".opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md",
  ".opencode/rules/CACHE_DISCIPLINE.md",
  ".opencode/rules/SESSION_DOCUMENTATION.md",
  ".opencode/rules/ENVIRONMENT.md",
  ".opencode/rules/AGENT_SELECTION.md",
  ".opencode/rules/PLANNING.md",
  ".opencode/rules/MVP_ARCHITECTURE.md"
]
```

Note: 9 entries (was 10), since three phantom rules were removed and two real rules were added.

- [ ] **Step 3: Validate JSON**

```bash
python3 -c "import json; print(len(json.load(open('.opencode/opencode.json'))['instructions']))"
```

Expected: `9`.

- [ ] **Step 4: Verify every instruction path exists**

```bash
python3 -c "
import json, os, sys
cfg = json.load(open('.opencode/opencode.json'))
errors = []
for p in cfg['instructions']:
    if not os.path.exists(p):
        errors.append(p)
if errors:
    print('FAIL: phantom instructions:', errors)
    sys.exit(1)
print('PASS: all', len(cfg['instructions']), 'instruction paths exist')
"
```

Expected: `PASS: all 9 instruction paths exist`.

- [ ] **Step 5: Commit**

```bash
git add .opencode/opencode.json
git commit -m "fix(opencode): remove phantom rule paths, add AGENTS.md + NO_PREMATURE_BACKWARDS_COMPAT.md"
```

---

### Task 1.5: Edit `.opencode/rules/AGENTS.md` skill table

**Files:**
- Modify: `.opencode/rules/AGENTS.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/rules/AGENTS.md
```

Locate the "Available Agent Skills" table (currently lists `pyqt5-desktop-patterns` and `qgis-plugin-conventions`, neither of which will exist after this plan lands — the former because it will be deleted in Task 2.2, the latter because it never existed). Also locate the "Command Code Resources" paragraph.

- [ ] **Step 2: Replace the skill table**

Use `edit` to replace the entire skill table with the corrected version:

```markdown
## Available Agent Skills

This repo ships the following domain skills (see `.agents/skills/`):

| Skill | File | When to use | Trigger |
|-------|------|-------------|---------|
| PyQt5 Desktop Patterns | `.agents/skills/pyqt5-desktop-patterns/SKILL.md` | (REMOVING — merged into hydra2dgpu-studio-ui) | — |
| HYDRA MCP Server | `.agents/skills/hydra-mcp-server/SKILL.md` | Live QGIS GUI tools, run sweeps, verify GUI | "use the hydra MCP tools", "launch a live QGIS session" |
| HYDRA2DGPU Studio UI | `.agents/skills/hydra2dgpu-studio-ui/SKILL.md` | Studio tabs, widgets, dock lifecycle, CLI/GUI parity | "edit Studio UI", "add a tab", "wire a widget" |
| HYDRA2DGPU GPKG Schema | `.agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md` | Read/write model or results GPKG | "inspect a results run", "store BC config" |
| FVM / CFD Solver Patterns | `.agents/skills/fvm-cfd-solver-patterns/SKILL.md` | Mesh structs, GPU kernels, BCs, coupling | (unchanged) |
| GPU Test Diagnostics | `.opencode/skills/gpu-test-diagnostics.md` | Diagnose CUDA test failures | (unchanged) |
| Mesh Quality Triage | `.opencode/skills/mesh-quality-triage.md` | Diagnose Gmsh mesh failures | (unchanged) |
| Subagent-Driven Development | `.agents/skills/subagent-driven-development/SKILL.md` | Run implementation plans | (unchanged) |
| Frontend Design | `.agents/skills/frontend-design/SKILL.md` | (unchanged) |
| Skills Discovery | `.agents/skills/skills-discovery/SKILL.md` | (unchanged) |
```

If `qgis-plugin-conventions` appears anywhere in the file, remove that row.

- [ ] **Step 3: Remove the Command Code Resources paragraph**

The `.commandcode/` paragraph is obsolete (directory doesn't exist). Delete it. The replacement paragraph is:

```markdown
## Agent Resources

- **Rules** — `.opencode/rules/`
- **Skills** — `.agents/skills/` (folder form) and `.opencode/skills/` (flat form)
- **Reference notes** — `.agents/*.md`
- **Agents** — `.opencode/agents/` (6 base + 48 model variants)
- **Plugin** — `.opencode/plugins/plan_dispatcher.ts` (reads the `instructions` + `dispatch_table` of every plan)
```

- [ ] **Step 4: Verify the changes**

```bash
! grep -q "qgis-plugin-conventions" .opencode/rules/AGENTS.md && echo "PASS no phantom"
! grep -q "pyqt5-desktop-patterns/SKILL.md\` | Studio UI" .opencode/rules/AGENTS.md && echo "PASS pyqt5 row removed"
! grep -q "\.commandcode/" .opencode/rules/AGENTS.md && echo "PASS no ghost"
grep -q "hydra-mcp-server/SKILL.md" .opencode/rules/AGENTS.md && echo "PASS new skill 1"
grep -q "hydra2dgpu-studio-ui/SKILL.md" .opencode/rules/AGENTS.md && echo "PASS new skill 2"
grep -q "hydra2dgpu-gpkg-schema-expert/SKILL.md" .opencode/rules/AGENTS.md && echo "PASS new skill 3"
```

Expected: six PASS lines.

- [ ] **Step 5: Commit**

```bash
git add .opencode/rules/AGENTS.md
git commit -m "fix(rules): update AGENTS.md skill table — drop phantom qgis-plugin-conventions, add three new skills"
```

---

### Task 1.6: Delete duplicated 12-principles block in `rules/PLANNING.md`

**Files:**
- Modify: `.opencode/rules/PLANNING.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/rules/PLANNING.md
```

Locate the "ALWAYS KEEP THESE PRINCIPLES OF GOOD SOFTWARE ENGINEERING IN MIND" section (between the "Cross-review rule" section and the "Docs lifecycle override" section). It is a verbatim duplicate of the same block in `rules/AGENTS.md`.

- [ ] **Step 2: Delete the block**

Use `edit` to remove the section. The deletion boundary is the heading line through the last numbered item (POLA) plus any trailing blank lines until the next heading ("Docs lifecycle override").

- [ ] **Step 3: Verify the deletion**

```bash
! grep -q "twelve fundamental software engineer principles" .opencode/rules/PLANNING.md && echo "PASS block deleted"
grep -q "Docs lifecycle override" .opencode/rules/PLANNING.md && echo "PASS next section intact"
grep -q "Cross-review rule" .opencode/rules/PLANNING.md && echo "PASS prev section intact"
wc -l .opencode/rules/PLANNING.md
```

Expected: three PASS lines. Line count drops by ~115 lines (the principles block was that long).

- [ ] **Step 4: Confirm the principles block still exists in AGENTS.md (single source of truth)**

```bash
grep -q "twelve fundamental software engineer principles" .opencode/rules/AGENTS.md && echo "PASS principles still in AGENTS.md"
```

Expected: one PASS line.

- [ ] **Step 5: Commit**

```bash
git add .opencode/rules/PLANNING.md
git commit -m "refactor(rules): dedupe 12-principles block — single source in AGENTS.md"
```

---

### Task 1.7: Edit `rules/MVP_ARCHITECTURE.md` (verify studio_dialog.py path + add cross-link)

**Files:**
- Modify: `.opencode/rules/MVP_ARCHITECTURE.md`

- [ ] **Step 1: Verify `studio_dialog.py` path**

```bash
test -f swe2d/workbench/studio_dialog.py && echo "PATH_OK" || echo "RENAMED"
ls swe2d/workbench/studio_dialog.py 2>&1
```

If PATH_OK: keep references to `studio_dialog.py` as-is. If RENAMED: search the file for any other `studio_dialog` reference and update to the new path.

- [ ] **Step 2: Add a cross-link to the new studio-ui skill**

Find the "## Enforcement" section and append a one-paragraph cross-link:

```markdown
## Related Skills

For Studio UI work — adding tabs, wiring widgets, structural changes, CLI/GUI
parity — see the `hydra2dgpu-studio-ui` skill at
`.agents/skills/hydra2dgpu-studio-ui/SKILL.md`. It absorbs the prior
`pyqt5-desktop-patterns` skill and adds the canonical-builder / parity rules
from the CLI-first refactor.
```

- [ ] **Step 3: Verify the edit**

```bash
test -f swe2d/workbench/studio_dialog.py || grep -q "## Related Skills" .opencode/rules/MVP_ARCHITECTURE.md
grep -q "hydra2dgpu-studio-ui" .opencode/rules/MVP_ARCHITECTURE.md && echo "PASS cross-link"
grep -q "## Related Skills" .opencode/rules/MVP_ARCHITECTURE.md && echo "PASS section"
```

Expected: at least two PASS lines (cross-link and section header present).

- [ ] **Step 4: Commit**

```bash
git add .opencode/rules/MVP_ARCHITECTURE.md
git commit -m "refactor(rules): verify studio_dialog.py path, cross-link hydra2dgpu-studio-ui skill"
```

---

## Phase 2: Parallel batch B (2 tasks)

Both tasks touch disjoint paths and have different Phase 1 prerequisites. Dispatch in parallel after Tasks 1.4 and 1.2 have committed.

### Task 2.1: Delete orphan `.opencode/AGENTS.md`

**Files:**
- Delete: `.opencode/AGENTS.md`

- [ ] **Step 1: Confirm the canonical version is now loaded via `opencode.json`**

```bash
grep -q "\.opencode/rules/AGENTS.md" .opencode/opencode.json && echo "PASS canonical loaded"
```

Expected: PASS. If this fails, Task 1.4 has not yet committed — halt and report BLOCKED.

- [ ] **Step 2: Delete the orphan**

```bash
git rm .opencode/AGENTS.md
```

- [ ] **Step 3: Verify**

```bash
! test -f .opencode/AGENTS.md && echo "PASS deleted"
test -f .opencode/rules/AGENTS.md && echo "PASS canonical still present"
```

Expected: two PASS lines.

- [ ] **Step 4: Commit (the `git rm` already staged the change)**

```bash
git commit -m "chore: delete orphan .opencode/AGENTS.md (now loaded via opencode.json instructions)"
```

---

### Task 2.2: Delete old `pyqt5-desktop-patterns` skill folder

**Files:**
- Delete: `.agents/skills/pyqt5-desktop-patterns/` (entire folder)

- [ ] **Step 1: Confirm the new studio-ui skill is committed**

```bash
git log --oneline -1 -- .agents/skills/hydra2dgpu-studio-ui/SKILL.md | grep -q "hydra2dgpu-studio-ui" && echo "PASS"
```

Expected: PASS. If this fails, Task 1.2 has not yet committed — halt and report BLOCKED.

- [ ] **Step 2: Verify all absorbed content is in the new skill**

```bash
for header in "Liveness guard" "StudioComponent" "Feature flags" "State persistence" "Structural changes checklist"; do
  grep -q "$header" .agents/skills/hydra2dgpu-studio-ui/SKILL.md || { echo "MISSING: $header"; exit 1; }
done
echo "PASS all absorbed content present"
```

Expected: PASS. If this fails, the new skill is incomplete — halt and escalate (the implementer of Task 1.2 needs to fix the absorb gap, not this task).

- [ ] **Step 3: Delete the old skill**

```bash
git rm -r .agents/skills/pyqt5-desktop-patterns/
```

- [ ] **Step 4: Verify**

```bash
! test -d .agents/skills/pyqt5-desktop-patterns && echo "PASS deleted"
test -f .agents/skills/hydra2dgpu-studio-ui/SKILL.md && echo "PASS replacement exists"
```

Expected: two PASS lines.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: delete pyqt5-desktop-patterns skill (merged into hydra2dgpu-studio-ui)"
```

---

## Phase 3: Parallel batch C (6 tasks)

Each task edits a disjoint base agent file. Dispatch in parallel after Tasks 1.1, 1.2, 1.3, 2.2 have all committed (the three new skills must exist; the old skill must be gone).

### Task 3.1: Update `.opencode/agents/python-pro.md`

**Files:**
- Modify: `.opencode/agents/python-pro.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/agents/python-pro.md
```

Locate any "Available Skills" / "Skills to consult" / "skill" reference section.

- [ ] **Step 2: Update skill references**

Apply these edits in one pass:

1. Add the three new skills to the "Available Skills" section (if one exists) with a one-line description each:
   ```markdown
   - **hydra-mcp-server**: live QGIS GUI tools, parameter sweeps, design patches. Use when driving Studio from an agent.
   - **hydra2dgpu-studio-ui**: Studio MVP architecture, CLI/GUI parity, dock lifecycle, widget liveness. Use for any Studio UI work.
   - **hydra2dgpu-gpkg-schema-expert**: model & results GeoPackage schemas. Use when reading or writing any GPKG.
   ```

2. If `qgis-plugin-conventions` or `.commandcode/` appears, remove the reference.

3. If `pyqt5-desktop-patterns` appears with a "use for Studio UI" annotation, update the annotation to point at `hydra2dgpu-studio-ui`.

- [ ] **Step 3: Verify**

```bash
! grep -q "qgis-plugin-conventions" .opencode/agents/python-pro.md && echo "PASS no phantom"
! grep -q "\.commandcode/" .opencode/agents/python-pro.md && echo "PASS no ghost"
grep -q "hydra-mcp-server" .opencode/agents/python-pro.md && echo "PASS new skill 1"
grep -q "hydra2dgpu-studio-ui" .opencode/agents/python-pro.md && echo "PASS new skill 2"
grep -q "hydra2dgpu-gpkg-schema-expert" .opencode/agents/python-pro.md && echo "PASS new skill 3"
```

Expected: five PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/python-pro.md
git commit -m "chore(agents): update python-pro base agent — add three new skills, drop stale refs"
```

---

### Task 3.2: Update `.opencode/agents/cpp-pro.md`

**Files:**
- Modify: `.opencode/agents/cpp-pro.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/agents/cpp-pro.md
```

- [ ] **Step 2: Update skill references**

Apply the same edits as Task 3.1, but only those relevant to a C++/CUDA agent:

1. Add `hydra2dgpu-gpkg-schema-expert` (useful when C++ code reads/writes results BLOBs)
2. Drop `qgis-plugin-conventions` if present
3. Drop `.commandcode/` references if present

Skip `hydra-mcp-server` and `hydra2dgpu-studio-ui` unless they have clear relevance to C++/CUDA work (verify by reading the agent file's existing scope; if it's purely C++/CUDA-focused, omit them).

- [ ] **Step 3: Verify**

```bash
! grep -q "qgis-plugin-conventions" .opencode/agents/cpp-pro.md && echo "PASS no phantom"
! grep -q "\.commandcode/" .opencode/agents/cpp-pro.md && echo "PASS no ghost"
```

Expected: two PASS lines. (The third assertion is conditional based on whether gpkg-schema-expert was added.)

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/cpp-pro.md
git commit -m "chore(agents): update cpp-pro base agent — drop stale refs"
```

---

### Task 3.3: Update `.opencode/agents/debugger.md`

**Files:**
- Modify: `.opencode/agents/debugger.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/agents/debugger.md
```

- [ ] **Step 2: Update skill references**

Apply edits:

1. Add `hydra-mcp-server` (debugger may need to use live GUI tools to reproduce issues)
2. Add `hydra2dgpu-gpkg-schema-expert` (debugging result-vs-expected often requires inspecting the GPKG)
3. Drop `qgis-plugin-conventions` if present
4. Drop `.commandcode/` references if present

- [ ] **Step 3: Verify**

```bash
! grep -q "qgis-plugin-conventions" .opencode/agents/debugger.md && echo "PASS no phantom"
! grep -q "\.commandcode/" .opencode/agents/debugger.md && echo "PASS no ghost"
grep -q "hydra-mcp-server" .opencode/agents/debugger.md && echo "PASS new skill 1"
grep -q "hydra2dgpu-gpkg-schema-expert" .opencode/agents/debugger.md && echo "PASS new skill 3"
```

Expected: four PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/debugger.md
git commit -m "chore(agents): update debugger base agent — add mcp-server + gpkg-schema skills"
```

---

### Task 3.4: Update `.opencode/agents/performance-engineer.md`

**Files:**
- Modify: `.opencode/agents/performance-engineer.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/agents/performance-engineer.md
```

- [ ] **Step 2: Update skill references**

Apply edits:

1. Add `hydra2dgpu-gpkg-schema-expert` (perf engineers reading BLOB layouts for size optimization)
2. Drop `qgis-plugin-conventions` if present
3. Drop `.commandcode/` references if present

- [ ] **Step 3: Verify**

```bash
! grep -q "qgis-plugin-conventions" .opencode/agents/performance-engineer.md && echo "PASS no phantom"
! grep -q "\.commandcode/" .opencode/agents/performance-engineer.md && echo "PASS no ghost"
grep -q "hydra2dgpu-gpkg-schema-expert" .opencode/agents/performance-engineer.md && echo "PASS new skill 3"
```

Expected: three PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/performance-engineer.md
git commit -m "chore(agents): update performance-engineer base agent — add gpkg-schema skill"
```

---

### Task 3.5: Update `.opencode/agents/build-engineer.md`

**Files:**
- Modify: `.opencode/agents/build-engineer.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/agents/build-engineer.md
```

- [ ] **Step 2: Update skill references**

Apply edits:

1. Add `hydra-mcp-server` (build engineer may set up the MCP server config)
2. Drop `qgis-plugin-conventions` if present
3. Drop `.commandcode/` references if present

- [ ] **Step 3: Verify**

```bash
! grep -q "qgis-plugin-conventions" .opencode/agents/build-engineer.md && echo "PASS no phantom"
! grep -q "\.commandcode/" .opencode/agents/build-engineer.md && echo "PASS no ghost"
grep -q "hydra-mcp-server" .opencode/agents/build-engineer.md && echo "PASS new skill 1"
```

Expected: three PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/build-engineer.md
git commit -m "chore(agents): update build-engineer base agent — add mcp-server skill"
```

---

### Task 3.6: Update `.opencode/agents/test-automator.md`

**Files:**
- Modify: `.opencode/agents/test-automator.md`

- [ ] **Step 1: Read the current file**

```bash
cat .opencode/agents/test-automator.md
```

- [ ] **Step 2: Update skill references**

Apply edits:

1. Add `hydra-mcp-server` (test-automator uses MCP tools for live GUI testing)
2. Add `hydra2dgpu-gpkg-schema-expert` (test-automator inspects test results in GPKG)
3. Drop `qgis-plugin-conventions` if present
4. Drop `.commandcode/` references if present

- [ ] **Step 3: Verify**

```bash
! grep -q "qgis-plugin-conventions" .opencode/agents/test-automator.md && echo "PASS no phantom"
! grep -q "\.commandcode/" .opencode/agents/test-automator.md && echo "PASS no ghost"
grep -q "hydra-mcp-server" .opencode/agents/test-automator.md && echo "PASS new skill 1"
grep -q "hydra2dgpu-gpkg-schema-expert" .opencode/agents/test-automator.md && echo "PASS new skill 3"
```

Expected: four PASS lines.

- [ ] **Step 4: Commit**

```bash
git add .opencode/agents/test-automator.md
git commit -m "chore(agents): update test-automator base agent — add mcp-server + gpkg-schema skills"
```

---

## Phase 4: Mechanical sync (1 task)

### Task 4.1: Mirror base agent bodies into 48 model variants

**Files:**
- Modify: 48 model-variant agent files in `.opencode/agents/*_*.md`

- [ ] **Step 1: Enumerate base + variant pairs**

```bash
cd .opencode/agents
for base in python-pro cpp-pro debugger performance-engineer build-engineer test-automator; do
  for variant in ${base}_*.md; do
    test -f "$variant" && echo "PAIR $base $variant"
  done
done
```

Record the count of variants per base. Expected total: 48.

- [ ] **Step 2: Verify each variant's current body matches its base, modulo `model:` line**

For each pair, compute the diff:

```bash
diff <(cat .opencode/agents/$base.md) \
     <(cat .opencode/agents/$variant | sed '/^model:/d')
```

Expected: empty diff (the only difference should be the `model:` line, which we stripped from the variant side). If any variant has additional drift, log it and report DONE_WITH_CONCERNS — do not auto-fix in this task; the orchestrator decides whether to escalate.

- [ ] **Step 3: For each variant, copy base body and overwrite `model:` line**

For each pair:

```bash
# Read the variant's existing model: line
model_line=$(grep '^model:' ".opencode/agents/$variant")
# Copy the base body into the variant, then replace any model: line with the variant's original
cp ".opencode/agents/$base.md" ".opencode/agents/$variant"
sed -i "s|^model:.*|$model_line|" ".opencode/agents/$variant"
```

- [ ] **Step 4: Verify all variants are now byte-equal to base, modulo `model:` line**

```bash
cd .opencode/agents
fail=0
for base in python-pro cpp-pro debugger performance-engineer build-engineer test-automator; do
  for variant in ${base}_*.md; do
    test -f "$variant" || continue
    if ! diff -q <(cat $base.md) <(sed '/^model:/d' $variant) >/dev/null; then
      echo "FAIL: $variant differs from $base beyond model: line"
      fail=$((fail+1))
    fi
  done
done
exit $fail
```

Expected: exit code 0 (no FAIL lines).

- [ ] **Step 5: Confirm variant count is unchanged (sanity check the loop)**

```bash
ls .opencode/agents/ | grep -E '^[a-z]+(_|-)[a-z0-9-]+\.md$' | wc -l
```

Expected: 48.

- [ ] **Step 6: Commit**

```bash
git add .opencode/agents/*_*.md
git commit -m "chore(agents): mechanically sync 48 model-variant bodies with their base agents"
```

---

## Phase 5: Verification gate (1 task)

### Task 5.1: Run the spec §6 verification gate

**Files:** None — read-only verification.

- [ ] **Step 1: Pre-flight — confirm all prior phase commits exist**

```bash
git log --oneline | grep -c "feat(skills): add hydra-" | grep -q 3 && echo "PASS three new skills"
git log --oneline | grep -q "fix(opencode)" && echo "PASS opencode.json edit"
git log --oneline | grep -q "fix(rules): update AGENTS.md" && echo "PASS rules/AGENTS.md edit"
git log --oneline | grep -q "chore: delete orphan .opencode/AGENTS.md" && echo "PASS orphan deleted"
git log --oneline | grep -q "chore: delete pyqt5-desktop-patterns" && echo "PASS pyqt5 deleted"
git log --oneline | grep -q "chore(agents): mechanically sync" && echo "PASS variants synced"
```

Expected: six PASS lines. If any FAIL, identify which Phase did not commit and report BLOCKED.

- [ ] **Step 2: Run shell block #1 (audit confirms the cleanup)**

```bash
! test -f .opencode/AGENTS.md
! grep -q "qgis-plugin-conventions" .opencode/rules/AGENTS.md
! grep -q "\.commandcode/" .opencode/rules/AGENTS.md
! grep -q "twelve fundamental software engineer principles" .opencode/rules/PLANNING.md
! test -d .agents/skills/pyqt5-desktop-patterns
test -f .agents/skills/hydra-mcp-server/SKILL.md
test -f .agents/skills/hydra2dgpu-studio-ui/SKILL.md
test -f .agents/skills/hydra2dgpu-gpkg-schema-expert/SKILL.md
echo "PASS block 1"
```

Expected: PASS block 1. Each command must exit 0 individually. If any command exits non-zero, identify which subagent failed.

- [ ] **Step 3: Run shell block #2 (opencode.json instructions sanity)**

```bash
python3 -c "
import json, os, sys
cfg = json.load(open('.opencode/opencode.json'))
for p in cfg.get('instructions', []):
    if not os.path.exists(p):
        print(f'FAIL: phantom instruction {p}')
        sys.exit(1)
if not any('AGENTS.md' in p for p in cfg.get('instructions', [])):
    print('FAIL: AGENTS.md not in instructions')
    sys.exit(1)
print('PASS block 2')
"
```

Expected: `PASS block 2`.

- [ ] **Step 4: Run shell block #3 (always-on tests)**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v \
  tests.test_workbench_gui \
  tests.test_workbench_persistence \
  tests.test_workbench_controller 2>&1 | tee /tmp/test_output.log
echo "TEST_EXIT=$?"
```

Expected: `TEST_EXIT=0`. The test output log is preserved for later inspection.

If any test fails, identify whether the failure is pre-existing (in which case the orchestrator decides whether to fix) or introduced by this plan (in which case the offending task must be re-dispatched).

- [ ] **Step 5: Run shell block #4 (each new SKILL.md is loadable)**

```bash
for f in .agents/skills/{hydra-mcp-server,hydra2dgpu-studio-ui,hydra2dgpu-gpkg-schema-expert}/SKILL.md; do
  test -s "$f" || { echo "FAIL: $f empty"; exit 1; }
done
echo "PASS block 4"
```

Expected: `PASS block 4`.

- [ ] **Step 6: Run shell block #5 (variant agents mirror base body)**

```bash
diff <(cat .opencode/agents/python-pro.md) \
     <(cat .opencode/agents/python-pro_kimi-for-coding_k3.md | sed '/^model:/d') \
     | grep -q '^>' \
  && echo "WARN: variant differs from base beyond model: line" \
  || echo "PASS block 5"
```

Expected: `PASS block 5`. (A WARN is acceptable if a recent base change intentionally diverges, but it should be flagged to the user.)

- [ ] **Step 7: Run shell block #6 (per-skill smoke check)**

```bash
for skill_name in "hydra-mcp-server" "hydra2dgpu-studio-ui" "hydra2dgpu-gpkg-schema-expert"; do
  test -f ".agents/skills/$skill_name/SKILL.md" || { echo "FAIL: $skill_name not in any skill"; exit 1; }
done
echo "PASS block 6"
```

Expected: `PASS block 6`.

- [ ] **Step 8: Produce verification report**

```text
=== VERIFICATION REPORT ===
Block 1 (cleanup confirmation): PASS
Block 2 (opencode.json sanity): PASS
Block 3 (always-on tests): PASS or FAIL (with test_output.log path)
Block 4 (new skills loadable): PASS
Block 5 (variant mirrors): PASS or WARN
Block 6 (per-skill triggers): PASS

Total: N/6 blocks PASS
```

- [ ] **Step 9: Report DONE if all blocks PASS, DONE_WITH_CONCERNS if any WARN, BLOCKED if any FAIL**

This task does NOT commit. The orchestrator inspects the report and updates the plan's `progress:` block.

---

## Plan Progress Tracking

After each task completes, the orchestrator updates the frontmatter `progress:` block:

```yaml
progress:
  total: 18
  done: M            # increment after each task's verification passes
  current: "K.L task title"   # next pending task
  blockers: []       # append any unresolved blocker; clear when resolved
  last_updated: YYYY-MM-DD
```

Per `docs/specs/2026-07-25-docs-lifecycle-design.md` §6.1:
- `progress.done` is incremented ONLY after the task's verification has passed.
- `progress.current` is the next pending step's short title, or `null` when fully done.
- `progress.last_updated` is set every time the block changes.
- `progress.total` is fixed at 18 (the count of plan-task items at authorship).
- After every task, both the task checkbox (in the body above) and the frontmatter `progress:` block are updated in the same edit.

---

## Cross-Phase Blocking Summary

| Task | Blocks (downstream) |
|---|---|
| 0.1 | All Phase 1–5 tasks |
| 1.1, 1.2, 1.3 (skill writes) | 1.7 (cross-link target), Phase 3 (skill references), Phase 5 (verification) |
| 1.4 (opencode.json) | 2.1 (orphan delete), Phase 5 (verification) |
| 1.5, 1.6 (rule edits) | Phase 5 (verification) |
| 1.7 (MVP_ARCHITECTURE) | Phase 5 (verification) |
| 2.1 (orphan delete) | Phase 5 (verification) |
| 2.2 (pyqt5 delete) | Phase 3 (skill references), Phase 5 (verification) |
| Phase 3 (base agents) | 4.1 (sync), Phase 5 (verification) |
| 4.1 (variant sync) | Phase 5 (verification) |

Per spec §9.10: failures BLOCK but do not auto-revert. The orchestrator dispatches a targeted fix subagent.