# Python Style Audit Waves Handoff

This document defines a repeatable, agent-friendly workflow for continuing the Python docstring/type-hint audit work in staged waves.

## Purpose

Use this guide when resuming or continuing the Python quality audit focused on:

- Missing Google-style docstrings.
- Missing function return annotations.
- Missing parameter annotations (excluding self and cls).

Audit source of truth: `tools/python_style_audit.py`.

## Current Baseline

As of 2026-05-09 after Wave 1 updates:

- Repo-wide total issues: 2085
- `backwater_model.py`: 0
- `backwater_qt.py`: 255
- Top remaining file: `swe2d_workbench_qt.py` (303)

Always refresh this baseline before new edits.

## Required Commands

Run from repository root.

```bash
python3 -m py_compile backwater_model.py backwater_qt.py swe2d_workbench_qt.py
python3 tools/python_style_audit.py
```

For per-file counts:

```bash
python3 - <<'PY'
from pathlib import Path
from tools.python_style_audit import _audit_file
for f in ['backwater_model.py','backwater_qt.py','swe2d_workbench_qt.py']:
    issues = _audit_file(Path(f))
    print(f'{f}: {len(issues)}')
PY
```

## Wave Plan

## Wave 1 (Completed)

Scope:

- `backwater_model.py` and first tranche of `backwater_qt.py`.

Outcome:

- `backwater_model.py` brought to zero audit issues.
- `backwater_qt.py` reduced substantially but still has a large remaining backlog.

## Wave 2 (Next)

Primary target:

- `backwater_qt.py`

Goal:

- Reduce this file from 255 to below 150 issues in one concentrated pass.

Tactics:

- Add signatures + Google-style docstrings for public/helper methods in contiguous blocks.
- Prioritize methods that fan out to many call paths (UI state, geometry edits, persistence helpers).
- Keep behavior unchanged; this is documentation/typing only.

Acceptance criteria:

- File compiles.
- No new runtime behavior changes.
- Audit count reduced by at least 100 from current baseline.

## Wave 3 (After Wave 2)

Primary target:

- `swe2d_workbench_qt.py`

Goal:

- Reduce this file from 303 to below 200 issues.

Tactics:

- Work in focused slices (top-level helpers, persistence block, run-loop helpers, mesh/topology UI handlers).
- Add lightweight, accurate annotations where Qt types are known; use object/Any sparingly when dynamic interfaces require it.

Acceptance criteria:

- File compiles.
- No changes to solver logic or GUI behavior.
- Audit count reduced by at least 100 from current baseline.

## Editing Guardrails

- Prefer minimal, non-functional edits.
- Preserve existing naming and public APIs.
- Use Google-style docstrings.
- Keep new comments concise and only where needed.
- Avoid mass reformatting.

## Handoff Checklist (For Any Agent)

Before editing:

1. Run repo audit and record current counts.
2. Pick one file and one contiguous tranche of methods.
3. Confirm no unrelated unstaged changes are modified.

After editing:

1. Run `python3 -m py_compile` on touched files.
2. Run `python3 tools/python_style_audit.py`.
3. Capture:
   - New repo total.
   - New per-file totals for touched files.
   - Exact files changed.

## Reporting Template

Use this format in handoff notes or PR text:

- Wave: `<wave number>`
- Files touched: `<list>`
- Repo issues: `<before> -> <after>`
- File issues:
  - `<file A>: <before> -> <after>`
  - `<file B>: <before> -> <after>`
- Validation:
  - `python3 -m py_compile ...` (pass/fail)
  - `python3 tools/python_style_audit.py` (pass/fail)
- Remaining priority order:
  1. `<file>`
  2. `<file>`
  3. `<file>`
