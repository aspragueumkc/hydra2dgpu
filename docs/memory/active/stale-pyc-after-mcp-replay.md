---
type: memory
status: active
created: 2026-07-26
topic: stale-pyc-after-mcp-replay
tags: [mcp, infra, lesson, decision, hazard]
evidence: tools/hydra_mcp/server.py:151-164; swe2d/services/mesh_persistence_service.py:1-100; .opencode/rules/CACHE_DISCIPLINE.md:1
related:
  - .opencode/rules/CACHE_DISCIPLINE.md
  - qgis_plugin/HYDRA2DGPU/__init__.py
  - swe2d/services/mesh_persistence_service.py
---

# Stale .pyc after MCP replay

## Context

On 2026-07-26, an MCP replay of `mcp_run_test.json` (run id
`swe2d_20260726T143634-0500`) was launched via `python -m swe2d.cli replay
--replay-file <tmp.json>` in the dev working tree. Later, a QGIS session
that had been running since before the replay failed when the user clicked
*Mesh → Import/Export → Load Mesh from GPKG* with the dialog message:

```
Could not read MESH1.gpkg: No module named 'swe2d.services.mesh_persistence_service'
```

The on-disk source `swe2d/services/mesh_persistence_service.py` existed
and was up to date with HEAD. A direct `from swe2d.services.mesh_persistence_service
import …` from a fresh interpreter succeeded. Purging
`swe2d/services/__pycache__/mesh_persistence_service.cpython-312.pyc`
fixed the dialog.

## Root cause

The MCP replay was the *first* process to import the post-merge module
graph from a Python interpreter that was also writing `__pycache__/`
into the dev tree. Python 3.12's bytecode cache validation uses source
mtime vs `.pyc` mtime. The replay subprocess's imports ran *after* the
QGIS-loaded source's mtime (00:52) but wrote a fresh `.pyc` (15:20),
producing a state where the on-disk bytecode referenced a module-set
inconsistent with the QGIS-process in-memory state. The next import in
the QGIS process — triggered by the button click — picked up the stale
bytecode and raised a generic `ModuleNotFoundError` whose real cause
(the .pyc being out of sync with the source) was hidden by the dialog
swallowing the traceback.

Note: the MCP subprocess does NOT mutate the QGIS session directly.
The interaction is purely through the shared dev tree's filesystem
and Python's `__pycache__` directory. The QGIS plugin's symlink chain
(`…/qgis_plugins/hydra2dgpu → private-repo-hydra2dgpu/qgis_plugin/HYDRA2DGPU`)
plus `_plugin_dir = realpath(__file__)` and the `…/…/…` dirname chain
in `qgis_plugin/HYDRA2DGPU/__init__.py` correctly add the repo root to
`sys.path` — import resolution is fine when caches are clean.

## Decision

- **Treat this as a one-off.** The merge from `feature/…` into
  `public-sanitize` did not make code changes in the strict sense, but
  the agent that performed the merge should have purged `__pycache__/`
  per `.opencode/rules/CACHE_DISCIPLINE.md` (the rule applies to any
  structural change to a Python module: signature changes, new return
  values, new classes, changed imports). Merge = structural change to
  the loaded module set. **Hold off** on adding
  `importlib.invalidate_caches()` or a Makefile target that auto-purges
  — the user wants to wait before introducing any always-on mitigation
  because the failure mode was rare and a single manual `find . -name
  '*.pyc' -delete` recovered the session.
- **Fix the silent-traceback regression that hid the root cause.** The
  `QMessageBox.critical(self, "Load Mesh Error", str(exc))` calls in
  `swe2d/workbench/studio_dialog.py::_load_mesh_from_gpkg` (line 739)
  and `::_save_mesh_to_gpkg` (line 685) only show the exception
  message. Replaced with: capture `traceback.format_exc()`, push the
  full trace into `self._log(...)`, and show a dialog with
  `{type(exc).__name__}: {exc}` plus the line *“See the runtime log
  for the full traceback.”*. Done 2026-07-26; review-pending still
  applies to the other ~10 dialog handlers in the same file (lines
  135, 256, 288, 309, 325, 342, 351, 360, 372, 884, 895, 916, 955,
  1026) that follow the same `str(exc)` pattern.

## What NOT to do (yet)

- Do **not** add `importlib.invalidate_caches()` to
  `qgis_plugin/HYDRA2DGPU/__init__.py` (the eager `_import_all`).
  Discussed on 2026-07-26; user wants to wait.
- Do **not** add a pre-commit hook that purges `__pycache__/`. Same
  reason.
- Do **not** add a Makefile target that purges on `make dev`. Same
  reason.

## Recovery procedure when this happens

1. Stop the MCP server (or wait for it to exit; it doesn't hold a
   lock on the .pyc files).
2. `cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu && find .
   -name __pycache__ -type d -exec rm -rf {} +` (or
   `find . -name '*.pyc' -delete`).
3. Restart QGIS so it re-imports from the now-clean source.
4. Re-run the action that failed; the dialog will now show the
   `{type(exc).__name__}: {exc}` line and refer you to the runtime
   log for the full traceback.

## Open questions

- Should the cache-purge rule in `.opencode/rules/CACHE_DISCIPLINE.md`
  be tightened to also cover the *merge* case explicitly (the current
  wording focuses on "structural change to a Python module: signature
  changes, new return values, new classes, changed imports" — a
  merge of feature branches may not be a code change in the strict
  sense but still changes which modules are loaded by which process)?
  Pending user decision.
- Should the same traceback-into-log fix be propagated to the
  remaining ~10 dialog handlers in `studio_dialog.py`? Pending user
  decision.
