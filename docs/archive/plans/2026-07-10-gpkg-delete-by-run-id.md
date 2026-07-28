---
type: plan
status: complete
created: 2026-07-10
completed: 2026-07-25
---

# GPKG Delete by Run ID Redesign — Implementation Plan

**Goal:** Replace single-select delete-by-run-ID with a two-step wizard: multi-select run IDs, then multi-select which table types to delete.

**Files:**
- `swe2d/workbench/services/gpkg_operations_service.py` — add `delete_run_partial()`
- `swe2d/workbench/dialogs/gpkg_explorer_dialog.py` — replace `_delete_by_run_id()`, add `_collect_run_ids()`, `_select_tables()`
- `tests/test_gpkg_operations.py` — add `TestDeleteRunPartial`

**Tasks:**
1. Add `delete_run_partial()` to service layer with tests
2. Replace dialog `_delete_by_run_id()` with two-step wizard UI
3. Run lint/typecheck, commit
