---
type: spec
status: complete
created: 2026-07-10
completed: 2026-07-25
---

# GPKG Explorer: Delete by Run ID Redesign

## Problem

The current "Delete by Run ID" flow only supports single run selection and always deletes all associated tables. Users need to:
1. Select multiple run IDs at once
2. Choose which table types to delete from (e.g. keep the mesh, only delete logs)

## Current State

- `gpkg_explorer_dialog.py:234-327` — `_delete_by_run_id()` shows a `QComboBox` (single select), previews affected tables, calls `delete_run()`
- `gpkg_operations_service.py:237-344` — `delete_run(gpkg_path, run_id)` drops legacy per-run tables, deletes from `swe2d_run_logs`, and deletes from 4 baked tables via `DELETE ... WHERE run_id=?`
- Baked schema tables: `swe2d_baked_mesh` (keyed by `mesh_name`, shared), `swe2d_baked_results`, `swe2d_baked_line_ts`, `swe2d_baked_line_profiles`, `swe2d_baked_coupling` (all keyed partially by `run_id`), `swe2d_simulation_configs` (no `run_id`)

## Design

### Flow

```
Click "Delete by Run ID"
  |
  v
Step 1: Select Run IDs dialog (multi-select QListWidget with checkboxes)
  |  - Lists all run_ids from swe2d_run_logs (or fallback table-name parsing)
  |  - Select All / Deselect All toggle
  |  - Next enabled when >= 1 selected
  v
Step 2: Select Tables dialog (multi-select QListWidget with checkboxes)
  |  - Dynamically computed from selected runs
  |  - Warnings on shared/global tables
  |  - All checked by default
  v
Confirmation prompt
  |
  v
delete_run_partial(gpkg_path, run_ids, table_kinds)
  |
  v
Refresh table list
```

### Step 1: Select Run IDs Dialog

- `QDialog` with `QListWidget` set to `QListWidget.SelectionMode.MultiSelection` or using `Qt.ItemIsUserCheckable` flags
- Each item: checkbox + run_id string + row count from `swe2d_run_logs` if available
- "Select All" / "Deselect All" buttons at top
- `QDialogButtonBox` with Next + Cancel
- Next disabled until ≥1 item checked

### Step 2: Select Tables Dialog

- `QDialog` with `QListWidget` using checkable items
- Table list built from the GPKG schema + selected run IDs:

| Table Kind | Table Name | Shown When | Warning |
|-----------|-----------|-----------|---------|
| run_log | `swe2d_run_logs` | table exists | — |
| baked_results | `swe2d_baked_results` | table exists | — |
| baked_line_ts | `swe2d_baked_line_ts` | table exists | — |
| baked_line_profiles | `swe2d_baked_line_profiles` | table exists | — |
| baked_coupling | `swe2d_baked_coupling` | table exists | — |
| baked_mesh | `swe2d_baked_mesh` | table exists | "Shared across runs — may orphan other results" |
| simulation_configs | `swe2d_simulation_configs` | table exists | "Contains ALL configs, not just selected runs" |
| legacy_tables | any `_*_{run_id}` tables for selected runs | any match | "N legacy per-run table(s) will be dropped" |

- All items checked by default
- Warnings shown as item foreground color (orange) or tooltip
- `QDialogButtonBox` with Delete + Cancel

### Service Layer

Add `delete_run_partial()` to `gpkg_operations_service.py`:

```python
def delete_run_partial(
    gpkg_path: str,
    run_ids: list[str],
    *,
    delete_run_logs: bool = True,
    delete_baked_results: bool = True,
    delete_baked_line_ts: bool = True,
    delete_baked_line_profiles: bool = True,
    delete_baked_coupling: bool = True,
    delete_baked_mesh: bool = False,
    delete_simulation_configs: bool = False,
    delete_legacy_tables: bool = True,
) -> list[str]:
```

Returns list of deleted table names for reporting.

Logic:
- Opens single connection, single transaction
- For each baked table (except mesh): `DELETE FROM {table} WHERE run_id IN ({placeholders})`
- For `swe2d_run_logs`: `DELETE FROM swe2d_run_logs WHERE run_id IN ({placeholders})`
- For `swe2d_baked_mesh`: `DELETE FROM swe2d_baked_mesh WHERE mesh_name NOT IN (SELECT DISTINCT mesh_name FROM swe2d_baked_results WHERE run_id IN (...))` — only removes orphaned meshes
- For `swe2d_simulation_configs`: `DELETE FROM swe2d_simulation_configs` — unfiltered, only if user opted in
- For legacy per-run tables: query `sqlite_master` for tables ending with `_{run_id}` for any selected run, DROP each + clean gpkg metadata + drop rtree sidecars
- Commit + VACUUM

### Dialog Code in `gpkg_explorer_dialog.py`

Replace `_delete_by_run_id()` with:

```python
def _delete_by_run_id(self):
    run_ids = self._collect_run_ids()       # Step 1
    if not run_ids:
        return
    table_kinds = self._select_tables(run_ids)  # Step 2
    if not table_kinds:
        return
    deleted = delete_run_partial(self._gpkg_path, run_ids, **table_kinds)
    self._log(f"Deleted {len(deleted)} table(s) for {len(run_ids)} run(s)")
    self.refresh_tables()
```

New helper methods:
- `_collect_run_ids() -> list[str]` — shows multi-select dialog, returns selected run_ids
- `_select_tables(run_ids) -> dict` — shows table selection dialog with warnings, returns kwargs for `delete_run_partial`

### Error Handling

- `delete_run_partial` wraps entire operation in try/except, rolls back on failure
- Dialog shows `QMessageBox.warning` on error with the exception message
- Individual table delete failures logged but don't abort the transaction (matching current `delete_run` behavior)

## Files to Modify

1. **`swe2d/workbench/services/gpkg_operations_service.py`** — add `delete_run_partial()`
2. **`swe2d/workbench/dialogs/gpkg_explorer_dialog.py`** — replace `_delete_by_run_id()`, add `_collect_run_ids()`, `_select_tables()`

## Testing

- Unit test `delete_run_partial` with baked tables (similar to existing `TestOrphanCleanupOnDeleteRun`)
- Unit test multi-run deletion
- Unit test selective table deletion (e.g. only run_logs, skip baked)
- Unit test legacy table cleanup in partial mode
