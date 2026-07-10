# Implementation Plan Revision 2: Consolidate the Three JSON Producers

| | |
|---|---|
| **Plan ID** | `SWE2D-RC-2026-07-10-r2` |
| **Status** | Revision (supersedes §1.2, §1.3, §4 of Revision 1) |
| **Owner** | SWE2D runtime + workbench teams |
| **Created** | 2026-07-10 |
| **Supersedes** | [Revision 1 §1.2, §1.3, §4](2026-07-10-unified-run-controller-revision-1.md) |
| **Target branch** | `feature/unified-run-controller` |
| **Companion reading** | [Revision 1](2026-07-10-unified-run-controller-revision-1.md), [base plan](2026-07-10-unified-run-controller.md) |

---

## 0. What changed

Revision 1 designed a new `swe2d_run_replays` table and proposed `RunContext.to_replay_json()` / `from_replay_json()` as the canonical JSON pair. But the **Batch Simulation Dialog** ([`workbench/dialogs/batch_simulation_dialog.py`](../../swe2d/workbench/dialogs/batch_simulation_dialog.py)) already does roughly half of this work in two places that aren't wired to the canonical path:

1. **"Snapshot Current Setup" button** — calls `_widget_params_to_run_params(widget_params)` (the **legacy CLI JSON shape** with `params.params.n_mann` etc.) and writes it into a table row.
2. **"From GPKG" button** — reads `swe2d_run_logs` (a different table from the proposed `swe2d_run_replays`) and creates rows from its `params` string.

Combined with the **already-existing** `swe2d_simulation_configs` widget-save table (from `RunController.on_save_simulation_config` in [the GUI run_controller.py](../../swe2d/workbench/controllers/run_controller.py)), we now have **three competing JSON producers and three competing GPKG tables**:

| Producer | Table | Shape | Purpose |
|----------|-------|-------|---------|
| `RunController.on_save_simulation_config` (existing GUI) | `swe2d_simulation_configs` | `widget_state` (typed by widget name) | User-initiated manual save |
| `BatchSimulationDialog._snapshot_current_setup` (existing GUI) | in-memory row only | `params.params.<key>` legacy CLI shape | Batch-row creation |
| `BatchSimulationDialog._import_from_gpkg` (existing GUI) | reads `swe2d_run_logs` | whatever was logged | Batch-row creation |
| `RunContext.to_replay_json` (Revision 1, NEW) | `swe2d_run_replays` (NEW) | full RunContext | Auto-saved per-run replay |

That's the duplicate-floating-JSON problem. Revision 2 collapses all four into **one producer / one consumer / one table.**

---

## 1. The three duplicates, mapped

### 1.1 Duplicate 1: widget-state save table (`swe2d_simulation_configs`)

**Location**: [`workbench/controllers/run_controller.py` `on_save_simulation_config`](../../swe2d/workbench/controllers/run_controller.py) (lines ~870–915).

**What it does**: User clicks "Save Config" → captures widget values via `collect_workbench_widget_state` → JSON-encodes → writes to `swe2d_simulation_configs`.

**Why it's a duplicate**: The same widget values are also captured by `RunContext.from_view()` into a `RunContext`, and `RunContext.to_replay_json()` produces a richer, more complete serialization. The `swe2d_simulation_configs` table is a strictly worse version of `swe2d_run_replays`.

**Why it can't simply be deleted**: It serves a different *intent* — user-initiated "save this config as a named recipe." Replay rows are auto-saved per run. Users may have many saved configs that aren't tied to a specific run.

**Resolution**: Keep the user-facing "Save Config" button, but **route the serialization through `RunContext.to_replay_json()`** and **write to the same `swe2d_run_replays` table** with a `kind` column distinguishing:

```
kind: 'auto'    — auto-saved per run (rows from RunController._finalize)
kind: 'manual'  — user-saved named config (rows from on_save_simulation_config)
```

### 1.2 Duplicate 2: legacy CLI JSON shape in `_widget_params_to_run_params`

**Location**: [`workbench/dialogs/batch_simulation_dialog.py` lines 32–157](../../swe2d/workbench/dialogs/batch_simulation_dialog.py).

**What it does**: Maps widget suffix names to CLI param names (`n_mann_spin` → `n_mann`, etc.) and produces a flat dict like `{"id": "...", "mesh": "...", "mesh_gpkg": "...", "params": {"n_mann": 0.035, ...}, "bc_lines": {...}}`.

**Why it's a duplicate**: This is **the old CLI params shape**, maintained here for compatibility with the existing `headless_runner.execute_run()`. Revision 1's `RunContext.to_replay_json()` produces a strictly richer version.

**Why it can't simply be deleted**: It's actively used by `BatchSimulationDialog._run_batch()` which passes these dicts to `BatchOrchestrator` which spawns CLI subprocesses.

**Resolution**: Replace `_widget_params_to_run_params` + the hand-rolled data-source assembly with a single call:

```python
# OLD (in _snapshot_current_setup):
widget_params = parent.collect_run_widget_params()
run_params = _widget_params_to_run_params(widget_params)  # legacy shape
entry = {"id": ..., "mesh": mesh_name, "mesh_gpkg": mesh_gpkg,
         "params": run_params, "bc_lines": bc_lines, ...}
self._add_row_from_entry(entry)

# NEW:
ctx = RunContext.from_view(parent, request=None)
ctx.run_id = "current_setup"  # override for the row id
entry = ctx.to_replay_json()  # canonical shape
self._add_row_from_entry(entry)
```

`_widget_params_to_run_params` and `_WIDGET_TO_CLI_MAP` are **deleted**.

### 1.3 Duplicate 3: GPKG import reads `swe2d_run_logs` instead of `swe2d_run_replays`

**Location**: [`workbench/dialogs/batch_simulation_dialog.py` `_import_from_gpkg`](../../swe2d/workbench/dialogs/batch_simulation_dialog.py) (lines ~480–530).

**What it does**: Reads `swe2d_run_logs` or `swe2d_baked_results` table and creates batch rows from the embedded `params` JSON string.

**Why it's a duplicate**: After Revision 1, every run has a corresponding `swe2d_run_replays` row with the canonical JSON. `swe2d_run_logs` is a different table for different content (run log text, not run inputs).

**Why it can't simply be deleted**: Old runs (pre-Revision-1) only have `swe2d_run_logs` entries, not `swe2d_run_replays`. Users with old result GPKGs still need to import them.

**Resolution**: `BatchSimulationDialog._import_from_gpkg` reads **`swe2d_run_replays` first**, falls back to `swe2d_run_logs` for legacy GPKGs with a logged deprecation notice.

---

## 2. The unified serialization contract

After Revision 2 there is **exactly one** function that produces run-serializable JSON:

```python
class RunContext:
    def to_replay_json(self) -> dict:
        """Canonical serialization of a run's inputs.

        Used by:
        - RunController._finalize() → auto-save per run
        - BatchSimulationDialog snapshot button → row creation
        - RunController.on_save_simulation_config() → user-named config
        - CLI replay file export
        """
```

And **exactly one** function that consumes it:

```python
class RunContext:
    @staticmethod
    def from_replay_json(payload: dict | str | Path) -> RunContext:
        """Build a RunContext from canonical JSON.

        Used by:
        - BatchSimulationDialog._import_from_gpkg() → row creation from GPKG
        - sse2d_cli replay command
        - RunController.on_load_simulation_config() → user-named config reload
        """
```

Both functions are **the** source of truth. Everything else (the three legacy producers above) is replaced by calls to these two.

---

## 3. The unified GPKG schema

### 3.1 The `swe2d_run_replays` table (was Revision 1 §4.1, extended)

```sql
CREATE TABLE IF NOT EXISTS swe2d_run_replays (
    run_id           TEXT PRIMARY KEY,
    mesh_name        TEXT,
    kind             TEXT NOT NULL DEFAULT 'auto',  -- 'auto' or 'manual'
    config_name      TEXT,                           -- only set for kind='manual'
    created_utc      TEXT NOT NULL,
    json_schema_ver  TEXT NOT NULL,                  -- 'swe2d-replay/1'
    replay_json      TEXT NOT NULL,                  -- canonical RunContext JSON
    cli_replay_cmd   TEXT NOT NULL                   -- copy-pasteable replay command
);

CREATE INDEX IF NOT EXISTS idx_replay_mesh ON swe2d_run_replays(mesh_name, created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_replay_kind ON swe2d_run_replays(kind, mesh_name);
```

### 3.2 Producers of this table

| Trigger | Source | `kind` | `config_name` |
|---------|--------|:------:|---------------|
| Run completion (auto) | `RunController._finalize()` | `auto` | NULL |
| User "Save Config" (GUI) | `RunController.on_save_simulation_config()` | `manual` | user-provided name |
| CLI replay export | `swe2d-cli replay --export` | `manual` | filename-derived |
| Batch row creation (in-memory only) | `BatchSimulationDialog._snapshot_current_setup` | (no row written) | n/a |

The "Snapshot Current Setup" button does **not** write to GPKG. It only produces an in-memory row. The row will be persisted as part of `swe2d_run_replays` if the user later runs that batch row through the CLI / GUI. (See §5.)

### 3.3 Consumers of this table

| Consumer | Use |
|----------|-----|
| `BatchSimulationDialog._import_from_gpkg` | Create batch rows from past runs / saved configs |
| `swe2d-cli replay --results-gpkg ...` | Replay an exact past run |
| `RunController.on_load_simulation_config` | Reapply a named config to widget state |
| GPKG Explorer dialog (future) | Browse replay history per mesh |

### 3.4 Migration: `swe2d_simulation_configs` → `swe2d_run_replays`

Old `swe2d_simulation_configs` rows have widget-state JSON in a different shape. Migration script:

```python
def migrate_simulation_configs_to_replays(gpkg_path: str) -> int:
    """One-shot migration: copy rows from swe2d_simulation_configs
    into swe2d_run_replays with kind='manual'."""
    n = 0
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        # Schema check
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_simulation_configs'"
        )
        if not cur.fetchone():
            return 0
        cur.execute(
            "SELECT config_id, mesh_name, created_utc, run_duration_s, description, widget_state "
            "FROM swe2d_simulation_configs"
        )
        for row in cur.fetchall():
            config_id, mesh_name, created, dur_s, desc, ws_str = row
            ws = json.loads(ws_str or "{}")
            # Convert widget_state into a replay payload. Best-effort: build
            # a partial RunContext from widgets and serialize.
            ctx = RunContext.from_widget_state_dict(ws, mesh_name, dur_s)
            payload = ctx.to_replay_json()
            cli_cmd = build_cli_replay_command(ctx, payload)
            cur.execute(
                "INSERT OR REPLACE INTO swe2d_run_replays "
                "(run_id, mesh_name, kind, config_name, created_utc, "
                " json_schema_ver, replay_json, cli_replay_cmd) "
                "VALUES (?, ?, 'manual', ?, ?, ?, ?, ?)",
                (
                    f"manual_{config_id}",
                    str(mesh_name or ""),
                    str(config_id),
                    str(created),
                    "swe2d-replay/1",
                    json.dumps(payload, default=str),
                    cli_cmd,
                ),
            )
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n
```

Old `swe2d_simulation_configs` table is **left in place** (for one release), with a deprecation warning emitted on read.

---

## 4. How each existing call-site changes

### 4.1 `RunController.on_save_simulation_config` (GUI)

**Before**: Calls `collect_workbench_widget_state` → writes to `swe2d_simulation_configs`.

**After**: Calls `RunContext.from_view(view, request=None)` → `ctx.to_replay_json()` → writes to `swe2d_run_replays` with `kind='manual'`. The old `swe2d_simulation_configs` write is removed.

```python
# NEW (replaces ~50 lines in on_save_simulation_config):
def on_save_simulation_config(self):
    view = self._view
    ctx = RunContext.from_view(view, request=None)
    config_name, ok = view.get_input_text("Save Config", "Configuration name:", "")
    if not ok:
        return
    if not config_name:
        config_name = datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z")
    ctx.run_id = f"manual_{config_name}"

    gpkg = self._prompt_gpkg_path()  # unchanged dialog logic
    payload = ctx.to_replay_json()
    cli_cmd = build_cli_replay_command(ctx, payload)
    persist_run_replay(
        gpkg_path=gpkg,
        run_id=ctx.run_id,
        mesh_name=ctx.mesh_name,
        kind="manual",
        config_name=config_name,
        replay_payload=payload,
        cli_replay_cmd=cli_cmd,
        log_fn=view._log,
    )
```

### 4.2 `RunController.on_load_simulation_config` (GUI)

**Before**: Reads `swe2d_simulation_configs`, builds dialog, applies to widget state via `view._apply_run_log_metadata_to_ui`.

**After**: Reads `swe2d_run_replays` (with fallback to `swe2d_simulation_configs` for legacy rows). Uses `RunContext.from_replay_json` → `ctx.apply_to_view(view)` (a new method that maps ctx fields back to widget state via the inverse of `collect_run_widget_params`).

```python
def on_load_simulation_config(self):
    view = self._view
    gpkg = self._prompt_gpkg_path()
    replays = load_run_replays(gpkg, log_fn=view._log)  # NEW unified loader
    if not replays:
        # Legacy fallback
        replays = load_simulation_configs(gpkg, log_fn=view._log)
        if replays:
            view._log("[DEPRECATION] Reading from swe2d_simulation_configs; "
                      "re-save configs to migrate.")
    if not replays:
        return

    dlg = SWE2DSimulationConfigDialog(
        configs=replays,
        db_path=gpkg,
        parent=view,
        apply_callback=lambda ctx: ctx.apply_to_view(view),
    )
    dlg.exec()
```

### 4.3 `BatchSimulationDialog._snapshot_current_setup`

**Before**: Calls `parent.collect_run_widget_params()` → `_widget_params_to_run_params(widget_params)` → manually assembles `entry` dict with `params`, `bc_lines`, `hyetograph`, `rain_cn`, `drainage`, `structures`, `sample_lines`, `mesh`, `mesh_gpkg` keys.

**After**: Calls `RunContext.from_view(parent, request=None)` → `ctx.to_replay_json()` → adds as row.

```python
def _snapshot_current_setup(self):
    parent = self.parent()
    if parent is None:
        return

    try:
        ctx = RunContext.from_view(parent, request=None)
    except Exception as exc:
        QtWidgets.QMessageBox.warning(self, "Snapshot Error", f"Failed: {exc}")
        return

    ctx.run_id = "current_setup"
    payload = ctx.to_replay_json()
    self._add_row_from_entry(payload)

    log = getattr(parent, "_log", None)
    if log:
        log("batch> snapshot added row from current setup")
```

**Net deletion**: The entire `_WIDGET_TO_CLI_MAP`, `_BOOL_WIDGET_TO_CLI_MAP`, `_COMBO_WIDGET_TO_CLI_MAP`, `_widget_params_to_run_params`, the manual `mtab` data-source capture block (~100 lines), and `_dict_with_gpkg` helper. **~150 lines deleted** from this dialog.

### 4.4 `BatchSimulationDialog._import_from_gpkg`

**Before**: Reads `swe2d_run_logs.params` column or `swe2d_baked_results.mesh_name`.

**After**: Reads `swe2d_run_replays.replay_json` column. Falls back to legacy tables for old GPKGs.

```python
def _import_from_gpkg(self):
    gpkg = self._gpkg_path()
    if not gpkg:
        return

    # Prefer the canonical replay table
    rows = load_run_replays(gpkg, log_fn=self._parent_log())  # NEW
    if not rows:
        # Legacy fallback: old run_logs / baked_results tables
        view = self._log_parent_warning()
        rows = self._query_runs_from_gpkg(gpkg)  # existing legacy reader
        if rows:
            QtWidgets.QMessageBox.information(
                self, "Import (legacy mode)",
                "GPKG uses legacy format; reading swe2d_run_logs. "
                "Re-run via current version to migrate to swe2d_run_replays."
            )

    for row in rows:
        payload = json.loads(row["replay_json"])
        payload["id"] = row["run_id"]
        self._add_row_from_entry(payload)

    QtWidgets.QMessageBox.information(
        self, "Import Complete",
        f"Imported {len(rows)} replay(s) from GPKG."
    )
```

### 4.5 `BatchSimulationDialog._run_batch` (subprocess invocation)

**Before**: Writes each row's JSON dict to a temp file → spawns `swe2d-cli ... --params <file>` subprocess.

**After**: Same flow, but the JSON written to the temp file is now the **canonical `replay_json` shape** (because rows are built via `to_replay_json`). The CLI subprocess reads it via `RunContext.from_replay_json` (per Revision 1).

**What changes**: Nothing — the subprocess invocation is shape-agnostic. It just reads whatever JSON file is given. The fact that the shape is now canonical is invisible to this code.

---

## 5. Phase plan additions (extends Revision 1 §5)

### Phase 8 — Consolidate JSON producers (3 days)

| Day | Task |
|----:|------|
| 1 | Implement `RunContext.from_view`, `RunContext.to_replay_json`, `RunContext.from_replay_json`, `RunContext.apply_to_view` in `runtime/run_context.py` |
| 1 | Implement `persist_run_replay`, `load_run_replays` in `services/gpkg_persistence_service.py` |
| 1 | Implement `RunContext.from_widget_state_dict` (legacy compat) |
| 2 | Refactor `BatchSimulationDialog._snapshot_current_setup` to use `from_view` / `to_replay_json` |
| 2 | Refactor `BatchSimulationDialog._import_from_gpkg` to read `swe2d_run_replays` first |
| 2 | Refactor `RunController.on_save_simulation_config` to write `swe2d_run_replays` |
| 2 | Refactor `RunController.on_load_simulation_config` to read `swe2d_run_replays` |
| 3 | Delete `_widget_params_to_run_params` + the three `*_WIDGET_TO_CLI_MAP` dicts |
| 3 | Add `cli/batch_runner.py` legacy-shape compat shim (one release deprecation) |
| 3 | Write `tests/test_replay_consolidation.py` (see §6) |

### Phase 9 — Migration tooling (1 day)

| Day | Task |
|----:|------|
| 1 | `swe2d.cli.migrate_replays.migrate_simulation_configs_to_replays()` — one-shot script |
| 1 | Auto-run migration on plugin upgrade (in plugin's `initGui`) |
| 1 | Deprecation warning when `swe2d_simulation_configs` is read |

---

## 6. Test additions (extends Revision 1 §7)

| Test | What | Acceptance |
|------|------|------------|
| `test_replay_consolidation_single_producer.py` | Monkey-patch `RunContext.to_replay_json`; confirm BatchDialog snapshot, save-config, and CLI replay all call it | Exactly one caller; the legacy producers are gone |
| `test_batch_snapshot_uses_canonical_json.py` | Snapshot button produces a row that round-trips through `RunContext.from_replay_json` to a valid context | Round-trip OK |
| `test_batch_from_gpkg_reads_replays.py` | Mock GPKG with `swe2d_run_replays` rows; confirm `_import_from_gpkg` reads them, not `swe2d_run_logs` | Reads from `swe2d_run_replays` |
| `test_legacy_simulation_configs_still_loads.py` | Mock GPKG with old `swe2d_simulation_configs` table; confirm fallback works + deprecation warning fires | Loads with warning |
| `test_save_config_writes_to_replays_table.py` | Save Config button writes to `swe2d_run_replays` with `kind='manual'`, not to `swe2d_simulation_configs` | New row in `swe2d_run_replays` |
| `test_load_config_reads_from_replays.py` | Load Config reads `swe2d_run_replays` and applies to widgets via `RunContext.apply_to_view` | Widgets reflect loaded state |

---

## 7. Revised acceptance criteria (extends Revision 1 §8)

- [ ] `grep -r '_widget_params_to_run_params' swe2d/` returns nothing.
- [ ] `grep -r '_WIDGET_TO_CLI_MAP' swe2d/` returns nothing.
- [ ] `grep -r 'swe2d_simulation_configs' swe2d/` returns only migration + deprecation-warning code.
- [ ] `grep -r 'swe2d_run_logs' swe2d/workbench` returns only the legacy fallback reader in `BatchSimulationDialog._import_from_gpkg`.
- [ ] All `RunContext`-related JSON goes through `to_replay_json` / `from_replay_json` (single producer pair).
- [ ] All run-config GPKG rows live in `swe2d_run_replays`.
- [ ] Round-trip test from Revision 1 still passes (no regression).
- [ ] GUI regression test from Revision 1 still passes (no behavior change).
- [ ] Migration script converts old `swe2d_simulation_configs` rows without loss.
- [ ] CHANGELOG notes deprecation of `swe2d_simulation_configs` (one release).

---

## 8. Migration user-impact summary

| Action user does today | What happens after Revision 2 |
|------------------------|-------------------------------|
| Click "Save Config" in GUI | Writes to `swe2d_run_replays` (kind=manual). Old `swe2d_simulation_configs` rows still load with a deprecation notice. |
| Click "Load Config" in GUI | Reads from `swe2d_run_replays` first; falls back to `swe2d_simulation_configs` with warning. |
| Click "Snapshot Current Setup" in batch dialog | Same UI; row now stores canonical JSON (vs. legacy CLI shape). |
| Click "From GPKG" in batch dialog | Reads from `swe2d_run_replays` first; falls back to `swe2d_run_logs` with notice. |
| Run `swe2d-cli replay --results-gpkg ...` | Reads from `swe2d_run_replays`. Works for all runs since Revision 1 merge. |
| Run `swe2d-cli replay --legacy-params-file <old.json>` | One-release compat shim translates old CLI shape → RunContext. Deprecation warning. |
| Auto-saved replay on every GUI run | Same as Revision 1 — auto-written to `swe2d_run_replays` (kind=auto). |

---

## 9. File-level deltas vs Revision 1

| File | Change |
|------|--------|
| `swe2d/runtime/run_context.py` | Add `to_replay_json`, `from_replay_json`, `apply_to_view`, `from_widget_state_dict`. |
| `swe2d/services/gpkg_persistence_service.py` | Add `persist_run_replay`, `load_run_replays`. Keep `persist_simulation_config` for one release. |
| `swe2d/workbench/dialogs/batch_simulation_dialog.py` | Delete `_WIDGET_TO_CLI_MAP`, `_BOOL_WIDGET_TO_CLI_MAP`, `_COMBO_WIDGET_TO_CLI_MAP`, `_widget_params_to_run_params`, the manual data-source assembly block, `_dict_with_gpkg`. Replace `_snapshot_current_setup` body with one call. Update `_import_from_gpkg` to prefer `swe2d_run_replays`. **Net: ~200 lines deleted.** |
| `swe2d/workbench/controllers/run_controller.py` | Replace `on_save_simulation_config` and `on_load_simulation_config` bodies to use canonical path. |
| `swe2d/cli/batch_runner.py` | Add legacy-shape compat shim (`_legacy_params_to_replay`). One-release deprecation. |
| `swe2d/cli/migrate_replays.py` (NEW) | One-shot migration script. |
| `swe2d/cli/__init__.py` | Wire `migrate_replays` subcommand. |
| `__init__.py` (plugin root) | Auto-run migration on `initGui` upgrade path. |

---

## 10. Open questions (revised)

1. **`kind='manual'` rows reuse `run_id` slot — could collide with future auto-saved runs.** Recommendation: prefix manual IDs with `manual_` (already in §3.2).
2. **`apply_to_view` requires an inverse of `collect_run_widget_params`.** The current mapping is partial (the GUI's `apply_run_log_metadata_to_ui` exists but is incomplete). Recommendation: in Phase 8, audit which widget keys are settable, document the gaps, and add the missing setters.
3. **Batch rows from `_snapshot_current_setup` are not persisted to GPKG — they live only in the table.** If the user closes the dialog before running the batch, the row is lost. Recommendation: keep current behavior (in-memory only); user can re-snapshot if needed.
4. **Does `swe2d-cli replay --export` need to exist?** Recommendation: yes — it produces a standalone JSON file from a `swe2d_run_replays` row, for sharing with collaborators who don't have the source GPKG.

---

## 11. Summary of the consolidation

After Revision 2:

| Concern | Single source |
|---------|---------------|
| Capture widget → JSON | `RunContext.to_replay_json` (used by RunController, BatchDialog snapshot, save-config) |
| Restore JSON → widgets | `RunContext.from_replay_json` + `RunContext.apply_to_view` (used by load-config, CLI replay, batch import) |
| Persist runnable config | `swe2d_run_replays` table (one row per run OR per named config) |
| Transport to subprocess | Canonical JSON file written by `to_replay_json`, read by `from_replay_json` |
| Translation tables for widget names | **Deleted.** All widget↔RunContext mapping lives in `from_view` / `apply_to_view`. |

**Three JSON producers → one. Three GPKG tables → one (with `kind` discriminator). Three translation tables → one method pair.**

Net code deletion: ~250 lines from `BatchSimulationDialog`, ~50 lines from `RunController.on_save/load_simulation_config`, plus the now-unneeded legacy compat tables. Net code addition: ~150 lines for the canonical `RunContext` methods + replay persistence helpers. **Net reduction: ~150 lines**, with strictly stronger guarantees (one round-trip test covers everything).