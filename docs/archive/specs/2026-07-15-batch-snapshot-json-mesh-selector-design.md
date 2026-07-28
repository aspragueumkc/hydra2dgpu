---
type: spec
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# Batch Simulation Snapshot / Mesh Selector Redesign

**Date:** 2026-07-15

## Goals

1. Make the Batch Simulation dialog's *Snapshot Current Setup* produce the same `swe2d-replay/1` JSON payload that the main dialog's *Export Config to JSON* produces.
2. Remove the GeoPackage path line edit from the Batch Simulation dialog.
3. Replace the two-widget GPKG path + mesh combo selector with a single read-only display + `Select Mesh...` button that opens a file picker followed by a GeoPackage mesh layer picker.
4. Ensure all data-source layer references in the snapshot (and export) JSON carry the full GeoPackage path, matching the CLI replay format.

## Background

The Batch Simulation dialog (`swe2d/workbench/dialogs/batch_simulation_dialog.py`) currently builds its snapshot JSON inline, duplicating logic that already exists in `RunController._build_replay_payload`. It also exposes a manual GeoPackage path line edit and a mesh combo populated from that GPKG. Users want a simpler, single-button mesh selector and parity with the main dialog's JSON export.

## Design

### 1. Snapshot payload — controller delegation

The snapshot logic must reuse the same payload builder as the main JSON export.

- Make `RunController._build_replay_payload` callable via a thin public wrapper `build_replay_payload(...)` with the same signature.
- Add `SWE2DStudioDialog.build_replay_payload(...)` that delegates to `self._controller.build_replay_payload(...)`.
- In `BatchSimulationDialog._snapshot_current_setup`, replace the manual `widget_state_to_flat_params` / `entry` construction with a call to `parent.build_replay_payload(...)`.
- The batch dialog passes the mesh name and GPKG path selected by the new mesh selector.

This keeps the JSON format in one place and respects the MVP boundary: the View asks the Controller for the payload, rather than duplicating the builder in the View.

### 2. Full layer paths in data sources

`SWE2DStudioDialog.collect_data_source_config()` currently only stores a `gpkg` key when the layer's source GPKG differs from the model GPKG. The CLI replay JSON should always include full paths so batch runs can resolve layers regardless of where the model GPKG is.

- Update `_dict_with_gpkg` in `collect_data_source_config` to always include `gpkg` when a non-empty path is available.
- This affects both *Export Config to JSON* and *Snapshot Current Setup* because both ultimately read `_data_sources` from the widget state.

### 3. UI replacement of GPKG path + mesh combo

In `BatchSimulationDialog`:

- **Remove:** the GeoPackage row (label, `QLineEdit`, Browse, Clear) and the Mesh row (`QComboBox`, Refresh, Apply to Selected).
- **Add:** a single top row containing:
  - `Mesh:` label
  - Read-only `QLineEdit` showing the selected mesh name and GPKG file name
  - `Select Mesh...` button
  - `Apply to All` button (moved from the data toolbar)
- **Keep:** `Apply to Selected` in the data toolbar.
- On dialog open, auto-populate the selector from the parent view's `_model_gpkg_path` and `_mesh_data["mesh_name"]` if they are available.

### 4. New lightweight mesh picker

Create `swe2d/workbench/dialogs/mesh_picker_dialog.py`:

- Input: a GeoPackage path.
- Query `swe2d_baked_mesh` for distinct `mesh_name` values, ordered by `created_utc DESC`.
- Present the names in a simple `QListWidget` or `QTableWidget`.
- Output: the selected `(mesh_name, gpkg_path)` tuple, or rejection if the user cancels.
- Reuse existing services where possible (e.g. `swe2d.workbench.services.gpkg_operations_service.list_tables` for validation, but direct sqlite is acceptable for the mesh list query).

The flow when the user clicks `Select Mesh...`:
1. `QFileDialog.getOpenFileName` for `*.gpkg`.
2. If a file is chosen, open `MeshPickerDialog` with that GPKG.
3. If a mesh is selected, update the read-only display and the dialog's internal `self._mesh_gpkg` / `self._mesh_name` state.

### 5. Files modified

- `swe2d/workbench/dialogs/batch_simulation_dialog.py` — UI rewrite, snapshot payload simplification.
- `swe2d/workbench/controllers/run_controller.py` — public `build_replay_payload` wrapper.
- `swe2d/workbench/studio_dialog.py` — delegate method and `_dict_with_gpkg` change.
- `swe2d/workbench/dialogs/mesh_picker_dialog.py` — new mesh picker dialog.

### 6. Error handling

- If the parent dialog does not expose `build_replay_payload`, show a warning and fall back gracefully (the dialog should still open).
- If mesh selection fails (file not found, no `swe2d_baked_mesh` table), show a `QMessageBox` and leave the previous selection unchanged.
- If `collect_widget_state_for_save` fails, keep the existing snapshot warning behavior.

### 7. Testing

- Run the existing batch dialog tests in `tests/test_results_path_wiring.py`.
- Verify that the snapshot JSON and a JSON exported from the main dialog have the same top-level keys (`schema_version`, `run_id`, `mesh`, `params`, `data_sources`, `results`, `units`, `widget_state`, `run_duration_s`).
- Verify that `data_sources` entries contain a `gpkg` key with the full path for layers loaded from an external GPKG and for layers loaded from the model GPKG.

## Open questions resolved

- Read-only display + Select Mesh button is the desired pattern.
- Keep both `Apply to All` and `Apply to Selected` buttons.
- Auto-populate the mesh selector from the currently loaded model on dialog open.

## Approach chosen

Approach 2 — Controller delegation. It avoids duplicating the replay-payload format, respects the MVP architecture, and keeps the UI changes localized to the batch dialog.
