# Workbench Session Persistence — Plan

## Goal

Make the HYDRA2DGPU workbench (dock layout + open/closed state) survive
across QGIS sessions. Today, on every QGIS launch the user must
re-open the workbench and re-arrange every dock panel.

## Approach

Use Qt's built-in `QMainWindow.saveState()` / `restoreState()` and
`saveGeometry()` / `restoreGeometry()` on `iface.mainWindow()`. Persist
the resulting bytes to `QSettings("HYDRA2DGPU", "HYDRA2DGPU")`.

Qt's `QMainWindow.saveState()` already serializes every `QDockWidget`
attached to the window by `objectName()`. Each dock in
`WorkbenchDialogBuilder._build_component` is named
`HYDRA2D{name.title()}Dock`, which is stable.

## Keys stored in QSettings

| Key | Type | Purpose |
|---|---|---|
| `workbench/open_on_startup` | bool | Auto-open the workbench when QGIS launches |
| `workbench/was_open` | bool | Whether the workbench was open at last unload |
| `workbench/dock_state` | QByteArray | `mainWindow.saveState()` (positions, sizes, float, tab stacks) |
| `workbench/geometry` | QByteArray | `mainWindow.saveGeometry()` (window size/position) |

## Files to modify

### 1. `swe2d/workbench/persistence.py` (new, ~120 lines)

Pure helper module encapsulating persistence — separately testable.

```python
def load_open_on_startup(settings) -> bool: ...
def save_open_on_startup(settings, value: bool) -> None: ...
def was_open(settings) -> bool | None: ...
def save_was_open(settings, value: bool) -> None: ...
def save_window_state(settings, main_window) -> None: ...
def restore_window_state(settings, main_window) -> bool: ...
def clear_window_state(settings) -> None: ...
```

### 2. `swe2d/workbench/views/studio_host_methods.py` (+~30 lines)

In `launch_swe2d_workbench_studio`:
- After all docks are built and added to `iface.mainWindow()`,
  call `restore_window_state(settings, iface.mainWindow())`.
- Save `was_open = True` immediately.

In `close_workbench_studio`:
- Save `was_open = False` and `save_window_state(...)` BEFORE tearing
  down the dialogs (so the layout captures the user's arrangement).

### 3. `swe2d/workbench/workbench_dialog_builder.py` (+~10 lines)

In `_build_component`, after `iface.addDockWidget(...)`:
- Hook a one-shot `QTimer.singleShot(0, ...)` to defer
  `restore_window_state` until all docks have been created (restoring
  mid-build would silently no-op the docks that haven't been added
  yet).

### 4. `hydra_plugin.py` (+~40 lines)

In `__init__`: open a `QSettings` once and store on `self`.

In `initGui`:
- Read `open_on_startup` and `was_open` from QSettings.
- If both true and workbench not already open, schedule `self.run()`
  via `QTimer.singleShot(0, self.run)`. The deferred call lets
  QGIS finish laying out its own panels first.

In `unload`:
- Save window state explicitly (covers the case where user
  closes QGIS without closing the workbench first).

In `HYDRASettingsDialog`:
- Add "Open Workbench on QGIS startup" checkbox bound to
  `open_on_startup`.
- Add "Reset Workbench Layout" button that calls
  `clear_window_state` — restores default QGIS dock arrangement
  on next launch.

## Test plan

### `tests/test_workbench_persistence.py` (rewrite, ~150 lines)

Replace the diagnostic-only script with real `pytest` tests:

1. `test_open_on_startup_defaults_false`
2. `test_save_and_load_open_on_startup`
3. `test_was_open_defaults_none`
4. `test_save_and_load_was_open`
5. `test_save_window_state_is_qbytearray`
6. `test_round_trip_window_state_restores_dock_layout`
   - Create a real `QMainWindow`, add 2 named `QDockWidget`s with
     known positions, save state, create a new `QMainWindow`, add
     the same docks, restore state, assert positions match.
7. `test_clear_window_state_removes_keys`
8. `test_initGui_auto_opens_when_open_on_startup_and_was_open`
   - Use a stub plugin + stub iface and verify the right method
     is called.

Tests must run under `QgsApplication`-less env (no QGIS required
for the helper-level tests; full initGui test may run headless
against `QApplication([])`).

## Failure modes to handle

- **`saveState()` returns empty bytes when no docks have `objectName`** —
  Verify every `_build_component` calls `setObjectName()`. (Yes,
  already done.) Wrap calls in try/except so a corrupted blob
  doesn't brick a session — log and ignore.
- **Stale dock objectNames after rename** — `restoreState` ignores
  unknown widget names. If `_build_overlay_page` rename from
  earlier had a dock with a different name, restore silently
  drops that dock. Verify by listing `restoreDockWidget` return
  values.
- **Multi-monitor layouts** — `restoreState` works fine across
  monitor configurations; geometry may position the window
  off-screen if the original monitor is gone. Fall back to
  `restoreGeometry(DOCK_STATE_ONLY)` only if we detect off-screen.

## Commit cadence

1. `Add persistence helper module` — `swe2d/workbench/persistence.py` + tests
2. `Wire save/restore into workbench launch/close` — `studio_host_methods.py`
3. `Restore dock layout from QSettings on launch` — `workbench_dialog_builder.py`
4. `Auto-open workbench on QGIS startup if previously open` — `hydra_plugin.py` initGui/unload
5. `Settings dialog: open-on-startup + reset layout` — `HYDRASettingsDialog`
6. (Separate) `Cleanup persistence diagnostic test` — rewrite `tests/test_workbench_persistence.py`
