# QGIS Backwater Plugin

Steady-flow 1D backwater modeling plugin for QGIS with GeoPackage-native model storage, culvert support, and integrated run/result tools.

## Current State

- Plugin-first architecture.
- Core solver is in `backwater_model.py` (GUI-free, CLI-capable).
- QGIS dock widget and UI workflows are in `backwater_qt.py`.
- QGIS menu integration is in `backwater_plugin.py`.
- Model I/O is GeoPackage-only in the plugin workflow.

## Key Features

- Create, load, save, and run backwater models directly from QGIS.
- Built-in main menu entries under **Backwater** for common actions:
  - Open Backwater Panel
  - Create New Model GeoPackage...
  - Load Model GeoPackage...
  - Save Model GeoPackage As...
  - Run Model
  - Open Results Plot
  - Open Results Table
  - Enable/Disable Layer Editing
  - Save Layer Edits
- New **Options** submenu in the Backwater menu:
  - Solver: `py` or `scipy`
  - Alpha Method: `conveyance` or `area`
- Downstream boundary controls in UI (`known_wse`, `normal_depth`) plus DS value and flow inputs.
- Result persistence to `model_results` and reload on model open.
- Culvert fields integrated into cross-section schema and forms.
- Matplotlib plotting support in plugin context with local runtime detection.

## GeoPackage Model Layers

The plugin expects these core layers:

- `cross_sections`
- `centerline` (required)
- `boundary_conditions`

The plugin also writes/reads:

- `model_results`

## Editing Workflow

- Enable editing with **Backwater > Enable/Disable Layer Editing**.
- Edit model data through QGIS attribute forms for loaded layers.
- Save edits with **Backwater > Save Layer Edits** before running.
- If unsaved layer edits exist, model run is blocked until edits are saved.

Form behavior includes:

- Cross-section and boundary-condition custom forms from `forms/`.
- Cross-section actions for terrain selection and Z updates.
- Conditional culvert field visibility based on `culvert_code`.

## Repository Layout

- `backwater_model.py`: hydraulic solver, GeoPackage I/O, CLI entry logic.
- `backwater_qt.py`: dock widget, run workflow, plotting/table UI.
- `backwater_plugin.py`: QGIS main menu and action wiring.
- `culvert_routine.py`: culvert hydraulics helpers.
- `forms/`: Qt Designer forms and form-init hooks.
- `expressions/`: QGIS expression helpers.
- `tests/`: solver and integration-oriented tests.
- `docs/`: design notes and hydraulic reference material.

## Development Notes

- In this environment, use `python3` for checks.
- Example syntax check:

```bash
python3 -m py_compile backwater_model.py backwater_qt.py backwater_plugin.py
```

## User Documentation

See `USER_GUIDE.md` for step-by-step usage in QGIS.

## Implementation Roadmap

For the current native-backend roadmap (1D C++ port + 2D SWE solver), see:

- `docs/IMPLEMENTATION_PLAN_6W_1D_CPP_AND_2D_SWE.md`
- `docs/TASK_BOARD_6W_SOLO_AND_AGENT.md`
