# SWE3D Phase 5 GUI Persistence Smoke Checklist

Purpose: verify that the new Experimental 3D projection divergence controls persist across QGIS project save/reload and are applied at runtime.

## Preconditions

- Native module is available in the active QGIS environment.
- A test project can open the HYDRA/SWE2D workbench.
- A mesh is loaded or generated so Experimental 3D controls are active.

## Controls under test

- `experimental_3d_projection_residual_sample_iters_spin`
- `experimental_3d_projection_divergence_gate_enable_chk`
- `experimental_3d_projection_divergence_ratio_target_spin`

## Step-by-step smoke flow

1. Open QGIS and load a test project.
2. Open the SWE2D workbench.
3. Enable Experimental 3D patch mode.
4. Set projection controls to non-default test values:
   - Residual sample stride: `7`
   - Divergence gate: `enabled`
   - Divergence ratio target: `0.85`
5. Save the QGIS project.
6. Close the workbench dialog.
7. Close and reopen the project in QGIS.
8. Reopen the workbench.
9. Confirm all three controls restored the exact values from step 4.
10. Start an Experimental 3D run.
11. In runtime logs, confirm `3D projection controls` line reports:
    - `residual_stride=7`
    - `divergence_gate=True`
    - `divergence_ratio_target=0.85`

## Negative toggle check

1. Change divergence gate to disabled.
2. Set divergence ratio target to `1.2`.
3. Save project, close, and reopen project.
4. Confirm values persisted.
5. Run again and confirm log line reports:
   - `divergence_gate=False`
   - `divergence_ratio_target=1.2`

## Expected result

- Widget values persist in project state across reload.
- Collected 3D env overrides reflect persisted values at run time.
- Runtime log includes projection-control observability line for each Experimental 3D run.

## Optional forensic check

In QGIS Python console, inspect persisted state payload:

```python
import json
from qgis.core import QgsProject
proj = QgsProject.instance()
raw, ok = proj.readEntry("Backwater2DWorkbench", "workbench_state_json", "")
print("state bytes:", len(raw or ""), "ok:", ok)
payload = json.loads(raw) if raw else {}
widgets = payload.get("widgets", {})
for k in (
    "experimental_3d_projection_residual_sample_iters_spin",
    "experimental_3d_projection_divergence_gate_enable_chk",
    "experimental_3d_projection_divergence_ratio_target_spin",
):
    print(k, "=>", widgets.get(k, {}).get("value"))
```
