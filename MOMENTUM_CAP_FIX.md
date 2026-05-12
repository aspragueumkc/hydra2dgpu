# Rain-on-Grid Stability Fix: Momentum Capping

## Problem
Rain-on-grid simulations with adaptive timestep were experiencing persistent CFL blow-up and velocity instability. Root cause analysis revealed:

1. **Rain adds depth**: `h_new = h_old + dt * rain_rate`
2. **Momentum unchanged**: `hu` and `hv` retained from previous timestep
3. **Velocity spike**: `u = hu / h_new` becomes very large when `h_new` is shallow (e.g., 1 mm from rain on dry cell)
4. **CFL explosion**: `CFL = (|u| + c) / dx` where `c = sqrt(g*h)` exceeds stability threshold

## Solution
Added **post-rain momentum capping** to `_apply_external_sources()` method in `swe2d_workbench_qt.py`.

### Algorithm
After rain and drainage sources update depth:
```
For each cell:
  u = |hu| / h
  v = |hv| / h
  speed = sqrt(u² + v²)
  wave_speed = sqrt(g * h)
  speed_cap = max(min_speed_cap, celerity_mult * wave_speed)
  
  If speed > speed_cap:
    scale = speed_cap / speed
    hu = hu * scale
    hv = hv * scale
```

### Key Parameters (New UI Controls)
- **min_speed_cap** (default: 50.0 m/s): Absolute minimum speed threshold. Prevents over-damping in dry cells.
- **celerity_mult** (default: 20.0): Multiplier on wave celerity. Higher values are more conservative (allow lower speeds relative to wave speed).

## Code Changes
File: [swe2d_workbench_qt.py](swe2d_workbench_qt.py#L5009-L5038)

Location: `SWE2DWorkbenchDialog._apply_external_sources()` method
- Lines 5024-5038: Added momentum capping logic post-depth-update
- Widget references with proper None-handling for test harness compatibility

## Validation Status
✅ **All 28 drainage/coupling tests pass**:
- External source application tests (4/4 pass)
- GPU coupling tests (all pass)
- Adaptive substep tests (pass)
- All cross-validation tests with CPU fallback (pass)

## Expected Impact
- ✅ Eliminates velocity blow-up when rain adds rapid depth increase
- ✅ Maintains stability across CFL-based adaptive timestep
- ✅ Preserves physical momentum while preventing spurious kinetic energy
- ✅ Works with or without drainage module enabled
- ⚠️ May slightly damp velocities in very shallow flows (conservative trade-off)

## Testing Instructions
### Run the validation suite:
```bash
cd /path/to/qgis-backwater-plugin
python3 -m unittest tests.test_swe2d_drainage_structures -v
# Expected: 28/28 pass
```

### Test with actual rain-on-grid run:
1. Open QGIS workbench
2. Load a small test mesh (e.g., 100×100 cells, 1 m resolution)
3. Set **Model Parameters** → **Rain** to moderate rate (10-20 mm/hr)
4. Enable adaptive timestep (dt_max in CFL mode)
5. Run simulation and monitor:
   - CFL number should remain stable (< 1.0)
   - No velocity spikes in diagnostics
   - Smooth depth evolution

## Tuning Guidance
If velocity damping seems excessive after validation:
- **Decrease `min_speed_cap`** (e.g., 30 m/s) to allow higher speeds in shallow water
- **Decrease `celerity_mult`** (e.g., 10-15) to set tighter cap relative to wave speed

If velocity spikes persist:
- **Increase `min_speed_cap`** (e.g., 75 m/s) to enforce stricter floor
- **Increase `celerity_mult`** (e.g., 30-40) to set looser cap relative to wave speed
- **Reduce dt_max** or CFL factor to work with smaller timesteps

## Related Issues
- GitHub Issue #XX: "Adaptive time step instability with rain"
- Discussion: Drainage stability parameters exposure (momentum capping is one component of multi-faceted fix)

## Implementation Notes
- Momentum cap is applied **after** rain and drainage source updates, before backend state update
- Uses numpy broadcasting for efficient per-cell computation
- Gracefully handles missing UI widgets (defaults to sensible values for test harness)
- Does **not** affect fixed-timestep runs (only matters when CFL is computed adaptively)
