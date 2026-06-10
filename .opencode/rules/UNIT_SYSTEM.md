# Unit System Conventions

- **Never assume a specific unit system** (SI or USC). All conversions must be based on the CRS-derived map units via `swe2d.units`.
- **C++ kernel accepts model units** for all geometry. Weir, orifice, bridge, and pump formulas are unit-agnostic — they produce correct results in whatever units the inputs are in, as long as the `gravity` parameter matches. Only the HDS-5 culvert path converts geometry to feet internally, computes in USC, then converts the result back to model units using the caller-supplied `model_to_ft` factor.
- **C++ kernel culvert output** is converted from CFS back to model units (÷ `model_to_ft³`) before returning. Non-culvert types return values directly in model units.
- **Python coupling controller** (`coupling.py`) converts kernel CFS output to model units via `SI_M3_PER_USC_FT3 / si_m3_per_model_volume()`.
- **Python structure module** (`swe2d/extensions/structures.py`) always returns **CMS** because culvert routines adopted from SWMM compute in USC and explicitly convert CFS→CMS.
- **Diagnostics stored in `SWE2DCouplingDiagnostics`** are in **model units** (not SI). The coupling controller converts from kernel/Python output units to model units before storing.
- **Runtime reporter** (`runtime_reporting.py`) displays diagnostics using `length_unit_name` and assumes values are already in model units.
- **Heap gravity bug fix**: Orifice/bridge formulas now use CRS-derived `gravity()` (9.81 m/s² for SI, 32.17 ft/s² for USC) instead of the old hardcoded 9.81 — this was a ~45% underestimation for USC projects.
- **`model_to_ft`**: Passed from Python to the C++ kernel as `units.model_to_ft()`. Needed so culvert code can convert model-unit geometry to feet internally.
