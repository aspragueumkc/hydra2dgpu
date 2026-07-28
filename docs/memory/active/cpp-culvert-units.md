---
type: memory
status: active
created: 2026-07-26
topic: cpp-culvert-units
tags: [cpp, units]
evidence: AGENTS.md:120
related:
  - AGENTS.md
---

# C++ Culvert Path Unit Convention

## Context

The C++ kernel's culvert path was historically hardcoded to gravity = 9.81
m/s², which underestimated flow by ~45% for USC projects. The HDS-5 culvert
routines compute in feet/CFS internally and convert results back via the
caller-supplied `model_to_ft`.

## Decision

- C++ culvert output is in CFS; `coupling.py` converts CFS → model units using
  `SI_M3_PER_USC_FT3 / si_m3_per_model_volume()`.
- `swe2d/extensions/structures.py` always returns **CMS** for culverts
  (explicit CFS → CMS conversion).
- Orifice/bridge formulas use CRS-derived `gravity()` (9.81 m/s² SI,
  32.17 ft/s² USC). Never hardcode gravity.
- `SWE2DCouplingDiagnostics` are stored in **model units**; the coupling
  controller converts before storing.

## Open questions

- Are there other kernel paths that still hardcode gravity? Audit
  `cpp/src/swe2d/` for the literal `9.81`.
