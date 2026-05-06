# Project Context: QGIS Backwater Plugin and SWE2D

## Overview
This repository is a QGIS plugin for backwater, culvert, and solver workflows.
The core engineering focus is shallow-water / hydraulic modeling, with SWE2D
treated as the primary high-performance path.

## Solver Conventions
- Prefer finite-volume shallow-water formulations.
- Prefer HLLC for edge fluxes unless a task explicitly requests another solver.
- Preserve wetting/drying logic and positivity enforcement.
- Use water surface elevation `eta = h + zb` when it improves well-balancing.
- Keep CUDA behavior and GPU robustness as the priority for SWE2D work.

## Repository-Specific Guidance
- Python plugin code lives in the top-level `*.py` modules.
- Native solver code lives under `cpp/src`.
- GPU validation is more important than CPU parity for SWE2D changes.
- The active SWE2D test targets are the GPU-focused suites in `tests/`.

## What Copilot Should Do
- Generate code consistent with the plugin architecture.
- Prefer small, local edits over large refactors.
- Keep numerical changes aligned with existing solver conventions.
- When adding PDE-related code, keep the design compatible with local LLM or CodePDE-style workflows.

## What Copilot Should Avoid
- Do not optimize SWE2D toward CPU parity unless explicitly asked.
- Do not remove wet/dry handling or positivity safeguards.
- Do not introduce unrelated architectural churn.