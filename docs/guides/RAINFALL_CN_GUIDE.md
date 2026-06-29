# Rainfall and CN Pipeline Guide

How rainfall and infiltration are modeled in SWE2D.

## Overview

The rainfall module applies spatially-distributed precipitation to the 2D
domain using the SCS Curve Number (CN) method for infiltration. Rainfall
is converted to excess (runoff) via CN-based initial abstraction and
continuous infiltration.

## Pipeline

```
Hyetograph (time series of rain intensity)
    │
    ▼
Rain gage locations ( Thiessen polygon assignment )
    │
    ▼
Per-cell rain rate [mm/hr → m/s]
    │
    ▼
SCS CN infiltration
    │
    ├─ Initial abstraction: Ia = 0.2 × S
    ├─ Potential max retention: S = (25400/CN) − 254
    └─ Excess rainfall: Pe = (P − Ia)² / (P − Ia + S)
    │
    ▼
Per-cell excess rain rate [m/s] → source term in solver
```

## Input Layers

| Layer | Purpose |
|-------|---------|
| `swe2d_hyetographs` | Time series of rain intensity (time, intensity columns) |
| `swe2d_rain_gages` | Point locations that assign hyetographs to cells |
| `swe2d_cn_zones` | Polygons with CN values for infiltration |
| `swe2d_storm_areas` | Polygons defining the rainfall application area |

## How It Works

### 1. Hyetograph Lookup

Each rain gage references a hyetograph by ID. The hyetograph is a time
series of rain intensities (mm/hr). At each coupling step, the current
intensity is interpolated from the hyetograph.

### 2. Thiessen Polygon Assignment

Rain gages are assigned to cells via Thiessen polygon interpolation. Each
cell receives rain from the nearest gage. When multiple gages exist, the
weighted average is used.

### 3. CN-Based Infiltration

For each cell within a CN zone:

1. **Initial abstraction**: `Ia = 0.2 × S` where `S = (25400/CN) − 254`
2. **Cumulative rainfall**: tracked per-cell across timesteps
3. **Excess rainfall**: `Pe = (P − Ia)² / (P − Ia + S)` when `P > Ia`
4. **Excess rate**: `Pe / dt` becomes the cell source term

Cells outside CN zones use CN=100 (impervious, no infiltration).

### 4. Extreme Rain Subcycling

When rain intensity exceeds a threshold (`source_rate_cap`), the solver
automatically subcycles the rainfall source term within a single 2D
timestep. This prevents numerical instability from very large rain rates.

The subcycle count is limited by `source_max_substeps` (default 100).

### 5. Stage-Rate Evaluation

The GPU kernel evaluates the CN excess using a stage-rate formulation
that provides smooth derivatives for the implicit time integrator. This
avoids the discontinuity at the `P = Ia` threshold.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Rain enabled** | off | Master switch for the rainfall module |
| **Default rain intensity** | 0 mm/hr | Uniform rain rate when no hyetograph is loaded |
| **Infiltration enabled** | off | Enable CN-based infiltration |
| **Infiltration model** | `green_ampt` | Infiltration model (CN is the primary method) |
| **Source rate cap** | solver default | Max rain rate before subcycling kicks in |
| **Source CFL beta** | solver default | CFL safety factor for rain subcycling |

## Troubleshooting

### No rain showing up

- Ensure **Rain enabled** is checked in the Parameters tab.
- Verify that rain gages are loaded and assigned to hyetographs.
- Check that the storm area polygon covers your mesh domain.

### Mass conservation errors with heavy rain

- Increase `source_max_substeps` if the log shows substep limit reached.
- Reduce `source_cfl_beta` for more conservative subcycling.
- Check that CN values are reasonable (30–98 range).

### Dry cells remain dry during rain

- Cells outside storm area polygons receive no rain.
- CN=100 zones are impervious — all rain becomes runoff.
- Very low rain intensities may not exceed initial abstraction.
