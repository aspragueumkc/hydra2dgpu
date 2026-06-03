# Mixed-Precision (FP32/FP64) GPU Optimization Plan

Date: 2026-06-01

## Objective

Reduce kernel execution time in the SWE2D GPU solver by using lower-precision arithmetic
(FP32/`float`) for compute-intensive but numerically stable sub-operations within the HLLC
flux kernel, while preserving FP64/`double` for state accumulation, mass conservation, and
near-cancellation-sensitive algebraic expressions.

## Target Architecture

- **Current hardware**: Ampere (SM 80) — FP64 throughput is 1/2 of FP32
- **Future**: Volta (SM 70) — FP64 is 1/32 of FP32, Hopper (SM 90) — FP64 is 1/2
- Gated by environment variable `BACKWATER_SWE2D_MIXED_PRECISION=1` (opt-in)

## Conservative Hybrid Approach

State I/O and critical arithmetic stays in FP64. The hot-path computation inside the flux
kernel is selectively demoted to FP32 where numerical analysis shows it is safe.

### Precision Budget

| Variable / Expression | Precision | Rationale |
|---|---|---|
| `cell_h[c]`, `cell_hu[c]`, `cell_hv[c]` (reads) | FP64 | Conserved variables accumulate across timesteps |
| `flux_h[e]`, `flux_hu[e]`, `flux_hv[e]` (writes) | FP64 | Accumulated across edges per cell; mass conservation |
| `edge_len[e]` | FP64 | Can span 0.01–1000 m; product magnitude matters |
| HLLC Riemann solver internals | **FP32** | Wave speeds (SL, SR, S_star) are well above FP32 epsilon for dynamically active edges |
| TVD limiter (all variants) | **FP32** | `fmax`/`fmin` chains and bounded dot products |
| WENO3-like smoothness weights | **FP32** | Nonlinear blend tolerates small errors |
| Gradient extrapolation (dx/dy dot) | **FP32** | Works at coordinate-difference scale |
| Ghost state construction | **FP32** | Simple arithmetic, no near-cancellation |
| Momentum capping (`sqrt(g*h)`) | **FP32** | Fine for h > 1e-6 m |
| **Hydrostatic reconstruction** | **FP64** | `eta - zb_face` — near-cancellation when cell is nearly dry |
| **Bed slope correction** | **FP64** | `0.5*g*(hL_star² - hL²)` — difference of squares; catastrophic cancellation at flat beds |
| **Final flux × len scaling** | **FP64** | Product magnitude matters for mass conservation |

### Implementation Pattern (inside `swe2d_flux_kernel`)

```
Read state (FP64) ──► if mixed_precision enabled ──► cast to float
                                    │
                         ┌─────────┼──────────┐
                         ▼         ▼          ▼
                   HLLC solver  TVD/WENO3  Ghost states
                   [float]      [float]    [float]
                         │         │          │
                         └─────────┼──────────┘
                                   ▼
                            flux_fh, flux_fhu, flux_fhv (float)
                                   │
                                   ▼ cast back to double
                                   │
              Hydrostatic recon [FP64] ◄── always FP64
              Bed slope corr    [FP64] ◄── always FP64
              Flux × len        [FP64] ◄── always FP64
              Write flux arrays [FP64]
```

The kernel signature gains one parameter: `bool use_mixed_precision`.  No template
instantiation or code duplication — the FP32 path is implemented as a local block
within the existing kernel.

### Shallow-Cell Protection

When `(hL + hR) < PROTECTION_THRESHOLD` (e.g. 10 × h_min ≈ 1e-5 m), the entire
edge-computation stays in FP64 regardless of the mixed-precision flag, guarding
against FP32 subnormal issues at wet/dry fronts.

## Accuracy Validation

Add a runtime validation mode (`BACKWATER_SWE2D_MIXED_PRECISION_VALIDATE=1`) that:

1. Samples a small fraction of edges (e.g. 1%, stride-based) each step
2. Computes the HLLC flux for those edges in both FP32 and FP64
3. Tracks the max relative difference `|q_f32 - q_f64| / max(1e-12, |q_f64|)`
4. Logs a warning if any edge exceeds a 1% threshold
5. Disables mixed precision for subsequent steps if the threshold is exceeded
   more than N times (backoff policy)

## Expected Throughput

| Architecture | HLLC portion speedup | Flux kernel speedup | End-to-end step speedup |
|---|---|---|---|
| Volta (SM 70) | 15–25× | ~2–3× | 50–70% |
| Ampere (SM 80) | 1.5–1.8× | 20–35% | 10–20% |
| Hopper (SM 90) | 1.5–1.8× | 20–35% | 10–20% |

Since the flux kernel is typically 40–60% of total step time (excluding coupling,
which is out of scope for this optimization), the end-to-end benefit on Ampere is
modest but worthwhile as an opt-in feature.

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| FP32 subnormals at thin-film depths | Spurious momentum spikes | Shallow-cell FP64 protection |
| HLLC wave-speed sign flip in FP32 | Wrong upwind direction | Validation sampling catches this |
| TVD slope-ratio denominator truncation | Ineffective limiter, oscillations | Pair-bounds clamp (FP64) after reconstruction |
| Code maintenance burden | Two code paths | No templates; local conversion block only |
