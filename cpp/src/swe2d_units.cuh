/**
 * swe2d_units.cuh — Centralized unit-conversion constants for SWE2D CUDA kernels.
 *
 * Dimensional notation in comments:
 *   L   = length            L²  = area
 *   L³  = volume            L³/T = volumetric flow
 *   L/T = velocity          L/T² = acceleration
 *   L/L = dimensionless slope
 *
 * All kernel computation uses US Customary (feet, ft², ft³/s) internally.
 * The constants below encode the conversion factors from SI → USC.
 */

#ifndef SWE2D_UNITS_CUH
#define SWE2D_UNITS_CUH

// ── SI ↔ USC length ──
constexpr double USC_FT_PER_SI_M    = 3.280839895013123;   // ft per m
constexpr double SI_M_PER_USC_FT   = 0.3048;               // m per ft
constexpr double SI_M2_PER_USC_FT2 = 0.09290304;           // m² per ft²
constexpr double SI_M3_PER_USC_FT3 = 0.028316846592;       // m³ per ft³

// ── Volumetric ──
constexpr double USC_FT3_PER_SI_M3  = 35.31466672148859;   // ft³/s per m³/s  (CFS per CMS)

// ── Gravity ──
constexpr double USC_GRAVITY        = 32.17404855643045;   // ft/s²  (L/T²)
constexpr double SI_GRAVITY         = 9.80665;             // m/s²   (L/T²)

// Historical aliases (prefer canonical names above)
constexpr double BW2D_GRAVITY        = USC_GRAVITY;         // ft/s²
constexpr double FT_PER_M           = USC_FT_PER_SI_M;     // ft per m
constexpr double CFS_PER_CMS        = USC_FT3_PER_SI_M3;   // ft³/s per m³/s

// ── Manning ──
constexpr double SI_MANNING_FACTOR  = 1.0;
constexpr double USC_MANNING_FACTOR = 1.486;                // 1.486/n^(1/2) for USC

#endif // SWE2D_UNITS_CUH