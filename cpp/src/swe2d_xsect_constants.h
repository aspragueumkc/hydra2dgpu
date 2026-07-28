#pragma once
// swe2d_xsect_constants.h
// Shared cross-section shape codes, surcharge method constants, and outfall
// mode constants. Used by both pipe1d.cu and swe2d_gpu.cu — defined here
// to avoid duplication.

namespace swe2d {
    constexpr int XSECT_CIRCULAR    = 0;
    constexpr int XSECT_RECTANGULAR = 1;
    constexpr int XSECT_ELLIPTICAL  = 2;

    constexpr int SURCHARGE_NONE    = 0;
    constexpr int SURCHARGE_SLOT    = 1;
    constexpr int SURCHARGE_EXTRAN  = 2;

    // Floor for γ in 1+γ·Δt denominator (1D semi-implicit friction,
    // matches the OMEGA_MIN floor concept used in the 2D solver).
    constexpr double OMEGA_MIN = 1e-6;

    // SPEC §2.8 — Outfall BC modes.
    constexpr int OUTFALL_FREE         = 0;
    constexpr int OUTFALL_NORMAL_DEPTH = 1;
    constexpr int OUTFALL_FIXED_WSE    = 2;
    constexpr int OUTFALL_RATING_CURVE = 3;
    constexpr int OUTFALL_TABULAR      = 4;

    constexpr int MAX_RATING_POINTS  = 32;
    constexpr int MAX_TABULAR_POINTS = 32;
}
