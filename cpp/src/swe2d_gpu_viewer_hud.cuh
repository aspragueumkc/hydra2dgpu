// Phase 4.2 — HUD render kernel + minimal 5×7 bitmap font.
//
// Renders one line of ASCII text onto a cudaSurfaceObject_t (the same
// GL-mapped cudaArray the Phase 3 color kernel writes into). The font is
// a hardcoded 5×7 bitmap covering ASCII 32 (space) .. 127 (DEL) — 96
// glyphs × 7 rows = 672 bytes total. Public-domain glyph data.
//
// Per-glyph encoding: 7 bytes per glyph, one per row. Within each byte,
// bit (4 - col) holds the pixel state for that column (0 = off,
// 1 = on).  LSB = leftmost column.
//
// Letter shapes were hand-traced from the public-domain "Tom Thumb"
// 5×7 font (CC0 / public domain).  Enough to render HUD strings like
//   "t=42.5s CFL=0.46 wet=16032 dt=0.035"

#pragma once
#include <cstdint>

typedef struct CUstream_st* cudaStream_t;

namespace swe2d_hud {

// Initialize the device font + LUT (idempotent).
void init();

// Render one null-terminated ASCII string at (x, y) pixel coords.
// bg_rgba = 0xAABBGGRR; fg_rgba = 0xAABBGGRR.  Caller must ensure the
// cudaArray_t has been wrapped in a cudaSurfaceObject_t (swe2d_hud_render
// does the wrap internally).
void render(
    cudaArray_t cu_array,
    int32_t fb_width, int32_t fb_height,
    const char* text,
    int32_t x, int32_t y,
    uint32_t bg_rgba, uint32_t fg_rgba);

}  // namespace swe2d_hud