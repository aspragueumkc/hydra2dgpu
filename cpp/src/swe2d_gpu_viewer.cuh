#pragma once
// Phase 2 — CLI headless renderer color kernel (swe2d_gpu_viewer.cu).
//
// One thread per mesh cell.  Maps field[c] → RGBA via a 256-entry colormap
// LUT, rasterizes to a pixel position in the framebuffer using the cell
// centroid.  Background pixels (no cell) are left untouched — the caller
// zero-fills the framebuffer before launching.
//
// Public API is `swe2d_viewer::launch_color_kernel`.  Caller (binding or
// test) is responsible for: uploading cell_x/cell_y, the colormap LUT,
// and the field buffer; allocating the output framebuffer; D2H after the
// launch returns.
#include <cstdint>

namespace swe2d_viewer {

// stream is `cudaStream_t` from cuda_runtime.h; declared `void*` here so
// callers (including the .cu file) don't need to include CUDA headers.
void launch_color_kernel(
    int32_t n_cells,
    const double* __restrict__ d_field,
    double vmin,
    double vmax,
    const double* __restrict__ d_cell_x,
    const double* __restrict__ d_cell_y,
    double x_min, double x_max,
    double y_min, double y_max,
    int32_t width, int32_t height,
    const uint8_t* __restrict__ d_colormap_lut,  // [256 * 4] RGBA bytes
    uint8_t* __restrict__ d_rgba_out,             // [width * height * 4] RGBA bytes
    void* stream);

}  // namespace swe2d_viewer