#pragma once
// Phase 5.1 — color-space utilities for NVENC.
#include <cstdint>

namespace swe2d_nv {

// Convert RGBA8 (device pointer, width × height) to NV12 (planar Y +
// interleaved UV at half-resolution, width × height / 2).  Width and
// height must be even (NVENC constraint).  All output pointers must be
// device-resident and pre-allocated to (width * height) and
// (width * height / 2) bytes.
void rgba_to_nv12(
    const uint8_t* d_rgba,
    int32_t width, int32_t height,
    uint8_t* d_y_plane,
    uint8_t* d_uv_plane,
    void* stream);

}  // namespace swe2d_nv