// Phase 5.1 — color-space utilities for NVENC.
//
// NVENC consumes NV12 (YUV 4:2:0) frames, not RGBA.  This file provides
// the device-side RGBA → NV12 conversion kernel and a small host helper
// to allocate the output buffers.
//
// The input is a device pointer (NOT a cudaArray) — Phase 5.1 MVP path.
// The binding does a host→device copy of the RGBA frame, then launches
// this kernel which writes the NV12 output.  The Phase 5+ refactor can
// chain the color kernel + this kernel for true zero-D2H.

#include "swe2d_nv_utils.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace swe2d_nv {

__global__ void rgba_to_nv12_kernel(
    const uint8_t* __restrict__ d_rgba,  // RGBA8, width * height * 4
    uint8_t* __restrict__ d_y_plane,     // width * height
    uint8_t* __restrict__ d_uv_plane,    // width * height / 2  (interleaved)
    int32_t width, int32_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Read RGBA from the device buffer (one pixel per thread).
    int idx = (y * width + x) * 4;
    int r = (int)d_rgba[idx + 0];
    int g = (int)d_rgba[idx + 1];
    int b = (int)d_rgba[idx + 2];

    // BT.601 conversion (approximate — fine for diagnostics).
    int yy = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    int uu = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    int vv = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    if (yy < 0)   yy = 0;
    if (yy > 255) yy = 255;
    if (uu < 0)   uu = 0;
    if (uu > 255) uu = 255;
    if (vv < 0)   vv = 0;
    if (vv > 255) vv = 255;
    d_y_plane[y * width + x] = (uint8_t)yy;

    // Subsample U/V to 2×2 block (every other pixel on both axes).
    if ((x & 1) == 0 && (y & 1) == 0) {
        int uv_index = (y / 2) * width + (x / 2) * 2;
        d_uv_plane[uv_index + 0] = (uint8_t)uu;
        d_uv_plane[uv_index + 1] = (uint8_t)vv;
    }
}

void rgba_to_nv12(
    const uint8_t* d_rgba,
    int32_t width, int32_t height,
    uint8_t* d_y_plane,
    uint8_t* d_uv_plane,
    void* stream)
{
    if (d_rgba == nullptr) throw std::runtime_error("rgba_to_nv12: null d_rgba");
    if (width <= 0 || height <= 0) throw std::runtime_error("rgba_to_nv12: bad dims");
    if ((width & 1) || (height & 1)) {
        // NVENC requires even width/height.
        throw std::runtime_error("rgba_to_nv12: width and height must be even");
    }
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rgba_to_nv12_kernel<<<grid, block, 0, s>>>(
        d_rgba, d_y_plane, d_uv_plane, width, height);
}

}  // namespace swe2d_nv