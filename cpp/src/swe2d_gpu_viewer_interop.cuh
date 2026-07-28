#pragma once
// Phase 3 — CUDA-OpenGL interop for zero-D2H GUI rendering.
//
// Uses the CUDA Driver API (cuGraphicsGLRegisterImage etc.) which lives in
// libcuda.so — already installed as the NVIDIA driver.  This avoids needing
// the deprecated libcudaGL.so (no longer in CUDA 13.x redist).
//
// Pattern:
//   1. Caller creates a GL_TEXTURE_2D (RGBA8, width × height).
//   2. Caller invokes register_texture(gl_tex) → returns resource handle.
//   3. Each frame: map_for_cuda_write(resource) → CUDA writes → unmap.
//   4. OpenGL renders a quad textured with the texture.
//   5. On shutdown: unregister(resource).
//
// Constraints (callers must satisfy):
//   - The OpenGL context must be current on the calling thread at
//     register / unregister / map / unmap time.

#include <cstdint>

// CUgraphicsResource is the driver-API opaque handle.  Forward-declared
// here so the .cu file (which includes cudaGL.h) is the only thing that
// pulls in the driver headers.
typedef struct CUgraphicsResource_st* CUgraphicsResource;
typedef struct cudaArray* cudaArray_t;

namespace swe2d_viewer_interop {

CUgraphicsResource register_texture(unsigned int gl_texture);
void unregister(CUgraphicsResource resource);

// Map the GL texture for CUDA write access; returns a cudaArray_t* that
// the kernel can write into (via cudaMemcpy2DToArray or surface writes).
// Caller must unmap before the OpenGL texture is used for rendering.
cudaArray_t* map_for_cuda_write(CUgraphicsResource resource);
void unmap(CUgraphicsResource resource);

// Synchronize a viewer stage and translate any CUDA error into a
// std::runtime_error carrying the stage name.  `stream` is opaque to
// avoid leaking cudaStream_t into the public header; the .cu file
// performs the cast.  On failure:
//   * cudaPeekAtLastError != cudaSuccess → "<stage> launch failed: ..."
//   * cudaStreamSynchronize != cudaSuccess → "<stage> execution failed: ..."
// Sticky errors are NOT cleared; callers must treat the throw as terminal.
void check_cuda_stage(const char* stage, void* stream);

// Launch the color kernel writing per-cell RGBA directly into the mapped
// cudaArray_t (no D2H).  All buffers must be device-resident:
//   d_field       — solver's d_h / d_hu / d_hv (device)
//   d_cell_x/y    — uploaded cell centroids (device, n_cells doubles each)
//   d_lut         — uploaded 256×4 RGBA colormap LUT (device)
//   d_vminmax     — uploaded vmin (idx 0) and vmax (idx 1) (device, 2 doubles).
//                  Written by compute_field_minmax_into_dev; the color kernel
//                  reads them once per block from device memory.  Keeps the
//                  colorization path fully device-side (zero D2H for the
//                  color pipeline).
void launch_color_kernel_into_array(
    int32_t n_cells,
    const double* __restrict__ d_field,
    const double* __restrict__ d_vminmax,
    const double* __restrict__ d_cell_x,
    const double* __restrict__ d_cell_y,
    double x_min, double x_max, double y_min, double y_max,
    int32_t width, int32_t height,
    const uint8_t* __restrict__ d_lut,
    cudaArray_t cu_array,
    void* stream);

/// Device-side reduction: compute min/max of d_field[0..n_cells-1] into
/// the 2-double buffer d_minmax_device (index 0 = min, index 1 = max).
/// Caller MUST seed the buffer before launch:
///   cudaMemset(d_minmax_device, 0xFF, sizeof(double));  // min = +INF bits
///   cudaMemset(...) for max = -INF bits.
/// Single-block reduction with atomic updates — sufficient for meshes up
/// to ~10K cells in a single launch.  Larger meshes loop within the block.
void compute_field_minmax_into_dev(
    int32_t n_cells,
    const double* __restrict__ d_field,
    double* __restrict__ d_minmax_device,
    void* stream);

}  // namespace swe2d_viewer_interop