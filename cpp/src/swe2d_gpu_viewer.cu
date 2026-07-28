// Phase 2 — CLI headless renderer color kernel.
// See swe2d_gpu_viewer.cuh for the contract.
#include "swe2d_gpu_viewer.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace swe2d_viewer {

__global__ void color_kernel(
    int32_t n_cells,
    const double* __restrict__ d_field,
    double vmin,
    double vmax,
    const double* __restrict__ d_cell_x,
    const double* __restrict__ d_cell_y,
    double x_min, double x_max,
    double y_min, double y_max,
    int32_t width, int32_t height,
    const uint8_t* __restrict__ d_lut,
    uint8_t* __restrict__ d_rgba_out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // Normalize field → [0, 1] → index into 256-entry LUT.
    double v = d_field[c];
    double t = (v - vmin) / fmax(vmax - vmin, 1.0e-12);
    t = fmax(0.0, fmin(1.0, t));
    int idx = (int)(t * 255.0 + 0.5);
    if (idx < 0)   idx = 0;
    if (idx > 255) idx = 255;

    int lut_off = idx * 4;
    uint8_t r = d_lut[lut_off + 0];
    uint8_t g = d_lut[lut_off + 1];
    uint8_t b = d_lut[lut_off + 2];
    uint8_t a = d_lut[lut_off + 3];

    // Cell centroid → pixel coords (flip y so up is up on screen).
    double px = (d_cell_x[c] - x_min) / (x_max - x_min) * (double)(width - 1);
    double py = (d_cell_y[c] - y_min) / (y_max - y_min) * (double)(height - 1);
    int ix = (int)(px + 0.5);
    int iy = height - 1 - (int)(py + 0.5);
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    int out_off = (iy * width + ix) * 4;
    d_rgba_out[out_off + 0] = r;
    d_rgba_out[out_off + 1] = g;
    d_rgba_out[out_off + 2] = b;
    d_rgba_out[out_off + 3] = a;
}

void launch_color_kernel(
    int32_t n_cells,
    const double* d_field,
    double vmin, double vmax,
    const double* d_cell_x,
    const double* d_cell_y,
    double x_min, double x_max,
    double y_min, double y_max,
    int32_t width, int32_t height,
    const uint8_t* d_lut,
    uint8_t* d_rgba_out,
    void* stream)
{
    if (n_cells <= 0 || width <= 0 || height <= 0) return;

    int block = 256;
    int grid = (n_cells + block - 1) / block;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    color_kernel<<<grid, block, 0, s>>>(
        n_cells, d_field, vmin, vmax, d_cell_x, d_cell_y,
        x_min, x_max, y_min, y_max,
        width, height, d_lut, d_rgba_out);
}

}  // namespace swe2d_viewer