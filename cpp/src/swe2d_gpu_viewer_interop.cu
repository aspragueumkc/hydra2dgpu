// Phase 3 — CUDA-OpenGL interop implementation.
// Uses the CUDA Driver API (cu* functions) which lives in libcuda.so
// (already installed as the NVIDIA driver) — no libcudaGL.so needed.
// See swe2d_gpu_viewer_interop.cuh for the contract.
#include "swe2d_gpu_viewer_interop.cuh"

// OpenGL header first — cudaGL.h needs GLuint / GLenum / GL_TEXTURE_2D defined.
#include <GL/gl.h>
#include <cudaGL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace swe2d_viewer_interop {

CUgraphicsResource register_texture(unsigned int gl_texture) {
    // CUDA driver state is initialized by the pybind11 module-load
    // cudaSetDevice(0) call (in swe2d_bindings.cpp).  We deliberately
    // don't `cuCtxSetCurrent` or `cuDevicePrimaryCtxRetain` here —
    // either would interfere with the CUDA runtime API's per-thread
    // primary context bookkeeping on the solver worker thread
    // (cudaMemcpyAsync then surfaces cudaErrorIllegalAddress from a
    // context the runtime doesn't recognize as its own).
    CUgraphicsResource res = nullptr;
    CUresult err = cuGraphicsGLRegisterImage(
        &res, gl_texture, GL_TEXTURE_2D,
        CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    if (err != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        std::string msg = "cuGraphicsGLRegisterImage failed: ";
        if (err_str) {
            msg += err_str;
        } else {
            // cuGetErrorString returned null — include the numeric code
            // so the actual CUDA error class is recoverable from logs.
            msg += "code=" + std::to_string((int)err);
        }
        msg += " (cuda_err=" + std::to_string((int)err) + ")";
        throw std::runtime_error(msg);
    }
    return res;
}

void unregister(CUgraphicsResource resource) {
    if (resource == nullptr) return;
    CUresult err = cuGraphicsUnregisterResource(resource);
    if (err != CUDA_SUCCESS) {
        // Log but don't throw — caller is tearing down.
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        (void)err_str;
    }
}

cudaArray_t* map_for_cuda_write(CUgraphicsResource resource) {
    if (resource == nullptr) {
        throw std::runtime_error("map_for_cuda_write: null resource");
    }
    CUresult err = cuGraphicsMapResources(1, &resource, 0);
    if (err != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        throw std::runtime_error(
            std::string("cuGraphicsMapResources failed: ")
            + (err_str ? err_str : "unknown error"));
    }
    // Driver API returns CUarray* (= CUarray_st**); same underlying type as
    // cudaArray_t* but we keep them typed correctly per API.
    CUarray* cu_array = new CUarray(nullptr);
    err = cuGraphicsSubResourceGetMappedArray(cu_array, resource, 0, 0);
    if (err != CUDA_SUCCESS) {
        cuGraphicsUnmapResources(1, &resource, 0);
        delete cu_array;
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        throw std::runtime_error(
            std::string("cuGraphicsSubResourceGetMappedArray failed: ")
            + (err_str ? err_str : "unknown error"));
    }
    // Cast: CUarray and cudaArray_t are typedef'd to the same underlying
    // struct cudaArray*.  The kernel writes via cudaMemcpy2DToArray which
    // takes cudaArray_t — both APIs share this opaque type.
    return reinterpret_cast<cudaArray_t*>(cu_array);
}

void unmap(CUgraphicsResource resource) {
    if (resource == nullptr) return;
    CUresult err = cuGraphicsUnmapResources(1, &resource, 0);
    if (err != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        throw std::runtime_error(
            std::string("cuGraphicsUnmapResources failed: ")
            + (err_str ? err_str : "unknown error"));
    }
}

__global__ void color_kernel_into_array(
    int32_t n_cells,
    const double* __restrict__ d_field,
    const double* __restrict__ d_vminmax,
    const double* __restrict__ d_cell_x,
    const double* __restrict__ d_cell_y,
    double x_min, double x_max,
    double y_min, double y_max,
    int32_t width, int32_t height,
    const uint8_t* __restrict__ d_lut,
    cudaSurfaceObject_t surf)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    // vmin / vmax live on device (written by compute_field_minmax_into_dev).
    // Read once per kernel invocation from global memory — single-thread
    // divergence is harmless since the value is constant across cells.
    double vmin = d_vminmax[0];
    double vmax = d_vminmax[1];
    double t = (d_field[c] - vmin) / fmax(vmax - vmin, 1.0e-12);
    t = fmax(0.0, fmin(1.0, t));
    int idx = (int)(t * 255.0 + 0.5);
    if (idx < 0)   idx = 0;
    if (idx > 255) idx = 255;
    int lut_off = idx * 4;
    uchar4 pix;
    pix.x = d_lut[lut_off + 0];
    pix.y = d_lut[lut_off + 1];
    pix.z = d_lut[lut_off + 2];
    pix.w = 255;

    double px = (d_cell_x[c] - x_min) / (x_max - x_min) * (double)(width - 1);
    double py = (d_cell_y[c] - y_min) / (y_max - y_min) * (double)(height - 1);
    int ix = (int)(px + 0.5);
    int iy = height - 1 - (int)(py + 0.5);
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    // Write per-cell RGBA directly into the GL-mapped cudaSurfaceObject.
    // `surf2Dwrite` is the device-callable surface write (cudaMemcpy2DToArray
    // is host-only).  The surface was bound to the GL texture's storage by
    // the binding's map step.
    surf2Dwrite(pix, surf, ix * sizeof(uchar4), iy);
}

void launch_color_kernel_into_array(
    int32_t n_cells,
    const double* d_field,
    const double* d_vminmax,
    const double* d_cell_x,
    const double* d_cell_y,
    double x_min, double x_max, double y_min, double y_max,
    int32_t width, int32_t height,
    const uint8_t* d_lut,
    cudaArray_t cu_array,  // underlying array — wraps it in a surface
    void* stream)
{
    if (n_cells <= 0 || width <= 0 || height <= 0) return;
    if (cu_array == nullptr) return;

    // Wrap the cudaArray_t in a cudaSurfaceObject_t so the kernel can
    // write via surf2Dwrite (device-callable).  cudaCreateSurfaceObject
    // is the runtime-API wrapper for the driver-API CUarray.
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;
    cudaSurfaceObject_t surf = 0;
    cudaError_t cerr = cudaCreateSurfaceObject(&surf, &res_desc);
    if (cerr != cudaSuccess) {
        throw std::runtime_error(
            std::string("viewer color surface creation failed: ")
            + cudaGetErrorString(cerr));
    }

    int block = 256;
    int grid = (n_cells + block - 1) / block;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    try {
        color_kernel_into_array<<<grid, block, 0, s>>>(
            n_cells, d_field, d_vminmax, d_cell_x, d_cell_y,
            x_min, x_max, y_min, y_max,
            width, height, d_lut, surf);

        // Synchronize the stage before tearing down the surface so the
        // surface lifetime stays bounded and explicit.
        check_cuda_stage("viewer color", stream);

        // Free the surface object now (after the kernel has actually
        // completed — check_cuda_stage above is a barrier).
        cudaError_t derr = cudaDestroySurfaceObject(surf);
        if (derr != cudaSuccess) {
            throw std::runtime_error(
                std::string("viewer color surface destroy failed: ")
                + cudaGetErrorString(derr));
        }
    } catch (...) {
        // Best-effort cleanup on any exception path — make sure the
        // surface handle is released even if the stage check or a later
        // destroy already partially failed.
        (void)cudaDestroySurfaceObject(surf);
        throw;
    }
}

// ─── Field min/max reduction (two-pass, zero D2H) ──────────────────────
//
// Single-block reduction with shared-memory partials + CAS-loop publish
// to the 2-element output buffer.  Caller MUST seed the buffer:
//   *d_minmax_device[0] = +INFINITY  (min slot)
//   *d_minmax_device[1] = -INFINITY  (max slot)
// before launch (use cudaMemset with the right bit pattern, or write
// via cudaMemcpy from host doubles).
//
// Supports arbitrary n_cells via per-thread loop.  Uses one block of 256
// threads; larger meshes serialize within the block — sufficient for
// the dambreak / small-project meshes this viewer targets.
//
// CAS via unsigned long long overload of atomicCAS (the double overload
// doesn't exist in CUDA).
__device__ __forceinline__ void cas_min_double(double* addr, double val) {
    unsigned long long* addr_as_ll =
        reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ll, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(static_cast<long long>(assumed));
        if (val >= cur) return;
        old = atomicCAS(addr_as_ll, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(val)));
    } while (assumed != old);
}

__device__ __forceinline__ void cas_max_double(double* addr, double val) {
    unsigned long long* addr_as_ll =
        reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ll, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(static_cast<long long>(assumed));
        if (val <= cur) return;
        old = atomicCAS(addr_as_ll, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(val)));
    } while (assumed != old);
}

__global__ void compute_field_minmax_kernel(
    int32_t n_cells,
    const double* __restrict__ d_field,
    double* __restrict__ d_minmax_device)
{
    __shared__ double s_min;
    __shared__ double s_max;
    if (threadIdx.x == 0) {
        s_min = d_minmax_device[0];
        s_max = d_minmax_device[1];
    }
    __syncthreads();

    double local_min =  1.0e300;
    double local_max = -1.0e300;
    for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
        double v = d_field[i];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    __syncthreads();
    if (local_min < s_min) s_min = local_min;
    if (local_max > s_max) s_max = local_max;
    __syncthreads();

    // Publish to device via CAS loops (portable across CC, no dependency
    // on atomicMin/atomicMax double support which is CC ≥6.0 only).
    if (threadIdx.x == 0) {
        cas_min_double(d_minmax_device + 0, s_min);
        cas_max_double(d_minmax_device + 1, s_max);
    }
}

void compute_field_minmax_into_dev(
    int32_t n_cells,
    const double* __restrict__ d_field,
    double* __restrict__ d_minmax_device,
    void* stream)
{
    if (n_cells <= 0) return;
    int block = 256;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    compute_field_minmax_kernel<<<1, block, 0, s>>>(
        n_cells, d_field, d_minmax_device);
    check_cuda_stage("viewer min/max", stream);
}

void check_cuda_stage(const char* stage, void* stream) {
    // cudaPeekAtLastError does NOT reset the sticky error (unlike
    // cudaGetLastError) — the contract here is "do not clear sticky
    // errors", so a downstream caller can still see the original cause.
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(stage) + " launch failed: " + cudaGetErrorString(err));
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(stage) + " execution failed: " + cudaGetErrorString(err));
    }
}

}  // namespace swe2d_viewer_interop