#pragma once
// Phase 5.1 — NVENC wrapper for zero-D2H recording.
//
// One NvencRecorder per recording session.  Owns the encode session,
// the registered input resource (CUDA device pointer → H.264 NALs),
// and the on-disk .ts muxer.  The Python binding in swe2d_bindings.cpp
// wraps this and exposes start / encode_rgba / finalize.

#include <cstdint>
#include <cstdio>
#include <string>

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUgraphicsResource_st* CUgraphicsResource;

namespace swe2d_nvenc {

struct NVencHandle;  // opaque; defined in .cu

// Returns true if the NVIDIA driver supports NVENC on this GPU +
// libnvidia-encode.so is loadable.  Cheap (one-time probe).
bool is_available();

// Create a recorder.  The CUDA device pointer `d_nv12_device_ptr` points
// to a width*height Y plane followed by a width*height/2 interleaved UV
// plane (NV12 layout, as written by swe2d_nv::rgba_to_nv12).  The
// recorder registers this pointer as an NVENC input resource and
// writes encoded H.264 NALs to the .ts muxer at `output_path`.
//
// Returns nullptr on failure.  Caller is responsible for calling
// encode_frame() + finalize() + delete.
NVencHandle* start(
    const std::string& output_path,
    int32_t width, int32_t height,
    int32_t fps,
    int32_t gop_size,
    void* d_nv12_device_ptr);

// Encode one frame.  `rgba_host` is a width*height*4 byte host buffer;
// the recorder copies it to the registered NV12 device pointer
// (Phase 5 MVP — could be eliminated with a host-to-device pinned
// buffer; non-blocking for now).
//
// Returns the number of encoded bytes (>0 on success).
int64_t encode_rgba(NVencHandle* h, const uint8_t* rgba_host);

// Flush the encoder (write a few IDR frames to drain the pipeline),
// finalize the TS stream, and free the recorder.  After this call the
// handle is invalid.
int64_t finalize(NVencHandle* h);

}  // namespace swe2d_nvenc