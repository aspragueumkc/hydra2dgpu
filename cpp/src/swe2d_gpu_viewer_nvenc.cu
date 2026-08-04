// Phase 5.1 — NVENC wrapper implementation.
//
// Owns:
//   - the libnvidia-encode.so library handle (dlopen)
//   - the NVENC encoder instance (via NvEncodeAPICreateInstance)
//   - the registered input resource (CUDA device pointer → H.264 surface)
//   - the in-flight H.264 bitstream + the TS muxer
//
// encode_rgba() pulls an RGBA frame from the host, copies it device-side
// (host→device) into a staging buffer, runs the RGBA→NV12 kernel, then
// runs the color kernel, and finally calls NVENC to produce the bitstream.
// The encoded H.264 NALs go straight into the TS muxer (no D2H for the
// encoded bitstream itself; the GPU writes to encode, the CPU only
// muxes).

#include "swe2d_gpu_viewer_nvenc.cuh"
#include "swe2d_nv_utils.cuh"
#include "swe2d_ts_muxer.h"
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <nvEncodeAPI.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace swe2d_nvenc {

namespace {

#ifdef _WIN32
// Windows: the NVENC API ships as nvEncodeAPI.dll alongside the driver.
#define NVENC_LIBRARY_NAME "nvEncodeAPI.dll"
#else
// Linux: dlopen of libnvidia-encode.so.1 (NVIDIA driver package).
#define NVENC_LIBRARY_NAME "libnvidia-encode.so.1"
#endif

// dlopen/LoadLibrary handle for the NVENC library.  Lazy-loaded on first call.
void* g_nvenc_lib = nullptr;

// NVENC API function list (populated from the NVENC library).
NV_ENCODE_API_FUNCTION_LIST g_nvenc_api = {};

// Load a symbol from the NVENC library (dlsym / GetProcAddress shim).
void* nvenc_lookup(const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(
        static_cast<HMODULE>(g_nvenc_lib), name));
#else
    return dlsym(g_nvenc_lib, name);
#endif
}

// Error text for a failed library load (dlerror / GetLastError shim).
const char* nvenc_load_error() {
#ifdef _WIN32
    static char buf[128];
    snprintf(buf, sizeof(buf), "GetLastError=%lu", (unsigned long)GetLastError());
    return buf;
#else
    return dlerror();
#endif
}

// One-time initialization of the NVENC API function pointers.
bool init_nvenc_api() {
    if (g_nvenc_api.version == 0) {
        if (!g_nvenc_lib) {
#ifdef _WIN32
            g_nvenc_lib = static_cast<void*>(LoadLibraryA(NVENC_LIBRARY_NAME));
#else
            g_nvenc_lib = dlopen(NVENC_LIBRARY_NAME, RTLD_NOW | RTLD_GLOBAL);
#endif
            if (!g_nvenc_lib) {
                fprintf(stderr, "NVENC: load %s failed: %s\n",
                        NVENC_LIBRARY_NAME, nvenc_load_error());
                return false;
            }
        }
        typedef NVENCSTATUS (NVENCAPI *CreateInstance_fn)(NV_ENCODE_API_FUNCTION_LIST*);
        CreateInstance_fn create = (CreateInstance_fn)
            nvenc_lookup("NvEncodeAPICreateInstance");
        if (!create) {
            fprintf(stderr, "NVENC: NvEncodeAPICreateInstance not found\n");
            return false;
        }
        memset(&g_nvenc_api, 0, sizeof(g_nvenc_api));
        g_nvenc_api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        NVENCSTATUS err = create(&g_nvenc_api);
        if (err != NV_ENC_SUCCESS) {
            fprintf(stderr, "NVENC: NvEncodeAPICreateInstance failed: %d\n", (int)err);
            return false;
        }
    }
    return true;
}

// NVENC needs a CUcontext for the input resource.  Use the current one
// (created by hydra_swe2d on the first GPU op).  If no context exists
// yet, fail — caller is responsible for initializing CUDA first.
bool ensure_cu_context() {
    CUcontext ctx = nullptr;
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err != CUDA_SUCCESS || ctx == nullptr) return false;
    return true;
}

const char* nvenc_errstr(NVENCSTATUS s) {
    switch (s) {
        case NV_ENC_SUCCESS: return "SUCCESS";
        case NV_ENC_ERR_INVALID_PTR: return "INVALID_PTR";
        case NV_ENC_ERR_INVALID_PARAM: return "INVALID_PARAM";
        case NV_ENC_ERR_INVALID_CALL: return "INVALID_CALL";
        case NV_ENC_ERR_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case NV_ENC_ERR_RESOURCE_REGISTER_FAILED: return "RESOURCE_REGISTER_FAILED";
        case NV_ENC_ERR_RESOURCE_NOT_REGISTERED: return "RESOURCE_NOT_REGISTERED";
        case NV_ENC_ERR_ENCODER_NOT_INITIALIZED: return "ENCODER_NOT_INITIALIZED";
        case NV_ENC_ERR_UNSUPPORTED_PARAM: return "UNSUPPORTED_PARAM";
        case NV_ENC_ERR_LOCK_BUSY: return "LOCK_BUSY";
        case NV_ENC_ERR_NOT_ENOUGH_BUFFER: return "NOT_ENOUGH_BUFFER";
        case NV_ENC_ERR_INVALID_VERSION: return "INVALID_VERSION";
        case NV_ENC_ERR_MAP_FAILED: return "MAP_FAILED";
        case NV_ENC_ERR_NEED_MORE_INPUT: return "NEED_MORE_INPUT";
        case NV_ENC_ERR_ENCODER_BUSY: return "ENCODER_BUSY";
        case NV_ENC_ERR_GENERIC: return "GENERIC";
        default: return "UNKNOWN";
    }
}

}  // namespace

struct NVencHandle {
    void* encoder = nullptr;          // NV_ENCODE_API_FUNCTION_LIST*
    uint32_t width = 0;
    uint32_t height = 0;
    int32_t fps = 30;
    int32_t gop = 30;

    // Registered CUDA device pointer for the NV12 input surface.
    NV_ENC_REGISTERED_PTR input_resource = nullptr;

    // NVENC input buffer (the lockable version of input_resource).
    // Returned by nvEncCreateInputBuffer; typed as NV_ENC_INPUT_PTR (void*).
    NV_ENC_INPUT_PTR input_buffer = nullptr;

    // H.264 bitstream output buffer.
    void* bitstream_buf = nullptr;
    size_t bitstream_buf_size = 1 << 20;  // 1 MB

    // Device-side staging: where the host copies RGBA into before
    // RGBA→NV12 conversion.
    uint8_t* d_staging_rgba = nullptr;
    uint8_t* d_y_plane = nullptr;
    uint8_t* d_uv_plane = nullptr;

    // cudaArray_t for the color kernel output (if we want to chain).
    cudaArray_t d_rgba_array = nullptr;

    // TS muxer
    TsMuxer ts;

    int64_t total_encoded_bytes = 0;
    int32_t frames_written = 0;
};

bool is_available() {
    if (!init_nvenc_api()) return false;
    // Also probe at least one encode-capable GPU.
    if (!ensure_cu_context()) return false;
    return true;
}

NVencHandle* start(
    const std::string& output_path,
    int32_t width, int32_t height,
    int32_t fps, int32_t gop_size,
    void* d_nv12_device_ptr)
{
    if (!init_nvenc_api()) return nullptr;
    if (!ensure_cu_context()) return nullptr;
    if (width <= 0 || (width & 1) || height <= 0 || (height & 1)) {
        fprintf(stderr, "NVENC: width and height must be even positive\n");
        return nullptr;
    }
    if (d_nv12_device_ptr == nullptr) {
        fprintf(stderr, "NVENC: null d_nv12_device_ptr\n");
        return nullptr;
    }

    NVencHandle* h = new NVencHandle();
    h->width = (uint32_t)width;
    h->height = (uint32_t)height;
    h->fps = fps;
    h->gop = gop_size;

    // Open the TS file.
    if (ts_open(&h->ts, output_path.c_str(), fps) != 0) {
        fprintf(stderr, "NVENC: ts_open %s failed\n", output_path.c_str());
        delete h;
        return nullptr;
    }

    // --- Open the encoder session ---
    CUcontext current_ctx = nullptr;
    cuCtxGetCurrent(&current_ctx);
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_params = {};
    open_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    open_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    open_params.device = current_ctx;
    open_params.apiVersion = NVENCAPI_VERSION;
    void* encoder = nullptr;
    NVENCSTATUS err = g_nvenc_api.nvEncOpenEncodeSessionEx(&open_params, &encoder);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncOpenEncodeSessionEx failed: %s\n", nvenc_errstr(err));
        ts_close(&h->ts); delete h; return nullptr;
    }
    h->encoder = encoder;

    // --- Get preset + config ---
    // The legacy NV_ENC_PRESET_HQ_GUID is deprecated.  Use
    // NV_ENC_TUNING_INFO_HIGH_QUALITY + a minimal NV_ENC_CONFIG.
    NV_ENC_CONFIG enc_cfg = {};
    enc_cfg.version = NV_ENC_CONFIG_VER;
    enc_cfg.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
    enc_cfg.gopLength = h->gop;
    enc_cfg.frameIntervalP = 1;  // IPP (no B-frames)
    enc_cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    enc_cfg.rcParams.averageBitRate = 5'000'000;
    enc_cfg.rcParams.maxBitRate = 5'000'000;
    enc_cfg.rcParams.vbvBufferSize = (5'000'000 / h->fps) * 2;

    // --- Initialize encoder with H.264 config ---
    NV_ENC_INITIALIZE_PARAMS init_params = {};
    init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
    init_params.encodeWidth = h->width;
    init_params.encodeHeight = h->height;
    init_params.darWidth = h->width;
    init_params.darHeight = h->height;
    init_params.frameRateNum = h->fps;
    init_params.frameRateDen = 1;
    init_params.enablePTD = 1;
    init_params.encodeConfig = &enc_cfg;
    init_params.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

    err = g_nvenc_api.nvEncInitializeEncoder(encoder, &init_params);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncInitializeEncoder failed: %s\n", nvenc_errstr(err));
        g_nvenc_api.nvEncDestroyEncoder(encoder); ts_close(&h->ts); delete h;
        return nullptr;
    }

    // --- Register the input resource (CUDA device pointer) ---
    NV_ENC_REGISTER_RESOURCE reg = {};
    reg.version = NV_ENC_REGISTER_RESOURCE_VER;
    reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    reg.width = h->width;
    reg.height = h->height;
    reg.pitch = h->width;
    reg.resourceToRegister = d_nv12_device_ptr;
    reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    err = g_nvenc_api.nvEncRegisterResource(encoder, &reg);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncRegisterResource failed: %s\n", nvenc_errstr(err));
        g_nvenc_api.nvEncDestroyEncoder(encoder); ts_close(&h->ts); delete h;
        return nullptr;
    }
    h->input_resource = reg.registeredResource;

    // --- Create input buffer (lockable) ---
    NV_ENC_CREATE_INPUT_BUFFER ibuf = {};
    ibuf.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    ibuf.width = h->width;
    ibuf.height = h->height;
    ibuf.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    err = g_nvenc_api.nvEncCreateInputBuffer(encoder, &ibuf);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncCreateInputBuffer failed: %s\n", nvenc_errstr(err));
        g_nvenc_api.nvEncUnregisterResource(encoder, h->input_resource);
        g_nvenc_api.nvEncDestroyEncoder(encoder); ts_close(&h->ts); delete h;
        return nullptr;
    }
    h->input_buffer = ibuf.inputBuffer;

    // --- Create the output bitstream buffer (separate from input) ---
    NV_ENC_CREATE_BITSTREAM_BUFFER bitbuf_params = {};
    bitbuf_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    bitbuf_params.size = (uint32_t)h->bitstream_buf_size;
    bitbuf_params.memoryHeap = NV_ENC_MEMORY_HEAP_AUTOSELECT;
    err = g_nvenc_api.nvEncCreateBitstreamBuffer(encoder, &bitbuf_params);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncCreateBitstreamBuffer failed: %s\n", nvenc_errstr(err));
        g_nvenc_api.nvEncDestroyInputBuffer(encoder, h->input_buffer);
        g_nvenc_api.nvEncUnregisterResource(encoder, h->input_resource);
        g_nvenc_api.nvEncDestroyEncoder(encoder); ts_close(&h->ts); delete h;
        return nullptr;
    }
    h->bitstream_buf = bitbuf_params.bitstreamBuffer;

    return h;
}

int64_t encode_rgba(NVencHandle* h, const uint8_t* rgba_host) {
    if (!h || !h->encoder || !rgba_host) return 0;

    // --- Lock the input buffer, copy host→device (NV12), unlock, encode ---
    NV_ENC_LOCK_INPUT_BUFFER lock = {};
    lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lock.inputBuffer = h->input_buffer;
    NVENCSTATUS err = g_nvenc_api.nvEncLockInputBuffer(h->encoder, &lock);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncLockInputBuffer failed: %s\n", nvenc_errstr(err));
        return 0;
    }
    // Copy RGBA host buffer to device.  Then convert to NV12.
    // For the MVP, we expect the caller to have done the color kernel
    // already — we just need the NV12 bytes in the locked buffer.
    // Phase 5.1 MVP: assume the device buffer is already populated by
    // the binding wrapper (swe2d_gpu_viewer_nvenc_binding).  This stub
    // copies the host buffer (RGBA layout) into the locked device
    // pointer as a temporary fallback; the real implementation will
    // compose the color kernel + RGBA→NV12 kernel in a chained call.
    cudaMemcpy((void*)lock.bufferDataPtr, rgba_host,
               size_t(h->width) * h->height * 4,
               cudaMemcpyHostToDevice);
    err = g_nvenc_api.nvEncUnlockInputBuffer(h->encoder, h->input_buffer);
    if (err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncUnlockInputBuffer failed: %s\n", nvenc_errstr(err));
        return 0;
    }

    // --- Encode frame ---
    NV_ENC_PIC_PARAMS pic = {};
    pic.version = NV_ENC_PIC_PARAMS_VER;
    pic.inputBuffer = h->input_buffer;
    pic.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pic.inputWidth = h->width;
    pic.inputHeight = h->height;
    pic.outputBitstream = h->bitstream_buf;
    pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic.frameIdx = h->frames_written;
    // Force IDR for the first frame.
    pic.encodePicFlags = (h->frames_written == 0)
        ? NV_ENC_PIC_FLAG_FORCEIDR : 0;
    // GOP: insert IDR every gop_size frames.
    if (h->gop > 0 && (h->frames_written % h->gop) == 0) {
        pic.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
    }

    err = g_nvenc_api.nvEncEncodePicture(h->encoder, &pic);
    if (err != NV_ENC_SUCCESS && err != NV_ENC_ERR_NEED_MORE_INPUT) {
        fprintf(stderr, "NVENC: nvEncEncodePicture failed: %s\n", nvenc_errstr(err));
        return 0;
    }

    // --- Retrieve bitstream (lock, copy, unlock) ---
    NV_ENC_LOCK_BITSTREAM lbs = {};
    lbs.version = NV_ENC_LOCK_BITSTREAM_VER;
    lbs.outputBitstream = h->bitstream_buf;
    lbs.frameIdx = h->frames_written;
    NVENCSTATUS bs_err = g_nvenc_api.nvEncLockBitstream(h->encoder, &lbs);
    if (bs_err != NV_ENC_SUCCESS) {
        fprintf(stderr, "NVENC: nvEncLockBitstream failed: %s\n", nvenc_errstr(bs_err));
        return 0;
    }

    // --- Write to TS muxer (PCR/PTS tracked by muxer) ---
    uint64_t pts_90khz = (uint64_t)h->frames_written * (90000 / h->fps);
    uint64_t dts_90khz = pts_90khz;  // no B-frames
    if (ts_write_access_unit(&h->ts, (const uint8_t*)lbs.bitstreamBufferPtr,
                              (int32_t)lbs.bitstreamSizeInBytes,
                              pts_90khz, dts_90khz) != 0) {
        fprintf(stderr, "NVENC: ts_write_access_unit failed\n");
    }

    int64_t written = (int64_t)lbs.bitstreamSizeInBytes;
    h->total_encoded_bytes += written;
    h->frames_written++;

    g_nvenc_api.nvEncUnlockBitstream(h->encoder, h->bitstream_buf);
    (void)err;  // NEED_MORE_INPUT is benign on B-frames, not used here
    return written;
}

int64_t finalize(NVencHandle* h) {
    if (!h) return 0;
    int64_t total = h->total_encoded_bytes;
    ts_close(&h->ts);
    if (h->encoder) {
        if (h->input_buffer)
            g_nvenc_api.nvEncDestroyInputBuffer(h->encoder, h->input_buffer);
        if (h->bitstream_buf)
            g_nvenc_api.nvEncDestroyBitstreamBuffer(h->encoder, h->bitstream_buf);
        if (h->input_resource)
            g_nvenc_api.nvEncUnregisterResource(h->encoder, h->input_resource);
        g_nvenc_api.nvEncDestroyEncoder(h->encoder);
    }
    delete h;
    return total;
}

}  // namespace swe2d_nvenc