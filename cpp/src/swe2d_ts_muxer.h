// Phase 5.1 — minimal MPEG Transport Stream (MPEG-TS) muxer.
//
// Writes a single-program, single-video-track .ts stream from H.264
// Annex B NALs.  No external library; ~200 lines of plain C.
//
// TS packet = 188 bytes:
//   byte  0       sync byte (0x47)
//   bytes 1-2     PID (13 bits) + flags (3 bits)
//   byte  3       adaptation_field_control + continuity_counter
//   bytes 4-187   payload (184 bytes)
//
// We emit three PID streams:
//   0x0001  PAT (Program Association Table)
//   0x0100  PMT (Program Map Table) — describes the video stream
//   0x0101  PES (Packetized Elementary Stream) — the H.264 NALs
//
// PES is wrapped around each H.264 access unit.  PCR is set in the
// adaptation field of every 7th PES packet (sufficient for the
// "diagnostic" use case; not strict broadcast spec).

#ifndef SWE2D_TS_MUXER_H
#define SWE2D_TS_MUXER_H

#include <stdint.h>
#include <stdio.h>

// Plain C struct — no extern "C" needed since this is consumed by the
// matching .c file.  The .cu wrapper (swe2d_gpu_viewer_nvenc.cu) calls
// the C functions directly; C++ name mangling is avoided by declaring
// them with extern "C" in the .cu file's include block if needed.
struct TsMuxer {
    FILE* fp;
    uint64_t total_bytes;
    uint32_t pcr_pid;
    uint32_t video_pid;
    uint32_t pcr_count;   // wraps every 32-bit

    // PCR / PTS / DTS are 33-bit, 90 kHz.  Track a monotonic counter.
    uint64_t pts_90khz;   // updated per frame
    uint32_t frame_count;
    int32_t fps;

    // 4-bit continuity counters (wrap 0-15).
    uint8_t pat_cc;
    uint8_t pmt_cc;
    uint8_t video_cc;
    uint8_t pcr_cc;
};

#ifdef __cplusplus
extern "C" {
#endif

// Open the .ts file and write the PAT + PMT (both 188 bytes).
int ts_open(struct TsMuxer* m, const char* path, int32_t fps);

// Write one H.264 access unit (one or more NALs prefixed with 00 00 00 01
// or 00 00 01).  `pts_90khz` is the presentation time in 90 kHz units.
// Adds a PCR insertion roughly every 7 PES packets.
int ts_write_access_unit(struct TsMuxer* m,
                          const uint8_t* nal_data, int32_t nal_size,
                          uint64_t pts_90khz, uint64_t dts_90khz);

// Close the file (flushes any pending bytes; MPEG-TS has no trailer).
int ts_close(struct TsMuxer* m);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SWE2D_TS_MUXER_H