// Phase 5.1 — minimal MPEG Transport Stream (MPEG-TS) muxer.
//
// Single-program, single-video-track .ts stream from H.264 Annex B NALs.
// Not a broadcast-spec implementation; sufficient for diagnostic recording
// (VLC / ffmpeg / mpv will play it).
//
// Packet structure (188 bytes each):
//   byte  0       sync (0x47)
//   bytes 1-2     PID (13 bits) + flags (3 bits: PUSI / random access / priority)
//   byte  3       adaptation_field_control (2 bits) + continuity_counter (4 bits)
//   bytes 4-187   payload (or adaptation field + payload)
//
// Streams emitted:
//   PID 0x0001  PAT  (one packet, points to PMT at 0x0100)
//   PID 0x0100  PMT  (one packet, H.264 video at 0x0101)
//   PID 0x0101  PES  (one or more packets per H.264 access unit)

#include "swe2d_ts_muxer.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

const uint8_t TS_SYNC = 0x47;
const uint32_t TS_PID_PAT  = 0x0001;
const uint32_t TS_PID_PMT  = 0x0100;
const uint32_t TS_PID_PCR  = 0x0101;
const uint32_t TS_PID_VIDEO = 0x0101;

const uint8_t H264_STREAM_TYPE = 0x1B;
const uint8_t PES_VIDEO = 0xE0;

inline void write16be(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)(v & 0xFF);
}

inline void write_pcr(uint8_t* payload, uint64_t pcr_27mhz) {
    // PCR is 42 bits base (27 MHz) + 9 bits extension (0).
    payload[0] = 0x80;  // PCR_flag = 1
    uint64_t base = pcr_27mhz >> 9;
    payload[1] = (uint8_t)(base >> 34);
    payload[2] = (uint8_t)(base >> 26);
    payload[3] = (uint8_t)(base >> 18);
    payload[4] = (uint8_t)(base >> 10);
    payload[5] = (uint8_t)(base >> 2);
    payload[6] = (uint8_t)((base << 6) | 0x00);
    payload[7] = 0x00;
}

inline void write_pts_dts(uint8_t* p, uint8_t marker, uint64_t ts_90khz) {
    p[0] = marker;
    p[1] = (uint8_t)(ts_90khz >> 29);
    p[2] = (uint8_t)(ts_90khz >> 22);
    p[3] = (uint8_t)(ts_90khz >> 14);
    p[4] = (uint8_t)(ts_90khz >> 7);
    p[5] = (uint8_t)((ts_90khz << 1) | 0x01);
}

void write_pat_packet(uint8_t* pkt) {
    pkt[0] = TS_SYNC;
    pkt[1] = 0x40;             // PUSI=1, PID high = 0x00
    pkt[2] = 0x01;             // PID low = 0x0001 (PAT)
    pkt[3] = 0x10;             // no adaptation field, CC=0
    // Section: table_id=0, section_length=9
    pkt[4] = 0x00;             // table_id = PAT
    pkt[5] = 0xB0 | 0x00;      // section_syntax_indicator=1, length high nibble
    pkt[6] = 0x09;             // section_length = 9
    // Program association body
    write16be(&pkt[7], 0x0001);  // transport_stream_id
    pkt[9]  = 0xC1;            // reserved + version=0 + current_next=1
    pkt[10] = 0x00;            // section_number
    pkt[11] = 0x00;            // last_section_number
    write16be(&pkt[12], 0x0001);  // program_number = 1
    pkt[14] = 0xE0 | 0x01;     // reserved + PMT pid high
    pkt[15] = 0x00;            // PMT pid low = 0x0100
    // Stuffing to 188 bytes
    memset(pkt + 16, 0xFF, 188 - 16);
}

void write_pmt_packet(uint8_t* pkt) {
    pkt[0] = TS_SYNC;
    pkt[1] = 0x41;             // PUSI=1, PID high = 0x01
    pkt[2] = 0x00;             // PID low = 0x0100 (PMT)
    pkt[3] = 0x10;             // no adaptation, CC=0
    // Section
    pkt[4] = 0x02;             // table_id = PMT
    pkt[5] = 0xB0 | 0x00;      // section_syntax, length high
    pkt[6] = 0x12;             // section_length = 18 bytes
    write16be(&pkt[7], 0x0001);  // program_number = 1
    pkt[9]  = 0xC1;            // reserved + version=0 + current_next=1
    pkt[10] = 0x00; pkt[11] = 0x00;  // section_number
    // PCR PID
    pkt[12] = 0xE0 | 0x01;     // reserved + PCR_PID high
    pkt[13] = 0x01;            // PCR_PID low = 0x0101
    // Elementary stream loop
    pkt[14] = H264_STREAM_TYPE;
    pkt[15] = 0xE0 | 0x01;     // reserved + ES_PID high
    pkt[16] = 0x01;            // ES_PID low = 0x0101
    write16be(&pkt[17], 0xF000);  // ES_info_length = 0 (no descriptors)
    // Stuffing
    memset(pkt + 19, 0xFF, 188 - 19);
}

inline int pes_header_len(bool has_dts) {
    // 9 (start code + flags + length + header_data_length) +
    //   5 (PTS) or 10 (PTS + DTS) - 3 (the bytes that PES length excludes)
    // = 14 with DTS, 9 with PTS only.
    return has_dts ? 14 : 9;
}

inline void write_pes_packet_first(
    uint8_t* pkt, uint64_t pts, uint64_t dts, bool has_dts,
    int payload_bytes_first_packet,
    int pes_total_len_field)
{
    pkt[0] = TS_SYNC;
    pkt[1] = 0x41;            // PUSI=1, PID=0x0101
    pkt[2] = 0x01;
    pkt[3] = 0x10;            // no adaptation, CC=0 (caller sets)
    // PES start code
    pkt[4] = 0x00; pkt[5] = 0x00; pkt[6] = 0x01; pkt[7] = PES_VIDEO;
    // PES packet length (PES bytes after this field).  0 if >65535.
    write16be(&pkt[8], pes_total_len_field);
    // PES header data
    pkt[10] = 0x80;           // '10' scrambling + '0' priority + '000' alignment + '0' copyright + '0' original_or_copy
    pkt[11] = 0x80;           // PTS_DTS_flags = '10' (PTS only)
    if (has_dts) pkt[11] = 0xC0;  // '11' = both PTS and DTS
    int header_data_len = has_dts ? 10 : 5;
    pkt[12] = (uint8_t)header_data_len;
    if (has_dts) {
        // PTS
        write_pts_dts(&pkt[13], 0x21, pts);
        // DTS
        write_pts_dts(&pkt[18], 0x11, dts);
    } else {
        write_pts_dts(&pkt[13], 0x21, pts);
    }
    (void)payload_bytes_first_packet;
}

inline void write_pes_packet_continue(
    uint8_t* pkt, uint8_t cc)
{
    pkt[0] = TS_SYNC;
    pkt[1] = 0x01;            // PUSI=0, PID=0x0101
    pkt[2] = 0x01;
    pkt[3] = 0x10 | (cc & 0x0F);
    // payload fills bytes 4..187
}

inline void write_pcr_packet(uint8_t* pkt, uint64_t pcr_27mhz, uint8_t cc) {
    pkt[0] = TS_SYNC;
    pkt[1] = 0x01; pkt[2] = 0x01; pkt[3] = 0x10 | (cc & 0x0F);
    // Adaptation field: 1 (length) + 6 (PCR base) + 1 (ext) + 1 (flags) = 9 bytes
    pkt[4] = 0x09;  // adaptation_field_length (including this byte)
    write_pcr(&pkt[5], pcr_27mhz);
    // 7 bytes PCR
    // After adaptation: 184 - (1 + 9) = 174 bytes of stuffing
    memset(pkt + 12, 0xFF, 188 - 12);
}

int ts_open(struct TsMuxer* m, const char* path, int32_t fps) {
    if (!m) return -1;
    memset(m, 0, sizeof(*m));
    m->fp = fopen(path, "wb");
    if (!m->fp) return -1;
    m->total_bytes = 0;
    m->frame_count = 0;
    m->pcr_count = 0;
    m->pts_90khz = 0;
    m->fps = (fps > 0) ? fps : 30;
    m->video_cc = 0;
    m->pat_cc = 0;
    m->pmt_cc = 0;
    m->pcr_cc = 0;

    // PAT
    uint8_t pat[188];
    write_pat_packet(pat);
    if (fwrite(pat, 1, 188, m->fp) != 188) { fclose(m->fp); m->fp = NULL; return -1; }
    m->total_bytes += 188;
    m->pat_cc = (m->pat_cc + 1) & 0x0F;

    // PMT
    uint8_t pmt[188];
    write_pmt_packet(pmt);
    if (fwrite(pmt, 1, 188, m->fp) != 188) { fclose(m->fp); m->fp = NULL; return -1; }
    m->total_bytes += 188;
    m->pmt_cc = (m->pmt_cc + 1) & 0x0F;
    return 0;
}

int ts_write_access_unit(struct TsMuxer* m,
                          const uint8_t* nal_data, int32_t nal_size,
                          uint64_t pts_90khz, uint64_t dts_90khz)
{
    if (!m || !m->fp || nal_size < 4) return -1;
    bool has_dts = (dts_90khz != pts_90khz);
    int hdr_len = pes_header_len(has_dts);

    // Compute total PES length (bytes after PES_length field).
    // = hdr_data_len (5 or 10) + NAL size.  PES_length is u16, 0 = unbounded.
    int hdr_data_len = has_dts ? 10 : 5;
    int total_pes_data = hdr_data_len + nal_size;
    uint16_t pes_length_field = (total_pes_data <= 65535)
        ? (uint16_t)total_pes_data : (uint16_t)0;
    int pes_packet_count = (4 + hdr_len + nal_size + 183) / 184;  // ceil

    // PCR every 7 access units
    bool need_pcr = ((m->frame_count % 7) == 0);
    if (need_pcr) {
        uint8_t pcr_pkt[188];
        write_pcr_packet(pcr_pkt, pts_90khz * 300, m->pcr_cc);
        if (fwrite(pcr_pkt, 1, 188, m->fp) != 188) return -1;
        m->total_bytes += 188;
        m->pcr_cc = (m->pcr_cc + 1) & 0x0F;
    }

    // First PES packet: TS header (4) + PES start (3) + PES flags (1) +
    //   PES length (2) + PES stream_id (0, already in pkt[7]) +
    //   PES header data (hdr_len) + NAL data
    // = 4 + 4 + 2 + 1 + hdr_len + NAL = 11 + hdr_len + NAL
    int first_payload_offset = 4 + 4 + 2 + 1 + hdr_len;  // 11 + hdr_len
    int first_nal_bytes = (188 - first_payload_offset) < nal_size
        ? (188 - first_payload_offset) : nal_size;
    {
        uint8_t pkt[188];
        memset(pkt, 0xFF, 188);
        pkt[0] = TS_SYNC;
        pkt[1] = 0x41; pkt[2] = 0x01;
        pkt[3] = 0x10 | (m->video_cc & 0x0F);
        // PES start code
        pkt[4] = 0x00; pkt[5] = 0x00; pkt[6] = 0x01; pkt[7] = PES_VIDEO;
        write16be(&pkt[8], pes_length_field);
        // PES header data
        pkt[10] = 0x80;
        if (has_dts) pkt[10] = 0xC0;
        pkt[11] = 0x80;
        if (has_dts) pkt[11] = 0xC0;
        pkt[12] = (uint8_t)hdr_data_len;
        if (has_dts) {
            write_pts_dts(&pkt[13], 0x21, pts_90khz);
            write_pts_dts(&pkt[18], 0x11, dts_90khz);
        } else {
            write_pts_dts(&pkt[13], 0x21, pts_90khz);
        }
        // First NAL bytes
        memcpy(pkt + first_payload_offset, nal_data, first_nal_bytes);
        if (fwrite(pkt, 1, 188, m->fp) != 188) return -1;
        m->total_bytes += 188;
        m->video_cc = (m->video_cc + 1) & 0x0F;
    }

    // Continuation packets
    int written = first_nal_bytes;
    while (written < nal_size) {
        uint8_t pkt[188];
        memset(pkt, 0xFF, 188);
        pkt[0] = TS_SYNC;
        pkt[1] = 0x01; pkt[2] = 0x01;
        pkt[3] = 0x10 | (m->video_cc & 0x0F);
        int copy = (184 < (nal_size - written)) ? 184 : (nal_size - written);
        memcpy(pkt + 4, nal_data + written, copy);
        if (fwrite(pkt, 1, 188, m->fp) != 188) return -1;
        m->total_bytes += 188;
        m->video_cc = (m->video_cc + 1) & 0x0F;
        written += copy;
    }

    m->frame_count++;
    m->pts_90khz = pts_90khz + (90000 / m->fps);
    (void)pes_packet_count;
    return 0;
}

int ts_close(struct TsMuxer* m) {
    if (!m || !m->fp) return -1;
    fclose(m->fp);
    m->fp = NULL;
    return 0;
}