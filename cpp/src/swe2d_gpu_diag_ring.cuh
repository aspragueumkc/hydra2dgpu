#pragma once
// Phase 4.1 — device-side diagnostic ring buffer.
//
// One record per simulation step. The host reads on demand (latest only
// for the GUI HUD; whole history for replay / batch post-processing).
// Mirrors the existing snapshot ring buffer in `swe2d_gpu.cu`.
#include <cstdint>

namespace swe2d_diag_ring {

// Per-step diagnostic record. Mirrors the structured fields the host
// already gets via pack_diag_kernel's result dict, plus t_s + dt_used.
struct DiagRecord {
    double t_s;             // simulation time at this step
    double dt_used;          // timestep taken
    int32_t gpu_active;      // 1 if GPU path active this step
    int32_t wet_cells;       // # wet cells
    double max_courant;      // max CFL this step
    double max_wse_error;    // max WSE error (from pack_diag_kernel)
    double mass_total;       // total mass (from pack_diag_kernel)
};

void init(int32_t initial_capacity);
void shutdown();

// Push the latest pack_diag values + step metadata into the ring.
// Caller responsibility to call once per step.
void push(
    double t_s,
    double dt_used,
    int32_t gpu_active,
    int32_t wet_cells,
    double max_courant,
    double max_wse_error,
    double mass_total);

int32_t count();

// Read the latest record into the host buffer.
void read_latest(DiagRecord* host_buf);

// Clear all records (next push starts at slot 0). Useful for new runs.
void clear();

}  // namespace swe2d_diag_ring