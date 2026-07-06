# On-Device Line Metrics — Kernel Interface Spec

## Goal

Replace the Python per-snapshot station interpolation + aggregate reduction with a device kernel that runs at `store_snapshot()` time. Zero Python computation per snapshot.

## Data Flow

```
init:  Python uploads sample map → device (once per run)
step:  backend.store_snapshot(t)  →  kernel reads h/hu/hv, writes line ring buffer
final: Python calls read_line_snapshots()  →  gets pre-computed arrays
```

## Sample Map Upload

New Python API exposed by the native module:

```c
// Upload per-line station data. Called once at init after the mesh is on device.
// Returns 0 on success.
int swe2d_gpu_configure_line_sampling(
    int                n_lines,           // number of sampled lines
    int const*         n_stations,        // [n_lines] stations per line
    int const*         cell_idx,          // flattened [sum(n_stations)] cell indices
    double const*      weights,           // flattened [sum(n_stations)] interpolation weights
    double const*      normal_x,          // [n_lines] edge normal x for flow computation
    double const*      normal_y,          // [n_lines] edge normal y
    double             gravity,           // model-unit gravity
    double             h_min,             // dry threshold
    int                max_snapshots      // ring buffer capacity
);
```

Each station is a (cell_idx, weight) pair. The kernel gathers `h/hu/hv` from that cell and blends linearly — no barycentric, just cell-centered value weighted by the station's interpolation weight. This matches the current Python `sample_line_aggregate_ts_row` which does `h[cell_idx]` with per-station `weights`.

## Kernel

Added at the end of `store_snapshot(t_accum)` — after the mesh ring buffer write:

```cuda
__global__ void compute_line_metrics_kernel(
    double const* h,              // (n_cells,) solver depth
    double const* hu,             // (n_cells,) solver discharge-x
    double const* hv,             // (n_cells,) solver discharge-y
    double const* cell_bed,       // (n_cells,) solver bed elevation
    int           n_lines,
    int const*    station_offsets, // [n_lines+1] prefix sum of n_stations
    int const*    cell_idx,       // [total_stations]
    double const* weights,        // [total_stations]
    double const* normal_x,       // [n_lines]
    double const* normal_y,       // [n_lines]
    double        gravity,
    double        h_min,
    int           snap_idx,       // current write position in ring buffer
    // Output ring buffers (pre-allocated, cyclical):
    double* line_ts_buf,          // [max_snaps × n_lines × 7]  TS fields
    double* line_prof_buf,        // [max_snaps × total_stations × 6]  profile fields
    int*    line_wet_buf          // [max_snaps × total_stations]  wet flags
);
```

### Per-station compute (one thread per station):

```cuda
int s = threadIdx.x + blockIdx.x * blockDim.x;
if (s >= total_stations) return;

int ci = cell_idx[s];
double hh = h[ci], huu = hu[ci], hvv = hv[ci], zb = cell_bed[ci];
int wet = (hh > h_min);
double safe_h = fmax(hh, 1e-12);
double vel = wet ? sqrt((huu/safe_h)*(huu/safe_h) + (hvv/safe_h)*(hvv/safe_h)) : 0.0;
double depth = hh - zb;
double wse = hh + zb;
double normal_v = wet ? (uu * nx + vv * ny) : 0.0;
double qn = wet ? hh * normal_v : 0.0;
double fr = wet ? vel / sqrt(fmax(gravity * hh, 1e-12)) : 0.0;

// Write profile fields
int po = snap_idx * total_stations * 6 + s * 6;
prof_buf[po + 0] = depth;     // depth_m
prof_buf[po + 1] = vel;       // velocity_ms
prof_buf[po + 2] = wse;       // wse_m
prof_buf[po + 3] = zb;        // bed_m
prof_buf[po + 4] = qn;        // flow_qn
prof_buf[po + 5] = fr;        // fr
wet_buf[snap_idx * total_stations + s] = wet;
```

### Per-line reduction (one thread block per line, or separate kernel launch):

- Weighted sum over stations in the line for: depth, velocity, wse, bed, fr
- Unweighted: wet_frac = mean(wet), flow_cms = sum(qn * weight)
- Write 7 scalars to `line_ts_buf[snap_idx * n_lines * 7 + line * 7 + field_offset]`

Total work: ~200 arithmetic ops per station + one reduction per line. A 100-line × 200-station config = 20K stations. GPU does this in < 50 µs.

## Ring Buffer

Same pattern as mesh snapshots — pre-allocated cyclical buffer with counter:

```c
struct LineMetricsRing {
    double* ts;       // [capacity × n_lines × 7]
    double* prof;     // [capacity × total_stations × 6]
    int*    wet;      // [capacity × total_stations]
    int     capacity;
    int     count;    // number of stored snapshots
    int     head;     // current write index (0..capacity-1)
};
```

## Readback

New Python API:

```python
def read_line_snapshots(self) -> dict:
    """Returns same keys as current Python-side live data, but all pre-computed."""
    return {
        "line_t_s":       np.ndarray (n_snaps,)          timestamps,
        "line_ts_data":   np.ndarray (n_snaps, n_lines, 7)   [depth, vel, wse, bed, flow, wet_frac, fr],
        "line_prof_data": np.ndarray (n_snaps, total_stations, 6) [depth, vel, wse, bed, qn, fr],
        "line_wet_data":  np.ndarray (n_snaps, total_stations)  wet flags,
        "station_offsets": np.ndarray (n_lines + 1,)      prefix sum for indexing profiles by line,
        "n_lines":        int,
        "station_m":      np.ndarray (total_stations,)    station distances (uploaded at init),
    }
```

Python side drops `populate_live_line_metrics`, `build_precomputed_line_results`, and the entire `_sample_line_metrics` closure — the data arrives already in viewer-compatible shape.

## Memory Budget

| Component | Size for 100 lines × 200 stations × 500 snaps |
|-----------|-----------------------------------------------|
| TS ring | 500 × 100 × 7 × 8 = 2.7 MB |
| Profile ring | 500 × 20000 × 6 × 8 = 480 MB |
| Wet ring | 500 × 20000 × 4 = 40 MB |
| Sample map | 20000 × (4 + 8) = 0.24 MB |
| **Total** | **~523 MB** |

Ring buffer capacity of 500 fits most runs. If memory is tight, reduce capacity or store profile as float32 (halves the 480 MB). The `h_min` threshold is a clamp that can be tuned — fewer wet stations once water drains.

## Implementation Order

1. Add `swe2d_gpu_configure_line_sampling()` upload function
2. Add `compute_line_metrics_kernel` — station gather + derived quantities
3. Add per-line reduction (block-level parallel sum)
4. Wire into `store_snapshot()` call site
5. Add `read_line_snapshots()` Python binding
6. Python: replace `populate_live_line_metrics` with direct readback
7. Delete `_sample_line_metrics` closure, `line_sampling_service.py`, `_agg_svc` code path
