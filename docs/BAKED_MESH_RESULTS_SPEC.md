# Baked Mesh & Results — GPKG BLOB Storage Specification

No backward compatibility. This is a clean-sheet replacement of the GPKG persistence
layer. Old tables (`swe2d_mesh`, `swe2d_mesh_results`, `swe2d_mesh_results_runs`,
`swe2d_mesh_max_results`, `swe2d_coupling_results`, `swe2d_coupling_results_runs`,
`swe2d_line_results_ts`, `swe2d_line_results_profile`, `swe2d_line_results_runs`,
`swe2d_conservation_forensics`) become dead schema — the code that reads or writes
them will be deleted, not deprecated.

---

## 1. Problem Statement

### 1.1 Current Mesh Persistence

Mesh geometry is stored in `swe2d_mesh` as individually compressed numpy arrays:

```sql
CREATE TABLE swe2d_mesh (
    mesh_name TEXT PRIMARY KEY,
    node_x BLOB, node_y BLOB, node_z BLOB,       -- zlib(float64)
    cell_nodes BLOB, face_offsets BLOB,           -- zlib(int32)
    bc_n0 BLOB, bc_n1 BLOB, bc_type BLOB, bc_val BLOB,  -- zlib
    ...
);
```

On GPKG load, these arrays are decompressed and the C++ builder (`swe2d_build_mesh_poly`)
**re-runs** the entire mesh construction pipeline:

- Polygon CCW enforcement (may reverse node order)
- Edge connectivity dedup + normal computation
- Boundary edge classification
- **RCMK cell renumbering** (different permutation each run)
- **Edge reordering** for GPU coalescing
- **2-ring stencil** for LSQ gradient

Any divergence between the save-time and load-time arrays causes the **headless runner
to fail** while the in-memory workbench path works — because the workbench never
round-trips through the GPKG.

### 1.2 Current Results Persistence

Results stored row-by-row in `swe2d_mesh_results`:

```sql
CREATE TABLE swe2d_mesh_results (
    run_id TEXT, t_s REAL, cell_id INTEGER,
    h REAL, hu REAL, hv REAL,
    PRIMARY KEY (run_id, t_s, cell_id)
);
```

For a 100K-cell mesh × 100 timesteps = **10M rows**, plus metadata tables, max-tracking
tables, and a `build_mesh_snapshot_rows()` that converts in-memory tuples → list-of-dicts
→ SQL `executemany` of 10M row tuples.

The `load_mesh_snapshot()` overlay path requires:

1. `SELECT t_s ... ORDER BY ABS(t_s - ?)` — find nearest timestep
2. `SELECT h, hu, hv ... WHERE run_id=? AND t_s=? ORDER BY cell_id` — fetch 100K rows
3. Python list comprehension: `np.asarray([float(r[0]) for r in rows])` — O(n) loop

### 1.3 Current GPKG Tables (All — to be Deleted)

| Table | Purpose |
|-------|---------|
| `swe2d_mesh` | Individual compressed array columns |
| `swe2d_mesh_results` | 1 row per (run, t_s, cell) |
| `swe2d_mesh_results_runs` | Run metadata |
| `swe2d_mesh_max_results` | Per-cell max tracking |
| `swe2d_mesh_max_results_runs` | Max tracking metadata |
| `swe2d_coupling_results` | 1 row per (run, t_s, component, object_id, metric) |
| `swe2d_coupling_results_runs` | Coupling run metadata |
| `swe2d_line_results_ts` | 1 row per (run, t_s, line) |
| `swe2d_line_results_profile` | 1 row per (run, t_s, line, station) |
| `swe2d_line_results_runs` | Line run metadata |
| `swe2d_conservation_forensics` | Per-timestep conservation accounting |

**11 tables → 5.**

---

## 2. Proposed Architecture

### 2.1 Principle

Replace every per-row table with **raw-byte BLOBs**. SQLite BLOBs accept arbitrary byte
sequences — no `pickle`, no `zlib`, no per-element iteration. `np.frombuffer()` restores
arrays in near-zero-copy time. The in-memory format (numpy arrays) **is** the persisted
format (raw bytes of those same arrays).

### 2.2 Target GPKG Schema — Five Tables

```sql
-- ─────────────────────────────────────────────────────────────
-- Table 1: Baked mesh (fully constructed SWE2DMesh, post C++)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE swe2d_baked_mesh (
    mesh_name    TEXT PRIMARY KEY,
    n_nodes      INTEGER NOT NULL,
    n_cells      INTEGER NOT NULL,
    n_edges      INTEGER NOT NULL,
    crs_wkt      TEXT DEFAULT '',
    created_utc  TEXT NOT NULL,
    baked_blob   BLOB NOT NULL       -- serialized SWE2DMesh (§3)
);

-- ─────────────────────────────────────────────────────────────
-- Table 2: Baked mesh results (all timesteps + per-step GPU max
--          tracking, one row per run)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE swe2d_baked_results (
    run_id       TEXT PRIMARY KEY,
    mesh_name    TEXT NOT NULL,
    n_cells      INTEGER NOT NULL,
    n_timesteps  INTEGER NOT NULL,
    created_utc  TEXT NOT NULL,
    times_blob   BLOB NOT NULL,      -- n_timesteps × float64
    h_blob       BLOB NOT NULL,      -- n_timesteps × n_cells × float64
    hu_blob      BLOB NOT NULL,      -- n_timesteps × n_cells × float64
    hv_blob      BLOB NOT NULL,      -- n_timesteps × n_cells × float64
    max_h_blob   BLOB,               -- n_cells × float64, GPU per-step max
    max_hu_blob  BLOB,               -- n_cells × float64, GPU per-step max
    max_hv_blob  BLOB                -- n_cells × float64, GPU per-step max
);

-- ─────────────────────────────────────────────────────────────
-- Table 3: Baked coupling (per-object timeseries, one row per
--          (run, component, object_id, metric))
-- ─────────────────────────────────────────────────────────────
CREATE TABLE swe2d_baked_coupling (
    run_id       TEXT,
    component    TEXT,       -- "drainage_node", "drainage_link", "structure"
    object_id    TEXT,
    object_name  TEXT,
    metric       TEXT,       -- "depth", "flow", "invert", "length"
    n_timesteps  INTEGER,
    times_blob   BLOB,       -- float64 [n_timesteps]
    values_blob  BLOB,       -- float64 [n_timesteps]
    PRIMARY KEY (run_id, component, object_id, metric)
);

-- ─────────────────────────────────────────────────────────────
-- Table 4: Baked line timeseries (per-line, one row per line)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE swe2d_baked_line_ts (
    run_id        TEXT,
    line_id       INTEGER,
    line_name     TEXT,
    n_timesteps   INTEGER,
    times_blob    BLOB,      -- float64 [n_timesteps]
    depth_blob    BLOB,      -- float64 [n_timesteps]
    vel_blob      BLOB,
    wse_blob      BLOB,
    bed_blob      BLOB,
    flow_blob     BLOB,
    wet_frac_blob BLOB,
    fr_blob       BLOB,
    PRIMARY KEY (run_id, line_id)
);

-- ─────────────────────────────────────────────────────────────
-- Table 5: Baked line profiles (per-line cross-section,
--          one row per line, 2-D timestep×station arrays)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE swe2d_baked_line_profiles (
    run_id        TEXT,
    line_id       INTEGER,
    line_name     TEXT,
    n_stations    INTEGER,
    n_timesteps   INTEGER,
    station_blob  BLOB,       -- float64 [n_stations] (fixed geometry)
    times_blob    BLOB,       -- float64 [n_timesteps]
    depth_blob    BLOB,       -- float64 [n_timesteps × n_stations]
    vel_blob      BLOB,
    wse_blob      BLOB,
    bed_blob      BLOB,
    flow_qn_blob  BLOB,
    fr_blob       BLOB,
    wet_blob      BLOB,       -- int32 [n_timesteps × n_stations]
    PRIMARY KEY (run_id, line_id)
);
```

**Five tables.** No `_runs` metadata tables, no max-tracking tables, no per-cell or
per-row indexing, no `ORDER BY cell_id`, no conservation forensics.

---

## 3. C++ Mesh Serialization

### 3.1 Serialization Format

A simple binary format: each `std::vector` member is stored as a **8-byte count prefix**
followed by **raw element bytes**. Scalars stored inline.

```
[n_nodes : int32]                         4 bytes
[n_cells : int32]                         4 bytes
[n_edges : int32]                         4 bytes

[count : uint64][node_x : double × count] 8 + N×8 bytes
[count : uint64][node_y : double × count] 8 + N×8 bytes
[count : uint64][node_z : double × count] 8 + N×8 bytes

[count : uint64][cell_face_offsets : int32 × count]  8 + (C+1)×4
[count : uint64][cell_face_nodes : int32 × count]    8 + sum(nv)×4
[count : uint64][cell_edge_offsets : int32 × count]  8 + (C+1)×4
[count : uint64][cell_edge_ids : int32 × count]      8 + sum(nv)×4

[count : uint64][cell_cx : double × count]   8 + C×8
[count : uint64][cell_cy : double × count]   8 + C×8
[count : uint64][cell_area : double × count] 8 + C×8
[count : uint64][cell_zb : double × count]   8 + C×8

[count : uint64][cell_inv_area : double × count]  8 + C×8

[count : uint64][edge_c0 : int32 × count]    8 + E×4
[count : uint64][edge_c1 : int32 × count]    8 + E×4
[count : uint64][edge_n0 : int32 × count]    8 + E×4
[count : uint64][edge_n1 : int32 × count]    8 + E×4
[count : uint64][edge_nx : double × count]   8 + E×8
[count : uint64][edge_ny : double × count]   8 + E×8
[count : uint64][edge_len : double × count]  8 + E×8
[count : uint64][edge_bc : int32 × count]    8 + E×4   (BCType → int32)
[count : uint64][edge_bc_val : double × count]  8 + E×8

[count : uint64][cell_perm : int32 × count]  8 + C×4

[count : uint64][cell_ring2_offsets : int32 × count]   8 + (C+1)×4
[count : uint64][cell_ring2_ids : int32 × count]       8 + R×4
[count : uint64][cell_ring2_dcx : double × count]      8 + R×8
[count : uint64][cell_ring2_dcy : double × count]      8 + R×8
[count : uint64][cell_ring2_inv_dist2 : double × count] 8 + R×8
```

Where N = n_nodes, C = n_cells, E = n_edges, R = sum(ring2_counts).

### 3.2 C++ API

```cpp
// swe2d_mesh.hpp
std::vector<uint8_t> swe2d_serialize_mesh(const SWE2DMesh& mesh);
SWE2DMesh swe2d_deserialize_mesh(const uint8_t* data, size_t size);
```

### 3.3 pybind11 Bindings

```python
blob: bytes   = mod.swe2d_serialize_mesh(pymesh_handle)
pymesh        = mod.swe2d_deserialize_mesh(blob)
```

Python accessor properties on `PyMesh` (needed for post-hoc line resampling and for
`SWE2DBackend.build_mesh_from_baked()` to repopulate host-side state):

```python
pm.node_x, pm.node_y, pm.node_z       # → np.array float64 (N,)
pm.cell_face_offsets                   # → np.array int32 (M+1,) or None
pm.cell_face_nodes                     # → np.array int32 (K,) or None
pm.cell_nodes                          # → np.array int32 (M*3,) for tri meshes
pm.cell_zb, pm.cell_cx, pm.cell_cy    # → np.array float64 (M,)
pm.cell_area, pm.cell_inv_area         # → np.array float64 (M,)
pm.cell_perm                           # → np.array int32 (M,) RCMK perm
```

### 3.4 Estimated Size

For a 100K-cell poly mesh (~200K edges): **≈ 20–25 MB**.

---

## 4. Python Persistence

### 4.1 Baked Mesh — Save (at mesh-build time)

```python
def persist_baked_mesh(gpkg_path, mesh_name, pymesh_handle, mod, crs_wkt=""):
    blob = mod.swe2d_serialize_mesh(pymesh_handle)
    info = mod.swe2d_mesh_info(pymesh_handle)
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS swe2d_baked_mesh (
            mesh_name TEXT PRIMARY KEY, n_nodes INTEGER, n_cells INTEGER,
            n_edges INTEGER, crs_wkt TEXT, created_utc TEXT, baked_blob BLOB)
    """)
    cur.execute("""
        INSERT OR REPLACE INTO swe2d_baked_mesh VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (mesh_name, info["n_nodes"], info["n_cells"], info["n_edges"],
          str(crs_wkt), datetime.now(timezone.utc).isoformat(), blob))
    conn.commit()
    conn.close()
```

### 4.2 Baked Mesh — Load (for headless runner / resampling)

```python
def load_baked_mesh(gpkg_path, mesh_name, mod):
    row = conn.execute(
        "SELECT baked_blob FROM swe2d_baked_mesh WHERE mesh_name=?", (mesh_name,)
    ).fetchone()
    return mod.swe2d_deserialize_mesh(row[0]) if row else None
```

### 4.3 Baked Results — Save (at end of run)

```python
def persist_baked_results(gpkg_path, run_id, mesh_name, snapshot_timesteps,
                           max_tracking=None):
    """snapshot_timesteps: list of (t_s, h_arr, hu_arr, hv_arr)

    max_tracking: optional dict from backend.get_max_tracking() with
                  keys "max_h", "max_hu", "max_hv" — GPU per-step maxima.
                  If provided, stored as extra BLOB columns.
    """
    times = np.array([t for t, _, _, _ in snapshot_timesteps], dtype=np.float64)
    n_steps = len(snapshot_timesteps)
    n_cells = snapshot_timesteps[0][1].size

    h_all = np.zeros(n_steps * n_cells, dtype=np.float64)
    hu_all = np.zeros(n_steps * n_cells, dtype=np.float64)
    hv_all = np.zeros(n_steps * n_cells, dtype=np.float64)
    for i, (_, h, hu, hv) in enumerate(snapshot_timesteps):
        s, e = i * n_cells, (i + 1) * n_cells
        h_all[s:e]  = np.asarray(h,  dtype=np.float64).ravel()
        hu_all[s:e] = np.asarray(hu, dtype=np.float64).ravel()
        hv_all[s:e] = np.asarray(hv, dtype=np.float64).ravel()

    max_h  = max_tracking.get("max_h")  if max_tracking else None
    max_hu = max_tracking.get("max_hu") if max_tracking else None
    max_hv = max_tracking.get("max_hv") if max_tracking else None

    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS swe2d_baked_results (
            run_id TEXT PRIMARY KEY, mesh_name TEXT,
            n_cells INTEGER, n_timesteps INTEGER, created_utc TEXT,
            times_blob BLOB, h_blob BLOB, hu_blob BLOB, hv_blob BLOB,
            max_h_blob BLOB, max_hu_blob BLOB, max_hv_blob BLOB)
    """)
    cur.execute("""
        INSERT OR REPLACE INTO swe2d_baked_results
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, mesh_name, n_cells, n_steps,
          datetime.now(timezone.utc).isoformat(),
          times.tobytes(), h_all.tobytes(), hu_all.tobytes(), hv_all.tobytes(),
          max_h.tobytes() if max_h is not None else None,
          max_hu.tobytes() if max_hu is not None else None,
          max_hv.tobytes() if max_hv is not None else None))
    conn.commit()
    conn.close()
```

### 4.4 Baked Results — Load (overlay / resampling)

```python
def load_baked_snapshot(gpkg_path, run_id, t_s):
    row = conn.execute(
        "SELECT n_timesteps, n_cells, times_blob, h_blob, hu_blob, hv_blob "
        "FROM swe2d_baked_results WHERE run_id=?", (run_id,)
    ).fetchone()
    if not row:
        return None
    n_steps, n_cells = int(row[0]), int(row[1])
    times  = np.frombuffer(row[2], dtype=np.float64)
    h_all  = np.frombuffer(row[3], dtype=np.float64).reshape(n_steps, n_cells)
    hu_all = np.frombuffer(row[4], dtype=np.float64).reshape(n_steps, n_cells)
    hv_all = np.frombuffer(row[5], dtype=np.float64).reshape(n_steps, n_cells)
    i = int(np.argmin(np.abs(times - t_s)))
    return {"t_s": float(times[i]), "h": h_all[i].copy(),
            "hu": hu_all[i].copy(), "hv": hv_all[i].copy(), "cell_count": n_cells}
```

### 4.5 Max Tracking — GPU Per-Step Maxima (Persisted)

The GPU update kernel tracks per-cell maxima on **every simulation timestep**
(not just snapshot intervals). At `swe2d_gpu.cu:2263`:

```c
if (h_new > d_max_h[c])  d_max_h[c]  = h_new;
if (hu_new > d_max_hu[c]) d_max_hu[c] = hu_new;
if (hv_new > d_max_hv[c]) d_max_hv[c] = hv_new;
```

These device arrays are downloaded at finalize time via
`backend.get_max_tracking()` and stored as optional BLOB columns
(`max_h_blob`, `max_hu_blob`, `max_hv_blob`) on `swe2d_baked_results`.

Overhead: 3 × n_cells × 8 bytes = ~2.4 MB for 100K cells. The GPU arrays
are already allocated and updated at zero additional kernel cost — this is
just persisting what the GPU already computes.

If the BLOB columns are NULL (e.g. older runs or runs where max tracking
was disabled), the viewer falls back to snapshot-resolution max:

```python
def load_max_tracking(row):
    """row: fetched row from swe2d_baked_results.
       Returns per-cell max dict from GPU data if available,
       otherwise computes from snapshot arrays."""
    if row[9] is not None:  # max_h_blob
        return {
            "max_h":  np.frombuffer(row[9],  dtype=np.float64),
            "max_hu": np.frombuffer(row[10], dtype=np.float64),
            "max_hv": np.frombuffer(row[11], dtype=np.float64),
        }
    # Fallback: snapshot-resolution max
    h_all  = np.frombuffer(row[5], dtype=np.float64).reshape(n_ts, n_cells)
    hu_all = np.frombuffer(row[6], dtype=np.float64).reshape(n_ts, n_cells)
    hv_all = np.frombuffer(row[7], dtype=np.float64).reshape(n_ts, n_cells)
    return {
        "max_h":  np.max(h_all, axis=0),
        "max_hu": np.max(hu_all, axis=0),
        "max_hv": np.max(hv_all, axis=0),
    }

### 4.6 Baked Coupling — Save

```python
def persist_baked_coupling(gpkg_path, run_id, component, object_id,
                           object_name, metric, times, values):
    conn.execute("""
        INSERT OR REPLACE INTO swe2d_baked_coupling
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, component, object_id, object_name, metric,
          len(times), times.tobytes(), values.tobytes()))
```

### 4.7 Baked Coupling — Load

```python
def load_baked_coupling_timeseries(gpkg_path, run_id, component, object_id, metric):
    row = conn.execute(
        "SELECT times_blob, values_blob FROM swe2d_baked_coupling "
        "WHERE run_id=? AND component=? AND object_id=? AND metric=?",
        (run_id, component, object_id, metric)
    ).fetchone()
    return (np.frombuffer(row[0], dtype=np.float64),
            np.frombuffer(row[1], dtype=np.float64)) if row else (None, None)
```

### 4.8 Baked Line TS — Save

```python
def persist_baked_line_ts(gpkg_path, run_id, line_id, line_name,
                          times, depth, vel, wse, bed, flow, wet_frac, fr):
    conn.execute("""
        INSERT OR REPLACE INTO swe2d_baked_line_ts
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, line_id, line_name, len(times),
          times.tobytes(), depth.tobytes(), vel.tobytes(),
          wse.tobytes(), bed.tobytes(), flow.tobytes(),
          wet_frac.tobytes(), fr.tobytes()))
```

### 4.9 Baked Line Profile — Save

```python
def persist_baked_line_profile(gpkg_path, run_id, line_id, line_name,
                               station_m, times, depth, vel, wse, bed,
                               flow_qn, fr, wet):
    conn.execute("""
        INSERT OR REPLACE INTO swe2d_baked_line_profiles
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, line_id, line_name,
          len(station_m), len(times),
          station_m.tobytes(), times.tobytes(),
          depth.tobytes(), vel.tobytes(), wse.tobytes(), bed.tobytes(),
          flow_qn.tobytes(), fr.tobytes(), wet.tobytes()))
```

### 4.10 Baked Line / Profile — Load (Unified, Live-or-GPKG)

```python
def load_baked_line_timeseries(source, run_id, line_id):
    """source: GPKG path str, or SWE2DResultsData with live arrays."""
    if isinstance(source, str):
        row = conn.execute(
            "SELECT n_timesteps, times_blob, depth_blob, vel_blob, "
            "wse_blob, bed_blob, flow_blob "
            "FROM swe2d_baked_line_ts WHERE run_id=? AND line_id=?",
            (run_id, line_id)
        ).fetchone()
        return {
            "t_s": np.frombuffer(row[1], dtype=np.float64),
            "depth_m": np.frombuffer(row[2], dtype=np.float64),
            "velocity_ms": np.frombuffer(row[3], dtype=np.float64),
            "wse_m": np.frombuffer(row[4], dtype=np.float64),
            "bed_m": np.frombuffer(row[5], dtype=np.float64),
            "flow_cms": np.frombuffer(row[6], dtype=np.float64),
        } if row else {}
    # Live path: arrays stored directly on data object
    return source.get_line_ts_arrays(run_id, line_id)

def load_baked_line_profile(source, run_id, line_id, t_sec):
    if isinstance(source, str):
        row = conn.execute(
            "SELECT n_stations, n_timesteps, station_blob, times_blob, "
            "wse_blob, bed_blob, depth_blob FROM swe2d_baked_line_profiles "
            "WHERE run_id=? AND line_id=?", (run_id, line_id)
        ).fetchone()
        if not row:
            return {}
        n_sta, n_ts = int(row[0]), int(row[1])
        stations = np.frombuffer(row[2], dtype=np.float64)
        times    = np.frombuffer(row[3], dtype=np.float64)
        i = int(np.argmin(np.abs(times - t_sec)))
        return {
            "station_m": stations,
            "wse_m":   np.frombuffer(row[4], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "bed_m":   np.frombuffer(row[5], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "depth_m": np.frombuffer(row[6], dtype=np.float64).reshape(n_ts, n_sta)[i],
        }
    # Live path
    return source.get_line_profile_arrays(run_id, line_id, t_sec)
```

---

## 5. Pipeline Integration Points

### 5.1 Baked Mesh — Save

After `backend.build_mesh()` succeeds, before `backend.initialize()`:

```python
backend.build_mesh(...)
persist_baked_mesh(gpkg_path, mesh_name, backend._mesh_h, mod)
backend.initialize(...)
```

For headless runner: same insertion point in `backend_initializer.build_and_initialize()`.

### 5.2 Baked Mesh — Load (headless runner)

```python
backend.build_mesh_from_baked(baked_blob)
# No C++ builder re-entry. Mesh is identical to original.
```

### 5.3 Baked Results — Save

In `run_finalizer.finalize_and_persist()`, replace:

```python
# Old:
rows = self._view.build_mesh_snapshot_rows()
self._view.persist_mesh_results(gpkg, run_id, rows, interval_s=...)

# New:
snapshots = _results_data.get_live_snapshot_timesteps()
max_tracking = backend.get_max_tracking() if backend is not None else None
persist_baked_results(gpkg, run_id, mesh_name, snapshots,
                       max_tracking=max_tracking)
```

Same pattern for coupling (`persist_baked_coupling`) and line results
(`persist_baked_line_ts` + `persist_baked_line_profile`).

### 5.4 Conservation Forensics — Removed

The `swe2d_conservation_forensics` table and all code that generates/reads it is
deleted. The final conservation summary (storage delta, source total, boundary
flux) is already recorded in the run log — that's sufficient.

### 5.5 Overlay Load

`overlay_controller.load_mesh_snapshot_for_overlay()` → `load_baked_snapshot()`.

### 5.6 Live Data Model

`SWE2DResultsData` stores live snapshots as numpy arrays, not list-of-dicts:

```python
# New live storage (same shape as GPKG blobs):
self._live_times: np.ndarray = np.empty(0)
self._live_h: np.ndarray = np.empty((0, n_cells))
self._live_hu: np.ndarray = np.empty((0, n_cells))
self._live_hv: np.ndarray = np.empty((0, n_cells))
self._live_line_ts: Dict[int, Dict[str, np.ndarray]] = {}
self._live_line_profile: Dict[int, Dict[str, np.ndarray]] = {}
self._live_coupling: Dict[Tuple[str, str, str], np.ndarray] = {}
```

The viewer calls `load_baked_line_timeseries(data, ...)` whether data is live or GPKG.
No `_data_source` branching.

### 5.7 Viewer Simplification (Before / After)

Every viewer/plotting file currently branches on `data_source`:

**`studio_viewer_pg.py` — line timeseries loading (`_load_timeseries_for_type`):**

```python
# Before: 2 import + branching + 2 code paths
from swe2d.results.queries import load_timeseries as _load_ts
from swe2d.results.queries import load_timeseries_from_live as _load_ts_live
is_live = getattr(self._result_data, "data_source", "") == "live"
raw = (_load_ts_live(data, run_id, lid) if is_live
       else _load_ts(gpkg_path, run_id, lid))

# After: 1 import, no branch
from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
raw = load_baked_line_timeseries(data, run_id, lid)
# data is SWE2DResultsData (live) or a GPKG path string — same function
```

**`studio_viewer_profile_pg.py` — profile loading:**

```python
# Before: 4 imports + branching
from swe2d.results.queries import (
    find_nearest_timestep, load_profile, load_profile_from_live,
    load_structure_flows_at_time,
)
t = find_nearest_timestep(gpkg_path, run_id, line_id, t_sec)
prof_data = (load_profile_from_live(data, run_id, line_id, t)
             if is_live else load_profile(gpkg_path, run_id, line_id, t))

# After: 1 import, no branching, no separate nearest-ts query
from swe2d.services.gpkg_persistence_service import load_baked_line_profile
prof_data = load_baked_line_profile(data, run_id, line_id, t_sec)
# find_nearest_timestep is internal to load_baked_line_profile (np.argmin)
```

**`results_render_service.py` — timeseries rendering:**

```python
# Before: same branching pattern as studio_viewer_pg
from swe2d.results.queries import load_timeseries as _load_ts
from swe2d.results.queries import load_timeseries_from_live as _load_ts_live
raw = (_load_ts_live(...) if is_live else _load_ts(...))

# After:
from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
raw = load_baked_line_timeseries(data_or_path, run_id, line_id)
```

**`structure_service.py` — structure flow overlay:**

```python
# Before: imports find_nearest_timestep + load_structure_flows_at_time
# Both do SQL queries that need discovery of shared/legacy table names
t = find_nearest_timestep(gpkg_path, run_id, line_id, t_sec)
flows = load_structure_flows_at_time(gpkg_path, run_id, t_sec)

# After: single function, no table-name discovery
from swe2d.services.gpkg_persistence_service import (
    load_baked_structure_flows_at_time
)
flows = load_baked_structure_flows_at_time(gpkg_path, run_id, t_sec)
# Uses np.argmin internally, no separate nearest-ts step
```

**`SWE2DResultsData.load_timeseries()` (in `results/data.py`):**

```python
# Before: branches on _data_source, imports load_timeseries / load_timeseries_from_live
# After: delegates to load_baked_line_timeseries, same function for both sources
def load_timeseries(self, run_record, line_id, var_key):
    raw = load_baked_line_timeseries(self, run_record.run_id, line_id)
    #                              ^ passes self (SWE2DResultsData) as source
    return raw if raw else {}
```

### 5.8 Files That Become Simpler

| File | Current | After | Imports removed |
|------|---------|-------|-----------------|
| `swe2d/workbench/views/studio_viewer_pg.py` | branches on `data_source`, 2 imports | no branch, 1 import | `load_timeseries`, `load_timeseries_from_live` |
| `swe2d/workbench/views/studio_viewer_profile_pg.py` | branches, 4 imports | no branch, 1 import | `find_nearest_timestep`, `load_profile`, `load_profile_from_live`, `load_structure_flows_at_time` |
| `swe2d/services/results_render_service.py` | branches, 2 imports | no branch, 1 import | `load_timeseries`, `load_timeseries_from_live` |
| `swe2d/results/structure_service.py` | imports `_find_prefixed_or_default_table`, `find_nearest_timestep`, `load_structure_flows_at_time` | imports `load_baked_structure_flows_at_time` | 3 legacy imports |
| `swe2d/results/data.py` | `_load_ts`/`_load_ts_live` branching in `load_timeseries()` | delegates to `load_baked_line_timeseries` | 2 legacy imports |
| `swe2d/results/timestep_service.py` | imports from `queries` | direct `load_baked_*` calls | various |

### 5.9 Temporal Controls — Slider → Overlay → Plot Chain

The current timestep propagation chain branches three ways on data source:

**`on_results_panel_timestep_changed()` (in `studio_results_panel.py:179`):**

```python
# Current: branches on live-vs-gpkg, snapshot availability, overlay state
# Step 1: sync slider
temporal.on_timestep_changed(t_s, frame_idx)

# Step 2: overlay — 3 code paths
_snapshots = getattr(results_data, "get_live_snapshot_timesteps", lambda: [])()
n_ts = len(_snapshots)
data_source = getattr(results_data, "data_source", "none")
if n_ts > 0 and data_source != "gpkg":
    # Live path: overlay data already in memory
    dialog._update_high_perf_overlay_time(float(t_s))
else:
    # GPKG path: load from GPKG, then update
    dialog._overlay_controller.load_mesh_snapshot_for_overlay(t_s)
    dialog._update_high_perf_overlay_time(float(t_s))

# Step 3: refresh plots
viewer.refresh()
```

**With baked format, this collapses to:**

```python
# After: no branching, single code path
temporal.on_timestep_changed(t_s, frame_idx)

# Overlay: load_baked_snapshot handles live & GPKG transparently
dialog._overlay_controller.load_mesh_snapshot_for_overlay(t_s)
dialog._update_high_perf_overlay_time(float(t_s))

viewer.refresh()
```

**Timestep array population** (`set_live_snapshot_timesteps` in `results/data.py`):

```python
# Current: extracts t_s from list of tuples
t_arr = np.array([float(s[0]) for s in snapshot_timesteps], dtype=np.float64)
self._all_timesteps = t_arr

# After: _live_times is already the numpy array
self._all_timesteps = self._live_times.copy()
```

**Frame count → slider range** (`temporal_dock.py:87`):

```python
# Current (unchanged — already consumed data.frame_count):
self._time_slider.setRange(0, max(0, data.frame_count - 1))

# frame_count comes from _all_timesteps.size, which is the same
# numpy array whether populated from live snapshots or loaded from GPKG.
```

**Animation controller** (`animation.py`):

The `ResultsAnimationController.set_timesteps()` receives the numpy array and emits
`current_timestep_changed(t_s, frame_idx)`. This signal is already data-source-agnostic
— it just carries a float and an int. No changes needed to the animation controller itself.

### 5.10 Run Discovery — "Add Results from GPKG" Button

The current discovery pipeline queries the **old** per-row table schemas:

```
studio_results_panel._on_results_add()
  → collect_runs_from_gpkg(gpkg_path)          [run_service.py]
    → discover_line_result_runs(gpkg_path)      [queries.py — ~100 lines]
      Scans swe2d_line_results_ts* table names
      Checks shared vs legacy per-run variants
      Queries swe2d_line_results_runs metadata
      → returns [{run_id, table_ts, table_profile, has_profile}, ...]

  → data.discover_runs()
    → collect_runs_from_gpkg() again
    → merge_run_records()
    → _rebuild_timestep_union()
      → load_timesteps(gpkg, run_id)           [timestep_service.py]
        → _resolve_ts_table()                  [queries.py — table name branching]
        → SELECT DISTINCT t_s FROM swe2d_line_results_ts WHERE run_id=?
        → returns np.array of timesteps

    → _load_coupling_for_first_enabled_run()
      → load_coupling_for_run(gpkg, run_id)    [timestep_service.py]
        → SELECT ... FROM swe2d_coupling_results WHERE run_id=?
```

**With the baked schema, this collapses to:**

```python
studio_results_panel._on_results_add()
  → collect_baked_runs_from_gpkg(gpkg_path)     [run_service.py — ~15 lines]
    SELECT DISTINCT run_id FROM swe2d_baked_results
    → for each run_id:
      SELECT n_timesteps FROM swe2d_baked_results WHERE run_id=?
      SELECT 1 FROM swe2d_baked_line_ts WHERE run_id=? LIMIT 1   → has_lines
      SELECT 1 FROM swe2d_baked_coupling WHERE run_id=? LIMIT 1  → has_coupling
    → returns [RunRecord(...)]

  → data.discover_runs()
    → _rebuild_timestep_union()
      → load_baked_timesteps(gpkg, run_id)
        SELECT times_blob FROM swe2d_baked_results WHERE run_id=?
        → np.frombuffer(blob, dtype=np.float64)  # no DISTINCT needed
```

**Key simplifications:**

| Current | Baked |
|---------|-------|
| `discover_line_result_runs()` — ~100 lines, scans table names, shared/legacy branching | `collect_baked_runs_from_gpkg()` — ~15 lines, single `SELECT DISTINCT run_id` |
| `load_timesteps()` — queries `DISTINCT t_s` from per-row table, returns array | `load_baked_timesteps()` — `np.frombuffer(times_blob)`, direct array |
| `_resolve_ts_table()` — table name discovery, shared/legacy branching | Removed entirely |
| `_rebuild_timestep_union()` — skips if `data_source == "live"` (branching) | Always runs same path — baked results have the same blob format live or in GPKG |
| No baked equivalent for `_load_coupling_for_first_enabled_run()` — queries per-row coupling | `SELECT DISTINCT run_id FROM swe2d_baked_coupling` — single query |

**Multiple runs still produce a timestep union** (e.g. comparing two runs with different
output intervals). `_rebuild_timestep_union()` calls `np.frombuffer(times_blob)` for each
run and unions the arrays — same logic, but the per-run load is a single blob read
instead of a `SELECT DISTINCT` over thousands of rows.

### 5.11 In-Memory Runs in the Results Panel

During a live simulation, the results panel shows a synthetic `RunRecord` with
`data_source == "live"`. The run record is created by `run_controller.py` and
inserted into `SWE2DResultsData._run_records`. This mechanism stays the same —
the baked format changes what data the `RunRecord` points to.

Currently the live `RunRecord` has no GPKG path; its data comes from the in-memory
`_live_snapshot_timesteps` list. With the baked format, the live data lives in
`_live_times`, `_live_h`, etc. — the same numpy arrays that will be serialized
to the GPKG BLOB at finalize time. The `RunRecord` doesn't need to change at all;
the load functions (`load_baked_snapshot`, `load_baked_line_timeseries`, etc.)
transparently read from live arrays when given a `SWE2DResultsData` instance
instead of a GPKG path.

The simplification is entirely in the **handler that receives the signal**: the three-way
branch in `on_results_panel_timestep_changed` becomes a single unconditional call to
`load_baked_snapshot()`.

### 5.12 GeoTIFF Export Overlay

The export method `export_high_perf_overlay_to_geotiff()` in `overlay_controller.py`
depends on two internal methods that reference old tables:

**`_get_snapshot_timesteps()`** (line 67):

```python
# Current: returns list of (t_s, h, hu, hv) tuples from _live_snapshot_timesteps
def _get_snapshot_timesteps(self) -> list:
    return self._data.get_live_snapshot_timesteps()
```

With baked arrays, this becomes a thin compatibility shim that reconstructs
the tuple format from the numpy arrays:

```python
# After: reconstruct from baked numpy arrays (transparent to callers)
def _get_snapshot_timesteps(self):
    d = self._data
    if d._live_times.size == 0:
        return []
    return [(float(t), h, hu, hv) for t, h, hu, hv in
            zip(d._live_times, d._live_h, d._live_hu, d._live_hv)]
```

Alternatively, update `render_unstructured_snapshot_image()` to accept the baked
array format directly — its `timesteps` parameter is `Sequence[Tuple[float, ndarray, ndarray, ndarray]]`.
The shim is simpler and avoids changing the render function's signature.

**`sync_high_perf_overlay_data()`** (line 79) — a ~100-line fallback chain:

```
1. Check _live_snapshot_timesteps for cell centroids + bed
2. If empty, try view._mesh_data
3. If empty, try self._cached_mesh_data
4. If empty, load from GPKG via:
   a. Query swe2d_mesh_results_runs for mesh_hash → OLD TABLE
   b. Look up swe2d_mesh by hash for mesh_name → OLD TABLE
   c. load_mesh_from_geopackage(gpkg, mesh_name) → OLD FUNCTION
```

With the baked format, steps 4a–4c become:

```python
# After: single baked mesh load
gpkg = str(getattr(data, "gpkg_path", "") or "")
if gpkg and os.path.isfile(gpkg):
    row = conn.execute(
        "SELECT mesh_name, baked_blob FROM swe2d_baked_mesh "
        "WHERE mesh_name = (SELECT mesh_name FROM swe2d_baked_results "
        "                   WHERE run_id = ? LIMIT 1)",
        (target_run,)
    ).fetchone()
    if row:
        pm = mod.swe2d_deserialize_mesh(row[1])
        self._data.overlay_cell_x = pm.cell_cx
        self._data.overlay_cell_y = pm.cell_cy
        self._data.overlay_cell_bed = pm.cell_zb
        self._data.overlay_node_x = pm.node_x
        self._data.overlay_node_y = pm.node_y
        # cell_nodes, tri_to_cell from pm.cell_face_nodes (triangulate if poly)
```

The entire fallback chain from ~100 lines to ~25 lines.

**`export_high_perf_overlay_to_geotiff()` itself** (line 452) requires no changes
to its own logic — it calls `_get_snapshot_timesteps()` and passes the result to
`render_unstructured_snapshot_image()`. As long as `_get_snapshot_timesteps()`
returns the expected tuple format, the export works unchanged.

---



## 6. Implementation Plan — Part 1 (Core)

### Phase 1: C++ Serialization

| Step | File | Change | Status |
|------|------|--------|--------|
| 1.1 | `cpp/src/swe2d_mesh.hpp` | Declare `swe2d_serialize_mesh()`, `swe2d_deserialize_mesh()` | ✅ |
| 1.2 | `cpp/src/swe2d_mesh.cpp` | Implement both (~130 lines) | ✅ |
| 1.3 | `cpp/src/swe2d_bindings.cpp` | pybind11 bindings returning/accepting `bytes` | ✅ |
| 1.4 | `cpp/src/swe2d_bindings.cpp` | Python accessor properties on PyMesh (node_x/y/z, cell_zb, etc.) | ✅ |

### Phase 2: Python Persistence

| Step | File | Change | Status |
|------|------|--------|--------|
| 2.1 | `swe2d/services/gpkg_persistence_service.py` | Add `persist_baked_mesh()`, `load_baked_mesh()` | ✅ |
| 2.2 | same | Add `persist_baked_results()`, `load_baked_snapshot()`, `compute_max_tracking()` | ✅ |
| 2.3 | same | Add `persist_baked_coupling()`, `load_baked_coupling_timeseries()` | ✅ |
| 2.4 | same | Add `persist_baked_line_ts()`, `persist_baked_line_profile()` | ✅ |
| 2.5 | same | Add `load_baked_line_timeseries()`, `load_baked_line_profile()` | ✅ |
| 2.6 | same | Update `__all__` | ✅ |

### Phase 3: Backend Integration

| Step | File | Change | Status |
|------|------|--------|--------|
| 3.1 | `swe2d/runtime/backend.py` | Add `build_mesh_from_baked()` method | ✅ |
| 3.2 | `swe2d/runtime/backend_initializer.py` | Insert `persist_baked_mesh()` call after `build_mesh()` | ✅ |
| 3.3 | `swe2d/runtime/run_finalizer.py` | Swap persistence calls to baked versions | ✅ |

**Phase 3 implementation notes:**
- The `use_baked` flag was removed entirely — baked persistence is the only path, no backward compat branching.
- The legacy `else` branch (conservation forensics, per-row line/coupling/mesh persistence) was deleted.
- Coupling persistence reads from `SWE2DResultsData._live_coupling` dict (accumulated numpy arrays per component/object/metric key) with a single `persist_baked_coupling()` call per key.
- Line TS persistence reads from `SWE2DResultsData._live_line_ts` dict (accumulated numpy arrays per line) with a single `persist_baked_line_ts()` call per line.
- The `update_run_snapshot_tag` call (referencing old `_runs` tables) was removed.
- Dead `storage_rows`/`boundary_rows`/`conservation_summary` blocks (only used by deleted conservation forensics path) were removed.

### Phase 4: Live Data Model & Viewer

| Step | File | Change | Status |
|------|------|--------|--------|
| 4.1 | `swe2d/results/data.py` | Replace list-of-dicts with numpy-array dicts; `load_timeseries()` delegates to `load_baked_line_timeseries()`; remove `_data_source` | ✅ |
| 4.2 | `swe2d/results/queries.py` | Replace all load functions with `load_baked_*` — no live/gpkg branching | ✅ |
| 4.3 | `swe2d/results/structure_service.py` | Replace `find_nearest_timestep` + `load_structure_flows_at_time` with baked delegates | ✅ |
| 4.4 | `swe2d/results/timestep_service.py` | Add `load_baked_timesteps()`, `load_baked_coupling_for_run()` | ✅ |
| 4.5 | `swe2d/workbench/services/non_gui_runtime_service.py` | Already feeds dicts → numpy arrays via append_line_snapshot/append_coupling_snapshot | ✅ |
| 4.6 | `swe2d/workbench/controllers/overlay_controller.py` | `_get_snapshot_timesteps()` reads baked numpy arrays | ✅ |
| 4.7 | `swe2d/workbench/views/studio_viewer_pg.py` | Remove `data_source` branching; replace with `load_baked_line_timeseries()` | ✅ |
| 4.8 | `swe2d/workbench/views/studio_viewer_profile_pg.py` | Remove `data_source` branching; replace with `load_baked_line_profile()` | ✅ |
| 4.9 | `swe2d/services/results_render_service.py` | Remove branching; replace with `load_baked_line_timeseries()` | ✅ |
| 4.10 | `swe2d/workbench/views/studio_results_panel.py` | Collapse three-way overlay branch to single path | ✅ |
| 4.11 | `swe2d/results/data.py` | `set_live_snapshot_timesteps()` populates `_live_times/_live_h/_live_hu/_live_hv` numpy arrays | ✅ |
| 4.12 | `swe2d/results/run_service.py` | `collect_baked_runs_from_gpkg()` already in gpkg_persistence_service.py | ✅ |
| 4.13 | `swe2d/results/data.py` | `discover_runs()`, `_rebuild_timestep_union()`, `_load_coupling_for_first_enabled_run()` use baked blob reads | ✅ |
| 4.14 | `swe2d/workbench/controllers/overlay_controller.py` | `sync_high_perf_overlay_data()` — baked mesh deserialize from `swe2d_baked_mesh`; legacy fallback chain (querying `swe2d_mesh_results_runs`, `swe2d_mesh`, `load_mesh_from_geopackage`) removed | ✅ |

### Phase 5: Delete Dead Code

| Step | File | Delete | Status |
|------|------|--------|--------|
| 5.1 | `swe2d/services/gpkg_persistence_service.py` | `build_mesh_rows_from_snapshots`, `load_coupling_results_from_geopackage`, `persist_line_results_to_geopackage`, `update_run_snapshot_tag` — deleted (dead code, no callers). `persist_mesh_to_geopackage`, `load_mesh_from_geopackage`, `persist_mesh_results_to_geopackage`, `persist_coupling_results_to_geopackage`, `persist_conservation_forensics_to_geopackage`, `persist_mesh_max_results_to_geopackage` — kept (still called by `studio_dialog.py`, `run_controller.py`, `headless_runner.py`, `cli/gpkg_adapter.py`). `__all__` cleaned up. | ✅ |
| 5.2 | `swe2d/workbench/services/non_gui_runtime_service.py` | `build_mesh_snapshot_rows()` — still called by `studio_dialog.py` | ⬜ |
| 5.3 | `swe2d/workbench/services/gpkg_service.py` | `load_mesh_snapshot()`, `delete_run_from_gpkg()` | ⬜ |
| 5.4 | `swe2d/results/queries.py` | Already rewritten in Phase 4 with baked delegates; old symbols still exported for backward compat | ✅ |
| 5.5 | `swe2d/results/timestep_service.py` | `load_timesteps()`, `load_line_timesteps()`, `load_coupling_for_run()` — kept for backward compat; baked wrappers added | ✅ |
| 5.5 | `swe2d/results/data.py` | `_data_source` kept for backward compat (auto-detects live); `_live_*_snapshot_rows` accessors kept for backward compat | ✅ |
| 5.6 | `swe2d/workbench/controllers/finalization_adapter.py` | Still references dialog wrappers that call old persistence | ⬜ |
| 5.7 | `swe2d/workbench/studio_dialog.py` | Still calls old persistence functions | ⬜ |
| 5.8 | `swe2d/workbench/controllers/overlay_controller.py` | Legacy fallback chain (`swe2d_mesh_results_runs`, `swe2d_mesh`, `load_mesh_from_geopackage`) removed from `sync_high_perf_overlay_data()` | ✅ |

---

## 7. Part 2 — Post-Hoc Line Resampling

After saved results are available in baked format, users can create new sample lines
and resample the saved results along them — no re-run needed.

### 7.1 Algorithm

The existing pure-numpy sampling pipeline in `mesh_service.py` already does the heavy
lifting. A glue function loops over timesteps and writes baked line BLOBs:

```
load_baked_mesh(gpkg)         → node_coords, cell_nodes, cell_zb
load_baked_results(gpkg)      → times, h_all, hu_all, hv_all

for each new line:
    build_line_sampling_map(…) → sample_map (once per line)
    for each timestep i:
        sample_line_metrics(h_all[i], …)     → profile dict
        sample_line_aggregate_ts_row(…)      → ts row
        accumulate
    persist_baked_line_ts(…)
    persist_baked_line_profile(…)
```

### 7.2 Core Function

```python
@dataclass
class LineDef:
    line_id: int
    line_name: str
    vertices: np.ndarray       # (M, 2) float64 polyline

def resample_lines_from_baked_results(
    gpkg_path: str,
    run_id: str,
    mesh_name: str,
    lines: List[LineDef],
    h_min: float = 1.0e-6,
    gravity: float = 9.81,
    mod=None,                  # hydra_swe2d module (for mesh deserialize)
    log_fn=None,
) -> Dict[int, str]:
    """Resample saved results along new lines. Writes baked line BLOBs."""
    # 1. Load mesh
    pm = load_baked_mesh(gpkg_path, mesh_name, mod)
    node_coords = np.column_stack([pm.node_x, pm.node_y])
    if hasattr(pm, 'cell_face_offsets') and pm.cell_face_offsets is not None:
        tri_nodes, tri_cell, nx_ext, ny_ext = _triangulate_poly_mesh(pm)
        cell_nodes = tri_nodes  # (total_tris, 3)
        cell_bed = pm.cell_zb[tri_cell]  # map tri → parent cell zb
    else:
        cell_nodes = pm.cell_nodes.reshape(-1, 3)
        cell_bed = pm.cell_zb
        nx_ext, ny_ext = pm.node_x, pm.node_y  # no extension needed

    # 2. Load results
    row = conn.execute(
        "SELECT n_timesteps, n_cells, times_blob, h_blob, hu_blob, hv_blob "
        "FROM swe2d_baked_results WHERE run_id=?", (run_id,)
    ).fetchone()
    n_ts, n_cells = int(row[0]), int(row[1])
    times  = np.frombuffer(row[2], dtype=np.float64)
    h_all  = np.frombuffer(row[3], dtype=np.float64).reshape(n_ts, n_cells)
    hu_all = np.frombuffer(row[4], dtype=np.float64).reshape(n_ts, n_cells)
    hv_all = np.frombuffer(row[5], dtype=np.float64).reshape(n_ts, n_cells)

    # 3. For each line
    for line in lines:
        sample_map = build_line_sampling_map(
            np.column_stack([nx_ext, ny_ext]), cell_nodes, line.vertices)

        # Pre-allocate TS arrays
        ts = {k: np.zeros(n_ts) for k in
              ["depth_m","velocity_ms","wse_m","bed_m","flow_cms","wet_frac","fr"]}
        profiles = []

        for i in range(n_ts):
            h, hu, hv = h_all[i], hu_all[i], hv_all[i]
            # TS row
            r = sample_line_aggregate_ts_row(sample_map, h, hu, hv,
                                             cell_bed, h_min, gravity, float(times[i]))
            if r:
                for k in ts: ts[k][i] = r[k]
            # Profile
            prof = sample_line_metrics(h, hu, hv, cell_bed,
                np.column_stack([nx_ext, ny_ext]), cell_nodes,
                line.vertices, h_min, float(times[i]), gravity, sample_map)
            profiles.append(prof)

        # Stack profiles → 2-D arrays
        n_sta = len(profiles[0]["station_m"])
        prof_2d = {k: np.zeros((n_ts, n_sta)) for k in
                   ["depth_m","velocity_ms","wse_m","bed_m","flow_qn","fr"]}
        wet_2d = np.zeros((n_ts, n_sta), dtype=np.int32)
        for i, p in enumerate(profiles):
            for k in prof_2d: prof_2d[k][i] = p[k]
            wet_2d[i] = p["wet"]

        persist_baked_line_ts(gpkg_path, run_id, line.line_id, line.line_name,
                              times, **ts)
        persist_baked_line_profile(gpkg_path, run_id, line.line_id, line.line_name,
                                   profiles[0]["station_m"], times, **prof_2d,
                                   wet=wet_2d)
```

### 7.3 Polygon Mesh Support

When the loaded mesh has `cell_face_offsets`, triangulate each polygon cell using
centroid-vertex fan before passing to `build_line_sampling_map()`:

```python
def _triangulate_poly_mesh(pm):
    """Convert polygon cells to centroid-vertex triangles for point-in-cell tests."""
    n_cells = len(pm.cell_face_offsets) - 1
    total_tris = sum(pm.cell_face_offsets[i+1] - pm.cell_face_offsets[i]
                     for i in range(n_cells))
    tri_nodes = np.empty((total_tris, 3), dtype=np.int32)
    tri_cell  = np.empty(total_tris, dtype=np.int32)
    idx = 0
    for c in range(n_cells):
        s, e = int(pm.cell_face_offsets[c]), int(pm.cell_face_offsets[c+1])
        for i in range(s, e):
            na, nb = int(pm.cell_face_nodes[i]), int(pm.cell_face_nodes[(i+1-s)%(e-s)+s])
            tri_nodes[idx] = [na, nb, len(pm.node_x) + c]  # centroid as virtual node
            tri_cell[idx] = c
            idx += 1
    nx_ext = np.concatenate([pm.node_x, pm.cell_cx])
    ny_ext = np.concatenate([pm.node_y, pm.cell_cy])
    return tri_nodes, tri_cell, nx_ext, ny_ext
```

### 7.4 Implementation

| Step | File | Change |
|------|------|--------|
| 7.1 | `swe2d/workbench/services/mesh_service.py` | Add `_triangulate_poly_mesh()` (~40 lines) |
| 7.2 | `swe2d/services/gpkg_persistence_service.py` | Add `LineDef` dataclass, `resample_lines_from_baked_results()` (~80 lines) |
| 7.3 | `swe2d/workbench/views/studio_results_panel.py` | Add "Resample Lines" button → line-layer picker → calls above |

Total: ~120 lines. All pure numpy except the UI button.

### 7.5 Key Property

The per-timestep loop reads from in-memory numpy arrays (`h_all[i]`), not from SQLite.
No I/O inside the loop. For a 100-ts run with 10 lines, the entire resample runs in
well under a second once `h_all` is loaded.

---

## 8. Summary of Complexity Reduction

| Metric | Current (11 tables) | Proposed (5 tables) |
|--------|--------------------|---------------------|
| GPKG tables | 11 | 5 |
| Rows for 100K cells × 100 steps × 10 lines | 10M+ | ~25 |
| `results/queries.py` lines | ~195 | ~60 |
| Live/GPKG dual code paths | 4 pairs | 0 pairs |
| `_data_source` flag | Required | Eliminated |
| Python per-cell iteration on load | O(n) everywhere | 0 (all `np.frombuffer`) |
| C++ mesh builder re-entry on GPKG load | Full rebuild | 0 (deserialize only) |
| Headless runner mesh round-trip bug | Exists | Eliminated |
| Conservation forensics table | Exists | Removed |
| Max-tracking tables | 2 | Removed (compute on load) |
| Table-name discovery functions | 4 | 0 |
| Post-hoc line resampling | Impossible | ~120 lines of glue code |
