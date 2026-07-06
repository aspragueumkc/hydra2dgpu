"""Pure data/logic layer for SWE2D results.

No widgets, no matplotlib, no Qt.  Owns run records, animation timing,
data queries, and state persistence.

The animation controller (ResultsAnimationController, a QObject) is
lazily created on first access via the ``anim`` property so this module
can be safely constructed from a worker thread without Qt affinity.
"""
from __future__ import annotations

import json
import logging
import os as _os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from swe2d.results.run_service import (
    RunRecord,
    collect_runs_from_gpkg,
    merge_run_records,
    next_color,
    remove_selected_runs,
)

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 4.0

_PERSISTENCE_GROUP = "Backwater2DWorkbench"
_PERSISTENCE_KEY = "swe2d_results_panel_state"


class SWE2DResultsData:
    """Pure data/logic layer for results.  No visible widgets."""

    def __init__(self, fps: float = _DEFAULT_FPS):
        self._run_records: List[RunRecord] = []
        self._manual_gpkg_paths: List[str] = []
        self._selected_run_keys: Set[str] = set()
        self._line_id: int = -1
        self._current_t_sec: float = 0.0
        self._all_timesteps: np.ndarray = np.empty(0, dtype=np.float64)
        self._anim_frame_idx: int = 0
        self._anim_fps: float = float(fps)

        # Animation controller is lazily created on first access via the
        # ``anim`` property so this class can be constructed from a worker
        # thread without creating a QObject (which would get the wrong
        # thread affinity).
        self._anim = None

        # Data source flag: "none", "live", "gpkg" (remains for backward compat)
        self._data_source: str = "none"

        # Baked live snapshot storage (numpy arrays, same shape as GPKG BLOBs)
        self._live_times: np.ndarray = np.empty(0, dtype=np.float64)
        self._live_run_id: str = ""
        self._overlay_selected_key: str = ""
        self._live_h: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._live_hu: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._live_hv: np.ndarray = np.empty((0, 0), dtype=np.float64)

        # Struct coupling data (populated on run discovery)
        self._coupling_records: list = []
        self._coupling_run_id: str = ""
        self._coupling_gpkg_path: str = ""

        # Overlay geometry arrays (populated by overlay controller)
        self.overlay_cell_x: Optional[np.ndarray] = None
        self.overlay_cell_y: Optional[np.ndarray] = None
        self.overlay_cell_bed: Optional[np.ndarray] = None
        self.overlay_node_x: Optional[np.ndarray] = None
        self.overlay_node_y: Optional[np.ndarray] = None
        self.overlay_cell_nodes: Optional[np.ndarray] = None
        self.overlay_tri_to_cell: Optional[np.ndarray] = None

        # In-memory snapshots during a live run (kept for backward compat)
        self._live_snapshot_timesteps: list = []
        # Line timeseries: {line_id: {t_s: np.ndarray (n_snaps,), depth_m: np.ndarray (n_snaps,), ...}}
        self._live_line_ts: Dict[int, Dict[str, np.ndarray]] = {}
        # Line profiles: {line_id: {n_stations: int, depth_m: np.ndarray (n_snaps, n_sta), ...}}
        self._live_line_profile: Dict[int, Dict[str, object]] = {}
        # Coupling: {(component, object_id, metric): {object_name: str, t_s: np.ndarray, values: np.ndarray}}
        self._live_coupling: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        # Display state for TS/Profile/Structure/Network renderers (plain data)
        self.ts_var_key: str = "flow_cms"
        self.prof_var_key: str = "wse_bed"
        self.prof_fill_key: str = "none"
        self.prof_cmap: str = "viridis"
        self.prof_show_structures: bool = True

    def _on_anim_timestep_changed(self, t_s: float, frame_idx: int) -> None:
        """Synchronize data-layer time state with the animation controller."""
        self._anim_frame_idx = int(frame_idx)
        self._current_t_sec = float(t_s)

    
    # ------------------------------------------------------------------
    # Public: run management
    # ------------------------------------------------------------------


        # Don't clear _run_records — live runs inject a synthetic RunRecord
        # that must survive.  Only the selected-run-keys and timesteps are
        # reset so the subsequent user-initiated add/discover flow is clean.



    def clear_live_snapshots(self) -> None:
        """Clear all in-memory snapshot data and reset to empty state.

        Called at the start of a new run to prepare fresh storage.
        Clears: mesh snapshots, line timeseries, line profiles, and coupling data.
        """
        self._live_snapshot_timesteps = []
        self._live_times = np.empty(0, dtype=np.float64)
        self._live_run_id = ""
        self._live_h = np.empty((0, 0), dtype=np.float64)
        self._live_hu = np.empty((0, 0), dtype=np.float64)
        self._live_hv = np.empty((0, 0), dtype=np.float64)
        self._live_line_ts.clear()
        self._live_line_profile.clear()
        self._live_coupling.clear()
        # Drop any GPKG-expanded coupling rows cached from a previous run
        # so viewers don't see stale data until load_coupling_records runs
        # again for the new run.
        self._coupling_records = []
        self._coupling_run_id = ""

    def init_coupling_storage(
        self,
        coupling_keys: List[Tuple[str, str, str]],
        coupling_object_names: Dict[Tuple[str, str, str], str],
    ) -> None:
        """Initialize coupling storage with dynamic lists.

        Called once before the run starts, after the coupling controller is
        built, so the object names are known.
        """
        self._live_coupling.clear()
        for key in coupling_keys:
            self._live_coupling[key] = {
                "object_name": coupling_object_names.get(key, key[1]),
                "t_s": [],
                "values": [],
            }

    def append_live_snapshot(self, t_s: float, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> None:
        """Append a single mesh snapshot to the live snapshot list.

        Parameters
        ----------
        t_s : float
            Simulation time in seconds.
        h : ndarray
            Water depth array.
        hu : ndarray
            x-momentum array.
        hv : ndarray
            y-momentum array.
        """
        self._live_snapshot_timesteps.append((t_s, h, hu, hv))
        self._data_source = "live"

    def set_live_snapshot_timesteps(
        self, timesteps: list, t_sec: float = 0.0
    ) -> None:
        """Bulk-replace the live snapshot list from device readback.

        ``timesteps`` is a list of ``(t_s, h, hu, hv)`` tuples as returned
        by :meth:`SWE2DBackend.read_snapshots`.  Also updates the animation
        timeline and the ``_live_times/_live_h/_live_hu/_live_hv`` arrays
        so the temporal dock slider and overlay can read the data.

        If *t_sec* > 0, seeks the animation to that time after populating.
        """
        self._live_snapshot_timesteps = list(timesteps)
        self._data_source = "live"
        if timesteps:
            n = len(timesteps)
            t_arr = np.array([float(t[0]) for t in timesteps], dtype=np.float64)
            n_cells = np.asarray(timesteps[0][1]).size if n > 0 else 0
            h_arr  = np.array([np.asarray(t[1], dtype=np.float64) for t in timesteps]) if n > 0 else np.empty((0, n_cells), dtype=np.float64)
            hu_arr = np.array([np.asarray(t[2], dtype=np.float64) for t in timesteps]) if n > 0 else np.empty((0, n_cells), dtype=np.float64)
            hv_arr = np.array([np.asarray(t[3], dtype=np.float64) for t in timesteps]) if n > 0 else np.empty((0, n_cells), dtype=np.float64)
            self._live_times = t_arr
            self._live_h = h_arr
            self._live_hu = hu_arr
            self._live_hv = hv_arr
            self._all_timesteps = t_arr
            self._current_t_sec = float(t_arr[0]) if t_arr.size > 0 else 0.0
            # Cast to float explicitly, using float(t) for each element
            float_times = [float(t) for t in t_arr]
            self._live_anim_count = len(float_times)
            if hasattr(self, "_anim") and self._anim is not None:
                self._anim.set_timesteps(t_arr)
        else:
            self._live_times = np.empty(0, dtype=np.float64)
            self._live_h = np.empty((0, 0), dtype=np.float64)
            self._live_hu = np.empty((0, 0), dtype=np.float64)
            self._live_hv = np.empty((0, 0), dtype=np.float64)
            self._all_timesteps = np.empty(0, dtype=np.float64)
            self._current_t_sec = 0.0
            self._live_anim_count = 0
            if hasattr(self, "_anim") and self._anim is not None:
                self._anim.set_timesteps(self._all_timesteps)
        if t_sec > 0.0 and hasattr(self, "_t_sec_to_frame_idx"):
            self._set_frame(self._t_sec_to_frame_idx(float(t_sec)))
        elif self._all_timesteps is not None and self._all_timesteps.size > 0:
            self._anim_frame_idx = 0
            if self._anim is not None:
                self._anim.set_index(0)

    def append_line_snapshot(self, row: dict, snap_idx: int) -> None:
        """Write a line timeseries row into pre-allocated _live_line_ts at snap_idx.

        snap_idx is the output-snapshot index (0-based), not the simulation timestep.
        """
        lid = int(row.get("line_id", -1))
        if lid < 0:
            return
        if lid not in self._live_line_ts:
            self._live_line_ts[lid] = {"line_name": str(row.get("line_name", f"line_{lid}"))}
        d = self._live_line_ts[lid]
        for key in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms", "wet_frac", "fr"):
            arr = d.get(key)
            if arr is not None and snap_idx < arr.size:
                arr[snap_idx] = float(row.get(key, 0.0))

    def append_line_profile_snapshot(self, row: dict, snap_idx: int) -> None:
        """Write a line profile row into pre-allocated _live_line_profile at snap_idx.

        snap_idx is the output-snapshot index (0-based).
        The per-line n_stations must already be set via preallocate_line_profile_nstations().
        """
        lid = int(row.get("line_id", -1))
        if lid < 0:
            return
        d = self._live_line_profile.get(lid)
        if d is None:
            return
        n_sta = int(d.get("n_stations", 0))
        if n_sta <= 0:
            return
        for key in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_qn", "fr"):
            arr = d.get(key)
            if arr is not None and snap_idx < arr.shape[0]:
                val = row.get(key, 0.0)
                if isinstance(val, np.ndarray) and val.size == n_sta:
                    arr[snap_idx, :] = val
                elif np.isscalar(val):
                    arr[snap_idx, :] = float(val)
        wet_arr = d.get("wet")
        if wet_arr is not None and snap_idx < wet_arr.shape[0]:
            wet_val = row.get("wet", 0)
            if isinstance(wet_val, np.ndarray) and wet_val.size == n_sta:
                wet_arr[snap_idx, :] = wet_val
            else:
                wet_arr[snap_idx, :] = int(wet_val)

    def append_coupling_snapshot(self, row: dict) -> None:
        """Write a coupling row into live coupling storage."""
        key = (str(row.get("component", "")),
               str(row.get("object_id", "")),
               str(row.get("metric", "")))
        if not key[0] or not key[1] or not key[2]:
            return
        d = self._live_coupling.get(key)
        if d is None:
            return
        d["t_s"].append(float(row.get("t_s", 0.0)))
        d["values"].append(float(row.get("value", 0.0)))

    def get_live_snapshot_timesteps(self) -> list:
        """Return the list of live mesh snapshots as (t_s, h, hu, hv) tuples."""
        return self._live_snapshot_timesteps

    def build_precomputed_line_results(self) -> dict:
        """Convert _live_line_ts / _live_line_profile to the dict format
        expected by ``SWE2DRunFinalizer.finalize_and_persist`` via the
        ``precomputed_line_results`` argument.
        """
        if not self._live_line_ts and not self._live_line_profile:
            return {}
        times = [float(s[0]) for s in self._live_snapshot_timesteps] if self._live_snapshot_timesteps else (
            list(self._live_times) if self._live_times.size > 0 else [])
        out: dict = {}
        for lid, d in self._live_line_ts.items():
            entry = {"line_name": d.get("line_name", f"line_{lid}")}
            if times:
                entry["t_s"] = list(times)
            for k in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms", "wet_frac", "fr"):
                v = d.get(k)
                if v is not None:
                    entry[f"ts_{k}"] = list(np.asarray(v, dtype=np.float64))
            out[lid] = entry
        for lid, d in self._live_line_profile.items():
            entry = out.setdefault(lid, {"line_name": d.get("line_name", f"line_{lid}")})
            sta = d.get("station_m")
            if sta is not None:
                entry["station_m"] = np.asarray(sta, dtype=np.float64)
            for k in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_qn", "fr"):
                arr = d.get(k)
                if arr is not None and hasattr(arr, "shape") and arr.ndim == 2:
                    entry[f"prof_{k}"] = arr
            wet = d.get("wet")
            if wet is not None and hasattr(wet, "shape") and wet.ndim == 2:
                entry["prof_wet"] = wet
        return out

    def get_live_line_snapshot_rows(self) -> list:
        """Reconstruct list-of-dicts from _live_line_ts numpy storage."""
        out = []
        for lid, d in self._live_line_ts.items():
            t_s = d.get("t_s")
            if t_s is None:
                continue
            n = t_s.size
            line_name = d.get("line_name", f"line_{lid}")
            for i in range(n):
                row = {"line_id": lid, "line_name": line_name, "t_s": float(t_s[i])}
                for key in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms", "wet_frac", "fr"):
                    arr = d.get(key)
                    row[key] = float(arr[i]) if arr is not None and i < arr.size else 0.0
                out.append(row)
        return out

    def get_live_line_profile_rows(self) -> list:
        """Reconstruct list-of-dicts from _live_line_profile numpy storage."""
        out = []
        for lid, d in self._live_line_profile.items():
            depth_m = d.get("depth_m")
            if depth_m is None:
                continue
            n_snaps = depth_m.shape[0]
            n_sta = depth_m.shape[1] if depth_m.ndim > 1 else 0
            station_m = d.get("station_m")
            station_arr = np.asarray(station_m) if station_m is not None else np.arange(n_sta)
            line_name = d.get("line_name", f"line_{lid}")
            for i in range(n_snaps):
                row = {"line_id": lid, "line_name": line_name}
                for key in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_qn", "fr"):
                    arr = d.get(key)
                    if arr is not None and arr.ndim == 2 and i < arr.shape[0]:
                        row[key] = arr[i]
                    else:
                        row[key] = np.array([])
                wet = d.get("wet")
                row["wet"] = wet[i] if wet is not None and wet.ndim == 2 and i < wet.shape[0] else np.array([], dtype=np.int32)
                row["station_m"] = station_arr
                out.append(row)
        return out

    def populate_live_line_metrics_from_gpu(
        self,
        lm_data: dict,
        line_names_by_id: Optional[Dict[int, str]] = None,
        line_ids_ordered: Optional[List[int]] = None,
    ) -> None:
        """Populate _live_line_ts/_live_line_profile from GPU-computed flat arrays.

        Called each time the GPU line metrics ring buffer is read back.
        ``line_ids_ordered`` maps the 0-based GPU index to the actual
        ``line_id`` from the geopackage sample-lines layer so the viewer
        can look up lines by their user-visible ID.
        """
        t_s = lm_data.get("t_s")
        profiles = lm_data.get("profiles")
        ts = lm_data.get("ts")
        wet = lm_data.get("wet")
        station_offsets = lm_data.get("station_offsets")
        station_m = lm_data.get("station_m")
        import logging as _lg
        _lg.warning("[LINE_DIAG] populate_from_gpu: t_s=%s profiles=%s ts=%s station_offsets=%s station_m=%s",
                     t_s.shape if hasattr(t_s, 'shape') else t_s,
                     profiles.shape if hasattr(profiles, 'shape') else profiles,
                     ts.shape if hasattr(ts, 'shape') else ts,
                     station_offsets.shape if hasattr(station_offsets, 'shape') else station_offsets,
                     station_m.shape if hasattr(station_m, 'shape') else station_m)
        if t_s is None or profiles is None or ts is None or station_offsets is None:
            _lg.warning("[LINE_DIAG] populate_from_gpu: EARLY RETURN — missing required keys")
            return
        total_stations = lm_data.get("total_stations", 0)
        if total_stations <= 0:
            _lg.warning("[LINE_DIAG] populate_from_gpu: EARLY RETURN — total_stations=%d", total_stations)
            return
        n_lines = int(station_offsets.size) - 1
        if n_lines <= 0:
            _lg.warning("[LINE_DIAG] populate_from_gpu: EARLY RETURN — n_lines=%d", n_lines)
            return

        # Resolve the ordered list of actual line IDs (matching the GPU 0-index).
        if not line_ids_ordered:
            line_ids_ordered = list(range(n_lines))

        new_ts: Dict[int, Dict[str, object]] = {}
        new_prof: Dict[int, Dict[str, object]] = {}
        _ts_keys = ("depth_m", "velocity_ms", "wse_m", "bed_m",
                     "flow_cms", "wet_frac", "fr")
        _prof_keys = ("depth_m", "velocity_ms", "wse_m", "bed_m",
                       "flow_qn", "fr")
        for gpu_idx in range(n_lines):
            lid = line_ids_ordered[gpu_idx] if gpu_idx < len(line_ids_ordered) else gpu_idx
            nm = (line_names_by_id or {}).get(lid, f"line_{lid}")
            s = int(station_offsets[gpu_idx])
            e = int(station_offsets[gpu_idx + 1])
            n_sta = e - s
            ts_line = ts[:, gpu_idx, :]
            t_dict = {"line_name": nm}
            for idx, key in enumerate(_ts_keys):
                t_dict[key] = np.ascontiguousarray(ts_line[:, idx])
            new_ts[lid] = t_dict
            p_dict: Dict[str, object] = {
                "line_name": nm,
                "station_m": np.ascontiguousarray(station_m[s:e]),
                "n_stations": n_sta,
            }
            if n_sta > 0:
                prof_slice = profiles[:, s:e, :]
                for idx, key in enumerate(_prof_keys):
                    p_dict[key] = np.ascontiguousarray(prof_slice[:, :, idx])
                p_dict["wet"] = (
                    np.ascontiguousarray(wet[:, s:e])
                    if wet is not None and wet.shape[1] >= total_stations
                    else np.zeros((ts.shape[0], n_sta), dtype=np.int32)
                )
            else:
                for key in _prof_keys:
                    p_dict[key] = np.empty((ts.shape[0], 0), dtype=np.float64)
                p_dict["wet"] = np.empty((ts.shape[0], 0), dtype=np.int32)
            new_prof[lid] = p_dict
        self._live_line_ts = new_ts
        self._live_line_profile = new_prof

    def get_live_coupling_snapshot_rows(self) -> list:
        """Reconstruct list-of-dicts from _live_coupling list storage."""
        out = []
        for (component, object_id, metric), d in self._live_coupling.items():
            t_s = d.get("t_s")
            values = d.get("values")
            if not t_s or not values:
                continue
            n = min(len(t_s), len(values))
            for i in range(n):
                out.append({
                    "component": component,
                    "object_id": object_id,
                    "metric": metric,
                    "object_name": d.get("object_name", ""),
                    "t_s": float(t_s[i]),
                    "value": float(values[i]),
                })
        return out

    def get_structure_flows_at_time(self, run_id: str, t_sec: float) -> list:
        """Return structure coupling rows at the nearest stored timestep."""
        out = []
        for (component, object_id, metric), d in self._live_coupling.items():
            if component != "structure" or metric != "flow":
                continue
            t_s = d.get("t_s")
            values = d.get("values")
            if not t_s or not values:
                continue
            n = min(len(t_s), len(values))
            if n == 0:
                continue
            i = int(np.argmin(np.abs(np.asarray(t_s[:n], dtype=np.float64) - t_sec)))
            out.append({
                "component": component,
                "object_id": object_id,
                "object_name": d.get("object_name", ""),
                "metric": metric,
                "t_s": float(t_s[i]),
                "value": float(values[i]),
            })
        return out

    def get_run_records(self) -> List[RunRecord]:
        """Return run records."""
        return list(self._run_records)

    def get_enabled_run_records(self) -> List[RunRecord]:
        """Return enabled run records."""
        return [r for r in self._run_records if r.enabled]

    def discover_runs(self, scan_paths: Optional[List[str]] = None) -> List[RunRecord]:
        """Scan GPKGs, return only user-selected runs.

        Parameters
        ----------
        scan_paths : list, optional
            Explicit list of GPKG paths to scan. When None (default), scans
            self._manual_gpkg_paths. When a list is provided, only those paths
            are scanned — use this when the caller wants to restrict scanning
            to specific GPKGs (e.g. only newly added ones).
        """
        if not self._selected_run_keys:
            self._run_records = []
            self._load_coupling_for_first_enabled_run()
            return []

        if scan_paths is None:
            manual_paths = [p for p in self._manual_gpkg_paths if p and _os.path.exists(p)]
            scan_paths = manual_paths
        else:
            scan_paths = [p for p in scan_paths if p and _os.path.exists(p)]

        all_candidates: List[RunRecord] = []
        for gpkg in scan_paths:
            if not gpkg:
                continue
            all_candidates.extend(collect_runs_from_gpkg(gpkg))

        self._run_records = merge_run_records(
            all_candidates, self._selected_run_keys, scan_paths,
        )
        self._load_coupling_for_first_enabled_run()
        return list(self._run_records)

    def keep_only_most_recent_run(self) -> None:
        """Keep only most recent run."""
        if self._run_records:
            first = self._run_records[0]
            first.enabled = True
            self._run_records = [first]
            self._rebuild_timestep_union()

    def toggle_run(self, run_key: str, enabled: bool) -> None:
        """Toggle run."""
        for rec in self._run_records:
            if rec.key == run_key:
                rec.enabled = enabled
                break
        self._rebuild_timestep_union()
        self._load_coupling_for_first_enabled_run()

    def remove_runs(self, run_keys: Set[str]) -> None:
        """Remove runs."""
        if not run_keys:
            return
        self._run_records, self._manual_gpkg_paths = remove_selected_runs(
            self._run_records, run_keys, self._manual_gpkg_paths,
        )
        self._selected_run_keys -= run_keys
        if self._overlay_selected_key in run_keys:
            self._overlay_selected_key = ""
        self._rebuild_timestep_union()

    def set_all_runs_visible(self) -> None:
        """Set all runs visible."""
        for rec in self._run_records:
            rec.enabled = True
        self._rebuild_timestep_union()

    def set_all_runs_hidden(self) -> None:
        """Set all runs hidden."""
        for rec in self._run_records:
            rec.enabled = False
        self._rebuild_timestep_union()

    def add_manual_gpkg(self, gpkg_path: str) -> None:
        """Add manual gpkg."""
        gpkg = str(gpkg_path or "").strip()
        if not gpkg:
            return
        if gpkg not in self._manual_gpkg_paths:
            self._manual_gpkg_paths.append(gpkg)

    def add_manual_selected_keys(self, keys: Set[str]) -> None:
        """Add manual selected keys."""
        self._selected_run_keys.update(keys)

    def add_results_files(self, file_paths: List[str]) -> Tuple[int, int]:
        """Add results from GPKG files. Returns (added_paths, added_runs)."""
        added_paths = 0
        added_runs = 0
        for fp in file_paths:
            gpkg = str(fp or "").strip()
            if not gpkg or not _os.path.exists(gpkg):
                continue
            candidates = collect_runs_from_gpkg(gpkg)
            if not candidates:
                continue
            if gpkg not in self._manual_gpkg_paths:
                self._manual_gpkg_paths.append(gpkg)
                added_paths += 1
            for rec in candidates:
                self._selected_run_keys.add(rec.key)
            added_runs += len(candidates)
        if added_paths > 0 or added_runs > 0:
            self.discover_runs()
        return added_paths, added_runs

    def run_ids_for_gpkg(self, gpkg_path: str, enabled_only: bool = False) -> List[str]:
        """Run ids for gpkg."""
        gpkg_norm = str(gpkg_path or "").strip()
        out: List[str] = []
        seen: Set[str] = set()
        for rec in self._run_records:
            if str(rec.gpkg_path or "").strip() != gpkg_norm:
                continue
            if enabled_only and not rec.enabled:
                continue
            rid = str(rec.run_id or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
        return out

    # ------------------------------------------------------------------
    # Public: line / timestep
    # ------------------------------------------------------------------

    @property
    def line_id(self) -> int:
        """line id."""
        return self._line_id

    def set_line_id(self, line_id: int) -> None:
        """Set line id."""
        new_id = int(line_id)
        if new_id == self._line_id:
            return
        self._line_id = new_id
        self._rebuild_timestep_union()

    def get_line_ids(self) -> List[int]:
        """Return sorted unique line IDs from enabled runs or live snapshots."""
        ids: Set[int] = set()

        # Live data path — keys of _live_line_ts
        for lid in self._live_line_ts:
            ids.add(lid)

        # GPKG baked table path — query each enabled run
        for rec in self._run_records:
            if not rec.enabled:
                continue
            try:
                conn = __import__('sqlite3').connect(rec.gpkg_path)
                try:
                    # Check if table exists first to avoid OperationalError
                    cur = conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='swe2d_baked_line_ts'"
                    )
                    if cur.fetchone() is None:
                        continue
                    rows = conn.execute(
                        "SELECT DISTINCT line_id FROM swe2d_baked_line_ts WHERE run_id=?",
                        (rec.run_id,),
                    ).fetchall()
                    ids.update(int(r[0]) for r in rows)
                finally:
                    conn.close()
            except Exception as e:
                logger.warning("[RESULTS] Failed to discover line IDs from %s: %s", rec.gpkg_path, e)
        return sorted(ids)

    @property
    def all_timesteps(self) -> np.ndarray:
        """all timesteps."""
        return self._all_timesteps.copy()

    @property
    def frame_count(self) -> int:
        """frame count."""
        return int(self._all_timesteps.size)

    @property
    def current_time_sec(self) -> float:
        """current time sec."""
        return float(self._current_t_sec)

    @property
    def current_frame_idx(self) -> int:
        """current frame idx."""
        return int(self._anim_frame_idx)

    def set_current_time(self, t_sec: float) -> None:
        """Set current time."""
        self._set_frame(self._t_sec_to_frame_idx(float(t_sec)))

    def t_sec_to_frame_idx(self, t_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return self._t_sec_to_frame_idx(float(t_sec))

    def frame_idx_to_t_sec(self, idx: int) -> float:
        """Convert frame index to time in seconds."""
        return self._frame_idx_to_t_sec(int(idx))

    # ------------------------------------------------------------------
    # Public: animation control
    # ------------------------------------------------------------------

    @property
    def anim(self):
        """Lazily create and return the animation controller (a QObject).

        Created on first access so that construction from a worker thread
        does not create a QObject with the wrong thread affinity.
        """
        if self._anim is None:
            from swe2d.results.animation import ResultsAnimationController
            self._anim = ResultsAnimationController(fps=self._anim_fps)
            self._anim.current_timestep_changed.connect(self._on_anim_timestep_changed)
        return self._anim

    @property
    def is_playing(self) -> bool:
        """Whether playing."""
        return self._anim is not None and self._anim.is_playing

    def play(self) -> None:
        """Start playback."""
        if self._anim is not None:
            self._anim.play()

    def pause(self) -> None:
        """Pause playback."""
        if self._anim is not None:
            self._anim.pause()

    def step_forward(self) -> None:
        """Step forward."""
        if self._anim is not None:
            self._anim.pause()
            idx = int(self._anim._index) + 1
        else:
            idx = self._anim_frame_idx + 1
        if self._all_timesteps is not None and idx >= self._all_timesteps.size:
            idx = 0
        self.set_index(idx)

    def step_backward(self) -> None:
        """Step backward."""
        if self._anim is not None:
            self._anim.pause()
            idx = int(self._anim._index) - 1
        else:
            idx = max(0, self._anim_frame_idx - 1)
        if idx < 0:
            idx = 0
        self.set_index(idx)

    def set_index(self, idx: int) -> None:
        """Set animation index and update current_time_sec BEFORE emitting.

        The previous implementation called ``self._anim.set_index()`` first,
        which synchronously emits ``current_timestep_changed``.  Plot widgets
        connected to that signal read ``data.current_time_sec`` while it still
        held the previous value.  We now update our own state first.
        """
        idx = int(idx)
        self._anim_frame_idx = idx
        if self._all_timesteps is not None and self._all_timesteps.size > 0:
            self._current_t_sec = float(self._all_timesteps[
                min(idx, self._all_timesteps.size - 1)
            ])
        if self._anim is not None:
            self._anim.set_index(idx)

    def set_frame_rate(self, fps: float) -> None:
        """Set frame rate."""
        self._anim_fps = float(fps)
        if self._anim is not None:
            self._anim.set_frame_rate(float(fps))

    # ------------------------------------------------------------------
    # Public: data queries
    # ------------------------------------------------------------------

    def load_timeseries(self, run_record: RunRecord, line_id: int, var_key: str) -> dict:
        """Load timeseries data for a run/line/variable (baked-aware)."""
        from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
        return load_baked_line_timeseries(run_record.gpkg_path, run_record.run_id, line_id)

    def get_coupling_records(self) -> list:
        """Return coupling records for the active run."""
        if self._coupling_records:
            return list(self._coupling_records)
        if self._data_source == "live":
            return self.get_live_coupling_snapshot_rows()
        return []

    def get_coupling_run_id(self) -> str:
        """Return the run ID for the current coupling data."""
        return self._coupling_run_id

    def load_coupling_for_first_enabled_run(self) -> list:
        """Load coupling for first enabled run."""
        return list(self._coupling_records)

    # ------------------------------------------------------------------
    # Public: overlay support
    # ------------------------------------------------------------------

    def active_overlay_run_id(self) -> str:
        """Return active overlay run id."""
        for rec in self._run_records:
            if rec.enabled:
                return str(rec.run_id)
        return ""

    @property
    def data_source(self) -> str:
        """Return the current data source: 'live', 'gpkg', or 'none'.

        When ``set_data_source`` has been called explicitly, honor that
        explicit choice even if ``_live_times`` is still populated from a
        previous run.
        """
        return self._data_source

    def set_data_source(self, source: str) -> None:
        """Set the current data source ('live', 'gpkg', or 'none')."""
        self._data_source = source

    def get_snapshot_at_time(self, t_sec: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load mesh snapshot (h, hu, hv) nearest to the given time."""
        # Try baked live arrays first
        if self._live_times.size > 0 and self._live_h.size > 0:
            i = int(np.argmin(np.abs(self._live_times - t_sec)))
            return (self._live_h[i], self._live_hu[i], self._live_hv[i])
        # Fall back to old list-of-tuples for backward compat
        if self._live_snapshot_timesteps:
            best = min(self._live_snapshot_timesteps, key=lambda s: abs(s[0] - t_sec))
            return (best[1], best[2], best[3])
        return None

    def first_enabled_record(self) -> Optional["RunRecord"]:
        """Return the first enabled RunRecord, or None."""
        for rec in self._run_records:
            if rec.enabled:
                return rec
        return None

    def overlay_selected_run(self):
        """Return the overlay-selected RunRecord, or first enabled if none selected."""
        if self._overlay_selected_key:
            for rec in self._run_records:
                if rec.key == self._overlay_selected_key:
                    return rec
        return self.first_enabled_record()

    def set_overlay_selected_key(self, key: str) -> None:
        """Set which run is selected for map overlay display."""
        self._overlay_selected_key = str(key or "")

    def enabled_overlay_targets(self) -> List[Tuple[str, str]]:
        """Return enabled overlay targets."""
        out: List[Tuple[str, str]] = []
        for rec in self._run_records:
            if not rec.enabled:
                continue
            gpkg = str(rec.gpkg_path or "").strip()
            run_id = str(rec.run_id or "").strip()
            if not gpkg or not run_id:
                continue
            out.append((gpkg, run_id))
        return out

    # ------------------------------------------------------------------
    # Public: state persistence (data only)
    # ------------------------------------------------------------------

    def save_data_state(self) -> dict:
        """Return serializable dict of data-only state."""
        return {
            "run_keys_enabled": [r.key for r in self._run_records if r.enabled],
            "run_ids_enabled": [r.run_id for r in self._run_records if r.enabled],
            "manual_gpkg_paths": list(self._manual_gpkg_paths),
            "selected_run_keys": sorted(self._selected_run_keys),
            "line_id": self._line_id,
            "t_sec": self._current_t_sec,
            "frame_idx": int(self._anim_frame_idx),
            "is_playing": bool(self._anim is not None and self._anim.is_playing),
            "overlay_selected_key": self._overlay_selected_key,
        }

    def restore_data_state(self, state: dict) -> None:
        """Restore data-only state from dict.  Does NOT touch widgets."""
        self._manual_gpkg_paths = [
            str(p) for p in state.get("manual_gpkg_paths", [])
            if isinstance(p, str) and p and _os.path.exists(p)
        ]
        self._selected_run_keys = {
            str(k) for k in state.get("selected_run_keys", [])
            if isinstance(k, str) and k
        }
        self.discover_runs()

        enabled_keys = set(state.get("run_keys_enabled", []))
        enabled_ids = set(state.get("run_ids_enabled", []))
        for rec in self._run_records:
            run_key = rec.key
            rid = run_key.split("::", 1)[1] if "::" in run_key else run_key
            should_enable = (run_key in enabled_keys) if enabled_keys else (rid in enabled_ids)
            rec.enabled = should_enable

        lid = state.get("line_id", -1)
        self._line_id = int(lid)

        self._overlay_selected_key = str(state.get("overlay_selected_key", "") or "")

        t_sec = float(state.get("t_sec", 0.0))
        self._set_frame(self._t_sec_to_frame_idx(t_sec))

    def save_to_project(self) -> None:
        """Persist data state to QgsProject."""
        try:
            from qgis.core import QgsProject
        except ImportError:
            return
        state = self.save_data_state()
        try:
            QgsProject.instance().writeEntry(
                _PERSISTENCE_GROUP, _PERSISTENCE_KEY, json.dumps(state)
            )
        except Exception as exc:
            logger.debug("[RESULTS] Failed to save data state: %s", exc)

    def restore_from_project(self) -> None:
        """Restore data state from QgsProject."""
        try:
            from qgis.core import QgsProject
        except ImportError:
            return
        try:
            raw, _ = QgsProject.instance().readEntry(
                _PERSISTENCE_GROUP, _PERSISTENCE_KEY, ""
            )
            if not raw:
                return
            state = json.loads(raw)
        except Exception as exc:
            logger.warning("[RESULTS] Failed to restore data state: %s", exc)
            return
        self.restore_data_state(state)

    # ------------------------------------------------------------------
    # Private: internal logic
    # ------------------------------------------------------------------

    def _rebuild_timestep_union(self) -> None:
        """Rebuild the union of all timesteps from baked results BLOBs.

        Uses load_baked_timesteps for efficient np.frombuffer reads.
        No longer skips live data — baked arrays handle both paths.
        """
        from swe2d.services.gpkg_persistence_service import load_baked_timesteps

        ts_sets: List[np.ndarray] = []
        for rec in self._run_records:
            if not rec.enabled:
                continue
            ts = load_baked_timesteps(rec.gpkg_path, rec.run_id)
            if ts.size:
                ts_sets.append(ts)

        self._all_timesteps = (
            np.unique(np.concatenate(ts_sets)) if ts_sets
            else np.empty(0, dtype=np.float64)
        )
        self._anim_frame_idx = 0
        if self._all_timesteps.size:
            self._current_t_sec = float(self._all_timesteps[0])
        if self._anim is not None:
            self._anim.set_timesteps(self._all_timesteps)

    def _t_sec_to_frame_idx(self, t_sec: float) -> int:
        """t sec to frame idx."""
        from swe2d.results.timestep_service import time_sec_to_frame_idx
        return time_sec_to_frame_idx(t_sec, self._all_timesteps)

    def _frame_idx_to_t_sec(self, idx: int) -> float:
        """frame idx to t sec."""
        from swe2d.results.timestep_service import frame_idx_to_time_sec
        return frame_idx_to_time_sec(idx, self._all_timesteps)

    def _set_frame(self, idx: int) -> None:
        """set frame."""
        idx = int(idx)
        if self._anim is not None:
            self._anim.set_index(idx)
            self._anim_frame_idx = int(self._anim._index)
        else:
            self._anim_frame_idx = idx
        if self._all_timesteps is not None and self._all_timesteps.size > 0:
            self._current_t_sec = float(self._all_timesteps[
                min(self._anim_frame_idx, self._all_timesteps.size - 1)
            ])

    @staticmethod
    def _expand_baked_coupling_rows(gpkg_path: str, run_id: str) -> list:
        """Load baked coupling BLOBs and expand them into per-row records.

        Each returned dict contains ``t_s``, ``value``, ``component``,
        ``object_id``, ``object_name``, and ``metric``.
        """
        from swe2d.services.gpkg_persistence_service import load_baked_coupling_timeseries

        records: list = []
        conn = __import__('sqlite3').connect(gpkg_path)
        try:
            has_coupling_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='swe2d_baked_coupling'"
            ).fetchone()
            if not has_coupling_table:
                has_coupling_input = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                    "name IN ('swe2d_drainage_nodes','swe2d_structures')"
                ).fetchone()
                if has_coupling_input:
                    logger.warning(
                        "[STRUCT] Run has drainage/structure input layers but no "
                        "swe2d_baked_coupling results table — coupling results were not persisted")
                return records
            meta_rows = conn.execute(
                "SELECT component, object_id, object_name, metric "
                "FROM swe2d_baked_coupling WHERE run_id=?",
                (run_id,),
            ).fetchall()
            for comp, oid, oname, metric in meta_rows:
                times, values = load_baked_coupling_timeseries(
                    gpkg_path, run_id, str(comp), str(oid), str(metric),
                )
                if times is None or times.size == 0:
                    continue
                for i in range(times.size):
                    records.append({
                        "t_s": float(times[i]),
                        "value": float(values[i]),
                        "component": str(comp),
                        "object_id": str(oid),
                        "object_name": str(oname),
                        "metric": str(metric),
                    })
            return records
        finally:
            conn.close()

    def _load_coupling_for_first_enabled_run(self) -> None:
        """load coupling for first enabled run (baked — expand BLOBs into per-row format)."""
        first = None
        for rec in self._run_records:
            if rec.enabled:
                first = rec
                break
        if first is None:
            self._coupling_records = []
            self._coupling_run_id = ""
            self._coupling_gpkg_path = ""
            return
        self._coupling_records = self._expand_baked_coupling_rows(
            first.gpkg_path, first.run_id
        )
        self._coupling_run_id = first.run_id
        self._coupling_gpkg_path = first.gpkg_path

    def load_coupling_records(self, run_id_or_key: str) -> None:
        """Load coupling records from GPKG baked table, falling back to live."""
        gpkg = ""
        run_id = ""
        for rec in self._run_records:
            if rec.run_id == run_id_or_key or rec.key == run_id_or_key:
                gpkg = rec.gpkg_path
                run_id = rec.run_id
                break
        if not gpkg:
            # Live run — use in-memory coupling snapshots if available
            live_rows = self.get_live_coupling_snapshot_rows()
            if live_rows:
                self._coupling_records = live_rows
                self._coupling_run_id = run_id_or_key
                self._coupling_gpkg_path = ""
            else:
                self._coupling_records = []
                self._coupling_run_id = ""
                self._coupling_gpkg_path = ""
            return
        self._coupling_records = self._expand_baked_coupling_rows(gpkg, run_id)
        self._coupling_run_id = run_id
        self._coupling_gpkg_path = gpkg
