"""Pure data/logic layer for SWE2D results.

No widgets, no matplotlib, no Qt (except ResultsAnimationController which
needs QTimer).  Owns run records, animation timing, data queries, and
state persistence.

Studio's View calls this module's public API to read/write data.
"""
from __future__ import annotations

import json
import logging
import os as _os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from swe2d.results.animation import ResultsAnimationController
from swe2d.results.run_service import (
    RunRecord,
    collect_runs_from_gpkg,
    merge_run_records,
    next_color,
    remove_selected_runs,
)

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 4.0

try:
    from qgis.core import QgsProject as _QgsProject
    _HAVE_QGSPROJECT = True
except ImportError:
    _QgsProject = None
    _HAVE_QGSPROJECT = False

_PERSISTENCE_GROUP = "Backwater2DWorkbench"
_PERSISTENCE_KEY = "swe2d_results_panel_state"


class SWE2DResultsData:
    """Pure data/logic layer for results.  No visible widgets."""

    def __init__(self, gpkg_path: str = "", fps: float = _DEFAULT_FPS):
        self._gpkg_path: str = str(gpkg_path or "")
        self._run_records: List[RunRecord] = []
        self._manual_gpkg_paths: List[str] = []
        self._selected_run_keys: Set[str] = set()
        self._line_id: int = -1
        self._current_t_sec: float = 0.0
        self._all_timesteps: np.ndarray = np.empty(0, dtype=np.float64)
        self._anim_frame_idx: int = 0
        self._anim_fps: float = float(fps)

        # Animation (needs QTimer — acceptable Qt dependency)
        self._anim = ResultsAnimationController(fps=self._anim_fps)

        # Data source flag: "none", "live", "gpkg" (remains for backward compat)
        self._data_source: str = "none"

        # Baked live snapshot storage (numpy arrays, same shape as GPKG BLOBs)
        self._live_times: np.ndarray = np.empty(0, dtype=np.float64)
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
        # Line timeseries: {line_id: {t_s: [], depth_m: [], vel: [], ...}}
        self._live_line_ts: Dict[int, Dict[str, list]] = {}
        # Line profiles: {line_id: {t_s: [], station_m: [], depth_m: [], ...}}
        self._live_line_profile: Dict[int, Dict[str, list]] = {}
        # Coupling: {(component, object_id, metric): {object_name, t_s: [], values: []}}
        self._live_coupling: Dict[Tuple[str, str, str], Dict] = {}

        # Display state for TS/Profile/Structure/Network renderers (plain data)
        self.ts_var_key: str = "flow_cms"
        self.prof_var_key: str = "wse_bed"
        self.prof_fill_key: str = "none"
        self.prof_cmap: str = "viridis"
        self.prof_show_structures: bool = True

    # ------------------------------------------------------------------
    # Public: run management
    # ------------------------------------------------------------------

    @property
    def gpkg_path(self) -> str:
        """gpkg path."""
        return self._gpkg_path

    def set_gpkg_path(self, gpkg_path: str) -> None:
        """Set gpkg path.  Does NOT auto-discover runs — use add_results_files or the Add Results dialog."""
        new = str(gpkg_path or "")
        if new == self._gpkg_path:
            return
        self._gpkg_path = new
        self._selected_run_keys.clear()
        self._all_timesteps = np.empty(0, dtype=np.float64)
        # Don't clear _run_records — live runs inject a synthetic RunRecord
        # that must survive.  Only the selected-run-keys and timesteps are
        # reset so the subsequent user-initiated add/discover flow is clean.

    def set_live_snapshot_timesteps(
        self, snapshot_timesteps: list, t_sec: float = 0.0
    ) -> None:
        """Feed in-memory snapshot timesteps for live-run temporal control.

        Called during a run so the temporal dock slider reflects the
        accumulating snapshots before they are persisted to GPKG.
        """
        if not snapshot_timesteps:
            self._all_timesteps = np.empty(0, dtype=np.float64)
            self._live_times = np.empty(0, dtype=np.float64)
            self._anim.set_timesteps(self._all_timesteps)
            return
        # Populate numpy arrays (same shape as baked BLOBs)
        n_steps = len(snapshot_timesteps)
        n_cells = int(np.asarray(snapshot_timesteps[0][1]).size) if n_steps > 0 else 0
        t_arr = np.array([float(s[0]) for s in snapshot_timesteps], dtype=np.float64)
        self._live_times = t_arr.copy()
        if n_cells > 0:
            self._live_h = np.zeros((n_steps, n_cells), dtype=np.float64)
            self._live_hu = np.zeros((n_steps, n_cells), dtype=np.float64)
            self._live_hv = np.zeros((n_steps, n_cells), dtype=np.float64)
            for i, (_, h, hu, hv) in enumerate(snapshot_timesteps):
                self._live_h[i] = np.asarray(h, dtype=np.float64).ravel()
                self._live_hu[i] = np.asarray(hu, dtype=np.float64).ravel()
                self._live_hv[i] = np.asarray(hv, dtype=np.float64).ravel()
        self._all_timesteps = t_arr
        self._anim.set_timesteps(self._all_timesteps)
        if t_sec > 0.0:
            self._set_frame(self._t_sec_to_frame_idx(float(t_sec)))

    def clear_live_snapshots(self) -> None:
        self._live_snapshot_timesteps = []
        self._live_times = np.empty(0, dtype=np.float64)
        self._live_h = np.empty((0, 0), dtype=np.float64)
        self._live_hu = np.empty((0, 0), dtype=np.float64)
        self._live_hv = np.empty((0, 0), dtype=np.float64)
        self._live_line_ts.clear()
        self._live_line_profile.clear()
        self._live_coupling.clear()

    def append_live_snapshot(self, t_s: float, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> None:
        self._live_snapshot_timesteps.append((t_s, h, hu, hv))
        self._data_source = "live"

    def append_line_snapshot(self, row: dict) -> None:
        """Append a line timeseries row, accumulating into _live_line_ts."""
        lid = int(row.get("line_id", -1))
        if lid < 0:
            return
        d = self._live_line_ts.setdefault(lid, {})
        for key in ("t_s", "depth_m", "velocity_ms", "wse_m", "bed_m",
                     "flow_cms", "wet_frac", "fr"):
            d.setdefault(key, []).append(float(row.get(key, 0.0)))
        if "line_name" not in d:
            d["line_name"] = str(row.get("line_name", f"line_{lid}"))

    def append_line_profile_snapshot(self, row: dict) -> None:
        """Append a line profile row, accumulating into _live_line_profile."""
        lid = int(row.get("line_id", -1))
        if lid < 0:
            return
        d = self._live_line_profile.setdefault(lid, {})
        for key in ("t_s", "station_m", "depth_m", "velocity_ms",
                     "wse_m", "bed_m", "flow_qn", "fr", "wet"):
            d.setdefault(key, []).append(float(row.get(key, 0.0)))
        if "line_name" not in d:
            d["line_name"] = str(row.get("line_name", f"line_{lid}"))

    def append_coupling_snapshot(self, row: dict) -> None:
        """Append a coupling row, accumulating into _live_coupling."""
        key = (str(row.get("component", "")),
               str(row.get("object_id", "")),
               str(row.get("metric", "")))
        if not key[0] or not key[1] or not key[2]:
            return
        d = self._live_coupling.setdefault(key, {"object_name": "", "t_s": [], "values": []})
        d["object_name"] = str(row.get("object_name", ""))
        d["t_s"].append(float(row.get("t_s", 0.0)))
        d["values"].append(float(row.get("value", 0.0)))

    def get_live_snapshot_timesteps(self) -> list:
        return self._live_snapshot_timesteps

    def get_live_line_snapshot_rows(self) -> list:
        """Backward-compat: reconstruct list-of-dicts from _live_line_ts."""
        out = []
        for lid, d in self._live_line_ts.items():
            n = len(d.get("t_s", []))
            for i in range(n):
                row = {"line_id": lid, "line_name": d.get("line_name", "")}
                for key in ("t_s", "depth_m", "velocity_ms", "wse_m",
                             "bed_m", "flow_cms", "wet_frac", "fr"):
                    arr = d.get(key, [])
                    row[key] = float(arr[i]) if i < len(arr) else 0.0
                out.append(row)
        return out

    def get_live_line_profile_rows(self) -> list:
        """Backward-compat: reconstruct list-of-dicts from _live_line_profile."""
        out = []
        for lid, d in self._live_line_profile.items():
            n = len(d.get("t_s", []))
            for i in range(n):
                row = {"line_id": lid, "line_name": d.get("line_name", "")}
                for key in ("t_s", "station_m", "depth_m", "velocity_ms",
                             "wse_m", "bed_m", "flow_qn", "fr", "wet"):
                    arr = d.get(key, [])
                    row[key] = float(arr[i]) if i < len(arr) else 0.0
                out.append(row)
        return out

    def get_live_coupling_snapshot_rows(self) -> list:
        """Backward-compat: reconstruct list-of-dicts from _live_coupling."""
        out = []
        for (component, object_id, metric), d in self._live_coupling.items():
            n = len(d.get("t_s", []))
            for i in range(n):
                out.append({
                    "component": component,
                    "object_id": object_id,
                    "metric": metric,
                    "object_name": d.get("object_name", ""),
                    "t_s": float(d["t_s"][i]) if i < len(d["t_s"]) else 0.0,
                    "value": float(d["values"][i]) if i < len(d["values"]) else 0.0,
                })
        return out

    def get_run_records(self) -> List[RunRecord]:
        """Return run records."""
        return list(self._run_records)

    def get_enabled_run_records(self) -> List[RunRecord]:
        """Return enabled run records."""
        return [r for r in self._run_records if r.enabled]

    def discover_runs(self) -> List[RunRecord]:
        """Scan all registered GPKGs, return only user-selected runs."""
        if not self._selected_run_keys:
            self._run_records = []
            self._load_coupling_for_first_enabled_run()
            return []

        manual_paths = [p for p in self._manual_gpkg_paths if p and _os.path.exists(p)]
        all_paths = [self._gpkg_path] + manual_paths if self._gpkg_path else manual_paths
        all_candidates: List[RunRecord] = []
        for gpkg in all_paths:
            if not gpkg:
                continue
            all_candidates.extend(collect_runs_from_gpkg(gpkg))

        self._run_records = merge_run_records(
            all_candidates, self._selected_run_keys, all_paths,
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
            except Exception:
                pass
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
    def anim(self) -> ResultsAnimationController:
        """anim."""
        return self._anim

    @property
    def is_playing(self) -> bool:
        """Whether playing."""
        return self._anim.is_playing

    def play(self) -> None:
        """Start playback."""
        self._anim.play()

    def pause(self) -> None:
        """Pause playback."""
        self._anim.pause()

    def step_forward(self) -> None:
        """Step forward."""
        self._anim.pause()
        self._anim.step_forward()

    def step_backward(self) -> None:
        """Step backward."""
        self._anim.pause()
        self._anim.step_backward()

    def set_index(self, idx: int) -> None:
        """Set index."""
        self._anim.set_index(int(idx))
        self._anim_frame_idx = int(self._anim._index)
        if self._all_timesteps is not None and self._all_timesteps.size > 0:
            self._current_t_sec = float(self._all_timesteps[
                min(int(self._anim_frame_idx), self._all_timesteps.size - 1)
            ])

    def set_frame_rate(self, fps: float) -> None:
        """Set frame rate."""
        self._anim_fps = float(fps)
        self._anim.set_frame_rate(float(fps))

    # ------------------------------------------------------------------
    # Public: data queries
    # ------------------------------------------------------------------

    def load_timeseries(self, run_record: RunRecord, line_id: int, var_key: str) -> dict:
        """Load timeseries data for a run/line/variable (baked-aware)."""
        from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries
        return load_baked_line_timeseries(run_record.gpkg_path, run_record.run_id, line_id)

    def get_coupling_records(self) -> list:
        """Return coupling records for the active run.

        Falls back to in-memory coupling snapshots during live runs
        when no GPKG coupling data has been loaded.
        """
        if self._coupling_records:
            return list(self._coupling_records)
        if self._data_source == "live":
            live_rows = self.get_live_coupling_snapshot_rows()
            if live_rows:
                return live_rows
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
        if self._live_times.size > 0:
            return "live"
        return self._data_source

    def set_data_source(self, source: str) -> None:
        self._data_source = source

    def get_snapshot_at_time(self, t_sec: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Try baked live arrays first
        if self._live_times.size > 0 and self._live_h.size > 0:
            i = int(np.argmin(np.abs(self._live_times - t_sec)))
            return (self._live_h[i], self._live_hu[i], self._live_hv[i])
        # Fall back to old list-of-tuples for backward compat
        if self._live_snapshot_timesteps:
            best = min(self._live_snapshot_timesteps, key=lambda s: abs(s[0] - t_sec))
            return (best[1], best[2], best[3])
        return None

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
            "is_playing": bool(self._anim.is_playing),
        }

    def restore_data_state(self, state: dict) -> None:
        """Restore data-only state from dict.  Does NOT touch widgets."""
        self._manual_gpkg_paths = [
            str(p) for p in state.get("manual_gpkg_paths", [])
            if isinstance(p, str) and p and _os.path.exists(p) and p != self._gpkg_path
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

        t_sec = float(state.get("t_sec", 0.0))
        self._set_frame(self._t_sec_to_frame_idx(t_sec))

    def save_to_project(self) -> None:
        """Persist data state to QgsProject."""
        if not _HAVE_QGSPROJECT or _QgsProject is None:
            return
        state = self.save_data_state()
        try:
            _QgsProject.instance().writeEntry(
                _PERSISTENCE_GROUP, _PERSISTENCE_KEY, json.dumps(state)
            )
        except Exception as exc:
            logger.debug("[RESULTS] Failed to save data state: %s", exc)

    def restore_from_project(self) -> None:
        """Restore data state from QgsProject."""
        if not _HAVE_QGSPROJECT or _QgsProject is None:
            return
        try:
            raw, _ = _QgsProject.instance().readEntry(
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
        self._anim.set_timesteps(self._all_timesteps)
        if self._all_timesteps.size:
            self._current_t_sec = float(self._all_timesteps[0])

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
        self._anim.set_index(idx)
        self._anim_frame_idx = int(self._anim._index)
        if self._all_timesteps is not None and self._all_timesteps.size > 0:
            self._current_t_sec = float(self._all_timesteps[
                min(self._anim_frame_idx, self._all_timesteps.size - 1)
            ])

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
        try:
            from swe2d.services.gpkg_persistence_service import load_baked_coupling_timeseries
            # Load all coupling records for this run via baked table
            conn = __import__('sqlite3').connect(first.gpkg_path)
            try:
                meta_rows = conn.execute(
                    "SELECT component, object_id, object_name, metric "
                    "FROM swe2d_baked_coupling WHERE run_id=?",
                    (first.run_id,),
                ).fetchall()
                records = []
                for comp, oid, oname, metric in meta_rows:
                    times, values = load_baked_coupling_timeseries(
                        first.gpkg_path, first.run_id,
                        str(comp), str(oid), str(metric),
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
                self._coupling_records = records
            finally:
                conn.close()
            self._coupling_run_id = first.run_id
            self._coupling_gpkg_path = first.gpkg_path
        except Exception as exc:
            logger.warning("[STRUCT] Failed to load coupling data: %s", exc)
            self._coupling_records = []

    def load_coupling_records(self, run_id_or_key: str) -> None:
        """Load coupling records from GPKG baked table, falling back to live."""
        gpkg = ""
        for rec in self._run_records:
            if rec.run_id == run_id_or_key or rec.key == run_id_or_key:
                gpkg = rec.gpkg_path
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
        try:
            import sqlite3
            conn = sqlite3.connect(gpkg)
            try:
                rows = conn.execute(
                    "SELECT component, object_id, object_name, metric, n_timesteps "
                    "FROM swe2d_baked_coupling WHERE run_id=?",
                    (run_id_or_key,),
                ).fetchall()
                self._coupling_records = [
                    {
                        "component": comp, "object_id": oid,
                        "object_name": oname, "metric": metric,
                        "n_timesteps": n_ts,
                    }
                    for comp, oid, oname, metric, n_ts in rows
                ]
                self._coupling_run_id = run_id_or_key
                self._coupling_gpkg_path = gpkg
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[STRUCT] Failed to load coupling data: %s", exc)
            self._coupling_records = []
