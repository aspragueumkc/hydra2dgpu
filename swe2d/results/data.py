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
        """Set gpkg path."""
        new = str(gpkg_path or "")
        if new == self._gpkg_path:
            return
        self._gpkg_path = new
        self.discover_runs()

    def set_live_snapshot_timesteps(
        self, snapshot_timesteps: list, t_sec: float = 0.0
    ) -> None:
        """Feed in-memory snapshot timesteps for live-run temporal control.

        Called during a run so the temporal dock slider reflects the
        accumulating snapshots before they are persisted to GPKG.
        """
        if not snapshot_timesteps:
            self._all_timesteps = np.empty(0, dtype=np.float64)
            self._anim.set_timesteps(self._all_timesteps)
            return
        t_arr = np.array([float(s[0]) for s in snapshot_timesteps], dtype=np.float64)
        self._all_timesteps = t_arr
        self._anim.set_timesteps(self._all_timesteps)
        if t_sec > 0.0:
            self._set_frame(self._t_sec_to_frame_idx(float(t_sec)))

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
        """Return sorted unique line IDs from enabled runs."""
        from swe2d.results.queries import load_line_ids
        ids: Set[int] = set()
        for rec in self._run_records:
            if not rec.enabled:
                continue
            try:
                ids.update(lid for lid, _ in load_line_ids(rec.gpkg_path, rec.run_id))
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
        """Load timeseries data for a run/line/variable."""
        from swe2d.results.queries import load_timeseries as _load_ts
        return _load_ts(run_record.gpkg_path, run_record.run_id, line_id)

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
        """rebuild timestep union."""
        from swe2d.results.timestep_service import (
            compute_timestep_union,
            load_timesteps,
        )
        if self._line_id < 0:
            self._all_timesteps = np.empty(0, dtype=np.float64)
            self._anim.set_timesteps(self._all_timesteps)
            return

        ts_sets: List[np.ndarray] = []
        for rec in self._run_records:
            if not rec.enabled:
                continue
            ts = load_timesteps(rec.gpkg_path, rec.run_id)
            if ts.size:
                ts_sets.append(ts)

        self._all_timesteps = (
            compute_timestep_union(ts_sets) if ts_sets
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
        """load coupling for first enabled run."""
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
            from swe2d.results.timestep_service import load_coupling_for_run
            self._coupling_records = load_coupling_for_run(first.gpkg_path, first.run_id)
            self._coupling_run_id = first.run_id
            self._coupling_gpkg_path = first.gpkg_path
        except Exception as exc:
            logger.warning("[STRUCT] Failed to load coupling data: %s", exc)
            self._coupling_records = []

    def load_coupling_records(self, run_id_or_key: str) -> None:
        """Load coupling records."""
        gpkg = ""
        for rec in self._run_records:
            if rec.run_id == run_id_or_key or rec.key == run_id_or_key:
                gpkg = rec.gpkg_path
                break
        if not gpkg:
            self._coupling_records = []
            self._coupling_run_id = ""
            self._coupling_gpkg_path = ""
            return
        try:
            from swe2d.results.timestep_service import load_coupling_for_run
            self._coupling_records = load_coupling_for_run(gpkg, run_id_or_key)
            self._coupling_run_id = run_id_or_key
            self._coupling_gpkg_path = gpkg
        except Exception as exc:
            logger.warning("[STRUCT] Failed to load coupling data: %s", exc)
            self._coupling_records = []
