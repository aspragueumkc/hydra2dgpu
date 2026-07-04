#!/usr/bin/env python3
"""Run finalization and persistence seam for SWE2D workbench.

Phase 10 goal: extract end-of-run mass balance, persistence, and summary
logging from `_on_run` into a focused helper module.
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
import datetime
import math
import time
from typing import Any, Dict, List, Protocol

import numpy as np

from swe2d.services.gpkg_persistence_service import (
    persist_baked_results,
    persist_baked_line_ts,
    persist_baked_coupling,
)


class RunFinalizationView(Protocol):
    """View protocol for run finalization — implemented by FinalizationAdapter."""

    def log_message(self, msg: str) -> None:
        """Log a message to the runtime log display."""
        ...

    def get_line_results_storage_path(self) -> str:
        """Return the current GeoPackage path for line results storage."""
        ...

    def sync_overlay_data(self) -> None:
        """Synchronize the high-performance overlay data from the results data layer."""
        ...

    def refresh_plot(self) -> None:
        """Refresh the results plot display."""
        ...

    def results_table_name(self) -> str:
        """Return the current results table name for persistence."""
        ...

    def length_unit_name(self) -> str:
        """Return the current length unit name (e.g., 'm' or 'ft')."""
        ...

    def length_scale_si_to_model(self) -> float:
        """Return the scale factor to convert SI units to model units."""
        ...

    def results_data(self) -> Any:
        """Return the current SWE2DResultsData instance."""
        ...

    def update_overlay_time(self, t: float) -> None:
        """Update the overlay display to show results at time t (seconds)."""
        ...

    def runtime_log_lines(self) -> List[str]:
        """Return the current list of runtime log lines."""
        ...

    def collect_run_log_metadata(self) -> Dict[str, object]:
        """Collect metadata dict for the run log entry."""
        ...

    def persist_run_log(self, gpkg_path: str, run_id: str, run_wallclock_start: str, run_wallclock_end: str, run_duration_wallclock_s: float, run_log_text: str, *, metadata: Dict[str, object]) -> None:
        """Persist the run log entry to the GeoPackage."""
        ...

    def is_cancel_requested(self) -> bool:
        """Return True if the user has requested cancellation of the run."""
        ...


class SWE2DRunFinalizer:
    """Owns end-of-run persistence, summaries, and final logs."""

    def __init__(self, view: RunFinalizationView):
        self._view = view

    def _flow_unit_label(self) -> str:
        """Derive flow unit label from length unit name (no widget access)."""
        unit = self._view.length_unit_name().strip().lower()
        if unit == "ft":
            return "cfs"
        if unit == "m":
            return "cms"
        return f"{unit}3/s"

    def finalize_and_persist(
        self,
        *,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        final_sim_time_s: float,
        n_area: int,
        area_model: np.ndarray,
        storage_start_model: float,
        source_budget_model: Dict[str, float],
        source_step_rows_model: List[Dict[str, float]],
        run_duration_s: float,
        boundary_flux_budget_model: Dict[str, float],
        boundary_flux_step_rows_model: List[Dict[str, float]],
        run_id: str,
        output_interval_s: float,
        line_output_interval_s: float,
        run_perf_start: float,
        run_wallclock_start: str,
        run_log_start_idx: int,
        thiessen_forcing: Any,
        rain_stats_acc: Dict[str, float],
        save_line_results: bool = False,
        save_coupling_results: bool = False,
        save_mesh_results: bool = False,
        save_run_log: bool = False,
        h_min: float = 1.0e-4,
        mesh_name: str = "",
        max_tracking: Optional[Dict[str, np.ndarray]] = None,
        coupling_controller: Any = None,
        sample_map: Any = None,
        cell_solver_z: Any = None,
        sample_line_metrics_callback: Any = None,
        snapshot_timesteps: Any = None,
        coupling_snapshots: Any = None,
        precomputed_line_results: Any = None,
    ) -> Dict[str, Any]:
        """Compute mass-balance summary, persist results to GeoPackage, and refresh UI.

        Returns a status dict with keys ``ok`` (bool), ``errors`` (list of
        persistence failures), and ``warnings`` (list of non-fatal issues).
        Callers that previously ignored the return value are unaffected.
        """
        status: Dict[str, Any] = {"ok": True, "errors": [], "warnings": []}

        def _record_error(msg: str, exc: Exception) -> None:
            status["ok"] = False
            status["errors"].append(f"{msg}: {exc}")
            self._view.log_message(f"[ERROR] {msg}: {exc}")

        def _record_warning(msg: str, exc: Exception) -> None:
            status["warnings"].append(f"{msg}: {exc}")
            self._view.log_message(f"[WARNING] {msg}: {exc}")
        h_end_model = np.asarray(h, dtype=np.float64).ravel()
        n_store_end = min(int(n_area), int(h_end_model.size))
        storage_end_model = float(np.sum(h_end_model[:n_store_end] * area_model[:n_store_end])) if n_store_end > 0 else 0.0
        storage_delta_model = storage_end_model - float(storage_start_model)
        source_total_model = (
            float(source_budget_model["rain"])
            + float(source_budget_model["cell"])
            + float(source_budget_model["coupling"])
        )
        implied_boundary_out_model = source_total_model - storage_delta_model
        avg_implied_boundary_q_model = implied_boundary_out_model / max(float(run_duration_s), 1.0e-12)

        vol_unit_label = f"{self._view.length_unit_name()}3"
        vol_to_si = 1.0 / (self._view.length_scale_si_to_model() ** 3)
        self._view.log_message(
            "Mass balance (explicit sources/storage): "
            f"source_total={source_total_model:.6f} {vol_unit_label} "
            f"(rain={source_budget_model['rain']:.6f}, cell={source_budget_model['cell']:.6f}, "
            f"coupling={source_budget_model['coupling']:.6f}), "
            f"dStorage={storage_delta_model:.6f} {vol_unit_label}, "
            f"implied_net_boundary_out={implied_boundary_out_model:.6f} {vol_unit_label} "
            f"(avg={avg_implied_boundary_q_model:.6f} {self._flow_unit_label()})"
        )
        self._view.log_message(
            "Mass balance (SI reference): "
            f"source_total={source_total_model * vol_to_si:.6f} m3, "
            f"dStorage={storage_delta_model * vol_to_si:.6f} m3, "
            f"implied_net_boundary_out={implied_boundary_out_model * vol_to_si:.6f} m3"
        )
        if boundary_flux_budget_model:
            self._view.log_message("Boundary flux volume by group (from flow-type BC edges):")
            for grp, vol_model in sorted(boundary_flux_budget_model.items(), key=lambda kv: abs(float(kv[1])), reverse=True):
                avg_q_model = float(vol_model) / max(float(run_duration_s), 1.0e-12)
                self._view.log_message(
                    f"  {grp}: volume={float(vol_model):.6f} {vol_unit_label}, "
                    f"avg_q={avg_q_model:.6f} {self._flow_unit_label()}"
                )

        _results_data = self._view.results_data()
        if snapshot_timesteps is not None:
            snapshot_timesteps = list(snapshot_timesteps)
        else:
            snapshot_timesteps = list(_results_data.get_live_snapshot_timesteps()) if _results_data else []
        terminal_t_s = max(0.0, float(final_sim_time_s))
        if not snapshot_timesteps:
            terminal_snapshot = (
                terminal_t_s,
                np.asarray(h, dtype=np.float64).copy(),
                np.asarray(hu, dtype=np.float64).copy(),
                np.asarray(hv, dtype=np.float64).copy(),
            )
            if _results_data is not None:
                _results_data.clear_live_snapshots()
                _results_data.append_live_snapshot(*terminal_snapshot)
            snapshot_timesteps = [terminal_snapshot]
            self._view.log_message(
                "Snapshot capture fallback: no interval snapshots recorded; "
                "stored terminal state snapshot for overlay/results."
            )
        else:
            try:
                last_t_s = float(snapshot_timesteps[-1][0])
            except Exception:
                last_t_s = terminal_t_s
            if terminal_t_s > last_t_s + 1.0e-6:
                terminal_snapshot = (
                    terminal_t_s,
                    np.asarray(h, dtype=np.float64).copy(),
                    np.asarray(hu, dtype=np.float64).copy(),
                    np.asarray(hv, dtype=np.float64).copy(),
                )
                if _results_data is not None:
                    _results_data.append_live_snapshot(*terminal_snapshot)
                snapshot_timesteps.append(terminal_snapshot)

        gpkg_results_path = self._view.get_line_results_storage_path()

        _t0 = time.perf_counter()
        if gpkg_results_path:
            # ── Baked persistence (GPKG BLOB format, only path) ──────────
            try:
                if save_mesh_results and snapshot_timesteps:
                    persist_baked_results(
                        gpkg_results_path, run_id, mesh_name,
                        snapshot_timesteps,
                        max_tracking=max_tracking,
                        log_fn=self._view.log_message,
                    )
                    self._view.log_message(
                        f"  baked mesh results saved to {gpkg_results_path} "
                        f"in {(time.perf_counter() - _t0) * 1000:.0f} ms"
                    )
            except Exception as exc:
                _record_error("Baked mesh results persistence failed", exc)

            _t0 = time.perf_counter()
            try:
                if save_line_results and snapshot_timesteps and (sample_line_metrics_callback is not None or precomputed_line_results is not None):
                    from collections import defaultdict
                    ts_by_line: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
                    prof_by_line: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
                    if precomputed_line_results is not None:
                        for lid, ld in precomputed_line_results.items():
                            if "t_s" in ld:
                                ts_by_line[lid]["line_name"] = ld.get("line_name", f"line_{lid}")
                                ts_by_line[lid]["t_s"] = list(ld["t_s"])
                                for k, pk in (("depth_m", "ts_depth_m"), ("velocity_ms", "ts_velocity_ms"),
                                              ("wse_m", "ts_wse_m"), ("bed_m", "ts_bed_m"),
                                              ("flow_cms", "ts_flow_cms"), ("wet_frac", "ts_wet_frac"),
                                              ("fr", "ts_fr")):
                                    if pk in ld:
                                        ts_by_line[lid][k] = list(ld[pk])
                            if "station_m" in ld:
                                pd = prof_by_line[lid]
                                pd["line_name"] = ld.get("line_name", f"line_{lid}")
                                pd["station_m"] = np.asarray(ld["station_m"], dtype=np.float64)
                                for k, pk in (("depth_m", "prof_depth_m"), ("velocity_ms", "prof_velocity_ms"),
                                              ("wse_m", "prof_wse_m"), ("bed_m", "prof_bed_m"),
                                              ("flow_qn", "prof_flow_qn"), ("fr", "prof_fr"),
                                              ("wet", "prof_wet")):
                                    if pk in ld:
                                        v = ld[pk]
                                        pd[k] = list(v) if isinstance(v, list) else list(v)
                    else:
                        for snap_t, h_snap, hu_snap, hv_snap in snapshot_timesteps:
                            h_arr = np.asarray(h_snap, dtype=np.float64)
                            hu_arr = np.asarray(hu_snap, dtype=np.float64)
                            hv_arr = np.asarray(hv_snap, dtype=np.float64)
                            cell_bed = np.asarray(cell_solver_z, dtype=np.float64) if cell_solver_z is not None else np.zeros_like(h_arr)
                            ts_rows, prof_rows = sample_line_metrics_callback(
                                sample_map, snap_t, h_arr, hu_arr, hv_arr, cell_bed,
                            )
                            for row in ts_rows:
                                lid = int(row.get("line_id", -1))
                                if lid < 0:
                                    continue
                                ld = ts_by_line[lid]
                                if "line_name" not in ld:
                                    ld["line_name"] = str(row.get("line_name", f"line_{lid}"))
                                ld.setdefault("t_s", []).append(float(snap_t))
                                for k in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms", "wet_frac", "fr"):
                                    ld.setdefault(k, []).append(float(row.get(k, 0.0)))
                            for row in prof_rows:
                                lid = int(row.get("line_id", -1))
                                if lid < 0:
                                    continue
                                pd = prof_by_line[lid]
                                if "line_name" not in pd:
                                    pd["line_name"] = str(row.get("line_name", f"line_{lid}"))
                                    pd["station_m"] = np.asarray(row["station_m"], dtype=np.float64)
                                for k in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_qn", "fr"):
                                    v = row.get(k)
                                    pd.setdefault(k, []).append(np.asarray(v, dtype=np.float64) if v is not None else np.array([]))
                                pd.setdefault("wet", []).append(np.asarray(row.get("wet", []), dtype=np.int32))
                    for lid, ld in ts_by_line.items():
                        times_arr = np.array(ld.get("t_s", []), dtype=np.float64)
                        if times_arr.size == 0:
                            continue
                        persist_baked_line_ts(
                            gpkg_results_path, run_id, lid, ld.get("line_name", f"line_{lid}"), times_arr,
                            np.array(ld.get("depth_m", []), dtype=np.float64),
                            np.array(ld.get("velocity_ms", []), dtype=np.float64),
                            np.array(ld.get("wse_m", []), dtype=np.float64),
                            np.array(ld.get("bed_m", []), dtype=np.float64),
                            np.array(ld.get("flow_cms", []), dtype=np.float64),
                            np.array(ld.get("wet_frac", []), dtype=np.float64),
                            np.array(ld.get("fr", []), dtype=np.float64),
                            log_fn=self._view.log_message,
                        )
                    for lid, pd in prof_by_line.items():
                        sm_list = pd.get("station_m", [])
                        if not sm_list:
                            continue
                        station_arr = np.asarray(sm_list, dtype=np.float64)
                        n_sta = station_arr.size
                        n_ts = len(pd.get("depth_m", []))
                        if n_ts == 0 or n_sta == 0:
                            continue
                        depth_flat = np.array(pd["depth_m"], dtype=np.float64)
                        vel_flat = np.array(pd["velocity_ms"], dtype=np.float64)
                        wse_flat = np.array(pd["wse_m"], dtype=np.float64)
                        bed_flat = np.array(pd["bed_m"], dtype=np.float64)
                        qn_flat = np.array(pd["flow_qn"], dtype=np.float64)
                        fr_flat = np.array(pd["fr"], dtype=np.float64)
                        wet_flat = np.array(pd["wet"], dtype=np.int32)
                        times_arr = np.array([float(s[0]) for s in snapshot_timesteps], dtype=np.float64)[:n_ts]
                        persist_baked_line_profile(
                            gpkg_results_path, run_id, lid, pd.get("line_name", f"line_{lid}"),
                            station_arr,
                            times_arr,
                            depth_flat.reshape(n_ts, n_sta),
                            vel_flat.reshape(n_ts, n_sta),
                            wse_flat.reshape(n_ts, n_sta),
                            bed_flat.reshape(n_ts, n_sta),
                            qn_flat.reshape(n_ts, n_sta),
                            fr_flat.reshape(n_ts, n_sta),
                            wet_flat.reshape(n_ts, n_sta),
                            log_fn=self._view.log_message,
                        )
                    self._view.log_message(
                        f"  baked line TS+profiles saved to {gpkg_results_path} "
                        f"in {(time.perf_counter() - _t0) * 1000:.0f} ms"
                    )
            except Exception as exc:
                _record_error("Baked line persistence failed", exc)

            _t0 = time.perf_counter()
            _coupling_data = coupling_snapshots if coupling_snapshots is not None else {}
            try:
                if save_coupling_results and _coupling_data:
                    for key, cd in _coupling_data.items():
                        component, object_id, metric = key
                        times_arr = np.array(cd.get("times", []), dtype=np.float64)
                        if times_arr.size == 0:
                            continue
                        persist_baked_coupling(
                            gpkg_results_path, run_id,
                            component, object_id,
                            cd.get("object_name", object_id),
                            metric,
                            times_arr,
                            np.array(cd.get("values", []), dtype=np.float64),
                            log_fn=self._view.log_message,
                        )
                    self._view.log_message(
                        f"  baked coupling saved to {gpkg_results_path} "
                        f"in {(time.perf_counter() - _t0) * 1000:.0f} ms"
                    )
            except Exception as exc:
                _record_error("Baked coupling persistence failed", exc)

        _t0 = time.perf_counter()
        try:
            self._view.sync_overlay_data()
            if snapshot_timesteps:
                self._view.update_overlay_time(float(snapshot_timesteps[-1][0]))
            self._view.log_message(f"  overlay sync + update in {(time.perf_counter() - _t0) * 1000:.0f} ms")
        except Exception as exc:
            _record_warning("Overlay sync failed", exc)

        run_wallclock_end = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")
        run_duration_wallclock_s = max(0.0, time.perf_counter() - float(run_perf_start))
        self._view.log_message(f"Run wallclock end: {run_wallclock_end}")
        self._view.log_message(f"Run wallclock duration: {run_duration_wallclock_s:.3f} s")
        if gpkg_results_path and save_run_log and run_id:
            _t0 = time.perf_counter()
            try:
                run_log_text = "\n".join(self._view.runtime_log_lines()[run_log_start_idx:])
            except Exception as exc:
                run_log_text = ""
                _record_warning("Could not collect runtime log lines", exc)
            run_log_metadata: Dict[str, object] = {}
            try:
                run_log_metadata = dict(self._view.collect_run_log_metadata() or {})
            except Exception:
                run_log_metadata = {}
            try:
                self._view.persist_run_log(
                    gpkg_results_path,
                    run_id,
                    run_wallclock_start,
                    run_wallclock_end,
                    run_duration_wallclock_s,
                    run_log_text,
                    metadata=run_log_metadata,
                )
                self._view.log_message(f"  run log saved to {gpkg_results_path} in {(time.perf_counter() - _t0) * 1000:.0f} ms")
            except Exception as exc:
                _record_error("Run log persistence failed", exc)

        if thiessen_forcing is not None and rain_stats_acc["samples"] > 0:
            avg_r = rain_stats_acc["rain_mm"] / rain_stats_acc["samples"]
            avg_e = rain_stats_acc["excess_mm"] / rain_stats_acc["samples"]
            self._view.log_message(
                "Spatial rain/CN summary: "
                f"mean rain={avg_r:.3f} mm/step, mean excess={avg_e:.3f} mm/step"
            )

        self._view.log_message("Run complete." if not self._view.is_cancel_requested() else "Run canceled by user.")
        h_min = float(h_min)
        wet = h > h_min
        safe_h = np.maximum(h, 1.0e-12)
        vel_mag = np.where(wet, np.sqrt((hu / safe_h) ** 2 + (hv / safe_h) ** 2), 0.0)
        self._view.log_message(
            f"Depth range: {float(np.min(h)):.6f} .. {float(np.max(h)):.6f} | "
            f"Velocity mag max (wet cells): {float(np.max(vel_mag)):.6f}"
        )
        # Snapshot cleanup intentionally removed — it was deleting user-requested
        # snapshots (persisted by on_snapshot button) alongside stale runtime
        # snapshots. GPKG persistence uses INSERT OR REPLACE, so stale data
        # is naturally overwritten on re-run. User-requested snapshots with
        # swe2d_snapshot_% run_ids are intentionally separate and must survive.
        # Experimental 3D patch mode removed
        try:
            self._view.refresh_plot()
        except Exception as exc:
            _record_warning("refresh_plot failed during finalization", exc)

        _rd = getattr(self._view, "_results_data", None)
        if _rd is not None:
            _rd.clear_live_snapshots()

        return status
