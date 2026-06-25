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

from swe2d.workbench.services.gpkg_persistence_service import persist_line_results_to_geopackage, update_run_snapshot_tag


class RunFinalizationView(Protocol):
    def log_message(self, msg: str) -> None: ...
    def get_line_results_storage_path(self) -> str: ...
    def sync_overlay_data(self) -> None: ...
    def refresh_plot(self) -> None: ...
    def results_table_name(self) -> str: ...
    def length_unit_name(self) -> str: ...
    def length_scale_si_to_model(self) -> float: ...
    def results_data(self) -> Any: ...
    def persist_coupling_results(self, gpkg_path: str, run_id: str, rows: Any, *, interval_s: float) -> None: ...
    def build_mesh_snapshot_rows(self) -> Any: ...
    def selected_mesh_results_table_name(self) -> str: ...
    def persist_mesh_results(self, gpkg_path: str, run_id: str, rows: Any, *, interval_s: float, table_name: str) -> None: ...
    def update_overlay_time(self, t: float) -> None: ...
    def runtime_log_lines(self) -> List[str]: ...
    def collect_run_log_metadata(self) -> Dict[str, object]: ...
    def persist_run_log(self, gpkg_path: str, run_id: str, run_wallclock_start: str, run_wallclock_end: str, run_duration_wallclock_s: float, run_log_text: str, *, metadata: Dict[str, object]) -> None: ...
    def is_cancel_requested(self) -> bool: ...
    def line_snapshot_rows(self) -> Any: ...
    def line_snapshot_profile_rows(self) -> Any: ...
    def coupling_snapshot_rows(self) -> Any: ...
    def persist_conservation_forensics(self, gpkg_path: str, run_id: str, storage_rows: List[Dict[str, float]], boundary_rows: List[Dict[str, float]], conservation_summary: Dict[str, float], *, source_step_rows: List[Dict[str, float]]) -> None: ...


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
    ) -> None:
        """finalize and persist."""
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

        storage_rows: List[Dict[str, float]] = [
            {
                "t_s": 0.0,
                "storage_model": float(storage_start_model),
                "storage_delta_model": 0.0,
            }
        ]
        _results_data = self._view.results_data()
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
        if snapshot_timesteps:
            for snap in snapshot_timesteps:
                try:
                    t_s, h_snap, _, _ = snap
                    hh = np.asarray(h_snap, dtype=np.float64).ravel()
                    n_snap = min(int(n_area), int(hh.size), int(area_model.size))
                    if n_snap <= 0:
                        continue
                    storage_t_model = float(np.sum(hh[:n_snap] * area_model[:n_snap]))
                    if not math.isfinite(storage_t_model):
                        continue
                    storage_rows.append(
                        {
                            "t_s": float(t_s),
                            "storage_model": storage_t_model,
                            "storage_delta_model": float(storage_t_model - float(storage_start_model)),
                        }
                    )
                except Exception:
                    continue

        boundary_rows: List[Dict[str, float]] = []
        for row in list(boundary_flux_step_rows_model or []):
            try:
                boundary_rows.append(
                    {
                        "t_s": float(row.get("t_s", 0.0)),
                        "group_name": str(row.get("group", "") or ""),
                        "q_effective_model": float(row.get("q_model", 0.0)),
                        "vol_effective_model": float(row.get("vol_model", 0.0)),
                    }
                )
            except Exception:
                continue

        conservation_summary = {
            "run_duration_s": float(run_duration_s),
            "source_rain_model": float(source_budget_model.get("rain", 0.0)),
            "source_cell_model": float(source_budget_model.get("cell", 0.0)),
            "source_coupling_model": float(source_budget_model.get("coupling", 0.0)),
            "source_total_model": float(source_total_model),
            "storage_start_model": float(storage_start_model),
            "storage_end_model": float(storage_end_model),
            "storage_delta_model": float(storage_delta_model),
            "implied_net_boundary_out_model": float(implied_boundary_out_model),
            "avg_implied_boundary_q_model": float(avg_implied_boundary_q_model),
            "boundary_group_volume_sum_model": float(sum(float(v) for v in boundary_flux_budget_model.values())),
        }

        gpkg_results_path = self._view.get_line_results_storage_path()

        _t0 = time.perf_counter()
        if gpkg_results_path:
            try:
                self._view.persist_conservation_forensics(
                    gpkg_results_path,
                    run_id,
                    storage_rows,
                    boundary_rows,
                    conservation_summary,
                    source_step_rows=list(source_step_rows_model or []),
                )
                self._view.log_message(f"  conservation persisted to {gpkg_results_path} in {(time.perf_counter() - _t0) * 1000:.0f} ms")
            except Exception as exc:
                self._view.log_message(f"Conservation forensic persistence warning: {exc}")

        _line_rows = _results_data.get_live_line_snapshot_rows() if _results_data else []
        _line_profile_rows = _results_data.get_live_line_profile_rows() if _results_data else []
        _coupling_rows = _results_data.get_live_coupling_snapshot_rows() if _results_data else []

        _t0 = time.perf_counter()
        if gpkg_results_path and save_line_results and _line_rows:
            persist_line_results_to_geopackage(
                gpkg_path=gpkg_results_path,
                run_id=run_id,
                rows=_line_rows,
                mesh_interval_s=output_interval_s,
                line_interval_s=line_output_interval_s,
                profile_rows=_line_profile_rows,
                log_fn=self._view.log_message,
                results_table_name_fn=self._view.results_table_name,
            )
            self._view.log_message(f"  line results saved to {gpkg_results_path} in {(time.perf_counter() - _t0) * 1000:.0f} ms")

        _t0 = time.perf_counter()
        if gpkg_results_path and save_coupling_results and _coupling_rows:
            self._view.persist_coupling_results(
                gpkg_results_path,
                run_id,
                _coupling_rows,
                interval_s=line_output_interval_s,
            )
            self._view.log_message(f"  coupling results saved to {gpkg_results_path} in {(time.perf_counter() - _t0) * 1000:.0f} ms")

        _t0 = time.perf_counter()
        if gpkg_results_path and save_mesh_results and snapshot_timesteps:
            mesh_rows = self._view.build_mesh_snapshot_rows()
            if mesh_rows:
                mesh_table_name = "swe2d_mesh_results"
                try:
                    mesh_table_name = self._view.selected_mesh_results_table_name() or "swe2d_mesh_results"
                except Exception:
                    mesh_table_name = "swe2d_mesh_results"
                self._view.persist_mesh_results(
                    gpkg_results_path,
                    run_id,
                    mesh_rows,
                    interval_s=output_interval_s,
                    table_name=mesh_table_name,
                )
                self._view.log_message(f"  mesh results saved to {gpkg_results_path} in {(time.perf_counter() - _t0) * 1000:.0f} ms")

        _t0 = time.perf_counter()
        try:
            self._view.sync_overlay_data()
            if snapshot_timesteps:
                self._view.update_overlay_time(float(snapshot_timesteps[-1][0]))
            self._view.log_message(f"  overlay sync + update in {(time.perf_counter() - _t0) * 1000:.0f} ms")
        except Exception:
            logger.warning("Unexpected error silently caught", exc_info=True)

        run_wallclock_end = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")
        run_duration_wallclock_s = max(0.0, time.perf_counter() - float(run_perf_start))
        self._view.log_message(f"Run wallclock end: {run_wallclock_end}")
        self._view.log_message(f"Run wallclock duration: {run_duration_wallclock_s:.3f} s")
        if gpkg_results_path and save_run_log and run_id:
            _t0 = time.perf_counter()
            run_log_text = "\n".join(self._view.runtime_log_lines()[run_log_start_idx:])
            run_log_metadata: Dict[str, object] = {}
            try:
                run_log_metadata = dict(self._view.collect_run_log_metadata() or {})
            except Exception:
                run_log_metadata = {}
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

        if gpkg_results_path:
            try:
                update_run_snapshot_tag(gpkg_results_path, run_id, is_snapshot=False,
                                        table_name_fn=self._view.results_table_name)
            except Exception:
                pass

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
        if self._view.line_snapshot_rows():
            self._view.log_message(
                f"Sample line rows captured: ts={len(self._view.line_snapshot_rows())}, "
                f"profile={len(self._view.line_snapshot_profile_rows())}"
            )
        if self._view.coupling_snapshot_rows():
            self._view.log_message(f"Coupling rows captured: {len(self._view.coupling_snapshot_rows())}")
        # Snapshot cleanup intentionally removed — it was deleting user-requested
        # snapshots (persisted by on_snapshot button) alongside stale runtime
        # snapshots. GPKG persistence uses INSERT OR REPLACE, so stale data
        # is naturally overwritten on re-run. User-requested snapshots with
        # swe2d_snapshot_% run_ids are intentionally separate and must survive.
        # Experimental 3D patch mode removed
        try:
            self._view.refresh_plot()
        except Exception as exc:
            logger.warning("refresh_plot failed during finalization: %s", exc, exc_info=True)
