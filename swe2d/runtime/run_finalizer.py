#!/usr/bin/env python3
"""Run finalization and persistence seam for SWE2D workbench.

Phase 10 goal: extract end-of-run mass balance, persistence, and summary
logging from `_on_run` into a focused helper module.
"""

from __future__ import annotations

import datetime
import math
import time
from typing import Any, Dict, List

import numpy as np


class SWE2DRunFinalizer:
    """Owns end-of-run persistence, summaries, and final logs."""

    def __init__(self, ui: Any):
        self._ui = ui

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
    ) -> None:
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

        vol_unit_label = f"{self._ui._length_unit_name}3"
        vol_to_si = 1.0 / (self._ui._length_scale_si_to_model() ** 3)
        self._ui._log(
            "Mass balance (explicit sources/storage): "
            f"source_total={source_total_model:.6f} {vol_unit_label} "
            f"(rain={source_budget_model['rain']:.6f}, cell={source_budget_model['cell']:.6f}, "
            f"coupling={source_budget_model['coupling']:.6f}), "
            f"dStorage={storage_delta_model:.6f} {vol_unit_label}, "
            f"implied_net_boundary_out={implied_boundary_out_model:.6f} {vol_unit_label} "
            f"(avg={avg_implied_boundary_q_model:.6f} {self._ui._flow_unit_label()})"
        )
        self._ui._log(
            "Mass balance (SI reference): "
            f"source_total={source_total_model * vol_to_si:.6f} m3, "
            f"dStorage={storage_delta_model * vol_to_si:.6f} m3, "
            f"implied_net_boundary_out={implied_boundary_out_model * vol_to_si:.6f} m3"
        )
        if boundary_flux_budget_model:
            self._ui._log("Boundary flux volume by group (from flow-type BC edges):")
            for grp, vol_model in sorted(boundary_flux_budget_model.items(), key=lambda kv: abs(float(kv[1])), reverse=True):
                avg_q_model = float(vol_model) / max(float(run_duration_s), 1.0e-12)
                self._ui._log(
                    f"  {grp}: volume={float(vol_model):.6f} {vol_unit_label}, "
                    f"avg_q={avg_q_model:.6f} {self._ui._flow_unit_label()}"
                )

        storage_rows: List[Dict[str, float]] = [
            {
                "t_s": 0.0,
                "storage_model": float(storage_start_model),
                "storage_delta_model": 0.0,
            }
        ]
        snapshot_timesteps = list(getattr(self._ui, "_snapshot_timesteps", []) or [])
        terminal_t_s = max(0.0, float(final_sim_time_s))
        if not snapshot_timesteps:
            terminal_snapshot = (
                terminal_t_s,
                np.asarray(h, dtype=np.float64).copy(),
                np.asarray(hu, dtype=np.float64).copy(),
                np.asarray(hv, dtype=np.float64).copy(),
            )
            self._ui._snapshot_timesteps = [terminal_snapshot]
            snapshot_timesteps = [terminal_snapshot]
            self._ui._log(
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
                self._ui._snapshot_timesteps.append(terminal_snapshot)
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

        gpkg_results_path = self._ui._current_line_results_storage_path()
        if gpkg_results_path and hasattr(self._ui, "_persist_conservation_forensics_to_geopackage"):
            try:
                self._ui._persist_conservation_forensics_to_geopackage(
                    gpkg_results_path,
                    run_id,
                    storage_rows,
                    boundary_rows,
                    conservation_summary,
                    source_step_rows=list(source_step_rows_model or []),
                )
            except Exception as exc:
                self._ui._log(f"Conservation forensic persistence warning: {exc}")
        if gpkg_results_path and bool(self._ui.save_line_results_to_gpkg_chk.isChecked()) and self._ui._line_snapshot_rows:
            self._ui._persist_line_results_to_geopackage(
                gpkg_results_path,
                run_id,
                self._ui._line_snapshot_rows,
                profile_rows=self._ui._line_snapshot_profile_rows,
                mesh_interval_s=output_interval_s,
                line_interval_s=line_output_interval_s,
            )
        if gpkg_results_path and bool(self._ui.save_coupling_results_to_gpkg_chk.isChecked()) and self._ui._coupling_snapshot_rows:
            self._ui._persist_coupling_results_to_geopackage(
                gpkg_results_path,
                run_id,
                self._ui._coupling_snapshot_rows,
                interval_s=line_output_interval_s,
            )
        if gpkg_results_path and bool(self._ui.save_mesh_results_to_gpkg_chk.isChecked()) and self._ui._snapshot_timesteps:
            mesh_rows = self._ui._build_mesh_snapshot_rows()
            if mesh_rows:
                mesh_table_name = "swe2d_mesh_results"
                if hasattr(self._ui, "_selected_mesh_results_table_name"):
                    try:
                        mesh_table_name = str(self._ui._selected_mesh_results_table_name() or "swe2d_mesh_results")
                    except Exception:
                        mesh_table_name = "swe2d_mesh_results"
                self._ui._persist_mesh_results_to_geopackage(
                    gpkg_results_path,
                    run_id,
                    mesh_rows,
                    interval_s=output_interval_s,
                    table_name=mesh_table_name,
                )
        if self._ui._results_mesh_mode_enabled and self._ui._snapshot_timesteps:
            try:
                self._ui._ensure_results_mesh_layer_mode()
            except Exception:
                pass
        try:
            self._ui._sync_high_perf_overlay_data()
            if self._ui._snapshot_timesteps:
                self._ui._update_high_perf_overlay_time(float(self._ui._snapshot_timesteps[-1][0]))
        except Exception:
            pass

        run_wallclock_end = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")
        run_duration_wallclock_s = max(0.0, time.perf_counter() - float(run_perf_start))
        self._ui._log(f"Run wallclock end: {run_wallclock_end}")
        self._ui._log(f"Run wallclock duration: {run_duration_wallclock_s:.3f} s")
        if gpkg_results_path and bool(self._ui.save_run_log_to_gpkg_chk.isChecked()) and run_id:
            run_log_text = "\n".join(self._ui._runtime_log_lines[run_log_start_idx:])
            run_log_metadata: Dict[str, object] = {}
            if hasattr(self._ui, "_collect_run_log_metadata"):
                try:
                    run_log_metadata = dict(self._ui._collect_run_log_metadata() or {})
                except Exception:
                    run_log_metadata = {}
            self._ui._persist_run_log_to_geopackage(
                gpkg_results_path,
                run_id,
                run_wallclock_start,
                run_wallclock_end,
                run_duration_wallclock_s,
                run_log_text,
                metadata=run_log_metadata,
            )

        if thiessen_forcing is not None and rain_stats_acc["samples"] > 0:
            avg_r = rain_stats_acc["rain_mm"] / rain_stats_acc["samples"]
            avg_e = rain_stats_acc["excess_mm"] / rain_stats_acc["samples"]
            self._ui._log(
                "Spatial rain/CN summary: "
                f"mean rain={avg_r:.3f} mm/step, mean excess={avg_e:.3f} mm/step"
            )

        self._ui._log("Run complete." if not self._ui._cancel_requested else "Run canceled by user.")
        h_min = float(self._ui.h_min_spin.value())
        wet = h > h_min
        safe_h = np.maximum(h, 1.0e-12)
        vel_mag = np.where(wet, np.sqrt((hu / safe_h) ** 2 + (hv / safe_h) ** 2), 0.0)
        self._ui._log(
            f"Depth range: {float(np.min(h)):.6f} .. {float(np.max(h)):.6f} | "
            f"Velocity mag max (wet cells): {float(np.max(vel_mag)):.6f}"
        )
        if self._ui._line_snapshot_rows:
            self._ui._log(
                f"Sample line rows captured: ts={len(self._ui._line_snapshot_rows)}, "
                f"profile={len(self._ui._line_snapshot_profile_rows)}"
            )
        if self._ui._coupling_snapshot_rows:
            self._ui._log(f"Coupling rows captured: {len(self._ui._coupling_snapshot_rows)}")
        if self._ui._three_d_patch_snapshots:
            self._ui._log(f"3D patch snapshots captured: {len(self._ui._three_d_patch_snapshots)}")
        self._ui._refresh_plot()
