#!/usr/bin/env python3
"""Runtime reporting seam for SWE2D workbench.

Phase 9 goal: extract post-step snapshot/progress/logging reporting from
`_on_run` into a focused helper module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SWE2DRuntimeReporter:
    """Handles per-step diagnostics, snapshot capture, and runtime logging."""

    def __init__(self) -> None:
        self._snapshot_requested = False
        self._snapshot_ready = False
        self._post_readback_callback: Optional[Callable[[], None]] = None
        # Tracks the current coupling snap index (NOT the row index).  Must
        # advance exactly once per coupling snapshot, not once per row.
        self._coupling_snap_idx: int = 0

    def request_snapshot_readback(self) -> None:
        """Set flag — next process_step will read accumulated device snapshots to host."""
        self._snapshot_requested = True

    def set_post_readback_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback invoked after a snapshot readback populates results_data."""
        self._post_readback_callback = callback

    def process_step(
        self,
        *,
        backend: Any,
        t_accum: float,
        dt_used: float,
        last_diag: Dict[str, Any],
        last_valid_cmax: float,
        last_valid_wse_res: float,
        sample_map: Any,
        cell_solver_z: Optional[np.ndarray],
        coupling_controller: Any,
        rain_src: Any,
        state_ms: float,
        ui_ms: float,
        step_wall_t0: float,
        step_ms: float,
        coupling_ms: float,
        source_ms: float,
        bc_ms: float,
        timing_totals_ms: Dict[str, float],
        timing_samples: int,
        i: int,
        run_duration_s: float,
        next_snap_t: float,
        next_line_snap_t: float,
        next_coupling_snap_t: float,
        output_interval_s: float,
        line_output_interval_s: float,
        process_events_interval_s: float,
        last_process_events_wall: float,
        h_min: float,
        length_unit_name: str,
        results_data: Any,
        sample_line_metrics_callback: Callable[..., Any],
        sample_coupling_object_metrics_callback: Callable[..., Any],
        process_events_callback: Callable[[], None],
        set_progress_callback: Callable[[int], None],
        log_callback: Callable[[str], None],
        perf_mode: bool = False,
    ) -> Dict[str, Any]:
        """process step."""
        step_cmax = float(last_diag.get("max_courant", float("nan")))
        if np.isfinite(step_cmax) and step_cmax >= 0.0:
            last_valid_cmax = step_cmax
        step_wse_res = float(
            last_diag.get(
                "max_depth_residual",
                last_diag.get("max_wse_elev_error", float("nan")),
            )
        )
        if np.isfinite(step_wse_res) and step_wse_res >= 0.0:
            last_valid_wse_res = step_wse_res
        t_accum += float(dt_used)

        need_mesh_snap = t_accum >= float(next_snap_t)
        need_line_snap = bool(sample_map) and t_accum >= float(next_line_snap_t)
        need_coupling_snap = (coupling_controller is not None) and (t_accum >= float(next_coupling_snap_t))

        h_s = hu_s = hv_s = None
        # Mesh snapshots: store on-device at every output interval.
        # No D2H readback — device ring buffer accumulates snapshot history.
        # Bulk readback happens only on explicit request (snapshot button / finalize).
        if need_mesh_snap:
            _t_state3 = time.perf_counter()
            backend.store_snapshot(t_accum)
            state_ms += (time.perf_counter() - _t_state3) * 1000.0
            next_snap_t += float(output_interval_s)

        # Line and coupling snapshots: metric computation is deferred to
        # readback time (finalize or snapshot request).  During the run,
        # only the mesh state is accumulated on-device in the ring buffer.
        # When snapshots are read back, line/coupling metrics can be
        # computed from the read-back h/hu/hv arrays.
        if need_line_snap:
            next_line_snap_t += float(line_output_interval_s)
        if need_coupling_snap:
            next_coupling_snap_t += float(line_output_interval_s)
            if results_data is not None and coupling_controller is not None:
                rows = sample_coupling_object_metrics_callback(coupling_controller, t_accum, None)
                snap_idx = self._coupling_snap_idx
                for row in rows:
                    results_data.append_coupling_snapshot(row, snap_idx=snap_idx)
                # Keep the data-layer counter in sync so readers that clamp
                # to _coupling_snap_idx see the right window of valid entries.
                results_data._coupling_snap_idx = max(
                    results_data._coupling_snap_idx, snap_idx + 1
                )
                self._coupling_snap_idx += 1

        # ── On-demand snapshot readback ──────────────────────────────────
        # When request_snapshot_readback() was called (from UI button press),
        # read all accumulated device snapshots to host and populate results_data.
        # This runs on the solver thread — safe for device access.
        if self._snapshot_requested:
            self._snapshot_requested = False
            timesteps = []
            t0 = time.perf_counter()
            n_cells = 0
            try:
                snap_data = backend.read_snapshots()
                if snap_data and "t_s" in snap_data:
                    ts = np.asarray(snap_data["t_s"], dtype=np.float64)
                    h_arr  = np.asarray(snap_data["h"],  dtype=np.float64)
                    hu_arr = np.asarray(snap_data["hu"], dtype=np.float64)
                    hv_arr = np.asarray(snap_data["hv"], dtype=np.float64)
                    n_snaps = int(ts.shape[0])
                    n_cells = int(h_arr.shape[1]) if h_arr.ndim >= 2 else 0
                    for si in range(n_snaps):
                        timesteps.append((
                            float(ts[si]),
                            np.ascontiguousarray(h_arr[si, :]),
                            np.ascontiguousarray(hu_arr[si, :]),
                            np.ascontiguousarray(hv_arr[si, :]),
                        ))
            except Exception:
                logger.warning("Snapshot readback failed", exc_info=True)
            t_read = time.perf_counter()
            if timesteps and results_data is not None:
                try:
                    # Merge with existing live snapshots from earlier
                    # readbacks so data accumulates across multiple fetches.
                    existing = results_data.get_live_snapshot_timesteps()
                    results_data.set_live_snapshot_timesteps(existing + timesteps)
                except Exception:
                    logger.warning("set_live_snapshot_timesteps failed", exc_info=True)
                t_set = time.perf_counter()
                # Compute line TS + profile from the read-back snapshots so
                # the line/profile viewers can render during live runs, not
                # just from GPKG after finalize.  Pre-fix the reporter never
                # invoked sample_line_metrics_callback, leaving
                # _live_line_ts / _live_line_profile empty.
                if sample_map and sample_line_metrics_callback is not None:
                    try:
                        results_data.populate_live_line_metrics(
                            sample_map=sample_map,
                            sample_callback=sample_line_metrics_callback,
                            cell_solver_z=cell_solver_z,
                        )
                    except Exception:
                        logger.warning(
                            "Live line metrics computation failed", exc_info=True
                        )
                t_metrics = time.perf_counter()
                self._snapshot_ready = True
                # Notify the UI that fresh snapshot data is available
                if self._post_readback_callback is not None:
                    try:
                        self._post_readback_callback()
                    except Exception:
                        logger.warning("Post-readback callback failed", exc_info=True)
                t_ui = time.perf_counter()
                logger.info(
                    "[Snapshot] read=%.3fs set=%.3fs metrics=%.3fs ui=%.3fs "
                    "snaps=%d cells=%d",
                    t_read - t0,
                    t_set - t_read,
                    t_metrics - t_set,
                    t_ui - t_metrics,
                    len(timesteps),
                    n_cells,
                )

        _now_wall = time.perf_counter()
        if _now_wall - float(last_process_events_wall) >= float(process_events_interval_s):
            _t_ui0 = time.perf_counter()
            process_events_callback()
            ui_ms += (time.perf_counter() - _t_ui0) * 1000.0
            last_process_events_wall = _now_wall

        step_wall_ms = (time.perf_counter() - step_wall_t0) * 1000.0
        timing_totals_ms["wall"] += step_wall_ms
        timing_totals_ms["step"] += step_ms
        timing_totals_ms["coupling"] += coupling_ms
        timing_totals_ms["source"] += source_ms
        timing_totals_ms["state"] += state_ms
        timing_totals_ms["bc"] += bc_ms
        timing_totals_ms["ui"] += ui_ms
        timing_samples += 1

        pct = int(min(100.0, (t_accum / max(float(run_duration_s), 1.0e-9)) * 100.0))
        set_progress_callback(pct)
        i += 1

        should_log_default = (i == 1 or i % 10 == 0 or pct >= 100)
        should_log_perf = (i == 1 or i % 200 == 0 or pct >= 100)
        if (not perf_mode and should_log_default) or (perf_mode and should_log_perf):
            max_courant = last_valid_cmax
            max_wse_res = last_valid_wse_res
            cmax_txt = f"{max_courant:.5f}" if np.isfinite(max_courant) and max_courant >= 0.0 else "n/a"
            wse_res_txt = f"{max_wse_res:.6e}" if np.isfinite(max_wse_res) and max_wse_res >= 0.0 else "n/a"
            rain_diag_txt = ""
            rain_arr_diag = np.asarray(rain_src, dtype=np.float64)
            if np.any(rain_arr_diag > 0.0):
                rain_diag_txt = " rain:active"
            log_callback(
                (
                    f"step={i} t={t_accum / 3600.0:.3f} hr / {run_duration_s / 3600.0:.3f} hr "
                    f"dt={float(last_diag.get('dt', 0.0)):.5f} "
                    f"gpu={bool(last_diag.get('gpu_active', False))} wet={last_diag.get('wet_cells', '?')} "
                    f"Cmax={cmax_txt} WSEres={wse_res_txt} "
                    f"graph_step={int(last_diag.get('gpu_graph_launches_step', 0))} "
                    f"graph_total={int(last_diag.get('gpu_graph_launches_total', 0))}"
                    f"{rain_diag_txt}"
                )
            )
            tiny_keys = (
                "tiny_mode_requested",
                "tiny_mode_selected",
                "tiny_mode_effective",
                "tiny_mode_fallback",
                "tiny_mode_fallback_count_total",
            )
            if all(k in last_diag for k in tiny_keys):
                log_callback(
                    "  tiny: "
                    f"req={int(last_diag.get('tiny_mode_requested', -1))} "
                    f"sel={int(last_diag.get('tiny_mode_selected', -1))} "
                    f"eff={int(last_diag.get('tiny_mode_effective', -1))} "
                    f"fallback={bool(last_diag.get('tiny_mode_fallback', False))} "
                    f"fallback_total={int(last_diag.get('tiny_mode_fallback_count_total', 0))}"
                )
            if perf_mode:
                return {
                    "t_accum": t_accum,
                    "last_valid_cmax": last_valid_cmax,
                    "last_valid_wse_res": last_valid_wse_res,
                    "next_snap_t": next_snap_t,
                    "next_line_snap_t": next_line_snap_t,
                    "next_coupling_snap_t": next_coupling_snap_t,
                    "last_process_events_wall": last_process_events_wall,
                    "timing_samples": timing_samples,
                    "i": i,
                    "state_ms": state_ms,
                    "ui_ms": ui_ms,
                }
            if timing_samples > 0:
                avg_wall = timing_totals_ms["wall"] / timing_samples
                avg_step = timing_totals_ms["step"] / timing_samples
                avg_cpl = timing_totals_ms["coupling"] / timing_samples
                avg_src = timing_totals_ms["source"] / timing_samples
                avg_state = timing_totals_ms["state"] / timing_samples
                avg_bc = timing_totals_ms["bc"] / timing_samples
                avg_ui = timing_totals_ms["ui"] / timing_samples
                step_gpu_frac = 100.0 * step_ms / max(step_wall_ms, 1.0e-9)
                avg_gpu_frac = 100.0 * avg_step / max(avg_wall, 1.0e-9)
                other_ms = max(0.0, step_wall_ms - (step_ms + coupling_ms + source_ms + state_ms + bc_ms + ui_ms))
                avg_other = max(0.0, avg_wall - (avg_step + avg_cpl + avg_src + avg_state + avg_bc + avg_ui))
                log_callback(
                    "  timing(ms): "
                    f"wall={step_wall_ms:.2f} step={step_ms:.2f} coupling={coupling_ms:.2f} "
                    f"source={source_ms:.2f} state={state_ms:.2f} bc={bc_ms:.2f} ui={ui_ms:.2f} other={other_ms:.2f} "
                    f"gpu_frac={step_gpu_frac:.1f}%"
                )
                log_callback(
                    "  timing-avg(ms): "
                    f"wall={avg_wall:.2f} step={avg_step:.2f} coupling={avg_cpl:.2f} "
                    f"source={avg_src:.2f} state={avg_state:.2f} bc={avg_bc:.2f} ui={avg_ui:.2f} other={avg_other:.2f} "
                    f"gpu_frac={avg_gpu_frac:.1f}%"
                )
            if coupling_controller is not None:
                cdiag = coupling_controller.last_diag
                limiter_events = float(cdiag.component_sums.get("drainage_limiter_events", 0.0))
                limiter_vol_model3 = float(cdiag.component_sums.get("drainage_limiter_volume_m3", 0.0))
                drain_substeps = float(cdiag.component_sums.get("drainage_substeps_used", 1.0))
                native_iterative = int(cdiag.component_sums.get("drainage_native_iterative", 0))
                log_callback(
                    "  coupling: "
                    f"drain_qmax={cdiag.drainage_max_link_flow:.4f} {length_unit_name}^3/s, "
                    f"drain_hmax={cdiag.drainage_max_node_depth:.4f} {length_unit_name}, "
                    f"struct_qsum={cdiag.structure_total_flow:.4f} {length_unit_name}^3/s, "
                    f"src_range=[{cdiag.source_min_mps:.3e}, {cdiag.source_max_mps:.3e}] {length_unit_name}/s, "
                    f"drain_substeps={drain_substeps:.0f}, "
                    f"native_iter={native_iterative}, "
                    f"limiter_events={limiter_events:.0f}, "
                    f"limiter_vol={limiter_vol_model3:.6f} {length_unit_name}^3"
                )

        return {
            "t_accum": t_accum,
            "last_valid_cmax": last_valid_cmax,
            "last_valid_wse_res": last_valid_wse_res,
            "next_snap_t": next_snap_t,
            "next_line_snap_t": next_line_snap_t,
            "next_coupling_snap_t": next_coupling_snap_t,
            "last_process_events_wall": last_process_events_wall,
            "timing_samples": timing_samples,
            "i": i,
            "state_ms": state_ms,
            "ui_ms": ui_ms,
        }
