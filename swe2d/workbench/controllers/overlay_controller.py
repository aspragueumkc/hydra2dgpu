"""High-perf canvas overlay controller.

MVP domain controller.  Methods extracted from the ``high_perf_overlay_bridge``
module and inlined here as ``OverlayController`` methods.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from typing import Any, Dict, Optional

import numpy as np

from swe2d.workbench.controllers.protocols_controller import OverlayView

logger = logging.getLogger(__name__)


# ── Utility functions (kept at module level, not methods) ────────────────


def mesh_fingerprint_from_arrays(
    node_x: np.ndarray, node_y: np.ndarray, cell_nodes: np.ndarray
) -> str:
    """Build a stable mesh fingerprint from node/cell topology arrays."""
    nx = np.asarray(node_x, dtype=np.float64).ravel()
    ny = np.asarray(node_y, dtype=np.float64).ravel()
    tri = np.asarray(cell_nodes, dtype=np.int32).ravel()
    if nx.size <= 0 or ny.size <= 0 or tri.size <= 0:
        return ""

    n_nodes = int(min(nx.size, ny.size))
    n_tri = int(tri.size // 3)
    if n_nodes <= 0 or n_tri <= 0:
        return ""

    sample_nodes = min(n_nodes, 4096)
    sample_tri = min(tri.size, 12288)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(nx[:sample_nodes], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(ny[:sample_nodes], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(tri[:sample_tri], dtype=np.int32).tobytes())
    h.update(f"|n_nodes={n_nodes}|n_tri={n_tri}|".encode("ascii"))
    return h.hexdigest()


# ── OverlayController ────────────────────────────────────────────────────


class OverlayController:
    """Domain controller for high-perf canvas overlay rendering."""

    def __init__(self, view: OverlayView):
        self._view = view
        self._cached_mesh_data: Optional[Dict[str, np.ndarray]] = None
        self._last_mesh_gpkg_run_id: str = ""
        from swe2d.results.data import SWE2DResultsData as _SWE2DRes
        if not hasattr(view, "_results_data") or view._results_data is None:
            view._results_data = _SWE2DRes()

    @property
    def _data(self):
        """Return the SWE2DResultsData instance from the view."""
        return getattr(self._view, "_results_data", None)

    def _get_snapshot_timesteps(self) -> list:
        """Return snapshot timesteps from results data (baked-aware).

        Reconstructs the tuple format from baked numpy arrays for
        compatibility with existing callers.
        """
        d = self._data
        if d is None:
            return []
        # Try baked live arrays first
        if hasattr(d, '_live_times') and d._live_times is not None and d._live_times.size > 0:
            n = d._live_times.size
            if hasattr(d, '_live_h') and d._live_h is not None and d._live_h.size > 0:
                return [(float(d._live_times[i]),
                         d._live_h[i], d._live_hu[i], d._live_hv[i])
                        for i in range(n)]
        # Fall back to old list-of-tuples
        return d.get_live_snapshot_timesteps()

    # ── Inlined bridge methods (converted from `dialog` → `self._view`) ──

    def _set_overlay_scalar_arrays(self, n_cells: int) -> None:
        """Set per-cell Manning's n and CN arrays from mesh metadata.

        Reads ``n_mann_cell`` and ``cn_cell`` from the view's mesh data
        dict and stores them on the results data using the shared naming
        contract.  Falls back to empty arrays when the mesh metadata does
        not contain the arrays or when the cell count is zero.

        Parameters
        ----------
        n_cells : int
            Expected number of cells; arrays are reshaped to ``(n_cells,)``.
        """
        if self._data is None:
            return
        if n_cells <= 0:
            self._data.overlay_cell_mannings_n = np.empty(0, dtype=np.float64)
            self._data.overlay_cell_curve_number = np.empty(0, dtype=np.float64)
            return

        mesh = getattr(self._view, "_mesh_data", {}) or {}

        mann_arr = mesh.get("n_mann_cell")
        if mann_arr is not None:
            mann_arr = np.asarray(mann_arr, dtype=np.float64).ravel()
            if mann_arr.size == n_cells:
                self._data.overlay_cell_mannings_n = mann_arr
            else:
                self._data.overlay_cell_mannings_n = np.empty(0, dtype=np.float64)
        else:
            self._data.overlay_cell_mannings_n = np.empty(0, dtype=np.float64)

        cn_arr = mesh.get("cn_cell")
        if cn_arr is not None:
            cn_arr = np.asarray(cn_arr, dtype=np.float64).ravel()
            if cn_arr.size == n_cells:
                self._data.overlay_cell_curve_number = cn_arr
            else:
                self._data.overlay_cell_curve_number = np.empty(0, dtype=np.float64)
        else:
            self._data.overlay_cell_curve_number = np.empty(0, dtype=np.float64)

    def sync_high_perf_overlay_data(self) -> None:
        """Refresh cached cell-center and bed arrays used by the canvas overlay.

        Two mutually exclusive paths selected by ``data_source``:
        1. Live run (data_source == "live") — reads mesh geometry from in-memory
           ``_mesh_data`` / ``_mesh_cell_centroids()``.  This is the ONLY
           correct source during a live simulation.  Fails loudly if missing.
        2. GPKG results (data_source != "live") — loads baked mesh BLOB from
           ``swe2d_baked_mesh`` in the results GPKG.  Fails loudly if
           the baked mesh entry is missing.  No fallback to in-memory mesh.
        """
        view = self._view
        data_source = self._data.data_source if self._data else "none"

        if data_source == "live":
            # ── Path 1: Live run — in-memory mesh only, fail loudly ──
            cx, cy = view._mesh_cell_centroids()
            if cx is None or cy is None or cx.size <= 0:
                raise RuntimeError(
                    "Live overlay requires mesh centroids — no mesh loaded?"
                )
            bed = view._mesh_cell_solver_bed()
            self._data.overlay_cell_x = np.asarray(cx, dtype=np.float64)
            self._data.overlay_cell_y = np.asarray(cy, dtype=np.float64)
            self._data.overlay_cell_bed = np.asarray(bed, dtype=np.float64)
            mesh = getattr(view, "_mesh_data", {}) or {}
            self._data.overlay_node_x = np.asarray(
                mesh.get("node_x", np.empty(0)), dtype=np.float64
            ).ravel()
            self._data.overlay_node_y = np.asarray(
                mesh.get("node_y", np.empty(0)), dtype=np.float64
            ).ravel()
            raw_cell_nodes = np.asarray(
                mesh.get("cell_nodes", np.empty(0)), dtype=np.int32
            ).ravel()
            if raw_cell_nodes.size <= 0:
                raise RuntimeError(
                    "Live overlay requires mesh cell_nodes — no mesh loaded?"
                )
            if "cell_face_offsets" in mesh and "cell_face_nodes" in mesh:
                offs = np.asarray(mesh["cell_face_offsets"], dtype=np.int32).ravel()
                faces = np.asarray(mesh["cell_face_nodes"], dtype=np.int32).ravel()
                tri_list = []
                tc_list = []
                for ci in range(int(offs.size) - 1):
                    s = int(offs[ci])
                    e = int(offs[ci + 1])
                    ns = faces[s:e]
                    for k in range(1, int(ns.size) - 1):
                        tri_list.append([int(ns[0]), int(ns[k]), int(ns[k + 1])])
                        tc_list.append(ci)
                if tri_list:
                    self._data.overlay_cell_nodes = np.asarray(
                        tri_list, dtype=np.int32
                    ).ravel()
                    self._data.overlay_tri_to_cell = np.asarray(tc_list, dtype=np.int32)
                else:
                    self._data.overlay_cell_nodes = raw_cell_nodes
                    self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)
            else:
                self._data.overlay_cell_nodes = raw_cell_nodes
                self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)
            self._set_overlay_scalar_arrays(int(cx.size))
            self.refresh_high_perf_canvas_overlay(None)

        else:
            # ── Path 2: GPKG results — baked mesh from GPKG only ──
            rec = self._data.overlay_selected_run()
            current_key = ""
            if rec:
                current_key = f"{rec.gpkg_path}::{rec.run_id}"
            needs_reload = (
                self._data.overlay_cell_x is None
                or self._data.overlay_cell_x.size <= 0
                or self._last_mesh_gpkg_run_id != current_key
            )
            if needs_reload:
                try:
                    if not rec:
                        raise ValueError("No enabled run records")
                    gpkg = rec.gpkg_path
                    run_id = rec.run_id
                    if not (gpkg and run_id and os.path.isfile(gpkg)):
                        raise ValueError(f"No valid GPKG for overlay: gpkg={gpkg!r} run={run_id!r}")
                    with sqlite3.connect(gpkg) as conn:
                        row = conn.execute(
                            "SELECT mesh_name, baked_blob FROM swe2d_baked_mesh "
                            "WHERE mesh_name = (SELECT mesh_name FROM swe2d_baked_results "
                            "                   WHERE run_id = ? LIMIT 1)",
                            (run_id,),
                        ).fetchone()
                    if not row:
                        raise ValueError(f"No baked mesh for run_id={run_id!r} in {gpkg}")
                    from hydra_swe2d import swe2d_deserialize_mesh
                    pm = swe2d_deserialize_mesh(row[1])
                    self._data.overlay_cell_x = np.asarray(pm.cell_cx, dtype=np.float64)
                    self._data.overlay_cell_y = np.asarray(pm.cell_cy, dtype=np.float64)
                    self._data.overlay_cell_bed = np.asarray(pm.cell_zb, dtype=np.float64)
                    self._data.overlay_node_x = np.asarray(pm.node_x, dtype=np.float64)
                    self._data.overlay_node_y = np.asarray(pm.node_y, dtype=np.float64)
                    cfn = pm.cell_face_nodes
                    cfo = pm.cell_face_offsets
                    if cfn is not None and cfo is not None:
                        cfn_arr = np.asarray(cfn, dtype=np.int32).ravel()
                        cfo_arr = np.asarray(cfo, dtype=np.int32).ravel()
                        tri_list = []
                        tc_list = []
                        for ci in range(int(cfo_arr.size) - 1):
                            s = int(cfo_arr[ci])
                            e = int(cfo_arr[ci + 1])
                            ring = cfn_arr[s:e]
                            for k in range(1, int(ring.size) - 1):
                                tri_list.append([int(ring[0]), int(ring[k]), int(ring[k + 1])])
                                tc_list.append(ci)
                        if tri_list:
                            self._data.overlay_cell_nodes = np.asarray(tri_list, dtype=np.int32).ravel()
                            self._data.overlay_tri_to_cell = np.asarray(tc_list, dtype=np.int32)
                    self._cached_mesh_data = {"node_x": pm.node_x, "node_y": pm.node_y,
                                              "cell_nodes": pm.cell_nodes}
                    self._last_mesh_gpkg_run_id = current_key
                    self._set_overlay_scalar_arrays(int(self._data.overlay_cell_x.size))
                    return
                except Exception as exc:
                    view._log(
                        f"[HighPerf Overlay] Baked mesh load from GPKG failed: {exc}"
                    )
            # Baked mesh not available — clear overlay
            self._data.overlay_cell_x = np.empty(0, dtype=np.float64)
            self._data.overlay_cell_y = np.empty(0, dtype=np.float64)
            self._data.overlay_cell_bed = np.empty(0, dtype=np.float64)
            self._data.overlay_node_x = np.empty(0, dtype=np.float64)
            self._data.overlay_node_y = np.empty(0, dtype=np.float64)
            self._data.overlay_cell_nodes = np.empty(0, dtype=np.int32)
            self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)
            self._set_overlay_scalar_arrays(0)
            self.refresh_high_perf_canvas_overlay(None)

    def update_high_perf_overlay_time(self, t_s: float) -> None:
        """Update overlay rendering at a specific simulation time."""
        self.refresh_high_perf_canvas_overlay(float(t_s))

    def destroy_high_perf_canvas_overlay_item(self) -> None:
        """Detach overlay canvas item and clear dialog-held references."""
        view = self._view
        item = getattr(view, "_high_perf_canvas_overlay_item", None)
        view._high_perf_canvas_overlay_item = None
        view._high_perf_canvas_overlay_enabled = False
        if item is None:
            return
        try:
            canvas = view._resolve_map_canvas()
            canvas.scene().removeItem(item)
        except Exception:
            logger.warning("Unexpected error silently caught", exc_info=True)

    def ensure_high_perf_canvas_overlay_item(self) -> Any:
        """Create or return the existing canvas overlay item."""
        view = self._view
        item = view._state.high_perf_canvas_overlay_item
        if item is not None:
            return item
        canvas = view._resolve_map_canvas()
        try:
            from swe2d.results.high_perf_viewer import SWE2DHighPerfCanvasOverlayItem

            item = SWE2DHighPerfCanvasOverlayItem(canvas)
            item.setVisible(True)
            item.setZValue(9999.0)
            view._state.high_perf_canvas_overlay_item = item
            return item
        except Exception as exc:
            view._log(f"[HighPerf Overlay] could not create canvas item: {exc}")
            view._state.high_perf_canvas_overlay_item = None
            return None

    def resolve_overlay_time(self, t_s: Any) -> Any:
        """Resolve the overlay time from an explicit value, results data, or snapshot."""
        view = self._view
        if t_s is not None:
            return float(t_s)
        data = getattr(view, "_results_data", None)
        if data is not None:
            try:
                return float(data.current_time_sec)
            except Exception as exc:
                view._log(f"[ERROR] resolve_overlay_time — current_time_sec failed: {exc}")
        _snapshots = self._get_snapshot_timesteps()
        if _snapshots:
            return float(_snapshots[-1][0])
        return None

    def apply_overlay_frame(self, frame: dict) -> None:
        """Apply a rendered overlay frame to the canvas."""
        from swe2d.workbench.services.mesh_data_prep_service import (
            overlay_frame_inputs,
            overlay_frame_is_valid,
        )

        view = self._view
        item = self.ensure_high_perf_canvas_overlay_item()
        if item is None:
            view._log("[HighPerf Overlay] No canvas item to render onto")
            return
        opacity = float(getattr(view, "_overlay_opacity", 1.0))
        qimage, extent, opacity = overlay_frame_inputs(frame, default_opacity=opacity)
        if not overlay_frame_is_valid(qimage):
            view._log("[HighPerf Overlay] rendered frame has null QImage")
            return
        item.setVisible(True)
        item.set_frame(qimage, extent, opacity)
        try:
            canvas = view._resolve_map_canvas()
            canvas.refresh()
            canvas.viewport().update()
        except Exception as exc:
            view._log(f"[ERROR] overlay canvas refresh failed: {exc}")

    def refresh_high_perf_canvas_overlay(self, t_s: Any) -> None:
        """Refresh the high-performance canvas overlay at time t_s."""
        view = self._view
        if not bool(getattr(view, "_high_perf_canvas_overlay_enabled", False)):
            return
        if self._data is None or self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0 or not self._get_snapshot_timesteps():
            return
        # Re-attempt scalar array population from mesh data when the overlay
        # is first rendered if the per-cell Manning/CN arrays are still empty.
        n_cells = int(self._data.overlay_cell_x.size)
        if n_cells > 0 and (
            getattr(self._data, "overlay_cell_mannings_n", None) is None
            or self._data.overlay_cell_mannings_n.size <= 0
            or getattr(self._data, "overlay_cell_curve_number", None) is None
            or self._data.overlay_cell_curve_number.size <= 0
        ):
            self._set_overlay_scalar_arrays(n_cells)
        t_use = self.resolve_overlay_time(t_s)
        if t_use is None:
            return
        try:
            from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image
            from swe2d.workbench.services.overlay_parameters_service import (
                collect_overlay_parameters,
            )

            params = collect_overlay_parameters(view, t_use)
            frame = render_unstructured_snapshot_image(**params)
            if not bool(frame.get("ok", False)):
                view._log(
                    f"[HighPerf Overlay] empty frame: {frame.get('message', 'unknown')}"
                )
                return
            # Store computed color range for reset-to-default feature
            self._data._overlay_computed_vmin = frame.get("computed_vmin", None)
            self._data._overlay_computed_vmax = frame.get("computed_vmax", None)
            # Push computed vmin/vmax back to the spin boxes so they show
            # the actual data range instead of the widget defaults (0/1).
            if self._data._overlay_computed_vmin is not None:
                view.set_overlay_color_range(
                    float(self._data._overlay_computed_vmin),
                    float(self._data._overlay_computed_vmax),
                )
            self.apply_overlay_frame(frame)
        except Exception as exc:
            view._log(f"[HighPerf Overlay] refresh failed: {exc}")

    def _clear_overlay_geometry(self) -> None:
        """Clear all overlay geometry arrays so the next refresh reloads mesh.

        Called when run records change (new GPKG loaded, runs toggled)
        so stale mesh geometry from a previous GPKG is not reused for
        a different run's snapshot data.
        """
        from swe2d.workbench.services.mesh_data_prep_service import (
            create_empty_overlay_arrays,
        )
        if self._data is None:
            return
        empty = create_empty_overlay_arrays()
        self._data.overlay_cell_x = empty["cell_x"]
        self._data.overlay_cell_y = empty["cell_y"]
        self._data.overlay_cell_bed = empty["cell_bed"]
        self._data.overlay_node_x = empty["node_x"]
        self._data.overlay_node_y = empty["node_y"]
        self._data.overlay_cell_nodes = empty["cell_nodes"]
        self._data.overlay_tri_to_cell = empty["tri_to_cell"]
        self._cached_mesh_data = None
        self._last_mesh_gpkg_run_id = ""

    def reset_runtime_snapshot_overlay_cache(self, reason: str = "") -> None:
        """Reset all snapshot/overlay cache state on the dialog."""
        from swe2d.workbench.services.mesh_data_prep_service import (
            create_empty_overlay_arrays,
        )

        view = self._view
        if self._data is None:
            return
        self._data.clear_live_snapshots()
        # Clear stale run records/selections so the next run starts clean.
        self._data._run_records = []
        self._data._selected_run_keys = set()
        self._data._overlay_selected_key = ""
        view._snapshot_mesh_fingerprint = ""
        self._data.set_data_source("none")
        view._overlay_last_loaded_t_s = None
        empty = create_empty_overlay_arrays()
        self._data.overlay_cell_x = empty["cell_x"]
        self._data.overlay_cell_y = empty["cell_y"]
        self._data.overlay_cell_bed = empty["cell_bed"]
        self._data.overlay_node_x = empty["node_x"]
        self._data.overlay_node_y = empty["node_y"]
        self._data.overlay_cell_nodes = empty["cell_nodes"]
        self._data.overlay_tri_to_cell = empty["tri_to_cell"]

        item = view._state.high_perf_canvas_overlay_item
        if item is not None:
            try:
                item.clear()
            except Exception as exc:
                view._log(f"[ERROR] reset runtime snapshot overlay cache failed: {exc}")
        if reason:
            view._log(f"[HighPerf Overlay] Cleared snapshot/overlay cache: {reason}")

    # ── Original methods ───────────────────────────────────────────────

    def on_high_perf_canvas_overlay_toggled(self, checked: bool) -> None:
        """Enable or disable the high-perf canvas overlay rendering.

        When disabled the overlay item is removed from the scene and
        the canvas is refreshed. When enabled the overlay is loaded
        for the current slider time and rendered immediately.
        """
        view = self._view
        view._high_perf_canvas_overlay_enabled = bool(checked)
        if not view._high_perf_canvas_overlay_enabled:
            item = view._state.high_perf_canvas_overlay_item
            if item is not None:
                try:
                    canvas = item._canvas()
                    canvas.scene().removeItem(item)
                except Exception as _e:

                    logger.warning(f"[ERROR] Exception in overlay_controller.py: {_e}")
                view._state.high_perf_canvas_overlay_item = None
            iface = view._resolve_qgis_iface()
            iface.mapCanvas().refresh()
            return

        data = getattr(view, "_results_data", None)
        t_s = 0.0
        if data is not None and hasattr(data, "current_time_sec"):
            try:
                t_s = float(data.current_time_sec)
            except Exception:
                t_s = 0.0

        # GPKG path: load_mesh_snapshot_for_overlay handles the full overlay
        # setup (mesh geometry + snapshot seeding + canvas refresh). When it
        # succeeds, skip the LIVE-path sync below so baked-mesh geometry is
        # not overwritten by view._mesh_cell_centroids() (Bug 1 fix).
        # Live path: snapshots already exist, go straight to sync+refresh.
        if not self._get_snapshot_timesteps():
            if self.load_mesh_snapshot_for_overlay(t_s):
                return
        self.sync_high_perf_overlay_data()
        self.refresh_high_perf_canvas_overlay(t_s)

    def on_high_perf_canvas_overlay_style_changed(self, *_: Any) -> None:
        """Enable/disable overlay controls based on the selected style."""
        view = self._view
        view.sync_overlay_widget_states()
        if bool(getattr(view, "_high_perf_canvas_overlay_enabled", False)):
            if not self._get_snapshot_timesteps() and (self._data is None or self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0):
                return
            self.refresh_high_perf_canvas_overlay(None)

    def export_high_perf_overlay_to_geotiff(self) -> None:
        """Export the current high-perf overlay frame to a GeoTIFF file.

        Uses the renderer to compute a scalar grid then writes it via
        GDAL. Aborts (with warning) when no overlay data is available
        or when GDAL is missing.
        """
        view = self._view
        _geotiff_snapshots = self._get_snapshot_timesteps()
        if self._data is None or self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0 or not _geotiff_snapshots:
            view.show_warning_message(
                "Export GeoTIFF",
                "No high-perf overlay data is available. "
                "Run a model with output intervals set, then enable the overlay.",
            )
            return

        start_dir = str(view._current_line_results_storage_path() or ".")
        if start_dir and os.path.exists(os.path.dirname(start_dir)):
            start_dir = os.path.dirname(start_dir)
        out_path = view.show_get_save_file(
            "Export High-Perf Overlay to GeoTIFF",
            start_dir,
            "GeoTIFF (*.tif *.tiff)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith((".tif", ".tiff")):
            out_path += ".tif"

        from swe2d.results.high_perf_viewer import render_unstructured_snapshot_image

        field_key = view.get_overlay_export_field()
        cmap_key = view.get_overlay_export_cmap()
        wse_render_mode = view.get_overlay_export_wse_render_mode()
        auto_contrast = view.get_overlay_auto_contrast()

        t_use = None
        data = getattr(view, "_results_data", None)
        if data is not None:
            try:
                t_use = float(data.current_time_sec)
            except Exception as e:
                view._log(f"[ERROR] export high perf overlay to geotiff failed: {e}")
                t_use = None
        if t_use is None and _geotiff_snapshots:
            t_use = float(_geotiff_snapshots[-1][0])

        pixel_size, ok = view.show_get_double(
            "Export GeoTIFF",
            "Pixel size (map units):",
            10.0,
            0.001,
            1.0e6,
        )
        if not ok:
            return
        pixel_size = max(1.0e-6, abs(pixel_size))

        cx = self._data.overlay_cell_x
        cy = self._data.overlay_cell_y
        x_min = float(np.nanmin(cx))
        x_max = float(np.nanmax(cx))
        y_min = float(np.nanmin(cy))
        y_max = float(np.nanmax(cy))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = 0.0, 1.0
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = 0.0, 1.0

        nx = max(32, int(np.ceil((x_max - x_min) / pixel_size)))
        ny = max(32, int(np.ceil((y_max - y_min) / pixel_size)))

        field_labels = {
            "depth": f"Depth ({view._length_unit_name})",
            "speed": f"Velocity ({view._length_unit_name}/s)",
            "wse": f"Water Surface ({view._length_unit_name})",
        }
        legend_label = field_labels.get(field_key, str(field_key))
        try:
            frame = render_unstructured_snapshot_image(
                cell_x=cx,
                cell_y=cy,
                cell_bed=self._data.overlay_cell_bed,
                node_x=self._data.overlay_node_x,
                node_y=self._data.overlay_node_y,
                cell_nodes=self._data.overlay_cell_nodes,
                tri_to_cell=self._data.overlay_tri_to_cell,
                timesteps=_geotiff_snapshots,
                current_time_s=float(t_use),
                field_key=field_key,
                wse_render_mode=wse_render_mode,
                cmap_key=cmap_key,
                resolution=(nx, ny),
                auto_contrast=auto_contrast,
                show_velocity_arrows=False,
                show_streamlines=False,
                render_extent_world=(x_min, x_max, y_min, y_max),
                show_legend=True,
                legend_label=legend_label,
            )
        except Exception as exc:
            view._log(f"[GeoTIFF Export] render error: {exc}")
            view.show_warning_message("Export GeoTIFF", f"Overlay render failed:\n{exc}")
            return

        if not bool(frame.get("ok", False)):
            msg = str(frame.get("message", "unknown render error"))
            view._log(f"[GeoTIFF Export] empty frame: {msg}")
            view.show_warning_message("Export GeoTIFF", f"Nothing rendered:\n{msg}")
            return

        scalar_grid = frame.get("grid")
        grid_mask = frame.get("grid_mask")
        if scalar_grid is None or grid_mask is None:
            view._log("[GeoTIFF Export] renderer did not return a scalar grid.")
            view.show_warning_message(
                "Export GeoTIFF",
                "The overlay renderer did not expose raw data values.\n"
                "A code update is required in swe2d.results.high_perf_viewer.",
            )
            return

        h_img, w = scalar_grid.shape
        grid_out = np.full((h_img, w), np.nan, dtype=np.float64)
        grid_out[grid_mask] = scalar_grid[grid_mask]

        crs_auth = "EPSG:4326"
        try:
            from qgis.core import QgsProject

            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception as e:
            view._log(f"[ERROR] export high perf overlay to geotiff failed: {e}")

        from swe2d.services.geotiff_export_service import export_overlay_grid_to_geotiff

        x_res = (x_max - x_min) / max(1, w)
        y_res = (y_max - y_min) / max(1, h_img)
        export_overlay_grid_to_geotiff(
            arr=grid_out,
            xmin=x_min,
            ymax=y_max,
            dx=x_res,
            dy=y_res,
            path=out_path,
            nodata=np.nan,
            crs_auth=crs_auth,
            band_description=str(field_key),
        )

        vmin = float(np.nanmin(grid_out))
        vmax = float(np.nanmax(grid_out))
        view._log(
            f"High-perf overlay exported to GeoTIFF: {out_path} "
            f"({w}x{h_img}, CRS={crs_auth}, field={field_key}, "
            f"t={t_use / 3600.0:.3f} hr, "
            f"range=[{vmin:.6g}, {vmax:.6g}])"
        )
        view.show_warning_message(
            "Export GeoTIFF",
            f"Exported {w}x{h_img} single-band Float64 to:\n{out_path}\n"
            f"CRS: {crs_auth}\n"
            f"Field: {field_key}  Time: {t_use / 3600.0:.3f} hr\n"
            f"Pixel size: {pixel_size:.4f} map units\n"
            f"Value range: [{vmin:.6g}, {vmax:.6g}] {view._length_unit_name}",
        )

    # ── Inlined results_bridge methods ────────────────────────────────

    def maybe_create_results_data(self) -> None:
        """Create the results data layer and wire it into the results view."""
        view = self._view
        from swe2d.results.data import SWE2DResultsData

        view._results_data = SWE2DResultsData()

    # ── End inlined results_bridge methods ────────────────────────────

    def load_mesh_snapshot_for_overlay(self, t_s: float) -> bool:
        """Load mesh snapshot for overlay rendering from GPKG.

        Reads the snapshot at time t_s from the GeoPackage and renders it
        on the map canvas. Does NOT replace _snapshot_timesteps — the
        in-memory list is preserved for animation playback.

        Returns True if snapshot was loaded and rendered, False otherwise.
        """
        view = self._view
        data = getattr(view, "_results_data", None)
        if data is None:
            return False
        # Resolve the per-run GPKG from the enabled RunRecord — NOT from
        # an overarching data-level path.  Each run's mesh lives in its
        # own GPKG and the overlay must read from there.
        rec = data.overlay_selected_run()
        if rec is None:
            return False
        gpkg = str(rec.gpkg_path or "")
        run_id = str(rec.run_id or "")
        if not gpkg or not os.path.exists(gpkg):
            return False

        # Load overlay mesh geometry from the baked mesh BLOB in GPKG only.
        # No fallback to in-memory _mesh_data — if the GPKG is missing
        # a swe2d_baked_mesh entry this fails loudly.
        current_gpkg_run_id = f"{gpkg}::{run_id}"
        if (self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0
                or self._last_mesh_gpkg_run_id != current_gpkg_run_id):
            self.sync_high_perf_overlay_data()
        if self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0:
            view._log(
                "[HighPerf Overlay] No mesh geometry available — "
                "cannot render overlay for GPKG results. "
                "The GPKG may be missing a swe2d_baked_mesh entry."
            )
            return False

        

        from swe2d.services.gpkg_persistence_service import load_baked_snapshot
        snapshot = load_baked_snapshot(gpkg, run_id, t_s)
        if snapshot is None:
            view._log("[HighPerf Overlay] No mesh snapshot found in GeoPackage.")
            return False

        h = snapshot["h"]
        hu = snapshot["hu"]
        hv = snapshot["hv"]
        nearest_ts = snapshot["t_s"]
        # Seed in-memory snapshot from GPKG so the overlay renderer has data.
        # data_source stays "gpkg" so that scrubbing to a different time
        # triggers a new GPKG load rather than reading stale in-memory data.
        # Save and restore coupling records around clear_live_snapshots so
        # slider scrubbing doesn't destroy the coupling cache (Bug 3 fix).
        _saved_coupling = self._data._coupling_records[:] if hasattr(self._data, '_coupling_records') else []
        _saved_coupling_run_id = str(getattr(self._data, '_coupling_run_id', ''))
        _saved_coupling_gpkg = str(getattr(self._data, '_coupling_gpkg_path', ''))
        self._data.clear_live_snapshots()
        self._data._coupling_records = _saved_coupling
        self._data._coupling_run_id = _saved_coupling_run_id
        self._data._coupling_gpkg_path = _saved_coupling_gpkg
        self._data.append_live_snapshot(
            float(nearest_ts),
            np.asarray(h, dtype=np.float64).copy(),
            np.asarray(hu, dtype=np.float64).copy(),
            np.asarray(hv, dtype=np.float64).copy(),
        )
        self._data.set_data_source("gpkg")
        view._overlay_last_loaded_t_s = nearest_ts
        self.update_high_perf_overlay_time(float(nearest_ts))
        logger.info(
            "[HighPerf Overlay] Loaded %d cells from GPKG run %s at t=%.2fs",
            h.size, run_id, nearest_ts,
        )
        return True
