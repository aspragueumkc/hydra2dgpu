"""High-perf canvas overlay controller.

MVP domain controller.  Methods extracted from the ``high_perf_overlay_bridge``
module and inlined here as ``OverlayController`` methods.
"""

from __future__ import annotations

import hashlib
import logging
import os
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
        from swe2d.results.data import SWE2DResultsData as _SWE2DRes
        if not hasattr(view, "_results_data") or view._results_data is None:
            view._results_data = _SWE2DRes()

    @property
    def _data(self):
        """Return the SWE2DResultsData instance from the view."""
        return getattr(self._view, "_results_data", None)

    def _get_snapshot_timesteps(self) -> list:
        """Return snapshot timesteps from results data."""
        return self._data.get_live_snapshot_timesteps()

    # ── Inlined bridge methods (converted from `dialog` → `self._view`) ──

    def sync_high_perf_overlay_data(self) -> None:
        """Refresh cached cell-center and bed arrays used by the canvas overlay.

        When snapshots exist, geometry is derived from them. When snapshots are
        empty (e.g., GPKG-loaded results), geometry is built from _mesh_data
        so the overlay can still render GPKG snapshots.
        """
        view = self._view
        _snapshots = self._data.get_live_snapshot_timesteps()
        if not _snapshots:
            # No in-memory snapshots — build geometry from mesh data or GPKG
            mesh = getattr(view, "_mesh_data", None) or {}
            if mesh.get("node_x") is None or mesh.get("cell_nodes") is None:
                mesh = self._cached_mesh_data or {}
                if mesh.get("node_x") is None:
                    # ponytail: load mesh from the same GPKG as the results
                    _data = getattr(view, "_results_data", None)
                    gpkg = str(getattr(_data, "gpkg_path", "") or "")
                    if gpkg and os.path.isfile(gpkg):
                        try:
                            from swe2d.workbench.services.gpkg_persistence_service import (
                                load_mesh_from_geopackage,
                            )
                            import sqlite3
                            conn = sqlite3.connect(gpkg)
                            try:
                                cur = conn.cursor()
                                cur.execute(
                                    "SELECT mesh_name FROM swe2d_mesh "
                                    "ORDER BY created_utc DESC LIMIT 1"
                                )
                                row = cur.fetchone()
                                if row:
                                    loaded = load_mesh_from_geopackage(gpkg, str(row[0]))
                                    if loaded and loaded.get("node_x") is not None:
                                        self._cached_mesh_data = loaded
                                        mesh = loaded
                            finally:
                                conn.close()
                        except Exception as exc:
                            logger.warning("Failed to load mesh from GPKG for overlay: %s", exc)
            if mesh.get("node_x") is not None and mesh.get("cell_nodes") is not None:
                try:
                    cx, cy = view._mesh_cell_centroids()
                    bed = view._mesh_cell_solver_bed()
                    self._data.overlay_cell_x = np.asarray(cx, dtype=np.float64)
                    self._data.overlay_cell_y = np.asarray(cy, dtype=np.float64)
                    self._data.overlay_cell_bed = np.asarray(bed, dtype=np.float64)
                    self._data.overlay_node_x = np.asarray(
                        mesh.get("node_x", np.empty(0)), dtype=np.float64
                    ).ravel()
                    self._data.overlay_node_y = np.asarray(
                        mesh.get("node_y", np.empty(0)), dtype=np.float64
                    ).ravel()
                    raw_cell_nodes = np.asarray(
                        mesh.get("cell_nodes", np.empty(0)), dtype=np.int32
                    ).ravel()
                    if "cell_face_offsets" in mesh and "cell_face_nodes" in mesh:
                        offs = np.asarray(mesh["cell_face_offsets"], dtype=np.int32).ravel()
                        faces = np.asarray(mesh["cell_face_nodes"], dtype=np.int32).ravel()
                        tri_list: list[list[int]] = []
                        tc_list: list[int] = []
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
                            self._data.overlay_tri_to_cell = np.asarray(
                                tc_list, dtype=np.int32
                            )
                        else:
                            self._data.overlay_cell_nodes = raw_cell_nodes
                            self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)
                    else:
                        self._data.overlay_cell_nodes = raw_cell_nodes
                        self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)
                except Exception as exc:
                    view._log(f"[HighPerf Overlay] Mesh-based geometry sync failed: {exc}")
                    self._data.overlay_cell_x = np.empty(0, dtype=np.float64)
                    self._data.overlay_cell_y = np.empty(0, dtype=np.float64)
                    self._data.overlay_cell_bed = np.empty(0, dtype=np.float64)
                    self._data.overlay_node_x = np.empty(0, dtype=np.float64)
                    self._data.overlay_node_y = np.empty(0, dtype=np.float64)
                    self._data.overlay_cell_nodes = np.empty(0, dtype=np.int32)
                    self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)

            else:
                self._data.overlay_cell_x = np.empty(0, dtype=np.float64)
                self._data.overlay_cell_y = np.empty(0, dtype=np.float64)
                self._data.overlay_cell_bed = np.empty(0, dtype=np.float64)
                self._data.overlay_node_x = np.empty(0, dtype=np.float64)
                self._data.overlay_node_y = np.empty(0, dtype=np.float64)
                self._data.overlay_cell_nodes = np.empty(0, dtype=np.int32)
                self._data.overlay_tri_to_cell = np.empty(0, dtype=np.int32)

            self.refresh_high_perf_canvas_overlay(None)
            return

        try:
            cx, cy = view._mesh_cell_centroids()
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
        except Exception as exc:
            view._log(f"[HighPerf Overlay] Data sync failed: {exc}")

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
            self.apply_overlay_frame(frame)
        except Exception as exc:
            view._log(f"[HighPerf Overlay] refresh failed: {exc}")

    def reset_runtime_snapshot_overlay_cache(self, reason: str = "") -> None:
        """Reset all snapshot/overlay cache state on the dialog."""
        from swe2d.workbench.services.mesh_data_prep_service import (
            create_empty_overlay_arrays,
        )

        view = self._view
        if self._data is None:
            return
        self._data.clear_live_snapshots()
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
                except Exception:
                    pass
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

        if not self._get_snapshot_timesteps() or (self._data is not None and self._data.data_source == "gpkg"):
            self.load_mesh_snapshot_for_overlay(t_s)
            self.sync_high_perf_overlay_data()
            self.refresh_high_perf_canvas_overlay(t_s)
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

        from osgeo import gdal, osr

        crs_auth = "EPSG:4326"
        try:
            from qgis.core import QgsProject

            proj_crs = QgsProject.instance().crs()
            if proj_crs is not None and proj_crs.isValid():
                crs_auth = proj_crs.authid() or crs_auth
        except Exception as e:
            view._log(f"[ERROR] export high perf overlay to geotiff failed: {e}")

        if gdal is not None:
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(out_path, w, h_img, 1, gdal.GDT_Float64)
            if ds is None:
                raise RuntimeError("GDAL could not create output dataset.")
            x_res = (x_max - x_min) / max(1, w)
            y_res = (y_max - y_min) / max(1, h_img)
            gt = (x_min, x_res, 0.0, y_max, 0.0, -y_res)
            ds.SetGeoTransform(gt)

            srs = osr.SpatialReference()
            srs.SetFromUserInput(crs_auth)
            ds.SetProjection(srs.ExportToWkt())

            band = ds.GetRasterBand(1)
            band.WriteArray(grid_out)
            band.SetNoDataValue(np.nan)
            band.SetDescription(str(field_key))
            ds.FlushCache()
            ds = None
        else:
            raise RuntimeError("GDAL is not available. Cannot write GeoTIFF.")

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

        gpkg = ""
        view._results_data = SWE2DResultsData(gpkg_path=gpkg)

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
        gpkg = getattr(data, "gpkg_path", None)
        if not gpkg or not os.path.exists(gpkg):
            return False
        run_ids = data.enabled_overlay_targets()
        if not run_ids:
            return False
        run_id = run_ids[0][1]

        if self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0:
            if getattr(view, "_mesh_data", None) is None:
                view._log(
                    "[HighPerf Overlay] No mesh loaded — cannot render overlay for GPKG results."
                )
                return False
            self.sync_high_perf_overlay_data()
        if self._data.overlay_cell_x is None or self._data.overlay_cell_x.size <= 0:
            return False

        from swe2d.workbench.services import gpkg_service

        snapshot = gpkg_service.load_mesh_snapshot(gpkg, run_id, t_s)
        if snapshot is None:
            view._log("[HighPerf Overlay] No mesh snapshot found in GeoPackage.")
            return False

        h = snapshot["h"]
        hu = snapshot["hu"]
        hv = snapshot["hv"]
        nearest_ts = snapshot["t_s"]
        self._data.clear_live_snapshots()
        self._data.append_live_snapshot(
            float(nearest_ts),
            np.asarray(h, dtype=np.float64).copy(),
            np.asarray(hu, dtype=np.float64).copy(),
            np.asarray(hv, dtype=np.float64).copy(),
        )
        if data is not None:
            data.set_data_source("gpkg")
        view._overlay_last_loaded_t_s = nearest_ts
        self.update_high_perf_overlay_time(float(nearest_ts))
        view._log(
            f"[HighPerf Overlay] Loaded {h.size} cells from GPKG run {run_id} "
            f"at t={nearest_ts:.2f}s"
        )
        return True
