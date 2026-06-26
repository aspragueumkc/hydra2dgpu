"""Mesh import, terrain assignment, GeoPackage creation, and exports.

MVP domain controller.  Currently a shell — methods are progressively
extracted from ``WorkbenchController`` as they are rewritten to use the
``MeshView`` protocol instead of direct widget access.
"""

from __future__ import annotations

import numpy as np

from swe2d.workbench.controllers.protocols_controller import MeshView


class MeshController:
    """Domain controller for mesh operations."""

    def __init__(self, view: MeshView):
        self._view = view


    # ── Results GeoPackage selection ───────────────────────────────────
    def on_select_results_gpkg(self) -> None:
        """Open a file dialog to select an existing results GeoPackage.

        Updates ``results_gpkg_path_edit`` if present and logs the
        selected path. Behaviour matches the legacy method exactly.
        """
        from qgis.PyQt import QtWidgets

        view = self._view
        default_path = str(view._current_line_results_storage_path() or "")
        if not os.path.exists(default_path):
            default_path = str(view._model_gpkg_path or "")
        gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            view,
            "Select Existing Results GeoPackage",
            default_path,
            "GeoPackage (*.gpkg)",
        )
        if not gpkg_path:
            return
        if not os.path.exists(gpkg_path):
            QtWidgets.QMessageBox.warning(
                view,
                "Results GeoPackage",
                "Please select an existing GeoPackage file.",
            )
            return
        view.set_results_gpkg_path(gpkg_path)
        view._log(f"Results GeoPackage override set: {gpkg_path}")

    # ── Mesh import orchestration ──────────────────────────────────────


    # ── Mesh import orchestration ──────────────────────────────────────
    def import_mesh_from_layers(self) -> None:
        """Import mesh from the currently-selected QGIS vector layers.

        The full implementation (formerly in ``_import_mesh_from_layers``
        in ``extracted/topology_and_io_methods.py``) is inlined here.
        Reads all widget/state from ``self._view``.
        """
        from swe2d.services.mesh_extraction_service import (
            extract_mesh_from_layer_data,
        )

        view = self._view
        try:
            from qgis.core import QgsProject
            _have_qgis = True
        except ImportError:
            _have_qgis = False
        if not _have_qgis:
            return
        nodes_layer = view.get_combo_selected_layer("nodes_layer_combo", "vector")
        cells_layer = view.get_combo_selected_layer("cells_layer_combo", "vector")
        if nodes_layer is None or cells_layer is None:
            view._log("Select both nodes and cells vector layers.")
            return

        extracted = extract_mesh_from_layer_data(
            nodes_layer=nodes_layer,
            cells_layer=cells_layer,
            log_fn=view._log,
        )
        if not extracted:
            view._log("No valid node or cell features found in selected layers.")
            return

        view._mesh_data = extracted
        view._reset_runtime_snapshot_overlay_cache("mesh imported from selected layers")
        n_faces = int(max(0, extracted.get("cell_face_offsets", np.zeros(1)).size - 1))
        n_tris = int(extracted.get("cell_nodes", np.zeros(1)).size // 3)
        node_x = extracted.get("node_x", np.zeros(1))

        try:
            view.set_layer_status_text(
                f"Loaded map mesh: nodes={node_x.size}, faces={n_faces}, triangles={n_tris}"
            )
        except RuntimeError:
            pass
        try:
            view.set_layer_status_text("Mesh loaded from selected map layers.")
        except RuntimeError:
            pass
        view._log(
            f"Imported mesh from map layers: nodes={node_x.size}, faces={n_faces}, triangles={n_tris}"
        )
        view._result_data = None
        try:
            viewer = getattr(view, "_studio_viewer", None)
            if viewer is not None:
                viewer.tab_widget.setCurrentWidget(
                    viewer.plot_widgets.get("Mesh"))
        except RuntimeError:
            pass
        try:
            view._refresh_plot()
        except RuntimeError:
            pass

    # ── 2D model GeoPackage creation ──────────────────────────────────


    # ── 2D model GeoPackage creation ──────────────────────────────────
    def create_2d_model_geopackage(self) -> None:
        """Create a new 2D model GeoPackage.

        The full implementation (formerly in ``_create_2d_model_geopackage``
        in ``extracted/topology_and_io_methods.py``) is inlined here.
        Reads all widget/state from ``self._view``.
        """
        from qgis.PyQt import QtWidgets
        from qgis.core import QgsProject

        view = self._view
        try:
            from qgis.core import QgsProject as _probe
            _have_qgis_local = True
        except ImportError:
            _have_qgis_local = False
        if not _have_qgis_local:
            view._log("QGIS layer API unavailable; cannot create model GeoPackage.")
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            view,
            "Create 2D Model GeoPackage",
            "swe2d_model.gpkg",
            "GeoPackage (*.gpkg)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".gpkg"):
            out_path += ".gpkg"

        crs_auth = "EPSG:4326"
        try:
            crs = QgsProject.instance().crs()
            if crs is not None and crs.isValid():
                crs_auth = crs.authid() or crs_auth
        except Exception as e:
            view._log(f"[ERROR] CRS auth for GPKG export: {e}")

        from swe2d.workbench.services.schema_definitions import (
            create_memory_layer,
            get_layer_names,
        )

        model_layers = [
            create_memory_layer(key, crs_auth)
            for key in get_layer_names()
        ]

        for i, lyr in enumerate(model_layers):
            view._write_memory_layer_to_gpkg(lyr, out_path, lyr.name(), create_file=(i == 0))

        # Store QML editor-widget styles in GPKG layer_styles table
        import os as _os
        from swe2d.workbench.services.gpkg_layer_styles_service import (
            write_qml_styles_to_gpkg as _write_styles,
        )
        _qml_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
            "QML",
        )
        if _os.path.isdir(_qml_dir):
            _write_styles(out_path, _qml_dir)

        view._log(f"Created 2D model GeoPackage: {out_path}")
        view.set_layer_status_text("2D model GeoPackage created.")
        view._load_2d_model_geopackage(path_override=out_path)

    # ── Layer combo orchestration ─────────────────────────────────────
    # ── High-perf canvas overlay toggle ───────────────────────────────


    # ── HEC-RAS HDF5 mesh export ─────────────────────────────────────
    def export_mesh_to_ugrid(self) -> None:
        """Export in-memory mesh geometry to UGRID NetCDF."""
        from swe2d.services.ugrid_export_service import write_ugrid_nc

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        if mesh is None:
            view._log("[ERROR] No mesh loaded for UGRID export.")
            return

        from qgis.PyQt import QtWidgets
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            view, "Export Mesh to UGRID NetCDF", "", "NetCDF (*.nc);;All (*.*)"
        )
        if not out_path:
            return

        try:
            from qgis.core import QgsProject
            crs_wkt = "LOCAL_CS[\"Unknown\"]"
            try:
                crs = QgsProject.instance().crs()
                if crs is not None and crs.isValid():
                    crs_wkt = crs.toWkt()
            except Exception as _e:

                try:

                    view._log(f"[ERROR] Exception in mesh_controller.py: {_e}")

                except Exception:

                    pass

            write_ugrid_nc(
                path=out_path,
                mesh_data=mesh,
                crs_wkt=crs_wkt,
                log_fn=view._log,
                gravity=getattr(view, "_gravity", 9.81),
                h_min=getattr(view, "_h_min", 0.01),
                n_mann=getattr(view, "_n_mann_default", 0.03),
                is_us_customary=bool(getattr(view, "_is_us_customary", False)),
                length_unit_name=getattr(view, "_length_unit_name", "m"),
            )
            view._log(f"Mesh exported to UGRID NetCDF: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(view, "UGRID Export Error", str(e))
            view._log(f"[ERROR] UGRID mesh export: {e}")

    # ── HEC-RAS HDF5 results export ──────────────────────────────────


    # ── HEC-RAS HDF5 results export ──────────────────────────────────
    def export_results_to_hdf5(self) -> None:
        """Export simulation results to HEC-RAS HDF5 format."""
        from swe2d.workbench.services.hecras_export_service import write_hecras_hdf5

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        rd = getattr(view, "_results_data", None)
        timesteps = rd.get_live_snapshot_timesteps()
        if mesh is None or not timesteps:
            from qgis.PyQt import QtWidgets
            QtWidgets.QMessageBox.warning(
                view, "Export HDF5", "No mesh or simulation results available."
            )
            return

        from qgis.PyQt import QtWidgets
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            view, "Export Results to HEC-RAS HDF5", "", "HDF5 (*.h5);;All (*.*)"
        )
        if not out_path:
            return

        try:
            from qgis.core import QgsProject
            projection_wkt = "LOCAL_CS[\"Unknown\"]"
            try:
                crs = QgsProject.instance().crs()
                if crs is not None and crs.isValid():
                    projection_wkt = crs.toWkt()
            except Exception as _e:

                try:

                    view._log(f"[ERROR] Exception in mesh_controller.py: {_e}")

                except Exception:

                    pass

            write_hecras_hdf5(
                path=out_path,
                mesh_data=mesh,
                timesteps=timesteps,
                projection_wkt=projection_wkt,
                log_fn=view._log,
                result_data=getattr(view, "_result_data", None),
                gravity=getattr(view, "_gravity", 9.81),
                h_min=getattr(view, "_h_min", 0.01),
                n_mann=getattr(view, "_n_mann_default", 0.03),
                is_us_customary=bool(getattr(view, "_is_us_customary", False)),
                length_unit_name=getattr(view, "_length_unit_name", "m"),
                include_extra=True,
            )
            view._log(f"Results exported to HEC-RAS HDF5: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(view, "HDF5 Export Error", str(e))
            view._log(f"[ERROR] HDF5 results export: {e}")

    # ── UGRID NetCDF results export ──────────────────────────────────


    # ── UGRID NetCDF results export ──────────────────────────────────
    def export_results_to_ugrid(self) -> None:
        """Export simulation results to UGRID NetCDF format."""
        from swe2d.services.ugrid_export_service import write_ugrid_nc

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        rd = getattr(view, "_results_data", None)
        timesteps = rd.get_live_snapshot_timesteps()

        from qgis.PyQt import QtWidgets
        if mesh is None or not timesteps:
            QtWidgets.QMessageBox.warning(
                view, "Export UGRID", "No mesh or simulation results available."
            )
            return

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            view, "Export Results to UGRID NetCDF",
            (getattr(view, "_current_line_results_storage_path", lambda: "")() or ""),
            "NetCDF (*.nc);;All (*.*)",
        )
        if not out_path:
            return

        try:
            write_ugrid_nc(out_path, mesh, timesteps=timesteps,
                           log_fn=view._log,
                           length_unit_name=getattr(view, "_length_unit_name", "m"),
                           is_us_customary=bool(getattr(view, "_is_us_customary", False)),
                           gravity=getattr(view, "_gravity", 9.81),
                           h_min=getattr(view, "_h_min", 0.01))
            view._log(f"Results exported to UGRID: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(view, "UGRID Export Error", str(e))
            view._log(f"[ERROR] UGRID export: {e}")

    # ── Run log viewer ────────────────────────────────────────────────


    # ── Lumped hydrology GeoPackage creation ──────────────────────────
    def create_lumped_hydrology_geopackage(self) -> None:
        """Open a file dialog and create a lumped hydrology GeoPackage."""
        from qgis.PyQt import QtWidgets
        from qgis.core import QgsProject
        from swe2d.services.lumped_hydrology_service import write_lumped_hydrology_geopackage

        view = self._view
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            view, "Create Lumped Hydrology GeoPackage",
            "lumped_hydrology_model.gpkg", "GeoPackage (*.gpkg)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".gpkg"):
            out_path += ".gpkg"

        crs_auth = "EPSG:4326"
        try:
            crs = QgsProject.instance().crs()
            if crs is not None and crs.isValid():
                crs_auth = crs.authid() or crs_auth
        except Exception as _e:

            try:

                view._log(f"[ERROR] Exception in mesh_controller.py: {_e}")

            except Exception:

                pass

        try:
            write_lumped_hydrology_geopackage(out_path, crs_auth=crs_auth)
            view._log("Created lumped hydrology GeoPackage: "
                       "lumped_subbasins, lumped_flow_paths, lumped_rain_events")
            view.set_layer_status_text(
                "Lumped hydrology GeoPackage created."
            )
        except Exception as e:
            view._log(f"[ERROR] Failed to create lumped hydrology gpkg: {e}")

    # ── Mesh-to-layers export ─────────────────────────────────────────


    # ── Mesh-to-layers export ─────────────────────────────────────────
    def export_mesh_to_layers(self) -> None:
        """Export in-memory mesh nodes and cells as QGIS map layers."""
        from qgis.core import QgsProject
        from swe2d.services.mesh_export_service import (
            build_nodes_vector_layer,
            build_cells_polygon_layer,
        )

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        if not mesh:
            view._log("[ERROR] No mesh data to export.")
            return

        try:
            crs = QgsProject.instance().crs()
            crs_auth = crs.authid() if crs is not None and crs.isValid() else "EPSG:4326"
        except Exception:
            crs_auth = "EPSG:4326"

        try:
            nodes_layer = build_nodes_vector_layer(
                mesh["node_x"], mesh["node_y"], mesh["node_z"],
                crs_auth=crs_auth,
            )
            cells_layer = build_cells_polygon_layer(
                mesh["node_x"], mesh["node_y"],
                mesh["cell_nodes"],
                cell_face_offsets=mesh.get("cell_face_offsets"),
                cell_face_nodes=mesh.get("cell_face_nodes"),
                cell_type_meta=mesh.get("cell_type"),
                region_meta=mesh.get("region_id"),
                size_meta=mesh.get("target_size"),
                crs_auth=crs_auth,
            )
            QgsProject.instance().addMapLayer(nodes_layer)
            QgsProject.instance().addMapLayer(cells_layer)
            view._mesh_nodes_layer_id = nodes_layer.id()
            view._mesh_cells_layer_id = cells_layer.id()
            self.refresh_layer_combos()
            view.set_layer_status_text(
                "Mesh exported to SWE2D_Mesh_Nodes and SWE2D_Mesh_Cells layers."
            )
            view._log("Mesh exported to map layers.")
        except Exception as e:
            view._log(f"[ERROR] Mesh export to layers failed: {e}")

    # ── Terrain assignment ────────────────────────────────────────────


    # ── Terrain assignment ────────────────────────────────────────────
    def assign_node_z_from_terrain(self) -> None:
        """Sample terrain raster at mesh node positions and assign node_z."""
        from qgis.PyQt import QtWidgets
        from swe2d.services.terrain_assignment_service import sample_raster_at_nodes

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        if not mesh:
            view._log("[ERROR] No mesh loaded for terrain assignment.")
            return

        terrain_combo = view.get_combo_widget("terrain_layer_combo")
        lyr = view._combo_layer(terrain_combo, "raster")
        if not lyr:
            QtWidgets.QMessageBox.warning(
                view, "Terrain Assignment", "Select a terrain raster layer first."
            )
            return

        try:
            provider = lyr.dataProvider()
            extent = lyr.extent()
            block = provider.block(1, extent, lyr.width(), lyr.height())
            if not block.isValid():
                view._log("[ERROR] Could not read terrain raster block.")
                return

            import numpy as np
            data_type_map = {1: np.uint8, 2: np.uint16, 3: np.int16,
                             4: np.uint32, 5: np.int32, 6: np.float32, 7: np.float64}
            dtype = data_type_map.get(block.dataType(), np.float64)
            raster_data = np.frombuffer(bytes(block.data()), dtype=dtype)
            raster_data = raster_data.reshape(block.height(), block.width())
            geo_transform = (
                extent.xMinimum(), lyr.rasterUnitsPerPixelX(), 0.0,
                extent.yMaximum(), 0.0, -lyr.rasterUnitsPerPixelY(),
            )
            mesh["node_z"] = sample_raster_at_nodes(
                mesh["node_x"], mesh["node_y"], raster_data, geo_transform,
            )
            view._result_data = None
            view._log(f"Assigned node z from terrain raster: {lyr.name()}")
            view._refresh_plot()
        except Exception as e:
            view._log(f"[ERROR] Terrain assignment failed: {e}")

    # ── Node Z pull from vector layer ─────────────────────────────────


    # ── Node Z pull from vector layer ─────────────────────────────────
    def pull_node_z_from_layer(self) -> None:
        """Pull bed_z from nodes vector layer into in-memory node_z."""
        from qgis.PyQt import QtWidgets
        from swe2d.services.terrain_assignment_service import (
            assign_node_z_from_layer_features,
        )

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        if not mesh:
            view._log("[ERROR] No mesh loaded for node_z pull.")
            return

        nodes_combo = view.get_combo_widget("nodes_layer_combo")
        lyr = view._combo_layer(nodes_combo, "vector")
        if not lyr:
            QtWidgets.QMessageBox.warning(
                view, "Pull Node Z", "Select a nodes layer first."
            )
            return

        features = []
        for feat in lyr.getFeatures():
            features.append({
                "node_id": feat["node_id"],
                "bed_z": feat["bed_z"],
            })

        try:
            updated = assign_node_z_from_layer_features(mesh["node_z"], features)
            view._result_data = None
            view._log(
                f"Pulled node_z from {updated} features in layer: {lyr.name()}"
            )
        except Exception as e:
            view._log(f"[ERROR] Pull node z failed: {e}")

    # ── Topology template layer creation ──────────────────────────────


    # ── 2D model GeoPackage loading ───────────────────────────────────
    def load_2d_model_geopackage(self, path_override: str | None = None) -> None:
        """Load a 2D model GeoPackage into the QGIS project.

        If path_override is None, opens a QFileDialog to select the file.
        """
        from qgis.core import QgsProject
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.services.model_gpkg_loader_service import load_layers_from_gpkg

        view = self._view
        gpkg_path = path_override
        if not gpkg_path:
            gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                view, "Load 2D Model GeoPackage", "", "GeoPackage (*.gpkg)",
            )
        if not gpkg_path:
            return

        try:
            layers = load_layers_from_gpkg(gpkg_path)
            if not layers:
                QtWidgets.QMessageBox.warning(
                    view, "Load Model",
                    "No valid model layers found in GeoPackage.",
                )
                return

            import os as _os
            from swe2d.workbench.services.gpkg_layer_styles_service import (
                apply_qml_style_from_gpkg as _apply_style,
            )
            for name, lyr in layers.items():
                QgsProject.instance().addMapLayer(lyr)
                _apply_style(lyr, gpkg_path)

            self._view._layer_controller.refresh_layer_combos()
            view._model_gpkg_path = str(gpkg_path)
            view._log(
                f"Loaded 2D model GeoPackage: {gpkg_path} "
                f"(layers loaded={len(layers)})"
            )
            view.set_layer_status_text(
                f"Loaded 2D model GeoPackage ({len(layers)} layers)."
            )
        except Exception as e:
            view._log(f"[ERROR] Failed to load model GeoPackage: {e}")

    # ── Preview BC / Manning overrides ────────────────────────────────

    def open_run_log_viewer(self) -> None:
        """Open file dialog, select GPKG, pick run, then show the run log viewer."""
        import os as _os
        from qgis.PyQt import QtCore, QtWidgets

        view = self._view
        db_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            view, "Select GeoPackage with run logs", "",
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        db_path = str(db_path or "").strip()
        if not db_path:
            return
        if not _os.path.exists(db_path):
            view._log(f"[ERROR] GeoPackage not found: {db_path}")
            return

        # Load full run-log records from the GPKG (not RunRecord list)
        from swe2d.results.run_log_storage import (
            load_run_logs_from_geopackage as _load_logs,
        )
        try:
            records = _load_logs(gpkg_path=db_path)
        except Exception as exc:
            view._log(f"[ERROR] Failed to load run logs: {exc}")
            return
        if not records:
            QtWidgets.QMessageBox.information(
                view, "Run Log Viewer",
                "No run logs found in the selected GeoPackage.",
            )
            return

        # If multiple runs, let user pick one via a simple selection dialog
        if len(records) > 1:
            run_ids = [str(r.get("run_id", "") or "") for r in records]
            run_id, ok = QtWidgets.QInputDialog.getItem(
                view, "Select Run", "Choose a run to view logs:",
                run_ids, 0, False,
            )
            if not ok or not run_id:
                return
        else:
            run_id = str(records[0].get("run_id", "") or "")
            if not run_id:
                return

        try:
            from swe2d.workbench.dialogs.run_log_viewer_dialog import (
                SWE2DRunLogViewerDialog,
            )
            dlg_viewer = SWE2DRunLogViewerDialog(
                records=records,
                run_id=run_id,
                db_path=db_path,
                parent=view,
            )
            dlg_viewer.exec()
        except ImportError:
            view._log("[ERROR] Run log viewer dialog not available.")
        except Exception:
            view._log("[ERROR] Run log viewer failed to open.")

