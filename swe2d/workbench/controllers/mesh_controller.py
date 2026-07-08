"""Mesh import, terrain assignment, GeoPackage creation, and exports.

MVP domain controller.  Currently a shell — methods are progressively
extracted from ``WorkbenchController`` as they are rewritten to use the
``MeshView`` protocol instead of direct widget access.
"""

from __future__ import annotations

import os

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
        view = self._view
        default_path = str(view._current_line_results_storage_path() or "")
        if not os.path.exists(default_path):
            default_path = str(view._model_gpkg_path or "")
        gpkg_path = view.get_open_file_name(
            "Select Existing Results GeoPackage",
            default_path,
            "GeoPackage (*.gpkg)",
        )
        if not gpkg_path:
            return
        if not os.path.exists(gpkg_path):
            view.show_warning_message(
                "Results GeoPackage",
                "Please select an existing GeoPackage file.",
            )
            return
        view.set_results_gpkg_path(gpkg_path)
        view._log(f"Results GeoPackage override set: {gpkg_path}")

    # ── Mesh import orchestration ──────────────────────────────────────


    # ── Mesh import orchestration ──────────────────────────────────────
    def import_mesh_from_layers(self) -> None:
        """Show a dialog to select nodes and cells layers, then import mesh."""
        from qgis.PyQt import QtWidgets
        from qgis.core import QgsProject, QgsVectorLayer

        view = self._view

        try:
            vector_layers = [
                lyr for lyr in QgsProject.instance().mapLayers().values()
                if isinstance(lyr, QgsVectorLayer)
            ]
        except Exception as exc:
            view._log(f"[ERROR] Could not read QGIS project layers: {exc}")
            return

        if not vector_layers:
            view.show_warning_message(
                "No Layers",
                "No vector layers found in the QGIS project."
            )
            return

        dlg = QtWidgets.QDialog(view)
        dlg.setWindowTitle("Load Mesh From Layers")
        dlg.setMinimumWidth(400)
        layout = QtWidgets.QFormLayout(dlg)

        nodes_combo = QtWidgets.QComboBox()
        cells_combo = QtWidgets.QComboBox()
        for combo in (nodes_combo, cells_combo):
            combo.addItem("(none)", None)
            for lyr in vector_layers:
                combo.addItem(lyr.name(), lyr.id())

        layout.addRow("Nodes layer:", nodes_combo)
        layout.addRow("Cells layer:", cells_combo)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        nodes_lid = nodes_combo.currentData()
        cells_lid = cells_combo.currentData()
        if not nodes_lid or not cells_lid:
            view._log("Select both a nodes layer and a cells layer.")
            return

        nodes_layer = None
        cells_layer = None
        for lyr in vector_layers:
            if lyr.id() == nodes_lid:
                nodes_layer = lyr
            if lyr.id() == cells_lid:
                cells_layer = lyr

        if nodes_layer is None or cells_layer is None:
            view._log("Could not resolve selected layers.")
            return

        from swe2d.services.mesh_extraction_service import (
            extract_mesh_from_layer_data,
        )
        extracted = extract_mesh_from_layer_data(
            nodes_layer=nodes_layer,
            cells_layer=cells_layer,
            log_fn=view._log,
        )
        if not extracted:
            view._log("No valid node or cell features found in selected layers.")
            return

        view._mesh_data = extracted

        try:
            self._auto_assign_node_z_from_elevation_source(extracted)
        except Exception as exc:
            view._log(f"[WARNING] Auto-assign node_z failed: {exc}")

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
        view._log(
            f"Imported mesh from map layers: nodes={node_x.size}, faces={n_faces}, triangles={n_tris}"
        )
        view._result_data = None
        try:
            view.show_mesh_tab()
        except RuntimeError:
            pass
        try:
            view._refresh_plot()
        except RuntimeError:
            pass

    def create_2d_model_geopackage(self) -> None:
        """Create a new 2D model GeoPackage.

        The full implementation (formerly in ``_create_2d_model_geopackage``
        in ``extracted/topology_and_io_methods.py``) is inlined here.
        Reads all widget/state from ``self._view``.
        """
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

        out_path = view.get_save_file_name(
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
            _os.path.dirname(
                _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
            ),
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

        out_path = view.get_save_file_name(
            "Export Mesh to UGRID NetCDF", "", "NetCDF (*.nc);;All (*.*)"
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
            view.show_critical_message("UGRID Export Error", str(e))
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
            view.show_warning_message(
                "Export HDF5", "No mesh or simulation results available."
            )
            return

        out_path = view.get_save_file_name(
            "Export Results to HEC-RAS HDF5", "", "HDF5 (*.h5);;All (*.*)"
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
            view.show_critical_message("HDF5 Export Error", str(e))
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

        if mesh is None or not timesteps:
            view.show_warning_message(
                "Export UGRID", "No mesh or simulation results available."
            )
            return

        out_path = view.get_save_file_name(
            "Export Results to UGRID NetCDF",
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
            view.show_critical_message("UGRID Export Error", str(e))
            view._log(f"[ERROR] UGRID export: {e}")

    # ── Run log viewer ────────────────────────────────────────────────


    # ── Lumped hydrology GeoPackage creation ──────────────────────────
    def create_lumped_hydrology_geopackage(self) -> None:
        """Open a file dialog and create a lumped hydrology GeoPackage."""
        from qgis.core import QgsProject
        from swe2d.services.lumped_hydrology_service import write_lumped_hydrology_geopackage

        view = self._view
        out_path = view.get_save_file_name(
            "Create Lumped Hydrology GeoPackage",
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
        """Sample terrain raster at mesh node positions and assign node_z.

        Shows a dialog with a raster-layer picker instead of relying on a
        pre-populated combo on the Layers page.
        """
        from qgis.PyQt import QtWidgets
        from qgis.core import QgsProject, QGisRasterLayer
        from swe2d.services.terrain_assignment_service import sample_raster_at_nodes

        view = self._view
        mesh = getattr(view, "_mesh_data", None)
        if not mesh:
            view._log("[ERROR] No mesh loaded for terrain assignment.")
            return

        try:
            raster_layers = [
                lyr for lyr in QgsProject.instance().mapLayers().values()
                if isinstance(lyr, (QgsRasterLayer,))
            ]
        except Exception as exc:
            view._log(f"[ERROR] Could not read QGIS project layers: {exc}")
            return

        if not raster_layers:
            view.show_warning_message(
                "No Raster Layers",
                "No raster layers found in the QGIS project."
            )
            return

        dlg = QtWidgets.QDialog(view)
        dlg.setWindowTitle("Assign Node Z From Terrain")
        dlg.setMinimumWidth(400)
        layout = QtWidgets.QFormLayout(dlg)

        raster_combo = QtWidgets.QComboBox()
        raster_combo.addItem("(none)", None)
        for lyr in raster_layers:
            raster_combo.addItem(lyr.name(), lyr.id())

        layout.addRow("Terrain raster:", raster_combo)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addRow(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        raster_lid = raster_combo.currentData()
        if not raster_lid:
            view._log("Select a terrain raster layer.")
            return

        raster_layer = None
        for lyr in raster_layers:
            if lyr.id() == raster_lid:
                raster_layer = lyr
                break

        if raster_layer is None:
            view._log("Could not resolve selected raster layer.")
            return

        try:
            provider = raster_layer.dataProvider()
            extent = raster_layer.extent()
            block = provider.block(1, extent, raster_layer.width(), raster_layer.height())
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
                extent.xMinimum(), raster_layer.rasterUnitsPerPixelX(), 0.0,
                extent.yMaximum(), 0.0, -raster_layer.rasterUnitsPerPixelY(),
            )
            mesh["node_z"] = sample_raster_at_nodes(
                mesh["node_x"], mesh["node_y"], raster_data, geo_transform,
            )
            view._result_data = None
            view._log(f"Assigned node z from terrain raster: {raster_layer.name()}")
            view._refresh_plot()
        except Exception as e:
            view._log(f"[ERROR] Terrain assignment failed: {e}")

    # ── Node Z pull from vector layer ─────────────────────────────────


    # ── Node Z pull from vector layer ─────────────────────────────────
    def pull_node_z_from_layer(self) -> None:
        """Pull bed_z from nodes vector layer into in-memory node_z."""
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
            view.show_warning_message(
                "Pull Node Z", "Select a nodes layer first."
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
        from swe2d.workbench.services.model_gpkg_loader_service import load_layers_from_gpkg

        view = self._view
        gpkg_path = path_override
        if not gpkg_path:
            gpkg_path = view.get_open_file_name(
                "Load 2D Model GeoPackage", "", "GeoPackage (*.gpkg)",
            )
        if not gpkg_path:
            return

        try:
            layers = load_layers_from_gpkg(gpkg_path)
            if not layers:
                view.show_warning_message(
                    "Load Model",
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


