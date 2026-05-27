from __future__ import annotations

# Extracted methods depend on symbols defined in swe2d_workbench_qt.
from swe2d_workbench_qt import *  # type: ignore F401,F403

def _refresh_layer_combos(self):
    if not _HAVE_QGIS_CORE:
        self.layer_status_lbl.setText("QGIS layer API unavailable in this runtime")
        return

    self._project_layer_state_blocked = True
    try:
        keep_nodes = self.nodes_layer_combo.currentData()
        keep_cells = self.cells_layer_combo.currentData()
        keep_terrain = self.terrain_layer_combo.currentData()
        keep_manning = self.manning_layer_combo.currentData() if hasattr(self, "manning_layer_combo") else None
        keep_cn = self.cn_layer_combo.currentData() if hasattr(self, "cn_layer_combo") else None
        keep_rain_gages = self.rain_gage_layer_combo.currentData() if hasattr(self, "rain_gage_layer_combo") else None
        keep_hyetograph = self.hyetograph_layer_combo.currentData() if hasattr(self, "hyetograph_layer_combo") else None
        keep_storm_area = self.storm_area_layer_combo.currentData() if hasattr(self, "storm_area_layer_combo") else None
        keep_topo_nodes = self.topo_nodes_combo.currentData() if hasattr(self, "topo_nodes_combo") else None
        keep_topo_arcs = self.topo_arcs_combo.currentData() if hasattr(self, "topo_arcs_combo") else None
        keep_topo_regions = self.topo_regions_combo.currentData() if hasattr(self, "topo_regions_combo") else None
        keep_topo_constraints = self.topo_constraints_combo.currentData() if hasattr(self, "topo_constraints_combo") else None
        keep_topo_quad_edges = self.topo_quad_edges_combo.currentData() if hasattr(self, "topo_quad_edges_combo") else None
        keep_bc_lines = self.bc_lines_layer_combo.currentData() if hasattr(self, "bc_lines_layer_combo") else None
        keep_internal_flow = self.internal_flow_layer_combo.currentData() if hasattr(self, "internal_flow_layer_combo") else None
        keep_sample_lines = self.sample_lines_layer_combo.currentData() if hasattr(self, "sample_lines_layer_combo") else None
        keep_drain_nodes = self.drain_nodes_layer_combo.currentData() if hasattr(self, "drain_nodes_layer_combo") else None
        keep_drain_links = self.drain_links_layer_combo.currentData() if hasattr(self, "drain_links_layer_combo") else None
        keep_drain_inlets = self.drain_inlets_layer_combo.currentData() if hasattr(self, "drain_inlets_layer_combo") else None
        keep_drain_node_inlets = self.drain_node_inlets_layer_combo.currentData() if hasattr(self, "drain_node_inlets_layer_combo") else None
        keep_structures = self.structures_layer_combo.currentData() if hasattr(self, "structures_layer_combo") else None
        keep_3d_obj_instances = self.experimental_3d_obj_layer_combo.currentData() if hasattr(self, "experimental_3d_obj_layer_combo") else None
        keep_3d_obj_inside_points = self.experimental_3d_obj_inside_points_layer_combo.currentData() if hasattr(self, "experimental_3d_obj_inside_points_layer_combo") else None

        self.nodes_layer_combo.clear()
        self.cells_layer_combo.clear()
        self.terrain_layer_combo.clear()
        if hasattr(self, "manning_layer_combo"):
            self.manning_layer_combo.clear()
            self.manning_layer_combo.addItem("(none)", None)
        if hasattr(self, "cn_layer_combo"):
            self.cn_layer_combo.clear()
            self.cn_layer_combo.addItem("(none)", None)
        if hasattr(self, "rain_gage_layer_combo"):
            self.rain_gage_layer_combo.clear()
            self.rain_gage_layer_combo.addItem("(none)", None)
        if hasattr(self, "hyetograph_layer_combo"):
            self.hyetograph_layer_combo.clear()
            self.hyetograph_layer_combo.addItem("(none)", None)
        if hasattr(self, "storm_area_layer_combo"):
            self.storm_area_layer_combo.clear()
            self.storm_area_layer_combo.addItem("(none)", None)
        if hasattr(self, "sample_lines_layer_combo"):
            self.sample_lines_layer_combo.clear()
            self.sample_lines_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_nodes_layer_combo"):
            self.drain_nodes_layer_combo.clear()
            self.drain_nodes_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_links_layer_combo"):
            self.drain_links_layer_combo.clear()
            self.drain_links_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_inlets_layer_combo"):
            self.drain_inlets_layer_combo.clear()
            self.drain_inlets_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_node_inlets_layer_combo"):
            self.drain_node_inlets_layer_combo.clear()
            self.drain_node_inlets_layer_combo.addItem("(none)", None)
        if hasattr(self, "structures_layer_combo"):
            self.structures_layer_combo.clear()
            self.structures_layer_combo.addItem("(none)", None)
        if hasattr(self, "experimental_3d_obj_layer_combo"):
            self.experimental_3d_obj_layer_combo.clear()
            self.experimental_3d_obj_layer_combo.addItem("(none)", None)
        if hasattr(self, "experimental_3d_obj_inside_points_layer_combo"):
            self.experimental_3d_obj_inside_points_layer_combo.clear()
            self.experimental_3d_obj_inside_points_layer_combo.addItem("(none)", None)
        if hasattr(self, "topo_nodes_combo"):
            self.topo_nodes_combo.clear()
        if hasattr(self, "topo_arcs_combo"):
            self.topo_arcs_combo.clear()
        if hasattr(self, "topo_regions_combo"):
            self.topo_regions_combo.clear()
        if hasattr(self, "topo_constraints_combo"):
            self.topo_constraints_combo.clear()
            self.topo_constraints_combo.addItem("(none)", None)
        if hasattr(self, "topo_quad_edges_combo"):
            self.topo_quad_edges_combo.clear()
            self.topo_quad_edges_combo.addItem("(none)", None)
        if hasattr(self, "bc_lines_layer_combo"):
            self.bc_lines_layer_combo.clear()
            self.bc_lines_layer_combo.addItem("(none)", None)
        if hasattr(self, "internal_flow_layer_combo"):
            self.internal_flow_layer_combo.clear()
            self.internal_flow_layer_combo.addItem("(none)", None)

        for lyr in self._iter_project_layers():
            try:
                if isinstance(lyr, QgsVectorLayer):
                    self._configure_swe2d_layer_editors(lyr)
                    if hasattr(self, "internal_flow_layer_combo"):
                        self.internal_flow_layer_combo.addItem(lyr.name(), lyr.id())
                    geom_type = lyr.geometryType()
                    if geom_type == QgsWkbTypes.GeometryType.PointGeometry:
                        self.nodes_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "rain_gage_layer_combo"):
                            self.rain_gage_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_nodes_combo"):
                            self.topo_nodes_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_nodes_layer_combo"):
                            self.drain_nodes_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "experimental_3d_obj_layer_combo"):
                            self.experimental_3d_obj_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "experimental_3d_obj_inside_points_layer_combo"):
                            self.experimental_3d_obj_inside_points_layer_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
                        self.cells_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "manning_layer_combo"):
                            self.manning_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "cn_layer_combo"):
                            self.cn_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "storm_area_layer_combo"):
                            self.storm_area_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_regions_combo"):
                            self.topo_regions_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_constraints_combo"):
                            self.topo_constraints_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type in (
                        QgsWkbTypes.GeometryType.UnknownGeometry,
                        getattr(QgsWkbTypes.GeometryType, "NullGeometry", QgsWkbTypes.GeometryType.UnknownGeometry),
                    ):
                        lname = str(lyr.name() or "").lower()
                        if hasattr(self, "hyetograph_layer_combo") and "hyetograph" in lname:
                            self.hyetograph_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_inlets_layer_combo") and "drainage_inlets" in lname:
                            self.drain_inlets_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_node_inlets_layer_combo") and "drainage_node_inlets" in lname:
                            self.drain_node_inlets_layer_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.LineGeometry:
                        if hasattr(self, "sample_lines_layer_combo"):
                            self.sample_lines_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_arcs_combo"):
                            self.topo_arcs_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_quad_edges_combo"):
                            self.topo_quad_edges_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "bc_lines_layer_combo"):
                            self.bc_lines_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_links_layer_combo"):
                            self.drain_links_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "structures_layer_combo"):
                            self.structures_layer_combo.addItem(lyr.name(), lyr.id())
                elif isinstance(lyr, QgsRasterLayer):
                    self.terrain_layer_combo.addItem(lyr.name(), lyr.id())
            except Exception:
                continue

        hydro_layer_map = {}
        for lyr in self._iter_project_layers():
            if isinstance(lyr, QgsVectorLayer):
                hydro_layer_map[str(lyr.name())] = str(lyr.name())
        for lyr in self._iter_project_layers():
            if isinstance(lyr, QgsVectorLayer) and "bc_lines" in str(lyr.name()).lower():
                self._set_value_map_editor(lyr, "hydrograph_layer", hydro_layer_map)

        def _restore(combo, keep_id):
            if not keep_id:
                return
            idx = combo.findData(keep_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _restore(self.nodes_layer_combo, keep_nodes)
        _restore(self.cells_layer_combo, keep_cells)
        _restore(self.terrain_layer_combo, keep_terrain)
        if hasattr(self, "manning_layer_combo"):
            _restore(self.manning_layer_combo, keep_manning)
        if hasattr(self, "cn_layer_combo"):
            _restore(self.cn_layer_combo, keep_cn)
        if hasattr(self, "rain_gage_layer_combo"):
            _restore(self.rain_gage_layer_combo, keep_rain_gages)
        if hasattr(self, "hyetograph_layer_combo"):
            _restore(self.hyetograph_layer_combo, keep_hyetograph)
        if hasattr(self, "storm_area_layer_combo"):
            _restore(self.storm_area_layer_combo, keep_storm_area)
        if hasattr(self, "topo_nodes_combo"):
            _restore(self.topo_nodes_combo, keep_topo_nodes)
        if hasattr(self, "topo_arcs_combo"):
            _restore(self.topo_arcs_combo, keep_topo_arcs)
        if hasattr(self, "topo_regions_combo"):
            _restore(self.topo_regions_combo, keep_topo_regions)
        if hasattr(self, "topo_constraints_combo") and keep_topo_constraints is not None:
            _restore(self.topo_constraints_combo, keep_topo_constraints)
        if hasattr(self, "topo_quad_edges_combo") and keep_topo_quad_edges is not None:
            _restore(self.topo_quad_edges_combo, keep_topo_quad_edges)
        if hasattr(self, "bc_lines_layer_combo") and keep_bc_lines is not None:
            _restore(self.bc_lines_layer_combo, keep_bc_lines)
        if hasattr(self, "internal_flow_layer_combo") and keep_internal_flow is not None:
            _restore(self.internal_flow_layer_combo, keep_internal_flow)
        if hasattr(self, "sample_lines_layer_combo") and keep_sample_lines is not None:
            _restore(self.sample_lines_layer_combo, keep_sample_lines)
        if hasattr(self, "drain_nodes_layer_combo") and keep_drain_nodes is not None:
            _restore(self.drain_nodes_layer_combo, keep_drain_nodes)
        if hasattr(self, "drain_links_layer_combo") and keep_drain_links is not None:
            _restore(self.drain_links_layer_combo, keep_drain_links)
        if hasattr(self, "drain_inlets_layer_combo") and keep_drain_inlets is not None:
            _restore(self.drain_inlets_layer_combo, keep_drain_inlets)
        if hasattr(self, "drain_node_inlets_layer_combo") and keep_drain_node_inlets is not None:
            _restore(self.drain_node_inlets_layer_combo, keep_drain_node_inlets)
        if hasattr(self, "structures_layer_combo") and keep_structures is not None:
            _restore(self.structures_layer_combo, keep_structures)
        if hasattr(self, "experimental_3d_obj_layer_combo") and keep_3d_obj_instances is not None:
            _restore(self.experimental_3d_obj_layer_combo, keep_3d_obj_instances)
        if hasattr(self, "experimental_3d_obj_inside_points_layer_combo") and keep_3d_obj_inside_points is not None:
            _restore(self.experimental_3d_obj_inside_points_layer_combo, keep_3d_obj_inside_points)

        self._update_unit_system_from_crs()
        self._refresh_layer_group_combo()
        self._update_topology_control_summary()
    finally:
        self._project_layer_state_blocked = False

    self._restore_project_layer_bindings()
    self._persist_project_layer_bindings()





def _refresh_velocity_vectors_overlay(self, t_s: float):
    self._velocity_overlay_refresh_token += 1
    refresh_token = int(self._velocity_overlay_refresh_token)
    frame_t0 = time.perf_counter()
    fetch_ms = 0.0
    build_ms = 0.0
    draw_ms = 0.0
    total_vectors = 0
    total_sources = 0
    panel = getattr(self, "_results_panel", None)
    if panel is None or not panel.velocity_overlay_enabled():
        self._clear_velocity_vectors_layers()
        return
    if not _HAVE_QGIS_CORE:
        self._clear_velocity_vectors_layers()
        return

    if not self._velocity_overlay_sources:
        self._clear_velocity_vectors_layers()
        return

    builder = self._get_velocity_vector_builder()
    if builder is None:
        self._clear_velocity_vectors_layers()
        return

    stride = max(1, int(panel.velocity_density_stride()))
    min_speed = max(0.0, float(panel.velocity_min_speed()))

    for source in list(self._velocity_overlay_sources):
        if refresh_token != self._velocity_overlay_refresh_token:
            return
        total_sources += 1
        gpkg_path = str(source.get("gpkg_path", "")).strip()
        run_id = str(source.get("run_id", "")).strip()
        table_name = str(source.get("table_name", "swe2d_mesh_results")).strip() or "swe2d_mesh_results"
        source_key = str(source.get("key", "")).strip()
        if not gpkg_path or not run_id or not source_key or not os.path.exists(gpkg_path):
            continue

        lyr = self._velocity_vectors_layer_for_source(source)
        if lyr is None:
            continue

        cell_to_fid = self._velocity_overlay_feature_ids.get(source_key)
        if cell_to_fid is None:
            cell_to_fid = {}
            self._velocity_overlay_feature_ids[source_key] = cell_to_fid

        dp = lyr.dataProvider()
        if not cell_to_fid:
            try:
                idx_cell = lyr.fields().indexFromName("cell_id")
                if idx_cell >= 0:
                    for f in lyr.getFeatures():
                        try:
                            cid = int(f["cell_id"])
                            cell_to_fid[cid] = int(f.id())
                        except Exception:
                            continue
            except Exception:
                pass

        _tf0 = time.perf_counter()
        snap = builder.load_snapshot(
            gpkg_path,
            run_id,
            float(t_s),
            t_tol=1.0,
            table_name=table_name,
        )
        fetch_ms += (time.perf_counter() - _tf0) * 1000.0
        if snap is None:
            lyr.triggerRepaint()
            continue

        if not self._velocity_overlay_source_mode_logged.get(source_key, False):
            try:
                support = self._velocity_data_support_for_run(gpkg_path, run_id, table_name)
                if str(getattr(snap, "source", "")) == "face_flux_reconstruction":
                    self._log(
                        "Velocity rendering mode: using face-centered reconstruction "
                        f"(run_id={run_id}, table={table_name}, face_table={support.get('face_table')}, "
                        f"face_rows={int(support.get('face_rows', 0))}, cell_rows={int(support.get('cell_rows', 0))})."
                    )
                else:
                    self._log(
                        "Velocity rendering mode: using cell-centered hu/hv "
                        f"(run_id={run_id}, table={table_name}, no usable face rows detected; "
                        f"cell_rows={int(support.get('cell_rows', 0))})."
                    )
            except Exception:
                pass
            self._velocity_overlay_source_mode_logged[source_key] = True

        cell_xy, base_len = self._mesh_cell_centers_for_gpkg(
            gpkg_path,
            run_id=run_id,
            table_name=table_name,
        )
        if not cell_xy:
            lyr.triggerRepaint()
            continue

        _tb0 = time.perf_counter()
        vecs = builder.build_vectors(
            snapshot=snap,
            cell_xy=cell_xy,
            stride=stride,
            min_depth=1.0e-6,
            min_speed=min_speed,
        )
        build_ms += (time.perf_counter() - _tb0) * 1000.0
        if not vecs:
            existing = list(cell_to_fid.values())
            if existing:
                dp.deleteFeatures(existing)
                self._velocity_overlay_feature_ids[source_key] = {}
            lyr.triggerRepaint()
            continue
        total_vectors += int(len(vecs))

        source_color = self._velocity_source_color(source_key)
        idx_speed = lyr.fields().indexFromName("speed")
        idx_u = lyr.fields().indexFromName("u")
        idx_v = lyr.fields().indexFromName("v")
        idx_ang = lyr.fields().indexFromName("angle_deg")
        idx_src = lyr.fields().indexFromName("source")
        idx_color = lyr.fields().indexFromName("color")
        idx_width = lyr.fields().indexFromName("width")

        new_feats = []
        geom_updates = {}
        attr_updates = {}
        seen_cells = set()
        for v in vecs:
            speed = float(v.get("speed", 0.0))
            if speed <= 1.0e-12:
                continue
            cid = int(v.get("cell_id", -1))
            if cid < 0:
                continue
            seen_cells.add(cid)
            dir_u = float(v.get("u", 0.0)) / speed
            dir_v = float(v.get("v", 0.0)) / speed
            line_len = float(base_len) * min(6.0, max(1.0, 1.25 + 1.15 * speed))

            x0 = float(v.get("x", 0.0))
            y0 = float(v.get("y", 0.0))
            x1 = x0 + dir_u * line_len
            y1 = y0 + dir_v * line_len
            geom = QgsGeometry.fromPolylineXY([
                QgsPointXY(x0, y0),
                QgsPointXY(x1, y1),
            ])

            fid = cell_to_fid.get(cid)
            if fid is not None:
                geom_updates[fid] = geom
                updates = {}
                if idx_speed >= 0:
                    updates[idx_speed] = speed
                if idx_u >= 0:
                    updates[idx_u] = float(v.get("u", 0.0))
                if idx_v >= 0:
                    updates[idx_v] = float(v.get("v", 0.0))
                if idx_ang >= 0:
                    updates[idx_ang] = float(v.get("angle_deg", 0.0))
                if idx_src >= 0:
                    updates[idx_src] = str(source.get("label", ""))
                if idx_color >= 0:
                    updates[idx_color] = source_color
                if idx_width >= 0:
                    updates[idx_width] = 0.8
                if updates:
                    attr_updates[fid] = updates
                continue

            feat = QgsFeature(lyr.fields())
            feat.setAttribute("cell_id", cid)
            feat.setAttribute("speed", speed)
            feat.setAttribute("u", float(v.get("u", 0.0)))
            feat.setAttribute("v", float(v.get("v", 0.0)))
            feat.setAttribute("angle_deg", float(v.get("angle_deg", 0.0)))
            feat.setAttribute("source", str(source.get("label", "")))
            feat.setAttribute("color", source_color)
            feat.setAttribute("width", 0.8)
            feat.setGeometry(geom)
            new_feats.append(feat)

        _td0 = time.perf_counter()
        if geom_updates:
            dp.changeGeometryValues(geom_updates)
        if attr_updates:
            dp.changeAttributeValues(attr_updates)
        if new_feats:
            ok, added = dp.addFeatures(new_feats)
            if ok:
                for f in added:
                    try:
                        cid = int(f["cell_id"])
                        cell_to_fid[cid] = int(f.id())
                    except Exception:
                        continue

        stale_cells = [cid for cid in list(cell_to_fid.keys()) if cid not in seen_cells]
        if stale_cells:
            stale_fids = [cell_to_fid[cid] for cid in stale_cells if cid in cell_to_fid]
            if stale_fids:
                dp.deleteFeatures(stale_fids)
            for cid in stale_cells:
                cell_to_fid.pop(cid, None)

        if new_feats or stale_cells:
            lyr.updateExtents()
        lyr.triggerRepaint()
        draw_ms += (time.perf_counter() - _td0) * 1000.0

    iface = getattr(self, "_iface", None)
    if iface is not None and hasattr(iface, "mapCanvas"):
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass

    self._velocity_overlay_frame_counter += 1
    frame_ms = (time.perf_counter() - frame_t0) * 1000.0
    if (
        self._velocity_overlay_frame_counter % max(1, int(self._velocity_overlay_perf_log_every)) == 0
        or frame_ms > 80.0
    ):
        self._log(
            "Velocity overlay perf: "
            f"frame_ms={frame_ms:.1f}, fetch_ms={fetch_ms:.1f}, build_ms={build_ms:.1f}, draw_ms={draw_ms:.1f}, "
            f"sources={total_sources}, vectors={total_vectors}, stride={stride}"
        )





def _refresh_streamline_traces_overlay(self, t_s: float):
    frame_t0 = time.perf_counter()
    fetch_ms = 0.0
    build_ms = 0.0
    draw_ms = 0.0
    total_traces = 0
    total_sources = 0

    panel = getattr(self, "_results_panel", None)
    if panel is None or not hasattr(panel, "streamline_overlay_enabled"):
        self._clear_streamline_traces_layers()
        return
    if not panel.streamline_overlay_enabled():
        self._clear_streamline_traces_layers()
        return
    if not _HAVE_QGIS_CORE:
        self._clear_streamline_traces_layers()
        return
    if not self._velocity_overlay_sources:
        self._clear_streamline_traces_layers()
        return

    builder = self._get_velocity_vector_builder()
    if builder is None:
        self._clear_streamline_traces_layers()
        return

    seed_count = 48
    max_steps = 30
    step_scale = 0.85
    try:
        seed_count = max(4, int(panel.streamline_seed_count()))
    except Exception:
        pass
    try:
        max_steps = max(4, int(panel.streamline_max_steps()))
    except Exception:
        pass
    try:
        step_scale = max(0.05, float(panel.streamline_step_scale()))
    except Exception:
        pass

    seed_stride = max(1, int(panel.velocity_density_stride()))
    min_speed = max(0.0, float(panel.velocity_min_speed()))

    for source in list(self._velocity_overlay_sources):
        total_sources += 1
        gpkg_path = str(source.get("gpkg_path", "")).strip()
        run_id = str(source.get("run_id", "")).strip()
        table_name = str(source.get("table_name", "swe2d_mesh_results")).strip() or "swe2d_mesh_results"
        source_key = str(source.get("key", "")).strip()
        if not gpkg_path or not run_id or not source_key or not os.path.exists(gpkg_path):
            continue

        lyr = self._streamline_traces_layer_for_source(source)
        if lyr is None:
            continue
        dp = lyr.dataProvider()

        existing_ids = [f.id() for f in lyr.getFeatures()]
        if existing_ids:
            try:
                dp.deleteFeatures(existing_ids)
            except Exception:
                pass

        _tf0 = time.perf_counter()
        snap = builder.load_snapshot(
            gpkg_path,
            run_id,
            float(t_s),
            t_tol=1.0,
            table_name=table_name,
        )
        fetch_ms += (time.perf_counter() - _tf0) * 1000.0
        if snap is None:
            lyr.triggerRepaint()
            continue

        cell_xy, _ = self._mesh_cell_centers_for_gpkg(
            gpkg_path,
            run_id=run_id,
            table_name=table_name,
        )
        if not cell_xy:
            lyr.triggerRepaint()
            continue

        _tb0 = time.perf_counter()
        traces = builder.build_streamline_traces(
            snapshot=snap,
            cell_xy=cell_xy,
            seed_count=seed_count,
            max_steps=max_steps,
            step_len_factor=step_scale,
            min_depth=1.0e-6,
            min_speed=min_speed,
            seed_stride=seed_stride,
        )
        build_ms += (time.perf_counter() - _tb0) * 1000.0
        if not traces:
            lyr.triggerRepaint()
            continue

        source_color = self._velocity_source_color(source_key)
        feats = []
        for tr in traces:
            pts = tr.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            qpts = []
            for xy in pts:
                try:
                    qpts.append(QgsPointXY(float(xy[0]), float(xy[1])))
                except Exception:
                    continue
            if len(qpts) < 2:
                continue

            mean_speed = float(tr.get("mean_speed", 0.0) or 0.0)
            style = builder.style_from_speed(mean_speed)
            feat = QgsFeature(lyr.fields())
            feat.setAttribute("trace_id", int(tr.get("trace_id", len(feats))))
            feat.setAttribute("speed", mean_speed)
            feat.setAttribute("length", float(tr.get("length", 0.0) or 0.0))
            feat.setAttribute("source", str(source.get("label", "")))
            feat.setAttribute("color", source_color)
            feat.setAttribute("width", float(style.get("width", 0.7) or 0.7))
            feat.setGeometry(QgsGeometry.fromPolylineXY(qpts))
            feats.append(feat)

        _td0 = time.perf_counter()
        if feats:
            try:
                dp.addFeatures(feats)
            except Exception:
                pass
            try:
                lyr.updateExtents()
            except Exception:
                pass
            total_traces += int(len(feats))

        lyr.triggerRepaint()
        draw_ms += (time.perf_counter() - _td0) * 1000.0

    iface = getattr(self, "_iface", None)
    if iface is not None and hasattr(iface, "mapCanvas"):
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass

    self._streamline_overlay_frame_counter += 1
    frame_ms = (time.perf_counter() - frame_t0) * 1000.0
    if (
        self._streamline_overlay_frame_counter % max(1, int(self._streamline_overlay_perf_log_every)) == 0
        or frame_ms > 100.0
    ):
        self._log(
            "Streamline overlay perf: "
            f"frame_ms={frame_ms:.1f}, fetch_ms={fetch_ms:.1f}, build_ms={build_ms:.1f}, draw_ms={draw_ms:.1f}, "
            f"sources={total_sources}, traces={total_traces}, seeds={seed_count}, steps={max_steps}"
        )

# ------------------------------------------------------------------





def _bind_map_tab_results_controls(self, map_tab_page: QtWidgets.QWidget, map_results_layout: QtWidgets.QGridLayout) -> None:
    def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
        w = map_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        return w

    def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
        w = map_tab_page.findChild(QtWidgets.QPushButton, name)
        if w is None:
            w = QtWidgets.QPushButton(text)
            w.setObjectName(name)
        return w

    def _find_or_create_combo(name: str) -> QtWidgets.QComboBox:
        w = map_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        return w

    def _find_or_create_double_spin(name: str) -> QtWidgets.QDoubleSpinBox:
        w = map_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        return w

    self.extended_outputs_chk = _find_or_create_check(
        "extended_outputs_chk",
        "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)",
    )
    self.save_mesh_results_to_gpkg_chk = _find_or_create_check(
        "save_mesh_results_to_gpkg_chk",
        "Save mesh snapshot results to GeoPackage",
    )
    self.save_line_results_to_gpkg_chk = _find_or_create_check(
        "save_line_results_to_gpkg_chk",
        "Save sampled line results to GeoPackage",
    )
    self.save_coupling_results_to_gpkg_chk = _find_or_create_check(
        "save_coupling_results_to_gpkg_chk",
        "Save drainage/structure results to GeoPackage",
    )
    self.save_run_log_to_gpkg_chk = _find_or_create_check(
        "save_run_log_to_gpkg_chk",
        "Save run log to GeoPackage",
    )
    self.open_results_viewer_btn = _find_or_create_button("open_results_viewer_btn", "Open 2D Results Viewer")
    self.open_results_panel_btn = _find_or_create_button("open_results_panel_btn", "Results Panel (multi-run)")
    self.high_perf_canvas_overlay_chk = _find_or_create_check(
        "high_perf_canvas_overlay_chk",
        "Show High-Perf Overlay On Map Canvas",
    )
    self.high_perf_canvas_overlay_field_combo = _find_or_create_combo("high_perf_canvas_overlay_field_combo")
    self.high_perf_canvas_overlay_cmap_combo = _find_or_create_combo("high_perf_canvas_overlay_cmap_combo")
    self.high_perf_canvas_overlay_lock_canvas_chk = _find_or_create_check(
        "high_perf_canvas_overlay_lock_canvas_chk",
        "Lock overlay resolution to current canvas size",
    )
    self.high_perf_canvas_overlay_res_combo = _find_or_create_combo("high_perf_canvas_overlay_res_combo")
    self.high_perf_canvas_overlay_auto_contrast_chk = _find_or_create_check(
        "high_perf_canvas_overlay_auto_contrast_chk",
        "Auto contrast",
    )
    self.high_perf_canvas_overlay_opacity_spin = _find_or_create_double_spin("high_perf_canvas_overlay_opacity_spin")
    self.high_perf_canvas_overlay_arrows_chk = _find_or_create_check(
        "high_perf_canvas_overlay_arrows_chk",
        "Draw velocity arrows",
    )
    self.high_perf_canvas_overlay_arrow_density_spin = _find_or_create_double_spin(
        "high_perf_canvas_overlay_arrow_density_spin"
    )
    self.high_perf_canvas_overlay_arrow_length_spin = _find_or_create_double_spin(
        "high_perf_canvas_overlay_arrow_length_spin"
    )
    self.high_perf_canvas_overlay_arrow_head_length_spin = _find_or_create_double_spin(
        "high_perf_canvas_overlay_arrow_head_length_spin"
    )
    self.high_perf_canvas_overlay_arrow_head_width_spin = _find_or_create_double_spin(
        "high_perf_canvas_overlay_arrow_head_width_spin"
    )
    self.high_perf_canvas_overlay_streamlines_chk = _find_or_create_check(
        "high_perf_canvas_overlay_streamlines_chk",
        "Draw streamlines",
    )
    self.high_perf_canvas_overlay_streamline_seed_spin = _find_or_create_double_spin(
        "high_perf_canvas_overlay_streamline_seed_spin"
    )
    self.high_perf_canvas_overlay_streamline_steps_spin = _find_or_create_double_spin(
        "high_perf_canvas_overlay_streamline_steps_spin"
    )
    self.high_perf_canvas_overlay_streamline_backend_combo = _find_or_create_combo(
        "high_perf_canvas_overlay_streamline_backend_combo"
    )

    if map_results_layout.indexOf(self.extended_outputs_chk) < 0:
        map_results_layout.addWidget(self.extended_outputs_chk, 0, 0, 1, 2)
    if map_results_layout.indexOf(self.save_mesh_results_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_mesh_results_to_gpkg_chk, 1, 0, 1, 2)
    if map_results_layout.indexOf(self.save_line_results_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_line_results_to_gpkg_chk, 2, 0, 1, 2)
    if map_results_layout.indexOf(self.save_coupling_results_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_coupling_results_to_gpkg_chk, 3, 0, 1, 2)
    if map_results_layout.indexOf(self.save_run_log_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_run_log_to_gpkg_chk, 4, 0, 1, 2)
    if map_results_layout.indexOf(self.open_results_viewer_btn) < 0:
        map_results_layout.addWidget(self.open_results_viewer_btn, 5, 0, 1, 2)
    if map_results_layout.indexOf(self.open_results_panel_btn) < 0:
        map_results_layout.addWidget(self.open_results_panel_btn, 6, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_chk, 7, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_field_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay field:"), 8, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_field_combo, 8, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_cmap_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay colormap:"), 9, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_cmap_combo, 9, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_lock_canvas_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_lock_canvas_chk, 10, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_res_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay resolution:"), 11, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_res_combo, 11, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_auto_contrast_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_auto_contrast_chk, 12, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_opacity_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay opacity:"), 13, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_opacity_spin, 13, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_arrows_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_arrows_chk, 14, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_arrow_density_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Arrow spacing (px):"), 15, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_arrow_density_spin, 15, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_arrow_length_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Arrow length scale:"), 16, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_arrow_length_spin, 16, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_arrow_head_length_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Arrow head length scale:"), 17, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_arrow_head_length_spin, 17, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_arrow_head_width_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Arrow head width scale:"), 18, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_arrow_head_width_spin, 18, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_streamlines_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_streamlines_chk, 19, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_streamline_backend_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Streamline backend:"), 20, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_streamline_backend_combo, 20, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_streamline_seed_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Streamline seeds:"), 21, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_streamline_seed_spin, 21, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_streamline_steps_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("Streamline steps:"), 22, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_streamline_steps_spin, 22, 1)

    self.extended_outputs_chk.setChecked(True)
    self.save_mesh_results_to_gpkg_chk.setChecked(True)
    self.save_line_results_to_gpkg_chk.setChecked(True)
    self.save_coupling_results_to_gpkg_chk.setChecked(True)
    self.save_run_log_to_gpkg_chk.setChecked(True)
    self.open_results_panel_btn.setToolTip("Open the dockable multi-run results panel")

    self.high_perf_canvas_overlay_chk.setChecked(False)
    self.high_perf_canvas_overlay_field_combo.clear()
    self.high_perf_canvas_overlay_field_combo.addItem("Depth", "depth")
    self.high_perf_canvas_overlay_field_combo.addItem("Velocity", "speed")
    self.high_perf_canvas_overlay_field_combo.addItem("Water Surface", "wse")
    self.high_perf_canvas_overlay_cmap_combo.clear()
    self.high_perf_canvas_overlay_cmap_combo.addItem("Turbo", "turbo")
    self.high_perf_canvas_overlay_cmap_combo.addItem("Viridis", "viridis")
    self.high_perf_canvas_overlay_cmap_combo.addItem("Plasma", "plasma")
    self.high_perf_canvas_overlay_cmap_combo.addItem("Gray", "gray")
    self.high_perf_canvas_overlay_res_combo.clear()
    self.high_perf_canvas_overlay_res_combo.addItem("640 x 360", (640, 360))
    self.high_perf_canvas_overlay_res_combo.addItem("960 x 540", (960, 540))
    self.high_perf_canvas_overlay_res_combo.addItem("1280 x 720", (1280, 720))
    self.high_perf_canvas_overlay_res_combo.addItem("1920 x 1080", (1920, 1080))
    self.high_perf_canvas_overlay_res_combo.setCurrentIndex(2)
    self.high_perf_canvas_overlay_lock_canvas_chk.setChecked(True)
    self.high_perf_canvas_overlay_auto_contrast_chk.setChecked(True)
    self.high_perf_canvas_overlay_opacity_spin.setDecimals(2)
    self.high_perf_canvas_overlay_opacity_spin.setRange(0.05, 1.0)
    self.high_perf_canvas_overlay_opacity_spin.setSingleStep(0.05)
    self.high_perf_canvas_overlay_opacity_spin.setValue(0.65)
    self.high_perf_canvas_overlay_arrows_chk.setChecked(True)
    self.high_perf_canvas_overlay_arrow_density_spin.setDecimals(0)
    self.high_perf_canvas_overlay_arrow_density_spin.setRange(8, 80)
    self.high_perf_canvas_overlay_arrow_density_spin.setSingleStep(2)
    self.high_perf_canvas_overlay_arrow_density_spin.setValue(28)
    self.high_perf_canvas_overlay_arrow_length_spin.setDecimals(2)
    self.high_perf_canvas_overlay_arrow_length_spin.setRange(0.2, 3.0)
    self.high_perf_canvas_overlay_arrow_length_spin.setSingleStep(0.1)
    self.high_perf_canvas_overlay_arrow_length_spin.setValue(1.0)
    self.high_perf_canvas_overlay_arrow_head_length_spin.setDecimals(2)
    self.high_perf_canvas_overlay_arrow_head_length_spin.setRange(0.2, 3.0)
    self.high_perf_canvas_overlay_arrow_head_length_spin.setSingleStep(0.1)
    self.high_perf_canvas_overlay_arrow_head_length_spin.setValue(1.0)
    self.high_perf_canvas_overlay_arrow_head_width_spin.setDecimals(2)
    self.high_perf_canvas_overlay_arrow_head_width_spin.setRange(0.2, 3.0)
    self.high_perf_canvas_overlay_arrow_head_width_spin.setSingleStep(0.1)
    self.high_perf_canvas_overlay_arrow_head_width_spin.setValue(1.0)
    self.high_perf_canvas_overlay_streamlines_chk.setChecked(False)
    self.high_perf_canvas_overlay_streamline_backend_combo.clear()
    self.high_perf_canvas_overlay_streamline_backend_combo.addItem("Auto (prefer compiled)", "auto")
    self.high_perf_canvas_overlay_streamline_backend_combo.addItem("CPU", "cpu")
    self.high_perf_canvas_overlay_streamline_backend_combo.addItem("CUDA", "cuda")
    self.high_perf_canvas_overlay_streamline_backend_combo.setCurrentIndex(0)
    self.high_perf_canvas_overlay_streamline_seed_spin.setDecimals(0)
    self.high_perf_canvas_overlay_streamline_seed_spin.setRange(8, 256)
    self.high_perf_canvas_overlay_streamline_seed_spin.setSingleStep(8)
    self.high_perf_canvas_overlay_streamline_seed_spin.setValue(48)
    self.high_perf_canvas_overlay_streamline_steps_spin.setDecimals(0)
    self.high_perf_canvas_overlay_streamline_steps_spin.setRange(4, 120)
    self.high_perf_canvas_overlay_streamline_steps_spin.setSingleStep(2)
    self.high_perf_canvas_overlay_streamline_steps_spin.setValue(24)

    for sig_obj, cb in (
        (self.open_results_viewer_btn.clicked, self._open_line_results_viewer),
        (self.open_results_panel_btn.clicked, self._show_results_panel),
        (self.high_perf_canvas_overlay_chk.toggled, self._on_high_perf_canvas_overlay_toggled),
        (self.high_perf_canvas_overlay_field_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_cmap_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_lock_canvas_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_res_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_auto_contrast_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_opacity_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_arrows_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_arrow_density_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_arrow_length_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_arrow_head_length_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_arrow_head_width_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_streamlines_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_streamline_backend_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_streamline_seed_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_streamline_steps_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
    ):
        try:
            sig_obj.disconnect(cb)
        except Exception:
            pass
        sig_obj.connect(cb)

    self._on_high_perf_canvas_overlay_style_changed()





def _bind_right_pane_controls(self, right_pane: QtWidgets.QWidget) -> None:
    def _ensure_root_layout() -> QtWidgets.QVBoxLayout:
        layout = right_pane.layout()
        if isinstance(layout, QtWidgets.QVBoxLayout):
            return layout
        layout = QtWidgets.QVBoxLayout(right_pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        return layout

    right_layout = _ensure_root_layout()

    view_row = right_pane.findChild(QtWidgets.QHBoxLayout, "view_row_layout")
    if view_row is None:
        view_row = QtWidgets.QHBoxLayout()
        view_row.setObjectName("view_row_layout")
        right_layout.insertLayout(0, view_row)

    view_mode_lbl = right_pane.findChild(QtWidgets.QLabel, "view_mode_lbl")
    if view_mode_lbl is None:
        view_mode_lbl = QtWidgets.QLabel("View:")
        view_mode_lbl.setObjectName("view_mode_lbl")
    if view_row.indexOf(view_mode_lbl) < 0:
        view_row.addWidget(view_mode_lbl)

    self.view_mode_combo = right_pane.findChild(QtWidgets.QComboBox, "view_mode_combo")
    if self.view_mode_combo is None:
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.setObjectName("view_mode_combo")
    if view_row.indexOf(self.view_mode_combo) < 0:
        view_row.addWidget(self.view_mode_combo)
    if view_row.count() < 3:
        view_row.addStretch(1)

    prev_view_text = self.view_mode_combo.currentText()
    self.view_mode_combo.blockSignals(True)
    try:
        self.view_mode_combo.clear()
        self.view_mode_combo.addItems(["Mesh", "Depth", "Velocity magnitude"])
        idx = self.view_mode_combo.findText(prev_view_text)
        if idx < 0:
            idx = 0
        self.view_mode_combo.setCurrentIndex(idx)
    finally:
        self.view_mode_combo.blockSignals(False)
    try:
        self.view_mode_combo.currentIndexChanged.disconnect(self._refresh_plot)
    except Exception:
        pass
    self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)

    popout_row = right_pane.findChild(QtWidgets.QHBoxLayout, "popout_row_layout")
    if popout_row is None:
        popout_row = QtWidgets.QHBoxLayout()
        popout_row.setObjectName("popout_row_layout")
        right_layout.insertLayout(1, popout_row)

    self.detach_mesh_view_btn = right_pane.findChild(QtWidgets.QPushButton, "detach_mesh_view_btn")
    if self.detach_mesh_view_btn is None:
        self.detach_mesh_view_btn = QtWidgets.QPushButton("Detach Mesh View")
        self.detach_mesh_view_btn.setObjectName("detach_mesh_view_btn")
    if popout_row.indexOf(self.detach_mesh_view_btn) < 0:
        popout_row.addWidget(self.detach_mesh_view_btn)

    self.detach_runtime_log_btn = right_pane.findChild(QtWidgets.QPushButton, "detach_runtime_log_btn")
    if self.detach_runtime_log_btn is None:
        self.detach_runtime_log_btn = QtWidgets.QPushButton("Detach Runtime Log")
        self.detach_runtime_log_btn.setObjectName("detach_runtime_log_btn")
    if popout_row.indexOf(self.detach_runtime_log_btn) < 0:
        popout_row.addWidget(self.detach_runtime_log_btn)
    if popout_row.count() < 3:
        popout_row.addStretch(1)

    for btn, cb in (
        (self.detach_mesh_view_btn, self._open_detached_mesh_view),
        (self.detach_runtime_log_btn, self._open_detached_runtime_log),
    ):
        try:
            btn.clicked.disconnect(cb)
        except Exception:
            pass
        btn.clicked.connect(cb)

    self._right_vertical_split = right_pane.findChild(QtWidgets.QSplitter, "right_vertical_split")
    if self._right_vertical_split is None:
        self._right_vertical_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._right_vertical_split.setObjectName("right_vertical_split")
    self._right_vertical_split.setOrientation(QtCore.Qt.Orientation.Vertical)
    self._right_vertical_split.setChildrenCollapsible(False)
    if right_layout.indexOf(self._right_vertical_split) < 0:
        right_layout.addWidget(self._right_vertical_split, stretch=1)

    right_plot_host = right_pane.findChild(QtWidgets.QWidget, "right_plot_host")
    if right_plot_host is None:
        right_plot_host = QtWidgets.QWidget()
        right_plot_host.setObjectName("right_plot_host")
    if self._right_vertical_split.indexOf(right_plot_host) < 0:
        self._right_vertical_split.insertWidget(0, right_plot_host)

    plot_layout = right_plot_host.layout()
    if not isinstance(plot_layout, QtWidgets.QVBoxLayout):
        plot_layout = QtWidgets.QVBoxLayout(right_plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
    while plot_layout.count():
        item = plot_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    if self._have_mpl:
        self._fig = self._Figure(figsize=(6.4, 4.2), tight_layout=True)
        self._canvas = self._FigureCanvas(self._fig)
        self._canvas.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        try:
            self._canvas.customContextMenuRequested.disconnect()
        except Exception:
            pass
        self._canvas.customContextMenuRequested.connect(
            lambda pos: self._show_panel_detach_menu("mesh", self._canvas.mapToGlobal(pos))
        )
        plot_layout.addWidget(self._canvas)
    else:
        self._fig = None
        self._canvas = None
        no_plot = QtWidgets.QLabel("Matplotlib Qt backend not available; results shown in text log only.")
        no_plot.setWordWrap(True)
        plot_layout.addWidget(no_plot)

    self.log_view = right_pane.findChild(QtWidgets.QPlainTextEdit, "log_view")
    if self.log_view is None:
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setObjectName("log_view")
    if self._right_vertical_split.indexOf(self.log_view) < 0:
        self._right_vertical_split.addWidget(self.log_view)
    self.log_view.setReadOnly(True)
    self.log_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    try:
        self.log_view.customContextMenuRequested.disconnect()
    except Exception:
        pass
    self.log_view.customContextMenuRequested.connect(
        lambda pos: self._show_panel_detach_menu("log", self.log_view.mapToGlobal(pos))
    )
    self._right_vertical_split.setSizes([520, 220])





def _build_line_sampling_map(self) -> List[Dict[str, object]]:
    if self._mesh_data is None or not _HAVE_QGIS_CORE:
        return []
    if not hasattr(self, "sample_lines_layer_combo"):
        return []
    line_layer = self._combo_layer(self.sample_lines_layer_combo, "vector")
    if line_layer is None:
        return []

    fields = set(line_layer.fields().names())
    id_field = "line_id" if "line_id" in fields else None
    name_field = "name" if "name" in fields else None
    enabled_field = "enabled" if "enabled" in fields else None

    cell_polys = self._mesh_cell_polygons()
    if not cell_polys:
        return []
    cell_bboxes = [g.boundingBox() if g is not None and not g.isEmpty() else None for g in cell_polys]

    sample_map: List[Dict[str, object]] = []
    for ft in line_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            if enabled_field is not None and int(ft[enabled_field]) <= 0:
                continue
        except Exception:
            pass

        line_len = float(geom.length())
        if line_len <= 0.0:
            continue
        try:
            p0 = geom.interpolate(0.0).asPoint()
            p1 = geom.interpolate(max(0.0, line_len - 1.0e-9)).asPoint()
            # Canonicalize orientation so line metrics do not depend on
            # digitizing direction (start/end click order).
            start_key = (float(p0.x()), float(p0.y()))
            end_key = (float(p1.x()), float(p1.y()))
            orient_sign = 1.0 if end_key >= start_key else -1.0
            dx = float(p1.x()) - float(p0.x())
            dy = float(p1.y()) - float(p0.y())
            if orient_sign < 0.0:
                dx = -dx
                dy = -dy
            mag = math.hypot(dx, dy)
            if mag <= 0.0:
                continue
            tx = dx / mag
            ty = dy / mag
            nx = ty
            ny = -tx
        except Exception:
            continue

        try:
            line_id = int(ft[id_field]) if id_field is not None else int(ft.id())
        except Exception:
            line_id = int(ft.id())
        line_name = str(ft[name_field]) if name_field is not None and ft[name_field] not in (None, "") else ""

        line_bbox = geom.boundingBox()
        idx: List[int] = []
        lens: List[float] = []
        station_m: List[float] = []
        flow_wx: List[float] = []
        flow_wy: List[float] = []
        overlap_keys_by_row: List[set] = []
        for ci, cell_geom in enumerate(cell_polys):
            bb = cell_bboxes[ci]
            if bb is None or not bb.intersects(line_bbox):
                continue
            try:
                inter = cell_geom.intersection(geom)
            except Exception:
                continue
            if inter is None or inter.isEmpty():
                continue
            seg_len = float(inter.length())
            if seg_len <= 0.0:
                continue

            wx = 0.0
            wy = 0.0
            seg_keys: set = set()
            try:
                parts = inter.asMultiPolyline()
            except Exception:
                parts = []
            if not parts:
                try:
                    poly = inter.asPolyline()
                    if poly:
                        parts = [poly]
                except Exception:
                    parts = []
            for seg in parts:
                if seg is None or len(seg) < 2:
                    continue
                for k in range(1, len(seg)):
                    p0 = seg[k - 1]
                    p1 = seg[k]
                    dx = float(p1.x()) - float(p0.x())
                    dy = float(p1.y()) - float(p0.y())
                    # Orient each intersection segment by along-line station,
                    # not raw geometry vertex order. This preserves a stable
                    # sign while respecting polyline curvature.
                    try:
                        s0 = float(geom.lineLocatePoint(QgsGeometry.fromPointXY(p0)))
                        s1 = float(geom.lineLocatePoint(QgsGeometry.fromPointXY(p1)))
                        if orient_sign < 0.0:
                            s0 = float(line_len) - s0
                            s1 = float(line_len) - s1
                        if s1 < s0:
                            dx = -dx
                            dy = -dy
                    except Exception:
                        if (dx * tx + dy * ty) < 0.0:
                            dx = -dx
                            dy = -dy
                    wx += dy
                    wy += -dx
                    x0 = float(p0.x())
                    y0 = float(p0.y())
                    x1 = float(p1.x())
                    y1 = float(p1.y())
                    if (x1, y1) < (x0, y0):
                        x0, y0, x1, y1 = x1, y1, x0, y0
                    seg_keys.add(
                        (
                            round(x0, 9),
                            round(y0, 9),
                            round(x1, 9),
                            round(y1, 9),
                        )
                    )

            s_loc = float("nan")
            try:
                cgeom = inter.centroid()
                if cgeom is not None and not cgeom.isEmpty():
                    s_loc = float(geom.lineLocatePoint(cgeom))
                    if orient_sign < 0.0:
                        s_loc = float(line_len) - s_loc
            except Exception:
                s_loc = float("nan")
            idx.append(ci)
            lens.append(seg_len)
            station_m.append(s_loc)
            flow_wx.append(wx)
            flow_wy.append(wy)
            overlap_keys_by_row.append(seg_keys)

        if idx and overlap_keys_by_row:
            owner_count = {}
            for key_set in overlap_keys_by_row:
                for key in key_set:
                    owner_count[key] = int(owner_count.get(key, 0)) + 1
            for j, key_set in enumerate(overlap_keys_by_row):
                if not key_set:
                    continue
                denom = max(owner_count.get(k, 1) for k in key_set)
                if denom > 1:
                    scale = 1.0 / float(denom)
                    lens[j] = float(lens[j]) * scale
                    flow_wx[j] = float(flow_wx[j]) * scale
                    flow_wy[j] = float(flow_wy[j]) * scale

        if idx:
            ord_idx = np.argsort(np.nan_to_num(np.asarray(station_m, dtype=np.float64), nan=0.0))
            sample_map.append(
                {
                    "line_id": int(line_id),
                    "line_name": line_name,
                    "normal_x": float(nx),
                    "normal_y": float(ny),
                    "cell_idx": np.asarray(idx, dtype=np.int32)[ord_idx],
                    "weights": np.asarray(lens, dtype=np.float64)[ord_idx],
                    "station_m": np.asarray(station_m, dtype=np.float64)[ord_idx],
                    "flow_wx": np.asarray(flow_wx, dtype=np.float64)[ord_idx],
                    "flow_wy": np.asarray(flow_wy, dtype=np.float64)[ord_idx],
                }
            )

    if sample_map:
        self._log(f"Sample line mapping ready: {len(sample_map)} line(s).")
    return sample_map





