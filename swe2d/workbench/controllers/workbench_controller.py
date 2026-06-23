#!/usr/bin/env python3
"""Thin orchestrator that routes dialog calls to domain controllers.

The dialog calls ``self._controller.<method>()`` for all orchestration.
This class delegates to the appropriate domain controller.
"""

from __future__ import annotations

from typing import Any


class WorkbenchController:
    """Orchestrator: routes calls to domain controllers."""

    def __init__(
        self,
        view: Any,
        run_controller: Any,
        mesh_controller: Any,
        overlay_controller: Any,
        topology_controller: Any,
        layer_controller: Any,
    ):
        self._view = view
        self._run = run_controller
        self._mesh = mesh_controller
        self._overlay = overlay_controller
        self._topo = topology_controller
        self._layer = layer_controller

    # ── Run domain ────────────────────────────────────────────────────

    def on_run(self, request: Any) -> None:
        """Delegate to the run controller."""
        self._run.on_run(request)

    def on_cancel(self) -> None:
        """Delegate to the run controller."""
        self._run.on_cancel()

    def on_preview_overrides(self) -> None:
        """Delegate to the run controller."""
        self._run.on_preview_overrides()

    def on_snapshot(self) -> None:
        """Delegate to the run controller."""
        self._run.on_snapshot()

    def on_load_run_settings_from_results(self) -> None:
        """Delegate to the run controller."""
        self._run.on_load_run_settings_from_results()

    def _preflight_validate_mesh(self) -> dict:
        """Delegate to the run controller."""
        return self._run._preflight_validate_mesh()

    def _collect_bc_for_edges(self, edge_n0, edge_n1) -> dict:
        """Delegate to the run controller."""
        return self._run._collect_bc_for_edges(edge_n0, edge_n1)

    def _prepare_run_inputs(self) -> dict:
        """Delegate to the run controller."""
        return self._run._prepare_run_inputs()

    def _collect_simulation_settings(self) -> dict:
        """Delegate to the run controller."""
        return self._run._collect_simulation_settings()

    # ── Mesh domain ───────────────────────────────────────────────────

    def create_2d_model_geopackage(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.create_2d_model_geopackage()

    def create_lumped_hydrology_geopackage(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.create_lumped_hydrology_geopackage()

    def export_mesh_to_layers(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.export_mesh_to_layers()

    def export_mesh_to_ugrid(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.export_mesh_to_ugrid()

    def export_results_to_hdf5(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.export_results_to_hdf5()

    def export_results_to_ugrid(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.export_results_to_ugrid()

    def assign_node_z_from_terrain(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.assign_node_z_from_terrain()

    def pull_node_z_from_layer(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.pull_node_z_from_layer()

    def open_run_log_viewer(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.open_run_log_viewer()

    def load_2d_model_geopackage(self, path_override: str | None = None) -> None:
        """Delegate to the mesh controller."""
        self._mesh.load_2d_model_geopackage(path_override=path_override)

    def import_mesh_from_layers(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.import_mesh_from_layers()

    def select_results_gpkg(self) -> None:
        """Delegate to the mesh controller."""
        self._mesh.on_select_results_gpkg()

    # ── Overlay domain ────────────────────────────────────────────────

    def on_high_perf_canvas_overlay_toggled(self, checked: bool) -> None:
        """Delegate to the overlay controller."""
        self._overlay.on_high_perf_canvas_overlay_toggled(checked)

    def on_high_perf_canvas_overlay_style_changed(self, *args: Any) -> None:
        """Delegate to the overlay controller."""
        self._overlay.on_high_perf_canvas_overlay_style_changed(*args)

    def export_high_perf_overlay_to_geotiff(self) -> None:
        """Delegate to the overlay controller."""
        self._overlay.export_high_perf_overlay_to_geotiff()

    def load_mesh_snapshot_for_overlay(self, t_s: float) -> bool:
        """Delegate to the overlay controller."""
        return self._overlay.load_mesh_snapshot_for_overlay(t_s)

    # ── Topology domain ───────────────────────────────────────────────

    def _refresh_topology_status(self) -> None:
        """Delegate to the topology controller."""
        self._topo._refresh_topology_status()

    def _populate_gmsh_quality_controls(self) -> None:
        """Delegate to the topology controller."""
        self._topo._populate_gmsh_quality_controls()

    def open_model_gpkg_explorer(self) -> None:
        """Delegate to the topology controller."""
        self._topo.open_model_gpkg_explorer()

    def create_topology_template_layers(self) -> None:
        """Delegate to the topology controller."""
        self._topo.create_topology_template_layers()

    def generate_mesh_from_topology_layers(self) -> None:
        """Delegate to the topology controller."""
        self._topo.generate_mesh_from_topology_layers()

    def on_terminate_topology_mesh(self) -> None:
        """Delegate to the topology controller."""
        self._topo.on_terminate_topology_mesh()

    def start_topology_mesh_async(self, **kwargs: Any) -> None:
        """Delegate to the topology controller."""
        self._topo.start_topology_mesh_async(**kwargs)

    def poll_topology_mesh_future(self, state: dict) -> dict:
        """Delegate to the topology controller."""
        return self._topo.poll_topology_mesh_future(state)

    # ── Layer domain ──────────────────────────────────────────────────

    def refresh_layer_combos(self) -> None:
        """Delegate to the layer controller."""
        self._layer.refresh_layer_combos()

    def autopopulate_layer_combos_from_group(self) -> None:
        """Delegate to the layer controller."""
        self._layer.autopopulate_layer_combos_from_group()
