"""GUI adapter: converts live widget state into a spec dict and delegates to the
canonical :func:`swe2d.runtime.run_context_builder.build_run_context`.

Phase 1.B also hosts QGIS-backed GPKG loading helpers here so the canonical
builder (:mod:`swe2d.runtime.run_context_builder`) stays QGIS-free.

This module is the **thin flip layer** for Phase 1.B: it translates GUI widget
values to a canonical ``swe2d-run/2`` spec dict, calls the single canonical
builder, and returns the resulting :class:`RunContext`.

GUI-supplied forcing/mesh objects (built from live QGIS layers by the
controller, inlined from the retired ``SWE2DRunDataBuilder`` /
``SWE2DRunOptionsBuilder``) flow through the spec under their canonical
RunContext field names — the canonical builder honors them via its
``_override`` helper.  No ``dataclasses.replace`` post-pass.

No duplicated defaults, normalization, or mesh-loading logic lives here.
"""
from __future__ import annotations

import datetime
import os
import threading
from typing import Any, Dict, Optional

import numpy as np

from swe2d.core.builder import build_run_context


def build_run_context_from_gui(
    widget_state: Dict[str, Any],
    *,
    mesh_data: Dict[str, Any],
    forcing: Optional[Dict[str, Any]] = None,
    run_input: Optional[Dict[str, Any]] = None,
    sample_map_data: list,
    inflow_progressive_enabled: bool,
    edge_groups: Dict[int, str],
    results_gpkg_path: str,
    model_gpkg_path: str,
    mesh_crs_wkt: str,
    parse_time_hours_fn,
) -> "RunContext":
    """Build a RunContext from live GUI widget state.

    This is the **Phase 1.B flip target**: all logic for translating widget
    values into a spec dict, calling the canonical builder, and attaching GUI
    callbacks lives here.  The controller's ``_build_run_context`` method is
    reduced to collecting widget values and calling this function.

    The returned ``RunContext`` is produced by the canonical builder with no
    post-build mutation — every GUI-supplied object flows through the spec
    dict under its canonical RunContext field name and is honored by the
    builder's ``_override`` helper.

    Parameters
    ----------
    widget_state : dict
        Flat dict of widget-name → value, as returned by
        ``view.collect_run_widget_params()``.
    mesh_data : dict
        Live mesh data dict (with node_x, node_y, node_z, cell_nodes, etc.)
        as held by the view.  ``n_mann_cell`` (if present) is forwarded to
        the spec.
    forcing : dict, optional
        Forcing / coupling objects built by the controller from live QGIS
        layers: ``internal_flow_forcing``, ``thiessen_forcing``,
        ``pipe_network_cfg``, ``hydraulic_structures_cfg``,
        ``cell_source_model``, ``rain_rate_model``.
    run_input : dict, optional
        Initial-condition / BC arrays built by the controller from live QGIS
        layers: ``h0``, ``hu0``, ``hv0``, ``bc_tp``, ``bc_vl``, ``bc_relax``,
        ``side_hydrographs``, ``edge_hydrographs``, ``edge_group_overrides``,
        ``n_mann_cell``.  These are computed from live QGIS layers (IC
        widgets, BC line layer, hydrograph tables) and forwarded through the
        spec under their canonical RunContext field names so the builder's
        ``_override`` helper uses them verbatim.
    sample_map_data : list
        Pre-built sample line map (from ``view._build_line_sampling_map()``),
        stored directly on the returned RunContext.
    inflow_progressive_enabled : bool
        ``_capture_inflow_progressive(view)`` result.
    edge_groups : dict
        ``_capture_edge_groups(view, run_input)`` result.
    results_gpkg_path : str
        Path to the results GPKG.
    model_gpkg_path : str
        Path to the model GPKG (used as ``mesh_gpkg``).
    mesh_crs_wkt : str
        CRS WKT string for the mesh.
    parse_time_hours_fn : callable
        ``view._parse_time_hours`` — used to resolve the ``run_duration_s``
        from the widget value stored in ``widget_state``.

    Returns
    -------
    RunContext
    """
    # ── Translate widget_state → canonical spec dict ────────────────────
    spec: Dict[str, Any] = {}

    # Identity
    spec["run_id"] = datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z")
    spec["run_wallclock_start"] = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")
    spec["run_log_start_idx"] = 0  # caller sets this properly via view._runtime_log_lines

    # Mesh (GUI has the data in memory; pass arrays so the canonical builder
    # uses them directly instead of re-reading from GPKG).
    spec["mesh_gpkg"] = str(model_gpkg_path or "")
    spec["mesh_name"] = str(mesh_data.get("mesh_name", "") or "")
    for _arr_key in ("node_x", "node_y", "node_z", "cell_nodes",
                      "cell_face_offsets", "cell_face_nodes"):
        _arr = mesh_data.get(_arr_key)
        if _arr is not None:
            spec[_arr_key] = np.asarray(_arr)
    spec["mesh_crs_wkt"] = mesh_crs_wkt

    # Time — resolve run_duration_s / output_interval_s
    # Honor pre-parsed values (set by the controller when request-level
    # overrides are applied); fall back to raw widget text for other paths.
    _dur_s = widget_state.get("run_duration_s")
    if _dur_s is None:
        _duration_text = str(widget_state.get("run_time_edit", "") or "").strip()
        _dur_s = parse_time_hours_fn(_duration_text) * 3600.0 if _duration_text else 0.0
    spec["run_duration_s"] = _dur_s

    _oi_s = widget_state.get("output_interval_s")
    if _oi_s is None:
        _oi_text = str(widget_state.get("output_interval_edit", "") or "").strip()
        _oi_s = max(1.0, parse_time_hours_fn(_oi_text) * 3600.0) if _oi_text else 1.0
    spec["output_interval_s"] = _oi_s

    # Widget params (using WIDGET_TO_RC mapping keys as-is — the builder
    # normalizes them via WIDGET_TO_RC internally)
    for _key in (
        "dt_spin", "initial_dt_spin", "cfl_spin", "cfl_lambda_cap_spin",
        "adaptive_cfl_dt_chk",
        "n_mann_spin", "h_min_spin",
        "reconstruction_combo", "reconstruction_combo_text",
        "temporal_order_combo", "temporal_order_combo_text",
        "drainage_gpu_method_combo",
        "culvert_solver_mode_combo",
        "bridge_coupling_mode",
        "enable_cuda_graphs_chk", "swe2d_perf_mode_chk",
        "rain_rate_spin",
        "max_rel_depth_increase_spin", "max_source_depth_step_spin",
        "max_source_rate_spin", "source_cfl_beta_spin", "source_max_substeps_spin",
        "shallow_damping_depth_spin", "depth_cap_spin",
        "momentum_cap_min_speed_spin", "momentum_cap_celerity_mult_spin",
        "max_inv_area_spin", "tiny_wet_cell_threshold_spin",
        "front_flux_damping_spin", "open_bc_relax_spin",
        "gpu_diag_sync_interval_spin",
        "active_set_hysteresis_chk", "source_true_subcycling_chk",
        "source_imex_split_chk", "use_redistribution_chk",
        "inflow_progressive_chk",
        "tiny_mode_combo", "degen_mode_combo",
        "gravity", "k_mann",
        "save_mesh_chk", "save_line_chk", "save_coupling_chk",
        "save_log_chk", "save_max_only_chk",
    ):
        if _key in widget_state:
            spec[_key] = widget_state[_key]

    # Special-case booleans that must become strings / enums
    # (the canonical builder reads these as strings, not as widget booleans).
    if "culvert_face_flux_chk" in widget_state:
        spec["culvert_face_flux_mode"] = (
            "face_flux" if widget_state["culvert_face_flux_chk"] else "off"
        )

    # Results
    spec["results_gpkg_path"] = str(results_gpkg_path or "")

    # Units (from view) — these become length_unit_name / length_scale_si_to_model
    # overrides on the RunContext.
    if "_length_unit_name" in widget_state:
        spec["length_unit_name"] = str(widget_state["_length_unit_name"])
    if "_length_scale_si_to_model" in widget_state:
        spec["length_scale_si_to_model"] = float(widget_state["_length_scale_si_to_model"])
    if "_rain_mm_to_model_depth" in widget_state:
        spec["rain_mm_to_model_depth"] = float(widget_state["_rain_mm_to_model_depth"])
    spec["uniform_inflow_enabled"] = bool(widget_state.get("_uniform_inflow_enabled", False))
    spec["rain_update_interval_s"] = float(widget_state.get("_rain_update_interval_s", 60.0))

    # ── GUI-built mesh scalars ─────────────────────────────────────────
    # The canonical builder derives n_mann_cell from the GPKG, but the GUI
    # may have an in-memory array built from a live spatial layer (n_mann
    # vector layer); forward it through the spec so the builder uses it
    # verbatim via _override.
    _n_mann_cell = mesh_data.get("n_mann_cell") if isinstance(mesh_data, dict) else None
    if _n_mann_cell is not None:
        spec["n_mann_cell"] = np.asarray(_n_mann_cell, dtype=np.float64)

    # ── GUI-built initial conditions / BC arrays / hydrographs ──────────
    # The controller computes these from live QGIS layers (the inline
    # equivalent of the retired SWE2DRunDataBuilder.build()):
    # h0/hu0/hv0 from the IC widgets (uniform_depth / uniform_wse / dry with
    # inflow priming), bc_tp/bc_vl/bc_relax from the live BC line layer plus
    # default_bc_type_combo, and the hydrograph dicts from the BC layer's
    # hydrograph tables.  The canonical builder cannot reproduce these from
    # the spec (the GPKG path uses default_bc_for_edges and h0 = zeros), so
    # forward them under their canonical RunContext field names — the
    # builder's _override helper then uses them verbatim.
    if run_input is not None:
        for _arr_key in ("h0", "hu0", "hv0", "bc_vl", "bc_relax"):
            _arr = run_input.get(_arr_key)
            if _arr is not None:
                spec[_arr_key] = np.asarray(_arr, dtype=np.float64)
        _bc_tp = run_input.get("bc_tp")
        if _bc_tp is not None:
            spec["bc_tp"] = np.asarray(_bc_tp, dtype=np.int32)
        _bc_n0 = run_input.get("bc_n0")
        if _bc_n0 is not None:
            spec["bc_edge_node0"] = np.asarray(_bc_n0, dtype=np.int32)
        _bc_n1 = run_input.get("bc_n1")
        if _bc_n1 is not None:
            spec["bc_edge_node1"] = np.asarray(_bc_n1, dtype=np.int32)
        for _dict_key in ("side_hydrographs", "edge_hydrographs", "edge_group_overrides"):
            _hg = run_input.get(_dict_key)
            if _hg is not None:
                spec[_dict_key] = dict(_hg)

    # ── GUI-built forcing / coupling objects ────────────────────────────
    # These were computed by the controller from live QGIS layers (the
    # inline equivalent of the retired SWE2DRunOptionsBuilder.build()).  They
    # flow through the spec under their canonical RunContext field names so
    # the builder's _override helper honors them without a post-build
    # replace().
    if forcing is not None:
        _internal_flow_forcing = forcing.get("internal_flow_forcing")
        if _internal_flow_forcing is not None:
            spec["internal_flow_forcing"] = _internal_flow_forcing
        _thiessen_forcing = forcing.get("thiessen_forcing")
        if _thiessen_forcing is not None:
            spec["thiessen_forcing"] = _thiessen_forcing
        _cell_source_model = forcing.get("cell_source_model")
        if _cell_source_model is not None:
            spec["cell_source_model"] = _cell_source_model
        _rain_rate_model = forcing.get("rain_rate_model", 0.0)
        if _rain_rate_model not in (None, 0.0):
            spec["rain_rate_model"] = _rain_rate_model
        _pipe_network_cfg = forcing.get("pipe_network_cfg")
        if _pipe_network_cfg is not None:
            spec["pipe_network_cfg"] = _pipe_network_cfg
        _hydraulic_structures_cfg = forcing.get("hydraulic_structures_cfg")
        if _hydraulic_structures_cfg is not None:
            spec["hydraulic_structures_cfg"] = _hydraulic_structures_cfg

    # ── Storage flags (raw checkboxes) ─────────────────────────────────
    # The builder normalizes these via WIDGET_TO_RC; mapping here is only
    # a safety net for any widget_state that skipped normalization.
    for _chk_key, _rc_key in (
        ("save_mesh_chk", "save_mesh_results"),
        ("save_line_chk", "save_line_results"),
        ("save_coupling_chk", "save_coupling_results"),
        ("save_log_chk", "save_run_log"),
        ("save_max_only_chk", "save_max_only"),
    ):
        if _chk_key in widget_state:
            spec[_rc_key] = bool(widget_state[_chk_key])

    # ── GUI runtime callbacks (per-run; not derivable from spec) ───────
    # Phase 3.5: the four view-bound callables (_apply_timeseries_bc,
    # _distribute_total_flow, _apply_external_sources, _build_line_sampling_map)
    # are no longer embedded in the spec.  The executor re-binds pure
    # versions internally from RunContext fields (edge_groups,
    # inflow_progressive_enabled, source_rate_cap, etc.).
    _mesh_cell_areas_fn = widget_state.get("_mesh_cell_areas_fn")
    if _mesh_cell_areas_fn is not None:
        spec["mesh_cell_areas"] = _mesh_cell_areas_fn
    _mesh_cell_min_bed_fn = widget_state.get("_mesh_cell_min_bed_fn")
    if _mesh_cell_min_bed_fn is not None:
        spec["mesh_cell_min_bed"] = _mesh_cell_min_bed_fn
    _mesh_cell_centroids_fn = widget_state.get("_mesh_cell_centroids_fn")
    if _mesh_cell_centroids_fn is not None:
        spec["mesh_cell_centroids"] = _mesh_cell_centroids_fn
    _internal_flow_cms_fn = widget_state.get("_internal_flow_cms_fn")
    if _internal_flow_cms_fn is not None:
        spec["internal_flow_source_cms_at_time"] = _internal_flow_cms_fn

    # ── Misc GUI-only fields ───────────────────────────────────────────
    spec["sample_map_data"] = list(sample_map_data or [])
    spec["inflow_progressive_enabled"] = bool(inflow_progressive_enabled)
    spec["edge_groups"] = dict(edge_groups or {})

    # ── Controller-only runtime values ─────────────────────────────────
    # The controller forwards these via ``widget_state`` (underscored keys)
    # because they are *not* widget values — they are derived from
    # request/runtime state in the controller.  Pipe them through the spec
    # so the canonical builder assigns them to the matching RunContext fields.
    if "_run_log_start_idx" in widget_state:
        spec["run_log_start_idx"] = int(widget_state["_run_log_start_idx"])
    if "_bridge_stacked_plans" in widget_state:
        spec["bridge_stacked_plans"] = list(widget_state["_bridge_stacked_plans"] or [])

    # ── Call the canonical builder ─────────────────────────────────────
    # Single entry point: every RunContext field the GUI has already built
    # arrives here via the spec; the builder assembles the final RunContext
    # without any post-build mutation.  This is the "true flip" of Phase 1.B.
    return build_run_context(
        spec,
        mesh_gpkg=model_gpkg_path,
        results_gpkg=results_gpkg_path,
        cancel_event=threading.Event(),
    )


# Note: the QGIS-backed drainage helper ``_build_drainage_config_from_gpkg_layers``
# has moved to ``swe2d.core.gpkg_io`` so the canonical builder can import it
# without depending on a workbench layer.
