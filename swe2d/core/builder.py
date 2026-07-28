"""Shared RunContext builder — bridges CLI JSON (and future GUI dict) to RunContext.

The CLI path follows the same architecture as the GUI: load layers from
GPKG via QgsVectorLayer (or raw sqlite3 for non-spatial data), then call
the identical service functions the GUI uses.  The only difference is that
the CLI supplies data from file paths / JSON values instead of QComboBoxes.
"""

from __future__ import annotations

import datetime
import difflib
import logging
import os
import sqlite3
import threading
from contextlib import closing
from typing import Any, Callable, Dict, List, Literal, Optional, Set

import numpy as np

from swe2d.core.run_context import RunContext
from swe2d.core.gpkg_io import query_mesh_from_gpkg

logger = logging.getLogger(__name__)


class BuildRunContextError(ValueError):
    """Raised when ``build_run_context`` rejects a spec at build time.

    Subclass of :class:`ValueError` so legacy callers that catch the
    broad type continue to work; named so test assertions and CLI
    callers can branch on the structured failure (e.g. "drainage spec
    rejected by GPKG loader") instead of substring-matching a generic
    ``ValueError``.
    """


def _build_error(spec_key: str, action: str, exc: Exception) -> BuildRunContextError:
    """Create an actionable typed error for a configured spec block."""
    return BuildRunContextError(
        f"spec key {spec_key!r} {action}: {exc}"
    )


def _require_gpkg_table(spec_key: str, gpkg_path: str, table_name: str) -> None:
    """Fail fast when a configured GeoPackage layer/table does not exist."""
    if not gpkg_path or not os.path.isfile(gpkg_path):
        raise BuildRunContextError(
            f"spec key {spec_key!r} references missing GeoPackage {gpkg_path!r}"
        )
    try:
        with closing(sqlite3.connect(gpkg_path)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name = ? COLLATE NOCASE",
                (table_name,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise _build_error(
            spec_key, f"could not inspect GeoPackage {gpkg_path!r}", exc
        ) from exc
    if exists is None:
        raise BuildRunContextError(
            f"spec key {spec_key!r} references missing table/layer "
            f"{table_name!r} in {gpkg_path!r}"
        )


def _unset_run_context_callback(*_args: Any, **_kwargs: Any) -> None:
    """Fail if an obsolete, unbound RunContext callback is invoked."""
    raise RuntimeError("RunContext callback was not configured")


# Identity sentinel checked by ``core.executor._validate_run_context`` when a
# callback is part of the executor's required callback set.
_UNSET_RUN_CONTEXT_CALLBACK = _unset_run_context_callback


def _norm_key(a, b):
    return (a, b) if a < b else (b, a)


# ── GUI-widget → RunContext parameter name mapping ─────────────────────────────
# Module-level so it can be shared by build_run_context_from_dict and
# widget_state_to_flat_params (for extracting flat params from versioned
# widget_state format saved by collect_workbench_widget_state).
WIDGET_TO_RC: Dict[str, str] = {
    # Spin boxes
    "n_mann_spin": "n_mann",
    "cfl_spin": "cfl",
    "h_min_spin": "h_min",
    "dt_spin": "dt_cfg",
    "initial_dt_spin": "initial_dt",
    "cfl_lambda_cap_spin": "cfl_lambda_cap",
    "gpu_diag_sync_interval_spin": "gpu_diag_sync_interval_steps",
    "max_rel_depth_increase_spin": "max_rel_depth_increase",
    "max_source_depth_step_spin": "source_depth_step_cap",
    "max_source_rate_spin": "source_rate_cap",
    "source_cfl_beta_spin": "source_cfl_beta",
    "source_max_substeps_spin": "source_max_substeps",
    "shallow_damping_depth_spin": "shallow_damping_depth",
    "depth_cap_spin": "depth_cap",
    "momentum_cap_min_speed_spin": "momentum_cap_min_speed",
    "momentum_cap_celerity_mult_spin": "momentum_cap_celerity_mult",
    "max_inv_area_spin": "max_inv_area",
    "tiny_wet_cell_threshold_spin": "tiny_wet_cell_threshold",
    "front_flux_damping_spin": "front_flux_damping",
    "open_bc_relax_spin": "open_bc_relaxation",
    # Initial condition (GUI → RunContext name,  h0 computed from these)
    "initial_wse_spin": "initial_wse",
    "initial_depth_spin": "initial_depth",
    "initial_condition_combo": "initial_condition_mode",
    "gpu_diag_sync_interval_raw": "gpu_diag_sync_interval_steps",
    # Checkboxes  (gui name → rc name)
    "adaptive_cfl_dt_chk": "adaptive_cfl_dt",
    "source_true_subcycling_chk": "source_true_subcycling",
    "source_imex_split_chk": "source_imex_split",
    "use_redistribution_chk": "use_redistribution",
    "swe2d_perf_mode_chk": "swe2d_perf_mode",
    "enable_cuda_graphs_chk": "cuda_graphs_enabled",
    # culvert_face_flux_chk: retired builder consumed the boolean and
    # derived culvert_face_flux_mode.  Map to itself so _normalize_spec
    # accepts the widget name (no-op mapping); the canonical builder
    # still reads culvert_face_flux_mode (a string), not this boolean.
    "culvert_face_flux_chk": "culvert_face_flux_chk",
    "active_set_hysteresis_chk": "active_set_hysteresis",
    "inflow_progressive_chk": "inflow_progressive",
    # Storage / results checkboxes
    "save_max_only_chk": "save_max_only",
    "save_mesh_results_to_gpkg_chk": "save_mesh_results",
    "save_line_results_to_gpkg_chk": "save_line_results",
    "save_coupling_results_to_gpkg_chk": "save_coupling_results",
    "save_run_log_to_gpkg_chk": "save_run_log",
    # Raw storage checkbox widget names (on _model_tab_view)
    "save_mesh_chk": "save_mesh_results",
    "save_line_chk": "save_line_results",
    "save_coupling_chk": "save_coupling_results",
    "save_log_chk": "save_run_log",
    # Combos
    "reconstruction_combo": "reconstruction_mode",
    "temporal_order_combo": "temporal_scheme",
    "tiny_mode_combo": "tiny_mode",
    "degen_mode_combo": "degen_mode",
    "drainage_gpu_method_combo": "drainage_gpu_method_mode",
    "culvert_solver_mode_combo": "culvert_solver_mode",
    "bridge_stacked_coupling_mode_combo": "bridge_stacked_coupling_mode",
    # bridge_coupling_mode: widget name emitted by collect_run_widget_params() in model_tab_view.
    # Must be in WIDGET_TO_RC so _normalize_spec translates it before validation.
    "bridge_coupling_mode": "bridge_stacked_coupling_mode",
    # culvert_face_flux_chk: the old SWE2DRunOptionsBuilder consumed this
    # boolean and derived culvert_face_flux_mode = "face_flux" from it.
    # That builder is retired; the canonical builder only reads
    # culvert_face_flux_mode (a string).  We intentionally do NOT map
    # culvert_face_flux_chk → culvert_face_flux_enabled here because no
    # such field exists on RunContext.  The widget name stays in the spec
    # as a harmless no-op (WIDGET_TO_RC.keys() is in _VALID_SPEC_KEYS).
    "default_bc_type_combo": "default_bc_type",
    # LineEdits (stored as strings, parsed by _v or callers)
    "run_time_edit": "run_duration_s",
    "output_interval_edit": "output_interval_s",
    # Note: results_gpkg_path_edit and results_table_name_edit are intentionally
    # NOT mapped here. The CLI replay JSON uses the original widget names in
    # params (matching the reference schema), and build_run_context_from_dict
    # resolves the results path via both keys (see results GPKG resolution).
    # Alternate keys (CLI JSON may use these)
    "duration_s": "run_duration_s",
    "dt_max": "dt_cfg",
}


# ── Combo display-text → param name mapping ──────────────────────────────────────
# collect_workbench_widget_state captures currentText() for combos as
# "{combo_name}_text" entries.  This map translates them to the
# RunContext param names that to_replay_json / from_replay_json expect.
_COMBO_TEXT_TO_PARAM: Dict[str, str] = {
    "reconstruction_combo_text": "reconstruction_name",
    "temporal_order_combo_text": "temporal_scheme_name",
    "tiny_mode_combo_text": "tiny_mode_name",
    "degen_mode_combo_text": "degen_mode_name",
    "drainage_gpu_method_combo_text": "drainage_gpu_method_name",
    "culvert_solver_mode_combo_text": "culvert_solver_mode_name",
    "bridge_stacked_coupling_mode_combo_text": "bridge_stacked_coupling_mode_name",
}

_GUI_METADATA_ALIASES: Dict[str, str] = {
    "culvert_face_flux_enabled": "culvert_face_flux_mode",
}


# ── Single defaults table ────────────────────────────────────────────────────────
# Canonical defaults shared by ALL RunContext constructors:
# build_run_context(), build_run_context_from_dict(), RunContext.from_replay_json(),
# and RunContext.from_widget_params().  ONE source of truth — eliminates the
# 0.2-vs-0.05 dt_cfg split between build_run_context_from_dict and from_replay_json.
#
# Array, callback, and container fields are NOT listed here — they are always
# computed or empty by the builder; specifying them in DEFAULTS would hide the
# fact that they must be derived.
_DEFAULTS: Dict[str, Any] = {
    # Identity / bookkeeping
    "run_id": "",
    "run_wallclock_start": "",
    "run_log_start_idx": 0,
    "results_gpkg_path": "",
    "model_gpkg_path": "",
    "mesh_name": "",
    "mesh_crs_wkt": "",
    # Time
    "run_duration_s": 0.0,
    "output_interval_s": 1.0,
    "dt_cfg": 0.05,
    # dt_request / dt_fixed are *derived* from adaptive_cfl_dt + dt_cfg at
    # assembly time (see build_run_context); no code-level default belongs
    # here because they are never set by a widget or a CLI JSON value
    # directly — only the (adaptive_cfl_dt, dt_cfg) pair is.  Hardcoding a
    # value here previously masked a refactor regression where the
    # adaptive-vs-fixed derivation was lost (dt_request=0.05 capped every
    # run at 0.05 s regardless of dt_cfg).  See commit 70561f9a.
    "initial_dt": 0.0,
    "adaptive_cfl_dt": False,
    # Solver modes
    "reconstruction_mode": 0,
    "reconstruction_name": "",
    "temporal_scheme": None,
    "temporal_scheme_name": "",
    "solver_backend_mode": "gpu",
    "coupling_loop_mode": "cuda",
    "drainage_solver_backend_mode": "gpu",
    "drainage_gpu_method_mode": "step",
    "culvert_solver_mode": 0,
    "cuda_graphs_enabled": False,
    "swe2d_perf_mode": False,
    "bridge_cuda_coupling": False,
    "bridge_stacked_coupling_mode": "phase3_spatial",
    "culvert_face_flux_mode": "off",
    # Numerics
    "gravity": 9.81,
    "k_mann": 1.0,
    "n_mann": 0.035,
    "cfl": 0.45,
    "h_min": 1e-4,
    "max_inv_area": 0.0,
    "cfl_lambda_cap": 0.0,
    "momentum_cap_min_speed": 0.0,
    "momentum_cap_celerity_mult": 0.0,
    "depth_cap": 0.0,
    "max_rel_depth_increase": 0.0,
    "shallow_damping_depth": 0.0,
    "source_cfl_beta": 0.0,
    "source_max_substeps": 1,
    "source_rate_cap": 0.0,
    "source_depth_step_cap": 0.0,
    "source_true_subcycling": False,
    "source_imex_split": False,
    "gpu_diag_sync_interval_steps": 0,
    "tiny_mode": 0,
    "tiny_wet_cell_threshold": 0,
    "degen_mode": 0,
    "front_flux_damping": 0.0,
    "open_bc_relaxation": 0.0,
    "active_set_hysteresis": False,
    "use_redistribution": False,
    "inflow_progressive": False,
    "uniform_inflow_enabled": False,
    "rain_update_interval_s": 60.0,
    # Storage flags
    "save_mesh_results": True,
    "save_line_results": False,
    "save_coupling_results": False,
    "save_run_log": False,
    "save_max_only": False,
    # Units (SI defaults; builder overrides via `_u` when mesh CRS is loaded)
    "length_unit_name": "m",
    "length_scale_si_to_model": 1.0,
    "rain_mm_to_model_depth": 1e-3,
    "rain_rate_si_to_model": 1.0,
    "flow_si_to_model": 1.0,
    # Misc (scalar only; arrays/callbacks are always derived)
    "inflow_progressive_enabled": False,
}

# RunContext scalar param keys that accept validation.  Array fields (node_x,
# cell_nodes, …), callback fields (mesh_cell_areas, …), and container fields
# (side_hydrographs, edge_hydrographs, bridge_stacked_plans, …) are excluded —
# they are always computed or derived by the builder; validating them would
# require knowing mesh shape before loading, which defeats the purpose.
_VALID_PARAM_KEYS: Set[str] = set(_DEFAULTS.keys())

# Keys that may appear in a canonical swe2d-run/2 spec at top level
# (includes mesh/data-source top-level keys and legacy aliases).
_VALID_SPEC_KEYS: Set[str] = _VALID_PARAM_KEYS | {
    "schema_version",
    "run_id", "id",
    "run_wallclock_start", "run_log_start_idx",
    "mesh", "mesh_gpkg", "mesh_name",
    "params", "results", "units", "data_sources", "_data_sources",
    "results_gpkg", "results_gpkg_path",
    "h0",
    # dt_request / dt_fixed are normally derived from (adaptive_cfl_dt,
    # dt_cfg) at build time, but a CLI / replay spec may pass them at
    # the top level to lock the timestep.  See the derivation block in
    # build_run_context.
    "dt_request", "dt_fixed",
    # Legacy aliases accepted during normalization
    "duration_s", "dt_max",
    "initial_water_surface_elevation",
    "results_gpkg_path_edit",
    "rain_rate_spin",
    "default_bc_type",
    # Data-source keys that appear as top-level entries after normalization
    "bc_lines", "drainage", "hyetograph", "rain_cn",
    "sample_lines", "structures", "infiltration_method",
    "storm_areas", "internal_flow_sources",
    "cancel_event",
    # GUI-override keys: pre-built forcing/mesh objects passed by the
    # GUI adapter through the spec (see ``_override`` in build_run_context).
    # These are full Python objects (numpy arrays, dataclasses, callables)
    # and never serialized — they live only in transient spec dicts.
    "pipe_network_cfg", "hydraulic_structures_cfg",
    "internal_flow_forcing", "cell_source_model", "rain_rate_model",
    "thiessen_forcing", "bridge_stacked_plans", "coupling_soa",
    "n_mann_cell", "cell_areas", "cell_centroids",
    "bc_n0", "bc_n1", "bc_tp", "bc_vl", "bc_relax",
    "bc_edge_node0", "bc_edge_node1",
    "node_x", "node_y", "node_z", "cell_nodes",
    "cell_face_offsets", "cell_face_nodes",
    "mesh_crs_wkt",
    "side_hydrographs", "edge_hydrographs", "edge_group_overrides",
    "hu0", "hv0",
    "length_unit_name", "length_scale_si_to_model",
    "rain_mm_to_model_depth", "rain_rate_si_to_model", "flow_si_to_model",
    "mesh_cell_areas", "mesh_cell_min_bed", "mesh_cell_centroids",
    "apply_timeseries_bc_values", "distribute_total_flow_to_unit_q",
    "apply_external_sources", "build_line_sampling_map",
    "internal_flow_source_cms_at_time",
    "sample_map_data", "inflow_progressive_enabled", "edge_groups",
    # Widget-name keys — accepted during normalization, flagged as suggestions
    *WIDGET_TO_RC.keys(),
    # Combo text keys
    *_COMBO_TEXT_TO_PARAM.keys(),
}

_VALID_DATA_SOURCE_KEYS: Set[str] = {
    "bc_lines",
    "drainage",
    "hyetograph",
    "rain_cn",
    "sample_lines",
    "structures",
    "infiltration_method",
    "storm_areas",
    "internal_flow_sources",
}
_VALID_NESTED_KEYS: Dict[str, Set[str]] = {
    "params": _VALID_PARAM_KEYS
    | set(WIDGET_TO_RC)
    | {
        *_COMBO_TEXT_TO_PARAM,
        *_COMBO_TEXT_TO_PARAM.values(),
        "culvert_face_flux_enabled",
        "culvert_face_flux_chk",
        "duration_s",
        "dt_max",
        # dt_request / dt_fixed are derived from (adaptive_cfl_dt, dt_cfg)
        # at build time, but a CLI / replay spec may still pass them
        # explicitly to lock the timestep.  Accept them here so the
        # validator does not reject legitimate replays.
        "dt_request",
        "dt_fixed",
        "initial_wse",
        "initial_depth",
        "initial_condition_mode",
        "initial_water_surface_elevation",
        "h0",
        "results_gpkg_path_edit",
        "results_table_name_edit",
        "rain_rate_spin",
        "default_bc_type",
        "coupling_substeps",
        "head_deadband",
        "dynamic_relaxation",
        "implicit_iters",
        "implicit_relax",
        "friction_method",
        "surcharge_method",
        "recon_method",
        "time_integrator",
        "friction_alpha",
    },
    "mesh": {"mesh_name", "gpkg_path", "crs_wkt"},
    "results": {
        "results_gpkg_path",
        "save_mesh_results",
        "save_line_results",
        "save_coupling_results",
        "save_run_log",
        "save_max_only",
    },
    "units": {
        "length_unit_name",
        "length_scale_si_to_model",
        "rain_mm_to_model_depth",
        "rain_rate_si_to_model",
        "flow_si_to_model",
    },
    "data_sources": _VALID_DATA_SOURCE_KEYS,
    "rain_cn": {"gpkg", "table", "cn_field", "ia_ratio"},
    "hyetograph": {"gpkg", "table", "gauge_layer"},
    "drainage": {
        "gpkg",
        "nodes_layer",
        "links_layer",
        "inlets_layer",
        "node_inlets_layer",
        "nodes",
        "links",
        "inlets",
        "inlet_types",
        "node_inlets",
        "outfalls",
        "gravity",
        "head_deadband_m",
        "dynamic_flow_relaxation",
        "pipe_solver_mode",
        "solver_mode",
        "coupling_substeps",
        "gpu_method",
        "head_deadband",
        "dynamic_relaxation",
        "implicit_iters",
        "implicit_relax",
        "friction_method",
        "surcharge_method",
        "recon_method",
        "time_integrator",
        "friction_alpha",
    },
    "structures": {
        "gpkg",
        "table",
        "enabled",
        "control_interval_s",
        "controller_name",
        "structures",
    },
    "sample_lines": {"gpkg", "table"},
    "bc_lines": {"gpkg", "table", "hydrograph_table"},
    "internal_flow_sources": {"gpkg", "table", "field", "hydrograph_table"},
    "storm_areas": {"gpkg", "table"},
}


def widget_state_to_flat_params(
    widget_state: dict,
    *,
    mesh_gpkg: str = "",
    mesh_name: str = "",
) -> dict:
    """Extract flat RunContext-params from versioned widget_state.

    ``collect_workbench_widget_state`` returns
    ``{"version": 1, "widgets": {"n_mann_spin": {"type": "...", "value": 0.035}, ...}}``.
    This function converts it to flat ``{rc_param_name: value}`` dict using
    ``WIDGET_TO_RC`` (for widget values) and ``_COMBO_TEXT_TO_PARAM`` (for
    combo display-text entries), so it can be stored in the ``params`` block
    of the ``swe2d-replay/1`` schema that ``build_run_context_from_dict`` reads.

    If ``mesh_gpkg`` and ``mesh_name`` are provided, the ``units`` block is
    also computed from the mesh CRS and returned as a side-effect in
    ``flat["_units_block"]``.
    """
    widgets = widget_state.get("widgets", {}) if isinstance(widget_state, dict) else {}
    flat: Dict[str, Any] = {}

    def _parse_duration(val: Any) -> Any:
        """Parse a time string (HH:MM or fraction-of-an-hour) to float seconds.

        If the value is not a recognisable time string it is returned unchanged,
        allowing callers to apply their own interpretation.
        """
        if not isinstance(val, str):
            return val
        s = val.strip()
        if ":" in s:
            try:
                parts = s.split(":")
                return (float(parts[0]) + float(parts[1]) / 60.0) * 3600.0
            except (ValueError, IndexError):
                return val
        return val

    for wname, winfo in widgets.items():
        if not isinstance(winfo, dict):
            continue
        value = winfo.get("value")
        if value is None:
            continue
        # Skip combo display-text entries — they're handled separately by
        # _COMBO_TEXT_TO_PARAM below (or ignored if there's no mapping).
        if wname.endswith("_text"):
            continue
        # Map GUI widget name → RunContext param name
        rc_name = WIDGET_TO_RC.get(wname, wname)
        # Time-edit widgets store HH:MM strings — convert to seconds
        flat[rc_name] = _parse_duration(value)

    # Handle combo display-text entries (e.g. reconstruction_combo_text → reconstruction_name)
    for text_key, param_name in _COMBO_TEXT_TO_PARAM.items():
        winfo = widgets.get(text_key)
        if isinstance(winfo, dict):
            text_val = winfo.get("value")
            if text_val is not None:
                flat[param_name] = text_val

    # Compute units from mesh CRS if mesh_gpkg is available
    if mesh_gpkg and mesh_name:
        from swe2d import units as _u2
        try:
            md = query_mesh_from_gpkg(mesh_gpkg, mesh_name)
            if md is not None:
                crs_wkt = str(md.get("crs_wkt", "") or "")
                si_m_per_model = _u2.si_m_per_model_from_wkt(crs_wkt) if crs_wkt else 1.0
                model_per_si_m = 1.0 / si_m_per_model if si_m_per_model else 1.0
                flat["_units_block"] = {
                    "length_unit_name": "ft" if si_m_per_model < 0.5 else "m",
                    "length_scale_si_to_model": si_m_per_model,
                    "rain_mm_to_model_depth": 1e-3 * model_per_si_m,
                    "rain_rate_si_to_model": model_per_si_m,
                    "flow_si_to_model": model_per_si_m ** 3,
                }
        except Exception as exc:
            raise _build_error(
                "units", "could not derive unit conversions from mesh", exc
            ) from exc

    return flat


# ── Inverse WIDGET_TO_RC: for RunContext → widget_state serialization ─────────
# Built from WIDGET_TO_RC by inverting the mapping.  Only entries where the
# widget name differs from the RunContext field name are kept — same-name
# entries (e.g. "bridge_coupling_mode" → "bridge_stacked_coupling_mode") are
# handled explicitly below.  Phase 3.7 uses this map in
# ``widget_state_to_widget_state_dict`` to walk a RunContext's scalar fields
# back to their corresponding widget names.
_RC_TO_WIDGET: Dict[str, str] = {}
for _wname, _rcname in WIDGET_TO_RC.items():
    if _wname != _rcname and _rcname not in _RC_TO_WIDGET:
        _RC_TO_WIDGET[_rcname] = _wname

# Storage checkboxes use a different naming convention than the WIDGET_TO_RC
# entries (which have ``save_*_to_gpkg_chk`` → ``save_*``).  The GUI's
# ``collect_widget_state_for_save`` reads the raw ``save_mesh_chk`` etc. from
# ``_model_tab_view``; the inverse serializer must emit those raw names so
# the round-trip works through the GUI's save path.
_STORAGE_CHK_WIDGETS: Dict[str, str] = {
    "save_mesh_results": "save_mesh_chk",
    "save_line_results": "save_line_chk",
    "save_coupling_results": "save_coupling_chk",
    "save_run_log": "save_log_chk",
    "save_max_only": "save_max_only_chk",
}

# Combo display-text keys that the GUI collects alongside the combo's
# currentData — the canonical inverse (widget_state_to_flat_params) reads
# these via _COMBO_TEXT_TO_PARAM; the forward serializer emits both forms.
_COMBO_TEXT_WIDGETS: Dict[str, str] = {v: k for k, v in _COMBO_TEXT_TO_PARAM.items()}


def _format_duration_hhmm(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM`` for ``QLineEdit`` widgets.

    The GUI's ``run_time_edit`` and ``output_interval_edit`` widgets store
    HH:MM strings; ``widget_state_to_flat_params._parse_duration`` parses
    them back.  Default ``1.0`` for ``output_interval_s`` is emitted as
    ``"0:01"`` (one minute) so the round-trip is exact.
    """
    if seconds <= 0:
        return "0:00"
    total_minutes = int(round(seconds / 60.0))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}:{minutes:02d}"


def widget_state_to_widget_state_dict(ctx) -> Dict[str, Any]:
    """Canonical serializer: ``RunContext`` → versioned widget_state dict.

    Produces the same format as
    :func:`swe2d.workbench.bridges.project_settings_bridge.collect_workbench_widget_state`:

    .. code-block:: python

        {"version": 1, "widgets": {wname: {"type": "...", "value": ...}}}

    The inverse is :func:`widget_state_to_flat_params` — together they form
    a round-trip pair:

      ctx  →  widget_state_dict  →  flat_params  →  build_run_context → ctx'

    Both sides agree on every scalar RunContext field.  Array fields (mesh,
    h0, hydrographs, callbacks) are not part of the widget_state — they live
    only in transient spec dicts (see ``_override`` in ``build_run_context``).
    """
    widgets: Dict[str, Dict[str, Any]] = {}

    # ── Time-edit widgets (HH:MM) ────────────────────────────────────
    widgets["run_time_edit"] = {
        "type": "QLineEdit",
        "value": _format_duration_hhmm(float(getattr(ctx, "run_duration_s", 0.0))),
    }
    widgets["output_interval_edit"] = {
        "type": "QLineEdit",
        "value": _format_duration_hhmm(float(getattr(ctx, "output_interval_s", 1.0))),
    }

    # ── Spin boxes (floats) ─────────────────────────────────────────
    # Walk the WIDGET_TO_RC inverse — every rc_name that maps to a spin
    # widget is emitted as a QDoubleSpinBox.  String fields (e.g.
    # ``drainage_gpu_method_mode``) and ``None`` values are skipped — those
    # live in combos and the inverse path handles them via the WIDGET_TO_RC
    # text-key round-trip.
    _TIME_EDIT_WIDGETS = {"run_time_edit", "output_interval_edit"}
    for _rcname, _wname in _RC_TO_WIDGET.items():
        # Skip non-spin widgets handled below (combos, checkboxes, edits).
        if _wname.endswith("_text") or _wname.endswith("_chk"):
            continue
        # Time-edit widgets are emitted above as QLineEdit HH:MM strings;
        # the spin loop would overwrite them with raw float seconds.
        if _wname in _TIME_EDIT_WIDGETS:
            continue
        val = getattr(ctx, _rcname, None)
        if val is None:
            continue
        if isinstance(val, str):
            # Combo text-mode value — handled via _COMBO_TEXT_WIDGETS below
            # (e.g. drainage_gpu_method_mode → drainage_gpu_method_combo).
            continue
        if isinstance(val, bool):
            # Booleans are emitted as checkboxes elsewhere; skip here to
            # avoid re-emitting under the spin slot.
            continue
        try:
            widgets[_wname] = {"type": "QDoubleSpinBox", "value": float(val)}
        except (TypeError, ValueError):
            # Non-numeric value (e.g. enum string) — leave to combo path.
            continue

    # ── Checkboxes (booleans) ───────────────────────────────────────
    for _rcname, _wname in _RC_TO_WIDGET.items():
        if not _wname.endswith("_chk"):
            continue
        # Skip storage_*_to_gpkg_chk aliases (mapped via _STORAGE_CHK_WIDGETS)
        if _wname.endswith("_to_gpkg_chk"):
            continue
        val = getattr(ctx, _rcname, False)
        widgets[_wname] = {"type": "QCheckBox", "value": bool(val)}

    # Storage checkboxes — raw widget names used by ``collect_widget_state_for_save``.
    for _rcname, _wname in _STORAGE_CHK_WIDGETS.items():
        widgets[_wname] = {
            "type": "QCheckBox",
            "value": bool(getattr(ctx, _rcname, False)),
        }

    # ── Combos ──────────────────────────────────────────────────────
    # reconstruction_combo stores the int (currentData); _text stores the label.
    if hasattr(ctx, "reconstruction_mode"):
        widgets["reconstruction_combo"] = {
            "type": "QComboBox",
            "value": int(getattr(ctx, "reconstruction_mode", 0)),
        }
    if getattr(ctx, "reconstruction_name", ""):
        widgets["reconstruction_combo_text"] = {
            "type": "QComboBox_text",
            "value": str(ctx.reconstruction_name),
        }
    if hasattr(ctx, "temporal_scheme"):
        widgets["temporal_order_combo"] = {
            "type": "QComboBox",
            "value": getattr(ctx, "temporal_scheme", None),
        }
    if getattr(ctx, "temporal_scheme_name", ""):
        widgets["temporal_order_combo_text"] = {
            "type": "QComboBox_text",
            "value": str(ctx.temporal_scheme_name),
        }

    return {"version": 1, "widgets": widgets}


# ── Normalization front-stage ───────────────────────────────────────────────────
# GUI-compat input normalization: NOT a second API.  All new code should emit
# canonical swe2d-run/2 specs directly.  This stage exists only to keep
# existing GUI-saved configs, replay JSON, and flat CLI JSON working without
# changes to their serialization format.
#
# The stage:
# 1. Accepts widget-name keys and combo-text keys (normalizes to spec keys).
# 2. Accepts legacy string-mesh form "mesh_name" → {mesh_name: ...}.
# 3. Accepts _data_sources / data_sources nested block (flattens to top-level).
# 4. Validates every nested block against the swe2d-run/2 schema.
# 5. Merges params sub-dict into top-level so callers can use either form.
# 6. Raises on unknown top-level keys with "did you mean" suggestions.
#    Widget-name keys produce a two-part suggestion (try the widget name AND
#    its spec equivalent) so the caller knows both what's wrong and what to use.


def _validate_nested_mapping(
    block_name: str,
    value: Any,
    allowed_keys: Set[str],
) -> None:
    """Validate one mapping-shaped nested spec block."""
    if not isinstance(value, dict):
        raise TypeError(
            f"spec block {block_name!r} must be an object, "
            f"got {type(value).__name__}"
        )

    unknown_keys = sorted(set(value) - allowed_keys, key=str)
    if not unknown_keys:
        return

    unknown = unknown_keys[0]
    suggestions = (
        difflib.get_close_matches(unknown, allowed_keys, n=3, cutoff=0.6)
        if isinstance(unknown, str)
        else []
    )
    hint = (
        f" (did you mean: {', '.join(repr(item) for item in suggestions)})"
        if suggestions
        else ""
    )
    raise ValueError(
        f"Unknown nested spec key {unknown!r} in block {block_name!r}{hint}. "
        "See docs/RUN_SPEC_SCHEMA.md."
    )


def _validate_nested_blocks(spec: Dict[str, Any]) -> None:
    """Validate every object/scalar block defined by swe2d-run/2."""
    for block_name, allowed_keys in _VALID_NESTED_KEYS.items():
        if block_name in spec:
            _validate_nested_mapping(block_name, spec[block_name], allowed_keys)

    if "_data_sources" in spec:
        _validate_nested_mapping(
            "_data_sources", spec["_data_sources"], _VALID_DATA_SOURCE_KEYS
        )

    if "infiltration_method" in spec:
        infiltration_method = spec["infiltration_method"]
        if isinstance(infiltration_method, dict):
            _validate_nested_mapping(
                "infiltration_method", infiltration_method, set()
            )
        if not isinstance(infiltration_method, str):
            raise TypeError(
                "spec block 'infiltration_method' must be a string, "
                f"got {type(infiltration_method).__name__}"
            )


def _normalize_spec(
    raw: Dict[str, Any],
    *,
    _mode: str = "cli",
) -> Dict[str, Any]:
    """Normalize a raw spec dict to canonical swe2d-run/2 form.

    This is the **GUI-compat input normalization front-stage**, not a second
    API.  It accepts legacy widget names, string meshes, and flat CLI JSON,
    then validates and re-emits a canonical dict.  New callers should emit
    canonical specs directly and skip this stage.

    Raises
    ------
    ValueError
        Unknown top-level key with "did you mean" suggestion from
        difflib.get_close_matches (widget-name keys suggest both the widget
        name AND its spec equivalent).
    TypeError
        A present scalar key has the wrong Python type for RunContext.
    """
    spec = dict(raw)  # shallow copy; we'll mutate it

    # ── 1. Normalize mesh string → dict ───────────────────────────────────
    mesh_val = spec.get("mesh")
    if isinstance(mesh_val, str):
        # String form: "mesh_name" — caller must still supply mesh_gpkg.
        spec["mesh"] = {"mesh_name": mesh_val}
    elif isinstance(mesh_val, dict):
        pass  # already canonical
    elif mesh_val is not None:
        raise TypeError(
            f"'mesh' must be a string or dict, got {type(mesh_val).__name__}: "
            f"{mesh_val!r}"
        )

    # ── 2. Normalize widget-name keys → spec keys ─────────────────────────
    # Apply WIDGET_TO_RC: widget names in top-level and inside params block.
    params_block = spec.get("params") or {}
    if isinstance(params_block, dict):
        for _gui_key, _rc_key in WIDGET_TO_RC.items():
            if _gui_key in params_block and _rc_key not in spec:
                spec[_rc_key] = params_block[_gui_key]

    for _gui_key, _rc_key in WIDGET_TO_RC.items():
        if _gui_key in spec and _rc_key not in spec:
            spec[_rc_key] = spec[_gui_key]

    # Normalize combo display-text keys → spec keys (top-level and inside params).
    for _text_key, _param_name in _COMBO_TEXT_TO_PARAM.items():
        if _text_key in spec and _param_name not in spec:
            spec[_param_name] = spec[_text_key]
        params_block_inner = spec.get("params")
        if (
            isinstance(params_block_inner, dict)
            and _text_key in params_block_inner
            and _param_name not in params_block_inner
        ):
            params_block_inner[_param_name] = params_block_inner[_text_key]

    # Translate GUI metadata aliases (e.g. ``culvert_face_flux_enabled`` → ``culvert_face_flux_mode``).
    for _alias_key, _param_name in _GUI_METADATA_ALIASES.items():
        if _alias_key in spec and _param_name not in spec:
            _value = spec[_alias_key]
            if isinstance(_value, bool) and _alias_key.endswith("_enabled"):
                spec[_param_name] = "face_flux" if _value else "off"
            else:
                spec[_param_name] = _value
        if (
            _alias_key.endswith("_enabled")
            and _alias_key in spec
            and _alias_key != _param_name
        ):
            spec.pop(_alias_key, None)
        params_block = spec.get("params")
        if isinstance(params_block, dict):
            if _alias_key in params_block and _param_name not in params_block:
                _value = params_block[_alias_key]
                if isinstance(_value, bool) and _alias_key.endswith("_enabled"):
                    params_block[_param_name] = "face_flux" if _value else "off"
                else:
                    params_block[_param_name] = _value
            if (
                _alias_key.endswith("_enabled")
                and _alias_key in params_block
                and _alias_key != _param_name
            ):
                params_block.pop(_alias_key, None)
            spec["params"] = params_block

    # ── 3. Flatten data_sources / _data_sources ──────────────────────────
    data_sources = spec.get("_data_sources") or spec.get("data_sources") or {}
    if isinstance(data_sources, dict):
        for _ds_key in (
            "bc_lines", "drainage", "hyetograph", "rain_cn",
            "sample_lines", "structures", "infiltration_method",
            "storm_areas", "internal_flow_sources",
        ):
            if _ds_key in data_sources and _ds_key not in spec:
                spec[_ds_key] = data_sources[_ds_key]

    # ── 4. Validate nested swe2d-run/2 blocks ─────────────────────────────
    _validate_nested_blocks(spec)

    # ── 5. Merge params into top-level for uniform access ─────────────────
    # After widget-name normalization, params values that have no top-level
    # override are accessible via the same _v() lookup that checks p first.
    # No mutation needed here — build_run_context_from_dict already handles
    # the top-level → params lookup via _v().

    # ── 6. Fail-fast validation: unknown top-level keys ───────────────────
    # Only validate keys that look like they might be a misspelled param name.
    # Known non-param / GUI-internal keys are explicitly skipped so they don't
    # trigger spurious rejections from replay payloads (widget_state, version, …).
    _param_like_keys = _VALID_PARAM_KEYS | set(WIDGET_TO_RC.values())
    _internal_skip_keys = {
        "widget_state",  # versioned GUI widget-state dict (from collect_widget_state_for_save)
        "version",        # widget_state version marker
        "widgets",        # nested widget dict inside widget_state
        "params",         # nested params block — already consumed above
        "mesh",           # already normalized to dict above
        "units",         # nested units block — consumed by builder
        "data_sources", "_data_sources",  # consumed above
        "results",        # consumed by builder
        "schema_version", # schema version marker
        "id",             # legacy alias for run_id (consumed above)
        "gpkg_path",      # may appear inside mesh/results dicts, not top-level param
        "cancel_event",   # threading event — not a spec field, passed separately
    }
    for _key in spec:
        if _key in _VALID_SPEC_KEYS or _key in _internal_skip_keys:
            continue
        if _key.startswith("_"):
            continue  # private/internal keys are never validated
        # Looks like a param but isn't known — generate suggestions.
        _suggestions = difflib.get_close_matches(_key, _param_like_keys, n=3, cutoff=0.6)
        if _suggestions:
            # If the key looks like a widget name, also suggest the spec equivalent.
            _widget_eq = WIDGET_TO_RC.get(_key)
            if _widget_eq and _widget_eq not in _suggestions:
                _suggestions = _suggestions + [_widget_eq]
            _hint = f" (did you mean: {', '.join(repr(s) for s in _suggestions)})"
        else:
            _hint = ""
        raise ValueError(
            f"Unknown spec key {type(spec[_key]).__name__} "
            f"'{_key}'{_hint}. "
            f"Use canonical RunContext field names from swe2d-run/2 schema "
            f"(see docs/RUN_SPEC_SCHEMA.md)."
        )

    return spec


# ── Drainage config dict (C-2) ─────────────────────────────────────────────────

def _drainage_config_dict(drainage_cfg: Dict[str, Any], _v) -> Dict[str, Any]:
    """Build the config dict passed to the GPKG-backed pipe-network loader.

    Includes the 5 GUI-parity keys that were missing from the CLI's
    pre-Phase-3 dict (friction_method, surcharge_method, recon_method,
    time_integrator, friction_alpha).  These are drainage-only — they live
    under the spec's ``drainage`` block first and fall back to top-level
    for legacy callers.  Defaults match
    ``pipe_network_service.build_pipe_network_config`` and the
    ``PipeNetworkConfig`` dataclass in ``swe2d/extensions/extension_models.py``.
    """
    cfg = drainage_cfg if isinstance(drainage_cfg, dict) else {}
    return {
        "solver_mode": int(_v("culvert_solver_mode", 0)),
        "coupling_substeps": int(_v("coupling_substeps", 1)),
        "gpu_method": str(_v("drainage_gpu_method_mode", "step")),
        "head_deadband": float(_v("head_deadband", 0.001)),
        "dynamic_relaxation": float(_v("dynamic_relaxation", 0.7)),
        "implicit_iters": int(_v("implicit_iters", 3)),
        "implicit_relax": float(_v("implicit_relax", 0.8)),
        "friction_method": int(
            cfg.get("friction_method", _v("friction_method", 0))
        ),
        "surcharge_method": int(
            cfg.get("surcharge_method", _v("surcharge_method", 0))
        ),
        "recon_method": int(
            cfg.get("recon_method", _v("recon_method", 0))
        ),
        "time_integrator": int(
            cfg.get("time_integrator", _v("time_integrator", 1))
        ),
        "friction_alpha": float(
            cfg.get("friction_alpha", _v("friction_alpha", 0.01))
        ),
    }


# ── Canonical builder ──────────────────────────────────────────────────────────

def build_run_context(
    spec: Dict[str, Any],
    *,
    mesh_gpkg: str = "",
    results_gpkg: str = "",
    cancel_event: Optional[threading.Event] = None,
) -> RunContext:
    """Build a fully-populated RunContext from a canonical ``swe2d-run/2`` spec dict.

    This is the **single canonical builder**: all RunContext construction routes
    through here.  It accepts canonical specs, legacy replay JSON, flat CLI JSON,
    and widget-state dicts — all normalized internally via the GUI-compat
    front-stage before any real work begins.

    Parameters
    ----------
    spec : dict
        A ``swe2d-run/2`` spec dict with optionally nested ``params``, ``mesh``,
        ``results``, ``units``, ``data_sources`` sub-dicts.  Top-level keys
        override nested ones.  Legacy formats (widget names, string meshes,
        flat CLI JSON) are accepted and normalized automatically.
    mesh_gpkg : str
        Path to the model GPKG.  May also be in ``spec["mesh_gpkg"]`` or
        ``spec["mesh"]["gpkg_path"]``.
    results_gpkg : str
        Path to the results GPKG.  May also be in ``spec["results_gpkg"]`` or
        ``spec["results"]["results_gpkg_path"]``.
    cancel_event : threading.Event or None
        Cancel signal propagated into the RunContext.

    Returns
    -------
    RunContext
    """
    # ── Normalization front-stage (GUI-compat input normalization) ─────────
    p = _normalize_spec(spec)
    # Override with explicit mesh/results GPKG args (higher priority than spec).
    if mesh_gpkg:
        p["mesh_gpkg"] = mesh_gpkg
    if results_gpkg:
        p["results_gpkg_path"] = results_gpkg
    if cancel_event is not None:
        p["cancel_event"] = cancel_event

    # ── Resolve mesh GPKG and name ─────────────────────────────────────
    mesh_dict = p.get("mesh", {})
    if isinstance(mesh_dict, dict):
        _mesh_gpkg = mesh_gpkg or str(p.get("mesh_gpkg", "") or mesh_dict.get("gpkg_path", ""))
        mesh_name = str(mesh_dict.get("mesh_name", "") or p.get("mesh_name", ""))
    else:
        # String form (already normalized to dict above, but guard anyway).
        _mesh_gpkg = mesh_gpkg or str(p.get("mesh_gpkg", ""))
        mesh_name = str(mesh_dict) if mesh_dict else ""

    # ── In-memory mesh path (GUI adapter forwards arrays via spec) ────
    _node_x_arr = p.get("node_x")
    if _node_x_arr is not None:
        md = {
            "node_x": np.asarray(_node_x_arr, dtype=np.float64),
            "node_y": np.asarray(p.get("node_y", np.empty(0)), dtype=np.float64),
            "node_z": np.asarray(p.get("node_z", np.empty(0)), dtype=np.float64),
            "cell_nodes": np.asarray(p.get("cell_nodes",
                np.empty(0, dtype=np.int32)), dtype=np.int32),
            "cell_face_offsets": np.asarray(p.get("cell_face_offsets",
                np.empty(0, dtype=np.int32)), dtype=np.int32),
            "cell_face_nodes": np.asarray(p.get("cell_face_nodes",
                np.empty(0, dtype=np.int32)), dtype=np.int32),
            "bc_edge_node0": np.asarray(p.get("bc_edge_node0",
                np.empty(0, dtype=np.int32)), dtype=np.int32),
            "bc_edge_node1": np.asarray(p.get("bc_edge_node1",
                np.empty(0, dtype=np.int32)), dtype=np.int32),
            "crs_wkt": str(p.get("mesh_crs_wkt", "") or p.get("crs_wkt", "")),
        }
        mesh_name = str(mesh_name or p.get("mesh_name", ""))
        _mesh_gpkg = _mesh_gpkg or ""
    else:
        if not _mesh_gpkg or not os.path.isfile(_mesh_gpkg):
            raise FileNotFoundError(f"Mesh GPKG not found: {_mesh_gpkg}")
        if not mesh_name:
            raise ValueError("mesh_name required in params")
        md = query_mesh_from_gpkg(_mesh_gpkg, mesh_name)
        if md is None:
            raise ValueError(f"Mesh '{mesh_name}' not found in {_mesh_gpkg}")

    # ── CRS / unit system ──────────────────────────────────────────────
    crs_wkt = str(md.get("crs_wkt", "") or (
        mesh_dict.get("crs_wkt", "") if isinstance(mesh_dict, dict) else ""
    ))
    from swe2d import units as _u
    si_m_per_model = _u.si_m_per_model_from_wkt(crs_wkt) if crs_wkt else 1.0
    _u.configure(si_m_per_model)

    units_cfg = p.get("units", {}) or {}
    params = p.get("params", {}) or {}
    results_cfg = p.get("results", {}) or {}

    # Helper: resolve param from top-level, then nested params, then default
    def _v(key: str, default: Any = None) -> Any:
        if key in p:
            return p[key]
        return params.get(key, default)

    # Helper: return a pre-built value from the spec (top-level only — these
    # are objects, not scalar params and never live under ``params``).
    # Used so the GUI adapter can pass already-built forcing/mesh objects
    # through the spec instead of overriding them via dataclasses.replace
    # after the build.  When the spec key is missing or None, the
    # GPKG-derived ``computed`` value is used.
    def _override(key: str, computed: Any) -> Any:
        val = p.get(key)
        return val if val is not None else computed

    # ── Resolve results GPKG ───────────────────────────────────────────
    _results_gpkg = results_gpkg or str(
        _v("results_gpkg_path", "")
        or params.get("results_gpkg_path_edit", "")
        or results_cfg.get("results_gpkg_path", "")
    )

    # ── Run identity ───────────────────────────────────────────────────
    run_id = str(p.get("run_id", "") or p.get("id", "") or
                 datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z"))
    run_wallclock_start = str(p.get("run_wallclock_start", ""))
    run_log_start_idx = int(p.get("run_log_start_idx", 0))

    # ── Pre-compute cell geometry callbacks from mesh ──────────────────
    node_x = md["node_x"]
    node_y = md["node_y"]
    node_z = md["node_z"]
    cell_nodes = md["cell_nodes"]
    face_offsets = md.get("cell_face_offsets")
    face_nodes = md.get("cell_face_nodes")
    n_cells = int(cell_nodes.shape[0]) if face_offsets is None else int(face_offsets.size - 1)

    from swe2d.services.mesh_computation_service import (
        mesh_cell_areas as _svc_cell_areas,
        mesh_cell_min_bed as _svc_cell_min_bed,
        mesh_cell_centroids as _svc_cell_centroids,
    )
    _cell_areas = _svc_cell_areas(md)
    _cell_bed = _svc_cell_min_bed(md)

    def _mesh_cell_areas():
        return _cell_areas

    def _mesh_cell_min_bed():
        return _cell_bed

    def _mesh_cell_centroids():
        return _svc_cell_centroids(md)

    # ── BC arrays ──────────────────────────────────────────────────────
    bc_n0 = md.get("bc_edge_node0", np.empty(0, dtype=np.int32))
    bc_n1 = md.get("bc_edge_node1", np.empty(0, dtype=np.int32))
    bc_relax = np.zeros(bc_n0.size, dtype=np.float64)

    default_bc_type = int(_v("default_bc_type", 1))
    if bc_n0.size > 0:
        md_for_bc = {"node_x": node_x, "node_y": node_y}
        from swe2d.services.mesh_computation_service import default_bc_for_edges
        bc_tp, bc_vl = default_bc_for_edges(md_for_bc, bc_n0, bc_n1, default_bc_type=default_bc_type)
    else:
        bc_tp = np.empty(0, dtype=np.int32)
        bc_vl = np.empty(0, dtype=np.float64)

    bc_cfg = p.get("bc_lines") or {}
    if isinstance(bc_cfg, dict) and bc_cfg.get("table"):
        _bc_gpkg = str(bc_cfg.get("gpkg") or _mesh_gpkg)
        _require_gpkg_table("bc_lines", _bc_gpkg, str(bc_cfg["table"]))
    if isinstance(bc_cfg, dict) and bc_cfg.get("table") and bc_n0.size > 0:
        try:
            from swe2d.core.boundary_qgis_adapter import (
                apply_bc_layer_overrides_from_gpkg as _bc_override,
            )
            bc_tp, bc_vl, bc_relax = _bc_override(
                gpkg_path=bc_cfg.get("gpkg", _mesh_gpkg),
                table_name=bc_cfg["table"],
                mesh_data={"node_x": node_x, "node_y": node_y},
                edge_n0=bc_n0,
                edge_n1=bc_n1,
                bc_type=bc_tp,
                bc_val=bc_vl,
                default_relax=float(_v("open_bc_relaxation", 0.0)),
                log_fn=logger.info,
            )
        except Exception as exc:
            raise _build_error(
                "bc_lines", "could not apply boundary overrides from GPKG", exc
            ) from exc

    # ── Hydrograph BCs ─────────────────────────────────────────────────
    side_hydrographs: Dict[str, Any] = {}
    edge_hydrographs: Dict[int, Any] = {}
    if isinstance(bc_cfg, dict) and bc_cfg.get("table") and bc_cfg.get("hydrograph_table"):
        _require_gpkg_table(
            "bc_lines", _bc_gpkg, str(bc_cfg["hydrograph_table"])
        )
        try:
            from swe2d.core.gpkg_io import collect_bc_layer_hydrographs_from_gpkg as _bc_hyd
            edge_hg_data = _bc_hyd(
                gpkg_path=bc_cfg.get("gpkg", _mesh_gpkg),
                bc_table=bc_cfg["table"],
                mesh_data={"node_x": node_x, "node_y": node_y},
                edge_n0=bc_n0,
                edge_n1=bc_n1,
                hydrograph_table=bc_cfg.get("hydrograph_table", "SWE2D_Hydrographs"),
                log_fn=logger.info,
            )
            if edge_hg_data:
                edge_hydrographs = edge_hg_data
                logger.info("Loaded %d edge hydrographs from GPKG", len(edge_hydrographs))
        except Exception as exc:
            raise _build_error(
                "bc_lines", "could not load boundary hydrographs", exc
            ) from exc

    # ── Thiessen forcing (rain + CN) ───────────────────────────────────
    # Phase 3.1: route the spec's `hyetograph`/`rain_cn` data sources through
    # the QGIS-based ``build_thiessen_rain_cn_forcing_from_gpkg`` shim — the
    # same builder the GUI dialog uses.  The raw-sqlite3 reimplementation
    # ``build_forced_thiessen_from_gpkg`` (raw cell→gauge mapping) is dead
    # and was removed in Phase 3.6.
    thiessen_forcing = None
    hyeto = p.get("hyetograph") or {}
    if isinstance(hyeto, dict) and hyeto.get("table") and hyeto.get("gauge_layer"):
        from swe2d.core.gpkg_io import build_thiessen_rain_cn_forcing_from_gpkg
        h_gpkg = hyeto.get("gpkg", _mesh_gpkg)
        htable = hyeto["table"]
        gtable = hyeto["gauge_layer"]
        cntable = p.get("rain_cn") or {}
        cn_table = cntable.get("table") if isinstance(cntable, dict) else None
        cn_field = cntable.get("cn_field", "cn") if isinstance(cntable, dict) else "cn"
        # ia_ratio lives in the spec under either ``rain_cn`` block or as
        # a top-level field; default to 0.2 (the SCS-CN standard).
        ia_ratio = float(cntable.get("ia_ratio", 0.2) if isinstance(cntable, dict) else 0.2)
        infil = str(_v("infiltration_method", "scs_cn"))
        _require_gpkg_table("hyetograph", str(h_gpkg), str(htable))
        _require_gpkg_table("hyetograph", str(h_gpkg), str(gtable))
        if cn_table:
            _require_gpkg_table("rain_cn", str(h_gpkg), str(cn_table))
        try:
            thiessen_forcing = build_thiessen_rain_cn_forcing_from_gpkg(
                gpkg_path=h_gpkg,
                hyetograph_table=htable,
                gauge_table=gtable,
                n_cells=n_cells,
                mesh_data=md,
                cn_table=cn_table,
                cn_field=cn_field,
                infiltration_method=infil,
                ia_ratio=ia_ratio,
                use_spatial_rain_cn=bool(cn_table),
                log_fn=logger.info,
            )
        except Exception as exc:
            raise _build_error(
                "hyetograph", "could not build rainfall/CN forcing", exc
            ) from exc
        if thiessen_forcing is None:
            raise BuildRunContextError(
                "spec key 'hyetograph' produced no rainfall/CN forcing"
            )

    # ── Internal flow sources ──────────────────────────────────────────
    internal_flow_forcing = None
    ifs_cfg = p.get("internal_flow_sources") or {}
    if isinstance(ifs_cfg, dict) and ifs_cfg.get("table"):
        ifs_gpkg = str(ifs_cfg.get("gpkg") or _mesh_gpkg)
        _require_gpkg_table(
            "internal_flow_sources", ifs_gpkg, str(ifs_cfg["table"])
        )
        if ifs_cfg.get("hydrograph_table"):
            _require_gpkg_table(
                "internal_flow_sources",
                ifs_gpkg,
                str(ifs_cfg["hydrograph_table"]),
            )
        try:
            from swe2d.core.gpkg_io import build_internal_flow_forcing_from_gpkg as _ifs_logic
            internal_flow_forcing = _ifs_logic(
                gpkg_path=ifs_gpkg,
                table_name=ifs_cfg["table"],
                mesh_data=md,
                requested_field_name=ifs_cfg.get("field", "src_value"),
                hydrograph_table=ifs_cfg.get("hydrograph_table", "SWE2D_Hydrographs"),
                log_fn=logger.info,
            )
            if internal_flow_forcing is None:
                raise BuildRunContextError(
                    "spec key 'internal_flow_sources' produced no forcing"
                )
        except BuildRunContextError:
            raise
        except Exception as exc:
            raise _build_error(
                "internal_flow_sources", "could not load forcing from GPKG", exc
            ) from exc

    # ── Drainage network config ────────────────────────────────────────
    # Phase 3.2: single builder path.  Two valid spec forms:
    #   1. ``nodes_layer`` form — opens nodes/links/inlets GPKG layers and
    #      delegates to ``build_pipe_network_config`` (the same GUI logic).
    #   2. Inline JSON ``{nodes, links, inlets, ...}`` form — delegates to
    #      ``build_drainage_config_from_json`` in ``extensions/drainage_network.py``.
    # Silently dropping the inline form was a real bug (Phase 0 allowlist
    # entry, now retired).  No third silent state: if a drainage block is
    # present but matches neither form, raise a typed ValueError.
    pipe_network_cfg = None
    drainage_cfg = p.get("drainage") or {}
    if isinstance(drainage_cfg, dict) and drainage_cfg:
        if "nodes_layer" in drainage_cfg:
            # Form 1: GPKG layers.
            _dgpkg = str(drainage_cfg.get("gpkg") or _mesh_gpkg)
            for layer_key in (
                "nodes_layer",
                "links_layer",
                "inlets_layer",
                "node_inlets_layer",
            ):
                if drainage_cfg.get(layer_key):
                    _require_gpkg_table(
                        "drainage", _dgpkg, str(drainage_cfg[layer_key])
                    )
            try:
                from swe2d.core.gpkg_io import (
                    _build_drainage_config_from_gpkg_layers as _drainage_logic,
                )
                pipe_network_cfg = _drainage_logic(
                    mesh_data=md,
                    drainage_gpkg=_dgpkg,
                    nodes_layer=drainage_cfg["nodes_layer"],
                    links_layer=drainage_cfg["links_layer"],
                    inlets_layer=drainage_cfg.get("inlets_layer"),
                    node_inlets_layer=drainage_cfg.get("node_inlets_layer"),
                    cell_min_bed=_cell_bed,
                    gravity=_u.gravity(),
                    config=_drainage_config_dict(drainage_cfg, _v),
                    log_fn=logger.info,
                )
                if pipe_network_cfg is None:
                    raise BuildRunContextError(
                        "spec key 'drainage' produced no pipe network config"
                    )
            except BuildRunContextError:
                raise
            except Exception as exc:
                raise _build_error(
                    "drainage",
                    "rejected GPKG layers "
                    f"(gpkg={_dgpkg!r}, "
                    f"nodes_layer={drainage_cfg.get('nodes_layer')!r}, "
                    f"links_layer={drainage_cfg.get('links_layer')!r})",
                    exc,
                ) from exc
        elif "nodes" in drainage_cfg or "links" in drainage_cfg:
            # Form 2: inline JSON.  No silent drop — if the inline data
            # is malformed, raise so the caller sees the failure.
            try:
                from swe2d.extensions.drainage_network import (
                    build_drainage_config_from_json,
                )
                pipe_network_cfg = build_drainage_config_from_json(
                    drainage_cfg, n_cells,
                )
                if pipe_network_cfg is None:
                    raise BuildRunContextError(
                        "spec key 'drainage' inline data is missing nodes or links"
                    )
            except BuildRunContextError:
                raise
            except Exception as exc:
                raise _build_error(
                    "drainage", "rejected inline drainage data", exc
                ) from exc
        else:
            raise BuildRunContextError(
                "spec key 'drainage' must be either "
                "{\"nodes_layer\": ..., \"links_layer\": ...} (GPKG layers) "
                "or {\"nodes\": [...], \"links\": [...]} (inline JSON); "
                f"got keys {sorted(drainage_cfg.keys())}"
            )

    # ── Hydraulic structures config ────────────────────────────────────
    hydraulic_structures_cfg = None
    structures_data = p.get("structures") or {}
    if isinstance(structures_data, dict) and structures_data.get("table"):
        hs_gpkg = str(structures_data.get("gpkg") or _mesh_gpkg)
        _require_gpkg_table(
            "structures", hs_gpkg, str(structures_data["table"])
        )
        try:
            from swe2d.core.gpkg_io import (
                build_hydraulic_structure_config_from_gpkg as _hs_logic,
            )
            hydraulic_structures_cfg = _hs_logic(
                gpkg_path=hs_gpkg,
                structures_table=structures_data["table"],
                mesh_data=md,
                log_fn=logger.info,
            )
            if hydraulic_structures_cfg is None:
                raise BuildRunContextError(
                    "spec key 'structures' produced no hydraulic structure config"
                )
        except BuildRunContextError:
            raise
        except Exception as exc:
            raise _build_error(
                "structures", "could not load hydraulic structures from GPKG", exc
            ) from exc
    elif isinstance(structures_data, dict) and structures_data:
        try:
            from swe2d.extensions.structures import build_structures_config_from_json
            hydraulic_structures_cfg = build_structures_config_from_json(structures_data, n_cells)
            if hydraulic_structures_cfg is None:
                raise BuildRunContextError(
                    "spec key 'structures' produced no hydraulic structure config"
                )
        except BuildRunContextError:
            raise
        except Exception as exc:
            raise _build_error(
                "structures", "could not build inline hydraulic structures", exc
            ) from exc

    # ── Bridge stacked plans ───────────────────────────────────────────
    bridge_stacked_plans: List[Any] = []
    try:
        from swe2d.runtime.bridge_stacked_runtime import (
            build_bridge_stacked_plans_for_runtime,
        )
        bridge_stacked_plans = build_bridge_stacked_plans_for_runtime(
            md, hydraulic_structures_cfg, log_fn=logger.info,
        )
    except Exception as exc:
        raise _build_error(
            "structures", "could not build bridge runtime plans", exc
        ) from exc

    # ── Derived model-space source rates ─────────────────────────────
    cell_source_model = None
    if internal_flow_forcing is not None:
        try:
            from swe2d.boundary_and_forcing.runtime_source_logic import (
                internal_flow_source_cms_at_time as _ifs_cms,
            )
            from swe2d.boundary_and_forcing.bc_logic import (
                interp_hydrograph as _interp_hydrograph,
            )
            cell_source_si = _ifs_cms(internal_flow_forcing, 0.0, _interp_hydrograph)
            if cell_source_si is not None:
                cell_source_model = _u.flow_si_to_model(cell_source_si)
        except Exception as exc:
            raise _build_error(
                "internal_flow_sources", "could not derive initial cell source", exc
            ) from exc

    rain_rate_mmhr = float(_v("rain_rate_spin", 0.0))
    try:
        rain_rate_model = _u.rain_si_to_model(rain_rate_mmhr / 1000.0 / 3600.0)
    except Exception as exc:
        raise _build_error(
            "rain_rate_spin", "could not convert rainfall rate to model units", exc
        ) from exc

    # bridge_cuda_coupling
    bridge_cuda_coupling = False
    if hydraulic_structures_cfg is not None:
        try:
            from swe2d.extensions.extension_models import StructureType
            structures = getattr(hydraulic_structures_cfg, "structures", None) or []
            has_bridge = any(
                int(getattr(s, "structure_type", 0)) == int(StructureType.BRIDGE)
                for s in structures
            )
            bridge_cuda_coupling = bool(has_bridge)
        except Exception as exc:
            raise _build_error(
                "structures", "could not determine bridge coupling mode", exc
            ) from exc

    # ── internal_flow_source_cms_at_time callback ─────────────────────
    from swe2d.boundary_and_forcing.runtime_source_logic import (
        internal_flow_source_cms_at_time as _internal_flow_source_cms_at_time_logic,
    )
    from swe2d.boundary_and_forcing.bc_logic import (
        interp_hydrograph as _interp_hydrograph,
    )

    def _internal_flow_source_cms_at_time(forcing, t_sec):
        return _internal_flow_source_cms_at_time_logic(
            forcing, t_sec, _interp_hydrograph,
        )

    # ── Edge group labels ──────────────────────────────────────────────
    edge_groups_dict: Dict[int, str] = {}
    if isinstance(bc_cfg, dict) and bc_cfg.get("table"):
        try:
            from swe2d.core.gpkg_io import (
                collect_bc_layer_edge_groups_from_gpkg as _bc_groups,
            )
            edge_groups_dict = _bc_groups(
                gpkg_path=bc_cfg.get("gpkg", _mesh_gpkg),
                bc_table=bc_cfg["table"],
                mesh_data={"node_x": node_x, "node_y": node_y},
                edge_n0=bc_n0,
                edge_n1=bc_n1,
                log_fn=logger.info,
            )
            if edge_groups_dict:
                logger.info("Loaded %d edge groups from GPKG", len(edge_groups_dict))
        except Exception as exc:
            raise _build_error(
                "bc_lines", "could not load boundary edge groups", exc
            ) from exc

    # ── Sample lines ──────────────────────────────────────────────────
    sample_lines_cfg = p.get("sample_lines") or {}
    sample_map_data: List[Dict[str, Any]] = []
    if isinstance(sample_lines_cfg, dict) and sample_lines_cfg.get("table"):
        sample_gpkg = str(sample_lines_cfg.get("gpkg") or _mesh_gpkg)
        _require_gpkg_table(
            "sample_lines", sample_gpkg, str(sample_lines_cfg["table"])
        )
        try:
            from swe2d.core.gpkg_io import (
                build_line_sampling_map_from_gpkg as _sample_map_logic,
            )
            sample_map_data = _sample_map_logic(
                gpkg_path=sample_lines_cfg.get("gpkg", _mesh_gpkg),
                sample_lines_table=sample_lines_cfg["table"],
                mesh_data=md,
                log_fn=logger.info,
            )
            if sample_map_data:
                logger.info("Built sample map with %d lines", len(sample_map_data))
            else:
                raise BuildRunContextError(
                    "spec key 'sample_lines' produced an empty sampling map"
                )
        except BuildRunContextError:
            raise
        except Exception as exc:
            raise _build_error(
                "sample_lines", "could not build line sampling map", exc
            ) from exc

    # ── Coupling SOA ──────────────────────────────────────────────────
    coupling_soa = None
    try:
        from swe2d.runtime.coupling import pack_coupling_soa
        if pack_coupling_soa is not None and (
            pipe_network_cfg is not None or hydraulic_structures_cfg is not None
        ):
            coupling_soa = pack_coupling_soa(
                n_cells=n_cells,
                pipe_network=pipe_network_cfg,
                hydraulic_structures=hydraulic_structures_cfg,
            )
    except Exception as exc:
        raise _build_error(
            "coupling_soa", "could not pack drainage/structure coupling data", exc
        ) from exc

    # ── n_mann_cell from mesh ──────────────────────────────────────────
    n_mann_cell = md.get("n_mann_cell")

    # ── Initial state ──────────────────────────────────────────────────
    # NOTE: use explicit None checks (not ``or``) — h0 may be a numpy array
    # (GUI adapter path) whose truth value is ambiguous.
    _h0_user = p.get("h0")
    if _h0_user is None:
        _h0_user = params.get("h0")
    if _h0_user is not None:
        h0 = np.asarray(_h0_user, dtype=np.float64)
        if h0.size != n_cells:
            raise ValueError(
                f"h0 has {h0.size} elements but mesh has {n_cells} cells"
            )
        hu0 = np.zeros(n_cells, dtype=np.float64)
        hv0 = np.zeros(n_cells, dtype=np.float64)
    else:
        _ic_mode = str(_v("initial_condition_mode", "")).strip().lower()
        _init_wse = _v("initial_wse", _v("initial_water_surface_elevation", None))
        _init_depth = _v("initial_depth", None)
        if _ic_mode in ("dry", "uniform_depth", "uniform_wse"):
            # Use the same initial_state logic the GUI uses — mode-aware,
            # handles dry-start priming for inflow-adjacent cells.
            from swe2d.mesh.mesh_runtime_logic import initial_state as _init_state
            _h_min_val = float(_v("h_min", _DEFAULTS["h_min"]))
            h0, hu0, hv0 = _init_state(
                mesh_data=md,
                mode=_ic_mode,
                initial_depth=float(_init_depth) if _init_depth is not None else 0.0,
                initial_wse=float(_init_wse) if _init_wse is not None else 0.0,
                h_min=_h_min_val,
                bc_n0=bc_n0,
                bc_n1=bc_n1,
                bc_tp=bc_tp,
                log_fn=logger.info,
            )
        else:
            # Legacy fallback: no mode specified — use whichever spin value is
            # present (preserves backward compat with specs that predate the
            # initial_condition_mode key).
            if _init_wse is not None:
                h0 = np.maximum(0.0, float(_init_wse) - _cell_bed)
            elif _init_depth is not None:
                h0 = np.full(n_cells, max(0.0, float(_init_depth)), dtype=np.float64)
            else:
                h0 = np.zeros(n_cells, dtype=np.float64)
            hu0 = np.zeros(n_cells, dtype=np.float64)
            hv0 = np.zeros(n_cells, dtype=np.float64)

    # ── Derive timestep mode (dt_request, dt_fixed) from adaptive_cfl_dt + dt_cfg ──
    # Restores the rule the retired swe2d/runtime/run_options_builder.py applied
    # (commit 70561f9a lost it; the C++ cap stayed at 0.05 forever).
    # C++ binding convention: ``dt_request = -1.0`` means "pure CFL", ``dt_fixed > 0``
    # means "ignore CFL, use this fixed dt".  An explicit caller-supplied
    # dt_request / dt_fixed (CLI / replay spec) still wins so replay can lock dt.
    _adaptive = bool(_v("adaptive_cfl_dt", _DEFAULTS["adaptive_cfl_dt"]))
    _derived_dt = -1.0 if _adaptive else float(
        _v("dt_cfg", _v("dt_max", _DEFAULTS["dt_cfg"]))
    )
    dt_request = (
        float(_v("dt_request", _derived_dt))
        if ("dt_request" in p) or ("dt_request" in params)
        else _derived_dt
    )
    dt_fixed = (
        float(_v("dt_fixed", _derived_dt))
        if ("dt_fixed" in p) or ("dt_fixed" in params)
        else _derived_dt
    )

    return RunContext(
        run_id=run_id,
        run_wallclock_start=run_wallclock_start,
        run_log_start_idx=run_log_start_idx,
        results_gpkg_path=_results_gpkg,
        model_gpkg_path=_mesh_gpkg,
        mesh_name=mesh_name,
        mesh_crs_wkt=crs_wkt,

        # Time — single defaults table (dt_cfg default 0.05, not 0.2;
        # output_interval_s default 1.0, NOT chained to run_duration_s)
        run_duration_s=float(_v("run_duration_s", _v("duration_s", _DEFAULTS["run_duration_s"]))),
        output_interval_s=float(_v("output_interval_s", _DEFAULTS["output_interval_s"])),
        dt_cfg=float(_v("dt_cfg", _v("dt_max", _DEFAULTS["dt_cfg"]))),
        dt_request=dt_request,
        dt_fixed=dt_fixed,
        initial_dt=float(_v("initial_dt", _DEFAULTS["initial_dt"])),
        adaptive_cfl_dt=_adaptive,

        # Solver modes
        reconstruction_mode=int(_v("reconstruction_mode", _DEFAULTS["reconstruction_mode"])),
        reconstruction_name=str(_v("reconstruction_name", _DEFAULTS["reconstruction_name"])),
        temporal_scheme=_v("temporal_scheme", _DEFAULTS["temporal_scheme"]),
        temporal_scheme_name=str(_v("temporal_scheme_name", _DEFAULTS["temporal_scheme_name"])),
        solver_backend_mode=str(_v("solver_backend_mode", _DEFAULTS["solver_backend_mode"])).strip().lower(),
        coupling_loop_mode=str(_v("coupling_loop_mode", _DEFAULTS["coupling_loop_mode"])).strip().lower(),
        drainage_solver_backend_mode=str(_v("drainage_solver_backend_mode", _DEFAULTS["drainage_solver_backend_mode"])).strip().lower(),
        drainage_gpu_method_mode=str(_v("drainage_gpu_method_mode", _DEFAULTS["drainage_gpu_method_mode"])).strip().lower(),
        culvert_solver_mode=int(_v("culvert_solver_mode", _DEFAULTS["culvert_solver_mode"])),
        cuda_graphs_enabled=bool(_v("cuda_graphs_enabled", _DEFAULTS["cuda_graphs_enabled"])),
        swe2d_perf_mode=bool(_v("swe2d_perf_mode", _DEFAULTS["swe2d_perf_mode"])),
        bridge_cuda_coupling=bridge_cuda_coupling,
        bridge_stacked_coupling_mode=str(_v("bridge_stacked_coupling_mode", _DEFAULTS["bridge_stacked_coupling_mode"])),
        culvert_face_flux_mode=str(_v("culvert_face_flux_mode", _DEFAULTS["culvert_face_flux_mode"])),

        # Numerics
        gravity=float(_v("gravity", _u.gravity())),
        k_mann=float(_v("k_mann", _u.manning_factor())),
        n_mann=float(_v("n_mann", _DEFAULTS["n_mann"])),
        cfl=float(_v("cfl", _DEFAULTS["cfl"])),
        h_min=float(_v("h_min", _DEFAULTS["h_min"])),
        max_inv_area=float(_v("max_inv_area", _DEFAULTS["max_inv_area"])),
        cfl_lambda_cap=float(_v("cfl_lambda_cap", _DEFAULTS["cfl_lambda_cap"])),
        momentum_cap_min_speed=float(_v("momentum_cap_min_speed", _DEFAULTS["momentum_cap_min_speed"])),
        momentum_cap_celerity_mult=float(_v("momentum_cap_celerity_mult", _DEFAULTS["momentum_cap_celerity_mult"])),
        depth_cap=float(_v("depth_cap", _DEFAULTS["depth_cap"])),
        max_rel_depth_increase=float(_v("max_rel_depth_increase", _DEFAULTS["max_rel_depth_increase"])),
        shallow_damping_depth=float(_v("shallow_damping_depth", _DEFAULTS["shallow_damping_depth"])),
        source_cfl_beta=float(_v("source_cfl_beta", _DEFAULTS["source_cfl_beta"])),
        source_max_substeps=int(_v("source_max_substeps", _DEFAULTS["source_max_substeps"])),
        source_rate_cap=float(_v("source_rate_cap", _DEFAULTS["source_rate_cap"])),
        source_depth_step_cap=float(_v("source_depth_step_cap", _DEFAULTS["source_depth_step_cap"])),
        source_true_subcycling=bool(_v("source_true_subcycling", _DEFAULTS["source_true_subcycling"])),
        source_imex_split=bool(_v("source_imex_split", _DEFAULTS["source_imex_split"])),
        gpu_diag_sync_interval_steps=int(_v("gpu_diag_sync_interval_steps", _DEFAULTS["gpu_diag_sync_interval_steps"])),
        tiny_mode=int(_v("tiny_mode", _DEFAULTS["tiny_mode"])),
        tiny_wet_cell_threshold=int(_v("tiny_wet_cell_threshold", _DEFAULTS["tiny_wet_cell_threshold"])),
        degen_mode=int(_v("degen_mode", _DEFAULTS["degen_mode"])),
        front_flux_damping=float(_v("front_flux_damping", _DEFAULTS["front_flux_damping"])),
        open_bc_relaxation=float(_v("open_bc_relaxation", _DEFAULTS["open_bc_relaxation"])),
        active_set_hysteresis=bool(_v("active_set_hysteresis", _DEFAULTS["active_set_hysteresis"])),
        use_redistribution=bool(_v("use_redistribution", _DEFAULTS["use_redistribution"])),
        inflow_progressive=bool(_v("inflow_progressive", _DEFAULTS["inflow_progressive"])),
        uniform_inflow_enabled=bool(_v("uniform_inflow_enabled", _DEFAULTS["uniform_inflow_enabled"])),
        rain_update_interval_s=float(_v("rain_update_interval_s", _DEFAULTS["rain_update_interval_s"])),

        # Mesh arrays — spec-supplied forcing/mesh values override the
        # GPKG-derived ones so the GUI adapter can pass pre-built objects
        # through the spec without post-overriding the RunContext.
        # See ``_override`` helper above.
        node_x=node_x,
        node_y=node_y,
        node_z=node_z,
        cell_nodes=cell_nodes,
        face_offsets=face_offsets,
        face_nodes=face_nodes,
        bc_n0=_override("bc_n0", bc_n0),
        bc_n1=_override("bc_n1", bc_n1),
        bc_tp=_override("bc_tp", bc_tp),
        bc_vl=_override("bc_vl", bc_vl),
        bc_relax=_override("bc_relax", bc_relax),
        side_hydrographs=_override("side_hydrographs", side_hydrographs),
        edge_hydrographs=_override("edge_hydrographs", edge_hydrographs),
        edge_group_overrides=_override("edge_group_overrides", {}),
        h0=_override("h0", h0),
        hu0=_override("hu0", hu0),
        hv0=_override("hv0", hv0),
        n_mann_cell=_override("n_mann_cell", n_mann_cell),
        cell_areas=_override("cell_areas", _cell_areas),
        cell_centroids=_override("cell_centroids", np.column_stack(_mesh_cell_centroids())),

        # Forcing / coupling
        rain_rate_model=_override("rain_rate_model", rain_rate_model),
        internal_flow_forcing=_override("internal_flow_forcing", internal_flow_forcing),
        cell_source_model=_override("cell_source_model", cell_source_model),
        thiessen_forcing=_override("thiessen_forcing", thiessen_forcing),
        pipe_network_cfg=_override("pipe_network_cfg", pipe_network_cfg),
        hydraulic_structures_cfg=_override("hydraulic_structures_cfg", hydraulic_structures_cfg),
        bridge_stacked_plans=_override("bridge_stacked_plans", bridge_stacked_plans),
        coupling_soa=_override("coupling_soa", coupling_soa),

        # Storage flags
        save_mesh_results=bool(_v("save_mesh_results", results_cfg.get("save_mesh_results", _DEFAULTS["save_mesh_results"]))),
        save_line_results=bool(_v("save_line_results", results_cfg.get("save_line_results", _DEFAULTS["save_line_results"]))),
        save_coupling_results=bool(_v("save_coupling_results", results_cfg.get("save_coupling_results", _DEFAULTS["save_coupling_results"]))),
        save_run_log=bool(_v("save_run_log", results_cfg.get("save_run_log", _DEFAULTS["save_run_log"]))),
        save_max_only=bool(_v("save_max_only", results_cfg.get("save_max_only", _DEFAULTS["save_max_only"]))),

        # Units
        length_unit_name=str(_override("length_unit_name", units_cfg.get("length_unit_name", _u.length_unit_name()))),
        length_scale_si_to_model=float(_override("length_scale_si_to_model", units_cfg.get("length_scale_si_to_model", _u.si_m_per_model()))),
        rain_mm_to_model_depth=float(_override("rain_mm_to_model_depth", units_cfg.get("rain_mm_to_model_depth", 1e-3 * _u.model_per_si_m()))),
        rain_rate_si_to_model=float(_override("rain_rate_si_to_model", units_cfg.get("rain_rate_si_to_model", _u.model_per_si_m()))),
        flow_si_to_model=float(_override("flow_si_to_model", units_cfg.get("flow_si_to_model", _u.model_per_si_m() ** 3))),

        # Callbacks — pre-computed from mesh data (no Qt dependency).
        # Legacy callback fields are rebound to pure implementations inside the
        # executor. Keep an explicit failing sentinel here instead of silent
        # no-op lambdas so accidental use cannot hide missing wiring.
        mesh_cell_areas=_override("mesh_cell_areas", _mesh_cell_areas),
        mesh_cell_min_bed=_override("mesh_cell_min_bed", _mesh_cell_min_bed),
        mesh_cell_centroids=_override("mesh_cell_centroids", _mesh_cell_centroids),
        apply_timeseries_bc_values=_override("apply_timeseries_bc_values", _UNSET_RUN_CONTEXT_CALLBACK),
        distribute_total_flow_to_unit_q=_override("distribute_total_flow_to_unit_q", _UNSET_RUN_CONTEXT_CALLBACK),
        apply_external_sources=_override("apply_external_sources", _UNSET_RUN_CONTEXT_CALLBACK),
        build_line_sampling_map=_override("build_line_sampling_map", _UNSET_RUN_CONTEXT_CALLBACK),
        internal_flow_source_cms_at_time=_override("internal_flow_source_cms_at_time", _internal_flow_source_cms_at_time),

        # Misc
        sample_map_data=_override("sample_map_data", sample_map_data),
        inflow_progressive_enabled=bool(_override("inflow_progressive_enabled", _v("inflow_progressive_enabled", _DEFAULTS["inflow_progressive_enabled"]))),
        edge_groups=_override("edge_groups", edge_groups_dict),
        cancel_event=cancel_event or threading.Event(),
    )


# ── RunContextBuilder — fluent builder with layered defaults ────────────────────
# Supports the GUI's need to layer multiple spec dicts (e.g. merge a base
# config with widget overrides) before calling build().
#
# Usage:
#   ctx = (RunContextBuilder.from_defaults()
#          .merge_context(base_ctx)
#          .with_params({"run_duration_s": 3600.0, "cfl": 0.5})
#          .build(mesh_gpkg="model.gpkg"))
#
# The builder normalizes every input spec via _normalize_spec() so callers
# can pass widget-state dicts, replay JSON, or canonical swe2d-run/2 specs.

class RunContextBuilder:
    """Fluent builder for RunContext with layered spec merging.

    Supports two construction modes:

    - **"cli"** — accepts a single spec dict (canonical swe2d-run/2 or
      flat CLI JSON).  ``with_params()`` layers additional specs on top.
      Call ``build()`` to produce the final ``RunContext``.

    - **"gui"** — first calls ``from_defaults()``, then layers one or more
      ``RunContext`` objects via ``merge_context()`` (capturing live GUI
      widget state), then layers additional spec dicts via
      ``with_params()``.  The final ``build()`` call uses the accumulated
      stack to compute the RunContext fields.

    The builder normalizes all spec inputs via the GUI-compat normalization
    front-stage, so widget-name keys, string meshes, and legacy flat JSON
    are all accepted transparently.
    """

    def __init__(
        self,
        mode: Literal["cli", "gui"],
        stack: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> None:
        if mode not in ("cli", "gui"):
            raise ValueError(f"mode must be 'cli' or 'gui', got {mode!r}")
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_stack", list(stack) if stack else [])
        object.__setattr__(self, "_vars", dict(vars) if vars else {})

    # ── factory constructors ───────────────────────────────────────────

    @classmethod
    def from_defaults(cls) -> "RunContextBuilder":
        """Create a builder pre-populated with the shared defaults table.

        Use this as the base for GUI paths that need to merge live widget
        state on top of defaults.  CLI callers can use ``build_run_context``
        directly without going through the builder.
        """
        return cls("gui", stack=[dict(_DEFAULTS)], vars={})

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "RunContextBuilder":
        """Create a builder from a single spec dict (canonical or legacy)."""
        normalized = _normalize_spec(spec)
        return cls("cli", stack=[normalized], vars={})

    # ── fluent mutation ────────────────────────────────────────────────

    def merge_context(self, other: RunContext) -> "RunContextBuilder":
        """Merge a RunContext's scalar fields into the builder stack.

        Produces a new builder; does not mutate ``other``.  Useful for the
        GUI path where a base RunContext (from defaults or a saved config)
        is layered with the current widget state captured as another
        RunContext.

        Only scalar param fields are merged; array/callback/container fields
        are always recomputed by ``build()`` from the resolved mesh.
        """
        scalar_fields = set(_DEFAULTS.keys())
        ctx_dict: Dict[str, Any] = {}
        for name in scalar_fields:
            val = getattr(other, name, None)
            if val is not None and val != _DEFAULTS.get(name):
                ctx_dict[name] = val
        # Normalize widget names that might appear in merged dicts.
        for _gui_key, _rc_key in WIDGET_TO_RC.items():
            if _gui_key in ctx_dict and _rc_key not in ctx_dict:
                ctx_dict[_rc_key] = ctx_dict.pop(_gui_key)
        new_stack = list(self._stack) + [ctx_dict]
        return RunContextBuilder(
            self._mode,
            stack=new_stack,
            vars=dict(self._vars),
        )

    def with_params(self, spec: Dict[str, Any]) -> "RunContextBuilder":
        """Layer an additional spec dict on top of the current builder stack.

        Later layers override earlier ones.  Accepts canonical swe2d-run/2
        dicts, flat CLI JSON, or widget-state dicts (all normalized).
        """
        normalized = _normalize_spec(spec)
        new_stack = list(self._stack) + [normalized]
        return RunContextBuilder(
            self._mode,
            stack=new_stack,
            vars=dict(self._vars),
        )

    def with_vars(self, vars: Dict[str, Any]) -> "RunContextBuilder":
        """Set arbitrary key-value metadata (not used in RunContext, available for callers)."""
        return RunContextBuilder(
            self._mode,
            stack=list(self._stack),
            vars={**self._vars, **vars},
        )

    # ── stack assembly ────────────────────────────────────────────────

    def _build_mode_stack(self) -> List[Dict[str, Any]]:
        """Assemble the ordered layer stack for the current mode."""
        if self._mode == "cli":
            return self._build_cli_stack()
        else:
            return self._build_gui_stack()

    def _build_cli_stack(self) -> List[Dict[str, Any]]:
        """CLI stack: defaults, then each user-supplied layer in order."""
        stack: List[Dict[str, Any]] = [dict(_DEFAULTS)]
        stack.extend(self._stack)
        return stack

    def _build_gui_stack(self) -> List[Dict[str, Any]]:
        """GUI stack: defaults base, then merged RunContexts, then user specs."""
        # GUI path starts from defaults (set up by from_defaults()).
        # Each merge_context() / with_params() call adds a layer.
        return list(self._stack)

    @staticmethod
    def _merge_stack(stack: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge an ordered stack of dicts (later layers override)."""
        result: Dict[str, Any] = {}
        for layer in stack:
            if not isinstance(layer, dict):
                continue
            result.update(layer)
        return result

    # ── build ─────────────────────────────────────────────────────────

    def build(
        self,
        *,
        mesh_gpkg: str = "",
        results_gpkg: str = "",
        cancel_event: Optional[threading.Event] = None,
    ) -> RunContext:
        """Assemble the final RunContext from the accumulated stack.

        The stack is merged in order (later layers override).  Mesh arrays,
        BC overrides, forcing, and coupling SOA are always recomputed from
        the resolved GPKG; they are never taken from the stack.
        """
        stack = self._build_mode_stack()
        merged = self._merge_stack(stack)
        return build_run_context(
            merged,
            mesh_gpkg=mesh_gpkg,
            results_gpkg=results_gpkg,
            cancel_event=cancel_event,
        )


# ── Legacy thin wrapper ─────────────────────────────────────────────────────────
# build_run_context_from_dict: thin wrapper around build_run_context().
# Preserved for backwards compatibility with existing CLI callers.  New code
# should call build_run_context() directly.

def build_run_context_from_dict(
    p: Dict[str, Any],
    *,
    mesh_gpkg: str = "",
    results_gpkg: str = "",
    cancel_event: Optional[threading.Event] = None,
) -> RunContext:
    """Build a RunContext from a flat parameter dict (legacy CLI path).

    .. deprecated::
        Use :func:`build_run_context` directly.  This function is a thin
        wrapper that calls ``build_run_context`` after normalising the flat
        dict format.  The signature and behaviour are identical; no caller
        needs to change.

    Parameters
    ----------
    p : dict
        Flat parameter dict with optionally nested ``params``, ``mesh``,
        ``results``, ``units``, ``drainage`` sub-dicts.  Top-level keys
        override nested ones.  ``mesh`` may be a string, a dict, or absent
        (in which case ``mesh_gpkg`` and ``mesh_name`` must be provided).
    mesh_gpkg, results_gpkg, cancel_event
        Forwarded to :func:`build_run_context`.

    Returns
    -------
    RunContext
    """
    return build_run_context(
        p,
        mesh_gpkg=mesh_gpkg,
        results_gpkg=results_gpkg,
        cancel_event=cancel_event,
    )

