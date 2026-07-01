"""Pure-Python run orchestration service — zero Qt imports.

Extracts run parameter collection, progress computation, and
configuration validation from swe2d_workbench_qt.py.
"""

from typing import Any, Dict, List, Optional


def collect_run_parameters(
    cfl: float,
    dt: float,
    run_duration: float,
    output_interval: float,
    n_mann: float,
    spatial_scheme: int,
    temporal_scheme: int,
    gravity: float,
    h_min: float,
    use_gpu: bool,
    checkpoint_interval: float,
    max_rel_depth_increase: Optional[float] = None,
    gpu_diag_sync_interval: Optional[int] = None,
    shallow_damping_depth: Optional[float] = None,
    depth_cap: Optional[float] = None,
    momentum_cap_min_speed: Optional[float] = None,
    momentum_cap_celerity_mult: Optional[float] = None,
    max_inv_area: Optional[float] = None,
    cfl_lambda_cap: Optional[float] = None,
    extreme_rain_mode: Optional[bool] = None,
    source_cfl_beta: Optional[float] = None,
    source_max_substeps: Optional[int] = None,
    source_true_subcycling: Optional[bool] = None,
    source_imex_split: Optional[bool] = None,
    tiny_mode: Optional[int] = None,
    tiny_wet_cell_threshold: Optional[int] = None,
    inflow_progressive: Optional[bool] = None,
) -> Dict[str, Any]:
    """Collect and validate run parameters into a typed dict.

    Parameters are flat scalar values (not Qt widgets) so this function
    can be used in headless and testing contexts.
    """
    params: Dict[str, Any] = {
        "cfl": float(cfl),
        "dt": float(dt),
        "run_duration_s": float(run_duration),
        "output_interval_s": float(output_interval),
        "n_mann": float(n_mann),
        "spatial_scheme": int(spatial_scheme),
        "temporal_scheme": int(temporal_scheme),
        "gravity": float(gravity),
        "h_min": float(h_min),
        "use_gpu": bool(use_gpu),
        "checkpoint_interval_s": float(checkpoint_interval),
    }
    if max_rel_depth_increase is not None:
        params["max_rel_depth_increase"] = float(max_rel_depth_increase)
    if gpu_diag_sync_interval is not None:
        params["gpu_diag_sync_interval"] = int(gpu_diag_sync_interval)
    if shallow_damping_depth is not None:
        params["shallow_damping_depth"] = float(shallow_damping_depth)
    if depth_cap is not None:
        params["depth_cap"] = float(depth_cap)
    if momentum_cap_min_speed is not None:
        params["momentum_cap_min_speed"] = float(momentum_cap_min_speed)
    if momentum_cap_celerity_mult is not None:
        params["momentum_cap_celerity_mult"] = float(momentum_cap_celerity_mult)
    if max_inv_area is not None:
        params["max_inv_area"] = float(max_inv_area)
    if cfl_lambda_cap is not None:
        params["cfl_lambda_cap"] = float(cfl_lambda_cap)
    if extreme_rain_mode is not None:
        params["extreme_rain_mode"] = bool(extreme_rain_mode)
    if source_cfl_beta is not None:
        params["source_cfl_beta"] = float(source_cfl_beta)
    if source_max_substeps is not None:
        params["source_max_substeps"] = int(source_max_substeps)
    if source_true_subcycling is not None:
        params["source_true_subcycling"] = bool(source_true_subcycling)
    if source_imex_split is not None:
        params["source_imex_split"] = bool(source_imex_split)
    if tiny_mode is not None:
        params["tiny_mode"] = int(tiny_mode)
    if tiny_wet_cell_threshold is not None:
        params["tiny_wet_cell_threshold"] = int(tiny_wet_cell_threshold)
    if inflow_progressive is not None:
        params["inflow_progressive"] = bool(inflow_progressive)
    return params


def compute_progress(
    run_time: float,
    total_duration: float,
    wall_elapsed: float,
) -> Dict[str, float]:
    """Compute simulation progress metrics.

    Returns
    -------
    dict with keys:
        percent         — fraction complete (0–100)
        eta_s           — estimated wall-clock seconds remaining
        wall_elapsed_s  — wall-clock seconds elapsed
        speedup         — simulation-time / wall-time ratio
    """
    total_duration = max(total_duration, 0.0)
    run_time = max(run_time, 0.0)
    wall_elapsed = max(wall_elapsed, 0.0)

    if total_duration > 0.0:
        percent = min(run_time / total_duration * 100.0, 100.0)
    else:
        percent = 0.0

    remaining_sim = max(total_duration - run_time, 0.0)

    if run_time > 0.0 and wall_elapsed >= 0.0:
        eta_s = remaining_sim * (wall_elapsed / run_time)
        speedup = run_time / wall_elapsed if wall_elapsed > 0.0 else 0.0
    else:
        eta_s = 0.0
        speedup = 0.0

    return {
        "percent": percent,
        "eta_s": eta_s,
        "wall_elapsed_s": wall_elapsed,
        "speedup": speedup,
    }


_REQUIRED_KEYS = {
    "cfl",
    "dt",
    "run_duration_s",
    "output_interval_s",
}

_VALID_SPATIAL_SCHEMES = {0, 1, 2, 3, 4, 5, 6}
_VALID_TEMPORAL_SCHEMES = {1, 2, 3, 4, 5, 6}


def validate_run_configuration(params: Dict[str, Any]) -> List[str]:
    """Validate run parameter configuration.

    Returns a list of error message strings. An empty list means
    the configuration is valid.
    """
    errors: List[str] = []

    for key in _REQUIRED_KEYS:
        if key not in params:
            errors.append(f"Missing required parameter: {key}")

    cfl = params.get("cfl")
    if cfl is not None:
        try:
            if float(cfl) <= 0.0:
                errors.append(f"CFL must be positive, got {cfl}")
        except (ValueError, TypeError):
            errors.append(f"CFL must be a number, got {cfl}")

    dt = params.get("dt")
    if dt is not None:
        try:
            if float(dt) <= 0.0:
                errors.append(f"Timestep dt must be positive, got {dt}")
        except (ValueError, TypeError):
            errors.append(f"dt must be a number, got {dt}")

    run_duration = params.get("run_duration_s")
    if run_duration is not None:
        try:
            if float(run_duration) <= 0.0:
                errors.append(f"Run duration must be positive, got {run_duration}")
        except (ValueError, TypeError):
            errors.append(f"Run duration must be a number, got {run_duration}")

    output_interval = params.get("output_interval_s")
    if output_interval is not None:
        try:
            if float(output_interval) <= 0.0:
                errors.append(f"Output interval must be positive, got {output_interval}")
        except (ValueError, TypeError):
            errors.append(f"Output interval must be a number, got {output_interval}")

    n_mann = params.get("n_mann")
    if n_mann is not None:
        try:
            if float(n_mann) < 0.0:
                errors.append(f"Manning's n must be non-negative, got {n_mann}")
        except (ValueError, TypeError):
            errors.append(f"Manning's n must be a number, got {n_mann}")

    spatial_scheme = params.get("spatial_scheme")
    if spatial_scheme is not None:
        try:
            if int(spatial_scheme) not in _VALID_SPATIAL_SCHEMES:
                errors.append(
                    f"Spatial scheme {spatial_scheme} is out of range "
                    f"(valid: {sorted(_VALID_SPATIAL_SCHEMES)})"
                )
        except (ValueError, TypeError):
            errors.append(f"Spatial scheme must be an integer, got {spatial_scheme}")

    temporal_scheme = params.get("temporal_scheme")
    if temporal_scheme is not None:
        try:
            if int(temporal_scheme) not in _VALID_TEMPORAL_SCHEMES:
                errors.append(
                    f"Temporal scheme {temporal_scheme} is out of range "
                    f"(valid: {sorted(_VALID_TEMPORAL_SCHEMES)})"
                )
        except (ValueError, TypeError):
            errors.append(f"Temporal scheme must be an integer, got {temporal_scheme}")

    return errors
