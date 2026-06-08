"""
Shared backend compat utilities for pybind11 native bindings.

Provides:
- ``call_solver_create_compat`` — three-tier fallback for ``swe2d_create_solver``
  that handles pybind11 signature mismatches when Python passes newer kwargs
  to an older compiled extension.
- ``log_feature_unavailable`` — check for a feature on a native module and log
  a ``[BACKEND]``-tagged warning if it is missing.

Usage::

    from swe2d.runtime.native_binding_compat import call_solver_create_compat

    solver_h = call_solver_create_compat(mod, mesh, h0, use_gpu=True, ...)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional

__all__: list[str] = [
    "call_solver_create_compat",
    "log_feature_unavailable",
]

logger = logging.getLogger(__name__)


def call_solver_create_compat(
    mod: Any,
    *args: Any,
    _compat_logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Any:
    """Call ``mod.swe2d_create_solver`` with kwargs compatible with the loaded
    extension.

    The three-tier fallback behaviour:

    1. Try the direct call with all kwargs.
    2. If a ``TypeError`` with "incompatible function arguments" is raised
       (the signature pybind11 emits for C++/Python argument mismatches),
       introspect the function signature and retry with only the kwargs that
       the function actually accepts.
    3. If signature introspection is unavailable (some pybind11 environments)
       fall back to a conservative hard-coded list of newer kwargs that are
       stripped before retrying.

    Parameters
    ----------
    mod :
        The loaded native extension module (e.g. ``hydra_swe2d``).
    *args :
        Positional arguments forwarded to ``swe2d_create_solver``.
    _compat_logger :
        Optional logger instance.  Defaults to ``native_binding_compat.logger``.
    **kwargs :
        Keyword arguments forwarded to ``swe2d_create_solver``.

    Returns
    -------
    object
        The solver handle returned by ``swe2d_create_solver``.
    """
    log = _compat_logger or logger

    try:
        return mod.swe2d_create_solver(*args, **kwargs)
    except TypeError as exc:
        if "incompatible function arguments" not in str(exc):
            raise

    sig: Optional[inspect.Signature] = None
    try:
        sig = inspect.signature(mod.swe2d_create_solver)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        allowed = {
            name
            for name, param in sig.parameters.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        log.info(
            "[BACKEND] Compat fallback (sig-based): stripped %d of %d kwargs",
            len(kwargs) - len(filtered),
            len(kwargs),
        )
        return mod.swe2d_create_solver(*args, **filtered)

    # Conservative fallback for environments where signature introspection
    # on a pybind11 function is unavailable.
    filtered = dict(kwargs)
    for key in (
        "extreme_rain_mode",
        "source_cfl_beta",
        "source_max_substeps",
        "source_rate_cap",
        "source_depth_step_cap",
        "source_true_subcycling",
        "source_imex_split",
        "enable_shallow_front_recon_fallback",
        "tiny_mode",
        "tiny_cell_threshold",
        "tiny_edge_threshold",
        "tiny_wet_cell_threshold",
        "tiny_persistent_chunk_substeps",
        "tiny_active_compaction_stride_steps",
        "tiny_enable_active_compaction",
        "godunov_mode",
        "equation_set",
        "coupling_mode",
        "three_d_solver_model",
        "enforce_gpu_only_advanced_modes",
        "three_d_single_phase_free_surface",
        "dt_initial",
        "friction_substep_enabled",
        "friction_target_courant",
        "friction_max_substeps",
        "shallow_friction_correction",
        "shallow_friction_depth_alpha",
        "shallow_friction_exponent",
    ):
        filtered.pop(key, None)
    log.info(
        "[BACKEND] Compat fallback (conservative): stripped %d of %d kwargs",
        len(kwargs) - len(filtered),
        len(kwargs),
    )
    return mod.swe2d_create_solver(*args, **filtered)


def log_feature_unavailable(
    mod: Any,
    feature_name: str,
    _log: Optional[logging.Logger] = None,
) -> bool:
    """Check whether *feature_name* exists on *mod* and log a ``[BACKEND]``
    warning if it is absent.

    Parameters
    ----------
    mod :
        The loaded native extension module.
    feature_name : str
        The fully qualified attribute name to check (e.g.
        ``"swe2d_solver_set_boundary_values"``).
    _log :
        Optional logger instance.  Defaults to ``native_binding_compat.logger``.

    Returns
    -------
    bool
        ``True`` if the attribute exists, ``False`` otherwise.
    """
    log = _log or logger
    available = hasattr(mod, feature_name)
    if not available:
        log.info("[BACKEND] Feature %r unavailable in loaded binary", feature_name)
    return available
