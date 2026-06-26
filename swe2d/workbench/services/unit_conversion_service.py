"""Unit conversion service for the SWE2D workbench.

Extracted from ``SWE2DWorkbenchStudioDialog`` — pure Python with zero Qt
imports.  All functions accept explicit parameters instead of reaching
into ``self``.

NO SILENT FALLBACKS:
    * ``update_unit_system_from_crs`` raises on unexpected errors.
    * ``detect_map_unit`` returns ``None`` only when QGIS is unavailable
      or the project CRS is invalid (documented, not a silent fallback).
"""
from __future__ import annotations

from typing import Callable, Optional

__all__ = [
    "update_unit_system_from_crs",
    "length_scale_si_to_model",
    "rain_mm_to_model_depth",
    "rain_rate_si_to_model",
    "flow_si_to_model",
    "detect_map_unit",
    "is_us_customary_units",
]


def detect_map_unit(
    *,
    have_qgis_core: bool,
    project=None,
    log_fn: Callable[[str], None] = lambda _: None,
) -> Optional[int]:
    """Return the CRS map-unit enum value, or *None* if unavailable.

    Parameters
    ----------
    have_qgis_core : bool
        Whether ``qgis.core`` was imported successfully.
    project : QgsProject or None
        The current QGIS project instance.
    log_fn : callable
        Logger callback (signature ``log_fn(message)``).
    """
    if not have_qgis_core or project is None:
        return None
    try:
        from qgis.core import QgsUnitTypes  # local import — service has no top-level Qt

        crs = project.crs()
        if crs is None or not crs.isValid() or QgsUnitTypes is None:
            return None
        return crs.mapUnits()
    except Exception as e:
        log_fn(f"[ERROR] detect map unit failed: {e}")
        return None


def length_scale_si_to_model() -> float:
    """Return the factor that converts one SI metre to one model-length unit."""
    from swe2d import units as _u

    return _u.model_per_si_m()


def rain_mm_to_model_depth() -> float:
    """Convert 1 mm of rain to model depth units."""
    return 1.0e-3 * length_scale_si_to_model()


def rain_rate_si_to_model(rain_rate_mps: float) -> float:
    """Convert a rain rate in m/s to model units."""
    from swe2d import units as _u

    return _u.rain_si_to_model(rain_rate_mps)


def flow_si_to_model(flow_cms: float) -> float:
    """Convert a flow in m³/s to model units."""
    from swe2d import units as _u

    return _u.flow_si_to_model(flow_cms)


def is_us_customary_units(length_unit_name: str) -> bool:
    """Return *True* when *length_unit_name* is ``'ft'`` (US Customary)."""
    return str(length_unit_name or "m").strip().lower() == "ft"


def update_unit_system_from_crs(
    *,
    have_qgis_core: bool,
    project=None,
    log_fn: Callable[[str], None] = lambda _: None,
) -> dict:
    """Detect CRS-based unit system and return the configuration dict.

    The caller is responsible for applying the returned values to its
    own state; this function has **no** side-effects on the dialog.

    Returns
    -------
    dict with keys:
        ``unit_name``   – ``'m'``, ``'ft'``, or CRS-derived label
        ``sys_name``    – ``'SI'``, ``'US Customary'``, or fallback label
        ``scale``       – SI-to-model length scale (float)
        ``gravity``     – gravitational acceleration in model units (float)
        ``k_mann``      – Manning factor (float)
        ``crs_desc``    – CRS description string (authid + description)
    """
    from swe2d import units as _u

    crs_desc = "(no CRS)"
    unit = detect_map_unit(
        have_qgis_core=have_qgis_core, project=project, log_fn=log_fn
    )
    unit_name = "m"
    sys_name = "SI"
    scale = 1.0
    if have_qgis_core and project is not None:
        try:
            crs = project.crs()
            if crs is not None and crs.isValid():
                crs_desc = f"{crs.authid()} {crs.description()}".strip()
        except Exception as _e:

            try:

                self._log(f"[ERROR] Exception in unit_conversion_service.py: {_e}")

            except Exception:

                pass
    if have_qgis_core and unit is not None:
        try:
            from qgis.core import QgsUnitTypes  # local import

            feet_candidates = {
                getattr(QgsUnitTypes, "DistanceFeet", None),
                getattr(QgsUnitTypes, "DistanceUSSurveyFeet", None),
            }
            unit_text = ""
            if hasattr(QgsUnitTypes, "toString"):
                unit_text = str(QgsUnitTypes.toString(unit) or "").strip().lower()
            is_feet_like_text = (
                "feet" in unit_text or "foot" in unit_text or "ft" in unit_text
            )
            if unit in feet_candidates or is_feet_like_text:
                unit_name = "ft"
                sys_name = "US Customary"
                scale = 0.3048
            elif unit == getattr(QgsUnitTypes, "DistanceMeters", None):
                unit_name = "m"
                sys_name = "SI"
                scale = 1.0
            else:
                unit_name = (
                    str(QgsUnitTypes.toString(unit))
                    if hasattr(QgsUnitTypes, "toString")
                    else "m"
                )
                sys_name = "SI (fallback)"
                scale = 1.0
        except Exception as e:
            log_fn(f"[ERROR] update unit system from crs failed: {e}")
    _u.configure(scale)
    return {
        "unit_name": unit_name,
        "sys_name": sys_name,
        "scale": scale,
        "gravity": _u.gravity(),
        "k_mann": _u.manning_factor(),
        "crs_desc": crs_desc,
    }
