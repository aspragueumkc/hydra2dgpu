"""
Unit system constants derived from the project CRS.
All conversions flow from a single LENGTH_SCALE (SI meters → model units).

Usage:
    from swe2d.units import configure, USC_FT_PER_SI_M, USC_FT3_PER_SI_M3
    configure(crs_length_scale)  # call once at startup

Dimensional notation in comments:
    L   = length
    L²  = area
    L³  = volume
    T   = time
    L/T = velocity
    L³/T = volumetric flow
"""

# ── Imperial/USC conversion constants (SI → US customary) ──
USC_FT_PER_SI_M: float = 3.280839895013123
SI_M_PER_USC_FT: float = 0.3048
SI_M2_PER_USC_FT2: float = 0.09290304
SI_M3_PER_USC_FT3: float = 0.028316846592
USC_FT3_PER_SI_M3: float = 35.31466672148859  # CFS per CMS

# ── Gravity ──
SI_GRAVITY: float = 9.80665  # m/s²  (L/T²)
USC_GRAVITY: float = 32.17404855643045  # ft/s²  (L/T²)

# ── Manning ──
SI_MANNING_FACTOR: float = 1.0
USC_MANNING_FACTOR: float = 1.486  # 1.486/n^(1/2) for USC units

# ── CRS-derived (set by configure()) ──
_si_m_per_model: float = 1.0
_model_per_si_m: float = 1.0
_si_m2_per_model_area: float = 1.0
_si_m3_per_model_volume: float = 1.0
_model_to_ft: float = USC_FT_PER_SI_M  # model units → ft (for HDS-5 culverts only)
_gravity: float = SI_GRAVITY           # L/T² in model units
_manning: float = SI_MANNING_FACTOR    # Manning multiplier in model units
_length_unit_name: str = "m"           # human-readable unit ("m" | "ft")


def configure(length_scale_si_to_model: float) -> None:
    """
    Configure CRS-derived unit conversion from the project length scale.

    Must be called once at startup before any other swe2d code runs.

    Parameters
    ----------
    length_scale_si_to_model : float
        How many SI meters per model unit.
        1.0 for metric CRS (m), 0.3048 for US-foot CRS (ft).
    """
    global _si_m_per_model, _model_per_si_m, _si_m2_per_model_area, _si_m3_per_model_volume
    global _model_to_ft, _gravity, _manning, _length_unit_name
    _si_m_per_model = float(length_scale_si_to_model)
    _model_per_si_m = 1.0 / _si_m_per_model
    _si_m2_per_model_area = _si_m_per_model ** 2
    _si_m3_per_model_volume = _si_m_per_model ** 3
    # Model-unit gravity: g_SI / si_m_per_model → ft/s² for USC, m/s² for SI
    _gravity = SI_GRAVITY * _model_per_si_m
    # Culvert-only: factor to convert model lengths → ft for HDS-5 tables.
    # For SI (1 m/model): _si_m_per_model * USC_FT_PER_SI_M = 1.0 * 3.28084 ft/m
    # For USC ft (0.3048 m/ft): 0.3048 * 3.28084 = 1.0 ft/ft
    _model_to_ft = _si_m_per_model * USC_FT_PER_SI_M
    # Manning multiplier: 1.0 for SI, 1.486 for USC
    _manning = USC_MANNING_FACTOR if _model_to_ft < 2.0 else SI_MANNING_FACTOR
    # Human-readable unit name: ft for USC, m for SI
    _length_unit_name = "ft" if _model_to_ft < 2.0 else "m"


def si_m_per_model() -> float:
    """SI meters per model-length unit. 1.0 for metric, 0.3048 for US-foot."""
    return _si_m_per_model


def model_per_si_m() -> float:
    """Model-length units per SI meter. 1.0 for metric, 3.28084 for US-foot."""
    return _model_per_si_m


def si_m2_per_model_area() -> float:
    """SI m² per model-length²."""
    return _si_m2_per_model_area


def si_m3_per_model_volume() -> float:
    """SI m³ per model-length³."""
    return _si_m3_per_model_volume


def gravity() -> float:
    """Gravity in model units (L/T²). 9.81 for metric, 32.17 for US-foot."""
    return _gravity


def model_to_ft() -> float:
    """
    Factor to convert model lengths → feet for HDS-5 culvert tables.
    3.28 for SI model (m→ft), 1.0 for US-foot model (ft stays ft).
    """
    return _model_to_ft


def manning_factor() -> float:
    """Manning multiplier for model units. 1.0 for metric, 1.486 for US-foot."""
    return _manning


def length_unit_name() -> str:
    """Human-readable model length unit. 'm' for SI, 'ft' for US-foot."""
    return _length_unit_name


def flow_si_to_model(flow_cms: float) -> float:
    """Convert volumetric flow from SI (m³/s) to model units (L³/T).

    Uses the CRS-derived length scale: 1.0 for SI (returns m³/s),
    ~35.315 for USC (returns ft³/s).
    """
    return flow_cms * (_model_per_si_m ** 3)


def rain_si_to_model(rain_rate_mps: float) -> float:
    """Convert rain rate from SI (m/s) to model depth units (L/T).

    Uses the CRS-derived length scale: 1.0 for SI (returns m/s),
    ~3.281 for USC (returns ft/s).
    """
    return rain_rate_mps * _model_per_si_m


def si_m_per_model_from_wkt(wkt: str) -> float:
    """Extract SI meters per model unit from a projected CRS WKT.

    Handles both WKT2 (``LENGTHUNIT``) and WKT1 (``UNIT``) formats.
    Returns 1.0 if no linear unit is found (e.g. geographic CRS).
    """
    import re
    try:
        # WKT2 format: CS[...LENGTHUNIT["name", factor]...]
        lu_match = re.search(r"LENGTHUNIT\[\"[^\"]*\",\s*([0-9.eE+-]+)", wkt)
        if lu_match:
            return float(lu_match.group(1))

        # WKT1 format: PROJCS["...", ... UNIT["name", factor, ...] ...]
        pj = wkt.find("PROJCS[")
        if pj < 0:
            pj = wkt.find("PROJCRS[")
        if pj >= 0:
            # The last UNIT entry in a projected CRS is the linear unit.
            # Parse backward through the WKT to find it.
            tail = wkt[pj:]
            units = list(re.finditer(r"UNIT\[\"[^\"]*\",\s*([0-9.eE+-]+)", tail))
            if units:
                # Skip angular units ("degree", "radian", "grad")
                for m in reversed(units):
                    unit_name_start = m.start() + len("UNIT[\"")
                    unit_name_end = tail.find("\"", unit_name_start)
                    unit_name = tail[unit_name_start:unit_name_end].lower()
                    if unit_name not in ("degree", "radian", "grad"):
                        return float(m.group(1))

        # If we reach here, no projected CRS with linear unit was found
        import logging
        logging.getLogger(__name__).warning(
            "No linear unit found in CRS WKT — assuming model units are SI meters. "
            "Geographic CRS or unsupported WKT format?"
        )
    except Exception:
        pass
    return 1.0


