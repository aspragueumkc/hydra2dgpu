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


def configure(length_scale_si_to_model: float) -> None:
    """
    Call once at startup with the CRS-derived length scale.
    
    Args:
        length_scale_si_to_model: How many model-length units per SI meter.
            1.0 for metric CRS, 3.28084 for US-foot CRS.
    """
    global _si_m_per_model, _model_per_si_m, _si_m2_per_model_area, _si_m3_per_model_volume
    _si_m_per_model = float(length_scale_si_to_model)
    _model_per_si_m = 1.0 / _si_m_per_model
    _si_m2_per_model_area = _si_m_per_model ** 2
    _si_m3_per_model_volume = _si_m_per_model ** 3


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


def compute_length_factor() -> float:
    """
    Factor to convert model metadata lengths → computation units.
    
    Computation units are:
      - SI model: feet (so kernel uses USC constant set)
      - USC model: feet (model is already feet)
    
    Always returns USC_FT_PER_SI_M / si_m_per_model().
    For SI model: 3.28 / 1.0 = 3.28  (m → ft)
    For USC model: 3.28 / 3.28 = 1.0  (ft stays ft)
    """
    return USC_FT_PER_SI_M / si_m_per_model()
