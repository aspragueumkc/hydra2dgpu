"""
test_culvert_swmm_validation.py

Validates our native (CPU/GPU) culvert solver against the FHWA HEC-5
reference equations from EPA SWMM5's culvert.c.

Strategy
--------
1. Create two minimal GeoPackages — one with a **meters** CRS and one
   with a **feet** CRS — so the test exercises the full production
   pipeline: read CRS → configure units → geometry in CRS units →
   kernel converts to feet internally via ``model_to_ft``.
2. Use ``culvert_routine.py`` (direct Python port of SWMM culvert.c)
   as the ground-truth reference for inlet-controlled flow.
3. Exercise the native CPU path via ``swe2d_cpu_compute_structure_flows``
   with inputs that force inlet control to dominate (short barrel,
   negligible friction) so the ``min(inlet, outlet)`` output equals
   inlet control.
4. Compare inlet control flow between Python reference and native path
   across a matrix of (code, diameter, head) combinations **for both
   CRS types**.
"""

import math
import os
import tempfile
import unittest

import numpy as np
from osgeo import ogr, osr

from culvert_routine import (
    CircularXsect,
    RectangularXsect,
    inlet_controlled_flow,
)
from swe2d.units import USC_FT_PER_SI_M, USC_FT3_PER_SI_M3

# ── Physical constants (SI, used by reference equations) ──────────────────────
G_SI = 9.80665

# ── Physical culvert dimensions (always in SI meters) ────────────────────────
DIAMETER_M = 1.2
LENGTH_M = 1.0
SLOPE = 0.001  # dimensionless slope
ROUGHNESS_N = 0.001  # very smooth — ensures inlet control dominates


# ── Helper conversions ────────────────────────────────────────────────────────
def _m_to_ft(x_m: float) -> float:
    return x_m * USC_FT_PER_SI_M


def _ft_to_m(x_ft: float) -> float:
    return x_ft * (1.0 / USC_FT_PER_SI_M)


def _cfs_to_cms(q_cfs: float) -> float:
    return q_cfs / USC_FT3_PER_SI_M3


# ── GeoPackage factory ───────────────────────────────────────────────────────
def _create_culvert_gpkg(path: str, epsg: int) -> None:
    """Create a minimal GeoPackage with a culvert polygon in the given CRS."""
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.CreateDataSource(path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    layer = ds.CreateLayer("structures", srs, ogr.wkbPolygon)
    feat = ogr.Feature(layer.GetLayerDefn())
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]:
        ring.AddPoint(x, y)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    feat.SetGeometry(poly)
    layer.CreateFeature(feat)
    feat = None
    layer = None
    ds = None


def _crs_length_scale_si_to_model(gpkg_path: str) -> float:
    """Read CRS from a GeoPackage → return SI-meters-per-model-unit.
    1.0 for metric CRS, ~0.3048 for US-foot CRS.
    """
    ds = ogr.Open(gpkg_path)
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    ds = None
    if srs is None:
        return 1.0
    return max(1.0e-6, srs.GetLinearUnits())


# ── Reference: Python port of SWMM's culvert_getInflow() ─────────────────────
def reference_inlet_control_cfs(
    diameter_m: float,
    slope_mm: float,
    head_m: float,
    culvert_code: int = 1,
    shape: str = "circular",
) -> float:
    """Inlet-controlled flow [CFS] from the SWMM reference equations.

    Parameters are in SI (meters); function converts to feet internally
    for the FHWA HEC-5 equations.
    """
    diam_ft = _m_to_ft(diameter_m)
    head_ft = _m_to_ft(head_m)
    if shape in ("box", "rect", "rectangular"):
        xsect = RectangularXsect(width_ft=diam_ft, height_ft=diam_ft,
                                 culvert_code=culvert_code)
    else:
        xsect = CircularXsect(diameter_ft=diam_ft, culvert_code=culvert_code)
    q_cfs, _dqh, _cond, _yr = inlet_controlled_flow(
        xsect,
        max(1.0e-6, slope_mm),
        head_ft,
    )
    return q_cfs


# ── Test geometry table (physical dimensions in metres) ───────────────────────
CULVERT_GEOMETRIES = [
    (1.2, 0.001, 1, "circular", "Circular concrete, square edge w/headwall"),
    (1.2, 0.001, 2, "circular", "Circular concrete, groove end w/headwall"),
    (1.2, 0.001, 3, "circular", "Circular concrete, groove end projecting"),
    (1.2, 0.001, 4, "circular", "Circular CMP, headwall"),
    (1.2, 0.001, 5, "circular", "Circular CMP, mitered to slope"),
    (0.9, 0.002, 1, "circular", "Smaller circular concrete"),
    (1.8, 0.0005, 1, "circular", "Larger circular concrete"),
    (1.2, 0.005, 1, "circular", "Steeper circular concrete"),
    (1.2, 0.001, 20, "rectangular", "Rectangular box, 30-75 deg wingwalls"),
    (1.2, 0.001, 25, "rectangular", "Rectangular box, 90 deg headwall"),
]

HEAD_VALUES_M = [0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0]


# ═════════════════════════════════════════════════════════════════════════════
# Reference self-consistency tests
# ═════════════════════════════════════════════════════════════════════════════
class TestCulvertInletControlReference(unittest.TestCase):
    """Verify the Python reference itself is self-consistent."""

    maxDiff = None

    def test_reference_basic_sanity(self):
        q_prev = 0.0
        for h in HEAD_VALUES_M:
            q = reference_inlet_control_cfs(DIAMETER_M, SLOPE, h, 1)
            self.assertGreaterEqual(q, q_prev,
                                    f"Flow should not decrease: h={h:.2f}m q={q:.6f}")
            q_prev = q

    def test_reference_code_variation(self):
        q1 = reference_inlet_control_cfs(DIAMETER_M, SLOPE, 1.5, 1)
        q4 = reference_inlet_control_cfs(DIAMETER_M, SLOPE, 1.5, 4)
        self.assertNotAlmostEqual(q1, q4, delta=0.5,
                                  msg="Code 1 and code 4 should differ")

    def test_reference_smaller_diameter_less_flow(self):
        q_big = reference_inlet_control_cfs(1.8, SLOPE, 1.5, 1)
        q_small = reference_inlet_control_cfs(0.9, SLOPE, 1.5, 1)
        self.assertGreater(q_big, q_small)


# ═════════════════════════════════════════════════════════════════════════════
# Native CPU vs SWMM reference  (parameterised over CRS)
# ═════════════════════════════════════════════════════════════════════════════
class TestCulvertNativeVsReference(unittest.TestCase):
    """Compare native CPU culvert flow against the SWMM reference.

    Two GeoPackages (meters + feet CRS) are created so the test exercises
    the full production path: CRS → configure units → geometry in CRS
    units → kernel converts to feet via model_to_ft.
    """

    _gpkg_m = None
    _gpkg_ft = None

    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import load_swe2d_native_module
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )

        try:
            cls.native_mod = load_swe2d_native_module(openmp_enabled=True)
        except Exception as e:
            raise unittest.SkipTest(f"Native module unavailable: {e}")
        if not hasattr(cls.native_mod, "swe2d_cpu_compute_structure_flows"):
            raise unittest.SkipTest(
                "swe2d_cpu_compute_structure_flows not found in native module"
            )

        cls._HydraulicStructure = HydraulicStructure
        cls._HydraulicStructureConfig = HydraulicStructureConfig
        cls._StructureType = StructureType

        cls._tmpdir = tempfile.mkdtemp(prefix="culvert_test_")
        cls._gpkg_m = os.path.join(cls._tmpdir, "culvert_meters.gpkg")
        cls._gpkg_ft = os.path.join(cls._tmpdir, "culvert_feet.gpkg")
        _create_culvert_gpkg(cls._gpkg_m, epsg=32633)  # UTM 33N (meters)
        _create_culvert_gpkg(cls._gpkg_ft, epsg=2229)   # CA zone 3 (US feet)

        cls._ls_m = _crs_length_scale_si_to_model(cls._gpkg_m)
        cls._ls_ft = _crs_length_scale_si_to_model(cls._gpkg_ft)
        assert abs(cls._ls_m - 1.0) < 0.01, f"Expected ~1.0 for meters, got {cls._ls_m}"
        assert abs(cls._ls_ft - 0.3048) < 0.01, f"Expected ~0.3048 for feet, got {cls._ls_ft}"

    def _native_flow_cfs(self, gpkg_path, diam_cr_units, head_cr_units,
                         code, shape="circular"):
        """Call the native CPU culvert solver and return flow in CFS.

        All geometry and WSE are passed **in CRS units** — exactly as
        the production code path does.  The kernel converts to feet
        internally via model_to_ft.
        """
        import swe2d.units as _u
        from swe2d.runtime.coupling import pack_structures_soa

        ls = _crs_length_scale_si_to_model(gpkg_path)
        _u.configure(ls)
        model_to_ft = _u.model_to_ft()
        gravity = _u.gravity()

        barrel_length = LENGTH_M if ls > 0.5 else _m_to_ft(LENGTH_M)

        st = self._HydraulicStructure(
            structure_id="TEST",
            structure_type=self._StructureType.CULVERT,
            upstream_cell=0, downstream_cell=1,
            crest_elev=0.0, enabled=True,
            metadata={
                "diameter": diam_cr_units,
                "length": barrel_length,
                "culvert_slope": SLOPE,
                "roughness_n": ROUGHNESS_N,
                "culvert_shape": "circular" if shape != "rectangular" else "rect",
                "culvert_code": code,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": barrel_length * SLOPE,
                "entrance_loss_k": 0.1,
                "exit_loss_k": 0.1,
                "culvert_barrels": 1.0,
            },
        )
        cfg = self._HydraulicStructureConfig(enabled=True, structures=[st])
        ssoa = pack_structures_soa(cfg, 2, model_to_ft=1.0)
        self.assertIsNotNone(ssoa, "pack_structures_soa returned None")

        cell_wse = np.array([head_cr_units, 0.1 * head_cr_units], dtype=np.float64)
        cell_bed = np.array([0.0, 0.0], dtype=np.float64)

        result = self.native_mod.swe2d_cpu_compute_structure_flows(
            cell_wse, cell_bed,
            ssoa.structure_type, ssoa.upstream_cell, ssoa.downstream_cell,
            ssoa.crest_elev, ssoa.width, ssoa.height, ssoa.diameter,
            ssoa.length, ssoa.roughness_n, ssoa.coeff, ssoa.cd, ssoa.opening,
            ssoa.q_pump, ssoa.max_flow,
            ssoa.culvert_code, ssoa.culvert_shape, ssoa.culvert_rise,
            ssoa.culvert_span, ssoa.culvert_area, ssoa.culvert_barrels,
            ssoa.culvert_slope, ssoa.inlet_invert_elev, ssoa.outlet_invert_elev,
            ssoa.entrance_loss_k, ssoa.exit_loss_k,
            ssoa.embankment_enabled, ssoa.embankment_crest_elev,
            ssoa.embankment_overflow_width, ssoa.embankment_weir_coeff,
            gravity,
            model_to_ft,
        )
        self.assertIsNotNone(result and result.size > 0, "Native returned empty")
        native_model_flow = float(result[0])

        # meters CRS → m³/s; feet CRS → ft³/s = CFS
        if ls > 0.5:
            return native_model_flow * USC_FT3_PER_SI_M3
        else:
            return native_model_flow

    def _check_matches_reference(self, gpkg_path, crs_label, diam_m, slope,
                                 code, head_m, shape, tol):
        """Compare native (CRS units) vs reference (SI meters) for one case."""
        ls = _crs_length_scale_si_to_model(gpkg_path)
        diam_cr = diam_m if ls > 0.5 else _m_to_ft(diam_m)
        head_cr = head_m if ls > 0.5 else _m_to_ft(head_m)

        q_native = self._native_flow_cfs(gpkg_path, diam_cr, head_cr, code, shape)
        q_ref = reference_inlet_control_cfs(diam_m, slope, head_m, code, shape)

        if q_ref < 1.0e-6 and q_native < 1.0e-6:
            return
        rel_err = abs(q_native - q_ref) / max(1.0e-9, q_ref)
        self.assertLess(
            rel_err, tol,
            f"[{crs_label}] code={code} D={diam_m}m h={head_m}m: "
            f"native={q_native:.4f} cfs, ref={q_ref:.4f} cfs, "
            f"rel_err={rel_err:.2%}"
        )

    # --- Code 1 (circular concrete) ---
    def test_code1_meters(self):
        for h in HEAD_VALUES_M:
            self._check_matches_reference(
                self._gpkg_m, "meters", DIAMETER_M, SLOPE, 1, h, "circular", 0.25)

    def test_code1_feet(self):
        for h in HEAD_VALUES_M:
            self._check_matches_reference(
                self._gpkg_ft, "feet", DIAMETER_M, SLOPE, 1, h, "circular", 0.25)

    # --- Code 4 (CMP) ---
    def test_code4_meters(self):
        for h in HEAD_VALUES_M:
            self._check_matches_reference(
                self._gpkg_m, "meters", DIAMETER_M, SLOPE, 4, h, "circular", 0.25)

    def test_code4_feet(self):
        for h in HEAD_VALUES_M:
            self._check_matches_reference(
                self._gpkg_ft, "feet", DIAMETER_M, SLOPE, 4, h, "circular", 0.25)

    # --- Spot-check across multiple codes and both CRS ---
    def test_across_codes_meters(self):
        for diam_m, slope, code, shape, desc in CULVERT_GEOMETRIES:
            for h in [0.9, 1.5, 3.0]:
                with self.subTest(desc=desc, h=h, crs="meters"):
                    self._check_matches_reference(
                        self._gpkg_m, "meters", diam_m, slope, code, h, shape, 0.30)

    def test_across_codes_feet(self):
        for diam_m, slope, code, shape, desc in CULVERT_GEOMETRIES:
            for h in [0.9, 1.5, 3.0]:
                with self.subTest(desc=desc, h=h, crs="feet"):
                    self._check_matches_reference(
                        self._gpkg_ft, "feet", diam_m, slope, code, h, shape, 0.30)


# ═════════════════════════════════════════════════════════════════════════════
# Python structure_details vs reference
# ═════════════════════════════════════════════════════════════════════════════
class TestCulvertPythonDetailsVsReference(unittest.TestCase):
    """Compare SWE2DStructureModule.structure_details() against the
    direct SWMM reference, verifying the Python plumbing works correctly.
    """

    def _python_structure_detail_flow(self, diam_m, slope, code, head_m,
                                      shape="circular",
                                      model_to_ft=USC_FT_PER_SI_M):
        from swe2d.extensions.extension_models import (
            HydraulicStructure,
            HydraulicStructureConfig,
            StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule

        inlet_invert = 0.0
        length_m = LENGTH_M
        outlet_invert = length_m * slope

        st = HydraulicStructure(
            structure_id="TEST",
            structure_type=StructureType.CULVERT,
            upstream_cell=0, downstream_cell=1,
            crest_elev=0.0, enabled=True,
            metadata={
                "diameter": diam_m,
                "length": length_m,
                "culvert_slope": slope,
                "roughness_n": ROUGHNESS_N,
                "culvert_shape": "circular" if shape != "rectangular" else "rect",
                "culvert_code": code,
                "inlet_invert_elev": inlet_invert,
                "outlet_invert_elev": outlet_invert,
                "entrance_loss_k": 0.1,
                "exit_loss_k": 0.1,
                "culvert_barrels": 1.0,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[st])
        smod = SWE2DStructureModule(cfg, model_to_ft=model_to_ft)

        cell_wse = np.array([head_m + inlet_invert, 0.1 * head_m], dtype=np.float64)
        details = smod.structure_details(cell_wse)
        d = details[0]
        return float(d.get("inlet_control_flow", 0.0))

    def test_python_details_inlet_control_si_model(self):
        """Python structure_details inlet control matches reference for SI model.

        This test uses model_to_ft=USC_FT_PER_SI_M (3.28) so the 1.2 m
        diameter is correctly converted to ~3.937 ft for the FHWA equations.
        """
        model_to_ft = USC_FT_PER_SI_M
        for h in HEAD_VALUES_M:
            q_ref = reference_inlet_control_cfs(1.2, 0.001, h, 1)
            q_py = self._python_structure_detail_flow(1.2, 0.001, 1, h, model_to_ft=model_to_ft)
            if q_ref < 1.0e-6 and q_py < 1.0e-6:
                continue
            q_ref_cms = _cfs_to_cms(q_ref)
            rel_err = abs(q_py - q_ref_cms) / max(1.0e-9, q_ref_cms)
            self.assertLess(
                rel_err, 0.10,
                f"Python details, h={h:.2f}m: py={q_py:.6f} cms, "
                f"ref={q_ref_cms:.6f} cms, rel_err={rel_err:.2%}"
            )


if __name__ == "__main__":
    unittest.main()
