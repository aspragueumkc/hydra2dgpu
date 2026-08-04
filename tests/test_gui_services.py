"""Behavioral tests for seven zero-coverage workbench services (Task B.1).

Spec: docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4.
Plan: docs/plans/2026-08-02-gui-test-coverage.md Task B.1.

Pattern P1 (pure service round-trip) throughout: real QgsVectorLayer /
QgsRasterLayer / QgsProject objects, real GeoPackages on disk, real HDF5
files.  No MagicMock substitutes for Qgs* types.  Mutations are verified by
reading back through the production read path.

Covers:
    swe2d/workbench/services/hecras_export_service.py
    swe2d/workbench/services/unit_conversion_service.py
    swe2d/workbench/services/non_gui_qgis_service.py
    swe2d/workbench/services/pipe_network_config_service.py
    swe2d/workbench/services/gpkg_layer_styles_service.py
    swe2d/workbench/services/topology_template_service.py
    swe2d/workbench/services/model_gpkg_loader_service.py
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import types
import unittest

import numpy as np

from tests.qgis_real_env import (
    ensure_qgis_app,
    make_memory_layer,
    make_temp_model_gpkg,
    requires_qgis,
)


# ===========================================================================
# 1. hecras_export_service.write_hecras_hdf5
# ===========================================================================


def _tri_mesh_data():
    """Small triangular mesh: 4 nodes, 2 cells, spatially varying bed."""
    return {
        "node_x": np.array([0.0, 10.0, 0.0, 10.0]),
        "node_y": np.array([0.0, 0.0, 10.0, 10.0]),
        "node_z": np.array([1.0, 2.0, 3.0, 4.0]),
        "cell_nodes": np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
    }


def _mixed_mesh_data():
    """CSR mesh: one quad + one triangle (cell_face_offsets path)."""
    return {
        "node_x": np.array([0.0, 10.0, 10.0, 0.0, 20.0]),
        "node_y": np.array([0.0, 0.0, 10.0, 10.0, 0.0]),
        "node_z": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "cell_face_offsets": np.array([0, 4, 7], dtype=np.int32),
        "cell_face_nodes": np.array([0, 1, 2, 3, 3, 2, 4], dtype=np.int32),
    }


_TS_BASE = (
    "Results/Unsteady/Output/Output Blocks/"
    "Base Output/Unsteady Time Series"
)


@unittest.skipUnless(
    importlib.util.find_spec("h5py") is not None, "h5py not installed"
)
class TestHecrasExportService(unittest.TestCase):
    """write_hecras_hdf5: HDF5 structure asserted per the writer's schema."""

    def setUp(self):
        import h5py  # noqa: F401 — loud skip above guarantees availability

        from swe2d.workbench.services.hecras_export_service import (
            write_hecras_hdf5,
        )

        self._write = write_hecras_hdf5
        self._tmpdir = tempfile.TemporaryDirectory(prefix="hydra_hecras_")
        self.addCleanup(self._tmpdir.cleanup)
        self._out = os.path.join(self._tmpdir.name, "out.h5")

    def _open(self):
        import h5py

        return h5py.File(self._out, "r")

    def test_mesh_data_none_raises(self):
        with self.assertRaises(RuntimeError):
            self._write(self._out, None)

    def test_geometry_only_triangular_mesh(self):
        mesh = _tri_mesh_data()
        self._write(
            self._out,
            mesh,
            timesteps=None,
            projection_wkt='PROJCS["Test CRS"]',
        )
        with self._open() as f:
            self.assertEqual(f.attrs["File Type"], b"HEC-RAS Results")
            self.assertEqual(f.attrs["Units System"], b"SI")
            self.assertEqual(f.attrs["Projection"], b'PROJCS["Test CRS"]')

            geo = f["Geometry"]
            self.assertEqual(geo.attrs["SI Units"], b"True")
            area = geo["2D Flow Areas/Perimeter 1"]

            fp = area["FacePoints Coordinate"][:]
            np.testing.assert_allclose(
                fp, np.column_stack([mesh["node_x"], mesh["node_y"]])
            )

            idx = area["Cells FacePoint Indexes"][:]
            np.testing.assert_array_equal(idx, mesh["cell_nodes"])

            centers = area["Cells Center Coordinate"][:]
            np.testing.assert_allclose(
                centers[0], [np.mean([0.0, 10.0, 0.0]), np.mean([0.0, 0.0, 10.0])]
            )
            np.testing.assert_allclose(
                centers[1],
                [np.mean([10.0, 10.0, 0.0]), np.mean([0.0, 10.0, 10.0])],
            )

            elev = area["Cells Minimum Elevation"][:]
            np.testing.assert_allclose(elev[0], np.mean([1.0, 2.0, 3.0]))
            np.testing.assert_allclose(elev[1], np.mean([2.0, 4.0, 3.0]))

            attrs = f["Geometry/2D Flow Areas/Attributes"]
            self.assertEqual(int(attrs["Cell Count"][0]), 2)

            # No timesteps → no Results tree.
            self.assertNotIn("Results", f)

    def test_mixed_cell_types_csr_path_pads_with_minus_one(self):
        self._write(self._out, _mixed_mesh_data(), timesteps=None)
        with self._open() as f:
            idx = f["Geometry/2D Flow Areas/Perimeter 1/Cells FacePoint Indexes"][:]
            self.assertEqual(idx.shape, (2, 4))
            np.testing.assert_array_equal(idx[0], [0, 1, 2, 3])
            np.testing.assert_array_equal(idx[1], [3, 2, 4, -1])

    def test_timestep_results_values(self):
        mesh = _tri_mesh_data()
        cell_z = np.array(
            [np.mean([1.0, 2.0, 3.0]), np.mean([2.0, 4.0, 3.0])]
        )
        h = np.array([1.0, 0.0])   # cell 1 dry
        hu = np.array([0.5, 1.0])
        hv = np.array([0.0, 0.0])
        timesteps = [(0.0, h, hu, hv), (3600.0, h, hu, hv)]
        self._write(self._out, mesh, timesteps=timesteps, n_mann=0.045)
        with self._open() as f:
            ts = f[_TS_BASE]
            np.testing.assert_allclose(ts["Time"][:], [0.0, 1.0])
            self.assertEqual(
                int(ts["Time"].attrs["Number of actual Time Steps"][0]), 2
            )
            self.assertEqual(ts["Time Date Stamp"].shape, (2,))

            area = ts[f"2D Flow Areas/Perimeter 1"]
            np.testing.assert_allclose(area["Depth"][0], h, rtol=1e-6)
            np.testing.assert_allclose(
                area["Water Surface"][0], h + cell_z, rtol=1e-6
            )
            # Wet cell: u = hu/h = 0.5; dry cell: zeroed.
            np.testing.assert_allclose(
                area["Cell Velocity - Magnitude"][0], [0.5, 0.0], rtol=1e-6
            )
            np.testing.assert_allclose(
                area["Cell Velocity - X"][0], [0.5, 0.0], rtol=1e-6
            )
            # include_extra defaults True → extended outputs present.
            np.testing.assert_allclose(
                area["Wet Mask"][0], [1.0, 0.0], rtol=1e-6
            )
            np.testing.assert_allclose(
                area["Cell Momentum - X"][0], hu, rtol=1e-6
            )
            self.assertIn("Cell Froude Number", area)
            # Manning fallback when result_data has no n_mann_cell.
            np.testing.assert_allclose(
                f["Geometry/2D Flow Areas/Perimeter 1/Cells Manning n"][:],
                [0.045, 0.045],
                rtol=1e-6,
            )
            # MDAL contract: Summary Output group exists.
            self.assertIn(
                "Results/Unsteady/Output/Output Blocks/"
                "Base Output/Summary Output/2D Flow Areas/Perimeter 1",
                f,
            )

    def test_result_data_n_mann_cell_overrides_fallback(self):
        self._write(
            self._out,
            _tri_mesh_data(),
            timesteps=None,
            result_data={"n_mann_cell": np.array([0.01, 0.09])},
        )
        with self._open() as f:
            np.testing.assert_allclose(
                f["Geometry/2D Flow Areas/Perimeter 1/Cells Manning n"][:],
                [0.01, 0.09],
                rtol=1e-6,
            )

    def test_us_customary_and_include_extra_false(self):
        self._write(
            self._out,
            _tri_mesh_data(),
            is_us_customary=True,
            include_extra=False,
            timesteps=[(0.0, np.array([1.0, 1.0]),
                        np.array([0.0, 0.0]), np.array([0.0, 0.0]))],
        )
        with self._open() as f:
            self.assertEqual(f.attrs["Units System"], b"US Customary")
            self.assertEqual(f["Geometry"].attrs["SI Units"], b"False")
            area = f[f"{_TS_BASE}/2D Flow Areas/Perimeter 1"]
            self.assertNotIn("Wet Mask", area)
            self.assertNotIn("Cell Froude Number", area)
            self.assertNotIn(
                "Cells Manning n", f["Geometry/2D Flow Areas/Perimeter 1"]
            )


# ===========================================================================
# 2. unit_conversion_service (all 7 public functions)
# ===========================================================================


@requires_qgis
class TestUnitConversionService(unittest.TestCase):
    """SI/USC matrix cross-checked against the swe2d.units ground truth."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from swe2d import units as _u

        self._u = _u
        self._prev_scale = _u.si_m_per_model()

    def tearDown(self):
        self._u.configure(self._prev_scale)

    @staticmethod
    def _project(authid):
        from qgis.core import QgsCoordinateReferenceSystem, QgsProject

        proj = QgsProject()
        crs = QgsCoordinateReferenceSystem(authid)
        if not crs.isValid():
            raise RuntimeError(f"test CRS {authid} is not valid in this env")
        proj.setCrs(crs)
        return proj

    # ---- detect_map_unit ----

    def test_detect_map_unit_meters_crs(self):
        from qgis.core import QgsUnitTypes

        from swe2d.workbench.services.unit_conversion_service import (
            detect_map_unit,
        )

        unit = detect_map_unit(
            have_qgis_core=True, project=self._project("EPSG:32616")
        )
        self.assertEqual(unit, QgsUnitTypes.DistanceMeters)

    def test_detect_map_unit_feet_crs(self):
        from qgis.core import QgsUnitTypes

        from swe2d.workbench.services.unit_conversion_service import (
            detect_map_unit,
        )

        unit = detect_map_unit(
            have_qgis_core=True, project=self._project("EPSG:2277")
        )
        # QGIS exposes the US-survey-foot enum without a named attr on
        # QgsUnitTypes in this version; match production's own feet-like
        # detection (enum candidate + toString text).
        feet_candidates = {
            getattr(QgsUnitTypes, "DistanceFeet", None),
            getattr(QgsUnitTypes, "DistanceUSSurveyFeet", None),
        }
        unit_text = QgsUnitTypes.toString(unit).lower()
        self.assertTrue(
            unit in feet_candidates
            or "feet" in unit_text
            or "foot" in unit_text,
            f"EPSG:2277 mapUnits not feet-like: {unit} ({unit_text!r})",
        )

    def test_detect_map_unit_geographic_crs_returns_degrees(self):
        from qgis.core import QgsUnitTypes

        from swe2d.workbench.services.unit_conversion_service import (
            detect_map_unit,
        )

        unit = detect_map_unit(
            have_qgis_core=True, project=self._project("EPSG:4326")
        )
        self.assertEqual(unit, QgsUnitTypes.DistanceDegrees)

    def test_detect_map_unit_unavailable_returns_none(self):
        from swe2d.workbench.services.unit_conversion_service import (
            detect_map_unit,
        )

        self.assertIsNone(detect_map_unit(have_qgis_core=False))
        self.assertIsNone(detect_map_unit(have_qgis_core=True, project=None))

    # ---- conversion functions vs swe2d.units ground truth ----

    def test_si_matrix(self):
        from swe2d.workbench.services import unit_conversion_service as svc

        self._u.configure(1.0)
        self.assertAlmostEqual(svc.length_scale_si_to_model(), 1.0)
        self.assertAlmostEqual(
            svc.length_scale_si_to_model(), self._u.model_per_si_m()
        )
        self.assertAlmostEqual(svc.rain_mm_to_model_depth(), 1.0e-3)
        self.assertAlmostEqual(svc.rain_rate_si_to_model(2.5e-6), 2.5e-6)
        self.assertAlmostEqual(
            svc.rain_rate_si_to_model(2.5e-6),
            self._u.rain_si_to_model(2.5e-6),
        )
        self.assertAlmostEqual(svc.flow_si_to_model(10.0), 10.0)
        self.assertAlmostEqual(
            svc.flow_si_to_model(10.0), self._u.flow_si_to_model(10.0)
        )

    def test_usc_matrix(self):
        from swe2d.workbench.services import unit_conversion_service as svc

        self._u.configure(0.3048)
        self.assertAlmostEqual(
            svc.length_scale_si_to_model(), 1.0 / 0.3048, places=12
        )
        self.assertAlmostEqual(
            svc.rain_mm_to_model_depth(), 1.0e-3 / 0.3048, places=12
        )
        self.assertAlmostEqual(
            svc.rain_rate_si_to_model(1.0), 1.0 / 0.3048, places=12
        )
        self.assertAlmostEqual(
            svc.flow_si_to_model(1.0), (1.0 / 0.3048) ** 3, places=9
        )
        self.assertAlmostEqual(
            svc.flow_si_to_model(1.0), self._u.flow_si_to_model(1.0)
        )

    def test_is_us_customary_units(self):
        from swe2d.workbench.services.unit_conversion_service import (
            is_us_customary_units,
        )

        self.assertTrue(is_us_customary_units("ft"))
        self.assertTrue(is_us_customary_units("FT"))
        self.assertTrue(is_us_customary_units("  ft  "))
        self.assertFalse(is_us_customary_units("m"))
        self.assertFalse(is_us_customary_units(""))

    # ---- update_unit_system_from_crs ----

    def test_update_from_meters_crs(self):
        from swe2d.workbench.services.unit_conversion_service import (
            update_unit_system_from_crs,
        )

        out = update_unit_system_from_crs(
            have_qgis_core=True, project=self._project("EPSG:32616")
        )
        self.assertEqual(out["unit_name"], "m")
        self.assertEqual(out["sys_name"], "SI")
        self.assertAlmostEqual(out["scale"], 1.0)
        self.assertAlmostEqual(out["gravity"], self._u.SI_GRAVITY)
        self.assertAlmostEqual(out["k_mann"], self._u.SI_MANNING_FACTOR)
        self.assertIn("EPSG:32616", out["crs_desc"])
        # Side-effect contract: global units are configured to the CRS scale.
        self.assertAlmostEqual(self._u.si_m_per_model(), 1.0)

    def test_update_from_feet_crs(self):
        from swe2d.workbench.services.unit_conversion_service import (
            update_unit_system_from_crs,
        )

        out = update_unit_system_from_crs(
            have_qgis_core=True, project=self._project("EPSG:2277")
        )
        self.assertEqual(out["unit_name"], "ft")
        self.assertEqual(out["sys_name"], "US Customary")
        self.assertAlmostEqual(out["scale"], 0.3048)
        self.assertAlmostEqual(out["gravity"], self._u.USC_GRAVITY, places=6)
        self.assertAlmostEqual(out["k_mann"], self._u.USC_MANNING_FACTOR)
        self.assertAlmostEqual(self._u.si_m_per_model(), 0.3048)

    def test_update_from_geographic_crs_falls_back(self):
        from swe2d.workbench.services.unit_conversion_service import (
            update_unit_system_from_crs,
        )

        out = update_unit_system_from_crs(
            have_qgis_core=True, project=self._project("EPSG:4326")
        )
        # Degrees are neither feet nor meters → documented fallback branch.
        self.assertEqual(out["sys_name"], "SI (fallback)")
        self.assertAlmostEqual(out["scale"], 1.0)
        self.assertEqual(out["unit_name"], "degrees")

    def test_update_without_qgis_returns_defaults(self):
        from swe2d.workbench.services.unit_conversion_service import (
            update_unit_system_from_crs,
        )

        out = update_unit_system_from_crs(have_qgis_core=False, project=None)
        self.assertEqual(out["unit_name"], "m")
        self.assertEqual(out["sys_name"], "SI")
        self.assertAlmostEqual(out["scale"], 1.0)
        self.assertEqual(out["crs_desc"], "(no CRS)")


# ===========================================================================
# 3. non_gui_qgis_service (all 4 public functions)
# ===========================================================================


@requires_qgis
class TestNonGuiQgisService(unittest.TestCase):
    """Real memory layers / raster layers against the helper functions."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def _fields_layer(self):
        from qgis.PyQt.QtCore import QVariant

        return make_memory_layer(
            "Point",
            fields=[("depth", QVariant.Double), ("Name", QVariant.String)],
            features=[("POINT(1 2)", (1.5, "gage_a"))],
            name="fields",
        )

    # ---- resolve_layer_field_name ----

    def test_resolve_layer_field_name_exact_and_case_insensitive(self):
        from swe2d.workbench.services.non_gui_qgis_service import (
            resolve_layer_field_name,
        )

        layer = self._fields_layer()
        self.assertEqual(resolve_layer_field_name(layer, "depth"), "depth")
        self.assertEqual(resolve_layer_field_name(layer, "DEPTH"), "depth")
        self.assertEqual(resolve_layer_field_name(layer, "name"), "Name")

    def test_resolve_layer_field_name_missing_returns_empty(self):
        from swe2d.workbench.services.non_gui_qgis_service import (
            resolve_layer_field_name,
        )

        layer = self._fields_layer()
        # Contract: unresolvable → "" (callers treat "" as absent).
        self.assertEqual(resolve_layer_field_name(layer, "bogus"), "")
        self.assertEqual(resolve_layer_field_name(layer, ""), "")
        self.assertEqual(resolve_layer_field_name(None, "depth"), "")

    # ---- parse_feature_float ----

    def test_parse_feature_float_value_and_defaults(self):
        from swe2d.workbench.services.non_gui_qgis_service import (
            parse_feature_float,
        )

        layer = self._fields_layer()
        feat = next(layer.getFeatures())
        self.assertAlmostEqual(parse_feature_float(feat, "depth", 9.0), 1.5)
        # Missing field → default.
        self.assertAlmostEqual(parse_feature_float(feat, "nope", 9.0), 9.0)
        # Unparseable string field → default.
        self.assertAlmostEqual(parse_feature_float(feat, "Name", 9.0), 9.0)
        # Blank field name → default.
        self.assertAlmostEqual(parse_feature_float(feat, "", 9.0), 9.0)

    def test_parse_feature_float_null_and_nonfinite(self):
        from qgis.PyQt.QtCore import QVariant

        from swe2d.workbench.services.non_gui_qgis_service import (
            parse_feature_float,
        )

        layer = make_memory_layer(
            "Point",
            fields=[("v", QVariant.Double)],
            features=[
                ("POINT(0 0)", (None,)),
                ("POINT(0 0)", (float("nan"),)),
            ],
            name="nulls",
        )
        feats = list(layer.getFeatures())
        # NULL attribute → default; NaN → default (non-finite guard).
        self.assertAlmostEqual(parse_feature_float(feats[0], "v", 7.0), 7.0)
        self.assertAlmostEqual(parse_feature_float(feats[1], "v", 7.0), 7.0)

    # ---- infer_obj_path_from_layer_3d_renderer ----

    def test_infer_obj_path_no_renderer(self):
        from swe2d.workbench.services.non_gui_qgis_service import (
            infer_obj_path_from_layer_3d_renderer,
        )

        self.assertEqual(infer_obj_path_from_layer_3d_renderer(None), "")
        layer = make_memory_layer("Point", name="plain")
        self.assertEqual(infer_obj_path_from_layer_3d_renderer(layer), "")

    def test_infer_obj_path_real_3d_renderer_with_model(self):
        from qgis._3d import QgsPoint3DSymbol, QgsVectorLayer3DRenderer

        from swe2d.workbench.services.non_gui_qgis_service import (
            infer_obj_path_from_layer_3d_renderer,
        )

        layer = make_memory_layer("Point", name="with3d")
        symbol = QgsPoint3DSymbol()
        symbol.setShape(QgsPoint3DSymbol.Model)
        symbol.setShapeProperties({"model": "/tmp/tree.obj"})
        layer.setRenderer3D(QgsVectorLayer3DRenderer(symbol))

        # The real QgsPoint3DSymbol stores its configured OBJ path in the
        # shape-properties mapping, not in a modelPath attribute.
        self.assertEqual(
            symbol.shapeProperties().get("model"), "/tmp/tree.obj"
        )
        self.assertEqual(
            infer_obj_path_from_layer_3d_renderer(layer), "/tmp/tree.obj"
        )

    # ---- build_patch_terrain_surface ----

    def _write_test_raster(self):
        """4x3 Float64 GeoTIFF; val[j, i] = j * 4 + i (row 0 at top)."""
        from osgeo import gdal

        vals = (np.arange(12, dtype=np.float64).reshape(3, 4)) + 0.25
        path = os.path.join(
            tempfile.mkdtemp(prefix="hydra_dem_"), "dem.tif"
        )
        ds = gdal.GetDriverByName("GTiff").Create(
            path, 4, 3, 1, gdal.GDT_Float64
        )
        # pixel 2 x 3 map units; top-left corner at (100, 209).
        ds.SetGeoTransform((100.0, 2.0, 0.0, 209.0, 0.0, -3.0))
        ds.GetRasterBand(1).WriteArray(vals)
        ds.FlushCache()
        ds = None
        self.addCleanup(os.unlink, path)
        return path, vals

    def test_build_patch_terrain_surface_real_raster(self):
        from qgis.core import QgsPointXY, QgsRasterLayer

        from swe2d.workbench.services.non_gui_qgis_service import (
            build_patch_terrain_surface,
        )

        path, vals = self._write_test_raster()
        raster = QgsRasterLayer(path, "dem")
        self.assertTrue(raster.isValid())

        spec = types.SimpleNamespace(
            nx=4, ny=3, dx=2.0, dy=3.0, origin_x=100.0, origin_y=200.0
        )
        terrain = build_patch_terrain_surface(
            spec=spec, raster_layer=raster, qgs_point_xy_cls=QgsPointXY
        )
        self.assertIsNotNone(terrain)
        self.assertEqual(terrain.shape, (3, 4))
        # terrain row 0 is the southern-most cell row (origin_y); GDAL
        # array row 0 is the northern-most raster row → rows are flipped.
        np.testing.assert_allclose(terrain, vals[::-1])

    def test_build_patch_terrain_surface_contract_nones(self):
        from qgis.core import QgsPointXY, QgsRasterLayer

        from swe2d.workbench.services.non_gui_qgis_service import (
            build_patch_terrain_surface,
        )

        path, _ = self._write_test_raster()
        raster = QgsRasterLayer(path, "dem")
        spec = types.SimpleNamespace(
            nx=2, ny=2, dx=1.0, dy=1.0, origin_x=100.0, origin_y=200.0
        )
        self.assertIsNone(
            build_patch_terrain_surface(
                spec=spec, raster_layer=None, qgs_point_xy_cls=QgsPointXY
            )
        )
        self.assertIsNone(
            build_patch_terrain_surface(
                spec=spec, raster_layer=raster, qgs_point_xy_cls=None
            )
        )
        bad = types.SimpleNamespace(
            nx=0, ny=2, dx=1.0, dy=1.0, origin_x=0.0, origin_y=0.0
        )
        self.assertIsNone(
            build_patch_terrain_surface(
                spec=bad, raster_layer=raster, qgs_point_xy_cls=QgsPointXY
            )
        )


# ===========================================================================
# 4. pipe_network_config_service.build_pipe_network_config_from_widgets
# ===========================================================================


@requires_qgis
class TestPipeNetworkConfigService(unittest.TestCase):
    """Widget-param dict → PipeNetworkConfig; guard contract returns None."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def _layers(self):
        from qgis.core import QgsFeature, QgsGeometry
        from swe2d.workbench.services.schema_definitions import (
            create_memory_layer,
        )

        nodes = create_memory_layer("swe2d_drainage_nodes")
        for nid, x, y, ntype in (
            ("N1", 5.0, 5.0, "junction"),
            ("N2", 15.0, 5.0, "outfall"),
        ):
            f = QgsFeature(nodes.fields())
            f.setGeometry(QgsGeometry.fromWkt(f"POINT({x} {y})"))
            f.setAttribute("node_id", nid)
            f.setAttribute("invert_elev", 10.0)
            f.setAttribute("rim_elev", 12.0)
            f.setAttribute("node_type", ntype)
            nodes.dataProvider().addFeatures([f])
        nodes.updateExtents()

        links = create_memory_layer("swe2d_drainage_links")
        f = QgsFeature(links.fields())
        f.setGeometry(
            QgsGeometry.fromWkt("LINESTRING(5 5, 15 5)")
        )
        f.setAttribute("link_id", "L1")
        f.setAttribute("from_node", "N1")
        f.setAttribute("to_node", "N2")
        f.setAttribute("link_type", "conduit")
        f.setAttribute("link_shape", "circular")
        f.setAttribute("length", 50.0)
        f.setAttribute("roughness_n", 0.013)
        f.setAttribute("diameter", 0.5)
        links.dataProvider().addFeatures([f])
        links.updateExtents()
        return nodes, links

    def _mesh(self):
        return {
            "node_x": np.array([0.0, 20.0, 0.0]),
            "node_y": np.array([0.0, 0.0, 20.0]),
            "node_z": np.array([9.0, 9.0, 9.0]),
            "cell_nodes": np.array([[0, 1, 2]], dtype=np.int32),
        }

    def _call(self, **overrides):
        from swe2d.extensions.extension_models import PipeNetworkConfig
        from swe2d.workbench.services.pipe_network_config_service import (
            build_pipe_network_config_from_widgets,
        )

        nodes, links = self._layers()
        kwargs = dict(
            mesh_data=self._mesh(),
            have_qgis_core=True,
            pipe_network_config_cls=PipeNetworkConfig,
            node_layer=nodes,
            link_layer=links,
            inlet_layer=None,
            node_inlet_layer=None,
            cell_min_bed=np.array([9.0]),
            nearest_cell_fn=lambda x, y: 0,
            gravity=9.81,
            solver_mode_name="Diffusion Wave",
            solver_mode=0,
            coupling_substeps=4,
            gpu_method="auto",
            head_deadband=0.002,
            dynamic_relaxation=0.6,
            implicit_iters=5,
            implicit_relax=0.7,
            friction_method=1,
            recon_method=1,
            time_integrator=0,
            friction_alpha=0.02,
            surcharge_method=1,
        )
        kwargs.update(overrides)
        return build_pipe_network_config_from_widgets(**kwargs)

    def test_realistic_widgets_build_config(self):
        from swe2d.extensions.extension_models import PipeNetworkConfig

        cfg = self._call()
        self.assertIsInstance(cfg, PipeNetworkConfig)
        self.assertTrue(cfg.enabled)
        self.assertEqual(len(cfg.nodes), 2)
        self.assertEqual(len(cfg.links), 1)
        self.assertEqual({n.node_id for n in cfg.nodes}, {"N1", "N2"})
        self.assertEqual(len(cfg.outfalls), 1)
        # Scalar widget params flow through unchanged.
        self.assertAlmostEqual(cfg.gravity, 9.81)
        self.assertEqual(cfg.coupling_substeps, 4)
        self.assertAlmostEqual(cfg.head_deadband_m, 0.002)
        self.assertAlmostEqual(cfg.dynamic_flow_relaxation, 0.6)
        self.assertEqual(cfg.implicit_coupling_iterations, 5)
        self.assertAlmostEqual(cfg.implicit_coupling_relaxation, 0.7)
        self.assertEqual(cfg.friction_method, 1)
        self.assertEqual(cfg.recon_method, 1)
        self.assertEqual(cfg.time_integrator, 0)
        self.assertAlmostEqual(cfg.friction_alpha, 0.02)
        self.assertEqual(cfg.surcharge_method, 1)
        self.assertEqual(cfg.pipe_solver_mode, "diffusion_wave")

    def test_fully_dynamic_solver_mode(self):
        cfg = self._call(solver_mode=1, solver_mode_name="Fully Dynamic")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.pipe_solver_mode, "fully_dynamic")

    def test_guard_contract_returns_none(self):
        # Contract: missing prerequisites return None (fail-closed — the
        # dialog treats None as "no drainage network"), never an exception
        # and never a half-populated config.
        self.assertIsNone(self._call(mesh_data=None))
        self.assertIsNone(self._call(have_qgis_core=False))
        self.assertIsNone(self._call(pipe_network_config_cls=None))
        self.assertIsNone(self._call(node_layer=None))
        self.assertIsNone(self._call(link_layer=None))
        self.assertIsNone(self._call(cell_min_bed=None))
        self.assertIsNone(self._call(nearest_cell_fn=None))

    def test_missing_cell_min_bed_logs_error(self):
        messages = []
        self.assertIsNone(
            self._call(cell_min_bed=None, log_fn=messages.append)
        )
        self.assertTrue(
            any("cell_min_bed" in m for m in messages),
            f"expected a loud log line, got {messages!r}",
        )


# ===========================================================================
# 5. gpkg_layer_styles_service.apply_qml_style_from_gpkg
# ===========================================================================


@requires_qgis
class TestGpkgLayerStylesService(unittest.TestCase):
    """Real layer + bundled real QML → True and style applied."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_apply_real_qml(self):
        from qgis.PyQt.QtGui import QColor

        from swe2d.workbench.services.gpkg_layer_styles_service import (
            apply_qml_style_from_gpkg,
        )

        layer = make_memory_layer("LineString", name="swe2d_bc_lines")
        before = layer.renderer().symbol().color()
        # gpkg_path is accepted for the public contract but the bundled
        # QML/ dir is the source of truth (implementation is repo-relative).
        ok = apply_qml_style_from_gpkg(layer, "/nonexistent/model.gpkg")
        self.assertTrue(ok)
        # Mutation verified by production readback: the renderer-v2 block
        # of QML/swe2d_bc_lines.qml paints lines 255,127,0 (orange).
        after = layer.renderer().symbol().color()
        self.assertNotEqual(before, after)
        self.assertEqual(after, QColor(255, 127, 0))

    def test_missing_qml_returns_false(self):
        from swe2d.workbench.services.gpkg_layer_styles_service import (
            apply_qml_style_from_gpkg,
        )

        layer = make_memory_layer("Point", name="no_such_layer_xyz")
        before = layer.renderer().symbol().color()
        self.assertFalse(
            apply_qml_style_from_gpkg(layer, "/nonexistent/model.gpkg")
        )
        self.assertEqual(layer.renderer().symbol().color(), before)


# ===========================================================================
# 6. topology_template_service.create_topology_template_layers
# ===========================================================================


@requires_qgis
class TestTopologyTemplateService(unittest.TestCase):
    """14 template layers: count, names, geometry types, field schemas."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_layers_match_canonical_schema(self):
        from qgis.core import QgsWkbTypes

        from swe2d.workbench.services.schema_definitions import (
            LAYER_SCHEMAS,
            get_display_name,
        )
        from swe2d.workbench.services.topology_template_service import (
            _TOPOLOGY_TEMPLATE_KEYS,
            create_topology_template_layers,
        )

        layers = create_topology_template_layers("EPSG:32616")
        self.assertEqual(len(layers), 14)
        self.assertEqual(len(layers), len(_TOPOLOGY_TEMPLATE_KEYS))

        geom_type_map = {
            "Point": QgsWkbTypes.PointGeometry,
            "LineString": QgsWkbTypes.LineGeometry,
            "Polygon": QgsWkbTypes.PolygonGeometry,
        }
        for key, (display_name, lyr) in zip(_TOPOLOGY_TEMPLATE_KEYS, layers):
            schema = LAYER_SCHEMAS[key]
            self.assertEqual(display_name, get_display_name(key))
            self.assertEqual(lyr.name(), display_name)
            self.assertTrue(lyr.isValid(), f"{key}: invalid layer")
            if schema["geom"] is None:
                # Table layers carry no CRS (memory URI "None?...").
                self.assertEqual(lyr.wkbType(), QgsWkbTypes.NoGeometry)
            else:
                self.assertEqual(lyr.crs().authid(), "EPSG:32616")
                self.assertEqual(
                    QgsWkbTypes.geometryType(lyr.wkbType()),
                    geom_type_map[schema["geom"]],
                    f"{key}: wrong geometry type",
                )
            self.assertEqual(
                list(lyr.fields().names()),
                [fname for fname, _ in schema["fields"]],
                f"{key}: field schema drift",
            )


# ===========================================================================
# 7. model_gpkg_loader_service
# ===========================================================================


@requires_qgis
class TestModelGpkgLoaderService(unittest.TestCase):
    """load_layers_from_gpkg round-trip against the production fixture."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_get_model_gpkg_layer_names(self):
        from swe2d.workbench.services.model_gpkg_loader_service import (
            get_model_gpkg_layer_names,
        )
        from swe2d.workbench.services.schema_definitions import (
            LAYER_SCHEMAS,
            get_layer_names,
        )

        names = get_model_gpkg_layer_names()
        self.assertEqual(names, get_layer_names())
        # Count follows the canonical schema, not a hardcoded number
        # (stale docstrings still say "18"; the schema has grown).
        self.assertEqual(len(names), len(LAYER_SCHEMAS))

    def test_load_layers_from_fixture_gpkg(self):
        from swe2d.workbench.services.model_gpkg_loader_service import (
            get_model_gpkg_layer_names,
            load_layers_from_gpkg,
        )

        with make_temp_model_gpkg() as path:
            loaded = load_layers_from_gpkg(path)
            self.assertEqual(
                sorted(loaded.keys()), sorted(get_model_gpkg_layer_names())
            )
            for name, lyr in loaded.items():
                self.assertTrue(lyr.isValid(), f"{name}: invalid layer")
                self.assertEqual(lyr.name(), name)
                self.assertEqual(lyr.providerType(), "ogr")

    def test_missing_file_returns_empty_dict(self):
        from swe2d.workbench.services.model_gpkg_loader_service import (
            load_layers_from_gpkg,
        )

        # CHARACTERIZATION FINDING: the docstring contract is "Missing
        # layers are silently skipped" — a nonexistent GPKG yields {} with
        # no error raised.  The caller (dialog) is responsible for
        # detecting the empty result and reporting it.
        loaded = load_layers_from_gpkg("/nonexistent/path/model.gpkg")
        self.assertEqual(loaded, {})


if __name__ == "__main__":
    unittest.main()
