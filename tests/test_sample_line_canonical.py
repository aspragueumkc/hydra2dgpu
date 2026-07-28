"""Failing tests for the canonical sample-line contract and independent flow reference.

Spec: docs/specs/2026-07-27-canonical-sample-line-sampling.md
Plan: docs/plans/2026-07-27-canonical-sample-line-sampling.md — Task 1

Scope of this module
--------------------
1. The canonical plain-data API on the line-sampling service
   (``build_canonical_line_sampling_map``) is reachable and produces the
   required per-line ``cell_idx``, ``weights``, ``station_m``, ``normal_x``,
   ``normal_y`` arrays.
2. The independent flow-rate oracle (``reference_line_flow``) computes
   ``Q = sum(weights * where(h > h_min, hu*nx + hv*ny, 0.0))`` for known
   state arrays without touching any line-metrics implementation.
3. The GPKG/batch path no longer raises
   ``AttributeError: 'tuple' object has no attribute 'isEmpty'`` for a
   valid enabled sample line over a valid mesh.

These tests intentionally fail today because the canonical plain-data
service and oracle are not yet implemented (Task 2) and the GPKG path
still feeds tuples into a QGIS-geometry-only callback (the bug in the
spec §2 / §12 "tuple/QGIS mismatch").
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Module-level fixture
# ---------------------------------------------------------------------------
def setUpModule() -> None:
    """Suppress GDAL's ``UseExceptions not called`` FutureWarning so the
    focused GPKG/tuple regression test isn't masked by upstream noise.
    Idempotent — repeated calls are a no-op.
    """
    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Deterministic rectangular mesh fixtures (plain records, no Qt)
# ---------------------------------------------------------------------------
def _rect_mesh_records(nx: int = 3, ny: int = 2, Lx: float = 30.0, Ly: float = 10.0):
    """Build a list of plain mesh-cell records covering an ``nx × ny`` grid.

    Cell index order is row-major (y asc, then x asc).  Cell ``i`` spans
    ``x ∈ [i*Lx/nx, (i+1)*Lx/nx]`` for ``i < nx`` and then jumps to the
    next row.
    """
    cells = []
    cell_idx = 0
    for j in range(ny):
        for i in range(nx):
            x0, x1 = i * Lx / nx, (i + 1) * Lx / nx
            y0, y1 = j * Ly / ny, (j + 1) * Ly / ny
            pts = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float64,
            )
            cells.append({"cell_idx": int(cell_idx), "points": pts})
            cell_idx += 1
    return cells


def _make_rect_mesh_data(nx: int = 3, ny: int = 2, Lx: float = 30.0, Ly: float = 10.0):
    """Return a mesh_data dict suitable for the existing mesh helpers."""
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().astype(np.float64)
    node_y = Yg.ravel().astype(np.float64)
    cell_face_nodes: list[int] = []
    cell_face_offsets = [0]
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n11 = (j + 1) * stride + i + 1
            n01 = (j + 1) * stride + i
            cell_face_nodes.extend([n00, n10, n11, n01])
            cell_face_offsets.append(len(cell_face_nodes))
    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_z": np.zeros_like(node_x),
        "cell_face_offsets": np.asarray(cell_face_offsets, dtype=np.int32),
        "cell_face_nodes": np.asarray(cell_face_nodes, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Canonical plain-data contract (spec §5, §6)
# ---------------------------------------------------------------------------
class TestCanonicalPlainDataContract(unittest.TestCase):
    """Spec §5, §6: plain-data in / plain-data out."""

    def test_canonical_api_is_exported(self):
        from swe2d.services import line_sampling_service as svc

        self.assertTrue(
            hasattr(svc, "build_canonical_line_sampling_map"),
            "Canonical plain-record API is missing from "
            "swe2d.services.line_sampling_service (spec §5, §6).",
        )

    def test_horizontal_line_crosses_three_bottom_row_cells(self):
        """A horizontal line at y=5 from x=5 to x=25 crosses cells 0,1,2
        in order, with equal intersection length and stations at the
        cell centroids along the oriented line.
        """
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        cells = _rect_mesh_records(nx=3, ny=2, Lx=30.0, Ly=10.0)
        sample_lines = [
            {
                "line_id": 7,
                "line_name": "transect_A",
                "enabled": True,
                "points": np.array([[5.0, 5.0], [25.0, 5.0]], dtype=np.float64),
            },
        ]
        result = build_canonical_line_sampling_map(
            sample_lines=sample_lines, mesh_cells=cells,
        )

        self.assertEqual(len(result), 1, "expected one map entry per line")
        rec = result[0]
        self.assertEqual(rec["line_id"], 7)
        self.assertEqual(rec["line_name"], "transect_A")
        # Tangent (1, 0); normal is the 90° rotation (sign-agnostic here).
        self.assertAlmostEqual(
            math.hypot(rec["normal_x"], rec["normal_y"]), 1.0, places=12
        )
        # Cell order is left-to-right along the oriented line.
        self.assertTrue(
            np.array_equal(rec["cell_idx"], np.array([0, 1, 2], dtype=np.int32)),
            f"cell_idx was {rec['cell_idx']!r}",
        )
        # Intersection length in each of the three crossed cells:
        # cell 0 [0,10] intersected by x∈[5,10] → 5; cell 1 [10,20] → 10;
        # cell 2 [20,30] intersected by x∈[20,25] → 5.
        self.assertTrue(
            np.allclose(rec["weights"], np.array([5.0, 10.0, 5.0])),
            f"weights were {rec['weights']!r}",
        )
        # Stations are the intersection midpoints projected onto the oriented
        # tangent, measured from the line start.  Line goes from x=5 to
        # x=25 with tangent (1, 0); intersection midpoints at x=7.5, 15,
        # 22.5 → stations 2.5, 10.0, 17.5.
        self.assertTrue(
            np.allclose(rec["station_m"], np.array([2.5, 10.0, 17.5])),
            f"station_m were {rec['station_m']!r}",
        )
        # Dtypes match spec §6.
        self.assertEqual(rec["cell_idx"].dtype, np.int32)
        self.assertEqual(rec["weights"].dtype, np.float64)
        self.assertEqual(rec["station_m"].dtype, np.float64)
        # Lengths match (spec §6).
        self.assertEqual(rec["cell_idx"].shape, rec["weights"].shape)
        self.assertEqual(rec["cell_idx"].shape, rec["station_m"].shape)

    def test_disabled_line_is_excluded(self):
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        cells = _rect_mesh_records(nx=3, ny=2, Lx=30.0, Ly=10.0)
        sample_lines = [
            {
                "line_id": 7,
                "line_name": "off",
                "enabled": False,
                "points": np.array([[5.0, 5.0], [25.0, 5.0]], dtype=np.float64),
            },
        ]
        result = build_canonical_line_sampling_map(
            sample_lines=sample_lines, mesh_cells=cells,
        )
        self.assertEqual(
            result, [],
            "disabled sample lines must not produce a map entry "
            "(spec §5.1 enabled field semantics).",
        )


# ---------------------------------------------------------------------------
# Reversed orientation and multi-cell crossing
# ---------------------------------------------------------------------------
class TestReversedOrientation(unittest.TestCase):
    def test_reversed_line_flips_normal_and_reverses_cell_order(self):
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        cells = _rect_mesh_records(nx=3, ny=2, Lx=30.0, Ly=10.0)
        fwd_lines = [{
            "line_id": 1, "line_name": "fwd", "enabled": True,
            "points": np.array([[5.0, 5.0], [25.0, 5.0]], dtype=np.float64),
        }]
        rev_lines = [{
            "line_id": 1, "line_name": "rev", "enabled": True,
            "points": np.array([[25.0, 5.0], [5.0, 5.0]], dtype=np.float64),
        }]
        fwd = build_canonical_line_sampling_map(
            sample_lines=fwd_lines, mesh_cells=cells,
        )
        rev = build_canonical_line_sampling_map(
            sample_lines=rev_lines, mesh_cells=cells,
        )
        self.assertEqual(len(fwd), 1)
        self.assertEqual(len(rev), 1)
        # Normal sign flips (spec §6: "preserve line orientation").
        self.assertAlmostEqual(fwd[0]["normal_x"], -rev[0]["normal_x"], places=12)
        self.assertAlmostEqual(fwd[0]["normal_y"], -rev[0]["normal_y"], places=12)
        # Weights are geometric → identical, just in reversed cell order.
        self.assertTrue(
            np.allclose(fwd[0]["weights"], rev[0]["weights"][::-1]),
            f"forward weights {fwd[0]['weights']!r} "
            f"vs reversed {rev[0]['weights']!r}",
        )
        # Cell order reverses.
        self.assertTrue(
            np.array_equal(fwd[0]["cell_idx"], rev[0]["cell_idx"][::-1]),
            f"forward cells {fwd[0]['cell_idx']!r} "
            f"vs reversed {rev[0]['cell_idx']!r}",
        )
        # Multi-cell contract: all per-line arrays share dtype + length
        # (spec §6).
        for label, rec_ in (("fwd", fwd[0]), ("rev", rev[0])):
            self.assertEqual(rec_["cell_idx"].dtype, np.int32,
                             f"{label} cell_idx dtype {rec_['cell_idx'].dtype}")
            self.assertEqual(rec_["weights"].dtype, np.float64,
                             f"{label} weights dtype {rec_['weights'].dtype}")
            self.assertEqual(rec_["station_m"].dtype, np.float64,
                             f"{label} station_m dtype {rec_['station_m'].dtype}")
            self.assertEqual(rec_["cell_idx"].shape, rec_["weights"].shape,
                             f"{label} cell_idx/weights length mismatch")
            self.assertEqual(rec_["cell_idx"].shape, rec_["station_m"].shape,
                             f"{label} cell_idx/station_m length mismatch")


class TestMultiCellCrossing(unittest.TestCase):
    def test_diagonal_line_crosses_expected_cells(self):
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        cells = _rect_mesh_records(nx=3, ny=2, Lx=30.0, Ly=10.0)
        # Diagonal from (0,0) to (30,10).  Cell height is 5; the line
        # crosses y=5 at x=15.  Bottom row [0..5] covers cells 0,1,2 by
        # x; the line is in cell 0 from x=0 to x=10, cell 1 from x=10
        # to x=15, then jumps to the top row at x=15 and is in cell 4
        # from x=15 to x=20, cell 5 from x=20 to x=30.  Cells 2 and 3
        # are never entered.
        sample_lines = [{
            "line_id": 11, "line_name": "diag", "enabled": True,
            "points": np.array([[0.0, 0.0], [30.0, 10.0]], dtype=np.float64),
        }]
        result = build_canonical_line_sampling_map(
            sample_lines=sample_lines, mesh_cells=cells,
        )
        self.assertEqual(len(result), 1)
        rec = result[0]
        self.assertEqual(rec["cell_idx"].size, 4)
        # Sum of intersection lengths must equal the total line length
        # (positive-length intersections only, no double-counting).
        self.assertAlmostEqual(
            float(np.sum(rec["weights"])),
            math.hypot(30.0, 10.0),
            places=6,
        )
        # Multi-cell contract: dtypes + equal lengths (spec §6).
        self.assertEqual(rec["cell_idx"].dtype, np.int32)
        self.assertEqual(rec["weights"].dtype, np.float64)
        self.assertEqual(rec["station_m"].dtype, np.float64)
        self.assertEqual(rec["cell_idx"].shape, rec["weights"].shape)
        self.assertEqual(rec["cell_idx"].shape, rec["station_m"].shape)


# ---------------------------------------------------------------------------
# Exact edge overlap and corner-only touches (spec §7 step 9)
# ---------------------------------------------------------------------------
class TestEdgeOverlapAndCornerTouch(unittest.TestCase):
    def test_exact_edge_overlap_intersects_two_cells_with_zero_double_count(self):
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        # A 2×1 mesh; line sits exactly along the shared interior edge
        # at y = Ly/2.  The line must intersect two cells, each for the
        # full length, with no duplicate weighting.
        cells = _rect_mesh_records(nx=2, ny=1, Lx=20.0, Ly=10.0)
        sample_lines = [{
            "line_id": 21, "line_name": "shared_edge", "enabled": True,
            "points": np.array([[0.0, 5.0], [20.0, 5.0]], dtype=np.float64),
        }]
        result = build_canonical_line_sampling_map(
            sample_lines=sample_lines, mesh_cells=cells,
        )
        self.assertEqual(len(result), 1)
        rec = result[0]
        self.assertEqual(rec["cell_idx"].size, 2)
        # Each intersection is 20 m long; weights sum to 20 (no double count).
        weights_arr = rec["weights"]
        self.assertAlmostEqual(
            float(np.sum(weights_arr)),
            20.0,
            places=6,
            msg=f"weights were {weights_arr!r}",
        )
        # Multi-cell contract: dtypes + equal lengths (spec §6).
        self.assertEqual(rec["cell_idx"].dtype, np.int32)
        self.assertEqual(rec["weights"].dtype, np.float64)
        self.assertEqual(rec["station_m"].dtype, np.float64)
        self.assertEqual(rec["cell_idx"].shape, rec["weights"].shape)
        self.assertEqual(rec["cell_idx"].shape, rec["station_m"].shape)

    def test_corner_only_touch_produces_no_intersection(self):
        """A sample line that touches a cell at exactly one vertex has
        zero-length intersection and must not appear in the map.
        """
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        cells = _rect_mesh_records(nx=3, ny=2, Lx=30.0, Ly=10.0)
        # Two-point degenerate: both endpoints coincide at a vertex that
        # belongs to exactly one cell (corner of the domain).
        sample_lines = [{
            "line_id": 31, "line_name": "corner_touch", "enabled": True,
            "points": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        }]
        result = build_canonical_line_sampling_map(
            sample_lines=sample_lines, mesh_cells=cells,
        )
        if result:
            self.assertEqual(
                result[0]["cell_idx"].size, 0,
                "a degenerate sample line must not produce a positive-length "
                "intersection; the canonical service must skip or empty it.",
            )


# ---------------------------------------------------------------------------
# Independent flow-rate oracle (spec §11.1)
# ---------------------------------------------------------------------------
class TestIndependentFlowOracle(unittest.TestCase):
    """Spec §11.1: independent reference for the kernel's wet-cell
    normal-discharge sum, no line-metrics implementation touched.
    """

    def test_oracle_function_exists(self):
        from swe2d.services.line_sampling_service import reference_line_flow

        self.assertTrue(callable(reference_line_flow))

    def test_uniform_flow_along_normal_direction(self):
        from swe2d.services.line_sampling_service import reference_line_flow

        # Three cells, uniform depth 2 m, uniform velocity v=(0, 1) so
        # qn = hu*nx + hv*ny with normal=(0, 1) → qn = 1*2 = 2.
        h = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        hu = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        hv = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        cell_idx = np.array([0, 1, 2], dtype=np.int32)
        weights = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        Q = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=0.0, normal_y=1.0,
            h_min=1e-4,
        )
        # Q = sum(weights * h * qn_unit_velocity) = 30 * 2 * 1 = 60.
        self.assertAlmostEqual(float(Q), 60.0, places=12)

    def test_uniform_flow_perpendicular_to_normal(self):
        """Flow perpendicular to the normal contributes zero discharge."""
        from swe2d.services.line_sampling_service import reference_line_flow

        h = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        hu = np.array([3.0, 3.0, 3.0], dtype=np.float64)
        hv = np.zeros(3, dtype=np.float64)
        cell_idx = np.array([0, 1, 2], dtype=np.int32)
        weights = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        Q = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=0.0, normal_y=1.0,
            h_min=1e-4,
        )
        # qn = 3*0 + 0*1 = 0 → Q = 0.
        self.assertAlmostEqual(float(Q), 0.0, places=12)

    def test_dry_cells_are_excluded_by_h_min(self):
        from swe2d.services.line_sampling_service import reference_line_flow

        # ``hu`` is treated as the per-cell momentum flux (h*u) — an
        # independent state variable in the canonical oracle.  We keep
        # hu>0 in dry cells on purpose so the wet predicate (h > h_min)
        # is the only thing that can exclude them; this isolates the
        # predicate from any coincidental hu=0 short-circuit.
        h = np.array([0.0, 1.5, 0.0], dtype=np.float64)  # cells 0 and 2 dry
        hu = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        hv = np.zeros(3, dtype=np.float64)
        cell_idx = np.array([0, 1, 2], dtype=np.int32)
        weights = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        Q = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=1.0, normal_y=0.0,
            h_min=1e-4,
        )
        # wet=[F,T,F]; qn=[1,1,1]; Q = 10*1 = 10 (hu already carries the
        # h*u factor; the oracle does not re-multiply by h).
        self.assertAlmostEqual(float(Q), 10.0, places=12)

    def test_h_min_boundary_excludes_cell_at_threshold(self):
        """h exactly at h_min must be treated as dry (kernel convention)."""
        from swe2d.services.line_sampling_service import reference_line_flow

        # ``hu`` is treated as the per-cell momentum flux (h*u) — an
        # independent state variable in the canonical oracle.
        h = np.array([1e-4, 1.5], dtype=np.float64)
        hu = np.array([2.0, 2.0], dtype=np.float64)
        hv = np.zeros(2, dtype=np.float64)
        cell_idx = np.array([0, 1], dtype=np.int32)
        weights = np.array([1.0, 1.0], dtype=np.float64)
        Q = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=1.0, normal_y=0.0,
            h_min=1e-4,
        )
        # Cell 0 at h=h_min is dry (strict >); only cell 1 contributes:
        # Q = 1 * (hu*nx + hv*ny) = 1 * (2*1 + 0*0) = 2.
        # (hu already carries the h*u factor; the oracle does not
        # re-multiply by h — see the dry-cell test for the same rule.)
        self.assertAlmostEqual(float(Q), 2.0, places=12)

    def test_nonuniform_manufactured_state(self):
        """Per-cell varying h, hu, hv; verify the weighted sum by hand."""
        from swe2d.services.line_sampling_service import reference_line_flow

        h = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float64)
        hu = np.array([0.4, 0.6, 0.8, 0.2], dtype=np.float64)
        hv = np.array([0.1, 0.2, 0.3, 0.05], dtype=np.float64)
        cell_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        weights = np.array([2.0, 4.0, 6.0, 1.0], dtype=np.float64)
        nx_, ny_ = 0.6, 0.8
        wet = h > 1e-4
        qn = hu * nx_ + hv * ny_
        expected = float(np.sum(weights * np.where(wet, qn, 0.0)))
        Q = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=nx_, normal_y=ny_,
            h_min=1e-4,
        )
        self.assertAlmostEqual(float(Q), expected, places=12)

    def test_reversed_orientation_flips_signed_flow(self):
        """Reversing the line normal flips the signed discharge; the
        unsigned magnitude is preserved.
        """
        from swe2d.services.line_sampling_service import reference_line_flow

        h = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        hu = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        hv = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        cell_idx = np.array([0, 1, 2], dtype=np.int32)
        weights = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        q_fwd = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=0.0, normal_y=1.0, h_min=1e-4,
        )
        q_rev = reference_line_flow(
            h=h, hu=hu, hv=hv,
            cell_idx=cell_idx, weights=weights,
            normal_x=0.0, normal_y=-1.0, h_min=1e-4,
        )
        self.assertAlmostEqual(float(q_fwd), 60.0, places=12)
        self.assertAlmostEqual(float(q_rev), -60.0, places=12)


# ---------------------------------------------------------------------------
# Direct tuple/QGSGeometry regression (spec §2 / §12)
# ---------------------------------------------------------------------------
class TestLineSamplingMapTupleRegression(unittest.TestCase):
    """Direct regression for the historical tuple/QgsGeometry mismatch
    (spec §2 / §12).

    The legacy ``build_line_sampling_map`` consumed
    ``mesh_cell_polygons_fn`` expecting each element to be a
    ``QgsGeometry`` polygon (with ``.isEmpty()``, ``.boundingBox()``,
    ``.intersection()``, etc.).  The plain-tuple shape produced by
    ``swe2d.mesh.mesh_runtime_logic.mesh_cell_polygons`` — and therefore
    passed by ``build_line_sampling_map_from_gpkg`` — used to explode
    at the first cell when this code path was exercised.

    Task 8 of the canonical sample-line plan removed
    ``build_line_sampling_map``; the canonical service is now the only
    path, and it must accept the legacy ``(xs, ys)`` tuple shape
    without a runtime type fallback.  This test pins that contract.
    """

    def test_canonical_service_consumes_plain_tuples_without_attribute_error(self):
        """Direct regression for spec §2 / §12.

        The canonical ``build_canonical_line_sampling_map`` must accept
        plain ``(xs, ys)`` tuple ``points`` arrays without raising the
        legacy ``AttributeError: 'tuple' object has no attribute
        'isEmpty'`` regression.
        """
        from swe2d.services.line_sampling_service import (
            build_canonical_line_sampling_map,
        )

        # The canonical plain-data path must accept exactly the
        # ``(xs, ys)`` tuple shape that
        # ``swe2d.mesh.mesh_runtime_logic.mesh_cell_polygons`` returns
        # today — that is the documented bug.  No
        # ``isEmpty()``, no ``boundingBox()``, no ``intersection()`` — just
        # plain coordinate arrays, which is what the bug fires on.

        try:
            result = build_canonical_line_sampling_map(
                sample_lines=[],
                mesh_cells=[{
                    "cell_idx": 0,
                    "points": (
                        np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float64),
                        np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64),
                    ),
                }],
            )
        except AttributeError as e:
            # The bug would fire if the canonical service tried to call
            # .isEmpty() on a tuple.  The plain-data path does not.
            self.fail(
                f"tuple/QGSGeometry regression still active: {e}. "
                f"Spec §2 + §12 require the canonical service to "
                f"consume plain data without a runtime type fallback."
            )
        # After the fix the canonical service must at least return a
        # well-formed list.
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# GUI/GPKG adapter parity (spec §8.2)
# ---------------------------------------------------------------------------
class TestGuiGpkgAdapterParity(unittest.TestCase):
    """GUI and GPKG adapters must feed identical plain records to the service."""

    @staticmethod
    def _line_fields() -> Any:
        class _Fields:
            @staticmethod
            def names():
                return ["line_id", "name", "enabled"]

        return _Fields()

    @staticmethod
    def _feature(
        feature_id: int,
        points: Sequence[Tuple[float, float]],
        *,
        line_id: int,
        name: str,
        enabled: int,
    ) -> Any:
        class _QgisGeometry:
            def isEmpty(self):
                return False

            def asPolyline(self):
                from qgis.core import QgsPointXY

                return [QgsPointXY(float(x), float(y)) for x, y in points]

        class _Feature:
            def id(self):
                return feature_id

            def geometry(self):
                return _QgisGeometry()

            def __getitem__(self, key):
                return {
                    "line_id": line_id,
                    "name": name,
                    "enabled": enabled,
                }[key]

        return _Feature()

    def _line_layer(self) -> Any:
        feature = self._feature(
            17,
            [(5.0, 5.0), (25.0, 5.0)],
            line_id=7,
            name="transect_A",
            enabled=1,
        )

        class _Layer:
            @staticmethod
            def fields():
                return TestGuiGpkgAdapterParity._line_fields()

            @staticmethod
            def getFeatures():
                return iter([feature])

            @staticmethod
            def featureCount():
                return 1

            @staticmethod
            def name():
                return "sample_lines"

        return _Layer()

    def test_gui_and_gpkg_adapters_produce_equal_canonical_arrays(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from swe2d.core.gpkg_io import build_line_sampling_map_from_gpkg
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

        mesh_data = _make_rect_mesh_data(nx=3, ny=2, Lx=30.0, Ly=10.0)
        gui_layer = self._line_layer()
        dialog = MagicMock()
        dialog._mesh_data = mesh_data
        dialog._model_tab_view = SimpleNamespace(sample_lines_layer_combo=object())
        dialog._combo_layer = lambda combo, expected_kind: gui_layer
        dialog._sample_line_records_from_layer = (
            lambda layer: SWE2DWorkbenchStudioDialog._sample_line_records_from_layer(
                dialog, layer
            )
        )
        dialog._log = lambda message: None

        gui_map = SWE2DWorkbenchStudioDialog._build_line_sampling_map(dialog)

        gpkg_feature = self._feature(
            17,
            [(5.0, 5.0), (25.0, 5.0)],
            line_id=7,
            name="transect_A",
            enabled=1,
        )

        gpkg_layer = MagicMock()
        gpkg_layer.fields.return_value = self._line_fields()
        gpkg_layer.getFeatures.return_value = iter([gpkg_feature])
        with patch(
            "swe2d.core.gpkg_io._open_gpkg_layer", return_value=gpkg_layer
        ):
            gpkg_map = build_line_sampling_map_from_gpkg(
                gpkg_path="/tmp/parity.gpkg",
                sample_lines_table="swe2d_sample_lines",
                mesh_data=mesh_data,
                log_fn=lambda message: None,
            )

        self.assertEqual(len(gui_map), len(gpkg_map))
        self.assertEqual(gui_map[0]["line_id"], gpkg_map[0]["line_id"])
        self.assertEqual(gui_map[0]["line_name"], gpkg_map[0]["line_name"])
        for key in ("cell_idx", "weights", "station_m"):
            self.assertTrue(
                np.array_equal(gui_map[0][key], gpkg_map[0][key]),
                f"GUI/GPKG mismatch for {key}: {gui_map[0][key]!r} vs "
                f"{gpkg_map[0][key]!r}",
            )
        self.assertAlmostEqual(gui_map[0]["normal_x"], gpkg_map[0]["normal_x"])
        self.assertAlmostEqual(gui_map[0]["normal_y"], gpkg_map[0]["normal_y"])
    def test_gui_sampling_method_has_no_mesh_qgis_geometry_path(self) -> None:
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

        source = __import__("inspect").getsource(
            SWE2DWorkbenchStudioDialog._build_line_sampling_map
        )
        self.assertNotIn("QgsGeometry", source)
        self.assertNotIn("mesh_cell_polygons", source)
        self.assertIn("build_canonical_line_sampling_map", source)


# ---------------------------------------------------------------------------
# Executor flatten contract (spec §5/§6, Task 5)
# ---------------------------------------------------------------------------
class TestExecutorFlattenContract(unittest.TestCase):
    """Executor contract: flatten canonical sample-line map to GPU arrays.

    Reads only the canonical fields produced by
    ``build_canonical_line_sampling_map`` and produces the exact
    ``station_offsets``, ``cell_idx``, ``weights``, ``normal_x``,
    ``normal_y``, ``station_m`` arrays sent to the GPU backend.
    Preserves line ordering and line IDs (consumed by
    ``SWE2DResultsData.populate_live_line_metrics_from_gpu``).
    """

    @staticmethod
    def _canonical_lines():
        return [
            {
                "line_id": 7,
                "line_name": "transect_A",
                "normal_x": 0.0,
                "normal_y": 1.0,
                "cell_idx": np.array([0, 1, 2], dtype=np.int32),
                "weights": np.array([5.0, 10.0, 5.0], dtype=np.float64),
                "station_m": np.array([2.5, 10.0, 17.5], dtype=np.float64),
            },
            {
                "line_id": 13,
                "line_name": "transect_B",
                "normal_x": -1.0,
                "normal_y": 0.0,
                "cell_idx": np.array([4, 5], dtype=np.int32),
                "weights": np.array([3.0, 7.0], dtype=np.float64),
                "station_m": np.array([1.0, 4.0], dtype=np.float64),
            },
        ]

    def test_flatten_helper_is_exported(self):
        from swe2d.services import line_sampling_service as svc

        self.assertTrue(
            hasattr(svc, "flatten_canonical_sample_line_map"),
            "Executor flatten helper is missing from "
            "swe2d.services.line_sampling_service (Task 5).",
        )

    def test_flatten_matches_executor_arrays(self):
        """The exact flattened arrays the executor ships to the backend."""
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        sample_map = self._canonical_lines()
        (
            station_offsets,
            cell_idx_arr,
            weights_arr,
            normal_x_arr,
            normal_y_arr,
            station_m_arr,
            line_ids_ordered,
            line_names_by_id,
        ) = flatten_canonical_sample_line_map(sample_map, n_cells=6)

        # Per-line flat dtypes.
        self.assertEqual(cell_idx_arr.dtype, np.int32)
        self.assertEqual(weights_arr.dtype, np.float64)
        self.assertEqual(normal_x_arr.dtype, np.float64)
        self.assertEqual(normal_y_arr.dtype, np.float64)
        self.assertEqual(station_m_arr.dtype, np.float64)
        self.assertEqual(station_offsets.dtype, np.int32)
        # Per-line cumulative offsets.
        self.assertTrue(
            np.array_equal(station_offsets, np.array([0, 3, 5], dtype=np.int32)),
            f"station_offsets was {station_offsets!r}",
        )
        # Flat arrays match the canonical input, in iteration order.
        self.assertTrue(
            np.array_equal(
                cell_idx_arr, np.array([0, 1, 2, 4, 5], dtype=np.int32),
            )
        )
        self.assertTrue(
            np.allclose(
                weights_arr, np.array([5.0, 10.0, 5.0, 3.0, 7.0]),
            )
        )
        self.assertTrue(
            np.allclose(
                station_m_arr, np.array([2.5, 10.0, 17.5, 1.0, 4.0]),
            )
        )
        # Per-line normal is broadcast to per-station.
        self.assertTrue(
            np.allclose(normal_x_arr, np.array([0.0, 0.0, 0.0, -1.0, -1.0])),
            f"normal_x_arr was {normal_x_arr!r}",
        )
        self.assertTrue(
            np.allclose(normal_y_arr, np.array([1.0, 1.0, 1.0, 0.0, 0.0])),
            f"normal_y_arr was {normal_y_arr!r}",
        )

    def test_configure_path_sends_exact_arrays_to_backend(self):
        """The executor's shared setup path must send exact canonical arrays."""
        from swe2d.services.line_sampling_service import (
            configure_canonical_sample_line_map,
        )

        class _FakeBackend:
            def __init__(self):
                self.kwargs = None

            def configure_line_sampling(self, **kwargs):
                self.kwargs = kwargs

        backend = _FakeBackend()
        configured = configure_canonical_sample_line_map(
            backend=backend,
            sample_map=self._canonical_lines(),
            n_cells=6,
            gravity=9.81,
            h_min=1.0e-4,
        )

        self.assertIsNotNone(backend.kwargs)
        expected_arrays = {
            "station_offsets": np.array([0, 3, 5], dtype=np.int32),
            "cell_idx": np.array([0, 1, 2, 4, 5], dtype=np.int32),
            "weights": np.array([5.0, 10.0, 5.0, 3.0, 7.0], dtype=np.float64),
            "normal_x": np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float64),
            "normal_y": np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64),
            "station_m": np.array([2.5, 10.0, 17.5, 1.0, 4.0], dtype=np.float64),
        }
        for key, expected in expected_arrays.items():
            np.testing.assert_array_equal(backend.kwargs[key], expected)
        self.assertEqual(backend.kwargs["gravity"], 9.81)
        self.assertEqual(backend.kwargs["h_min"], 1.0e-4)
        self.assertEqual(configured[6], [7, 13])
        self.assertEqual(
            configured[7], {7: "transect_A", 13: "transect_B"}
        )

    def test_line_ordering_and_ids_are_preserved(self):
        """The executor's line_ids_ordered + line_names_by_id must mirror
        iteration order so populate_live_line_metrics_from_gpu maps GPU
        indices back to user-visible line IDs.
        """
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        # Reverse the input order — flatten must NOT re-sort.
        sample_map = list(reversed(self._canonical_lines()))
        (
            _station_offsets,
            _cell_idx_arr,
            _weights_arr,
            _normal_x_arr,
            _normal_y_arr,
            _station_m_arr,
            line_ids_ordered,
            line_names_by_id,
        ) = flatten_canonical_sample_line_map(sample_map, n_cells=6)

        self.assertEqual(line_ids_ordered, [13, 7])
        self.assertEqual(line_names_by_id, {13: "transect_B", 7: "transect_A"})

    def test_consistent_lengths_are_accepted(self):
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        sample_map = self._canonical_lines()
        # Should not raise — cell_idx, weights, station_m all 3 for line 7,
        # all 2 for line 13.
        (
            station_offsets,
            cell_idx_arr,
            weights_arr,
            _,
            _,
            station_m_arr,
            _,
            _,
        ) = flatten_canonical_sample_line_map(sample_map, n_cells=6)
        # Per-line lengths must match within each line (spec §6).
        self.assertEqual(station_offsets.size, 3)
        self.assertEqual(cell_idx_arr.size, 5)
        self.assertEqual(weights_arr.size, 5)
        self.assertEqual(station_m_arr.size, 5)

    def test_inconsistent_lengths_raise_no_silent_clipping(self):
        """No silent clipping — per-line length mismatch must raise."""
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        sample_map = [{
            "line_id": 7,
            "line_name": "bad",
            "normal_x": 0.0,
            "normal_y": 1.0,
            "cell_idx": np.array([0, 1, 2], dtype=np.int32),
            "weights": np.array([5.0, 10.0], dtype=np.float64),  # wrong length
            "station_m": np.array([2.5, 10.0, 17.5], dtype=np.float64),
        }]
        with self.assertRaises(ValueError) as cm:
            flatten_canonical_sample_line_map(sample_map, n_cells=6)
        self.assertIn("inconsistent", str(cm.exception).lower())

    def test_out_of_range_indices_raise_no_silent_clipping(self):
        """No silent clipping — cell_idx outside [0, n_cells) must raise."""
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        sample_map = [{
            "line_id": 7,
            "line_name": "bad",
            "normal_x": 0.0,
            "normal_y": 1.0,
            "cell_idx": np.array([0, 1, 6], dtype=np.int32),  # 6 out of range
            "weights": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "station_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        }]
        with self.assertRaises(ValueError) as cm:
            flatten_canonical_sample_line_map(sample_map, n_cells=6)
        self.assertIn("out-of-range", str(cm.exception).lower())

    def test_negative_cell_idx_raises(self):
        """cell_idx < 0 must raise — negative indices are not legal
        unsigned int32 GPU input and must never be silently coerced.
        """
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        sample_map = [{
            "line_id": 7,
            "line_name": "bad",
            "normal_x": 0.0,
            "normal_y": 1.0,
            "cell_idx": np.array([-1, 0], dtype=np.int32),
            "weights": np.array([1.0, 1.0], dtype=np.float64),
            "station_m": np.array([0.0, 1.0], dtype=np.float64),
        }]
        with self.assertRaises(ValueError):
            flatten_canonical_sample_line_map(sample_map, n_cells=6)

    def test_cell_idx_must_be_integral_and_exactly_int32_representable(self):
        """Fractional and int32-overflow values must fail before casting."""
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        base = {
            "line_id": 7,
            "line_name": "bad",
            "normal_x": 0.0,
            "normal_y": 1.0,
            "weights": np.array([1.0], dtype=np.float64),
            "station_m": np.array([0.0], dtype=np.float64),
        }
        invalid_cell_indices = (
            np.array([2**32 + 1], dtype=np.int64),
            np.array([-(2**32)], dtype=np.int64),
            np.array([1.5], dtype=np.float64),
        )
        for cell_idx in invalid_cell_indices:
            with self.subTest(cell_idx=cell_idx):
                with self.assertRaises(ValueError):
                    flatten_canonical_sample_line_map(
                        [dict(base, cell_idx=cell_idx)], n_cells=6,
                    )

    def test_missing_canonical_field_raises_no_silent_default(self):
        """Missing canonical field must raise KeyError, not silently
        substitute a default — the previous executor flatten silently
        used ``[]``/``0.0``/``1.0`` for missing fields (Task 5).
        """
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        sample_map = [{
            "line_id": 7,
            "line_name": "incomplete",
            "normal_x": 0.0,
            "normal_y": 1.0,
            # cell_idx missing
            "weights": np.array([1.0], dtype=np.float64),
            "station_m": np.array([0.0], dtype=np.float64),
        }]
        with self.assertRaises(KeyError):
            flatten_canonical_sample_line_map(sample_map, n_cells=6)

    def test_empty_sample_map_returns_empty_arrays(self):
        """No lines → empty arrays, station_offsets = [0]."""
        from swe2d.services.line_sampling_service import (
            flatten_canonical_sample_line_map,
        )

        (
            station_offsets,
            cell_idx_arr,
            weights_arr,
            normal_x_arr,
            normal_y_arr,
            station_m_arr,
            line_ids_ordered,
            line_names_by_id,
        ) = flatten_canonical_sample_line_map([], n_cells=4)

        self.assertTrue(
            np.array_equal(station_offsets, np.array([0], dtype=np.int32))
        )
        self.assertEqual(cell_idx_arr.size, 0)
        self.assertEqual(weights_arr.size, 0)
        self.assertEqual(normal_x_arr.size, 0)
        self.assertEqual(normal_y_arr.size, 0)
        self.assertEqual(station_m_arr.size, 0)
        self.assertEqual(line_ids_ordered, [])
        self.assertEqual(line_names_by_id, {})

    def test_executor_uses_configure_helper(self):
        """Executor must delegate canonical setup to the configure helper."""
        import inspect

        from swe2d.core import executor as _executor

        source = inspect.getsource(_executor.execute_run)
        self.assertIn(
            "configure_canonical_sample_line_map",
            source,
            "executor.execute_run must call the canonical configure helper.",
        )
        # Old inline silent-fallback pattern (sm.get("..._x", 0.0)) must
        # be gone from the execute_run line-sampling setup path.
        self.assertNotIn(
            'sm.get("normal_x", 0.0)', source,
            "executor must not silently default normal_x to 0.0.",
        )
        self.assertNotIn(
            'sm.get("cell_idx", [])', source,
            "executor must not silently default cell_idx to [].",
        )


# ---------------------------------------------------------------------------
# Byte / schema compatibility for baked line tables (Task 5)
# ---------------------------------------------------------------------------
class TestBakedLineTableSchemaCompat(unittest.TestCase):
    """The persist_*_line_ts and persist_*_line_profile writers must
    produce ``swe2d_baked_line_ts`` and ``swe2d_baked_line_profiles``
    tables whose schema (column set, order, types) and BLOBs are
    byte-compatible with the loaders (``load_baked_line_timeseries``,
    ``load_baked_line_profile``).

    Task 5 calls for proving persistence/readback remains byte/schema
    compatible without an unrelated schema change.
    """

    _LINE_TS_COLUMNS = (
        "run_id", "line_id", "line_name", "n_timesteps",
        "times_blob", "depth_blob", "vel_blob", "wse_blob",
        "bed_blob", "flow_blob", "wet_frac_blob", "fr_blob",
    )
    _LINE_TS_TYPES = (
        "TEXT", "INTEGER", "TEXT", "INTEGER",
        "BLOB", "BLOB", "BLOB", "BLOB",
        "BLOB", "BLOB", "BLOB", "BLOB",
    )
    _LINE_TS_PRIMARY_KEY = (1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    _LINE_PROFILE_COLUMNS = (
        "run_id", "line_id", "line_name", "n_stations", "n_timesteps",
        "station_blob", "times_blob", "depth_blob", "vel_blob",
        "wse_blob", "bed_blob", "flow_qn_blob", "fr_blob", "wet_blob",
    )
    _LINE_PROFILE_TYPES = (
        "TEXT", "INTEGER", "TEXT", "INTEGER", "INTEGER",
        "BLOB", "BLOB", "BLOB", "BLOB", "BLOB", "BLOB", "BLOB",
        "BLOB", "BLOB",
    )
    _LINE_PROFILE_PRIMARY_KEY = (
        1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    )

    @staticmethod
    def _table_info(conn, table: str):
        cur = conn.execute(f"PRAGMA table_info({table})")
        return tuple(cur.fetchall())

    @classmethod
    def _assert_schema_contract(cls, conn, table, columns, types, primary_key):
        info = cls._table_info(conn, table)
        assert len(info) == len(columns)
        actual_columns = tuple(row[1] for row in info)
        actual_types = tuple(str(row[2]).upper() for row in info)
        actual_notnull = tuple(int(row[3]) for row in info)
        actual_pk = tuple(int(row[5]) for row in info)
        cls_expected_notnull = (0,) * len(columns)
        if actual_columns != columns:
            raise AssertionError((actual_columns, columns))
        if actual_types != types:
            raise AssertionError((actual_types, types))
        if actual_notnull != cls_expected_notnull:
            raise AssertionError((actual_notnull, cls_expected_notnull))
        if actual_pk != primary_key:
            raise AssertionError((actual_pk, primary_key))

    def _open_gpkg(self, path: str):
        import sqlite3

        return sqlite3.connect(path)

    def test_line_ts_table_schema_matches_loader_columns(self):
        """``swe2d_baked_line_ts`` columns must match the loader's SELECT."""
        from swe2d.services.gpkg_persistence_service import persist_baked_line_ts

        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        try:
            gpkg = os.path.join(tmpdir, "ts.gpkg")
            times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
            arr = np.zeros(3, dtype=np.float64)
            persist_baked_line_ts(
                gpkg, "run", 7, "ln",
                times, arr, arr, arr, arr, arr, arr, arr,
            )
            conn = self._open_gpkg(gpkg)
            try:
                self._assert_schema_contract(
                    conn,
                    "swe2d_baked_line_ts",
                    self._LINE_TS_COLUMNS,
                    self._LINE_TS_TYPES,
                    self._LINE_TS_PRIMARY_KEY,
                )
            finally:
                conn.close()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_line_profile_table_schema_matches_loader_columns(self):
        """``swe2d_baked_line_profiles`` columns must match the loader's SELECT."""
        from swe2d.services.gpkg_persistence_service import persist_baked_line_profile

        import tempfile, os, shutil

        tmpdir = tempfile.mkdtemp()
        try:
            gpkg = os.path.join(tmpdir, "profile.gpkg")
            station = np.array([0.0, 1.0], dtype=np.float64)
            times = np.array([0.0, 1.0], dtype=np.float64)
            arr2 = np.zeros((2, 2), dtype=np.float64)
            wet2 = np.zeros((2, 2), dtype=np.int32)
            persist_baked_line_profile(
                gpkg, "run", 7, "ln",
                station, times, arr2, arr2, arr2, arr2, arr2, arr2, wet2,
            )
            conn = self._open_gpkg(gpkg)
            try:
                self._assert_schema_contract(
                    conn,
                    "swe2d_baked_line_profiles",
                    self._LINE_PROFILE_COLUMNS,
                    self._LINE_PROFILE_TYPES,
                    self._LINE_PROFILE_PRIMARY_KEY,
                )
            finally:
                conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_line_ts_blob_byte_compatible_with_loader(self):
        """BLOBs written by ``persist_baked_line_ts`` must round-trip
        through ``load_baked_line_timeseries`` byte-for-byte.
        """
        from swe2d.services.gpkg_persistence_service import (
            load_baked_line_timeseries,
            persist_baked_line_ts,
        )

        import tempfile, os, shutil

        tmpdir = tempfile.mkdtemp()
        try:
            gpkg = os.path.join(tmpdir, "ts.gpkg")
            times = np.linspace(0.0, 60.0, 7, dtype=np.float64)
            depth = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3], dtype=np.float64)
            velocity = np.array([1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0], dtype=np.float64)
            wse = depth + 10.0
            bed = np.full(7, 10.0, dtype=np.float64)
            flow = depth * velocity
            wet_frac = np.linspace(1.0, 0.5, 7, dtype=np.float64)
            fr = np.linspace(0.1, 0.5, 7, dtype=np.float64)
            persist_baked_line_ts(
                gpkg, "byte_run", 11, "byte_ln",
                times, depth, velocity, wse, bed, flow, wet_frac, fr,
            )
            conn = self._open_gpkg(gpkg)
            try:
                stored_blobs = conn.execute(
                    "SELECT times_blob, depth_blob, vel_blob, wse_blob, "
                    "bed_blob, flow_blob, wet_frac_blob, fr_blob "
                    "FROM swe2d_baked_line_ts WHERE run_id=? AND line_id=?",
                    ("byte_run", 11),
                ).fetchone()
            finally:
                conn.close()
            expected_blobs = tuple(
                np.asarray(values, dtype=np.float64).tobytes()
                for values in (
                    times, depth, velocity, wse, bed, flow, wet_frac, fr,
                )
            )
            self.assertEqual(stored_blobs, expected_blobs)
            loaded = load_baked_line_timeseries(gpkg, "byte_run", 11)
            np.testing.assert_array_equal(loaded["t_s"], times)
            np.testing.assert_array_equal(loaded["depth"], depth)
            np.testing.assert_array_equal(loaded["velocity"], velocity)
            np.testing.assert_array_equal(loaded["wse"], wse)
            np.testing.assert_array_equal(loaded["bed"], bed)
            np.testing.assert_array_equal(loaded["flow"], flow)
            np.testing.assert_array_equal(loaded["wet_frac"], wet_frac)
            np.testing.assert_array_equal(loaded["fr"], fr)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_line_profile_blob_byte_compatible_with_loader(self):
        """BLOBs written by ``persist_baked_line_profile`` must round-trip
        through ``load_baked_line_profile`` byte-for-byte.
        """
        from swe2d.services.gpkg_persistence_service import (
            load_baked_line_profile,
            persist_baked_line_profile,
        )

        import tempfile, os, shutil

        tmpdir = tempfile.mkdtemp()
        try:
            gpkg = os.path.join(tmpdir, "profile.gpkg")
            station = np.array([0.0, 5.0, 10.0], dtype=np.float64)
            times = np.array([0.0, 60.0], dtype=np.float64)
            n_ts, n_sta = times.size, station.size
            depth = np.tile([0.1, 0.5, 0.9], (n_ts, 1)).astype(np.float64)
            velocity = np.tile([0.5, 1.0, 1.5], (n_ts, 1)).astype(np.float64)
            wse = depth + 5.0
            bed = np.full((n_ts, n_sta), 5.0, dtype=np.float64)
            flow_qn = depth * velocity
            fr = np.full((n_ts, n_sta), 0.3, dtype=np.float64)
            wet = np.ones((n_ts, n_sta), dtype=np.int32)
            persist_baked_line_profile(
                gpkg, "byte_run", 11, "byte_ln",
                station, times, depth, velocity, wse, bed, flow_qn, fr, wet,
            )
            conn = self._open_gpkg(gpkg)
            try:
                stored_blobs = conn.execute(
                    "SELECT station_blob, times_blob, depth_blob, vel_blob, "
                    "wse_blob, bed_blob, flow_qn_blob, fr_blob, wet_blob "
                    "FROM swe2d_baked_line_profiles "
                    "WHERE run_id=? AND line_id=?",
                    ("byte_run", 11),
                ).fetchone()
            finally:
                conn.close()
            expected_blobs = tuple(
                np.asarray(values, dtype=dtype).tobytes()
                for values, dtype in (
                    (station, np.float64),
                    (times, np.float64),
                    (depth, np.float64),
                    (velocity, np.float64),
                    (wse, np.float64),
                    (bed, np.float64),
                    (flow_qn, np.float64),
                    (fr, np.float64),
                    (wet, np.int32),
                )
            )
            self.assertEqual(stored_blobs, expected_blobs)
            loaded = load_baked_line_profile(gpkg, "byte_run", 11, t_sec=0.0)
            np.testing.assert_array_equal(loaded["station"], station)
            np.testing.assert_array_equal(loaded["depth"], depth[0])
            np.testing.assert_array_equal(loaded["velocity"], velocity[0])
            np.testing.assert_array_equal(loaded["wse"], wse[0])
            np.testing.assert_array_equal(loaded["bed"], bed[0])
            np.testing.assert_array_equal(loaded["flow_qn"], flow_qn[0])
            np.testing.assert_array_equal(loaded["fr"], fr[0])
            np.testing.assert_array_equal(loaded["wet"], wet[0])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
