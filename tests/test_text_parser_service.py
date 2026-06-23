"""Tests for the pure text parser service.

Verifies that ``swe2d.workbench.text_parser_service`` provides:
- ``parse_time_hours(token)`` -> ``float`` (raises on invalid input)
- ``parse_hydrograph_text(text)`` -> ``tuple[np.ndarray, np.ndarray]``
  (raises on invalid input; no silent fallbacks)

The service must have ZERO Qt imports.  The workbench dialog uses
``_parse_time_hours`` and ``_parse_hydrograph_text`` methods; once the
service is wired in (Task 4), the dialog methods will delegate here.
"""
from __future__ import annotations

import ast
import os
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_PATH = os.path.join(
    REPO_ROOT, "swe2d", "workbench", "services", "text_parser_service.py"
)


def _load_module():
    """Import the service module (post-impl)."""
    from swe2d.workbench.services import text_parser_service
    return text_parser_service


class TestNoQtImports(unittest.TestCase):
    """The service must be a pure-Python module — zero Qt imports."""

    def test_module_file_exists(self):
        self.assertTrue(
            os.path.isfile(SERVICE_PATH),
            f"text_parser_service.py not found at {SERVICE_PATH}",
        )

    def test_module_has_no_qt_imports(self):
        with open(SERVICE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        forbidden = {
            "PyQt5", "PyQt4", "PySide2", "PySide6",
            "qgis", "qgis.PyQt", "qgis.core", "qgis.gui",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.assertNotIn(
                        top, forbidden,
                        f"forbidden Qt/QGIS import: {alias.name}",
                    )
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                self.assertNotIn(
                    top, forbidden,
                    f"forbidden Qt/QGIS from-import: {node.module}",
                )

    def test_module_exports_required_symbols(self):
        mod = _load_module()
        self.assertTrue(hasattr(mod, "parse_time_hours"))
        self.assertTrue(hasattr(mod, "parse_hydrograph_text"))


class TestParseTimeHours(unittest.TestCase):
    """parse_time_hours must return hours as float, or raise ValueError."""

    def setUp(self):
        self.parse_time_hours = _load_module().parse_time_hours

    def test_plain_float_hours(self):
        self.assertAlmostEqual(self.parse_time_hours("0.5"), 0.5)

    def test_integer_hours(self):
        self.assertAlmostEqual(self.parse_time_hours("2"), 2.0)

    def test_strips_whitespace(self):
        self.assertAlmostEqual(self.parse_time_hours("  1.25  "), 1.25)

    def test_hh_colon_mm(self):
        self.assertAlmostEqual(self.parse_time_hours("1:30"), 1.5)

    def test_hh_colon_mm_with_whitespace(self):
        self.assertAlmostEqual(self.parse_time_hours(" 1 : 30 "), 1.5)

    def test_hh_colon_mm_colon_ss(self):
        # 1h 30m 30s == 1 + 30/60 + 30/3600 == 1.508333...
        self.assertAlmostEqual(
            self.parse_time_hours("1:30:30"),
            1.0 + 30.0 / 60.0 + 30.0 / 3600.0,
            places=9,
        )

    def test_zero_hours(self):
        self.assertAlmostEqual(self.parse_time_hours("0"), 0.0)

    def test_zero_hh_mm(self):
        self.assertAlmostEqual(self.parse_time_hours("0:00"), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            self.parse_time_hours("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            self.parse_time_hours("   ")

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            self.parse_time_hours("abc")

    def test_invalid_hhmm_token_raises(self):
        with self.assertRaises(ValueError):
            self.parse_time_hours("1:xx")

    def test_too_many_colons_raises(self):
        with self.assertRaises(ValueError):
            self.parse_time_hours("1:2:3:4")

    def test_returns_float(self):
        self.assertIsInstance(self.parse_time_hours("0.5"), float)


class TestParseHydrographText(unittest.TestCase):
    """parse_hydrograph_text returns (times_s, values) ndarrays or raises."""

    def setUp(self):
        self.parse_hydrograph_text = _load_module().parse_hydrograph_text

    def test_simple_comma_separated(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("0,0;1,10")
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(t.dtype, np.float64)
        self.assertEqual(v.dtype, np.float64)
        np.testing.assert_array_equal(t, np.array([0.0, 3600.0]))
        np.testing.assert_array_equal(v, np.array([0.0, 10.0]))

    def test_equals_separator(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("0=0;1=10")
        np.testing.assert_array_equal(t, np.array([0.0, 3600.0]))
        np.testing.assert_array_equal(v, np.array([0.0, 10.0]))

    def test_newline_separator(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("0,0\n1,10")
        np.testing.assert_array_equal(t, np.array([0.0, 3600.0]))
        np.testing.assert_array_equal(v, np.array([0.0, 10.0]))

    def test_hhmm_time_token(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("0:30,5")
        np.testing.assert_array_equal(t, np.array([1800.0]))
        np.testing.assert_array_equal(v, np.array([5.0]))

    def test_sorts_by_time(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("2,20;0,0;1,10")
        np.testing.assert_array_equal(t, np.array([0.0, 3600.0, 7200.0]))
        np.testing.assert_array_equal(v, np.array([0.0, 10.0, 20.0]))

    def test_dedupes_close_times(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("0,1;0,2;0,3")
        np.testing.assert_array_equal(t, np.array([0.0]))
        np.testing.assert_array_equal(v, np.array([3.0]))

    def test_extra_whitespace_tolerated(self):
        import numpy as np
        t, v = self.parse_hydrograph_text("  0 , 0 ; 1 , 10  ")
        np.testing.assert_array_equal(t, np.array([0.0, 3600.0]))
        np.testing.assert_array_equal(v, np.array([0.0, 10.0]))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            self.parse_hydrograph_text("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            self.parse_hydrograph_text("   \n\t  ")

    def test_no_valid_entries_raises(self):
        with self.assertRaises(ValueError):
            self.parse_hydrograph_text(";;;")

    def test_invalid_chunk_no_separator_raises(self):
        with self.assertRaises(ValueError):
            self.parse_hydrograph_text("0_5")

    def test_invalid_time_raises(self):
        with self.assertRaises(ValueError):
            self.parse_hydrograph_text("abc,5")

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            self.parse_hydrograph_text("0,abc")

    def test_returns_tuple_of_ndarrays(self):
        result = self.parse_hydrograph_text("0,0;1,10")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        import numpy as np
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)


if __name__ == "__main__":
    unittest.main()
