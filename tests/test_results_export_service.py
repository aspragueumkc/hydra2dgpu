"""Tests for swe2d.results.export_service — pure CSV export, no Qt."""

import csv
import io
import os
import tempfile
import unittest

import numpy as np

from swe2d.results.export_service import export_table_to_csv, export_timeseries_to_csv


class TestExportTableToCsv(unittest.TestCase):
    def test_writes_headers_and_rows(self):
        headers = ["Name", "Value", "Unit"]
        rows = [
            ["h_max", "1.25", "m"],
            ["v_max", "3.0", "m/s"],
        ]
        buf = io.StringIO()
        export_table_to_csv(buf, headers, rows)
        buf.seek(0)
        reader = csv.reader(buf)
        self.assertEqual(next(reader), headers)
        self.assertEqual(next(reader), rows[0])
        self.assertEqual(next(reader), rows[1])

    def test_empty_rows(self):
        headers = ["A", "B"]
        buf = io.StringIO()
        export_table_to_csv(buf, headers, [])
        buf.seek(0)
        reader = csv.reader(buf)
        self.assertEqual(next(reader), headers)
        with self.assertRaises(StopIteration):
            next(reader)

    def test_empty_headers(self):
        buf = io.StringIO()
        export_table_to_csv(buf, [], [["a", "b"]])
        buf.seek(0)
        contents = buf.read()
        self.assertEqual(contents, "\r\n" + "a,b\r\n")

    def test_numeric_values_populated(self):
        headers = ["x", "y"]
        rows = [[1, 2.5], [3, 4.5]]
        buf = io.StringIO()
        export_table_to_csv(buf, headers, rows)
        buf.seek(0)
        reader = csv.reader(buf)
        self.assertEqual(next(reader), ["x", "y"])
        self.assertEqual(next(reader), ["1", "2.5"])
        self.assertEqual(next(reader), ["3", "4.5"])

    def test_writes_to_real_file(self):
        headers = ["Col1"]
        rows = [["val1"]]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            path = f.name
            export_table_to_csv(f, headers, rows)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            self.assertEqual(next(reader), headers)
            self.assertEqual(next(reader), rows[0])
        os.unlink(path)


class TestExportTimeseriesToCsv(unittest.TestCase):
    def test_writes_labels_as_header_and_data_rows(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = ["Time", "Stage"]
        buf = io.StringIO()
        export_timeseries_to_csv(buf, data, labels)
        buf.seek(0)
        content = buf.read()
        expected = "Time,Stage\r\n1.0,2.0\r\n3.0,4.0\r\n5.0,6.0\r\n"
        self.assertEqual(content, expected)

    def test_single_column(self):
        data = np.array([[1.0], [2.0], [3.0]])
        labels = ["Val"]
        buf = io.StringIO()
        export_timeseries_to_csv(buf, data, labels)
        buf.seek(0)
        content = buf.read()
        expected = "Val\r\n1.0\r\n2.0\r\n3.0\r\n"
        self.assertEqual(content, expected)

    def test_empty_data(self):
        data = np.empty((0, 2))
        labels = ["A", "B"]
        buf = io.StringIO()
        export_timeseries_to_csv(buf, data, labels)
        buf.seek(0)
        content = buf.read()
        self.assertEqual(content, "A,B\r\n")

    def test_writes_to_real_file(self):
        data = np.array([[0.0, 10.0], [1.0, 11.0]])
        labels = ["t", "h"]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
            path = f.name
            export_timeseries_to_csv(f, data, labels)
        with open(path, newline="") as f:
            content = f.read()
            expected = "t,h\r\n0.0,10.0\r\n1.0,11.0\r\n"
            self.assertEqual(content, expected)
        os.unlink(path)

    def test_integer_data_converted_to_string(self):
        data = np.array([[0, 5], [1, 6]])
        labels = ["idx", "val"]
        buf = io.StringIO()
        export_timeseries_to_csv(buf, data, labels)
        buf.seek(0)
        content = buf.read()
        self.assertEqual(content, "idx,val\r\n0,5\r\n1,6\r\n")


if __name__ == "__main__":
    unittest.main()
