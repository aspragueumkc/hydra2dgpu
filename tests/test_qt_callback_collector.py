#!/usr/bin/env python3
"""Small self-tests for the gui-behavioral callback-error collector."""
from __future__ import annotations

import contextlib
import io
import unittest

from qgis.PyQt import QtCore

from tests import fail_on_qt_callback_errors as collector


class TestQtCallbackErrorCollector(unittest.TestCase):
    def setUp(self):
        collector.clear_callback_errors()
        self.addCleanup(collector.clear_callback_errors)

    def test_python_callback_exception_is_recorded_and_fails(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            try:
                raise TypeError("synthetic callback failure")
            except TypeError as exc:
                collector.record_python_exception(type(exc), exc, exc.__traceback__)

            self.assertEqual(len(collector.callback_errors()), 1)
            with self.assertRaises(SystemExit) as raised:
                collector.fail_if_callback_errors()
            self.assertEqual(raised.exception.code, 1)

        self.assertIn("synthetic callback failure", stderr.getvalue())

    def test_benign_qt_warning_is_not_a_callback_failure(self):
        collector.handle_qt_message(
            QtCore.QtWarningMsg,
            None,
            "This plugin does not support propagateSizeHints()",
        )
        self.assertEqual(collector.callback_errors(), ())


if __name__ == "__main__":
    unittest.main()
