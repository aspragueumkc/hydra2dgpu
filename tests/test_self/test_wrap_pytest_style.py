"""Verify tools/wrap_pytest_style.py correctly wraps pytest-style tests.

Spec: docs/specs/2026-07-26-test-discipline-design.md §6.3
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


class TestWrapPytestStyle(unittest.TestCase):
    def test_wrap_pytest_style_creates_testcase(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytest_file = os.path.join(tmp, "test_x.py")
            with open(pytest_file, "w") as f:
                f.write(textwrap.dedent("""\
                    def test_a():
                        assert 1 + 1 == 2

                    def test_b():
                        assert True is True
                """))
            # Apply the wrap (--path flag targets the tmpdir)
            result = subprocess.run(
                [sys.executable, "tools/wrap_pytest_style.py", "--path", tmp],
                capture_output=True, text=True, cwd=os.getcwd(),
            )
            self.assertEqual(result.returncode, 0, f"wrap failed: {result.stderr}")
            # Read the wrapped file
            with open(pytest_file) as f:
                wrapped = f.read()
            self.assertIn("_PytestStyleWrapper", wrapped,
                          "wrap did not add _PytestStyleWrapper class")
            self.assertIn("import unittest", wrapped,
                          "wrap did not add `import unittest`")
            # Verify the wrapper has the correct structure: it should
            # iterate globals and attach `test_*` functions as staticmethods.
            # The wrapper code itself is verified by fast_fail.sh Gate 1
            # on the real corpus; here we just verify the AST mutation
            # happened correctly.
            self.assertIn("setattr(_PytestStyleWrapper, _name, staticmethod(_obj))", wrapped)
            # Note: the wrap does NOT delete the original `def test_*` from the
            # static text on disk — the `del globals()[_name]` loop in the
            # wrapper template removes them from `globals()` at *runtime*
            # (when unittest imports the module). Static text remains.
            # Runtime behavior is verified by fast_fail.sh Gate 1 on the
            # real corpus.
            self.assertIn("def test_a():", wrapped,
                          "wrap should preserve the original def statements in text")
            self.assertIn("def test_b():", wrapped,
                          "wrap should preserve the original def statements in text")
            # Parse the wrapped file as a module to verify it's syntactically valid
            import ast as _ast
            try:
                _ast.parse(wrapped, filename=pytest_file)
            except SyntaxError as e:
                self.fail(f"wrapped file has syntax error: {e}")

    def test_wrap_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytest_file = os.path.join(tmp, "test_x.py")
            with open(pytest_file, "w") as f:
                f.write(textwrap.dedent("""\
                    def test_a():
                        assert True
                """))
            # First wrap
            subprocess.run(
                [sys.executable, "tools/wrap_pytest_style.py", "--path", tmp],
                check=True, cwd=os.getcwd(),
            )
            # Second wrap should be a no-op
            result = subprocess.run(
                [sys.executable, "tools/wrap_pytest_style.py", "--path", tmp],
                capture_output=True, text=True, cwd=os.getcwd(),
            )
            self.assertIn("0 file(s) changed", result.stdout,
                          f"Second wrap should be idempotent: {result.stdout}")


if __name__ == "__main__":
    unittest.main(verbosity=2)