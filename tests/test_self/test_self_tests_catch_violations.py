"""Meta-test: each self-test must actually catch its target violation.

Spec: docs/specs/2026-07-26-test-discipline-design.md §6.1

This is the bootstrap that makes the fast-fail set trustworthy. A self-test
that doesn't fail on a real violation is the worst kind of test theatre:
it pretends to enforce a rule while letting violations through.

Strategy: for each self-test, create a synthetic violation in a tempdir
and verify the corresponding detector (grep, AST check, or subprocess
invocation) catches it. This is a "positive control" in the testing-the-tests
sense.
"""
import ast
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


class TestSelfTestsCatchViolations(unittest.TestCase):
    """Each test creates a synthetic violation in a tempdir and asserts the
    self-test's detection logic (a grep, AST check, or subprocess run) flags it.
    """

    def test_mvp_rule1_violation_caught(self):
        """A controller reaching into view._model_tab_view should match the
        MVP Rule 1 grep."""
        with tempfile.TemporaryDirectory() as tmp:
            controller = os.path.join(tmp, "fake_controller.py")
            with open(controller, "w") as f:
                f.write(textwrap.dedent("""\
                    class FakeController:
                        def go(self, view):
                            return view._model_tab_view.foo
                """))
            result = subprocess.run(
                ["grep", "-rnE", r"view\._model_tab_view\.", controller],
                capture_output=True,
            )
            self.assertEqual(
                result.returncode, 0,
                "Synthetic MVP Rule 1 violation should match the grep. "
                "If it doesn't, the test_mvp_grep_rules self-test isn't catching "
                "the violation it claims to catch.",
            )

    def test_stale_skip_caught(self):
        """A test file with @unittest.skip('retired') should be caught."""
        with tempfile.TemporaryDirectory() as tmp:
            test_file = os.path.join(tmp, "test_x.py")
            with open(test_file, "w") as f:
                f.write(textwrap.dedent("""\
                    import unittest

                    @unittest.skip("retired: do not use")
                    class TestX(unittest.TestCase):
                        def test_y(self):
                            self.assertTrue(True)
                """))
            result = subprocess.run(
                ["grep", "-rnE",
                 r"@unittest\.skip\(\s*['\"](retired|deprecated|TODO|FIXME)",
                 test_file],
                capture_output=True,
            )
            self.assertEqual(
                result.returncode, 0,
                "Synthetic stale skip should match the grep.",
            )

    def test_broken_import_caught(self):
        """A test file that imports a non-existent module should be caught by
        the test_collect_only self-test's subprocess."""
        with tempfile.TemporaryDirectory() as tmp:
            broken_test = os.path.join(tmp, "test_broken.py")
            with open(broken_test, "w") as f:
                f.write(textwrap.dedent("""\
                    import unittest
                    from swe2d.nonexistent_module_xyz import Foo

                    class TestBroken(unittest.TestCase):
                        def test_x(self):
                            self.assertTrue(True)
                """))
            result = subprocess.run(
                ["python3", "-c",
                 "import sys, unittest\n"
                 "try:\n"
                 "    loader = unittest.TestLoader()\n"
                 "    suite = loader.discover('" + tmp + "', pattern='test_*.py', top_level_dir='.')\n"
                 "except Exception as e:\n"
                 "    print(f'Collection failed: {e}', file=sys.stderr)\n"
                 "    sys.exit(1)\n"
                 "print(f'Collected {suite.countTestCases()} tests OK')\n"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(
                result.returncode, 0,
                f"`TestLoader.discover()` should fail on broken imports. "
                f"If it doesn't, the test_collect_only self-test isn't catching the "
                f"violation it claims to catch. rc={result.returncode}, stderr={result.stderr[-500:]}",
            )

    def test_assertTrue_after_assertRaises_caught(self):
        """A test that does `with assertRaises(): pass; assertTrue(True)` is theatre."""
        from tests.test_self.test_no_test_theatre import _has_after_context_noop
        with tempfile.TemporaryDirectory() as tmp:
            test_file = os.path.join(tmp, "test_x.py")
            with open(test_file, "w") as f:
                f.write(textwrap.dedent("""\
                    import unittest

                    class TestTheatre(unittest.TestCase):
                        def test_noop(self):
                            with self.assertRaises(ValueError):
                                pass
                            self.assertTrue(True)
                """))
            with open(test_file) as f:
                tree = ast.parse(f.read(), filename=test_file)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "test_noop":
                    self.assertTrue(
                        _has_after_context_noop(node),
                        "Synthetic no-op should be detected by _has_after_context_noop.",
                    )
                    return
            self.fail("test_noop method not found in synthetic fixture")


if __name__ == "__main__":
    unittest.main(verbosity=2)
