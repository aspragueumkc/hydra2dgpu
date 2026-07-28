"""Collect-only gate + tests/test_self/ non-empty invariant.

Spec: docs/specs/2026-07-26-test-discipline-design.md §3.2
"""
import os
import subprocess
import sys
import unittest


class TestCollectOnly(unittest.TestCase):
    """The collect-only stage is Gate 1 of fast_fail.sh. It must pass before
    any other stage runs. A failure here means the test corpus won't even
    import — usually a worktree-merge artifact or a moved-source reference.
    """

    def test_test_self_directory_is_nonempty(self):
        """tests/test_self/ must always contain ≥5 test files.

        This prevents silent loss of the fast-fail signal: if someone deletes
        all self-tests, this test fails and the developer notices.
        """
        py_files = [f for f in os.listdir("tests/test_self/") if f.endswith(".py")]
        self.assertGreaterEqual(
            len(py_files),
            5,
            "tests/test_self/ must contain at least 5 test files "
            "(4 self-tests + 1 meta-test). Found: " + ", ".join(py_files),
        )

    def test_collect_only_passes(self):
        """All tests under tests/ must collect without import errors.

        This is the single most important gate. If this fails, no other
        test runs. The failure message includes the import error so the
        agent can find the broken import.

        Note: ``--collect-only`` was removed from ``unittest discover`` in
        Python 3.12. We use a small Python helper that calls
        ``unittest.TestLoader().discover()`` and exits non-zero on any
        collection-time error.
        """
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys, unittest\n"
                "try:\n"
                "    loader = unittest.TestLoader()\n"
                "    suite = loader.discover('tests', pattern='test_*.py', top_level_dir='.')\n"
                "except Exception as e:\n"
                "    print(f'Collection failed: {e}', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "print(f'Collected {suite.countTestCases()} tests OK')\n",
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Test collection failed (rc={result.returncode}). "
            f"This indicates broken imports — usually a worktree-merge "
            f"artifact or a moved-source reference. stderr:\n"
            f"{result.stderr[-2000:]}\nstdout:\n{result.stdout[-500:]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
