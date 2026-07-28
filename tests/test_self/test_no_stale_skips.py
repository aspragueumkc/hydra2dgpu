"""Detect stale @unittest.skip markers (retired/deprecated/TODO/FIXME).

Spec: docs/specs/2026-07-26-test-discipline-design.md §3.2
Catches audit §4 row #7 (Skipped-and-forgotten).
"""
import subprocess
import unittest


class TestNoStaleSkips(unittest.TestCase):
    PATTERN = r"@unittest\.skip\(\s*(['\"])(retired|deprecated|TODO|FIXME)"

    def test_no_stale_skip_markers(self):
        """No @unittest.skip marker should use retired/deprecated/TODO/FIXME reasons.

        These markers silently disable a test forever. The fix is to either
        delete the test or replace the skip with a real fix.

        Note: ``tests/test_self/`` is excluded because the meta-test
        (``test_self_tests_catch_violations.py``) intentionally contains
        the string ``@unittest.skip("retired: ...")`` as a synthetic
        violation to verify THIS test's grep catches the pattern.
        """
        result = subprocess.run(
            ["grep", "-rnE", "--exclude-dir=test_self",
             self.PATTERN, "tests/"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            self.fail(
                "Stale @unittest.skip markers found (delete the test or "
                "replace the skip with a real fix):\n\n" + result.stdout
            )
        # grep returns 1 when no match found — that's the success path.
        # Any other code is an actual error.
        self.assertEqual(
            result.returncode, 1,
            f"grep failed unexpectedly (rc={result.returncode}): {result.stderr}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
