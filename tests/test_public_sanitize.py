import tempfile
import unittest
from pathlib import Path

from tools.public_sanitize import is_excluded, scan_tree


class PublicSanitizeTests(unittest.TestCase):
    def test_allowed_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.py"
            path.write_text("from pathlib import Path\nROOT = Path(__file__).resolve().parents[1]\n")
            self.assertEqual(scan_tree(Path(directory)), [])

    def test_rejects_private_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests" / "module.py"
            path.parent.mkdir()
            path.write_text("QGIS_BINARY_PATH = '/home/developer/qgis'\n")
            self.assertTrue(any("private absolute path" in item for item in scan_tree(Path(directory))))

    def test_uses_pathlib_for_intrinsic_repo_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tests" / "module.py"
            path.parent.mkdir()
            path.write_text("ROOT = Path(__file__).resolve().parents[1]\n")
            self.assertEqual(scan_tree(Path(directory)), [])

    def test_directory_exclusion_does_not_reject_parent(self):
        self.assertTrue(is_excluded("docs/archive/session/log.md", ["docs/archive/"]))
        self.assertFalse(is_excluded("docs/USER_GUIDE.md", ["docs/archive/"]))

    def test_missing_exclusion_is_not_present(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(is_excluded("docs/USER_GUIDE.md", ["reference/"]))


if __name__ == "__main__":
    unittest.main()
