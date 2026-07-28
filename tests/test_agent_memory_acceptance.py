"""End-to-end happy path for the memory CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PY = REPO_ROOT / "tools" / "memory.py"


class Acceptance(unittest.TestCase):
    def test_full_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for sub in ("active", "review-pending", "superseded", ".memory"):
                (root / sub).mkdir()
            env = {
                "PYTHONPATH": str(REPO_ROOT),
                "HYDRA_MEMORY_ACTIVE_DIR": str(root / "active"),
                "HYDRA_MEMORY_REVIEW_PENDING_DIR": str(root / "review-pending"),
                "HYDRA_MEMORY_SUPERSEDED_DIR": str(root / "superseded"),
                "HYDRA_MEMORY_INDEX_DIR": str(root / ".memory"),
                "HYDRA_MEMORY_ENCODER": "mock",
            }
            (root / "active" / "tag-allowlist.md").write_text(
                "---\ntype: memory\nstatus: active\ncreated: 2026-07-26\n"
                "topic: tag-allowlist\ntags: [infra]\nevidence: AGENTS.md:1\n---\n"
                "## Decision\nlesson decision hazard infra.\n",
                encoding="utf8",
            )

            def run(*args):
                return subprocess.run(
                    [sys.executable, str(MEMORY_PY), *args],
                    cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                )

            r = run("draft", "--text", "Capture from test.", "--topic", "happy-path",
                    "--tags", "lesson")
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            src = next((root / "review-pending").glob("*.md"))
            # Fill in the citation stub (mirror the user flow).
            text = src.read_text(encoding="utf8")
            text = text.replace(
                "evidence: PENDING: add citation before approve",
                "evidence: AGENTS.md:1",
            )
            src.write_text(text, encoding="utf8")

            r = run("validate", str(src))
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)

            r = run("approve", str(src))
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            self.assertTrue((root / "active" / "happy-path.md").exists())

            r = run("reindex")
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            self.assertTrue((root / ".memory" / "index.sqlite").exists())

            r = run("search", "capture from test", "--k", "5")
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("happy-path", r.stdout)

            r = run("status")
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("ok", r.stdout.lower())

            r = run("list")
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("happy-path", r.stdout)
