"""End-to-end tests for reindex/search/status using a mocked encoder."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PY = REPO_ROOT / "tools" / "memory.py"


def _run_cli(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(MEMORY_PY), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _mocked_env(root: Path) -> dict:
    return {
        "HYDRA_MEMORY_ACTIVE_DIR": str(root / "active"),
        "HYDRA_MEMORY_REVIEW_PENDING_DIR": str(root / "review-pending"),
        "HYDRA_MEMORY_SUPERSEDED_DIR": str(root / "superseded"),
        "HYDRA_MEMORY_INDEX_DIR": str(root / ".memory"),
        "HYDRA_MEMORY_ENCODER": "mock",
    }


@contextmanager
def _fresh_tree():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for sub in ("active", "review-pending", "superseded", ".memory"):
            (root / sub).mkdir()
        (root / "active" / "tag-allowlist.md").write_text(
            "---\ntype: memory\nstatus: active\ncreated: 2026-07-26\n"
            "topic: tag-allowlist\ntags: [infra]\nevidence: AGENTS.md:1\n---\n"
            "## Decision\nlesson decision hazard infra cpp units doc.\n",
            encoding="utf8",
        )
        (root / "active" / "culvert-units.md").write_text(
            "---\ntype: memory\nstatus: active\ncreated: 2026-07-26\n"
            "topic: culvert-units\ntags: [cpp, units]\nevidence: AGENTS.md:120\n---\n"
            "## Decision\nThe C++ culvert path returns CFS; convert via units.cms_to_model().\n",
            encoding="utf8",
        )
        (root / "active" / "cfl.md").write_text(
            "---\ntype: memory\nstatus: active\ncreated: 2026-07-26\n"
            "topic: cfl\ntags: [kernel, swe2d]\nevidence: AGENTS.md:60\n---\n"
            "## Decision\nCFL target is 0.5 for the GPU explicit solver.\n",
            encoding="utf8",
        )
        yield root


class ReindexAndSearch(unittest.TestCase):
    def test_reindex_writes_index_and_metadata(self) -> None:
        with _fresh_tree() as root:
            r = _run_cli("reindex", env_extra=_mocked_env(root))
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            index = root / ".memory" / "index.sqlite"
            self.assertTrue(index.exists())
            self.assertTrue((root / ".memory" / "last_indexed_at.json").exists())
            meta = json.loads((root / ".memory" / "last_indexed_at.json").read_text())
            self.assertIn("commit", meta)
            self.assertIn("chunk_count", meta)
            self.assertIn("backend", meta)

    def test_search_ranks_exact_match_first(self) -> None:
        with _fresh_tree() as root:
            _run_cli("reindex", env_extra=_mocked_env(root))
            r = _run_cli("search", "culvert units", "--k", "5", env_extra=_mocked_env(root))
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("culvert-units", r.stdout)
            self.assertIn("AGENTS.md:120", r.stdout)

    def test_search_max_tokens_truncates_with_footer(self) -> None:
        with _fresh_tree() as root:
            _run_cli("reindex", env_extra=_mocked_env(root))
            r = _run_cli(
                "search", "decision", "--k", "20", "--max-tokens", "10",
                env_extra=_mocked_env(root),
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("+", r.stdout)
            self.assertIn("rerun with --max-tokens higher", r.stdout)

    def test_unbounded_warns(self) -> None:
        with _fresh_tree() as root:
            _run_cli("reindex", env_extra=_mocked_env(root))
            r = _run_cli(
                "search", "decision", "--k", "20", "--unbounded",
                env_extra=_mocked_env(root),
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("WARNING: --unbounded may inject", r.stdout + r.stderr)

    def test_status_ok_after_reindex(self) -> None:
        with _fresh_tree() as root:
            r = _run_cli("status", env_extra=_mocked_env(root))
            # No index yet → stale, blocking=never_indexed, exit 1.
            self.assertEqual(r.returncode, 1, msg=r.stderr)
            self.assertIn("stale", r.stdout.lower())
            _run_cli("reindex", env_extra=_mocked_env(root))
            r = _run_cli("status", env_extra=_mocked_env(root))
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("ok", r.stdout.lower())

    def test_cpu_fallback_banner_fires(self) -> None:
        with _fresh_tree() as root:
            env = _mocked_env(root)
            # Remove the explicit mock override so the loader falls through to
            # _try_real, which fires the loud CPU banner when the encoder
            # stack is missing.
            env.pop("HYDRA_MEMORY_ENCODER", None)
            env["HYDRA_MEMORY_FORCE_CPU"] = "1"
            r = _run_cli("reindex", env_extra=env)
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("CPU embedding fallback", r.stdout + r.stderr)
            meta = json.loads((root / ".memory" / "last_indexed_at.json").read_text())
            self.assertEqual(meta["backend"], "cpu")
            r = _run_cli("status", env_extra=env)
            self.assertIn("cpu_fallback_active", r.stdout + r.stderr)
