"""Unit tests for tools/memory.py frontmatter validation."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PY = REPO_ROOT / "tools" / "memory.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MEMORY_PY), *args],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )


def _write_entry(path: Path, *, body: str, front: dict) -> None:
    lines = ["---"]
    for k, v in front.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


class FrontmatterValidation(unittest.TestCase):
    def test_minimal_valid_entry_passes(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Decision\nAll good.",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["lesson"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("ok", r.stdout.lower())

    def test_missing_evidence_fails(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Decision\nBad.",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["lesson"],
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 1)
            self.assertIn("evidence", r.stdout.lower() + r.stderr.lower())

    def test_secret_in_body_fails(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Decision\nThe token is aws_secret=AKIA1234.",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["lesson"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 1)
            self.assertIn("secret", r.stdout.lower() + r.stderr.lower())

    def test_secret_in_fenced_code_passes(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Decision\nExample:\n```\npassword=hunter2\n```",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["lesson"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_invalid_tag_fails(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Decision\nBad tag.",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["not-in-allowlist"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 1)
            self.assertIn("tag", r.stdout.lower() + r.stderr.lower())

    def test_invalid_topic_slug_fails(self) -> None:
        with self._tmp_active() as active:
            entry = active / "BAD-SLUG.md"
            _write_entry(
                entry,
                body="## Decision\nBad slug.",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "bad-slug",
                    "tags": ["lesson"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 1)
            self.assertIn("topic", r.stdout.lower() + r.stderr.lower())

    def test_status_enum_enforced(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Decision\nBad status.",
                front={
                    "type": "memory",
                    "status": "garbage",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["lesson"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 1)
            self.assertIn("status", r.stdout.lower() + r.stderr.lower())

    def test_missing_decision_heading_fails(self) -> None:
        with self._tmp_active() as active:
            entry = active / "topic.md"
            _write_entry(
                entry,
                body="## Context\nNo decision here.",
                front={
                    "type": "memory",
                    "status": "active",
                    "created": "2026-07-26",
                    "topic": "topic",
                    "tags": ["lesson"],
                    "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("validate", str(entry))
            self.assertEqual(r.returncode, 1)
            self.assertIn("decision", r.stdout.lower() + r.stderr.lower())

    def _tmp_active(self):
        import tempfile
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            with tempfile.TemporaryDirectory() as td:
                active = Path(td) / "active"
                active.mkdir()
                taglist = active / "tag-allowlist.md"
                taglist.write_text(
                    "---\ntype: memory\nstatus: active\ncreated: 2026-07-26\n"
                    "topic: tag-allowlist\ntags: [infra]\nevidence: AGENTS.md:1\n---\n"
                    "## Decision\nlesson, decision, hazard, infra, cpp, units, gui, doc.\n",
                    encoding="utf8",
                )
                saved = os.environ.copy()
                os.environ["HYDRA_MEMORY_ACTIVE_DIR"] = str(active)
                try:
                    yield active
                finally:
                    for k in ("HYDRA_MEMORY_ACTIVE_DIR",):
                        os.environ.pop(k, None)
                    os.environ.update(saved)
        return _cm()


class CaptureFlow(unittest.TestCase):
    def test_draft_writes_review_pending_with_timestamp(self) -> None:
        with self._env() as env:
            r = _run_cli(
                "draft", "--text", "Culvert path returns CFS.", "--topic", "culvert-units",
                "--tags", "cpp,units",
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            rp = env["review_pending"]
            files = sorted(p.name for p in rp.glob("*.md"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith("-culvert-units.md"))
            text = (rp / files[0]).read_text(encoding="utf8")
            self.assertIn("type: memory", text)
            self.assertIn("status: review-pending", text)
            self.assertIn("Culvert path returns CFS.", text)
            self.assertIn("cpp", text)

    def test_approve_moves_to_active_and_drops_prefix(self) -> None:
        with self._env() as env:
            _run_cli(
                "draft", "--text", "Body.", "--topic", "topic", "--tags", "lesson",
            )
            src = next(env["review_pending"].glob("*.md"))
            # Replace the stub evidence with a real citation so validate passes.
            text = src.read_text(encoding="utf8")
            text = text.replace("evidence: PENDING: add citation before approve",
                                "evidence: AGENTS.md:1")
            src.write_text(text, encoding="utf8")
            r = _run_cli("approve", str(src))
            self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
            self.assertFalse(src.exists())
            target = env["active"] / "topic.md"
            self.assertTrue(target.exists())
            text = target.read_text(encoding="utf8")
            self.assertIn("status: active", text)

    def test_approve_topic_collision_merges_into_open_questions(self) -> None:
        with self._env() as env:
            _write_entry(
                env["active"] / "topic.md",
                body="## Decision\nFirst body.\n\n## Open questions\n",
                front={
                    "type": "memory", "status": "active", "created": "2026-07-26",
                    "topic": "topic", "tags": ["lesson"], "evidence": "AGENTS.md:1",
                },
            )
            _run_cli(
                "draft", "--text", "New evidence.", "--topic", "topic", "--tags", "lesson",
            )
            src = next(env["review_pending"].glob("*.md"))
            text = src.read_text(encoding="utf8")
            text = text.replace("evidence: PENDING: add citation before approve",
                                "evidence: AGENTS.md:1")
            src.write_text(text, encoding="utf8")
            r = _run_cli("approve", str(src))
            self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
            active_text = (env["active"] / "topic.md").read_text(encoding="utf8")
            self.assertIn("New evidence.", active_text)
            self.assertIn("## Open questions", active_text)
            supers = list(env["superseded"].glob("*.md"))
            self.assertEqual(len(supers), 1)
            self.assertIn(
                "superseded_by: docs/memory/active/topic.md",
                supers[0].read_text(encoding="utf8"),
            )

    def test_list_returns_active_topics(self) -> None:
        with self._env() as env:
            _write_entry(
                env["active"] / "alpha.md",
                body="## Decision\nA.",
                front={
                    "type": "memory", "status": "active", "created": "2026-07-26",
                    "topic": "alpha", "tags": ["lesson"], "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli("list")
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("alpha", r.stdout)

    def test_archive_moves_to_superseded_with_metadata(self) -> None:
        with self._env() as env:
            _write_entry(
                env["active"] / "old.md",
                body="## Decision\nOld.",
                front={
                    "type": "memory", "status": "active", "created": "2026-07-26",
                    "topic": "old", "tags": ["lesson"], "evidence": "AGENTS.md:1",
                },
            )
            _write_entry(
                env["active"] / "new.md",
                body="## Decision\nNew.",
                front={
                    "type": "memory", "status": "active", "created": "2026-07-26",
                    "topic": "new", "tags": ["lesson"], "evidence": "AGENTS.md:1",
                },
            )
            r = _run_cli(
                "archive", "old", "--reason", "superseded", "--by", "new",
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertFalse((env["active"] / "old.md").exists())
            sup = env["superseded"] / "old.md"
            self.assertTrue(sup.exists())
            text = sup.read_text(encoding="utf8")
            self.assertIn("status: superseded", text)
            self.assertIn("superseded_by: docs/memory/active/new.md", text)
            self.assertIn("completed:", text)

    def _env(self):
        import tempfile
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                env = {
                    "root": root,
                    "active": root / "active",
                    "review_pending": root / "review-pending",
                    "superseded": root / "superseded",
                }
                for k in ("active", "review_pending", "superseded"):
                    env[k].mkdir()
                (env["active"] / "tag-allowlist.md").write_text(
                    "---\ntype: memory\nstatus: active\ncreated: 2026-07-26\n"
                    "topic: tag-allowlist\ntags: [infra]\nevidence: AGENTS.md:1\n---\n"
                    "## Decision\nlesson decision hazard infra cpp units doc.\n",
                    encoding="utf8",
                )
                saved = os.environ.copy()
                os.environ["HYDRA_MEMORY_ACTIVE_DIR"] = str(env["active"])
                os.environ["HYDRA_MEMORY_REVIEW_PENDING_DIR"] = str(env["review_pending"])
                os.environ["HYDRA_MEMORY_SUPERSEDED_DIR"] = str(env["superseded"])
                try:
                    yield env
                finally:
                    for k in (
                        "HYDRA_MEMORY_ACTIVE_DIR",
                        "HYDRA_MEMORY_REVIEW_PENDING_DIR",
                        "HYDRA_MEMORY_SUPERSEDED_DIR",
                    ):
                        os.environ.pop(k, None)
                    os.environ.update(saved)
        return _cm()


class HookInstall(unittest.TestCase):
    def test_install_hook_writes_files(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True, capture_output=True,
            )
            (root / ".git" / "hooks").mkdir(parents=True, exist_ok=True)
            # Provide a stub memory.py in the temp root so the hook script's
            # path resolves; the actual hook content is what we test.
            (root / "tools").mkdir()
            (root / "tools" / "memory.py").write_text(
                "# stub for install-hook test\n", encoding="utf8"
            )
            r = subprocess.run(
                [sys.executable, str(MEMORY_PY), "install-hook"],
                cwd=str(root),
                env={
                    **os.environ,
                    "PYTHONPATH": str(root),
                    "HYDRA_MEMORY_ACTIVE_DIR": str(root / "active"),
                },
                capture_output=True, text=True,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertTrue((root / ".git" / "hooks" / "pre-commit").exists())
            self.assertTrue((root / ".git" / "hooks" / "post-merge").exists())
            text = (root / ".git" / "hooks" / "pre-commit").read_text()
            self.assertIn("validate", text)
            text2 = (root / ".git" / "hooks" / "post-merge").read_text()
            self.assertIn("reindex", text2)
