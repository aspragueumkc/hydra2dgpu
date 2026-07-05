"""Tests for swe2d.cli.batch_runner.BatchOrchestrator (extracted from dialog).

The orchestrator owns subprocess pool lifecycle and status-file polling. The
dialog receives callbacks.
"""
import pathlib
import time

import pytest


class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


def test_orchestrator_runs_each_param_set(tmp_path, monkeypatch):
    from swe2d.cli import batch_runner

    seen = []

    def fake_popen(cmd, **kw):
        seen.append(tuple(cmd))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", fake_popen)

    orch = batch_runner.BatchOrchestrator(
        param_sets=[{"id": "a"}, {"id": "b"}],
        workdir=str(tmp_path),
    )
    results = orch.run()
    assert len(results) == 2
    assert seen and "run" in seen[0]


def test_orchestrator_emits_progress_callbacks(tmp_path, monkeypatch):
    from swe2d.cli import batch_runner

    def fake_popen(cmd, **kw):
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", fake_popen)

    progress = []
    orch = batch_runner.BatchOrchestrator(
        param_sets=[{"id": "a"}],
        workdir=str(tmp_path),
        on_progress=lambda done, total: progress.append((done, total)),
    )
    orch.run()
    assert progress[-1][0] == 1
    assert progress[-1][1] == 1


def test_orchestrator_collects_failures(tmp_path, monkeypatch):
    from swe2d.cli import batch_runner

    def fake_popen(cmd, **kw):
        return _FakeCompleted(returncode=1)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", fake_popen)

    orch = batch_runner.BatchOrchestrator(
        param_sets=[{"id": "a"}],
        workdir=str(tmp_path),
    )
    results = orch.run()
    assert results[0]["status"] == "failed"
