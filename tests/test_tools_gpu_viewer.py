from __future__ import annotations
import unittest
"""Tests for tools/hydra_mcp/tools_gpu_viewer.py — MCP tool surface.

The bridge client is mocked (no real QGIS / GPU required). Each test
verifies the tool returns a structured error when no bridge is active,
and that valid args are passed through to the bridge.
"""

import os
import sys

import pytest

# Make tools/ importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.hydra_mcp.tools_gpu_viewer import (  # noqa: E402
    gpu_viewer_open,
    gpu_viewer_set_field,
    gpu_viewer_read_snapshot,
    gpu_viewer_screenshot,
)


class _FakeClient:
    def __init__(self, response=None, raise_exc=None):
        self.calls = []
        self.response = response or {"ok": True}
        self.raise_exc = raise_exc

    def call(self, method, params=None):
        self.calls.append((method, params))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def _patch_get_bridge_client(monkeypatch, client):
    monkeypatch.setattr(
        "tools.hydra_mcp.tools_gpu_viewer._get_bridge_client",
        lambda *a, **kw: client,
    )


def test_all_four_tools_are_callable():
    """All four MCP tool functions are importable and callable."""
    assert callable(gpu_viewer_open)
    assert callable(gpu_viewer_set_field)
    assert callable(gpu_viewer_read_snapshot)
    assert callable(gpu_viewer_screenshot)


def test_no_bridge_returns_structured_error(monkeypatch):
    """When no QGIS bridge is active, every tool returns ok=False with a message."""
    monkeypatch.setattr(
        "tools.hydra_mcp.tools_gpu_viewer._get_bridge_client",
        lambda *a, **kw: None,
    )
    r = gpu_viewer_open()
    assert r.get("ok") is False
    assert "no active" in r["error"].lower()
    r = gpu_viewer_read_snapshot()
    assert r.get("ok") is False
    assert "no active" in r["error"].lower()
    r = gpu_viewer_screenshot("/tmp/dummy_viewer.png")
    assert r.get("ok") is False
    assert "no active" in r["error"].lower()


def test_open_passes_method(monkeypatch):
    client = _FakeClient()
    _patch_get_bridge_client(monkeypatch, client)
    gpu_viewer_open()
    assert client.calls == [("gpu_viewer_open", None)]


def test_set_field_passes_method_and_arg(monkeypatch):
    client = _FakeClient()
    _patch_get_bridge_client(monkeypatch, client)
    r = gpu_viewer_set_field("speed")
    assert client.calls == [("gpu_viewer_set_field", {"field": "speed"})]
    assert r == {"ok": True}


def test_set_field_invalid_returns_error(monkeypatch):
    """Validation happens client-side before calling the bridge."""
    client = _FakeClient()
    _patch_get_bridge_client(monkeypatch, client)
    r = gpu_viewer_set_field("not-a-field")
    assert r["ok"] is False
    assert "invalid" in r["error"].lower()
    # No bridge call should have happened.
    assert client.calls == []


def test_read_snapshot_passes_method(monkeypatch):
    client = _FakeClient(response={
        "ok": True, "t_s": 1.5, "n_cells": 4,
        "h_b64": "", "hu_b64": "", "hv_b64": "",
    })
    _patch_get_bridge_client(monkeypatch, client)
    r = gpu_viewer_read_snapshot()
    assert client.calls == [("gpu_viewer_read_snapshot", None)]
    assert r["ok"] is True
    assert r["t_s"] == 1.5


def test_screenshot_passes_method_and_args(monkeypatch, tmp_path):
    client = _FakeClient()
    _patch_get_bridge_client(monkeypatch, client)
    out = str(tmp_path / "viewer.png")
    r = gpu_viewer_screenshot(out)
    assert client.calls == [(  # method, params
        "gpu_viewer_screenshot",
        {"out_path": out, "format": "png"},
    )]
    assert r == {"ok": True}


def test_screenshot_creates_parent_dir(monkeypatch, tmp_path):
    """Tool creates the parent dir for the screenshot if missing."""
    client = _FakeClient()
    _patch_get_bridge_client(monkeypatch, client)
    out = tmp_path / "nested" / "dir" / "viewer.png"
    gpu_viewer_screenshot(str(out))
    assert out.parent.is_dir()


def test_bridge_exception_returns_error(monkeypatch):
    """When the bridge client raises, the tool returns a structured error."""
    client = _FakeClient(raise_exc=RuntimeError("bridge dropped"))
    _patch_get_bridge_client(monkeypatch, client)
    r = gpu_viewer_open()
    assert r["ok"] is False
    assert "bridge dropped" in r["error"]

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
