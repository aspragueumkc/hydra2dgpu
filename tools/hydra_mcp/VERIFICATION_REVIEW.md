# Verification Review — Post-Fix (feature/hydra-mcp @ 07d5880)

Second-pass review verifying the 4 blocker fixes from `COMPREHENSIVE_REVIEW.md`.
Fix commits: `d271ae8` (B-1 + B-4), `bc586f1` (B-2), `07d5880` (B-3). No other
commits since `c775d53`; the 3 fix commits touch exactly
`.kimi-code/mcp.json`, `tests/test_hydra_mcp.py`,
`tools/hydra_mcp/tools_gui.py`, `tools/hydra_mcp/widget_screenshot.py`.

Test run: `PYTHONPATH=. python3 -m unittest tests.test_hydra_mcp` →
**Ran 65 tests, OK (skipped=1)**.

## Fix Verification Matrix

| # | Status | Evidence | Issues |
|---|--------|----------|--------|
| B-1 | VERIFIED | `tools_gui.py:150-205` — `_find_qgis_binary()` (`tools_gui.py:255-299`) returns the `qgis` launcher via `QGIS_BINARY` env → conda paths → `shutil.which("qgis")` → `/usr/bin/qgis`; command built as `[qgis_binary, "--noplugins", "--noversioncheck", "--project", ..., "--code", bridge_script]`. No `python3`-with-QGIS-flags path remains; `_find_qgis_python` is gone. Module + helper docstrings updated to match. | No new test covers binary lookup or command construction (first review explicitly asked for a subprocess integration test — still missing). |
| B-2 | VERIFIED | `.kimi-code/mcp.json:8-26` — args now include `--with PyQt5 --with PyQt5-Qt5`; `disabledTools: ["design_apply_patch"]` present. Matches plan (`docs/HYDRA_MCP_SERVER_PLAN.md:213`). Commit message documents a manual `uv run … import QLocalSocket` smoke check. | `design_apply_patch` does not exist yet (Tier C, planned) — disabling a non-existent tool is harmless and plan-compliant. |
| B-3 | VERIFIED | `widget_screenshot.py:75-91` — uses `QtCore.QBuffer` + `QIODevice.WriteOnly`, JPEG quality passed as 3rd positional arg (`85`), `save()` boolean checked with a clear error on False, `buffer.close()` in `finally`. Two new tests (`tests/test_hydra_mcp.py:1507-1608`) assert the buffer is a real `QBuffer`/`QIODevice`, quality==85, and base64 round-trip. Both pass. | Still no test against a real `QPixmap` under `QApplication` (mock pixmap only) — acceptable per test-docstring rationale, but real-encoding confidence comes only from the API being correct now. |
| B-4 | PARTIAL | `tools_gui.py:318-374` — `_wait_for_bridge_ready` uses a daemon thread draining `proc.stdout` line-by-line into a `queue.Queue`; main loop polls the queue (0.25 s) plus `proc.poll()` for early death; raises `RuntimeError` on premature exit or timeout. No `/proc/<pid>/fd/1` anywhere (grep confirms). Empirically exercised with a live child: timeout raises correctly. | **New issue N-1 below**: the error path in `gui_launch` can hang, and stderr is never drained, so the claimed deadlock elimination is only half true. No unit test for `_wait_for_bridge_ready`. |

## Regression Checks (unchanged invariants)

- **Bridge protocol** — intact. `encode_message`/`decode_messages`
  (`bridge_client.py:22-42`) are still the single shared wire-format
  implementation, imported by `qgis_bridge.py:36`. 4-byte big-endian length
  prefix + UTF-8 JSON. Token checked per-message
  (`qgis_bridge.py:259-263`); wrong token → `-32001` + disconnect.
- **Single-flight** — intact. `_busy` flag rejects concurrent GUI requests
  with `-32000` and is reset in `finally` (`qgis_bridge.py:277-304`). `ping`
  remains allowed while busy (harmless, no widget access).
- **Token hygiene** — intact. `secrets.token_urlsafe(32)`
  (`qgis_bridge.py:161`), file chmod 0600 (`qgis_bridge.py:199`), client
  auto-discovery UID-gated on `st_uid == os.getuid()`
  (`bridge_client.py:123`). (Note: first-review HIGH #7, atomic 0600-at-create,
  remains unfixed — out of scope for these 4 blockers.)
- **Silent absorbs** — two new `except Exception: pass` blocks in
  `gui_launch`'s cleanup path (`tools_gui.py:227,231`) and one
  `except (OSError, ValueError): pass` in the reader thread
  (`tools_gui.py:348-351`). The reader-thread one is fine (main loop detects
  death/timeout). The cleanup-path ones mask the N-1 drain problem (see below).
- **Docstrings** — updated consistently: module docstring now documents
  `qgis --code` injection; `_find_qgis_binary` and `_wait_for_bridge_ready`
  docstrings describe the new behavior and explicitly call out the replaced
  approaches. README examples already used `qgis … --code` and remain accurate.
- **Git** — only the 3 fix commits since `c775d53`; no other mutations.
  Untracked files are review docs only.

## New Issues Found

### N-1 (MEDIUM) — `gui_launch` error path can hang forever on the timeout case

`tools_gui.py:214-237`: when `_wait_for_bridge_ready` raises `RuntimeError`,
the handler drains the child pipes **before** killing the child:

```python
stdout_tail = (proc.stdout.read() or b"").decode(...)  # line 224
stderr_tail = (proc.stderr.read() or b"").decode(...)  # line 225
...
proc.kill()                                            # line 229
```

In the **timeout** case the child is still alive and holds its stdout open, so
`proc.stdout.read()` blocks until EOF — which never arrives. The tool call
hangs indefinitely instead of returning the structured error, and the
`except Exception: pass` around it cannot help (blocking is not an exception).
Verified empirically: a live child that never prints the ready line causes
`_wait_for_bridge_ready` to raise correctly at 1 s, but the subsequent
`proc.stdout.read()` is still blocked after 5 s. (The premature-exit case is
fine because the pipes are already at EOF.) Additionally, the main thread now
races the still-running daemon reader thread on the same buffered stream —
harmless for correctness of the error, but the tail it captures may be empty.

Related: **stderr is never drained during the wait** (only stdout, by the
reader thread). If QGIS writes more than one pipe buffer (~64 KB) to stderr
before printing the ready line, the child blocks on the stderr write, never
prints ready → guaranteed timeout → the hang above. The B-4 commit message
claims "pipe-buffer deadlock risk is eliminated" — true only for stdout.

Suggested fix: kill/reap the child first (`proc.kill(); proc.wait(timeout=5)`),
then drain the pipes (after kill, reads hit EOF promptly); and either drain
stderr in a second daemon thread during the wait or start the child with
`stderr=subprocess.STDOUT` so a single drained pipe carries everything.

### N-2 (LOW) — No test coverage for the B-1/B-4 fixes

The 104 added test lines cover only B-3 (screenshot). Nothing exercises
`_find_qgis_binary`, the launch-command construction, or
`_wait_for_bridge_ready` — the last is easily testable with a real
`subprocess.Popen(["printf", "HYDRA_MCP_BRIDGE_READY s t\n"])`. The first
review's CRITICAL #1 explicitly requested a subprocess integration test for
the launch path; it was not added. (65 tests pass, but they would not catch a
regression of B-1 or B-4.)

### N-3 (LOW) — Stale references in `MCP_PHASE2_REVIEW.md`

`tools/hydra_mcp/MCP_PHASE2_REVIEW.md:32,127,141,158` still discusses
`_find_qgis_python`, which no longer exists. Untracked working doc, not code —
flagging only so it isn't mistaken for current truth.

## Verdict

**NEEDS_FIXES** — all 4 blockers are genuinely fixed (B-4 only partially: the
readiness detection itself is correct, but its caller's error path introduces
hang N-1). N-1 should be fixed before shipping `gui_launch`; N-2/N-3 are
follow-ups.
