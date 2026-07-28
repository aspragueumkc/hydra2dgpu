#!/usr/bin/env bash
# ── Launch the FULL QGIS desktop under NVIDIA Compute Sanitizer.
#
# This wraps `qgis` so every CUDA call is checked.  You keep clicking
# buttons just like always — when the simulation crashes, the
# sanitizer dumps the offending line + backtrace to a file.
#
# Usage:
#   bash tests/run_full_qgis_under_sanitizer.sh
#
# Override sim duration / output paths:
#   GUI_RUN_DURATION_S=300 SANITIZER_LOG=/tmp/san.log \
#       bash tests/run_full_qgis_under_sanitizer.sh
#
# What this script does:
#   1. Kills any stale Xvfb / qgis / qgis_process
#   2. Rebuilds the .so so the binary on disk matches your source
#   3. Purges __pycache__ so QGIS reimports edited Python modules
#   4. Starts Xvfb on :99
#   5. Spawns the full `qgis` desktop via compute-sanitizer --tool memcheck
#   6. Waits for QGIS to exit (Ctrl-C the GUI to trigger cleanup)
#   7. Tails the sanitizer log + the GUI's stdout/stderr
#
# Where things land:
#   $SANITIZER_LOG  (default: /tmp/gui_sanitizer.log) — sanitizer report
#   /tmp/qgis_gui.out                              — QGIS / plugin stdout
#   /tmp/gui_crash_repro.out                       — runtime_reporter log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${CONDA_ENV:-qgis_stable}"

SANITIZER_LOG="${SANITIZER_LOG:-/tmp/gui_sanitizer.log}"
QGIS_STDOUT_LOG="${QGIS_STDOUT_LOG:-/tmp/qgis_gui.out}"
GUI_RUN_DURATION_S="${GUI_RUN_DURATION_S:-30.0}"

# ── 1. Kill stale processes ────────────────────────────────────────────────
echo "[setup] killing any stale qgis (NOT Xvfb — we may share it with other apps)"
pkill -f qgis_process 2>/dev/null || true
pkill -f '^qgis-bin' 2>/dev/null || true
# Only kill Xvfb if we end up using it
sleep 1

# ── 2. Rebuild the native module so source == on-disk binary ────────────────
echo "[setup] rebuild hydra_swe2d (fast, ~30s for an unchanged tree)"
cd "$REPO_ROOT/build"
mamba run -n "$ENV_NAME" \
    env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    /usr/bin/cmake --build . -j"$(nproc)" --target hydra_swe2d 2>&1 | tail -5
cd "$REPO_ROOT"

# ── 3. Purge stale bytecode ────────────────────────────────────────────────
echo "[setup] purging __pycache__"
find "$REPO_ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── 4. Pick a usable display ──────────────────────────────────────────────
# Honour the user's current DISPLAY (=:10.0 on xrdp, etc.) so the GUI
# window actually appears on screen.  Fall back to :0 if it points to a
# valid socket, or fall back to a fresh Xvfb only if nothing else works.
if [ -z "${DISPLAY:-}" ] || ! ls "/tmp/.X11-unix/X${DISPLAY#:}" 2>/dev/null; then
  # No DISPLAY or invalid — probe for a live socket
  for try in :10 :0 :1; do
    if ls "/tmp/.X11-unix/X${try#:}" 2>/dev/null; then
      export DISPLAY="$try"
      echo "[setup] detected live display: $DISPLAY"
      break
    fi
  done
fi

if [ -z "${DISPLAY:-}" ] || ! ls "/tmp/.X11-unix/X${DISPLAY#:}" 2>/dev/null; then
  # Last resort: virtual framebuffer (NOT visible to the user)
  export DISPLAY=":99"
  if ! pgrep -x Xvfb >/dev/null 2>&1; then
    echo "[setup] WARNING: spawning Xvfb on $DISPLAY — connect a VNC viewer to see this"
    Xvfb "$DISPLAY" -screen 0 1600x1200x24 -nolisten tcp >/dev/null 2>&1 &
    XVFB_PID=$!
    sleep 1
  fi
else
  echo "[setup] using DISPLAY=$DISPLAY (visible)"
fi

# ── 5. Locate compute-sanitizer ───────────────────────────────────────────
SANITIZER_BIN="$(mamba run -n "$ENV_NAME" which compute-sanitizer 2>/dev/null || true)"
if [ -z "$SANITIZER_BIN" ]; then
  echo "[error] compute-sanitizer not found in env $ENV_NAME"
  exit 1
fi
echo "[setup] sanitizer: $SANITIZER_BIN"

# ── 6. Locate qgis binary in the env ──────────────────────────────────────
QGIS_BIN="$(mamba run -n "$ENV_NAME" which qgis 2>/dev/null || true)"
if [ -z "$QGIS_BIN" ]; then
  echo "[error] qgis not found in env $ENV_NAME"
  exit 1
fi
echo "[setup] qgis: $QGIS_BIN"
echo "[setup] sim duration pass-through: GUI_RUN_DURATION_S=${GUI_RUN_DURATION_S}s"
echo "[setup] sanitizer log: $SANITIZER_LOG"

# ── 7. Run QGIS under sanitizer (script keeps running till you quit QGIS) ──
# All env vars needed by the plugin and the workbench go via mamba run -e.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export QGIS_PLUGINPATH="$REPO_ROOT"

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  Launching QGIS under compute-sanitizer --tool memcheck ..."
echo "  Click Run / Cancel / etc in the GUI normally.  When the simulator"
echo "  crashes, the sanitizer dumps a backtrace and exits."
echo "  Press Ctrl-C here to kill QGIS manually."
echo "════════════════════════════════════════════════════════════════════"
echo

# `unbuffered` so the log fills in real time.  Don't pass -x to qgis
# (would suppress plugin UI output).  Pass --noversioncheck to keep
# startup fast.
"$SANITIZER_BIN" \
    --tool=memcheck \
    --leak-check=full \
    --show-backtrace=host \
    --print-limit=100 \
    --log-file="$SANITIZER_LOG" \
    mamba run -n "$ENV_NAME" \
        env PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
               QGIS_PLUGINPATH="$REPO_ROOT" \
               GUI_RUN_DURATION_S="$GUI_RUN_DURATION_S" \
        "$QGIS_BIN" --noversioncheck \
            > "$QGIS_STDOUT_LOG" 2>&1 || true
RC=$?

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  QGIS exited (rc=$RC)"
echo
echo "  Tail of plugin/runtime stdout ($QGIS_STDOUT_LOG):"
echo "  ────────────────────────────────────────────────"
tail -40 "$QGIS_STDOUT_LOG" 2>/dev/null || echo "(empty)"
echo
echo "  Tail of sanitizer report ($SANITIZER_LOG):"
echo "  ────────────────────────────────────────────────"
tail -80 "$SANITIZER_LOG" 2>/dev/null || echo "(empty — sanitizer did not log; means no mem error)"
echo
echo "  Full files:"
echo "    $QGIS_STDOUT_LOG"
echo "    $SANITIZER_LOG"
echo
if grep -qE "(Invalid|out of bounds|illegal|Memory access)" "$SANITIZER_LOG" 2>/dev/null; then
  echo "  ⚠ Memcheck flagged an error.  See $SANITIZER_LOG for the full"
  echo "    line + backtrace.  That IS the bug — paste the relevant"
  echo "    ~30 lines back to me and we can fix it for real."
else
  echo "  Sanitizer ran clean — no illegal memory access detected."
  echo "  If QGIS crashed anyway (rc!=0), share $QGIS_STDOUT_LOG instead."
fi
