#!/usr/bin/env bash
# Production-path release install test.
#
# Builds the wheel + plugin zip locally, serves the wheel from a local
# server, installs the plugin zip into the isolated qgis_clean env
# (release_test profile — no dev symlinks, no repo PYTHONPATH), runs the
# production BackendInstaller (wheel pulled from the local host), then
# runs the FULL test suite against the installed code:
#   - fast_fail.sh gates 1-4 (incl. the MCP widget walk, which needs a
#     live bridge autostarted by the INSTALLED plugin)
#   - the GPU validation tests (needs cuda-cudart in qgis_clean +
#     system NVIDIA driver)
#
# The wheel bundles swe2d/ + tests/ + tools/ (see pyproject.toml) so the
# suite imports everything from the installed wheel, not the repo.
#
# Usage: bash tools/release_install_test.sh
# Requires: mamba, qgis_stable (build), qgis_clean (isolated install),
#           system NVIDIA driver, ~10 min.

set -euo pipefail

# No developer-specific paths in this file (public-sanitized repo policy):
# the mamba binary and CUDA compiler come from the environment, fail-fast
# when absent.
MAMBA="${MAMBA:-}"
if [ -z "$MAMBA" ]; then
  MAMBA="$(command -v mamba || true)"
fi
if [ -z "$MAMBA" ]; then
  die "mamba not found — set MAMBA to the mamba binary path"
fi
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VERSION=$(sed -nE 's/^version\s*=\s*"([^"]+)".*/\1/p' pyproject.toml | head -1)
WORK=/tmp/hydra-release-test
WHEEL_DIR="$WORK/wheels"
HTTP_DIR="$WORK/http"
RUN_DIR="$WORK/run"
PORT=8765
DISPLAY_NUM=:99
ENV_NAME=qgis_clean
PROFILE=release_test
PROFILE_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/$PROFILE"
PROFILE_PLUGINS="$PROFILE_DIR/python/plugins"
INSTALL_DIR="$HOME/.hydra2dgpu"
VENV_SP="$INSTALL_DIR/lib/python3.12/site-packages"
WHEEL_URL_BASE="http://127.0.0.1:$PORT"
# Where the bridge writes its token file (XDG_RUNTIME_DIR, else /tmp).
TOKEN_DIR="${XDG_RUNTIME_DIR:-/tmp}"

log() { printf '\n== %s ==\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# shellcheck disable=SC2016
HOOK='eval "$("'"$MAMBA"'" shell hook --shell bash)" && mamba activate'

XVFB_PID=""

cleanup() {
  set +e
  pkill -f "http.server $PORT" 2>/dev/null
  pkill -f "qgis --noversioncheck --profile $PROFILE" 2>/dev/null
  # Kill only the Xvfb WE started — a global `pkill -x Xvfb` would take
  # down displays that other sessions (and the next run) depend on.
  if [ -n "$XVFB_PID" ]; then
    kill "$XVFB_PID" 2>/dev/null
  fi
  rm -f "$TOKEN_DIR"/hydra_mcp_bridge_*.json
  set -e
}
trap cleanup EXIT

ensure_display() {
  if ! DISPLAY="$DISPLAY_NUM" xdpyinfo > /dev/null 2>&1; then
    log "starting Xvfb on $DISPLAY_NUM"
    setsid Xvfb "$DISPLAY_NUM" -screen 0 1280x1024x24 \
      > "$WORK/xvfb.log" 2>&1 < /dev/null &
    XVFB_PID=$!
    sleep 2
    DISPLAY="$DISPLAY_NUM" xdpyinfo > /dev/null 2>&1 \
      || die "Xvfb failed to start on $DISPLAY_NUM"
  fi
}

# ── Phase 1: build wheel + plugin zip (qgis_stable) ──────────────────────────
log "Phase 1: build wheel + plugin zip"
rm -rf "$WORK"
mkdir -p "$WHEEL_DIR" "$RUN_DIR"
bash -c "$HOOK qgis_stable && \
  CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  CMAKE_ARGS=\"-DCMAKE_CUDA_COMPILER=\$(command -v nvcc);-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13\" \
  NVCC_PREPEND_FLAGS='' \
  python3 -m build --wheel --no-isolation -o '$WHEEL_DIR'"
bash -c "$HOOK qgis_stable && python3 tools/package_plugin.py"
ZIP="$(ls "$REPO_ROOT"/dist/HYDRA2DGPU-*.zip 2>/dev/null | head -1)"
test -f "$ZIP" || die "package_plugin.py did not produce a dist/HYDRA2DGPU-*.zip"
log "plugin zip: $ZIP"
WHEEL="$(ls "$WHEEL_DIR"/*.whl | head -1)"
[ -n "$WHEEL" ] || die "no wheel built"
log "wheel: $WHEEL"

# ── Phase 2: local wheel server (manylinux name; see installer.wheel_name) ──
log "Phase 2: local wheel server on $WHEEL_URL_BASE"
MANYLINUX_NAME="hydra_swe2d-$VERSION-cp312-cp312-manylinux_2_28_x86_64.whl"
# installer.py resolves RELEASE_TAG = f"v{WHEEL_VERSION}" — the URL is
# /v<version>/<wheel>, so the served directory MUST carry the v prefix.
mkdir -p "$HTTP_DIR/v$VERSION"
ln -sf "$WHEEL" "$HTTP_DIR/v$VERSION/$MANYLINUX_NAME"
setsid python3 -m http.server "$PORT" --directory "$HTTP_DIR" \
  > "$WORK/http.log" 2>&1 < /dev/null &
sleep 1
curl -fsSI "$WHEEL_URL_BASE/v$VERSION/$MANYLINUX_NAME" > /dev/null \
  || die "local wheel server unreachable"

# ── Phase 3: ensure qgis_clean has QGIS + CUDA runtime ───────────────────────
log "Phase 3: ensure $ENV_NAME env (qgis, numpy, gmsh, cuda-cudart=13.2)"
if ! mamba env list | grep -q "$ENV_NAME"; then
  bash -c "$HOOK base && mamba create -n $ENV_NAME -c conda-forge -y \
    python=3.12 qgis=3.44.9 numpy=2.4.6 gmsh=4.15.2 pyqt=5.15.11"
fi
# The wheel's hydra_swe2d*.so links libcudart.so.13 (built against CUDA 13.2).
if ! bash -c "$HOOK $ENV_NAME && python3 -c 'import ctypes; ctypes.CDLL(\"libcudart.so.13\")'" \
    2>/dev/null; then
  bash -c "$HOOK $ENV_NAME && mamba install -n $ENV_NAME -c conda-forge -y cuda-cudart=13.2"
fi
# Test-env deps that qgis/conda-forge do not pull in: pyqtgraph (workbench
# runtime, also a wheel dep), scipy + netCDF4 (GPU test result export).
# pip, not mamba: the conda solver stalls for many minutes on the qgis
# env's huge dependency graph.
if ! bash -c "$HOOK $ENV_NAME && python3 -c 'import pyqtgraph, scipy, netCDF4'" \
    2>/dev/null; then
  bash -c "$HOOK $ENV_NAME && pip install pyqtgraph scipy netCDF4"
fi

# ── Phase 4: install the plugin zip into the release_test profile ────────────
log "Phase 4: install $ZIP into $PROFILE profile (no dev symlinks)"
rm -rf "$PROFILE_DIR" "$INSTALL_DIR"
mkdir -p "$PROFILE_PLUGINS"
unzip -q "$ZIP" -d "$PROFILE_PLUGINS"
test -f "$PROFILE_PLUGINS/HYDRA2DGPU/metadata.txt" \
  || die "zip did not extract a HYDRA2DGPU/ plugin folder"
# Enable the plugin in the profile (Plugin Manager does this on install).
mkdir -p "$PROFILE_DIR/QGIS"
cat > "$PROFILE_DIR/QGIS/QGIS3.ini" <<EOF
[PythonPlugins]
hydra2dgpu=true
EOF

# ── Phase 5: production BackendInstaller pulls the wheel from the local host ─
log "Phase 5: headless BackendInstaller (production install path)"
bash -c "$HOOK $ENV_NAME && cd /tmp && \
  HYDRA_SWE2D_WHEEL_URL='$WHEEL_URL_BASE' \
  PYTHONPATH='$PROFILE_PLUGINS' \
  python3 -c '
from HYDRA2DGPU.installer import BackendInstaller
BackendInstaller(\"$PROFILE_PLUGINS/HYDRA2DGPU\", version=\"$VERSION\").install(progress=print)
'"
test -f "$VENV_SP/hydra_swe2d/__init__.py" || die "wheel not installed into $VENV_SP"
grep -q "GET .*$MANYLINUX_NAME" "$WORK/http.log" \
  || die "wheel server log shows no download for $MANYLINUX_NAME"

# ── Phase 6: launch the INSTALLED plugin's QGIS (autostarts MCP bridge) ─────
log "Phase 6: launch $ENV_NAME QGIS ($PROFILE profile) with MCP bridge"
ensure_display
# First boot initializes the profile and writes its own QGIS3.ini —
# hand-seeded [PythonPlugins] entries are dropped on a fresh profile.
# NOTE: no HYDRA_MCP_BRIDGE here — this boot is only for profile init,
# and its bridge token would confuse the Phase-6 wait below.
bash -c "$HOOK $ENV_NAME && \
  env -u PYTHONPATH DISPLAY=$DISPLAY_NUM timeout 25 \
  qgis --noversioncheck --profile $PROFILE \
  > '$WORK/qgis-firstboot.log' 2>&1 < /dev/null" || true
# The timeout(1) SIGTERM only begins shutdown — the first-boot QGIS can
# still hold the profile lock when it returns.  A second instance on the
# same profile exits silently (QGIS 3.44 profile lock), so wait for the
# process to be fully gone before relaunching.
for _ in $(seq 1 20); do
  pgrep -x qgis > /dev/null || break
  sleep 1
done
for _ in $(seq 1 30); do
  [ -f "$PROFILE_DIR/QGIS/QGIS3.ini" ] && break
  sleep 1
done
# Enable the plugin the way Plugin Manager would (GUI action -> ini flag).
# The key MUST be the folder name ("HYDRA2DGPU"): QGIS 3.44 matches
# [PythonPlugins] keys against the plugin FOLDER, and the production zip
# extracts to HYDRA2DGPU/ (the dev symlink is lowercase hydra2dgpu, which
# is why the lowercase key works in dev but not here).  Note a fresh
# profile inherits a *lowercase* hydra2dgpu=true from the default profile
# (QgsSettings fallback) — useless for the uppercase folder.
if ! grep -q "^HYDRA2DGPU=true" "$PROFILE_DIR/QGIS/QGIS3.ini"; then
  sed -i '/^\[PythonPlugins\]/a HYDRA2DGPU=true' "$PROFILE_DIR/QGIS/QGIS3.ini"
fi
grep -q "^HYDRA2DGPU=true" "$PROFILE_DIR/QGIS/QGIS3.ini" \
  || die "could not enable HYDRA2DGPU in $PROFILE_DIR/QGIS/QGIS3.ini"
# Second boot: the real session with the bridge. setsid + disown keep
# QGIS alive after the inner bash -c exits (a plain `&` leaves it
# vulnerable to SIGHUP when the wrapper shell dies).
rm -f "$TOKEN_DIR"/hydra_mcp_bridge_*.json
bash -c "$HOOK $ENV_NAME && \
  env -u PYTHONPATH HYDRA_MCP_BRIDGE=1 DISPLAY=$DISPLAY_NUM \
  setsid qgis --noversioncheck --profile $PROFILE \
  > '$WORK/qgis.log' 2>&1 < /dev/null & disown"
sleep 5
if ! pgrep -x qgis > /dev/null; then
  echo "--- qgis.log tail:"; tail -10 "$WORK/qgis.log"
  die "second-boot QGIS exited early (profile lock?)"
fi
for _ in $(seq 1 120); do
  BRIDGE_TOKEN_FILE="$(ls "$TOKEN_DIR"/hydra_mcp_bridge_*.json 2>/dev/null | head -1 || true)"
  [ -n "$BRIDGE_TOKEN_FILE" ] && break
  if grep -q "died on signal" "$WORK/qgis.log" 2>/dev/null; then
    echo "--- qgis.log tail:"; tail -5 "$WORK/qgis.log"
    die "second-boot QGIS crashed (signal 11 boot flake)"
  fi
  sleep 1
done
[ -n "$BRIDGE_TOKEN_FILE" ] || { echo "--- qgis.log tail:"; tail -20 "$WORK/qgis.log"; die "bridge token never appeared"; }
log "bridge token: $BRIDGE_TOKEN_FILE"

# ── Phase 7: run the FULL suite against the installed code ───────────────────
log "Phase 7: full test suite against the production-path install"
cp "$REPO_ROOT/tools/fast_fail.sh" "$RUN_DIR/"
# fast_fail.sh cds to the parent of its own dir, so its CWD must contain
# tests/ (a symlink to the wheel's copy — the wheel is the source of
# truth here, NOT the repo).
ln -sfn "$VENV_SP/tests" "$WORK/tests"
# Some self-tests read static repo fixtures by relative path (they are
# repo-layout self-tests, not production code).  fast_fail.sh cds to the
# parent of its own dir ($WORK), so the fixtures must live THERE.  Stage
# single files/dirs only — a full tools/ symlink would shadow the
# wheel's tools/ via CWD on sys.path.
ln -sfn "$REPO_ROOT/.opencode" "$WORK/.opencode"
mkdir -p "$WORK/tools"
ln -sfn "$REPO_ROOT/tools/wrap_pytest_style.py" "$WORK/tools/wrap_pytest_style.py"
bash -c "$HOOK $ENV_NAME && cd '$RUN_DIR' && \
  export PYTHONPATH='$VENV_SP:$PROFILE_PLUGINS:'\"\$PYTHONPATH\" && \
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\" && \
  export HYDRA_MCP_INTEGRATION=1 && \
  bash ./fast_fail.sh"
log "Phase 7b: GPU validation tests"
bash -c "$HOOK $ENV_NAME && cd '$RUN_DIR' && \
  export PYTHONPATH='$VENV_SP:$PROFILE_PLUGINS:'\"\$PYTHONPATH\" && \
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\" && \
  python3 -m unittest -v \
    tests.test_swe2d_gpu_validation_perf \
    tests.test_swe2d_gpu_unstructured \
    tests.test_swe2d_gpu_dambreak"

log "ALL DONE — production-path install + full test suite passed"
