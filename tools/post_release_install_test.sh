#!/usr/bin/env bash
# Post-release production-path install test (real GitHub artifacts).
#
# Unlike release_install_test.sh (which builds the wheel locally and serves
# it from 127.0.0.1), this gate pulls the ACTUAL assets from the published
# GitHub Release and runs the production install path end-to-end against them:
#
#   - gh release download fetches HYDRA2DGPU-<ver>.zip from the release
#   - the zip is installed into the isolated qgis_clean release_test profile
#   - the production BackendInstaller runs WITHOUT HYDRA_SWE2D_WHEEL_URL, so
#     it probes api.github.com and pip-installs the manylinux wheel straight
#     from the real GitHub release (the exact end-user first-launch path)
#   - the FULL test suite runs against the installed code
#
# Usage: bash tools/post_release_install_test.sh [vX.Y.Z]
#   (default tag: from pyproject.toml version as v<version>)
# Requires: gh (authenticated), mamba, qgis_clean env, system NVIDIA driver,
#           ~10 min.
#
# This script deliberately avoids local-build shortcuts: a passing run means
# the shipped zip + wheel on GitHub actually install and pass the suite.

set -euo pipefail

MAMBA="${MAMBA:-}"
if [ -z "$MAMBA" ]; then
  MAMBA="$(command -v mamba || true)"
fi
if [ -z "$MAMBA" ]; then
  die "mamba not found — set MAMBA to the mamba binary path"
fi
command -v gh >/dev/null 2>&1 || die "gh CLI not found — required to download release assets"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VERSION=$(sed -nE 's/^version\s*=\s*"([^"]+)".*/\1/p' pyproject.toml | head -1)
TAG="${1:-v$VERSION}"
REPO="aspragueumkc/hydra2dgpu"

WORK=/tmp/hydra-post-release-test
RUN_DIR="$WORK/run"
DISPLAY_NUM=:99
ENV_NAME=qgis_clean
PROFILE=release_test
PROFILE_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/$PROFILE"
PROFILE_PLUGINS="$PROFILE_DIR/python/plugins"
INSTALL_DIR="$HOME/.hydra2dgpu"
VENV_SP="$INSTALL_DIR/lib/python3.12/site-packages"
TOKEN_DIR="${XDG_RUNTIME_DIR:-/tmp}"

log() { printf '\n== %s ==\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# shellcheck disable=SC2016
HOOK='eval "$("'"$MAMBA"'" shell hook --shell bash)" && mamba activate'

XVFB_PID=""

cleanup() {
  set +e
  pkill -f "qgis --noversioncheck --profile $PROFILE" 2>/dev/null
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

# ── Phase 1: verify + download the release artifacts from GitHub ─────────────
log "Phase 1: verify assets on release $TAG + download plugin zip"
rm -rf "$WORK"
mkdir -p "$RUN_DIR"
gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name' \
  > "$WORK/assets.txt"
grep -q "HYDRA2DGPU-$VERSION.zip" "$WORK/assets.txt" \
  || die "release $TAG missing HYDRA2DGPU-$VERSION.zip (assets: $(tr '\n' ' ' < "$WORK/assets.txt"))"
grep -q "hydra_swe2d-$VERSION-cp312-cp312-manylinux_2_28_x86_64.whl" "$WORK/assets.txt" \
  || die "release $TAG missing the manylinux wheel (assets: $(tr '\n' ' ' < "$WORK/assets.txt"))"

gh release download "$TAG" --repo "$REPO" \
  --pattern "HYDRA2DGPU-$VERSION.zip" --dir "$WORK" --skip-existing
ZIP="$WORK/HYDRA2DGPU-$VERSION.zip"
test -f "$ZIP" || die "gh release download did not produce $ZIP"
log "downloaded: $ZIP"
unzip -l "$ZIP" >/dev/null || die "downloaded zip is corrupt"

# ── Phase 2: ensure qgis_clean env (qgis, numpy, gmsh, cuda-cudart) ─────────
log "Phase 2: ensure $ENV_NAME env"
if ! mamba env list | grep -q "$ENV_NAME"; then
  bash -c "$HOOK base && mamba create -n $ENV_NAME -c conda-forge -y \
    python=3.12 qgis=3.44.9 numpy=2.4.6 gmsh=4.15.2 pyqt=5.15.11"
fi
# The wheel's hydra_swe2d*.so links libcudart.so.13 (built against CUDA 13.2).
if ! bash -c "$HOOK $ENV_NAME && python3 -c 'import ctypes; ctypes.CDLL(\"libcudart.so.13\")'" \
    2>/dev/null; then
  bash -c "$HOOK $ENV_NAME && mamba install -n $ENV_NAME -c conda-forge -y cuda-cudart=13.2"
fi
if ! bash -c "$HOOK $ENV_NAME && python3 -c 'import pyqtgraph, scipy, netCDF4'" \
    2>/dev/null; then
  bash -c "$HOOK $ENV_NAME && pip install pyqtgraph scipy netCDF4"
fi

# ── Phase 3: install the downloaded plugin zip into the release_test profile ─
log "Phase 3: install $ZIP into $PROFILE profile"
rm -rf "$PROFILE_DIR" "$INSTALL_DIR"
mkdir -p "$PROFILE_PLUGINS"
unzip -q "$ZIP" -d "$PROFILE_PLUGINS"
test -f "$PROFILE_PLUGINS/HYDRA2DGPU/metadata.txt" \
  || die "zip did not extract a HYDRA2DGPU/ plugin folder"
mkdir -p "$PROFILE_DIR/QGIS"
cat > "$PROFILE_DIR/QGIS/QGIS3.ini" <<EOF
[PythonPlugins]
hydra2dgpu=true
EOF

# ── Phase 4: production BackendInstaller pulls the wheel from GitHub ─────────
log "Phase 4: headless BackendInstaller (real GitHub release URL)"
# No HYDRA_SWE2D_WHEEL_URL override → installer probes api.github.com and
# pip-installs the manylinux wheel straight from the release.
bash -c "$HOOK $ENV_NAME && cd /tmp && \
  PYTHONPATH='$PROFILE_PLUGINS' \
  python3 -c '
from HYDRA2DGPU.installer import BackendInstaller
BackendInstaller(\"$PROFILE_PLUGINS/HYDRA2DGPU\", version=\"$VERSION\").install(progress=print)
'"
test -f "$VENV_SP/hydra_swe2d/__init__.py" \
  || die "wheel not installed into $VENV_SP from GitHub release"

# ── Phase 5: launch the INSTALLED plugin's QGIS (autostarts MCP bridge) ─────
log "Phase 5: launch $ENV_NAME QGIS ($PROFILE profile) with MCP bridge"
ensure_display
bash -c "$HOOK $ENV_NAME && \
  env -u PYTHONPATH DISPLAY=$DISPLAY_NUM timeout 25 \
  qgis --noversioncheck --profile $PROFILE \
  > '$WORK/qgis-firstboot.log' 2>&1 < /dev/null" || true
for _ in $(seq 1 20); do
  pgrep -x qgis > /dev/null || break
  sleep 1
done
for _ in $(seq 1 30); do
  [ -f "$PROFILE_DIR/QGIS/QGIS3.ini" ] && break
  sleep 1
done
if ! grep -q "^HYDRA2DGPU=true" "$PROFILE_DIR/QGIS/QGIS3.ini"; then
  sed -i '/^\[PythonPlugins\]/a HYDRA2DGPU=true' "$PROFILE_DIR/QGIS/QGIS3.ini"
fi
grep -q "^HYDRA2DGPU=true" "$PROFILE_DIR/QGIS/QGIS3.ini" \
  || die "could not enable HYDRA2DGPU in $PROFILE_DIR/QGIS/QGIS3.ini"
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
BRIDGE_TOKEN_FILE=""
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

# ── Phase 6: run the FULL suite against the installed code ───────────────────
log "Phase 6: full test suite against the release artifacts"
cp "$REPO_ROOT/tools/fast_fail.sh" "$RUN_DIR/"
ln -sfn "$VENV_SP/tests" "$WORK/tests"
ln -sfn "$REPO_ROOT/.opencode" "$WORK/.opencode"
mkdir -p "$WORK/tools"
ln -sfn "$REPO_ROOT/tools/wrap_pytest_style.py" "$WORK/tools/wrap_pytest_style.py"
bash -c "$HOOK $ENV_NAME && cd '$RUN_DIR' && \
  export PYTHONPATH='$VENV_SP:$PROFILE_PLUGINS:'\"\$PYTHONPATH\" && \
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\" && \
  export HYDRA_MCP_INTEGRATION=1 && \
  bash ./fast_fail.sh"
log "Phase 6b: GPU validation tests"
bash -c "$HOOK $ENV_NAME && cd '$RUN_DIR' && \
  export PYTHONPATH='$VENV_SP:$PROFILE_PLUGINS:'\"\$PYTHONPATH\" && \
  export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib\" && \
  python3 -m unittest -v \
    tests.test_swe2d_gpu_validation_perf \
    tests.test_swe2d_gpu_unstructured \
    tests.test_swe2d_gpu_dambreak"

log "ALL DONE — post-release install from GitHub artifacts + full suite passed"
