#!/usr/bin/env bash
set -euo pipefail

# Launch QGIS from the conda qgis_stable env without conflicting virtualenv contamination.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure conda activation works in non-interactive shells.
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
  echo "Missing conda init script at ${HOME}/miniforge3/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "qgis_stable" ]]; then
  set +u
  conda activate qgis_stable
  set -u
fi

# If a venv is active, remove it from PATH and clear VIRTUAL_ENV.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PATH=":${PATH}:"
  PATH="${PATH//:${VIRTUAL_ENV//\//\/}\/bin:/:}"
  PATH="${PATH#:}"
  PATH="${PATH%:}"
  unset VIRTUAL_ENV
fi

# Avoid Python interpreter/path contamination that breaks PyQGIS startup.
unset PYTHONHOME || true
unset PYTHONPATH || true
unset PYTHONSTARTUP || true
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

if [[ "${1:-}" == "--doctor" ]]; then
  python - <<'PY'
import sys
import math
print("python:", sys.executable)
print("math:", math.__file__ if hasattr(math, "__file__") else "built-in")
print("sys.path:")
for p in sys.path:
    print(" ", p)
PY
  exit 0
fi

exec qgis "$@"
