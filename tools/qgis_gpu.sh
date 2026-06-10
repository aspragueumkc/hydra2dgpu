#!/usr/bin/env bash
# Launch QGIS from the qgis_stable conda environment with clean .pyc cache.
set -e

ENV_NAME="qgis_stable"
# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# Auto-detect conda base from the active environment
if [ -n "${CONDA_PREFIX:-}" ]; then
    ENV_BASE="$(dirname "$CONDA_PREFIX")"
else
    echo "ERROR: conda not active. Please activate a conda environment first." >&2
    exit 1
fi

# If not already in the target env, try to activate it
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    # shellcheck disable=SC1091
    source "$ENV_BASE/../etc/profile.d/conda.sh" 2>/dev/null || \
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || {
        echo "ERROR: Cannot locate conda.sh. Ensure conda is installed." >&2
        exit 1
    }
    conda activate "$ENV_NAME"
fi

# Purge all Python bytecode cache in the workspace
find "$WORKSPACE" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find "$WORKSPACE" -name "*.pyc" -delete 2>/dev/null

# Launch QGIS using the env's qgis binary
exec "$ENV_BASE/bin/qgis" "$@"
