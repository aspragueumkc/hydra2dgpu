#!/usr/bin/env bash
# Launch QGIS from the qgis_stable conda environment with clean .pyc cache.
set -e

ENV_NAME="qgis_stable"
ENV_BASE="/home/aaron/miniforge3/envs/$ENV_NAME"
WORKSPACE="/home/aaron/QGIS_Plugins_dev/qgis-backwater-plugin-GPU_ONLY"

# If not already in the target env, try to activate it
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    # shellcheck disable=SC1091
    source /home/aaron/miniforge3/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
fi

# Purge all Python bytecode cache in the workspace
find "$WORKSPACE" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find "$WORKSPACE" -name "*.pyc" -delete 2>/dev/null

# Launch QGIS using the env's qgis binary
exec "$ENV_BASE/bin/qgis" "$@"
