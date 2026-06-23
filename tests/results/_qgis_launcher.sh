#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH=":$REPO_ROOT:$REPO_ROOT/build:$REPO_ROOT/tests"
exec qgis --noplugins --noversioncheck      --project "$REPO_ROOT/tests/mocks/empty_project.qgs"      --code "$REPO_ROOT/tests/results/_qgis_exec_script.py"
