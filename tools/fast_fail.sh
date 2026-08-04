#!/usr/bin/env bash
# tools/fast_fail.sh — HYDRA fast-fail test orchestrator
#
# Stages ordered cheapest-to-most-expensive. Stage failure halts.
# Pre-commit runs this with HYDRA_MCP_INTEGRATION unset (stages 1-3).
# Pre-PR / qgis-smoke-test sets HYDRA_MCP_INTEGRATION=1 (stages 1-4).
#
# Spec: docs/specs/2026-07-26-test-discipline-design.md §3.1
set -uo pipefail
cd "$(dirname "$0")/.."

stage() {
  local name=$1; shift
  printf "\n== %s ==\n" "$name"
  if "$@"; then
    return 0
  else
    return 1
  fi
}

# Gate 1: collect-only (catches broken imports, worktree-merge drift)
# Note: --collect-only was removed from unittest.discover in 3.12. Use a
# small Python helper that calls TestLoader.discover() and exits 1 on any
# collection-time error.
stage "collect-only" python3 -c "
import sys, unittest
try:
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py', top_level_dir='.')
except Exception as e:
    print(f'Collection failed: {e}', file=sys.stderr)
    sys.exit(1)
print(f'Collected {suite.countTestCases()} tests OK')
" || exit 1

# Gate 2: self-tests (catches LLM-spec drift, test theatre, stale skips)
stage "self-tests" python3 -m unittest discover \
  -s tests/test_self -p 'test_*.py' -v || exit 1

# Gate 3: wired-in (catches Qt shim regression, CLI/GUI parity).
# Two separate invocations — running both in one python process causes a
# QApplication singleton conflict between the two test modules.
stage "wired-in:qt-shim" python3 -m unittest -v \
  tests.test_workbench_persistence.TestQtModuleClassifierUnderRealQGIS || exit 1
stage "wired-in:parity" python3 -m unittest -v \
  tests.test_run_context_parity || exit 1

# Gate 3.5: gui-behavioral (catches real headless-QGIS GUI regressions).
# One process — all modules share the ensure_qgis_app() singleton safely.
# Spec: docs/specs/2026-08-02-gui-test-coverage-design.md §6.
# Import the collector before unittest imports any Qt/pyqtgraph module.  It
# turns asynchronous callback tracebacks into a hard stage failure.
stage "gui-behavioral" env QT_API=pyqt5 PYQTGRAPH_QT_LIB=PyQt5 \
  python3 -m tests.fail_on_qt_callback_errors \
  tests.test_qgis_real_env tests.test_graph_editor_service \
  tests.test_graph_editor_dialog tests.test_hydrograph_editor \
  tests.test_topo_attr_table_dialog tests.test_gui_services \
  tests.test_studio_viewer_pg tests.test_studio_viewer \
  tests.test_temporal_dock tests.test_map_tools \
  tests.test_profile_chain_widget tests.test_detached_dialogs \
  tests.test_gpkg_inspection_dialogs \
  tests.test_viewer_dialogs tests.test_workbench_wiring \
  tests.test_workbench_api_misc tests.test_signal_helpers \
  tests.test_pyqtgraph_qvariant tests.test_qt_callback_collector || exit 1

# Gate 4: MCP widget walk (only when HYDRA_MCP_INTEGRATION=1)
if [ "${HYDRA_MCP_INTEGRATION:-0}" = "1" ]; then
  stage "mcp-smoke" python3 -m unittest -v \
    tests.test_mcp_widget_walk_smoke || exit 1
  stage "mcp-walk" python3 -m unittest -v \
    tests.test_mcp_widget_walk || exit 1
else
  printf "\n== mcp-walk (SKIPPED, set HYDRA_MCP_INTEGRATION=1 to enable) ==\n"
fi

printf "\n== fast_fail: ALL STAGES PASSED ==\n"
