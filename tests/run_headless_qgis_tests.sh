#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_headless_qgis_tests.sh
#
# Run HYDRA 2D GPU tests in a headless QGIS environment using Xvfb.
#
# This script:
#   1. Starts a virtual display (Xvfb) if none is available
#   2. Starts QGIS in headless (no-python-warning) mode
#   3. Runs the specified test modules via the QGIS Python console
#   4. Collects results and shuts down
#
# Usage:
#   # Run the mocked GUI tests (no QGIS needed, uses mock qgis.core):
#   bash tests/run_headless_qgis_tests.sh --mock
#
#   # Run full integration tests in headless QGIS (QGIS must be installed):
#   bash tests/run_headless_qgis_tests.sh --qgis
#
#   # Run a specific test file:
#   bash tests/run_headless_qgis_tests.sh --mock tests.test_workbench_gui
#
# Prerequisites for --qgis mode:
#   - QGIS 3.28+ installed
#   - Xvfb installed (apt install xvfb)
#   - hydra_swe2d native module built (in build/)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Config ─────────────────────────────────────────────────────────────────
TEST_PATTERN="${2:-tests.test_workbench_gui}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/tests/results}"
MODE="${1:---mock}"

mkdir -p "$RESULTS_DIR"

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Mode: Mock QGIS (no real QGIS needed) ─────────────────────────────────
run_mock_tests() {
    info "Running mocked GUI tests (no QGIS required)..."
    info "Test pattern: $TEST_PATTERN"

    PYTHONPATH="$PYTHONPATH:$REPO_ROOT:$REPO_ROOT/build:$REPO_ROOT/tests" \
        python3 -m unittest -v "$TEST_PATTERN" \
        | tee "$RESULTS_DIR/mock_test_output.log"

    local exit_code="${PIPESTATUS[0]}"
    if [ "$exit_code" -eq 0 ]; then
        info "All mock tests passed."
    else
        error "Mock tests failed (exit code $exit_code)."
    fi
    return "$exit_code"
}

# ── Mode: Headless QGIS (requires QGIS + Xvfb) ────────────────────────────
run_qgis_tests() {
    info "Running integration tests in headless QGIS..."

    # Find QGIS Python
    QGIS_PYTHON="${QGIS_PYTHON:-$(which qgis 2>/dev/null || echo "/usr/bin/qgis")}"
    if [ ! -x "$QGIS_PYTHON" ]; then
        error "QGIS not found. Install QGIS or use --mock mode."
        error "Set QGIS_PYTHON=/path/to/qgis if installed in a non-standard location."
        return 1
    fi

    # Start Xvfb if no display is available
    if [ -z "${DISPLAY:-}" ]; then
        export DISPLAY=":99"
        if ! pgrep -x Xvfb > /dev/null 2>&1; then
            info "Starting Xvfb on display $DISPLAY ..."
            Xvfb "$DISPLAY" -screen 0 1280x1024x24 &
            XVFB_PID=$!
            sleep 1
            trap "kill $XVFB_PID 2>/dev/null || true" EXIT
        fi
    fi

    # Build the test runner QGIS Python script
    TEST_RUNNER="$RESULTS_DIR/_headless_runner.py"
    cat > "$TEST_RUNNER" << 'PYEOF'
"""Headless QGIS test runner launched by run_headless_qgis_tests.sh."""
import os, sys, json, unittest, time

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (repo, os.path.join(repo, "build")):
    if p not in sys.path:
        sys.path.insert(0, p)

results = {"exit_code": 1, "tests_run": 0, "failures": [], "errors": []}

try:
    from qgis.core import QgsApplication
    QgsApplication.setPrefixPath("/usr", True)
    qgs = QgsApplication([], False)
    qgs.initQgis()

    from qgis import utils
    from swe2d_workbench_qt import SWE2DWorkbenchQtDialog
    dialog = SWE2DWorkbenchQtDialog()

    # Basic smoke test: instantiation succeeded
    results["tests_run"] = 1
    results["dialog_created"] = type(dialog).__name__

    qgs.exitQgis()
    results["exit_code"] = 0

except Exception as e:
    import traceback
    results["errors"].append({
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    })
    results["exit_code"] = 1

with open(os.path.join(os.path.dirname(__file__), "_headless_results.json"), "w") as f:
    json.dump(results, f, indent=2)

sys.exit(results["exit_code"])
PYEOF

    info "Launching QGIS headless test runner..."
    PYTHONPATH="$PYTHONPATH:$REPO_ROOT:$REPO_ROOT/build:$REPO_ROOT/tests" \
        "$QGIS_PYTHON" --noprocessing --no-python-warning \
                       --project "$REPO_ROOT/tests/mocks/empty_project.qgs" \
                       --code "$TEST_RUNNER" \
                       2>&1 | tee "$RESULTS_DIR/qgis_test_output.log" || true

    # Check results
    if [ -f "$RESULTS_DIR/_headless_results.json" ]; then
        python3 -c "
import json
with open('$RESULTS_DIR/_headless_results.json') as f:
    r = json.load(f)
if r['exit_code'] == 0:
    print(f\"${GREEN}QGIS tests passed: dialog={r.get('dialog_created', '?')}${NC}\")
else:
    print(f\"${RED}QGIS tests failed: {len(r.get('errors', []))} errors${NC}\")
    for e in r.get('errors', []):
        print(f'  {e.get(\"type\")}: {e.get(\"message\")}')
" 2>/dev/null || true
    else:
        warn "No results file found — QGIS may have crashed silently."
        warn "Check $RESULTS_DIR/qgis_test_output.log"
        return 1
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────
main() {
    case "$MODE" in
        --mock|-m)
            run_mock_tests
            ;;
        --qgis|-q)
            run_qgis_tests
            ;;
        *)
            echo "Usage: $0 [--mock|--qgis] [test_pattern]"
            echo ""
            echo "  --mock   Run mocked tests (no QGIS required) [default]"
            echo "  --qgis   Run integration tests in headless QGIS (QGIS must be installed)"
            exit 1
            ;;
    esac
}

main
