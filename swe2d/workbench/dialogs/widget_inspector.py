"""Widget Inspector — polling-only. Zero event filters. Cannot crash QGIS."""
from qgis.PyQt.QtCore import QObject, QTimer
from qgis.PyQt.QtCore import Qt as _Qt
from qgis.PyQt.QtGui import QCursor
from qgis.PyQt.QtWidgets import QApplication, QMessageBox
import os, subprocess

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_POLLER = None


class _ClickWatcher(QObject):
    STATE_IDLE = 0       # waiting for left-button to be NOT pressed (all clear)
    STATE_ARMED = 1      # detected clear state, now waiting for press
    STATE_PRESSED = 2    # detected press, now waiting for release

    def __init__(self):
        super().__init__()
        self._state = _ClickWatcher.STATE_IDLE
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)

    def _poll(self):
        """Poll mouse button state and advance the click-detection state machine."""
        pressed = bool(QApplication.mouseButtons() & _Qt.LeftButton)
        if self._state == _ClickWatcher.STATE_IDLE:
            if not pressed:
                self._state = _ClickWatcher.STATE_ARMED
        elif self._state == _ClickWatcher.STATE_ARMED:
            if pressed:
                self._state = _ClickWatcher.STATE_PRESSED
        elif self._state == _ClickWatcher.STATE_PRESSED:
            if not pressed:
                self._timer.stop()
                QTimer.singleShot(0, _probe)


def _search_dirs():
    """Return project source directories for grep-based class search."""
    return [os.path.join(_PROJECT_ROOT, d) for d in ("swe2d", "cpp", "tests", "hydra_plugin.py")
            if os.path.exists(os.path.join(_PROJECT_ROOT, d))]


def _append_line(lines, text):
    """Append a line to an output list (helper for _probe)."""
    lines.append(text)


def _probe():
    """Identify the widget under the cursor and show details in a message box."""
    w = QApplication.widgetAt(QCursor.pos())
    if not w:
        QMessageBox.information(None, "Widget Inspector", "No widget at cursor position.")
        return
    cls_name = type(w).__name__
    oname = w.objectName()
    lines = [f"Class: {cls_name}", f"ObjectName: \"{oname}\""]
    dirs = _search_dirs()
    lines.append("Defined in:")
    found_def = False
    try:
        r = subprocess.run(["grep", "-rn", "-F", f"class {cls_name}"] + dirs,
                           capture_output=True, text=True, timeout=5)
        for ln in r.stdout.strip().splitlines()[:1]:
            _append_line(lines, ln)
            found_def = True
    except Exception as _e:

        logger.warning(f"[ERROR] Exception in widget_inspector.py: {_e}")
    try:
        r = subprocess.run(["grep", "-rln", "-F", f"{cls_name}("] + dirs,
                           capture_output=True, text=True, timeout=5)
        src_files = set(r.stdout.strip().splitlines())
        if oname:
            r2 = subprocess.run(["grep", "-rln", f'setObjectName("{oname}")'] + dirs,
                                capture_output=True, text=True, timeout=5)
            src_files |= set(r2.stdout.strip().splitlines())
        shown = 0
        for fn in sorted(src_files):
            if fn and shown < 5:
                rel = os.path.relpath(fn, _PROJECT_ROOT)
                _append_line(lines, f"  {rel}")
                shown += 1
    except Exception as _e:

        logger.warning(f"[ERROR] Exception in widget_inspector.py: {_e}")
    if not found_def and not src_files:
        _append_line(lines, "  (not found in project source)")
    cur = w.parent()
    chain = []
    while cur and len(chain) < 8:
        cn = cur.objectName()
        chain.append(f"{type(cur).__name__} \"{cn}\"" if cn else type(cur).__name__)
        if cn:
            break
        cur = cur.parent()
    if chain:
        lines.append("Parent: " + " → ".join(chain))
    QMessageBox.information(None, "Widget Inspector", "\n".join(lines))


def arm():
    """Arm one-shot widget inspector via polling (no event filters).

    Sequence:
      1. Wait 800ms for trigger click to fully settle.
      2. Start 80ms poll cycle: IDLE → wait for button clear →
         ARMED → wait for button press → PRESSED → wait for button release → PROBE.

    The 3-state cycle ensures the trigger click is always ignored.
    """
    global _POLLER
    QTimer.singleShot(800, _start)


def _start():
    """Start the click-watcher poller after the initial settle delay."""
    global _POLLER
    _POLLER = _ClickWatcher()
