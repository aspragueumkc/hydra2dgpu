"""Pure-Python service: detect, fetch, and install the hydra-swe2d backend.

No Qt widgets here at the service level (MVP: the dialog is a separate
View class within this same module).
"""
from __future__ import annotations
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
from typing import Callable

WHEEL_VERSION = "1.3.0"
RELEASE_TAG = f"v{WHEEL_VERSION}"
GITHUB_RELEASES = "http://127.0.0.1:8765"
ENV_DIR_NAME = ".hydra2dgpu"
CACHE_DIR_ENV = "HYDRA2DGPU_CACHE_DIR"

ProgressFn = Callable[[str], None]


class BackendInstaller:
    """Service: detect, fetch, and install the hydra_swe2d wheel.

    Deliberately Qt-free. Receives data via constructor and method args;
    never reaches through to widgets.
    """

    def __init__(self, plugin_dir: str, version: str = WHEEL_VERSION):
        self.plugin_dir = plugin_dir
        self.version = version

    def env_dir(self) -> Path:
        override = os.environ.get(CACHE_DIR_ENV, "").strip()
        base = Path(override) if override else Path.home()
        return base / ENV_DIR_NAME

    def site_packages(self, env_dir: Path) -> Path | None:
        lib = env_dir / "lib"
        if not lib.exists():
            return None
        matches = sorted(lib.glob("python*/site-packages"))
        return matches[0] if matches else None

    def add_env_to_path(self, env_dir: Path) -> None:
        sp = self.site_packages(env_dir)
        if sp and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

    def backend_available(self) -> bool:
        try:
            import hydra_swe2d  # noqa: F401
            return True
        except Exception:
            return False

    def wheel_name(self) -> str:
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        sysname = platform.system().lower()
        if sysname == "linux":
            plat = "manylinux_2_28_x86_64"
        elif sysname == "windows":
            plat = "win_amd64"
        else:
            raise RuntimeError(f"Unsupported platform: {sysname}")
        return f"hydra_swe2d-{self.version}-{py}-{py}-{plat}.whl"

    def wheel_url(self) -> str:
        return f"{GITHUB_RELEASES}/{RELEASE_TAG}/{self.wheel_name()}"

    def install(self, progress: ProgressFn) -> None:
        env = self.env_dir()
        if not env.exists():
            progress(f"Creating isolated environment at {env}")
            venv.create(env, with_pip=True)

        sp = self.site_packages(env)
        if not sp:
            raise RuntimeError("site-packages not found in created environment")

        scripts = "Scripts" if platform.system() == "Windows" else "bin"
        pip = env / scripts / ("pip.exe" if platform.system() == "Windows" else "pip")

        progress("Upgrading pip in isolated environment...")
        self._run_pip([str(pip), "install", "--upgrade", "pip"], "pip upgrade", progress)

        url = self.wheel_url()
        progress(f"Downloading {url}")
        self._run_pip([str(pip), "install", url], f"wheel {url}", progress)

        self.add_env_to_path(env)
        import hydra_swe2d  # noqa: F401
        progress(f"Installed hydra_swe2d {self.version}")

    def _run_pip(self, args: list[str], desc: str, progress: ProgressFn) -> None:
        """Run pip, streaming each stdout/stderr line into the GUI progress
        log and surfacing the full captured output on failure.

        Bypasses `capture_output=True` (the previous silent-failure form
        that discarded pip's actual error message). Captured here only
        so we can raise a useful RuntimeError on non-zero exit; otherwise
        output streams line-by-line so the user sees pip's progress live.
        """
        progress(f"$ {' '.join(args)}")
        try:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"{desc}: failed to launch pip ({exc})") from exc
        assert proc.stdout is not None
        captured: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            captured.append(line)
            progress(f"  {line}")
        proc.wait()
        if proc.returncode != 0:
            joined = "\n".join(captured)
            raise RuntimeError(
                f"{desc} failed with exit code {proc.returncode}.\n"
                f"pip output:\n{joined}"
            )


# ─── View layer: InstallDialog ──────────────────────────────────────────────
# Qt imports are kept below this point and are lazy — the service class
# BackendInstaller above is importable in a Qt-less environment. The View
# classes (InstallDialog, _InstallThread) are built on first access via
# module-level __getattr__ (PEP 562). BackendInstaller never needs Qt.

_QT_BINDINGS = None  # populated lazily by _import_qt()


def _import_qt():
    """Lazy Qt import — keeps BackendInstaller service Qt-free at module
    load time. The InstallDialog View triggers this on demand. Cached
    after the first call so later attribute lookups stay cheap."""
    global _QT_BINDINGS
    if _QT_BINDINGS is not None:
        return _QT_BINDINGS
    from qgis.PyQt.QtCore import QThread, pyqtSignal
    from qgis.PyQt.QtWidgets import (
        QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton,
        QTextEdit, QMessageBox,
    )
    _QT_BINDINGS = (
        QThread, pyqtSignal, QDialog, QVBoxLayout, QLabel, QProgressBar,
        QPushButton, QTextEdit, QMessageBox,
    )
    return _QT_BINDINGS


def _build_view_classes():
    """Build the lazy Qt-dependent View classes. Triggers Qt import on
    first call. Result is cached so subsequent lookups are O(1)."""
    (_QThread, _pyqtSignal, _QDialog, _QVBoxLayout, _QLabel,
     _QProgressBar, _QPushButton, _QTextEdit, _QMessageBox) = _import_qt()

    class _InstallThread(_QThread):
        """Worker thread that runs BackendInstaller.install() off the UI
        thread. Avoids freezing QGIS during pip download."""

        progress = _pyqtSignal(str)
        finished = _pyqtSignal()
        error = _pyqtSignal(str)

        def __init__(self, installer: BackendInstaller, parent=None):
            super().__init__(parent)
            self._installer = installer

        def run(self) -> None:  # noqa: D401  (QThread API)
            try:
                self._installer.install(self.progress.emit)
                self.finished.emit()
            except Exception as exc:
                self.error.emit(str(exc))

    class InstallDialog(_QDialog):
        """View: a thin progress dialog that drives BackendInstaller via
        callbacks."""

        def __init__(self, installer: BackendInstaller, parent=None):
            super().__init__(parent)
            self._installer = installer
            self.setWindowTitle("Install HYDRA2DGPU Backend")
            self.setMinimumWidth(480)

            layout = _QVBoxLayout(self)
            self._label = _QLabel(
                "Downloading the HYDRA2DGPU solver backend.\n"
                f"Isolated environment: {self._installer.env_dir()}"
            )
            self._progress = _QProgressBar()
            self._progress.setRange(0, 0)
            self._progress.setVisible(False)
            self._log = _QTextEdit()
            self._log.setReadOnly(True)
            self._log.setVisible(False)
            self._btn = _QPushButton("Install")
            self._btn.clicked.connect(self._start)
            for w in (self._label, self._progress, self._log, self._btn):
                layout.addWidget(w)
            self._thread = None

        def _start(self) -> None:
            self._btn.setEnabled(False)
            self._progress.setVisible(True)
            self._log.setVisible(True)
            self._thread = _InstallThread(self._installer)
            self._thread.progress.connect(self._log.append)
            self._thread.finished.connect(self._on_finished)
            self._thread.error.connect(self._on_error)
            self._thread.start()

        def _on_finished(self) -> None:
            _QMessageBox.information(
                self, "Success",
                "Backend installed. Please restart QGIS.",
            )
            self.accept()

        def _on_error(self, msg: str) -> None:
            self._progress.setVisible(False)
            self._btn.setEnabled(True)
            _QMessageBox.critical(self, "Installation Failed", msg)

    return InstallDialog, _InstallThread


# Module-level lazy attribute access (PEP 562): looking up InstallDialog
# or _InstallThread triggers Qt import + class construction on first
# access, then caches in module globals for subsequent lookups.
_VIEW_CLASSES = None  # (InstallDialog, _InstallThread) once built


def __getattr__(name):
    """Lazy View-class loader. Keeps BackendInstaller importable in a
    Qt-less environment; the View classes are built only when first
    accessed by attribute lookup on this module."""
    global _VIEW_CLASSES
    if name in ("InstallDialog", "_InstallThread"):
        if _VIEW_CLASSES is None:
            _VIEW_CLASSES = _build_view_classes()
        # Inject into module globals so ``from .installer import X``
        # works on subsequent imports (PEP 562 only fires on miss).
        InstallDialog, _InstallThread = _VIEW_CLASSES
        globals()["InstallDialog"] = InstallDialog
        globals()["_InstallThread"] = _InstallThread
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")