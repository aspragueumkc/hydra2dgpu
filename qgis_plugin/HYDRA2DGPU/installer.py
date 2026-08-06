"""Pure-Python service: detect, fetch, and install the hydra-swe2d backend.

No Qt widgets here at the service level (MVP: the dialog is a separate
View class within this same module).
"""
from __future__ import annotations
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

WHEEL_VERSION = "0.3.0"
RELEASE_TAG = f"v{WHEEL_VERSION}"
# Source of truth for the wheel download URL. The plugin zip (built by
# tools/package_plugin.py) is installed by the QGIS Plugin Manager; the
# first-launch BackendInstaller then downloads the matching wheel from
# this URL into ~/.hydra2dgpu/. Override locally by setting
# HYDRA_SWE2D_WHEEL_URL — useful for CI smoke tests.
#
# GitHub release assets are served under /releases/download/<tag>/<name>,
# NOT /releases/<tag>/<name> (the latter 404s). The trailing
# "/releases/download" is part of the download base.
GITHUB_RELEASES = os.environ.get(
    "HYDRA_SWE2D_WHEEL_URL",
    "https://github.com/aspragueumkc/hydra2dgpu/releases/download",
)
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
        """Locate site-packages inside a venv.

        Layout differs by OS:
          - Windows: <env>/Lib/site-packages
          - POSIX:   <env>/lib/pythonX.Y/site-packages
        """
        win = env_dir / "Lib" / "site-packages"
        if win.is_dir():
            return win
        lib = env_dir / "lib"
        if lib.is_dir():
            matches = sorted(lib.glob("python*/site-packages"))
            if matches:
                return matches[0]
        return None

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

    def _wheel_candidates(self) -> list[str]:
        """Return plausible wheel filenames for this Python + platform.

        cibuildwheel emits filenames like
        ``hydra_swe2d-<v>-cp<python>-cp<python>-<plat>.whl``. We probe
        the GitHub release asset list at install-time and pick the first
        one that matches this interpreter + platform. Listing every
        candidate up front keeps the release upload free to use any
        cibuildwheel output naming without coordinating with this code.
        """
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        sysname = platform.system().lower()
        if sysname == "linux":
            # Newest cibuildwheel uses ``manylinux_2_28``; the
            # pre-2.20 toolchain used ``manylinux2014``. We try the
            # newer one first because that is what
            # .github/workflows/build-wheels.yml emits today.
            return [
                f"hydra_swe2d-{self.version}-{py}-{py}-manylinux_2_28_x86_64.whl",
                f"hydra_swe2d-{self.version}-{py}-{py}-manylinux2014_x86_64.whl",
                f"hydra_swe2d-{self.version}-{py}-{py}-manylinux_2_17_x86_64.whl",
            ]
        if sysname == "windows":
            return [
                f"hydra_swe2d-{self.version}-{py}-{py}-win_amd64.whl",
                f"hydra_swe2d-{self.version}-{py}-{py}-win_64.whl",
            ]
        raise RuntimeError(f"Unsupported platform: {sysname}")

    def wheel_url(self) -> str:
        """Resolve the wheel download URL by probing the GitHub Release
        asset list for the first candidate that exists.

        Falls back to the first candidate if the probe fails (offline,
        private mirror, rate-limited): ``pip install <url>`` still surfaces
        a clear 404 in the install dialog.
        """
        names = self._wheel_candidates()
        base = GITHUB_RELEASES.rstrip("/")
        if base.startswith("http://127.0.0.1"):
            # Local QA mirror — probe is meaningless; take the first.
            return f"{base}/{RELEASE_TAG}/{names[0]}"
        # GitHub release pages are /github.com/<owner>/<repo>/releases[/download]
        # — derive the API owner/repo from the path so HYDRA_SWE2D_WHEEL_URL
        # overrides keep working regardless of the trailing segment.
        segments = base.split("/")
        try:
            gh = segments.index("github.com")
            owner, repo = segments[gh + 1], segments[gh + 2]
        except (ValueError, IndexError):
            owner = repo = None
        if not owner or not repo:
            return f"{base}/{RELEASE_TAG}/{names[0]}"
        try:
            from urllib.request import urlopen
            from urllib.error import URLError
            from urllib.parse import urlparse
            api = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{RELEASE_TAG}"
            # Only ever open https: the URL is built from a fixed github.com
            # host plus owner/repo parsed out of the download base, so an
            # https-only guard here is the intended audit boundary (no
            # file:// or other schemes reach urlopen).
            if urlparse(api).scheme != "https":
                return f"{base}/{RELEASE_TAG}/{names[0]}"
            with urlopen(api, timeout=15) as resp:
                import json
                data = json.loads(resp.read().decode("utf-8"))
            assets = {a["name"] for a in data.get("assets", [])}
        except (URLError, TimeoutError, ValueError, KeyError):
            return f"{base}/{RELEASE_TAG}/{names[0]}"
        for name in names:
            if name in assets:
                return f"{base}/{RELEASE_TAG}/{name}"
        # Probe succeeded but no candidate matched — fall back to the
        # newest one and let pip surface a useful 404.
        return f"{base}/{RELEASE_TAG}/{names[0]}"

    def install(self, progress: ProgressFn) -> None:
        env = self.env_dir()
        if not env.exists():
            progress(f"Creating isolated environment at {env}")
            python = self._real_python()
            progress(f"  base interpreter: {python}")
            # Create the venv as a subprocess of the REAL Python. Running
            # venv.create() in-process bases the venv on sys._base_executable,
            # which inside OSGeo4W QGIS is the qgis-ltr-bin.exe launcher — the
            # venv's 'python.exe' becomes a copy of the QGIS binary and
            # 'python -m ensurepip' crashes (0xC0000005) or spawns a QGIS
            # window instead of installing pip.
            # Note: pip is installed by default in 3.12 venvs; the CLI flag is
            # --without-pip (to disable), not --with-pip.
            r = subprocess.run(
                [str(python), "-m", "venv", str(env)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"venv creation failed with exit code {r.returncode}.\n"
                    f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
                )

        sp = self.site_packages(env)
        if not sp:
            raise RuntimeError("site-packages not found in created environment")

        # Invoke pip via the venv's python (python -m pip), NOT the pip.exe /
        # pip console-script launcher. On Windows, running pip.exe to upgrade
        # itself refuses with "To modify pip, please run:
        # <venv>\Scripts\python.exe -m pip install --upgrade pip" — pip.exe
        # is a frozen console-script wrapper that cannot safely replace the
        # file it is running from. python -m pip always works.
        scripts = "Scripts" if platform.system() == "Windows" else "bin"
        py = env / scripts / ("python.exe" if platform.system() == "Windows" else "python")

        progress("Upgrading pip in isolated environment...")
        self._run_pip([str(py), "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade", progress)

        url = self.wheel_url()
        progress(f"Downloading {url}")
        self._run_pip([str(py), "-m", "pip", "install", url], f"wheel {url}", progress)

        self.add_env_to_path(env)
        import hydra_swe2d  # noqa: F401
        progress(f"Installed hydra_swe2d {self.version}")

    @staticmethod
    def _real_python() -> Path:
        """Locate a real python.exe to base the isolated venv on.

        Inside OSGeo4W QGIS on Windows, sys.executable and
        sys._base_executable point at the QGIS launcher (qgis-ltr-bin.exe),
        NOT a Python interpreter. venv.create() would copy that launcher
        into the venv, and 'python -m ensurepip' would crash (0xC0000005)
        or open a QGIS window. Resolve the actual Python instead.
        """
        candidates: list[str] = []
        # sys._base_executable, when it is actually a python binary
        bex = getattr(sys, "_base_executable", "") or ""
        if bex and os.path.basename(bex).lower().startswith("python"):
            candidates.append(bex)
        # python.exe beside the real prefix (OSGeo4W: C:\OSGeo4W\apps\Python312)
        for prefix in (
            getattr(sys, "base_prefix", "") or "",
            getattr(sys, "base_exec_prefix", "") or "",
        ):
            if prefix:
                candidates.append(os.path.join(prefix, "python.exe"))
        # python.exe beside the launcher
        if sys.executable:
            candidates.append(os.path.join(os.path.dirname(sys.executable), "python.exe"))
        for cand in candidates:
            if cand and os.path.isfile(cand):
                return Path(cand)
        # Last resort: the running executable (correct on normal installs).
        return Path(sys.executable)

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
        if proc.stdout is None:
            raise RuntimeError(f"{desc}: pip produced no stdout pipe")
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