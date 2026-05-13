"""swe2d_results_animation.py

Animation controller for SWE2D results playback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
try:
    from qgis.PyQt import QtCore
    from qgis.PyQt.QtCore import pyqtSignal
except Exception:
    from PyQt5 import QtCore
    from PyQt5.QtCore import pyqtSignal


class ResultsAnimationController(QtCore.QObject):
    """Drive timestep playback with play/pause/step controls."""

    current_timestep_changed = pyqtSignal(float, int)
    play_state_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QtCore.QObject] = None, fps: float = 4.0):
        super().__init__(parent)
        self._timesteps = np.empty(0, dtype=np.float64)
        self._index = 0
        self._playing = False
        self._fps = max(0.25, float(fps))
        self._adaptive_interval_ms = 0
        self._last_tick_clock = QtCore.QElapsedTimer()
        self._tick_ewma_ms = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    @property
    def current_index(self) -> int:
        return int(self._index)

    @property
    def is_playing(self) -> bool:
        return bool(self._playing)

    def set_timesteps(self, timesteps: np.ndarray) -> None:
        arr = np.asarray(timesteps, dtype=np.float64).ravel()
        self._timesteps = arr.copy() if arr.size else np.empty(0, dtype=np.float64)
        if self._timesteps.size <= 0:
            self.pause()
            self._index = 0
            self.current_timestep_changed.emit(0.0, 0)
            return
        self._index = max(0, min(self._index, self._timesteps.size - 1))
        self.current_timestep_changed.emit(float(self._timesteps[self._index]), int(self._index))

    def set_index(self, index: int) -> None:
        if self._timesteps.size <= 0:
            self._index = 0
            self.current_timestep_changed.emit(0.0, 0)
            return
        idx = max(0, min(int(index), self._timesteps.size - 1))
        if idx == self._index:
            return
        self._index = idx
        self.current_timestep_changed.emit(float(self._timesteps[self._index]), int(self._index))

    def set_frame_rate(self, fps: float) -> None:
        self._fps = max(0.25, float(fps))
        self._adaptive_interval_ms = 0
        if self._playing:
            self._timer.setInterval(self._target_interval_ms())

    def play(self) -> None:
        if self._timesteps.size <= 0:
            return
        if self._playing:
            return
        self._playing = True
        self._adaptive_interval_ms = 0
        self._tick_ewma_ms = 0.0
        self._last_tick_clock.restart()
        self._timer.start(self._target_interval_ms())
        self.play_state_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self._adaptive_interval_ms = 0
        self._tick_ewma_ms = 0.0
        self.play_state_changed.emit(False)

    def step_forward(self) -> None:
        if self._timesteps.size <= 0:
            return
        idx = (self._index + 1) % self._timesteps.size
        self.set_index(idx)

    def step_backward(self) -> None:
        if self._timesteps.size <= 0:
            return
        idx = (self._index - 1) % self._timesteps.size
        self.set_index(idx)

    def _target_interval_ms(self) -> int:
        return max(33, int(1000.0 / self._fps))

    def _update_adaptive_interval(self, dt_ms: float) -> int:
        base_ms = self._target_interval_ms()
        dt_ms = max(0.0, float(dt_ms))
        alpha = 0.25
        if self._tick_ewma_ms <= 0.0:
            self._tick_ewma_ms = dt_ms
        else:
            self._tick_ewma_ms = ((1.0 - alpha) * self._tick_ewma_ms) + (alpha * dt_ms)

        guard_ms = int(max(base_ms, round(self._tick_ewma_ms * 1.2)))
        guard_ms = max(base_ms, min(1000, guard_ms))
        self._adaptive_interval_ms = guard_ms
        return guard_ms

    def _on_tick(self) -> None:
        dt_ms = float(self._last_tick_clock.restart()) if self._last_tick_clock.isValid() else 0.0
        self.step_forward()
        if self._playing:
            self._timer.setInterval(self._update_adaptive_interval(dt_ms))
