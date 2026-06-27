"""Temporal dock — animation bar for time-stepping through results.

Owns: step-back, play/pause, step-forward, slider, time label, speed combo.
Wires directly to ResultsAnimationController signals via SWE2DResultsData.
"""
from __future__ import annotations

from typing import Any, Optional

from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import Qt

_TIME_UNIT = "hr"
_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


class TemporalDockWidget(QtWidgets.QWidget):
    """Animation bar for stepping through simulation time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Any = None

        self._play_btn: QtWidgets.QPushButton = None
        self._time_slider: QtWidgets.QSlider = None
        self._time_lbl: QtWidgets.QLabel = None
        self._speed_combo: QtWidgets.QComboBox = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the temporal control bar layout: step buttons, slider, time label, speed combo."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._step_back_btn = QtWidgets.QPushButton("\u25c4")
        self._step_back_btn.setFixedSize(24, 22)
        self._step_back_btn.setToolTip("Step back one frame")
        self._step_back_btn.clicked.connect(self._on_step_back)

        self._play_btn = QtWidgets.QPushButton("\u25b6")
        self._play_btn.setFixedSize(24, 22)
        self._play_btn.setCheckable(True)
        self._play_btn.setToolTip("Play / Pause animation")
        self._play_btn.clicked.connect(self._on_play_pause)

        self._step_fwd_btn = QtWidgets.QPushButton("\u25b6")
        self._step_fwd_btn.setFixedSize(28, 22)
        self._step_fwd_btn.setToolTip("Step forward one frame")
        self._step_fwd_btn.clicked.connect(self._on_step_fwd)

        self._time_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setRange(0, 0)
        self._time_slider.setValue(0)
        self._time_slider.setTracking(True)
        self._time_slider.setToolTip("Drag to navigate through simulation time steps.")
        self._time_slider.valueChanged.connect(self._on_slider_changed)

        self._time_lbl = QtWidgets.QLabel(f"T = 0.000 {_TIME_UNIT}")
        self._time_lbl.setFixedWidth(100)
        self._time_lbl.setStyleSheet("font-size: 9px;")

        self._speed_combo = QtWidgets.QComboBox()
        for spd in _SPEEDS:
            self._speed_combo.addItem(f"{spd}\u00d7", spd)
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.setFixedWidth(56)
        self._speed_combo.setToolTip("Playback speed")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        layout.addWidget(self._step_back_btn)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._step_fwd_btn)
        layout.addWidget(self._time_slider, 1)
        layout.addWidget(self._time_lbl)
        layout.addWidget(self._speed_combo)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_data(self, data: Any) -> None:
        """Bind animation data and set the slider range from frame count."""
        self._data = data
        if data is not None and hasattr(data, "frame_count"):
            self._time_slider.setRange(0, max(0, data.frame_count - 1))

    def on_timestep_changed(self, t_sec: float, frame_idx: int) -> None:
        """Update slider position and time label when the current timestep changes."""
        if self._data is None:
            return
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(int(frame_idx))
        self._time_slider.blockSignals(False)
        self._time_lbl.setText(f"T = {t_sec / 3600.0:.3f} {_TIME_UNIT}")

    def on_play_state_changed(self, playing: bool) -> None:
        """Update play button icon and checked state when playback toggles."""
        self._play_btn.blockSignals(True)
        self._play_btn.setChecked(bool(playing))
        self._play_btn.setText("\u23f8" if playing else "\u25b6")
        self._play_btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_play_pause(self, _checked: bool) -> None:
        """Toggle play/pause on the animation data object."""
        if self._data is None:
            return
        if self._data.is_playing:
            self._data.pause()
        else:
            self._data.play()

    def _on_step_back(self) -> None:
        """Step the animation back one frame."""
        if self._data is None:
            return
        self._data.step_backward()

    def _on_step_fwd(self) -> None:
        """Step the animation forward one frame."""
        if self._data is None:
            return
        self._data.step_forward()

    def _on_slider_changed(self, value: int) -> None:
        """Seek the animation to the frame index from the slider."""
        if self._data is None:
            return
        self._data.set_index(int(value))

    def _on_speed_changed(self, index: int) -> None:
        """Update the animation frame rate when the speed combo changes."""
        if self._data is None:
            return
        speed = _SPEEDS[max(0, min(index, len(_SPEEDS) - 1))]
        self._data.set_frame_rate(4.0 * speed)
