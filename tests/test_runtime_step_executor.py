"""Regression tests for SWE2DRuntimeStepExecutor source paths."""

import numpy as np
import pytest

from swe2d.runtime.runtime_step_executor import SWE2DRuntimeStepExecutor


class _FakeBackend:
    def __init__(self, n_cells=4, areas=None):
        self.n_cells = n_cells
        self.areas = np.full(n_cells, 10.0, dtype=np.float64) if areas is None else np.asarray(areas, dtype=np.float64)
        self.accumulated = []
        self.step_dt = 1.0

    def cell_areas(self):
        return self.areas

    def accumulate_external_sources_native(self, src):
        self.accumulated.append(np.asarray(src, dtype=np.float64).copy())

    def step(self, dt_request):
        return {"dt": self.step_dt}


class _FakeCouplingController:
    def __init__(self, native=True):
        self.native = native

    def apply_native_device_sources(self, t, dt):
        return self.native


def test_native_device_path_accumulates_internal_flow():
    """Internal-flow source must be applied on the native coupling path."""
    executor = SWE2DRuntimeStepExecutor()
    n = 4
    backend = _FakeBackend(n_cells=n)
    internal_flow = np.array([0.0, 5.0, 0.0, 20.0], dtype=np.float64)
    volume_calls = []
    external_calls = []

    def rain_cb(t0, t1, accumulate, mutate_state):
        return np.zeros(n, dtype=np.float64)

    def cell_cb(t):
        return internal_flow

    def volume_cb(dt, rain, cell, coupled):
        volume_calls.append((dt, rain, cell, coupled))

    def external_cb(backend, dt, rain, cell, coupled):
        external_calls.append((dt, rain, cell, coupled))

    result = executor.execute_step(
        backend=backend,
        t_accum=0.0,
        last_diag=None,
        dt_cfg=1.0,
        dt_request=1.0,
        coupling_controller=_FakeCouplingController(native=True),
        rain_source_for_window_callback=rain_cb,
        cell_source_model_at_time_callback=cell_cb,
        accumulate_source_volume_model_callback=volume_cb,
        apply_external_sources_callback=external_cb,
        native_source_injection_mode=True,
    )

    assert result["dt_used"] == 1.0
    assert len(external_calls) == 0, "external source callback should not be used in native path"
    assert len(backend.accumulated) == 1
    np.testing.assert_allclose(backend.accumulated[0], internal_flow / backend.areas)
    assert len(volume_calls) == 1
    assert volume_calls[0][2] is internal_flow


def test_native_device_path_accumulates_rain_and_internal_flow():
    """Both rain and internal-flow sources are accumulated natively."""
    executor = SWE2DRuntimeStepExecutor()
    n = 4
    backend = _FakeBackend(n_cells=n)
    internal_flow = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    rain = np.array([0.0, 0.1, 0.0, 0.2], dtype=np.float64)

    def rain_cb(t0, t1, accumulate, mutate_state):
        return rain

    def cell_cb(t):
        return internal_flow

    def volume_cb(dt, r, c, coupled):
        pass

    def external_cb(backend, dt, r, c, coupled):
        pytest.fail("external callback should not be called in native path")

    executor.execute_step(
        backend=backend,
        t_accum=0.0,
        last_diag=None,
        dt_cfg=1.0,
        dt_request=1.0,
        coupling_controller=_FakeCouplingController(native=True),
        rain_source_for_window_callback=rain_cb,
        cell_source_model_at_time_callback=cell_cb,
        accumulate_source_volume_model_callback=volume_cb,
        apply_external_sources_callback=external_cb,
        native_source_injection_mode=True,
    )

    assert len(backend.accumulated) == 2
    np.testing.assert_allclose(backend.accumulated[0], internal_flow / backend.areas)
    np.testing.assert_allclose(backend.accumulated[1], rain)


def test_non_native_path_uses_external_source_callback():
    """Standard path should call the external source callback, not native accumulation."""
    executor = SWE2DRuntimeStepExecutor()
    n = 4
    backend = _FakeBackend(n_cells=n)
    internal_flow = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    rain = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    external_calls = []

    def rain_cb(t0, t1, accumulate, mutate_state):
        return rain

    def cell_cb(t):
        return internal_flow

    def volume_cb(dt, r, c, coupled):
        pass

    def external_cb(bk, dt, r, c, coupled):
        external_calls.append((dt, r, c, coupled))

    executor.execute_step(
        backend=backend,
        t_accum=0.0,
        last_diag=None,
        dt_cfg=1.0,
        dt_request=1.0,
        coupling_controller=_FakeCouplingController(native=False),
        rain_source_for_window_callback=rain_cb,
        cell_source_model_at_time_callback=cell_cb,
        accumulate_source_volume_model_callback=volume_cb,
        apply_external_sources_callback=external_cb,
        native_source_injection_mode=False,
    )

    assert len(external_calls) == 1
    assert external_calls[0][2] is internal_flow
    assert len(backend.accumulated) == 0
