import threading
import time

import numpy as np

from raft_control.backends import NIDAQmxBackend
from raft_control.models import WaveformResult


class FakeChannels:
    def __init__(self):
        self.calls = []

    def add_ao_voltage_chan(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def add_do_chan(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeTiming:
    def __init__(self):
        self.calls = []

    def cfg_samp_clk_timing(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeTask:
    instances = []

    def __init__(self):
        self.ao_channels = FakeChannels()
        self.do_channels = FakeChannels()
        self.timing = FakeTiming()
        self.out_stream = type("Stream", (), {"regen_mode": None})()
        self.started = False
        self.closed = False
        self.scalar_writes = []
        FakeTask.instances.append(self)

    def write(self, value, auto_start=False):
        self.scalar_writes.append((value, auto_start))

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeWriter:
    writes = []

    def __init__(self, stream, auto_start=False):
        self.stream = stream

    def write_many_sample(self, values, timeout=None):
        FakeWriter.writes.append(values.copy())
        time.sleep(0.001)


class DryBackend(NIDAQmxBackend):
    def _import(self):
        driver = type("Driver", (), {"Task": FakeTask})
        acquisition = type("Acquisition", (), {"CONTINUOUS": "continuous"})
        regeneration = type(
            "Regeneration",
            (),
            {"ALLOW_REGENERATION": "allowed", "DONT_ALLOW_REGENERATION": "disabled"},
        )
        return driver, acquisition, regeneration, FakeWriter


class FlakyWriter(FakeWriter):
    fail_next = True

    def write_many_sample(self, values, timeout=None):
        if type(self).fail_next:
            type(self).fail_next = False
            raise RuntimeError("simulated DAQ write failure")
        super().write_many_sample(values, timeout)


class DryFlakyBackend(DryBackend):
    def _import(self):
        driver, acquisition, regeneration, _ = super()._import()
        return driver, acquisition, regeneration, FlakyWriter


def waveform(value):
    samples = np.full((4, 20), value, dtype=float)
    return WaveformResult(
        np.arange(20) / 1000,
        samples,
        samples,
        samples,
        samples,
        np.zeros_like(samples, dtype=bool),
        1000,
        "test",
    )


def test_one_task_streams_replacement_and_ramps_to_zero():
    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    first_started = threading.Event()
    second_started = threading.Event()

    backend.start(waveform(1), first_started.set, lambda: None, lambda error: None)
    assert first_started.wait(1)
    backend.replace(waveform(2), second_started.set, lambda: None, lambda error: None)
    assert second_started.wait(1)
    backend.stop()

    assert len(FakeTask.instances) == 1
    assert FakeTask.instances[0].closed
    assert FakeTask.instances[0].out_stream.regen_mode == "disabled"
    # 25 x 20 ms chunks provide 500 ms scheduling headroom.
    assert FakeWriter.writes[0].shape == (4, 500)
    assert any(np.all(values == 2) for values in FakeWriter.writes)
    np.testing.assert_allclose(FakeWriter.writes[-1][:, -1], 0)


def test_300_second_keep_time_uses_bounded_stream_and_is_not_zeroed():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    config = ControllerConfig(sample_rate_hz=10_000, stream_window_s=10)
    request = ActionRequest(30, 53, 45, 46, 0, 0, 300)
    long_waveform = build_waveform(request, config)

    assert long_waveform.duration_s == 300
    assert long_waveform.clipped_currents_a.shape == (4, 100_000)
    assert not np.all(long_waveform.clipped_currents_a == 0)
    # The lifecycle timer must use the keep time, not the bounded stream window.
    from raft_control.backends import _keep_duration
    assert _keep_duration(long_waveform) == 300


def test_amplifier_enable_line_is_safe_low_then_tracks_lifecycle():
    FakeTask.instances.clear()
    backend = DryBackend(
        ["ao0", "ao1", "ao2", "ao3"],
        "Dev1",
        amplifier_enable_line="port0/line0",
    )

    backend.initialize()
    enable_task = FakeTask.instances[0]
    assert enable_task.do_channels.calls == [(("Dev1/port0/line0",), {})]
    assert enable_task.scalar_writes == [(False, True)]

    backend.enable()
    backend.disable()
    backend.close()

    zero_task = FakeTask.instances[1]
    assert zero_task.ao_channels.calls == [
        (("Dev1/ao0,Dev1/ao1,Dev1/ao2,Dev1/ao3",), {"min_val": -10.0, "max_val": 10.0})
    ]
    assert zero_task.scalar_writes == [([0.0, 0.0, 0.0, 0.0], True)]
    assert zero_task.closed
    assert enable_task.scalar_writes == [
        (False, True),
        (True, True),
        (False, True),
        (False, True),
    ]
    assert enable_task.closed


def test_amplifier_stays_disabled_when_zeroing_fails():
    FakeTask.instances.clear()
    backend = DryBackend(
        ["ao0", "ao1", "ao2", "ao3"],
        "Dev1",
        amplifier_enable_line="port0/line0",
    )
    backend.initialize()
    enable_task = FakeTask.instances[0]

    original_write = FakeTask.write

    def fail_analog_write(task, value, auto_start=False):
        if task is not enable_task:
            raise RuntimeError("simulated zero write failure")
        original_write(task, value, auto_start)

    FakeTask.write = fail_analog_write
    try:
        try:
            backend.enable()
        except RuntimeError as exc:
            assert "simulated zero write failure" in str(exc)
        else:
            raise AssertionError("enable should fail when outputs cannot be zeroed")
    finally:
        FakeTask.write = original_write
        backend.close()

    assert enable_task.scalar_writes == [(False, True), (False, True)]
    assert FakeTask.instances[1].closed


def test_failed_worker_reports_error_and_can_start_fresh_task():
    FakeTask.instances.clear()
    FlakyWriter.fail_next = True
    backend = DryFlakyBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    failed = threading.Event()
    errors = []

    backend.start(waveform(1), lambda: None, lambda: None, lambda error: (errors.append(error), failed.set()))
    assert failed.wait(1)
    assert "simulated DAQ write failure" in errors[0]

    restarted = threading.Event()
    backend.start(waveform(1), restarted.set, lambda: None, lambda error: None)
    assert restarted.wait(1)
    backend.stop()
    assert len(FakeTask.instances) >= 2


def _wait_for_writes(count, timeout=2.0):
    deadline = time.monotonic() + timeout
    while len(FakeWriter.writes) < count and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(FakeWriter.writes) >= count


def test_streamed_chunks_are_phase_continuous_across_window_boundaries():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    # A 0.1 s preview window is far shorter than the 500 ms DAQ buffer, so a
    # repeating window would wrap several times within the written samples.
    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=0.1)
    long_waveform = build_waveform(ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300), config)
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)

    backend.start(long_waveform, lambda: None, lambda: None, lambda error: None)
    _wait_for_writes(20)
    backend.stop()

    streamed = np.concatenate(FakeWriter.writes[:-1], axis=1)  # last write is the ramp
    assert streamed.shape[1] > 5 * long_waveform.clipped_currents_a.shape[1]
    np.testing.assert_allclose(streamed, long_waveform.stream.samples(0, streamed.shape[1]), atol=1e-12)


def test_replacement_restarts_phase_at_zero():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=0.1)
    first = build_waveform(ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300), config)
    second = build_waveform(ActionRequest(20, 40, 12.34, 56.78, 0, 0, 300), config)
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    replaced_at = []
    second_started = threading.Event()

    backend.start(first, lambda: None, lambda: None, lambda error: None)
    _wait_for_writes(5)
    backend.replace(
        second,
        lambda: (replaced_at.append(len(FakeWriter.writes)), second_started.set()),
        lambda: None,
        lambda error: None,
    )
    assert second_started.wait(1)
    _wait_for_writes(replaced_at[0] + 3)
    backend.stop()

    chunk = FakeWriter.writes[replaced_at[0]].shape[1]
    after = np.concatenate(FakeWriter.writes[replaced_at[0]:replaced_at[0] + 3], axis=1)
    np.testing.assert_allclose(after, second.stream.samples(0, 3 * chunk), atol=1e-12)
