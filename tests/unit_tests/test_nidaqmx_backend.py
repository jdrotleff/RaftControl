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


class FakeStream:
    # None: the device generates samples as soon as they are written.
    # Otherwise: the device's generated-sample count is held at this value.
    generated_override = None

    def __init__(self):
        self.regen_mode = None
        self.written = 0

    @property
    def total_samp_per_chan_generated(self):
        if FakeStream.generated_override is None:
            return self.written
        return FakeStream.generated_override


class FakeTask:
    instances = []

    def __init__(self):
        self.ao_channels = FakeChannels()
        self.do_channels = FakeChannels()
        self.timing = FakeTiming()
        self.out_stream = FakeStream()
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
        self.stream.written += values.shape[1]
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

    # Replacement reuses the one clocked streaming task; the only other task
    # is the on-demand zero write after stopping.
    assert [bool(t.timing.calls) for t in FakeTask.instances] == [True, False]
    assert FakeTask.instances[1].scalar_writes == [([0.0, 0.0, 0.0, 0.0], True)]
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


def _switch_offset(second, size=20, timeout=2.0):
    """Sample index where the replacement begins, found from the written output.

    The first write that equals the replacement evaluated at its own offset
    (and not the old action) marks the switch; this does not depend on when
    the started callback fires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        offset = 0
        for values in list(FakeWriter.writes):
            n = values.shape[1]
            if n >= size and np.allclose(values[:, :size], second.stream.samples(offset, size), atol=1e-12):
                return offset
            offset += n
        time.sleep(0.001)
    raise AssertionError("replacement was never written")


def _run_replacement(first, second, extra_writes=3):
    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    started = threading.Event()
    seen = {}

    def on_started():
        seen["switch"] = dict(backend.started_switch or {})
        started.set()

    backend.start(first, lambda: None, lambda: None, lambda error: None)
    _wait_for_writes(5)
    backend.replace(second, on_started, lambda: None, lambda error: None)
    assert started.wait(1)
    position = _switch_offset(second)
    _wait_for_writes(len(FakeWriter.writes) + extra_writes)
    backend.stop()
    return position, seen["switch"]


def test_replacement_continues_absolute_time():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=0.1)
    first = build_waveform(ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300), config)
    second = build_waveform(ActionRequest(20, 40, 12.34, 56.78, 0, 0, 300), config)

    position, _ = _run_replacement(first, second)

    assert position > 0
    # Everything from the switch on is the replacement at absolute time.
    offset, tail = 0, []
    for values in FakeWriter.writes[:-1]:  # last write is the ramp
        if offset >= position:
            tail.append(values)
        offset += values.shape[1]
    after = np.concatenate(tail, axis=1)
    np.testing.assert_allclose(after, second.stream.samples(position, after.shape[1]), atol=1e-12)


def test_identical_action_replacement_is_seamless():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=0.1)
    request = ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300)
    first, second = build_waveform(request, config), build_waveform(request, config)

    _, switch = _run_replacement(first, second)

    streamed = np.concatenate(FakeWriter.writes[:-1], axis=1)
    np.testing.assert_allclose(streamed, first.stream.samples(0, streamed.shape[1]), atol=1e-12)
    np.testing.assert_allclose(switch["phase_jump_rad"], 0, atol=1e-9)


def test_frequency_change_jump_follows_the_sim_formula():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=0.1)
    first = build_waveform(ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300), config)
    second = build_waveform(ActionRequest(30.038, 53.65, 12.34, 56.78, 0, 0, 300), config)

    position, switch = _run_replacement(first, second)

    # Same rule as the engine, cos(2 pi f t - phi) on global time: a changed
    # frequency jumps by 2 pi (f2 - f1) t at the switch, it is not carried over.
    t = position / config.sample_rate_hz
    expected = 2 * np.pi * np.array([12.34 - 45.99, 56.78 - 46.35]) * t
    expected = (expected + np.pi) % (2 * np.pi) - np.pi
    assert switch["sample_index"] == position
    np.testing.assert_allclose(switch["phase_jump_rad"], expected, atol=1e-9)
    assert np.abs(expected).max() > 0.1  # the test would be vacuous without a jump


def test_replacement_started_waits_until_buffered_samples_are_generated():
    from raft_control.config import ControllerConfig
    from raft_control.models import ActionRequest
    from raft_control.waveform import build_waveform

    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    FakeStream.generated_override = 0
    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=0.1)
    first = build_waveform(ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300), config)
    second = build_waveform(ActionRequest(20, 40, 12.34, 56.78, 0, 0, 0.05), config)
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    second_started = threading.Event()
    second_duration = threading.Event()
    try:
        backend.start(first, lambda: None, lambda: None, lambda error: None)
        _wait_for_writes(5)
        backend.replace(second, second_started.set, second_duration.set, lambda error: None)

        # Locate where the replacement's samples begin in the output stream.
        position = _switch_offset(second)

        # Queued but not yet generated: neither started nor timing keep time.
        FakeStream.generated_override = position - 1
        time.sleep(0.15)
        assert not second_started.is_set()
        assert not second_duration.is_set()

        # Once the device reaches the first replacement sample, it has started,
        # and the 50 ms keep time is measured from that moment.
        FakeStream.generated_override = position
        assert second_started.wait(1)
        assert second_duration.wait(1)
    finally:
        FakeStream.generated_override = None
        backend.stop()


def _stop_in_thread(backend):
    thread = threading.Thread(target=backend.stop)
    thread.start()
    return thread


def test_stop_waits_for_buffered_ramp_then_zeroes_outputs():
    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    started = threading.Event()
    done = []
    try:
        backend.start(waveform(1), started.set, lambda: None, done.append)
        assert started.wait(1)
        _wait_for_writes(3)
        stream_task = FakeTask.instances[0]
        # The device lags the writer: the ramp is still queued behind samples.
        FakeStream.generated_override = stream_task.out_stream.written
        stopping = _stop_in_thread(backend)
        time.sleep(0.15)
        assert stream_task.started, "task must not stop before the ramp is generated"

        FakeStream.generated_override = None
        stopping.join(2)
        assert not stopping.is_alive()
    finally:
        FakeStream.generated_override = None
        backend.stop()

    assert not stream_task.started and stream_task.closed
    np.testing.assert_allclose(FakeWriter.writes[-1][:, -1], 0)
    zero_task = FakeTask.instances[1]
    assert zero_task.scalar_writes == [([0.0, 0.0, 0.0, 0.0], True)]
    assert zero_task.closed
    assert done == ["stopped"]


def test_stop_is_bounded_when_device_never_catches_up():
    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005)
    started = threading.Event()
    try:
        backend.start(waveform(1), started.set, lambda: None, lambda error: None)
        assert started.wait(1)
        FakeStream.generated_override = 0
        _wait_for_writes(50)
        began = time.monotonic()
        backend.stop()
        elapsed = time.monotonic() - began
    finally:
        FakeStream.generated_override = None
        backend.stop()

    # At most one 500 ms buffer plus the ramp and a margin, never unbounded.
    assert elapsed < 1.0
    assert FakeTask.instances[0].closed
    assert FakeTask.instances[1].scalar_writes == [([0.0, 0.0, 0.0, 0.0], True)]


class RealtimeStream(FakeStream):
    """Device that generates samples in real time from task start."""

    rate = 1000

    def __init__(self):
        super().__init__()
        self.started_at = None

    @property
    def total_samp_per_chan_generated(self):
        if self.started_at is None:
            return 0
        return min(self.written, int((time.monotonic() - self.started_at) * self.rate))


class RealtimeTask(FakeTask):
    def __init__(self):
        super().__init__()
        self.out_stream = RealtimeStream()

    def start(self):
        super().start()
        self.out_stream.started_at = time.monotonic()


class RealtimeWriter(FakeWriter):
    """Blocks like DAQmx until the host buffer has room; can stall once."""

    stall_after_chunks = None
    stall_s = 0.0

    def __init__(self, stream, auto_start=False):
        super().__init__(stream, auto_start)
        self.calls = 0

    def write_many_sample(self, values, timeout=None):
        task = next(t for t in FakeTask.instances if t.out_stream is self.stream)
        capacity = task.timing.calls[0][1]["samps_per_chan"]
        while self.stream.written - self.stream.total_samp_per_chan_generated + values.shape[1] > capacity:
            time.sleep(0.001)
        self.calls += 1
        if self.calls == type(self).stall_after_chunks:
            time.sleep(type(self).stall_s)
        FakeWriter.writes.append(values.copy())
        self.stream.written += values.shape[1]


class DryRealtimeBackend(DryBackend):
    def _import(self):
        _, acquisition, regeneration, _ = super()._import()
        driver = type("Driver", (), {"Task": RealtimeTask})
        return driver, acquisition, regeneration, RealtimeWriter


def test_buffer_size_is_configurable():
    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    backend = DryBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005, buffer_s=0.1)
    started = threading.Event()
    backend.start(waveform(1), started.set, lambda: None, lambda error: None)
    assert started.wait(1)
    backend.stop()

    assert FakeTask.instances[0].timing.calls[0][1]["samps_per_chan"] == 100
    assert FakeWriter.writes[0].shape == (4, 100)
    assert backend.stream_stats["buffer_s"] == 0.1


def test_stream_stats_reveal_stall_depth():
    FakeTask.instances.clear()
    FakeWriter.writes.clear()
    RealtimeWriter.stall_after_chunks = 10
    RealtimeWriter.stall_s = 0.12
    backend = DryRealtimeBackend(["ao0", "ao1", "ao2", "ao3"], "Dev1", safe_ramp_s=0.005, buffer_s=0.2)
    started = threading.Event()
    try:
        backend.start(waveform(1), started.set, lambda: None, lambda error: None)
        assert started.wait(1)
        time.sleep(0.6)
    finally:
        RealtimeWriter.stall_after_chunks = None
        backend.stop()

    stats = backend.stream_stats
    assert stats["buffer_s"] == 0.2
    assert stats["chunks"] > 10
    assert stats["max_loop_gap_s"] >= 0.12
    # A 120 ms stall consumes ~120 ms of the 200 ms queue (plus up to one chunk).
    assert 0.03 <= stats["min_headroom_s"] <= 0.09
