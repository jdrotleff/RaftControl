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


class FakeTiming:
    def __init__(self):
        self.calls = []

    def cfg_samp_clk_timing(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeTask:
    instances = []

    def __init__(self):
        self.ao_channels = FakeChannels()
        self.timing = FakeTiming()
        self.out_stream = type("Stream", (), {"regen_mode": None})()
        self.started = False
        self.closed = False
        FakeTask.instances.append(self)

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
    assert FakeTask.instances[0].out_stream.regen_mode == "allowed"
    assert any(np.all(values == 2) for values in FakeWriter.writes)
    np.testing.assert_allclose(FakeWriter.writes[-1][:, -1], 0)


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
