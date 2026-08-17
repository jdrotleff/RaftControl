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
        regeneration = type("Regeneration", (), {"DONT_ALLOW_REGENERATION": "disabled"})
        return driver, acquisition, regeneration, FakeWriter


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
    assert any(np.all(values == 2) for values in FakeWriter.writes)
    np.testing.assert_allclose(FakeWriter.writes[-1][:, -1], 0)
