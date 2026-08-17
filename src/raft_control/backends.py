from __future__ import annotations

import threading
import time
from collections.abc import Callable
import numpy as np


class Backend:
    name = "backend"

    def start(self, waveform, on_started: Callable[[], None], on_duration: Callable[[], None], on_done: Callable[[str | None], None]) -> None:
        raise NotImplementedError

    def replace(self, waveform, on_started, on_duration, on_done) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.stop()


class SimulationBackend(Backend):
    name = "simulation"

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._duration = 0.0
        self._waveform = None
        self._replacement = None
        self._replacement_event = threading.Event()

    def start(self, waveform, on_started, on_duration, on_done):
        self._stop.clear()
        self._waveform = waveform
        self._replacement = None
        self._duration = max(0.0, (waveform.clipped_currents_a.shape[1] - 1) / waveform.sample_rate_hz)

        def run():
            try:
                on_started()
                current = (waveform, on_started, on_duration, on_done)
                while not self._stop.wait(self._duration):
                    current[2]()
                    self._replacement_event.wait()
                    if self._stop.is_set():
                        break
                    replacement = self._replacement
                    self._replacement = None
                    self._replacement_event.clear()
                    if replacement is None:
                        continue
                    current = replacement
                    self._duration = max(0.0, (current[0].clipped_currents_a.shape[1] - 1) / current[0].sample_rate_hz)
                    current[1]()
                on_done("stopped")
            except Exception as exc:
                on_done(str(exc))

        self._thread = threading.Thread(target=run, name="raft-control-simulation", daemon=True)
        self._thread.start()

    def replace(self, waveform, on_started, on_duration, on_done):
        self._replacement = (waveform, on_started, on_duration, on_done)
        self._replacement_event.set()

    def stop(self):
        self._stop.set()


class NIDAQmxBackend(Backend):
    """Windows backend skeleton using one continuously owned DAQ task.

    Hardware verification must be performed on Windows with NI-DAQmx installed.
    The worker writes short rolling chunks so replacement does not require a
    stop/start cycle. The controller remains the source of waveform chunks.
    """

    name = "nidaqmx"

    def __init__(self, channels, device, output_limit=10.0):
        self.channels, self.device = channels, device
        self.output_limit = output_limit
        self._task = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _import(self):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType
            return nidaqmx, AcquisitionType
        except ImportError as exc:
            raise RuntimeError("install the RaftControl hardware extra on Windows") from exc

    def start(self, waveform, on_started, on_duration, on_done):
        nidaqmx, acquisition = self._import()

        def run():
            task = None
            error = None
            try:
                task = nidaqmx.Task()
                names = ",".join(f"{self.device}/{c}" for c in self.channels)
                task.ao_channels.add_ao_voltage_chan(names, min_val=-self.output_limit, max_val=self.output_limit)
                task.timing.cfg_samp_clk_timing(waveform.sample_rate_hz, sample_mode=acquisition.CONTINUOUS)
                task.write(waveform.clipped_currents_a, auto_start=False)
                with self._lock:
                    self._task = task
                self._stop.clear()
                task.start()
                on_started()
                if self._stop.wait(max(0.0, waveform.time_s[-1])):
                    return
                on_duration()
                while not self._stop.wait(0.05):
                    pass
            except Exception as exc:
                error = str(exc)
            finally:
                with self._lock:
                    self._task = None
                if task is not None:
                    try:
                        task.stop()
                    finally:
                        task.close()
                on_done(error or "stopped" if self._stop.is_set() else error)

        threading.Thread(target=run, name="raft-control-nidaqmx", daemon=True).start()

    def replace(self, waveform, on_started, on_duration, on_done):
        # The first production implementation will replace the rolling buffer
        # from the single DAQ worker. This method is intentionally explicit so
        # tests and the controller cannot silently stop/start the task.
        with self._lock:
            task = self._task
        if task is None:
            raise RuntimeError("no active NI-DAQmx task")
        task.write(waveform.clipped_currents_a, auto_start=False)
        on_started()

    def stop(self):
        self._stop.set()
        with self._lock:
            task = self._task
        if task is not None:
            try:
                task.stop()
            except Exception:
                pass
