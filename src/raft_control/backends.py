from __future__ import annotations

import threading
import time
from collections.abc import Callable
import numpy as np


def _keep_duration(waveform) -> float:
    duration = getattr(waveform, "duration_s", None)
    if duration is not None:
        return max(0.0, float(duration))
    return max(0.0, (waveform.clipped_currents_a.shape[1] - 1) / waveform.sample_rate_hz)


class Backend:
    name = "backend"

    def initialize(self) -> None:
        pass

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

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
        self.last_output = np.zeros(4)
        self.ramped_to_zero = False

    def start(self, waveform, on_started, on_duration, on_done):
        self._stop.clear()
        self._waveform = waveform
        self._replacement = None
        self._duration = _keep_duration(waveform)

        def run():
            try:
                on_started()
                current = (waveform, on_started, on_duration, on_done)
                started_at = time.monotonic()
                duration_sent = False
                while not self._stop.wait(0.005):
                    self.last_output = current[0].clipped_currents_a[:, -1].copy()
                    if not duration_sent and time.monotonic() - started_at >= self._duration:
                        current[2]()
                        duration_sent = True
                    if self._replacement_event.is_set():
                        replacement = self._replacement
                        self._replacement = None
                        self._replacement_event.clear()
                        if replacement is not None:
                            current = replacement
                            self._duration = _keep_duration(current[0])
                            started_at = time.monotonic()
                            duration_sent = False
                            current[1]()
                self.last_output[:] = 0.0
                self.ramped_to_zero = True
                current[3]("stopped")
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
    """Continuously streamed four-channel NI analog-output backend."""

    name = "nidaqmx"

    def __init__(
        self,
        channels,
        device,
        output_limit=10.0,
        safe_ramp_s=0.05,
        amplifier_enable_line=None,
    ):
        self.channels, self.device = channels, device
        self.output_limit = output_limit
        self.amplifier_enable_line = amplifier_enable_line
        self._enable_task = None
        self._task = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._replacement = None
        self._replacement_event = threading.Event()
        self._thread = None
        self._safe_ramp_s = safe_ramp_s

    def _import(self):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, RegenerationMode
            from nidaqmx.stream_writers import AnalogMultiChannelWriter
            return nidaqmx, AcquisitionType, RegenerationMode, AnalogMultiChannelWriter
        except ImportError as exc:
            raise RuntimeError("install the RaftControl hardware extra on Windows") from exc

    def initialize(self):
        """Own the amplifier-enable line and put it in LabVIEW's safe-low state."""
        if self.amplifier_enable_line is None:
            return
        with self._lock:
            if self._enable_task is not None:
                return
            nidaqmx, _, _, _ = self._import()
            task = nidaqmx.Task()
            try:
                task.do_channels.add_do_chan(
                    f"{self.device}/{self.amplifier_enable_line}"
                )
                task.write(False, auto_start=True)
            except Exception:
                task.close()
                raise
            self._enable_task = task

    def enable(self):
        self.initialize()
        with self._lock:
            # NI analog outputs can retain their last voltage after the
            # previous task or process exits.  Establish a known-safe output
            # while the amplifier is still disabled before exposing the
            # coils to those channels.
            nidaqmx, _, _, _ = self._import()
            zero_task = nidaqmx.Task()
            try:
                names = ",".join(f"{self.device}/{c}" for c in self.channels)
                zero_task.ao_channels.add_ao_voltage_chan(
                    names, min_val=-self.output_limit, max_val=self.output_limit
                )
                zero_task.write([0.0] * len(self.channels), auto_start=True)
                time.sleep(self._safe_ramp_s)
            finally:
                zero_task.close()
            if self._enable_task is not None:
                self._enable_task.write(True, auto_start=True)

    def disable(self):
        with self._lock:
            if self._enable_task is not None:
                self._enable_task.write(False, auto_start=True)

    def start(self, waveform, on_started, on_duration, on_done):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("NI-DAQmx task is already running")
        nidaqmx, acquisition, regeneration, writer_type = self._import()
        self._stop.clear()
        self._replacement_event.clear()
        self._replacement = None

        def run():
            task = None
            error = None
            try:
                task = nidaqmx.Task()
                names = ",".join(f"{self.device}/{c}" for c in self.channels)
                task.ao_channels.add_ao_voltage_chan(names, min_val=-self.output_limit, max_val=self.output_limit)
                # A 500 ms device buffer tolerates normal Windows scheduling
                # stalls.  Writes stay at 20 ms so replacement checks remain
                # frequent without relying on a fragile 40 ms total buffer.
                chunk_samples = max(2, int(round(waveform.sample_rate_hz * 0.02)))
                buffer_samples = chunk_samples * 25
                task.timing.cfg_samp_clk_timing(
                    waveform.sample_rate_hz,
                    sample_mode=acquisition.CONTINUOUS,
                    samps_per_chan=buffer_samples,
                )
                # Stream only into space the device has finished consuming.
                # Rewriting a regenerating buffer can make the device alternate
                # between old and new samples and raises DAQmx warning 200015.
                task.out_stream.regen_mode = regeneration.DONT_ALLOW_REGENERATION
                writer = writer_type(task.out_stream, auto_start=False)
                with self._lock:
                    self._task = task
                current = (waveform, on_started, on_duration, on_done)
                sample_index = 0
                last_sample = np.zeros(len(self.channels), dtype=float)

                def next_chunk(source, index):
                    # Evaluate on the absolute sample index so the output stays
                    # phase-continuous for any frequency and any keep time.
                    stream = getattr(source, "stream", None)
                    if stream is not None:
                        return np.ascontiguousarray(stream.samples(index, chunk_samples)), index + chunk_samples
                    values = source.clipped_currents_a
                    indices = (np.arange(chunk_samples) + index) % values.shape[1]
                    return np.ascontiguousarray(values[:, indices]), (index + chunk_samples) % values.shape[1]

                # Fill the complete output buffer before starting.
                initial_parts = []
                for _ in range(buffer_samples // chunk_samples):
                    part, sample_index = next_chunk(waveform, sample_index)
                    initial_parts.append(part)
                initial = np.ascontiguousarray(np.concatenate(initial_parts, axis=1))
                writer.write_many_sample(initial, timeout=2.0)
                written = initial.shape[1]
                last_sample = initial[:, -1].copy()
                task.start()
                on_started()
                started_at = time.monotonic()
                duration_sent = False
                # Replacements queue behind up to one buffer (~500 ms) of the
                # previous action.  Each waits here with the absolute output
                # position of its first sample until the device generates it.
                pending = []
                while not self._stop.is_set():
                    if self._replacement_event.is_set():
                        with self._lock:
                            replacement = self._replacement
                            self._replacement = None
                            self._replacement_event.clear()
                        if replacement is not None:
                            current = replacement
                            # A replacement starts its own waveform at phase 0.
                            sample_index = 0
                            started_at = None
                            duration_sent = False
                            pending.append((written, replacement))
                    if pending:
                        generated = task.out_stream.total_samp_per_chan_generated
                        while pending and generated >= pending[0][0]:
                            position, action = pending.pop(0)
                            if action is current:
                                # Back-date to when the first sample was generated.
                                started_at = time.monotonic() - (generated - position) / action[0].sample_rate_hz
                            action[1]()
                    if started_at is not None and not duration_sent and time.monotonic() - started_at >= _keep_duration(current[0]):
                        current[2]()
                        duration_sent = True
                    chunk, sample_index = next_chunk(current[0], sample_index)
                    writer.write_many_sample(chunk, timeout=2.0)
                    written += chunk.shape[1]
                    last_sample = chunk[:, -1].copy()

                ramp_samples = max(2, int(round(current[0].sample_rate_hz * self._safe_ramp_s)))
                ramp = np.linspace(last_sample, np.zeros_like(last_sample), ramp_samples, axis=0).T
                writer.write_many_sample(np.ascontiguousarray(ramp), timeout=2.0)
                written += ramp_samples
                # The ramp is queued behind up to one buffer of waveform.
                # Stopping the task discards unplayed samples, so wait until
                # the device has generated the final zero.  The deadline is
                # bounded by what the device buffer can hold.
                rate = current[0].sample_rate_hz
                deadline = time.monotonic() + (buffer_samples + ramp_samples) / rate + 0.25
                while time.monotonic() < deadline:
                    remaining = written - task.out_stream.total_samp_per_chan_generated
                    if remaining <= 0:
                        break
                    time.sleep(min(0.005, remaining / rate))
            except Exception as exc:
                error = str(exc)
            finally:
                with self._lock:
                    self._task = None
                if task is not None:
                    try:
                        task.stop()
                    except Exception as exc:
                        if error is None and not self._stop.is_set():
                            error = str(exc)
                    try:
                        task.close()
                    except Exception as exc:
                        if error is None:
                            error = str(exc)
                # NI analog outputs hold their last value after a buffered task
                # stops or fails.  Always make a best-effort on-demand zero
                # write after releasing the task.
                zero_task = None
                try:
                    zero_task = nidaqmx.Task()
                    names = ",".join(f"{self.device}/{c}" for c in self.channels)
                    zero_task.ao_channels.add_ao_voltage_chan(
                        names, min_val=-self.output_limit, max_val=self.output_limit
                    )
                    zero_task.write([0.0] * len(self.channels), auto_start=True)
                except Exception:
                    pass
                finally:
                    if zero_task is not None:
                        try:
                            zero_task.close()
                        except Exception:
                            pass
                current_done = current[3] if "current" in locals() else on_done
                try:
                    current_done(error or "stopped" if self._stop.is_set() else error)
                except Exception:
                    # A callback must not prevent the DAQ worker from finishing
                    # and becoming restartable.
                    pass

        self._thread = threading.Thread(target=run, name="raft-control-nidaqmx", daemon=True)
        self._thread.start()

    def replace(self, waveform, on_started, on_duration, on_done):
        with self._lock:
            if self._task is None:
                raise RuntimeError("no active NI-DAQmx task")
            self._replacement = (waveform, on_started, on_duration, on_done)
            self._replacement_event.set()
        if self._task is None:
            raise RuntimeError("no active NI-DAQmx task")

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self._safe_ramp_s + 1.0))

    def close(self):
        self.stop()
        with self._lock:
            task = self._enable_task
            if task is None:
                return
            try:
                task.write(False, auto_start=True)
            finally:
                task.close()
                self._enable_task = None
