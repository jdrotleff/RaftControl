# Output streaming, latency and buffer sizing

How RaftControl streams coil currents to the NI USB-6363, why the current
settings were chosen, and how to re-measure them.

## How output is generated

- **Continuous phase.** Hardware output is computed chunk by chunk from an
  absolute sample index, `t = n / sample_rate_hz`
  (`WaveformStream.samples()` in `src/raft_control/waveform.py`). Arbitrary
  fractional frequencies such as 45.99 Hz and 46.35 Hz stay phase-continuous
  for the whole keep time. Earlier versions repeated a fixed 10 s window, which
  jumped by 324° (X) and 180° (Y) at every window boundary for that example.
- **Bounded memory.** Only a preview of at most `stream_window_s` is stored as
  arrays (plots, min/max currents). Streamed chunks are computed on demand, so a
  300 s action costs no more memory than a 3 s one. Action records keep only the
  per-channel min/max currents, not sample arrays.
- **Replacement.** A new action replaces the running one inside the same DAQ
  task and starts its own waveform at phase 0.
- **Honest `started` time.** A replacement's samples queue behind whatever is
  already buffered. `started` fires, and the keep time starts, only once the
  device reports (`total_samp_per_chan_generated`) that the replacement's first
  sample has been output.
- **Safe stop.** On stop, a `safe_ramp_s` ramp to zero is queued. The worker
  waits until the device has generated it (bounded by the buffer size), then
  stops the task and always writes zero to the outputs. NI analog outputs
  otherwise hold their last value after a task stops.

## Where the latency comes from

Time from `send` until the new field reaches the coils is roughly:

```
build_waveform  +  host buffer (stream_buffer_s)  +  device FIFO  +  one 20 ms chunk
```

The USB-6363 AO FIFO holds 8,191 samples shared across channels, about 2,047
per channel with 4 channels:

| sample rate | FIFO latency |
|---|---|
| 10 kHz | ~205 ms |
| 40 kHz | ~51 ms |

The FIFO is always kept full by the driver. The `AO.DataXferReqCond` setting
that would keep it emptier is not supported on this device over USB; the
driver coerces it (warning 209803). Raising the sample rate is therefore the
only way to shorten the FIFO's share of the latency.

## Rig measurements (2026-10-21, USB-6363, amplifier disabled)

Measured with `scripts/buffer_probe.py`, replacing the action every 3 s like an
RL agent. *Latency* is replace → first sample generated. *Min headroom* is the
least output that was still queued at any point; 0 means underflow.

| sample rate | buffer | priority | CPU load | latency (median/max) | min headroom | result |
|---|---|---|---|---|---|---|
| 10 kHz | 500 ms | normal | idle | 743 / 813 ms | 431 ms | ok |
| 10 kHz | 200 ms | normal | idle | 406 / 516 ms | 198 ms | ok |
| 10 kHz | 100 ms | normal | idle | 297 / 328 ms | 99 ms | ok |
| 10 kHz | 50 ms | normal | idle | 281 / 282 ms | 39 ms | ok |
| 10 kHz | 200–50 ms | normal | 4 of 8 cores | as idle | as idle | ok |
| 10 kHz | 500 ms | normal | 8 of 8 cores | – | 20 ms | **underflow** |
| 10 kHz | 100 ms | normal | 8 of 8 cores | – | 20 ms | **write timeout** |
| 10 kHz | 100 ms | high | 8 of 8 cores | 297 / 329 ms | 99 ms | ok |
| 40 kHz | 100 ms | normal | idle | 172 / 188 ms | 84 ms | ok |
| 40 kHz | 50 ms | normal | idle | 110 / 125 ms | 39 ms | ok |
| 40 kHz | 100 ms | high | 8 of 8 cores | 171 / 188 ms | 99 ms | ok |
| 40 kHz | 100 ms | high | 8 of 8 cores, **300 s soak** | 172 / 188 ms | 99 ms | ok |

Conclusions:

- **Starvation is a priority problem, not a buffer problem.** With every core
  busy, even the original 500 ms buffer failed at normal priority. At high
  priority, 50 ms buffers survived.
- **Headroom tracks the host buffer.** The minimum headroom was always close
  to `stream_buffer_s`, so the FIFO adds latency but no measurable margin.
- **Buffers below about 40 ms do not shrink further.** The worker always keeps
  at least two 20 ms chunks queued.

`build_waveform` takes about 24 ms per `send` at 40 kHz (3 s keep, idle). A
send previously built the waveform twice.

## Chosen hardware settings (`configs/windows.json`)

| setting | value | why |
|---|---|---|
| `sample_rate_hz` | 40000 | FIFO latency ~51 ms instead of ~205 ms |
| `stream_buffer_s` | 0.1 | ~100 ms margin against scheduling stalls |
| `high_priority` | true | prevents DAQ starvation under CPU load |

Expected `send` → coil latency is about 0.2 s, down from about 0.8 s.

Signals are at most 100 Hz, so 40 kHz does not change the field at the signal
frequency. It only makes the DAC staircase finer. The DAC's half-sample hold
delay drops from 50 µs to 12.5 µs, about 0.6° of phase at 46 Hz. To return to
10 kHz, change `sample_rate_hz` alone.

## Re-measuring

Each run logs a `stream_stats` event (`buffer_s`, `min_headroom_s`,
`max_headroom_s`, `max_loop_gap_s`, `chunks`) to the event log. If
`min_headroom_s` approaches 0 in normal operation, increase `stream_buffer_s`.

To sweep buffer sizes on the rig (the probe holds the amplifier-enable line
low, so the coils carry no current):

```powershell
python scripts/buffer_probe.py --config configs/windows.json --buffers 0.05 0.1 0.2 --seconds 120 --high-priority
python scripts/buffer_probe.py --config configs/windows.json --buffers 0.1 --seconds 300 --high-priority --cpu-load 8
```

Run it while the programs that normally share this PC are active. Choose a
buffer whose minimum headroom stays well above zero under the worst realistic
load.

## Tests covering this behaviour

All tests are dry: they use a fake NI-DAQmx driver and never open `Dev1`.

`tests/unit_tests/test_waveform_stream.py`

- Chunks continue the absolute time axis and match an independent reference
  calculation.
- No discontinuity at the 10 s preview-window boundary for 45.99 / 46.35 Hz.
- Samples at the end of a 300 s action are correct; memory stays bounded.
- Preview and stream clip identically at the ±10 A limit.
- Invalid sample ranges are rejected.

`tests/unit_tests/test_nidaqmx_backend.py`

- Streamed DAQ writes are phase-continuous across many preview-window lengths.
- A replacement restarts at phase 0.
- `started` and the keep-time clock wait until the device has generated the
  replacement's first sample.
- Stop waits for the queued ramp, then always writes zero; it is bounded if the
  device never catches up.
- `buffer_s` sets the DAQ buffer size.
- `stream_stats` reveals an injected 120 ms stall, using a fake device that
  generates samples in real time and blocks writes like DAQmx.

`tests/unit_tests/test_controller_lifecycle.py`

- A send builds its waveform exactly once, including the replacement path.
- Action records keep no sample arrays.
- `configs/windows.json` carries the measured settings.
- The server raises process priority only when `high_priority` is set.
