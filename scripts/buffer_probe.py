"""Measure how much NI-DAQmx output buffer RaftControl needs on this machine.

For each buffer size the probe streams a real action, replaces it periodically
(like an RL agent) and reports:

- worst stall:   largest drop of the output queue below its steady level, i.e.
                 the longest time the DAQ worker was not refilling the buffer
- min headroom:  output left queued at that worst moment (0 = underflow)
- latency:       time from replace() until the replacement reached the outputs

Choose a buffer comfortably larger than the worst stall seen under realistic
load.  Run it while the GUI, the RL agent and anything else normally running on
this PC are active, or add --cpu-load to stress the machine.

The amplifier-enable line is held low throughout: analog outputs carry the
waveform voltage, but the amplifier does not drive the coils.

Example:
    python scripts/buffer_probe.py --config configs/windows.json --buffers 0.05 0.1 0.2 0.5
"""

from __future__ import annotations

import argparse
import multiprocessing
import statistics
import threading
import time

from raft_control.backends import NIDAQmxBackend
from raft_control.config import load_config
from raft_control.models import ActionRequest
from raft_control.waveform import build_waveform


def _burn():
    while True:
        pass


def probe(config, buffer_s, seconds, replace_every, xfer_cond, actions):
    backend = NIDAQmxBackend(
        config.channels,
        config.daq_device,
        config.current_limit_a,
        config.safe_ramp_s,
        config.amplifier_enable_line,
        buffer_s,
        xfer_cond,
    )
    waveforms = [build_waveform(action, config) for action in actions]
    started = threading.Event()
    errors = []
    latencies = []
    try:
        backend.initialize()  # enable line low; never call enable() here
        backend.start(waveforms[0], started.set, lambda: None, lambda e: e and e != "stopped" and errors.append(e))
        if not started.wait(5):
            raise RuntimeError("stream did not start")
        end = time.monotonic() + seconds
        index = 0
        while time.monotonic() < end and not errors:
            time.sleep(min(replace_every, max(0.0, end - time.monotonic())))
            if time.monotonic() >= end or errors:
                break
            index += 1
            reached = threading.Event()
            queued = time.monotonic()
            backend.replace(waveforms[index % len(waveforms)], reached.set, lambda: None, lambda e: None)
            if reached.wait(5):
                latencies.append(time.monotonic() - queued)
    finally:
        backend.stop()
        stats = dict(backend.stream_stats)
        backend.close()
    return stats, latencies, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/windows.json")
    parser.add_argument("--buffers", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5], help="host buffer sizes in s")
    parser.add_argument("--seconds", type=float, default=60.0, help="duration per buffer size")
    parser.add_argument("--replace-every", type=float, default=3.0, help="replacement period in s (RL step)")
    parser.add_argument("--xfer-cond", default=None,
                        help="OutputDataTransferCondition name, e.g. ON_BOARD_MEMORY_EMPTY or ON_BOARD_MEMORY_HALF_FULL_OR_LESS")
    parser.add_argument("--cpu-load", type=int, default=0, help="number of CPU-burning processes")
    parser.add_argument("--allow-without-enable-line", action="store_true",
                        help="run although no amplifier_enable_line is configured (make sure the amplifier is off)")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.backend != "nidaqmx":
        parser.error("the probe needs a config with backend 'nidaqmx'")
    if config.amplifier_enable_line is None and not args.allow_without_enable_line:
        parser.error("no amplifier_enable_line configured; switch the amplifier off and pass --allow-without-enable-line")

    actions = [
        ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, args.replace_every),
        ActionRequest(20.0, 40.0, 12.34, 56.78, 0, 0, args.replace_every),
    ]
    burners = [multiprocessing.Process(target=_burn, daemon=True) for _ in range(args.cpu_load)]
    for process in burners:
        process.start()
    try:
        print(f"{config.sample_rate_hz:g} Hz, {len(config.channels)} channels, xfer-cond={args.xfer_cond or 'default'}, "
              f"cpu-load={args.cpu_load}, {args.seconds:g} s per buffer")
        print(f"{'buffer':>8} {'queue':>8} {'worst stall':>12} {'min headroom':>13} {'latency med/max':>16}  result")
        for buffer_s in args.buffers:
            stats, latencies, errors = probe(config, buffer_s, args.seconds, args.replace_every, args.xfer_cond, actions)
            queue = stats.get("max_headroom_s")
            low = stats.get("min_headroom_s")
            stall = queue - low if queue is not None and low is not None else None
            latency = (f"{statistics.median(latencies) * 1e3:.0f}/{max(latencies) * 1e3:.0f} ms" if latencies else "-")
            fmt = lambda v: f"{v * 1e3:.0f} ms" if v is not None else "-"
            result = f"FAILED: {errors[0]}" if errors else "ok"
            print(f"{buffer_s * 1e3:>5.0f} ms {fmt(queue):>8} {fmt(stall):>12} {fmt(low):>13} {latency:>16}  {result}")
    finally:
        for process in burners:
            process.terminate()


if __name__ == "__main__":
    main()
