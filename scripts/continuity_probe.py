"""Check on the real NI backend that replacements continue on the absolute sample clock.

Streams an action and replaces it every --replace-every seconds, first with the
identical action, then with an action of the same frequencies but another
amplitude.  Every switch reported by the backend (started_switch, the data
behind the controller's phase_at_switch log event) is checked:

- sample_index is strictly increasing (the clock never restarts at 0)
- the carrier phase jump is 0 for actions with the same frequencies

The amplifier-enable line is held low throughout: analog outputs carry the
waveform voltage, but the amplifier does not drive the coils.

Example:
    python scripts/continuity_probe.py --config configs/windows.json
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from raft_control.backends import NIDAQmxBackend
from raft_control.config import load_config
from raft_control.models import ActionRequest
from raft_control.waveform import build_waveform


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/windows.json")
    parser.add_argument("--replacements", type=int, default=8)
    parser.add_argument("--replace-every", type=float, default=1.5)
    parser.add_argument("--buffer", type=float, default=0.1, help="host buffer in s")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.backend != "nidaqmx":
        parser.error("needs a config with backend 'nidaqmx'")
    if config.amplifier_enable_line is None:
        parser.error("no amplifier_enable_line configured; the probe never enables the amplifier")

    # Same frequencies on purpose; fractional so a restart at index 0 would jump.
    same = ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, args.replace_every)
    other_amplitude = ActionRequest(20.0, 40.0, 45.99, 46.35, 0, 0, args.replace_every)
    waveforms = [build_waveform(a, config) for a in (same, same, other_amplitude, same)]

    backend = NIDAQmxBackend(
        config.channels, config.daq_device, config.current_limit_a, config.safe_ramp_s,
        config.amplifier_enable_line, args.buffer,
    )
    switches = []
    errors = []
    started = threading.Event()

    def on_done(error):
        if error and error != "stopped":
            errors.append(error)

    try:
        backend.initialize()  # enable line low; never call enable() here
        backend.start(waveforms[0], started.set, lambda: None, on_done)
        if not started.wait(5):
            time.sleep(0.5)
            raise SystemExit(f"stream did not start: {errors[0] if errors else 'no error reported'}")
        for i in range(1, args.replacements + 1):
            time.sleep(args.replace_every)
            reached = threading.Event()

            def on_started():
                switches.append(dict(backend.started_switch or {}))
                reached.set()

            backend.replace(waveforms[i % len(waveforms)], on_started, lambda: None, on_done)
            if not reached.wait(5) or errors:
                break
    finally:
        backend.stop()
        backend.close()

    if errors:
        raise SystemExit(f"FAILED: {errors[0]}")
    print(f"{'#':>2} {'sample_index':>12} {'time_s':>9} {'phase jump x/y (mrad)':>24}")
    ok = True
    previous = -1
    for i, s in enumerate(switches, 1):
        jump = np.array(s.get("phase_jump_rad", [np.nan, np.nan]))
        bad = s.get("sample_index", 0) <= previous or not np.allclose(jump, 0, atol=1e-6)
        ok &= not bad
        previous = s.get("sample_index", previous)
        print(f"{i:>2} {s.get('sample_index'):>12} {s.get('time_s', float('nan')):>9.3f} "
              f"{jump[0] * 1e3:>11.4f} {jump[1] * 1e3:>11.4f}  {'FAIL' if bad else 'ok'}")
    if len(switches) < args.replacements:
        ok = False
        print(f"only {len(switches)} of {args.replacements} replacements reached the output")
    print("PASS: sample clock continuous, no phase jumps" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
