# RaftControl

Standalone Windows field-control service for the NI USB-6363.

RaftControl contains no RL, tracking, LabVIEW, or SwarmRL dependencies. It
accepts canonical magnetic-field actions over a length-prefixed JSON TCP
protocol and converts them into calibrated, safety-limited coil currents.

The existing LabVIEW protocol remains in the Rafts/SwarmRL-MPI-IS repositories
and is not modified by this project.

## Windows setup

Install Python, `uv`, and NI-DAQmx on Windows. From this repository:

```powershell
uv sync --extra hardware
uv run python -m raft_control.server --config configs/windows.json
```

Confirm the NI device name and AO channels in NI MAX before enabling output.
The service starts disabled. A client must explicitly send `enable` before any
action can be accepted. Hardware testing must first use a disconnected load
and a measured low-amplitude signal.

## Manual GUI

Install the GUI extra and run it from a second terminal while the server is
running:

```powershell
uv sync --extra hardware --extra gui
uv run python scripts/gui.py
```

The GUI has two modes:

```powershell
# Local manual controller owning the hardware (default)
uv run python scripts/gui.py --mode controller

# Equivalent explicit local form
uv run python scripts/gui.py --mode controller --local

# Local controller that queues actions and runs them sequentially
uv run python scripts/gui.py --mode controller --queue

# Show a button that generates random field/frequency parameters
uv run python scripts/gui.py --mode controller --shuffle

# Remote manual controller; the RaftControl server must already be running
uv run python scripts/gui.py --mode controller --remote

# Read-only view of actions submitted by RL or another controller
uv run python scripts/gui.py --mode viewer
```

The GUI previews waveforms locally. Controller mode owns the hardware directly
unless `--remote` is supplied. Press **Enable** before **Send action**. Viewer mode has
no enable, send, disable, or stop controls and polls recent server actions.
When a new action appears, viewer mode reads its original request parameters
from the server record and regenerates the plots locally. It does not render
samples copied from the NI output buffer.

Each submitted action receives a unique action ID. The GUI shows its short ID
and lifecycle status (`queued`, `started`, `duration_elapsed`, `replaced`,
`stopped`, or `failed`). A new action replaces the active action without
recreating the NI-DAQmx task. After its duration elapses, the last action keeps
running until another action replaces it or the operator presses **Stop** or
**Disable**. Shutdown ramps all four outputs to zero over `safe_ramp_s`.

Controller mode overwrites the active action by default. With `--queue`, the
GUI stores subsequent actions locally and submits the next one when the active
action reaches `duration_elapsed`. **Stop** and **Disable** clear that pending
queue. Queue mode works with both `--local` and `--remote`; its queue belongs to
the GUI process and is not persisted by the server.

With `--shuffle`, the GUI shows a **Shuffle and send** button. It independently
samples `bx` and `by` from 0–80 G and `fx` and `fy` from 0–60 Hz, sets both
gradients to zero, preserves the current duration, updates the preview, and
immediately sends or queues the generated action through the normal controller
flow.

## Protocol

Each frame is a 4-byte unsigned big-endian payload length followed by UTF-8
JSON. See `src/raft_control/server.py` for the request types. The controller
generates action IDs; client request IDs are only correlation metadata.

RaftControl uses TCP port 6340 by default, matching the existing Rafts
configuration. The LabVIEW process must be stopped before RaftControl starts;
the two protocols are not intended to run simultaneously.

## Simulation tests

```powershell
uv sync
uv run pytest
```

Automated tests are dry: they use the simulation backend or a fake NI-DAQmx
driver and never open `Dev1`. Unit tests live in `tests/unit_tests`; combined
controller-flow tests live in `tests/integration_tests`. Physical coil testing
is an explicit operator action performed through the GUI.
