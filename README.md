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

The GUI previews waveforms locally and communicates with hardware only through
the TCP client. Press **Enable** before **Send action**. **Stop** and closing
the window stop and disable the remote controller.

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
