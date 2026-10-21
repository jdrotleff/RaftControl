import time
from pathlib import Path

from raft_control.backends import SimulationBackend
from raft_control.config import ControllerConfig
from raft_control.controller import FieldController
from raft_control.models import ActionRequest, ActionStatus


ARTIFACTS = Path(__file__).parents[1] / ".artifacts"


def action(**changes):
    values = dict(bx=1, by=1, fx=25, fy=25, FX=0, FY=0, duration=0.03)
    values.update(changes)
    return ActionRequest(**values)


def controller(name):
    ARTIFACTS.mkdir(exist_ok=True)
    log_path = ARTIFACTS / name
    log_path.unlink(missing_ok=True)
    config = ControllerConfig(sample_rate_hz=1000, log_path=str(log_path))
    backend = SimulationBackend()
    return FieldController(config, backend), backend, log_path


def test_replacement_is_immediate_and_final_action_keeps_running():
    control, backend, _ = controller("replacement.jsonl")
    control.enable()
    first = control.send(action())
    second = control.send(action(bx=2))
    time.sleep(0.06)

    assert control.status(first.action_id).status == ActionStatus.REPLACED
    assert control.status(first.action_id).replacement_action_id == second.action_id
    assert control.status(second.action_id).status == ActionStatus.DURATION_ELAPSED
    assert control.status(second.action_id).tail_active

    control.stop()
    assert backend.ramped_to_zero
    assert control.status(second.action_id).status == ActionStatus.STOPPED


def test_lifecycle_log_is_compact():
    control, _, log_path = controller("compact.jsonl")
    control.enable()
    record = control.send(action())
    control.stop(record.action_id)
    log = log_path.read_text(encoding="utf-8")

    assert record.action_id in log
    assert '"event":"rejected"' not in log
    assert "calculated_currents_a" not in log
    assert "transmitted_currents_a" not in log


def test_send_builds_waveform_once_per_action(monkeypatch):
    import raft_control.controller as controller_module

    calls = []
    original = controller_module.build_waveform
    monkeypatch.setattr(
        controller_module, "build_waveform", lambda *a, **k: (calls.append(a[0]), original(*a, **k))[1]
    )
    control, _, _ = controller("build-once.jsonl")
    control.enable()
    control.send(action())
    control.send(action(bx=2))  # replacement path
    control.stop()

    assert len(calls) == 2


def test_records_do_not_retain_sample_arrays():
    control, _, _ = controller("no-arrays.jsonl")
    control.enable()
    record = control.send(action(duration=5))
    control.stop()

    assert record.calculated_currents_a is None
    assert record.transmitted_currents_a is None
    # The compact per-channel extremes are kept.
    assert len(record.min_currents_a) == 4 and len(record.max_currents_a) == 4
    assert record.as_dict()["calculated_currents_a"] is None


def test_hardware_config_uses_measured_low_latency_settings():
    from raft_control.config import load_config

    config = load_config(Path(__file__).parents[2] / "configs" / "windows.json")

    # Measured on the USB-6363 rig: 40 kHz, 100 ms buffer and high priority
    # held 99 ms headroom under full CPU load with ~170 ms output latency.
    assert config.sample_rate_hz == 40_000
    assert config.stream_buffer_s == 0.1
    assert config.high_priority is True


def test_server_raises_priority_only_when_configured(monkeypatch):
    import raft_control.server as server_module

    raised = []
    monkeypatch.setattr(server_module, "raise_process_priority", lambda: raised.append(True) or True)
    server_module.apply_process_priority(ControllerConfig(high_priority=False))
    assert raised == []
    server_module.apply_process_priority(ControllerConfig(high_priority=True))
    assert raised == [True]


def test_started_switch_is_logged_as_phase_at_switch():
    control, backend, log_path = controller("phase_at_switch.jsonl")
    backend.started_switch = {"sample_index": 5, "time_s": 0.005, "carrier_phase_rad": [0.1, 0.2], "phase_jump_rad": [0.0, 0.3]}
    control.enable()
    record = control.send(action())
    time.sleep(0.05)
    control.stop(record.action_id)

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if '"phase_at_switch"' in line]
    assert lines
    assert f'"action_id":"{record.action_id}"' in lines[0]
    assert '"phase_jump_rad":[0.0,0.3]' in lines[0]
