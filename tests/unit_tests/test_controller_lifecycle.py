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
