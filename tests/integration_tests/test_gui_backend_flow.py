import time
from pathlib import Path

from raft_control.backends import SimulationBackend
from raft_control.config import ControllerConfig
from raft_control.controller import FieldController
from raft_control.models import ActionRequest, ActionStatus


def test_gui_equivalent_enable_send_replace_stop_flow():
    artifacts = Path(__file__).parents[1] / ".artifacts"
    artifacts.mkdir(exist_ok=True)
    control = FieldController(
        ControllerConfig(sample_rate_hz=1000, log_path=str(artifacts / "gui-flow.jsonl")),
        SimulationBackend(),
    )
    request = ActionRequest(50, 50, 25, 25, 0, 0, 0.02)

    control.enable()
    first = control.send(request)
    second = control.send(request)
    time.sleep(0.04)

    assert first.action_id != second.action_id
    assert control.status(first.action_id).status == ActionStatus.REPLACED
    assert control.status(second.action_id).status == ActionStatus.DURATION_ELAPSED
    control.stop(second.action_id)
    assert control.status(second.action_id).status == ActionStatus.STOPPED
