from raft_control.config import ControllerConfig
from raft_control.controller import FieldController
from raft_control.models import ActionRequest
from raft_control.server import RaftControlServer


def test_recent_is_compact_but_contains_original_request():
    controller = FieldController(ControllerConfig(log_path="tests/.artifacts/server-viewer.jsonl"))
    controller.enable()
    record = controller.send(ActionRequest(1, 2, 25, 25, 0, 0, 0.02))
    server = RaftControlServer(controller, "127.0.0.1", 0)

    response = server._dispatch({"type": "recent", "request_id": "viewer"})
    viewed = response["actions"][-1]

    assert viewed["action_id"] == record.action_id
    assert viewed["request"]["bx"] == 1
    assert viewed["request"]["by"] == 2
    assert "calculated_currents_a" not in viewed
    assert "transmitted_currents_a" not in viewed
    assert response["active_action_id"] == record.action_id
    assert response["enabled"] is True
    controller.close()


def test_recent_identifies_active_action_instead_of_later_rejection():
    controller = FieldController(ControllerConfig(log_path="tests/.artifacts/server-active.jsonl"))
    controller.enable()
    active = controller.send(ActionRequest(1, 2, 25, 25, 0, 0, 0.02))
    controller.enabled = False
    rejected = controller.send(ActionRequest(3, 4, 25, 25, 0, 0, 0.02))
    controller.enabled = True
    server = RaftControlServer(controller, "127.0.0.1", 0)

    response = server._dispatch({"type": "recent", "request_id": "viewer"})

    assert response["actions"][-1]["action_id"] == rejected.action_id
    assert response["active_action_id"] == active.action_id
    controller.close()
