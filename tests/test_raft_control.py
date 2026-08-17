import json
import numpy as np
from pathlib import Path

from raft_control.backends import SimulationBackend
from raft_control.config import ControllerConfig
from raft_control.controller import FieldController
from raft_control.models import ActionRequest
from raft_control.protocol import FrameDecoder, encode_message


ARTIFACTS = Path(__file__).parent / ".artifacts"


def make_config(**kwargs):
    ARTIFACTS.mkdir(exist_ok=True)
    log_path = ARTIFACTS / "legacy-events.jsonl"
    log_path.unlink(missing_ok=True)
    values = {"backend": "simulation", "sample_rate_hz": 1000, "log_path": str(log_path)}
    values.update(kwargs)
    return ControllerConfig(**values)


def req(**kwargs):
    values = {"bx": 1, "by": 1, "fx": 1, "fy": 1, "FX": 0, "FY": 0, "duration": 0.02}
    values.update(kwargs)
    return ActionRequest(**values)


def test_protocol_handles_fragmented_frames():
    payload = encode_message({"type": "heartbeat", "request_id": "x"})
    decoder = FrameDecoder()
    assert decoder.feed(payload[:2]) == []
    assert decoder.feed(payload[2:7]) == []
    assert decoder.feed(payload[7:]) == [{"type": "heartbeat", "request_id": "x"}]


def test_reference_waveform_and_safety():
    controller = FieldController(make_config(sample_rate_hz=10000), backend=SimulationBackend())
    wave = controller.preview(req(bx=10, by=20, fx=1, fy=2, duration=1, FX=.5, FY=-.5)).waveform
    np.testing.assert_allclose(np.abs(wave.clipped_currents_a).max(axis=1), [3.83720259, 1.57570633, 3.60082197, 4.59213331], atol=1e-7)
    assert np.max(np.abs(wave.clipped_currents_a)) <= 10


def test_disabled_send_is_rejected():
    controller = FieldController(make_config(), backend=SimulationBackend())
    assert controller.send(req()).status.value == "rejected"


def test_lifecycle_and_logging():
    controller = FieldController(make_config(), backend=SimulationBackend())
    controller.enable()
    record = controller.send(req())
    assert record.action_id
    controller.stop(record.action_id)
    log = (ARTIFACTS / "legacy-events.jsonl").read_text()
    assert '"event":"validated"' in log
    assert '"event":"queued"' in log
