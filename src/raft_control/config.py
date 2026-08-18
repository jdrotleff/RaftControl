from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "configs" / "field_config" / "AM_new_coil_system.txt"


@dataclass(slots=True)
class ControllerConfig:
    calibration_path: str = str(DEFAULT_CALIBRATION_PATH)
    calibration_version: str = "AM_new_coil_system"
    backend: str = "simulation"
    daq_device: str = "Dev1"
    channels: list[str] = field(default_factory=lambda: ["ao0", "ao1", "ao2", "ao3"])
    amplifier_enable_line: str | None = None
    sample_rate_hz: float = 1000.0
    current_limit_a: float = 10.0
    phase_x_deg: float = 0.0
    phase_y_deg: float = 0.0
    force_phase_x_deg: float = 0.0
    force_phase_y_deg: float = 0.0
    direction_x: float = 1.0
    direction_y: float = 1.0
    python_port: int = 6340
    bind_address: str = "127.0.0.1"
    heartbeat_timeout_s: float = 5.0
    safe_ramp_s: float = 0.05
    log_path: str = "logs/raft_control.jsonl"

    def __post_init__(self) -> None:
        self.calibration_path = str(DEFAULT_CALIBRATION_PATH)
        if self.current_limit_a != 10.0:
            raise ValueError("RaftControl requires a ±10 A current limit")

    @property
    def channel_mapping(self) -> list[str]:
        return [f"{self.daq_device}/{channel}" for channel in self.channels]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ControllerConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("calibration_path", None)
    return ControllerConfig(**data)
