from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ActionRequest:
    bx: float
    by: float
    fx: float
    fy: float
    FX: float
    FY: float
    duration: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ActionRequest":
        fields = {"bx", "by", "fx", "fy", "FX", "FY", "duration"}
        missing = fields - set(values)
        extra = set(values) - fields
        if missing:
            raise ValueError(f"missing action fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"unknown action fields: {', '.join(sorted(extra))}")
        try:
            return cls(**{name: float(values[name]) for name in fields})
        except (TypeError, ValueError) as exc:
            raise ValueError("action fields must be numeric") from exc

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class ActionStatus(str, Enum):
    VALIDATED = "validated"
    QUEUED = "queued"
    STARTED = "started"
    DURATION_ELAPSED = "duration_elapsed"
    COMPLETED = "completed"
    REPLACED = "replaced"
    STOPPED = "stopped"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(slots=True)
class WaveformResult:
    time_s: Any
    requested_components: Any
    currents_a: Any
    gradient_currents_a: Any
    clipped_currents_a: Any
    clipped: Any
    sample_rate_hz: float
    calibration_version: str


@dataclass(slots=True)
class PreviewResult:
    request: ActionRequest
    waveform: WaveformResult
    backend: str
    channel_mapping: list[str]


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    request: ActionRequest
    status: ActionStatus
    backend: str
    channel_mapping: list[str]
    calibration_version: str
    calculated_currents_a: Any = None
    transmitted_currents_a: Any = None
    clipped: bool = False
    min_currents_a: list[float] = field(default_factory=list)
    max_currents_a: list[float] = field(default_factory=list)
    timestamps: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    tail_active: bool = False
    duration_elapsed: bool = False
    replacement_action_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["request"] = self.request.as_dict()
        result["status"] = self.status.value
        for key in ("calculated_currents_a", "transmitted_currents_a"):
            if hasattr(result[key], "tolist"):
                result[key] = result[key].tolist()
        return result
