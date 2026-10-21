from __future__ import annotations

import math
import numpy as np

from .calibration import load_calibration
from .config import ControllerConfig
from .models import ActionRequest, WaveformResult


def _components(request: ActionRequest, config: ControllerConfig, t: np.ndarray) -> np.ndarray:
    px, py = np.deg2rad(config.phase_x_deg), np.deg2rad(config.phase_y_deg)
    fpx, fpy = np.deg2rad(config.force_phase_x_deg), np.deg2rad(config.force_phase_y_deg)
    return np.vstack([
        request.bx * np.cos(2 * np.pi * request.fx * t + px) * config.direction_x,
        request.by * np.sin(2 * np.pi * request.fy * t + py) * config.direction_y,
        request.FX * np.cos(2 * np.pi * request.fx * t + fpy),
        request.FY * np.sin(2 * np.pi * request.fy * t + fpx),
        # np.full_like(t, request.FX) * np.sin(2 * np.pi * request.fy * t + fpx),
        # np.full_like(t, request.FY) * np.cos(2 * np.pi * request.fx * t + fpy),
    ])
    #if request.fx == 0 and request.fy == 0:
    #    components = np.vstack([
    #        np.full_like(t, request.bx),
    #        np.full_like(t, request.by),
    #        np.full_like(t, request.FX),
    #       np.full_like(t, request.FY),
    #    ])


class WaveformStream:
    """Procedural hardware output evaluated on an absolute sample index.

    Chunks are computed from t = n / sample_rate, so consecutive chunks are
    phase-continuous for arbitrary fractional frequencies and memory use does
    not depend on the keep time.
    """

    __slots__ = ("request", "config", "calibration", "sample_rate_hz", "current_limit_a")

    def __init__(self, request: ActionRequest, config: ControllerConfig, calibration: np.ndarray):
        self.request = request
        self.config = config
        self.calibration = calibration
        self.sample_rate_hz = config.sample_rate_hz
        self.current_limit_a = config.current_limit_a

    def samples(self, start_index: int, sample_count: int) -> np.ndarray:
        """Clipped coil currents for samples [start_index, start_index + sample_count)."""
        if start_index < 0 or sample_count <= 0:
            raise ValueError("start_index must be >= 0 and sample_count > 0")
        t = (start_index + np.arange(sample_count, dtype=float)) / self.sample_rate_hz
        currents = self.calibration @ _components(self.request, self.config, t)
        return np.clip(currents, -self.current_limit_a, self.current_limit_a)


def build_waveform(request: ActionRequest, config: ControllerConfig, duration: float | None = None) -> WaveformResult:
    if request.duration <= 0 or not math.isfinite(request.duration):
        raise ValueError("duration must be finite and greater than zero")
    if not all(math.isfinite(float(v)) for v in request.as_dict().values()):
        raise ValueError("all action values must be finite")
    if config.sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    # The arrays are a bounded preview (plots, action records).  Hardware
    # output is generated chunk by chunk from the stream, so long keep times
    # (for example 300 s at 10 kHz) never allocate the full waveform.  The
    # requested keep time is carried separately in duration_s.
    output_duration = min(
        request.duration if duration is None else duration,
        config.stream_window_s,
    )
    count = max(2, int(round(output_duration * config.sample_rate_hz)))
    t = np.arange(count, dtype=float) / config.sample_rate_hz
    components = _components(request, config, t)
    calibration = load_calibration(config.calibration_path)
    currents = calibration @ components
    gradient = components.copy()
    gradient[:2] = 0.0
    limit = config.current_limit_a
    clipped = np.clip(currents, -limit, limit)
    return WaveformResult(
        time_s=t,
        requested_components=components,
        currents_a=currents,
        gradient_currents_a=calibration @ gradient,
        clipped_currents_a=clipped,
        clipped=np.not_equal(currents, clipped),
        sample_rate_hz=config.sample_rate_hz,
        calibration_version=config.calibration_version,
        duration_s=request.duration,
        stream=WaveformStream(request, config, calibration),
    )
