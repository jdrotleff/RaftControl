from __future__ import annotations

import math
import numpy as np

from .calibration import load_calibration
from .config import ControllerConfig
from .models import ActionRequest, WaveformResult


def build_waveform(request: ActionRequest, config: ControllerConfig, duration: float | None = None) -> WaveformResult:
    if request.duration <= 0 or not math.isfinite(request.duration):
        raise ValueError("duration must be finite and greater than zero")
    if not all(math.isfinite(float(v)) for v in request.as_dict().values()):
        raise ValueError("all action values must be finite")
    if config.sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    output_duration = request.duration if duration is None else duration
    count = max(2, int(round(output_duration * config.sample_rate_hz)))
    t = np.arange(count, dtype=float) / config.sample_rate_hz
    px, py = np.deg2rad(config.phase_x_deg), np.deg2rad(config.phase_y_deg)
    fpx, fpy = np.deg2rad(config.force_phase_x_deg), np.deg2rad(config.force_phase_y_deg)
    components = np.vstack([
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
    )

