import numpy as np

from raft_control.calibration import load_calibration
from raft_control.config import ControllerConfig
from raft_control.models import ActionRequest
from raft_control.waveform import WaveformStream, build_waveform


# Fractional frequencies whose cycle count is not an integer within a 10 s
# window: a repeating window would jump by 324 deg (X) and 180 deg (Y).
EXAMPLE = ActionRequest(30.038, 53.65, 45.99, 46.35, 0, 0, 300)


def expected_currents(request, config, indices):
    """Independent reference: currents at absolute sample indices."""
    t = np.asarray(indices, dtype=float) / config.sample_rate_hz
    px, py = np.deg2rad(config.phase_x_deg), np.deg2rad(config.phase_y_deg)
    fpx, fpy = np.deg2rad(config.force_phase_x_deg), np.deg2rad(config.force_phase_y_deg)
    components = np.vstack([
        request.bx * np.cos(2 * np.pi * request.fx * t + px) * config.direction_x,
        request.by * np.sin(2 * np.pi * request.fy * t + py) * config.direction_y,
        request.FX * np.cos(2 * np.pi * request.fx * t + fpy),
        request.FY * np.sin(2 * np.pi * request.fy * t + fpx),
    ])
    currents = load_calibration(config.calibration_path) @ components
    return np.clip(currents, -config.current_limit_a, config.current_limit_a)


def test_build_waveform_carries_stream_with_keep_time():
    config = ControllerConfig(sample_rate_hz=10_000, stream_window_s=10)
    waveform = build_waveform(EXAMPLE, config)

    assert isinstance(waveform.stream, WaveformStream)
    assert waveform.stream.sample_rate_hz == 10_000
    assert waveform.duration_s == 300


def test_chunks_continue_the_absolute_time_axis():
    config = ControllerConfig(sample_rate_hz=10_000, stream_window_s=10)
    stream = build_waveform(EXAMPLE, config).stream
    chunk = 200

    parts = [stream.samples(start, chunk) for start in range(0, 2_000, chunk)]
    joined = np.concatenate(parts, axis=1)

    np.testing.assert_allclose(joined, stream.samples(0, 2_000), atol=1e-12)
    np.testing.assert_allclose(joined, expected_currents(EXAMPLE, config, np.arange(2_000)), atol=1e-9)
    # First sample of each chunk is the exact continuation of the previous one.
    for i in range(1, len(parts)):
        np.testing.assert_allclose(
            parts[i][:, 0], expected_currents(EXAMPLE, config, [i * chunk])[:, 0], atol=1e-9
        )


def test_no_discontinuity_at_stream_window_boundary():
    config = ControllerConfig(sample_rate_hz=10_000, stream_window_s=10)
    stream = build_waveform(EXAMPLE, config).stream
    boundary = int(10 * config.sample_rate_hz)

    around = stream.samples(boundary - 100, 200)

    np.testing.assert_allclose(
        around, expected_currents(EXAMPLE, config, np.arange(boundary - 100, boundary + 100)), atol=1e-9
    )
    steps = np.abs(np.diff(around, axis=1))
    # The step across the boundary is no larger than the ordinary sample steps
    # around it, per channel.
    ordinary = np.delete(steps, 99, axis=1).max(axis=1)
    assert np.all(steps[:, 99] <= ordinary * 1.01)
    # A 180 deg Y jump (as the old repeating window produced) would be far
    # larger than one 10 kHz sample step of a ~46 Hz signal.
    assert steps.max() < 0.1


def test_samples_at_end_of_300_second_action_match_reference():
    config = ControllerConfig(sample_rate_hz=10_000, stream_window_s=10)
    stream = build_waveform(EXAMPLE, config).stream
    start = int(300 * config.sample_rate_hz) - 200

    np.testing.assert_allclose(
        stream.samples(start, 200),
        expected_currents(EXAMPLE, config, np.arange(start, start + 200)),
        atol=1e-9,
    )


def test_300_second_action_memory_is_bounded():
    config = ControllerConfig(sample_rate_hz=10_000, stream_window_s=10)
    waveform = build_waveform(EXAMPLE, config)

    # Preview arrays are limited to the stream window.
    assert waveform.clipped_currents_a.shape == (4, 100_000)
    # The stream stores no precomputed samples; chunks are sized on request.
    stored = [getattr(waveform.stream, name) for name in WaveformStream.__slots__]
    assert all(np.asarray(value).size <= 16 for value in stored)
    assert waveform.stream.samples(2_000_000, 200).shape == (4, 200)


def test_clipping_is_identical_for_preview_and_stream():
    config = ControllerConfig(sample_rate_hz=1_000, stream_window_s=2)
    request = ActionRequest(160, 160, 45.99, 46.35, 0.8, 0.8, 5)
    waveform = build_waveform(request, config)
    count = waveform.clipped_currents_a.shape[1]

    assert waveform.clipped.any()
    np.testing.assert_array_equal(waveform.stream.samples(0, count), waveform.clipped_currents_a)
    assert np.abs(waveform.stream.samples(0, 10_000)).max() <= config.current_limit_a


def test_stream_rejects_invalid_ranges():
    config = ControllerConfig()
    stream = build_waveform(EXAMPLE, config).stream
    for start, count in ((-1, 10), (0, 0), (0, -5)):
        try:
            stream.samples(start, count)
        except ValueError:
            pass
        else:
            raise AssertionError(f"samples({start}, {count}) should be rejected")
