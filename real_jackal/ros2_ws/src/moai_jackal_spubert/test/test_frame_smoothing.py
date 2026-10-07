"""Rate-limited map->odom following: pure functions."""
import importlib.util
import math
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "frame_smoothing_under_test", Path(__file__).parents[1] / "moai_jackal_spubert/frame_smoothing.py")
fs = importlib.util.module_from_spec(spec)
sys.modules["frame_smoothing_under_test"] = fs
spec.loader.exec_module(fs)


def test_first_use_and_disabled_rate_take_the_latest_transform():
    assert fs.slew_transform(None, (10.0, -0.5, 0.02), 0.1, 0.1, 0.05) == (10.0, -0.5, 0.02)
    assert fs.slew_transform((9.0, 0.0, 0.0), (10.0, -0.5, 0.02), 0.1, 0.0, 0.05) == (10.0, -0.5, 0.02)


def test_small_corrections_pass_straight_through():
    out = fs.slew_transform((10.0, -0.50, 0.0), (10.004, -0.503, 0.001), 0.1, 0.1, 0.05)
    assert out == (10.004, -0.503, 0.001)


def test_large_correction_is_followed_at_the_rate_limit_not_at_once():
    applied = (10.0, 0.0, 0.0)
    for _ in range(10):  # 1 s at 0.1 m/s
        applied = fs.slew_transform(applied, (10.0, -0.6, 0.0), 0.1, 0.1, 0.05)
    assert applied[1] == -0.1 or math.isclose(applied[1], -0.1, abs_tol=1e-9)
    for _ in range(60):  # 6 s more: arrives and stops exactly
        applied = fs.slew_transform(applied, (10.0, -0.6, 0.0), 0.1, 0.1, 0.05)
    assert math.isclose(applied[1], -0.6, abs_tol=1e-9)


def test_direction_is_toward_the_target_and_yaw_wraps_the_short_way():
    out = fs.slew_transform((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), 1.0, 1.0, 1.0)
    assert math.isclose(out[0], 0.6) and math.isclose(out[1], 0.8)
    out = fs.slew_transform((0.0, 0.0, math.radians(179)), (0.0, 0.0, math.radians(-179)), 1.0, 1.0, math.radians(1.0))
    assert math.isclose(math.degrees(out[2]) % 360.0, 180.0, abs_tol=1e-6)  # crossed +-180 forward


def test_bad_input_keeps_the_previous_transform():
    prev = (10.0, 0.0, 0.0)
    assert fs.slew_transform(prev, (float("nan"), 0.0, 0.0), 0.1, 0.1, 0.05) == prev
    assert fs.slew_transform(None, (float("nan"), 0.0, 0.0), 0.1, 0.1, 0.05)[0] != fs.slew_transform(None, (1.0, 0.0, 0.0), 0.1, 0.1, 0.05)[0]


def test_negative_dt_never_moves_backwards():
    out = fs.slew_transform((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), -5.0, 0.1, 0.05)
    assert out == (0.0, 0.0, 0.0)


def test_large_jump_detector():
    assert fs.is_large_jump((0, 0, 0), (3.5, 0, 0), 3.0) and not fs.is_large_jump((0, 0, 0), (2.0, 0, 0), 3.0)
