"""Choose a live model frame without treating millimetre noise as heading.

The default preserves the training frame (last observed displacement) during
reliable motion, and uses the current odometry yaw during stops and noisy motion.
``odom_yaw`` always follows the body frame and ``adaptive`` uses a longer motion
window; these alternatives change normalization during normal moving operation.
All modes still need evaluation with the existing checkpoint in closed loop.
It uses measured sample times, motion gates and hysteresis, but never holds an
old heading while the latest segment is stationary or inconsistent with yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


XY = Tuple[float, float]


def angle_difference_rad(left: float, right: float) -> float:
    """Return the signed shortest angular difference in [-pi, pi]."""
    return math.atan2(math.sin(left - right), math.cos(left - right))


@dataclass(frozen=True)
class HeadingConfig:
    mode: str = "motion_guarded"
    window_s: float = 1.2
    min_displacement_m: float = 0.05
    min_recent_displacement_m: float = 0.01
    enter_speed_mps: float = 0.10
    exit_speed_mps: float = 0.05
    max_yaw_disagreement_rad: float = math.radians(30.0)
    max_sample_gap_s: float = 0.8

    def __post_init__(self) -> None:
        if self.mode not in {"odom_yaw", "adaptive", "motion_guarded"}:
            raise ValueError("heading mode must be odom_yaw, adaptive or motion_guarded")
        for name in (
            "window_s", "min_displacement_m", "min_recent_displacement_m",
            "enter_speed_mps", "exit_speed_mps", "max_yaw_disagreement_rad",
            "max_sample_gap_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.exit_speed_mps >= self.enter_speed_mps:
            raise ValueError("exit_speed_mps must be below enter_speed_mps")
        if self.max_yaw_disagreement_rad > math.pi:
            raise ValueError("max_yaw_disagreement_rad must not exceed pi")


@dataclass(frozen=True)
class HeadingSelection:
    theta_rad: float
    source: str
    diagnostics: Dict[str, Any]


class HeadingSelector:
    def __init__(self, config: Optional[HeadingConfig] = None) -> None:
        self.config = config if config is not None else HeadingConfig()
        self.reset()

    def reset(self) -> None:
        """Clear motion hysteresis after a new goal, clock or frame reset."""
        self._using_motion = False
        self._last_sample_time_s: Optional[float] = None

    def select(
        self,
        *,
        history_xy: Sequence[XY],
        sample_times_s: Optional[Sequence[float]],
        odom_yaw_rad: float,
    ) -> HeadingSelection:
        yaw = float(odom_yaw_rad)
        if not math.isfinite(yaw):
            self.reset()
            raise ValueError("odom_yaw_rad must be finite")
        points = [(float(x), float(y)) for x, y in history_xy]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            self.reset()
            raise ValueError("history_xy must contain finite coordinates")

        dx = dy = 0.0
        if len(points) >= 2:
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        recent_displacement = math.hypot(dx, dy)
        # Exactly mirrors the preserved runtime function for comparison only.
        legacy = math.atan2(dy, dx) if recent_displacement > 1e-3 else yaw
        diagnostics: Dict[str, Any] = {
            "mode": self.config.mode,
            "odom_yaw_rad": yaw,
            "legacy_heading_rad": legacy,
            "legacy_minus_odom_yaw_rad": angle_difference_rad(legacy, yaw),
            "history_count": len(points),
            "recent_displacement_m": recent_displacement,
            "recent_speed_mps": None,
            "window_displacement_m": None,
            "window_speed_mps": None,
            "window_duration_s": None,
            "window_heading_rad": None,
            "motion_active_before": self._using_motion,
        }

        def finish(theta: float, source: str, motion: bool = False) -> HeadingSelection:
            self._using_motion = motion
            diagnostics.update({
                "theta_rad": float(theta),
                "source": source,
                "theta_minus_odom_yaw_rad": angle_difference_rad(theta, yaw),
                "motion_active_after": motion,
            })
            return HeadingSelection(float(theta), source, diagnostics)

        times = None if sample_times_s is None else [float(t) for t in sample_times_s]
        if times is not None and len(times) != len(points):
            self.reset()
            raise ValueError("sample_times_s must correspond one-to-one to history_xy")
        valid_times = (
            times is not None
            and all(math.isfinite(t) for t in times)
            and all(b > a for a, b in zip(times, times[1:]))
        )
        diagnostics["sample_times_valid"] = bool(valid_times)
        diagnostics["sample_times_s"] = times if valid_times else None
        if valid_times and times:
            if self._last_sample_time_s is not None and times[-1] < self._last_sample_time_s:
                self._using_motion = False
                diagnostics["sample_clock_reset"] = True
            self._last_sample_time_s = times[-1]
        if len(points) < 2:
            return finish(yaw, "odom_yaw_insufficient_history")
        if not valid_times:
            return finish(yaw, "odom_yaw" if self.config.mode == "odom_yaw" else "odom_yaw_invalid_timestamps")

        assert times is not None
        recent_dt = times[-1] - times[-2]
        diagnostics["recent_dt_s"] = recent_dt
        recent_speed = recent_displacement / recent_dt
        diagnostics["recent_speed_mps"] = recent_speed
        start = len(times) - 2
        while start > 0 and times[-1] - times[start - 1] <= self.config.window_s + 1e-9:
            start -= 1
        duration = times[-1] - times[start]
        window_dx = points[-1][0] - points[start][0]
        window_dy = points[-1][1] - points[start][1]
        displacement = math.hypot(window_dx, window_dy)
        speed = displacement / duration
        motion_heading = math.atan2(window_dy, window_dx)
        disagreement = abs(angle_difference_rad(motion_heading, yaw))
        recent_heading = math.atan2(dy, dx)
        recent_disagreement = abs(angle_difference_rad(recent_heading, yaw))
        diagnostics.update({
            "window_start_index": start,
            "window_duration_s": duration,
            "window_displacement_m": displacement,
            "window_speed_mps": speed,
            "window_heading_rad": motion_heading,
            "window_yaw_disagreement_rad": disagreement,
            "recent_heading_rad": recent_heading,
            "recent_yaw_disagreement_rad": recent_disagreement,
        })
        if self.config.mode == "odom_yaw":
            return finish(yaw, "odom_yaw")
        if any(b - a > self.config.max_sample_gap_s for a, b in zip(times[start:], times[start + 1:])):
            return finish(yaw, "odom_yaw_sample_gap")
        # A historical moving window must not freeze the frame during a stop
        # or turn in place: use the latest physical yaw immediately.
        if recent_displacement < self.config.min_recent_displacement_m or recent_speed < self.config.exit_speed_mps:
            return finish(yaw, "odom_yaw_low_recent_motion")
        if displacement < self.config.min_displacement_m:
            return finish(yaw, "odom_yaw_low_window_displacement")
        if max(disagreement, recent_disagreement) > self.config.max_yaw_disagreement_rad:
            return finish(yaw, "odom_yaw_disagreement")
        threshold = self.config.exit_speed_mps if self._using_motion else self.config.enter_speed_mps
        if speed < threshold or recent_speed < threshold:
            return finish(yaw, "odom_yaw_below_motion_threshold")
        if self.config.mode == "motion_guarded":
            return finish(recent_heading, "guarded_recent_motion", motion=True)
        return finish(motion_heading, "adaptive_motion_window", motion=True)
