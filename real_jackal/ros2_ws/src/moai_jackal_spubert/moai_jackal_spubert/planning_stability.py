"""Rank freshly validated paths; this module never reuses a rejected path."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .navigation_core import lookahead_point, normalize_angle

XY = Tuple[float, float]


@dataclass(frozen=True)
class SelectionConfig:
    mode: str = "continuous"
    lookahead_m: float = 0.70
    guidance_weight: float = 0.20
    heading_weight: float = 0.50
    turning_weight: float = 0.15
    continuity_weight: float = 2.00

    def __post_init__(self):
        if self.mode not in {"continuous", "first_valid"}:
            raise ValueError("candidate_selection_mode must be continuous or first_valid")
        for name in ("lookahead_m", "guidance_weight", "heading_weight", "turning_weight", "continuity_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lookahead_m <= 0:
            raise ValueError("lookahead_m must be positive")


@dataclass(frozen=True)
class PathOption:
    index: int
    path: Sequence[XY]
    guidance_distance_m: float
    valid: bool


@dataclass(frozen=True)
class PathScore:
    index: int
    total: float
    heading_error_rad: float
    turning_rad: float
    continuity_distance_m: float
    guidance_distance_m: float


def rank_valid_paths(
    options: Sequence[PathOption],
    *,
    current: XY,
    robot_yaw: float,
    previous_path: Optional[Sequence[XY]] = None,
    config: SelectionConfig = SelectionConfig(),
) -> list[PathScore]:
    """Safety is a prerequisite, not a weighted cost that another term can offset.

    Compare short forward portions at equal arc lengths. The previous path is
    only a scoring reference, never an executable fallback or an averaged path.
    All returned indices refer to the supplied, independently validated options.
    """
    scores = []
    if not all(math.isfinite(v) for v in (*current, robot_yaw)):
        return scores
    distances = [config.lookahead_m * scale for scale in (0.5, 1.0, 1.5)]
    previous_samples = (
        [lookahead_point(previous_path, current, distance) for distance in distances]
        if previous_path else []
    )
    for option in options:
        if not option.valid or len(option.path) < 2:
            continue
        if not all(math.isfinite(v) for point in option.path for v in point):
            continue
        if not math.isfinite(option.guidance_distance_m) or option.guidance_distance_m < 0:
            continue
        # Include the current-to-first-future-point segment that was safety checked.
        points = [current, *option.path]
        samples = [lookahead_point(points, current, distance) for distance in distances]
        target = samples[1]
        heading_error = normalize_angle(math.atan2(target[1] - current[1], target[0] - current[0]) - robot_yaw)
        headings = []
        for start, end in zip([current, *samples[:-1]], samples):
            if math.dist(start, end) > 1e-6:
                headings.append(math.atan2(end[1] - start[1], end[0] - start[0]))
        turning = sum(abs(normalize_angle(b - a)) for a, b in zip(headings, headings[1:]))
        continuity = (
            sum(math.dist(a, b) for a, b in zip(samples, previous_samples)) / len(samples)
            if previous_samples else 0.0
        )
        total = (
            config.guidance_weight * option.guidance_distance_m
            + config.heading_weight * abs(heading_error)
            + config.turning_weight * turning
            + config.continuity_weight * continuity
        )
        scores.append(PathScore(option.index, total, heading_error, turning, continuity, option.guidance_distance_m))
    if config.mode == "continuous":
        scores.sort(key=lambda item: (item.total, item.index))
    return scores
