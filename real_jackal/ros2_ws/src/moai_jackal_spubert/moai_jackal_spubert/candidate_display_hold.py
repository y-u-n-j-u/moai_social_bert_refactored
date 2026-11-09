"""Decide when the RViz candidate display may change (display only, no influence on the planner).

The bridge replans about once a second, so the candidate paths drawn in RViz changed every second. With the hold on, the set that is
drawn stays until the robot has reached the goal sample (MGP) of the candidate that was selected from it, the goal changed, the hold ran
out (the robot did not get there), or the shown set had no selected candidate (nothing to wait for).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

XY = Tuple[float, float]


class CandidateDisplayHold:
    def __init__(self, enabled: bool = False, reach_radius_m: float = 0.5, max_hold_s: float = 20.0) -> None:
        self.enabled = bool(enabled)
        self.reach_radius_m = float(reach_radius_m)
        self.max_hold_s = float(max_hold_s)
        self._shown_at_s: Optional[float] = None
        self._shown_generation: Optional[int] = None
        self._shown_mgp: Optional[XY] = None

    def should_refresh(self, now_s: float, robot_xy: Sequence[float], goal_generation: int) -> bool:
        if not self.enabled or self._shown_at_s is None:
            return True
        if goal_generation != self._shown_generation:
            return True
        if self._shown_mgp is None:
            return True
        if now_s - self._shown_at_s > self.max_hold_s:
            return True
        return math.hypot(robot_xy[0] - self._shown_mgp[0], robot_xy[1] - self._shown_mgp[1]) <= self.reach_radius_m

    def note_shown(self, now_s: float, goal_generation: int, mgp_xy: Optional[Sequence[float]]) -> None:
        self._shown_at_s = float(now_s)
        self._shown_generation = goal_generation
        self._shown_mgp = None if mgp_xy is None else (float(mgp_xy[0]), float(mgp_xy[1]))
