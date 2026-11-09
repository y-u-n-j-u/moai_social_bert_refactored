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
    def __init__(self, enabled: bool = False, reach_radius_m: float = 0.5, max_hold_s: float = 20.0,
                 same_cycle_s: float = 0.4) -> None:
        self.enabled = bool(enabled)
        self.same_cycle_s = float(same_cycle_s)
        self._decided_at_s: Optional[float] = None
        self._decision = True
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

    def decide(self, now_s: float, robot_xy: Sequence[float], goal_generation: int, mgp_xy: Optional[Sequence[float]]) -> bool:
        """One decision per planning cycle, shared by every display that is drawn from the same candidates.

        The first caller of a cycle decides (and, when the set is replaced, records its goal sample); the others within
        ``same_cycle_s`` get the same answer, so the candidate lines, the goal-sample spheres, the guidance sphere and the selected
        path change together.
        """
        if self._decided_at_s is not None and 0.0 <= now_s - self._decided_at_s < self.same_cycle_s:
            return self._decision
        refresh = self.should_refresh(now_s, robot_xy, goal_generation)
        if refresh:
            self.note_shown(now_s, goal_generation, mgp_xy)
        self._decided_at_s = float(now_s)
        self._decision = refresh
        return refresh

    def note_shown(self, now_s: float, goal_generation: int, mgp_xy: Optional[Sequence[float]]) -> None:
        self._shown_at_s = float(now_s)
        self._shown_generation = goal_generation
        self._shown_mgp = None if mgp_xy is None else (float(mgp_xy[0]), float(mgp_xy[1]))


def display_mgp(candidates: Sequence[dict], chosen) -> Optional[XY]:
    """The goal sample (MGP) the robot has to reach before the shown set is replaced.

    The goal of the selected candidate; when no candidate was selected (every candidate failed) the goal of the first candidate that
    has one, so that such a set is held too instead of being redrawn at every replan.
    """
    first = None
    for row in candidates:
        goal = row.get("goal")
        if goal is None or len(goal) < 2 or not all(math.isfinite(float(v)) for v in goal[:2]):
            continue
        point = (float(goal[0]), float(goal[1]))
        if row.get("valid") and chosen is not None and row.get("candidate_index") == chosen:
            return point
        if first is None:
            first = point
    return first
