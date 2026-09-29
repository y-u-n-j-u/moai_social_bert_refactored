"""ROS-independent diagnostic helpers; never participate in motion decisions."""
from __future__ import annotations

import json
import math
from typing import Optional


def diagnostic_json(record: dict) -> str:
    """Emit standards-compliant JSON (missing/nonfinite measurements are null)."""
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        return value

    return json.dumps(clean(record), allow_nan=False, separators=(",", ":"))


class StopEpisode:
    """Keep the initiating stop reason even if a follow-up status masks it."""
    def __init__(self) -> None:
        self.first_reason: Optional[str] = None
        self.since_s: Optional[float] = None
        self.last_completed: Optional[dict] = None

    def finish(self, now_s: float) -> None:
        if self.first_reason is not None:
            self.last_completed = {
                "first_reason": self.first_reason,
                "started_at_s": self.since_s,
                "duration_s": max(0.0, now_s - self.since_s),
            }
        self.first_reason = None
        self.since_s = None

    def observe(self, reason: str, now_s: float, stopped: bool) -> dict:
        # A reset simulation clock must not produce a negative duration.
        if self.since_s is not None and now_s < self.since_s:
            self.since_s = now_s
        if not stopped:
            self.finish(now_s)
        elif self.first_reason is None:
            self.first_reason = reason
            self.since_s = now_s
        return {
            "first_reason": self.first_reason,
            "started_at_s": self.since_s,
            "duration_s": None if self.since_s is None else max(0.0, now_s - self.since_s),
            "last_completed": self.last_completed,
        }
