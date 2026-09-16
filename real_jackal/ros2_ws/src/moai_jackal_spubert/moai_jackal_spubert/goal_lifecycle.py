"""Goal completion latch with a process-unique identity for delayed ROS messages."""
from __future__ import annotations

from uuid import uuid4


class GoalLifecycle:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid4().hex
        self.generation = 0
        self.goal_id = ""
        self.completed = False

    def start(self) -> str:
        self.generation += 1
        self.goal_id = f"{self.session_id}:{self.generation}"
        self.completed = False
        return self.goal_id

    def complete(self, goal_id: str) -> bool:
        if not self.goal_id or goal_id != self.goal_id:
            return False
        self.completed = True
        return True
