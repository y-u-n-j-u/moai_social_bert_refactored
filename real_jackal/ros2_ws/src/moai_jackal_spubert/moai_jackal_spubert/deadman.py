"""Receive-time joystick gate. Releasing or losing the signal requires rearming."""
import math


class Deadman:
    def __init__(self, button: int, timeout_s: float):
        if button < 0 or not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("Configure a nonnegative deadman button and positive timeout")
        self.button = button
        self.timeout_s = timeout_s
        self.last_s = -math.inf
        self.pressed = False

    def observe(self, buttons, now_s: float) -> None:
        self.last_s = now_s
        self.pressed = self.button < len(buttons) and buttons[self.button] == 1

    def ready(self, now_s: float) -> bool:
        return self.pressed and 0.0 <= now_s - self.last_s <= self.timeout_s
