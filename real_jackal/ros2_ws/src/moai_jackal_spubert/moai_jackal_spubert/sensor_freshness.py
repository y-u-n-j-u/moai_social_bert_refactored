"""Check ROS source timestamps as well as callback receipt times."""
import math


def stamp_error(stamp, *, now_ns: int, timeout_sec: float, future_tolerance_sec: float = 0.1) -> str:
    """Return an empty string or missing/invalid/stale/in_future.

    Sources and nodes must share the same ROS clock. A zero stamp is unspecified
    in this real-robot deployment and must not be refreshed by receiving it again.
    """
    if stamp is None:
        return "missing"
    try:
        sec, nanosec = stamp.sec, stamp.nanosec
        if type(sec) is not int or type(nanosec) is not int or sec < 0 or not 0 <= nanosec < 1_000_000_000:
            return "invalid"
        stamp_ns = sec * 1_000_000_000 + nanosec
        if stamp_ns == 0:
            return "missing"
        if not math.isfinite(timeout_sec) or timeout_sec <= 0 or not math.isfinite(future_tolerance_sec) or future_tolerance_sec < 0:
            return "invalid"
        age = (int(now_ns) - stamp_ns) * 1e-9
        if age > timeout_sec:
            return "stale"
        if age < -future_tolerance_sec:
            return "in_future"
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "invalid"
    return ""
