"""Best-effort diagnostics; none of these values participate in motion checks.

Durations use a monotonic clock. Message/history ages use the node's ROS clock
and the supplied headers; they do not measure upstream state-estimator latency.
"""
import math
import time


def phase_now():
    try:
        value = float(time.perf_counter())
        return value if math.isfinite(value) else None
    except Exception:
        return None


def record_duration(timings, name, started, finished=None):
    """Keep missing/broken timers missing, rather than inventing zero latency."""
    try:
        finished = phase_now() if finished is None else finished
        elapsed = float(finished) - float(started)
        if math.isfinite(elapsed) and elapsed >= 0:
            timings[name] = elapsed
    except Exception:
        pass


def _difference(now, then):
    try:
        difference = float(now) - float(then)
        return difference if math.isfinite(difference) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _stamp_ns(stamp):
    try:
        sec, nanosec = stamp.sec, stamp.nanosec
        if (type(sec) is not int or type(nanosec) is not int
                or sec < 0 or not 0 <= nanosec < 1_000_000_000):
            return None
        value = sec * 1_000_000_000 + nanosec
        return value if value > 0 else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def input_age_snapshot(*, now_ns, stamps, receipt_times_s, history_times_s, history_odom_stamps_ns):
    """Signed ages distinguish future headers from delayed ones.

    Integer subtraction precedes conversion for nanosecond header precision.
    A history sample's collection age and its odometry header age are separate.
    """
    result = {"ros_now_ns": int(now_ns)}
    now_s = int(now_ns) * 1e-9
    for name, stamp in stamps.items():
        source_ns = _stamp_ns(stamp)
        result[f"{name}_header_age_sec"] = None if source_ns is None else (int(now_ns) - source_ns) * 1e-9
        result[f"{name}_receipt_age_sec"] = _difference(now_s, receipt_times_s.get(name))
    samples, source_stamps = list(history_times_s), list(history_odom_stamps_ns)
    result["history_latest_sample_age_sec"] = _difference(now_s, samples[-1]) if samples else None
    result["history_sample_span_sec"] = _difference(samples[-1], samples[0]) if samples else None
    source_ns = source_stamps[-1] if source_stamps else None
    source_valid = type(source_ns) is int and source_ns > 0
    result["history_latest_odom_header_age_sec"] = (int(now_ns) - source_ns) * 1e-9 if source_valid else None
    result["history_latest_sample_minus_odom_header_sec"] = (
        _difference(samples[-1], source_ns * 1e-9) if samples and source_valid else None
    )
    return result
