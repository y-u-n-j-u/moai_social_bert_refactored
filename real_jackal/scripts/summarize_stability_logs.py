#!/usr/bin/env python3
"""Summarize bridge/tracker JSONL without ROS, numpy or model dependencies.

Example: python3 summarize_stability_logs.py --bridge bridge.jsonl \
    --tracker tracker.jsonl --output report.json

Exit 0: no detected format/required-field/time-order errors; exit 2: report has
data-quality errors. Statistics are descriptive, not evidence of better model
performance, physical wheel motion, or safety. Angular values use radians.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

MISSING = object()


class Quality:
    def __init__(self):
        self.counts = Counter()
        self.examples = []

    def issue(self, line, kind, field):
        self.counts[f"{kind}:{field}"] += 1
        if len(self.examples) < 10:
            self.examples.append({"line": line, "kind": kind, "field": field})

    def report(self):
        return {"counts": dict(sorted(self.counts.items())),
                "first_errors": self.examples, "error_count": sum(self.counts.values())}


def read_rows(path, quality):
    rows, blank = [], 0
    with Path(path).open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.strip():
                blank += 1
                continue
            try:
                def reject_constant(value):
                    raise ValueError(f"Non-JSON numeric constant {value}")
                record = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
                if not isinstance(record, dict):
                    raise ValueError("JSONL record is not an object")
            except (ValueError, UnicodeError):
                quality.issue(line, "malformed", "json_object")
                record = None
            rows.append((line, record))
    return rows, blank


def field(record, key, line, quality):
    value = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            quality.issue(line, "missing", key)
            return MISSING
        value = value[part]
    return value


def number(record, key, line, quality):
    value = field(record, key, line, quality)
    if value is MISSING:
        return None
    if value is None:
        quality.issue(line, "null_or_invalid", key)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        quality.issue(line, "null_or_invalid", key)
        return None
    try:
        converted = float(value)
    except OverflowError:
        converted = math.inf
    if not math.isfinite(converted):
        quality.issue(line, "null_or_invalid", key)
        return None
    return value if isinstance(value, int) else converted


def timestamps(rows, key, scale, quality):
    times, previous, reversals, duplicates = [], None, 0, 0
    for line, record in rows:
        time = None if record is None else number(record, key, line, quality)
        if time is not None and time < 0:
            quality.issue(line, "null_or_invalid", key)
            time = None
        if time is not None:
            if previous is not None:
                if time < previous:
                    quality.issue(line, "time_reversal", key)
                    reversals += 1
                elif time == previous:
                    duplicates += 1
            previous = time
        # Check integer nanoseconds before converting to seconds: float(epoch_ns)
        # can hide a small backwards jump at real ROS epoch timestamps.
        times.append(None if time is None else time * scale)
    return times, reversals, duplicates


def wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def magnitude_stats(values):
    values = [abs(v) for v in values]
    return {"samples": len(values), "mean_abs_rad": sum(values) / len(values) if values else None,
            "max_abs_rad": max(values) if values else None}


def goal_key(record, line, quality):
    value = field(record, "goal_id", line, quality)
    if value is MISSING or value is None:
        return None
    if not isinstance(value, str) or not value:
        quality.issue(line, "null_or_invalid", "goal_id")
        return None
    return value


def timing_stats(values):
    ordered = sorted(values)
    size = len(ordered)
    return {"samples": size, "mean_sec": sum(ordered) / size if size else None,
            "min_sec": ordered[0] if size else None,
            "p50_sec": ordered[max(0, math.ceil(size * 0.50) - 1)] if size else None,
            "p95_sec": ordered[max(0, math.ceil(size * 0.95) - 1)] if size else None,
            "max_sec": ordered[-1] if size else None}


def summarize_phase_timing(rows, quality):
    """Optional fields keep older logs valid; unavailable measurements stay absent."""
    phases, ages, previous_io = defaultdict(list), defaultdict(list), defaultdict(list)

    def collect(value, destination, prefix, line, *, signed=False):
        if not isinstance(value, dict):
            quality.issue(line, "null_or_invalid", prefix)
            return
        for key, seconds in value.items():
            if not key.endswith("_sec") or seconds is None:
                continue
            try:
                finite = math.isfinite(float(seconds))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not finite or (not signed and seconds < 0)):
                quality.issue(line, "null_or_invalid", f"{prefix}.{key}")
                continue
            destination[key].append(seconds)

    for line, record in rows:
        if not isinstance(record, dict):
            continue
        if "phase_timing_sec" in record:
            collect(record["phase_timing_sec"], phases, "phase_timing_sec", line)
        if "previous_diagnostic_io_sec" in record:
            collect(record["previous_diagnostic_io_sec"], previous_io, "previous_diagnostic_io_sec", line)
        if "input_ages" in record:
            snapshots = record["input_ages"]
            if not isinstance(snapshots, dict):
                quality.issue(line, "null_or_invalid", "input_ages")
                continue
            for stage, snapshot in snapshots.items():
                stage_values = defaultdict(list)
                collect(snapshot, stage_values, f"input_ages.{stage}", line, signed=True)
                for key, values in stage_values.items():
                    ages[f"{stage}.{key}"].extend(values)
    return {
        "phases_sec": {key: timing_stats(values) for key, values in sorted(phases.items())},
        "input_ages_sec": {key: timing_stats(values) for key, values in sorted(ages.items())},
        "previous_record_io_sec": {key: timing_stats(values) for key, values in sorted(previous_io.items())},
        "interpretation": (
            "Samples are available log records, not control cycles or unique jobs; "
            "one job can have multiple records. Total phases overlap subphases. "
            "Percentiles use nearest rank. Header ages use the reported ROS headers, "
            "not measured upstream state-estimator latency; negative ages mean future headers. "
            "Previous-record I/O excludes the final record's unreported write."
        ),
    }


def summarize_bridge(rows, quality):
    times, reversals, duplicates = timestamps(rows, "stamp_ns", 1e-9, quality)
    counts, reasons, candidate_counts, candidate_reasons = Counter(), Counter(), Counter(), Counter()
    differences, steps = [], []
    previous = None
    events = Counter()
    for (line, record), time in zip(rows, times):
        if record is None:
            previous = None
            continue
        event = field(record, "event", line, quality)
        if not isinstance(event, str):
            if event is not MISSING:
                quality.issue(line, "null_or_invalid", "event")
            previous = None
            continue
        events[event] += 1
        if time is None:
            previous = None
        if event != "prediction":
            # inference_error/preparation_rejected contain no comparable model
            # heading. Unknown events also cannot establish continuity; do not
            # join the predictions on either side of an unobserved interval.
            previous = None
            continue
        counts["total"] += 1
        valid = field(record, "valid", line, quality)
        if type(valid) is bool:
            counts["valid" if valid else "invalid"] += 1
        else:
            if valid is not MISSING:
                quality.issue(line, "null_or_invalid", "valid")
            counts["validity_unknown"] += 1
        if valid is False:
            reason = field(record, "reason", line, quality)
            if isinstance(reason, str) and reason:
                reasons[reason] += 1
                if reason == "no_candidates":
                    counts["no_candidates"] += 1
            else:
                if reason is not MISSING:
                    quality.issue(line, "null_or_invalid", "reason")
        attempts = field(record, "attempts", line, quality)
        if isinstance(attempts, list):
            if not attempts:
                counts["empty_candidate_lists"] += 1
            all_invalid = bool(attempts)
            for attempt in attempts:
                if not isinstance(attempt, dict) or type(attempt.get("valid")) is not bool:
                    quality.issue(line, "missing_or_invalid", "attempts[].valid")
                    candidate_counts["validity_unknown"] += 1
                    all_invalid = False
                    continue
                candidate_counts["valid" if attempt["valid"] else "invalid"] += 1
                all_invalid = all_invalid and not attempt["valid"]
                if not attempt["valid"]:
                    reason = attempt.get("reason")
                    if isinstance(reason, str) and reason:
                        candidate_reasons[reason] += 1
                    else:
                        quality.issue(line, "missing_or_invalid", "attempts[].reason")
            if all_invalid and valid is False:
                counts["all_candidates_invalid"] += 1
        else:
            if attempts is not MISSING:
                quality.issue(line, "null_or_invalid", "attempts")
        # Pending/expired/cancelled jobs can be rejected without collecting a
        # model result. Missing angles in these explicit records are expected;
        # do not invent a heading or connect comparisons across the gap.
        if record.get("model_output_collected") is False:
            counts["model_output_not_collected"] += 1
            if valid is True:
                quality.issue(line, "inconsistent", "model_output_collected")
            previous = None
            continue
        theta = number(record, "model_input.theta_rad", line, quality)
        legacy = number(record, "model_input.legacy_heading_rad", line, quality)
        goal = goal_key(record, line, quality)
        if theta is not None and legacy is not None:
            differences.append(wrapped(wrapped(theta) - wrapped(legacy)))
        if theta is None or time is None or goal is None:
            previous = None
        else:
            if previous is not None and previous[0] == goal:
                steps.append(wrapped(wrapped(theta) - wrapped(previous[1])))
            previous = (goal, theta)
    names = ("total", "valid", "invalid", "validity_unknown", "all_candidates_invalid", "no_candidates", "empty_candidate_lists", "model_output_not_collected")
    return {
        "events": dict(sorted(events.items())),
        "predictions": {name: counts[name] for name in names},
        "invalid_prediction_reasons": dict(sorted(reasons.items())),
        "candidates": dict(candidate_counts), "invalid_candidate_reasons": dict(sorted(candidate_reasons.items())),
        "actual_vs_legacy_heading": magnitude_stats(differences),
        "consecutive_actual_heading_change": None if reversals else magnitude_stats(steps),
        "time_order_valid": not reversals, "duplicate_timestamps": duplicates,
        "timing": summarize_phase_timing(rows, quality),
    }


class SignChanges:
    def __init__(self, epsilon):
        self.epsilon, self.previous, self.flips, self.samples = epsilon, None, 0, 0
        self.comparisons = 0

    def reset(self):
        self.previous = None

    def add(self, value):
        if value is None:
            self.reset()
            return
        self.samples += 1
        sign = 1 if value > self.epsilon else -1 if value < -self.epsilon else 0
        if sign:
            if self.previous is not None:
                self.comparisons += 1
                if sign != self.previous:
                    self.flips += 1
            self.previous = sign

    def report(self):
        return {"available_samples": self.samples, "nonzero_comparisons": self.comparisons,
                "sign_reversals": self.flips if self.comparisons else None}


def summarize_tracker(rows, quality, epsilon):
    times, reversals, duplicates = timestamps(rows, "time_sec", 1.0, quality)
    states, durations = Counter(), defaultdict(float)
    requested, applied = SignChanges(epsilon), SignChanges(epsilon)
    previous, previous_goal = None, object()
    interval_count, unattributed, max_interval = 0, 0.0, None
    for (line, record), time in zip(rows, times):
        if record is None:
            previous = None
            requested.reset(); applied.reset()
            continue
        event = field(record, "event", line, quality)
        state = None
        if event == "tracking":
            state = "tracking"
        elif event == "stop":
            reason = field(record, "stop_reason", line, quality)
            if isinstance(reason, str) and reason:
                state = f"stop:{reason}"
            else:
                if reason is not MISSING:
                    quality.issue(line, "null_or_invalid", "stop_reason")
        else:
            if event is not MISSING:
                quality.issue(line, "null_or_invalid", "event")
        if state is not None:
            states[state] += 1
        goal = goal_key(record, line, quality)
        if time is None or goal is None or goal != previous_goal:
            requested.reset(); applied.reset()
        req = number(record, "requested_angular", line, quality)
        app = number(record, "applied_angular", line, quality)
        if time is not None:
            requested.add(req); applied.add(app)
        previous_goal = goal
        if previous is not None and time is not None and time >= previous[0]:
            dt = time - previous[0]
            interval_count += 1
            max_interval = dt if max_interval is None else max(max_interval, dt)
            if previous[1] is None:
                unattributed += dt
            else:
                durations[previous[1]] += dt
        previous = (time, state) if time is not None else None
    intervals = {
        "interval_count": interval_count,
        "seconds_by_state": dict(sorted(durations.items())),
        "unattributed_seconds": unattributed if interval_count else None,
        "covered_seconds": sum(durations.values()) + unattributed if interval_count else None,
        "maximum_sample_interval_seconds": max_interval,
        "last_interval_unknown": True,
    }
    return {
        "state_samples": dict(sorted(states.items())),
        "time_order_valid": not reversals, "duplicate_timestamps": duplicates,
        "estimated_state_intervals": None if reversals else intervals,
        "requested_angular": None if reversals else requested.report(),
        "applied_angular": None if reversals else applied.report(),
    }


def summarize(path, kind, epsilon):
    quality = Quality()
    rows, blank = read_rows(path, quality)
    summary = summarize_bridge(rows, quality) if kind == "bridge" else summarize_tracker(rows, quality, epsilon)
    return {"file": str(Path(path)), "records": sum(r is not None for _, r in rows),
            "blank_lines_skipped": blank, **summary, "data_quality": quality.report()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--tracker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--angular-zero-epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if not args.bridge and not args.tracker:
        parser.error("Provide --bridge and/or --tracker")
    if not math.isfinite(args.angular_zero_epsilon) or args.angular_zero_epsilon < 0:
        parser.error("--angular-zero-epsilon must be finite and nonnegative")
    if args.output and any(p and p.resolve() == args.output.resolve() for p in (args.bridge, args.tracker)):
        parser.error("Output must not overwrite an input log")
    report = {
        "format": 1,
        "definitions": {
            "heading": "Wrapped differences in [-pi, pi]; consecutive predictions within the same known goal; missing rows/angles/times or unknown goals break continuity.",
            "angular_reversal": "Nonzero sign changes; zeros ignored, including a zero crossing. Missing rows/values/times, unknown goals and goal changes reset the comparison. No comparable pairs yields null.",
            "angular_zero_epsilon_rad_s": args.angular_zero_epsilon,
            "state_time": "Each sample's state is held until the next adjacent valid timestamp; no interpolation across malformed/missing timestamps; no last-interval extrapolation.",
            "time_reversal": "Reject all temporal/sequence metrics for that log; do not sort or repair timestamps.",
            "limits": "Command/state logs do not measure physical wheel stop or motion, prove model improvement, or establish safety. Compare like-for-like scenarios and coverage.",
        },
        "bridge": summarize(args.bridge, "bridge", args.angular_zero_epsilon) if args.bridge else None,
        "tracker": summarize(args.tracker, "tracker", args.angular_zero_epsilon) if args.tracker else None,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 2 if any(report[k] and report[k]["data_quality"]["error_count"] for k in ("bridge", "tracker")) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as exc:
        print(f"Log summary failed: {exc}", file=sys.stderr)
        sys.exit(1)
