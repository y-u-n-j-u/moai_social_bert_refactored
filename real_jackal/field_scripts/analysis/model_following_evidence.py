#!/usr/bin/env python3
"""Read-only model-following audit of a ROS 2 bag or normalized JSONL.

The output distinguishes exact path provenance from temporal command/odometry
corroboration. Matching command values do not establish publisher identity.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
from statistics import median


NS = 1_000_000_000
PATH = "/spu_bert/predicted_path"
DECISION = "/spu_bert/selection_evidence"
TRACKER = "/spu_bert/tracker_diagnostics"
ODOM = "/aft_mapped_to_init"
UPSTREAM = "/j100_0519/cmd_vel"
DOWNSTREAM = "/j100_0519/platform/cmd_vel_unstamped"
DEMO_MARKER = "/__synthetic_fixture"


def stamp_ns(header):
    return header.stamp.sec * NS + header.stamp.nanosec


def load_bag(path, kinds, storage_id):
    # Import ROS only when reading a bag. The audit also runs on Windows JSONL.
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    if storage_id is None:
        storage_id = "mcap" if path.suffix == ".mcap" or (
            path.is_dir() and any(path.glob("*.mcap"))) else "sqlite3"
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(path), storage_id=storage_id), ConverterOptions("cdr", "cdr"))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in kinds}
    if types:
        reader.set_filter(StorageFilter(topics=list(types)))
    while types and reader.has_next():
        topic, raw, t = reader.read_next()
        if topic not in types:
            continue
        m = deserialize_message(raw, types[topic])
        kind = kinds[topic]
        if kind in ("decision", "tracker"):
            try:
                data = json.loads(m.data)
            except (TypeError, ValueError):
                # Keep the bad sample visible as unverified evidence.
                data = {}
            if not isinstance(data, dict):
                data = {}
        elif kind == "path":
            data = {"stamp_ns": stamp_ns(m.header), "frame_id": m.header.frame_id,
                    "points": [[p.pose.position.x, p.pose.position.y] for p in m.poses]}
        elif kind == "odom":
            p = m.pose.pose.position
            data = {"stamp_ns": stamp_ns(m.header), "frame_id": m.header.frame_id, "x": p.x, "y": p.y}
        elif kind == "joy":
            data = {"axes": list(m.axes), "buttons": list(m.buttons)}
        else:
            twist = getattr(m, "twist", m)
            data = {"v": twist.linear.x, "w": twist.angular.z}
        yield {"topic": topic, "bag_ns": t, "message": data}


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if (not isinstance(data, dict) or not isinstance(data.get("topic"), str)
                    or type(data.get("bag_ns")) is not int or not isinstance(data.get("message"), dict)):
                raise ValueError(f"{path}:{line_no}: expected topic, integer bag_ns, object message")
            yield data


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    return value


class Series:
    def __init__(self, samples):
        self.items = sorted(samples, key=lambda e: e[0])
        self.times = [t for t, _ in self.items]

    def nearest(self, t, limit_ns):
        i = bisect_left(self.times, t)
        indices = range(max(i - 1, 0), min(i + 1, len(self.times)))
        best = min(indices, default=None, key=lambda j: abs(self.times[j] - t))
        if best is None or abs(self.times[best] - t) > limit_ns:
            return None
        stamp = self.times[best]
        left = bisect_left(self.times, stamp)
        right = bisect_right(self.times, stamp)
        sample = self.items[best]
        if any(item[1] != sample[1] for item in self.items[left:right]):
            return None  # different samples with indistinguishable receipt times
        return sample

    def within(self, t, limit_ns):
        return self.items[bisect_left(self.times, t - limit_ns):bisect_right(self.times, t + limit_ns)]


def path_key(message):
    stamp, frame = message.get("stamp_ns"), message.get("frame_id")
    if type(stamp) is int and stamp > 0 and isinstance(frame, str) and frame:
        return stamp, frame
    return None


def geometry(data):
    if not isinstance(data, (list, tuple)) or not data:
        return ()
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(map(finite, p)) for p in data):
        return ()
    return tuple(tuple(p) for p in data)


def geometry_matches(a, b):
    # Path and evidence both serialize these same double precision XY values.
    return bool(a) and len(a) == len(b) and all(math.dist(p, q) <= 1e-6 for p, q in zip(a, b))


def command(data):
    if isinstance(data, dict) and all(finite(data.get(k)) for k in ("v", "w")):
        return data["v"], data["w"]
    return None


def nearest_path_point(point, points):
    if not points:
        return None, None
    best = math.dist(point, points[0])
    along = 0.0
    arc = 0.0
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 == 0:
            continue
        u = max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / length2))
        distance = math.dist(point, (ax + u * dx, ay + u * dy))
        length = math.sqrt(length2)
        if distance < best:
            best = distance
            along = arc + u * length
        arc += length
    return best, along


def candidates_from(decision):
    candidates = decision.get("candidates")
    return [p for p in candidates if isinstance(p, dict)] if isinstance(candidates, list) else []


def audit(events, kinds, *, max_skew_ns=150_000_000, command_tolerance=0.005):
    indexed = defaultdict(list)
    lookup = {"decision": defaultdict(list), "path": defaultdict(list)}
    synthetic = False
    for e in events:
        synthetic |= e["topic"] == DEMO_MARKER
        kind = kinds.get(e["topic"])
        if not kind:
            continue
        t, m = e["bag_ns"], e["message"]
        indexed[kind].append((t, m))
        if kind in lookup and (key := path_key(m)) is not None:
            lookup[kind][key].append((t, m))
    series = {k: Series(indexed[k]) for k in ("upstream", "downstream", "tracker", "odom")}
    manual = Series(indexed["manual"])
    # Account for every recorded downstream command, including any overrides.
    basis = next((k for k in ("downstream", "upstream", "tracker") if series[k].items), None)
    clock = series[basis].items if basis else []
    t0 = clock[0][0] if clock else 0
    upstream_topic = next((tp for tp, kind in kinds.items() if kind == "upstream"), None)
    rows = []
    snapshots = []
    for t, m in clock:
        samples = {k: ((t, m) if k == basis else s.nearest(t, max_skew_ns)) for k, s in series.items()}
        upstream = command(samples["upstream"][1]) if samples["upstream"] else None
        actual = command(samples["downstream"][1]) if samples["downstream"] else None
        track_t, diag = samples["tracker"] or (None, {})
        expected = command(diag.get("published_command"))
        motion_command = actual if basis == "downstream" else upstream if basis == "upstream" else expected
        moving = motion_command is not None and any(abs(x) > 0.01 for x in motion_command)
        control = diag.get("control")
        control = control if isinstance(control, dict) else {}
        path_info = diag.get("path")
        path_info = path_info if isinstance(path_info, dict) else {}
        key = path_key(path_info)
        matched = {}
        for k in lookup:
            options = lookup[k].get(key, [])
            if len(options) == 1 and options[0][0] <= t + max_skew_ns:
                matched[k] = options[0][1]
        path = geometry(matched.get("path", {}).get("points"))
        decision = matched.get("decision", {})
        executed = geometry(decision.get("executed_path"))
        source = decision.get("executed_source")
        selected_id = decision.get("executed_candidate_index")
        selected = [c for c in candidates_from(decision) if type(selected_id) is int
                    and c.get("candidate_index") == selected_id]
        steer = control.get("steering_source")
        odom = samples["odom"][1] if samples["odom"] else {}
        odom_xy = (odom.get("x"), odom.get("y"))
        diag_robot = diag.get("robot")
        frame_ok = (key is not None and isinstance(diag_robot, dict)
                    and key[1] == diag_robot.get("frame_id") == odom.get("frame_id"))
        xy_ok = all(map(finite, odom_xy))
        error, progress = nearest_path_point(odom_xy, path) if frame_ok and xy_ok else (None, None)
        command_agrees = all(a is not None and b is not None and max(abs(x-y) for x, y in zip(a,b)) <= command_tolerance
                             for a, b in ((expected, upstream), (upstream, actual)))
        manual_active = any(any(abs(x) > 0.01 for x in c)
                            for _, m in manual.within(t, max_skew_ns) if (c := command(m)) is not None)

        if motion_command is None:
            label = "missing_command"
        elif not moving:
            label = "zero_command"
        elif manual_active:
            label = "manual_command_nearby"
        elif not diag or diag.get("event") != "tracking" or diag.get("motion_enabled") is not True:
            label = "unverified_tracker"
        elif not path or not decision or not path_info.get("fresh"):
            label = "unverified_path"
        elif source in ("retained", "fallback", "recovery", "hold"):
            label = f"source_{source}"
        elif source != "model" or len(selected) != 1 or selected[0].get("valid") is not True:
            label = "unverified_candidate"
        elif not geometry_matches(path, executed) or not geometry_matches(path, geometry(selected[0].get("path"))):
            label = "candidate_path_mismatch"
        elif steer in ("gap", "avoidance"):
            label = f"steering_{steer}"
        elif (steer != "path" or control.get("mode") != "model_evidence"
              or decision.get("model_evidence_only") is not True):
            label = "unverified_control_mode"
        elif control.get("deadman_configured") is not True or control.get("deadman_pressed") is not True:
            label = "unverified_deadman"
        elif diag.get("cmd_vel_topic") != upstream_topic:
            label = "unverified_command_topic"
        elif actual is None or upstream is None or expected is None:
            label = "unverified_command_chain"
        elif not command_agrees:
            label = "command_mismatch"
        elif error is None:
            label = "unverified_odom"
        else:
            label = "model_chain_consistent"
        row = {"bag_ns": t, "time_s": (t-t0)/NS, "category": label, "moving_command": moving,
               "path_stamp_ns": key[0] if key else None, "frame_id": key[1] if key else None,
               "executed_source": source, "executed_candidate_index": selected_id, "steering_source": steer,
               "decision_reason": decision.get("decision_reason"),
               "selection_reason": decision.get("selection_reason"),
               "control_mode": control.get("mode"), "manual_command_nearby": manual_active, "tracker_bag_ns": track_t,
               "upstream_bag_ns": samples["upstream"][0] if samples["upstream"] else None,
               "downstream_bag_ns": samples["downstream"][0] if samples["downstream"] else None,
               "odom_bag_ns": samples["odom"][0] if samples["odom"] else None,
               "odom_x": odom_xy[0] if xy_ok else None, "odom_y": odom_xy[1] if xy_ok else None,
               "cross_track_m": error, "projected_path_progress_m": progress,
               "same_path_odom_step_m": None, "same_path_forward_progress_m": None}
        for name, pair in (("upstream", upstream), ("downstream", actual), ("tracker", expected)):
            row[f"{name}_v"], row[f"{name}_w"] = pair or (None, None)
        rows.append(row)
        snapshots.append((decision, path))

    # Only consecutive corroborated ticks with independent odom observations
    # on the same path contribute to motion/progress (no interpolation over gaps).
    for previous, current in zip(rows, rows[1:]):
        if (previous["category"] == current["category"] == "model_chain_consistent"
                and previous["path_stamp_ns"] == current["path_stamp_ns"]
                and previous["frame_id"] == current["frame_id"]
                and 0 < current["odom_bag_ns"] - previous["odom_bag_ns"] <= 2 * max_skew_ns):
            current["same_path_odom_step_m"] = math.dist(
                (previous["odom_x"], previous["odom_y"]), (current["odom_x"], current["odom_y"]))
            current["same_path_forward_progress_m"] = (
                current["projected_path_progress_m"] - previous["projected_path_progress_m"])
    moving = Counter(r["category"] for r in rows if r["moving_command"])
    errors = sorted(r["cross_track_m"] for r in rows if r["category"] == "model_chain_consistent")
    summary = {
        "synthetic": synthetic,
        "sample_basis": basis,
        "topics": dict(kinds),
        "message_counts": {k: len(indexed[k]) for k in kinds.values()},
        "max_receipt_skew_s": max_skew_ns / NS,
        "command_tolerance": command_tolerance,
        "all_sample_categories": dict(Counter(r["category"] for r in rows)),
        "moving_sample_categories": dict(moving),
        "decision_sources": dict(Counter(str(d.get("executed_source") or "unverified") for _, d in indexed["decision"])),
        "model_chain_consistent_samples": len(errors),
        "consistent_fraction_of_moving_samples": len(errors) / sum(moving.values()) if moving else None,
        "cross_track_m": {"median": median(errors), "p95": errors[math.ceil(0.95 * len(errors))-1],
                          "max": errors[-1]} if errors else None,
        "same_path_motion": {
            "paired_odom_intervals": sum(r["same_path_odom_step_m"] is not None for r in rows),
            "observed_displacement_m": sum(r["same_path_odom_step_m"] or 0 for r in rows),
            "net_forward_progress_m": sum(r["same_path_forward_progress_m"] or 0 for r in rows),
        },
        "limitations": [
            "Counts are recorded samples, not a time or distance share.",
            "Exact path stamp/frame and geometry establish path provenance; command and odometry joins use bag receipt time.",
            "Matching commands are consistent with model following, but do not identify the publisher or exclude a coincident joystick command.",
            "A recorded nonzero manual command near a sample is excluded; absence of recorded manual commands is not proof of absence.",
            "A consistent nonzero command is not proof of movement; observed displacement and projected progress use only adjacent same-path odometry.",
            "The downstream topic must be verified against the actual robot mux/controller wiring for each setup.",
            "Odometry is an estimate. Cross-track distance is point-to-current-polyline distance, not ground truth or safety clearance.",
            "The audit covers the recorded interval only and cannot establish safe behavior.",
        ],
    }
    return summary, rows, snapshots


def render_plot(path, summary, rows, snapshots, at_s=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, cmd) = plt.subplots(1, 2, figsize=(14, 5.7), layout="constrained")
    if summary["synthetic"]:
        fig.suptitle("SYNTHETIC DEMO — NOT A ROBOT TEST", color="darkred", fontweight="bold")
    indices = [i for i, (d, _) in enumerate(snapshots) if d]
    i = None
    if indices:
        if at_s is None:
            i = next((j for j in indices if rows[j]["category"] == "model_chain_consistent"), indices[0])
        else:
            i = min(indices, key=lambda j: abs(rows[j]["time_s"] - at_s))
    if i is None:
        ax.text(0.5, 0.5, "No joined decision evidence", transform=ax.transAxes, ha="center")
    else:
        row = rows[i]
        d, path_points = snapshots[i]
        for c in candidates_from(d):
            pts = geometry(c.get("path"))
            if not pts:
                continue
            chosen = c.get("candidate_index") == d.get("selected_candidate_index")
            color = "tab:green" if chosen else "tab:blue" if c.get("valid") else "tab:red"
            score = c.get("score")
            total = score.get("total") if isinstance(score, dict) else None
            annotation = f" | cost {total:.3f}" if finite(total) else ""
            ax.plot(*zip(*pts), marker=".", alpha=0.85 if chosen else 0.55, color=color,
                    label=f"Candidate {c.get('candidate_index')} ({c.get('reason', '?')}){annotation}")
        if path_points:
            ax.plot(*zip(*path_points), color="k", linestyle="--", linewidth=2, label="Published path")
        local = [r for r in rows if row["frame_id"] and r["frame_id"] == row["frame_id"]
                 and abs(r["time_s"]-row["time_s"]) <= 3 and r["cross_track_m"] is not None]
        if local:
            ax.plot([r["odom_x"] for r in local], [r["odom_y"] for r in local], color="tab:purple",
                    linewidth=2, label="Recorded odometry (±3 s)")
        ax.set_title(f"t={row['time_s']:.2f}s | {d.get('executed_source')} | {d.get('decision_reason')}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)
    ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
    for key, color, style in (("tracker", "tab:green", ":"), ("upstream", "tab:blue", "--"),
                              ("downstream", "black", "-")):
        cmd.plot([r["time_s"] for r in rows], [r[f"{key}_w"] for r in rows], color=color, ls=style,
                 label=f"{key} angular command")
    if i is not None:
        cmd.axvline(rows[i]["time_s"], color="tab:purple", alpha=0.5)
    cmd.set(xlabel="Bag time since first command [s]", ylabel="Angular command [rad/s]",
            title=f"Matched evidence: {summary['model_chain_consistent_samples']} moving samples")
    cmd.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def demo_events(kinds):
    """A fabricated sample with a model segment, fallback, and manual override."""
    names = {k: tp for tp, k in kinds.items()}
    yield {"topic": DEMO_MARKER, "bag_ns": NS, "message": {"synthetic": True}}

    def event(kind, t, **message):
        return {"topic": names[kind], "bag_ns": t, "message": message}

    for i in range(40):
        t = NS + (i + 1) * (NS // 10)
        source = "fallback" if i >= 24 else "model"
        manual = i in (17, 18)
        if i in (0, 12, 24, 36):
            path_stamp = t - NS // 100
            x = i * 0.018
            points = [[x + j * 0.1, 0.04 + 0.007 * j] for j in range(12)]
            rejected = [[x + j * 0.09, -0.06 * j] for j in range(12)]
            alternate = [[x + j * 0.10, 0.04 + 0.018 * j] for j in range(12)]
            key = {"stamp_ns": path_stamp, "frame_id": "odom"}
            yield event("path", path_stamp, **key, points=points)
            yield event("decision", path_stamp + 1, **key, executed_source=source,
                        model_evidence_only=(source == "model"), executed_path=points,
                        selected_candidate_index=0 if source == "model" else None,
                        executed_candidate_index=0 if source == "model" else None,
                        decision_reason="lowest valid cost" if source == "model" else "demo fallback",
                        selection_reason="minimum total cost",
                        tie_break="earlier_selector_input",
                        candidates=[
                            {"candidate_index": 0, "rank": 1, "path": points, "valid": True, "reason": "valid",
                             "score": {"eligible": True, "rank_cost": 0.0, "continuity_cost": 0.03,
                                       "heading_cost": 0.07, "soft_margin_cost": 0.0, "total": 0.1}},
                            {"candidate_index": 1, "rank": 2, "path": rejected, "valid": False,
                             "reason": "footprint collision"},
                            {"candidate_index": 2, "rank": 3, "path": alternate, "valid": True, "reason": "valid",
                             "score": {"eligible": True, "rank_cost": 0.3, "continuity_cost": 0.15,
                                       "heading_cost": 0.07, "soft_margin_cost": 0.0, "total": 0.52}},
                        ])
        cmd = {"v": 0.18, "w": -0.25 if source == "fallback" else 0.08}
        yield event("upstream", t, **cmd)
        yield event("tracker", t + 1, event="tracking", motion_enabled=True, path={**key, "fresh": True},
                    robot={"frame_id": "odom"}, cmd_vel_topic=names["upstream"],
                    control={"mode": "model_evidence" if source == "model" else "normal",
                             "steering_source": "path", "deadman_configured": True,
                             "deadman_pressed": True}, published_command=cmd)
        actual = {"v": 0.08, "w": -0.35} if manual else cmd
        if manual and "manual" in names:
            yield event("manual", t + 1, **actual)
        yield event("downstream", t + 2, **actual)
        yield event("odom", t + 3, frame_id="odom", x=i * 0.018, y=0.035 + 0.001 * i)


def report_html(summary, decisions, *, has_plot):
    def cell(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True)
        return escape("—" if value is None else str(value))

    def table(headers, rows):
        return ("<table><thead><tr>" + "".join(f"<th>{cell(h)}</th>" for h in headers)
                + "</tr></thead><tbody>" + "".join(
                    "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows)
                + "</tbody></table>")

    count = summary["model_chain_consistent_samples"]
    errors = summary["cross_track_m"]
    motion = summary["same_path_motion"]
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>Jackal model-following audit</title>',
            '<style>body{font:16px system-ui,sans-serif;margin:2rem;line-height:1.5;max-width:90rem}'
            'table{border-collapse:collapse;margin:0.5rem 0 1rem}td,th{border:1px solid #ccd4dd;padding:.35rem .55rem;'
            'text-align:left;overflow-wrap:anywhere}th{background:#edf2f7}details{margin:.6rem 0}'
            'img{max-width:100%;height:auto}.warn{color:#8e210f;font-weight:700}</style>',
            '<h1>Jackal model-following audit</h1>']
    if summary["synthetic"]:
        page.append('<p class="warn">SYNTHETIC DEMO — NOT A ROBOT TEST</p>')
    page.extend([f'<p>Input: {cell(summary["input"])}</p>',
                 f'<p>{count} recorded moving-command samples match the selected model path, '
                 'path-based steering, the recorded downstream command, and odometry.</p>',
                 '<p><a href="summary.json">Summary JSON</a> · '
                 '<a href="samples.csv">Sample CSV</a> · <a href="decisions.jsonl">Decision JSONL</a></p>',
                 '<h2>Moving-command sample classifications</h2>',
                 table(("Category", "Samples"), summary["moving_sample_categories"].items()),
                 '<h2>Odometry in matching samples</h2>',
                 table(("Metric", "Value"), [
                     ("Cross-track median [m]", round(errors["median"], 4) if errors else None),
                     ("Cross-track p95 [m]", round(errors["p95"], 4) if errors else None),
                     ("Paired odometry intervals", motion["paired_odom_intervals"]),
                     ("Observed displacement [m]", round(motion["observed_displacement_m"], 4)),
                     ("Net projected progress [m]", round(motion["net_forward_progress_m"], 4)),
                 ]),
                 '<h2>Interpretation limits</h2><ul>'
                 + "".join(f'<li>{cell(line)}</li>' for line in summary["limitations"]) + '</ul>'])
    if has_plot:
        page.append('<p><img src="evidence.png" alt="Candidate paths, recorded odometry and angular commands"></p>')
    page.append('<h2>Recorded decisions</h2>')
    for event in sorted(decisions, key=lambda e: e["bag_ns"]):
        d = event["message"]
        candidate_rows = []
        for c in candidates_from(d):
            score = c.get("score")
            score = score if isinstance(score, dict) else {}
            costs = [score.get(k) for k in ("rank_cost", "continuity_cost", "heading_cost", "soft_margin_cost", "total")]
            candidate_rows.append((c.get("candidate_index"), c.get("rank"), c.get("valid"), c.get("reason"),
                                   score.get("eligible"), *(round(v, 4) if finite(v) else None for v in costs)))
        page.append(
            f'<details><summary>Path stamp {cell(d.get("stamp_ns"))} · {cell(d.get("executed_source"))}'
            f' · {cell(d.get("decision_reason"))}</summary>'
            f'<p>Selection: {cell(d.get("selection_reason"))} · '
            f'proposed #{cell(d.get("selected_candidate_index"))} · executed #{cell(d.get("executed_candidate_index"))}'
            f' · tie break: {cell(d.get("tie_break"))}</p>'
            + table(("Candidate", "Rank", "Valid", "Reason", "Eligible", "Rank cost", "Continuity",
                     "Heading", "Soft margin", "Total cost"), candidate_rows) + '</details>')
    return "\n".join(page) + "\n</html>\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, nargs="?", help="ROS 2 bag directory/file, or normalized JSONL")
    p.add_argument("--demo", action="store_true", help="Generate explicitly labeled synthetic data, never robot evidence")
    p.add_argument("--out", type=Path, required=True, help="New report directory (will not overwrite)")
    p.add_argument("--path-topic", default=PATH)
    p.add_argument("--decision-topic", default=DECISION)
    p.add_argument("--tracker-topic", default=TRACKER)
    p.add_argument("--odom-topic", default=ODOM)
    p.add_argument("--upstream-topic", default=UPSTREAM)
    p.add_argument("--actual-cmd-topic", default=DOWNSTREAM, help="Verify this is the robot-side mux output")
    p.add_argument("--joy-topic", help="Optional recorded Joy topic; presence is reported, not source attribution")
    p.add_argument("--manual-cmd-topic", help="Optional teleop Twist topic; nearby nonzero commands are excluded")
    p.add_argument("--storage-id", choices=("sqlite3", "mcap"))
    p.add_argument("--max-skew-s", type=float, default=0.15)
    p.add_argument("--command-tolerance", type=float, default=0.005)
    p.add_argument("--export-events", action="store_true", help="Write normalized events.jsonl for offline sharing")
    p.add_argument("--plot", action="store_true", help="Render evidence.png with Matplotlib")
    p.add_argument("--at-s", type=float, help="Candidate snapshot at the nearest recorded joined command time")
    a = p.parse_args()
    if a.demo == (a.input is not None):
        p.error("provide exactly one input bag/JSONL or --demo")
    if not (math.isfinite(a.max_skew_s) and 0 < a.max_skew_s <= 1
            and math.isfinite(a.command_tolerance) and 0 <= a.command_tolerance <= 0.05):
        p.error("require 0 < --max-skew-s <= 1 and 0 <= --command-tolerance <= 0.05")
    topics = [(a.path_topic, "path"), (a.decision_topic, "decision"), (a.tracker_topic, "tracker"),
              (a.odom_topic, "odom"), (a.upstream_topic, "upstream"), (a.actual_cmd_topic, "downstream")]
    if a.joy_topic:
        topics.append((a.joy_topic, "joy"))
    if a.manual_cmd_topic or a.demo:
        topics.append((a.manual_cmd_topic or "/demo/manual_cmd", "manual"))
    kinds = dict(topics)
    if len(kinds) != len(topics):
        p.error("topic names must be distinct; an upstream topic is not a mux output")
    if a.out.exists():
        p.error(f"report directory already exists: {a.out}")
    try:
        if a.plot:
            import matplotlib  # ensure plotting dependencies before writing output
        if a.demo:
            events = list(demo_events(kinds))
        else:
            events = list(load_jsonl(a.input) if a.input.suffix.lower() == ".jsonl"
                          else load_bag(a.input, kinds, a.storage_id))
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        p.error(str(exc))
    summary, rows, snapshots = audit(events, kinds, max_skew_ns=int(a.max_skew_s * NS),
                                     command_tolerance=a.command_tolerance)
    summary["input"] = str(a.input.resolve()) if a.input is not None else "SYNTHETIC DEMO — NOT A ROBOT TEST"
    a.out.mkdir(parents=True)
    (a.out / "summary.json").write_text(json.dumps(clean_json(summary), indent=2, ensure_ascii=False,
                                                     allow_nan=False)+"\n", encoding="utf-8")
    with (a.out / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("bag_ns,time_s,category\n")
    decisions = [e for e in events if e["topic"] == a.decision_topic]
    with (a.out / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for event in decisions:
            f.write(json.dumps(clean_json(event), allow_nan=False, separators=(",", ":"))+"\n")
    if a.export_events:
        with (a.out / "events.jsonl").open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(clean_json(event), allow_nan=False, separators=(",", ":"))+"\n")
    if a.plot:
        render_plot(a.out / "evidence.png", summary, rows, snapshots, a.at_s)
    (a.out / "report.html").write_text(report_html(summary, decisions, has_plot=a.plot), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
