"""Synthetic records are never accepted as model evidence when a link is absent."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[4] / "field_scripts/analysis/model_following_evidence.py"
M = runpy.run_path(str(SCRIPT))
TOPICS = {M[k.upper()]: k for k in ("path", "decision", "tracker", "odom", "upstream", "downstream")}
NAMES = {kind: topic for topic, kind in TOPICS.items()}


def event(kind, t, **message):
    return {"topic": NAMES[kind], "bag_ns": t, "message": message}


def run_events():
    points = [[0, 0], [1, 0], [2, 0]]
    key = {"stamp_ns": 1_000_000_000, "frame_id": "odom"}
    result = [
        event("path", 1_001_000_000, **key, points=points),
        event("decision", 1_002_000_000, **key, executed_source="model",
              model_evidence_only=True,
              executed_path=points, executed_candidate_index=0, selected_candidate_index=0,
              candidates=[{"candidate_index": 0, "path": points, "valid": True, "reason": "valid"}]),
    ]
    for t, x in ((1_100_000_000, 0.1), (1_200_000_000, 0.12)):
        cmd = {"v": 0.2, "w": -0.1}
        result.extend([
            event("upstream", t, **cmd),
            event("tracker", t + 1, event="tracking", motion_enabled=True,
                  path={**key, "fresh": True}, cmd_vel_topic=NAMES["upstream"],
                  robot={"frame_id": "odom"},
                  control={"mode": "model_evidence", "steering_source": "path",
                           "deadman_configured": True, "deadman_pressed": True},
                  published_command=cmd),
            event("downstream", t + 2, **cmd),
            event("odom", t + 3, frame_id="odom", x=x, y=0.03),
        ])
    return deepcopy(result)


def audit(events):
    return M["audit"](events, TOPICS, max_skew_ns=60_000_000)


def test_matching_chain_measures_cross_track_and_same_path_motion():
    summary, rows, _ = audit(run_events())
    assert summary["moving_sample_categories"] == {"model_chain_consistent": 2}
    assert summary["cross_track_m"]["p95"] == pytest.approx(0.03)
    assert summary["same_path_motion"]["observed_displacement_m"] == pytest.approx(0.02)
    assert rows[1]["same_path_forward_progress_m"] == pytest.approx(0.02)


@pytest.mark.parametrize("kind, mutation, category", [
    ("decision", lambda m: m.update(executed_source="retained"), "source_retained"),
    ("decision", lambda m: m.update(executed_source="fallback"), "source_fallback"),
    ("decision", lambda m: m.update(stamp_ns=1_000_000_001), "unverified_path"),
    ("decision", lambda m: m["candidates"][0].update(path=[[0, 1], [1, 1]]), "candidate_path_mismatch"),
    ("tracker", lambda m: m["control"].update(steering_source="gap"), "steering_gap"),
    ("tracker", lambda m: m["control"].update(steering_source="avoidance"), "steering_avoidance"),
    ("tracker", lambda m: m.pop("control"), "unverified_control_mode"),
    ("decision", lambda m: m.pop("model_evidence_only"), "unverified_control_mode"),
    ("tracker", lambda m: m["control"].update(deadman_pressed=False), "unverified_deadman"),
    ("downstream", lambda m: m.update(w=0.3), "command_mismatch"),
    ("odom", lambda m: m.update(frame_id="map"), "unverified_odom"),
])
def test_evidence_is_conservative(kind, mutation, category):
    events = run_events()
    for e in events:
        if e["topic"] == NAMES[kind]:
            mutation(e["message"])
    summary, _, _ = audit(events)
    assert summary["moving_sample_categories"] == {category: 2}
    assert summary["model_chain_consistent_samples"] == 0


def test_missing_mux_output_never_confirms_controller_input():
    events = [e for e in run_events() if e["topic"] != NAMES["downstream"]]
    summary, _, _ = audit(events)
    assert summary["sample_basis"] == "upstream"
    assert summary["moving_sample_categories"] == {"unverified_command_chain": 2}


def test_ambiguous_path_provenance_is_not_matched():
    events = run_events()
    events.append(deepcopy(events[0]))
    summary, _, _ = audit(events)
    assert summary["moving_sample_categories"] == {"unverified_path": 2}


def test_matching_command_with_manual_teleop_present_is_excluded():
    events = run_events()
    events.append({"topic": "/manual", "bag_ns": 1_100_000_001, "message": {"v": 0.2, "w": -0.1}})
    summary, _, _ = M["audit"](events, {**TOPICS, "/manual": "manual"}, max_skew_ns=60_000_000)
    assert summary["moving_sample_categories"] == {"manual_command_nearby": 1, "model_chain_consistent": 1}


def test_demo_cannot_be_mistaken_for_a_robot_run():
    kinds = {**TOPICS, "/demo/manual_cmd": "manual"}
    summary, _, _ = M["audit"](M["demo_events"](kinds), kinds)
    assert summary["synthetic"] is True
    assert "manual_command_nearby" in summary["moving_sample_categories"]
    assert "source_fallback" in summary["moving_sample_categories"]
