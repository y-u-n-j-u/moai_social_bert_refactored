"""The all-rejected fallback with curves to the goal samples (MGP), run through the actual bridge callbacks (no DDS, GPU or robot)."""
import json
from types import SimpleNamespace as NS

from test_collision_bridge_offline import Marker, Node, Runtime, bridge  # noqa: F401  (the fixture builds the real bridge with doubles)

GOALS = [(2.0, 0.0), (2.4, 0.9), (2.4, -0.9), (-1.5, 0.0), (3.0, 0.2)]


def predictions(node):
    return [json.loads(line) for line in node._diagnostics_file.getvalue().splitlines()
            if json.loads(line)["event"] == "prediction"]


def make_all_model_candidates_fail(node, safe=(True, True, True, True, True)):
    """Model candidates that fail the map check of the model itself, so the fallback has to act."""
    node._runtime.candidates = [NS(
        selected_goal_valid=True, trajectory_map_safe=False, execution_valid=True,
        path_world=[(0.2 * (i + 1), 0.01 * rank) for i in range(12)], candidate_rank=rank, candidate_index=rank + 10,
        selected_goal_world=GOALS[rank], candidate_goals_world=list(GOALS), candidate_safe_mask=tuple(safe),
    ) for rank in range(5)]
    node.map_collision = False                    # free space: the curves can pass the checks
    node.global_path_fallback = True


def evidence_of(node):
    return json.loads(node.selection_evidence_pub.messages[-1].data) if getattr(node, "selection_evidence_pub", None) \
        and node.selection_evidence_pub.messages else None


def test_curves_to_the_goal_samples_are_built_validated_and_one_is_published(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node._on_timer()
    evidence = evidence_of(node)
    assert evidence is not None, "no selection evidence published"
    rows = evidence["fallback_candidates"]
    assert rows, "no fallback candidates recorded"
    mgp = [r for r in rows if r["kind"] == "mgp"]
    assert len(mgp) >= 2
    behind = [r for r in rows if r["kind"] == "mgp" and r["goal"] == [-1.5, 0.0]]
    assert not behind, "a goal sample behind the robot must not get a curve"
    selected = [r for r in rows if r["selected"]]
    assert len(selected) == 1 and selected[0]["valid"]
    assert evidence["executed_source"] == "fallback"
    assert evidence["selected_fallback_index"] == selected[0]["candidate_index"]
    published = node.path_pub.messages[-1].poses
    assert [list(p) for p in published] == selected[0]["path"]


def test_an_mgp_curve_ends_at_its_goal_sample(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node._on_timer()
    for row in evidence_of(node)["fallback_candidates"]:
        if row["kind"] == "mgp":
            end = row["path"][-1]
            assert abs(end[0] - row["goal"][0]) < 0.3 and abs(end[1] - row["goal"][1]) < 0.3


def test_off_by_default_leaves_the_old_fallback_untouched(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    assert node.mgp_fallback is False
    node._on_timer()
    assert not (evidence_of(node) or {}).get("fallback_candidates")


def test_no_safe_goal_samples_means_only_the_old_fallback_competes(bridge):
    node = bridge
    make_all_model_candidates_fail(node, safe=(False,) * 5)
    node.mgp_fallback = True
    node._on_timer()
    rows = (evidence_of(node) or {}).get("fallback_candidates") or []
    assert all(r["kind"] != "mgp" for r in rows)


def test_nothing_is_published_when_every_curve_collides(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node.map_collision = True                      # everything occupied: every path fails the footprint check
    node._on_timer()
    assert node.path_pub.messages[-1].poses == []
    rows = (evidence_of(node) or {}).get("fallback_candidates") or []
    assert all(not r["valid"] for r in rows)


def drawn(node):
    array = node.selection_marker_pub.messages[-1]
    return [m for m in array.markers if getattr(m, "ns", "") == "fallback_candidates"]


def test_rviz_draws_the_fallback_paths_blue_and_the_published_one_green(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node.rejected_display_max = 0
    node._on_timer()
    lines = drawn(node)
    assert lines, "no fallback line was drawn"
    green = [m for m in lines if m.color.g > 0.9 and m.color.r < 0.3]
    blue = [m for m in lines if m.color.b > 0.9 and m.color.r < 0.3]
    assert len(green) == 1 and abs(green[0].scale.x - 0.075) < 1e-9
    assert len(blue) >= 1
    # the published path is the green one: no second, thicker copy of it is drawn
    array = node.selection_marker_pub.messages[-1]
    assert not [m for m in array.markers if getattr(m, "ns", "") == "published_nonmodel_path"]


def test_rejected_fallback_lines_count_toward_the_red_limit(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node.rejected_display_max = 2
    node.map_collision = True                      # every candidate fails
    node._on_timer()
    array = node.selection_marker_pub.messages[-1]
    red = [m for m in array.markers if getattr(m, "ns", "") in ("model_candidates", "fallback_candidates") and m.color.r > 0.9 and m.color.g < 0.3]
    assert len(red) <= 2


def test_when_the_previous_path_is_kept_the_fallback_candidates_are_still_shown_but_none_is_selected(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node.path_retention_max_age_s = 2.0
    node._last_published_path = [(0.0, 0.0)] + [(0.3 * i, 0.0) for i in range(1, 9)]
    node._last_published_s = node.now_s - 0.3
    node._on_timer()
    evidence = evidence_of(node)
    assert evidence["executed_source"] == "retained"
    rows = evidence["fallback_candidates"]
    assert rows and not any(r["selected"] for r in rows) and evidence.get("selected_fallback_index") is None
    # the thick published line stays (it is the retained path) and the fallback lines are drawn as candidates
    array = node.selection_marker_pub.messages[-1]
    assert [m for m in array.markers if getattr(m, "ns", "") == "published_nonmodel_path"]
    assert drawn(node)


def test_selection_markers_survive_a_missing_robot_pose(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node.candidate_hold_enabled = True
    node._candidate_hold.enabled = True
    evidence = {"executed_source": "hold", "selected_candidate_index": None, "robot_pose": None, "goal_generation": 1,
                "candidates": [], "executed_path": [], "fallback_candidates": []}
    header = NS(frame_id="odom", stamp=None)
    node._publish_selection_markers(header, evidence)             # must not raise
    assert node.selection_marker_pub.messages


def test_no_candidate_number_appears_twice_in_one_cycle(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    # model candidates numbered like the goal samples (0..4) so that the fallback curves would collide with them
    for rank, candidate in enumerate(node._runtime.candidates):
        candidate.candidate_index = rank
    node._on_timer()
    evidence = evidence_of(node)
    numbers = [c["candidate_index"] for c in evidence["candidates"]] + [r["candidate_index"] for r in evidence["fallback_candidates"]]
    assert len(numbers) == len(set(numbers)), numbers
    assert all(isinstance(n, int) for n in numbers)
    assert all(r["candidate_index"] >= len(GOALS) for r in evidence["fallback_candidates"])     # taken beyond every goal-sample number


def test_a_curve_keeps_its_goal_sample_number_when_that_number_is_free(bridge):
    node = bridge
    make_all_model_candidates_fail(node)
    node.mgp_fallback = True
    node._on_timer()
    for row in evidence_of(node)["fallback_candidates"]:
        if row["kind"] == "mgp":
            assert GOALS[row["candidate_index"]] == tuple(row["goal"])        # model numbers are 10..14 here, so 0..4 are free
