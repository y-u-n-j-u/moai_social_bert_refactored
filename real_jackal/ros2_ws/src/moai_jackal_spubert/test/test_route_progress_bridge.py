"""Exercise route-aware progress through real bridge callbacks and publication."""
import json
import math
from dataclasses import replace

import pytest

from test_bridge_stability import PoseStamped, candidate, plan_tick, ready_bridge


DETOUR = [(0, 0), (0, 2), (4, 2), (4, 0)]


def detour_bridge():
    node = ready_bridge()
    goal = PoseStamped()
    goal.header.frame_id = 'odom'
    goal.pose.position.x = 4.0
    node._on_goal(goal)
    node._route_in_frame = lambda frame: list(DETOUR)
    result = replace(candidate(), path_world=[(0.0, step * 0.1) for step in range(1, 13)],
                     selected_goal_world=(0.0, 1.2))
    node._runtime.candidates = [result]
    return node


def context(node):
    return json.loads(node.plan_context_pub.messages[-1].data)


def record(node):
    return json.loads(node._diagnostics_file.getvalue().splitlines()[-1])


def test_detour_reaches_tracker_with_legacy_and_route_progress_separately_logged():
    node = detour_bridge()
    plan_tick(node)
    assert context(node)['state'] == 'active'
    prediction = record(node)
    assert prediction['progress_mode'] == 'route'
    assert prediction['attempts'][0]['goal_progress_m'] == 0.0
    assert prediction['attempts'][0]['route_progress']['effective_progress_m'] == pytest.approx(1.2)
    assert prediction['validation']['progress_route']['points'] == [list(p) for p in DETOUR]
    assert prediction['validation']['progress_route']['frame'] == 'odom'


def test_explicit_legacy_comparison_mode_preserves_original_rejection():
    node = detour_bridge()
    node.candidate_progress_mode = 'final_goal'
    plan_tick(node)
    assert context(node)['state'] == 'hold'
    assert record(node)['attempts'][0]['reason'] == 'insufficient_goal_progress'


def test_route_revalidation_uses_new_route_instead_of_dispatch_snapshot():
    node = detour_bridge()
    node._on_timer()
    new_route = [(0, 0), (0, -2), (4, -2), (4, 0)]
    node._route_in_frame = lambda frame: new_route
    node._inference_executor.finish()
    node._on_timer()
    assert context(node)['state'] == 'hold'
    assert record(node)['validation']['progress_route']['points'] == [list(p) for p in new_route]
    assert record(node)['attempts'][0]['reason'] == 'insufficient_route_progress'


@pytest.mark.parametrize('route,reason', [
    (None, 'progress_route_unavailable'),
    ([], 'empty_progress_route'),
    ([(0, 0), (math.nan, 0)], 'nonfinite_progress_route'),
    ([(0, 0), (0, 2)], 'progress_route_goal_mismatch'),
])
def test_required_route_cannot_disappear_or_become_invalid_during_inference(route, reason):
    node = detour_bridge()
    node._on_timer()
    node._route_in_frame = lambda frame: route
    node._inference_executor.finish()
    node._on_timer()
    assert context(node)['reason'] == reason
    assert context(node)['state'] == 'hold'
    assert node.path_pub.messages[-1].poses == []


def test_same_frame_tf_like_route_displacement_is_not_counted_as_candidate_progress():
    node = detour_bridge()
    node._on_timer()
    node._route_in_frame = lambda frame: [(2, 0), (2, 2), (4, 2), (4, 0)]
    node._inference_executor.finish()
    node._on_timer()
    assert context(node)['reason'] == 'progress_route_too_far'
    assert node.path_pub.messages[-1].poses == []
