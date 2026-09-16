"""Deterministic attribution of blocking phases, without ROS or a model/GPU."""
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from moai_jackal_spubert import phase_timing
from test_bridge_stability import plan_tick, ready_bridge


class FakeElapsedClock:
    def __init__(self):
        self.value = 1000.0

    def now(self):
        return self.value

    def advance(self, duration):
        self.value += duration

    def timed(self, function, duration):
        def call(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            finally:
                self.advance(duration)
        return call


def records(node):
    return [json.loads(line) for line in node._diagnostics_file.getvalue().splitlines()]


def test_header_receipt_and_history_ages_are_distinct_signed_ros_clock_measurements():
    result = phase_timing.input_age_snapshot(
        now_ns=100_000_000_000,
        stamps={"odom": NS(sec=99, nanosec=750_000_000),
                "scan": NS(sec=100, nanosec=100_000_000), "tracks": NS(sec=0, nanosec=0)},
        receipt_times_s={"odom": 99.9, "scan": 100.0, "tracks": float("-inf")},
        history_times_s=[99.0, 99.6], history_odom_stamps_ns=[98_950_000_000, 99_550_000_000],
    )
    assert result["odom_header_age_sec"] == pytest.approx(0.25)
    assert result["odom_receipt_age_sec"] == pytest.approx(0.1)
    assert result["scan_header_age_sec"] == pytest.approx(-0.1)
    assert result["scan_receipt_age_sec"] == 0
    assert result["tracks_header_age_sec"] is None
    assert result["tracks_receipt_age_sec"] is None
    assert result["history_latest_sample_age_sec"] == pytest.approx(0.4)
    assert result["history_latest_odom_header_age_sec"] == pytest.approx(0.45)
    assert result["history_latest_sample_minus_odom_header_sec"] == pytest.approx(0.05)


def test_missing_bad_or_backwards_timers_do_not_invent_zero_duration():
    timings = {}
    phase_timing.record_duration(timings, "missing_sec", None, 4.0)
    phase_timing.record_duration(timings, "backwards_sec", 5.0, 4.0)
    phase_timing.record_duration(timings, "nan_sec", float("nan"), 4.0)
    with patch.object(phase_timing.time, "perf_counter", side_effect=RuntimeError("timer fault")):
        phase_timing.record_duration(timings, "broken_sec", 2.0)
    assert timings == {}


def test_queue_prediction_collection_validation_and_publish_are_separate_phases():
    node, clock = ready_bridge(), FakeElapsedClock()
    functions = node._on_timer.__func__.__globals__
    node._refresh_map = clock.timed(node._refresh_map, 0.02)
    node.heading_selector.select = clock.timed(node.heading_selector.select, 0.03)
    node._runtime.predict_candidates = clock.timed(node._runtime.predict_candidates, 0.40)
    node._publish_markers = clock.timed(node._publish_markers, 0.06)
    node.plan_context_pub.publish = clock.timed(node.plan_context_pub.publish, 0.04)
    with patch.object(phase_timing.time, "perf_counter", clock.now), patch.dict(functions, {
        "validate_candidate_path": clock.timed(functions["validate_candidate_path"], 0.08),
        "rank_valid_paths": clock.timed(functions["rank_valid_paths"], 0.03),
    }):
        node._on_timer()
        assert node._pending_inference is not None
        clock.advance(0.07)  # Executor scheduling, not model compute.
        node._inference_executor.finish()
        clock.advance(0.11)  # Result ready, but ROS timer has not collected it.
        node._on_timer()
    result = records(node)[-1]
    assert result["valid"] is True
    measured = result["phase_timing_sec"]
    expected = {
        "preparation_map_sec": 0.02, "runtime_inputs_prepare_sec": 0.03,
        "preparation_total_sec": 0.05, "worker_queue_sec": 0.07,
        "worker_predict_sec": 0.40, "worker_total_sec": 0.40,
        "result_collection_wait_sec": 0.11, "validation_map_sec": 0.02,
        "candidate_validation_sec": 0.08, "candidate_ranking_sec": 0.03,
        "marker_publish_sec": 0.06, "path_context_publish_sec": 0.04,
        "validation_and_publish_total_sec": 0.23, "prepare_to_decision_total_sec": 0.86,
    }
    for name, value in expected.items():
        assert measured[name] == pytest.approx(value), name
    # The monotonic fake elapsed time must not be mixed into ROS header ages.
    assert result["input_ages"]["dispatch"]["odom_header_age_sec"] == pytest.approx(0)
    assert result["input_ages"]["validation"]["odom_header_age_sec"] == pytest.approx(0)
    assert json.loads(node.plan_context_pub.messages[-1].data)["state"] == "active"


def test_failed_model_still_reports_worker_time_and_holds():
    node, clock = ready_bridge(), FakeElapsedClock()

    def fail(**kwargs):
        clock.advance(0.4)
        raise RuntimeError("model failed")

    node._runtime.predict_candidates = fail
    with patch.object(phase_timing.time, "perf_counter", clock.now):
        plan_tick(node)
    result = records(node)[-1]
    assert result["event"] == "inference_error"
    assert result["phase_timing_sec"]["worker_predict_sec"] == pytest.approx(0.4)
    assert json.loads(node.plan_context_pub.messages[-1].data)["state"] == "hold"


def test_age_instrumentation_failure_does_not_change_a_valid_plan():
    node = ready_bridge()
    functions = node._on_timer.__func__.__globals__
    with patch.dict(functions, {"input_age_snapshot": lambda **kwargs: 1 / 0}):
        plan_tick(node)
    result = records(node)[-1]
    assert result["valid"] is True
    assert result["input_ages"]["dispatch"] == {"unavailable": True}


def test_previous_io_is_labeled_and_log_failure_does_not_stop_path_publication():
    node, clock = ready_bridge(), FakeElapsedClock()

    class SlowStream(io.StringIO):
        def write(self, value):
            clock.advance(0.2)
            return super().write(value)

    node._diagnostics_file = SlowStream()
    with patch.object(phase_timing.time, "perf_counter", clock.now):
        node._record("first", {})
        node._record("second", {})
    first, second = records(node)
    assert first["previous_diagnostic_io_sec"] == {}
    assert second["previous_diagnostic_io_sec"]["file_write_sec"] == pytest.approx(0.2)
    node._diagnostics_file.close()  # ValueError from a closed sink, not just OSError.
    plan_tick(node)
    assert node._diagnostics_file is None
    assert json.loads(node.plan_context_pub.messages[-1].data)["state"] == "active"


def test_summary_handles_optional_missing_timings_future_headers_and_invalid_durations():
    summary = runpy.run_path(str(Path(__file__).resolve().parents[4] / "scripts/summarize_stability_logs.py"))
    rows = [(1, {"phase_timing_sec": {"worker_predict_sec": 0.2},
                 "input_ages": {"dispatch": {"odom_header_age_sec": -0.1, "scan_header_age_sec": None}}}),
            (2, {}),
            (3, {"phase_timing_sec": {"worker_predict_sec": 0.8, "bad_sec": -1},
                 "previous_diagnostic_io_sec": {"file_write_sec": 0.02}})]
    quality = summary["Quality"]()
    result = summary["summarize_phase_timing"](rows, quality)
    timing = result["phases_sec"]["worker_predict_sec"]
    assert timing["samples"] == 2
    assert timing["mean_sec"] == pytest.approx(0.5)
    assert timing["p95_sec"] == pytest.approx(0.8)
    assert result["input_ages_sec"]["dispatch.odom_header_age_sec"]["min_sec"] == -0.1
    assert "dispatch.scan_header_age_sec" not in result["input_ages_sec"]
    assert quality.report()["counts"] == {"null_or_invalid:phase_timing_sec.bad_sec": 1}
