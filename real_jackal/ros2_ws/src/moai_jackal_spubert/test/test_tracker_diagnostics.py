import json
import math

from moai_jackal_spubert.tracker_diagnostics import StopEpisode, diagnostic_json


def test_diagnostics_are_strict_json_with_explicit_missing_values():
    record = json.loads(diagnostic_json({"nested": {"age": math.inf}, "pose": (math.nan, 1)}))
    assert record == {"nested": {"age": None}, "pose": [None, 1]}


def test_first_stop_reason_survives_followup_status_and_is_saved_on_resume():
    episode = StopEpisode()
    episode.observe("planner_hold", 10.0, True)
    snapshot = episode.observe("predicted_path_missing_or_stale", 10.1, True)
    assert snapshot["first_reason"] == "planner_hold"
    resumed = episode.observe("tracking", 12.0, False)
    assert resumed["first_reason"] is None
    assert resumed["last_completed"]["first_reason"] == "planner_hold"
    assert resumed["last_completed"]["duration_s"] == 2.0
    assert episode.observe("scan_missing_or_stale", 13.0, True)["first_reason"] == "scan_missing_or_stale"


def test_clock_reset_never_produces_negative_stop_duration():
    episode = StopEpisode()
    episode.observe("disarmed", 10, True)
    assert episode.observe("disarmed", 1, True)["duration_s"] == 0
    episode.finish(0)
    assert episode.last_completed["duration_s"] == 0
