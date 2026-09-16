from copy import deepcopy
import json
from pathlib import Path
import runpy
import tempfile


summary = runpy.run_path(str(Path(__file__).resolve().parents[4] / "scripts/summarize_stability_logs.py"))


def metadata(**changes):
    value = {
        "policy": "preserve_safe_samples", "raw_count": 40, "original_count": 20,
        "raw_safe_count": 3, "original_safe_count": 0, "output_safe_count": 3,
        "repaired_count": 3, "output_count": 20,
        "raw": {"rejection_counts": {"unknown_model_map": 37, "footprint_unknown": 20}},
        "original": {"rejection_counts": {"unknown_model_map": 20, "footprint_unknown": 12}},
    }
    value.update(changes)
    return value


def row(value):
    return {"event": "prediction", "stamp_ns": 1_000_000_000, "goal_id": "mission:1",
            "valid": True, "attempts": [{"valid": True}], "model_output_collected": True,
            "model_input": {"theta_rad": 0.0, "legacy_heading_rad": 0.0, "goal_sampling": value}}


def summarize(*records):
    quality = summary["Quality"]()
    report = summary["summarize_goal_sampling"](list(enumerate(records, 1)), quality)
    json.dumps(report, allow_nan=False)
    return report, quality.report()


def test_complete_rows_count_recovery_without_inflating_overlapping_reasons():
    report, quality = summarize(row(metadata()), row(metadata(
        original_safe_count=2, output_safe_count=4, repaired_count=2,
        original={"rejection_counts": {"unknown_model_map": 18, "footprint_unknown": 10}})))
    group = report["by_policy"]["preserve_safe_samples"]
    assert quality["error_count"] == 0
    assert group["comparable_predictions"] == 2
    assert group["zero_original_safe"] == 1
    assert group["zero_output_safe"] == 0
    assert group["recovered_zero_to_positive"] == 1
    assert group["total_repaired_slots"] == 5
    assert group["fields"]["raw_count"] == {"prediction_samples": 2, "total": 80}
    raw = group["rejections"]["raw"]
    assert raw["prediction_samples"] == 2 and raw["candidate_samples"] == 80
    assert raw["reason_totals"] == {"footprint_unknown": 40, "unknown_model_map": 74}
    assert sum(raw["reason_totals"].values()) > raw["candidate_samples"]  # Reasons overlap.
    assert group["rejections"]["original"]["candidate_samples"] == 40


def test_policy_only_legacy_and_missing_metadata_remain_unavailable():
    old = row(metadata())
    del old["model_input"]["goal_sampling"]
    pending = {"event": "prediction", "model_output_collected": False}
    report, quality = summarize(old, pending, row({"policy": "legacy", "output_safe_count": 0, "output_count": 20}))
    group = report["by_policy"]["legacy"]
    assert quality["error_count"] == 0
    assert report["metadata_unavailable_predictions"] == 2
    assert group["comparable_predictions"] == 0 and group["incomplete_predictions"] == 1
    assert group["zero_original_safe"] is None and group["zero_output_safe"] is None
    assert group["total_repaired_slots"] is None
    assert group["fields"]["raw_count"]["total"] is None
    assert group["fields"]["output_safe_count"] == {"prediction_samples": 1, "total": 0}
    assert group["rejections"]["raw"]["reason_totals"] is None


def test_policies_and_nonprediction_events_cannot_amplify_recovery_totals():
    legacy = metadata(policy="legacy", repaired_count=0, output_safe_count=0)
    repeated = row(metadata())
    repeated["event"] = "inference_error"
    report, quality = summarize(row(metadata()), row(legacy), repeated)
    assert quality["error_count"] == 0
    assert report["prediction_records"] == 2
    assert report["by_policy"]["preserve_safe_samples"]["recovered_zero_to_positive"] == 1
    assert report["by_policy"]["legacy"]["recovered_zero_to_positive"] == 0
    assert report["by_policy"]["legacy"]["zero_output_safe"] == 1
    assert "total_repaired_slots" not in report  # No mixed-policy headline metric.


def test_nonfinite_negative_noninteger_or_boolean_counts_do_not_contribute():
    for value in (float("nan"), float("inf"), -1, True, "40", 40.5):
        report, quality = summarize(row(metadata(raw_count=value)))
        group = report["by_policy"]["preserve_safe_samples"]
        assert quality["error_count"] > 0
        assert report["invalid_metadata_predictions"] == 1
        assert group["comparable_predictions"] == 0
        assert group["fields"]["original_count"]["total"] is None
        assert group["rejections"]["original"]["reason_totals"] is None


def test_inconsistent_count_balances_and_candidate_amplification_are_rejected():
    changes = ({"output_safe_count": 21}, {"original_safe_count": 21},
               {"raw_safe_count": 41}, {"repaired_count": 21},
               {"output_safe_count": 2}, {"raw_safe_count": 0}, {"output_count": 21})
    for change in changes:
        report, quality = summarize(row(metadata(**change)))
        assert quality["error_count"] > 0
        assert report["by_policy"]["preserve_safe_samples"]["comparable_predictions"] == 0
        assert report["by_policy"]["preserve_safe_samples"]["total_repaired_slots"] is None


def test_invalid_rejection_counts_exclude_the_record_instead_of_amplifying_totals():
    for counts in ({"unknown": 38}, {"unknown": -1}, {"unknown": float("nan")},
                   {"unknown": True}, {"": 1}, [], None):
        report, quality = summarize(row(metadata(raw={"rejection_counts": counts})))
        assert quality["error_count"] > 0
        group = report["by_policy"]["preserve_safe_samples"]
        assert group["comparable_predictions"] == 0
        assert group["rejections"]["raw"]["candidate_samples"] is None


def test_missing_rejection_counts_and_explicit_empty_counts_have_distinct_coverage():
    no_reasons = metadata()
    del no_reasons["raw"]
    all_safe = metadata(raw_safe_count=40, original_safe_count=20, output_safe_count=20,
                        repaired_count=0, raw={"rejection_counts": {}}, original={"rejection_counts": {}})
    report, quality = summarize(row(no_reasons), row(all_safe))
    assert quality["error_count"] == 0
    raw = report["by_policy"]["preserve_safe_samples"]["rejections"]["raw"]
    assert raw == {"prediction_samples": 1, "candidate_samples": 40, "reason_totals": {}}


def test_malformed_metadata_and_missing_policy_report_quality_errors():
    for value in (None, [], {}, {"policy": ""}, {"policy": 123}):
        report, quality = summarize(row(value))
        assert quality["error_count"] == 1
        assert report["invalid_metadata_predictions"] == 1
        assert report["by_policy"] == {}


def test_jsonl_integration_and_invalid_constant_never_fabricate_a_zero_sample():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bridge.jsonl"
        good = row(metadata())
        bad = deepcopy(good)
        bad["model_input"]["goal_sampling"]["raw_count"] = float("nan")
        path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
        report = summary["summarize"](path, "bridge", 1e-6)
    assert report["records"] == 1
    assert report["data_quality"]["counts"] == {"malformed:json_object": 1}
    group = report["goal_sampling"]["by_policy"]["preserve_safe_samples"]
    assert group["comparable_predictions"] == 1 and group["total_repaired_slots"] == 3
