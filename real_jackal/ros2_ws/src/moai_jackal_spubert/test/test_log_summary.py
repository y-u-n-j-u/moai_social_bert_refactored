from pathlib import Path
import runpy


summary = runpy.run_path(str(Path(__file__).resolve().parents[4] / 'scripts/summarize_stability_logs.py'))


def record(stamp, theta):
    return {'event': 'prediction', 'stamp_ns': stamp, 'goal_id': 'mission:1',
            'valid': True, 'attempts': [{'valid': True}],
            'model_output_collected': True,
            'model_input': {'theta_rad': theta, 'legacy_heading_rad': theta}}


def test_cancelled_inference_is_not_missing_data_and_breaks_heading_comparison():
    rows = [(1, record(1_000_000_000, 0)),
            (2, {'event': 'prediction', 'stamp_ns': 2_000_000_000,
                 'goal_id': 'mission:1', 'valid': False,
                 'reason': 'inference_result_expired', 'attempts': [],
                 'model_output_collected': False}),
            (3, record(3_000_000_000, 1))]
    quality = summary['Quality']()
    report = summary['summarize_bridge'](rows, quality)
    assert quality.report()['error_count'] == 0
    assert report['predictions']['model_output_not_collected'] == 1
    assert report['actual_vs_legacy_heading']['samples'] == 2
    assert report['consecutive_actual_heading_change']['samples'] == 0
    assert report['invalid_prediction_reasons'] == {'inference_result_expired': 1}


def test_completed_inference_missing_heading_still_reports_bad_data():
    row = record(1_000_000_000, 0)
    del row['model_input']
    quality = summary['Quality']()
    summary['summarize_bridge']([(1, row)], quality)
    assert quality.report()['error_count'] == 2


def test_active_result_cannot_claim_model_was_not_collected():
    row = record(1_000_000_000, 0)
    row['model_output_collected'] = False
    quality = summary['Quality']()
    summary['summarize_bridge']([(1, row)], quality)
    assert quality.report()['counts'] == {'inconsistent:model_output_collected': 1}


def test_failed_preparation_inference_and_unknown_events_break_heading_comparison():
    for event in ('inference_error', 'preparation_rejected', 'unknown_status'):
        rows = [(1, record(1_000_000_000, 0)),
                (2, {'event': event, 'stamp_ns': 2_000_000_000,
                     'model_output_collected': False, 'error': 'unobserved model heading'}),
                (3, record(3_000_000_000, 1))]
        quality = summary['Quality']()
        report = summary['summarize_bridge'](rows, quality)
        assert quality.report()['error_count'] == 0, event
        assert report['events'][event] == 1
        assert report['predictions']['total'] == 2
        assert report['actual_vs_legacy_heading']['samples'] == 2
        assert report['consecutive_actual_heading_change']['samples'] == 0


def test_heading_comparison_resumes_between_predictions_after_a_failed_job():
    rows = [(1, record(1_000_000_000, 0)),
            (2, {'event': 'inference_error', 'stamp_ns': 2_000_000_000}),
            (3, record(3_000_000_000, 1)),
            (4, record(4_000_000_000, 1.25))]
    quality = summary['Quality']()
    report = summary['summarize_bridge'](rows, quality)
    assert quality.report()['error_count'] == 0
    comparison = report['consecutive_actual_heading_change']
    assert comparison['samples'] == 1
    assert abs(comparison['max_abs_rad'] - 0.25) < 1e-12
