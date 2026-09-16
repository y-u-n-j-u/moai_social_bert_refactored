from types import SimpleNamespace as NS
from moai_jackal_spubert.sensor_freshness import stamp_error


def test_source_timestamp_cannot_be_refreshed_by_receiving_it_again():
    assert stamp_error(NS(sec=95, nanosec=0), now_ns=100_000_000_000, timeout_sec=0.5) == "stale"
    assert stamp_error(NS(sec=100, nanosec=0), now_ns=100_000_000_000, timeout_sec=0.5) == ""


def test_source_timestamp_boundaries_and_future_clock_mismatch():
    assert stamp_error(NS(sec=99, nanosec=500_000_000), now_ns=100_000_000_000, timeout_sec=0.5) == ""
    assert stamp_error(NS(sec=99, nanosec=499_999_999), now_ns=100_000_000_000, timeout_sec=0.5) == "stale"
    assert stamp_error(NS(sec=101, nanosec=0), now_ns=100_000_000_000, timeout_sec=0.5) == "in_future"


def test_unspecified_and_malformed_stamps_fail_closed():
    assert stamp_error(None, now_ns=100_000_000_000, timeout_sec=0.5) == "missing"
    assert stamp_error(NS(sec=0, nanosec=0), now_ns=100_000_000_000, timeout_sec=0.5) == "missing"
    assert stamp_error(NS(sec=99, nanosec=1_000_000_000), now_ns=100_000_000_000, timeout_sec=0.5) == "invalid"
