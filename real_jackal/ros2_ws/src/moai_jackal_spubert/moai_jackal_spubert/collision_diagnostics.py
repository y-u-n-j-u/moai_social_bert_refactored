"""Read-only collision evidence helpers; not used to select or reject paths."""
from __future__ import annotations

from .navigation_core import CandidateCheck


def collision_evidence(check: CandidateCheck | None) -> dict:
    if check is None:
        return {"swept_check_performed": False}
    return {
        "swept_check_performed": check.current_footprint_collision is not None,
        "footprint_collision_count": check.footprint_collision_count,
        "current_footprint_collision": check.current_footprint_collision,
        "first_collision_point": check.first_collision_point,
        "first_collision_sample_index": check.first_collision_sample_index,
        "first_collision_segment_index": check.first_collision_segment_index,
        "first_collision_distance_m": check.first_collision_distance_m,
        "sample_spacing_m": check.collision_sample_spacing_m,
        "swept_sample_count": check.swept_sample_count,
    }


def header_evidence(message, received_s: float, now_s: float) -> dict:
    stamp = message.header.stamp
    stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return {
        "frame_id": message.header.frame_id,
        "stamp_ns": stamp_ns,
        "source_age_s": now_s - stamp_ns / 1e9,
        "received_age_s": now_s - received_s,
    }
