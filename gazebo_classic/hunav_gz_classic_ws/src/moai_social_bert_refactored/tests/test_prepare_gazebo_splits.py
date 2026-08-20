from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_gazebo_splits.py"
SPEC = importlib.util.spec_from_file_location("prepare_gazebo_splits", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREPARE)


def sample(map_name: str, episode_id: int = 0) -> dict:
    return {
        "target_past": [[0.0, 0.0]],
        "meta": {
            "episode_id": episode_id,
            "map_yaml_path": f"/maps/{map_name}.yaml",
        },
    }


def write_run(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump({"samples": samples}, stream)


def write_qualification(run_dir: Path, passed: bool) -> None:
    (run_dir / "qualification.json").write_text(
        __import__("json").dumps({"passed": passed})
    )


def write_pilot_summary(
    pilot_dir: Path, passed: bool, run_count: int
) -> None:
    (pilot_dir / "pilot_summary.json").write_text(
        __import__("json").dumps(
            {
                "passed": passed,
                "required_runs": run_count,
                "report_count": run_count,
                "passed_run_count": run_count if passed else run_count - 1,
            }
        )
    )


def test_load_runs_excludes_held_out_map(tmp_path: Path) -> None:
    write_run(tmp_path / "route_choice_clean_social.pkl", [sample("training_route_choice")])
    write_run(tmp_path / "dual_route_clean_social.pkl", [sample("training_dual_route")])

    samples, sources, excluded = PREPARE.load_runs(
        tmp_path,
        {"training_dual_route"},
    )

    assert len(samples) == 1
    assert samples[0]["meta"]["map_name"] == "training_route_choice"
    assert len(sources) == 1
    assert len(excluded) == 1
    assert excluded[0]["maps"] == ["training_dual_route"]


def test_mixed_training_and_held_out_run_is_rejected(tmp_path: Path) -> None:
    write_run(
        tmp_path / "mixed_clean_social.pkl",
        [sample("training_route_choice"), sample("training_dual_route")],
    )

    with pytest.raises(ValueError, match="mixes training and held-out maps"):
        PREPARE.load_runs(tmp_path, {"training_dual_route"})


def test_scenario_name_fallback_extracts_map() -> None:
    value = PREPARE.sample_map_name(
        {
            "meta": {
                "scenario_name": "agents_training_route_choice_safe_teacher_low.yaml"
            }
        },
        Path("sample.pkl"),
    )

    assert value == "training_route_choice"


def test_load_runs_requires_passed_qualification_when_enabled(tmp_path: Path) -> None:
    passed = tmp_path / "passed" / "processed" / "passed_clean_social.pkl"
    failed = tmp_path / "failed" / "processed" / "failed_clean_social.pkl"
    missing = tmp_path / "missing" / "processed" / "missing_clean_social.pkl"
    write_run(passed, [sample("training_route_choice")])
    write_run(failed, [sample("training_route_choice")])
    write_run(missing, [sample("training_route_choice")])
    write_qualification(passed.parent.parent, True)
    write_qualification(failed.parent.parent, False)

    samples, sources, excluded = PREPARE.load_runs(
        tmp_path,
        set(),
        require_qualification=True,
    )

    assert len(samples) == 1
    assert sources == [str(passed)]
    assert {item["reason"] for item in excluded} == {
        "qualification_failed",
        "qualification_missing",
    }


def test_failed_qualification_is_excluded_even_for_legacy_mode(tmp_path: Path) -> None:
    passed = tmp_path / "legacy" / "processed" / "legacy_clean_social.pkl"
    failed = tmp_path / "failed" / "processed" / "failed_clean_social.pkl"
    write_run(passed, [sample("training_route_choice")])
    write_run(failed, [sample("training_route_choice")])
    write_qualification(failed.parent.parent, False)

    samples, sources, excluded = PREPARE.load_runs(
        tmp_path,
        set(),
        require_qualification=False,
    )

    assert len(samples) == 1
    assert sources == [str(passed)]
    assert [item["reason"] for item in excluded] == ["qualification_failed"]


def test_load_runs_requires_approved_minimum_pilot_size(tmp_path: Path) -> None:
    approved = tmp_path / "approved" / "seed_1" / "processed" / "seed_1_clean_social.pkl"
    smoke = tmp_path / "smoke" / "seed_1" / "processed" / "seed_1_clean_social.pkl"
    write_run(approved, [sample("training_route_choice")])
    write_run(smoke, [sample("training_route_choice")])
    write_qualification(approved.parent.parent, True)
    write_qualification(smoke.parent.parent, True)
    write_pilot_summary(approved.parents[2], True, 10)
    write_pilot_summary(smoke.parents[2], True, 2)

    samples, sources, excluded = PREPARE.load_runs(
        tmp_path,
        set(),
        require_qualification=True,
        minimum_pilot_runs=10,
    )

    assert len(samples) == 1
    assert sources == [str(approved)]
    assert [item["reason"] for item in excluded] == [
        "pilot_run_count_below_minimum"
    ]


def test_recording_id_uses_relative_run_path(tmp_path: Path) -> None:
    first = tmp_path / "pilot_a" / "seed_1" / "processed" / "seed_1_clean_social.pkl"
    second = tmp_path / "pilot_b" / "seed_1" / "processed" / "seed_1_clean_social.pkl"
    write_run(first, [sample("training_route_choice")])
    write_run(second, [sample("training_route_choice")])

    samples, _, _ = PREPARE.load_runs(tmp_path, set())
    recording_ids = {item["meta"]["recording_id"] for item in samples}

    assert len(recording_ids) == 2
    assert all("pilot_" in value for value in recording_ids)
