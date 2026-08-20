#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$script_dir/hunav_gz_classic_ws"
wrapper_dir="$workspace_dir/src/hunav_gazebo_wrapper"
recordings_dir="$workspace_dir/moai_recordings"
catalog="$wrapper_dir/scenarios/joint_interaction_catalog.json"

scenario="${HUNAV_JOINT_SCENARIO:-agents_training_route_choice_joint_lower_crossing.yaml}"
scenario="$(basename "$scenario")"
if [[ "$scenario" != *.yaml ]]; then
    scenario="${scenario}.yaml"
fi
scenario_stem="${scenario%.yaml}"
pilot_name="${HUNAV_PILOT_NAME:-${scenario_stem}_smoke}"
output_dir="$recordings_dir/$pilot_name"
container_output_dir="/home/hunav_gz_classic_ws/moai_recordings/$pilot_name"
seeds_text="${HUNAV_PILOT_SEEDS:-401}"
run_timeout="${HUNAV_PILOT_RUN_TIMEOUT:-210}"
max_attempts="${HUNAV_PILOT_MAX_ATTEMPTS:-2}"
profile_offset="${HUNAV_PILOT_PROFILE_OFFSET:-0}"
local_map_size_m="${HUNAV_TRAINING_LOCAL_MAP_SIZE_M:-20.0}"
local_map_grid_size="${HUNAV_TRAINING_LOCAL_MAP_GRID_SIZE:-32}"
human_safety_margin="${HUNAV_JOINT_HUMAN_SAFETY_MARGIN:-0.45}"
human_prediction_horizon="${HUNAV_JOINT_HUMAN_PREDICTION_HORIZON:-2.4}"
human_prediction_step="${HUNAV_JOINT_HUMAN_PREDICTION_STEP:-0.3}"
pedestrians_avoid_robot="${HUNAV_JOINT_PEDESTRIANS_AVOID_ROBOT:-False}"
human_avoidance_mode="${HUNAV_JOINT_HUMAN_AVOIDANCE_MODE:-auto}"
human_yield_supervisor="${HUNAV_JOINT_HUMAN_YIELD_SUPERVISOR:-auto}"
human_yield_safety_distance="${HUNAV_JOINT_HUMAN_YIELD_SAFETY_DISTANCE:-1.45}"
human_yield_release_distance="${HUNAV_JOINT_HUMAN_YIELD_RELEASE_DISTANCE:-1.60}"
human_yield_prediction_horizon="${HUNAV_JOINT_HUMAN_YIELD_PREDICTION_HORIZON:-6.0}"
minimum_clean_avoidance_windows="${HUNAV_PILOT_MIN_CLEAN_AVOIDANCE_WINDOWS:-0}"
max_robot_stop_duration="${HUNAV_PILOT_MAX_ROBOT_STOP_DURATION:-}"
stop_speed_threshold="${HUNAV_PILOT_STOP_SPEED_THRESHOLD:-0.10}"
minimum_route_deviation="${HUNAV_PILOT_MIN_ROUTE_DEVIATION:-}"
nav_params_file="${HUNAV_JOINT_NAV_PARAMS_FILE:-}"
use_gazebo_gui="${HUNAV_PILOT_USE_GAZEBO_GUI:-False}"
use_rviz="${HUNAV_PILOT_USE_RVIZ:-False}"
validate_only="${HUNAV_PILOT_VALIDATE_ONLY:-False}"
postprocessor="$workspace_dir/src/moai_hunav_bridge/scripts/postprocess_pedestrian_dataset.py"
quality_gate="$script_dir/scripts/teacher_quality_gate.py"

if [ ! -f "$catalog" ]; then
    echo "Missing joint-interaction catalog: $catalog" >&2
    echo "Run scripts/generate_training_maps.py from the repository root." >&2
    exit 2
fi
if [ ! -f "$wrapper_dir/scenarios/$scenario" ]; then
    echo "Unknown joint scenario: $scenario" >&2
    exit 2
fi
if ! [[ "$profile_offset" =~ ^[0-9]+$ ]]; then
    echo "HUNAV_PILOT_PROFILE_OFFSET must be a non-negative integer" >&2
    exit 2
fi
case "${pedestrians_avoid_robot,,}" in
    true|1|yes|on)
        expected_coupling="reciprocal"
        ;;
    false|0|no|off)
        expected_coupling="one-way"
        ;;
    *)
        echo "HUNAV_JOINT_PEDESTRIANS_AVOID_ROBOT must be True or False" >&2
        exit 2
        ;;
esac
if ! [[ "$minimum_clean_avoidance_windows" =~ ^[0-9]+$ ]]; then
    echo "HUNAV_PILOT_MIN_CLEAN_AVOIDANCE_WINDOWS must be a non-negative integer" >&2
    exit 2
fi

profile_values="$(python3 -c '
import json
import sys

catalog = json.load(open(sys.argv[1], encoding="utf-8"))
scenario = sys.argv[2]
if scenario not in catalog:
    raise SystemExit(f"scenario is not in joint catalog: {scenario}")
profile = catalog[scenario]
values = [
    profile["map"],
    *profile["robot_start"],
    *profile["robot_goal"],
    profile["agent_count"],
    ",".join(profile["tags"]),
]
print("\t".join(str(value) for value in values))
' "$catalog" "$scenario")"
IFS=$'\t' read -r map_name base_start_x base_start_y goal_x goal_y agent_count scenario_tags <<< "$profile_values"
case "${human_yield_supervisor,,}" in
    auto)
        if [[ ",$scenario_tags," == *",yield,"* ]]; then
            human_yield_supervisor=True
        else
            human_yield_supervisor=False
        fi
        ;;
    true|1|yes|on)
        human_yield_supervisor=True
        ;;
    false|0|no|off)
        human_yield_supervisor=False
        ;;
    *)
        echo "HUNAV_JOINT_HUMAN_YIELD_SUPERVISOR must be auto, True, or False" >&2
        exit 2
        ;;
esac
case "${human_avoidance_mode,,}" in
    auto)
        if [[ ",$scenario_tags," == *",moving_avoidance,"* ]]; then
            human_avoidance_mode=continuous
        elif [[ "${human_yield_supervisor,,}" == "true" ]]; then
            human_avoidance_mode=yield
        else
            human_avoidance_mode=off
        fi
        ;;
    off|yield|continuous)
        human_avoidance_mode="${human_avoidance_mode,,}"
        ;;
    *)
        echo "HUNAV_JOINT_HUMAN_AVOIDANCE_MODE must be auto, off, yield, or continuous" >&2
        exit 2
        ;;
esac
if [[ "$human_avoidance_mode" == "yield" ]]; then
    human_yield_supervisor=True
else
    human_yield_supervisor=False
fi
map_yaml="$wrapper_dir/maps/$map_name.yaml"
if [ ! -f "$map_yaml" ]; then
    echo "Scenario map is missing: $map_yaml" >&2
    exit 2
fi
if [[ ",$scenario_tags," == *",moving_avoidance,"* ]]; then
    max_robot_stop_duration="${max_robot_stop_duration:-0.8}"
    minimum_route_deviation="${minimum_route_deviation:-0.8}"
    if [ -z "${HUNAV_PILOT_MIN_CLEAN_AVOIDANCE_WINDOWS+x}" ]; then
        minimum_clean_avoidance_windows=1
    fi
fi
if [ -z "$nav_params_file" ]; then
    if [[ "$human_avoidance_mode" == "continuous" ]]; then
        nav_params_file=/home/hunav_gz_classic_ws/install/hunav_gazebo_wrapper/share/hunav_gazebo_wrapper/launch/pmb2_params/pmb2_nav_continuous_avoidance_teacher.yaml
    else
        nav_params_file=/home/hunav_gz_classic_ws/install/hunav_gazebo_wrapper/share/hunav_gazebo_wrapper/launch/pmb2_params/pmb2_nav_route_choice_teacher.yaml
    fi
fi

if [[ "${validate_only,,}" == "true" ]]; then
    echo "scenario=$scenario map=$map_name start=($base_start_x,$base_start_y) goal=($goal_x,$goal_y)"
    echo "agents=$agent_count tags=$scenario_tags map_yaml=$map_yaml avoidance_mode=$human_avoidance_mode yield_supervisor=$human_yield_supervisor"
    exit 0
fi

mkdir -p "$output_dir"
read -r -a seeds <<< "$seeds_text"
run_index=0

stop_container() {
    docker stop hunavsim_pmb2 >/dev/null 2>&1 || true
}
trap stop_container EXIT
trap 'stop_container; exit 130' INT TERM

for seed in "${seeds[@]}"; do
    profile_index=$(((run_index + profile_offset) % 10))
    start_values="$(python3 -c '
import math
import sys

sx, sy, gx, gy = map(float, sys.argv[1:5])
index = int(sys.argv[5])
forward_offsets = (0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35)
lateral_offsets = (0.00, 0.08, -0.08, 0.12, -0.12, 0.04, -0.04, 0.10, -0.10, 0.00)
# Route-choice starts above or below the central block. Keep profile variation
# on the outside of that block so a reactive detour does not begin biased
# toward the obstacle. Centered maps retain symmetric lateral variation.
if abs(sy) >= 0.5 and abs(gy) >= 0.5 and sy * gy > 0.0:
    side = 1.0 if sy > 0.0 else -1.0
    outward = (0.00, 0.04, 0.08, 0.12, 0.16, 0.02, 0.06, 0.10, 0.14, 0.18)
    lateral_offsets = tuple(side * value for value in outward)
dx, dy = gx - sx, gy - sy
length = math.hypot(dx, dy)
ux, uy = dx / length, dy / length
px, py = -uy, ux
start_x = sx + ux * forward_offsets[index] + px * lateral_offsets[index]
start_y = sy + uy * forward_offsets[index] + py * lateral_offsets[index]
yaw = math.atan2(gy - start_y, gx - start_x)
print(f"{start_x:.3f}\t{start_y:.3f}\t{yaw:.6f}")
' "$base_start_x" "$base_start_y" "$goal_x" "$goal_y" "$profile_index")"
    IFS=$'\t' read -r start_x start_y start_yaw <<< "$start_values"
    goal_waypoints="${goal_x},${goal_y};${start_x},${start_y}"
    run_index=$((run_index + 1))

    run_name="seed_${seed}"
    run_dir="$output_dir/$run_name"
    processed_dir="$run_dir/processed"
    raw_path="$run_dir/${run_name}_raw.pkl"
    container_raw_path="$container_output_dir/$run_name/${run_name}_raw.pkl"
    report_path="$run_dir/qualification.json"

    mkdir -p "$processed_dir"
    if [ -e "$raw_path" ] || [ -e "$report_path" ]; then
        echo "Refusing to overwrite existing pilot result: $run_dir" >&2
        exit 2
    fi

    completed=false
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        log_path="$run_dir/runner_attempt_${attempt}.log"
        echo "[joint-teacher] scenario=$scenario map=$map_name seed=$seed profile=$profile_index attempt=$attempt"
        echo "[joint-teacher] start=($start_x,$start_y) goal=($goal_x,$goal_y) agents=$agent_count tags=$scenario_tags"
        (
            HUNAV_DOCKER_TTY=false \
            HUNAV_DOCKER_USE_GPU=false \
            HUNAV_SCENARIO="$scenario" \
            HUNAV_NAVIGATION=True \
            HUNAV_PMB2_NAV_PARAMS_FILE="$nav_params_file" \
            HUNAV_USE_GAZEBO_GUI="$use_gazebo_gui" \
            HUNAV_USE_RVIZ="$use_rviz" \
            HUNAV_USE_EVALUATOR=False \
            HUNAV_PEDESTRIANS_AVOID_ROBOT="$pedestrians_avoid_robot" \
            HUNAV_USE_NAVGOAL_TO_START=True \
            HUNAV_HUMAN_OBSTACLE_PUBLISH_RATE=10.0 \
            HUNAV_HUMAN_OBSTACLE_CURRENT_RING_POINTS=16 \
            HUNAV_HUMAN_OBSTACLE_SAFETY_MARGIN="$human_safety_margin" \
            HUNAV_HUMAN_OBSTACLE_FILL_SPACING=0.20 \
            HUNAV_HUMAN_OBSTACLE_PREDICTED_RING_POINTS=16 \
            HUNAV_HUMAN_OBSTACLE_PREDICTION_HORIZON="$human_prediction_horizon" \
            HUNAV_HUMAN_OBSTACLE_PREDICTION_STEP="$human_prediction_step" \
            HUNAV_HUMAN_AVOIDANCE_MODE="$human_avoidance_mode" \
            HUNAV_HUMAN_YIELD_SUPERVISOR="$human_yield_supervisor" \
            HUNAV_HUMAN_YIELD_SAFETY_DISTANCE="$human_yield_safety_distance" \
            HUNAV_HUMAN_YIELD_RELEASE_DISTANCE="$human_yield_release_distance" \
            HUNAV_HUMAN_YIELD_PREDICTION_HORIZON="$human_yield_prediction_horizon" \
            HUNAV_HUMAN_YIELD_PREDICTION_STEP=0.10 \
            HUNAV_GZPOSE_X="$start_x" \
            HUNAV_GZPOSE_Y="$start_y" \
            HUNAV_GZPOSE_YAW="$start_yaw" \
            HUNAV_ROBOT_PATH_PLANNER=nav2 \
            HUNAV_ROBOT_SAVE_TRAINING_PKL=True \
            HUNAV_ROBOT_TRAINING_PKL_PATH="$container_raw_path" \
            HUNAV_ROBOT_TRAINING_SAMPLE_STRIDE=4 \
            HUNAV_ROBOT_TRAINING_FLUSH_EVERY=1 \
            HUNAV_AUTO_GOAL=True \
            HUNAV_AUTO_GOAL_MODE=waypoint \
            HUNAV_AUTO_GOAL_WAYPOINTS="$goal_waypoints" \
            HUNAV_AUTO_GOAL_SEED="$seed" \
            HUNAV_AUTO_GOAL_MIN_EPISODE_DURATION=12.0 \
            HUNAV_AUTO_GOAL_TIMEOUT=90.0 \
            HUNAV_AUTO_GOAL_NO_PROGRESS_TIMEOUT=30.0 \
            HUNAV_AUTO_GOAL_MAX_GOALS=1 \
            "$script_dir/run-hunav_gz_classic11_pmb2.bash"
        ) >"$log_path" 2>&1 &
        launcher_pid=$!
        deadline=$((SECONDS + run_timeout))

        while kill -0 "$launcher_pid" >/dev/null 2>&1; do
            if grep -q "Automatic goal publisher completed 1 goals" "$log_path"; then
                completed=true
                sleep 3
                break
            fi
            if (( SECONDS >= deadline )); then
                echo "[joint-teacher] seed=$seed attempt=$attempt timed out" >&2
                break
            fi
            sleep 2
        done

        stop_container
        wait "$launcher_pid" 2>/dev/null || true
        if [ "$completed" = true ] && [ -s "$raw_path" ]; then
            break
        fi
    done

    if [ "$completed" != true ] || [ ! -s "$raw_path" ]; then
        echo "[joint-teacher] seed=$seed produced no completed recording" >&2
        continue
    fi

    python3 "$postprocessor" \
        --input "$raw_path" \
        --map-yaml "$map_yaml" \
        --out-dir "$processed_dir" \
        --name "$run_name" \
        --map-size-m "$local_map_size_m" \
        --map-grid-size "$local_map_grid_size" \
        --guidance-policy route \
        --visualize 0 \
        --skip-quality-report >/dev/null

    quality_args=(
        evaluate
        --raw "$raw_path"
        --processed-summary "$processed_dir/${run_name}_summary.json"
        --minimum-clean-avoidance-windows "$minimum_clean_avoidance_windows"
        --stop-speed-threshold "$stop_speed_threshold"
        --expected-coupling "$expected_coupling"
        --expected-human-avoidance-mode "$human_avoidance_mode"
        --expected-human-yield-supervisor "${human_yield_supervisor,,}"
        --output "$report_path"
    )
    if [ -n "$max_robot_stop_duration" ]; then
        quality_args+=(--max-robot-stop-duration "$max_robot_stop_duration")
    fi
    if [ -n "$minimum_route_deviation" ]; then
        quality_args+=(--minimum-route-deviation "$minimum_route_deviation")
    fi
    python3 "$quality_gate" "${quality_args[@]}" >/dev/null || true
done

reports=("$output_dir"/seed_*/qualification.json)
if [ ! -e "${reports[0]}" ]; then
    echo "No qualification reports were produced." >&2
    exit 1
fi

python3 "$quality_gate" aggregate \
    --reports "${reports[@]}" \
    --required-runs "${#seeds[@]}" \
    --output "$output_dir/pilot_summary.json"
