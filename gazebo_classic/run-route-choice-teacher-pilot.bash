#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$script_dir/hunav_gz_classic_ws"
recordings_dir="$workspace_dir/moai_recordings"
pilot_name="${HUNAV_PILOT_NAME:-route_choice_safe_teacher_v3}"
output_dir="$recordings_dir/$pilot_name"
container_output_dir="/home/hunav_gz_classic_ws/moai_recordings/$pilot_name"
scenario="${HUNAV_PILOT_SCENARIO:-agents_training_route_choice_safe_teacher_low.yaml}"
seeds_text="${HUNAV_PILOT_SEEDS:-201 202 203 204 205 206 207 208 209 210}"
run_timeout="${HUNAV_PILOT_RUN_TIMEOUT:-180}"
max_attempts="${HUNAV_PILOT_MAX_ATTEMPTS:-2}"
profile_offset="${HUNAV_PILOT_PROFILE_OFFSET:-0}"
local_map_size_m="${HUNAV_TRAINING_LOCAL_MAP_SIZE_M:-20.0}"
local_map_grid_size="${HUNAV_TRAINING_LOCAL_MAP_GRID_SIZE:-32}"
postprocessor="$workspace_dir/src/moai_hunav_bridge/scripts/postprocess_pedestrian_dataset.py"
quality_gate="$script_dir/scripts/teacher_quality_gate.py"
map_yaml="$workspace_dir/src/hunav_gazebo_wrapper/maps/training_route_choice.yaml"

mkdir -p "$output_dir"
read -r -a seeds <<< "$seeds_text"

if ! [[ "$profile_offset" =~ ^[0-9]+$ ]]; then
    echo "HUNAV_PILOT_PROFILE_OFFSET must be a non-negative integer" >&2
    exit 2
fi

# Direct start-goal lines intersect the asymmetric block. Positive y profiles
# force the upper bypass; negative y profiles force the lower bypass. Varying
# start x changes encounter timing without changing the held-out test map.
start_x_profiles=(-8.6 -8.4 -8.2 -8.0 -7.8 -7.6 -7.4 -7.2 -7.0 -6.8)
robot_y_profiles=(-1.6 1.6 -1.4 1.4 -1.2 1.2 -1.5 1.5 -1.3 1.3)
run_index=0

stop_container() {
    docker stop hunavsim_pmb2 >/dev/null 2>&1 || true
}
trap stop_container EXIT
trap 'stop_container; exit 130' INT TERM

for seed in "${seeds[@]}"; do
    profile_index=$(((run_index + profile_offset) % ${#start_x_profiles[@]}))
    start_x="${start_x_profiles[$profile_index]}"
    robot_y="${robot_y_profiles[$profile_index]}"
    goal_waypoints="9.0,${robot_y};${start_x},${robot_y}"
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
        echo "[route-choice-teacher] seed=$seed profile=$profile_index attempt=$attempt start=($start_x,$robot_y) scenario=$scenario"
        (
            HUNAV_DOCKER_TTY=false \
            HUNAV_DOCKER_USE_GPU=false \
            HUNAV_SCENARIO="$scenario" \
            HUNAV_NAVIGATION=True \
            HUNAV_PMB2_NAV_PARAMS_FILE=/home/hunav_gz_classic_ws/install/hunav_gazebo_wrapper/share/hunav_gazebo_wrapper/launch/pmb2_params/pmb2_nav_route_choice_teacher.yaml \
            HUNAV_USE_GAZEBO_GUI=False \
            HUNAV_USE_RVIZ=False \
            HUNAV_USE_EVALUATOR=False \
            HUNAV_PEDESTRIANS_AVOID_ROBOT=False \
            HUNAV_USE_NAVGOAL_TO_START=True \
            HUNAV_HUMAN_OBSTACLE_PUBLISH_RATE=10.0 \
            HUNAV_HUMAN_OBSTACLE_CURRENT_RING_POINTS=16 \
            HUNAV_HUMAN_OBSTACLE_SAFETY_MARGIN=0.65 \
            HUNAV_HUMAN_OBSTACLE_PREDICTED_RING_POINTS=16 \
            HUNAV_HUMAN_OBSTACLE_PREDICTION_HORIZON=2.4 \
            HUNAV_HUMAN_OBSTACLE_PREDICTION_STEP=0.3 \
            HUNAV_GZPOSE_X="$start_x" \
            HUNAV_GZPOSE_Y="$robot_y" \
            HUNAV_GZPOSE_YAW=0.0 \
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
            HUNAV_AUTO_GOAL_TIMEOUT=75.0 \
            HUNAV_AUTO_GOAL_NO_PROGRESS_TIMEOUT=24.0 \
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
                echo "[route-choice-teacher] seed=$seed attempt=$attempt timed out" >&2
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
        echo "[route-choice-teacher] seed=$seed produced no completed recording" >&2
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

    python3 "$quality_gate" evaluate \
        --raw "$raw_path" \
        --processed-summary "$processed_dir/${run_name}_summary.json" \
        --expected-coupling one-way \
        --output "$report_path" >/dev/null || true
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
