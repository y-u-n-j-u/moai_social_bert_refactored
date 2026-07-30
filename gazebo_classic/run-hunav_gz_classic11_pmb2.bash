#!/usr/bin/env bash
set -euo pipefail

container_name="hunavsim_pmb2"

cleanup_container() {
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        echo "Removing existing container: $container_name"
        docker rm -f "$container_name" >/dev/null
    fi
}

trap cleanup_container EXIT INT TERM
cleanup_container
xhost +local:docker >/dev/null 2>&1 || true

# Resolve paths relative to this script so the launcher works from any directory.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cwd="$script_dir"
default_static_map_odom=False
if [ "${HUNAV_ROBOT_TYPE:-pmb2}" = "jackal" ]; then
    default_static_map_odom=True
    if [ "${HUNAV_JACKAL_LASER:-0}" = "1" ]; then
        default_static_map_odom=False
    fi
fi

gpu_args=()
case "${HUNAV_DOCKER_USE_GPU:-true}" in
    1|true|TRUE|yes|YES|on|ON)
        gpu_args=(--gpus all)
        ;;
esac

docker run -it \
    --rm \
    --name "$container_name" \
    "${gpu_args[@]}" \
    --env="RMW_IMPLEMENTATION=rmw_fastrtps_cpp" \
    --env="HUNAV_ROBOT_TYPE=${HUNAV_ROBOT_TYPE:-pmb2}" \
    --env="HUNAV_ROBOT_NAME=${HUNAV_ROBOT_NAME:-${HUNAV_ROBOT_TYPE:-pmb2}}" \
    --env="HUNAV_AGENT_MOTION_MODEL=${HUNAV_AGENT_MOTION_MODEL:-hunav}" \
    --env="HUNAV_NAVIGATION=${HUNAV_NAVIGATION:-False}" \
    --env="HUNAV_USE_GAZEBO_GUI=${HUNAV_USE_GAZEBO_GUI:-True}" \
    --env="HUNAV_USE_RVIZ=${HUNAV_USE_RVIZ:-True}" \
    --env="HUNAV_USE_STATIC_MAP_ODOM=${HUNAV_USE_STATIC_MAP_ODOM:-$default_static_map_odom}" \
    --env="HUNAV_UPDATE_RATE=${HUNAV_UPDATE_RATE:-10.0}" \
    --env="HUNAV_JACKAL_LASER=${HUNAV_JACKAL_LASER:-0}" \
    --env="HUNAV_JACKAL_REALSENSE=${HUNAV_JACKAL_REALSENSE:-0}" \
    --env="HUNAV_JACKAL_SPUBERT_CONTROLLER=${HUNAV_JACKAL_SPUBERT_CONTROLLER:-False}" \
    --env="HUNAV_ROBOT_PATH_PLANNER=${HUNAV_ROBOT_PATH_PLANNER:-nav2}" \
    --env="HUNAV_ROBOT_SPUBERT_REPO_PATH=${HUNAV_ROBOT_SPUBERT_REPO_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored}" \
    --env="HUNAV_ROBOT_SPUBERT_CONFIG_PATH=${HUNAV_ROBOT_SPUBERT_CONFIG_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml}" \
    --env="HUNAV_ROBOT_SPUBERT_CHECKPOINT=${HUNAV_ROBOT_SPUBERT_CHECKPOINT:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_moai_gazebo_guided_mgp_fs/model_best.pth}" \
    --env="HUNAV_ROBOT_SPUBERT_CUDA=${HUNAV_ROBOT_SPUBERT_CUDA:-true}" \
    --env="HUNAV_ROBOT_SPUBERT_D_SAMPLE=${HUNAV_ROBOT_SPUBERT_D_SAMPLE:-40}" \
    --env="HUNAV_ROBOT_SPUBERT_REPLAN_PERIOD=${HUNAV_ROBOT_SPUBERT_REPLAN_PERIOD:-0.8}" \
    --env="HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2=${HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2:-True}" \
    --env="HUNAV_ROBOT_SAVE_TRAINING_PKL=${HUNAV_ROBOT_SAVE_TRAINING_PKL:-False}" \
    --env="HUNAV_ROBOT_TRAINING_PKL_PATH=${HUNAV_ROBOT_TRAINING_PKL_PATH:-/home/hunav_gz_classic_ws/moai_recordings/robot_target_all_trajs.pkl}" \
    --env="HUNAV_ROBOT_TRAINING_RECORD_DT=${HUNAV_ROBOT_TRAINING_RECORD_DT:-0.4}" \
    --env="HUNAV_ROBOT_TRAINING_SAMPLE_STRIDE=${HUNAV_ROBOT_TRAINING_SAMPLE_STRIDE:-1}" \
    --env="HUNAV_ROBOT_TRAINING_FLUSH_EVERY=${HUNAV_ROBOT_TRAINING_FLUSH_EVERY:-10}" \
    --env="HUNAV_ROBOT_TRAINING_MAX_SAMPLES=${HUNAV_ROBOT_TRAINING_MAX_SAMPLES:-0}" \
    --env="HUNAV_SOCIAL_BERT_PREDICTOR=${HUNAV_SOCIAL_BERT_PREDICTOR:-spubert}" \
    --env="HUNAV_SPUBERT_MODEL_PATH=${HUNAV_SPUBERT_MODEL_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/ethucy/univ/spubert.pth}" \
    --env="HUNAV_SPUBERT_REPO_PATH=${HUNAV_SPUBERT_REPO_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/runtime/SPUBERT}" \
    --env="HUNAV_MOAI_SPUBERT_REPO_PATH=${HUNAV_MOAI_SPUBERT_REPO_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored}" \
    --env="HUNAV_SPUBERT_CUDA=${HUNAV_SPUBERT_CUDA:-true}" \
    --env="HUNAV_SPUBERT_K_SAMPLE=${HUNAV_SPUBERT_K_SAMPLE:-3}" \
    --env="HUNAV_SPUBERT_D_SAMPLE=${HUNAV_SPUBERT_D_SAMPLE:-40}" \
    --env="HUNAV_SPUBERT_CACHE_TTL=${HUNAV_SPUBERT_CACHE_TTL:-5.0}" \
    --env="HUNAV_SPUBERT_SYNC_ON_CACHE_MISS=${HUNAV_SPUBERT_SYNC_ON_CACHE_MISS:-False}" \
    --env="HUNAV_SPUBERT_CANDIDATE_SELECTION_MODE=${HUNAV_SPUBERT_CANDIDATE_SELECTION_MODE:-guidance_point}" \
    --env="HUNAV_SPUBERT_GUIDANCE_RADIUS=${HUNAV_SPUBERT_GUIDANCE_RADIUS:-8.0}" \
    --env="HUNAV_SOCIAL_BERT_LOOKAHEAD_STEP=${HUNAV_SOCIAL_BERT_LOOKAHEAD_STEP:-6}" \
    --env="HUNAV_SOCIAL_BERT_MIN_GOAL_SPEED=${HUNAV_SOCIAL_BERT_MIN_GOAL_SPEED:-0.35}" \
    --env="HUNAV_SOCIAL_BERT_MAX_SPEED=${HUNAV_SOCIAL_BERT_MAX_SPEED:-1.8}" \
    --env="HUNAV_SOCIAL_BERT_MAX_ACCEL=${HUNAV_SOCIAL_BERT_MAX_ACCEL:-1.8}" \
    --env="HUNAV_SOCIAL_BERT_GOAL_VELOCITY_BLEND=${HUNAV_SOCIAL_BERT_GOAL_VELOCITY_BLEND:-0.25}" \
    --env="HUNAV_SOCIAL_BERT_GOAL_ARRIVAL_DISTANCE=${HUNAV_SOCIAL_BERT_GOAL_ARRIVAL_DISTANCE:-1.5}" \
    --env="HUNAV_SOCIAL_BERT_GOAL_ARRIVAL_MIN_SPEED=${HUNAV_SOCIAL_BERT_GOAL_ARRIVAL_MIN_SPEED:-0.08}" \
    --env="HUNAV_SOCIAL_BERT_ROBOT_PERSONAL_SPACE=${HUNAV_SOCIAL_BERT_ROBOT_PERSONAL_SPACE:-1.25}" \
    --env="HUNAV_SOCIAL_BERT_ROBOT_COLLISION_BUFFER=${HUNAV_SOCIAL_BERT_ROBOT_COLLISION_BUFFER:-0.35}" \
    --env="HUNAV_SOCIAL_BERT_ROBOT_HARD_COLLISION_GUARD=${HUNAV_SOCIAL_BERT_ROBOT_HARD_COLLISION_GUARD:-True}" \
    --env="HUNAV_SOCIAL_BERT_AGENT_PERSONAL_SPACE=${HUNAV_SOCIAL_BERT_AGENT_PERSONAL_SPACE:-1.25}" \
    --env="HUNAV_SOCIAL_BERT_AGENT_AVOIDANCE_GAIN=${HUNAV_SOCIAL_BERT_AGENT_AVOIDANCE_GAIN:-0.9}" \
    --env="HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_DISTANCE=${HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_DISTANCE:-1.35}" \
    --env="HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_GAIN=${HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_GAIN:-0.45}" \
    --env="HUNAV_SOCIAL_BERT_OBSTACLE_COLLISION_BUFFER=${HUNAV_SOCIAL_BERT_OBSTACLE_COLLISION_BUFFER:-0.42}" \
    --env="HUNAV_SOCIAL_BERT_VELOCITY_SMOOTHING_ALPHA=${HUNAV_SOCIAL_BERT_VELOCITY_SMOOTHING_ALPHA:-0.35}" \
    --env="HUNAV_SOCIAL_BERT_COLLISION_BUFFER=${HUNAV_SOCIAL_BERT_COLLISION_BUFFER:-0.16}" \
    --env="HUNAV_SOCIAL_BERT_MAX_LATERAL_SPEED_RATIO=${HUNAV_SOCIAL_BERT_MAX_LATERAL_SPEED_RATIO:-0.70}" \
    --env="HUNAV_SOCIAL_BERT_MAX_YAW_RATE=${HUNAV_SOCIAL_BERT_MAX_YAW_RATE:-1.0}" \
    --env="HUNAV_SOCIAL_BERT_CONTROLLER_AVOIDANCE_ENABLED=${HUNAV_SOCIAL_BERT_CONTROLLER_AVOIDANCE_ENABLED:-False}" \
    --env="HUNAV_SOCIAL_BERT_SAVE_TRAINING_PKL=${HUNAV_SOCIAL_BERT_SAVE_TRAINING_PKL:-False}" \
    --env="HUNAV_SOCIAL_BERT_TRAINING_PKL_PATH=${HUNAV_SOCIAL_BERT_TRAINING_PKL_PATH:-/home/hunav_gz_classic_ws/moai_recordings/spubert_guidance_point_all_trajs.pkl}" \
    --env="HUNAV_SOCIAL_BERT_TRAINING_RECORD_DT=${HUNAV_SOCIAL_BERT_TRAINING_RECORD_DT:-0.4}" \
    --env="HUNAV_SOCIAL_BERT_TRAINING_SAMPLE_STRIDE=${HUNAV_SOCIAL_BERT_TRAINING_SAMPLE_STRIDE:-1}" \
    --env="HUNAV_SOCIAL_BERT_TRAINING_FLUSH_EVERY=${HUNAV_SOCIAL_BERT_TRAINING_FLUSH_EVERY:-10}" \
    --env="HUNAV_SOCIAL_BERT_TRAINING_MAX_SAMPLES=${HUNAV_SOCIAL_BERT_TRAINING_MAX_SAMPLES:-0}" \
    --env="HUNAV_SOCIAL_BERT_PUBLISH_DEBUG_MARKERS=${HUNAV_SOCIAL_BERT_PUBLISH_DEBUG_MARKERS:-False}" \
    --env="HUNAV_SOCIAL_BERT_DEBUG_MARKER_PUBLISH_EVERY=${HUNAV_SOCIAL_BERT_DEBUG_MARKER_PUBLISH_EVERY:-5}" \
    --env="HUNAV_SOCIAL_BERT_DEBUG_FOCUS_AGENT_ID=${HUNAV_SOCIAL_BERT_DEBUG_FOCUS_AGENT_ID:-1}" \
    --env="HUNAV_SOCIAL_BERT_DEBUG_FOCUS_AGENT_ONLY=${HUNAV_SOCIAL_BERT_DEBUG_FOCUS_AGENT_ONLY:-False}" \
    --env="HUNAV_SOCIAL_BERT_DEBUG_GUIDANCE_ONLY=${HUNAV_SOCIAL_BERT_DEBUG_GUIDANCE_ONLY:-False}" \
    --env="HUNAV_SOCIAL_BERT_DEBUG_SHOW_ALL_CANDIDATE_PATHS=${HUNAV_SOCIAL_BERT_DEBUG_SHOW_ALL_CANDIDATE_PATHS:-False}" \
    --env="HUNAV_GZPOSE_X=${HUNAV_GZPOSE_X:-0.0}" \
    --env="HUNAV_GZPOSE_Y=${HUNAV_GZPOSE_Y:-0.0}" \
    --env="HUNAV_GZPOSE_Z=${HUNAV_GZPOSE_Z:-0.25}" \
    --env="HUNAV_GZPOSE_YAW=${HUNAV_GZPOSE_YAW:-0.0}" \
    --env="DISPLAY=${DISPLAY:-:0}" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="__GLX_VENDOR_LIBRARY_NAME=nvidia" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --net=host \
    --privileged \
    --mount type=bind,source=$cwd/entrypoint.bash,target=/entrypoint.bash,readonly \
    --mount type=bind,source=$cwd/hunav_gz_classic_ws,target=/home/hunav_gz_classic_ws \
    --mount type=bind,source=$cwd/pmb2_overrides/sick_tim571_laser_gpu.gazebo.xacro,target=/home/pmb2_ws/install/pmb2_description/share/pmb2_description/urdf/sensors/sick_tim571_laser_gpu.gazebo.xacro,readonly \
    --mount type=bind,source=$cwd/pmb2_overrides/mobile_base_controller_public_sim_fast.yaml,target=/home/pmb2_ws/install/pmb2_controller_configuration/share/pmb2_controller_configuration/config/mobile_base_controller_public_sim.yaml,readonly \
    pmb2_hunavsim \
    bash
