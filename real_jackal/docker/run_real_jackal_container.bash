#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_JACKAL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

IMAGE="${IMAGE:-moai-jackal-spubert:social005}"
CONTAINER_NAME="${CONTAINER_NAME:-moai_jackal_spubert}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
MAP_DIR="${MAP_DIR:-$HOME/jackal_maps}"
LOG_DIR="${LOG_DIR:-$HOME/jackal_logs}"
JACKAL_DDS_PROFILE="${JACKAL_DDS_PROFILE:-}"
CAPSTONE_CALIB="${CAPSTONE_CALIB:-}"

runtime_args=(
  --env "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
  --volume "$REAL_JACKAL_DIR/scripts:/root/jackal_runtime/scripts:ro"
  --volume "$REAL_JACKAL_DIR/ros2_ws/src/moai_jackal_spubert/config/nav2_route_planner.yaml:/root/robot_ws/install/moai_jackal_spubert/share/moai_jackal_spubert/config/nav2_route_planner.yaml:ro"
)

if [[ -n "$JACKAL_DDS_PROFILE" ]]; then
  if [[ ! -r "$JACKAL_DDS_PROFILE" ]]; then
    echo "Jackal Fast DDS profile is not readable: $JACKAL_DDS_PROFILE" >&2
    exit 1
  fi
  JACKAL_DDS_PROFILE="$(realpath "$JACKAL_DDS_PROFILE")"
  dds_container_profile="/root/jackal_runtime/fastdds_laptop.xml"
  runtime_args+=(
    --env "ROS_LOCALHOST_ONLY=0"
    --env "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
    --env "FASTRTPS_DEFAULT_PROFILES_FILE=$dds_container_profile"
    --env "FASTDDS_DEFAULT_PROFILES_FILE=$dds_container_profile"
    --volume "$JACKAL_DDS_PROFILE:$dds_container_profile:ro"
  )
fi

if [[ -n "$CAPSTONE_CALIB" ]]; then
  if [[ ! -s "$CAPSTONE_CALIB/extrinsic.txt" || ! -s "$CAPSTONE_CALIB/intrinsic.txt" ]]; then
    echo "Calibration directory must contain non-empty extrinsic.txt and intrinsic.txt: $CAPSTONE_CALIB" >&2
    exit 1
  fi
  CAPSTONE_CALIB="$(realpath "$CAPSTONE_CALIB")"
  runtime_args+=(--volume "$CAPSTONE_CALIB:/root/data/calib:ro")
fi

mkdir -p "$MAP_DIR" "$LOG_DIR"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  echo "Stop it explicitly before creating another deployment container." >&2
  exit 1
fi

gui_args=()
if [[ "${MOAI_ENABLE_GUI:-0}" == "1" ]]; then
  gui_args+=(--env "DISPLAY=${DISPLAY:-}")
  gui_args+=(--volume /tmp/.X11-unix:/tmp/.X11-unix)
fi

docker run -dit --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --network host \
  --privileged \
  --volume /dev:/dev \
  --volume "$MAP_DIR:/root/jackal_maps:rw" \
  --volume "$LOG_DIR:/root/jackal_logs" \
  "${runtime_args[@]}" \
  "${gui_args[@]}" \
  "$IMAGE" \
  sleep infinity

echo "Container started: $CONTAINER_NAME (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
if [[ -n "$JACKAL_DDS_PROFILE" ]]; then
  echo "Jackal Fast DDS profile: $JACKAL_DDS_PROFILE"
fi
echo "Open a shell: docker exec -it $CONTAINER_NAME bash"
