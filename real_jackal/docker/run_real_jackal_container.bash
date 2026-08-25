#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-moai-jackal-spubert:social005}"
CONTAINER_NAME="${CONTAINER_NAME:-moai_jackal_spubert}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
MAP_DIR="${MAP_DIR:-$HOME/jackal_maps}"
LOG_DIR="${LOG_DIR:-$HOME/jackal_logs}"

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
  --env "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" \
  "${gui_args[@]}" \
  "$IMAGE" \
  sleep infinity

echo "Container started: $CONTAINER_NAME (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
echo "Open a shell: docker exec -it $CONTAINER_NAME bash"
