#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-moai_jackal_spubert}"

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null \
    | grep -Fxq true; then
  printf 'Container is not running: %s\n' "$CONTAINER_NAME" >&2
  exit 1
fi

docker exec -it "$CONTAINER_NAME" \
  bash --noprofile --norc -c \
  'source /root/jackal_runtime/scripts/jackal_ros_env.bash && exec bash --noprofile --norc -i'
