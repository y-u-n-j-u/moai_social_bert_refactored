#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${HUNAVSIM_CONTAINER:-hunavsim_pmb2}"

container_names="$(docker ps --format '{{.Names}}')" || {
  echo "Could not access Docker. Check that Docker is running and your user can use it." >&2
  exit 1
}

if ! grep -qx "$CONTAINER_NAME" <<<"$container_names"; then
  echo "Container '$CONTAINER_NAME' is not running." >&2
  echo "Start it first with: cd gazebo_classic && ./run-hunav_gz_classic11_pmb2.bash" >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  exec docker exec -it "$CONTAINER_NAME" bash -lc '
    source /opt/ros/humble/setup.bash
    source /usr/share/gazebo/setup.sh
    source /home/pmb2_ws/install/setup.bash
    if [ -f /root/jackal_ws/install/setup.bash ]; then
      source /root/jackal_ws/install/setup.bash
    fi
    source /home/hunav_gz_classic_ws/install/setup.bash
    exec bash
  '
fi

quoted_cmd=$(printf '%q ' "$@")
docker_args=(exec)
if [ -t 0 ]; then
  docker_args+=(-it)
fi

exec docker "${docker_args[@]}" "$CONTAINER_NAME" bash -lc "
  source /opt/ros/humble/setup.bash
  source /usr/share/gazebo/setup.sh
  source /home/pmb2_ws/install/setup.bash
  if [ -f /root/jackal_ws/install/setup.bash ]; then
    source /root/jackal_ws/install/setup.bash
  fi
  source /home/hunav_gz_classic_ws/install/setup.bash
  $quoted_cmd
"
