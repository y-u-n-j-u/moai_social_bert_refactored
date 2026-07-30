#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gazebo_data_root="${GAZEBO_DATA_ROOT:-${repo_root}/data/processed/gazebo/splits_episode}"
image="${MOAI_IMAGE:-moai-social-bert:cu124}"
config="${MOAI_CONFIG:-configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml}"

if [[ ! -d "${gazebo_data_root}" ]]; then
  echo "Gazebo split directory not found: ${gazebo_data_root}" >&2
  echo "Set GAZEBO_DATA_ROOT to the directory containing the *_train/val/test.pkl files." >&2
  exit 2
fi

docker build --tag "${image}" "${repo_root}"
docker run --rm --gpus all --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env MPLCONFIGDIR=/tmp/moai-matplotlib-cache \
  --volume "${repo_root}:/workspace" \
  --volume "${gazebo_data_root}:/data/gazebo:ro" \
  --workdir /workspace \
  "${image}" \
  python train.py --config "${config}" "$@"
