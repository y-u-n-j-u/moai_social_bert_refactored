#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_root="${repo_root}/gazebo_classic"
model_root="${sim_root}/hunav_gz_classic_ws/src/moai_social_bert_refactored"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Engine first." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is unavailable or this user has no Docker permission." >&2
  echo "After adding the user to the docker group, log out and back in." >&2
  exit 2
fi

echo "[1/3] Checking NVIDIA GPU access from Docker"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "[2/3] Building HuNavSim + Gazebo + Nav2 image"
docker build \
  --tag pmb2_hunavsim \
  --file "${sim_root}/Dockerfile.hunav_gz_classic11_pmb2" \
  "${sim_root}"

echo "[3/3] Building guided SPU-BERT training image"
docker build \
  --tag moai-social-bert:cu124 \
  "${model_root}"

mkdir -p \
  "${sim_root}/hunav_gz_classic_ws/moai_recordings" \
  "${model_root}/data/processed/gazebo/runs" \
  "${model_root}/data/processed/gazebo/splits_episode"

echo
echo "Setup complete."
echo "Next: ./scripts/run_simulation.sh pmb2_run_001"
