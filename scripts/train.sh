#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${repo_root}/gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored"
split_root="${model_root}/data/processed/gazebo/splits_episode"

for split in train val test; do
  path="${split_root}/pmb2_true_goal_gp_v1_clean_social_${split}.pkl"
  if [[ ! -f "${path}" ]]; then
    echo "Missing dataset split: ${path}" >&2
    echo "Run ./scripts/prepare_dataset.sh first." >&2
    exit 2
  fi
done

exec "${model_root}/scripts/run_gazebo_train_docker.sh" "$@"
