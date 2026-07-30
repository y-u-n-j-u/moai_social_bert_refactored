#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${repo_root}/gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored"

exec "${model_root}/scripts/run_gazebo_validation_docker.sh" "$@"
