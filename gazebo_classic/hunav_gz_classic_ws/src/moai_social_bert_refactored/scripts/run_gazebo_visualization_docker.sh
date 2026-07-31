#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gazebo_data_root="${GAZEBO_DATA_ROOT:-${repo_root}/data/processed/gazebo/splits_episode}"
checkpoint="${MOAI_CHECKPOINT:-${repo_root}/output/spubert_moai_gazebo_guided_mgp_fs/model_best.pth}"
output_dir="${GAZEBO_GUIDED_FIGURES:-${repo_root}/figures/gazebo_guided_mgp_results}"
config="${MOAI_CONFIG:-configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml}"
font_path="${MOAI_KOREAN_FONT:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}"
image="${MOAI_IMAGE:-moai-social-bert:cu124}"

if [[ ! -d "${gazebo_data_root}" ]]; then
  echo "Gazebo split directory not found: ${gazebo_data_root}" >&2
  echo "Set GAZEBO_DATA_ROOT to the directory containing the *_train/val/test.pkl files." >&2
  exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
docker build --tag "${image}" "${repo_root}"

font_mount=()
font_option=()
if [[ -f "${font_path}" ]]; then
  font_mount=(--volume "${font_path}:/fonts/NotoSansCJK-Regular.ttc:ro")
  font_option=(--font-path /fonts/NotoSansCJK-Regular.ttc)
fi

docker run --rm --gpus all --shm-size=4g \
  --volume "${repo_root}:/workspace:ro" \
  --volume "${gazebo_data_root}:/data/gazebo:ro" \
  --volume "${checkpoint}:/checkpoints/model_best.pth:ro" \
  --volume "${output_dir}:/results" \
  "${font_mount[@]}" \
  "${image}" \
  python scripts/visualize_gazebo_guided_results.py \
    --config "${config}" \
    --checkpoint /checkpoints/model_best.pth \
    --output-dir /results \
    "${font_option[@]}"
