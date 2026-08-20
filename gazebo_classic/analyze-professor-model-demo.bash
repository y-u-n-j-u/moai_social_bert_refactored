#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 RUN_NAME" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_name="$1"
diagnostics="${script_dir}/hunav_gz_classic_ws/moai_recordings/${run_name}_spubert_diagnostics.jsonl"
output_dir="${script_dir}/hunav_gz_classic_ws/moai_recordings/${run_name}_analysis"
map_yaml="${script_dir}/hunav_gz_classic_ws/src/hunav_gazebo_wrapper/maps/training_route_choice.yaml"
analyzer="${script_dir}/hunav_gz_classic_ws/src/moai_social_bert_refactored/scripts/analyze_spubert_runtime_diagnostics.py"

if [ ! -f "${diagnostics}" ]; then
  echo "Diagnostics not found: ${diagnostics}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/moai_mpl}" python3 "${analyzer}" \
  "${diagnostics}" \
  --output-dir "${output_dir}" \
  --map-yaml "${map_yaml}"
