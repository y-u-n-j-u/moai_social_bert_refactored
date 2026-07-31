#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/prepare_dataset.sh RAW_PKL MAP [RUN_NAME]

Examples:
  ./scripts/prepare_dataset.sh \
    gazebo_classic/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl \
    training_corridor

  ./scripts/prepare_dataset.sh /absolute/run.pkl /absolute/map.yaml run_001

MAP may be a map name from hunav_gazebo_wrapper/maps or a YAML path.
All processed runs already present in data/processed/gazebo/runs are merged,
then split by (recording_id, RViz-goal episode_id).
EOF
}

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_root="${repo_root}/gazebo_classic"
workspace="${sim_root}/hunav_gz_classic_ws"
model_root="${workspace}/src/moai_social_bert_refactored"
map_root="${workspace}/src/hunav_gazebo_wrapper/maps"

raw_input="$1"
if [[ "${raw_input}" != /* ]]; then
  raw_input="${repo_root}/${raw_input}"
fi
raw_input="$(realpath "${raw_input}")"

map_input="$2"
if [[ "${map_input}" != */* && "${map_input}" != *.yaml ]]; then
  map_input="${map_root}/${map_input}.yaml"
elif [[ "${map_input}" != /* ]]; then
  map_input="${repo_root}/${map_input}"
fi
map_input="$(realpath "${map_input}")"

run_name="${3:-$(basename "${raw_input}" .pkl)}"
if [[ ! "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run name may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
if ! docker image inspect pmb2_hunavsim >/dev/null 2>&1; then
  echo "Image pmb2_hunavsim is missing. Run ./scripts/setup.sh first." >&2
  exit 2
fi
if ! docker image inspect moai-social-bert:cu124 >/dev/null 2>&1; then
  echo "Image moai-social-bert:cu124 is missing. Run ./scripts/setup.sh first." >&2
  exit 2
fi

runs_root="${model_root}/data/processed/gazebo/runs"
run_output="${runs_root}/${run_name}"
splits_root="${model_root}/data/processed/gazebo/splits_episode"
mkdir -p "${run_output}" "${splits_root}"

echo "[1/3] Postprocessing robot, pedestrians, RViz goal, guidance point, and local map"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env MPLBACKEND=Agg \
  --env MPLCONFIGDIR=/tmp/moai-matplotlib-cache \
  --volume "${workspace}:/home/hunav_gz_classic_ws:ro" \
  --volume "$(dirname "${raw_input}"):/input:ro" \
  --volume "$(dirname "${map_input}"):/maps:ro" \
  --volume "${run_output}:/output" \
  --entrypoint python3 \
  pmb2_hunavsim \
  /home/hunav_gz_classic_ws/src/moai_hunav_bridge/scripts/postprocess_pedestrian_dataset.py \
  --input "/input/$(basename "${raw_input}")" \
  --map-yaml "/maps/$(basename "${map_input}")" \
  --out-dir /output \
  --name "${run_name}" \
  --map-size-m 20.0 \
  --map-grid-size 32 \
  --guidance-radius 8.0

echo "[2/3] Merging all runs and creating episode-safe train/val/test splits"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "${model_root}:/workspace:ro" \
  --volume "${runs_root}:/data/runs:ro" \
  --volume "${splits_root}:/data/splits" \
  --workdir /workspace \
  moai-social-bert:cu124 \
  python scripts/prepare_gazebo_splits.py \
  --input-dir /data/runs \
  --out-dir /data/splits \
  --prefix pmb2_true_goal_gp_v1_clean_social

echo "[3/3] Validating every sample against the Yunju guided SPU-BERT input contract"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "${splits_root}:/data/gazebo:ro" \
  moai-social-bert:cu124 \
  python scripts/validate_gazebo_guided_dataset.py \
  --config configs/spubert/moai_social_nav_ext_scene_smoke_fs.yaml

echo
echo "Dataset ready: ${splits_root}"
echo "Next: ./scripts/train.sh --dry_run"
