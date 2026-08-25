#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
COLLEAGUE_REPO="${COLLEAGUE_REPO:-/home/junwoo/capstone/spu_deploy_docker_colleague}"
BASE_IMAGE="${BASE_IMAGE:-spubert_deploy:latest}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-moai-jackal-spubert:social005}"
MODEL_REPO="$PROJECT_ROOT/gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored"
CHECKPOINT="$MODEL_REPO/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth"
PRETRAIN_CHECKPOINT="$MODEL_REPO/output/spubert_ethucy_all_scene_pretrain/pretrain_model_best.pth"
PACKAGE_SOURCE="$PROJECT_ROOT/real_jackal/ros2_ws/src/moai_jackal_spubert"
EXPECTED_CHECKPOINT_SHA256="03d88a96db9abe46c47d98837a57785f01c197a7338f0784510e88bec6a10a95"

for required in \
  "$COLLEAGUE_REPO/Dockerfile" \
  "$MODEL_REPO/SPU-BERT/spubert/model.py" \
  "$CHECKPOINT" \
  "$PRETRAIN_CHECKPOINT" \
  "$PACKAGE_SOURCE/package.xml"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required deployment input: $required" >&2
    exit 1
  fi
done

actual_checkpoint_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_checkpoint_sha256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "Checkpoint SHA256 mismatch: $actual_checkpoint_sha256" >&2
  echo "Expected: $EXPECTED_CHECKPOINT_SHA256" >&2
  echo "Run 'git lfs pull' before building the deployment image." >&2
  exit 1
fi
if (( $(stat -c '%s' "$PRETRAIN_CHECKPOINT") < 1000000 )); then
  echo "Pretrain checkpoint is too small; run 'git lfs pull'." >&2
  exit 1
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Deployment inputs: PASS"
  echo "Checkpoint SHA256: $actual_checkpoint_sha256"
  exit 0
fi

mkdir -p "$COLLEAGUE_REPO/context/model_assets"
cp "$CHECKPOINT" "$COLLEAGUE_REPO/context/model_assets/model_best.pth"
cp "$PRETRAIN_CHECKPOINT" "$COLLEAGUE_REPO/context/model_assets/pretrain_model_best.pth"

if [[ "${SKIP_BASE_BUILD:-0}" != "1" ]]; then
  echo "[1/2] Building colleague sensor/perception base image: $BASE_IMAGE"
  docker build -t "$BASE_IMAGE" "$COLLEAGUE_REPO"
else
  echo "[1/2] Reusing existing base image: $BASE_IMAGE"
  docker image inspect "$BASE_IMAGE" >/dev/null
fi

BUILD_CONTEXT="$(mktemp -d /tmp/moai_jackal_build.XXXXXX)"
trap 'rm -rf "$BUILD_CONTEXT"' EXIT
mkdir -p "$BUILD_CONTEXT/model_repo" "$BUILD_CONTEXT/moai_jackal_spubert"

rsync -a \
  --exclude='.git' \
  --exclude='data' \
  --exclude='output' \
  --exclude='logs' \
  --exclude='figures' \
  --exclude='__pycache__' \
  "$MODEL_REPO/" "$BUILD_CONTEXT/model_repo/"
mkdir -p "$BUILD_CONTEXT/model_repo/output/spubert_route_gp_continue_b21_col01_social005"
cp "$CHECKPOINT" \
  "$BUILD_CONTEXT/model_repo/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth"
rsync -a --exclude='__pycache__' "$PACKAGE_SOURCE/" "$BUILD_CONTEXT/moai_jackal_spubert/"
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_CONTEXT/Dockerfile"
cp "$SCRIPT_DIR/docker_entrypoint.sh" "$BUILD_CONTEXT/docker_entrypoint.sh"
cp -r "$PROJECT_ROOT/real_jackal/scripts" "$BUILD_CONTEXT/scripts"

echo "[2/2] Building latest Adaptive-GP/Social-Loss image: $OUTPUT_IMAGE"
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$OUTPUT_IMAGE" \
  "$BUILD_CONTEXT"

echo "Built $OUTPUT_IMAGE"
echo "Checkpoint SHA256: $actual_checkpoint_sha256"
