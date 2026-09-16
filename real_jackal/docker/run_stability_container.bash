#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${IMAGE:?Set IMAGE to the versioned stability overlay image}"
: "${CAPSTONE_CALIB:?Set CAPSTONE_CALIB to the existing lab calibration directory}"
if [[ "$(docker image inspect --format '{{index .Config.Labels "org.moai.stability.overlay"}}' "$IMAGE")" != true ]]; then
  echo "IMAGE is not a stability overlay built by build_stability_overlay.bash" >&2
  exit 1
fi
export IMAGE CAPSTONE_CALIB
export MOAI_RUNTIME_FROM_IMAGE=1
# Keep the original container available for rollback; select the target explicitly.
export CONTAINER_NAME="${CONTAINER_NAME:-moai_jackal_spubert_stability}"
exec bash "$SCRIPT_DIR/run_real_jackal_with_dds.bash"
