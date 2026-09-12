#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_JACKAL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_DIR="$(cd -- "$REAL_JACKAL_DIR/.." && pwd)"

"$REAL_JACKAL_DIR/scripts/configure_jackal_network.bash" check

export JACKAL_DDS_PROFILE="${JACKAL_DDS_PROFILE:-$REAL_JACKAL_DIR/config/fastdds_laptop_udp_discovery.xml}"
if [[ -z "${CAPSTONE_CALIB:-}" \
    && -s "$REPOSITORY_DIR/spu_deploy_docker/context/calib/extrinsic.txt" \
    && -s "$REPOSITORY_DIR/spu_deploy_docker/context/calib/intrinsic.txt" ]]; then
  export CAPSTONE_CALIB="$REPOSITORY_DIR/spu_deploy_docker/context/calib"
fi

exec "$SCRIPT_DIR/run_real_jackal_container.bash"
