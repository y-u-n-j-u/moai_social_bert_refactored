#!/usr/bin/env bash
set -euo pipefail

# Build only: no container is stopped, launched or armed by this script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_JACKAL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$REAL_JACKAL_DIR/.." && pwd)"
: "${BASE_IMAGE:?Set BASE_IMAGE to the audited existing deployment image ID or snapshot tag}"
PACKAGE_SOURCE="$REAL_JACKAL_DIR/ros2_ws/src/moai_jackal_spubert"

for command in docker python3 tar; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
BASE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"
# A local tag works with Dockerfile FROM across Docker/BuildKit versions, including
# when BASE_IMAGE was supplied as a bare image ID. Never retag an existing name.
BASE_PIN_TAG="moai-jackal-spubert-base:sha-${BASE_IMAGE_ID#sha256:}"
if docker image inspect "$BASE_PIN_TAG" >/dev/null 2>&1; then
  [[ "$(docker image inspect --format '{{.Id}}' "$BASE_PIN_TAG")" == "$BASE_IMAGE_ID" ]] \
    || { echo "Base snapshot tag collision: $BASE_PIN_TAG" >&2; exit 1; }
else
  docker image tag "$BASE_IMAGE_ID" "$BASE_PIN_TAG"
fi
BUILD_CONTEXT="$(mktemp -d /tmp/moai_stability_build.XXXXXX)"
trap 'rm -rf -- "$BUILD_CONTEXT"' EXIT
mkdir -p "$BUILD_CONTEXT/package" "$BUILD_CONTEXT/scripts"
tar -C "$PACKAGE_SOURCE" --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='test' --exclude='.pytest_cache' -cf - . \
  | tar -C "$BUILD_CONTEXT/package" -xf -
cp "$REAL_JACKAL_DIR/scripts/"*.bash "$BUILD_CONTEXT/scripts/"
cp "$REAL_JACKAL_DIR/scripts/"*.py "$BUILD_CONTEXT/scripts/"
cp "$SCRIPT_DIR/Dockerfile.stability" "$BUILD_CONTEXT/Dockerfile"
cp "$SCRIPT_DIR/docker_entrypoint.sh" "$BUILD_CONTEXT/docker_entrypoint.sh"
REVISION="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || printf 'file-transfer')"
SOURCE_SHA="$(python3 "$REAL_JACKAL_DIR/scripts/deployment_manifest.py" create \
  --package "$BUILD_CONTEXT/package" --output "$BUILD_CONTEXT/source_manifest.json" \
  --revision "$REVISION")"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-moai-jackal-spubert:stability-$(date -u +%Y%m%d-%H%M%S)-${SOURCE_SHA:0:12}}"
if docker image inspect "$OUTPUT_IMAGE" >/dev/null 2>&1; then
  echo "Refusing to replace existing image tag: $OUTPUT_IMAGE (choose a new tag)." >&2
  exit 1
fi
printf 'Base image: %s\nOutput image: %s\nPackage SHA256: %s\n' \
  "$BASE_IMAGE_ID" "$OUTPUT_IMAGE" "$SOURCE_SHA"
docker build --build-arg "BASE_IMAGE=$BASE_PIN_TAG" --build-arg "BASE_IMAGE_ID=$BASE_IMAGE_ID" \
  --build-arg "SOURCE_SHA=$SOURCE_SHA" --build-arg "SOURCE_REVISION=$REVISION" \
  -t "$OUTPUT_IMAGE" "$BUILD_CONTEXT"
printf 'Built %s\nNo ROS nodes or drive commands were started.\n' "$OUTPUT_IMAGE"
