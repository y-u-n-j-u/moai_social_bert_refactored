#!/usr/bin/env bash
set -euo pipefail

# Read the old deployment before choosing its image or making a manual snapshot.
# No container/ROS process is stopped, started, paused or changed here.
SOURCE_CONTAINER="${SOURCE_CONTAINER:-moai_jackal_spubert}"
AUDIT_ROOT="${AUDIT_ROOT:-$HOME/jackal_deployment_audits}"
AUDIT_DIR="$AUDIT_ROOT/$(date -u +%Y%m%d-%H%M%S)-$$"
mkdir -p "$AUDIT_DIR"
docker inspect "$SOURCE_CONTAINER" > "$AUDIT_DIR/container-inspect.json"
BASE_ID="$(docker inspect --format '{{.Image}}' "$SOURCE_CONTAINER")"
printf '%s\n' "$BASE_ID" > "$AUDIT_DIR/base-image-id.txt"
docker image inspect "$BASE_ID" > "$AUDIT_DIR/image-inspect.json"
docker diff "$SOURCE_CONTAINER" > "$AUDIT_DIR/container-filesystem-diff.txt"
docker inspect --format '{{json .Mounts}}' "$SOURCE_CONTAINER" > "$AUDIT_DIR/mounts.json"
if [[ "$(docker inspect --format '{{.State.Running}}' "$SOURCE_CONTAINER")" != true ]]; then
  echo "Metadata saved: $AUDIT_DIR" >&2
  echo "The container is stopped; source/import audit was not performed. Do not start robot processes for this script." >&2
  exit 1
fi

# --dereference preserves the actual contents behind colcon symlinks, including
# manually edited installed files. Host calibration/maps remain untouched.
docker exec "$SOURCE_CONTAINER" bash --noprofile --norc -c '
  set -e
  paths=()
  for p in root/robot_ws/src/moai_jackal_spubert \
           root/robot_ws/install/moai_jackal_spubert \
           root/moai_stability_ws/src/moai_jackal_spubert \
           root/moai_stability_ws/install/moai_jackal_spubert root/data/calib; do
    [[ ! -e "/$p" ]] || paths+=("$p")
  done
  ((${#paths[@]} > 0)) || exit 1
  tar --exclude="__pycache__" --exclude="*.pyc" -chf - -C / "${paths[@]}"
' > "$AUDIT_DIR/package-and-calibration.tar"

docker exec -i "$SOURCE_CONTAINER" bash --noprofile --norc -c '
  set -e
  source /opt/ros/humble/setup.bash
  source /root/robot_ws/install/setup.bash
  if [[ -n "${MOAI_STABILITY_OVERLAY:-}" ]]; then source "$MOAI_STABILITY_OVERLAY"; fi
  exec /usr/bin/python3 -
' > "$AUDIT_DIR/source-installed-imports.json" <<'PY'
import hashlib, importlib.util, json, os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
name = "moai_jackal_spubert"
spec = importlib.util.find_spec(name)
result = {"package_import": spec.origin if spec else None,
          "package_share": get_package_share_directory(name), "files": {}}
roots = [Path("/root/robot_ws/src") / name, Path("/root/robot_ws/install") / name,
         Path("/root/moai_stability_ws/src") / name,
         Path("/root/moai_stability_ws/install") / name]
if spec and spec.submodule_search_locations:
    roots += [Path(p) for p in spec.submodule_search_locations]
for root in roots:
    if not root.is_dir():
        continue
    for directory, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", "test")]
        for filename in files:
            path = Path(directory) / filename
            if path.suffix not in (".py", ".yaml", ".rviz", ".xml", ".cfg"):
                continue
            result["files"][str(path)] = {
                "resolved": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(result, indent=2, sort_keys=True))
PY
printf 'Audit saved: %s\nExisting image: %s\n' "$AUDIT_DIR" "$BASE_ID"
echo 'Review container-filesystem-diff.txt and the package backup before selecting BASE_IMAGE.'
echo 'Image-only reuse does not include edits made inside the container writable layer.'
