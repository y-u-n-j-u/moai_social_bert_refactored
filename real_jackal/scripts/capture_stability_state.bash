#!/usr/bin/env bash
set -euo pipefail

# Run inside the new container. This records evidence, never sets parameters,
# publishes velocity, calls enable_motion or changes localization.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jackal_ros_env.bash"
OUTPUT_DIR="${OUTPUT_DIR:-/root/jackal_logs/stability_check_$(date -u +%Y%m%d-%H%M%S)}"
CMD_TOPIC="${CMD_TOPIC:-/spu_bert/cmd_vel_dryrun}"
mkdir -p "$OUTPUT_DIR"
/usr/bin/python3 "$SCRIPT_DIR/deployment_manifest.py" verify \
  | tee "$OUTPUT_DIR/deployment-verification.json"
cp /root/jackal_deployment/source_manifest.json "$OUTPUT_DIR/source_manifest.json"
failures=0
capture() {
  local name="$1"
  shift
  if timeout 30 "$@" > "$OUTPUT_DIR/$name" 2>&1; then
    printf '[PASS] %s\n' "$name"
  else
    printf '[INCOMPLETE] %s (see saved output)\n' "$name" >&2
    failures=$((failures + 1))
  fi
}
capture bridge-parameters.yaml ros2 param dump /real_jackal_spubert_bridge
capture tracker-parameters.yaml ros2 param dump /safe_path_tracker
capture runtime-status.txt ros2 topic echo /spu_bert/runtime_status --once
capture tracker-status.txt ros2 topic echo /spu_bert/tracker_status --once
capture tracker-diagnostics.txt ros2 topic echo /spu_bert/tracker_diagnostics --once
capture plan-context.txt ros2 topic echo /spu_bert/plan_context std_msgs/msg/String \
  --once --qos-durability transient_local --qos-reliability reliable
# No completion event is normal before a goal has been reached. Record any
# latched event, but do not treat its absence as a failed pre-drive check.
if ! timeout 30 ros2 topic echo /spu_bert/goal_completion std_msgs/msg/String \
    --once --qos-durability transient_local --qos-reliability reliable \
    > "$OUTPUT_DIR/goal-completion.txt" 2>&1; then
  printf 'No completion event captured; expected if no goal has completed.\n' \
    >> "$OUTPUT_DIR/goal-completion.txt"
fi
capture cmd-vel-endpoints.txt ros2 topic info "$CMD_TOPIC" --verbose
if [[ -f /root/jackal_logs/spubert_real_jackal_diagnostics.jsonl ]]; then
  tail -n 200 /root/jackal_logs/spubert_real_jackal_diagnostics.jsonl \
    > "$OUTPUT_DIR/bridge-diagnostics-tail.jsonl"
fi
if [[ -f /root/jackal_logs/tracker_stability.jsonl ]]; then
  tail -n 200 /root/jackal_logs/tracker_stability.jsonl \
    > "$OUTPUT_DIR/tracker-diagnostics-tail.jsonl"
fi
printf 'Evidence saved: %s\n' "$OUTPUT_DIR"
echo 'A source hash PASS is not a drive validation. Review parameters, states and bag data.'
((failures == 0))
