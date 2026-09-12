#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-preflight}"
CMD_TOPIC="/j100_0519/cmd_vel"
PLATFORM_CMD_TOPIC="/j100_0519/platform/cmd_vel_unstamped"

if [[ "$MODE" != "preflight" && "$MODE" != "ready" ]]; then
  printf 'Usage: verify_jackal_link.bash <preflight|ready>\n' >&2
  exit 2
fi

source /root/jackal_runtime/scripts/jackal_ros_env.bash
set -u

failures=0
pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1" >&2; failures=$((failures + 1)); }

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1

if ! command -v ping >/dev/null 2>&1; then
  printf '[INFO] ping is not installed in this image; ROS discovery is the peer test.\n'
elif ping -c 2 -W 1 192.168.50.2 >/dev/null 2>&1; then
  pass 'Jackal DDS peer 192.168.50.2 is reachable'
else
  fail 'Jackal DDS peer 192.168.50.2 is unreachable'
fi

nodes=""
topics=""
graph_ready=false
for attempt in {1..12}; do
  nodes="$(timeout 12 ros2 node list 2>/dev/null || true)"
  topics="$(timeout 12 ros2 topic list -t 2>/dev/null || true)"
  if grep -q 'platform_velocity_controller' <<<"$nodes" \
      && grep -q 'twist_mux' <<<"$nodes" \
      && grep -Fq "$CMD_TOPIC [geometry_msgs/msg/Twist]" <<<"$topics" \
      && grep -Fq "$PLATFORM_CMD_TOPIC [geometry_msgs/msg/Twist]" <<<"$topics" \
      && grep -Fq '/j100_0519/platform/motors/feedback [' <<<"$topics" \
      && grep -Fq '/j100_0519/platform/motors/cmd_drive [' <<<"$topics" \
      && grep -Fq '/j100_0519/platform/mcu/status [' <<<"$topics" \
      && grep -Fq '/j100_0519/platform/emergency_stop [' <<<"$topics"; then
    graph_ready=true
    break
  fi
  sleep 5
done
if [[ "$graph_ready" == true ]]; then
  pass 'required Jackal DDS graph is discovered'
else
  fail 'required Jackal DDS graph was not discovered within 60 seconds'
fi

topic_type="$(timeout 8 ros2 topic type "$CMD_TOPIC" 2>/dev/null || true)"
if [[ "$topic_type" == "geometry_msgs/msg/Twist" ]]; then
  pass "$CMD_TOPIC type is geometry_msgs/msg/Twist"
else
  fail "$CMD_TOPIC type is '${topic_type:-unknown}'"
fi

topic_info="$(timeout 8 ros2 topic info "$CMD_TOPIC" --verbose 2>/dev/null || true)"
printf '%s\n' "$topic_info"
publisher_count="$(awk '/^Publisher count:/ {print $3; exit}' <<<"$topic_info")"
subscriber_count="$(awk '/^Subscription count:/ {print $3; exit}' <<<"$topic_info")"

if [[ "${subscriber_count:-0}" == "1" ]] \
    && grep -q 'Node name: twist_mux' <<<"$topic_info"; then
  pass 'twist_mux is the only cmd_vel subscriber'
else
  fail 'expected exactly one twist_mux subscriber on cmd_vel'
fi

platform_info="$(timeout 8 ros2 topic info "$PLATFORM_CMD_TOPIC" --verbose 2>/dev/null || true)"
printf '%s\n' "$platform_info"
platform_publisher_count="$(awk '/^Publisher count:/ {print $3; exit}' <<<"$platform_info")"
platform_subscriber_count="$(awk '/^Subscription count:/ {print $3; exit}' <<<"$platform_info")"
if [[ "${platform_publisher_count:-0}" == "1" ]] \
    && [[ "${platform_subscriber_count:-0}" == "1" ]] \
    && grep -q 'Node name: twist_mux' <<<"$platform_info" \
    && grep -q 'Node name: platform_velocity_controller' <<<"$platform_info"; then
  pass 'twist_mux -> platform_velocity_controller command chain is complete'
else
  fail 'platform command chain is incomplete'
fi

if [[ "$MODE" == "preflight" ]]; then
  if [[ "${publisher_count:-0}" == "0" ]]; then
    pass 'no cmd_vel publisher exists before SPU-BERT launch'
  else
    fail 'a cmd_vel publisher already exists before SPU-BERT launch'
  fi
else
  if [[ "${publisher_count:-0}" == "1" ]] \
      && grep -q 'Node name: safe_path_tracker' <<<"$topic_info"; then
    pass 'safe_path_tracker is the only cmd_vel publisher'
  else
    fail 'safe_path_tracker is not the sole cmd_vel publisher'
  fi
fi

feedback_info="$(timeout 8 ros2 topic info \
  /j100_0519/platform/motors/feedback --verbose 2>/dev/null || true)"
feedback_publishers="$(awk '/^Publisher count:/ {print $3; exit}' <<<"$feedback_info")"
if [[ "${feedback_publishers:-0}" -ge 1 ]]; then
  pass 'MCU motor feedback publisher is present'
else
  fail 'MCU motor feedback publisher is missing'
fi

drive_info="$(timeout 8 ros2 topic info \
  /j100_0519/platform/motors/cmd_drive --verbose 2>/dev/null || true)"
drive_subscribers="$(awk '/^Subscription count:/ {print $3; exit}' <<<"$drive_info")"
if [[ "${drive_subscribers:-0}" -ge 1 ]]; then
  pass 'MCU motor command subscriber is present'
else
  fail 'MCU motor command subscriber is missing'
fi

mcu_info="$(timeout 8 ros2 topic info \
  /j100_0519/platform/mcu/status --verbose 2>/dev/null || true)"
mcu_publishers="$(awk '/^Publisher count:/ {print $3; exit}' <<<"$mcu_info")"
if [[ "${mcu_publishers:-0}" -ge 1 ]]; then
  pass 'MCU status publisher is present'
else
  fail 'MCU status publisher is missing'
fi

estop_info="$(timeout 8 ros2 topic info \
  /j100_0519/platform/emergency_stop --verbose 2>/dev/null || true)"
estop_publishers="$(awk '/^Publisher count:/ {print $3; exit}' <<<"$estop_info")"
if [[ "${estop_publishers:-0}" -lt 1 ]]; then
  fail 'physical emergency stop publisher is missing'
else
  # The MCU endpoint may need several SIMPLE-discovery cycles after a fresh
  # participant starts.  Pin the known message type and sensor-data QoS so a
  # short graph-discovery delay is not misreported as an unsafe E-stop state.
  estop="$(timeout 60 ros2 topic echo /j100_0519/platform/emergency_stop \
    std_msgs/msg/Bool --once --field data \
    --qos-reliability best_effort --qos-durability volatile \
    2>/dev/null | tr -d '[:space:]-' || true)"
  case "${estop,,}" in
    false) pass 'physical emergency stop is released' ;;
    true)
      if [[ "$MODE" == "preflight" ]]; then
        pass 'physical emergency stop is engaged for preflight'
      else
        fail 'physical emergency stop is engaged'
      fi
      ;;
    *) fail 'could not read /j100_0519/platform/emergency_stop' ;;
  esac
fi

if ((failures > 0)); then
  printf 'JACKAL LINK CHECK FAILED: keep /spu_bert/enable_motion false.\n' >&2
  exit 1
fi
printf 'JACKAL LINK CHECK PASSED (%s).\n' "$MODE"
