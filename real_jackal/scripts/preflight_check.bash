#!/bin/bash
set -u

failures=0
expected_checkpoint_sha256="03d88a96db9abe46c47d98837a57785f01c197a7338f0784510e88bec6a10a95"
required_files=(
  /root/moai_social_bert_refactored/configs/spubert/moai_social_nav_route_gp_continue_b21_col01_social005.yaml
  /root/moai_social_bert_refactored/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth
)
required_topics=(/aft_mapped_to_init /scan /ped_tracking)

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
for path in "${required_files[@]}"; do
  if [[ -s "$path" ]]; then
    echo "PASS file $path"
  else
    echo "FAIL file $path"
    failures=$((failures + 1))
  fi
done

checkpoint="${required_files[1]}"
if [[ -s "$checkpoint" ]]; then
  actual_checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
  if [[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]]; then
    echo "PASS checkpoint SHA256 $actual_checkpoint_sha256"
  else
    echo "FAIL checkpoint SHA256 $actual_checkpoint_sha256"
    echo "     expected $expected_checkpoint_sha256"
    failures=$((failures + 1))
  fi
fi

if nvidia-smi >/dev/null 2>&1; then
  echo "PASS NVIDIA GPU visible"
else
  echo "FAIL NVIDIA GPU is not visible inside the container"
  failures=$((failures + 1))
fi

for topic in "${required_topics[@]}"; do
  type="$(ros2 topic type "$topic" 2>/dev/null || true)"
  if [[ -n "$type" ]]; then
    echo "PASS topic $topic [$type]"
  else
    echo "FAIL topic $topic is missing"
    failures=$((failures + 1))
  fi
done

if timeout 4 ros2 run tf2_ros tf2_echo odom base_link >/dev/null 2>&1; then
  echo "PASS TF odom -> base_link"
else
  echo "FAIL TF odom -> base_link"
  failures=$((failures + 1))
fi

if ros2 action list 2>/dev/null | grep -Fxq /compute_path_to_pose; then
  echo "PASS action /compute_path_to_pose"
else
  echo "FAIL action /compute_path_to_pose (start the route planner first)"
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  echo "PREFLIGHT FAIL: $failures required checks failed"
  exit 1
fi
echo "PREFLIGHT PASS: keep motion disarmed and run the monitor test next"
