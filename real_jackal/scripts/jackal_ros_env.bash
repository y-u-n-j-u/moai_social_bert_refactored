#!/usr/bin/env bash

# Source this file inside moai_jackal_spubert. It intentionally does not use
# either the host or container user's .bashrc.
_capstone_jackal_env_main() {
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'This file must be sourced: source %s\n' "${BASH_SOURCE[0]}" >&2
    return 2
  fi

  local profile="${JACKAL_DDS_PROFILE_IN_CONTAINER:-/root/jackal_runtime/fastdds_laptop.xml}"
  if [[ ! -r "$profile" ]]; then
    printf 'Fast DDS profile is not mounted: %s\n' "$profile" >&2
    return 1
  fi
  local address_present=false
  if command -v ip >/dev/null 2>&1; then
    if ip -o -4 address show | awk '{print $4}' | cut -d/ -f1 \
        | grep -Fxq 192.168.50.1; then
      address_present=true
    fi
  elif [[ -r /proc/net/fib_trie ]] \
      && grep -Fq -- '192.168.50.1' /proc/net/fib_trie; then
    address_present=true
  fi
  if [[ "$address_present" != true ]]; then
    printf 'Required laptop DDS address 192.168.50.1 is not configured.\n' >&2
    return 1
  fi

  export ROS_DOMAIN_ID=1
  export ROS_LOCALHOST_ONLY=0
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTRTPS_DEFAULT_PROFILES_FILE="$profile"
  export FASTDDS_DEFAULT_PROFILES_FILE="$profile"
  unset ROS_DISCOVERY_SERVER CYCLONEDDS_URI

  source /opt/ros/humble/setup.bash
  source /root/robot_ws/install/setup.bash

  printf 'Jackal ROS environment ready: domain=%s rmw=%s profile=%s\n' \
    "$ROS_DOMAIN_ID" "$RMW_IMPLEMENTATION" "$profile"
}

_capstone_jackal_env_main "$@"
_capstone_jackal_env_status=$?
unset -f _capstone_jackal_env_main
return "$_capstone_jackal_env_status"
