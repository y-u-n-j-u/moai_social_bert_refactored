#!/usr/bin/env bash
set -euo pipefail

INTERFACE="${JACKAL_INTERFACE:-enp131s0}"
DDS_PEER_IP="${JACKAL_DDS_PEER_IP:-192.168.50.2}"
REQUIRED_ADDRESSES=(
  "192.168.1.50/24"
  "192.168.131.50/24"
  "192.168.50.1/24"
)

usage() {
  cat <<'EOF'
Usage: configure_jackal_network.bash <up|check>

  up     Add only missing Capstone IP aliases, then run checks.
  check  Read-only validation. No network settings are changed.

Overrides:
  JACKAL_INTERFACE=enp131s0
  JACKAL_DDS_PEER_IP=192.168.50.2

This script never changes NetworkManager ownership, routes, gateways, DNS, or
existing addresses. It never removes another team's settings.
EOF
}

has_address() {
  local address_without_prefix="${1%/*}"
  ip -o -4 address show dev "$INTERFACE" \
    | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$address_without_prefix"
}

carrier_is_up() {
  [[ -r "/sys/class/net/$INTERFACE/carrier" ]] \
    && [[ "$(<"/sys/class/net/$INTERFACE/carrier")" == "1" ]]
}

check_network() {
  local failures=0
  if [[ ! -d "/sys/class/net/$INTERFACE" ]]; then
    printf '[FAIL] Interface does not exist: %s\n' "$INTERFACE" >&2
    return 1
  fi

  ip -br link show dev "$INTERFACE"
  ip -br address show dev "$INTERFACE"

  if carrier_is_up; then
    printf '[PASS] Ethernet carrier is up on %s\n' "$INTERFACE"
  else
    printf '[FAIL] No Ethernet carrier on %s; connect/power the Jackal network first.\n' \
      "$INTERFACE" >&2
    failures=$((failures + 1))
  fi

  local address
  for address in "${REQUIRED_ADDRESSES[@]}"; do
    if has_address "$address"; then
      printf '[PASS] Address %s is configured\n' "$address"
    else
      printf '[FAIL] Address %s is missing\n' "$address" >&2
      failures=$((failures + 1))
    fi
  done

  if carrier_is_up && ping -I "$INTERFACE" -c 2 -W 1 "$DDS_PEER_IP" >/dev/null 2>&1; then
    printf '[PASS] Jackal DDS peer %s responds on %s\n' "$DDS_PEER_IP" "$INTERFACE"
  else
    printf '[FAIL] Jackal DDS peer %s is unreachable on %s\n' \
      "$DDS_PEER_IP" "$INTERFACE" >&2
    failures=$((failures + 1))
  fi

  ((failures == 0)) || return 1
}

bring_up_network() {
  if [[ ! -d "/sys/class/net/$INTERFACE" ]]; then
    printf 'Interface does not exist: %s\n' "$INTERFACE" >&2
    return 1
  fi

  sudo -v
  sudo ip link set "$INTERFACE" up

  local address
  for address in "${REQUIRED_ADDRESSES[@]}"; do
    if has_address "$address"; then
      printf '[KEEP] Existing address %s\n' "$address"
    else
      sudo ip address add "$address" dev "$INTERFACE"
      printf '[ADD] Address %s\n' "$address"
    fi
  done

  local attempt
  for attempt in 1 2 3 4 5; do
    carrier_is_up && break
    sleep 1
  done
  check_network
}

case "${1:-}" in
  up) bring_up_network ;;
  check) check_network ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
