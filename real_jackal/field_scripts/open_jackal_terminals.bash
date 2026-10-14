#!/usr/bin/env bash
# Open the terminals needed for a Jackal drive in ONE terminator window split into 10 panes.
#
# Every tab is already at a prompt with its command typed in: press Enter to run it (or edit it first).
# Nothing is executed by this script except the read-only STATUS tab.
# The RUN tab (monitor_run.bash) is never run automatically: it starts the drive (the robot moves), so only press
# Enter there when the e-stop operator is ready.
#
# Usage:
#   open_jackal_terminals.bash                 # one terminator window, 10 panes (left: set-up, right: drive)
#   open_jackal_terminals.bash --only status   # a single terminal (names below)
#   open_jackal_terminals.bash --dry-run       # print what would be opened
#   CONTAINER_NAME=moai_jackal_spubert open_jackal_terminals.bash
#
# Pane names: host sensors nav rviz trail | status check run stop after
set -euo pipefail

CONTAINER="${CONTAINER_NAME:-moai_jackal_spubert}"
IMAGE="${IMAGE:-drive-1012}"
REPO="${CAPSTONE_REPO:-$HOME/capstone_navi}"
STAB="${STABILITY_REPO:-$HOME/capstone_navi_stability_20260916}"
ONLY=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) shift; ONLY="${1:-}" ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

command -v terminator >/dev/null || { echo "terminator not found" >&2; exit 1; }
DIR="$(mktemp -d /tmp/jackal_tabs.XXXXXX)"

# tab <name> <title> <where: host|container> <banner> <prefilled command> <autorun: 0|1>
tab_script() {
  local name="$1" title="$2" where="$3" banner="$4" prefill="$5" auto="$6" file="$DIR/$1.sh"
  {
    echo '#!/usr/bin/env bash'
    printf 'printf "\\033]0;%%s\\007" %q\n' "$title"
    printf 'echo %q\n' "== $title =="
    printf 'echo %q\n' "$banner"
    if [ "$where" = host ]; then
      printf 'PREFILL=%q\n' "$prefill"
      echo 'if [ "'"$auto"'" = 1 ]; then eval "$PREFILL"; else read -e -i "$PREFILL" -p "$ " CMD; eval "$CMD"; fi'
      echo 'exec bash -i'
    else
      printf 'export PREFILL=%q AUTO=%q\n' "$prefill" "$auto"
      cat <<EOF
if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -Fxq true; then
  echo "container $CONTAINER is not running - start it in the HOST tab first"; exec bash -i
fi
exec docker exec -it -e PREFILL -e AUTO "$CONTAINER" bash --noprofile --norc -c '
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null
source /root/moai_stability_ws/install/setup.bash
if [ "\$AUTO" = 1 ]; then eval "\$PREFILL"; else read -e -i "\$PREFILL" -p "\$ " CMD; eval "\$CMD"; fi
exec bash --noprofile --norc -i'
EOF
    fi
  } > "$file"
  chmod +x "$file"
}

STATUS_LOOP='while true; do clear; date; uptime | sed "s/.*load/load/"; R=$(readlink -f /root/jackal_logs/drive_latest); echo "--- fast-livo: $(tail -1 $R/fastlivo_rss.log)"; echo "--- procs: bridge=$(ps -eo cmd | grep -c "real_jackal_spubert_brid[g]e") tracker=$(ps -eo cmd | grep -c "safe_path_tracke[r]") rviz=$(ps -eo stat,cmd | grep "rviz[2]" | grep -vc "^Z") bag=$(ps -eo cmd | grep -c "ros2 bag reco[r]d") monitor=$(ps -eo cmd | grep -c "monitor_run.ba[s]h")"; L=$(ls -t $R/spubert_real_*.log 2>/dev/null | head -1); echo "--- bridge:"; grep -E "path_valid|hold:|new_goal" $L 2>/dev/null | tail -2 | sed -E "s/.*\]: //" | cut -c1-110; echo "--- tracker:"; grep tracking $L 2>/dev/null | tail -1 | sed -E "s/.*\]: //" | cut -c1-110; sleep 3; done'

tab_script host    "1 HOST: network + container" host \
  "Run once after boot: network (sudo), then the container - skipped with a message if it is already running." \
  "sudo bash $REPO/real_jackal/scripts/configure_jackal_network.bash up && { if docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null | grep -Fxq true; then echo 'container $CONTAINER is already running - nothing to start'; else (cd $STAB && CAPSTONE_CALIB=$REPO/spu_deploy_docker/context/calib IMAGE=$IMAGE CONTAINER_NAME=$CONTAINER MOAI_ENABLE_GUI=1 ./real_jackal/docker/run_stability_container.bash); fi; }" 0
tab_script sensors "2 SENSORS (lidar + LIO)" container \
  "Lidar driver, then LIO (FAST-LIVO, VIO off). Restart LIO about every hour (CPU grows)." \
  "bash /root/start_livox_only.bash && sleep 10 && bash /root/start_lio_fixed.bash" 0
tab_script nav     "3 NAV2 + BRIDGE" container \
  "Nav2 (map, AMCL, planner) and the bridge/tracker. Re-run the bridge part after a config change; goal must be set again." \
  "bash /root/start_nav2_bridge.bash" 0
tab_script rviz    "4 RVIZ" container \
  "Recording layout. Close it before the run if the computer lags (software GL costs ~2 cores)." \
  "bash /root/start_rviz.bash" 0
tab_script trail   "5 ODOM TRAIL" container \
  "Draws the travelled path (blue). Cleared when a drive starts or a new goal is set." \
  "python3 /root/odom_trail.py" 0
tab_script status  "6 STATUS (read-only)" container \
  "Runs by itself: load, FAST-LIVO, processes, last bridge and tracker lines." \
  "$STATUS_LOOP" 1
tab_script check   "7 CHECK before the run" host \
  "Settles AMCL, shows AMCL std, robot link, sensor rates, load. Run it after the goal is set (and RViz closed if it lags)." \
  "docker exec -i $CONTAINER bash --noprofile --norc -s < $REPO/real_jackal/field_scripts/check_ready.bash" 0
tab_script run     "8 RUN (!!)" container \
  "!!! THE ROBOT WILL DRIVE after this. Only press Enter when the e-stop operator is ready and AMCL has converged." \
  "bash /root/monitor_run.bash" 0
tab_script stop    "9 STOP" container \
  "Press Enter to stop the robot immediately (safe to run any time)." \
  'ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"' 0
tab_script after   "10 AFTER THE RUN" container \
  "Stop the rosbag cleanly (never open a live bag), then analyse it." \
  'pkill -INT -f "ros2 bag reco[r]d"; sleep 5; ls -dt /root/jackal_logs/drive_latest/bag_[0-9]* | grep -v "\.log" | head -3' 0

ALL="host sensors nav rviz trail status check run stop after"
LIST="${ONLY:-$ALL}"
for n in $LIST; do [ -f "$DIR/$n.sh" ] || { echo "unknown pane: $n (choose from: $ALL)" >&2; exit 2; }; done

# one terminator layout: two columns, the panes stacked evenly in each
# the terminals take the left LEFT_PERCENT % of the primary screen, RViz (see its Window Geometry) the right part
read SCREEN_W SCREEN_H < <(xrandr --current 2>/dev/null | awk '/ connected/ && match($0, /[0-9]+x[0-9]+\+[0-9]+\+[0-9]+/) {split(substr($0, RSTART, RLENGTH), a, /[x+]/); print a[1], a[2]; exit}')
SCREEN_W="${SCREEN_W:-1704}"; SCREEN_H="${SCREEN_H:-1065}"
export TERM_W=$(( SCREEN_W * ${LEFT_PERCENT:-46} / 100 )) TERM_H=$(( SCREEN_H - 100 ))
python3 - "$DIR" $LIST <<'PY'
import sys
d, names = sys.argv[1], sys.argv[2:]
import os
TW, TH = int(os.environ['TERM_W']), int(os.environ['TERM_H'])
cols = [names[:5], names[5:]] if len(names) > 5 else [names]
out = ["[global_config]", "[keybindings]", "[profiles]", "  [[default]]", "[layouts]", "  [[jackal]]",
       "    [[[window0]]]", "      type = Window", '      parent = ""', f"      size = {TW}, {TH}", "      position = 0:0"]


def section(name, kind, parent, order, **extra):
    out.extend([f"    [[[{name}]]]", f"      type = {kind}", f"      parent = {parent}", f"      order = {order}"])
    out.extend(f"      {k} = {v}" for k, v in extra.items())


def terminal(tag, name, parent, order):
    section(f"{tag}_{name}", "Terminal", parent, order, profile="default", command=f"bash {d}/{name}.sh")


def column(col, tag, parent, order):
    """Stack the panes of one column with nested VPaned so that they all get the same height."""
    if len(col) == 1:
        terminal(tag, col[0], parent, order)
        return
    for k, name in enumerate(col[:-1]):
        remaining = len(col) - k
        section(f"{tag}_v{k}", "VPaned", parent, order, ratio=f"{1.0 / remaining:.4f}")
        terminal(tag, name, f"{tag}_v{k}", 0)
        parent, order = f"{tag}_v{k}", 1
    terminal(tag, col[-1], parent, 1)


if len(cols) == 2:
    section("hp", "HPaned", "window0", 0, ratio="0.5")
    column(cols[0], "L", "hp", 0)
    column(cols[1], "R", "hp", 1)
else:
    column(cols[0], "T", "window0", 0)
open(d + "/terminator.conf", "w").write("\n".join(out) + "\n")
PY
# --no-dbus: an already running terminator would take the new window over and ignore our layout file
CMD=(env GDK_BACKEND=x11 terminator --no-dbus -g "$DIR/terminator.conf" -l jackal -T "Jackal drive")
if [ $DRY -eq 1 ]; then echo "${CMD[*]}"; echo "(dry run - layout in $DIR/terminator.conf)"; exit 0; fi
"${CMD[@]}" >/dev/null 2>&1 &
echo "opened: $LIST   (scripts in $DIR)"
# GNOME on Wayland ignores the window position that terminator asks for, but a move request to a mapped X11 window
# is honoured: place the terminal on the left part of the screen (RViz opens on the right, see its Window Geometry).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
place() { GDK_BACKEND=x11 /usr/bin/python3 "$HERE/place_window.py" "Jackal drive" 0 0 "$TERM_W" "$TERM_H" terminator 2>/dev/null; }
for _ in $(seq 1 30); do
  sleep 0.5
  if place; then
    # the window manager places a new window itself once it is fully set up: move it after that, and again to be sure
    sleep 4; place || true
    sleep 2; place || true
    echo "terminal window placed on the left (${TERM_W}x${TERM_H})"; exit 0
  fi
done
echo "note: could not place the window - tile by hand: Super+Left on the terminal, Super+Right on RViz"
