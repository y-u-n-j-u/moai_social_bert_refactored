#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
RUN=$(readlink -f /root/jackal_logs/drive_latest)
setsid nohup rviz2 -d /root/real_jackal_spubert_record.rviz > "$RUN/rviz.log" 2>&1 < /dev/null &
echo "rviz $!" >> "$RUN/pids"; echo "started rviz pid $!"
