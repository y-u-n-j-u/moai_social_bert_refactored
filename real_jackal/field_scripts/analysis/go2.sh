#!/usr/bin/env bash
R=$(readlink -f /root/jackal_logs/drive_latest)
setsid nohup bash --noprofile --norc /root/monitor_run.bash > $R/monitor_run.out 2>&1 < /dev/null &
sleep 20; cut -c1-200 $R/monitor_run.out | tail -9
