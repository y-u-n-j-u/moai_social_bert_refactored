# Field scripts (run inside the Jackal container, /root/)

- `monitor_run.bash` - gate + run + supervised drive. Refuses to run unless AMCL has converged
  (x std <= 0.60 m, y std <= 0.20 m, yaw <= 10 deg) and a rosbag recorder is running (starts one if not).
  Auto-stops on FAST-LIVO memory > 2000 MB, odom > 80 m, map->odom shift > 1 m, bridge ERROR,
  obstacle < 0.26 m, or when the monitor ends for any reason other than goal arrival.
  Never pipe its output into `head` (SIGPIPE ends it while running).
- `start_bag3.bash` - rosbag of the topics needed to analyse a run. Do not open the live .db3 while recording
  (SQLite "database is locked" kills the recorder); stop it with SIGINT first.
- `odom_trail.py` - display only: accumulates `/aft_mapped_to_init` into `/spu_bert/odom_trail` (a Path) so RViz can show where the
  robot has actually driven. A new `/goal_pose` or an odom jump > 5 m (LIO restart) starts a new trail. Run it in the container:
  `python3 /root/odom_trail.py`.
- `real_jackal_spubert_record.rviz` - RViz layout for recording: Nav2 global path hidden, executed path (`Final TGP`) thicker,
  blue travelled-path trail added. The "EXECUTED: <source>" label and the rejected-candidate markers stay on, so the recording shows
  which planner produced each path. `start_rviz.bash` loads this layout.
