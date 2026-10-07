# Field scripts (run inside the Jackal container, /root/)

- `monitor_run.bash` - gate + arm + supervised drive. Refuses to arm unless AMCL has converged
  (x std <= 0.60 m, y std <= 0.20 m, yaw <= 10 deg) and a rosbag recorder is running (starts one if not).
  Auto-disarms on FAST-LIVO memory > 2000 MB, odom > 80 m, map->odom shift > 1 m, bridge ERROR,
  obstacle < 0.26 m, or when the monitor ends for any reason other than goal arrival.
  Never pipe its output into `head` (SIGPIPE ends it while armed).
- `start_bag3.bash` - rosbag of the topics needed to analyse a run. Do not open the live .db3 while recording
  (SQLite "database is locked" kills the recorder); stop it with SIGINT first.
