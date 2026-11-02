#!/usr/bin/env python3
"""Small always-on-top status window for a drive (a few % of one core, unlike RViz): what the robot is doing and WHY it stops.

Shows the tracker state (driving / holding / stopped) with the reason in plain Korean, the bridge's path source, speed,
distance to the nearest obstacle / person, distance to the goal, and the last events. Every state change is appended to
/root/jackal_logs/drive_events.log (one line each) so it can be followed from outside. Display only: publishes nothing.
"""
import math
import re
import threading
import time
import tkinter as tk
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from moai_nav_msgs.msg import Tracks
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

EVENT_LOG = "/root/jackal_logs/drive_events.log"

EXPLAIN = [
    (r"predicted_path_missing_or_stale", "새 경로가 안 옴(브릿지가 경로를 못 냄)"),
    (r"hold:adaptive_guidance_failed", "몸체가 안전하게 들어갈 유도점이 없음"),
    (r"hold:recovery_blocked_obstacle_in_rotation_circle", "제자리 회전할 공간이 막힘"),
    (r"hold:all_candidates_rejected:robot_footprint_collision", "후보 경로 전부 몸체 충돌(지도 점유)"),
    (r"hold:all_candidates_rejected:insufficient_goal_progress", "후보 경로가 목표로 안 나아감"),
    (r"hold:all_candidates_rejected:predicted_human_clearance", "후보 경로 전부 사람과 가까움"),
    (r"hold:all_candidates_rejected:no_map_safe_goal", "지도상 안전한 목표점 없음"),
    (r"hold:all_candidates_rejected:model_map_check_failed", "모델 경로가 지도 검사에서 탈락"),
    (r"hold:odom_missing_or_stale", "오도메트리(LIO)가 오래됨"),
    (r"hold:scan_missing_or_stale", "스캔이 오래됨"),
    (r"hold:waiting_for_goal", "목표 대기"),
    (r"hold:goal_reached", "목표 도착"),
    (r"obstacle_too_close:([0-9.]+)m", "장애물이 앞 \\1m 안(추종기 정지)"),
    (r"goal_reached_stopped", "목표 도착, 정지"),
    (r"emergency", "비상정지 래치"),
    (r"human_too_close", "사람이 정지 거리 안"),
    (r"planner_hold", "경로 대기(브릿지 판단 대기)"),
    (r"^stopped", "정지 상태(움직임 꺼짐)"),
]


def explain(text):
    for pattern, message in EXPLAIN:
        match = re.search(pattern, text)
        if match:
            return re.sub(pattern, message, match.group(0)) if "\\1" in message else message
    return text[:60]


class Hud(Node):
    def __init__(self):
        super().__init__("drive_hud")
        self.tracker = ""
        self.runtime = ""
        self.cmd = (0.0, 0.0, 0.0)
        self.people = []
        self.odom = None
        self.goal = None
        self.events = deque(maxlen=9)
        self.last_key = None
        self.last_runtime_hold = None
        self.create_subscription(String, "/spu_bert/tracker_status", self.on_tracker, 10)
        self.create_subscription(String, "/spu_bert/runtime_status", self.on_runtime, 10)
        self.create_subscription(Twist, "/j100_0519/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Tracks, "/ped_tracking", self.on_tracks, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/aft_mapped_to_init", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/spu_bert/final_goal_odom", self.on_goal, 10)

    def log(self, level, text):
        stamp = time.strftime("%H:%M:%S")
        self.events.appendleft((stamp, level, text))
        try:
            with open(EVENT_LOG, "a") as handle:
                handle.write(f"{stamp} {level} {text}\n")
        except OSError:
            pass

    def on_tracker(self, msg):
        self.tracker = msg.data
        key = "RUN" if msg.data.startswith("tracking") else msg.data.split(" ")[0]
        key = re.sub(r"obstacle_too_close:[0-9.]+m", "obstacle_too_close", key)
        if key in ("planner_hold", "predicted_path_missing_or_stale"):
            # one event per reason: the tracker alternates these two statuses every second while it waits
            key = "HOLD|" + (self.last_runtime_hold or "")[:48]
        if key != self.last_key:
            if key == "RUN":
                self.log("RUN", "주행 재개/진행")
            elif key.startswith("HOLD|"):
                self.log("HOLD", explain(msg.data) + (f" | 브릿지: {explain(self.last_runtime_hold)}" if self.last_runtime_hold else ""))
            else:
                self.log("STOP", explain(msg.data))
            self.last_key = key

    def on_runtime(self, msg):
        self.runtime = msg.data
        if msg.data.startswith("hold:"):
            self.last_runtime_hold = msg.data

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z, time.time())

    def on_tracks(self, msg):
        self.people = [(float(k.pose.position.x), float(k.pose.position.y)) for k in msg.tracks]

    def on_odom(self, msg):
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def on_goal(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)


def main():
    rclpy.init()
    node = Hud()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    root = tk.Tk()
    root.title("주행 상태")
    root.geometry("760x560+900+30")
    root.attributes("-topmost", True)
    root.configure(bg="#202020")
    banner = tk.Label(root, text="...", font=("Noto Sans CJK KR", 30, "bold"), fg="white", bg="#444444", pady=14)
    banner.pack(fill="x")
    reason = tk.Label(root, text="", font=("Noto Sans CJK KR", 17), fg="white", bg="#202020", wraplength=720, justify="left")
    reason.pack(fill="x", padx=14, pady=(10, 4))
    info = tk.Label(root, text="", font=("Noto Sans CJK KR", 15), fg="#dddddd", bg="#202020", justify="left", anchor="w")
    info.pack(fill="x", padx=14, pady=4)
    log_box = tk.Label(root, text="", font=("Noto Sans CJK KR", 12), fg="#bbbbbb", bg="#202020", justify="left", anchor="nw", wraplength=730)
    log_box.pack(fill="both", expand=True, padx=14, pady=6)
    colors = {"RUN": "#1b7f3b", "HOLD": "#b58900", "STOP": "#b3261e"}

    def tick():
        text = node.tracker
        moving = time.time() - node.cmd[2] < 1.0 and abs(node.cmd[0]) > 0.01
        if text.startswith("tracking"):
            banner.config(text="주행 중", bg=colors["RUN"])
            reason.config(text=text)
        elif text:
            level = "HOLD" if text.startswith("planner_hold") or "stale" in text else "STOP"
            banner.config(text="멈춤(경로 대기)" if level == "HOLD" else "정지", bg=colors[level])
            extra = node.last_runtime_hold or ""
            reason.config(text=explain(text) + (f"\n브릿지: {explain(extra)}" if extra and level == "HOLD" else ""))
        else:
            banner.config(text="움직임 꺼짐", bg="#444444")
            reason.config(text="(\"run\" 전이거나 정지된 상태)")
        goal = "-"
        if node.odom and node.goal:
            goal = "%.1f m" % math.hypot(node.goal[0] - node.odom[0], node.goal[1] - node.odom[1])
        nearest = min((math.hypot(x, y) for x, y in node.people), default=None)
        obstacle = re.search(r"obstacle=([0-9.inf]+)", text)
        info.config(text="속도 %.2f m/s  회전 %.2f rad/s%s\n앞 장애물 %s m   사람 %s   목표까지 %s\n브릿지: %s" % (
            node.cmd[0] if moving else 0.0, node.cmd[1] if moving else 0.0, "" if moving else "  (명령 없음)",
            obstacle.group(1) if obstacle else "-", ("%.2f m" % nearest) if nearest is not None else "없음", goal, node.runtime[:62]))
        log_box.config(text="최근 이벤트\n" + "\n".join(f"{s} [{l}] {t}" for s, l, t in node.events))
        root.after(250, tick)

    tick()
    root.mainloop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
