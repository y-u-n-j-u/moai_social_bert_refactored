#!/usr/bin/env python3
"""Light top-down drive view (Tk, no GPU) with the drive status: replaces RViz while driving (RViz costs ~5 cores).

Draws, in the map frame and following the robot: the saved map, the live scan (yellow; points the people mask removed are
magenta), the rolling occupancy map the bridge plans on (orange), people with their walked trail, the Nav2 path (blue), the
model path (green), the goal (red), and the robot body (0.34 m hull, 0.44 m planning circle). Occupied / unknown rolling-map
cells INSIDE the 0.44 m circle are painted red: they are what makes the bridge say "no footprint-safe guidance point".
Mouse wheel zooms. Display only: publishes nothing; start it after the stack is up and the pose is set.
"""
import math
import os
import time
import threading
import tkinter as tk
from collections import deque

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from drive_hud import Hud, explain

MAP_YAML = os.environ.get("DRIVE_VIEW_MAP", "/root/jackal_maps/world_map.yaml")
SIZE = 720                       # canvas edge in pixels
ZOOMS = (2, 3, 4, 6)             # pixels per saved-map cell -> 18 / 12 / 9 / 6 m across
HULL_R, PLAN_R = 0.34, 0.44


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_map(path):
    spec = yaml.safe_load(open(path))
    image = spec["image"] if os.path.isabs(spec["image"]) else os.path.join(os.path.dirname(path), spec["image"])
    with open(image, "rb") as handle:
        tokens, data = [], handle.read()
    # P5 header: magic, width, height, maxval, then one whitespace byte and the pixels
    pos, tokens = 0, []
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1
    width, height = int(tokens[1]), int(tokens[2])
    gray = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=pos).reshape(height, width)
    if spec.get("negate", 0):
        gray = 255 - gray
    rgb = np.empty((height, width, 3), np.uint8)
    rgb[:] = (150, 150, 150)                     # unknown
    rgb[gray >= 250] = (232, 232, 232)           # free
    rgb[gray <= 80] = (30, 30, 30)               # occupied
    return rgb, float(spec["resolution"]), float(spec["origin"][0]), float(spec["origin"][1])


class ViewNode(Hud):
    def __init__(self):
        super().__init__()
        self.scan = None
        self.people_ids = []
        self.scan_nohuman = None
        self.rolling = None
        self.global_path = None
        self.model_path = None
        self.goal_odom = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/spu_bert/scan_nohuman", lambda m: setattr(self, "scan_nohuman", m), qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, "/spu_bert/debug_rolling_map", lambda m: setattr(self, "rolling", m), latched)
        self.create_subscription(Path, "/spu_bert/global_path_odom", lambda m: setattr(self, "global_path", m), 5)
        self.create_subscription(Path, "/spu_bert/predicted_path", lambda m: setattr(self, "model_path", m), 10)
        self.create_subscription(PoseStamped, "/spu_bert/final_goal_odom", lambda m: setattr(self, "goal_odom", m), 10)

    def on_tracks(self, msg):
        super().on_tracks(msg)
        self.people_ids = [int(k.id) for k in msg.tracks]

    def lookup(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)


def main():
    rclpy.init()
    node = ViewNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    base, res, ox, oy = load_map(MAP_YAML)
    height, width = base.shape[:2]
    state = {"zoom": 1, "trails": {}, "seen": {}}

    root = tk.Tk()
    root.title("주행 화면")
    root.geometry("1010x800+20+20")
    root.configure(bg="#202020")
    root.attributes("-topmost", True)
    banner = tk.Label(root, text="...", font=("Noto Sans CJK KR", 26, "bold"), fg="white", bg="#444444", pady=8)
    banner.pack(fill="x")
    body = tk.Frame(root, bg="#202020")
    body.pack(fill="both", expand=True)
    canvas = tk.Canvas(body, width=SIZE, height=SIZE, bg="#464646", highlightthickness=0)
    canvas.pack(side="left", padx=6, pady=6)
    side = tk.Label(body, text="", font=("Noto Sans CJK KR", 12), fg="#dddddd", bg="#202020", justify="left", anchor="nw", wraplength=262)
    side.pack(side="left", fill="both", expand=True, padx=4, pady=6)
    canvas.create_image(0, 0, anchor="nw", tags="bg")
    colors = {"RUN": "#1b7f3b", "HOLD": "#b58900", "STOP": "#b3261e"}

    def on_wheel(event):
        step = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4) else -1
        state["zoom"] = min(max(state["zoom"] + step, 0), len(ZOOMS) - 1)

    canvas.bind("<MouseWheel>", on_wheel)
    canvas.bind("<Button-4>", on_wheel)
    canvas.bind("<Button-5>", on_wheel)

    def frame():
        z = ZOOMS[state["zoom"]]
        n = SIZE // z
        pose = node.lookup("map", "base_link")
        mo = node.lookup("map", "odom")
        ob = node.lookup("odom", "base_link")
        text = node.tracker
        if text.startswith("tracking"):
            banner.config(text="주행 중", bg=colors["RUN"])
        elif text:
            level = "HOLD" if text.startswith("planner_hold") or "stale" in text else "STOP"
            banner.config(text="멈춤 (경로 대기)" if level == "HOLD" else "정지", bg=colors[level])
        else:
            banner.config(text="움직임 꺼짐", bg="#444444")
        lines = []
        if pose is None or mo is None:
            canvas.itemconfig("bg", image="")
            canvas.delete("dyn")
            canvas.create_text(SIZE // 2, SIZE // 2, text="위치 추정 대기 (map→odom 없음)", fill="white", font=("Noto Sans CJK KR", 18), tags="dyn")
        else:
            cx, cy, cyaw = pose
            tx, ty, tyaw = mo
            c0 = int(round((cx - ox) / res)) - n // 2
            r0 = int(round(height - 1 - (cy - oy) / res)) - n // 2
            view = np.full((n, n, 3), 70, np.uint8)
            rs, re, cs, ce = max(r0, 0), min(r0 + n, height), max(c0, 0), min(c0 + n, width)
            if rs < re and cs < ce:
                view[rs - r0:re - r0, cs - c0:ce - c0] = base[rs:re, cs:ce]

            def put(xs, ys, color):
                cols = np.round((xs - ox) / res - c0).astype(int)
                rows = np.round((height - 1 - (ys - oy) / res) - r0).astype(int)
                ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
                view[rows[ok], cols[ok]] = color

            def odom_to_map(xs, ys):
                co, so = math.cos(tyaw), math.sin(tyaw)
                return tx + co * xs - so * ys, ty + so * xs + co * ys

            blocked = []
            counts = (0, 0)
            rolling = node.rolling
            if rolling is not None:
                grid = np.array(rolling.data, dtype=np.int8).reshape(rolling.info.height, rolling.info.width)
                rr = rolling.info.resolution
                ii, jj = np.nonzero(grid == 100)
                xs = rolling.info.origin.position.x + (jj + 0.5) * rr
                ys = rolling.info.origin.position.y + (ii + 0.5) * rr
                mx, my = odom_to_map(xs, ys)
                put(mx, my, (235, 125, 0))
                if ob is not None:
                    gi, gj = np.nonzero((grid == 100) | (grid == -1))
                    gx = rolling.info.origin.position.x + (gj + 0.5) * rr
                    gy = rolling.info.origin.position.y + (gi + 0.5) * rr
                    inside = np.hypot(gx - ob[0], gy - ob[1]) < PLAN_R
                    occ_in = int(((grid[gi, gj] == 100) & inside).sum())
                    unk_in = int(((grid[gi, gj] == -1) & inside).sum())
                    counts = (occ_in, unk_in)
                    bx, by = odom_to_map(gx[inside], gy[inside])
                    blocked = list(zip(bx, by))
            scan, nh = node.scan, node.scan_nohuman
            if scan is not None:
                ranges = np.array(scan.ranges, dtype=np.float64)
                angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
                good = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= min(scan.range_max, 12.0))
                removed = np.zeros(len(ranges), bool)
                if nh is not None and len(nh.ranges) == len(ranges):
                    removed = good & ~np.isfinite(np.array(nh.ranges, dtype=np.float64))
                px = cx + ranges * np.cos(angles + cyaw)
                py = cy + ranges * np.sin(angles + cyaw)
                put(px[good & ~removed], py[good & ~removed], (255, 215, 0))
                put(px[removed], py[removed], (255, 0, 255))
            image = np.repeat(np.repeat(view, z, axis=0), z, axis=1)
            photo = tk.PhotoImage(data=b"P6 %d %d 255\n" % (image.shape[1], image.shape[0]) + image.tobytes())
            canvas.itemconfig("bg", image=photo)
            canvas.photo = photo
            canvas.delete("dyn")
            scale = z / res

            def w2c(x, y):
                return ((x - ox) / res - c0 + 0.5) * z, ((height - 1 - (y - oy) / res) - r0 + 0.5) * z

            for bx, by in blocked:
                u, v = w2c(bx, by)
                canvas.create_rectangle(u - 0.05 * scale, v - 0.05 * scale, u + 0.05 * scale, v + 0.05 * scale, fill="#ff2020", outline="", tags="dyn")

            def draw_path(path, color, width_px):
                if path is None or len(path.poses) < 2:
                    return
                pts = []
                for p in path.poses[:: max(len(path.poses) // 200, 1)]:
                    mx_, my_ = odom_to_map(np.array([p.pose.position.x]), np.array([p.pose.position.y]))
                    pts.extend(w2c(mx_[0], my_[0]))
                canvas.create_line(*pts, fill=color, width=width_px, tags="dyn")

            draw_path(node.global_path, "#4aa3ff", 2)
            draw_path(node.model_path, "#22dd66", 5)
            if node.goal_odom is not None:
                gx_, gy_ = odom_to_map(np.array([node.goal_odom.pose.position.x]), np.array([node.goal_odom.pose.position.y]))
                u, v = w2c(gx_[0], gy_[0])
                canvas.create_polygon(u, v - 12, u + 12, v, u, v + 12, u - 12, v, fill="#ff3030", outline="white", tags="dyn")
            now = time.time()
            for index, (px_, py_) in enumerate(node.people):
                mxp, myp = cx + math.cos(cyaw) * px_ - math.sin(cyaw) * py_, cy + math.sin(cyaw) * px_ + math.cos(cyaw) * py_
                key = node.people_ids[index] if len(node.people_ids) == len(node.people) else index
                trail = state["trails"].setdefault(key, deque(maxlen=300))
                if not trail or math.hypot(trail[-1][0] - mxp, trail[-1][1] - myp) > 0.08:
                    trail.append((mxp, myp))
                state["seen"][key] = now
                u, v = w2c(mxp, myp)
                canvas.create_oval(u - 0.3 * scale, v - 0.3 * scale, u + 0.3 * scale, v + 0.3 * scale, outline="#d070ff", width=3, tags="dyn")
            for key in [k for k, t in state["seen"].items() if now - t > 4.0]:
                state["trails"].pop(key, None)
                state["seen"].pop(key, None)
            for trail in state["trails"].values():
                if len(trail) > 1:
                    canvas.create_line(*[c for pt in trail for c in w2c(*pt)], fill="#d070ff", width=2, tags="dyn")
            u, v = w2c(cx, cy)
            canvas.create_oval(u - PLAN_R * scale, v - PLAN_R * scale, u + PLAN_R * scale, v + PLAN_R * scale, outline="#ffffff", dash=(4, 3), tags="dyn")
            canvas.create_oval(u - HULL_R * scale, v - HULL_R * scale, u + HULL_R * scale, v + HULL_R * scale, outline="#00e5ff", width=2, tags="dyn")
            canvas.create_line(u, v, u + 0.5 * scale * math.cos(cyaw), v - 0.5 * scale * math.sin(cyaw), fill="#00e5ff", width=3, tags="dyn")
            canvas.create_line(12, SIZE - 14, 12 + scale, SIZE - 14, fill="white", width=3, tags="dyn")
            canvas.create_text(12 + scale / 2, SIZE - 26, text="1 m", fill="white", tags="dyn")
            lines.append("몸체 반경 %.2f m 안의 지도 셀:\n  점유 %d / 모름 %d  (빨간 점)" % (PLAN_R, counts[0], counts[1]))
        goal = "-"
        if node.odom and node.goal:
            goal = "%.1f m" % math.hypot(node.goal[0] - node.odom[0], node.goal[1] - node.odom[1])
        nearest = min((math.hypot(x, y) for x, y in node.people), default=None)
        moving = time.time() - node.cmd[2] < 1.0 and abs(node.cmd[0]) > 0.01
        reason = explain(text) + ((" | 브릿지: " + explain(node.last_runtime_hold)) if node.last_runtime_hold and not text.startswith("tracking") else "") if text else "\"run\" 전이거나 정지된 상태"
        side.config(text="%s\n\n속도 %.2f m/s  회전 %.2f\n사람 %s   목표까지 %s\n\n%s\n\n브릿지: %s\n\n최근 이벤트\n%s\n\n범례: 노랑=스캔, 자홍=사람 때문에 지운 점,\n주황=계획용 지도 점유, 빨강=몸체 안 침범,\n파랑=전역 경로, 초록=모델 경로,\n보라=사람/궤적, 빨간 마름모=목표" % (
            reason, node.cmd[0] if moving else 0.0, node.cmd[1] if moving else 0.0,
            ("%.2f m" % nearest) if nearest is not None else "없음", goal, "\n".join(lines), node.runtime[:44],
            "\n".join("%s [%s] %s" % e for e in list(node.events)[:6])))
        root.after(250, frame)

    frame()
    root.mainloop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
