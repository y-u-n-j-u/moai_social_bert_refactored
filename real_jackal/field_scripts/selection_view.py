#!/usr/bin/env python3
"""Read-only desktop view of the bridge's candidate selection evidence."""

import errno
import getpass
import json
import math
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TOPIC = "/spu_bert/selection_evidence"
REASONS = {
    "minimum_weighted_cost": "선택 가능한 후보 중 총비용 최소",
    "only_valid_option": "유일하게 선택 가능한 경로",
    "first_valid_by_rank": "선택기 입력 순서에서 첫 유효 경로",
    "previous_path_revalidated": "이전 경로가 현재 검사도 통과",
    "robot_footprint_collision": "로봇 몸체 기준 지도 충돌",
    "predicted_human_clearance": "예측 보행자와 거리 부족",
    "insufficient_goal_progress": "목표 방향 진행량 부족",
    "kinematic_jump": "인접 경로점 간 이동량 초과",
    "model_map_check_failed": "모델 내부 지도 검사 탈락",
    "no_map_safe_goal": "지도상 안전한 후보 목표 없음",
    "invalid_path_length": "경로점 개수 불일치",
    "all_candidates_rejected": "모든 새 후보 탈락",
    "no_candidates": "새 모델 후보 없음",
    "waiting_for_goal": "목표 입력 대기",
    "goal_reached": "목표 도달",
}
SOURCES = {
    "model": "새 모델 경로", "retained": "이전 경로 유지",
    "fallback": "대체 경로", "recovery": "복구 경로", "hold": "정지 대기",
}
COSTS = ("total", "rank_cost", "continuity_cost", "heading_cost", "soft_margin_cost")


def acquire_instance_lock():
    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    lock_path = Path(tempfile.gettempdir()) / f"moai_selection_view_{getpass.getuser()}_{domain}.lock"
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return None
        raise
    return handle


def reason_text(code):
    if not code:
        return "—"
    return " · ".join(REASONS.get(part, part) for part in str(code).split(":"))


def candidate_name(index):
    return f"C{index}" if index is not None else "—"


def cost_text(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def candidate_rows(report):
    """Format emitted facts, including safety failures with no computed cost."""
    chosen = report.get("selected_candidate_index")
    published = report.get("executed_candidate_index")
    selection_reason = report.get("selection_reason")
    retained = report.get("retained_option")
    winner = next((row for row in report["candidates"]
                   if chosen is not None and row.get("candidate_index") == chosen), None)
    if winner is None and report.get("executed_source") == "retained":
        winner = retained
    winning_cost = ((winner or {}).get("score") or {}).get("total")
    for row in report["candidates"]:
        index = row.get("candidate_index")
        score = row.get("score") or {}
        if not row.get("valid"):
            tag, state, reason = "rejected", "검사 탈락", reason_text(row.get("reason"))
        elif index is not None and index == chosen:
            sent = report.get("executed_source") == "model" and index == published
            tag, state = ("selected", "선택·발행") if sent else ("replaced", "선택 후 교체")
            reason = reason_text(selection_reason)
        elif score.get("eligible") is False:
            tag, state, reason = "excluded", "선택 제외", "시작 방향 제한 초과"
        else:
            tag, state, reason = "valid", "검사 통과", "다른 경로가 선택됨"
            total = score.get("total")
            if selection_reason == "minimum_weighted_cost" and total is not None and winning_cost is not None:
                if total > winning_cost + 1e-12:
                    reason = "선택 경로보다 총비용이 높음"
                elif abs(total - winning_cost) <= 1e-12:
                    reason = "동점에서 앞선 입력 경로 우선"
        yield tag, (candidate_name(index), row.get("rank", "—"), state,
                    *(cost_text(score.get(key)) for key in COSTS), reason)

    if retained:
        score = retained.get("score") or {}
        sent = report.get("executed_source") == "retained"
        state = "발행" if sent else "비교 옵션"
        reason = reason_text(report.get("decision_reason")) if sent else "이전 경로 검사 통과"
        yield "replaced", ("이전 경로", "—", state,
                           *(cost_text(score.get(key)) for key in COSTS), reason)


class EvidenceSubscriber(Node):
    def __init__(self):
        super().__init__("selection_view")
        self.latest = None
        self.received_at = None
        self.create_subscription(String, TOPIC, self.on_evidence, 1)

    def on_evidence(self, message):
        try:
            report = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if (not isinstance(report, dict) or report.get("schema_version") != 1
                or report.get("event") != "selection_evidence"
                or not isinstance(report.get("stamp_ns"), int)
                or not isinstance(report.get("candidates"), list)
                or not all(isinstance(row, dict) for row in report["candidates"])):
            return
        self.latest = report
        self.received_at = time.monotonic()


class SelectionView:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.displayed = None
        self.displayed_rows = ()
        root.title("Jackal Candidate Decisions")
        root.geometry("1060x440")
        root.minsize(900, 360)
        root.configure(background="#f2f5f8")

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Candidate.Treeview", rowheight=32, font=("TkDefaultFont", 11),
                        background="white", fieldbackground="white")
        style.configure("Candidate.Treeview.Heading", font=("TkDefaultFont", 10, "bold"))

        summary = tk.Frame(root, background="#172638", padx=20, pady=12)
        summary.pack(fill="x")
        self.source = tk.StringVar(value="선택 증거 수신 대기")
        self.choice = tk.StringVar(value="모델 후보 선택")
        self.decision = tk.StringVar(value=f"{TOPIC} 구독 중")
        for var, size, color, weight in ((self.source, 10, "#a9bfd4", "normal"),
                                         (self.choice, 17, "white", "bold"),
                                         (self.decision, 11, "#cfe0ef", "normal")):
            tk.Label(summary, textvariable=var, background="#172638", foreground=color,
                     font=("TkDefaultFont", size, weight), anchor="w").pack(fill="x", pady=2)

        table = ttk.Frame(root, padding=(16, 12, 16, 0))
        table.pack(fill="both", expand=True)
        columns = ("후보", "모델 순위", "결과", "총비용", "입력 순서", "연속성", "시작 방향", "장애물 여유", "판단 이유")
        widths = (76, 76, 116, 71, 74, 68, 76, 88, 270)
        self.rows = ttk.Treeview(table, columns=columns, show="headings", selectmode="none",
                                 height=6, style="Candidate.Treeview")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.rows.yview)
        self.rows.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.rows.pack(side="left", fill="both", expand=True)
        for title, width in zip(columns, widths):
            self.rows.heading(title, text=title)
            self.rows.column(title, width=width, minwidth=width if title != "판단 이유" else 128,
                             anchor="w" if title == "판단 이유" else "center", stretch=title == "판단 이유")
        for tag, bg, fg in (("selected", "#e1f5e8", "#165b39"), ("rejected", "#fce9e9", "#8b2727"),
                            ("valid", "#e9f2fc", "#234f80"), ("excluded", "#fff0dc", "#8a490e"),
                            ("replaced", "#fff5d8", "#755614")):
            self.rows.tag_configure(tag, background=bg, foreground=fg)
        tk.Label(root, text="총비용 = 입력 순서 + 연속성 + 시작 방향 + 장애물 여유   ·   — 는 미산정 또는 해당 없음",
                 background="#f2f5f8", foreground="#455669", anchor="w").pack(fill="x", padx=18, pady=12)

    def refresh(self):
        if not rclpy.ok():
            self.root.destroy()
            return
        rclpy.spin_once(self.node, timeout_sec=0)
        report = self.node.latest
        if report is not None:
            mode = "합성 시연" if report.get("synthetic") else "ROS 선택 증거"
            stale = "  |  새 데이터 없음 · 마지막 수신 표시" if time.monotonic() - self.node.received_at > 3.0 else ""
            sec, ns = divmod(report["stamp_ns"], 1_000_000_000)
            self.source.set(f"{mode}  |  ROS {sec}.{ns:09d}{stale}")
            if report is not self.displayed:
                self.show_report(report)
                self.displayed = report
        self.root.after(40, self.refresh)

    def show_report(self, report):
        chosen = report.get("selected_candidate_index")
        source = report.get("executed_source")
        sent = report.get("executed_candidate_index")
        chooser_reason = report.get("selection_reason")
        decision_reason = report.get("decision_reason")
        published_text = SOURCES.get(source, source or "—")
        if sent is not None:
            published_text += f" {candidate_name(sent)}"
        reason = reason_text(chooser_reason) if chosen is not None else "새 모델 후보 선택 없음"
        self.choice.set(f"모델 선택  {candidate_name(chosen)}  ·  {reason}")
        self.decision.set(f"추종기에 발행  {published_text}  |  {reason_text(decision_reason)}")
        rows = tuple(candidate_rows(report))
        if rows != self.displayed_rows:
            self.rows.delete(*self.rows.get_children())
            for tag, values in rows:
                self.rows.insert("", "end", values=values, tags=(tag,))
            self.displayed_rows = rows


def main():
    lock = acquire_instance_lock()
    if lock is None:
        print("Candidate decisions window is already running in this ROS domain.", flush=True)
        return
    with lock:
        root = tk.Tk()
        rclpy.init()
        node = EvidenceSubscriber()
        try:
            window = SelectionView(root, node)
            root.after(0, window.refresh)
            root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
