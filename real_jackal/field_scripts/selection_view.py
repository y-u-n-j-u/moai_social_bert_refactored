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
    return f"C{index}" if index is not None else "—"        # fallback paths: CF1.. / CFP / CFA


def cost_text(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


HOLD_ENABLED = os.environ.get("SELECTION_VIEW_HOLD", "1") != "0"
HOLD_REACH_M = float(os.environ.get("SELECTION_VIEW_REACH_M", "0.5"))
HOLD_MAX_S = float(os.environ.get("SELECTION_VIEW_HOLD_MAX_S", "20"))


def display_mgp(candidates, chosen):
    """Goal sample (MGP) of the selected candidate, else of the first candidate that has one (same rule as the bridge's RViz hold)."""
    first = None
    for row in candidates:
        goal = row.get("goal")
        if not goal or len(goal) < 2:
            continue
        point = (float(goal[0]), float(goal[1]))
        if row.get("valid") and chosen is not None and row.get("candidate_index") == chosen:
            return point
        if first is None:
            first = point
    return first


class TableHold:
    """Keep the shown table until the robot is within HOLD_REACH_M of its goal sample, the goal changed or HOLD_MAX_S passed
    (operator request 11-09: the table must not change at every replan). Display only."""

    def __init__(self):
        self.mgp = None
        self.generation = None
        self.shown_at = None

    def distance(self, report):
        pose = report.get("robot_pose")
        if self.mgp is None or not pose:
            return None
        return math.hypot(pose[0] - self.mgp[0], pose[1] - self.mgp[1])

    def should_refresh(self, report, now):
        if not HOLD_ENABLED or self.shown_at is None or self.mgp is None:
            return True
        if report.get("goal_generation") != self.generation or now - self.shown_at > HOLD_MAX_S:
            return True
        distance = self.distance(report)
        return distance is None or distance <= HOLD_REACH_M

    def note_shown(self, report, now):
        self.mgp = display_mgp(report.get("candidates", []), report.get("selected_candidate_index"))
        self.generation = report.get("goal_generation")
        self.shown_at = now


MAX_REJECTED_ROWS = int(os.environ.get("SELECTION_VIEW_MAX_REJECTED", "2"))   # 0 or negative = show all


def hidden_rejected(report, max_rejected):
    """How many failed candidates are left out of the table."""
    if max_rejected is None:
        return 0
    failed = sum(1 for row in [*report.get("candidates", []), *(report.get("fallback_candidates") or [])] if not row.get("valid"))
    return max(failed - max_rejected, 0)


def row_result(row, chosen, sent, selection_reason, winning_cost, selector, uniform_reason=None):
    """(tag, state, reason) of one candidate row; the same rules for the model candidates and for the fallback paths."""
    index = row.get("candidate_index")
    score = row.get("score") or {}
    if not row.get("valid"):
        return "rejected", "검사 탈락", reason_text(row.get("reason"))
    if index is not None and index == chosen:
        if sent:
            return "selected", "선택·발행", reason_text(selection_reason)
        # operator request 11-09: only three results are shown; a pick that was replaced stays "검사 통과"
        return "valid", "검사 통과", f"{selector}가 골랐지만 다른 경로가 발행됨"
    if uniform_reason:                                  # operator request 11-09: fallback paths that passed but were not picked
        return "valid", "검사 통과", uniform_reason
    if score.get("eligible") is False:
        return "valid", "검사 통과", "시작 방향 제한 초과로 선택 제외"
    reason = "다른 경로가 선택됨"
    total = score.get("total")
    if selection_reason == "minimum_weighted_cost" and total is not None and winning_cost is not None:
        if total > winning_cost + 1e-12:
            reason = "선택 경로보다 총비용이 높음"
        elif abs(total - winning_cost) <= 1e-12:
            reason = "동점에서 앞선 입력 경로 우선"
    return "valid", "검사 통과", reason


MAX_ROWS = int(os.environ.get("SELECTION_VIEW_MAX_ROWS", "5"))          # operator request 11-09: never more than 5 candidates in the table


def cap_rows(rows, maximum):
    """At most ``maximum`` rows, in the original order; the selected row is always kept (the others fill up in order)."""
    rows = list(rows)
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    keep = [i for i, (tag, _) in enumerate(rows) if tag == "selected"][:maximum]
    for i in range(len(rows)):
        if len(keep) >= maximum:
            break
        if i not in keep:
            keep.append(i)
    return [rows[i] for i in sorted(keep)]


def candidate_rows(report, max_rejected=None):
    """Format emitted facts, including safety failures with no computed cost.

    ``max_rejected`` (operator request 11-09): show at most this many rows of candidates that failed the checks (the first ones, in
    the evidence order, model candidates and fallback paths together); None = all. The number hidden is reported by ``hidden_rejected``.
    Fallback paths (built only when every model candidate was rejected) are named C + F1.. (curves to the goal samples), CFP (Nav2 route
    prefix), CFA (arc) and get the same three results and the same reasons as the model candidates.
    """
    shown_rejected = 0

    def block(rows, chosen, sent, selection_reason, selector, uniform_reason=None):
        nonlocal shown_rejected
        winner = next((r for r in rows if chosen is not None and r.get("candidate_index") == chosen), None)
        winning_cost = ((winner or {}).get("score") or {}).get("total")
        for row in rows:
            tag, state, reason = row_result(row, chosen, sent, selection_reason, winning_cost, selector, uniform_reason)
            if tag == "rejected":
                if max_rejected is not None and shown_rejected >= max_rejected:
                    continue
                shown_rejected += 1
            score = row.get("score") or {}
            yield tag, (candidate_name(row.get("candidate_index")), row.get("rank", "—"), state,
                        *(cost_text(score.get(key)) for key in COSTS), reason)

    executed_source = report.get("executed_source")
    yield from block(report["candidates"], report.get("selected_candidate_index"),
                     executed_source == "model" and report.get("selected_candidate_index") == report.get("executed_candidate_index"),
                     report.get("selection_reason"), "모델")
    fallback = report.get("fallback_candidates") or []
    if fallback:
        yield from block(fallback, report.get("selected_fallback_index"), executed_source == "fallback",
                         report.get("fallback_selection_reason"), "선택기", "선택 경로보다 총비용이 높음")


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
        self.hold = TableHold()
        self.hold_note = ""
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
        # operator request 11-09: above the table only the model choice line is shown (self.source / self.decision are still
        # updated but not displayed)
        tk.Label(summary, textvariable=self.choice, background="#172638", foreground="white",
                 font=("TkDefaultFont", 17, "bold"), anchor="w").pack(fill="x", pady=2)

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
            self.source.set(f"{mode}  |  ROS {sec}.{ns:09d}{stale}{self.hold_note}")
            if report is not self.displayed:
                now = time.monotonic()
                if self.hold.should_refresh(report, now):
                    self.show_report(report)
                    self.hold.note_shown(report, now)
                    self.hold_note = ""
                else:
                    distance = self.hold.distance(report)
                    self.hold_note = ("  |  표 고정 중" + (f" · 목표 샘플까지 {distance:.1f}m" if distance is not None else ""))
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
        self.choice.set("모델 선택")                                   # operator request 11-09: only these words, no explanation
        self.decision.set(f"추종기에 발행  {published_text}  |  {reason_text(decision_reason)}")
        limit = MAX_REJECTED_ROWS if MAX_REJECTED_ROWS > 0 else None
        hidden = hidden_rejected(report, limit)
        if hidden:
            self.decision.set(self.decision.get() + f"   (검사 탈락 {hidden}개 더 있음, 표에는 {limit}개만 표시)")
        rows = tuple(cap_rows(candidate_rows(report, limit), MAX_ROWS))
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
