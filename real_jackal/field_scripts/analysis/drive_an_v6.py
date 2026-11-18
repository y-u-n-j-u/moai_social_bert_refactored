#!/usr/bin/env python3
"""주행 기록(rosbag) 분석 v6: 자율 구간의 속도, 정지, 회전, 경로 출처, never_hold / never_hold_route, fallback, 사람, 센서 끊김.

사용: python3 drive_an_v6.py <bag 폴더>   (컨테이너 안, ROS 환경 필요. 열려 있는(녹화 중인) bag은 열지 말 것)
자율 구간 = 처음 움직인 시각 ~ 브릿지가 goal_reached를 낸 시각(없으면 마지막 증거). 도착 뒤 컨트롤러로 옮긴 구간은 뺀다.
"""
import collections
import json
import math
import statistics as st
import sys

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

KEEP = ("/j100_0519/cmd_vel", "/aft_mapped_to_init", "/spu_bert/tracker_status", "/spu_bert/selection_evidence",
        "/ped_tracking", "/scan")


def load(bag):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id="sqlite3"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    data, t0 = collections.defaultdict(list), None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        t = stamp * 1e-9
        t0 = t if t0 is None else t0
        if topic in KEEP:
            data[topic].append((t - t0, deserialize_message(raw, get_message(types[topic]))))
    return data


def main():
    data = load(sys.argv[1])
    odom = [(t, m.pose.pose.position.x, m.pose.pose.position.y) for t, m in data["/aft_mapped_to_init"]]
    cmd = [(t, m.linear.x, m.angular.z) for t, m in data["/j100_0519/cmd_vel"]]
    ev = [(t, json.loads(m.data)) for t, m in data["/spu_bert/selection_evidence"]]
    status = [(t, m.data) for t, m in data["/spu_bert/tracker_status"]]
    start = next((t for t, v, w in cmd if abs(v) > 0.05), cmd[0][0] if cmd else 0.0)
    reached = [t for t, d in ev if d.get("decision_reason") == "goal_reached"]
    end = reached[0] if reached else ev[-1][0]
    print("기록 길이 %.0f초 | 첫 움직임 %.0f초 | 목표 도착 판정 %s" % (odom[-1][0], start, "%.0f초" % end if reached else "없음(마지막 증거 시각 사용)"))

    seg = [x for x in odom if start <= x[0] <= end]
    dist = sum(math.dist(seg[i][1:], seg[i + 1][1:]) for i in range(len(seg) - 1))
    dur = max(end - start, 1e-6)
    # 속도: 1초 간격 위치 차이(실제 속도) 와 명령 속도
    per_sec = []
    for a in range(int(start) + 1, int(end)):
        p0 = min(odom, key=lambda x: abs(x[0] - a))
        p1 = min(odom, key=lambda x: abs(x[0] - (a + 1)))
        per_sec.append(math.dist(p0[1:], p1[1:]))
    cv = [v for t, v, w in cmd if start <= t <= end]
    cw = [w for t, v, w in cmd if start <= t <= end]
    print("\n== 속도 ==")
    print("자율 구간 %.0f초, 이동 %.1f m, 평균 %.2f m/s" % (dur, dist, dist / dur))
    if per_sec:
        s = sorted(per_sec)
        print("실제 속도(1초 단위): 중앙값 %.2f, 상위 10%% %.2f, 최대 %.2f m/s | 0.35 m/s 이상이었던 초 %d%%" % (
            st.median(s), s[int(0.9 * (len(s) - 1))], s[-1], 100 * sum(1 for x in s if x >= 0.35) // len(s)))
    if cv:
        cs = sorted(cv)
        print("명령 속도: 최대 %.2f, 중앙값 %.2f | 0.38 이상인 명령 %d%%" % (cs[-1], st.median(cs), 100 * sum(1 for x in cv if x >= 0.38) // len(cv)))

    slow = [a for a in range(int(start), int(end)) if (lambda L: (sum(L) / len(L) if L else 0.0) < 0.10)([v for t, v, w in cmd if a <= t < a + 1])]
    print("\n== 정지/느림 ==")
    print("명령 속도가 0.10 m/s 미만인 초: %d (%.0f%%) %s" % (len(slow), 100 * len(slow) / dur, slow[:20]))
    rot = sum(abs(w) for w in cw) * (dur / max(len(cw), 1))
    print("회전량 %.0f도 = 10 m당 %.0f도 (지난 주행 11-09: 312~481도)" % (math.degrees(rot), math.degrees(rot) / max(dist / 10, 1e-3)))

    window = [(t, d) for t, d in ev if start <= t <= end]
    src = collections.Counter(d["executed_source"] for _, d in window)
    print("\n== 경로 출처 (주기 %d개) ==" % len(window))
    print(dict(src))
    keep = [d for _, d in window if str(d.get("decision_reason", "")).startswith("keep_moving")]
    route = [d for _, d in window if str(d.get("decision_reason", "")).startswith("never_hold_route")]
    print("never_hold 마지막 경로 따라가기: %d주기 %s" % (len(keep), dict(collections.Counter(d["decision_reason"][12:46] for d in keep))))
    print("never_hold_route(global path 추종): %d주기 %s" % (len(route), dict(collections.Counter(d["decision_reason"][17:52] for d in route))))
    holds = [(round(t), d["decision_reason"]) for t, d in window if d["executed_source"] == "hold"]
    print("빈 경로(보류) 주기: %d %s" % (len(holds), holds[:8]))
    V = sum(1 for _, d in window for c in d["candidates"] if c["valid"])
    I = sum(1 for _, d in window for c in d["candidates"] if not c["valid"])
    alli = [d for _, d in window if d["candidates"] and not any(c["valid"] for c in d["candidates"])]
    print("모델 후보 탈락 %.0f%% | 전부 탈락한 주기 %d (%.0f%%) -> 발행 %s" % (
        100 * I / max(V + I, 1), len(alli), 100 * len(alli) / max(len(window), 1), dict(collections.Counter(d["executed_source"] for d in alli))))
    fb = [d for _, d in window if d.get("fallback_candidates")]
    mg = [r for d in fb for r in d["fallback_candidates"] if r["kind"] == "mgp"]
    sel = collections.Counter(r["kind"] for d in fb for r in d["fallback_candidates"] if r["selected"])
    print("fallback 주기 %d | MGP 곡선 %d개 중 통과 %d%% | 선택된 종류 %s" % (len(fb), len(mg), 100 * sum(r["valid"] for r in mg) // max(len(mg), 1), dict(sel)))

    stl = collections.Counter(m.split(" ")[0][:34] for t, m in status if start <= t <= end and not m.startswith("tracking"))
    print("추종기 비추종 상태(window): %s" % dict(stl.most_common(5)))

    print("\n== 사람, 센서 ==")
    tr = [(t, m) for t, m in data["/ped_tracking"] if start <= t <= end]
    dm = [min(math.hypot(k.pose.position.x, k.pose.position.y) for k in m.tracks) for t, m in tr if m.tracks]
    print("사람: 시야 %.0f%% 프레임, 최소 거리 %s" % (100 * len(dm) / max(len(tr), 1), "%.2f m" % min(dm) if dm else "없음"))
    for name, series in (("odom", odom), ("scan", [(t, 0) for t, m in data["/scan"]])):
        gaps = [(round(series[i][0], 1), round(series[i + 1][0] - series[i][0], 2)) for i in range(len(series) - 1)
                if series[i + 1][0] - series[i][0] > 0.4 and start <= series[i][0] <= end]
        print("%s 끊김(0.4초 초과): %s" % (name, gaps[:6] or "없음"))
    print("\n도착 뒤 컨트롤러로 옮긴 구간은 제외함. 끝 위치 odom x %.1f" % odom[-1][1])


if __name__ == "__main__":
    main()
