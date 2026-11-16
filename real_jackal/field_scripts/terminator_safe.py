#!/usr/bin/python3
"""terminator를 그대로 실행하되, 레이아웃을 만들 때 가끔 무한 반복에 빠지는 결함만 막는다.

원인: terminatorlib/paned.py 의 set_position_by_ratio 에는
    while doing_layout and get_length() == 1: (이벤트 처리 반복)
가 있는데, 창 크기가 정해지지 않으면 끝나지 않는다(CPU 100%, 칸이 중간까지만 열리고 입력 불가).
여기서는 같은 반복을 최대 MAX_WAIT_S 초까지만 하고 넘어간다. 이후 창 크기가 정해지면 terminator가 비율을 다시 맞춘다.
사용법은 terminator와 같다:  terminator_safe.py --no-dbus -g 설정파일 -l 레이아웃 -T 제목
환경변수 TERMINATOR_SAFE_DEBUG=파일 이 있으면 6초 뒤 각 칸의 크기를 그 파일에 적는다.
"""
import os
import runpy
import sys
import time

MAX_WAIT_S = float(os.environ.get("TERMINATOR_SAFE_MAX_WAIT_S", "0.4"))

import gi  # noqa: E402

gi.require_version("Gtk", "3.0")  # terminator와 같은 GTK 3 (먼저 지정하지 않으면 4.0이 불려서 오류)
from gi.repository import GLib, Gtk  # noqa: E402
from terminatorlib import paned  # noqa: E402


def set_position_by_ratio(self):
    ratio = self.ratio
    deadline = time.monotonic() + MAX_WAIT_S
    while self.terminator.doing_layout and self.get_length() == 1 and time.monotonic() < deadline:
        while Gtk.events_pending():
            Gtk.main_iteration()
        time.sleep(0.002)
    self.ratio = ratio
    self.set_pos(self.position_by_ratio(self.get_length(), self.get_handlesize(), self.ratio))


paned.Paned.set_position_by_ratio = set_position_by_ratio

debug_file = os.environ.get("TERMINATOR_SAFE_DEBUG")
if debug_file:
    def dump():
        from terminatorlib.terminator import Terminator
        with open(debug_file, "w") as handle:
            for number, term in enumerate(Terminator().terminals):
                a = term.get_allocation()
                handle.write(f"{number} {a.width}x{a.height}\n")
        return False
    GLib.timeout_add(6000, dump)

sys.argv[0] = "/usr/bin/terminator"
runpy.run_path("/usr/bin/terminator", run_name="__main__")
