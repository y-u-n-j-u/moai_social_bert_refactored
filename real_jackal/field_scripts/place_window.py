#!/usr/bin/python3
"""Move/resize an X11 (XWayland) window found by title: place_window.py "<title>" X Y W H

GNOME on Wayland ignores the position terminator asks for, but a move request sent to an already mapped X11 window
is honoured. Needs only the system python3 with GTK (gi + GdkX11); no wmctrl/xdotool.
"""
import re
import subprocess
import sys

import gi

gi.require_version("Gdk", "3.0")
gi.require_version("GdkX11", "3.0")
from gi.repository import Gdk, GdkX11  # noqa: E402


def find_window(title, wm_class):
    """id of the client window (not the mutter frame) whose title matches and whose class is ``wm_class``"""
    tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
    for line in tree.splitlines():
        m = re.match(r'\s+(0x[0-9a-f]+) "(.*)": \("(.*?)" "(.*?)"\)', line)
        if m and title in m.group(2) and m.group(3) == wm_class:
            return int(m.group(1), 16)
    return None


def main():
    title, x, y, w, h = sys.argv[1], *map(int, sys.argv[2:6])
    wm_class = sys.argv[6] if len(sys.argv) > 6 else "terminator"
    xid = find_window(title, wm_class)
    if xid is None:
        print("window not found", file=sys.stderr)
        return 1
    display = GdkX11.X11Display.get_default() or Gdk.Display.get_default()
    window = GdkX11.X11Window.foreign_new_for_display(display, xid)
    window.move_resize(x, y, w, h)
    display.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
