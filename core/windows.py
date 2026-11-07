"""Win32 窗口枚举与查询（从 capture.py 拆出，职责单一）。"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

user32 = getattr(ctypes, "windll", None).user32 if hasattr(ctypes, "windll") else None

# 常见的无意义顶层窗口标题
_EXCLUDED_TITLES = {"Default IME", "MSCTFIME UI", ""}


def get_foreground_hwnd(own_process_only: bool = True) -> int:
    """返回前台窗口句柄。

    own_process_only=True 时只认本进程自己的窗口（BUG-10 修复）：
    「截图时隐藏程序窗口」的语义是隐藏我们自己的窗口；旧版无差别
    取前台窗口，用户在游戏里按 F12 时会每帧隐藏-恢复游戏窗口。
    前台不是本进程窗口时返回 0（调用方按不隐藏处理）。
    """
    if user32 is None:
        return 0
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return 0
    if own_process_only:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value != os.getpid():
            return 0
    return hwnd


def get_window_title(hwnd: int) -> str:
    if user32 is None:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _enum_windows(callback) -> None:
    if user32 is None:
        return

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def shim(hwnd, _):
        return callback(hwnd)

    user32.EnumWindows(shim, 0)


def find_window(title_part: str) -> int:
    """按标题关键字查找可见顶层窗口，未找到返回 0。"""
    if user32 is None:
        return 0
    result = [0]
    needle = title_part.lower()

    def callback(hwnd: int) -> bool:
        title = get_window_title(hwnd)
        if title and needle in title.lower() and user32.IsWindowVisible(hwnd):
            result[0] = hwnd
            return False
        return True

    _enum_windows(callback)
    return result[0]


def list_visible_windows() -> list[tuple[int, str]]:
    """列出所有有标题的可见顶层窗口。"""
    if user32 is None:
        return []
    windows: list[tuple[int, str]] = []

    def callback(hwnd: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            title = get_window_title(hwnd).strip()
            if title not in _EXCLUDED_TITLES:
                windows.append((hwnd, title))
        return True

    _enum_windows(callback)
    return windows
