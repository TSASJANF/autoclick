"""鼠标点击（Win32 SendInput）。

修复历史缺陷：旧实现用 GetSystemMetrics(0/1)（仅主显示器）做绝对坐标
换算，多显示器环境下副屏坐标会被错误映射到主屏。现使用虚拟桌面
（SM_*VIRTUALSCREEN）+ MOUSEEVENTF_VIRTUALDESK；单显示器下虚拟桌面
即主屏，行为与旧版一致。
"""

from __future__ import annotations

import ctypes

# Windows SendInput 常量
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

user32 = ctypes.windll.user32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_input", _INPUT),
    ]


def virtual_screen_rect() -> tuple[int, int, int, int]:
    """返回 (x, y, w, h)：整个虚拟桌面的原点与尺寸（供安全边界判定）。"""
    return (
        user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
    )


def _to_absolute(x: int, y: int) -> tuple[int, int]:
    """屏幕像素坐标 -> 虚拟桌面绝对坐标 (0-65535)。"""
    vx, vy, vw, vh = virtual_screen_rect()
    abs_x = int((x - vx) * 65535 / max(vw, 1))
    abs_y = int((y - vy) * 65535 / max(vh, 1))
    return abs_x, abs_y


def _send_input(*inputs: INPUT) -> None:
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    user32.SendInput(n, arr, ctypes.sizeof(INPUT))


def click(x: int, y: int, move_first: bool = True) -> None:
    """在屏幕像素坐标 (x, y) 处模拟一次左键点击。"""
    abs_x, abs_y = _to_absolute(x, y)
    flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    inputs: list[INPUT] = []

    if move_first:
        move = INPUT()
        move.type = INPUT_MOUSE
        move._input.mi.dx = abs_x
        move._input.mi.dy = abs_y
        move._input.mi.dwFlags = MOUSEEVENTF_MOVE | flags
        inputs.append(move)

    for event_flag in (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP):
        ev = INPUT()
        ev.type = INPUT_MOUSE
        ev._input.mi.dx = abs_x
        ev._input.mi.dy = abs_y
        ev._input.mi.dwFlags = event_flag | flags
        inputs.append(ev)

    _send_input(*inputs)
