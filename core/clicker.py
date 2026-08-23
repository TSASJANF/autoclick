import ctypes
import time

# Windows SendInput constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

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


def _send_input(*inputs: INPUT):
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    user32.SendInput(n, arr, ctypes.sizeof(INPUT))


class Clicker:
    @staticmethod
    def click(x: int, y: int, move_first: bool = True):
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)

        # Convert to absolute coordinates (0-65535)
        abs_x = int(x * 65535 / screen_w)
        abs_y = int(y * 65535 / screen_h)

        inputs = []

        if move_first:
            move = INPUT()
            move.type = INPUT_MOUSE
            move._input.mi.dx = abs_x
            move._input.mi.dy = abs_y
            move._input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
            inputs.append(move)

        down = INPUT()
        down.type = INPUT_MOUSE
        down._input.mi.dx = abs_x
        down._input.mi.dy = abs_y
        down._input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE
        inputs.append(down)

        up = INPUT()
        up.type = INPUT_MOUSE
        up._input.mi.dx = abs_x
        up._input.mi.dy = abs_y
        up._input.mi.dwFlags = MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE
        inputs.append(up)

        _send_input(*inputs)
