"""全局热键（零 Qt）：GetAsyncKeyState 轮询实现。

选择轮询而非 RegisterHotKey 的原因：RegisterHotKey 在热键被其他程序
注册时失败；轮询永远可用。50ms 周期（20Hz）CPU 占用可忽略，延迟可接受。

与旧版的区别：本类不依赖任何 UI 框架，通过回调通知；Qt 信号桥接由
UI 层完成（ui/main_window.HotkeyBridge）。
"""

from __future__ import annotations

import threading
import time
import ctypes

user32 = ctypes.windll.user32

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

_POLL_INTERVAL = 0.05  # 秒

# 修饰键虚拟键码
_MOD_KEYS = {
    MOD_CONTROL: (0x11,),          # VK_CONTROL
    MOD_ALT: (0x12,),              # VK_MENU (Alt)
    MOD_SHIFT: (0x10,),            # VK_SHIFT
    MOD_WIN: (0x5B, 0x5C),         # VK_LWIN, VK_RWIN
}

# 主键名 -> 虚拟键码
_VK_MAP = {
    'escape': 0x1B, 'esc': 0x1B, 'tab': 0x09, 'space': 0x20,
    'return': 0x0D, 'enter': 0x0D, 'backspace': 0x08, 'delete': 0x2E,
    'insert': 0x2D, 'home': 0x24, 'end': 0x23,
    'pageup': 0x21, 'pagedown': 0x22,
    'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
    'printscreen': 0x2C, 'scrolllock': 0x91, 'pause': 0x13,
    'capslock': 0x14, 'numlock': 0x90,
}
for _i in range(1, 25):
    _VK_MAP[f'f{_i}'] = 0x70 + _i - 1

# Qt QKeySequence 输出的键名 -> 虚拟键码
_QT_KEY_ALIASES = {
    'ctrl': 'control', 'meta': 'win', 'super': 'win', 'del': 'delete',
    'ins': 'insert', 'pgup': 'pageup', 'pgdown': 'pagedown',
    'esc': 'escape', 'return': 'enter',
}


def parse_key(key_str: str) -> tuple[int | None, int]:
    """解析人类可读的热键串为 (vk_code, modifier_flags)。

    接受 "F9"、"Ctrl+F9"、"Shift+Alt+A" 等格式；无法解析返回 (None, 0)。
    """
    vk: int | None = None
    mods = 0
    for raw in str(key_str).split('+'):
        part = raw.strip().lower()
        if not part:
            continue
        part = _QT_KEY_ALIASES.get(part, part)
        if part in ('ctrl', 'control'):
            mods |= MOD_CONTROL
        elif part == 'alt':
            mods |= MOD_ALT
        elif part == 'shift':
            mods |= MOD_SHIFT
        elif part == 'win':
            mods |= MOD_WIN
        elif part in _VK_MAP:
            vk = _VK_MAP[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
    return vk, mods


class HotkeyWatcher:
    """后台线程轮询热键，按下沿触发回调（回调在轮询线程执行）。"""

    def __init__(self, on_pressed):
        self._on_pressed = on_pressed
        self._thread: threading.Thread | None = None
        self._running = False
        self._vk: int | None = None
        self._mods = 0
        self._was_down = False

    def register(self, key_str: str) -> bool:
        """开始监听给定热键，返回是否解析成功。"""
        self.unregister()
        vk, mods = parse_key(key_str)
        if vk is None:
            return False
        self._vk, self._mods = vk, mods
        self._was_down = False
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop,
                                        name="hotkey-watcher", daemon=True)
        self._thread.start()
        return True

    def unregister(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _poll_loop(self) -> None:
        while self._running:
            down = self._is_down()
            if down and not self._was_down:
                try:
                    self._on_pressed()
                except Exception:  # 回调异常不能杀死轮询线程
                    pass
            self._was_down = down
            time.sleep(_POLL_INTERVAL)

    def _is_down(self) -> bool:
        """所有要求的修饰键与主键同时按下时为 True。"""
        for mod_flag, vk_codes in _MOD_KEYS.items():
            if self._mods & mod_flag:
                if not any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in vk_codes):
                    return False
        return bool(user32.GetAsyncKeyState(self._vk) & 0x8000)
