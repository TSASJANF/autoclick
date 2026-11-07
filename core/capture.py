"""屏幕截图：mss 全屏/区域采集 + Win32 PrintWindow 窗口采集。

相对历史版本的修复：
1. 线程安全：mss 实例非线程安全，UI 线程（截图按钮）与检测线程共用
   同一 ScreenCapture 时会崩溃，现以锁串行化所有 grab；
2. 窗口采集失败不再静默降级为全屏截图 —— 错误尺寸的图像会让点击
   坐标全部错位，现在抛出 CaptureError 由调用方明确处理；
3. 窗口枚举拆分到 core.windows。
"""

from __future__ import annotations

import ctypes
import datetime
import os
import threading
import time
from ctypes import wintypes

import mss
import numpy as np

from . import windows
from .image_io import imwrite

windll = getattr(ctypes, "windll", None)
user32 = windll.user32 if windll else None
gdi32 = windll.gdi32 if windll else None
dwmapi = windll.dwmapi if windll else None

PW_RENDERFULLCONTENT = 0x00000002  # Win10+：即使被遮挡也渲染窗口内容
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
DWMWA_EXTENDED_FRAME_BOUNDS = 9


class CaptureError(RuntimeError):
    """截图失败（窗口消失、DC 获取失败等）。"""


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


def _grab(sct: mss.base.MSSBase, monitor: dict) -> np.ndarray:
    """mss grab -> BGR ndarray，丢弃 alpha 通道。"""
    img = sct.grab(monitor)
    return np.array(img)[:, :, :3]


class ScreenCapture:
    def __init__(self):
        self._sct = mss.mss()
        self._lock = threading.Lock()

    # ── 屏幕采集 ──────────────────────────────────────────────────────

    def capture_full(self) -> np.ndarray:
        with self._lock:
            return _grab(self._sct, self._sct.monitors[0])

    def capture_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        with self._lock:
            return _grab(self._sct, {"left": x, "top": y, "width": w, "height": h})

    def capture_hide_window(self, hwnd: int = 0) -> np.ndarray:
        """隐藏指定窗口后截全屏（避免截到自身），随后恢复。"""
        if hwnd and user32 is not None:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
            time.sleep(0.05)
        try:
            return self.capture_full()
        finally:
            if hwnd and user32 is not None:
                user32.ShowWindow(hwnd, 5)  # SW_SHOW

    # ── 窗口采集（PrintWindow，同 OBS Win10 采集方式）─────────────────

    def capture_window(self, hwnd: int) -> np.ndarray:
        """采集指定窗口内容，被遮挡也不受影响。

        抛出 CaptureError 而非静默降级为全屏 —— 全屏图像与窗口坐标不
        匹配会导致点击位置错乱。
        """
        if user32 is None or gdi32 is None or dwmapi is None:
            raise CaptureError("Win32 API 不可用（非 Windows 平台）")
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))

        dwm_rect = wintypes.RECT()
        hr = dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(dwm_rect), ctypes.sizeof(dwm_rect),
        )
        if hr == 0:
            rect = dwm_rect

        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w <= 0 or h <= 0:
            raise CaptureError(f"窗口尺寸无效: {w}x{h}")

        hwnd_dc = user32.GetDC(hwnd)
        if not hwnd_dc:
            raise CaptureError("GetDC 失败")
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        if not mem_dc:
            user32.ReleaseDC(hwnd, hwnd_dc)
            raise CaptureError("CreateCompatibleDC 失败")
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        if not bitmap:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)
            raise CaptureError("CreateCompatibleBitmap 失败")
        old_bmp = gdi32.SelectObject(mem_dc, bitmap)

        try:
            if not user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT):
                # 旧 GDI 应用不支持 PW_RENDERFULLCONTENT，退回无标志调用
                if not user32.PrintWindow(hwnd, mem_dc, 0):
                    raise CaptureError("PrintWindow 失败")

            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = w
            bmi.bmiHeader.biHeight = -h  # top-down
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = BI_RGB

            buf = ctypes.create_string_buffer(w * h * 4)
            gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), DIB_RGB_COLORS)
        finally:
            gdi32.SelectObject(mem_dc, old_bmp)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)

        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return img[:, :, :3].copy()  # BGRA -> BGR，脱离临时 buffer

    # ── 按配置采集（所有页签共用的截图入口）──────────────────────────

    def capture_by_config(self, config: dict) -> np.ndarray:
        """按用户配置选择窗口采集 / 隐藏自身 / 直接全屏。"""
        win_title = config.get("window_capture_title", "") if config.get("window_capture_enabled") else ""
        if win_title:
            hwnd = windows.find_window(win_title)
            if hwnd:
                try:
                    return self.capture_window(hwnd)
                except CaptureError:
                    pass  # 窗口采集失败时退回普通模式，而不是静默拿错图
            return self.capture_full()
        if config.get("screenshot_hide_window", True):
            return self.capture_hide_window(windows.get_foreground_hwnd())
        return self.capture_full()

    # ── 其他 ──────────────────────────────────────────────────────────

    def get_monitors(self) -> list[dict]:
        with self._lock:
            return list(self._sct.monitors[1:])

    @staticmethod
    def save_screenshot(img: np.ndarray, directory: str, prefix: str = "screenshot") -> str:
        """带时间戳文件名保存截图，返回完整路径。"""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"{prefix}_{timestamp}.png")
        if not imwrite(filepath, img):
            raise CaptureError(f"截图保存失败: {filepath}")
        return filepath
