import os
import time
import datetime
import ctypes
from ctypes import wintypes
import numpy as np
import mss
import cv2

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
dwmapi = ctypes.windll.dwmapi

PW_RENDERFULLCONTENT = 0x00000002  # Win10+: render window content even if occluded

# GDI constants
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


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


class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()

    def capture_full(self) -> np.ndarray:
        monitor = self.sct.monitors[0]
        img = self.sct.grab(monitor)
        return np.array(img)[:, :, :3]

    def capture_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        region = {"left": x, "top": y, "width": w, "height": h}
        img = self.sct.grab(region)
        return np.array(img)[:, :, :3]

    def capture_hide_window(self, hwnd: int = 0) -> np.ndarray:
        if hwnd:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
            time.sleep(0.05)
        img = self.capture_full()
        if hwnd:
            user32.ShowWindow(hwnd, 5)  # SW_SHOW
        return img

    def capture_window(self, hwnd: int) -> np.ndarray:
        """Capture a specific window using PrintWindow (same as OBS Win10 capture).

        Uses PW_RENDERFULLCONTENT to render the window's own content pipeline,
        avoiding occlusion from other windows in front of it.
        """
        # Get window rect
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))

        # Try DWM extended frame bounds for accurate size (excludes shadow)
        dwm_rect = wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        hr = dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(dwm_rect), ctypes.sizeof(dwm_rect),
        )
        if hr == 0:
            rect = dwm_rect

        x, y = rect.left, rect.top
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return self.capture_full()

        # Create memory DC and bitmap
        hwnd_dc = user32.GetDC(hwnd)
        if not hwnd_dc:
            return self.capture_full()
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        if not mem_dc:
            user32.ReleaseDC(hwnd, hwnd_dc)
            return self.capture_full()
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
        if not bitmap:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)
            return self.capture_full()
        old_bmp = gdi32.SelectObject(mem_dc, bitmap)

        # PrintWindow with PW_RENDERFULLCONTENT
        # This is the key: renders window content from its own pipeline,
        # not from the screen, so occluding windows don't matter.
        result = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        if not result:
            # Fallback: try without the flag (works for GDI apps)
            user32.PrintWindow(hwnd, mem_dc, 0)

        # Read bitmap data
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), DIB_RGB_COLORS)

        # Cleanup
        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)

        # Convert BGRA -> BGR
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return img[:, :, :3].copy()

    def get_monitors(self) -> list[dict]:
        return self.sct.monitors[1:]

    @staticmethod
    def get_foreground_hwnd() -> int:
        return user32.GetForegroundWindow()

    @staticmethod
    def get_window_title(hwnd: int) -> str:
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    @staticmethod
    def find_window(title_part: str) -> int:
        result = [0]

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_callback(hwnd, _):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if title_part.lower() in buf.value.lower() and user32.IsWindowVisible(hwnd):
                    result[0] = hwnd
                    return False
            return True

        user32.EnumWindows(enum_callback, 0)
        return result[0]

    @staticmethod
    def list_visible_windows() -> list[tuple[int, str]]:
        windows = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_callback(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    if title and title != "Default IME" and title != "MSCTFIME UI":
                        windows.append((hwnd, title))
            return True

        user32.EnumWindows(enum_callback, 0)
        return windows

    @staticmethod
    def save_screenshot(img: np.ndarray, directory: str, prefix: str = "screenshot") -> str:
        """Save a screenshot to the specified directory with a timestamp filename.

        Args:
            img: The image array (BGR format from OpenCV)
            directory: The directory to save to
            prefix: Filename prefix (default: "screenshot")

        Returns:
            The full path of the saved screenshot
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, img)
        return filepath
