"""ndarray <-> Qt 图像转换工具。

修复历史缺陷：旧代码 ``QImage(rgb.data, ...)`` 不拥有 numpy 缓冲区，
 QImage 与源数组生命周期解耦后是悬空指针（依赖恰好同作用域的侥幸）。
 此处一律 .copy() 让 QImage 拥有自己的数据。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qimage(img: np.ndarray) -> QImage:
    """BGR ndarray -> 拥有独立数据的 QImage。"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(bgr_to_qimage(img))


def hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)。"""
    return (int(color_hex[1:3], 16),
            int(color_hex[3:5], 16),
            int(color_hex[5:7], 16))


def bgr_color(color_hex: str) -> tuple[int, int, int]:
    """'#rrggbb' -> OpenCV 的 (b, g, r)。"""
    r, g, b = hex_to_rgb(color_hex)
    return (b, g, r)
