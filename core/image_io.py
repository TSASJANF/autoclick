"""Unicode 安全的图像读写。

真实缺陷修复：Windows 上 cv2.imread/imwrite 不支持含非 ASCII 字符的
路径（本项目目录名含中文即触发，表现为模板/图片全部加载失败而程序
不报任何错）。统一改用 np.fromfile + imdecode / imencode + tofile，
行为与 cv2 原生函数一致且对任意路径安全。

全项目禁止直接调用 cv2.imread/cv2.imwrite，一律使用本模块。
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图像（任意路径安全）。失败返回 None，与 cv2.imread 一致。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path: str, img: np.ndarray) -> bool:
    """写图像（任意路径安全）。成功返回 True。"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except OSError:
        logger.exception("图像写入失败: %s", path)
        return False
