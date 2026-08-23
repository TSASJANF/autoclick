"""数据生命周期管理（SEC-03）。

历史问题：data/screenshots/（含 opencv 子目录）保存用户真实屏幕截图，
无保留策略、无清理入口——分发/共享项目目录时直接泄漏用户屏幕历史。
现启动时按保留上限自动清理最旧的截图。
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import paths

logger = logging.getLogger(__name__)

DEFAULT_KEEP = 200          # 截图保留上限（按修改时间保留最新）
SCREENSHOTS_DIR = Path("data") / "screenshots"


def cleanup_screenshots(keep: int = DEFAULT_KEEP) -> int:
    """清理截图目录，保留最新 keep 张，返回删除数量。"""
    root = (paths.APP_ROOT / SCREENSHOTS_DIR).resolve()
    if not root.is_dir():
        return 0
    files = [p for p in root.rglob("*.png") if p.is_file()]
    if len(files) <= keep:
        return 0
    files.sort(key=lambda p: p.stat().st_mtime)
    removed = 0
    for old in files[:len(files) - keep]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            continue
    if removed:
        logger.info("已清理 %d 张过期截图（保留最新 %d 张）", removed, keep)
    return removed
