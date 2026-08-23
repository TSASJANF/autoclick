"""Qt 线程适配层：AI 模型后台加载器。

修复历史 UX 缺陷：旧版在 UI 线程同步加载 YOLO 模型（每个 1~3 秒），
目标多时启动/刷新会卡住界面数秒；且模型文件缺失时 ultralytics 会尝试
联网下载，每次页签可见（showEvent）都重复失败流程。

本加载器在后台线程逐个加载模型：
- 文件不存在的目标直接记为失败（不触发网络下载），路径变化前不重试；
- 加载完成后一次性发回结果字典，由 DetectTab 原子替换。
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QThread, Signal

from core.detector import YoloDetector

logger = logging.getLogger(__name__)


class ModelLoader(QThread):
    """后台加载一组目标的 AI 模型。"""

    loaded = Signal(dict)       # {target_id: YoloDetector}
    failed = Signal(list)       # [(target_id, model_path, error)]

    def __init__(self, jobs: list[tuple[str, str]], parent=None):
        """jobs: [(target_id, model_path)]"""
        super().__init__(parent)
        self._jobs = jobs

    def run(self):
        ok: dict[str, YoloDetector] = {}
        errors: list[tuple[str, str, str]] = []
        for target_id, model_path in self._jobs:
            try:
                det = YoloDetector()
                det.load(model_path)
                ok[target_id] = det
            except FileNotFoundError as e:
                errors.append((target_id, model_path, str(e)))
            except Exception as e:
                logger.exception("模型加载失败: %s", model_path)
                errors.append((target_id, model_path, str(e)))
        if ok:
            self.loaded.emit(ok)
        if errors:
            self.failed.emit(errors)
