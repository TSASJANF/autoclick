"""YOLO（ultralytics）检测器封装。

修复历史缺陷：旧版对任意路径直接调 YOLO(path)，文件不存在时
ultralytics 会把它当作模型名尝试联网下载 —— UI 里表现为切个页签就
触发网络请求。现先校验文件存在，不存在直接抛 FileNotFoundError。
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .models import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """单个目标的 YOLO 模型。每个 Target 独立持有实例（推理无跨线程共享）。"""

    def __init__(self, model_path: str = ""):
        self.model = None
        self.model_path = model_path
        self.names: dict[int, str] = {}

    def load(self, model_path: str | None = None) -> None:
        path = model_path or self.model_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        from ultralytics import YOLO  # 延迟导入，缩短应用启动时间
        self.model = YOLO(path)
        self.names = self.model.names
        self.model_path = path
        logger.info("AI 模型已加载: %s", path)

    def detect(self, image: np.ndarray, target_id: str = "", target_name: str = "",
               color: str = "", conf: float = 0.5) -> list[Detection]:
        if self.model is None:
            return []
        results = self.model(image, conf=conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(
                    x1, y1, x2, y2, float(box.conf[0]),
                    target_id, target_name, color, "ai",
                ))
        return detections
