"""OpenCV 模板匹配检测器。

相对历史版本的性能修复（本模块运行在 10FPS 检测热路径上）：
1. 移除每帧对 matchTemplate 结果矩阵的 min/mean 全图归约 —— 旧版这两
   个值只为 debug 日志计算，纯浪费；
2. 阈值命中点不再全量生成 Detection 再 O(n²) NMS。平坦匹配区会产生
   数千个相邻命中点，旧版先构造数千对象再两两IoU。现按匹配分数取
   top-K 候选再抑制，复杂度与候选数解耦；
3. 结构体统一为 core.models.Detection。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

from .image_io import imread
from .models import Detection, nms

logger = logging.getLogger(__name__)

# 每个模板最多保留的候选命中数（按分数排序），随后做 NMS
_MAX_CANDIDATES = 32


class TemplateDetector:
    """多模板管理 + matchTemplate 检测。目标间模板互不影响，线程安全
    （dict 读写依赖 GIL 原子性；检测期间模板集快照使用）。"""

    def __init__(self):
        self.templates: dict[str, np.ndarray] = {}   # target_id -> BGR 模板
        self.thresholds: dict[str, float] = {}       # target_id -> 阈值

    def load_template(self, target_id: str, template_path: str,
                      threshold: float = 0.8) -> bool:
        """加载模板图，成功返回 True。"""
        img = imread(template_path)
        if img is None:
            logger.warning("模板加载失败: %s, 路径: %s", target_id, template_path)
            return False
        self.templates[target_id] = img
        self.thresholds[target_id] = threshold
        return True

    def remove_template(self, target_id: str) -> None:
        self.templates.pop(target_id, None)
        self.thresholds.pop(target_id, None)

    def has_template(self, target_id: str) -> bool:
        return target_id in self.templates

    def detect(self, image: np.ndarray, target_id: str = "",
               target_name: str = "", color: str = "",
               threshold_override: Optional[float] = None) -> list[Detection]:
        """对给定目标 ID 列表（缺省为全部已加载模板）做模板匹配。"""
        ids = [target_id] if target_id else list(self.templates.keys())
        detections: list[Detection] = []

        templates = dict(self.templates)   # 快照，避免与并发加载竞争
        thresholds = dict(self.thresholds)

        for tid in ids:
            template = templates.get(tid)
            if template is None:
                continue
            threshold = threshold_override or thresholds.get(tid, 0.8)
            th, tw = template.shape[:2]
            if image.shape[0] < th or image.shape[1] < tw:
                continue  # 模板比图像还大，必然无匹配

            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            t_name = target_name if (tid == target_id or not target_id) else tid
            t_color = color if (tid == target_id or not target_id) else ""
            detections.extend(self._extract_hits(
                result, tw, th, threshold, tid, t_name, t_color))

        # 同一目标内的重叠命中去重
        return nms(detections, iou_threshold=0.3, same_target_only=True)

    @staticmethod
    def _extract_hits(result: np.ndarray, tw: int, th: int, threshold: float,
                      tid: str, target_name: str, color: str) -> list[Detection]:
        """从匹配分数矩阵提取 top-K 候选命中。"""
        n = result.size
        k = min(_MAX_CANDIDATES, n)
        if k < n:
            # argpartition 取 top-K 分数位置，避免全矩阵排序
            flat_idx = np.argpartition(result.ravel(), n - k)[n - k:]
        else:
            flat_idx = np.arange(n)

        hits: list[Detection] = []
        for idx in flat_idx:
            y, x = divmod(int(idx), result.shape[1])
            score = float(result[y, x])
            if score >= threshold:
                hits.append(Detection(x, y, x + tw, y + th, score,
                                      tid, target_name, color, "opencv"))
        return hits
