"""领域数据模型（纯数据，零框架依赖）。

统一了历史上三个几乎相同的检测结果类（core.detector.Detection、
core.opencv_detector.OpenCVDetection、ui.detect_tab.DetectionResult），
并集中 Target / SampleSet 数据类，供存储层与业务层共享。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Optional

# 检测模式白名单
DETECTION_MODES = ("ai", "opencv", "both")

# 目标默认配色（循环取用）
TARGET_COLORS = [
    "#2ecc71", "#e74c3c", "#3498db", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#ec407a", "#00bcd4", "#8bc34a",
]


@dataclass
class Detection:
    """一次检测命中。source 为 "ai" 或 "opencv"。"""

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    target_id: str
    target_name: str
    color: str
    source: str

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def iou(self, other: "Detection") -> float:
        """与另一检测框的交并比。"""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0

    def __repr__(self) -> str:
        return (f"Detection({self.target_name}[{self.source}] "
                f"{self.conf:.2f} [{self.x1},{self.y1},{self.x2},{self.y2}])")


def nms(detections: list[Detection], iou_threshold: float = 0.3,
        same_target_only: bool = False) -> list[Detection]:
    """非极大值抑制。same_target_only=True 时仅对同一目标的框互相抑制。

    用于模板匹配的框内去重，以及 "ai+opencv" 双源结果的跨源去重
    （此时 same_target_only=False，同目标不同源的重复命中被合并）。
    """
    ranked = sorted(detections, key=lambda d: -d.conf)
    kept: list[Detection] = []
    for det in ranked:
        duplicate = False
        for k in kept:
            if same_target_only and det.target_id != k.target_id:
                continue
            if det.iou(k) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


@dataclass
class Target:
    """一个检测目标（AI 模型 / OpenCV 模板 / 双模式）。"""

    id: str
    name: str
    color: str
    priority: int = 0                      # 数字越小优先级越高
    enabled: bool = True
    detection_mode: str = "ai"             # ai | opencv | both
    # AI 模型
    model_path: str = ""                   # 存储格式：相对 APP_ROOT 或绝对
    # OpenCV 模板
    template_path: str = ""
    match_threshold: float = 0.8
    source_image_path: str = ""
    # 训练数据
    data_dir: str = ""
    set_id: str = ""                       # 所属样本集
    model_base: str = ""                   # 训练用基础模型名

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if known.get("detection_mode") not in DETECTION_MODES:
            known["detection_mode"] = "ai"
        return cls(**known)


@dataclass
class SampleSet:
    """样本集：图片 + 若干目标的标注数据 + 训练负样本策略。"""

    id: str
    name: str
    created_at: str
    negative_sample_ratio: int = 20        # 负样本占训练集百分比 (10-30)
    negative_min_positive: int = 5         # 启用负样本所需最小正样本数

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleSet":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
