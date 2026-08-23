"""统一检测引擎：采集 -> 检测 -> 点击决策，纯逻辑、零 UI 依赖。

替代历史版本中 ui/detect_tab.py 里的 SingleDetectThread 与
ParallelDetectThread（两者约 200 行重复代码，仅点击策略不同）。

关键设计：
- 每帧流程：capture -> 并行逐目标检测 -> 跨源去重 -> 点击决策 -> 点击；
- 线程池在引擎生命周期内复用（旧版每帧创建/销毁一次线程池，10FPS 下
  每秒 10 次）；
- 点击策略为纯函数 select_click_targets，可独立单元测试；
- "ai+opencv" 双模式目标的两路结果做跨源 NMS，避免同一目标被点两次
  （旧版缺陷）。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import clicker
from .capture import ScreenCapture
from .detector import YoloDetector
from .models import Detection, Target, nms
from .opencv_detector import TemplateDetector

logger = logging.getLogger(__name__)

# 每个目标的检测入口。engine 持有的 ai_detectors 以 target_id 为键，
# 各目标模型独立，因此并行推理无共享状态。
Region = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class EngineConfig:
    confidence: float = 0.5
    auto_click: bool = False
    use_priority: bool = False
    click_all_instances: bool = False   # 聚焦目标组内：点全部还是只点最高分
    region: Region | None = None        # 区域检测（None = 全屏）
    hide_window: bool = False           # 采集时隐藏前台窗口（通常是自身）
    target_fps: int = 10


@dataclass
class FrameResult:
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)


def select_click_targets(detections: list[Detection],
                         priorities: dict[str, int],
                         use_priority: bool,
                         click_all: bool) -> list[Detection]:
    """点击决策（纯函数）。

    规则（与旧版两种线程的行为保持一致）：
    1. use_priority：聚焦优先级数字最小的目标组（并列时按组内最高置信
       度决胜）；否则所有结果视为一组；
    2. click_all=True：点击组内全部实例（按置信度降序）；否则只点组内
       置信度最高的一个。
    """
    if not detections:
        return []

    if use_priority:
        groups: dict[str, list[Detection]] = {}
        for det in detections:
            groups.setdefault(det.target_id, []).append(det)
        focus = min(
            groups.values(),
            key=lambda g: (priorities.get(g[0].target_id, 999), -max(d.conf for d in g)),
        )
    else:
        focus = detections

    if click_all:
        return sorted(focus, key=lambda d: -d.conf)
    return [max(focus, key=lambda d: d.conf)]


class DetectionEngine:
    """可长期复用的检测引擎。一次 run 循环对应一个引擎实例。"""

    def __init__(self, capture: ScreenCapture,
                 target_provider,
                 ai_detectors: dict[str, YoloDetector],
                 template_detector: TemplateDetector,
                 config: EngineConfig):
        """target_provider: 返回 list[Target] 的可调用（每帧取启用目标）"""
        self.capture = capture
        self.target_provider = target_provider
        self.ai_detectors = ai_detectors
        self.template_detector = template_detector
        self.config = config
        self._executor: ThreadPoolExecutor | None = None
        self._hide_hwnd = 0

    # ── 生命周期 ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="detect")
        if self.config.hide_window:
            # 记住启动时的前台窗口（通常是我们自己），采集时隐藏避免入镜
            from . import windows
            self._hide_hwnd = windows.get_foreground_hwnd()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._hide_hwnd = 0

    # ── 单帧 ──────────────────────────────────────────────────────────

    def process_frame(self) -> FrameResult:
        """执行一帧完整流程（采集/检测/点击），供 worker 循环调用。"""
        cfg = self.config
        all_targets = self.target_provider()
        img = self._capture(cfg)
        detections = self.detect(img, all_targets)
        detections = self._dedupe_cross_source(detections)

        if cfg.auto_click and detections:
            priorities = {t.id: t.priority for t in all_targets}
            enabled_ids = {t.id for t in all_targets if t.enabled}
            # 旧版单目标线程的行为：只有一个启用目标时点其全部实例
            click_all = cfg.click_all_instances or len(enabled_ids) == 1
            targets = select_click_targets(
                detections, priorities, cfg.use_priority, click_all)
            for det in targets:
                cx, cy = det.center
                if cfg.region:
                    cx += cfg.region[0]
                    cy += cfg.region[1]
                clicker.click(cx, cy)

        return FrameResult(image=img, detections=detections)

    def detect(self, img: np.ndarray,
               targets: list[Target] | None = None) -> list[Detection]:
        """并行执行所有启用目标的检测。"""
        if targets is None:
            targets = self.target_provider()
        targets = [t for t in targets if t.enabled]
        if not targets:
            return []
        if len(targets) == 1 or self._executor is None:
            results = [self._detect_target(t, img) for t in targets]
        else:
            futures = [self._executor.submit(self._detect_target, t, img)
                       for t in targets]
            results = [f.result() for f in futures]

        detections = [d for r in results for d in r]
        if not detections:
            return []

        if self.config.use_priority:
            priorities = {t.id: t.priority for t in targets}
            detections.sort(key=lambda d: (priorities.get(d.target_id, 999), -d.conf))
        else:
            detections.sort(key=lambda d: -d.conf)
        return detections

    def _detect_target(self, target: Target, img: np.ndarray) -> list[Detection]:
        """单目标检测（可在线程池中并行执行）。"""
        results: list[Detection] = []
        try:
            if target.detection_mode in ("ai", "both"):
                detector = self.ai_detectors.get(target.id)
                if detector is not None and detector.model is not None:
                    results.extend(detector.detect(
                        img, target.id, target.name, target.color,
                        conf=self.config.confidence))
            if target.detection_mode in ("opencv", "both"):
                results.extend(self.template_detector.detect(
                    img, target.id, target.name, target.color))
        except Exception:
            logger.exception("目标 %s 检测失败", target.name)
        return results

    @staticmethod
    def _dedupe_cross_source(detections: list[Detection]) -> list[Detection]:
        """同目标 ai/opencv 双源重叠命中去重（保留高分）。"""
        return nms(detections, iou_threshold=0.5)

    def _capture(self, cfg: EngineConfig) -> np.ndarray:
        if self._hide_hwnd:
            return self.capture.capture_hide_window(self._hide_hwnd)
        if cfg.region:
            return self.capture.capture_region(*cfg.region)
        return self.capture.capture_full()

    # ── 帧率控制 ──────────────────────────────────────────────────────

    @staticmethod
    def sleep_for_fps(frame_elapsed: float, target_fps: int) -> float:
        """返回距下一帧还需睡眠的秒数（已超时返回 0）。"""
        interval = 1.0 / max(target_fps, 1)
        return max(0.0, interval - frame_elapsed)


def run_loop_step(engine: DetectionEngine,
                  on_frame, on_fps) -> None:
    """单步执行一帧并回调。worker 循环调用，便于统一计时逻辑。"""
    t0 = time.perf_counter()
    result = engine.process_frame()
    elapsed = time.perf_counter() - t0
    if on_frame:
        on_frame(result)
    if on_fps:
        on_fps(1.0 / elapsed if elapsed > 0 else 0.0)
    time.sleep(engine.sleep_for_fps(elapsed, engine.config.target_fps))
