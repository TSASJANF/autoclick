"""Qt 线程适配层：检测循环 worker。

core.detection_engine 是纯逻辑引擎；本模块只负责把引擎循环搬到
QThread 上并通过信号回传帧与 FPS。替代旧版 ui/detect_tab.py 中职责
重复的 SingleDetectThread / ParallelDetectThread 两个类。
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.detection_engine import DetectionEngine, FrameResult


class DetectionWorker(QThread):
    """驱动 DetectionEngine 的帧循环。stop() 为协作式，不使用 terminate。"""

    frame_ready = Signal(np.ndarray, list)   # image, detections
    fps_update = Signal(float)

    def __init__(self, engine: DetectionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.running = False

    def run(self):
        self.running = True
        self.engine.start()
        try:
            while self.running:
                t0 = time.perf_counter()
                result: FrameResult = self.engine.process_frame()
                elapsed = time.perf_counter() - t0

                self.frame_ready.emit(result.image, result.detections)
                self.fps_update.emit(1.0 / elapsed if elapsed > 0 else 0.0)

                remaining = self.engine.sleep_for_fps(
                    elapsed, self.engine.config.target_fps)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.engine.shutdown()

    def stop(self):
        """请求停止并等待线程退出（等待当前帧完成，通常 < 1 帧）。"""
        self.running = False
        self.wait(5000)
