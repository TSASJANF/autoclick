"""Qt 线程适配层：检测循环 worker。

core.detection_engine 是纯逻辑引擎；本模块只负责把引擎循环搬到
QThread 上并通过信号回传帧与 FPS。

相对旧版检测线程的修复：
- BUG-02：主循环逐帧异常保护（锁屏/RDP/UAC/显存耗尽等一帧异常不再
  杀死线程），连续失败达阈值时发 error 信号并自动停止，UI 复位；
- BUG-03：running 在构造时置 True（QThread.start 到 run 首行之间存在
  竞态窗口，旧版可在窗口期内二次启动两个检测线程 → 双倍点击），
  run() 的 finally 统一复位；
- BUG-08：stop() 带超时，不无限阻塞 UI 线程。
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.detection_engine import DetectionEngine, FrameResult

logger = logging.getLogger(__name__)

# 连续异常多少帧后判定检测环境不可用，自动停止
MAX_CONSECUTIVE_ERRORS = 10


class DetectionWorker(QThread):
    """驱动 DetectionEngine 的帧循环。stop() 为协作式，不使用 terminate。"""

    frame_ready = Signal(np.ndarray, list)   # image, detections
    fps_update = Signal(float)
    error = Signal(str)                      # 致命错误（连续失败自动停止时）

    def __init__(self, engine: DetectionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        # 构造即置位：消除 start() 到 run() 首行之间的竞态窗口（BUG-03）
        self.running = True

    def run(self):
        self.engine.start()
        consecutive_errors = 0
        try:
            while self.running:
                t0 = time.perf_counter()
                try:
                    result: FrameResult = self.engine.process_frame()
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    logger.exception("检测帧失败（连续第 %d 次）", consecutive_errors)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        msg = (f"检测连续失败 {consecutive_errors} 次，已自动停止"
                               f"（常见原因：锁屏/显示器休眠/远程会话切换）")
                        self.error.emit(msg)
                        break
                    time.sleep(0.2)
                    continue

                elapsed = time.perf_counter() - t0
                self.frame_ready.emit(result.image, result.detections)
                self.fps_update.emit(1.0 / elapsed if elapsed > 0 else 0.0)

                remaining = self.engine.sleep_for_fps(
                    elapsed, self.engine.config.target_fps)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.engine.shutdown()
            self.running = False

    def stop(self, timeout_ms: int = 2000) -> bool:
        """请求停止并等待线程退出（等待当前帧完成）。

        返回线程是否已在超时内退出；超时时线程会自行退出，
        UI 不应无限阻塞等待（BUG-08）。
        """
        self.running = False
        if not self.isRunning():
            return True
        return self.wait(timeout_ms)
