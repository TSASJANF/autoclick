"""Qt 线程适配层：训练 worker。

修复历史缺陷：
1. 旧 TrainWorker 定义了与 QThread 内置同名的 ``finished`` 信号将其
   遮蔽 —— 现更名为 succeeded / stopped，不再与基类冲突；
2. 旧版 stop 逻辑失效必然 terminate() 强杀线程；现在 core.TrainRunner
   通过 ultralytics 回调实现真正的协作停止，worker 无需 terminate。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from core.train_runner import TrainConfig, TrainRunner

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(str)     # 模型路径（训练完成）
    stopped = Signal(str)       # 模型路径（用户停止，使用已存权重）
    failed = Signal(str)        # 错误信息

    def __init__(self, config: TrainConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._runner = TrainRunner(config, on_progress=self.progress.emit)

    def request_stop(self):
        self._runner.stop()

    def run(self):
        try:
            status, model_path = self._runner.run()
        except Exception as e:
            logger.exception("训练失败")
            self.failed.emit(str(e))
            return
        if status == "stopped":
            self.stopped.emit(model_path)
        else:
            self.succeeded.emit(model_path)
