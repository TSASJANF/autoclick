"""训练运行器（零 Qt）：封装 ultralytics 训练全流程。

修复历史严重缺陷：旧 TrainWorker.request_stop() 只设置了一个从未被
任何代码读取的 _stop 标志 —— 所谓「协作停止」完全无效，UI 层等待
10 秒后必然调用 QThread.terminate() 强杀线程，可能导致 CUDA 上下文
损坏和半写文件。

现通过 ultralytics 回调（on_fit_epoch_end）在每个 epoch 结束时检查
停止标志并抛出异常中止训练，训练器正常收尾（checkpoint 已保存），
进程优雅退出。停止后返回已保存的 best/last 权重。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingStopped(Exception):
    """用户请求停止，训练在 epoch 边界中止。"""


def get_device_info() -> tuple[str, str]:
    """返回 (device_str, 描述信息)。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "0", f"GPU: {torch.cuda.get_device_name(0)} (CUDA)"
        return "cpu", "GPU 不可用, 使用 CPU 训练 (安装 CUDA 版 PyTorch 可加速)"
    except ImportError:
        return "cpu", "PyTorch 未安装, 使用 CPU"


@dataclass
class TrainConfig:
    data_yaml: str
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    model_base: str = "yolov8n.pt"
    project_dir: str = ""


class TrainRunner:
    """单次训练的运行器。run() 可在任意线程执行，stop() 可从其他线程调用。"""

    def __init__(self, config: TrainConfig, on_progress=None):
        self.config = config
        self._on_progress = on_progress or (lambda msg: None)
        self._stop_requested = False

    def stop(self) -> None:
        """请求协作停止（epoch 边界生效）。线程安全：只写一个布尔。"""
        self._stop_requested = True

    def run(self) -> tuple[str, str]:
        """执行训练，返回 (状态, 模型路径)。状态: "completed" | "stopped"。

        抛出异常表示训练失败。
        """
        cfg = self.config
        try:
            return self._run_inner(cfg)
        finally:
            from .dataset import cleanup_training_dir
            cleanup_training_dir(cfg.data_yaml)

    # ── 内部 ──────────────────────────────────────────────────────────

    def _run_inner(self, cfg: TrainConfig) -> tuple[str, str]:
        from ultralytics import YOLO

        device, device_info = get_device_info()
        self._progress(f"设备: {device_info}")
        self._progress(f"加载模型: {cfg.model_base}")
        model = self._load_model_resilient(YOLO, cfg.model_base)

        # 协作停止：epoch 边界抛 TrainingStopped 中止训练
        def _check_stop(trainer):
            if self._stop_requested:
                raise TrainingStopped

        model.add_callback("on_fit_epoch_end", _check_stop)

        self._progress(f"开始训练: epochs={cfg.epochs}, batch={cfg.batch_size}, "
                       f"imgsz={cfg.img_size}")
        train_kwargs = dict(
            data=cfg.data_yaml,
            epochs=cfg.epochs,
            batch=cfg.batch_size,
            imgsz=cfg.img_size,
            device=device,
            verbose=True,
            workers=0,  # Windows 下避免多进程 DataLoader 问题
        )
        if cfg.project_dir:
            train_kwargs["project"] = cfg.project_dir

        status = "completed"
        try:
            model.train(**train_kwargs)
            # ultralytics 个别路径会吞掉回调异常：以停止标志为准
            if self._stop_requested:
                status = "stopped"
        except TrainingStopped:
            status = "stopped"
        except Exception:
            if self._stop_requested:
                status = "stopped"   # 停止过程中引出的次生异常按停止处理
            else:
                raise

        best_path = self._locate_weights(model)
        if status == "stopped":
            self._progress("训练已被用户停止，使用已保存的权重")
        else:
            self._progress(f"训练完成! 模型保存到: {best_path}")
        return status, best_path

    def _load_model_resilient(self, yolo_cls, model_base: str):
        """加载基础模型；损坏的本地文件自动删除后重试一次（重新下载）。"""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return yolo_cls(model_base)
            except Exception as e:
                if ("PytorchStreamReader" in str(e)
                        and "failed finding central directory" in str(e)):
                    last_error = e
                    if attempt == 0 and os.path.isfile(model_base):
                        self._progress(f"模型文件损坏: {model_base}，已删除并重新下载")
                        try:
                            os.remove(model_base)
                        except OSError:
                            pass
                    else:
                        raise
                else:
                    raise
        raise RuntimeError(f"模型加载失败: {last_error}")

    def _locate_weights(self, model) -> str:
        """定位训练产出的权重文件（优先 best.pt，回退 last.pt）。"""
        save_dir = getattr(model.trainer, "save_dir", None) if getattr(model, "trainer", None) else None
        candidates = []
        if save_dir:
            candidates.append(Path(save_dir) / "weights" / "best.pt")
            candidates.append(Path(save_dir) / "weights" / "last.pt")
        if self.config.project_dir:
            candidates.append(Path(self.config.project_dir) / "train" / "weights" / "best.pt")
        candidates.append(Path("runs") / "detect" / "train" / "weights" / "best.pt")
        for c in candidates:
            if c.exists():
                return str(c)
        return str(candidates[0]) if candidates else ""

    def _progress(self, msg: str) -> None:
        logger.info("[train] %s", msg)
        self._on_progress(msg)
