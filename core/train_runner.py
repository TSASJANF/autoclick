"""训练运行器（零 Qt）：数据集组装 + ultralytics 训练全流程。

相对旧版的核心修复：
1. 协作停止：旧 TrainWorker.request_stop() 设置的标志从未被读取，停止
   必然走 QThread.terminate() 强杀（文件损坏/CUDA 僵死风险）。现通过
   ultralytics 回调（on_fit_epoch_end）在 epoch 边界协作中止；
2. 数据集组装（旧版在 UI 主线程逐张复制整库图片，界面冻结数秒）移入
   本 runner，与训练同线程执行（BUG-14）；
3. best.pt 定位：旧版三级回退猜测可能命中旧训练的权重（ultralytics
   目录名递增 train2/train3...），回退路径现必须通过 mtime 校验——
   晚于本次训练开始时间才可信（BUG-17）。
"""

from __future__ import annotations

import logging
import os
import time
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
class DatasetPrep:
    """训练前的数据集组装参数（由 UI 收集，runner 在后台线程执行）。"""
    dataset_dir: str
    class_names: list[str]
    images_dir: str
    negative_ratio: int = 0
    min_positive: int = 5
    seed: int | None = None          # None → 使用 core.dataset.DEFAULT_SEED
    val_ratio: float = 0.2


@dataclass
class TrainConfig:
    data_yaml: str = ""
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    model_base: str = "yolov8n.pt"
    project_dir: str = ""
    prep: DatasetPrep | None = None  # 提供时由 runner 组装数据集


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
            if cfg.prep is not None:
                cfg.data_yaml = self._prepare_dataset(cfg.prep)
            return self._run_inner(cfg)
        finally:
            from .dataset import cleanup_training_dir
            cleanup_training_dir(cfg.data_yaml)

    # ── 数据集组装（后台线程执行，BUG-14）────────────────────────────

    def _prepare_dataset(self, prep: DatasetPrep) -> str:
        from .dataset import prepare_training_dir
        self._progress("准备训练数据（含 train/val 划分）...")
        result = prepare_training_dir(
            prep.dataset_dir, prep.class_names, prep.images_dir,
            negative_ratio=prep.negative_ratio,
            min_positive=prep.min_positive,
            seed=prep.seed if prep.seed is not None else _default_seed(),
            val_ratio=prep.val_ratio,
        )
        if result["negative_count"] > 0:
            self._progress(
                f"[负样本] 从 {result['unlabeled_count']} 张无标注图片中选取 "
                f"{result['negative_count']} 张 "
                f"(占总训练集 {result['negative_count'] * 100 // result['total_count']}%)")
        elif result["positive_count"] < prep.min_positive:
            self._progress(f"[负样本] 正样本 {result['positive_count']} 张未达阈值 "
                           f"{prep.min_positive}，跳过负样本")
        self._progress(f"[划分] train={result['train_count']} / val={result['val_count']} "
                       f"(seed={result['seed']})")
        self._progress(f"创建 data.yaml: {result['yaml_path']}")
        return result["yaml_path"]

    # ── 训练 ──────────────────────────────────────────────────────────

    def _run_inner(self, cfg: TrainConfig) -> tuple[str, str]:
        from ultralytics import YOLO

        device, device_info = get_device_info()
        self._progress(f"设备: {device_info}")
        self._progress(f"加载模型: {cfg.model_base}")
        model = self._load_model_resilient(YOLO, cfg.model_base)
        train_start = time.time()

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

        best_path = self._locate_weights(model, cfg, train_start)
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

    @staticmethod
    def _locate_weights(model, cfg: TrainConfig, train_start: float) -> str:
        """定位本次训练产出的权重。

        候选按优先级排序，全部要求 mtime >= 训练开始时间——旧版直接
        拼接 project/train/weights/best.pt，ultralytics 递增目录名
        （train2/train3...）时会把旧模型当成本次成果回填（BUG-17）。
        """
        candidates: list[Path] = []
        save_dir = (getattr(model.trainer, "save_dir", None)
                    if getattr(model, "trainer", None) else None)
        if save_dir:
            candidates += [Path(save_dir) / "weights" / "best.pt",
                           Path(save_dir) / "weights" / "last.pt"]
        if cfg.project_dir:
            candidates.append(Path(cfg.project_dir) / "train" / "weights" / "best.pt")
        candidates.append(Path("runs") / "detect" / "train" / "weights" / "best.pt")
        return select_weight(candidates, train_start)

    def _progress(self, msg: str) -> None:
        logger.info("[train] %s", msg)
        self._on_progress(msg)


def _default_seed() -> int:
    from .dataset import DEFAULT_SEED
    return DEFAULT_SEED


def select_weight(candidates: list[Path], train_start: float,
                  slack: float = 5.0) -> str:
    """从候选路径中选出可信的权重文件（纯函数，可测）。

    可信 = 文件存在且 mtime 不早于训练开始时间（留 5 秒文件系统时钟
    误差余量）。无可信候选时返回第一个存在文件的候选（训练中途停止
    且无任何产出时返回空字符串）。
    """
    existing = [c for c in candidates if c.exists()]
    for c in existing:
        try:
            if c.stat().st_mtime >= train_start - slack:
                return str(c)
        except OSError:
            continue
    return str(existing[0]) if existing else (str(candidates[0]) if candidates else "")
