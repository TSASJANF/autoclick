"""应用上下文（组合根）：所有共享服务在此组装，各页签统一持有。

取代历史模式：MainWindow 把 manager/config 以构造参数层层下传，页签
各自缓存 config 引用、SettingsTab 另行 load_config 产生第二份副本，
配置更新靠直接改别人的属性（``annotate_tab.config = cfg``）注入。

现在：配置只有 context.config 一份；更新走 update_config()，页签每
次使用时从 context 现取，不存在过期副本。
"""

from __future__ import annotations

import logging

from core.capture import ScreenCapture
from core.config import load_config, save_config
from core.detector import YoloDetector
from core.models import Target
from core.opencv_detector import TemplateDetector
from core.sample_set_manager import SampleSetManager
from core.state_manager import load_state, save_state
from core.target_manager import TargetManager

logger = logging.getLogger(__name__)


class AppContext:
    def __init__(self):
        self.config: dict = load_config()
        self.state: dict = load_state()
        self.tm = TargetManager()
        self.tm.load()
        self.ssm = SampleSetManager()
        self.ssm.load()
        self.ssm.ensure_default()
        self.opencv_detector = TemplateDetector()
        self.capture = ScreenCapture()
        # target_id -> 已加载的 YOLO 模型（由 ModelLoader 后台填充）
        self.ai_detectors: dict[str, YoloDetector] = {}
        # 加载失败记录: target_id -> 当时尝试的 model_path（路径变化才重试）
        self.failed_models: dict[str, str] = {}

    # ── 配置 ──────────────────────────────────────────────────────────

    def update_config(self, new_config: dict) -> None:
        """更新并持久化配置。页签持有 context，读到的永远是新值。"""
        self.config = new_config
        save_config(new_config)

    # ── 模型装载 ──────────────────────────────────────────────────────

    def pending_model_jobs(self) -> list[tuple[str, str]]:
        """待加载的 AI 模型任务：目标开启 AI 且路径有效、尚未加载/未失败。"""
        jobs = []
        for t in self.tm.targets:
            if t.detection_mode not in ("ai", "both") or not t.model_path:
                continue
            if t.id in self.ai_detectors:
                continue
            if self.failed_models.get(t.id) == t.model_path:
                continue  # 同一路径已失败，不重试（避免反复触发下载）
            jobs.append((t.id, t.model_path))
        return jobs

    def apply_loaded_models(self, loaded: dict[str, YoloDetector]) -> None:
        self.ai_detectors.update(loaded)
        for tid in loaded:
            self.failed_models.pop(tid, None)
        logger.info("AI 模型就绪: %d 个", len(loaded))

    def apply_failed_models(self, errors: list[tuple[str, str, str]]) -> None:
        for target_id, model_path, _err in errors:
            self.failed_models[target_id] = model_path

    def drop_model(self, target_id: str) -> None:
        self.ai_detectors.pop(target_id, None)
        self.failed_models.pop(target_id, None)

    # ── 状态 ──────────────────────────────────────────────────────────

    def save_app_state(self) -> None:
        save_state(self.state)

    # ── 便捷查询 ──────────────────────────────────────────────────────

    def set_display_name(self, set_id: str) -> str:
        ss = self.ssm.get(set_id)
        return ss.name if ss else ""

    def target_model_label(self, target: Target) -> str:
        """目标表格「模型」列的短标签。"""
        if target.detection_mode == "opencv":
            return "opencv"
        if target.model_base:
            return target.model_base
        if target.model_path:
            return self._model_label_from_args(target.model_path)
        return "ai"

    @staticmethod
    def _model_label_from_args(model_path: str) -> str:
        import os
        args_file = os.path.join(os.path.dirname(os.path.dirname(model_path)),
                                 "args.yaml")
        try:
            with open(args_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("model:"):
                        val = line.split(":", 1)[1].strip()
                        return os.path.splitext(os.path.basename(val))[0]
        except OSError:
            pass
        return os.path.splitext(os.path.basename(model_path))[0]
