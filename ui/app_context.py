"""应用上下文（组合根）：所有共享服务在此组装，各页签统一持有。

取代历史模式：MainWindow 把 manager/config 以构造参数层层下传，页签
各自缓存 config 引用、SettingsTab 另行 load_config 产生第二份副本，
配置更新靠直接改别人的属性注入。

现在：配置只有 context.config 一份；更新走 update_config()，页签每次
使用时从 context 现取，不存在过期副本。

状态同步（ARCH-02）：变更通知统一走 hub（ChangeHub 信号），MainWindow
订阅一次 hub.targets_changed 即可覆盖所有目标增删改；新增页签不再需
要在 MainWindow 里逐个补连散装信号。
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QObject, Signal

from core.capture import ScreenCapture
from core.config import load_config, save_config
from core.detector import YoloDetector
from core.models import Target
from core.opencv_detector import TemplateDetector
from core.sample_set_manager import SampleSetManager
from core.state_manager import load_state, save_state
from core.target_manager import TargetManager

logger = logging.getLogger(__name__)

# 常驻 AI 模型数量上限（PERF-06：每模型显存/内存占用可观，N 目标 =
# N 模型线性增长；超出时不再加载并提示）
MAX_LOADED_MODELS = 8


class ChangeHub(QObject):
    """领域变更信号中心：任何页签修改目标/样本集后都从这里广播。"""

    targets_changed = Signal()        # 目标增删改（各页签刷新目标视图）
    sets_changed = Signal()           # 样本集增删改


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
        # 变更信号中心（ARCH-02）
        self.hub = ChangeHub()

    # ── 配置 ──────────────────────────────────────────────────────────

    def update_config(self, new_config: dict) -> None:
        """更新并持久化配置。页签持有 context，读到的永远是新值。"""
        self.config = new_config
        save_config(new_config)

    # ── 模型装载 ──────────────────────────────────────────────────────

    def pending_model_jobs(self) -> list[tuple[str, str]]:
        """待加载的 AI 模型任务。

        - 已加载且路径未变的目标跳过；路径变化（如重新训练后回填了新
          best.pt）会重新装载 —— 修复「训练完成但检测仍用旧模型」
          （BUG-05）；
        - 同一路径加载失败过的目标跳过（路径变化才重试），避免对失效
          路径反复触发 ultralytics 联网下载；
        - 超过 MAX_LOADED_MODELS 后不再装载（PERF-06）。
        """
        jobs = []
        for t in self.tm.targets:
            if t.detection_mode not in ("ai", "both") or not t.model_path:
                continue
            current_path = os.path.abspath(t.model_path)
            loaded = self.ai_detectors.get(t.id)
            if loaded is not None and loaded.model_path == current_path:
                continue  # 已加载且路径一致
            if loaded is None and self.failed_models.get(t.id) == current_path:
                continue  # 同一路径已失败，不重试
            if len(self.ai_detectors) >= MAX_LOADED_MODELS:
                logger.warning(
                    "已加载模型数达到上限 %d，跳过 %s 的模型装载",
                    MAX_LOADED_MODELS, t.name)
                continue
            jobs.append((t.id, current_path))
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

    # ── 启动自检（BIZ-05）────────────────────────────────────────────

    def startup_health_check(self) -> list[str]:
        """启动完整性扫描：失效的模型/模板路径、损坏备份残留。"""
        issues: list[str] = []
        for t in self.tm.targets:
            if t.detection_mode in ("ai", "both") and t.model_path \
                    and not os.path.isfile(t.model_path):
                issues.append(f"目标 '{t.name}' 的模型文件不存在: {t.model_path}")
            if t.detection_mode in ("opencv", "both") and t.template_path \
                    and not os.path.isfile(t.template_path):
                issues.append(f"目标 '{t.name}' 的模板文件不存在: {t.template_path}")
        for corrupt in self._find_corrupt_backups():
            issues.append(f"发现损坏的配置备份: {corrupt}")
        for issue in issues:
            logger.warning("[启动自检] %s", issue)
        return issues

    @staticmethod
    def _find_corrupt_backups() -> list[str]:
        from core import paths
        return [str(p) for p in paths.APP_ROOT.glob("*.corrupt-*")]

    # ── 状态 ──────────────────────────────────────────────────────────

    def save_app_state(self) -> None:
        save_state(self.state)

    # ── 便捷查询 ──────────────────────────────────────────────────────

    def set_display_name(self, set_id: str) -> str:
        ss = self.ssm.get(set_id)
        return ss.name if ss else ""

    def target_model_label(self, target: Target) -> str:
        """目标表格「模型」列的短标签（含加载失败标注，BUG-09）。"""
        if target.detection_mode == "opencv":
            return "opencv"
        if target.id in self.ai_detectors:
            label = target.model_base or self._model_label_from_args(target.model_path)
            return label if target.model_path else "ai"
        if target.id in self.failed_models:
            return "加载失败"
        if target.model_base:
            return target.model_base
        if target.model_path:
            return "未加载"
        return "ai"

    @staticmethod
    def _model_label_from_args(model_path: str) -> str:
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
        return os.path.splitext(os.path.basename(model_path))[0] if model_path else "ai"
