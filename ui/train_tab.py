"""训练页：选择目标、配置参数、启动/停止 YOLO 训练。

布局：ScrollArea 卡片式布局。训练逻辑（后台线程协作停止、数据集组装、
自动调参）与上一版一致，仅视图层换用 Fluent 组件。
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CardWidget, FluentIcon, LineEdit, PrimaryPushButton,
    ProgressBar, PushButton, ScrollArea, SpinBox, SubtitleLabel, TextEdit,
)

from core import dataset as ds
from core.annotations import read_class_ids
from core.image_io import imread
from core.models import SampleSet
from core.target_manager import TargetManager
from core.train_runner import DatasetPrep, TrainConfig
from ui.app_context import AppContext
from ui.workers.training_worker import TrainingWorker

logger = logging.getLogger(__name__)

_SIZE_SAMPLE_LIMIT = 8


class TrainTab(ScrollArea):
    training_complete = Signal(str, str)  # model_path, target_id

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.setObjectName("trainInterface")
        self.ctx = ctx
        self.tm: TargetManager = ctx.tm
        self.train_worker: TrainingWorker | None = None
        self._init_ui()

    # ========== UI ==========

    def _init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        self.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(8)

        title = QLabel("训练")
        title.setObjectName("heading")
        root.addWidget(title)
        root.addWidget(QLabel("为 AI 检测目标训练专属 YOLO 模型；停止为协作式（epoch 边界生效）"))

        root.addWidget(self._build_target_card())
        root.addWidget(self._build_params_card())
        root.addWidget(self._build_control_card())
        root.addWidget(self._build_log_card())
        root.addStretch()

    def _card(self, title: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)
        if title:
            lay.addWidget(SubtitleLabel(title))
        return card, lay

    def _build_target_card(self) -> CardWidget:
        card, lay = self._card("选择目标")
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        lay.addWidget(self.target_combo)

        self.dataset_info = BodyLabel("请先选择目标")
        lay.addWidget(self.dataset_info)

        refresh_btn = PushButton(FluentIcon.SYNC, "刷新数据集信息")
        refresh_btn.clicked.connect(self._update_dataset_info)
        lay.addWidget(refresh_btn)
        return card

    def _build_params_card(self) -> CardWidget:
        card, lay = self._card("训练参数")

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_row.addWidget(BodyLabel("基础模型"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        browse_model_btn = PushButton(FluentIcon.FOLDER, "浏览...")
        browse_model_btn.clicked.connect(self._browse_base_model)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(browse_model_btn)
        lay.addLayout(model_row)

        save_row = QHBoxLayout()
        save_row.setSpacing(8)
        save_row.addWidget(BodyLabel("保存位置"))
        self.save_path_edit = LineEdit()
        self.save_path_edit.setPlaceholderText("留空则使用默认路径 (runs/detect/train/weights/)")
        browse_save_btn = PushButton(FluentIcon.FOLDER, "浏览...")
        browse_save_btn.clicked.connect(self._browse_save_path)
        save_row.addWidget(self.save_path_edit, 1)
        save_row.addWidget(browse_save_btn)
        lay.addLayout(save_row)

        # 训练轮数 / 批次 / 图片尺寸：标签旁控件，两行三组紧凑排列
        self.epochs_spin = SpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.batch_spin = SpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        self.imgsize_spin = SpinBox()
        self.imgsize_spin.setRange(320, 1280)
        self.imgsize_spin.setValue(640)
        self.imgsize_spin.setSingleStep(32)

        params_row1 = QHBoxLayout()
        params_row1.setSpacing(8)
        params_row1.addWidget(BodyLabel("训练轮数"))
        self.epochs_spin.setMaximumWidth(100)
        params_row1.addWidget(self.epochs_spin)
        params_row1.addSpacing(16)
        params_row1.addWidget(BodyLabel("批次大小"))
        self.batch_spin.setMaximumWidth(100)
        params_row1.addWidget(self.batch_spin)
        params_row1.addStretch()
        lay.addLayout(params_row1)

        params_row2 = QHBoxLayout()
        params_row2.setSpacing(8)
        params_row2.addWidget(BodyLabel("图片尺寸"))
        self.imgsize_spin.setMaximumWidth(100)
        params_row2.addWidget(self.imgsize_spin)
        params_row2.addStretch()
        lay.addLayout(params_row2)
        return card

    def _build_control_card(self) -> CardWidget:
        card, lay = self._card("控制")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.train_btn = PrimaryPushButton(FluentIcon.PLAY, "开始训练")
        self.train_btn.clicked.connect(self._start_training)
        row.addWidget(self.train_btn)
        self.stop_btn = PushButton(FluentIcon.CLOSE, "停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        row.addWidget(self.stop_btn)
        row.addStretch()
        lay.addLayout(row)

        self.progress_bar = ProgressBar()
        lay.addWidget(self.progress_bar)
        return card

    def _build_log_card(self) -> CardWidget:
        card, lay = self._card("训练日志")
        self.log_text = TextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(220)
        lay.addWidget(self.log_text)
        return card

    # ========== 浏览与选择 ==========

    def _browse_base_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择基础模型", "", "模型文件 (*.pt *.onnx *.yaml)")
        if path:
            idx = self.model_combo.findText(path)
            if idx < 0:
                self.model_combo.addItem(path)
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
            else:
                self.model_combo.setCurrentIndex(idx)

    def _browse_save_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型保存位置")
        if path:
            self.save_path_edit.setText(path)

    def refresh_targets(self):
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        for ss in self.ctx.ssm.get_all():
            set_targets = [t for t in self.tm.get_by_set(ss.id)
                           if t.detection_mode in ("ai", "both")]
            if set_targets:
                self.target_combo.addItem(f"── {ss.name} ──", None)
                for t in set_targets:
                    self.target_combo.addItem(f"  {t.name}", t.id)
        self.target_combo.blockSignals(False)
        for i in range(self.target_combo.count()):
            if self.target_combo.itemData(i) is not None:
                self.target_combo.setCurrentIndex(i)
                self._on_target_changed(i)
                break

    def select_target_by_id(self, target_id: str):
        for i in range(self.target_combo.count()):
            if self.target_combo.itemData(i) == target_id:
                self.target_combo.setCurrentIndex(i)
                self._on_target_changed(i)
                return

    def _get_current_target(self):
        tid = self.target_combo.currentData()
        return self.tm.get(tid) if tid else None

    def _get_current_set(self) -> SampleSet | None:
        target = self._get_current_target()
        if target and target.set_id:
            return self.ctx.ssm.get(target.set_id)
        return None

    def _negative_settings(self) -> tuple[str, int, int]:
        ss = self._get_current_set()
        if ss:
            return (self.ctx.ssm.images_dir(ss.id),
                    ss.negative_sample_ratio, ss.negative_min_positive)
        return (os.path.join("data", "shared_images"), 20, 5)

    def _on_target_changed(self, index: int):
        if index < 0 or self.target_combo.currentData() is None:
            return
        self._update_dataset_info()
        self._auto_adjust_params()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_dataset_info()

    # ========== 数据集信息与自动调参 ==========

    def _update_dataset_info(self):
        target = self._get_current_target()
        if not target:
            self.dataset_info.setText("请先选择目标")
            return
        ds_dir = target.data_dir
        if not os.path.exists(ds_dir):
            self.dataset_info.setText(f"目标: {target.name}  |  数据目录不存在: {ds_dir}")
            return

        images_dir, neg_ratio, min_positive = self._negative_settings()
        stats = ds.get_dataset_stats(ds_dir, images_dir)
        positive_count = len(ds.find_labeled_images(ds_dir, images_dir))
        unlabeled_count = stats["image_count"] - positive_count

        negative_count = ds.count_negative_candidates(
            positive_count, unlabeled_count, neg_ratio, min_positive)
        if neg_ratio > 0 and negative_count == 0:
            if positive_count < min_positive:
                neg_status = f"  |  负样本: 正样本不足 (需 ≥{min_positive})"
            else:
                neg_status = "  |  负样本: 无可用图片"
        else:
            neg_status = f"  |  负样本: {negative_count} (占 {neg_ratio}%)"

        class_info = ", ".join(f"class_{k}:{v}"
                               for k, v in sorted(stats["class_counts"].items())) or "无"
        self.dataset_info.setText(
            f"目标: {target.name}  |  正样本: {positive_count}/{stats['image_count']}  |  "
            f"训练总数: {positive_count + negative_count}{neg_status}  |  "
            f"标注: {stats['label_count']} 条  |  类别: {class_info}")

    def _auto_adjust_params(self):
        target = self._get_current_target()
        if not target:
            return
        images_dir, neg_ratio, min_positive = self._negative_settings()
        if not os.path.exists(images_dir):
            return

        ds_dir = target.data_dir
        positive_count = (len(ds.find_labeled_images(ds_dir, images_dir))
                          if os.path.exists(ds_dir) else 0)
        all_images = ds.list_images(images_dir)
        negative_count = ds.count_negative_candidates(
            positive_count, max(0, len(all_images) - positive_count),
            neg_ratio, min_positive)

        widths, heights = [], []
        for fname in all_images[:_SIZE_SAMPLE_LIMIT]:
            img = imread(os.path.join(images_dir, fname))
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
        if not widths:
            return

        img_size, batch, epochs = ds.suggest_hyperparams(
            positive_count, negative_count,
            sum(widths) / len(widths), sum(heights) / len(heights),
            min(min(widths), min(heights)))

        self.epochs_spin.setValue(epochs)
        self.batch_spin.setValue(batch)
        self.imgsize_spin.setValue(img_size)

        self.log_text.append(
            f"[自动调整] 采样图片平均尺寸: {sum(widths)/len(widths):.0f}x"
            f"{sum(heights)/len(heights):.0f} → img_size={img_size}, "
            f"batch={batch}, epochs={epochs} "
            f"(正样本 {positive_count}, 负样本 {negative_count})")

    # ========== 训练启停 ==========

    def _start_training(self):
        if self.train_worker is not None and self.train_worker.isRunning():
            return
        target = self._get_current_target()
        if not target:
            self.log_text.append("错误: 请先选择目标")
            return
        ds_dir = target.data_dir
        if not os.path.exists(ds_dir):
            self.log_text.append(f"错误: 数据目录不存在: {ds_dir}")
            return

        if target.model_path and os.path.exists(target.model_path):
            reply = QMessageBox.question(
                self, "确认训练",
                f"目标 '{target.name}' 已有模型:\n{target.model_path}\n\n"
                f"重新训练将生成新模型，是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        class_ids = read_class_ids(os.path.join(ds_dir, "labels"))
        max_cls = max(class_ids) + 1 if class_ids else 1
        class_names = [f"class_{i}" for i in range(max_cls)]

        images_dir, neg_ratio, min_positive = self._negative_settings()
        save_path = self.save_path_edit.text().strip()
        project_dir = save_path if save_path else os.path.join(ds_dir, "runs")

        self.train_worker = TrainingWorker(TrainConfig(
            model_base=self.model_combo.currentText(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            img_size=self.imgsize_spin.value(),
            project_dir=project_dir,
            prep=DatasetPrep(
                dataset_dir=ds_dir,
                class_names=class_names,
                images_dir=images_dir,
                negative_ratio=neg_ratio,
                min_positive=min_positive,
            ),
        ))
        model_base_name = os.path.splitext(os.path.basename(self.model_combo.currentText()))[0]
        self.train_worker.progress.connect(self._on_progress)
        self.train_worker.succeeded.connect(
            lambda path: self._on_train_done(path, target.id, model_base_name))
        self.train_worker.stopped.connect(
            lambda path: self._on_train_done(path, target.id, model_base_name, stopped=True))
        self.train_worker.failed.connect(self._on_train_error)
        self.train_worker.start()

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, 0)

    def _stop_training(self):
        if self.train_worker is None:
            return
        self.log_text.append("正在停止训练（等待当前 epoch 结束）...")
        self.stop_btn.setEnabled(False)
        self.train_worker.request_stop()

    def _on_train_done(self, model_path: str, target_id: str,
                       model_base: str = "", stopped: bool = False):
        self._reset_controls()
        if not model_path or not os.path.exists(model_path):
            self.log_text.append(f"\n✗ 训练结束但未找到模型文件: {model_path or '(空)'}")
            return
        mark = "▪ 训练已停止，使用已保存的权重" if stopped else "✓ 训练完成!"
        self.log_text.append(f"\n{mark} 模型: {model_path}")
        target = self.tm.get(target_id)
        if target:
            target.model_path = model_path
            target.model_base = model_base
            self.tm.save()
        self.training_complete.emit(model_path, target_id)

    @Slot(str)
    def _on_train_error(self, err: str):
        self.log_text.append(f"\n✗ 错误: {err}")
        self._reset_controls()

    def _reset_controls(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

    @Slot(str)
    def _on_progress(self, msg: str):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
