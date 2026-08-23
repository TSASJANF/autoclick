"""训练页：选择目标、准备数据集、启动/停止 YOLO 训练。

相对历史版本的改动：
- 停止训练真正生效：core.train_runner 通过 ultralytics 回调在 epoch
  边界协作中止（旧版的停止标志从未被检查，必然走 terminate 强杀）；
- worker 信号不再遮蔽 QThread.finished；
- 数据集准备与超参推荐使用 core.dataset 纯函数；
- 采样图片尺寸时限制数量，避免 UI 线程长时间读盘。
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QTextEdit, QVBoxLayout, QWidget,
)

from core import dataset as ds
from core.image_io import imread
from core.annotations import read_class_ids
from core.models import SampleSet
from core.target_manager import TargetManager
from core.train_runner import DatasetPrep, TrainConfig
from ui.app_context import AppContext
from ui.workers.training_worker import TrainingWorker

logger = logging.getLogger(__name__)

# 自动调参时最多采样读取的图片数（控制 UI 线程耗时）
_SIZE_SAMPLE_LIMIT = 8


class TrainTab(QWidget):
    training_complete = Signal(str, str)  # model_path, target_id

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.tm: TargetManager = ctx.tm
        self.train_worker: TrainingWorker | None = None
        self._init_ui()

    # ========== UI ==========

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        target_group = QGroupBox("选择目标")
        target_layout = QHBoxLayout(target_group)
        target_layout.setSpacing(12)
        target_label = QLabel("训练目标:")
        target_label.setStyleSheet("color: #8888a8;")
        target_layout.addWidget(target_label)

        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_combo)
        layout.addWidget(target_group)

        dataset_group = QGroupBox("数据集")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setSpacing(8)
        self.dataset_info = QLabel("请先选择目标")
        self.dataset_info.setStyleSheet("color: #e8e8f0;")
        dataset_layout.addWidget(self.dataset_info)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._update_dataset_info)
        dataset_layout.addWidget(refresh_btn)
        layout.addWidget(dataset_group)

        params_group = QGroupBox("训练参数")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(12)
        params_layout.setLabelAlignment(Qt.AlignRight)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.setMaximumWidth(70)
        browse_model_btn.clicked.connect(self._browse_base_model)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(browse_model_btn)
        params_layout.addRow("基础模型:", model_row)

        save_path_layout = QHBoxLayout()
        save_path_layout.setSpacing(8)
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setPlaceholderText("留空则使用默认路径 (runs/detect/train/weights/)")
        browse_save_btn = QPushButton("浏览...")
        browse_save_btn.setMaximumWidth(70)
        browse_save_btn.clicked.connect(self._browse_save_path)
        save_path_layout.addWidget(self.save_path_edit)
        save_path_layout.addWidget(browse_save_btn)
        params_layout.addRow("模型保存位置:", save_path_layout)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        params_layout.addRow("训练轮数:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(16)
        params_layout.addRow("批次大小:", self.batch_spin)

        self.imgsize_spin = QSpinBox()
        self.imgsize_spin.setRange(320, 1280)
        self.imgsize_spin.setValue(640)
        self.imgsize_spin.setSingleStep(32)
        params_layout.addRow("图片尺寸:", self.imgsize_spin)
        layout.addWidget(params_group)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.train_btn = QPushButton("开始训练")
        self.train_btn.setObjectName("success_btn")
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("danger_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setSpacing(8)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

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
        """按数据集规模/尺寸推荐超参（采样受限，不会长时间阻塞 UI）。"""
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

        # 数据集组装（含负样本与 train/val 划分）在训练 worker 的后台
        # 线程执行，不在 UI 线程逐张复制文件（BUG-14）
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
        """协作停止：epoch 边界生效，训练器正常收尾，无需 terminate。"""
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
