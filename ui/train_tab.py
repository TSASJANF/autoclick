import os
import cv2

from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QTextEdit, QProgressBar, QFileDialog,
    QLineEdit, QFormLayout, QComboBox, QFrame, QMessageBox,
)

from core.trainer import Trainer, TrainWorker
from core.target_manager import TargetManager
from core.sample_set_manager import SampleSetManager


class TrainTab(QWidget):
    training_complete = Signal(str, str)  # model_path, target_id

    def __init__(self, target_manager: TargetManager, sample_set_manager: SampleSetManager | None = None):
        super().__init__()
        self.tm = target_manager
        self.ssm = sample_set_manager or SampleSetManager()
        self.train_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Target selector
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

        # Dataset info
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

        # Training parameters
        params_group = QGroupBox("训练参数")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(12)
        params_layout.setLabelAlignment(Qt.AlignRight)

        # Base model
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

        # Model save path
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

        # Training parameters
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

        # Controls
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

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Log
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setSpacing(8)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

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
        for ss in self.ssm.get_all():
            set_targets = [t for t in self.tm.get_by_set(ss.id) if t.detection_mode in ("ai", "both")]
            if set_targets:
                # Add a separator-style header (disabled, non-selectable)
                self.target_combo.addItem(f"── {ss.name} ──", None)
                for t in set_targets:
                    self.target_combo.addItem(f"  {t.name}", t.id)
        self.target_combo.blockSignals(False)
        # Select first real target (skip header items with data=None)
        for i in range(self.target_combo.count()):
            if self.target_combo.itemData(i) is not None:
                self.target_combo.setCurrentIndex(i)
                self._on_target_changed(i)
                break

    def select_target_by_id(self, target_id: str):
        """Select a target in the combo by its ID."""
        for i in range(self.target_combo.count()):
            if self.target_combo.itemData(i) == target_id:
                self.target_combo.setCurrentIndex(i)
                self._on_target_changed(i)
                return

    def _on_target_changed(self, index: int):
        if index < 0:
            return
        # Skip separator items (data is None)
        if self.target_combo.currentData() is None:
            return
        self._update_dataset_info()
        self._auto_adjust_params()

    def showEvent(self, event):
        """Refresh dataset info when tab becomes visible (e.g. after settings change)."""
        super().showEvent(event)
        self._update_dataset_info()

    def _get_current_target(self):
        tid = self.target_combo.currentData()
        return self.tm.get(tid) if tid else None

    def _get_current_set(self):
        """Get the sample set that the current target belongs to."""
        target = self._get_current_target()
        if target and target.set_id:
            return self.ssm.get(target.set_id)
        return None

    def _update_dataset_info(self):
        target = self._get_current_target()
        if not target:
            self.dataset_info.setText("请先选择目标")
            return
        ds_dir = target.data_dir
        if not os.path.exists(ds_dir):
            self.dataset_info.setText(f"目标: {target.name}  |  数据目录不存在: {ds_dir}")
            return

        sample_set = self._get_current_set()
        if sample_set:
            images_dir = sample_set.images_dir
            neg_ratio = sample_set.negative_sample_ratio
            min_positive = sample_set.negative_min_positive
        else:
            images_dir = os.path.join("data", "shared_images")
            neg_ratio = 20
            min_positive = 5

        stats = Trainer.get_dataset_stats(ds_dir, images_dir)
        positive_count = len(Trainer.find_labeled_images(ds_dir, images_dir))
        total_images = stats['image_count']
        unlabeled_count = total_images - positive_count

        negative_count = 0
        neg_status = ""
        if neg_ratio > 0 and positive_count >= min_positive and unlabeled_count > 0:
            negative_count = min(
                round(positive_count * neg_ratio / (100 - neg_ratio)),
                unlabeled_count,
            )
            neg_status = f"  |  负样本: {negative_count} (占 {neg_ratio}%)"
        elif neg_ratio > 0 and positive_count < min_positive:
            neg_status = f"  |  负样本: 正样本不足 (需 ≥{min_positive})"
        elif neg_ratio > 0 and unlabeled_count == 0:
            neg_status = "  |  负样本: 无可用图片"

        effective_count = positive_count + negative_count
        class_info = ", ".join(f"class_{k}:{v}" for k, v in sorted(stats['class_counts'].items())) or "无"

        self.dataset_info.setText(
            f"目标: {target.name}  |  正样本: {positive_count}/{total_images}  |  "
            f"训练总数: {effective_count}{neg_status}  |  "
            f"标注: {stats['label_count']} 条  |  类别: {class_info}"
        )

    def _auto_adjust_params(self):
        """Analyze dataset and auto-adjust training parameters for best quality."""
        target = self._get_current_target()
        if not target:
            return

        sample_set = self._get_current_set()
        if sample_set:
            images_dir = sample_set.images_dir
            neg_ratio = sample_set.negative_sample_ratio
            min_positive = sample_set.negative_min_positive
        else:
            images_dir = os.path.join("data", "shared_images")
            neg_ratio = 20
            min_positive = 5

        if not os.path.exists(images_dir):
            return

        # Count positive and estimate negative samples
        ds_dir = target.data_dir
        positive_count = len(Trainer.find_labeled_images(ds_dir, images_dir)) if os.path.exists(ds_dir) else 0
        IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        all_images = [f for f in os.listdir(images_dir)
                      if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        total_images = len(all_images)
        negative_count = 0
        if neg_ratio > 0 and positive_count >= min_positive:
            unlabeled_count = total_images - positive_count
            negative_count = round(positive_count * neg_ratio / (100 - neg_ratio))
            negative_count = min(negative_count, unlabeled_count)

        effective_count = positive_count + negative_count
        if effective_count == 0:
            return

        # Sample images to determine typical dimensions
        widths, heights = [], []
        sample_count = min(20, total_images)
        for fname in all_images[:sample_count]:
            img_path = os.path.join(images_dir, fname)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)

        if not widths:
            return

        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        avg_side = (avg_w + avg_h) / 2
        min_side = min(min(widths), min(heights))

        # Determine img_size: prioritize detecting small objects
        # Match or exceed the smallest image dimension for better detail
        if min_side <= 320:
            img_size = 320
        elif min_side <= 416:
            img_size = 416
        elif avg_side <= 600:
            img_size = 640
        elif avg_side <= 900:
            img_size = 800
        else:
            img_size = 1280

        # Adjust batch and epochs based on effective training count
        # Small dataset: more epochs, smaller batch (more gradient updates per epoch)
        # Large dataset: fewer epochs, larger batch (stable convergence)
        if effective_count < 30:
            batch, epochs = 4, 150
        elif effective_count < 80:
            batch, epochs = 4, 120
        elif effective_count < 200:
            batch, epochs = 8, 80
        elif effective_count < 500:
            batch, epochs = 16, 60
        else:
            batch, epochs = 16, 40

        # Large img_size consumes more VRAM — reduce batch to avoid OOM
        if img_size >= 1280 and batch > 8:
            batch = 8
        elif img_size >= 800 and batch > 16:
            batch = 16

        # Apply to UI
        self.epochs_spin.setValue(epochs)
        self.batch_spin.setValue(batch)
        self.imgsize_spin.setValue(img_size)

        neg_info = f", 负样本 {negative_count}" if negative_count > 0 else ""
        self.log_text.append(
            f"[自动调整] 图片平均尺寸: {avg_w:.0f}x{avg_h:.0f} → img_size={img_size}, "
            f"batch={batch}, epochs={epochs} "
            f"(正样本 {positive_count}{neg_info}, 训练总数 {effective_count})"
        )

    def _start_training(self):
        target = self._get_current_target()
        if not target:
            self.log_text.append("错误: 请先选择目标")
            return
        ds_dir = target.data_dir
        if not os.path.exists(ds_dir):
            self.log_text.append(f"错误: 数据目录不存在: {ds_dir}")
            return

        # Confirm if target already has a model
        if target.model_path and os.path.exists(target.model_path):
            reply = QMessageBox.question(
                self, "确认训练",
                f"目标 '{target.name}' 已有模型:\n{target.model_path}\n\n"
                f"重新训练将生成新模型，是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        labels_dir = os.path.join(ds_dir, "labels")
        class_ids = set()
        if os.path.exists(labels_dir):
            for f in os.listdir(labels_dir):
                if f.endswith('.txt'):
                    with open(os.path.join(labels_dir, f)) as fp:
                        for line in fp:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
        max_cls = max(class_ids) + 1 if class_ids else 1
        class_names = [f"class_{i}" for i in range(max_cls)]

        sample_set = self._get_current_set()
        if sample_set:
            images_dir = sample_set.images_dir
            neg_ratio = sample_set.negative_sample_ratio
            min_positive = sample_set.negative_min_positive
        else:
            images_dir = os.path.join("data", "shared_images")
            neg_ratio = 20
            min_positive = 5

        try:
            result = Trainer.create_data_yaml(
                ds_dir, class_names, images_dir=images_dir,
                negative_ratio=neg_ratio, min_positive=min_positive,
            )
        except ValueError as e:
            self.log_text.append(f"错误: {e}")
            return
        yaml_path = result["yaml_path"]
        if result["negative_count"] > 0:
            self.log_text.append(
                f"[负样本] 从 {result['unlabeled_count']}"
                f" 张无标注图片中随机选取 {result['negative_count']} 张"
                f" (占总训练集 {result['negative_count'] * 100 // result['total_count']}%, "
                f"正样本 {result['positive_count']} 张)"
            )
        elif result["positive_count"] < min_positive:
            self.log_text.append(
                f"[负样本] 正样本仅 {result['positive_count']} 张，未达阈值"
                f" {min_positive}，跳过负样本"
            )
        self.log_text.append(f"创建 data.yaml: {yaml_path}")

        # Determine project save dir
        save_path = self.save_path_edit.text().strip()
        project_dir = save_path if save_path else os.path.join(ds_dir, "runs")

        self.train_worker = TrainWorker(
            data_yaml=yaml_path,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            img_size=self.imgsize_spin.value(),
            model_base=self.model_combo.currentText(),
            project_dir=project_dir,
        )
        self.train_worker.progress.connect(self._on_progress)
        self.train_worker.finished.connect(lambda path: self._on_finished(path, target.id, os.path.splitext(os.path.basename(self.model_combo.currentText()))[0]))
        self.train_worker.error.connect(self._on_error)
        self.train_worker.start()

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, 0)

    def _stop_training(self):
        if self.train_worker:
            self.train_worker.request_stop()
            self.train_worker.wait(10000)
            if self.train_worker.isRunning():
                self.train_worker.terminate()
                self.train_worker.wait()
            # Clean up temp files
            self.train_worker._cleanup_temp_files()
            self.log_text.append("训练已被用户停止")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

    @Slot(str)
    def _on_progress(self, msg: str):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, model_path: str, target_id: str, model_base: str = ""):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)

        if not os.path.exists(model_path):
            self.log_text.append(f"\n✗ 训练完成但模型文件未找到: {model_path}")
            self.progress_bar.setValue(0)
            return

        self.log_text.append(f"\n✓ 训练完成! 模型: {model_path}")
        self.progress_bar.setValue(100)
        target = self.tm.get(target_id)
        if target:
            target.model_path = model_path
            target.model_base = model_base
            self.tm.save()
        self.training_complete.emit(model_path, target_id)

    @Slot(str)
    def _on_error(self, err: str):
        self.log_text.append(f"\n✗ 错误: {err}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
