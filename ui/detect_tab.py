"""检测页：屏幕预览、目标选择、检测启停与结果展示。

相对历史版本的改动：
- 检测循环从本文件的两个 QThread 子类移入 core.detection_engine +
  ui.workers.DetectionWorker（本文件只做视图与信号转发）；
- AI 模型后台加载（旧版在 UI 线程同步加载，多目标时卡界面数秒）；
- 模型加载失败有负缓存，不再每次 showEvent 重试（旧版会对不存在
  的路径反复触发 ultralytics 联网下载）；
- 截图逻辑统一走 capture_by_config。
"""

from __future__ import annotations

import logging
import os
import shutil

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QMessageBox, QPushButton, QMenu, QSlider,
    QSplitter, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.capture import ScreenCapture
from core.detection_engine import DetectionEngine, EngineConfig
from core.target_manager import TargetManager
from ui.app_context import AppContext
from ui.widgets.image_utils import bgr_color, bgr_to_qimage
from ui.workers.detection_worker import DetectionWorker
from ui.workers.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# 目标表格列索引
COL_CHECK, COL_NAME, COL_PRIORITY, COL_MODEL, COL_SET = 0, 1, 2, 3, 4


class DetectTab(QWidget):
    target_deleted = Signal()
    status_message = Signal(str)
    floating_window_requested = Signal(bool, bool)  # show, detecting

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.tm: TargetManager = ctx.tm
        self.detect_thread: DetectionWorker | None = None
        self._model_loader: ModelLoader | None = None
        self._init_ui()

    # ========== UI 构建 ==========

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 左：预览
        preview_group = QGroupBox("屏幕预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(8)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a16;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
            }
        """)
        preview_layout.addWidget(self.preview_label)

        fps_frame = QFrame()
        fps_layout = QHBoxLayout(fps_frame)
        fps_layout.setContentsMargins(8, 4, 8, 4)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        fps_layout.addWidget(self.fps_label)
        fps_layout.addStretch()
        preview_layout.addWidget(fps_frame)

        # 右：控制
        right_splitter = QSplitter(Qt.Vertical)

        target_group = QGroupBox("检测目标")
        target_layout = QVBoxLayout(target_group)
        target_layout.setSpacing(8)

        self.target_table = QTableWidget(0, 5)
        self.target_table.setHorizontalHeaderLabels(["", "名称", "优先级", "模型", "样本集"])
        self.target_table.setColumnWidth(COL_CHECK, 30)
        self.target_table.setColumnWidth(COL_NAME, 120)
        self.target_table.setColumnWidth(COL_PRIORITY, 60)
        self.target_table.setColumnWidth(COL_MODEL, 80)
        self.target_table.setColumnWidth(COL_SET, 100)
        self.target_table.horizontalHeader().setStretchLastSection(True)
        self.target_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.target_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.target_table.cellDoubleClicked.connect(self._on_target_cell_double_clicked)
        self.target_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.target_table.customContextMenuRequested.connect(self._on_target_context_menu)
        target_layout.addWidget(self.target_table)

        target_btn_layout = QHBoxLayout()
        target_btn_layout.setSpacing(8)

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        target_btn_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("全不选")
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_checked(False))
        target_btn_layout.addWidget(self.deselect_all_btn)

        target_layout.addLayout(target_btn_layout)
        right_splitter.addWidget(target_group)

        # 检测设置
        settings_group = QGroupBox("检测设置")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)

        conf_frame = QFrame()
        conf_layout = QHBoxLayout(conf_frame)
        conf_layout.setContentsMargins(0, 0, 0, 0)

        conf_label = QLabel("置信度:")
        conf_label.setStyleSheet("color: #8888a8;")
        conf_layout.addWidget(conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 99)
        self.conf_slider.setValue(50)
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setStyleSheet("color: #a5b4fc; font-weight: 600;")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f"{v / 100:.2f}"))
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        settings_layout.addWidget(conf_frame)

        fps_frame2 = QFrame()
        fps_layout2 = QHBoxLayout(fps_frame2)
        fps_layout2.setContentsMargins(0, 0, 0, 0)

        fps_setting_label = QLabel("目标帧率:")
        fps_setting_label.setStyleSheet("color: #8888a8;")
        fps_layout2.addWidget(fps_setting_label)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        fps_layout2.addWidget(self.fps_spin)
        settings_layout.addWidget(fps_frame2)

        self.priority_cb = QCheckBox("启用优先级 (数字越小优先级越高)")
        settings_layout.addWidget(self.priority_cb)

        self.auto_click_cb = QCheckBox("自动点击")
        settings_layout.addWidget(self.auto_click_cb)

        multi_click_frame = QFrame()
        multi_click_layout = QHBoxLayout(multi_click_frame)
        multi_click_layout.setContentsMargins(0, 0, 0, 0)

        multi_click_label = QLabel("多目标同优先级点击:")
        multi_click_label.setStyleSheet("color: #8888a8;")
        multi_click_layout.addWidget(multi_click_label)

        self.multi_click_combo = QComboBox()
        self.multi_click_combo.addItems(["只点击一个", "点击所有实例"])
        self.multi_click_combo.setToolTip("当多个相同优先级的目标被检测到时的点击策略")
        multi_click_layout.addWidget(self.multi_click_combo)
        settings_layout.addWidget(multi_click_frame)

        self.floating_auto_switch_cb = QCheckBox("开始检测时自动切换到悬浮窗")
        self.floating_auto_switch_cb.setToolTip("勾选后点击开始检测会自动进入悬浮窗模式")
        settings_layout.addWidget(self.floating_auto_switch_cb)

        right_splitter.addWidget(settings_group)

        # 控制按钮 + 结果表
        btn_group = QWidget()
        btn_layout = QVBoxLayout(btn_group)
        btn_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("success_btn")
        self.start_btn.clicked.connect(self._toggle_detection)
        btn_row.addWidget(self.start_btn)

        self.screenshot_btn = QPushButton("截图")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        btn_row.addWidget(self.screenshot_btn)

        self.minimize_btn = QPushButton("缩小到悬浮窗")
        self.minimize_btn.clicked.connect(self._minimize_to_floating)
        btn_row.addWidget(self.minimize_btn)

        btn_layout.addLayout(btn_row)

        results_label = QLabel("检测结果:")
        results_label.setStyleSheet("color: #8888a8; font-size: 12px; margin-top: 8px;")
        btn_layout.addWidget(results_label)

        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["目标", "来源", "置信度", "位置", "大小"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        btn_layout.addWidget(self.result_table)

        right_splitter.addWidget(btn_group)
        right_splitter.setSizes([250, 180, 200])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(preview_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([700, 300])
        layout.addWidget(main_splitter)

        self.screenshot_shortcut = QShortcut(QKeySequence("F12"), self)
        self.screenshot_shortcut.activated.connect(self._take_screenshot)

    # ========== 目标表格 ==========

    def refresh_targets(self):
        self.target_table.setRowCount(0)
        for t in self.tm.targets:
            row = self.target_table.rowCount()
            self.target_table.insertRow(row)

            cb = QCheckBox()
            cb.setChecked(t.enabled)
            cb.stateChanged.connect(lambda state, tid=t.id: self._on_check_changed(tid, state))
            self.target_table.setCellWidget(row, COL_CHECK, cb)

            name_item = QTableWidgetItem(t.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, t.id)
            self.target_table.setItem(row, COL_NAME, name_item)

            pri_item = QTableWidgetItem(str(t.priority))
            pri_item.setTextAlignment(Qt.AlignCenter)
            self.target_table.setItem(row, COL_PRIORITY, pri_item)

            model_item = QTableWidgetItem(self.ctx.target_model_label(t))
            model_item.setFlags(model_item.flags() & ~Qt.ItemIsEditable)
            self.target_table.setItem(row, COL_MODEL, model_item)

            set_item = QTableWidgetItem(self.ctx.set_display_name(t.set_id))
            set_item.setFlags(set_item.flags() & ~Qt.ItemIsEditable)
            self.target_table.setItem(row, COL_SET, set_item)

        self._load_opencv_templates()
        self._start_model_loader()

    def _on_check_changed(self, target_id: str, state: int):
        target = self.tm.get(target_id)
        if target:
            target.enabled = (state == Qt.Checked)

    def _set_all_checked(self, checked: bool):
        for row in range(self.target_table.rowCount()):
            cb = self.target_table.cellWidget(row, COL_CHECK)
            if cb:
                cb.setChecked(checked)

    def _selected_target_ids(self) -> list[str]:
        ids = []
        for row in range(self.target_table.rowCount()):
            item = self.target_table.item(row, COL_NAME)
            cb = self.target_table.cellWidget(row, COL_CHECK)
            if item and cb and cb.isChecked():
                ids.append(item.data(Qt.UserRole))
        return ids

    def _row_target(self, row: int):
        if 0 <= row < len(self.tm.targets):
            return self.tm.targets[row]
        return None

    def _on_target_cell_double_clicked(self, row: int, col: int):
        target = self._row_target(row)
        if target is None:
            return
        if col == COL_NAME:
            new_name, ok = QInputDialog.getText(self, "重命名目标", "新名称:", text=target.name)
            if ok and new_name:
                target.name = new_name
                self.tm.save()
                self.refresh_targets()
        elif col == COL_PRIORITY:
            self.target_table.itemChanged.connect(self._on_priority_changed)

    def _on_priority_changed(self, item: QTableWidgetItem):
        try:
            self.target_table.itemChanged.disconnect(self._on_priority_changed)
        except (RuntimeError, TypeError):
            pass
        if item.column() != COL_PRIORITY:
            return
        target = self._row_target(item.row())
        if target is None:
            return
        try:
            target.priority = int(item.text())
            self.tm.save()
        except ValueError:
            pass

    def _on_target_context_menu(self, pos):
        row = self.target_table.rowAt(pos.y())
        if self._row_target(row) is None:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("删除目标（彻底删除文件）")
        action = menu.exec(self.target_table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self._delete_target_at(row)

    def _delete_target_at(self, row: int):
        target = self._row_target(row)
        if target is None:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除目标 '{target.name}' 吗？\n\n"
            f"这将删除以下所有内容：\n"
            f"  • 训练数据: {target.data_dir}\n"
            f"  • 模型文件: {target.model_path or '无'}\n"
            f"  • 模板文件: {target.template_path or '无'}\n\n"
            f"此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        text, ok = QInputDialog.getText(
            self, "最终确认",
            f"请输入目标名称 '{target.name}' 以确认删除：")
        if not ok or text != target.name:
            QMessageBox.information(self, "取消", "名称不匹配，删除已取消")
            return
        if target.data_dir and os.path.isdir(target.data_dir):
            shutil.rmtree(target.data_dir, ignore_errors=True)
        if target.model_path and os.path.isfile(target.model_path):
            try:
                os.remove(target.model_path)
            except OSError:
                logger.warning("模型文件删除失败: %s", target.model_path)
        self.tm.remove(target.id)
        self.tm.save()
        self.ctx.drop_model(target.id)
        self.ctx.opencv_detector.remove_template(target.id)
        self.refresh_targets()
        self.target_deleted.emit()

    # ========== 模型装载 ==========

    def _start_model_loader(self):
        """后台加载待装载的 AI 模型（无任务时为空操作）。"""
        jobs = self.ctx.pending_model_jobs()
        if not jobs or (self._model_loader is not None and self._model_loader.isRunning()):
            return
        self._model_loader = ModelLoader(jobs)
        self._model_loader.loaded.connect(self._on_models_loaded)
        self._model_loader.failed.connect(self._on_models_failed)
        self._model_loader.start()

    def _on_models_loaded(self, loaded: dict):
        self.ctx.apply_loaded_models(loaded)
        for tid in loaded:
            t = self.tm.get(tid)
            if t:
                logger.info("AI 模型已加载: %s", t.name)

    def _on_models_failed(self, errors: list):
        self.ctx.apply_failed_models(errors)
        for target_id, model_path, err in errors:
            t = self.tm.get(target_id)
            name = t.name if t else target_id
            self.status_message.emit(f"模型加载失败 [{name}]: {err}")

    def _load_opencv_templates(self):
        detector = self.ctx.opencv_detector
        for t in self.tm.targets:
            if t.detection_mode in ("opencv", "both") and t.template_path:
                path = os.path.abspath(t.template_path)
                if not os.path.exists(path):
                    logger.warning("模板文件不存在: %s", path)
                    continue
                if detector.has_template(t.id):
                    detector.thresholds[t.id] = t.match_threshold
                else:
                    detector.load_template(t.id, path, t.match_threshold)

    # ========== 检测启停 ==========

    def _toggle_detection(self):
        if self.detect_thread and self.detect_thread.running:
            self._stop_detection()
            return
        self._start_detection()

    def _start_detection(self):
        selected = set(self._selected_target_ids())
        if not selected:
            QMessageBox.warning(self, "提示", "请先选择至少一个检测目标")
            return
        if len(selected) > 4:
            reply = QMessageBox.question(
                self, "确认",
                f"已选择 {len(selected)} 个目标，同时运行可能导致性能问题。\n是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        for t in self.tm.targets:
            t.enabled = t.id in selected
        self.tm.save()

        # 模型没就绪的 AI 目标给出提示（后台加载未完成或失败）
        missing = [t.name for t in self.tm.get_enabled()
                   if t.detection_mode in ("ai", "both")
                   and t.id not in self.ctx.ai_detectors]
        if missing:
            logger.warning("以下目标的 AI 模型未就绪: %s", ", ".join(missing))

        engine = DetectionEngine(
            capture=self.ctx.capture,
            target_provider=self.tm.get_enabled,
            ai_detectors=self.ctx.ai_detectors,
            template_detector=self.ctx.opencv_detector,
            config=EngineConfig(
                confidence=self.conf_slider.value() / 100,
                auto_click=self.auto_click_cb.isChecked(),
                use_priority=self.priority_cb.isChecked(),
                click_all_instances=self.multi_click_combo.currentIndex() == 1,
                hide_window=self.ctx.config.get("screenshot_hide_window", True),
                target_fps=self.fps_spin.value(),
            ),
        )
        self.detect_thread = DetectionWorker(engine)
        self.detect_thread.frame_ready.connect(self._on_frame)
        self.detect_thread.fps_update.connect(self._on_fps)
        self.detect_thread.start()

        self.start_btn.setText("停止检测")
        self.start_btn.setObjectName("danger_btn")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        if self.floating_auto_switch_cb.isChecked():
            self.floating_window_requested.emit(True, True)

    def _stop_detection(self):
        if self.detect_thread:
            self.detect_thread.stop()
            self.detect_thread = None
        self.start_btn.setText("开始检测")
        self.start_btn.setObjectName("success_btn")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        if self.floating_auto_switch_cb.isChecked():
            self.floating_window_requested.emit(False, False)

    def stop_if_running(self) -> bool:
        """若检测正在运行则停止，返回之前是否在运行。"""
        was_running = bool(self.detect_thread and self.detect_thread.running)
        if was_running:
            self._stop_detection()
        return was_running

    @property
    def is_detecting(self) -> bool:
        return bool(self.detect_thread and self.detect_thread.running)

    # ========== 帧回调 ==========

    @Slot(np.ndarray, list)
    def _on_frame(self, img: np.ndarray, detections: list):
        for det in detections:
            b, g, r = bgr_color(det.color)
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (b, g, r), 2)
            label = f"{det.target_name}[{det.source}] {det.conf:.2f}"
            cv2.putText(img, label, (det.x1, det.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)

        qt_img = bgr_to_qimage(img)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

        self.result_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            self.result_table.setItem(i, 0, QTableWidgetItem(det.target_name))
            self.result_table.setItem(i, 1, QTableWidgetItem(det.source))
            self.result_table.setItem(i, 2, QTableWidgetItem(f"{det.conf:.2f}"))
            self.result_table.setItem(i, 3, QTableWidgetItem(f"({det.x1},{det.y1})"))
            self.result_table.setItem(i, 4, QTableWidgetItem(f"{det.width}x{det.height}"))

    @Slot(float)
    def _on_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    # ========== 其他动作 ==========

    def _take_screenshot(self):
        try:
            img = self.ctx.capture.capture_by_config(self.ctx.config)
            default_path = self.ctx.config.get("screenshot_default_path", "data/screenshots")
            saved_path = ScreenCapture.save_screenshot(img, default_path, "detect")
            self.status_message.emit(f"截图已保存: {saved_path}")
        except Exception as e:
            logger.exception("截图失败")
            self.status_message.emit(f"截图失败: {e}")

    def _minimize_to_floating(self):
        self.floating_window_requested.emit(True, self.is_detecting)

    def showEvent(self, event):
        """页签可见时仅刷新表格；模型加载由 refresh_targets 触发且有负缓存。"""
        super().showEvent(event)
        self.refresh_targets()
