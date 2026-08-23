"""检测页：屏幕预览、目标选择、检测启停与结果展示。

布局：ScrollArea 内的卡片式布局（预览/目标/设置/控制/结果五张卡片），
窗口过小时纵向滚动，不再压缩空间。检测引擎、模型后台加载等逻辑与
上一版完全一致，仅视图层换用 Fluent 组件。
"""

from __future__ import annotations

import logging
import os
import shutil

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHeaderView, QHBoxLayout, QInputDialog,
    QLabel, QMessageBox, QPushButton, QMenu, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, CheckBox, FluentIcon,
    PrimaryPushButton, PushButton, ScrollArea, Slider, SpinBox, SubtitleLabel,
    TableWidget,
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


class DetectTab(ScrollArea):
    status_message = Signal(str)
    floating_window_requested = Signal(bool, bool)  # show, detecting
    add_target_requested = Signal()
    import_model_requested = Signal()

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.setObjectName("detectInterface")
        self.ctx = ctx
        self.tm: TargetManager = ctx.tm
        self.detect_thread: DetectionWorker | None = None
        self._model_loader: ModelLoader | None = None
        self._preview_enabled = True
        self._fps_warned = False
        self._init_ui()

    # ========== UI 构建 ==========

    def _init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        self.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(20, 16, 20, 24)
        root.setSpacing(12)

        # 标题
        title = QLabel("检测")
        title.setObjectName("heading")
        root.addWidget(title)
        sub = CaptionLabel("勾选目标 → 设置参数 → 开始检测；全局热键 F9 快速切换")
        root.addWidget(sub)

        root.addWidget(self._build_preview_card())
        root.addWidget(self._build_target_card())
        root.addWidget(self._build_settings_card())
        root.addWidget(self._build_control_card())
        root.addWidget(self._build_result_card())
        root.addStretch()

        self.screenshot_shortcut = QShortcut(QKeySequence("F12"), self)
        self.screenshot_shortcut.activated.connect(self._take_screenshot)

    def _card(self, title: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)
        if title:
            t = SubtitleLabel(title)
            lay.addWidget(t)
        return card, lay

    def _build_preview_card(self) -> CardWidget:
        card, lay = self._card("屏幕预览")
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(400)
        self.preview_label.setStyleSheet(
            "background-color: #0a0a16; border: 1px solid #2a2a4a; border-radius: 8px;")
        lay.addWidget(self.preview_label)

        self.fps_label = BodyLabel("FPS: --")
        lay.addWidget(self.fps_label)
        return card

    def _build_target_card(self) -> CardWidget:
        card, lay = self._card("检测目标")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        add_btn = PrimaryPushButton(FluentIcon.ADD, "添加目标")
        add_btn.clicked.connect(self.add_target_requested.emit)
        btn_row.addWidget(add_btn)
        import_btn = PushButton(FluentIcon.FOLDER, "导入模型")
        import_btn.clicked.connect(self.import_model_requested.emit)
        btn_row.addWidget(import_btn)
        self.select_all_btn = PushButton("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        btn_row.addWidget(self.select_all_btn)
        self.deselect_all_btn = PushButton("全不选")
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_checked(False))
        btn_row.addWidget(self.deselect_all_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.target_table = TableWidget()
        self.target_table.setHorizontalHeaderLabels(["", "名称", "优先级", "模型", "样本集"])
        header = self.target_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.target_table.setMinimumHeight(200)
        self.target_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.target_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.target_table.cellDoubleClicked.connect(self._on_target_cell_double_clicked)
        self.target_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.target_table.customContextMenuRequested.connect(self._on_target_context_menu)
        lay.addWidget(self.target_table)
        return card

    def _build_settings_card(self) -> CardWidget:
        card, lay = self._card("检测设置")

        conf_label = BodyLabel("置信度")
        lay.addWidget(conf_label)
        conf_row = QHBoxLayout()
        conf_row.setSpacing(12)
        self.conf_slider = Slider(Qt.Horizontal)
        self.conf_slider.setRange(10, 99)
        self.conf_slider.setValue(50)
        self.conf_value_label = BodyLabel("0.50")
        self.conf_value_label.setMinimumWidth(48)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f"{v / 100:.2f}"))
        conf_row.addWidget(self.conf_slider, 1)
        conf_row.addWidget(self.conf_value_label)
        lay.addLayout(conf_row)

        fps_label = BodyLabel("目标帧率")
        lay.addWidget(fps_label)
        self.fps_spin = SpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        lay.addWidget(self.fps_spin)

        self.priority_cb = CheckBox("启用优先级（数字越小优先级越高）")
        lay.addWidget(self.priority_cb)

        self.auto_click_cb = CheckBox("自动点击")
        lay.addWidget(self.auto_click_cb)

        self.floating_auto_switch_cb = CheckBox("开始检测时自动切换到悬浮窗")
        lay.addWidget(self.floating_auto_switch_cb)

        lay.addWidget(BodyLabel("多目标同优先级点击策略"))
        self.multi_click_combo = QComboBox()
        self.multi_click_combo.addItems(["只点击一个", "点击所有实例"])
        self.multi_click_combo.setMinimumWidth(200)
        lay.addWidget(self.multi_click_combo)
        return card

    def _build_control_card(self) -> CardWidget:
        card, lay = self._card("控制")
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("success_btn")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self._toggle_detection)
        btn_row.addWidget(self.start_btn, 1)

        self.screenshot_btn = PushButton(FluentIcon.CAMERA, "截图")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        btn_row.addWidget(self.screenshot_btn)

        self.minimize_btn = PushButton(FluentIcon.MINIMIZE, "缩小到悬浮窗")
        self.minimize_btn.clicked.connect(self._minimize_to_floating)
        btn_row.addWidget(self.minimize_btn)
        lay.addLayout(btn_row)
        return card

    def _build_result_card(self) -> CardWidget:
        card, lay = self._card("检测结果")
        self.result_table = TableWidget()
        self.result_table.setHorizontalHeaderLabels(["目标", "来源", "置信度", "位置", "大小"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setMinimumHeight(160)
        lay.addWidget(self.result_table)
        return card

    # ========== 目标表格 ==========

    def refresh_targets(self):
        self._refresh_table_only()
        self._load_opencv_templates()
        self._start_model_loader()

    def _refresh_table_only(self):
        self.target_table.blockSignals(True)
        try:
            selected = {row: (self.target_table.cellWidget(row, COL_CHECK).isChecked()
                              if self.target_table.cellWidget(row, COL_CHECK) else False)
                        for row in range(self.target_table.rowCount())}
            self.target_table.setRowCount(0)
            for i, t in enumerate(self.tm.targets):
                row = self.target_table.rowCount()
                self.target_table.insertRow(row)
                cb = CheckBox()
                cb.setChecked(selected.get(i, t.enabled))
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
        finally:
            self.target_table.blockSignals(False)

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

    def _on_priority_changed(self, item):
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
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        text, ok = QInputDialog.getText(
            self, "最终确认", f"请输入目标名称 '{target.name}' 以确认删除：")
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
        self.ctx.hub.targets_changed.emit()

    # ========== 模型装载 ==========

    def _start_model_loader(self):
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
        self._refresh_table_only()

    def _on_models_failed(self, errors: list):
        self.ctx.apply_failed_models(errors)
        for target_id, model_path, err in errors:
            t = self.tm.get(target_id)
            name = t.name if t else target_id
            self.status_message.emit(f"模型加载失败 [{name}]: {err}")
        self._refresh_table_only()

    def _load_opencv_templates(self):
        detector = self.ctx.opencv_detector
        for t in self.tm.targets:
            if t.detection_mode in ("opencv", "both") and t.template_path:
                path = os.path.abspath(t.template_path)
                if not os.path.exists(path):
                    logger.warning("模板文件不存在: %s", path)
                    self.status_message.emit(f"模板文件不存在 [{t.name}]: {path}")
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
            self.status_message.emit("请先选择至少一个检测目标")
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
        self.detect_thread.error.connect(self._on_detection_error)
        self.detect_thread.start()
        self._fps_warned = False

        self.start_btn.setText("停止检测")
        self.start_btn.setObjectName("danger_btn")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        if self.floating_auto_switch_cb.isChecked():
            self.floating_window_requested.emit(True, True)

    def _stop_detection(self):
        if self.detect_thread:
            worker, self.detect_thread = self.detect_thread, None
            if not worker.stop(timeout_ms=2000):
                self.status_message.emit("正在等待当前检测帧完成…")
        self.start_btn.setText("开始检测")
        self.start_btn.setObjectName("success_btn")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        if self.floating_auto_switch_cb.isChecked():
            self.floating_window_requested.emit(False, False)

    def stop_if_running(self, timeout_ms: int = 8000) -> bool:
        was_running = self.is_detecting
        if was_running and self.detect_thread is not None:
            worker = self.detect_thread
            self.detect_thread = None
            worker.stop(timeout_ms=timeout_ms)
        return was_running

    @Slot(str)
    def _on_detection_error(self, message: str):
        self.detect_thread = None
        self.start_btn.setText("开始检测")
        self.start_btn.setObjectName("success_btn")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.status_message.emit(message)

    def set_preview_enabled(self, enabled: bool):
        self._preview_enabled = enabled

    @property
    def is_detecting(self) -> bool:
        return bool(self.detect_thread and self.detect_thread.running)

    # ========== 帧回调 ==========

    @Slot(np.ndarray, list)
    def _on_frame(self, img: np.ndarray, detections: list):
        if self._preview_enabled:
            for det in detections:
                b, g, r = bgr_color(det.color)
                cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (b, g, r), 2)
                label = f"{det.target_name}[{det.source}] {det.conf:.2f}"
                cv2.putText(img, label, (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
            qt_img = bgr_to_qimage(img)
            from PySide6.QtGui import QPixmap
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
        if not self._fps_warned and self.detect_thread is not None:
            target = self.detect_thread.engine.config.target_fps
            if fps < target * 0.7:
                self._fps_warned = True
                self.status_message.emit(
                    f"实测帧率 ({fps:.1f}) 低于目标 ({target})，"
                    f"可降低帧率/换更小模型/减少同时检测的目标数")

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
        super().showEvent(event)
        self.refresh_targets()
