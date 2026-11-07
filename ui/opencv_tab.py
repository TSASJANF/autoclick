"""OpenCV 模板页：模板的创建、框选、上传、测试与删除。

布局：ScrollArea 卡片式布局。模板匹配/保存等业务逻辑不变。
"""

from __future__ import annotations

import logging
import os

import numpy as np
from PySide6.QtCore import QRect, Qt, Slot
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    DoubleSpinBox,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SubtitleLabel,
    TableWidget,
)

from core.capture import CaptureError, ScreenCapture
from core.image_io import imread, imwrite
from core.models import Target
from core.target_manager import TargetManager
from ui.app_context import AppContext
from ui.widgets.canvas import ColorDelegate, ImageCanvas
from ui.widgets.image_utils import bgr_color, bgr_to_pixmap, hex_to_rgb

logger = logging.getLogger(__name__)

COL_COLOR, COL_NAME, COL_THRESHOLD, COL_STATUS = 0, 1, 2, 3

SCREENSHOTS_DIR = os.path.join("data", "screenshots", "opencv")


class OpenCVTab(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.setObjectName("opencvInterface")
        self.ctx = ctx
        self.tm: TargetManager = ctx.tm
        self.detector = ctx.opencv_detector
        self._current_screenshot: np.ndarray | None = None
        self._block_save = False
        self._pending_target_id: str | None = None
        self._init_ui()

    # ========== UI ==========

    def _init_ui(self):
        # 双栏：左截取画布，右侧可滚动模板管理 —— 更紧凑
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(10)

        root.addWidget(self._build_capture_card(), 3)

        right_scroll = ScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_content = QWidget()
        right_scroll.setWidget(right_content)
        right_lay = QVBoxLayout(right_content)
        right_lay.setContentsMargins(0, 0, 4, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(self._build_template_card())
        right_lay.addStretch()
        root.addWidget(right_scroll, 2)

        self.screenshot_shortcut = QShortcut(QKeySequence("F12"), self)
        self.screenshot_shortcut.activated.connect(self._take_screenshot)

    def _card(self, title: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)
        if title:
            lay.addWidget(SubtitleLabel(title))
        return card, lay

    def _build_capture_card(self) -> CardWidget:
        card, lay = self._card("模板截取")
        self.canvas = ImageCanvas()
        self.canvas.region_drawn.connect(self._on_region_selected)
        lay.addWidget(self.canvas)

        self.capture_btn = PushButton(FluentIcon.CAMERA, "截取屏幕 (F12)")
        self.capture_btn.clicked.connect(self._take_screenshot)
        lay.addWidget(self.capture_btn)

        self.selection_info = BodyLabel("请截取屏幕后框选模板区域")
        lay.addWidget(self.selection_info)
        return card

    def _build_template_card(self) -> CardWidget:
        card, lay = self._card("模板管理")

        self.template_table = TableWidget()
        self.template_table.setColumnCount(4)
        self.template_table.setHorizontalHeaderLabels(["颜色", "名称", "阈值", "状态"])
        self.template_table.setItemDelegateForColumn(COL_COLOR, ColorDelegate(self.template_table))
        header = self.template_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.template_table.setMinimumHeight(200)
        self.template_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.template_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.template_table.currentCellChanged.connect(self._on_template_selected)
        self.template_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.template_table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.template_table)

        lay.addWidget(BodyLabel("新模板匹配阈值"))
        self.threshold_spin = DoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setValue(0.8)
        self.threshold_spin.setSingleStep(0.05)
        lay.addWidget(self.threshold_spin)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.add_btn = PrimaryPushButton(FluentIcon.ADD, "增加")
        self.add_btn.clicked.connect(self._add_template)
        row.addWidget(self.add_btn)
        self.upload_btn = PushButton(FluentIcon.UP, "上传")
        self.upload_btn.clicked.connect(self._upload_template)
        row.addWidget(self.upload_btn)
        self.remove_template_btn = PushButton(FluentIcon.DELETE, "删除")
        self.remove_template_btn.clicked.connect(self._remove_template)
        row.addWidget(self.remove_template_btn)
        self.test_btn = PrimaryPushButton(FluentIcon.PLAY, "测试匹配")
        self.test_btn.clicked.connect(self._test_match)
        row.addWidget(self.test_btn)
        row.addStretch()
        lay.addLayout(row)
        return card

    # ========== 截图与显示 ==========

    def _take_screenshot(self):
        try:
            img = self.ctx.capture.capture_by_config(self.ctx.config)
        except CaptureError as e:
            self.selection_info.setText(f"截图失败: {e}")
            return
        self._current_screenshot = img
        saved_path = ScreenCapture.save_screenshot(img, SCREENSHOTS_DIR, "opencv")
        self.selection_info.setText(f"截图已保存: {saved_path}")

        if self._pending_target_id:
            target = self.tm.get(self._pending_target_id)
            if target:
                self.canvas.set_draw_color(target.color)
        self.canvas.set_image(bgr_to_pixmap(img))

    def _display_image(self, img: np.ndarray):
        self._current_screenshot = img
        self.canvas.set_image(bgr_to_pixmap(img))

    # ========== 模板管理 ==========

    def _opencv_targets(self) -> list[Target]:
        return self.tm.get_opencv_targets()

    def _add_template(self):
        name, ok = QInputDialog.getText(self, "新建模板", "模板名称:")
        if not ok or not name:
            return
        target = self.tm.add(name=name, detection_mode="opencv",
                             match_threshold=self.threshold_spin.value())
        os.makedirs(os.path.join(target.data_dir, "templates"), exist_ok=True)
        self.tm.save()
        self._refresh_table()
        self._pending_target_id = target.id
        self.canvas.set_draw_color(target.color)

        idx = self._opencv_targets().index(target)
        self.template_table.setCurrentCell(idx, 0)
        self.selection_info.setText(f"已创建模板 '{name}'，请截图后框选模板区域")

    def _upload_template(self):
        row = self.template_table.currentRow()
        targets = self._opencv_targets()
        if row < 0 or row >= len(targets):
            QMessageBox.warning(self, "提示", "请先选择一个模板")
            return
        target = targets[row]

        if self._template_has_image(target):
            reply = QMessageBox.question(
                self, "确认覆盖",
                f"模板 '{target.name}' 已有图片，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        choice = self._choose_upload_type()
        if choice == "image":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片 (*.png *.jpg *.bmp *.jpeg *.webp)")
            if path:
                img = imread(path)
                if img is not None:
                    self._pending_target_id = target.id
                    target.source_image_path = path
                    self.tm.save()
                    self.canvas.set_draw_color(target.color)
                    self._display_image(img)
                    self.selection_info.setText("已加载图片，请框选模板区域")
        elif choice == "template":
            path, _ = QFileDialog.getOpenFileName(
                self, "选择模板图片", "", "图片 (*.png *.jpg *.bmp *.jpeg *.webp)")
            if path:
                self._save_template_direct(target, path)

    @staticmethod
    def _choose_upload_type() -> str | None:
        dialog = QDialog()
        dialog.setWindowTitle("上传模板")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("选择上传方式:"))

        result = {"choice": None}
        btn_box = QDialogButtonBox()
        img_btn = btn_box.addButton("上传图片（框选）", QDialogButtonBox.AcceptRole)
        tmpl_btn = btn_box.addButton("上传模板（直接使用）", QDialogButtonBox.AcceptRole)
        btn_box.addButton("取消", QDialogButtonBox.RejectRole)
        layout.addWidget(btn_box)

        img_btn.clicked.connect(lambda: result.update(choice="image"))
        tmpl_btn.clicked.connect(lambda: result.update(choice="template"))
        if dialog.exec() != QDialog.Accepted:
            return None
        return result["choice"]

    @staticmethod
    def _template_has_image(target: Target) -> bool:
        return bool(target.template_path and os.path.exists(target.template_path))

    def _save_template_direct(self, target: Target, image_path: str):
        img = imread(image_path)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return
        tmpl_path = self._write_template_file(target, img)
        target.template_path = tmpl_path
        target.source_image_path = tmpl_path
        self.detector.load_template(target.id, tmpl_path, target.match_threshold)
        self.tm.save()
        self._refresh_table()
        QMessageBox.information(self, "保存成功", f"模板 '{target.name}' 已保存")

    def _write_template_file(self, target: Target, template: np.ndarray) -> str:
        tmpl_dir = os.path.join(target.data_dir, "templates")
        os.makedirs(tmpl_dir, exist_ok=True)
        tmpl_path = os.path.join(tmpl_dir, f"{target.id}_template.png")
        if not imwrite(tmpl_path, template):
            raise RuntimeError(f"模板写入失败: {tmpl_path}")
        return tmpl_path

    @Slot(QRect)
    def _on_region_selected(self, rect: QRect):
        if self._current_screenshot is None:
            self.selection_info.setText("错误: 当前没有截图")
            return
        h, w = self._current_screenshot.shape[:2]
        x1 = max(0, min(rect.x(), w))
        y1 = max(0, min(rect.y(), h))
        x2 = max(0, min(rect.x() + rect.width(), w))
        y2 = max(0, min(rect.y() + rect.height(), h))
        if x2 <= x1 or y2 <= y1:
            logger.debug("选择区域无效: (%d,%d)-(%d,%d)", x1, y1, x2, y2)
            return

        if not self._pending_target_id:
            logger.debug("没有待处理的目标 ID")
            return
        target = self.tm.get(self._pending_target_id)
        if target is None:
            self._pending_target_id = None
            return

        if self._template_has_image(target):
            reply = QMessageBox.question(
                self, "确认覆盖",
                f"模板 '{target.name}' 已有图片，是否覆盖为新选择的区域？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        template = self._current_screenshot[y1:y2, x1:x2]
        try:
            target.template_path = self._write_template_file(target, template)
        except RuntimeError as e:
            QMessageBox.warning(self, "错误", str(e))
            return
        self.detector.load_template(target.id, target.template_path, target.match_threshold)
        self.tm.save()
        self._refresh_table()
        self.canvas.set_selection(QRect(x1, y1, x2 - x1, y2 - y1))
        self.selection_info.setText(
            f"已选择区域: ({x1},{y1})-({x2},{y2})  大小: {x2 - x1}x{y2 - y1}"
            f"  → 模板 '{target.name}'")

    def _on_template_selected(self, row: int, *_args):
        targets = self._opencv_targets()
        if row < 0 or row >= len(targets):
            return
        target = targets[row]
        self._pending_target_id = target.id
        self.canvas.set_draw_color(target.color)

        if target.source_image_path and os.path.exists(target.source_image_path):
            img = imread(target.source_image_path)
            if img is not None:
                self._display_image(img)
                self._show_template_location(img, target)
                self.selection_info.setText(f"模板 '{target.name}' 的源图片")
                return

        if self._template_has_image(target):
            img = imread(target.template_path)
            if img is not None:
                self._display_image(img)
                self.selection_info.setText(f"模板 '{target.name}' 的模板图片")
                return

        self.selection_info.setText(f"模板 '{target.name}' 暂无图片，请截图或上传")

    def _show_template_location(self, source_img: np.ndarray, target: Target):
        import cv2
        if not self._template_has_image(target):
            return
        template_img = imread(target.template_path)
        if template_img is None:
            return
        th, tw = template_img.shape[:2]
        if source_img.shape[0] < th or source_img.shape[1] < tw:
            return
        result = cv2.matchTemplate(source_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= target.match_threshold:
            self.canvas.set_selection(QRect(max_loc[0], max_loc[1], tw, th))

    def _remove_template(self):
        row = self.template_table.currentRow()
        targets = self._opencv_targets()
        if row < 0 or row >= len(targets):
            return
        target = targets[row]
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除模板 '{target.name}' 吗？")
        if reply != QMessageBox.Yes:
            return
        self.detector.remove_template(target.id)
        self.tm.remove(target.id)
        self.tm.save()
        self._refresh_table()
        if self._pending_target_id == target.id:
            self._pending_target_id = None

    def _on_cell_double_clicked(self, row: int, col: int):
        targets = self._opencv_targets()
        if row < 0 or row >= len(targets):
            return
        target = targets[row]
        if col == COL_NAME:
            new_name, ok = QInputDialog.getText(self, "重命名", "新名称:", text=target.name)
            if ok and new_name:
                target.name = new_name
                self.tm.save()
                self._refresh_table()

    def _on_item_changed(self, item):
        if self._block_save:
            return
        targets = self._opencv_targets()
        row = item.row()
        if row < 0 or row >= len(targets):
            return
        target = targets[row]
        if item.column() == COL_THRESHOLD:
            try:
                val = float(item.text())
                if 0.1 <= val <= 1.0:
                    target.match_threshold = val
                    if target.template_path:
                        self.detector.load_template(target.id, target.template_path, val)
                    self.tm.save()
            except ValueError:
                pass

    # ========== 测试匹配 ==========

    def _test_match(self):
        if self._current_screenshot is None:
            QMessageBox.warning(self, "提示", "请先截取屏幕")
            return
        detections = self.detector.detect(self._current_screenshot)
        if not detections:
            self.selection_info.setText("未匹配到任何目标")
            return

        import cv2
        img = self._current_screenshot.copy()
        for det in detections:
            target = self.tm.get(det.target_id)
            color_hex = target.color if target else "#00ff00"
            b, g, r = bgr_color(color_hex)
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (b, g, r), 2)
            cv2.putText(img, f"{det.target_name} {det.conf:.2f}",
                        (det.x1, det.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
        self.canvas.set_image(bgr_to_pixmap(img))
        self.selection_info.setText(f"匹配到 {len(detections)} 个目标")

    # ========== 表格刷新 ==========

    def _refresh_table(self):
        self._block_save = True
        targets = self._opencv_targets()
        self.template_table.setRowCount(len(targets))
        for i, t in enumerate(targets):
            color_item = QTableWidgetItem()
            color_item.setBackground(QBrush(QColor(*hex_to_rgb(t.color))))
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            self.template_table.setItem(i, COL_COLOR, color_item)

            name_item = QTableWidgetItem(t.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, t.id)
            self.template_table.setItem(i, COL_NAME, name_item)

            thr_item = QTableWidgetItem(f"{t.match_threshold:.2f}")
            thr_item.setTextAlignment(Qt.AlignCenter)
            self.template_table.setItem(i, COL_THRESHOLD, thr_item)

            has_tmpl = self._template_has_image(t)
            status_item = QTableWidgetItem("✓" if has_tmpl else "等待框选")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            status_item.setForeground(QBrush(
                QColor(34, 197, 94) if has_tmpl else QColor(234, 179, 8)))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.template_table.setItem(i, COL_STATUS, status_item)
        self._block_save = False

    def refresh_from_manager(self):
        self._refresh_table()
        for t in self.tm.targets:
            if t.detection_mode in ("opencv", "both") and t.template_path:
                path = os.path.abspath(t.template_path)
                if not os.path.exists(path):
                    logger.warning("模板文件不存在: %s", path)
                    continue
                self.detector.load_template(t.id, path, t.match_threshold)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_table()
