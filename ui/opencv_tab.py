import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QSplitter,
    QDoubleSpinBox, QInputDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QFrame, QDialog, QDialogButtonBox,
    QStyledItemDelegate,
)

from core.capture import ScreenCapture
from core.opencv_detector import OpenCVDetector
from core.target_manager import TargetManager, Target
from core.sample_set_manager import SampleSetManager

# Column indices
COL_COLOR = 0
COL_NAME = 1
COL_THRESHOLD = 2
COL_STATUS = 3


class _ColorDelegate(QStyledItemDelegate):
    """Delegate that always paints the item's background color, even when selected."""
    def paint(self, painter, option, index):
        color = index.data(Qt.BackgroundRole)
        if color:
            painter.fillRect(option.rect, color)
        else:
            super().paint(painter, option, index)


class _DrawOverlay(QWidget):
    """Transparent overlay for real-time drag preview."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._start = QPoint()
        self._end = QPoint()
        self._active = False
        self._color = QColor(255, 255, 255)

    def set_color(self, color: QColor):
        self._color = color
        self.update()

    def start(self, start: QPoint, end: QPoint):
        self._start, self._end, self._active = start, end, True
        self.update()

    def move(self, end: QPoint):
        self._end = end
        self.update()

    def stop(self):
        self._active = False
        self.update()

    def paintEvent(self, event):
        if not self._active:
            return
        painter = QPainter(self)
        painter.setPen(QPen(self._color, 2, Qt.DashLine))
        painter.drawRect(QRect(self._start, self._end).normalized())
        painter.end()


class TemplateCanvas(QLabel):
    region_selected = Signal(QRect)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("""
            QLabel {
                background-color: #0a0a16;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
            }
        """)
        self._pixmap = None
        self._drawing = False
        self._start = QPoint()
        self._end = QPoint()
        self._selection = QRect()
        self._selection_color = QColor(0, 255, 0)
        self._overlay = _DrawOverlay(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._selection = QRect()
        self._update_display()

    def set_selection_color(self, color_hex: str):
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        self._selection_color = QColor(r, g, b)
        self._overlay.set_color(QColor(r, g, b))

    def set_selection(self, rect: QRect):
        """Set selection rectangle and update display."""
        self._selection = rect
        self._update_display()

    def _update_display(self):
        if self._pixmap is None:
            return
        canvas = self._pixmap.copy()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self._selection.isNull():
            pen = QPen(self._selection_color, 2)
            painter.setPen(pen)
            painter.drawRect(self._selection)
        painter.end()
        scaled = canvas.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._pixmap:
            self._drawing = True
            self._start = self._map_to_image(ev.pos())
            self._end = self._start
            self._overlay.start(ev.pos(), ev.pos())

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._end = self._map_to_image(ev.pos())
            self._overlay.move(ev.pos())

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self._end = self._map_to_image(ev.pos())
            self._overlay.stop()
            rect = QRect(self._start, self._end).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self._selection = rect
                self.region_selected.emit(rect)
            self._update_display()

    def _map_to_image(self, widget_pos: QPoint) -> QPoint:
        if self._pixmap is None:
            return widget_pos
        pw, ph = self._pixmap.width(), self._pixmap.height()
        lw, lh = self.width(), self.height()
        scale = min(lw / pw, lh / ph)
        offset_x = (lw - pw * scale) / 2
        offset_y = (lh - ph * scale) / 2
        ix = int((widget_pos.x() - offset_x) / scale)
        iy = int((widget_pos.y() - offset_y) / scale)
        return QPoint(max(0, min(ix, pw)), max(0, min(iy, ph)))


class OpenCVTab(QWidget):
    def __init__(self, target_manager: TargetManager, opencv_detector: OpenCVDetector,
                 config: dict = None, sample_set_manager=None):
        super().__init__()
        self.tm = target_manager
        self.detector = opencv_detector
        self.capture = ScreenCapture()
        self.config = config or {}
        self._current_screenshot = None
        self._block_save = False
        self._pending_target_id = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Left: template preview / capture
        left_group = QGroupBox("模板截取")
        left_layout = QVBoxLayout(left_group)
        left_layout.setSpacing(8)

        self.canvas = TemplateCanvas()
        self.canvas.region_selected.connect(self._on_region_selected)
        left_layout.addWidget(self.canvas)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.capture_btn = QPushButton("截取屏幕 (F12)")
        self.capture_btn.clicked.connect(self._take_screenshot)
        btn_layout.addWidget(self.capture_btn)

        left_layout.addLayout(btn_layout)

        self.selection_info = QLabel("请截取屏幕后框选模板区域")
        self.selection_info.setStyleSheet("color: #8888a8;")
        left_layout.addWidget(self.selection_info)

        layout.addWidget(left_group, stretch=2)

        # Right: template table + settings
        right_group = QGroupBox("模板管理")
        right_layout = QVBoxLayout(right_group)
        right_layout.setSpacing(12)

        # Template table
        table_label = QLabel("已添加的模板:")
        table_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        right_layout.addWidget(table_label)

        self.template_table = QTableWidget(0, 4)
        self.template_table.setHorizontalHeaderLabels(["颜色", "名称", "阈值", "状态"])
        self.template_table.setItemDelegateForColumn(COL_COLOR, _ColorDelegate(self.template_table))
        self.template_table.setColumnWidth(COL_COLOR, 40)
        self.template_table.setColumnWidth(COL_NAME, 100)
        self.template_table.setColumnWidth(COL_THRESHOLD, 50)
        self.template_table.setColumnWidth(COL_STATUS, 40)
        self.template_table.horizontalHeader().setStretchLastSection(True)
        self.template_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.template_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.template_table.currentCellChanged.connect(self._on_template_selected)
        self.template_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.template_table.itemChanged.connect(self._on_item_changed)
        right_layout.addWidget(self.template_table)

        # Match threshold
        threshold_label = QLabel("新模板匹配阈值:")
        threshold_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        right_layout.addWidget(threshold_label)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setValue(0.8)
        self.threshold_spin.setSingleStep(0.05)
        right_layout.addWidget(self.threshold_spin)

        # Buttons
        target_btn_layout = QHBoxLayout()
        target_btn_layout.setSpacing(8)

        self.add_btn = QPushButton("增加")
        self.add_btn.setObjectName("primary_btn")
        self.add_btn.clicked.connect(self._add_template)
        target_btn_layout.addWidget(self.add_btn)

        self.upload_btn = QPushButton("上传")
        self.upload_btn.clicked.connect(self._upload_template)
        target_btn_layout.addWidget(self.upload_btn)

        self.remove_template_btn = QPushButton("删除")
        self.remove_template_btn.setObjectName("danger_btn")
        self.remove_template_btn.clicked.connect(self._remove_template)
        target_btn_layout.addWidget(self.remove_template_btn)

        right_layout.addLayout(target_btn_layout)

        # Test button
        self.test_btn = QPushButton("测试匹配")
        self.test_btn.setObjectName("success_btn")
        self.test_btn.clicked.connect(self._test_match)
        right_layout.addWidget(self.test_btn)

        # Verify button (for debugging)
        self.verify_btn = QPushButton("验证模板")
        self.verify_btn.clicked.connect(self._verify_template)
        right_layout.addWidget(self.verify_btn)

        right_layout.addStretch()
        layout.addWidget(right_group, stretch=1)

        # Keyboard shortcut for screenshot
        self.screenshot_shortcut = QShortcut(QKeySequence("F12"), self)
        self.screenshot_shortcut.activated.connect(self._take_screenshot)

    def _take_screenshot(self):
        """Take screenshot using global config settings."""
        win_capture = self.config.get("window_capture_enabled", False)
        win_title = self.config.get("window_capture_title", "")

        if win_capture and win_title:
            hwnd = self.capture.find_window(win_title)
            if hwnd:
                img = self.capture.capture_window(hwnd)
            else:
                img = self.capture.capture_full()
        else:
            hide = self.config.get("screenshot_hide_window", True)
            if hide:
                hwnd = ScreenCapture.get_foreground_hwnd()
                img = self.capture.capture_hide_window(hwnd)
            else:
                img = self.capture.capture_full()

        self._current_screenshot = img

        # Save screenshot to file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        screenshots_dir = os.path.join(base_dir, "data", "screenshots", "opencv")
        saved_path = ScreenCapture.save_screenshot(img, screenshots_dir, "opencv")
        self.selection_info.setText(f"截图已保存: {saved_path}")

        # Set selection color based on pending target
        if self._pending_target_id:
            target = self.tm.get(self._pending_target_id)
            if target:
                self.canvas.set_selection_color(target.color)

        self._display_image(img)

    def _display_image(self, img: np.ndarray):
        """Display an image on the canvas."""
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qt_img))

    def _add_template(self):
        """Create empty template with name, then wait for screenshot + selection."""
        name, ok = QInputDialog.getText(self, "新建模板", "模板名称:")
        if not ok or not name:
            return

        # Create target with empty template (OpenCV targets have no sample set)
        target = self.tm.add(
            name=name,
            detection_mode="opencv",
            match_threshold=self.threshold_spin.value(),
        )
        os.makedirs(os.path.join(target.data_dir, "templates"), exist_ok=True)
        self.tm.save()
        self._refresh_table()
        self._pending_target_id = target.id

        # Set selection color
        self.canvas.set_selection_color(target.color)

        # Select the new template in table
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        idx = opencv_targets.index(target)
        self.template_table.setCurrentCell(idx, 0)

        self.selection_info.setText(f"已创建模板 '{name}'，请截图后框选模板区域")

    def _upload_template(self):
        """Upload template with dialog to choose upload type."""
        row = self.template_table.currentRow()
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        if row < 0 or row >= len(opencv_targets):
            QMessageBox.warning(self, "提示", "请先选择一个模板")
            return

        target = opencv_targets[row]

        # Check if template already has image
        has_image = self._template_has_image(target)
        if has_image:
            reply = QMessageBox.question(
                self, "确认覆盖",
                f"模板 '{target.name}' 已有图片，是否覆盖？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Show upload type dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("上传模板")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        label = QLabel("选择上传方式:")
        layout.addWidget(label)

        btn_box = QDialogButtonBox()
        img_btn = btn_box.addButton("上传图片（框选）", QDialogButtonBox.AcceptRole)
        tmpl_btn = btn_box.addButton("上传模板（直接使用）", QDialogButtonBox.AcceptRole)
        cancel_btn = btn_box.addButton("取消", QDialogButtonBox.RejectRole)
        layout.addWidget(btn_box)

        result = {"choice": None}
        img_btn.clicked.connect(lambda: result.update(choice="image") or dialog.accept())
        tmpl_btn.clicked.connect(lambda: result.update(choice="template") or dialog.accept())
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.Accepted:
            return

        if result["choice"] == "image":
            # Upload full image, then user selects region
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.png *.jpg *.bmp *.jpeg *.webp)")
            if path:
                img = cv2.imread(path)
                if img is not None:
                    self._current_screenshot = img
                    self._pending_target_id = target.id
                    # Save source image path
                    target.source_image_path = path
                    self.tm.save()
                    # Set selection color
                    self.canvas.set_selection_color(target.color)
                    self._display_image(img)
                    self.selection_info.setText(f"已加载图片，请框选模板区域")
        elif result["choice"] == "template":
            # Upload pre-cropped template directly
            path, _ = QFileDialog.getOpenFileName(self, "选择模板图片", "", "图片 (*.png *.jpg *.bmp *.jpeg *.webp)")
            if path:
                self._save_template_direct(target, path)

    def _template_has_image(self, target: Target) -> bool:
        """Check if target already has a template image."""
        if target.template_path and os.path.exists(target.template_path):
            return True
        return False

    def _save_template_direct(self, target: Target, image_path: str):
        """Save an image directly as template (no selection needed)."""
        tmpl_dir = os.path.join(target.data_dir, "templates")
        os.makedirs(tmpl_dir, exist_ok=True)
        tmpl_path = os.path.join(tmpl_dir, f"{target.id}_template.png")

        # Copy image
        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return

        cv2.imwrite(tmpl_path, img)
        target.template_path = tmpl_path
        # For direct upload, source_image_path is the same as template_path
        target.source_image_path = tmpl_path
        self.detector.load_template(target.id, tmpl_path, target.match_threshold)
        self.tm.save()
        self._refresh_table()
        QMessageBox.information(self, "保存成功", f"模板 '{target.name}' 已保存")

    @Slot(QRect)
    def _on_region_selected(self, rect: QRect):
        if self._current_screenshot is None:
            self.log_debug("错误: 当前没有截图")
            return
        h, w = self._current_screenshot.shape[:2]
        x1 = max(0, min(rect.x(), w))
        y1 = max(0, min(rect.y(), h))
        x2 = max(0, min(rect.x() + rect.width(), w))
        y2 = max(0, min(rect.y() + rect.height(), h))
        if x2 <= x1 or y2 <= y1:
            self.log_debug(f"错误: 选择区域无效: ({x1},{y1})-({x2},{y2})")
            return

        template = self._current_screenshot[y1:y2, x1:x2]
        self.log_debug(f"选择区域: ({x1},{y1})-({x2},{y2}), 模板尺寸: {template.shape}")

        # If we have a pending target, save to it
        if self._pending_target_id:
            target = self.tm.get(self._pending_target_id)
            if not target:
                self._pending_target_id = None
                self.log_debug("错误: 未找到目标")
                return

            # Check if this is first selection (empty template) or overwrite
            is_first = not self._template_has_image(target)

            if not is_first:
                # Confirm overwrite
                reply = QMessageBox.question(
                    self, "确认覆盖",
                    f"模板 '{target.name}' 已有图片，是否覆盖为新选择的区域？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            # Save template
            self._save_template(target, template)
            # Update canvas to show selection rectangle
            self.canvas.set_selection(rect)
            self.selection_info.setText(
                f"已选择区域: ({x1},{y1})-({x2},{y2})  大小: {x2-x1}x{y2-y1}  → 模板 '{target.name}'"
            )
        else:
            self.log_debug("警告: 没有待处理的目标ID")

    def _save_template(self, target: Target, template: np.ndarray):
        """Save template image to target."""
        tmpl_dir = os.path.join(target.data_dir, "templates")
        os.makedirs(tmpl_dir, exist_ok=True)
        tmpl_path = os.path.join(tmpl_dir, f"{target.id}_template.png")

        # Debug: print template info
        self.log_debug(f"保存模板: 目标={target.name}, 尺寸={template.shape}, 路径={tmpl_path}")

        # Save template
        success = cv2.imwrite(tmpl_path, template)
        self.log_debug(f"cv2.imwrite 结果: {success}")

        # Verify file exists
        if os.path.exists(tmpl_path):
            file_size = os.path.getsize(tmpl_path)
            self.log_debug(f"模板文件已保存: {tmpl_path}, 大小={file_size} bytes")

            # Verify file can be read back
            verify_img = cv2.imread(tmpl_path)
            if verify_img is not None:
                self.log_debug(f"模板文件验证通过: 尺寸={verify_img.shape}")
            else:
                self.log_debug(f"错误: 模板文件无法读取验证!")
        else:
            self.log_debug(f"错误: 模板文件未保存!")

        target.template_path = tmpl_path
        self.detector.load_template(target.id, tmpl_path, target.match_threshold)
        self.tm.save()
        self._refresh_table()

    def _on_template_selected(self, row: int, col: int, prev_row: int, prev_col: int):
        """When a template is selected in the table, show its source image."""
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        if row < 0 or row >= len(opencv_targets):
            return

        target = opencv_targets[row]
        self._pending_target_id = target.id

        # Set selection color
        self.canvas.set_selection_color(target.color)

        # Show source image if available
        if target.source_image_path and os.path.exists(target.source_image_path):
            img = cv2.imread(target.source_image_path)
            if img is not None:
                self._current_screenshot = img
                self._display_image(img)

                # Show selection if template exists
                if target.template_path and os.path.exists(target.template_path):
                    # Find the template location in the source image
                    template_img = cv2.imread(target.template_path)
                    if template_img is not None:
                        result = cv2.matchTemplate(img, template_img, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        if max_val >= target.match_threshold:
                            th, tw = template_img.shape[:2]
                            x1, y1 = max_loc
                            rect = QRect(x1, y1, tw, th)
                            self.canvas.set_selection(rect)

                self.selection_info.setText(f"模板 '{target.name}' 的源图片")
                return

        # If no source image, show template directly if it exists
        if target.template_path and os.path.exists(target.template_path):
            img = cv2.imread(target.template_path)
            if img is not None:
                self._current_screenshot = img
                self._display_image(img)
                self.selection_info.setText(f"模板 '{target.name}' 的模板图片")
                return

        # No image available
        self.selection_info.setText(f"模板 '{target.name}' 暂无图片，请截图或上传")

    def _remove_template(self):
        row = self.template_table.currentRow()
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        if row < 0 or row >= len(opencv_targets):
            return
        target = opencv_targets[row]
        reply = QMessageBox.question(self, "确认删除", f"确定删除模板 '{target.name}' 吗？")
        if reply == QMessageBox.Yes:
            self.detector.remove_template(target.id)
            self.tm.remove(target.id)
            self.tm.save()
            self._refresh_table()
            if self._pending_target_id == target.id:
                self._pending_target_id = None

    def _on_cell_double_clicked(self, row: int, col: int):
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        if row < 0 or row >= len(opencv_targets):
            return
        target = opencv_targets[row]

        if col == COL_NAME:
            new_name, ok = QInputDialog.getText(self, "重命名", "新名称:", text=target.name)
            if ok and new_name:
                target.name = new_name
                self.tm.save()
                self._refresh_table()

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._block_save:
            return
        row = item.row()
        col = item.column()
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        if row < 0 or row >= len(opencv_targets):
            return
        target = opencv_targets[row]

        if col == COL_THRESHOLD:
            try:
                val = float(item.text())
                if 0.1 <= val <= 1.0:
                    target.match_threshold = val
                    if target.template_path:
                        self.detector.load_template(target.id, target.template_path, val)
                    self.tm.save()
            except ValueError:
                pass

    def _test_match(self):
        if self._current_screenshot is None:
            QMessageBox.warning(self, "提示", "请先截取屏幕")
            return

        # Debug info
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        loaded = list(self.detector.templates.keys())
        self.log_debug(f"测试匹配: 输入图像尺寸={self._current_screenshot.shape}, 已加载模板={loaded}, 共{len(opencv_targets)}个模板")

        # Check each template
        for t in opencv_targets:
            tmpl_exists = t.template_path and os.path.exists(t.template_path)
            in_detector = t.id in self.detector.templates
            self.log_debug(f"  模板 '{t.name}': template_path={t.template_path}, 文件存在={tmpl_exists}, 检测器中={in_detector}")

            # Verify template file if it exists
            if tmpl_exists:
                try:
                    tmpl_img = cv2.imread(t.template_path)
                    if tmpl_img is not None:
                        self.log_debug(f"    模板文件可读取: 尺寸={tmpl_img.shape}")

                        # Test matching directly
                        result = cv2.matchTemplate(self._current_screenshot, tmpl_img, cv2.TM_CCOEFF_NORMED)
                        max_val = np.max(result)
                        self.log_debug(f"    直接匹配测试: 最大匹配度={max_val:.4f}, 阈值={t.match_threshold}")

                        if max_val < t.match_threshold:
                            self.log_debug(f"    警告: 最大匹配度低于阈值!")
                    else:
                        self.log_debug(f"    错误: 模板文件无法读取!")
                except Exception as e:
                    self.log_debug(f"    错误: 读取模板文件异常: {e}")

        detections = self.detector.detect(self._current_screenshot)
        if detections:
            img = self._current_screenshot.copy()
            for det in detections:
                target = self.tm.get(det.target_id)
                color_hex = target.color if target else "#00ff00"
                r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (b, g, r), 2)
                cv2.putText(img, f"{det.target_name} {det.conf:.2f}", (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
            h, w, ch = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.canvas.set_image(QPixmap.fromImage(qt_img))
            self.selection_info.setText(f"匹配到 {len(detections)} 个目标")
        else:
            self.selection_info.setText("未匹配到任何目标")

    def log_debug(self, msg: str):
        """Print debug message to console."""
        print(f"[OpenCV Tab] {msg}")

    def _verify_template(self):
        """Verify all templates are correctly saved and loaded."""
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        self.log_debug(f"验证模板: 共 {len(opencv_targets)} 个模板")

        for t in opencv_targets:
            self.log_debug(f"\n--- 模板 '{t.name}' (ID: {t.id}) ---")
            self.log_debug(f"  template_path: {t.template_path}")
            self.log_debug(f"  match_threshold: {t.match_threshold}")
            self.log_debug(f"  detection_mode: {t.detection_mode}")

            # Check if file exists
            if t.template_path:
                if os.path.exists(t.template_path):
                    file_size = os.path.getsize(t.template_path)
                    self.log_debug(f"  文件存在: 是, 大小={file_size} bytes")

                    # Try to read the file
                    try:
                        img = cv2.imread(t.template_path)
                        if img is not None:
                            self.log_debug(f"  文件可读取: 是, 尺寸={img.shape}")
                        else:
                            self.log_debug(f"  文件可读取: 否 (cv2.imread 返回 None)")
                    except Exception as e:
                        self.log_debug(f"  文件读取异常: {e}")
                else:
                    self.log_debug(f"  文件存在: 否")
            else:
                self.log_debug(f"  template_path: 未设置")

            # Check if loaded in detector
            if t.id in self.detector.templates:
                tmpl = self.detector.templates[t.id]
                self.log_debug(f"  检测器中: 是, 尺寸={tmpl.shape}")
            else:
                self.log_debug(f"  检测器中: 否")

    def _refresh_table(self):
        self._block_save = True
        opencv_targets = [t for t in self.tm.targets if t.detection_mode in ("opencv", "both")]
        self.template_table.setRowCount(len(opencv_targets))
        for i, t in enumerate(opencv_targets):
            # Color cell — solid color block
            color_item = QTableWidgetItem()
            r, g, b = int(t.color[1:3], 16), int(t.color[3:5], 16), int(t.color[5:7], 16)
            color_item.setBackground(QBrush(QColor(r, g, b)))
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            self.template_table.setItem(i, COL_COLOR, color_item)

            # Name
            name_item = QTableWidgetItem(t.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, t.id)
            self.template_table.setItem(i, COL_NAME, name_item)

            # Threshold
            thr_item = QTableWidgetItem(f"{t.match_threshold:.2f}")
            thr_item.setTextAlignment(Qt.AlignCenter)
            self.template_table.setItem(i, COL_THRESHOLD, thr_item)

            # Status
            has_tmpl = t.template_path and os.path.exists(t.template_path)
            status_item = QTableWidgetItem("✓" if has_tmpl else "等待框选")
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            status_item.setForeground(QBrush(QColor(34, 197, 94) if has_tmpl else QColor(234, 179, 8)))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.template_table.setItem(i, COL_STATUS, status_item)

        self._block_save = False

    def refresh_from_manager(self):
        self._refresh_table()
        loaded_count = 0
        for t in self.tm.targets:
            if t.detection_mode in ("opencv", "both") and t.template_path:
                path = os.path.abspath(t.template_path)
                if not os.path.exists(path):
                    self.log_debug(f"模板文件不存在: {path}")
                    continue
                for attempt in range(3):
                    if self.detector.load_template(t.id, path, t.match_threshold):
                        loaded_count += 1
                        break
                    self.log_debug(f"模板加载重试 {attempt+1}/3: {t.name}")
        self.log_debug(f"从管理器加载了 {loaded_count} 个模板")

    def showEvent(self, event):
        """Refresh template table when tab becomes visible."""
        super().showEvent(event)
        self._refresh_table()
