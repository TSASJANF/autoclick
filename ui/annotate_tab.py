import os
import json
import shutil
import datetime
import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QLineEdit, QGroupBox, QSplitter,
    QInputDialog, QMessageBox, QListWidgetItem, QComboBox, QMenu,
    QDialog, QFormLayout, QSpinBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QStyledItemDelegate, QFrame, QSizePolicy,
)

from core.capture import ScreenCapture
from core.target_manager import TargetManager, Target, TARGET_COLORS
from core.sample_set_manager import SampleSetManager, SampleSet

NEW_TARGET_SENTINEL = "__new_target__"


class _ColorDelegate(QStyledItemDelegate):
    """Delegate that always paints the item's background color, even when selected."""
    def paint(self, painter, option, index):
        color = index.data(Qt.BackgroundRole)
        if color:
            painter.fillRect(option.rect, color)
        else:
            super().paint(painter, option, index)


class _DrawOverlay(QWidget):
    """Transparent overlay for real-time drag preview — avoids full pixmap redraw."""

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

    def start(self, start: QPoint, end: QPoint):
        self._start = start
        self._end = end
        self._active = True
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
        pen = QPen(self._color, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRect(self._start, self._end).normalized())
        painter.end()


class ImageCanvas(QLabel):
    box_drawn = Signal(QRect)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        self._boxes: list[tuple[QRect, int, str, str]] = []  # (rect, cls_id, color_hex, label)
        self._overlay = _DrawOverlay(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())

    def sizeHint(self):
        """Override to prevent label from expanding based on pixmap content."""
        return self.minimumSize()

    def set_image(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._boxes.clear()
        self._update_display()

    def add_box(self, rect: QRect, class_id: int, color: str = "#2ecc71", label: str = ""):
        self._boxes.append((rect, class_id, color, label))
        self._update_display()

    def clear_boxes(self):
        self._boxes.clear()
        self._update_display()

    def set_draw_color(self, color_hex: str):
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        self._overlay.set_color(QColor(r, g, b))

    def get_boxes(self) -> list[tuple[QRect, int, str, str]]:
        return list(self._boxes)

    def _update_display(self):
        if self._pixmap is None:
            return
        canvas = self._pixmap.copy()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        for rect, cls_id, color_hex, lbl in self._boxes:
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            pen = QPen(QColor(r, g, b), 2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.setFont(QFont("Segoe UI", 10))
            text = lbl if lbl else f"cls:{cls_id}"
            painter.drawText(rect.topLeft() + QPoint(2, -4), text)

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
                self.box_drawn.emit(rect)
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


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


class AnnotateTab(QWidget):
    target_deleted = Signal()
    target_added = Signal()
    sample_set_renamed = Signal(str)  # emits set_id when renamed

    def __init__(self, target_manager: TargetManager, config: dict | None = None,
                 sample_set_manager: SampleSetManager | None = None):
        super().__init__()
        self.tm = target_manager
        self.ssm = sample_set_manager or SampleSetManager()
        self.config = config or {}
        self.current_image_path = None
        self.current_image = None
        self.capture = ScreenCapture()
        self._image_list: list[str] = []
        self._current_index: int = -1
        self._current_target: Target | None = None
        self._current_set: SampleSet | None = None
        self._init_ui()
        # Populate sample sets and select first one
        self._populate_set_combo()
        if self.set_combo.count() > 0:
            self.set_combo.setCurrentIndex(0)
            # Force-initialize: setCurrentIndex(0) is a no-op if index
            # is already 0 (Qt won't emit currentIndexChanged), so we
            # must call the handler manually on first init.
            if self._current_set is None:
                self._on_sample_set_changed(0)

    # ========== UI ==========

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # === Left: image canvas ===
        canvas_group = QGroupBox("图片")
        canvas_layout = QVBoxLayout(canvas_group)
        canvas_layout.setSpacing(8)

        self.canvas = ImageCanvas()
        self.canvas.box_drawn.connect(self._on_box_drawn)
        canvas_layout.addWidget(self.canvas)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)

        self.prev_btn = QPushButton("上一张 (A)")
        self.prev_btn.clicked.connect(self._prev_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.save_btn = QPushButton("保存并下一张 (S)")
        self.save_btn.setObjectName("success_btn")
        self.save_btn.clicked.connect(self._save_and_next)
        self.save_btn.setEnabled(False)
        nav_layout.addWidget(self.save_btn)

        self.next_btn = QPushButton("下一张 (D)")
        self.next_btn.clicked.connect(self._next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        canvas_layout.addLayout(nav_layout)

        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.open_btn = QPushButton("打开图片")
        self.open_btn.clicked.connect(self._open_image)
        action_layout.addWidget(self.open_btn)

        self.open_folder_btn = QPushButton("打开文件夹")
        self.open_folder_btn.clicked.connect(self._open_folder)
        action_layout.addWidget(self.open_folder_btn)

        self.screenshot_btn = QPushButton("截图")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        action_layout.addWidget(self.screenshot_btn)

        self.reannotate_btn = QPushButton("重新标注")
        self.reannotate_btn.setToolTip("清除当前图片在当前目标下的标注，重新开始")
        self.reannotate_btn.clicked.connect(self._reannotate)
        self.reannotate_btn.setEnabled(False)
        action_layout.addWidget(self.reannotate_btn)

        canvas_layout.addLayout(action_layout)

        # Progress label
        self.progress_label = QLabel("未加载图片")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        canvas_layout.addWidget(self.progress_label)

        # === Right panel ===
        right_splitter = QSplitter(Qt.Vertical)

        # --- Sample set selector ---
        set_group = QGroupBox("样本集")
        set_layout = QHBoxLayout(set_group)
        set_layout.setSpacing(8)

        self.set_combo = QComboBox()
        self.set_combo.currentIndexChanged.connect(self._on_sample_set_changed)
        set_layout.addWidget(self.set_combo, 1)

        self.add_set_btn = QPushButton("新建")
        self.add_set_btn.clicked.connect(self._create_sample_set)
        set_layout.addWidget(self.add_set_btn)

        self.rename_set_btn = QPushButton("重命名")
        self.rename_set_btn.clicked.connect(self._rename_sample_set)
        set_layout.addWidget(self.rename_set_btn)

        self.delete_set_btn = QPushButton("删除")
        self.delete_set_btn.setObjectName("danger_btn")
        self.delete_set_btn.clicked.connect(self._delete_sample_set)
        set_layout.addWidget(self.delete_set_btn)

        right_splitter.addWidget(set_group)

        # --- Target table ---
        target_group = QGroupBox("当前标注目标")
        target_layout = QVBoxLayout(target_group)
        target_layout.setSpacing(8)

        self.target_table = QTableWidget(0, 2)
        self.target_table.setHorizontalHeaderLabels(["颜色", "名称"])
        self.target_table.setItemDelegateForColumn(0, _ColorDelegate(self.target_table))
        self.target_table.setColumnWidth(0, 40)
        self.target_table.horizontalHeader().setStretchLastSection(True)
        self.target_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.target_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.target_table.setMaximumHeight(180)
        self.target_table.currentCellChanged.connect(self._on_target_row_changed)
        self.target_table.cellDoubleClicked.connect(self._on_target_cell_double_clicked)
        self.target_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.target_table.customContextMenuRequested.connect(self._on_target_context_menu)
        target_layout.addWidget(self.target_table)

        # Target info and add button
        target_btn_layout = QHBoxLayout()
        target_btn_layout.setSpacing(8)

        self.target_info_label = QLabel("选择一个目标进行标注")
        self.target_info_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        target_btn_layout.addWidget(self.target_info_label)

        add_target_btn = QPushButton("增加")
        add_target_btn.setObjectName("primary_btn")
        add_target_btn.clicked.connect(self._create_new_target)
        target_btn_layout.addWidget(add_target_btn)

        target_layout.addLayout(target_btn_layout)
        right_splitter.addWidget(target_group)

        # --- Image list ---
        img_list_group = QGroupBox("图片列表")
        img_list_layout = QVBoxLayout(img_list_group)
        img_list_layout.setSpacing(8)

        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        self.image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list_widget.customContextMenuRequested.connect(self._on_image_list_context_menu)
        img_list_layout.addWidget(self.image_list_widget)

        self.annotated_count_label = QLabel("已标注: 0 / 0")
        self.annotated_count_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        img_list_layout.addWidget(self.annotated_count_label)

        right_splitter.addWidget(img_list_group)

        # --- Box list ---
        box_group = QGroupBox("标注框")
        box_layout = QVBoxLayout(box_group)
        box_layout.setSpacing(8)

        self.box_list = QListWidget()
        box_layout.addWidget(self.box_list)

        box_btn_layout = QHBoxLayout()
        box_btn_layout.setSpacing(8)

        self.delete_box_btn = QPushButton("删除选中框")
        self.delete_box_btn.setObjectName("danger_btn")
        self.delete_box_btn.clicked.connect(self._delete_selected_box)
        box_btn_layout.addWidget(self.delete_box_btn)

        self.clear_btn = QPushButton("清除所有框")
        self.clear_btn.clicked.connect(self._clear_all_boxes)
        box_btn_layout.addWidget(self.clear_btn)

        box_layout.addLayout(box_btn_layout)
        right_splitter.addWidget(box_group)

        # --- Dataset info ---
        info_group = QGroupBox("数据集")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #e8e8f0;")
        info_layout.addWidget(self.info_label)

        right_splitter.addWidget(info_group)

        right_splitter.setSizes([50, 80, 250, 150, 60])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(canvas_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([700, 300])
        main_layout.addWidget(main_splitter)

        # Keyboard shortcuts
        self.screenshot_shortcut = QShortcut(QKeySequence("F12"), self)
        self.screenshot_shortcut.activated.connect(self._take_screenshot)

    # ========== Sample set management ==========

    def _populate_set_combo(self):
        """Fill the sample set combo box from SampleSetManager."""
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for ss in self.ssm.get_all():
            self.set_combo.addItem(ss.name, ss.id)
        self.set_combo.blockSignals(False)

    def select_set_by_id(self, set_id: str):
        """Programmatically select a sample set by its ID."""
        for i in range(self.set_combo.count()):
            if self.set_combo.itemData(i) == set_id:
                self.set_combo.setCurrentIndex(i)
                # Ensure state is initialized even if signal was suppressed
                # (setCurrentIndex is a no-op when index doesn't change)
                if self._current_set is None or self._current_set.id != set_id:
                    self._on_sample_set_changed(i)
                return

    def select_target_by_id(self, target_id: str):
        """Programmatically select a target by its ID in the target table."""
        set_targets = self._get_set_targets()
        for i, t in enumerate(set_targets):
            if t.id == target_id:
                self.target_table.setCurrentCell(i, 0)
                return

    def _on_sample_set_changed(self, index: int):
        """Switch to a different sample set."""
        if index < 0 or index >= self.set_combo.count():
            self._current_set = None
            return

        set_id = self.set_combo.currentData()
        new_set = self.ssm.get(set_id)
        if new_set is None:
            return

        # Save old set's session before switching
        if self._current_set is not None and self._current_set.id != new_set.id:
            self._save_session()

        self._current_set = new_set

        # Load new set's session
        self._load_session()

        # Refresh targets for this set
        self.refresh_targets()

        # Update info
        self._update_info()

    def _create_sample_set(self):
        name, ok = QInputDialog.getText(self, "新建样本集", "样本集名称:")
        if not ok or not name:
            return
        ss = self.ssm.add(name)
        self._populate_set_combo()
        self.select_set_by_id(ss.id)

    def _rename_sample_set(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        new_name, ok = QInputDialog.getText(
            self, "重命名样本集", "新名称:", text=self._current_set.name)
        if not ok or not new_name:
            return
        self.ssm.rename(self._current_set.id, new_name)
        self._populate_set_combo()
        self.select_set_by_id(self._current_set.id)
        self.sample_set_renamed.emit(self._current_set.id)

    def _delete_sample_set(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        ss = self._current_set
        targets = self.tm.get_by_set(ss.id)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除样本集 '{ss.name}' 吗？\n\n"
            f"这将删除以下所有内容：\n"
            f"  • 样本集目录: {ss.dir_path}\n"
            f"  • 包含的 {len(targets)} 个目标及其标注\n"
            f"  • 样本集中的所有图片\n\n"
            f"此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Delete targets in this set
        for t in targets:
            if t.model_path and os.path.isfile(t.model_path):
                os.remove(t.model_path)
            self.tm.remove(t.id)
        self.tm.save()

        # Delete sample set (files + index entry)
        self.ssm.remove(ss.id, delete_files=True)

        self._current_set = None
        self._populate_set_combo()
        self.target_deleted.emit()

    # ========== Target management ==========

    def refresh_targets(self):
        """Refresh the target table for the current sample set."""
        self.target_table.blockSignals(True)
        self.target_table.setRowCount(0)

        if self._current_set is None:
            self.target_table.blockSignals(False)
            self._current_target = None
            return

        set_targets = self.tm.get_by_set(self._current_set.id)
        for t in set_targets:
            row = self.target_table.rowCount()
            self.target_table.insertRow(row)
            # Color cell
            color_item = QTableWidgetItem()
            r, g, b = int(t.color[1:3], 16), int(t.color[3:5], 16), int(t.color[5:7], 16)
            color_item.setBackground(QBrush(QColor(r, g, b)))
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            self.target_table.setItem(row, 0, color_item)
            # Name cell
            name_item = QTableWidgetItem(t.name)
            name_item.setData(Qt.UserRole, t.id)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.target_table.setItem(row, 1, name_item)

        self.target_table.blockSignals(False)
        if set_targets:
            self._on_target_row_changed(0, 0, -1, -1)
        else:
            self._current_target = None
            self.target_info_label.setText("当前样本集无目标，请先创建")

    def _get_set_targets(self) -> list[Target]:
        """Get targets belonging to the current sample set."""
        if self._current_set is None:
            return []
        return self.tm.get_by_set(self._current_set.id)

    def _on_target_row_changed(self, row: int, col: int, prev_row: int, prev_col: int):
        set_targets = self._get_set_targets()
        if row < 0 or row >= len(set_targets):
            self._current_target = None
            return
        self._current_target = set_targets[row]
        self.target_info_label.setText(f"目标: {self._current_target.name}")
        self.canvas.set_draw_color(self._current_target.color)

        # Load session if not loaded yet
        if not self._image_list and self._current_set is not None:
            self._load_session()

        # Refresh annotation status for all images with new target
        self._refresh_image_list_status()

        # Reload current image to show new target's labels
        if self._current_index >= 0 and self._image_list:
            self._load_image(self._image_list[self._current_index])
            self._update_progress()

    def _refresh_image_list_status(self):
        """Update annotation status for all images based on current target."""
        for i in range(self.image_list_widget.count()):
            if i < len(self._image_list):
                item = self.image_list_widget.item(i)
                name = os.path.basename(self._image_list[i])
                status = self._get_label_status(self._image_list[i])
                if status == "selected":
                    item.setText("[已标注] " + name)
                    item.setForeground(QColor(34, 197, 94))   # green
                elif status == "other":
                    item.setText("[其他目标] " + name)
                    item.setForeground(QColor(234, 179, 8))   # yellow
                else:
                    item.setText("[未标注] " + name)
                    item.setForeground(QColor(239, 68, 68))   # red

    def _on_target_cell_double_clicked(self, row: int, col: int):
        set_targets = self._get_set_targets()
        if row < 0 or row >= len(set_targets):
            return
        target = set_targets[row]
        if col == 1:  # Name column
            new_name, ok = QInputDialog.getText(self, "重命名目标", "新名称:", text=target.name)
            if ok and new_name:
                target.name = new_name
                self.tm.save()
                self.refresh_targets()

    def _on_target_context_menu(self, pos):
        row = self.target_table.rowAt(pos.y())
        set_targets = self._get_set_targets()
        if row < 0 or row >= len(set_targets):
            return
        menu = QMenu(self)
        delete_action = menu.addAction("删除目标（彻底删除文件）")
        action = menu.exec(self.target_table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self._delete_target_at(row)

    def _delete_target_at(self, row: int):
        set_targets = self._get_set_targets()
        target = set_targets[row]
        reply1 = QMessageBox.question(
            self, "确认删除",
            f"确定要删除目标 '{target.name}' 吗？\n\n"
            f"这将删除以下所有内容：\n"
            f"  • 训练数据: {target.data_dir}\n"
            f"  • 模型文件: {target.model_path or '无'}\n"
            f"  • 模板文件: {target.template_path or '无'}\n\n"
            f"此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply1 != QMessageBox.Yes:
            return
        text, ok = QInputDialog.getText(
            self, "最终确认",
            f"请输入目标名称 '{target.name}' 以确认删除：",
        )
        if not ok or text != target.name:
            QMessageBox.information(self, "取消", "名称不匹配，删除已取消")
            return
        if target.data_dir and os.path.isdir(target.data_dir):
            shutil.rmtree(target.data_dir, ignore_errors=True)
        if target.model_path and os.path.isfile(target.model_path):
            os.remove(target.model_path)
        self.tm.remove(target.id)
        self.tm.save()
        self.refresh_targets()
        if self._current_target and self._current_target.id == target.id:
            self._current_target = None
            self.canvas.clear_boxes()
            self.box_list.clear()
        self.target_deleted.emit()

    def _create_new_target(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        name, ok = QInputDialog.getText(self, "新建目标", "目标名称:")
        if not ok or not name:
            return
        used_colors = {t.color for t in self._get_set_targets()}
        color = "#2ecc71"
        for c in TARGET_COLORS:
            if c not in used_colors:
                color = c
                break
        target = self.tm.add(name=name, set_id=self._current_set.id, color=color)
        os.makedirs(os.path.join(target.data_dir, "labels"), exist_ok=True)
        self.tm.save()
        self.refresh_targets()
        self.target_added.emit()
        set_targets = self._get_set_targets()
        idx = set_targets.index(target)
        self.target_table.setCurrentCell(idx, 0)

    def _get_current_data_dir(self) -> str:
        if self._current_target:
            return self._current_target.data_dir
        return "data"

    def _get_current_color(self) -> str:
        if self._current_target:
            return self._current_target.color
        return "#2ecc71"

    # ========== Session persistence (per sample set) ==========

    def _get_session_file(self) -> str:
        """Return the session.json path for the current sample set."""
        if self._current_set is not None:
            return self._current_set.session_file
        # Fallback: use first available set
        sets = self.ssm.get_all()
        if sets:
            return sets[0].session_file
        return os.path.join("data", "session.json")

    def _save_session(self):
        """Save session to the current sample set's session.json."""
        session_file = self._get_session_file()
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        session = {
            "image_list": self._image_list,
            "current_index": self._current_index,
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

    def _load_session(self):
        """Load session from the current sample set's session.json."""
        session_file = self._get_session_file()
        if not os.path.exists(session_file):
            self._image_list = []
            self._current_index = -1
            self.image_list_widget.clear()
            self.canvas.set_image(QPixmap())
            self.canvas.clear_boxes()
            self.box_list.clear()
            self._update_progress()
            self._update_nav_buttons()
            return
        with open(session_file, "r", encoding="utf-8") as f:
            session = json.load(f)
        paths = session.get("image_list", [])
        # Filter out files that no longer exist
        valid_paths = [p for p in paths if os.path.exists(p)]
        saved_index = session.get("current_index", -1)
        self._image_list = valid_paths
        self._current_index = -1
        self.image_list_widget.clear()
        for p in valid_paths:
            self.image_list_widget.addItem(self._create_image_list_item(p))
        self._update_progress()
        self._update_nav_buttons()
        # Restore index
        if valid_paths and 0 <= saved_index < len(valid_paths):
            self._current_index = saved_index
        elif valid_paths:
            self._current_index = 0
        # Display the restored image (if any)
        if self._current_index >= 0 and self._image_list:
            self._load_image(self._image_list[self._current_index])

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self._prev_image()
        elif event.key() == Qt.Key_D:
            self._next_image()
        elif event.key() == Qt.Key_S:
            self._save_and_next()
        else:
            super().keyPressEvent(event)

    # ========== Image list ==========

    def _create_image_list_item(self, image_path: str) -> QListWidgetItem:
        """Create a QListWidgetItem with proper label status coloring."""
        name = os.path.basename(image_path)
        status = self._get_label_status(image_path)
        if status == "selected":
            item = QListWidgetItem("[已标注] " + name)
            item.setForeground(QColor(34, 197, 94))
        elif status == "other":
            item = QListWidgetItem("[其他目标] " + name)
            item.setForeground(QColor(234, 179, 8))
        else:
            item = QListWidgetItem("[未标注] " + name)
            item.setForeground(QColor(239, 68, 68))
        return item

    def _load_image_list(self, paths: list[str]):
        """Load images into the current set's image list."""
        for p in paths:
            if p not in self._image_list:
                self._image_list.append(p)
                self.image_list_widget.addItem(self._create_image_list_item(p))
        self._update_progress()
        self._update_nav_buttons()
        self._save_session()

    def _has_label(self, image_path: str) -> bool:
        """Check if the currently selected target has a label for this image."""
        return self._get_label_status(image_path) == "selected"

    def _get_label_status(self, image_path: str) -> str:
        """Check label status: 'selected' (current target), 'other' (other targets in set), or 'none'."""
        if self._current_set is None:
            return "none"
        img_name = os.path.basename(image_path)
        name_no_ext = os.path.splitext(img_name)[0]
        for target in self._get_set_targets():
            label_path = os.path.join(target.data_dir, "labels", name_no_ext + ".txt")
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                if self._current_target and target.id == self._current_target.id:
                    return "selected"
                else:
                    return "other"
        return "none"

    def _update_image_status(self, index: int):
        if 0 <= index < len(self._image_list):
            item = self.image_list_widget.item(index)
            name = os.path.basename(self._image_list[index])
            status = self._get_label_status(self._image_list[index])
            if status == "selected":
                item.setText("[已标注] " + name)
                item.setForeground(QColor(34, 197, 94))   # green
            elif status == "other":
                item.setText("[其他目标] " + name)
                item.setForeground(QColor(234, 179, 8))   # yellow
            else:
                item.setText("[未标注] " + name)
                item.setForeground(QColor(239, 68, 68))   # red
        self._update_progress()

    def _update_progress(self):
        total = len(self._image_list)
        annotated = sum(1 for p in self._image_list if self._has_label(p))
        self.annotated_count_label.setText(f"已标注: {annotated} / {total}")
        if self._current_index >= 0:
            self.progress_label.setText(f"第 {self._current_index + 1} / {total} 张")

    def _update_nav_buttons(self):
        has_images = len(self._image_list) > 0
        has_prev = self._current_index > 0
        has_next = self._current_index < len(self._image_list) - 1
        self.prev_btn.setEnabled(has_prev)
        self.next_btn.setEnabled(has_next)
        self.save_btn.setEnabled(has_images)
        self.reannotate_btn.setEnabled(has_images and self._current_index >= 0)

    # ========== Open / copy images ==========

    def _get_current_images_dir(self) -> str:
        """Return the images directory for the current sample set."""
        if self._current_set is not None:
            return self._current_set.images_dir
        # Fallback (should not happen)
        return os.path.join("data", "shared_images")

    def _open_image(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "打开图片", "", "图片 (*.png *.jpg *.bmp *.jpeg *.webp)")
        if paths:
            copied_paths = self._copy_to_current_set(paths)
            for p in copied_paths:
                if p not in self._image_list:
                    self._image_list.append(p)
                    self.image_list_widget.addItem(self._create_image_list_item(p))
            if copied_paths:
                self._switch_to(self._image_list.index(copied_paths[0]))
            self._save_session()

    def _open_folder(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)]
            if paths:
                copied_paths = self._copy_to_current_set(paths)
                for p in copied_paths:
                    if p not in self._image_list:
                        self._image_list.append(p)
                        self.image_list_widget.addItem(self._create_image_list_item(p))
                if copied_paths:
                    self._switch_to(self._image_list.index(copied_paths[0]))
                self._save_session()
            else:
                QMessageBox.warning(self, "提示", "该文件夹中没有找到图片文件")

    def _copy_to_current_set(self, paths: list[str]) -> list[str]:
        """Copy images to current sample set's images directory and return new paths."""
        images_dir = self._get_current_images_dir()
        os.makedirs(images_dir, exist_ok=True)
        copied = []
        for src_path in paths:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(images_dir, filename)
            # If file already exists with same name, add timestamp
            if os.path.exists(dst_path) and os.path.abspath(src_path) != os.path.abspath(dst_path):
                name, ext = os.path.splitext(filename)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}{ext}"
                dst_path = os.path.join(images_dir, filename)
            # Copy if not already in directory
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)
            copied.append(dst_path)
        return copied

    def _take_screenshot(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        # Check if window capture is enabled
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

        # Save to current sample set's images directory
        images_dir = self._get_current_images_dir()
        os.makedirs(images_dir, exist_ok=True)
        tmp_path = ScreenCapture.save_screenshot(img, images_dir, "screenshot")
        if tmp_path not in self._image_list:
            self._image_list.append(tmp_path)
            item = QListWidgetItem("[未标注] " + os.path.basename(tmp_path))
            item.setForeground(QColor(239, 68, 68))
            self.image_list_widget.addItem(item)
        self._switch_to(self._image_list.index(tmp_path))

    # ========== Navigation ==========

    def _switch_to(self, index: int):
        if index < 0 or index >= len(self._image_list):
            return
        self._current_index = index
        self.image_list_widget.blockSignals(True)
        self.image_list_widget.setCurrentRow(index)
        self.image_list_widget.blockSignals(False)
        self._load_image(self._image_list[index])
        self._update_nav_buttons()
        self._update_progress()
        self._save_session()

    def _prev_image(self):
        if self._current_index > 0:
            self._switch_to(self._current_index - 1)

    def _next_image(self):
        if self._current_index < len(self._image_list) - 1:
            self._switch_to(self._current_index + 1)

    def _save_and_next(self):
        self._save_labels(silent=True)
        self._next_image()

    def _on_image_selected(self, row: int):
        if row >= 0 and row != self._current_index:
            self._switch_to(row)

    # ========== Context menu ==========

    def _on_image_list_context_menu(self, pos):
        item = self.image_list_widget.itemAt(pos)
        if item is None:
            return
        row = self.image_list_widget.row(item)
        menu = QMenu(self)
        remove_action = menu.addAction("从列表中删除")
        remove_action.triggered.connect(lambda: self._delete_image_from_list(row))
        delete_file_action = menu.addAction("彻底删除文件")
        delete_file_action.triggered.connect(lambda: self._delete_file_permanently(row))
        menu.exec(self.image_list_widget.mapToGlobal(pos))

    def _delete_image_from_list(self, row: int):
        if row < 0 or row >= len(self._image_list):
            return
        self._image_list.pop(row)
        self.image_list_widget.takeItem(row)
        self._adjust_after_removal(row)

    def _delete_file_permanently(self, row: int):
        if row < 0 or row >= len(self._image_list):
            return
        path = self._image_list[row]
        name = os.path.basename(path)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要彻底删除文件 '{name}' 吗？\n\n"
            f"此操作将删除图片和当前样本集中所有目标的标注，不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # Delete image file
        if os.path.exists(path):
            os.remove(path)
        # Delete associated label files for targets in current sample set
        img_name = os.path.basename(path)
        name_no_ext = os.path.splitext(img_name)[0]
        for target in self._get_set_targets():
            label_path = os.path.join(target.data_dir, "labels", name_no_ext + ".txt")
            if os.path.exists(label_path):
                os.remove(label_path)
        # Remove from list
        self._image_list.pop(row)
        self.image_list_widget.takeItem(row)
        self._adjust_after_removal(row)

    def _adjust_after_removal(self, removed_row: int):
        if not self._image_list:
            self._current_index = -1
            self.current_image_path = None
            self.current_image = None
            self.canvas.set_image(QPixmap())
            self.box_list.clear()
        elif self._current_index == removed_row:
            self._current_index = -1
            self._switch_to(min(removed_row, len(self._image_list) - 1))
        elif self._current_index > removed_row:
            self._current_index -= 1
        self._update_progress()
        self._update_nav_buttons()
        self._save_session()

    # ========== Image loading & labels ==========

    def _load_image(self, path: str):
        self.current_image_path = path
        self.current_image = cv2.imread(path)
        if self.current_image is None:
            return
        h, w, ch = self.current_image.shape
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qt_img))
        self.canvas.clear_boxes()
        self.box_list.clear()

        # Load labels from targets in current sample set only
        img_name = os.path.basename(path)
        name_no_ext = os.path.splitext(img_name)[0]
        for target in self._get_set_targets():
            label_path = os.path.join(target.data_dir, "labels", name_no_ext + ".txt")
            if os.path.exists(label_path):
                self._load_labels_with_color(label_path, w, h, target.color, target.name)

    def _get_label_path(self, image_path: str) -> str:
        img_name = os.path.basename(image_path)
        name_no_ext = os.path.splitext(img_name)[0]
        data_dir = self._get_current_data_dir()
        return os.path.join(data_dir, "labels", name_no_ext + ".txt")

    def _load_labels_with_color(self, label_path: str, img_w: int, img_h: int, color: str, target_name: str):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - w / 2) * img_w)
                    y1 = int((cy - h / 2) * img_h)
                    x2 = int((cx + w / 2) * img_w)
                    y2 = int((cy + h / 2) * img_h)
                    rect = QRect(x1, y1, x2 - x1, y2 - y1)
                    label = f"{target_name}:cls{cls_id}"
                    self.canvas.add_box(rect, cls_id, color, label)
                    self.box_list.addItem(f"[{target_name}] cls:{cls_id} ({x1},{y1})-({x2},{y2})")

    @Slot(QRect)
    def _on_box_drawn(self, rect: QRect):
        if self._current_target is None:
            QMessageBox.warning(self, "提示", "请先选择标注目标")
            return
        color = self._get_current_color()
        name = self._current_target.name
        self.canvas.add_box(rect, 0, color, f"{name}:cls0")
        self.box_list.addItem(f"[{name}] cls:0 ({rect.x()},{rect.y()})-({rect.x()+rect.width()},{rect.y()+rect.height()})")

    def _save_labels(self, silent: bool = False):
        if self.current_image_path is None or self.current_image is None or self._current_target is None:
            return
        h, w = self.current_image.shape[:2]

        # Only save boxes that belong to current target (by color)
        current_color = self._get_current_color()
        all_boxes = self.canvas.get_boxes()
        my_boxes = [(rect, cls_id, color, lbl) for rect, cls_id, color, lbl in all_boxes if color == current_color]

        # Create labels directory if needed
        label_dir = os.path.join(self._get_current_data_dir(), "labels")
        os.makedirs(label_dir, exist_ok=True)

        # Save label file
        label_path = self._get_label_path(self.current_image_path)
        with open(label_path, "w") as f:
            for rect, cls_id, _, _ in my_boxes:
                cx = (rect.x() + rect.width() / 2) / w
                cy = (rect.y() + rect.height() / 2) / h
                rw = rect.width() / w
                rh = rect.height() / h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}\n")

        if self._current_index >= 0:
            self._update_image_status(self._current_index)

        self._update_info()
        if not silent:
            QMessageBox.information(self, "保存成功", f"标注已保存到 {label_path}")

    def _reannotate(self):
        if self.current_image_path is None or self._current_target is None:
            return
        reply = QMessageBox.question(
            self, "重新标注",
            f"确定要清除当前图片在目标 '{self._current_target.name}' 下的所有标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            # Remove only boxes of current target's color
            current_color = self._get_current_color()
            all_boxes = self.canvas.get_boxes()
            remaining = [(r, c, co, l) for r, c, co, l in all_boxes if co != current_color]
            self.canvas.clear_boxes()
            self.box_list.clear()
            for rect, cls_id, color, label in remaining:
                self.canvas.add_box(rect, cls_id, color, label)
                self.box_list.addItem(f"[{label}] cls:{cls_id}")
            # Delete label file
            label_path = self._get_label_path(self.current_image_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            if self._current_index >= 0:
                self._update_image_status(self._current_index)

    def _clear_all_boxes(self):
        self.canvas.clear_boxes()
        self.box_list.clear()

    def _delete_selected_box(self):
        row = self.box_list.currentRow()
        if row >= 0:
            self.box_list.takeItem(row)
            boxes = self.canvas.get_boxes()
            if row < len(boxes):
                boxes.pop(row)
                self.canvas.clear_boxes()
                for rect, cls_id, color, label in boxes:
                    self.canvas.add_box(rect, cls_id, color, label)

    def _update_info(self):
        from core.trainer import Trainer
        if self._current_set is None:
            self.info_label.setText("请先选择一个样本集")
            return
        if self._current_target is None:
            self.info_label.setText(
                f"样本集: {self._current_set.name}  "
                f"图片: {len(self._image_list)}  目标: {len(self._get_set_targets())}"
            )
            return
        data_dir = self._current_target.data_dir
        images_dir = self._get_current_images_dir()
        if os.path.exists(data_dir):
            stats = Trainer.get_dataset_stats(data_dir, images_dir)
            self.info_label.setText(
                f"样本集: {self._current_set.name}  "
                f"目标: {self._current_target.name}  "
                f"图片: {stats['image_count']}  标注: {stats['label_count']}"
            )
        else:
            self.info_label.setText(f"样本集: {self._current_set.name}  数据目录不存在")
