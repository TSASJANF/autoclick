"""标注页：样本集管理、目标标注、图片导航与会话记忆。

布局：ScrollArea 内的卡片式布局（样本集/图片/标注目标/图片列表/标注框/
数据集六张卡片），窗口过小时纵向滚动。标注归属、缓存等业务逻辑不变。
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SubtitleLabel,
    TableWidget,
)

from core import annotations as anno
from core.image_io import imread
from core.models import SampleSet, Target
from core.sample_set_manager import SampleSetManager
from core.target_manager import TargetManager
from ui.app_context import AppContext
from ui.widgets.canvas import ColorDelegate, ImageCanvas
from ui.widgets.image_utils import bgr_to_pixmap

logger = logging.getLogger(__name__)

STATUS_TEXT = {
    "selected": ("[已标注] ", QColor(34, 197, 94)),
    "other": ("[其他目标] ", QColor(234, 179, 8)),
    "none": ("[未标注] ", QColor(239, 68, 68)),
}


def _status_item(image_path: str, status: str) -> QListWidgetItem:
    prefix, color = STATUS_TEXT[status]
    item = QListWidgetItem(prefix + os.path.basename(image_path))
    item.setForeground(color)
    return item


class AnnotateTab(QWidget):
    sample_set_renamed = Signal(str)

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.setObjectName("annotateInterface")
        self.ctx = ctx
        self.tm: TargetManager = ctx.tm
        self.ssm: SampleSetManager = ctx.ssm
        self.current_image_path: str | None = None
        self.current_image = None
        self._image_list: list[str] = []
        self._current_index = -1
        self._current_target: Target | None = None
        self._current_set: SampleSet | None = None
        self._stems_cache: dict[str, set[str]] | None = None
        self._init_ui()
        self._populate_set_combo()
        if self.set_combo.count() > 0:
            self.set_combo.setCurrentIndex(0)
            if self._current_set is None:
                self._on_sample_set_changed(0)

    # ========== UI ==========

    def _init_ui(self):
        # 双栏：左画布，右侧可滚动面板 —— 看图与操作同屏，更紧凑
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(10)

        root.addWidget(self._build_canvas_card(), 3)

        right_scroll = ScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_content = QWidget()
        right_scroll.setWidget(right_content)
        right_lay = QVBoxLayout(right_content)
        right_lay.setContentsMargins(0, 0, 4, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(self._build_set_card())
        right_lay.addWidget(self._build_target_card())
        right_lay.addWidget(self._build_image_list_card())
        right_lay.addWidget(self._build_box_card())
        right_lay.addWidget(self._build_info_card())
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

    def _build_set_card(self) -> CardWidget:
        card, lay = self._card("样本集")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.set_combo = QComboBox()
        self.set_combo.currentIndexChanged.connect(self._on_sample_set_changed)
        row.addWidget(self.set_combo, 1)
        self.add_set_btn = PushButton(FluentIcon.ADD, "新建")
        self.add_set_btn.clicked.connect(self._create_sample_set)
        row.addWidget(self.add_set_btn)
        self.rename_set_btn = PushButton("重命名")
        self.rename_set_btn.clicked.connect(self._rename_sample_set)
        row.addWidget(self.rename_set_btn)
        self.delete_set_btn = PushButton(FluentIcon.DELETE, "删除")
        self.delete_set_btn.clicked.connect(self._delete_sample_set)
        row.addWidget(self.delete_set_btn)
        lay.addLayout(row)
        return card

    def _build_canvas_card(self) -> CardWidget:
        card, lay = self._card("图片")
        self.canvas = ImageCanvas()
        self.canvas.region_drawn.connect(self._on_box_drawn)
        lay.addWidget(self.canvas)

        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.prev_btn = PushButton(FluentIcon.LEFT_ARROW, "上一张 (A)")
        self.prev_btn.clicked.connect(self._prev_image)
        self.prev_btn.setEnabled(False)
        nav.addWidget(self.prev_btn)
        self.save_btn = QPushButton("保存并下一张 (S)")
        self.save_btn.setObjectName("success_btn")
        self.save_btn.clicked.connect(self._save_and_next)
        self.save_btn.setEnabled(False)
        nav.addWidget(self.save_btn)
        self.next_btn = PushButton(FluentIcon.RIGHT_ARROW, "下一张 (D)")
        self.next_btn.clicked.connect(self._next_image)
        self.next_btn.setEnabled(False)
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

        action = QHBoxLayout()
        action.setSpacing(6)
        self.open_btn = PushButton(FluentIcon.FOLDER, "打开图片")
        self.open_btn.clicked.connect(self._open_image)
        action.addWidget(self.open_btn)
        self.open_folder_btn = PushButton(FluentIcon.FOLDER, "打开文件夹")
        self.open_folder_btn.clicked.connect(self._open_folder)
        action.addWidget(self.open_folder_btn)
        self.screenshot_btn = PushButton(FluentIcon.CAMERA, "截图")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        action.addWidget(self.screenshot_btn)
        self.reannotate_btn = PushButton("重新标注")
        self.reannotate_btn.setToolTip("清除当前图片在当前目标下的标注，重新开始")
        self.reannotate_btn.clicked.connect(self._reannotate)
        self.reannotate_btn.setEnabled(False)
        action.addWidget(self.reannotate_btn)
        action.addStretch()
        lay.addLayout(action)

        self.progress_label = BodyLabel("未加载图片")
        lay.addWidget(self.progress_label)
        return card

    def _build_target_card(self) -> CardWidget:
        card, lay = self._card("当前标注目标")
        self.target_table = TableWidget()
        self.target_table.setColumnCount(2)
        self.target_table.setHorizontalHeaderLabels(["颜色", "名称"])
        self.target_table.setItemDelegateForColumn(0, ColorDelegate(self.target_table))
        header = self.target_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.target_table.setMaximumHeight(220)
        self.target_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.target_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.target_table.currentCellChanged.connect(self._on_target_row_changed)
        self.target_table.cellDoubleClicked.connect(self._on_target_cell_double_clicked)
        self.target_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.target_table.customContextMenuRequested.connect(self._on_target_context_menu)
        lay.addWidget(self.target_table)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.target_info_label = BodyLabel("选择一个目标进行标注")
        row.addWidget(self.target_info_label, 1)
        add_target_btn = PrimaryPushButton(FluentIcon.ADD, "增加")
        add_target_btn.clicked.connect(self._create_new_target)
        row.addWidget(add_target_btn)
        lay.addLayout(row)
        return card

    def _build_image_list_card(self) -> CardWidget:
        card, lay = self._card("图片列表")
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        self.image_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_list_widget.customContextMenuRequested.connect(self._on_image_list_context_menu)
        self.image_list_widget.setMinimumHeight(140)
        lay.addWidget(self.image_list_widget)
        self.annotated_count_label = BodyLabel("已标注: 0 / 0")
        lay.addWidget(self.annotated_count_label)
        return card

    def _build_box_card(self) -> CardWidget:
        card, lay = self._card("标注框")
        self.box_list = QListWidget()
        self.box_list.setMaximumHeight(140)
        lay.addWidget(self.box_list)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.delete_box_btn = QPushButton("删除选中框")
        self.delete_box_btn.setObjectName("danger_btn")
        self.delete_box_btn.clicked.connect(self._delete_selected_box)
        row.addWidget(self.delete_box_btn)
        self.clear_btn = PushButton("清除所有框")
        self.clear_btn.clicked.connect(self._clear_all_boxes)
        row.addWidget(self.clear_btn)
        row.addStretch()
        lay.addLayout(row)
        return card

    def _build_info_card(self) -> CardWidget:
        card, lay = self._card("数据集")
        self.info_label = BodyLabel()
        lay.addWidget(self.info_label)
        return card

    # ========== 样本集管理 ==========

    def _populate_set_combo(self):
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for ss in self.ssm.get_all():
            self.set_combo.addItem(ss.name, ss.id)
        self.set_combo.blockSignals(False)

    def select_set_by_id(self, set_id: str):
        for i in range(self.set_combo.count()):
            if self.set_combo.itemData(i) == set_id:
                self.set_combo.setCurrentIndex(i)
                if self._current_set is None or self._current_set.id != set_id:
                    self._on_sample_set_changed(i)
                return

    def select_target_by_id(self, target_id: str):
        for i, t in enumerate(self._get_set_targets()):
            if t.id == target_id:
                self.target_table.setCurrentCell(i, 0)
                return

    def _on_sample_set_changed(self, index: int):
        if index < 0 or index >= self.set_combo.count():
            self._current_set = None
            return
        set_id = self.set_combo.currentData()
        new_set = self.ssm.get(set_id)
        if new_set is None:
            return
        if self._current_set is not None and self._current_set.id != new_set.id:
            self._save_session()
        self._current_set = new_set
        self._invalidate_stems_cache()
        self._load_session()
        self.refresh_targets()
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
        set_root = self.ssm.set_root(ss.id)
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除样本集 '{ss.name}' 吗？\n\n"
            f"这将删除以下所有内容：\n"
            f"  • 样本集目录: {set_root}\n"
            f"  • 包含的 {len(targets)} 个目标及其标注\n"
            f"  • 样本集中的所有图片\n\n"
            f"此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for t in targets:
            if t.model_path and os.path.isfile(t.model_path):
                try:
                    os.remove(t.model_path)
                except OSError:
                    logger.warning("模型文件删除失败: %s", t.model_path)
            self.tm.remove(t.id)
        self.tm.save()
        self.ssm.remove(ss.id, delete_files=True)
        self._current_set = None
        self._populate_set_combo()
        self.ctx.hub.targets_changed.emit()

    # ========== 目标管理 ==========

    def _get_set_targets(self) -> list[Target]:
        if self._current_set is None:
            return []
        return self.tm.get_by_set(self._current_set.id)

    def refresh_targets(self):
        self._invalidate_stems_cache()
        self.target_table.blockSignals(True)
        self.target_table.setRowCount(0)
        if self._current_set is not None:
            set_targets = self._get_set_targets()
            for t in set_targets:
                row = self.target_table.rowCount()
                self.target_table.insertRow(row)
                from ui.widgets.image_utils import hex_to_rgb
                color_item = QTableWidgetItem()
                color_item.setBackground(QColor(*hex_to_rgb(t.color)))
                color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
                self.target_table.setItem(row, 0, color_item)
                name_item = QTableWidgetItem(t.name)
                name_item.setData(Qt.UserRole, t.id)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.target_table.setItem(row, 1, name_item)
        self.target_table.blockSignals(False)

        if self._current_set is None:
            self._current_target = None
            return
        set_targets = self._get_set_targets()
        if set_targets:
            self._on_target_row_changed(0, 0, -1, -1)
        else:
            self._current_target = None
            self.target_info_label.setText("当前样本集无目标，请先创建")

    def _on_target_row_changed(self, row: int, col: int, prev_row: int, prev_col: int):
        set_targets = self._get_set_targets()
        if row < 0 or row >= len(set_targets):
            self._current_target = None
            return
        self._current_target = set_targets[row]
        self.target_info_label.setText(f"目标: {self._current_target.name}")
        self.canvas.set_draw_color(self._current_target.color)

        if not self._image_list and self._current_set is not None:
            self._load_session()

        self._refresh_image_list_status()
        if self._current_index >= 0 and self._image_list:
            self._load_image(self._image_list[self._current_index])
            self._update_progress()

    def _refresh_image_list_status(self):
        status_map = self._compute_status_map()
        for i in range(self.image_list_widget.count()):
            if i < len(self._image_list):
                item = self.image_list_widget.item(i)
                prefix, color = STATUS_TEXT[status_map[self._image_list[i]]]
                item.setText(prefix + os.path.basename(self._image_list[i]))
                item.setForeground(color)

    def _get_stems_map(self) -> dict[str, set[str]]:
        if self._stems_cache is None:
            self._stems_cache = anno.labeled_stems_map(
                {t.id: t.data_dir for t in self._get_set_targets()})
        return self._stems_cache

    def _invalidate_stems_cache(self):
        self._stems_cache = None

    def _compute_status_map(self) -> dict[str, str]:
        current_id = self._current_target.id if self._current_target else ""
        return anno.status_from_stems(self._get_stems_map(), self._image_list, current_id)

    def _on_target_cell_double_clicked(self, row: int, col: int):
        set_targets = self._get_set_targets()
        if row < 0 or row >= len(set_targets):
            return
        target = set_targets[row]
        if col == 1:
            new_name, ok = QInputDialog.getText(self, "重命名目标", "新名称:", text=target.name)
            if ok and new_name:
                target.name = new_name
                self.tm.save()
                self.refresh_targets()

    def _on_target_context_menu(self, pos):
        row = self.target_table.rowAt(pos.y())
        if row < 0 or row >= len(self._get_set_targets()):
            return
        menu = QMenu(self)
        delete_action = menu.addAction("删除目标（彻底删除文件）")
        action = menu.exec(self.target_table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self._delete_target_at(row)

    def _delete_target_at(self, row: int):
        target = self._get_set_targets()[row]
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
        self._invalidate_stems_cache()
        self.refresh_targets()
        if self._current_target and self._current_target.id == target.id:
            self._current_target = None
            self.canvas.set_boxes([])
            self.box_list.clear()
        self.ctx.hub.targets_changed.emit()

    def _create_new_target(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        name, ok = QInputDialog.getText(self, "新建目标", "目标名称:")
        if not ok or not name:
            return
        target = self.tm.add(name=name, set_id=self._current_set.id,
                             color=self.tm.next_color())
        os.makedirs(os.path.join(target.data_dir, "labels"), exist_ok=True)
        self.tm.save()
        self._invalidate_stems_cache()
        self.refresh_targets()
        self.ctx.hub.targets_changed.emit()
        idx = self._get_set_targets().index(target)
        self.target_table.setCurrentCell(idx, 0)

    # ========== 会话持久化 ==========

    def _get_session_file(self) -> str:
        if self._current_set is not None:
            return self.ssm.session_file(self._current_set.id)
        sets = self.ssm.get_all()
        if sets:
            return self.ssm.session_file(sets[0].id)
        return os.path.join("data", "session.json")

    def _save_session(self):
        session = {"image_list": self._image_list, "current_index": self._current_index}
        session_file = self._get_session_file()
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        from core import json_store
        json_store.write_json(session_file, session)

    def _load_session(self):
        from core import json_store
        session = json_store.read_json(self._get_session_file(), {})
        paths = [p for p in session.get("image_list", []) if os.path.exists(p)]
        saved_index = session.get("current_index", -1)
        self._image_list = paths
        self._current_index = -1

        status_map = self._compute_status_map()
        self.image_list_widget.clear()
        for p in paths:
            self.image_list_widget.addItem(_status_item(p, status_map[p]))
        self._update_progress()
        self._update_nav_buttons()
        if paths and 0 <= saved_index < len(paths):
            self._current_index = saved_index
        elif paths:
            self._current_index = 0
        if self._current_index >= 0 and self._image_list:
            self._load_image(self._image_list[self._current_index])

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.NoModifier:
            if event.key() == Qt.Key_A:
                self._prev_image()
                return
            if event.key() == Qt.Key_D:
                self._next_image()
                return
            if event.key() == Qt.Key_S:
                self._save_and_next()
                return
        super().keyPressEvent(event)

    # ========== 图片载入与列表 ==========

    def _get_current_images_dir(self) -> str:
        if self._current_set is not None:
            return self.ssm.images_dir(self._current_set.id)
        return os.path.join("data", "shared_images")

    def _add_images(self, copied_paths: list[str]):
        status_map = self._compute_status_map()
        for p in copied_paths:
            if p not in self._image_list:
                self._image_list.append(p)
                self.image_list_widget.addItem(_status_item(p, status_map.get(p, "none")))
        self._update_progress()
        self._update_nav_buttons()

    def _open_image(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "打开图片", "", "图片 (*.png *.jpg *.bmp *.jpeg *.webp)")
        if paths:
            copied = self._copy_to_current_set(paths)
            self._add_images(copied)
            if copied:
                self._switch_to(self._image_list.index(copied[0]))
            self._save_session()

    def _open_folder(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if os.path.splitext(f)[1].lower() in anno.IMAGE_EXTENSIONS]
        if not paths:
            QMessageBox.warning(self, "提示", "该文件夹中没有找到图片文件")
            return
        copied = self._copy_to_current_set(paths)
        self._add_images(copied)
        if copied:
            self._switch_to(self._image_list.index(copied[0]))
        self._save_session()

    def _copy_to_current_set(self, paths: list[str]) -> list[str]:
        images_dir = self._get_current_images_dir()
        os.makedirs(images_dir, exist_ok=True)
        copied = []
        for src_path in paths:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(images_dir, filename)
            if os.path.exists(dst_path) and os.path.abspath(src_path) != os.path.abspath(dst_path):
                stem, ext = os.path.splitext(filename)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dst_path = os.path.join(images_dir, f"{stem}_{timestamp}{ext}")
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.copy2(src_path, dst_path)
            copied.append(dst_path)
        return copied

    def _take_screenshot(self):
        if self._current_set is None:
            QMessageBox.warning(self, "提示", "请先选择一个样本集")
            return
        try:
            img = self.ctx.capture.capture_by_config(self.ctx.config)
        except Exception as e:
            logger.exception("截图失败")
            QMessageBox.warning(self, "错误", f"截图失败: {e}")
            return
        images_dir = self._get_current_images_dir()
        from core.capture import ScreenCapture
        saved_path = ScreenCapture.save_screenshot(img, images_dir, "screenshot")
        self._add_images([saved_path])
        self._switch_to(self._image_list.index(saved_path))

    # ========== 导航 ==========

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

    def _on_image_list_context_menu(self, pos):
        item = self.image_list_widget.itemAt(pos)
        if item is None:
            return
        row = self.image_list_widget.row(item)
        menu = QMenu(self)
        menu.addAction("从列表中删除", lambda: self._delete_image_from_list(row))
        menu.addAction("彻底删除文件", lambda: self._delete_file_permanently(row))
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
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要彻底删除文件 '{os.path.basename(path)}' 吗？\n\n"
            f"此操作将删除图片和当前样本集中所有目标的标注，不可撤销！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        if os.path.exists(path):
            os.remove(path)
        for target in self._get_set_targets():
            label_path = anno.label_path_for(target.data_dir, path)
            if os.path.exists(label_path):
                os.remove(label_path)
        self._invalidate_stems_cache()
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

    # ========== 图像与标注 ==========

    def _load_image(self, path: str):
        self.current_image_path = path
        self.current_image = imread(path)
        if self.current_image is None:
            logger.warning("图片读取失败: %s", path)
            return
        h, w = self.current_image.shape[:2]
        self.canvas.set_image(bgr_to_pixmap(self.current_image))
        self.box_list.clear()

        boxes: list[anno.Box] = []
        for target in self._get_set_targets():
            label_path = anno.label_path_for(target.data_dir, path)
            boxes.extend(anno.read_labels(
                label_path, w, h,
                owner_target_id=target.id, color=target.color,
                label_prefix=target.name))
        self.canvas.set_boxes(boxes)
        self._sync_box_list()

    def _sync_box_list(self):
        self.box_list.clear()
        for b in self.canvas.get_boxes():
            self.box_list.addItem(
                f"[{b.label or b.owner_target_id}] cls:{b.cls_id} "
                f"({b.x1},{b.y1})-({b.x2},{b.y2})")

    def _on_box_drawn(self, rect: QRect):
        if self._current_target is None:
            QMessageBox.warning(self, "提示", "请先选择标注目标")
            return
        t = self._current_target
        self.canvas.add_box(anno.Box(
            rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height(),
            cls_id=0, owner_target_id=t.id, color=t.color, label=f"{t.name}:cls0"))
        self._sync_box_list()

    def _save_labels(self, silent: bool = False):
        if (self.current_image_path is None or self.current_image is None
                or self._current_target is None):
            return
        h, w = self.current_image.shape[:2]
        t = self._current_target
        my_boxes = [b for b in self.canvas.get_boxes()
                    if b.owner_target_id == t.id]
        label_path = anno.label_path_for(t.data_dir, self.current_image_path)
        anno.write_labels(label_path, my_boxes, w, h)
        self._invalidate_stems_cache()

        if self._current_index >= 0:
            self._refresh_image_list_status()
            self._update_progress()
        self._update_info()
        if not silent:
            QMessageBox.information(self, "保存成功", f"标注已保存到 {label_path}")

    def _reannotate(self):
        if self.current_image_path is None or self._current_target is None:
            return
        reply = QMessageBox.question(
            self, "重新标注",
            f"确定要清除当前图片在目标 '{self._current_target.name}' 下的所有标注吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        t = self._current_target
        remaining = [b for b in self.canvas.get_boxes() if b.owner_target_id != t.id]
        self.canvas.set_boxes(remaining)
        self._sync_box_list()
        label_path = anno.label_path_for(t.data_dir, self.current_image_path)
        if os.path.exists(label_path):
            os.remove(label_path)
        self._invalidate_stems_cache()
        if self._current_index >= 0:
            self._refresh_image_list_status()
            self._update_progress()

    def _clear_all_boxes(self):
        self.canvas.set_boxes([])
        self.box_list.clear()

    def _delete_selected_box(self):
        row = self.box_list.currentRow()
        if row < 0:
            return
        boxes = self.canvas.get_boxes()
        if row < len(boxes):
            boxes.pop(row)
            self.canvas.set_boxes(boxes)
            self._sync_box_list()
        else:
            self.box_list.takeItem(row)

    # ========== 进度与信息 ==========

    def _update_progress(self):
        total = len(self._image_list)
        status_map = self._compute_status_map()
        annotated = sum(1 for p in self._image_list if status_map[p] == "selected")
        self.annotated_count_label.setText(f"已标注: {annotated} / {total}")
        if self._current_index >= 0:
            self.progress_label.setText(f"第 {self._current_index + 1} / {total} 张")

    def _update_nav_buttons(self):
        has_images = len(self._image_list) > 0
        self.prev_btn.setEnabled(self._current_index > 0)
        self.next_btn.setEnabled(self._current_index < len(self._image_list) - 1)
        self.save_btn.setEnabled(has_images)
        self.reannotate_btn.setEnabled(has_images and self._current_index >= 0)

    def _update_info(self):
        from core.dataset import get_dataset_stats
        if self._current_set is None:
            self.info_label.setText("请先选择一个样本集")
            return
        if self._current_target is None:
            self.info_label.setText(
                f"样本集: {self._current_set.name}  "
                f"图片: {len(self._image_list)}  目标: {len(self._get_set_targets())}")
            return
        data_dir = self._current_target.data_dir
        if os.path.exists(data_dir):
            stats = get_dataset_stats(data_dir, self._get_current_images_dir())
            self.info_label.setText(
                f"样本集: {self._current_set.name}  "
                f"目标: {self._current_target.name}  "
                f"图片: {stats['image_count']}  标注: {stats['label_count']}")
        else:
            self.info_label.setText(f"样本集: {self._current_set.name}  数据目录不存在")
