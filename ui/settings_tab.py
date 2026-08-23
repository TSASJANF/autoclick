"""设置页：截图、窗口捕获、快捷键与训练数据参数。

相对历史版本的改动：不再独立 load_config 产生第二份配置副本，统一从
AppContext 读取、经 context.update_config 写回。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from core.windows import list_visible_windows
from ui.app_context import AppContext

logger = logging.getLogger(__name__)


class SettingsTab(QWidget):
    config_changed = Signal(dict)

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.config = ctx.config
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # 截图设置
        ss_group = QGroupBox("截图设置")
        ss_layout = QFormLayout(ss_group)
        ss_layout.setSpacing(12)
        ss_layout.setLabelAlignment(Qt.AlignRight)

        self.default_path_cb = QCheckBox("启用默认截图路径")
        self.default_path_cb.setChecked(self.config.get("screenshot_default_enabled", False))
        ss_layout.addRow(self.default_path_cb)

        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)
        self.path_edit = QLineEdit(self.config.get("screenshot_default_path", "data/screenshots"))
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        ss_layout.addRow("默认路径:", path_layout)

        self.hide_window_cb = QCheckBox("截图时隐藏程序窗口")
        self.hide_window_cb.setChecked(self.config.get("screenshot_hide_window", True))
        ss_layout.addRow(self.hide_window_cb)
        layout.addWidget(ss_group)

        # 窗口捕获设置
        win_group = QGroupBox("窗口捕获设置")
        win_layout = QFormLayout(win_group)
        win_layout.setSpacing(12)
        win_layout.setLabelAlignment(Qt.AlignRight)

        self.window_capture_cb = QCheckBox("启用窗口捕获 (只截取指定软件窗口)")
        self.window_capture_cb.setChecked(self.config.get("window_capture_enabled", False))
        self.window_capture_cb.toggled.connect(self._on_window_capture_toggled)
        win_layout.addRow(self.window_capture_cb)

        win_combo_layout = QHBoxLayout()
        win_combo_layout.setSpacing(8)
        self.window_combo = QComboBox()
        self.window_combo.setEditable(True)
        self.window_combo.setPlaceholderText("输入窗口标题关键字或从列表选择")
        self._load_window_list()
        refresh_btn = QPushButton("刷新窗口列表")
        refresh_btn.clicked.connect(self._load_window_list)
        win_combo_layout.addWidget(self.window_combo)
        win_combo_layout.addWidget(refresh_btn)
        win_layout.addRow("目标窗口:", win_combo_layout)

        self._update_window_controls_state()
        layout.addWidget(win_group)

        # 快捷键设置
        hotkey_group = QGroupBox("快捷键设置")
        hotkey_layout = QFormLayout(hotkey_group)
        hotkey_layout.setSpacing(12)
        hotkey_layout.setLabelAlignment(Qt.AlignRight)

        from PySide6.QtWidgets import QKeySequenceEdit
        self.hotkey_toggle_detection = QKeySequenceEdit()
        self.hotkey_toggle_detection.setKeySequence(
            self.config.get("hotkey_toggle_detection", "F9"))
        hotkey_layout.addRow("开始/停止检测:", self.hotkey_toggle_detection)

        hotkey_hint = QLabel("提示：点击输入框后按下想要设置的快捷键")
        hotkey_hint.setStyleSheet("color: #8888a8; font-size: 11px;")
        hotkey_layout.addRow(hotkey_hint)
        layout.addWidget(hotkey_group)

        # 训练数据设置
        neg_group = QGroupBox("训练数据设置")
        neg_layout = QFormLayout(neg_group)
        neg_layout.setSpacing(12)
        neg_layout.setLabelAlignment(Qt.AlignRight)

        ratio_frame = QFrame()
        ratio_layout = QHBoxLayout(ratio_frame)
        ratio_layout.setContentsMargins(0, 0, 0, 0)
        self.negative_ratio_slider = QSlider(Qt.Horizontal)
        self.negative_ratio_slider.setRange(10, 30)
        self.negative_ratio_slider.setValue(self.config.get("negative_sample_ratio", 20))
        self.negative_ratio_slider.valueChanged.connect(
            lambda v: self.negative_ratio_label.setText(f"{v}%"))
        ratio_layout.addWidget(self.negative_ratio_slider)

        self.negative_ratio_label = QLabel(f"{self.negative_ratio_slider.value()}%")
        self.negative_ratio_label.setStyleSheet(
            "color: #a5b4fc; font-weight: 600; min-width: 36px;")
        self.negative_ratio_label.setFixedWidth(36)
        ratio_layout.addWidget(self.negative_ratio_label)
        neg_layout.addRow("负样本比例:", ratio_frame)

        ratio_hint = QLabel("负样本 = 无目标标注的图片，可减少误报。占总训练集的比例。")
        ratio_hint.setStyleSheet("color: #8888a8; font-size: 11px;")
        neg_layout.addRow(ratio_hint)

        self.negative_min_spin = QSpinBox()
        self.negative_min_spin.setRange(1, 100)
        self.negative_min_spin.setValue(self.config.get("negative_min_positive", 5))
        neg_layout.addRow("最小正样本数:", self.negative_min_spin)

        min_hint = QLabel("正样本数量低于此值时，不添加负样本。")
        min_hint.setStyleSheet("color: #8888a8; font-size: 11px;")
        neg_layout.addRow(min_hint)
        layout.addWidget(neg_group)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        save_btn = QPushButton("保存设置")
        save_btn.setObjectName("primary_btn")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)
        layout.addStretch()

    def _on_window_capture_toggled(self, checked: bool):
        self._update_window_controls_state()
        if checked:
            # 窗口捕获模式下「隐藏自身窗口」无意义
            self.hide_window_cb.setChecked(False)

    def _update_window_controls_state(self):
        self.window_combo.setEnabled(self.window_capture_cb.isChecked())

    def _load_window_list(self):
        self.window_combo.clear()
        for _hwnd, title in list_visible_windows():
            self.window_combo.addItem(title)
        saved = self.config.get("window_capture_title", "")
        if saved:
            idx = self.window_combo.findText(saved)
            if idx >= 0:
                self.window_combo.setCurrentIndex(idx)
            else:
                self.window_combo.setCurrentText(saved)

    def _browse_path(self):
        d = QFileDialog.getExistingDirectory(self, "选择截图默认路径")
        if d:
            self.path_edit.setText(d)

    def _save(self):
        cfg = dict(self.config)
        # QKeySequenceEdit 可录入多段序列（如 "F9, F10"），只取第一段，
        # 避免解析失败导致热键静默失效（BUG-18）
        seq = self.hotkey_toggle_detection.keySequence()
        hotkey = seq[0].toString() if seq.count() > 0 else ""
        cfg.update({
            "screenshot_default_enabled": self.default_path_cb.isChecked(),
            "screenshot_default_path": self.path_edit.text(),
            "screenshot_hide_window": self.hide_window_cb.isChecked(),
            "window_capture_enabled": self.window_capture_cb.isChecked(),
            "window_capture_title": self.window_combo.currentText(),
            "hotkey_toggle_detection": hotkey,
            "negative_sample_ratio": self.negative_ratio_slider.value(),
            "negative_min_positive": self.negative_min_spin.value(),
        })
        self.config = cfg
        self.ctx.update_config(cfg)
        self.config_changed.emit(cfg)
        QMessageBox.information(self, "保存成功", "设置已保存")
