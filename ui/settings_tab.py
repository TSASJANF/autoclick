import os
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QCheckBox, QFileDialog,
    QMessageBox, QComboBox, QFrame, QKeySequenceEdit,
    QSlider, QSpinBox,
)

from core.config import load_config, save_config
from core.capture import ScreenCapture


class SettingsTab(QWidget):
    config_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.config = load_config()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Screenshot settings
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

        # Window capture settings
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

        # Hotkey settings
        hotkey_group = QGroupBox("快捷键设置")
        hotkey_layout = QFormLayout(hotkey_group)
        hotkey_layout.setSpacing(12)
        hotkey_layout.setLabelAlignment(Qt.AlignRight)

        self.hotkey_toggle_detection = QKeySequenceEdit()
        self.hotkey_toggle_detection.setKeySequence(self.config.get("hotkey_toggle_detection", "F9"))
        hotkey_layout.addRow("开始/停止检测:", self.hotkey_toggle_detection)

        hotkey_hint = QLabel("提示：点击输入框后按下想要设置的快捷键")
        hotkey_hint.setStyleSheet("color: #8888a8; font-size: 11px;")
        hotkey_layout.addRow(hotkey_hint)

        layout.addWidget(hotkey_group)

        # Negative sample settings
        neg_group = QGroupBox("训练数据设置")
        neg_layout = QFormLayout(neg_group)
        neg_layout.setSpacing(12)
        neg_layout.setLabelAlignment(Qt.AlignRight)

        # Ratio slider
        ratio_frame = QFrame()
        ratio_layout = QHBoxLayout(ratio_frame)
        ratio_layout.setContentsMargins(0, 0, 0, 0)

        self.negative_ratio_slider = QSlider(Qt.Horizontal)
        self.negative_ratio_slider.setRange(10, 30)
        self.negative_ratio_slider.setValue(self.config.get("negative_sample_ratio", 20))
        self.negative_ratio_slider.valueChanged.connect(self._on_ratio_changed)
        ratio_layout.addWidget(self.negative_ratio_slider)

        self.negative_ratio_label = QLabel(f"{self.negative_ratio_slider.value()}%")
        self.negative_ratio_label.setStyleSheet("color: #a5b4fc; font-weight: 600; min-width: 36px;")
        self.negative_ratio_label.setFixedWidth(36)
        ratio_layout.addWidget(self.negative_ratio_label)

        neg_layout.addRow("负样本比例:", ratio_frame)

        ratio_hint = QLabel("负样本 = 无目标标注的图片，可减少误报。占总训练集的比例。")
        ratio_hint.setStyleSheet("color: #8888a8; font-size: 11px;")
        neg_layout.addRow(ratio_hint)

        # Min positive threshold
        self.negative_min_spin = QSpinBox()
        self.negative_min_spin.setRange(1, 100)
        self.negative_min_spin.setValue(self.config.get("negative_min_positive", 5))
        neg_layout.addRow("最小正样本数:", self.negative_min_spin)

        min_hint = QLabel("正样本数量低于此值时，不添加负样本。")
        min_hint.setStyleSheet("color: #8888a8; font-size: 11px;")
        neg_layout.addRow(min_hint)

        layout.addWidget(neg_group)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Save button
        save_btn = QPushButton("保存设置")
        save_btn.setObjectName("primary_btn")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        layout.addStretch()

    def _on_window_capture_toggled(self, checked: bool):
        self._update_window_controls_state()
        if checked:
            # When window capture is enabled, hide_window is irrelevant
            self.hide_window_cb.setChecked(False)

    def _on_ratio_changed(self, value: int):
        self.negative_ratio_label.setText(f"{value}%")

    def _update_window_controls_state(self):
        enabled = self.window_capture_cb.isChecked()
        self.window_combo.setEnabled(enabled)

    def _load_window_list(self):
        self.window_combo.clear()
        windows = ScreenCapture.list_visible_windows()
        for hwnd, title in windows:
            self.window_combo.addItem(title, hwnd)
        # Restore saved value
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
        self.config["screenshot_default_enabled"] = self.default_path_cb.isChecked()
        self.config["screenshot_default_path"] = self.path_edit.text()
        self.config["screenshot_hide_window"] = self.hide_window_cb.isChecked()
        self.config["window_capture_enabled"] = self.window_capture_cb.isChecked()
        self.config["window_capture_title"] = self.window_combo.currentText()
        self.config["hotkey_toggle_detection"] = self.hotkey_toggle_detection.keySequence().toString()
        self.config["negative_sample_ratio"] = self.negative_ratio_slider.value()
        self.config["negative_min_positive"] = self.negative_min_spin.value()
        save_config(self.config)
        self.config_changed.emit(self.config)
        QMessageBox.information(self, "保存成功", "设置已保存")

    def get_config(self) -> dict:
        return self.config
