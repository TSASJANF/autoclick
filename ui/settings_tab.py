"""设置页：截图、窗口捕获、快捷键与训练数据参数。

布局：ScrollArea 卡片式布局。配置读写走 AppContext，逻辑不变。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CardWidget, CheckBox, FluentIcon, LineEdit, PrimaryPushButton,
    PushButton, ScrollArea, Slider, SpinBox, SubtitleLabel, SwitchButton,
)
from PySide6.QtWidgets import QKeySequenceEdit

from core.windows import list_visible_windows
from ui.app_context import AppContext

logger = logging.getLogger(__name__)


class SettingsTab(ScrollArea):
    config_changed = Signal(dict)

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsInterface")
        self.ctx = ctx
        self.config = ctx.config
        self._init_ui()

    def _init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        self.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(8)

        title = QLabel("设置")
        title.setObjectName("heading")
        root.addWidget(title)
        root.addWidget(QLabel("截图与窗口捕获、全局快捷键、训练数据参数"))

        root.addWidget(self._build_screenshot_card())
        root.addWidget(self._build_window_card())
        root.addWidget(self._build_hotkey_card())
        root.addWidget(self._build_training_card())
        root.addWidget(self._build_save_row())
        root.addStretch()

    def _card(self, title: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)
        if title:
            lay.addWidget(SubtitleLabel(title))
        return card, lay

    def _label(self, text: str) -> BodyLabel:
        return BodyLabel(text)

    def _build_screenshot_card(self) -> CardWidget:
        card, lay = self._card("截图设置")
        self.default_path_cb = CheckBox("启用默认截图路径")
        self.default_path_cb.setChecked(self.config.get("screenshot_default_enabled", False))
        lay.addWidget(self.default_path_cb)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        path_row.addWidget(self._label("默认路径"))
        self.path_edit = LineEdit()
        self.path_edit.setText(self.config.get("screenshot_default_path", "data/screenshots"))
        browse_btn = PushButton(FluentIcon.FOLDER, "浏览")
        browse_btn.clicked.connect(self._browse_path)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_btn)
        lay.addLayout(path_row)

        self.hide_window_cb = CheckBox("截图时隐藏程序窗口")
        self.hide_window_cb.setChecked(self.config.get("screenshot_hide_window", True))
        lay.addWidget(self.hide_window_cb)
        return card

    def _build_window_card(self) -> CardWidget:
        card, lay = self._card("窗口捕获设置")
        self.window_capture_cb = SwitchButton("启用窗口捕获（只截取指定软件窗口）")
        self.window_capture_cb.setChecked(self.config.get("window_capture_enabled", False))
        self.window_capture_cb.checkedChanged.connect(self._on_window_capture_toggled)
        lay.addWidget(self.window_capture_cb)

        lay.addWidget(self._label("目标窗口"))
        win_row = QHBoxLayout()
        win_row.setSpacing(8)
        self.window_combo = QComboBox()
        self.window_combo.setEditable(True)
        self.window_combo.setPlaceholderText("输入窗口标题关键字或从列表选择")
        self._load_window_list()
        refresh_btn = PushButton(FluentIcon.SYNC, "刷新窗口列表")
        refresh_btn.clicked.connect(self._load_window_list)
        win_row.addWidget(self.window_combo, 1)
        win_row.addWidget(refresh_btn)
        lay.addLayout(win_row)

        self._update_window_controls_state()
        return card

    def _build_hotkey_card(self) -> CardWidget:
        card, lay = self._card("快捷键设置")
        hk_row = QHBoxLayout()
        hk_row.setSpacing(8)
        hk_row.addWidget(self._label("开始/停止检测"))
        self.hotkey_toggle_detection = QKeySequenceEdit()
        self.hotkey_toggle_detection.setKeySequence(
            self.config.get("hotkey_toggle_detection", "F9"))
        self.hotkey_toggle_detection.setMaximumWidth(200)
        hk_row.addWidget(self.hotkey_toggle_detection)
        hk_row.addStretch()
        lay.addLayout(hk_row)
        lay.addWidget(QLabel("提示：点击输入框后按下想要设置的快捷键"))
        return card

    def _build_training_card(self) -> CardWidget:
        card, lay = self._card("训练数据设置")
        ratio_row = QHBoxLayout()
        ratio_row.setSpacing(8)
        ratio_row.addWidget(self._label("负样本比例"))
        self.negative_ratio_slider = Slider(Qt.Horizontal)
        self.negative_ratio_slider.setRange(10, 30)
        self.negative_ratio_slider.setValue(self.config.get("negative_sample_ratio", 20))
        self.negative_ratio_label = BodyLabel(f"{self.negative_ratio_slider.value()}%")
        self.negative_ratio_label.setMinimumWidth(40)
        self.negative_ratio_slider.valueChanged.connect(
            lambda v: self.negative_ratio_label.setText(f"{v}%"))
        ratio_row.addWidget(self.negative_ratio_slider, 1)
        ratio_row.addWidget(self.negative_ratio_label)
        lay.addLayout(ratio_row)
        lay.addWidget(QLabel("负样本 = 无目标标注的图片，可减少误报，占总训练集比例"))

        min_row = QHBoxLayout()
        min_row.setSpacing(8)
        min_row.addWidget(self._label("最小正样本数"))
        self.negative_min_spin = SpinBox()
        self.negative_min_spin.setRange(1, 100)
        self.negative_min_spin.setValue(self.config.get("negative_min_positive", 5))
        self.negative_min_spin.setMaximumWidth(110)
        min_row.addWidget(self.negative_min_spin)
        min_row.addStretch()
        lay.addLayout(min_row)
        lay.addWidget(QLabel("正样本数量低于此值时，不添加负样本"))
        return card

    def _build_save_row(self) -> CardWidget:
        card, lay = self._card("")
        save_btn = PrimaryPushButton(FluentIcon.SAVE, "保存设置")
        save_btn.clicked.connect(self._save)
        lay.addWidget(save_btn)
        return card

    # ========== 交互 ==========

    def _on_window_capture_toggled(self, checked: bool):
        self._update_window_controls_state()
        if checked:
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
