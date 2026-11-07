"""共享对话框：添加/编辑目标。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.models import TARGET_COLORS, SampleSet

MODE_MAP = {"仅 AI": "ai", "仅 OpenCV": "opencv", "AI + OpenCV": "both"}
MODE_LABELS = {v: k for k, v in MODE_MAP.items()}


class AddTargetDialog(QDialog):
    """创建/编辑检测目标。编辑时通过 set_target() 预填。"""

    def __init__(self, sample_sets: list[SampleSet] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加目标")
        self.setMinimumWidth(420)
        self.setContentsMargins(24, 24, 24, 24)
        self._sample_sets = sample_sets or []

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("创建新检测目标")
        header.setObjectName("heading")
        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入目标名称...")
        form_layout.addRow("目标名称:", self.name_edit)

        if self._sample_sets:
            self.set_combo = QComboBox()
            for ss in self._sample_sets:
                self.set_combo.addItem(ss.name, ss.id)
            form_layout.addRow("所属样本集:", self.set_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(MODE_MAP.keys()))
        form_layout.addRow("检测模式:", self.mode_combo)

        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 999)
        self.priority_spin.setValue(0)
        form_layout.addRow("优先级 (越小越高):", self.priority_spin)

        self.color_combo = QComboBox()
        for c in TARGET_COLORS:
            self.color_combo.addItem(c)
        form_layout.addRow("颜色:", self.color_combo)

        layout.addLayout(form_layout)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        layout.addWidget(sep2)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        ok_btn = QPushButton("确定")
        ok_btn.setObjectName("primary_btn")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def set_target(self, target) -> None:
        """用已有目标预填（编辑模式）。"""
        self.setWindowTitle("编辑目标")
        self.name_edit.setText(target.name)
        self.mode_combo.setCurrentText(MODE_LABELS.get(target.detection_mode, "仅 AI"))
        self.priority_spin.setValue(target.priority)
        if target.color:
            idx = self.color_combo.findText(target.color)
            if idx >= 0:
                self.color_combo.setCurrentIndex(idx)
        if target.set_id and hasattr(self, "set_combo"):
            for i in range(self.set_combo.count()):
                if self.set_combo.itemData(i) == target.set_id:
                    self.set_combo.setCurrentIndex(i)
                    break

    def get_data(self) -> dict:
        data = {
            "name": self.name_edit.text().strip(),
            "detection_mode": MODE_MAP[self.mode_combo.currentText()],
            "priority": self.priority_spin.value(),
            "color": self.color_combo.currentText(),
        }
        if self._sample_sets and hasattr(self, "set_combo"):
            data["set_id"] = self.set_combo.currentData()
        return data
