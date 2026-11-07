from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FloatingWindow(QWidget):
    """Compact floating window for detection mode."""

    stop_clicked = Signal()
    start_clicked = Signal()
    expand_clicked = Signal()
    settings_changed = Signal(dict)  # emits changed settings

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragging = False
        self._drag_pos = QPoint(-1, -1)
        self._detecting = False
        self._sync_connected = False
        self._init_ui()

    def _init_ui(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setFixedSize(280, 250)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Main container with dark styling
        container = QFrame()
        container.setObjectName("floating_container")
        container.setStyleSheet("""
            #floating_container {
                background-color: rgba(15, 15, 26, 230);
                border: 1px solid #6366f1;
                border-radius: 12px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Title bar with drag and buttons
        title_bar = QHBoxLayout()
        title_bar.setSpacing(6)

        self.title_label = QLabel("悬浮窗")
        self.title_label.setStyleSheet("color: #a5b4fc; font-size: 12px; font-weight: 600;")
        title_bar.addWidget(self.title_label)

        title_bar.addStretch()

        expand_btn = QPushButton("放大")
        expand_btn.setFixedSize(40, 22)
        expand_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #818cf8; }
        """)
        expand_btn.clicked.connect(self.expand_clicked.emit)
        title_bar.addWidget(expand_btn)

        self.action_btn = QPushButton("开始")
        self.action_btn.setFixedSize(40, 22)
        self.action_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #4ade80; }
        """)
        self.action_btn.clicked.connect(self._on_action_clicked)
        title_bar.addWidget(self.action_btn)

        layout.addLayout(title_bar)

        # FPS display
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #8888a8; font-size: 11px;")
        layout.addWidget(self.fps_label)

        # Confidence slider
        conf_layout = QHBoxLayout()
        conf_layout.setSpacing(6)

        conf_lbl = QLabel("置信度:")
        conf_lbl.setStyleSheet("color: #8888a8; font-size: 11px;")
        conf_layout.addWidget(conf_lbl)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 99)
        self.conf_slider.setValue(50)
        self.conf_slider.setFixedHeight(16)
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #2a2a4a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                height: 12px;
                margin: -4px 0;
                background: #6366f1;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #6366f1;
                border-radius: 2px;
            }
        """)
        conf_layout.addWidget(self.conf_slider)

        self.conf_val = QLabel("0.50")
        self.conf_val.setStyleSheet("color: #a5b4fc; font-size: 11px; font-weight: 600;")
        self.conf_val.setFixedWidth(32)
        conf_layout.addWidget(self.conf_val)

        self.conf_slider.valueChanged.connect(self._on_conf_changed)

        layout.addLayout(conf_layout)

        # FPS spin
        fps_layout = QHBoxLayout()
        fps_layout.setSpacing(6)

        fps_lbl = QLabel("帧率:")
        fps_lbl.setStyleSheet("color: #8888a8; font-size: 11px;")
        fps_layout.addWidget(fps_lbl)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setFixedHeight(22)
        self.fps_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1a1a2e;
                color: #e8e8f0;
                border: 1px solid #2a2a4a;
                border-radius: 4px;
                padding: 2px 4px;
                font-size: 11px;
            }
        """)
        fps_layout.addWidget(self.fps_spin)

        fps_layout.addStretch()
        layout.addLayout(fps_layout)

        # Checkboxes
        self.priority_cb = QCheckBox("优先级")
        self.priority_cb.setStyleSheet("color: #8888a8; font-size: 11px;")
        layout.addWidget(self.priority_cb)

        self.auto_click_cb = QCheckBox("自动点击")
        self.auto_click_cb.setStyleSheet("color: #8888a8; font-size: 11px;")
        layout.addWidget(self.auto_click_cb)

    def _on_action_clicked(self):
        if self._detecting:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    def set_detecting(self, detecting: bool):
        self._detecting = detecting
        if detecting:
            self.title_label.setText("检测中")
            self.action_btn.setText("停止")
            self.action_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #f87171; }
            """)
        else:
            self.title_label.setText("悬浮窗")
            self.action_btn.setText("开始")
            self.action_btn.setStyleSheet("""
                QPushButton {
                    background-color: #22c55e;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #4ade80; }
            """)

    def _on_conf_changed(self, v):
        self.conf_val.setText(f"{v/100:.2f}")
        self.settings_changed.emit({"confidence": v})

    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def sync_from_detect_tab(self, detect_tab):
        """Sync settings from DetectTab."""
        self.conf_slider.setValue(detect_tab.conf_slider.value())
        self.fps_spin.setValue(detect_tab.fps_spin.value())
        self.priority_cb.setChecked(detect_tab.priority_cb.isChecked())
        self.auto_click_cb.setChecked(detect_tab.auto_click_cb.isChecked())

        # Only connect once to prevent signal accumulation
        if not self._sync_connected:
            self.conf_slider.valueChanged.connect(
                lambda v: detect_tab.conf_slider.setValue(v))
            self.fps_spin.valueChanged.connect(
                lambda v: detect_tab.fps_spin.setValue(v))
            self.priority_cb.toggled.connect(detect_tab.priority_cb.setChecked)
            self.auto_click_cb.toggled.connect(detect_tab.auto_click_cb.setChecked)
            self._sync_connected = True

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def closeEvent(self, event):
        self.expand_clicked.emit()
        event.accept()
