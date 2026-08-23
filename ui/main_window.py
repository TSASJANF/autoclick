import ctypes
import os
import threading
import time

from PySide6.QtCore import Qt, Slot, Signal, QObject
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QMenuBar, QMenu,
    QInputDialog, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QComboBox, QPushButton, QFormLayout,
    QCheckBox, QFileDialog, QFrame,
)

from core.target_manager import TargetManager, Target, TARGET_COLORS
from core.sample_set_manager import SampleSetManager
from core.opencv_detector import OpenCVDetector
from core.config import load_config
from core.state_manager import load_state, save_state

from ui.detect_tab import DetectTab
from ui.annotate_tab import AnnotateTab
from ui.train_tab import TrainTab
from ui.opencv_tab import OpenCVTab
from ui.settings_tab import SettingsTab
from ui.floating_window import FloatingWindow


# Windows API constants for hotkey polling
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

user32 = ctypes.windll.user32


class NativeHotkeyManager(QObject):
    """System-wide global hotkey using GetAsyncKeyState polling.

    Unlike RegisterHotKey, this always works regardless of whether other
    programs have already registered the same hotkey.  A background thread
    polls key state every 50 ms and emits hotkey_pressed on the rising edge.
    """
    hotkey_pressed = Signal()

    # Virtual key codes for modifier keys
    _MOD_KEYS = {
        MOD_CONTROL: (0x11,),          # VK_CONTROL
        MOD_ALT: (0x12,),              # VK_MENU (Alt)
        MOD_SHIFT: (0x10,),            # VK_SHIFT
        MOD_WIN: (0x5B, 0x5C),         # VK_LWIN, VK_RWIN
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._running = False
        self._vk = None
        self._mods = 0
        self._was_pressed = False

    def register(self, key_str: str) -> bool:
        """Start polling for the given hotkey. Returns True on success."""
        self.unregister()
        vk, mods = self._parse_key(key_str)
        if vk is None:
            return False
        self._vk = vk
        self._mods = mods
        self._was_pressed = False
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return True

    def unregister(self):
        """Stop the polling thread."""
        self._running = False
        # Thread is daemon — will exit automatically, but join for cleanliness
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ── internal ────────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            pressed = self._is_hotkey_down()
            if pressed and not self._was_pressed:
                self.hotkey_pressed.emit()
            self._was_pressed = pressed
            time.sleep(0.05)        # 20 Hz — negligible CPU, ~50 ms latency

    def _is_hotkey_down(self) -> bool:
        """Return True when every required modifier AND the main key are held."""
        # Check modifiers
        for mod_flag, vk_codes in self._MOD_KEYS.items():
            if self._mods & mod_flag:
                if not any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in vk_codes):
                    return False
        # Check main key
        return bool(user32.GetAsyncKeyState(self._vk) & 0x8000)

    @staticmethod
    def _parse_key(key_str: str):
        """Parse a human-readable hotkey string into (vk_code, modifier_flags).

        Accepts formats like ``"F9"``, ``"Ctrl+F9"``, ``"Shift+Alt+A"``.
        """
        VK_MAP = {
            'escape': 0x1B, 'esc': 0x1B, 'tab': 0x09, 'space': 0x20,
            'return': 0x0D, 'enter': 0x0D, 'backspace': 0x08, 'delete': 0x2E,
            'insert': 0x2D, 'home': 0x24, 'end': 0x23,
            'pageup': 0x21, 'pagedown': 0x22,
            'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
            'printscreen': 0x2C, 'scrolllock': 0x91, 'pause': 0x13,
            'capslock': 0x14, 'numlock': 0x90,
        }
        for i in range(1, 25):
            VK_MAP[f'f{i}'] = 0x70 + i - 1

        parts = [p.strip().lower() for p in key_str.split('+')]
        mods = 0
        vk = None
        for part in parts:
            if part in ('ctrl', 'control'):
                mods |= MOD_CONTROL
            elif part == 'alt':
                mods |= MOD_ALT
            elif part == 'shift':
                mods |= MOD_SHIFT
            elif part in ('win', 'meta', 'super'):
                mods |= MOD_WIN
            elif part in VK_MAP:
                vk = VK_MAP[part]
            elif len(part) == 1:
                vk = ord(part.upper())
        return vk, mods


class AddTargetDialog(QDialog):
    def __init__(self, sample_sets=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加目标")
        self.setMinimumWidth(420)
        self.setContentsMargins(24, 24, 24, 24)
        self._sample_sets = sample_sets or []

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header
        header = QLabel("创建新检测目标")
        header.setObjectName("heading")
        layout.addWidget(header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # Form
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
        self.mode_combo.addItems(["仅 AI", "仅 OpenCV", "AI + OpenCV"])
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

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        layout.addWidget(sep2)

        # Buttons
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

    def get_data(self) -> dict:
        mode_map = {"仅 AI": "ai", "仅 OpenCV": "opencv", "AI + OpenCV": "both"}
        data = {
            "name": self.name_edit.text(),
            "detection_mode": mode_map[self.mode_combo.currentText()],
            "priority": self.priority_spin.value(),
            "color": self.color_combo.currentText(),
        }
        if self._sample_sets and hasattr(self, 'set_combo'):
            data["set_id"] = self.set_combo.currentData()
        return data


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动点击 - 屏幕识别工具")
        self.setMinimumSize(1200, 800)

        # Core managers
        self.tm = TargetManager()
        self.tm.load()
        self.ssm = SampleSetManager()
        self.ssm.load()
        self.ssm.ensure_default()
        self.opencv_detector = OpenCVDetector()

        # Config and state
        self.config = load_config()
        self.state = load_state()

        # Restore window geometry
        self._restore_window_state()

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.detect_tab = DetectTab(self.tm, self.opencv_detector, self.config, self.ssm)
        self.annotate_tab = AnnotateTab(self.tm, self.config, self.ssm)
        self.train_tab = TrainTab(self.tm, self.ssm)
        self.opencv_tab = OpenCVTab(self.tm, self.opencv_detector, self.config, self.ssm)
        self.settings_tab = SettingsTab()

        self.tabs.addTab(self.detect_tab, "检测")
        self.tabs.addTab(self.annotate_tab, "标注")
        self.tabs.addTab(self.train_tab, "训练")
        self.tabs.addTab(self.opencv_tab, "OpenCV 模板")
        self.tabs.addTab(self.settings_tab, "设置")
        self.setCentralWidget(self.tabs)

        # Floating window (created but not shown)
        self.floating_window = FloatingWindow()
        self._setup_floating_window()

        # Status bar (must be created before connections)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

        # Connections
        self.train_tab.training_complete.connect(self._on_training_complete)
        self.settings_tab.config_changed.connect(self._on_config_changed)
        self.annotate_tab.target_deleted.connect(self._refresh_all)
        self.annotate_tab.target_added.connect(self._refresh_all)
        self.annotate_tab.sample_set_renamed.connect(self._on_sample_set_renamed)
        self.detect_tab.target_deleted.connect(self._refresh_all)
        self.detect_tab.status_message.connect(self.status.showMessage)

        # Setup global hotkey for detection toggle
        self._setup_global_hotkey()

        # Menu bar
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        file_menu.addAction("导入模型到目标...", self._import_model)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        # Initial refresh
        self._refresh_all()

        # Restore tab state
        self._restore_tab_states()

    def _add_target(self):
        dlg = AddTargetDialog(sample_sets=self.ssm.get_all(), parent=self)
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "提示", "请输入目标名称")
                return
            target = self.tm.add(**data)
            # Create data directory
            os.makedirs(os.path.join(target.data_dir, "labels"), exist_ok=True)
            self.tm.save()
            self._refresh_all()
            self.status.showMessage(f"已添加目标: {target.name}")

    def _edit_target(self):
        if not self.tm.targets:
            QMessageBox.information(self, "提示", "暂无目标")
            return
        names = [t.name for t in self.tm.targets]
        name, ok = QInputDialog.getItem(self, "编辑目标", "选择目标:", names, 0, False)
        if not ok:
            return
        target = next((t for t in self.tm.targets if t.name == name), None)
        if not target:
            return
        dlg = AddTargetDialog(sample_sets=self.ssm.get_all(), parent=self)
        dlg.name_edit.setText(target.name)
        dlg.priority_spin.setValue(target.priority)
        if target.set_id and hasattr(dlg, 'set_combo'):
            for i in range(dlg.set_combo.count()):
                if dlg.set_combo.itemData(i) == target.set_id:
                    dlg.set_combo.setCurrentIndex(i)
                    break
        if dlg.exec() == QDialog.Accepted:
            data = dlg.get_data()
            target.name = data["name"]
            target.detection_mode = data["detection_mode"]
            target.priority = data["priority"]
            target.color = data["color"]
            if "set_id" in data and data["set_id"]:
                target.set_id = data["set_id"]
                # Update data_dir to match new set
                target.data_dir = os.path.join("data", "sets", data["set_id"], "targets", target.id)
            self.tm.save()
            self._refresh_all()

    def _delete_target(self):
        if not self.tm.targets:
            QMessageBox.information(self, "提示", "暂无目标")
            return
        names = [t.name for t in self.tm.targets]
        name, ok = QInputDialog.getItem(self, "删除目标", "选择目标:", names, 0, False)
        if not ok:
            return
        target = next((t for t in self.tm.targets if t.name == name), None)
        if target:
            reply = QMessageBox.question(self, "确认删除", f"确定删除目标 '{target.name}' 吗？")
            if reply == QMessageBox.Yes:
                self.tm.remove(target.id)
                self.tm.save()
                self._refresh_all()

    def _import_model(self):
        if not self.tm.targets:
            QMessageBox.information(self, "提示", "请先添加目标")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt *.onnx)")
        if not path:
            return
        names = [t.name for t in self.tm.targets]
        name, ok = QInputDialog.getItem(self, "选择目标", "将模型导入到:", names, 0, False)
        if not ok:
            return
        target = next((t for t in self.tm.targets if t.name == name), None)
        if target:
            target.model_path = path
            self.tm.save()
            self._refresh_all()
            self.status.showMessage(f"已将模型导入到目标 '{target.name}': {path}")

    def _refresh_all(self):
        self.annotate_tab.refresh_targets()
        self.train_tab.refresh_targets()
        self.detect_tab.refresh_targets()
        self.opencv_tab.refresh_from_manager()
        target_count = len(self.tm.targets)
        self.status.showMessage(f"已加载 {target_count} 个目标")

    def _on_sample_set_renamed(self, set_id: str):
        """When a sample set is renamed, refresh tabs that display set names."""
        self.train_tab.refresh_targets()
        self.detect_tab.refresh_targets()

    def _setup_floating_window(self):
        """Connect floating window signals."""
        self.detect_tab.floating_window_requested.connect(self._toggle_floating_window)
        self.floating_window.stop_clicked.connect(self._stop_from_floating)
        self.floating_window.start_clicked.connect(self._start_from_floating)
        self.floating_window.expand_clicked.connect(self._expand_from_floating)
        self.floating_window.fps_spin.valueChanged.connect(
            lambda v: self.detect_tab.fps_spin.setValue(v))

    def _toggle_floating_window(self, show: bool, detecting: bool):
        if show:
            self.floating_window.sync_from_detect_tab(self.detect_tab)
            self.floating_window.set_detecting(detecting)
            # Restore saved position or center on screen
            fw_state = self.state.get("floating_window", {})
            x = fw_state.get("x")
            y = fw_state.get("y")
            if x is not None and y is not None:
                self.floating_window.move(x, y)
            else:
                screen = self.screen()
                if screen:
                    geo = screen.availableGeometry()
                    self.floating_window.move(
                        geo.width() - 280, geo.height() - 240
                    )
            # Connect FPS updates if detecting
            if detecting and self.detect_tab.detect_thread:
                try:
                    self.detect_tab.detect_thread.fps_update.disconnect(
                        self.floating_window.update_fps)
                except (RuntimeError, TypeError):
                    pass
                self.detect_tab.detect_thread.fps_update.connect(
                    self.floating_window.update_fps)
            self.floating_window.show()
            self.showMinimized()
        else:
            self._expand_from_floating()

    def _stop_from_floating(self):
        """Stop detection from floating window."""
        if self.detect_tab.detect_thread and self.detect_tab.detect_thread.running:
            self.detect_tab._toggle_detection()
            # _toggle_detection will emit floating_window_requested(False, False)
            # which will trigger _expand_from_floating if auto-switch is enabled
            if not self.detect_tab.floating_auto_switch_cb.isChecked():
                self._expand_from_floating()
        else:
            self._expand_from_floating()

    def _start_from_floating(self):
        """Start detection from floating window."""
        self.detect_tab._toggle_detection()
        is_running = self.detect_tab.detect_thread and self.detect_tab.detect_thread.running
        self.floating_window.set_detecting(is_running)

    def _expand_from_floating(self):
        """Restore main window from floating mode."""
        # Save floating window position
        pos = self.floating_window.pos()
        self.state.setdefault("floating_window", {})
        self.state["floating_window"]["x"] = pos.x()
        self.state["floating_window"]["y"] = pos.y()
        self.floating_window.hide()
        self.showNormal()
        self.activateWindow()

    @Slot(str, str)
    def _on_training_complete(self, model_path: str, target_id: str):
        self.detect_tab.refresh_targets()
        self.status.showMessage(f"训练完成: {model_path}")

    @Slot(dict)
    def _on_config_changed(self, cfg: dict):
        self.config = cfg
        self.detect_tab.update_config(cfg)
        self.annotate_tab.config = cfg
        self.opencv_tab.config = cfg
        self._update_global_hotkey()

    def _restore_window_state(self):
        """Restore window position, size, and active tab."""
        ws = self.state.get("window", {})
        x = ws.get("x")
        y = ws.get("y")
        width = ws.get("width", 1200)
        height = ws.get("height", 800)

        if x is not None and y is not None:
            self.move(x, y)
        self.resize(width, height)

        if ws.get("maximized", False):
            self.showMaximized()

    def _restore_tab_states(self):
        """Restore tab-specific states."""
        # Active tab
        active_tab = self.state.get("window", {}).get("active_tab", 0)
        if 0 <= active_tab < self.tabs.count():
            self.tabs.setCurrentIndex(active_tab)

        # Detect tab
        detect_state = self.state.get("detect", {})
        self.detect_tab.conf_slider.setValue(detect_state.get("confidence", 50))
        self.detect_tab.fps_spin.setValue(detect_state.get("fps", 10))
        self.detect_tab.priority_cb.setChecked(detect_state.get("priority_enabled", False))
        self.detect_tab.auto_click_cb.setChecked(detect_state.get("auto_click", False))
        self.detect_tab.floating_auto_switch_cb.setChecked(detect_state.get("floating_auto_switch", False))

        # Train tab
        train_state = self.state.get("train", {})
        self.train_tab.epochs_spin.setValue(train_state.get("epochs", 50))
        self.train_tab.batch_spin.setValue(train_state.get("batch_size", 16))
        self.train_tab.imgsize_spin.setValue(train_state.get("img_size", 640))
        save_path = train_state.get("save_path", "")
        if save_path:
            self.train_tab.save_path_edit.setText(save_path)
        last_train_target = train_state.get("last_target_id", "")
        if last_train_target:
            self.train_tab.select_target_by_id(last_train_target)

        # OpenCV tab
        opencv_state = self.state.get("opencv", {})
        self.opencv_tab.threshold_spin.setValue(opencv_state.get("threshold", 0.8))

        # Annotate tab - restore last target selection and sample set
        annotate_state = self.state.get("annotate", {})
        selected_set_id = annotate_state.get("selected_set_id", "")
        if selected_set_id:
            self.annotate_tab.select_set_by_id(selected_set_id)
        # Restore last selected target by ID (not index, since index is per-set now)
        last_target_id = annotate_state.get("last_target_id", "")
        if last_target_id:
            self.annotate_tab.select_target_by_id(last_target_id)

    def _save_state(self):
        """Save current application state."""
        self.state["window"] = {
            "x": self.pos().x(),
            "y": self.pos().y(),
            "width": self.width(),
            "height": self.height(),
            "maximized": self.isMaximized(),
            "active_tab": self.tabs.currentIndex(),
        }

        self.state["detect"] = {
            "confidence": self.detect_tab.conf_slider.value(),
            "fps": self.detect_tab.fps_spin.value(),
            "priority_enabled": self.detect_tab.priority_cb.isChecked(),
            "auto_click": self.detect_tab.auto_click_cb.isChecked(),
            "floating_auto_switch": self.detect_tab.floating_auto_switch_cb.isChecked(),
        }

        self.state["train"] = {
            "epochs": self.train_tab.epochs_spin.value(),
            "batch_size": self.train_tab.batch_spin.value(),
            "img_size": self.train_tab.imgsize_spin.value(),
            "save_path": self.train_tab.save_path_edit.text(),
            "last_target_id": self.train_tab.target_combo.currentData() or "",
        }

        self.state["opencv"] = {
            "threshold": self.opencv_tab.threshold_spin.value(),
        }

        # Annotate tab - save last selected target ID and sample set
        annotate_state = self.state.get("annotate", {})
        if self.annotate_tab._current_target:
            annotate_state["last_target_id"] = self.annotate_tab._current_target.id
        if self.annotate_tab._current_set:
            annotate_state["selected_set_id"] = self.annotate_tab._current_set.id
        self.state["annotate"] = annotate_state

        save_state(self.state)

    def closeEvent(self, event):
        """Save state and cleanup when closing the application."""
        try:
            self._native_hotkey.unregister()
        except Exception:
            pass
        try:
            self._save_state()
        except Exception:
            pass
        super().closeEvent(event)

    def _setup_global_hotkey(self):
        """Setup system-wide global hotkey using Windows RegisterHotKey."""
        self._native_hotkey = NativeHotkeyManager(self)
        self._native_hotkey.hotkey_pressed.connect(self._on_hotkey_toggle_detection)
        self._update_global_hotkey()

    def _update_global_hotkey(self):
        """Update global hotkey based on config."""
        hotkey = self.config.get("hotkey_toggle_detection", "F9")
        if hotkey:
            ok = self._native_hotkey.register(hotkey)
            if not ok:
                self.status.showMessage(f"快捷键 {hotkey} 无效，请检查格式", 5000)
            else:
                print(f"[Hotkey] 全局快捷键已注册: {hotkey}")

    def _on_hotkey_toggle_detection(self):
        """Handle hotkey press - toggle detection without restoring window."""
        was_running = self.detect_tab.detect_thread and self.detect_tab.detect_thread.running
        self.detect_tab._toggle_detection()
        is_running = self.detect_tab.detect_thread and self.detect_tab.detect_thread.running
        # Sync floating window state if visible
        if self.floating_window.isVisible():
            if was_running and not is_running:
                self.floating_window.set_detecting(False)
            elif not was_running and is_running:
                self.floating_window.sync_from_detect_tab(self.detect_tab)
                self.floating_window.set_detecting(True)
