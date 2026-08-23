"""主窗口：页签组装、全局热键、悬浮窗与状态持久化。

相对历史版本的改动：
- NativeHotkeyManager（Qt+线程混合体）移出：轮询逻辑在 core.hotkey
  （零 Qt），本文件仅保留一个信号桥接对象；
- 各管理器/配置由 AppContext 组装，页签构造签名统一；
- 添加/编辑目标对话框抽到 ui.dialogs 共享。
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog, QInputDialog, QMainWindow, QMessageBox, QStatusBar, QTabWidget,
)

from core.hotkey import HotkeyWatcher
from ui.app_context import AppContext
from ui.annotate_tab import AnnotateTab
from ui.detect_tab import DetectTab
from ui.dialogs import AddTargetDialog
from ui.floating_window import FloatingWindow
from ui.opencv_tab import OpenCVTab
from ui.settings_tab import SettingsTab
from ui.train_tab import TrainTab

logger = logging.getLogger(__name__)


class HotkeyBridge(QObject):
    """core.HotkeyWatcher 回调 -> Qt 信号（跨线程自动走队列连接）。"""

    hotkey_pressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._watcher = HotkeyWatcher(self._emit_pressed)

    def register(self, key_str: str) -> bool:
        return self._watcher.register(key_str)

    def unregister(self) -> None:
        self._watcher.unregister()

    def _emit_pressed(self):
        self.hotkey_pressed.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动点击 - 屏幕识别工具")
        self.setMinimumSize(1200, 800)

        # 组合根
        self.ctx = AppContext()
        self._restore_window_state()

        # 页签
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.detect_tab = DetectTab(self.ctx)
        self.annotate_tab = AnnotateTab(self.ctx)
        self.train_tab = TrainTab(self.ctx)
        self.opencv_tab = OpenCVTab(self.ctx)
        self.settings_tab = SettingsTab(self.ctx)

        self.tabs.addTab(self.detect_tab, "检测")
        self.tabs.addTab(self.annotate_tab, "标注")
        self.tabs.addTab(self.train_tab, "训练")
        self.tabs.addTab(self.opencv_tab, "OpenCV 模板")
        self.tabs.addTab(self.settings_tab, "设置")
        self.setCentralWidget(self.tabs)

        # 悬浮窗（创建但不显示）
        self.floating_window = FloatingWindow()
        self._setup_floating_window()

        # 状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

        # 信号连接
        self.train_tab.training_complete.connect(self._on_training_complete)
        self.settings_tab.config_changed.connect(self._on_config_changed)
        # 目标变更统一走 hub 广播（ARCH-02：新增页签修改目标后无需在此补连）
        self.ctx.hub.targets_changed.connect(self._refresh_all)
        self.annotate_tab.sample_set_renamed.connect(self._on_sample_set_renamed)
        self.detect_tab.status_message.connect(self.status.showMessage)

        # 启动自检（BIZ-05）：失效路径/损坏备份汇总提示
        self._run_startup_health_check()

        # 全局热键
        self._hotkey_bridge = HotkeyBridge(self)
        self._hotkey_bridge.hotkey_pressed.connect(self._on_hotkey_toggle_detection)
        self._update_global_hotkey()

        # 菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("导入模型到目标...", self._import_model)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        self._refresh_all()
        self._restore_tab_states()

    # ========== 目标操作 ==========

    def _add_target(self):
        dlg = AddTargetDialog(sample_sets=self.ctx.ssm.get_all(), parent=self)
        if dlg.exec() == AddTargetDialog.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "提示", "请输入目标名称")
                return
            target = self.ctx.tm.add(**data)
            os.makedirs(os.path.join(target.data_dir, "labels"), exist_ok=True)
            self.ctx.tm.save()
            self._refresh_all()
            self.status.showMessage(f"已添加目标: {target.name}")

    def _edit_target(self):
        tm = self.ctx.tm
        if not tm.targets:
            QMessageBox.information(self, "提示", "暂无目标")
            return
        names = [t.name for t in tm.targets]
        name, ok = QInputDialog.getItem(self, "编辑目标", "选择目标:", names, 0, False)
        if not ok:
            return
        target = next((t for t in tm.targets if t.name == name), None)
        if not target:
            return
        dlg = AddTargetDialog(sample_sets=self.ctx.ssm.get_all(), parent=self)
        dlg.set_target(target)
        if dlg.exec() == AddTargetDialog.Accepted:
            data = dlg.get_data()
            target.name = data["name"]
            target.detection_mode = data["detection_mode"]
            target.priority = data["priority"]
            target.color = data["color"]
            if data.get("set_id"):
                target.set_id = data["set_id"]
                target.data_dir = os.path.join(
                    "data", "sets", data["set_id"], "targets", target.id)
            self.ctx.tm.save()
            self._refresh_all()

    def _import_model(self):
        tm = self.ctx.tm
        if not tm.targets:
            QMessageBox.information(self, "提示", "请先添加目标")
            return
        # SEC-01：.pt 是 pickle 序列化格式，加载即执行其中代码。
        # 从他人处获取的模型等同于运行未知程序，必须明示风险。
        reply = QMessageBox.warning(
            self, "安全提示",
            "模型文件 (.pt) 加载时会执行其中嵌入的代码。\n"
            "仅导入来源可信的模型（自己训练或完全信任的提供者），\n"
            "从网络/群组下载的模型可能包含恶意代码。\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt *.onnx)")
        if not path:
            return
        names = [t.name for t in tm.targets]
        name, ok = QInputDialog.getItem(self, "选择目标", "将模型导入到:", names, 0, False)
        if not ok:
            return
        target = next((t for t in tm.targets if t.name == name), None)
        if target:
            target.model_path = path
            self.ctx.drop_model(target.id)   # 让后台加载器重新装载
            tm.save()
            self._refresh_all()
            self.status.showMessage(f"已将模型导入到目标 '{target.name}': {path}")

    def _run_startup_health_check(self):
        issues = self.ctx.startup_health_check()
        if issues:
            self.status.showMessage(f"启动自检: 发现 {len(issues)} 项异常（详见日志）", 8000)

    def _refresh_all(self):
        self.annotate_tab.refresh_targets()
        self.train_tab.refresh_targets()
        self.detect_tab.refresh_targets()
        self.opencv_tab.refresh_from_manager()
        self.status.showMessage(f"已加载 {len(self.ctx.tm.targets)} 个目标")

    def _on_sample_set_renamed(self, set_id: str):
        self.train_tab.refresh_targets()
        self.detect_tab.refresh_targets()

    @Slot(str, str)
    def _on_training_complete(self, model_path: str, target_id: str):
        self.detect_tab.refresh_targets()
        self.status.showMessage(f"训练完成: {model_path}")

    @Slot(dict)
    def _on_config_changed(self, cfg: dict):
        self._update_global_hotkey()

    # ========== 悬浮窗 ==========

    def _setup_floating_window(self):
        self.detect_tab.floating_window_requested.connect(self._toggle_floating_window)
        self.floating_window.stop_clicked.connect(self._stop_from_floating)
        self.floating_window.start_clicked.connect(self._start_from_floating)
        self.floating_window.expand_clicked.connect(self._expand_from_floating)
        self.floating_window.fps_spin.valueChanged.connect(
            lambda v: self.detect_tab.fps_spin.setValue(v))

    def _toggle_floating_window(self, show: bool, detecting: bool):
        if show:
            # 悬浮窗模式：主窗口已最小化，断开预览绘制节省主线程开销
            self.detect_tab.set_preview_enabled(False)
            self.floating_window.sync_from_detect_tab(self.detect_tab)
            self.floating_window.set_detecting(detecting)
            fw_state = self.ctx.state.get("floating_window", {})
            x, y = fw_state.get("x"), fw_state.get("y")
            if x is not None and y is not None:
                self.floating_window.move(x, y)
            else:
                screen = self.screen()
                if screen:
                    geo = screen.availableGeometry()
                    self.floating_window.move(geo.width() - 280, geo.height() - 240)
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
        if self.detect_tab.is_detecting:
            self.detect_tab._toggle_detection()
            if not self.detect_tab.floating_auto_switch_cb.isChecked():
                self._expand_from_floating()
        else:
            self._expand_from_floating()

    def _start_from_floating(self):
        self.detect_tab._toggle_detection()
        self.floating_window.set_detecting(self.detect_tab.is_detecting)

    def _expand_from_floating(self):
        pos = self.floating_window.pos()
        self.ctx.state.setdefault("floating_window", {})
        self.ctx.state["floating_window"]["x"] = pos.x()
        self.ctx.state["floating_window"]["y"] = pos.y()
        self.floating_window.hide()
        self.detect_tab.set_preview_enabled(True)
        self.showNormal()
        self.activateWindow()

    # ========== 全局热键 ==========

    def _update_global_hotkey(self):
        hotkey = self.ctx.config.get("hotkey_toggle_detection", "F9")
        if hotkey and not self._hotkey_bridge.register(hotkey):
            self.status.showMessage(f"快捷键 {hotkey} 无效，请检查格式", 5000)

    def _on_hotkey_toggle_detection(self):
        was_running = self.detect_tab.is_detecting
        self.detect_tab._toggle_detection()
        is_running = self.detect_tab.is_detecting
        if self.floating_window.isVisible():
            if was_running and not is_running:
                self.floating_window.set_detecting(False)
            elif not was_running and is_running:
                self.floating_window.sync_from_detect_tab(self.detect_tab)
                self.floating_window.set_detecting(True)

    # ========== 状态持久化 ==========

    def _restore_window_state(self):
        ws = self.ctx.state.get("window", {})
        x, y = ws.get("x"), ws.get("y")
        if x is not None and y is not None:
            self.move(x, y)
        self.resize(ws.get("width", 1200), ws.get("height", 800))
        if ws.get("maximized", False):
            self.showMaximized()

    def _restore_tab_states(self):
        state = self.ctx.state
        active_tab = state.get("window", {}).get("active_tab", 0)
        if 0 <= active_tab < self.tabs.count():
            self.tabs.setCurrentIndex(active_tab)

        detect_state = state.get("detect", {})
        self.detect_tab.conf_slider.setValue(detect_state.get("confidence", 50))
        self.detect_tab.fps_spin.setValue(detect_state.get("fps", 10))
        self.detect_tab.priority_cb.setChecked(detect_state.get("priority_enabled", False))
        self.detect_tab.auto_click_cb.setChecked(detect_state.get("auto_click", False))
        self.detect_tab.floating_auto_switch_cb.setChecked(
            detect_state.get("floating_auto_switch", False))

        train_state = state.get("train", {})
        self.train_tab.epochs_spin.setValue(train_state.get("epochs", 50))
        self.train_tab.batch_spin.setValue(train_state.get("batch_size", 16))
        self.train_tab.imgsize_spin.setValue(train_state.get("img_size", 640))
        save_path = train_state.get("save_path", "")
        if save_path:
            self.train_tab.save_path_edit.setText(save_path)
        last_train_target = train_state.get("last_target_id", "")
        if last_train_target:
            self.train_tab.select_target_by_id(last_train_target)

        opencv_state = state.get("opencv", {})
        self.opencv_tab.threshold_spin.setValue(opencv_state.get("threshold", 0.8))

        annotate_state = state.get("annotate", {})
        selected_set_id = annotate_state.get("selected_set_id", "")
        if selected_set_id:
            self.annotate_tab.select_set_by_id(selected_set_id)
        last_target_id = annotate_state.get("last_target_id", "")
        if last_target_id:
            self.annotate_tab.select_target_by_id(last_target_id)

    def _save_state(self):
        state = self.ctx.state
        state["window"] = {
            "x": self.pos().x(),
            "y": self.pos().y(),
            "width": self.width(),
            "height": self.height(),
            "maximized": self.isMaximized(),
            "active_tab": self.tabs.currentIndex(),
        }
        state["detect"] = {
            "confidence": self.detect_tab.conf_slider.value(),
            "fps": self.detect_tab.fps_spin.value(),
            "priority_enabled": self.detect_tab.priority_cb.isChecked(),
            "auto_click": self.detect_tab.auto_click_cb.isChecked(),
            "floating_auto_switch": self.detect_tab.floating_auto_switch_cb.isChecked(),
        }
        state["train"] = {
            "epochs": self.train_tab.epochs_spin.value(),
            "batch_size": self.train_tab.batch_spin.value(),
            "img_size": self.train_tab.imgsize_spin.value(),
            "save_path": self.train_tab.save_path_edit.text(),
            "last_target_id": self.train_tab.target_combo.currentData() or "",
        }
        state["opencv"] = {"threshold": self.opencv_tab.threshold_spin.value()}

        annotate_state = dict(state.get("annotate", {}))
        if self.annotate_tab._current_target:
            annotate_state["last_target_id"] = self.annotate_tab._current_target.id
        if self.annotate_tab._current_set:
            annotate_state["selected_set_id"] = self.annotate_tab._current_set.id
        state["annotate"] = annotate_state

        self.ctx.save_app_state()

    def closeEvent(self, event):
        # 退出收尾（BUG-01）：检测与训练线程都必须显式停止，
        # 否则退出瞬间检测线程仍在 SendInput 点击屏幕（幽灵点击），
        # 或 QThread 在析构阶段被强杀导致崩溃
        try:
            self.detect_tab.stop_if_running(timeout_ms=8000)
        except Exception:
            logger.exception("停止检测线程失败")
        try:
            if self.train_tab.train_worker is not None and self.train_tab.train_worker.isRunning():
                logger.info("退出前等待训练线程协作停止…")
                self.train_tab.train_worker.request_stop()
                self.train_tab.train_worker.wait(15000)
        except Exception:
            logger.exception("停止训练线程失败")
        try:
            self._hotkey_bridge.unregister()
        except Exception:
            logger.exception("注销热键失败")
        try:
            self._save_state()
        except Exception:
            logger.exception("保存应用状态失败")
        super().closeEvent(event)
