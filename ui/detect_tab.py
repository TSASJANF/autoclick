import os
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QColor, QBrush, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QFileDialog, QGroupBox,
    QTableWidget, QTableWidgetItem, QCheckBox, QSplitter,
    QAbstractItemView, QMenu, QMessageBox, QInputDialog, QFrame,
    QComboBox,
)

from core.capture import ScreenCapture
from core.detector import Detector
from core.opencv_detector import OpenCVDetector
from core.clicker import Clicker
from core.target_manager import TargetManager, Target


class DetectionResult:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float,
                 target_id: str, target_name: str, source: str, color: str):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.conf = conf
        self.target_id = target_id
        self.target_name = target_name
        self.source = source
        self.color = color

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def _get_model_label(target: Target) -> str:
    if target.detection_mode == "opencv":
        return "opencv"
    if target.model_base:
        return target.model_base
    # Try to read from training args.yaml
    if target.model_path:
        model_dir = os.path.dirname(target.model_path)  # weights/
        train_dir = os.path.dirname(model_dir)           # train-N/
        args_file = os.path.join(train_dir, "args.yaml")
        if os.path.exists(args_file):
            try:
                with open(args_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("model:"):
                            val = line.split(":", 1)[1].strip()
                            return os.path.splitext(os.path.basename(val))[0]
            except Exception:
                pass
        return os.path.splitext(os.path.basename(target.model_path))[0]
    return "ai"


def _detect_single_target(target: Target, ai_detectors: dict, opencv_detector: OpenCVDetector,
                          img: np.ndarray, conf: float) -> list[DetectionResult]:
    """Run detection for a single target. Used in parallel execution."""
    results = []
    if target.detection_mode in ("ai", "both"):
        detector = ai_detectors.get(target.id)
        if detector and detector.model:
            dets = detector.detect(img, conf=conf)
            for d in dets:
                results.append(DetectionResult(
                    d.x1, d.y1, d.x2, d.y2, d.conf,
                    target.id, target.name, "ai", target.color
                ))
    if target.detection_mode in ("opencv", "both"):
        dets = opencv_detector.detect(img, target_ids=[target.id])
        for d in dets:
            results.append(DetectionResult(
                d.x1, d.y1, d.x2, d.y2, d.conf,
                target.id, target.name, "opencv", target.color
            ))
    return results


class SingleDetectThread(QThread):
    """Detection thread for single target mode - sequential, click all instances by confidence."""
    frame_ready = Signal(np.ndarray, list)
    fps_update = Signal(float)

    def __init__(self, ai_detectors: dict[str, Detector], opencv_detector: OpenCVDetector,
                 target_manager: TargetManager, capture: ScreenCapture,
                 conf: float = 0.5, fps: int = 10, region=None,
                 auto_click: bool = False, use_priority: bool = False,
                 hide_window: bool = False):
        super().__init__()
        self.ai_detectors = ai_detectors
        self.opencv_detector = opencv_detector
        self.tm = target_manager
        self.capture = capture
        self.conf = conf
        self.target_fps = fps
        self.region = region
        self.auto_click = auto_click
        self.use_priority = use_priority
        self.hide_window = hide_window
        self.running = False
        self.clicker = Clicker()

    def run(self):
        self.running = True
        interval = 1.0 / max(self.target_fps, 1)
        hwnd = 0
        if self.hide_window:
            hwnd = ScreenCapture.get_foreground_hwnd()

        while self.running:
            t0 = time.perf_counter()

            if self.hide_window and hwnd:
                img = self.capture.capture_hide_window(hwnd)
            elif self.region:
                img = self.capture.capture_region(*self.region)
            else:
                img = self.capture.capture_full()

            results = self._detect(img)

            # Single target mode: click all instances by confidence (high to low)
            if self.auto_click and results:
                for target in results:
                    cx, cy = target.center
                    if self.region:
                        cx += self.region[0]
                        cy += self.region[1]
                    self.clicker.click(cx, cy)

            self.frame_ready.emit(img, results)

            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_update.emit(fps)
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _detect(self, img: np.ndarray) -> list[DetectionResult]:
        results = []
        enabled_targets = self.tm.get_enabled()

        for target in enabled_targets:
            if target.detection_mode in ("ai", "both"):
                detector = self.ai_detectors.get(target.id)
                if detector and detector.model:
                    dets = detector.detect(img, conf=self.conf)
                    for d in dets:
                        results.append(DetectionResult(
                            d.x1, d.y1, d.x2, d.y2, d.conf,
                            target.id, target.name, "ai", target.color
                        ))
            if target.detection_mode in ("opencv", "both"):
                dets = self.opencv_detector.detect(img, target_ids=[target.id])
                for d in dets:
                    results.append(DetectionResult(
                        d.x1, d.y1, d.x2, d.y2, d.conf,
                        target.id, target.name, "opencv", target.color
                    ))

        # Sort by confidence (high to low) for single target mode
        results.sort(key=lambda r: -r.conf)
        return results

    def stop(self):
        self.running = False
        self.wait()


class ParallelDetectThread(QThread):
    """Detection thread for multi-target mode - parallel execution per target."""
    frame_ready = Signal(np.ndarray, list)
    fps_update = Signal(float)

    def __init__(self, ai_detectors: dict[str, Detector], opencv_detector: OpenCVDetector,
                 target_manager: TargetManager, capture: ScreenCapture,
                 conf: float = 0.5, fps: int = 10, region=None,
                 auto_click: bool = False, use_priority: bool = False,
                 hide_window: bool = False, click_all_instances: bool = False):
        super().__init__()
        self.ai_detectors = ai_detectors
        self.opencv_detector = opencv_detector
        self.tm = target_manager
        self.capture = capture
        self.conf = conf
        self.target_fps = fps
        self.region = region
        self.auto_click = auto_click
        self.use_priority = use_priority
        self.hide_window = hide_window
        self.click_all_instances = click_all_instances
        self.running = False
        self.clicker = Clicker()

    def run(self):
        self.running = True
        interval = 1.0 / max(self.target_fps, 1)
        hwnd = 0
        if self.hide_window:
            hwnd = ScreenCapture.get_foreground_hwnd()

        while self.running:
            t0 = time.perf_counter()

            # Capture screen once
            if self.hide_window and hwnd:
                img = self.capture.capture_hide_window(hwnd)
            elif self.region:
                img = self.capture.capture_region(*self.region)
            else:
                img = self.capture.capture_full()

            # Run detection for all enabled targets in parallel
            results = self._detect_parallel(img)

            # Click logic
            if self.auto_click and results:
                self._click_targets(results)

            self.frame_ready.emit(img, results)

            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_update.emit(fps)
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _detect_parallel(self, img: np.ndarray) -> list[DetectionResult]:
        """Run detection for all targets in parallel using thread pool."""
        enabled_targets = self.tm.get_enabled()
        if not enabled_targets:
            return []

        all_results = []

        # Use thread pool to run detections in parallel
        with ThreadPoolExecutor(max_workers=len(enabled_targets)) as executor:
            futures = {
                executor.submit(
                    _detect_single_target,
                    target, self.ai_detectors, self.opencv_detector, img, self.conf
                ): target
                for target in enabled_targets
            }

            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception:
                    pass  # Skip failed detections

        # Sort by priority then confidence
        if self.use_priority:
            priority_map = {t.id: t.priority for t in self.tm.targets}
            all_results.sort(key=lambda r: (priority_map.get(r.target_id, 999), -r.conf))
        else:
            all_results.sort(key=lambda r: -r.conf)

        return all_results

    def _click_targets(self, results: list[DetectionResult]):
        """Click targets based on priority and click_all_instances setting."""
        if not results:
            return

        if self.use_priority:
            # Group by target_id, find highest priority
            priority_map = {t.id: t.priority for t in self.tm.targets}
            target_groups = {}
            for r in results:
                if r.target_id not in target_groups:
                    target_groups[r.target_id] = []
                target_groups[r.target_id].append(r)

            # Find highest priority target (lowest number)
            highest_priority_target = min(
                target_groups.keys(),
                key=lambda tid: priority_map.get(tid, 999)
            )

            if self.click_all_instances:
                # Click all instances of highest priority target
                targets_to_click = target_groups[highest_priority_target]
                targets_to_click.sort(key=lambda r: -r.conf)  # Sort by confidence
            else:
                # Click only the highest confidence instance
                targets_to_click = [max(
                    target_groups[highest_priority_target],
                    key=lambda r: r.conf
                )]
        else:
            # No priority, click highest confidence
            targets_to_click = [results[0]]

        # Perform clicks
        for target in targets_to_click:
            cx, cy = target.center
            if self.region:
                cx += self.region[0]
                cy += self.region[1]
            self.clicker.click(cx, cy)

    def stop(self):
        self.running = False
        self.wait()


# Column indices for target table
COL_CHECK = 0
COL_NAME = 1
COL_PRIORITY = 2
COL_MODEL = 3
COL_SET = 4


class DetectTab(QWidget):
    target_deleted = Signal()
    status_message = Signal(str)
    floating_window_requested = Signal(bool, bool)  # show, detecting

    def __init__(self, target_manager: TargetManager, opencv_detector: OpenCVDetector,
                 config: dict | None = None, sample_set_manager=None):
        super().__init__()
        self.tm = target_manager
        self.opencv_detector = opencv_detector
        self.ssm = sample_set_manager
        self.ai_detectors: dict[str, Detector] = {}
        self.capture = ScreenCapture()
        self.detect_thread = None
        self.config = config or {}
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Left: preview
        preview_group = QGroupBox("屏幕预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(8)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a16;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
            }
        """)
        preview_layout.addWidget(self.preview_label)

        # FPS counter
        fps_frame = QFrame()
        fps_layout = QHBoxLayout(fps_frame)
        fps_layout.setContentsMargins(8, 4, 8, 4)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #8888a8; font-size: 12px;")
        fps_layout.addWidget(self.fps_label)
        fps_layout.addStretch()
        preview_layout.addWidget(fps_frame)

        # Right: controls
        right_splitter = QSplitter(Qt.Vertical)

        # Target table
        target_group = QGroupBox("检测目标")
        target_layout = QVBoxLayout(target_group)
        target_layout.setSpacing(8)

        self.target_table = QTableWidget(0, 5)
        self.target_table.setHorizontalHeaderLabels(["", "名称", "优先级", "模型", "样本集"])
        self.target_table.setColumnWidth(COL_CHECK, 30)
        self.target_table.setColumnWidth(COL_NAME, 120)
        self.target_table.setColumnWidth(COL_PRIORITY, 60)
        self.target_table.setColumnWidth(COL_MODEL, 80)
        self.target_table.setColumnWidth(COL_SET, 100)
        self.target_table.horizontalHeader().setStretchLastSection(True)
        self.target_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.target_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.target_table.cellDoubleClicked.connect(self._on_target_cell_double_clicked)
        self.target_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.target_table.customContextMenuRequested.connect(self._on_target_context_menu)
        target_layout.addWidget(self.target_table)

        target_btn_layout = QHBoxLayout()
        target_btn_layout.setSpacing(8)

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self._select_all_targets)
        target_btn_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("全不选")
        self.deselect_all_btn.clicked.connect(self._deselect_all_targets)
        target_btn_layout.addWidget(self.deselect_all_btn)

        target_layout.addLayout(target_btn_layout)
        right_splitter.addWidget(target_group)

        # Settings
        settings_group = QGroupBox("检测设置")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)

        # Confidence slider
        conf_frame = QFrame()
        conf_layout = QHBoxLayout(conf_frame)
        conf_layout.setContentsMargins(0, 0, 0, 0)

        conf_label = QLabel("置信度:")
        conf_label.setStyleSheet("color: #8888a8;")
        conf_layout.addWidget(conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 99)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
        conf_layout.addWidget(self.conf_slider)

        self.conf_label = QLabel("0.50")
        self.conf_label.setStyleSheet("color: #a5b4fc; font-weight: 600;")
        conf_layout.addWidget(self.conf_label)

        settings_layout.addWidget(conf_frame)

        # FPS setting
        fps_frame = QFrame()
        fps_layout = QHBoxLayout(fps_frame)
        fps_layout.setContentsMargins(0, 0, 0, 0)

        fps_setting_label = QLabel("目标帧率:")
        fps_setting_label.setStyleSheet("color: #8888a8;")
        fps_layout.addWidget(fps_setting_label)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        fps_layout.addWidget(self.fps_spin)

        settings_layout.addWidget(fps_frame)

        # Checkboxes
        self.priority_cb = QCheckBox("启用优先级 (数字越小优先级越高)")
        settings_layout.addWidget(self.priority_cb)

        self.auto_click_cb = QCheckBox("自动点击")
        settings_layout.addWidget(self.auto_click_cb)

        # Multi-target click mode
        multi_click_frame = QFrame()
        multi_click_layout = QHBoxLayout(multi_click_frame)
        multi_click_layout.setContentsMargins(0, 0, 0, 0)

        multi_click_label = QLabel("多目标同优先级点击:")
        multi_click_label.setStyleSheet("color: #8888a8;")
        multi_click_layout.addWidget(multi_click_label)

        self.multi_click_combo = QComboBox()
        self.multi_click_combo.addItems(["只点击一个", "点击所有实例"])
        self.multi_click_combo.setToolTip("当多个相同优先级的目标被检测到时的点击策略")
        multi_click_layout.addWidget(self.multi_click_combo)

        settings_layout.addWidget(multi_click_frame)

        self.floating_auto_switch_cb = QCheckBox("开始检测时自动切换到悬浮窗")
        self.floating_auto_switch_cb.setToolTip("勾选后点击开始检测会自动进入悬浮窗模式")
        settings_layout.addWidget(self.floating_auto_switch_cb)

        right_splitter.addWidget(settings_group)

        # Control buttons
        btn_group = QWidget()
        btn_layout = QVBoxLayout(btn_group)
        btn_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("success_btn")
        self.start_btn.clicked.connect(self._toggle_detection)
        btn_row.addWidget(self.start_btn)

        self.screenshot_btn = QPushButton("截图")
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        btn_row.addWidget(self.screenshot_btn)

        self.minimize_btn = QPushButton("缩小到悬浮窗")
        self.minimize_btn.clicked.connect(self._minimize_to_floating)
        btn_row.addWidget(self.minimize_btn)

        btn_layout.addLayout(btn_row)

        # Results section
        results_label = QLabel("检测结果:")
        results_label.setStyleSheet("color: #8888a8; font-size: 12px; margin-top: 8px;")
        btn_layout.addWidget(results_label)

        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(["目标", "来源", "置信度", "位置", "大小"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        btn_layout.addWidget(self.result_table)

        right_splitter.addWidget(btn_group)

        right_splitter.setSizes([250, 180, 200])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(preview_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([700, 300])
        layout.addWidget(main_splitter)

        # Keyboard shortcut for screenshot
        self.screenshot_shortcut = QShortcut(QKeySequence("F12"), self)
        self.screenshot_shortcut.activated.connect(self._take_screenshot)

    def refresh_targets(self):
        self.target_table.setRowCount(0)
        for t in self.tm.targets:
            row = self.target_table.rowCount()
            self.target_table.insertRow(row)

            # Checkbox
            cb = QCheckBox()
            cb.setChecked(t.enabled)
            cb.stateChanged.connect(lambda state, tid=t.id: self._on_check_changed(tid, state))
            self.target_table.setCellWidget(row, COL_CHECK, cb)

            # Name
            name_item = QTableWidgetItem(t.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, t.id)
            self.target_table.setItem(row, COL_NAME, name_item)

            # Priority
            pri_item = QTableWidgetItem(str(t.priority))
            pri_item.setTextAlignment(Qt.AlignCenter)
            self.target_table.setItem(row, COL_PRIORITY, pri_item)

            # Model type
            model_label = _get_model_label(t)
            model_item = QTableWidgetItem(model_label)
            model_item.setFlags(model_item.flags() & ~Qt.ItemIsEditable)
            model_item.setForeground(QBrush(QColor(136, 136, 168)))
            self.target_table.setItem(row, COL_MODEL, model_item)

            # Sample set
            set_name = ""
            if t.set_id and self.ssm:
                ss = self.ssm.get(t.set_id)
                set_name = ss.name if ss else ""
            set_item = QTableWidgetItem(set_name)
            set_item.setFlags(set_item.flags() & ~Qt.ItemIsEditable)
            set_item.setForeground(QBrush(QColor(136, 136, 168)))
            self.target_table.setItem(row, COL_SET, set_item)

        self._load_ai_models()
        self._load_opencv_templates()

    def _on_check_changed(self, target_id: str, state: int):
        target = self.tm.get(target_id)
        if target:
            target.enabled = (state == Qt.Checked)

    def _on_target_cell_double_clicked(self, row: int, col: int):
        if row < 0 or row >= len(self.tm.targets):
            return
        target = self.tm.targets[row]

        if col == COL_NAME:
            new_name, ok = QInputDialog.getText(self, "重命名目标", "新名称:", text=target.name)
            if ok and new_name:
                target.name = new_name
                self.tm.save()
                self.refresh_targets()
        elif col == COL_PRIORITY:
            try:
                self.target_table.itemChanged.disconnect(self._on_priority_changed)
            except (RuntimeError, TypeError):
                pass
            self.target_table.itemChanged.connect(self._on_priority_changed)

    def _on_priority_changed(self, item: QTableWidgetItem):
        if item.column() != COL_PRIORITY:
            return
        row = item.row()
        if row < 0 or row >= len(self.tm.targets):
            return
        try:
            new_pri = int(item.text())
            self.tm.targets[row].priority = new_pri
            self.tm.save()
        except ValueError:
            pass
        self.target_table.itemChanged.disconnect(self._on_priority_changed)

    def _on_target_context_menu(self, pos):
        row = self.target_table.rowAt(pos.y())
        if row < 0 or row >= len(self.tm.targets):
            return
        menu = QMenu(self)
        delete_action = menu.addAction("删除目标（彻底删除文件）")
        action = menu.exec(self.target_table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self._delete_target_at(row)

    def _delete_target_at(self, row: int):
        target = self.tm.targets[row]
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
        import shutil
        if target.data_dir and os.path.isdir(target.data_dir):
            shutil.rmtree(target.data_dir, ignore_errors=True)
        if target.model_path and os.path.isfile(target.model_path):
            try:
                os.remove(target.model_path)
            except OSError:
                pass
        self.tm.remove(target.id)
        self.tm.save()
        self.ai_detectors.pop(target.id, None)
        self.refresh_targets()
        self.target_deleted.emit()

    def _load_ai_models(self):
        for t in self.tm.targets:
            if t.detection_mode in ("ai", "both") and t.model_path and t.id not in self.ai_detectors:
                path = os.path.abspath(t.model_path)
                for attempt in range(3):
                    det = Detector()
                    try:
                        det.load(path)
                        self.ai_detectors[t.id] = det
                        print(f"[DetectTab] AI 模型已加载: {t.name}, 路径: {path}")
                        break
                    except Exception as e:
                        print(f"[DetectTab] AI 模型加载失败 {t.name} (尝试 {attempt+1}/3): {e}")

    def _load_opencv_templates(self):
        """Load all OpenCV templates into the detector with retry logic."""
        for t in self.tm.targets:
            if t.detection_mode in ("opencv", "both") and t.template_path:
                path = os.path.abspath(t.template_path)
                if not os.path.exists(path):
                    print(f"[DetectTab] 模板文件不存在: {path}")
                    continue
                for attempt in range(3):
                    if self.opencv_detector.load_template(t.id, path, t.match_threshold):
                        break
                    print(f"[DetectTab] 模板加载重试 {attempt+1}/3: {t.name}")

    def _select_all_targets(self):
        for row in range(self.target_table.rowCount()):
            cb = self.target_table.cellWidget(row, COL_CHECK)
            if cb:
                cb.setChecked(True)

    def _deselect_all_targets(self):
        for row in range(self.target_table.rowCount()):
            cb = self.target_table.cellWidget(row, COL_CHECK)
            if cb:
                cb.setChecked(False)

    def _get_selected_target_ids(self) -> list[str]:
        ids = []
        for row in range(self.target_table.rowCount()):
            item = self.target_table.item(row, COL_NAME)
            if item:
                tid = item.data(Qt.UserRole)
                cb = self.target_table.cellWidget(row, COL_CHECK)
                if cb and cb.isChecked():
                    ids.append(tid)
        return ids

    def _toggle_detection(self):
        if self.detect_thread and self.detect_thread.running:
            self.detect_thread.stop()
            self.detect_thread = None
            self.start_btn.setText("开始检测")
            self.start_btn.setObjectName("success_btn")
            self.start_btn.style().unpolish(self.start_btn)
            self.start_btn.style().polish(self.start_btn)
            if self.floating_auto_switch_cb.isChecked():
                self.floating_window_requested.emit(False, False)
        else:
            selected = set(self._get_selected_target_ids())
            if not selected:
                QMessageBox.warning(self, "提示", "请先选择至少一个检测目标")
                return

            # Check if more than 4 targets selected
            if len(selected) > 4:
                reply = QMessageBox.question(
                    self, "确认",
                    f"已选择 {len(selected)} 个目标，同时运行可能导致性能问题。\n是否继续？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            for t in self.tm.targets:
                t.enabled = t.id in selected
            self.tm.save()

            hide_window = self.config.get("screenshot_hide_window", True)

            # Choose between single and parallel mode
            if len(selected) == 1:
                # Single target mode - sequential
                self.detect_thread = SingleDetectThread(
                    ai_detectors=self.ai_detectors,
                    opencv_detector=self.opencv_detector,
                    target_manager=self.tm,
                    capture=self.capture,
                    conf=self.conf_slider.value() / 100,
                    fps=self.fps_spin.value(),
                    auto_click=self.auto_click_cb.isChecked(),
                    use_priority=self.priority_cb.isChecked(),
                    hide_window=hide_window,
                )
            else:
                # Multi-target mode - parallel
                click_all = self.multi_click_combo.currentIndex() == 1
                self.detect_thread = ParallelDetectThread(
                    ai_detectors=self.ai_detectors,
                    opencv_detector=self.opencv_detector,
                    target_manager=self.tm,
                    capture=self.capture,
                    conf=self.conf_slider.value() / 100,
                    fps=self.fps_spin.value(),
                    auto_click=self.auto_click_cb.isChecked(),
                    use_priority=self.priority_cb.isChecked(),
                    hide_window=hide_window,
                    click_all_instances=click_all,
                )

            self.detect_thread.frame_ready.connect(self._on_frame)
            self.detect_thread.fps_update.connect(self._on_fps)
            self.detect_thread.start()
            self.start_btn.setText("停止检测")
            self.start_btn.setObjectName("danger_btn")
            self.start_btn.style().unpolish(self.start_btn)
            self.start_btn.style().polish(self.start_btn)
            if self.floating_auto_switch_cb.isChecked():
                self.floating_window_requested.emit(True, True)

    @Slot(np.ndarray, list)
    def _on_frame(self, img, detections):
        for det in detections:
            color_hex = det.color
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (b, g, r), 2)
            label = f"{det.target_name}[{det.source}] {det.conf:.2f}"
            cv2.putText(img, label, (det.x1, det.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)

        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

        self.result_table.setRowCount(len(detections))
        for i, det in enumerate(detections):
            self.result_table.setItem(i, 0, QTableWidgetItem(det.target_name))
            self.result_table.setItem(i, 1, QTableWidgetItem(det.source))
            self.result_table.setItem(i, 2, QTableWidgetItem(f"{det.conf:.2f}"))
            self.result_table.setItem(i, 3, QTableWidgetItem(f"({det.x1},{det.y1})"))
            self.result_table.setItem(i, 4, QTableWidgetItem(f"{det.x2-det.x1}x{det.y2-det.y1}"))

    @Slot(float)
    def _on_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _take_screenshot(self):
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

        # Always save screenshot to file
        default_path = self.config.get("screenshot_default_path", "data/screenshots")
        saved_path = ScreenCapture.save_screenshot(img, default_path, "detect")
        self.status_message.emit(f"截图已保存: {saved_path}")

    def _minimize_to_floating(self):
        """Switch to floating window mode."""
        detecting = self.detect_thread is not None and self.detect_thread.running
        self.floating_window_requested.emit(True, detecting)

    def update_config(self, config: dict):
        self.config = config

    def showEvent(self, event):
        """Auto-refresh targets when the tab becomes visible."""
        super().showEvent(event)
        self.refresh_targets()
