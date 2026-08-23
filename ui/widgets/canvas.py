"""共享画布组件：标注页与 OpenCV 模板页共用的图像画布。

合并历史重复：ui/annotate_tab.ImageCanvas、ui/opencv_tab.TemplateCanvas、
两份 _DrawOverlay、两份 _ColorDelegate 共 5 个类合并为 3 个。

关键修复：标注框归属键从「颜色字符串」改为 target_id（两个目标颜色相
同时旧版会互相污染标注数据），颜色只负责显示。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyledItemDelegate, QWidget

from core.annotations import Box

from .image_utils import hex_to_rgb


class ColorDelegate(QStyledItemDelegate):
    """让列表项的背景色在选中时依然可见。"""

    def paint(self, painter, option, index):
        color = index.data(Qt.BackgroundRole)
        if color:
            painter.fillRect(option.rect, color)
        else:
            super().paint(painter, option, index)


class _DrawOverlay(QWidget):
    """拖拽预览覆盖层 —— 避免拖动过程中整图重绘。"""

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

    def begin(self, start: QPoint, end: QPoint):
        self._start, self._end, self._active = start, end, True
        self.update()

    def move_to(self, end: QPoint):
        self._end = end
        self.update()

    def finish(self):
        self._active = False
        self.update()

    def paintEvent(self, event):
        if not self._active:
            return
        painter = QPainter(self)
        painter.setPen(QPen(self._color, 2, Qt.DashLine))
        painter.drawRect(QRect(self._start, self._end).normalized())
        painter.end()


class ImageCanvas(QLabel):
    """显示图像 + 拖框交互的画布。

    - 拖拽完成发出 region_drawn(QRect)（图像坐标系）；
    - 支持叠加标注框（set_boxes 批量刷新，避免逐框全图重绘）；
    - 支持持久化选择框（set_selection，模板页用）。
    """

    region_drawn = Signal(QRect)

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
        self._boxes: list[Box] = []
        self._selection = QRect()
        self._selection_color = QColor("#2ecc71")
        self._drawing = False
        self._start = QPoint()
        self._end = QPoint()
        self._overlay = _DrawOverlay(self)

    # ── Qt 事件 ───────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())

    def sizeHint(self):
        return self.minimumSize()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._pixmap:
            self._drawing = True
            self._start = self._map_to_image(ev.pos())
            self._end = self._start
            self._overlay.begin(ev.pos(), ev.pos())

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._end = self._map_to_image(ev.pos())
            self._overlay.move_to(ev.pos())

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            self._end = self._map_to_image(ev.pos())
            self._overlay.finish()
            rect = QRect(self._start, self._end).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.region_drawn.emit(rect)
            self._render()

    # ── 内容设置 ──────────────────────────────────────────────────────

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._boxes = []
        self._selection = QRect()
        self._render()

    def set_draw_color(self, color_hex: str):
        self._selection_color = QColor(*hex_to_rgb(color_hex))
        self._overlay.set_color(QColor(*hex_to_rgb(color_hex)))

    def set_boxes(self, boxes: list[Box]):
        """批量设置标注框（单次重绘）。"""
        self._boxes = list(boxes)
        self._render()

    def add_box(self, box: Box):
        self._boxes.append(box)
        self._render()

    def get_boxes(self) -> list[Box]:
        return list(self._boxes)

    def set_selection(self, rect: QRect):
        self._selection = rect
        self._render()

    # ── 渲染 ──────────────────────────────────────────────────────────

    def _render(self):
        if self._pixmap is None:
            return
        canvas = self._pixmap.copy()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        for box in self._boxes:
            rect = QRect(box.x1, box.y1, box.width, box.height)
            painter.setPen(QPen(QColor(*hex_to_rgb(box.color)), 2))
            painter.drawRect(rect)
            painter.setFont(QFont("Segoe UI", 10))
            text = box.label or f"cls:{box.cls_id}"
            painter.drawText(rect.topLeft() + QPoint(2, -4), text)

        if not self._selection.isNull():
            painter.setPen(QPen(self._selection_color, 2))
            painter.drawRect(self._selection)

        painter.end()
        scaled = canvas.scaled(self.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        super().setPixmap(scaled)

    def _map_to_image(self, widget_pos: QPoint) -> QPoint:
        """部件坐标 -> 图像坐标（等比缩放 + 居中偏移换算）。"""
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
