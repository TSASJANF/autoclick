"""应用主题：PySide6-Fluent-Widgets（Windows 11 Fluent Design）。

历史：此前依次试过自定义 style.qss（半成品，加载即压坏布局）与
qt-material（纯样式表，无法解决布局结构问题）。现改用 Fluent 组件库：
自带卡片 / 滚动区 / 导航 / 输入控件，视觉统一且支持内容滚动。

语义色（primary/success/danger）通过一层薄补充样式覆盖在 Fluent 主题
之上，供 dialogs 与历史 objectName 沿用。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ACCENT = "#5b8ff9"          # Fluent 蓝（主色）
SUCCESS = "#22c55e"
DANGER = "#ef4444"

# 叠加的语义按钮样式（qfluentwidgets 全局样式之外的补充）
SUPPLEMENT_QSS = f"""
QPushButton#primary_btn {{
    background-color: {ACCENT}; color: white; font-weight: 600;
    border: none; border-radius: 6px; padding: 7px 16px;
}}
QPushButton#primary_btn:hover {{ background-color: #6ea0ff; }}
QPushButton#primary_btn:pressed {{ background-color: #4a7fdb; }}
QPushButton#primary_btn:disabled {{ background-color: #3a3a4a; color: #7a7a8a; }}

QPushButton#success_btn {{
    background-color: {SUCCESS}; color: white; font-weight: 600;
    border: none; border-radius: 6px; padding: 7px 16px;
}}
QPushButton#success_btn:hover {{ background-color: #4ade80; }}
QPushButton#success_btn:pressed {{ background-color: #16a34a; }}
QPushButton#success_btn:disabled {{ background-color: #2f6b48; color: #9fbfae; }}

QPushButton#danger_btn {{
    background-color: {DANGER}; color: white; font-weight: 600;
    border: none; border-radius: 6px; padding: 7px 16px;
}}
QPushButton#danger_btn:hover {{ background-color: #f87171; }}
QPushButton#danger_btn:pressed {{ background-color: #dc2626; }}
QPushButton#danger_btn:disabled {{ background-color: #6b2f2f; color: #bfa0a0; }}

QLabel#heading {{ font-size: 18px; font-weight: 600; color: #e8e8f0; }}
"""


def apply_theme(app) -> None:
    """应用 Fluent 暗色主题 + 主色 + 语义按钮补充样式。"""
    from qfluentwidgets import setTheme, setThemeColor, Theme
    setTheme(Theme.DARK)
    setThemeColor(ACCENT)
    app.setStyleSheet(app.styleSheet() + SUPPLEMENT_QSS)
    logger.info("主题已应用: PySide6-Fluent-Widgets (accent=%s)", ACCENT)
