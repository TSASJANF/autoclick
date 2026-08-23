"""应用主题：qt-material（Material Design, MIT）+ 少量补充样式。

历史背景：项目里那份 633 行的 style.qss 从未被加载过（孤儿文件）；
重构时接入后暴露出它与实际控件不匹配——全局字体/内边距改变挤压了
所有按旧尺寸写死的布局，文字截断、视觉破碎。

现改为成熟组件主题库 qt-material，并只保留一层很薄的补充样式：
- 语义按钮（primary/success/danger objectName，业务代码已广泛使用）；
- 次要文字色、标题、分隔线等设计令牌。

注意：qt_material 必须在 PySide 之后导入。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 设计令牌（与库内各页签内联样式保持同族）
ACCENT = "#6366f1"          # 主色（靛蓝）
ACCENT_HOVER = "#818cf8"
SUCCESS = "#22c55e"
SUCCESS_HOVER = "#4ade80"
DANGER = "#ef4444"
DANGER_HOVER = "#f87171"
TEXT_MUTED = "#8888a8"
BORDER = "#2a2a4a"
SURFACE_DARK = "#0a0a16"    # 画布深色底

# 叠加在 qt-material 之上的补充样式（语义按钮 + 设计令牌）。
# 必须拼接在主题样式表之后整体设置——QApplication.setStyleSheet 是
# 整体替换而非叠加。
SUPPLEMENT_QSS = f"""
QPushButton#primary_btn {{
    background-color: {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton#primary_btn:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primary_btn:disabled {{ background-color: #4b4b6b; color: #9999b5; }}

QPushButton#success_btn {{
    background-color: {SUCCESS};
    color: white;
    font-weight: 600;
}}
QPushButton#success_btn:hover {{ background-color: {SUCCESS_HOVER}; }}
QPushButton#success_btn:disabled {{ background-color: #2f6b48; color: #9fbfae; }}

QPushButton#danger_btn {{
    background-color: {DANGER};
    color: white;
    font-weight: 600;
}}
QPushButton#danger_btn:hover {{ background-color: {DANGER_HOVER}; }}
QPushButton#danger_btn:disabled {{ background-color: #6b2f2f; color: #bfa0a0; }}

QLabel#heading {{
    font-size: 16px;
    font-weight: 600;
    color: #e8e8f0;
}}

QLabel#muted {{ color: {TEXT_MUTED}; }}
QFrame#hline {{
    color: {BORDER};
    max-height: 1px;
    border: none;
    border-top: 1px solid {BORDER};
}}
"""


def apply_theme(app) -> None:
    """应用 Material 主题并叠加语义组件样式。"""
    from qt_material import apply_stylesheet

    extra = {
        # 中文优先字体，回退 Segoe UI
        "font_family": "Microsoft YaHei UI",
        # 紧凑密度：表格/表单类工具界面默认密度偏大，-1 档更合适
        "density_scale": "-1",
        "primaryColor": ACCENT,
        "primaryLightColor": ACCENT_HOVER,
    }
    apply_stylesheet(app, theme="dark_cyan.xml", extra=extra)
    app.setStyleSheet(app.styleSheet() + SUPPLEMENT_QSS)
    logger.info("主题已应用: qt-material dark_cyan (accent=%s)", ACCENT)
