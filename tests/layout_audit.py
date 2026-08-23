"""确定性布局审计：用字体度量找出所有被截断的文字（开发工具）。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QGroupBox,
    QLabel, QPushButton, QRadioButton, QTableWidget, QTabWidget, QListWidget,
)


def audit_widget(w, path, issues):
    if isinstance(w, QGroupBox):
        title = w.title()
        if title:
            fm = w.fontMetrics()
            need = fm.horizontalAdvance(title) + 24
            if need > w.width() + 2:
                issues.append(f"[QGroupBox] {path}: 标题 '{title}' 需~{need}px 实际 {w.width()}px")
    fm = w.fontMetrics()
    if isinstance(w, QLabel):
        text = w.text()
        if text and not w.wordWrap():
            need = fm.horizontalAdvance(text)
            if need > w.width() + 2:
                issues.append(f"[QLabel] {path}: '{text}' 需 {need}px 实际 {w.width()}px")
    elif isinstance(w, (QCheckBox, QRadioButton)):
        text = w.text()
        if text:
            need = w.sizeHint().width()
            if need > w.width() + 2:
                issues.append(f"[{w.__class__.__name__}] {path}: '{text}' 需 {need}px 实际 {w.width()}px")
    elif isinstance(w, QPushButton):
        text = w.text()
        if text:
            need = w.sizeHint().width()
            if need > w.width() + 2:
                issues.append(f"[QPushButton] {path}: '{text}' 需 {need}px 实际 {w.width()}px")
    elif isinstance(w, QComboBox):
        text = w.currentText()
        if text:
            need = fm.horizontalAdvance(text) + 40  # 箭头区
            if need > w.width() + 2:
                issues.append(f"[QComboBox] {path}: '{text}' 需~{need}px 实际 {w.width()}px")


def audit_table(t, path, issues):
    fm = t.fontMetrics()
    for row in range(t.rowCount()):
        for col in range(t.columnCount()):
            item = t.item(row, col)
            if item is None:
                continue
            text = item.text()
            need = fm.horizontalAdvance(text) + 14
            actual = t.columnWidth(col)
            if need > actual + 2:
                issues.append(f"[表格] {path} 行{row}列{col}: '{text}' 需~{need}px 列宽{actual}px")


def walk(w, path, issues):
    audit_widget(w, path, issues)
    if isinstance(w, QTableWidget):
        audit_table(w, path, issues)
    for child in w.findChildren(type(w)) if False else w.findChildren(QLabel) + \
            w.findChildren(QCheckBox) + w.findChildren(QPushButton) + \
            w.findChildren(QComboBox) + w.findChildren(QTableWidget):
        pass
    return issues


def audit_root(root, name, issues, counter):
    def rec(w, path):
        counter[0] += 1
        audit_widget(w, path, issues)
        if isinstance(w, QTableWidget):
            audit_table(w, path, issues)
        for ch in w.children():
            if ch.isWidgetType() and ch.parent() is w:
                rec(ch, f"{path}/{ch.objectName() or ch.__class__.__name__}")
    rec(root, name)
    counter[0] += 1


def main() -> int:
    import main as app_main
    app_main.setup_logging()
    app = QApplication(sys.argv)
    from ui.theme import apply_theme
    apply_theme(app)

    from ui.main_window import MainWindow
    w = MainWindow()
    w.resize(int(sys.argv[1]) if len(sys.argv) > 1 else 1600, int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    w.show()
    for _ in range(30):
        app.processEvents()

    names = ["检测", "标注", "训练", "OpenCV模板", "设置"]
    total = 0
    for i, name in enumerate(names):
        w.switchTo(w.interface_order[i])
        for _ in range(30):
            app.processEvents()
        issues = []
        counter = [0]
        audit_root(w.interface_order[i], name, issues, counter)
        print(f"   (审计覆盖 {counter[0]} 个控件)")
        print(f"== {name}: {len(issues)} 处截断")
        for msg in issues:
            print("   " + msg)
        total += len(issues)
    print(f"TOTAL: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
