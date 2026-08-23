"""离屏渲染各页签截图，用于人工目检布局（开发工具，不参与 CI）。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication


def main() -> int:
    import main as app_main
    app_main.setup_logging()
    app = QApplication(sys.argv)

    from ui.theme import apply_theme
    apply_theme(app)

    from ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()

    out_dir = os.path.join("logs", "ui_shots")

    # 让后台模型加载跑一会儿，表格状态更接近真实
    for _ in range(40):
        app.processEvents()

    names = ["detect", "annotate", "train", "opencv", "settings"]
    for i, name in enumerate(names):
        w.tabs.setCurrentIndex(i)
        for _ in range(20):
            app.processEvents()
        pix = w.grab()
        path = os.path.join(out_dir, f"{name}.png")
        pix.save(path)
        print(f"saved {path} ({pix.width()}x{pix.height()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
