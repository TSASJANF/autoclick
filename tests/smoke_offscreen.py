"""离屏冒烟测试：完整构建主窗口并断言核心装配（不进入人工交互）。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer


def main() -> int:
    import main as app_main
    app_main.setup_logging()
    app = QApplication(sys.argv)
    from ui.theme import apply_theme
    apply_theme(app)

    from ui.main_window import MainWindow
    w = MainWindow()
    w.show()

    result = {"ok": False, "msg": ""}

    def check():
        try:
            assert w.stackedWidget.count() == 5, f"界面数: {w.stackedWidget.count()}"
            assert len(w.ctx.tm.targets) >= 1, "目标未加载"
            assert w.ctx.ssm.get_all(), "样本集未加载"
            assert w.detect_tab.target_table.rowCount() == len(w.ctx.tm.targets)
            assert w.annotate_tab._current_set is not None
            assert w.train_tab.target_combo.count() >= 1
            for t in w.ctx.tm.targets:
                print(f"OK-TARGET [{t.name}] mode={t.detection_mode} "
                      f"template={w.ctx.opencv_detector.has_template(t.id)} "
                      f"ai={t.id in w.ctx.ai_detectors} "
                      f"failed={t.id in w.ctx.failed_models}")
            result["ok"] = True
        except Exception as e:  # noqa: BLE001
            result["msg"] = str(e)
            import traceback
            traceback.print_exc()
        finally:
            app.quit()

    t = QTimer()
    t.setSingleShot(True)
    t.timeout.connect(check)
    t.start(6000)
    app.exec()

    if result["ok"]:
        print("=== SMOKE TEST PASSED ===")
        return 0
    print(f"=== SMOKE TEST FAILED: {result['msg']} ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
