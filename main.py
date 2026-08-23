"""应用入口：日志初始化、全局样式、QApplication 启动。

相对历史版本的改动：
1. 配置 logging（控制台 + logs/app.log 轮转文件）——旧版全是 print，
   排障只能盯控制台；
2. 加载 ui/style.qss —— 该 633 行设计系统文件一直存在却从未被任何
   代码引用（primary_btn/success_btn/danger_btn 等 objectName 选择器
   全部失效），现在真正接上；
3. 捕获顶层异常写入日志，而不是让 Qt 静默吞掉。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from core.paths import APP_ROOT


def setup_logging() -> None:
    log_dir = APP_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=2 * 1024 * 1024,
        backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 三方库降噪
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def load_stylesheet(app) -> None:
    qss_path = Path(__file__).parent / "ui" / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


def main() -> int:
    setup_logging()
    logger = logging.getLogger("main")

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    load_stylesheet(app)

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    try:
        return app.exec()
    except Exception:
        logger.exception("应用发生未捕获异常")
        raise
    finally:
        # 状态由 MainWindow.closeEvent 保存
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
