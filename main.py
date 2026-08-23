"""应用入口：日志初始化、单实例锁、Material 主题、QApplication 启动。

相对历史版本的改动：
1. logging（控制台 + logs/app.log 轮转 + Qt 消息路由）——旧版全是
   print，GUI 进程无控制台时所有可诊断信息实际全部丢失；
2. 单实例锁（SEC-02）：双开进程会相互覆盖 targets.json/sets_index.json
   （后写者抹掉先写者的全部变更），现以 QLocalServer 拒绝第二实例；
3. 主题：PySide6-Fluent-Widgets（Windows 11 Fluent Design）+ 语义按钮
   补充样式（ui/theme.py），取代此前的自定义 style.qss 与 qt-material；
4. 启动时清理过期截图（SEC-03）。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from core.paths import APP_ROOT

_SINGLE_INSTANCE_KEY = "auto-clicker-single-instance"


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

    # Qt 内部警告（qpainter/qimage 等）路由到日志，GUI 下不再丢失
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    def qt_message_handler(mode, _context, message):
        qt_logger = logging.getLogger("qt")
        if mode == QtMsgType.QtWarningMsg:
            qt_logger.warning(message)
        elif mode == QtMsgType.QtCriticalMsg or mode == QtMsgType.QtFatalMsg:
            qt_logger.critical(message)
        else:
            qt_logger.info(message)

    qInstallMessageHandler(qt_message_handler)


def acquire_single_instance_lock(app):
    """单实例锁（SEC-02）。成功返回持有的 QLocalServer，已有实例返回 None。"""
    from PySide6.QtNetwork import QLocalServer, QLocalSocket

    probe = QLocalSocket()
    probe.connectToServer(_SINGLE_INSTANCE_KEY)
    if probe.waitForConnected(200):
        probe.disconnectFromServer()
        return None   # 已有实例在运行
    QLocalServer.removeServer(_SINGLE_INSTANCE_KEY)   # 清理崩溃残留
    server = QLocalServer()
    server.listen(_SINGLE_INSTANCE_KEY)
    return server


def main() -> int:
    setup_logging()
    logger = logging.getLogger("main")

    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)

    from ui.theme import apply_theme
    apply_theme(app)

    lock = acquire_single_instance_lock(app)
    if lock is None:
        QMessageBox.warning(None, "程序已在运行",
                            "自动点击工具已在运行中。\n双开会相互覆盖配置并导致数据丢失。")
        return 0

    # 启动清理：过期截图（SEC-03）
    from core.data_lifecycle import cleanup_screenshots
    try:
        cleanup_screenshots()
    except Exception:
        logger.exception("截图清理失败（不影响启动）")

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    try:
        return app.exec()
    except Exception:
        logger.exception("应用发生未捕获异常")
        raise


if __name__ == "__main__":
    sys.exit(main())
