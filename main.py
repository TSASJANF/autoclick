import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    # State is saved in MainWindow.closeEvent
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
