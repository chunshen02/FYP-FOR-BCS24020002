import sys
from PyQt6.QtWidgets import QApplication
from runtime.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Deepfake Detector")
    app.setApplicationVersion("1.0.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
