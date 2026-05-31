import os
import sys

# Suppress MediaPipe / TFLite internal telemetry and verbose logs before any import
os.environ.setdefault("GLOG_minloglevel", "3")       # glog: suppress INFO/WARNING/ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # TFLite: suppress all C++ logs
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")  # avoid GPU probe logs on CPU-only machines

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
