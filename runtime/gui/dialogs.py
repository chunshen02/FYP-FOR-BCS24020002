"""
dialogs.py

Modal dialogs used by the main window.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)
from PyQt6.QtCore import Qt


class SaveSessionDialog(QDialog):
    """Asks the user whether to save or discard the just-finished session."""

    def __init__(self, duration: str, alert_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save session results?")
        self.setModal(True)
        self.setFixedWidth(320)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #dddddd;
            }
            """
        )
        self._build_ui(duration, alert_count)

    def _build_ui(self, duration: str, alert_count: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Save session results?")
        title.setStyleSheet("font-size: 14px; font-weight: 500; color: #ffffff;")
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333333;")
        layout.addWidget(line)

        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(6)

        def make_row(label: str, value: str) -> QHBoxLayout:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 12px; color: #888888;")
            val = QLabel(value)
            val.setStyleSheet("font-size: 12px; color: #dddddd;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            return row

        stats_layout.addLayout(make_row("Duration", duration))
        stats_layout.addLayout(make_row("Alerts triggered", str(alert_count)))

        result_row = QHBoxLayout()
        result_lbl = QLabel("Result")
        result_lbl.setStyleSheet("font-size: 12px; color: #888888;")
        result_badge = QLabel(
            "Deepfake detected" if alert_count > 0 else "Clean"
        )
        if alert_count > 0:
            result_badge.setStyleSheet(
                """
                font-size: 11px; font-weight: 500;
                background-color: #FCEBEB; color: #A32D2D;
                border-radius: 3px; padding: 2px 8px;
                """
            )
        else:
            result_badge.setStyleSheet(
                """
                font-size: 11px; font-weight: 500;
                background-color: #EAF3DE; color: #3B6D11;
                border-radius: 3px; padding: 2px 8px;
                """
            )
        result_row.addWidget(result_lbl)
        result_row.addStretch()
        result_row.addWidget(result_badge)
        stats_layout.addLayout(result_row)

        layout.addLayout(stats_layout)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #333333;")
        layout.addWidget(line2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        discard_btn = QPushButton("Discard")
        discard_btn.setFixedHeight(34)
        discard_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: 1px solid #555555;
                border-radius: 6px;
                color: #aaaaaa;
                font-size: 12px;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #2a2a2a; }
            """
        )
        discard_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Save")
        save_btn.setFixedHeight(34)
        save_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #E1F5EE;
                border: 1px solid #1D9E75;
                border-radius: 6px;
                color: #085041;
                font-size: 12px;
                font-weight: 500;
                padding: 0 20px;
            }
            QPushButton:hover { background-color: #9FE1CB; }
            """
        )
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)

        btn_row.addWidget(discard_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
