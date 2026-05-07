"""
session_log.py

Read-only QTextEdit that displays a timestamped session event log.
"""

from datetime import datetime
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import pyqtSlot


class SessionLog(QTextEdit):
    """
    Displays a timestamped log of detection events.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setStyleSheet(
            "background-color: #111; color: #ccc; font-family: monospace; font-size: 11px;"
        )
        self.log_event("SESSION", "Deepfake Detector started.")

    @pyqtSlot(str, str)
    def log_event(self, event_type: str, detail: str):
        """Append a timestamped entry and scroll to bottom."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colour_map = {
            "ALERT": "#ff4444",
            "DISMISS": "#44ff44",
            "SESSION": "#aaaaaa",
            "INFO": "#cccccc",
            "ERROR": "#ff8800",
        }
        colour = colour_map.get(event_type, "#cccccc")
        entry = (
            f'<span style="color:#666;">[{timestamp}]</span> '
            f'<span style="color:{colour};">[{event_type}]</span> '
            f'<span style="color:#ccc;">{detail}</span>'
        )
        self.append(entry)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
