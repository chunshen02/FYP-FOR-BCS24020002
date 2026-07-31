"""
session_log.py

Read-only QTextEdit that displays a timestamped session event log.
"""

from datetime import datetime
from PyQt6.QtWidgets import QTextEdit, QFrame
from PyQt6.QtCore import pyqtSlot


class SessionLog(QTextEdit):
    """
    Displays a timestamped log of detection events.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.log_event("SESSION", "Deepfake Detector started.")

    @pyqtSlot(str, str)
    def log_event(self, event_type: str, detail: str):
        """Append a timestamped entry and scroll to bottom."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colour_map = {
            "ALERT": "#F87171",
            "DISMISS": "#4ADE80",
            "SESSION": "#8A93A3",
            "INFO": "#C4C9D4",
            "ERROR": "#FBBF24",
        }
        colour = colour_map.get(event_type, "#C4C9D4")
        # Multi-line details (e.g. "Reasons: a; b") get an indented continuation
        # so bullet-style detail lines read clearly under the event line.
        detail_html = detail.replace(" Reasons: ", "<br>&nbsp;&nbsp;&nbsp;&nbsp;Reasons: ")
        entry = (
            f'<p style="margin: 0 0 6px 0; line-height: 145%;">'
            f'<span style="color:#4B5563;">[{timestamp}]</span> '
            f'<span style="color:{colour}; font-weight: bold;">[{event_type}]</span> '
            f'<span style="color:#C4C9D4;">{detail_html}</span>'
            f'</p>'
        )
        self.append(entry)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
