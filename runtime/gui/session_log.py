"""
session_log.py

Read-only QTextEdit that displays a timestamped session event log.
"""

from datetime import datetime
from PyQt6.QtWidgets import QTextEdit, QFrame
from PyQt6.QtCore import pyqtSlot

from runtime.gui import theme

_EVENT_COLOURS = {
    "ALERT": theme.RED,
    "DISMISS": theme.GREEN,
    "SESSION": theme.TEXT_DIM,
    "INFO": theme.TEXT,
    "ERROR": theme.AMBER,
}


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
        colour = _EVENT_COLOURS.get(event_type, theme.TEXT)
        # Kept single-line by design: full reasoning lives in the Detection
        # Analysis card, not in this event timeline (see main_window.py).
        entry = (
            f'<p style="margin: 0 0 6px 0; line-height: 145%;">'
            f'<span style="color:{theme.TEXT_MUTED};">[{timestamp}]</span> '
            f'<span style="color:{colour}; font-weight: bold;">[{event_type}]</span> '
            f'<span style="color:{theme.TEXT};">{detail}</span>'
            f'</p>'
        )
        self.append(entry)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
