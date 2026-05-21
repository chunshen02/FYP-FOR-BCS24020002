"""
alert_system.py

Manages the visual alert overlay and Windows toast notifications.
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSlot
import sys


class AlertOverlay(QWidget):
    """
    Applies a red border overlay to the parent window on alert.
    """

    ALERT_STYLE = "border: 4px solid red;"
    NORMAL_STYLE = "border: none;"

    def __init__(self, target_widget):
        super().__init__()
        self._target = target_widget
        _ = sys

    @pyqtSlot(float)
    def on_alert_triggered(self, peak_p_fake: float):
        """Show red border and send toast notification."""
        self._target.setStyleSheet(self.ALERT_STYLE)
        self._send_toast(peak_p_fake)

    @pyqtSlot()
    def on_alert_dismissed(self):
        """Remove red border."""
        self._target.setStyleSheet(self.NORMAL_STYLE)

    def _send_toast(self, peak_p_fake: float):
        try:
            from plyer import notification

            notification.notify(
                title="Deepfake Detected",
                message=f"Deepfake face detected. Confidence: {peak_p_fake:.1%}",
                app_name="Deepfake Detector",
                timeout=5,
            )
        except Exception:
            pass  # Toast is non-critical; silently skip if unavailable
