"""
alert_system.py

Out-of-window alert cues (taskbar flash, beep, desktop notification), so an
alert is noticed even while the meeting window has focus.
"""

import threading
from PyQt6.QtWidgets import QApplication, QWidget
from plyer import notification

import config


def notify_alert(window: QWidget, peak_p_fake: float):
    """Flash the taskbar entry, plus the beep and desktop notification if enabled in Settings."""
    QApplication.alert(window)
    if config.ALERT_SOUND_ENABLED:
        QApplication.beep()
    if config.ALERT_NOTIFICATION_ENABLED:
        # plyer's Windows backend sleeps for the full notification timeout, so it
        # must not run on the GUI thread.
        threading.Thread(target=_send_toast, args=(peak_p_fake,), daemon=True).start()


def _send_toast(peak_p_fake: float):
    try:
        notification.notify(
            title="Deepfake Detected",
            message=f"Deepfake face detected. Confidence: {peak_p_fake:.1%}",
            app_name="Deepfake Detector",
            timeout=5,
        )
    except Exception:
        pass  # Toast is non-critical; the in-app alert card is still shown
